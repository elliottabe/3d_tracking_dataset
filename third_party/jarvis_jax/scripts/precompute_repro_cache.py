"""Precompute reprojected volumes for the HybridNet cached trainer.

Sweeps all framesets in the requested split, runs frozen ViTPose + reproject
(HybridNet3D.reproject_volume), and streams fp16 volumes to a memmap so the
v2vNet trainer can skip the expensive front-end on every step.

Usage
-----
    python scripts/precompute_repro_cache.py \\
        --root     /data/red_data_unified_V3 \\
        --vitpose-ckpt /data/jax_vitpose_runs/v3_8gpu_20260620/final \\
        --cache-dir /tmp/repro_cache \\
        --split    val \\
        --batch    8 \\
        --limit    0       # 0 = all
        --force            # overwrite existing cache

Idempotent: if <split>_meta.json exists and the 'vitpose_ckpt' key matches
(and --limit matches n), the precompute is skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# Add repo root to path when run directly (not as a module)
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Precompute reprojected volumes for HybridNet cached training.")
    ap.add_argument("--root", required=True,
                    help="Root of the V3 dataset (contains annotations/, train/, val/).")
    ap.add_argument("--vitpose-ckpt", required=True,
                    help="Orbax ViTPose checkpoint directory.")
    ap.add_argument("--cache-dir", required=True,
                    help="Output directory for the cache files.")
    ap.add_argument("--split", default="val",
                    help="Dataset split: 'train' or 'val' (default: val).")
    ap.add_argument("--batch", type=int, default=8,
                    help="Framesets per batch (default: 8).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cache only the first N framesets; 0 = all (default: 0).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing cache even if meta matches.")
    return ap.parse_args(argv)


def _cache_is_valid(cache_dir: str, split: str, vitpose_ckpt: str, n: int) -> bool:
    """Return True if a valid, matching cache already exists."""
    import json
    meta_path = os.path.join(cache_dir, f"{split}_meta.json")
    if not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return (
            meta.get("vitpose_ckpt") == vitpose_ckpt
            and int(meta.get("n", -1)) == n
        )
    except Exception:
        return False


def main(argv=None):
    args = _parse_args(argv)

    from flax import nnx

    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.convert.build_checkpoint import load_vitpose
    from jarvis_jax.data.repro_cache import load_cache, write_cache
    from jarvis_jax.data.v3_3d import V3FramesetDataset, frameset_batches
    from jarvis_jax.hybridnet.model import HybridNet3D
    from jarvis_jax.hybridnet.v2vnet import V2VNet
    from jarvis_jax.sharding import data_parallel_mesh, replicate, shard_batch

    cfg = ViTPoseConfig()

    # ------------------------------------------------------------------
    # Load dataset to determine n
    # ------------------------------------------------------------------
    print(f"[precompute] Loading dataset: {args.root} split={args.split}")
    ds = V3FramesetDataset(args.root, args.split)
    total = len(ds)
    n = min(args.limit, total) if args.limit > 0 else total
    print(f"[precompute] {total} framesets in split; will cache {n}")

    # ------------------------------------------------------------------
    # Idempotency check
    # ------------------------------------------------------------------
    if not args.force and _cache_is_valid(args.cache_dir, args.split,
                                           args.vitpose_ckpt, n):
        print(f"[precompute] Cache already valid at {args.cache_dir} "
              f"(split={args.split}, n={n}). Use --force to recompute.")
        return

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    print(f"[precompute] Loading ViTPose from {args.vitpose_ckpt}")
    vitpose = load_vitpose(args.vitpose_ckpt, cfg)
    v2vnet = V2VNet(cfg.num_keypoints, cfg.num_keypoints, rngs=nnx.Rngs(0))
    model = HybridNet3D(vitpose, v2vnet, cfg)

    # ------------------------------------------------------------------
    # Replicate model across devices
    # ------------------------------------------------------------------
    mesh = data_parallel_mesh()
    num_devices = len(jax.devices())
    print(f"[precompute] {num_devices} device(s); replicating model")
    graphdef, state = nnx.split(model)
    state = replicate(state, mesh)
    model = nnx.merge(graphdef, state)

    # JIT the reproject_volume call for performance
    @jax.jit
    def _reproject(crops4, center3D, centerHM, cameraMatrices):
        return model.reproject_volume(crops4, center3D, centerHM, cameraMatrices)

    # ------------------------------------------------------------------
    # Allocate label accumulators
    # ------------------------------------------------------------------
    kp3d_acc = np.zeros((n, 50, 3), dtype=np.float32)
    c3d_acc = np.zeros((n, 3), dtype=np.float32)
    vis_acc = np.zeros((n, 50), dtype=bool)

    # ------------------------------------------------------------------
    # Sweep batches, stream volumes to memmap
    # ------------------------------------------------------------------
    os.makedirs(args.cache_dir, exist_ok=True)
    vol_path = os.path.join(args.cache_dir, f"{args.split}_volumes.f16")
    mm = np.memmap(vol_path, dtype=np.float16, mode="w+",
                   shape=(n, 50, 48, 48, 48))

    # Build an index list limited to n framesets
    indices = np.arange(n)  # first n, in order (shuffle=False)

    written = 0
    t0 = time.perf_counter()

    # Use a manual index sweep so we can enforce the --limit cutoff exactly.
    batch_size = args.batch
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sel = indices[start:end]
        B = len(sel)

        # Load samples manually (frameset_batches shuffles, we want ordered)
        samples = [ds[int(i)] for i in sel]
        batch = {key: np.stack([s[key] for s in samples], axis=0) for key in samples[0]}

        # Shard inputs across devices
        with mesh:
            crops4 = shard_batch(jnp.asarray(batch["crops4"]), mesh)
            center3D_b = shard_batch(jnp.asarray(batch["center3D"]), mesh)
            centerHM = shard_batch(jnp.asarray(batch["centerHM"]), mesh)
            camMat = shard_batch(jnp.asarray(batch["cameraMatrices"]), mesh)

            vol = _reproject(crops4, center3D_b, centerHM, camMat)

        # vol: (B, 50, 48, 48, 48) jax array — pull to host, cast to fp16
        vol_np = np.asarray(vol).astype(np.float16)
        mm[start:end] = vol_np

        # Accumulate labels
        kp3d_acc[start:end] = batch["kp3d"]
        c3d_acc[start:end] = batch["center3D"]
        vis_acc[start:end] = batch["vis"]

        written += B
        elapsed = time.perf_counter() - t0
        fps = written / elapsed
        print(f"[precompute] {written}/{n} framesets  "
              f"({fps:.1f} fs/s)  vol_range=[{float(vol_np.min()):.3f}, "
              f"{float(vol_np.max()):.3f}]")

    mm.flush()
    del mm

    # ------------------------------------------------------------------
    # Write labels + meta
    # ------------------------------------------------------------------
    meta = {
        "vitpose_ckpt": args.vitpose_ckpt,
        "grid_size": 48,
        "grid_spacing": 1,
        "roi_cube": 48,
        "heatmap_size": 226,
        "num_cameras": int(ds[0]["cameraMatrices"].shape[0]),
        "n": n,
        "split": args.split,
    }

    np.savez(
        os.path.join(args.cache_dir, f"{args.split}_labels.npz"),
        kp3d=kp3d_acc,
        center3D=c3d_acc,
        vis=vis_acc,
    )
    import json
    with open(os.path.join(args.cache_dir, f"{args.split}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.perf_counter() - t0
    vol_size_gb = os.path.getsize(vol_path) / 1e9
    print(f"\n[precompute] Done in {elapsed:.1f}s")
    print(f"[precompute] {vol_path}  ({vol_size_gb:.2f} GB)")
    print(f"[precompute] labels: {args.split}_labels.npz")
    print(f"[precompute] meta:   {args.split}_meta.json")

    # ------------------------------------------------------------------
    # Quick load_cache round-trip check
    # ------------------------------------------------------------------
    print("\n[precompute] Round-trip check via load_cache ...")
    c = load_cache(args.cache_dir, args.split)
    assert c["volumes"].shape == (n, 50, 48, 48, 48), (
        f"Unexpected shape: {c['volumes'].shape}")
    assert c["volumes"].dtype == np.float16
    vols = np.asarray(c["volumes"])
    finite_frac = float(np.isfinite(vols).mean())
    vmin = float(vols.min())
    vmax = float(vols.max())
    print(f"[precompute] volumes shape={c['volumes'].shape}  "
          f"dtype={c['volumes'].dtype}  finite={finite_frac:.6f}  "
          f"range=[{vmin:.4f}, {vmax:.4f}]")
    assert finite_frac == 1.0, f"Non-finite values in cached volumes: {1-finite_frac:.2%}"
    print("[precompute] Round-trip OK.")


if __name__ == "__main__":
    main()
