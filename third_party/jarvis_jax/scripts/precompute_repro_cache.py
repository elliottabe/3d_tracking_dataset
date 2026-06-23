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

Idempotent: if <split>_meta.json exists and the 'vitpose_ckpt', 'grid_size',
'grid_spacing', 'roi_cube', 'heatmap_size' keys match (and --limit matches n),
and the volume file has the expected size, the precompute is skipped unless
--force is given.
"""

from __future__ import annotations

import argparse
import json
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

# Fixed volume shape constants — kept in sync with repro_cache.py
_NUM_JOINTS = 50
_GRID = 48


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


# Grid params written into meta — single source of truth for the staleness check
_META_GRID_PARAMS = {
    "grid_size": 48,
    "grid_spacing": 1,
    "roi_cube": 48,
    # heatmap_size is BOUNDING_BOX/2 + 2 = 448/2 + 2 = 226
    "heatmap_size": 226,
}


def _cache_is_valid(
    cache_dir: str,
    split: str,
    vitpose_ckpt: str,
    n: int,
) -> bool:
    """Return True if a valid, matching, complete cache already exists.

    Checks:
      - meta.json matches vitpose_ckpt, n, and all grid params
      - volume file exists and has the expected byte size (guards truncated files)
    """
    meta_path = os.path.join(cache_dir, f"{split}_meta.json")
    vol_path = os.path.join(cache_dir, f"{split}_volumes.f16")
    if not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("vitpose_ckpt") != vitpose_ckpt:
            return False
        if int(meta.get("n", -1)) != n:
            return False
        for key, expected in _META_GRID_PARAMS.items():
            if meta.get(key) != expected:
                return False
        # Guard against truncated volume files (e.g. killed mid-write)
        expected_bytes = n * _NUM_JOINTS * _GRID * _GRID * _GRID * 2  # fp16 = 2 bytes
        if not os.path.isfile(vol_path):
            return False
        if os.path.getsize(vol_path) != expected_bytes:
            return False
        return True
    except Exception:
        return False


def main(argv=None):
    args = _parse_args(argv)

    from flax import nnx

    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.convert.build_checkpoint import load_vitpose
    from jarvis_jax.data.repro_cache import load_cache, write_cache
    from jarvis_jax.data.v3_3d import V3FramesetDataset
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
    # Build meta dict
    # ------------------------------------------------------------------
    meta = dict(_META_GRID_PARAMS)
    meta["vitpose_ckpt"] = args.vitpose_ckpt
    meta["num_cameras"] = int(ds[0]["cameraMatrices"].shape[0])
    # Carry the skeleton so the cached trainer can wire the graph-Laplacian
    # bone prior (train_3d_cached reads meta["keypoint_names"]/["skeleton"]).
    # Without this the cached run silently trains with zero edges.
    meta["keypoint_names"] = list(ds.keypoint_names)
    meta["skeleton"] = ds.skeleton

    # ------------------------------------------------------------------
    # Preallocate label accumulators (filled as volumes are generated)
    # ------------------------------------------------------------------
    kp3d_acc = np.zeros((n, _NUM_JOINTS, 3), dtype=np.float32)
    c3d_acc = np.zeros((n, 3), dtype=np.float32)
    vis_acc = np.zeros((n, _NUM_JOINTS), dtype=bool)

    # ------------------------------------------------------------------
    # Generator: yields per-frameset volumes, fills label accumulators
    # as a side effect so write_cache can consume it fully before labels
    # are passed.  Labels are preallocated above and passed by reference;
    # write_cache consumes the generator (filling them), then writes labels.
    # ------------------------------------------------------------------
    indices = np.arange(n)  # first n, in order (shuffle=False)
    batch_size = args.batch
    t0 = time.perf_counter()
    written_count = [0]  # mutable counter accessible inside generator

    def _volumes_generator():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sel = indices[start:end]
            B = len(sel)

            # Load samples (manual index sweep to enforce --limit exactly)
            samples = [ds[int(i)] for i in sel]
            batch = {key: np.stack([s[key] for s in samples], axis=0)
                     for key in samples[0]}

            # Save original labels before padding inputs
            kp3d_batch = batch["kp3d"]
            c3d_batch = batch["center3D"]
            vis_batch = batch["vis"]

            # Pad inputs on axis 0 to device-multiple if necessary (fixes IndivisibleError)
            # on partial batches. shard_batch requires axis-0 divisible by len(jax.devices()).
            nd = len(jax.devices())
            pad = (-B) % nd
            if pad > 0:
                batch["crops4"] = np.concatenate(
                    [batch["crops4"], np.repeat(batch["crops4"][-1:], pad, axis=0)], axis=0)
                batch["center3D"] = np.concatenate(
                    [batch["center3D"], np.repeat(batch["center3D"][-1:], pad, axis=0)], axis=0)
                batch["centerHM"] = np.concatenate(
                    [batch["centerHM"], np.repeat(batch["centerHM"][-1:], pad, axis=0)], axis=0)
                batch["cameraMatrices"] = np.concatenate(
                    [batch["cameraMatrices"], np.repeat(batch["cameraMatrices"][-1:], pad, axis=0)], axis=0)

            # Shard inputs across devices
            with mesh:
                crops4 = shard_batch(jnp.asarray(batch["crops4"]), mesh)
                center3D_b = shard_batch(jnp.asarray(batch["center3D"]), mesh)
                centerHM = shard_batch(jnp.asarray(batch["centerHM"]), mesh)
                camMat = shard_batch(jnp.asarray(batch["cameraMatrices"]), mesh)

                vol = _reproject(crops4, center3D_b, centerHM, camMat)

            # Pull to host; vol shape: (B_padded, 50, 48, 48, 48)
            vol_np = np.asarray(vol)
            # Slice off padding to restore original batch size
            vol_np = vol_np[:B]

            # Accumulate labels for this batch (use original unpadded batch)
            kp3d_acc[start:end] = kp3d_batch
            c3d_acc[start:end] = c3d_batch
            vis_acc[start:end] = vis_batch

            # Yield individual frameset volumes (fp16)
            for b in range(B):
                vol_b = vol_np[b].astype(np.float16)
                written_count[0] += 1
                elapsed = time.perf_counter() - t0
                fps = written_count[0] / elapsed
                print(f"[precompute] {written_count[0]}/{n} framesets  "
                      f"({fps:.1f} fs/s)  vol_range=["
                      f"{float(vol_b.min()):.3f}, {float(vol_b.max()):.3f}]")
                yield vol_b

    # ------------------------------------------------------------------
    # write_cache: consumes the generator (filling label accumulators),
    # then writes labels + meta in one place (no duplicated logic here)
    # ------------------------------------------------------------------
    write_cache(
        args.cache_dir,
        args.split,
        volumes_iter=_volumes_generator(),
        n=n,
        kp3d=kp3d_acc,
        center3D=c3d_acc,
        vis=vis_acc,
        meta=meta,
    )

    elapsed = time.perf_counter() - t0
    vol_path = os.path.join(args.cache_dir, f"{args.split}_volumes.f16")
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
    assert c["volumes"].shape == (n, _NUM_JOINTS, _GRID, _GRID, _GRID), (
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
