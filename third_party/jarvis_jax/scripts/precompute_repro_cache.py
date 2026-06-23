"""Precompute reprojected volumes for the HybridNet cached trainer.

Sweeps all framesets in the requested split, runs frozen ViTPose + reproject
(HybridNet3D.reproject_volume), and streams fp16 volumes to a memmap so the
v2vNet trainer can skip the expensive front-end on every step.

Usage (Hydra)
-------------
    python scripts/precompute_repro_cache.py \\
        paths=hyak cache=default \\
        cache.split=val \\
        cache.batch=8 \\
        cache.limit=0 \\
        cache.force=false

Idempotent: if <split>_meta.json exists and the 'vitpose_ckpt', 'grid_size',
'grid_spacing', 'roi_cube', 'heatmap_size' keys match (and --limit matches n),
and the volume file has the expected size, the precompute is skipped unless
cache.force=true is given.
"""

from __future__ import annotations

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

import hydra
from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, build_dataclass

register_resolvers()

# Fixed volume shape constants — kept in sync with repro_cache.py
_NUM_JOINTS = 50
_GRID = 48


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


def run_precompute(
    *,
    root: str,
    vitpose_ckpt: str,
    cache_dir: str,
    split: str,
    batch: int,
    limit: int,
    force: bool,
    vitpose_cfg=None,
):
    """Precompute reprojected volumes and write them to a memmap cache.

    Args:
        root:        Root of the V3 dataset (contains annotations/, train/, val/).
        vitpose_ckpt: Orbax ViTPose checkpoint directory.
        cache_dir:   Output directory for the cache files.
        split:       Dataset split: 'train' or 'val'.
        batch:       Framesets per batch.
        limit:       Cache only the first N framesets; 0 = all.
        force:       Overwrite existing cache even if meta matches.
        vitpose_cfg: Optional ViTPoseConfig instance; built from defaults if None.
    """
    from flax import nnx

    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.convert.build_checkpoint import load_vitpose
    from jarvis_jax.data.repro_cache import load_cache, write_cache
    from jarvis_jax.data.v3_3d import V3FramesetDataset
    from jarvis_jax.hybridnet.model import HybridNet3D
    from jarvis_jax.hybridnet.v2vnet import V2VNet
    from jarvis_jax.sharding import data_parallel_mesh, replicate, shard_batch

    if vitpose_cfg is None:
        vitpose_cfg = ViTPoseConfig()

    # ------------------------------------------------------------------
    # Load dataset to determine n
    # ------------------------------------------------------------------
    print(f"[precompute] Loading dataset: {root} split={split}")
    ds = V3FramesetDataset(root, split)
    total = len(ds)
    n = min(limit, total) if limit > 0 else total
    print(f"[precompute] {total} framesets in split; will cache {n}")

    # ------------------------------------------------------------------
    # Idempotency check
    # ------------------------------------------------------------------
    if not force and _cache_is_valid(cache_dir, split, vitpose_ckpt, n):
        print(f"[precompute] Cache already valid at {cache_dir} "
              f"(split={split}, n={n}). Use cache.force=true to recompute.")
        return

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    print(f"[precompute] Loading ViTPose from {vitpose_ckpt}")
    vitpose = load_vitpose(vitpose_ckpt, vitpose_cfg)
    v2vnet = V2VNet(vitpose_cfg.num_keypoints, vitpose_cfg.num_keypoints, rngs=nnx.Rngs(0))
    model = HybridNet3D(vitpose, v2vnet, vitpose_cfg)

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
    meta["vitpose_ckpt"] = vitpose_ckpt
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
    batch_size = batch
    t0 = time.perf_counter()
    written_count = [0]  # mutable counter accessible inside generator

    def _volumes_generator():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sel = indices[start:end]
            B = len(sel)

            # Load samples (manual index sweep to enforce --limit exactly)
            samples = [ds[int(i)] for i in sel]
            batch_data = {key: np.stack([s[key] for s in samples], axis=0)
                         for key in samples[0]}

            # Save original labels before padding inputs
            kp3d_batch = batch_data["kp3d"]
            c3d_batch = batch_data["center3D"]
            vis_batch = batch_data["vis"]

            # Pad inputs on axis 0 to device-multiple if necessary (fixes IndivisibleError)
            # on partial batches. shard_batch requires axis-0 divisible by len(jax.devices()).
            nd = len(jax.devices())
            pad = (-B) % nd
            if pad > 0:
                batch_data["crops4"] = np.concatenate(
                    [batch_data["crops4"], np.repeat(batch_data["crops4"][-1:], pad, axis=0)], axis=0)
                batch_data["center3D"] = np.concatenate(
                    [batch_data["center3D"], np.repeat(batch_data["center3D"][-1:], pad, axis=0)], axis=0)
                batch_data["centerHM"] = np.concatenate(
                    [batch_data["centerHM"], np.repeat(batch_data["centerHM"][-1:], pad, axis=0)], axis=0)
                batch_data["cameraMatrices"] = np.concatenate(
                    [batch_data["cameraMatrices"], np.repeat(batch_data["cameraMatrices"][-1:], pad, axis=0)], axis=0)

            # Shard inputs across devices
            with mesh:
                crops4 = shard_batch(jnp.asarray(batch_data["crops4"]), mesh)
                center3D_b = shard_batch(jnp.asarray(batch_data["center3D"]), mesh)
                centerHM = shard_batch(jnp.asarray(batch_data["centerHM"]), mesh)
                camMat = shard_batch(jnp.asarray(batch_data["cameraMatrices"]), mesh)

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
        cache_dir,
        split,
        volumes_iter=_volumes_generator(),
        n=n,
        kp3d=kp3d_acc,
        center3D=c3d_acc,
        vis=vis_acc,
        meta=meta,
    )

    elapsed = time.perf_counter() - t0
    vol_path = os.path.join(cache_dir, f"{split}_volumes.f16")
    vol_size_gb = os.path.getsize(vol_path) / 1e9
    print(f"\n[precompute] Done in {elapsed:.1f}s")
    print(f"[precompute] {vol_path}  ({vol_size_gb:.2f} GB)")
    print(f"[precompute] labels: {split}_labels.npz")
    print(f"[precompute] meta:   {split}_meta.json")

    # ------------------------------------------------------------------
    # Quick load_cache round-trip check
    # ------------------------------------------------------------------
    print("\n[precompute] Round-trip check via load_cache ...")
    c = load_cache(cache_dir, split)
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


def main_from_cfg(cfg):
    """Map a composed Hydra config into run_precompute kwargs and run."""
    from jarvis_jax.config import ViTPoseConfig
    vitpose_cfg = build_dataclass(ViTPoseConfig, cfg.model.vitpose)
    return run_precompute(
        root=cfg.paths.data_root,
        vitpose_ckpt=cfg.paths.vitpose_ckpt,
        cache_dir=cfg.paths.cache_dir,
        split=cfg.cache.split,
        batch=cfg.cache.batch,
        limit=cfg.cache.limit,
        force=cfg.cache.force,
        vitpose_cfg=vitpose_cfg,
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
