"""Precompute PER-CAMERA 2-D HEATMAPS (not reprojected volumes) for the
coarse-to-fine / rotation-augmentation training fan-out (Task 15).

WHY A SECOND CACHE. ``precompute_repro_cache.py`` bakes the frozen 2-D
front-end AND a FIXED, axis-aligned, spacing=1, grid_size=48 reprojection
into its cache -- exactly what an unaugmented, single-stage arm (A1_base)
needs, and nothing more. Two of Task 15's arm features cannot be built from
that cache no matter how it is resampled after the fact:

  1. ROTATION AUGMENTATION (``train.rot_augment``) means re-sampling the
     world grid along a ROTATED basis (``reproject.py::reproject_heatmaps``'s
     ``rotation`` kwarg) -- i.e. calling the DLT projection + heatmap gather
     again with different grid points, not resampling an already-gathered
     volume. ``rot_augment.py``'s own module docstring is explicit about this
     ("Rotating the grid basis during training... removes that dependence").
  2. COARSE-TO-FINE REFINEMENT (``model.refine.enabled``,
     ``hybridnet/refine.py``) re-reprojects a SMALL per-joint window at
     spacing=0.25 (vs. the coarse spacing=1) -- "matched to the 2-D detail
     that ALREADY EXISTS" in the raw heatmaps. At spacing=1 each output voxel
     already truncates its heatmap lookup to whole-pixel granularity
     (``reproject.py``: ``(val/2).astype(int32)``), so the sub-voxel detail a
     spacing=0.25 re-projection recovers is NOT present in the coarse cached
     volume; interpolating that volume more finely would manufacture detail,
     not recover it.

Caching the per-camera heatmaps INSTEAD (this script) lets every arm redo
just the reprojection (cheap: a DLT matmul + bilinear coordinate-map upsample
+ integer gather, no front-end forward pass) at train time, with whatever
grid_size/grid_spacing/rotation that arm needs -- exactly the primitives
``reproject.py``, ``rot_augment.py`` and ``refine.py`` already implement.
This is what makes ALL 8 Task-15 arms possible from ONE cache, including
A5_hires's grid_size=96/spacing=0.5 (same physical FOV as the default
48/spacing=1, twice the sampling -- a per-arm REPROJECTION hyperparameter,
not a property baked into the cache).

COST. The expensive step (frozen-front-end forward pass over ~3.7k
framesets x 7 cameras, I/O-bound on per-sample crop/mask loading -- see
precompute_repro_cache.py's own measured ~1.4-2.0 framesets/s) is IDENTICAL
to the volume cache's build; this script pays it again because it needs a
DIFFERENT front-end output (heatmaps, not the fixed-grid reprojection), not
because anything is redundant to skip. Heatmaps are stored fp16 UNPADDED
(224x224, ViTPoseConfig.heatmap_size) and un-transposed exactly as
``HybridNet3D.predict_heatmaps`` returns them -- padding to 226 and the
train-time gather-index transpose are redone on GPU at train time (see
``jarvis_jax/train/train_3d_cached.py``), matching ``reproject_volume``'s own
pad+transpose+``/255`` steps exactly so numbers stay comparable across the
two caches.

Layout on disk
--------------
<cache_dir>/<split>_heatmaps.f16      memmap (n, num_cam, J, 224, 224) fp16
<cache_dir>/<split>_hmlabels.npz      kp3d (n,J,3) f32, vis (n,J) bool,
                                      center3D (n,3) f32,
                                      centerHM (n,num_cam,2) f32,
                                      cameraMatrices (n,num_cam,4,3) f32,
                                      fly_id (n,) int32,
                                      calib_group (n,) '<U1',
                                      is_female (n,) bool
<cache_dir>/<split>_hmmeta.json       front_end, frontend_ckpt, dataset_version,
                                      num_cameras, heatmap_size, keypoint_names,
                                      skeleton, n, split

Idempotent the same way precompute_repro_cache.py is: a matching
<split>_hmmeta.json + correctly-sized volume file skips the rebuild unless
--force.

Usage
-----
    python scripts/precompute_heatmap_cache.py \
        --v5-root $V5 --vitpose-ckpt $VIT --cache-dir $HMCACHE \
        --split train --batch 64
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_HM = 224  # ViTPoseConfig.heatmap_size (unpadded)
_NUM_JOINTS = 50


def _hm_meta_path(cache_dir, split):
    return os.path.join(cache_dir, f"{split}_hmmeta.json")


def _hm_vol_path(cache_dir, split):
    return os.path.join(cache_dir, f"{split}_heatmaps.f16")


def _hm_labels_path(cache_dir, split):
    return os.path.join(cache_dir, f"{split}_hmlabels.npz")


def _cache_is_valid(cache_dir, split, vitpose_ckpt, n, num_cam):
    meta_path = _hm_meta_path(cache_dir, split)
    vol_path = _hm_vol_path(cache_dir, split)
    if not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("vitpose_ckpt") != vitpose_ckpt:
            return False
        if int(meta.get("n", -1)) != n:
            return False
        if int(meta.get("num_cameras", -1)) != num_cam:
            return False
        if int(meta.get("heatmap_size", -1)) != _HM:
            return False
        expected_bytes = n * num_cam * _NUM_JOINTS * _HM * _HM * 2  # fp16
        if not os.path.isfile(vol_path):
            return False
        if os.path.getsize(vol_path) != expected_bytes:
            return False
        return True
    except Exception:
        return False


def run_precompute(*, v5_root, vitpose_ckpt, cache_dir, split, batch, limit, force):
    from flax import nnx

    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.convert.build_checkpoint import load_vitpose
    from jarvis_jax.data.device import normalize_image
    from jarvis_jax.data.v5_3d import V5FramesetDataset
    from jarvis_jax.sharding import data_parallel_mesh, replicate, shard_batch

    vitpose_cfg = ViTPoseConfig()

    print(f"[hmcache] Loading dataset: {v5_root} split={split} (v5)")
    ds = V5FramesetDataset(v5_root, split)
    total = len(ds)
    n = min(limit, total) if limit > 0 else total
    num_cam = int(ds[0]["cameraMatrices"].shape[0])
    print(f"[hmcache] {total} framesets in split; will cache {n}; num_cam={num_cam}")

    if not force and _cache_is_valid(cache_dir, split, vitpose_ckpt, n, num_cam):
        print(f"[hmcache] Cache already valid at {cache_dir} (split={split}, n={n}). "
              f"Use --force to recompute.")
        return

    print(f"[hmcache] Loading ViTPose from {vitpose_ckpt}")
    model = load_vitpose(vitpose_ckpt, vitpose_cfg)

    mesh = data_parallel_mesh()
    num_devices = len(jax.devices())
    print(f"[hmcache] {num_devices} device(s); replicating model")
    graphdef, state = nnx.split(model)
    state = replicate(state, mesh)
    model = nnx.merge(graphdef, state)

    @jax.jit
    def _predict_hm(crops4):
        # crops4: (B, num_cam, 448, 448, 4) uint8
        B, C = crops4.shape[0], crops4.shape[1]
        flat = crops4.reshape(B * C, 448, 448, 4)
        imgs = jax.vmap(normalize_image)(flat)
        hm_flat = model(imgs, use_running_average=True)   # (B*C, 224, 224, J)
        J = hm_flat.shape[-1]
        hm = hm_flat.reshape(B, C, 224, 224, J)
        return jnp.transpose(hm, (0, 1, 4, 2, 3))          # (B, C, J, 224, 224)

    os.makedirs(cache_dir, exist_ok=True)
    mm = np.memmap(_hm_vol_path(cache_dir, split), dtype=np.float16, mode="w+",
                   shape=(n, num_cam, _NUM_JOINTS, _HM, _HM))

    kp3d_acc = np.zeros((n, _NUM_JOINTS, 3), np.float32)
    vis_acc = np.zeros((n, _NUM_JOINTS), bool)
    center3D_acc = np.zeros((n, 3), np.float32)
    centerHM_acc = np.zeros((n, num_cam, 2), np.float32)
    camMat_acc = np.zeros((n, num_cam, 4, 3), np.float32)
    fly_id_acc = np.zeros((n,), np.int32)
    calib_group_acc = np.empty((n,), dtype="<U1")
    is_female_acc = np.zeros((n,), bool)

    indices = np.arange(n)
    t0 = time.perf_counter()
    written = 0
    for start in range(0, n, batch):
        end = min(start + batch, n)
        sel = indices[start:end]
        B = len(sel)
        samples = [ds[int(i)] for i in sel]
        batch_data = {key: np.stack([s[key] for s in samples], axis=0)
                     for key in ("crops4", "center3D", "centerHM", "cameraMatrices",
                                 "kp3d", "vis", "fly_id")}

        nd = len(jax.devices())
        pad = (-B) % nd
        crops4 = batch_data["crops4"]
        if pad > 0:
            crops4 = np.concatenate([crops4, np.repeat(crops4[-1:], pad, axis=0)], axis=0)

        with mesh:
            crops4_dev = shard_batch(jnp.asarray(crops4), mesh)
            hm = _predict_hm(crops4_dev)

        hm_np = np.asarray(hm)[:B]

        kp3d_acc[start:end] = batch_data["kp3d"]
        vis_acc[start:end] = batch_data["vis"]
        center3D_acc[start:end] = batch_data["center3D"]
        centerHM_acc[start:end] = batch_data["centerHM"]
        camMat_acc[start:end] = batch_data["cameraMatrices"]
        fly_id_acc[start:end] = batch_data["fly_id"]
        for k, i in enumerate(sel):
            rec = ds._fs[i]["recording"]
            calib_group_acc[start + k] = ds.manifest[rec]["calib_group"]
            is_female_acc[start + k] = ds.is_female(int(i))

        mm[start:end] = hm_np.astype(np.float16)
        written += B
        elapsed = time.perf_counter() - t0
        fps = written / elapsed
        print(f"[hmcache] {written}/{n} framesets ({fps:.2f} fs/s) "
              f"hm_range=[{float(hm_np.min()):.3f}, {float(hm_np.max()):.3f}]")

    mm.flush()
    del mm

    np.savez(_hm_labels_path(cache_dir, split),
             kp3d=kp3d_acc, vis=vis_acc, center3D=center3D_acc,
             centerHM=centerHM_acc, cameraMatrices=camMat_acc,
             fly_id=fly_id_acc, calib_group=calib_group_acc,
             is_female=is_female_acc)

    meta = dict(
        front_end="vitpose",
        vitpose_ckpt=vitpose_ckpt,
        dataset_version="v5",
        num_cameras=num_cam,
        heatmap_size=_HM,
        keypoint_names=list(ds.keypoint_names),
        skeleton=ds.skeleton,
        n=n,
        split=split,
    )
    with open(_hm_meta_path(cache_dir, split), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.perf_counter() - t0
    vol_size_gb = os.path.getsize(_hm_vol_path(cache_dir, split)) / 1e9
    print(f"\n[hmcache] Done in {elapsed:.1f}s")
    print(f"[hmcache] {_hm_vol_path(cache_dir, split)}  ({vol_size_gb:.2f} GB)")

    # Round-trip check
    mm2 = np.memmap(_hm_vol_path(cache_dir, split), dtype=np.float16, mode="r",
                    shape=(n, num_cam, _NUM_JOINTS, _HM, _HM))
    sample = np.asarray(mm2[:min(32, n)]).astype(np.float32)
    finite_frac = float(np.isfinite(sample).mean())
    print(f"[hmcache] round-trip sample finite={finite_frac:.6f} "
          f"range=[{sample.min():.4f}, {sample.max():.4f}]")
    assert finite_frac == 1.0, "Non-finite values in cached heatmaps"
    print("[hmcache] Round-trip OK.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-root", required=True)
    ap.add_argument("--vitpose-ckpt", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_precompute(v5_root=args.v5_root, vitpose_ckpt=args.vitpose_ckpt,
                   cache_dir=args.cache_dir, split=args.split,
                   batch=args.batch, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
