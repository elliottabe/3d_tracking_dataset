"""Reprojected-volume cache: writer and reader.

Stores per-frameset reprojected volumes (fp16) and labels so the v2vNet
trainer can skip frozen ViTPose + reproject every step.

Layout on disk
--------------
<cache_dir>/
    <split>_volumes.f16      np.memmap (n, 50, 48, 48, 48) float16
    <split>_labels.npz       kp3d (n,50,3) f32, center3D (n,3) f32, vis (n,50) bool
    <split>_meta.json        dict with vitpose_ckpt, grid params, n, split

Public API
----------
write_cache(cache_dir, split, *, volumes_iter, n, kp3d, center3D, vis, meta)
    Stream n volumes from `volumes_iter` (each (50,48,48,48) fp16 or f32)
    to the memmap WITHOUT holding all data in RAM, then write labels.npz
    and meta.json.

load_cache(cache_dir, split) -> dict
    Returns:
        volumes   : np.memmap (n,50,48,48,48) float16 (mode='r')
        kp3d      : np.ndarray (n,50,3) float32
        center3D  : np.ndarray (n,3)   float32
        vis       : np.ndarray (n,50)  bool
        meta      : dict
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np

# Fixed volume shape constants — kept in sync with HybridNet3D.reproject_volume
_NUM_JOINTS = 50
_GRID = 48  # spatial dimension of the reprojected volume

_VOL_SHAPE = (_NUM_JOINTS, _GRID, _GRID, _GRID)  # per-frameset


def _vol_path(cache_dir: str, split: str) -> str:
    return os.path.join(cache_dir, f"{split}_volumes.f16")


def _labels_path(cache_dir: str, split: str) -> str:
    return os.path.join(cache_dir, f"{split}_labels.npz")


def _meta_path(cache_dir: str, split: str) -> str:
    return os.path.join(cache_dir, f"{split}_meta.json")


# ---------------------------------------------------------------------------
# write_cache
# ---------------------------------------------------------------------------

def write_cache(
    cache_dir: str,
    split: str,
    *,
    volumes_iter: Iterable[np.ndarray],
    n: int,
    kp3d: np.ndarray,        # (n, 50, 3) float32
    center3D: np.ndarray,    # (n, 3)     float32
    vis: np.ndarray,         # (n, 50)    bool
    meta: dict,
) -> None:
    """Write reprojected volumes and labels to cache_dir.

    Volumes are streamed one-by-one from *volumes_iter* to a memmap so that
    the full (n, 50, 48, 48, 48) array (up to ~29 GB for n~3 k) is never
    held in RAM simultaneously.

    Args:
        cache_dir:     Directory to write files into (created if absent).
        split:         Dataset split name, e.g. 'train' or 'val'.
        volumes_iter:  Iterable yielding n arrays of shape (50,48,48,48).
                       Each element may be float16 or float32; stored as fp16.
        n:             Total number of framesets expected from the iterator.
        kp3d:          ``(n, 50, 3)`` float32 triangulated 3-D keypoints.
        center3D:      ``(n, 3)`` float32 3-D ROI centres.
        vis:           ``(n, 50)`` bool visibility flags.
        meta:          Dict written to meta.json (must include 'vitpose_ckpt'
                       and 'grid_size'; 'n' and 'split' are injected/overridden).
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Open memmap for writing — shape (n, J, G, G, G) fp16
    mm = np.memmap(
        _vol_path(cache_dir, split),
        dtype=np.float16,
        mode="w+",
        shape=(n, *_VOL_SHAPE),
    )

    written = 0
    for vol in volumes_iter:
        if written >= n:
            break
        vol = np.asarray(vol)
        if vol.dtype != np.float16:
            vol = vol.astype(np.float16)
        mm[written] = vol
        written += 1

    if written != n:
        raise ValueError(
            f"write_cache: volumes_iter yielded {written} items but n={n}")

    # Flush to disk (no-op for w+ mode but explicit is clearer)
    mm.flush()
    del mm  # release file handle

    # Write labels
    np.savez(
        _labels_path(cache_dir, split),
        kp3d=np.asarray(kp3d, dtype=np.float32),
        center3D=np.asarray(center3D, dtype=np.float32),
        vis=np.asarray(vis, dtype=bool),
    )

    # Write meta — inject / normalise n and split
    meta_out = dict(meta)
    meta_out["n"] = n
    meta_out["split"] = split
    with open(_meta_path(cache_dir, split), "w") as f:
        json.dump(meta_out, f, indent=2)


# ---------------------------------------------------------------------------
# load_cache
# ---------------------------------------------------------------------------

def load_cache(cache_dir: str, split: str) -> dict:
    """Load a previously-written cache.

    Returns a dict with:
        volumes   : np.memmap (n, 50, 48, 48, 48) float16, mode='r'
        kp3d      : np.ndarray (n, 50, 3) float32
        center3D  : np.ndarray (n, 3)     float32
        vis       : np.ndarray (n, 50)    bool
        meta      : dict
    """
    meta_file = _meta_path(cache_dir, split)
    if not os.path.isfile(meta_file):
        raise FileNotFoundError(
            f"Cache meta not found: {meta_file}. "
            f"Run precompute_repro_cache.py first.")

    with open(meta_file) as f:
        meta = json.load(f)

    n = int(meta["n"])

    mm = np.memmap(
        _vol_path(cache_dir, split),
        dtype=np.float16,
        mode="r",
        shape=(n, *_VOL_SHAPE),
    )

    with np.load(_labels_path(cache_dir, split)) as f:
        kp3d = f["kp3d"].copy()
        center3D = f["center3D"].copy()
        vis = f["vis"].copy()

    return {
        "volumes": mm,
        "kp3d": kp3d,
        "center3D": center3D,
        "vis": vis,
        "meta": meta,
    }
