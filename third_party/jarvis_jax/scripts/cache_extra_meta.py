"""Companion per-frameset metadata for an already-built v5 reprojected-volume
cache (scripts/precompute_repro_cache.py).

WHY A SEPARATE FILE. write_cache()/load_cache() (jarvis_jax/data/repro_cache.py)
only ever stored kp3d/center3D/vis -- enough to train+eval V2VNet, but not
enough to break results down by calibration group or sex, or to drop the
264 "second fly" framesets (Task 15, arm A6). Re-running precompute_repro_cache.py
to add these fields would redo the EXPENSIVE part (frozen-front-end inference +
GPU reprojection over ~3.7k framesets, ~35 min) just to pick up three cheap,
purely-metadata fields that don't depend on it at all. V5FramesetDataset's
frameset order is a deterministic sort over `coco["framesets"].items()`
(jarvis_jax/data/v5_3d.py), so re-instantiating it here reproduces the EXACT
same index order the cache was built with (no shuffling anywhere in that
path) -- this script is safe to run standalone, any time, without touching
the cache's volumes/labels files.

Writes ``<cache_dir>/<split>_extra_meta.npz`` with:
    fly_id      (n,) int32   -- 0 = primary fly, 1 = the second fly in a
                                two-fly frameset (see V5FramesetDataset
                                module docstring: 264 such framesets overall).
    calib_group (n,) '<U1'   -- rig calibration group letter (manifest.json).
    is_female   (n,) bool    -- V5FramesetDataset.is_female(idx): the SAME
                                resolved-sex fallback chain used to filter/
                                weight by sex elsewhere (_resolve_sex).

Usage:
    python scripts/cache_extra_meta.py \
        --v5-root $V5 --cache-dir $CACHE --split train
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from jarvis_jax.data.v5_3d import V5FramesetDataset


def build_extra_meta(v5_root: str, split: str) -> dict:
    ds = V5FramesetDataset(v5_root, split)
    n = len(ds)
    fly_id = np.array([int(fsv["fly_id"]) for fsv in ds._fs], dtype=np.int32)
    calib_group = np.array(
        [ds.manifest[fsv["recording"]]["calib_group"] for fsv in ds._fs])
    is_female = np.array([ds.is_female(i) for i in range(n)], dtype=bool)
    assert len(fly_id) == n and len(calib_group) == n and len(is_female) == n
    return {"fly_id": fly_id, "calib_group": calib_group, "is_female": is_female}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-root", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    args = ap.parse_args()

    extra = build_extra_meta(args.v5_root, args.split)
    out = os.path.join(args.cache_dir, f"{args.split}_extra_meta.npz")
    np.savez(out, **extra)
    print(f"[cache_extra_meta] wrote {out}  n={len(extra['fly_id'])}  "
          f"fly_id!=0: {int((extra['fly_id'] != 0).sum())}  "
          f"female: {int(extra['is_female'].sum())}  "
          f"groups: {sorted(set(extra['calib_group'].tolist()))}")


if __name__ == "__main__":
    main()
