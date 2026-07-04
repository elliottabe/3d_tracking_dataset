#!/usr/bin/env python3
"""Remap tool for LEGACY `sam3_masks.npz` files whose camera (C) axis order
was never recorded and may not match the calibration -- the courtship SAM3
camera-order-scramble bug (triangulating the stored per-camera mask
centroids under the identity mapping gave 46px reprojection residual; the
correct permutation gave 6.8px -- see
jarvis_jax.cse.courtship_bout_masks.detect_camera_order).

For each bout under --predictions-dir, this GEOMETRICALLY DETECTS the true
per-camera order of the stored masks (brute-force permutation search against
the calibration), permutes the packed masks/valid/centroids C axis into
--cameras order, and writes a corrected sam3_masks.npz (now carrying a
`cameras` name array, so it never needs this again) to --out. Prints the
identity-mapping residual vs the corrected residual for every bout so a
scramble (large gap) is obvious.

NEVER overwrites the input --predictions-dir -- always writes to a separate
--out directory. Does not touch mask/pixel *content*, only which C-axis slot
each camera's data lives in.

Usage:
    python scripts/fix_mask_camera_order.py \\
        --predictions-dir /path/to/Session0/.../Predictions_3D_36233268 \\
        --calib-dir /path/to/Session0/.../calibration \\
        --cameras Cam2012630,Cam2012631,Cam2012853,Cam2012855,Cam2012857,Cam2012861,Cam2012862 \\
        --out /path/to/Predictions_3D_36233268_fixed
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.cse.courtship_bout_masks import detect_camera_order


def bout_dirs(predictions_dir: str) -> list[int]:
    """Sorted bout indices discovered as bout_<idx> dirs under predictions_dir
    (mirrors scripts/run_courtship_bout.py's `_bout_dirs`)."""
    idxs = []
    for d in sorted(glob.glob(os.path.join(predictions_dir, "bout_*"))):
        m = re.match(r"bout_(\d+)$", os.path.basename(d))
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)


def stored_axis_to_expected_order(best_perm, rt_camera_names, expected_cameras) -> list[int]:
    """Index list `sel` (len C) such that indexing an array's camera axis
    (currently in the file's ORIGINAL STORED order) with `arr[:, sel]`
    reorders it into `expected_cameras` order.

    `best_perm` is `detect_camera_order`'s mask-index -> rt-camera-index
    mapping (rt-camera-index indexes into `rt_camera_names`, i.e.
    `list(rt.cameras)` -- NOT necessarily the same order as
    `expected_cameras`, e.g. if the calibration's own dict/glob order
    differs from the caller's desired output order).
    """
    best_perm = list(best_perm)
    inv = [0] * len(best_perm)                 # inv[rt_cam_idx] -> stored mask index
    for stored_i, rt_j in enumerate(best_perm):
        inv[rt_j] = stored_i
    name_to_rt_idx = {name: j for j, name in enumerate(rt_camera_names)}
    missing = [c for c in expected_cameras if c not in name_to_rt_idx]
    if missing:
        raise ValueError(
            f"expected camera(s) {missing} not in calibration cameras {rt_camera_names}")
    return [inv[name_to_rt_idx[c]] for c in expected_cameras]


def fix_one_bout(npz_path: str, rt: ReprojectionTool, expected_cameras: list[str], *,
                 fly: int = 0, n_frames: int = 20, max_resid: float = 20.0,
                 tol: float = 2.0):
    """Detect + correct one bout's sam3_masks.npz camera order.

    Returns (fixed_arrays: dict, identity_resid: float, best_resid: float,
    scrambled: bool). `fixed_arrays` is ready to `np.savez_compressed`.
    """
    with np.load(npz_path) as z:
        packed = np.asarray(z["packed"])
        valid = np.asarray(z["valid"])
        centroids = np.asarray(z["centroids"])
        shape = np.asarray(z["shape"])
        version = np.asarray(z["version"]) if "version" in z.files else None

    A = packed.shape[0]
    if not (0 <= fly < A):
        raise ValueError(f"{npz_path}: fly={fly} out of range (A={A})")

    centroids_fly = centroids[fly].transpose(1, 0, 2)   # (T,C,2)
    valid_fly = valid[fly].transpose(1, 0)              # (T,C)

    det = detect_camera_order(centroids_fly, valid_fly, rt, n_frames=n_frames)
    rt_camera_names = list(rt.cameras)
    sel = stored_axis_to_expected_order(det["best_perm"], rt_camera_names, expected_cameras)

    scrambled = (det["identity_resid"] - det["best_resid"] > tol
                and det["best_resid"] <= max_resid)

    fixed = dict(
        packed=packed[:, sel],
        valid=valid[:, sel],
        centroids=centroids[:, sel],
        shape=shape,
        cameras=np.array(list(expected_cameras)),
    )
    if version is not None:
        fixed["version"] = version
    return fixed, det["identity_resid"], det["best_resid"], scrambled


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-dir", required=True,
                    help="Session predictions dir holding bout_*/sam3_masks.npz")
    ap.add_argument("--calib-dir", required=True,
                    help="Calibration dir (Cam*.yaml) for this session/recording")
    ap.add_argument("--cameras", required=True,
                    help="Comma-separated expected camera names, in the desired "
                         "output C-axis order (e.g. matching cfg.recording.cameras)")
    ap.add_argument("--out", required=True,
                    help="Output root for corrected bout_*/sam3_masks.npz "
                         "(must NOT be --predictions-dir)")
    ap.add_argument("--fly", type=int, default=0,
                    help="Fly index whose centroids drive detection (default 0; "
                         "the permutation is applied to ALL flies)")
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--max-resid", type=float, default=20.0)
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--bout-ids", default="",
                    help="Comma-separated bout indices; default = every bout found")
    args = ap.parse_args()

    predictions_dir = os.path.abspath(args.predictions_dir)
    out_dir = os.path.abspath(args.out)
    if out_dir == predictions_dir:
        raise SystemExit(
            "--out must not be the same as --predictions-dir "
            "(never overwrite legacy masks in place)")
    expected_cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

    rt = ReprojectionTool(args.calib_dir)
    if rt.num_cameras != len(expected_cameras):
        raise SystemExit(
            f"--cameras has {len(expected_cameras)} name(s) but calibration "
            f"{args.calib_dir} has {rt.num_cameras} camera(s)")

    ids = ([int(x) for x in args.bout_ids.split(",") if x.strip()]
          if args.bout_ids.strip() else bout_dirs(predictions_dir))
    if not ids:
        raise SystemExit(f"no bout_* dirs found under {predictions_dir}")

    os.makedirs(out_dir, exist_ok=True)
    n_scrambled = 0
    n_done = 0
    for bi in ids:
        bout_name = f"bout_{bi:05d}"
        npz_path = os.path.join(predictions_dir, bout_name, "sam3_masks.npz")
        if not os.path.isfile(npz_path):
            print(f"[fix-mask-order] {bout_name}: no sam3_masks.npz -- skipping")
            continue

        fixed, identity_resid, best_resid, scrambled = fix_one_bout(
            npz_path, rt, expected_cameras, fly=args.fly, n_frames=args.n_frames,
            max_resid=args.max_resid, tol=args.tol)
        tag = "SCRAMBLED -> FIXED" if scrambled else "aligned"
        print(f"[fix-mask-order] {bout_name}: identity_resid={identity_resid:.2f}px "
             f"corrected_resid={best_resid:.2f}px [{tag}]")
        n_scrambled += int(scrambled)
        n_done += 1

        out_bout_dir = os.path.join(out_dir, bout_name)
        os.makedirs(out_bout_dir, exist_ok=True)
        out_npz = os.path.join(out_bout_dir, "sam3_masks.npz")
        tmp = out_npz + ".tmp.npz"
        np.savez_compressed(tmp, **fixed)
        os.replace(tmp, out_npz)

    print(f"[fix-mask-order] done: {n_done} bout(s) processed, "
         f"{n_scrambled} scramble(s) corrected -> {out_dir}")


if __name__ == "__main__":
    main()
