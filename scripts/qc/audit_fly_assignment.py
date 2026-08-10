#!/usr/bin/env python3
"""Measure whether each fly's 2D keypoints land on that fly's OWN mask.

This is the check that caught the `predictions_dir` defect: if pose reads a
different mask set than the one identity was assigned on, every stage still
"succeeds" while fly0's keypoints track fly1.

Scored only on frames where both masks exist, their centroids are >`--sep` px
apart, and the keypoint centroid lies within `--margin` x that separation of the
nearer mask. When the flies overlap, "nearer its own centroid" is not a
meaningful question; the margin gate additionally drops frames where the
keypoints sit on NEITHER mask (a lost detector), which is a coverage failure
rather than an identity swap.

`--sep` defaults to 200 px, not 400. Measured over 798k valid (camera, frame)
pairs the median mask separation is 277 px and only 9.5% of frames exceed
400 px, so a 400 px gate scores just 54 of 160 bouts. Keypoint centroids sit
22-49 px from their own mask centroid, so 200 px separation under a 0.35 margin
is still decisive (own ~35 px vs other ~165 px) while covering all 160 bouts.

NOTE ON A BUG THIS SCRIPT EXISTS TO AVOID: `np.linalg.norm(v, -1)` passes -1 as
`ord`, not `axis`. For a (T,2) array that is a legal matrix norm returning a
SCALAR, silently comparing one number instead of T frames -- which shows up as
suspiciously exact 50.0% rates. Always `axis=-1`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "third_party" / "jarvis_jax"))

VIDEO_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship"


def kp_centroids(npz, ci, thresh):
    """(T,2) mean of confident keypoints in camera `ci`; NaN where none."""
    g = npz["conf"][:, ci] >= thresh
    return np.nanmean(np.where(g[..., None], npz["kp2d"][:, ci], np.nan), axis=1)


def audit_bout(mask_npz, pose_dir, bout, cams, sep_px=200.0, thresh=0.5,
               margin=0.35):
    with np.load(mask_npz, allow_pickle=True) as z:
        order = [[str(x) for x in z["cameras"]].index(c) for c in cams]
        cent = np.asarray(z["centroids"])[:, order]
        val = np.asarray(z["valid"], bool)[:, order]
    if cent.shape[0] < 2:
        return None
    try:
        kps = {f: np.load(Path(pose_dir) / "bouts" / f"bout_{bout:05d}"
                          / f"fly{f}" / "kp2d.npz") for f in (0, 1)}
    except (FileNotFoundError, OSError):
        return None
    per_cam, wrong, total = {}, 0, 0
    for ci, cam in enumerate(cams):
        kc = {f: kp_centroids(kps[f], ci, thresh) for f in (0, 1)}
        T = min(len(kc[0]), cent.shape[2])
        sep = np.linalg.norm(cent[0, ci, :T] - cent[1, ci, :T], axis=-1)
        base = val[0, ci, :T] & val[1, ci, :T] & (sep > sep_px)
        cw = ct = 0
        for f in (0, 1):
            own = np.linalg.norm(kc[f][:T] - cent[f, ci, :T], axis=-1)
            oth = np.linalg.norm(kc[f][:T] - cent[1 - f, ci, :T], axis=-1)
            # On ONE of the two masks, not stranded between them.
            s = (base & np.isfinite(own) & np.isfinite(oth)
                 & (np.minimum(own, oth) < margin * sep))
            ct += int(s.sum()); cw += int((s & (oth < own)).sum())
        if ct:
            per_cam[cam] = (cw, ct)
        wrong += cw; total += ct
    return {"wrong": wrong, "total": total, "per_cam": per_cam}


def main(argv=None):
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/gscratch/portia/eabe/data/Johnson_lab/processed/courtship")
    ap.add_argument("--pose-dirs", nargs="+", default=["pose", "pose_v3"])
    ap.add_argument("--sep", type=float, default=200.0)
    ap.add_argument("--margin", type=float, default=0.35,
                    help="keypoints must be within margin*sep of the nearer mask")
    ap.add_argument("--min-frames", type=int, default=100)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)

    results = {pd: {} for pd in a.pose_dirs}
    for md in sorted(glob.glob(f"{a.root}/*/*/sam3_masks")):
        rec = os.path.dirname(md)
        sess = os.path.basename(os.path.dirname(rec))
        cal = os.path.join(VIDEO_ROOT, sess, os.path.basename(rec), "calibration")
        if not os.path.isdir(cal):
            continue
        cams = list(ReprojectionTool(cal).cameras.keys())
        for npz in sorted(glob.glob(f"{md}/bout_*/sam3_masks.npz")):
            b = int(os.path.basename(os.path.dirname(npz)).split("_")[1])
            tag = f"{sess}/{os.path.basename(rec)}#{b}"
            for pd in a.pose_dirs:
                r = audit_bout(npz, os.path.join(rec, pd), b, cams, a.sep,
                               margin=a.margin)
                if r and r["total"] >= a.min_frames:
                    results[pd][tag] = r

    summary = {}
    for pd, byb in results.items():
        w = sum(r["wrong"] for r in byb.values())
        t = sum(r["total"] for r in byb.values())
        fr = {k: r["wrong"] / r["total"] for k, r in byb.items()}
        bad = {k: v for k, v in fr.items() if v > 0.10}
        summary[pd] = {"bouts": len(byb), "wrong_frac": w / max(t, 1),
                       "frames": t, "bouts_over_10pct": len(bad),
                       "worst": sorted(fr.items(), key=lambda x: -x[1])[:8]}
        print(f"{pd}:  {len(byb)} bouts scored, {t} fly-frames")
        print(f"   overall wrong-fly: {100*w/max(t,1):.2f}%   "
              f"bouts >10% wrong: {len(bad)}")
        for k, v in summary[pd]["worst"]:
            if v > 0.01:
                print(f"      {k:42s} {100*v:5.1f}%")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {a.json_out}")
    return summary


if __name__ == "__main__":
    main()
