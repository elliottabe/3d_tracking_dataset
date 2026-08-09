#!/usr/bin/env python3
"""Calibrate the per-view confidence gate for triangulation.

The 2D detector emits all 50 keypoint channels for whatever crop it is handed,
including a crop containing no fly, and it reports high per-keypoint peak
confidence while doing so (see docs/benchmark/2026-08-08-detector-v4-retrain.md).
The pipeline's per-keypoint `detector.conf_thresh` therefore does not stop a
misplaced crop from entering triangulation, where it produces a confident but
wrong 3D -- worse than a dropout, because a NaN is visibly missing and a
plausible wrong point is not.

Per-VIEW median confidence separates the two cases much better than any
per-keypoint value. This script measures both distributions on real data so the
threshold is chosen from evidence rather than from one example:

  positives: crops centred on an annotated fly (fly definitely present)
  negatives: crops from the same images at locations that do NOT overlap any
             annotated fly bbox (fly definitely absent)

Reports both distributions, the separation, and the threshold that keeps a
chosen fraction of genuine views (default 99%).

Usage:
    python scripts/calibrate_view_gate.py --ckpt <run>/final --per-subset 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"
for p in (str(PROJECT_DIR), str(PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_SUBSETS = [
    "wall_frames",              # the hard, train-only case
    "courtship_11_50_female",   # held-out female
    "courtship_25_51_female",
    "headless_22_50",           # missing anatomy
    "S8_male_R_amp",            # missing anatomy
    "S6male",                   # easy baseline
    "grooming",
]


def sample_negative_origin(bboxes, img_w, img_h, crop, rng, tries=40):
    """A crop origin whose `crop`x`crop` box overlaps no annotated fly bbox."""
    if img_w <= crop and img_h <= crop:
        return None
    for _ in range(tries):
        x0 = int(rng.integers(0, max(1, img_w - crop)))
        y0 = int(rng.integers(0, max(1, img_h - crop)))
        if all(x0 + crop <= bx or x0 >= bx + bw or
               y0 + crop <= by or y0 >= by + bh
               for (bx, by, bw, bh) in bboxes):
            return x0, y0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=("/gscratch/portia/eabe/data/Johnson_lab/"
                                       "jax_vitpose_runs/v4_8gpu_20260808/final"))
    ap.add_argument("--source-root",
                    default="/gscratch/portia/eabe/data/Johnson_lab/red_data/general_model")
    ap.add_argument("--subsets", nargs="*", default=DEFAULT_SUBSETS)
    ap.add_argument("--per-subset", type=int, default=40,
                    help="max annotated views sampled per subset")
    ap.add_argument("--keep-frac", type=float, default=0.99,
                    help="fraction of genuine views the gate must keep")
    ap.add_argument("--crop", type=int, default=448)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scripts.viz.detector_predictions import (
        load_subset, load_model, predict_crops, load_crop, crop_origin_np)

    rng = np.random.default_rng(args.seed)
    model, _ = load_model(args.ckpt)
    print(f"checkpoint: {args.ckpt}\n")

    pos_all, neg_all = [], []
    print(f"{'subset':<24}{'views':>7}{'pos median':>12}{'neg median':>12}")
    for subset in args.subsets:
        data = load_subset(args.source_root, subset)
        items = [(im, data["anns"][k]) for k, im in data["images"].items()
                 if k in data["anns"]]
        if not items:
            continue
        rng.shuffle(items)
        items = items[:args.per_subset]

        # group annotated bboxes per image file, so negatives avoid every fly
        by_file = {}
        for _k, im in data["images"].items():
            by_file.setdefault(im["file_name"], [])
        for k, ann in data["anns"].items():
            im = data["images"][k]
            by_file[im["file_name"]].append(ann["bbox"])

        pos_crops, neg_crops = [], []
        for im, ann in items:
            w, h = im["width"], im["height"]
            x0, y0 = crop_origin_np(ann["bbox"], w, h, args.crop)
            pos_crops.append(load_crop(args.source_root, subset, im, x0, y0, args.crop))
            org = sample_negative_origin(by_file[im["file_name"]], w, h,
                                         args.crop, rng)
            if org is not None:
                neg_crops.append(load_crop(args.source_root, subset, im,
                                           org[0], org[1], args.crop))

        def medians(crops):
            out = []
            for i in range(0, len(crops), 16):
                _kp, conf = predict_crops(model, np.stack(crops[i:i + 16]))
                out.extend(np.median(conf, axis=1).tolist())
            return np.asarray(out)

        pos = medians(pos_crops) if pos_crops else np.array([])
        neg = medians(neg_crops) if neg_crops else np.array([])
        pos_all.append(pos); neg_all.append(neg)
        print(f"{subset:<24}{len(pos):>7}"
              f"{(np.median(pos) if len(pos) else float('nan')):>12.3f}"
              f"{(np.median(neg) if len(neg) else float('nan')):>12.3f}")

    pos = np.concatenate([p for p in pos_all if len(p)])
    neg = np.concatenate([n for n in neg_all if len(n)])

    print(f"\npositives (fly present): n={len(pos)}")
    for q in (1, 5, 25, 50):
        print(f"    p{q:<3} {np.percentile(pos, q):.3f}")
    print(f"negatives (no fly):      n={len(neg)}")
    for q in (50, 75, 95, 99):
        print(f"    p{q:<3} {np.percentile(neg, q):.3f}")

    thresh = float(np.percentile(pos, 100 * (1 - args.keep_frac)))
    kept = float((pos >= thresh).mean())
    rejected = float((neg < thresh).mean())
    print(f"\nthreshold keeping {100*args.keep_frac:.0f}% of genuine views: "
          f"{thresh:.3f}")
    print(f"    keeps    {100*kept:5.1f}% of positives")
    print(f"    rejects  {100*rejected:5.1f}% of negatives")
    for t in (0.4, 0.5, 0.6, 0.7, 0.8):
        print(f"    t={t:.1f}: keeps {100*(pos>=t).mean():5.1f}% pos, "
              f"rejects {100*(neg<t).mean():5.1f}% neg")


if __name__ == "__main__":
    main()
