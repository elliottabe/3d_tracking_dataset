#!/usr/bin/env python3
"""Per-keypoint val MPJPE breakdown for a trained JAX ViTPose run.

Loads <run-dir>/final, runs over the FULL val split, and prints each keypoint's
mean MPJPE (px on the 448 crop) sorted worst-first, plus the overall mean and a
best-vs-worst summary. Tells you whether a high overall MPJPE is uniform or
dominated by a few keypoints (e.g. distal leg tips).

Usage:
    python scripts/diag_vit_perkp.py --run-dir \
        /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/myrun \
        --out /tmp/perkp_myrun.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"
sys.path.insert(0, str(PKG_DIR))

DEFAULT_DATA_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="ViTPose run dir (loads <run-dir>/final)")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--recordings", default=None,
                    help="comma-separated recording names to restrict to (default: all val)")
    ap.add_argument("--out", default=None, help="optional CSV path for the per-keypoint table")
    args = ap.parse_args()

    import jax.numpy as jnp
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.convert.build_checkpoint import load_vitpose
    from jarvis_jax.data.v3 import V3Dataset, batches
    from jarvis_jax.data.device import normalize_image
    from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints

    meta = json.load(open(f"{args.data_root}/annotations/instances_{args.split}.json"))
    names = meta["keypoint_names"]
    K = len(names)
    recs = [r for r in args.recordings.split(",") if r] if args.recordings else None
    ds = V3Dataset(args.data_root, args.split, recordings=recs)
    scale = 448 / float(ds.heatmap_size)
    print(f"{args.split}: {len(ds)} annotations, {K} keypoints"
          f"{f' (recordings={recs})' if recs else ''}", flush=True)

    cfg = ViTPoseConfig()
    ckpt = str(Path(args.run_dir) / "final")
    print(f"loading {ckpt} ...", flush=True)
    model = load_vitpose(ckpt, cfg)
    model.eval()

    err_sum = np.zeros(K)
    err_cnt = np.zeros(K)
    for img4_u8, kp_xy, vis in batches(ds, args.batch, shuffle=False, drop_last=False):
        img = normalize_image(jnp.asarray(img4_u8))
        pk = np.asarray(heatmaps_to_keypoints(model(img, use_running_average=True)))
        gk = np.asarray(kp_xy) * scale
        d = np.linalg.norm(pk - gk, axis=-1)              # (B,K) px on 448 crop
        m = np.asarray(vis).astype(float)
        err_sum += (d * m).sum(0)
        err_cnt += m.sum(0)

    per_kp = np.where(err_cnt > 0, err_sum / np.maximum(err_cnt, 1), np.nan)
    overall = float(np.nansum(err_sum) / max(np.nansum(err_cnt), 1))
    med = float(np.nanmedian(per_kp))
    order = np.argsort(np.nan_to_num(per_kp, nan=-1.0))[::-1]   # worst first

    print("\n=== per-keypoint val MPJPE (px on 448 crop), worst first ===", flush=True)
    for i in order:
        print(f"  {names[i]:<24} {per_kp[i]:6.1f}px   (n={int(err_cnt[i])})")

    finite = per_kp[np.isfinite(per_kp)]
    print(f"\noverall mean MPJPE : {overall:.2f}px")
    print(f"median over kps    : {med:.2f}px")
    print(f"best 5 kp mean     : {np.sort(finite)[:5].mean():.1f}px")
    print(f"worst 10 kp mean   : {np.sort(finite)[-10:].mean():.1f}px")
    # If the mean is dominated by a tail, the median << mean.
    print(f"\nInterpretation: median<<mean => a few keypoints dominate; "
          f"median≈mean => error is uniform.", flush=True)

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["keypoint", "mpjpe_px", "n_visible"])
            for i in range(K):
                w.writerow([names[i], round(float(per_kp[i]), 2), int(err_cnt[i])])
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
