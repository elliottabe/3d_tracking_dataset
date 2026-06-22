#!/usr/bin/env python3
"""
Visualize a trained JAX ViTPose KeypointDetect model on COURTSHIP, comparing
the female vs the male fly on held-out validation frames.

Expands scripts/viz_keypoints.py (single recording) to the paired courtship
recordings (female + male) and writes two figures:
  * <out>/viz_courtship_frames.png - example crops, one row per sex, with
    predicted keypoints (red + skeleton) vs GT (green x) and per-frame MPJPE.
  * <out>/viz_courtship_perkp.png  - per-keypoint mean error, female vs male
    (grouped bars, sorted by female-minus-male gap) - shows which landmarks
    (e.g. distal leg tips) are worse on the female.

Errors are in pixels of the 448x448 crop (heatmap coords x 2).

Usage:
    python scripts/viz_courtship_mf.py --run-dir \
        /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v3_8gpu_20260620

    # Custom recordings / frame count / output:
    python scripts/viz_courtship_mf.py --run-dir <RUN> \
        --female-rec 2026_06_15_12_12_34 --male-rec 2026_06_18_19_23_03 \
        --n-frames 5 --out-dir /tmp/myviz
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
DEFAULT_FEMALE_REC = "2026_05_27_11_56_05"   # held-out female-courtship recording
DEFAULT_MALE_REC = "2026_05_27_11_57_05"     # held-out male-courtship recording
SCALE = 448 / 224.0                          # heatmap coords -> 448 crop coords
SEX_COLOR = {"female": "#ff4da6", "male": "#3aa0ff"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', default=None,
                   help='Training run dir; loads <run-dir>/final and writes figures there')
    p.add_argument('--ckpt', default=None,
                   help='Explicit Orbax model checkpoint dir (overrides --run-dir/final)')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--female-rec', default=DEFAULT_FEMALE_REC)
    p.add_argument('--male-rec', default=DEFAULT_MALE_REC)
    p.add_argument('--split', default='val')
    p.add_argument('--n-frames', type=int, default=5,
                   help='Example frames per sex (default: 5)')
    p.add_argument('--batch', type=int, default=8)
    args = p.parse_args()

    if args.ckpt:
        ckpt = args.ckpt
    elif args.run_dir:
        ckpt = str(Path(args.run_dir) / "final")
    else:
        p.error("provide --run-dir or --ckpt")
    out_dir = Path(args.out_dir or args.run_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.convert.build_checkpoint import load_vitpose
    from jarvis_jax.data.v3 import V3Dataset, batches
    from jarvis_jax.data.device import normalize_image
    from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints

    meta = json.load(open(f"{args.data_root}/annotations/instances_{args.split}.json"))
    names = meta["keypoint_names"]
    nidx = {n: i for i, n in enumerate(names)}
    edges = [(nidx[e["keypointA"]], nidx[e["keypointB"]]) for e in meta["skeleton"]
             if e["keypointA"] in nidx and e["keypointB"] in nidx]
    K = len(names)

    cfg = ViTPoseConfig()
    print(f"loading model from {ckpt} ...", flush=True)
    model = load_vitpose(ckpt, cfg)
    model.eval()

    def predict(img4_u8):
        img = normalize_image(jnp.asarray(img4_u8)[None])
        return np.asarray(heatmaps_to_keypoints(model(img, use_running_average=True)))[0]

    sexes = [("female", args.female_rec), ("male", args.male_rec)]
    per_kp = {}     # sex -> (K,) mean error
    overall = {}    # sex -> float
    datasets = {}   # sex -> V3Dataset
    for sex, rec in sexes:
        ds = V3Dataset(args.data_root, args.split, recordings=[rec])
        if len(ds) == 0:
            p.error(f"no {args.split} frames for {sex} recording {rec!r}")
        datasets[sex] = ds
        print(f"{sex}: {rec}  {len(ds)} {args.split} frames", flush=True)
        err_sum = np.zeros(K)
        err_cnt = np.zeros(K)
        for img4_u8, kp_xy, vis in batches(ds, args.batch, shuffle=False, drop_last=False):
            img = normalize_image(jnp.asarray(img4_u8))
            pk = np.asarray(heatmaps_to_keypoints(model(img, use_running_average=True)))
            gk = np.asarray(kp_xy) * SCALE
            d = np.linalg.norm(pk - gk, axis=-1)
            m = np.asarray(vis)
            err_sum += (d * m).sum(0)
            err_cnt += m.sum(0)
        per_kp[sex] = np.where(err_cnt > 0, err_sum / np.maximum(err_cnt, 1), np.nan)
        overall[sex] = float(np.nansum(err_sum) / max(np.nansum(err_cnt), 1))
        print(f"  {sex} overall MPJPE: {overall[sex]:.2f}px", flush=True)

    # ---- Figure 1: example frames, one row per sex ----
    n = args.n_frames
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 9.4), squeeze=False)
    for r, (sex, rec) in enumerate(sexes):
        ds = datasets[sex]
        idxs = np.unique(np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int))
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(idxs):
                continue
            img4_u8, kp_xy, vis = ds[int(idxs[c])]
            pk = predict(img4_u8)
            gk = kp_xy * SCALE
            ax.imshow(img4_u8[:, :, :3])
            for a, b in edges:
                if vis[a] and vis[b]:
                    ax.plot([pk[a, 0], pk[b, 0]], [pk[a, 1], pk[b, 1]], "-",
                            color="red", lw=0.9, alpha=0.6)
            ax.scatter(pk[vis, 0], pk[vis, 1], c="red", s=14, zorder=3)
            ax.scatter(gk[vis, 0], gk[vis, 1], c="lime", s=14, marker="x", zorder=4)
            fe = (np.linalg.norm(pk[vis] - gk[vis], axis=-1).mean()
                  if vis.any() else float("nan"))
            ax.set_title(f"{sex} frame {idxs[c]}\nMPJPE {fe:.1f}px", fontsize=9,
                         color=SEX_COLOR[sex])
        axes[r, 0].text(-0.08, 0.5, f"{sex.upper()}\n{overall[sex]:.1f}px",
                        transform=axes[r, 0].transAxes, rotation=90, va="center",
                        ha="center", fontsize=12, color=SEX_COLOR[sex], weight="bold")
    fig.suptitle("JAX ViTPose on courtship - female vs male (val)\n"
                 "red = pred (+skeleton)   green x = GT", fontsize=14)
    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    f1 = out_dir / "viz_courtship_frames.png"
    plt.savefig(f1, dpi=110, bbox_inches="tight")
    print(f"SAVED {f1}", flush=True)

    # ---- Figure 2: per-keypoint female vs male (sorted by gap) ----
    pf, pm = per_kp["female"], per_kp["male"]
    gap = np.nan_to_num(pf, nan=0) - np.nan_to_num(pm, nan=0)
    order = np.argsort(gap)                       # worst-for-female at bottom
    yy = np.arange(K)
    fig2, ax2 = plt.subplots(figsize=(11, max(6, 0.30 * K)))
    ax2.barh(yy - 0.2, pf[order], height=0.4, color=SEX_COLOR["female"], label="female")
    ax2.barh(yy + 0.2, pm[order], height=0.4, color=SEX_COLOR["male"], label="male")
    ax2.set_yticks(yy)
    ax2.set_yticklabels([names[i] for i in order], fontsize=7)
    ax2.axvline(overall["female"], color=SEX_COLOR["female"], ls="--", lw=1)
    ax2.axvline(overall["male"], color=SEX_COLOR["male"], ls="--", lw=1)
    ax2.set_xlabel("mean error (px in 448 crop)")
    ax2.set_ylim(-1, K)
    ax2.set_title("Per-keypoint mean error: female vs male courtship\n"
                  f"(sorted by female-minus-male gap; dashed = overall  "
                  f"F {overall['female']:.1f} / M {overall['male']:.1f}px)")
    ax2.legend(loc="lower right")
    plt.tight_layout()
    f2 = out_dir / "viz_courtship_perkp.png"
    plt.savefig(f2, dpi=110, bbox_inches="tight")
    print(f"SAVED {f2}", flush=True)

    worst_f = [(names[i], round(float(pf[i]), 1)) for i in np.argsort(gap)[::-1][:6]]
    print(f"overall  female {overall['female']:.2f}px  |  male {overall['male']:.2f}px")
    print(f"biggest female-worse keypoints (F px, M px): "
          f"{[(n, round(float(pf[nidx[n]]),1), round(float(pm[nidx[n]]),1)) for n,_ in worst_f]}")


if __name__ == "__main__":
    main()
