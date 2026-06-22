#!/usr/bin/env python3
"""
Visualize a trained JAX ViTPose KeypointDetect model: predicted vs ground-truth
keypoints on validation frames, plus a per-keypoint mean-error breakdown.

Loads the model from a run's `final/` Orbax checkpoint (or an explicit --ckpt),
runs it over one recording's validation frames, and writes two figures:
  * <out>/viz_frames.png  - N example frames, pred (red + skeleton) vs GT (green x),
    per-frame MPJPE in the title.
  * <out>/viz_perkp.png   - bar chart of mean error per keypoint (sorted), which
    shows exactly which landmarks are accurate vs hard (e.g. tarsal tips).

Errors are in pixels of the 448x448 crop (heatmap coords x 2). The model loads
on whatever GPU count is available (load_vitpose is topology-agnostic).

Usage:
    python scripts/viz_keypoints.py --run-dir \
        /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v3_8gpu_20260620

    # A different recording / more frames / custom output dir:
    python scripts/viz_keypoints.py --run-dir <RUN> --recording <rec> \
        --n-frames 12 --out-dir /tmp/myviz
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
DEFAULT_RECORDING = "2026_05_27_11_56_05"   # held-out female-courtship recording
SCALE = 448 / 224.0                          # heatmap coords -> 448 crop coords


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', default=None,
                   help='Training run dir; loads <run-dir>/final and writes figures there')
    p.add_argument('--ckpt', default=None,
                   help='Explicit Orbax model checkpoint dir (overrides --run-dir/final)')
    p.add_argument('--out-dir', default=None,
                   help='Where to write figures (default: --run-dir, else cwd)')
    p.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--recording', default=DEFAULT_RECORDING,
                   help='Validation recording name to evaluate/visualize')
    p.add_argument('--split', default='val', help='Dataset split (default: val)')
    p.add_argument('--n-frames', type=int, default=8,
                   help='Number of example frames in the frames figure (default: 8)')
    p.add_argument('--batch', type=int, default=8,
                   help='Eval batch size for the per-keypoint sweep (default: 8)')
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
    ds = V3Dataset(args.data_root, args.split, recordings=[args.recording])
    if len(ds) == 0:
        p.error(f"no {args.split} frames for recording {args.recording!r}")
    print(f"{args.recording}: {len(ds)} {args.split} frames", flush=True)

    def predict(img4_u8):
        img = normalize_image(jnp.asarray(img4_u8)[None])
        return np.asarray(heatmaps_to_keypoints(model(img, use_running_average=True)))[0]

    # ---- per-keypoint error over the whole recording ----
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
    per_kp = np.where(err_cnt > 0, err_sum / np.maximum(err_cnt, 1), np.nan)
    overall = float(np.nansum(err_sum) / max(np.nansum(err_cnt), 1))
    print(f"overall MPJPE: {overall:.2f}px", flush=True)

    # ---- Figure 1: example frames ----
    n = min(args.n_frames, len(ds))
    idxs = np.unique(np.linspace(0, len(ds) - 1, n).astype(int))
    cols = min(4, len(idxs))
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.6 * rows), squeeze=False)
    for ax, ix in zip(axes.ravel(), idxs):
        img4_u8, kp_xy, vis = ds[int(ix)]
        pk = predict(img4_u8)
        gk = kp_xy * SCALE
        ax.imshow(img4_u8[:, :, :3])
        for a, b in edges:
            if vis[a] and vis[b]:
                ax.plot([pk[a, 0], pk[b, 0]], [pk[a, 1], pk[b, 1]], "-",
                        color="red", lw=0.9, alpha=0.6)
        ax.scatter(pk[vis, 0], pk[vis, 1], c="red", s=16, label="pred", zorder=3)
        ax.scatter(gk[vis, 0], gk[vis, 1], c="lime", s=16, marker="x", label="GT", zorder=4)
        fe = (np.linalg.norm(pk[vis] - gk[vis], axis=-1).mean()
              if vis.any() else float("nan"))
        ax.set_title(f"frame {ix}  MPJPE {fe:.1f}px", fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[len(idxs):]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="upper right", fontsize=9)
    fig.suptitle(f"JAX ViTPose - {args.recording} - overall MPJPE {overall:.1f}px\n"
                 f"red=pred(+skeleton)  green x=GT", fontsize=13)
    plt.tight_layout()
    f1 = out_dir / "viz_frames.png"
    plt.savefig(f1, dpi=110, bbox_inches="tight")
    print(f"SAVED {f1}", flush=True)

    # ---- Figure 2: per-keypoint mean error (sorted) ----
    order = np.argsort(np.nan_to_num(per_kp, nan=-1))
    vals = per_kp[order]
    fig2, ax2 = plt.subplots(figsize=(11, max(6, 0.24 * K)))
    yy = np.arange(K)
    ax2.barh(yy, vals, color=plt.cm.RdYlGn_r(np.clip(vals / 60.0, 0, 1)))
    ax2.set_yticks(yy)
    ax2.set_yticklabels([names[i] for i in order], fontsize=7)
    ax2.axvline(overall, color="k", ls="--", lw=1, label=f"overall {overall:.1f}px")
    ax2.set_xlabel("mean error (px in 448 crop)")
    ax2.set_ylim(-1, K)
    ax2.set_title(f"Per-keypoint mean error - {args.recording} (sorted)")
    ax2.legend()
    plt.tight_layout()
    f2 = out_dir / "viz_perkp.png"
    plt.savefig(f2, dpi=110, bbox_inches="tight")
    print(f"SAVED {f2}", flush=True)

    worst = [(names[i], round(float(per_kp[i]), 1)) for i in order[::-1][:5]]
    best = [(names[i], round(float(per_kp[i]), 1)) for i in order[:5]]
    print(f"worst 5: {worst}")
    print(f"best 5:  {best}")


if __name__ == "__main__":
    main()
