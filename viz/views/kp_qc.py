"""kp-qc view: ViTPose pred-vs-GT keypoints + per-keypoint error QC.

Ported verbatim (model-eval + matplotlib logic) from two reference scripts:
  * scripts/viz_keypoints.py    -- single recording, pred-vs-GT + per-kp bars.
  * scripts/viz_courtship_mf.py -- paired female-vs-male recordings.

`run(args)` dispatches on `args.female_vs_male`: False runs the single-
recording layout (_run_single, port of viz_keypoints.main), True runs the
paired layout (_run_paired, port of viz_courtship_mf.main). Only the
argparse-driven `main()`/CLI wrapper was dropped; the model load, per-keypoint
error sweep, and figure code are unchanged from the source scripts.

Like the reference scripts, all heavy imports (matplotlib, jax, jarvis_jax)
stay LAZY inside the run functions below -- module level here is stdlib +
numpy only (+ the third_party/jarvis_jax sys.path insert), so
`import viz.views.kp_qc` stays cheap even under JAX_PLATFORMS=cpu.
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"
sys.path.insert(0, str(PKG_DIR))

DEFAULT_DATA_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
DEFAULT_RECORDING = "2026_05_27_11_56_05"    # held-out female-courtship recording
DEFAULT_FEMALE_REC = "2026_05_27_11_56_05"   # held-out female-courtship recording
DEFAULT_MALE_REC = "2026_05_27_11_57_05"     # held-out male-courtship recording
SCALE = 448 / 224.0                          # heatmap coords -> 448 crop coords
SEX_COLOR = {"female": "#ff4da6", "male": "#3aa0ff"}
_SPLIT = "val"     # QC-internal (not exposed on the CLI); matches script defaults
_BATCH = 8         # QC-internal (not exposed on the CLI); matches script defaults


def _resolve_ckpt(args):
    ckpt = getattr(args, "ckpt", None)
    run_dir = getattr(args, "run_dir", None)
    if ckpt:
        return ckpt
    if run_dir:
        return str(Path(run_dir) / "final")
    raise ValueError("kp-qc requires --run-dir or --ckpt (checkpoint dir to load)")


def _resolve_data_root(args):
    data_root = getattr(args, "data_root", None) or DEFAULT_DATA_ROOT
    if not Path(data_root).is_dir():
        raise ValueError(f"kp-qc --data-root not found: {data_root}")
    return data_root


def run(args) -> int:
    if getattr(args, "female_vs_male", False):
        return _run_paired(args)
    return _run_single(args)


def _run_single(args) -> int:
    """Port of scripts/viz_keypoints.py main(): single-recording pred-vs-GT
    keypoints + per-keypoint error bars."""
    ckpt = _resolve_ckpt(args)
    data_root = _resolve_data_root(args)
    recording = getattr(args, "recording", None) or DEFAULT_RECORDING
    n_frames = getattr(args, "n", 8) or 8
    out_dir = Path(getattr(args, "out", None) or getattr(args, "run_dir", None) or ".")
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

    meta = json.load(open(f"{data_root}/annotations/instances_{_SPLIT}.json"))
    names = meta["keypoint_names"]
    nidx = {n: i for i, n in enumerate(names)}
    edges = [(nidx[e["keypointA"]], nidx[e["keypointB"]]) for e in meta["skeleton"]
             if e["keypointA"] in nidx and e["keypointB"] in nidx]
    K = len(names)

    cfg = ViTPoseConfig()
    print(f"loading model from {ckpt} ...", flush=True)
    model = load_vitpose(ckpt, cfg)
    model.eval()
    ds = V3Dataset(data_root, _SPLIT, recordings=[recording])
    if len(ds) == 0:
        raise ValueError(f"no {_SPLIT} frames for recording {recording!r}")
    print(f"{recording}: {len(ds)} {_SPLIT} frames", flush=True)

    def predict(img4_u8):
        img = normalize_image(jnp.asarray(img4_u8)[None])
        return np.asarray(heatmaps_to_keypoints(model(img, use_running_average=True)))[0]

    # ---- per-keypoint error over the whole recording ----
    err_sum = np.zeros(K)
    err_cnt = np.zeros(K)
    for img4_u8, kp_xy, vis in batches(ds, _BATCH, shuffle=False, drop_last=False):
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
    n = min(n_frames, len(ds))
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
    fig.suptitle(f"JAX ViTPose - {recording} - overall MPJPE {overall:.1f}px\n"
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
    ax2.set_title(f"Per-keypoint mean error - {recording} (sorted)")
    ax2.legend()
    plt.tight_layout()
    f2 = out_dir / "viz_perkp.png"
    plt.savefig(f2, dpi=110, bbox_inches="tight")
    print(f"SAVED {f2}", flush=True)

    worst = [(names[i], round(float(per_kp[i]), 1)) for i in order[::-1][:5]]
    best = [(names[i], round(float(per_kp[i]), 1)) for i in order[:5]]
    print(f"worst 5: {worst}")
    print(f"best 5:  {best}")
    return 0


def _run_paired(args) -> int:
    """Port of scripts/viz_courtship_mf.py main(): female-vs-male paired
    recordings, comparing per-keypoint error and example frames per sex."""
    ckpt = _resolve_ckpt(args)
    data_root = _resolve_data_root(args)
    female_rec = getattr(args, "female_rec", None) or DEFAULT_FEMALE_REC
    male_rec = getattr(args, "male_rec", None) or DEFAULT_MALE_REC
    n_frames = getattr(args, "n", 5) or 5
    out_dir = Path(getattr(args, "out", None) or getattr(args, "run_dir", None) or ".")
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

    meta = json.load(open(f"{data_root}/annotations/instances_{_SPLIT}.json"))
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

    sexes = [("female", female_rec), ("male", male_rec)]
    per_kp = {}     # sex -> (K,) mean error
    overall = {}    # sex -> float
    datasets = {}   # sex -> V3Dataset
    for sex, rec in sexes:
        ds = V3Dataset(data_root, _SPLIT, recordings=[rec])
        if len(ds) == 0:
            raise ValueError(f"no {_SPLIT} frames for {sex} recording {rec!r}")
        datasets[sex] = ds
        print(f"{sex}: {rec}  {len(ds)} {_SPLIT} frames", flush=True)
        err_sum = np.zeros(K)
        err_cnt = np.zeros(K)
        for img4_u8, kp_xy, vis in batches(ds, _BATCH, shuffle=False, drop_last=False):
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
    n = n_frames
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
    return 0
