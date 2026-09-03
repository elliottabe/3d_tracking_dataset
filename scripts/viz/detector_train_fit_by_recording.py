#!/usr/bin/env python3
"""Pre-flight: which TRAIN recordings could v12_bal_maskoff not fit?

v12's final train loss (0.0069) is 1.85x v5's (0.0037) on the same recipe, so
part of the new root is being fitted poorly. Evaluating the finished checkpoint
on a sample of its OWN training split, per recording, separates two readings:
  * hard-but-consistent content: uniformly moderate train error, tips worst,
    left/right symmetric;
  * label defects in one export (L/R swapped, wrong camera, wrong fly): ONE or
    a few recordings with train error far above the rest, and/or a strong
    left-vs-right asymmetry that no flip augmentation can produce.

EXPECTATION: if labels are sound, per-recording train MPJPE is a few px
everywhere (the net has seen every one of these frames ~1.5-5x) and the
L-minus-R tarsal-tip error is ~0. A recording at 10+ px train error, or a L-R
gap of several px concentrated in one recording, is a label problem to open.

Channel 3 zeroed (mask-ablation checkpoint). Outputs to
figures/2026-09-03-maskoff-attention/train_fit_by_recording.{png,json}.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO), str(_REPO / "third_party" / "jarvis_jax")):
    sys.path.insert(0, p)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.data.v5_2d import V5Dataset
from jarvis_jax.data.mask_zero import ZeroMaskDataset
from jarvis_jax.data.device import normalize_image
from jarvis_jax.scripts.eval_keypoints_2d import restore_model
from jarvis_jax.tracking.predict_2d import verify_detector_kp_order
from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints
from jarvis_jax.train.train import _eval_forward


class Subset:
    def __init__(self, ds, idx):
        self._ds, self.idx = ds, list(idx)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i): return self._ds[self.idx[i]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902")
    ap.add_argument("--ckpt", default="/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v12_bal_maskoff/final")
    ap.add_argument("--per-rec", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="figures/2026-09-03-maskoff-attention")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = json.load(open(os.path.join(a.root, "annotations", "keypoint_names.json")))
    assert verify_detector_kp_order(a.ckpt, names, strict=True) is not None
    nidx = {n: i for i, n in enumerate(names)}
    tipsL = [nidx[f"{l}_TaTip"] for l in ("T1L", "T2L", "T3L")]
    tipsR = [nidx[f"{l}_TaTip"] for l in ("T1R", "T2R", "T3R")]
    manifest = json.load(open(os.path.join(a.root, "manifest.json")))["recordings"]

    model = restore_model(a.ckpt, "vitpose", ViTPoseConfig()); model.eval()
    rng = np.random.default_rng(0)
    results = {}
    for split in ("train", "val"):
        raw = V5Dataset(a.root, split); ds = ZeroMaskDataset(raw)
        recs = np.asarray([f.split("/")[0] for f in raw.file_names])
        for rec in sorted(set(recs)):
            idx = np.where(recs == rec)[0]
            idx = rng.choice(idx, min(a.per_rec, len(idx)), replace=False)
            errs = []
            for s in range(0, len(idx), a.batch):
                chunk = [ds[int(i)] for i in idx[s:s + a.batch]]
                img = jnp.asarray(np.stack([c[0] for c in chunk]))
                kp = np.stack([c[1] for c in chunk]) * 2.0          # heatmap px -> crop px
                vis = np.stack([c[2] for c in chunk]) > 0
                pk = np.asarray(heatmaps_to_keypoints(_eval_forward(model, normalize_image(img)), in_size=448))
                e = np.linalg.norm(pk - kp, axis=-1); e[~vis] = np.nan
                errs.append(e)
            e = np.concatenate(errs)
            m = manifest.get(rec, {})
            results[f"{split}/{rec}"] = {
                "split": split, "recording": rec, "n": int(len(idx)), "n_total": int((recs == rec).sum()),
                "subset": m.get("subset", "?"), "sex": m.get("sex", "?"), "behavior": m.get("behavior", "?"),
                "mpjpe": float(np.nanmean(e)), "p90": float(np.nanpercentile(e, 90)),
                "tips_L": float(np.nanmean(e[:, tipsL])), "tips_R": float(np.nanmean(e[:, tipsR])),
                "per_kp": [float(x) for x in np.nanmean(e, axis=0)],
            }
            r = results[f"{split}/{rec}"]
            print(f"{split:5s} {rec} {r['subset'][:28]:28s} n={r['n']:3d}/{r['n_total']:5d} "
                  f"mpjpe {r['mpjpe']:6.2f} p90 {r['p90']:6.2f} tipsL {r['tips_L']:6.2f} tipsR {r['tips_R']:6.2f}")
    json.dump({"ckpt": a.ckpt, "root": a.root, "kp_names": names, "results": results},
              open(out / "train_fit_by_recording.json", "w"), indent=1)

    keys = sorted(results, key=lambda k: (results[k]["split"], -results[k]["mpjpe"]))
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.42 * len(keys) + 2), gridspec_kw={"width_ratios": [2, 1]})
    y = np.arange(len(keys))
    cols = ["C0" if results[k]["split"] == "train" else "C3" for k in keys]
    axes[0].barh(y, [results[k]["mpjpe"] for k in keys], color=cols, alpha=0.85)
    axes[0].barh(y, [results[k]["p90"] for k in keys], color=cols, alpha=0.25)
    axes[0].set_yticks(y); axes[0].set_yticklabels(
        [f"{results[k]['recording']}  {results[k]['subset'][:30]}  ({results[k]['sex']}, n={results[k]['n']})" for k in keys], fontsize=7)
    axes[0].invert_yaxis(); axes[0].set_xlabel("MPJPE px (dark) and p90 (light), channel 3 zeroed")
    axes[0].set_title("v12_bal_maskoff on its OWN split, per recording. blue = TRAIN (should all be low), red = val", fontsize=9)
    axes[0].grid(axis="x", alpha=0.3)
    d = [results[k]["tips_L"] - results[k]["tips_R"] for k in keys]
    axes[1].barh(y, d, color=cols, alpha=0.85); axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_yticks(y); axes[1].set_yticklabels([]); axes[1].invert_yaxis()
    axes[1].set_xlabel("tarsal-tip MPJPE, LEFT minus RIGHT (px)")
    axes[1].set_title("L-R asymmetry per recording (expect ~0 if labels are consistent)", fontsize=9)
    axes[1].grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "train_fit_by_recording.png", dpi=130)
    print("wrote", out / "train_fit_by_recording.png")


if __name__ == "__main__":
    main()
