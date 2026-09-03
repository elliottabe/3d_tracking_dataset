#!/usr/bin/env python3
"""Show the distractor-aware training stream on REAL v12 crops before training on it.

Three changes land together (jarvis_jax/data/distractor.py, copy_paste.py,
train/losses.py) and each is visible on a crop. EXPECTATIONS, per column:

  raw          white = target keypoints (leg chains drawn), ORANGE = the other
               fly's keypoints appended as rows [K:]. Orange must sit on the
               OTHER fly, white on the target; a two-fly row shows both, a
               single-fly row shows only white.
  gray-fill    the other fly's BODY is flat crop-mean grey (SAM mask dilated
               15 px, target protected 60 px); its LEGS remain visible; the
               target is intact. Single-fly rows are unchanged.
  repulsion    the TaTip-channel footprint: yellow blobs on every one of the
               OTHER fly's six tarsal tips (all "TaTip" = one part), NOTHING on
               the target's own tips, nothing on a single-fly row. It is what
               the loss charges the target's tip channels for firing on.
  copy-paste   single-fly rows only: a same-camera donor fly composited
               40-300 px from the host at the same scale/lighting with a soft
               edge; its orange keypoints on its own legs/wings; the host's
               white keypoints unmoved. Two-fly rows: "n/a".

Rows: 3 real two-fly train crops (closest pairs first) + 3 single-fly crops
(one female, one wall/climbing, one male). Reads only the dataset root; no GPU.

    python scripts/viz/distractor_aug_preview.py \
        --out figures/2026-09-03-distractor-aug
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO), str(_REPO / "third_party" / "jarvis_jax")):
    sys.path.insert(0, p)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp

from jarvis_jax.data.v5_2d import V5Dataset
from jarvis_jax.data.distractor import (DistractorKeypointDataset, DistractorGrayFillDataset,
                                        build_part_index, repulsion_footprint)
from jarvis_jax.data.copy_paste import CopyPasteKeypointDataset
from viz.core.colors import leg_chains


def chains(ax, kp, vis, names, color, ls="-"):
    for chain in leg_chains(names).values():
        pts = np.asarray([kp[k] for k in chain if vis[k]])
        if len(pts) >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=0.9, ls=ls)
    v = np.asarray(vis, bool)
    ax.plot(kp[v, 0], kp[v, 1], ".", color=color, ms=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902")
    ap.add_argument("--out", default="figures/2026-09-03-distractor-aug")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = json.load(open(os.path.join(a.root, "annotations", "keypoint_names.json")))
    K = len(names); part_of_k, parts = build_part_index(names)
    tip_ch = names.index("T1L_TaTip")

    base = V5Dataset(a.root, "train")
    dk = DistractorKeypointDataset(base)
    fill = DistractorGrayFillDataset(dk, a.root, p=1.0, seed=a.seed)
    cp = CopyPasteKeypointDataset(dk, root=a.root, p=1.0, seed=a.seed)

    # rows: closest 3 two-fly pairs (by bbox centre distance) + 3 singles
    two = [i for i in range(len(dk)) if dk.has_distractor(i)]
    def sep(i):
        j = dk._others[i][0]
        c = lambda b: np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])
        return np.linalg.norm(c(base.bboxes[i]) - c(base.bboxes[j]))
    two = sorted(two, key=sep)[:3]
    rng = np.random.default_rng(a.seed)
    sexes = np.asarray(base.sex); beh = np.asarray(base.behavior)
    singles = [i for i in range(len(dk)) if not dk.has_distractor(i)]
    pick = lambda m: int(rng.choice([i for i in singles if m(i)]))
    single = [pick(lambda i: sexes[i] == "female"), pick(lambda i: beh[i] in ("wall", "climbing")),
              pick(lambda i: sexes[i] == "male")]
    rows = [(i, "two-fly") for i in two] + [(i, "single") for i in single]

    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.7 * len(rows)))
    for r, (i, kind) in enumerate(rows):
        img4, kp, vis = dk[i]
        rgb = img4[..., :3]; kpx = kp * 2                        # heatmap px -> crop px
        ax = axes[r, 0]; ax.imshow(rgb)
        chains(ax, kpx[:K], vis[:K], names, "white"); chains(ax, kpx[K:], vis[K:], names, "orange", "--")
        ax.set_title(f"{kind}: {base.file_names[i]}\nann {base.ann_ids[i]} {sexes[i]} {beh[i]}"
                     + (f"  sep {sep(i):.0f}px" if kind == "two-fly" else ""), fontsize=7)
        ax.set_ylabel("raw", fontsize=9)
        # gray-fill
        f = fill.fill(img4, i)
        ax = axes[r, 1]; ax.imshow((f if f is not None else img4)[..., :3])
        chains(ax, kpx[:K], vis[:K], names, "white"); chains(ax, kpx[K:], vis[K:], names, "orange", "--")
        ax.set_title("gray-fill" + ("" if f is not None else " (no distractor mask -> unchanged)"), fontsize=8)
        # repulsion footprint for the TaTip channel
        fp = np.asarray(repulsion_footprint(jnp.asarray(kp[None, K:]), jnp.asarray(vis[None, K:]),
                                            part_of_k, heatmap_size=224, sigma=7.0))[0, ..., tip_ch]
        ax = axes[r, 2]; ax.imshow(rgb, alpha=0.6)
        ax.imshow(fp, extent=(0, 448, 448, 0), cmap="viridis", alpha=0.7, vmin=0, vmax=1)
        chains(ax, kpx[:K], vis[:K], names, "white")
        ax.set_title(f"repulsion footprint, {names[tip_ch]} channel\nmax {fp.max():.2f} (0 = nothing to repel)", fontsize=8)
        # copy-paste
        ax = axes[r, 3]
        if kind == "single":
            img_cp, kp_cp, vis_cp, meta = cp.sample_with_meta(i)
            ax.imshow(img_cp[..., :3]); kc = kp_cp * 2
            chains(ax, kc[:K], vis_cp[:K], names, "white"); chains(ax, kc[K:], vis_cp[K:], names, "orange", "--")
            ax.set_title(f"copy-paste: donor {meta['donor_file_name'].split('/')[0]}  sep {meta['sep_achieved_px']:.0f}px  "
                         f"{'donor on top' if meta['top_is_donor'] else 'host on top'}", fontsize=7)
        else:
            ax.imshow(np.zeros_like(rgb)); ax.set_title("copy-paste: n/a (real two-fly)", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Distractor-aware training stream on real v12 train crops. white = target keypoints, orange dashed = "
                 "the OTHER fly's keypoints (rows [K:]); expectations in the script docstring", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "distractor_aug_preview.png", dpi=110)
    json.dump({"rows": [{"idx": int(i), "kind": k, "file": base.file_names[i]} for i, k in rows],
               "parts": parts}, open(out / "distractor_aug_preview.json", "w"), indent=1)
    print("wrote", out / "distractor_aug_preview.png")


if __name__ == "__main__":
    main()
