#!/usr/bin/env python3
"""Hard-case overlay: mask-populated vs mask-zeroed 2D predictions on real
frames, for the mask-channel-ablation task
(.superpowers/sdd/2026-08-29-coarse-to-fine-3d/mask-channel-ablation.md).

Reads a mask_channel_eval.py .npz (already has pred_populated/pred_zeroed/gt
per annotation -- no GPU needed here) and picks:
  * the FEMALE-recording annotation(s) whose MPJPE gets WORST under mask
    zeroing (the hard case CLAUDE.md asks for -- occlusion/wall/OOD-prone),
  * the female-recording annotation with the SMALLEST (or most negative)
    degradation, for contrast -- not an independent "well separated" split
    (this val set has no per-frame proximity signal, see
    mask_channel_report.py's module docstring), just the least-affected frame.
Reconstructs each one's exact 448x448 RGB crop from the on-disk image (same
bbox + crop_origin the dataset loader uses) and draws GT (white), populated
prediction (cyan) and zeroed prediction (orange), coloured by
viz/core/colors.py body group via marker shape, side by side.

STATED EXPECTATION (write this before looking, per CLAUDE.md): if the mask
channel matters most on hard/occluded frames, the zeroed markers should
visibly drift off the true anatomy (especially legs/wings) on the worst-case
frame while staying close to GT on the well-separated one; if the mask
contributes little even here, zeroed and populated should overlap almost
exactly on BOTH frames.

Usage:
    python scripts/analysis/mask_channel_overlay.py \\
        --npz figures/2026-08-31-mask-ablation/v5_s70_bal_augdef_full.npz \\
        --data-root /gscratch/.../red_data_3d_v5 \\
        --out-dir figures/2026-08-31-mask-ablation
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"
for p in (str(PROJECT_DIR), str(PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def reconstruct_crop(data_root, file_name, bbox, crop=448):
    import numpy as np
    from PIL import Image
    from jarvis_jax.data.transforms import crop_origin

    img_path = os.path.join(data_root, "images", file_name)
    with Image.open(img_path) as pil:
        img = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    img_h, img_w = img.shape[:2]
    x0, y0 = crop_origin(bbox, img_w, img_h, crop)
    return img[y0:y0 + crop, x0:x0 + crop], (x0, y0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-hard", type=int, default=2)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from viz.core.colors import keypoint_groups

    z = np.load(a.npz, allow_pickle=True)
    kp_names = [str(x) for x in z["kp_names"]]
    val_recording = str(z["val_recording"])
    groups = keypoint_groups(kp_names)
    idx2group = {i: g for g, idxs in groups.items() for i in idxs}
    group_colors = {"head": "red", "thorax": "gold", "abdomen": "magenta", "legs": "orange"}

    fem = z["recording"] == val_recording
    pred_p, pred_z, gt = z["pred_populated"], z["pred_zeroed"], z["gt_populated"]
    vis = z["vis_populated"] & z["vis_zeroed"]
    err_p = np.linalg.norm(pred_p - gt, axis=-1)
    err_z = np.linalg.norm(pred_z - gt, axis=-1)

    def per_ann_mpjpe(err):
        v = vis.astype(np.float64)
        denom = v.sum(1)
        return np.divide((err * v).sum(1), denom, out=np.full(denom.shape, np.nan), where=denom > 0)

    mp_p, mp_z = per_ann_mpjpe(err_p), per_ann_mpjpe(err_z)
    degradation = mp_z - mp_p
    fem_idx = np.where(fem)[0]
    if fem_idx.size == 0:
        print(f"no annotations for val_recording={val_recording!r}; nothing to render")
        return
    order = fem_idx[np.argsort(-degradation[fem_idx])]
    hard_picks = [i for i in order if np.isfinite(degradation[i])][:a.n_hard]
    easy_picks = [i for i in order[::-1] if np.isfinite(degradation[i])][:1]

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for tag, picks in (("hard", hard_picks), ("easy", easy_picks)):
        for rank, i in enumerate(picks):
            fn = str(z["file_name"][i])
            bbox = z["bbox"][i]
            crop, (x0, y0) = reconstruct_crop(a.data_root, fn, bbox)
            fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
            for ax, pred, label in ((axes[0], pred_p[i], "mask POPULATED"),
                                    (axes[1], pred_z[i], "mask ZEROED")):
                ax.imshow(crop)
                for k in range(len(kp_names)):
                    if not vis[i, k]:
                        continue
                    c = group_colors.get(idx2group.get(k, ""), "grey")
                    gx, gy = gt[i, k]
                    px, py = pred[k]
                    ax.plot(gx, gy, "o", color="white", mec="black", ms=5, mew=0.6)
                    ax.plot(px, py, "x", color=c, ms=7, mew=2)
                ax.set_title(f"{label}\nMPJPE={mp_p[i] if 'POPULATED' in label else mp_z[i]:.2f}px",
                            fontsize=10)
                ax.axis("off")
            fig.suptitle(f"{fn}  (crop origin {x0},{y0})  degradation(zeroed-populated)="
                        f"{degradation[i]:+.2f}px  [{tag} #{rank}]", fontsize=9)
            handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                                  markeredgecolor="black", label="GT", ms=6),
                      *[plt.Line2D([0], [0], marker="x", color=c, label=g, ms=7, mew=2)
                        for g, c in group_colors.items()]]
            fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8)
            fig.tight_layout(rect=(0, 0.06, 1, 1))
            out_path = Path(a.out_dir) / f"overlay_{tag}{rank}.png"
            fig.savefig(out_path, dpi=140)
            plt.close(fig)
            print(f"wrote {out_path}  (degradation {degradation[i]:+.2f}px, "
                 f"populated {mp_p[i]:.2f}px, zeroed {mp_z[i]:.2f}px)")


if __name__ == "__main__":
    main()
