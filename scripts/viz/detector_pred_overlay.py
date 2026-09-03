#!/usr/bin/env python3
"""Render ONE detector's predictions against GT on real val frames.

scripts/viz/detector_ckpt_overlay.py is a two-arm A/B and refuses a single
checkpoint (and refuses any arm whose keypoint order was only warn-verified --
correctly: v5vf's training root is deleted, so its axis cannot be checked).
This renders one model whose order IS verified.

EXPECTATION, given v12's aggregate is 15.19 px but its body keypoints are
7.6-12 px and its tarsal tips 32-37 px: the green prediction skeleton should
sit cleanly on the body, thorax and femurs in every panel, and diverge from
cyan GT at the LEG EXTREMITIES. If instead whole skeletons are displaced, or
predictions land on the wrong fly, the error is not a tarsal-precision problem
and the per-keypoint reading is wrong.

Cyan = GT label. Green = prediction. Red segments join a GT/pred pair more
than `--flag-px` apart, so the eye goes to the failures rather than the bulk.

    python scripts/viz/detector_pred_overlay.py \
        --npz figures/2026-09-03-v12-detector-ab/v12_bal_maskoff.npz \
        --root <v12 root> --out figures/2026-09-03-v12-detector-ab
"""
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--flag-px", type=float, default=20.0)
    a = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "third_party", "jarvis_jax"))
    from jarvis_jax.data.transforms import crop_origin

    z = np.load(a.npz, allow_pickle=True)
    # THE NPZ IS IN CROP SPACE, NOT IMAGE SPACE. mask_channel_eval stores
    # coordinates relative to the 448x448 crop the detector saw; drawing them
    # straight onto the 1936x448 frame puts every skeleton in the left third of
    # the image, on empty floor, while GT and prediction still agree with each
    # other -- so the panel looks like a catastrophic model failure and the
    # MPJPE in its title still reads 2.3 px. Map back through the repo's own
    # crop_origin so the convention cannot drift from the one the eval used.
    pred, gt = z["pred_zeroed"].copy(), z["gt_zeroed"].copy()
    for i in range(len(pred)):
        w, h = int(z["img_wh"][i][0]), int(z["img_wh"][i][1])
        x0, y0 = crop_origin(z["bbox"][i], w, h, crop=448)
        pred[i] += (x0, y0); gt[i] += (x0, y0)
    ann_id, rec, fn, kpn = z["ann_id"], z["recording"], z["file_name"], z["kp_names"]
    sex = z["sex"]
    d = json.load(open(os.path.join(a.root, "annotations", "instances_val.json")))
    visv = {an["id"]: (np.asarray(an["keypoints"], float).reshape(-1, 3)[:, 2] > 0)
            for an in d["annotations"]}
    vis = np.stack([visv[int(i)] for i in ann_id])
    per = np.array([np.linalg.norm(pred[i] - gt[i], axis=-1)[vis[i]].mean()
                    if vis[i].any() else np.nan for i in range(len(pred))])

    order = np.argsort(np.nan_to_num(per, nan=-1))
    ok = ~np.isnan(per)
    med_i = order[ok.sum() // 2 - a.n // 2: ok.sum() // 2 - a.n // 2 + a.n]
    bands = [("BEST", order[:a.n]), ("MEDIAN", med_i), ("WORST", order[-a.n:][::-1])]

    fig, axes = plt.subplots(len(bands), a.n, figsize=(5.6 * a.n, 2.0 * len(bands)))
    axes = np.atleast_2d(axes)
    for bi, (label, idxs) in enumerate(bands):
        for ci, i in enumerate(idxs):
            ax = axes[bi][ci]; ax.axis("off")
            img = np.asarray(Image.open(os.path.join(a.root, "images", str(fn[i]))).convert("L"))
            ax.imshow(img, cmap="gray")
            v = vis[i]
            ax.scatter(gt[i][v, 0], gt[i][v, 1], s=7, c="#00d5ff", linewidths=0, label="GT", zorder=3)
            ax.scatter(pred[i][v, 0], pred[i][v, 1], s=7, c="#39ff14", linewidths=0, label="pred", zorder=4)
            dd = np.linalg.norm(pred[i] - gt[i], axis=-1)
            for k in np.where(v & (dd > a.flag_px))[0]:
                ax.plot([gt[i][k, 0], pred[i][k, 0]], [gt[i][k, 1], pred[i][k, 1]],
                        "-", c="red", lw=0.9, zorder=5)
            worst_k = str(kpn[np.argmax(np.where(v, dd, -1))])
            ax.set_title(f"{label if ci == 0 else ''}  {str(rec[i])[:19]} {str(fn[i]).split('/')[-1][:14]}"
                         f"\n{str(sex[i])}  MPJPE {per[i]:.1f}px  worst: {worst_k} {dd[v].max():.0f}px",
                         fontsize=7,
                         color=("crimson" if label == "WORST" else "black"))
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle("v12 detector on held-out val.  cyan = GT, green = prediction, "
                 "red line = pair >20 px apart.\n"
                 "Aggregate 15.19 px; body keypoints 7.6-12 px, tarsal tips 32-37 px "
                 "-- so the divergence should be at the LEG TIPS.", fontsize=11)
    fig.tight_layout(rect=(0, 0.035, 1, 0.90))
    p = os.path.join(a.out, "v12_pred_overlay.png")
    fig.savefig(p, dpi=125)
    print("wrote", p)
    print(f"per-annotation MPJPE: best {np.nanmin(per):.2f}  median {np.nanmedian(per):.2f}  "
          f"worst {np.nanmax(per):.2f}")


if __name__ == "__main__":
    main()
