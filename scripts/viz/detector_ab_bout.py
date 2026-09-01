#!/usr/bin/env python3
"""A/B two 2-D detector checkpoints on real bout frames, per fly and per camera.

EXPECTATION (write this down before looking, per CLAUDE.md):
The candidate is v5vf_maskoff -- trained with the 4th (SAM-mask) channel
ALWAYS ZERO, val MPJPE 5.29px overall / 9.93px female, against 5.49 / 19.08
for the identical recipe WITH the mask channel. The female roughly halved, so
on a courtship frame the visible difference should be concentrated in the
FEMALE (fly0) and in her legs/wings, not in the male:
  * fly1 (male): both columns should look about the same. A large male
    difference means something other than the mask channel changed.
  * fly0 (female), especially against a wall or partly occluded: the candidate
    should put leg chains on actual legs where the baseline scatters them. Leg
    chains (T1L_ThxCx -> ... -> T1L_TaTip) are drawn as connected polylines
    precisely so a scrambled chain is visible as a zig-zag rather than hiding
    in a cloud of dots.
FAILURE MODES this must be able to show, or it proves nothing:
  * the candidate fed a REAL mask channel (flag not set) -> keypoints degrade
    everywhere, both flies, which is the train/inference mismatch not a win;
  * anatomy scrambled (a keypoint-order bug) -> chains connect across the body
    rather than down a limb. LOO/IoU/MPJPE are all blind to this.

Renders one row per (fly, camera) with the two detectors side by side, keypoint
names/colours from viz.core.colors so it reads like every other figure here.

    python scripts/viz/detector_ab_bout.py \
        --run-root <.../pose> --bout 28 --frames 400,1000,1450 \
        --cameras Cam2012630,Cam2012855 \
        --det baseline=<ckpt dir> --det maskoff=<ckpt dir>:zeromask \
        --out figures/<topic>/detector_ab
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True, help="bout's sam3_masks.npz")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--start-frame", type=int, required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--frames", default="400,1000,1450",
                    help="LOCAL bout frames (comma list)")
    ap.add_argument("--cameras", default="Cam2012630,Cam2012855")
    ap.add_argument("--flies", default="0,1")
    ap.add_argument("--det", action="append", required=True,
                    metavar="LABEL=CKPT[:zeromask]")
    ap.add_argument("--kp-config", default="configs/detector/vitpose_v3.yaml")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--decode-sharpen", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from omegaconf import OmegaConf
    from viz.core.colors import PALETTE, keypoint_groups, leg_chains
    from jarvis_jax.tracking.predict_2d import load_detector, predict_bout_2d
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

    kp_names = list(OmegaConf.load(args.kp_config).kp_names)
    groups = keypoint_groups(kp_names)
    chains = leg_chains(kp_names)
    # BGR palette -> matplotlib RGB 0..1
    col = {k: (v[2] / 255, v[1] / 255, v[0] / 255) for k, v in PALETTE.items()}

    z = np.load(args.masks, allow_pickle=True)
    cams_all = [str(c) for c in z["cameras"]]
    packed, shp = z["packed"], tuple(int(x) for x in z["shape"])
    cen, val = z["centroids"], z["valid"]
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    flies = [int(f) for f in args.flies.split(",") if f.strip() != ""]
    locals_ = [int(f) for f in args.frames.split(",") if f.strip() != ""]
    cam_mats = np.asarray(ReprojectionTool(args.calib_dir).camera_matrices, np.float32)

    def unpack(fly, t):
        return np.unpackbits(packed[fly, :, t], axis=-1)[:, :shp[1]].astype(bool)

    # frames: (n, C, H, W, 3)
    frames = []
    for p in locals_:
        per = []
        for c in cams_all:
            capv = cv2.VideoCapture(os.path.join(args.video_dir, f"{c}.mp4"))
            capv.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame + p))
            ok, im = capv.read(); capv.release()
            if not ok:
                raise SystemExit(f"read fail {c} @{args.start_frame+p}")
            per.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        frames.append(np.stack(per))
    frames = np.stack(frames)

    dets = []
    for spec in args.det:
        lab, rest = spec.split("=", 1)
        zero = rest.endswith(":zeromask")
        dets.append((lab, rest[:-len(":zeromask")] if zero else rest, zero))

    results = {}
    for lab, ckpt, zero in dets:
        vit = load_detector(ckpt, num_keypoints=len(kp_names))
        for fly in flies:
            m = np.stack([unpack(fly, p) for p in locals_])            # (n,C,H,W)
            other = np.stack([unpack(1 - fly, p) for p in locals_])
            kp2d, conf = predict_bout_2d(
                vit, list(frames), m, cen[fly][:, locals_].transpose(1, 0, 2),
                val[fly][:, locals_].T, cam_mats, decode_sharpen=args.decode_sharpen,
                distractor_masks=other, distractor_dilate=15,
                zero_mask_channel=zero)
            results[(lab, fly)] = (kp2d, conf)
            print(f"  {lab} fly{fly}: kp2d{kp2d.shape} "
                  f"median conf {np.median(conf[conf>0]):.3f} "
                  f"frac>thr {float((conf>args.conf).mean()):.3f}"
                  f"{'  [mask channel ZEROED]' if zero else ''}", flush=True)
        del vit

    rows = [(fly, cam, p) for fly in flies for cam in cams for p in locals_]
    fig, axes = plt.subplots(len(rows), len(dets),
                            figsize=(7.6 * len(dets), 2.4 * len(rows)), squeeze=False)
    HALF = 190
    report = {}
    for ci_d, (lab, ckpt, zero) in enumerate(dets):
        for ri, (fly, cam, p) in enumerate(rows):
            ax = axes[ri][ci_d]
            cidx = cams_all.index(cam); tidx = locals_.index(p)
            img = frames[tidx, cidx]
            kp2d, conf = results[(lab, fly)]
            k, c = kp2d[tidx, cidx], conf[tidx, cidx]
            cx, cy = cen[fly, cidx, p]
            ax.imshow(img)
            for g, cname in (("head", "head"), ("thorax", "thorax"),
                            ("abdomen", "abdomen"), ("legs", "legs")):
                ii = [i for i in groups[g] if c[i] > args.conf]
                if ii:
                    ax.scatter(k[ii, 0], k[ii, 1], s=13, c=[col[cname]],
                               edgecolors="k", linewidths=0.3, zorder=3,
                               label=g if ri == 0 and ci_d == 0 else None)
            for leg, idxs in chains.items():
                ok = [i for i in idxs if c[i] > args.conf]
                if len(ok) >= 2:
                    ax.plot(k[ok, 0], k[ok, 1], "-", lw=1.1,
                            color=col["legs"], alpha=0.9, zorder=2)
            ax.plot(cx, cy, "o", mfc="none", mec="white", ms=13, mew=1.4, zorder=4)
            ax.set_xlim(cx - HALF, cx + HALF); ax.set_ylim(cy + HALF * 0.55, cy - HALF * 0.55)
            who = "male" if fly == 1 else "FEMALE"
            ax.set_title(f"{lab}{' [ch3=0]' if zero else ''} | fly{fly} ({who}) | "
                         f"{cam} | frame {args.start_frame+p} | "
                         f"{int((c>args.conf).sum())}/{len(kp_names)} kp > {args.conf}",
                         fontsize=7.5)
            ax.set_xticks([]); ax.set_yticks([])
            report[f"{lab}|fly{fly}|{cam}|{args.start_frame+p}"] = {
                "n_kp_above_conf": int((c > args.conf).sum()),
                "median_conf": float(np.median(c[c > 0])) if (c > 0).any() else None,
                "zero_mask_channel": zero}
    axes[0][0].legend(fontsize=7, loc="upper left", framealpha=0.8)
    # NOTE viz.core.colors.PALETTE is BGR (cv2 order): "legs" (255,140,0) is
    # AZURE here, not the "orange" its comment in colors.py claims -- the
    # comment is wrong, the tuple is what renders. Caption says what is drawn.
    fig.suptitle("2-D detector A/B on bout frames -- colours from viz.core.colors "
                 "(red=head, yellow=thorax, magenta=abdomen, blue=legs w/ chains)\n"
                 "white circle = SAM3 fly centroid; male = fly1, FEMALE = fly0",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=125)
    json.dump(report, open(f"{args.out}.json", "w"), indent=2)
    print("wrote", f"{args.out}.png / .json")


if __name__ == "__main__":
    main()
