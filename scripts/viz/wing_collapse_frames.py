#!/usr/bin/env python3
"""The corrected frames themselves: 3-D wing BEFORE vs AFTER the collapse gate.

EXPECTATION on a flip frame, in a view that RESOLVES the two wings
(630/853/862 on Session0 bout 28):
  * MAGENTA x (3-D WingL V12/V13 reprojected BEFORE the gate) sits away from the
    left wing -- pulled toward the body/right wing, because 4 of 7 views had
    both wing labels on one wing and the consensus followed them.
  * LIME o (AFTER) sits ON the left wing, next to that view's own cyan 2-D.
  * the collapsed view (855) is shown last for contrast: its OWN cyan and red
    2-D sit on top of each other, which is why it was dropped.
If magenta and lime coincide, the gate changed nothing on this frame. If lime is
off the wing, the remaining views were not enough.
"""
import argparse, os, json, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _m in [k for k in list(sys.modules) if k == "viz" or k.startswith("viz.")]:
    del sys.modules[_m]
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout-dir", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--start-frame", type=int, required=True)
    ap.add_argument("--fly", type=int, default=1)
    ap.add_argument("--cameras", default="Cam2012630,Cam2012853,Cam2012862,Cam2012855")
    ap.add_argument("--max-frame", type=int, default=300)
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--frames", default="")
    ap.add_argument("--anatomy", default="configs/anatomy/v1.yaml")
    ap.add_argument("--recording", default="configs/recording/session0.yaml")
    ap.add_argument("--detector", default="configs/detector/vitpose_v3.yaml")
    ap.add_argument("--min-views-kept", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from omegaconf import OmegaConf
    from viz.core.bout_artifacts import load_bout_kp, model_kp_names, centroids_canonical
    from jarvis_jax.tracking.wing_lr_assign import (detect_wing_lr_collapse,
                                                    mask_collapsed_wing_views)
    from jarvis_jax.tracking.triangulate import triangulate_keypoints
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

    canon = list(OmegaConf.load(args.recording).cameras)
    d = OmegaConf.load(args.detector)
    kp = model_kp_names(args.anatomy)
    bk = load_bout_kp(args.bout_dir, args.fly, anatomy_cfg=args.anatomy, cameras=canon)
    cen, _, _ = centroids_canonical(args.masks, canon)
    P = np.asarray(ReprojectionTool(args.calib_dir).camera_matrices, np.float32)
    T = min(args.max_frame, bk._kp2d.shape[0])
    k2, c2 = bk._kp2d[:T], bk._conf2d[:T]

    col, _ = detect_wing_lr_collapse(k2, c2, kp, conf_thresh=float(d.conf_thresh))
    c2m, minfo = mask_collapsed_wing_views(c2, col, kp,
                                           min_views_kept=args.min_views_kept)
    tri = lambda cc: triangulate_keypoints(
        k2, cc, P, conf_thresh=float(d.conf_thresh),
        view_conf_thresh=float(d.view_conf_thresh),
        reproj_resid_px=float(d.reproj_resid_px))[0]
    before, after = tri(c2), tri(c2m)
    iL12, iL13 = kp.index("WingL_V12"), kp.index("WingL_V13")
    vb = np.linalg.norm(before[:, iL12] - before[:, iL13], axis=-1)
    va = np.linalg.norm(after[:, iL12] - after[:, iL13], axis=-1)

    if args.frames.strip():
        picks = [int(x) for x in args.frames.split(",")]
    else:
        flip = np.flatnonzero(vb > 10)
        # prefer frames the gate actually CORRECTED, so the panel shows the fix
        fixed = [int(t) for t in flip if va[t] < 10]
        picks = (fixed or [int(t) for t in flip])[:args.n_frames]
    if not picks:
        raise SystemExit("no flipped frames found in this window")
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]

    def rp(X, c):
        h = np.concatenate([np.asarray(X, float), [1.0]])
        uvw = h @ P[c]
        return uvw[:2] / max(uvw[2], 1e-9)

    fig, axes = plt.subplots(len(picks), len(cams),
                             figsize=(3.4 * len(cams), 2.9 * len(picks)), squeeze=False)
    caps = {c: cv2.VideoCapture(os.path.join(args.video_dir, f"{c}.mp4")) for c in cams}
    rep = {}
    for r, p in enumerate(picks):
        for j, cam in enumerate(cams):
            c = canon.index(cam)
            caps[cam].set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame + p))
            ok, im = caps[cam].read()
            ax = axes[r][j]
            if not ok:
                ax.axis("off"); continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            ax.imshow(im)
            for side, cl in (("L", "cyan"), ("R", "red")):
                pts = np.array([k2[p, c, kp.index(f"Wing{side}_{q}")]
                                for q in ("base", "V12", "V13")])
                ax.plot(pts[:, 0], pts[:, 1], "-o", color=cl, ms=4, lw=1.2,
                        alpha=0.9, zorder=3)
            for i, mk in ((iL12, "x"), (iL13, "x")):
                q = rp(before[p, i], c)
                ax.plot(q[0], q[1], mk, color="magenta", ms=11, mew=2.4, zorder=5)
            for i in (iL12, iL13):
                q = rp(after[p, i], c)
                ax.plot(q[0], q[1], "o", mfc="none", mec="lime", ms=12, mew=2.2,
                        zorder=6)
            cx, cy = cen[args.fly, c, p]
            ax.plot(cx, cy, "o", mfc="none", mec="white", ms=12, mew=1.2, zorder=4)
            ax.set_xlim(cx - 175, cx + 175); ax.set_ylim(im.shape[0], 0)
            ax.set_xticks([]); ax.set_yticks([])
            tag = "DROPPED" if col[p, c] else "kept"
            ax.set_title(f"{cam[-3:]} [{tag}]  local {p}\n"
                         f"vein {vb[p]:.1f} -> {va[p]:.1f}u", fontsize=8)
            rep[f"{p}|{cam}"] = dict(dropped=bool(col[p, c]),
                                     vein_before=float(vb[p]), vein_after=float(va[p]))
    for cp in caps.values():
        cp.release()
    fig.suptitle(f"Male wing L/R collapse -- CORRECTED frames. cyan/red = that "
                 f"view's own 2-D WingL/WingR;  MAGENTA x = 3-D WingL V12/V13 "
                 f"reprojected BEFORE the gate;  LIME o = AFTER.\n"
                 f"Lime should sit on the left wing beside cyan; magenta should "
                 f"not. min_views_kept={args.min_views_kept}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=130)
    json.dump(rep, open(f"{args.out}.json", "w"), indent=2)
    print(f"wrote {args.out}.png  frames {picks}  "
          f"vein before {np.round(vb[picks],1).tolist()} -> after {np.round(va[picks],1).tolist()}")


if __name__ == "__main__":
    main()
