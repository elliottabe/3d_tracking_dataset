#!/usr/bin/env python3
"""All cameras at once: does WingL land on the RIGHT wing in some views?

The male's WingL_V12/V13 jump on ~20 of the first 40 frames of Session0 bout 28,
with the V12-V13 length alternating 4.6u <-> 21.6u (21.6 ~= base->V12 + V12->V13,
i.e. V12 sitting on the wing BASE). run_bout's displacement-gate comment calls
this "a correlated majority-flip artifact ... a landmark alternating between two
real-but-wrong visible structures at high confidence, camera majority flipping
frame to frame".

THE QUESTION THIS ANSWERS. Triangulation with consensus rejection
(reproj_resid_px) is SUPPOSED to fix a per-view error: a minority of wrong views
gets gated out. So either
  (a) only a MINORITY of cameras swaps L<->R, and the gate failed -- then the
      fix is the gate (threshold, or it is not reached for this keypoint); or
  (b) the MAJORITY swaps, and no consensus can help -- the 3-D faithfully
      reconstructs a wrong correspondence, and the fix must be 2-D/temporal.
Counting swapped views per frame decides it, and that is what this renders.

EXPECTATION: on a flip frame, cyan (WingL) markers sit on the fly's RIGHT wing
in some panels and the left in others. Row title reports each camera's own
reprojection residual for that keypoint, so a swapped view that the gate did
NOT reject is visible as a large residual that survived.
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
    ap.add_argument("--frames", required=True, help="LOCAL bout frames")
    ap.add_argument("--anatomy", default="configs/anatomy/v1.yaml")
    ap.add_argument("--recording", default="configs/recording/session0.yaml")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from omegaconf import OmegaConf
    from viz.core.bout_artifacts import load_bout_kp, model_kp_names, centroids_canonical
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

    cams = list(OmegaConf.load(args.recording).cameras)
    kp = model_kp_names(args.anatomy)
    bk = load_bout_kp(args.bout_dir, args.fly, anatomy_cfg=args.anatomy, cameras=cams)
    cen, val, _ = centroids_canonical(args.masks, cams)
    P = np.asarray(ReprojectionTool(args.calib_dir).camera_matrices, np.float32)
    picks = [int(x) for x in args.frames.split(",")]
    NAMES = ["WingL_base", "WingL_V12", "WingL_V13",
             "WingR_base", "WingR_V12", "WingR_V13"]
    I = {n: kp.index(n) for n in NAMES}

    fig, axes = plt.subplots(len(picks), len(cams),
                             figsize=(3.0 * len(cams), 2.5 * len(picks)),
                             squeeze=False)
    report = {}
    caps = {c: cv2.VideoCapture(os.path.join(args.video_dir, f"{c}.mp4")) for c in cams}
    for r, p in enumerate(picks):
        vL = np.linalg.norm(bk._kp3d[p, I["WingL_V12"]] - bk._kp3d[p, I["WingL_V13"]])
        for c, cam in enumerate(cams):
            caps[cam].set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame + p))
            ok, im = caps[cam].read()
            ax = axes[r][c]
            if not ok:
                ax.axis("off"); continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            ax.imshow(im)
            resid = {}
            for side, col in (("L", "cyan"), ("R", "red")):
                pts = np.array([bk._kp2d[p, c, I[f"Wing{side}_{q}"]]
                                for q in ("base", "V12", "V13")])
                ax.plot(pts[:, 0], pts[:, 1], "-o", color=col, ms=5, lw=1.5, zorder=4)
                for q, xy in zip(("b", "12", "13"), pts):
                    ax.annotate(q, xy, color=col, fontsize=7, zorder=5,
                                xytext=(3, 3), textcoords="offset points")
                # this camera's own reprojection residual for V12
                X = bk._kp3d[p, I[f"Wing{side}_V12"]]
                if np.isfinite(X).all():
                    uvw = np.concatenate([X, [1.0]]) @ P[c]
                    pr = uvw[:2] / max(uvw[2], 1e-9)
                    ax.plot(pr[0], pr[1], "s", mfc="none", mec="lime", ms=9,
                            mew=1.6, zorder=6)
                    resid[side] = float(np.linalg.norm(pr - pts[1]))
            cx, cy = cen[args.fly, c, p]
            ax.plot(cx, cy, "o", mfc="none", mec="white", ms=13, mew=1.3, zorder=3)
            ax.set_xlim(cx - 170, cx + 170); ax.set_ylim(im.shape[0], 0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{cam[-3:]} f{args.start_frame+p}\n"
                         f"V12 resid L {resid.get('L', float('nan')):.0f}p "
                         f"R {resid.get('R', float('nan')):.0f}p", fontsize=7)
            report[f"f{args.start_frame+p}|{cam}"] = dict(
                resid_L=resid.get("L"), resid_R=resid.get("R"),
                vein3d_L=float(vL))
        axes[r][0].set_ylabel(f"local {p}\nvein L {vL:.1f}u", fontsize=8)
    for cp in caps.values():
        cp.release()
    fig.suptitle("All cameras, male WingL (cyan) vs WingR (red): base-12-13. "
                 "GREEN square = the 3-D V12 reprojected into that view.\n"
                 "A view where CYAN sits on the fly's RIGHT wing is an L<->R swap; "
                 "count them to see if the majority flipped.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=125)
    json.dump(report, open(f"{args.out}.json", "w"), indent=2)
    print("wrote", f"{args.out}.png")


if __name__ == "__main__":
    main()
