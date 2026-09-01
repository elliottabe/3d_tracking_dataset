#!/usr/bin/env python3
"""Show the wing L/R collapse gate: which views are dropped, and what it fixes.

EXPECTATION, if the gate is doing what it claims:
  * LEFT panel: the per-camera separation |WingL_* - WingR_*| (body lengths)
    over time. Cameras that RESOLVE the wings sit high; a collapsed camera sits
    near zero. On the male, 630/853/862 should stay high while 631/855/857/861
    dip to ~0.3 exactly on the frames whose 3-D vein length flips.
  * MIDDLE: 3-D WingL_V12-V13 length before vs after. The flip spikes (~21.6u
    against a ~4.8u baseline) should mostly vanish; the flat part must NOT move,
    or the gate is changing frames that were already fine.
  * RIGHT: the frames themselves, marking which cameras were dropped, so a
    dropped view can be checked to actually have both labels on one wing.
If the spikes survive, the collapse is not the mechanism. If the flat region
moves, the gate is over-firing.
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
    ap.add_argument("--max-frame", type=int, default=1500)
    ap.add_argument("--frames", default="", help="LOCAL frames to render; "
                    "default = the worst flipped frames found")
    ap.add_argument("--n-render", type=int, default=3)
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

    cams = list(OmegaConf.load(args.recording).cameras)
    d = OmegaConf.load(args.detector)
    kp = model_kp_names(args.anatomy)
    bk = load_bout_kp(args.bout_dir, args.fly, anatomy_cfg=args.anatomy, cameras=cams)
    cen, _, _ = centroids_canonical(args.masks, cams)
    P = np.asarray(ReprojectionTool(args.calib_dir).camera_matrices, np.float32)
    T = min(args.max_frame, bk._kp2d.shape[0])
    k2, c2 = bk._kp2d[:T], bk._conf2d[:T]

    col, info = detect_wing_lr_collapse(k2, c2, kp, conf_thresh=float(d.conf_thresh))
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

    # separation per camera (recompute for plotting, same definition)
    ib, ia = kp.index("Scutellum"), kp.index("Abd_tip")
    body = np.linalg.norm(k2[:, :, ia, :] - k2[:, :, ib, :], axis=-1)
    body = np.where(body > 1, body, np.nan)
    seps = []
    for L, R in (("WingL_base", "WingR_base"), ("WingL_V12", "WingR_V12"),
                 ("WingL_V13", "WingR_V13")):
        seps.append(np.linalg.norm(k2[:, :, kp.index(L), :]
                                   - k2[:, :, kp.index(R), :], axis=-1) / body)
    sep = np.nanmean(np.stack(seps), axis=0)                # (T,C)

    if args.frames.strip():
        picks = [int(x) for x in args.frames.split(",")]
    else:
        # Pick the worst frames the gate can actually act on. Without the
        # min_views_kept mask the "worst" frames are the degenerate tail (the
        # female's wings stop being resolvable after ~frame 1100), where the
        # guard correctly does nothing -- so the panel would show the guard
        # holding rather than the gate working.
        actionable = (col.sum(1) > 0) & ((col.shape[1] - col.sum(1))
                                         >= args.min_views_kept)
        score = np.where(actionable, vb, -np.inf)
        worst = np.argsort(-score)[:args.n_render * 4]
        picks = sorted(set(int(x) for x in worst if np.isfinite(score[x])))[:args.n_render]
    if not picks:
        picks = [0]

    ncol = 2 + len(picks)
    fig = plt.figure(figsize=(4.6 * 2 + 3.0 * len(picks), 3.8))
    gs = fig.add_gridspec(1, ncol, width_ratios=[1.5, 1.5] + [1.0] * len(picks))

    ax = fig.add_subplot(gs[0, 0])
    for c, cam in enumerate(cams):
        drop = 100 * col[:, c].mean()
        ax.plot(sep[:, c], lw=0.7, label=f"{cam[-3:]} ({drop:.0f}% flagged)")
    ax.axhline(0.6, ls="--", color="k", lw=1.0, label="abs_floor 0.6")
    ax.set_title("per-camera |WingL - WingR| separation\n(body lengths; low = "
                 "both labels on one wing)", fontsize=9)
    ax.set_xlabel(f"bout frame (0-{T-1})"); ax.set_ylabel("separation")
    ax.legend(fontsize=6, ncol=2)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(vb, lw=0.8, color="tab:red", label="before gate")
    ax2.plot(va, lw=0.8, color="tab:green", label="after gate")
    ax2.set_title("3-D WingL V12-V13 length\n(RIGID: spikes are the flip)", fontsize=9)
    ax2.set_xlabel(f"bout frame (0-{T-1})"); ax2.set_ylabel("length (world u)")
    ax2.legend(fontsize=7)
    for p in picks:
        ax2.axvline(p, color="0.7", lw=0.8, zorder=0)

    caps = {c: cv2.VideoCapture(os.path.join(args.video_dir, f"{c}.mp4")) for c in cams}
    for j, p in enumerate(picks):
        # show the camera with the LARGEST separation (a view that resolved them)
        c = int(np.nanargmax(sep[p]))
        caps[cams[c]].set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame + p))
        ok, im = caps[cams[c]].read()
        axi = fig.add_subplot(gs[0, 2 + j])
        if ok:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            axi.imshow(im)
            for side, cl in (("L", "cyan"), ("R", "red")):
                pts = np.array([k2[p, c, kp.index(f"Wing{side}_{q}")]
                                for q in ("base", "V12", "V13")])
                axi.plot(pts[:, 0], pts[:, 1], "-o", color=cl, ms=5, lw=1.4)
            cx, cy = cen[args.fly, c, p]
            axi.plot(cx, cy, "o", mfc="none", mec="white", ms=13, mew=1.3)
            axi.set_xlim(cx - 190, cx + 190); axi.set_ylim(im.shape[0], 0)
        axi.set_xticks([]); axi.set_yticks([])
        dropped = [cams[k][-3:] for k in range(len(cams)) if col[p, k]]
        axi.set_title(f"local {p} | best view {cams[c][-3:]} (sep {sep[p,c]:.1f})\n"
                      f"vein {vb[p]:.1f} -> {va[p]:.1f}u\n"
                      f"dropped: {','.join(dropped) or 'none'}", fontsize=7)
    for cp in caps.values():
        cp.release()
    fig.suptitle(f"Wing L/R collapse gate -- fly{args.fly}, min_views_kept="
                 f"{args.min_views_kept}: masked {minfo['views_masked']} views over "
                 f"{minfo['frames_masked']} frames, left "
                 f"{minfo['frames_left_alone_too_few_views']} frames alone",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=130)
    json.dump({"detect": info, "mask": minfo,
               "flip_before": int((vb > 10).sum()), "flip_after": int((va > 10).sum())},
              open(f"{args.out}.json", "w"), indent=2, default=float)
    print(f"wrote {args.out}.png   flip frames {int((vb>10).sum())} -> {int((va>10).sum())}")


if __name__ == "__main__":
    main()
