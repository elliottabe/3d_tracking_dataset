#!/usr/bin/env python3
"""Are the wing keypoints on the wing? Rigid-vein length + where they land.

WHY: on Session0 bout 28 the shipped artifacts give V12-V13 vein lengths of
21.58u (male left) but 2.53u (male right), and 7.25u / 3.81u with 86-122% CV
on the female. A rigid vein cannot change length and two wings cannot differ
8.5x, so some of these landmarks are collapsed onto one point -- which is ALSO
why they score the BEST jitter in the table (0.06u step, zero spikes). Smooth
and wrong.

EXPECTATION, if the cause is unilateral wing extension (the male sings with one
wing, the other stays folded over the abdomen):
  * male: the EXTENDED wing's V12 and V13 sit apart, along the visible vein;
    the FOLDED wing's V12/V13 sit on top of each other ON the folded wing over
    the abdomen. Length flat-ish for the extended wing, tiny for the folded.
  * female: both wings folded, so both pairs collapse and their length wanders.
If instead a collapsed pair lands somewhere that is NOT a wing (on the thorax,
the other fly, the arena), it is a detector failure rather than an occlusion
limit, and the fix is training data, not interpretation.

Left: vein length vs frame for both wings/flies (a rigid vein must be a
horizontal line). Right: the wing keypoints drawn on real frames.
"""
import argparse, os, json, sys
import numpy as np

# `python scripts/viz/wing_vein_check.py` puts scripts/viz/ at sys.path[0], and
# it has no `core` submodule -- the failing `import viz` caches an EMPTY
# namespace package in sys.modules, which a later path insert does NOT undo.
# Drop the stale binding, then put the repo root first. Same fix as
# scripts/run_bout.py's qc-report import.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _m in [k for k in list(sys.modules) if k == "viz" or k.startswith("viz.")]:
    del sys.modules[_m]
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout-dir", required=True, help=".../bouts/bout_000NN")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--start-frame", type=int, required=True)
    ap.add_argument("--camera", default="Cam2012630")
    ap.add_argument("--calib-dir", default=None,
                    help="if given, also draw the REPROJECTED 3-D wing points in "
                         "green: where green and the 2-D marker disagree, the views "
                         "disagreed and triangulation could not resolve the pair")
    ap.add_argument("--masks", default=None, help="sam3_masks.npz for the centroid")
    ap.add_argument("--cameras", default=None,
                    help="COMMA LIST in the pipeline's CANONICAL order "
                         "(cfg.recording.cameras). REQUIRED with --masks: kp2d's "
                         "camera axis is canonical order, but the npz stores its "
                         "own order, and load_bout_masks reorders BY NAME. Taking "
                         "the index from the npz list plots one camera's keypoints "
                         "on another camera's image -- which is exactly the error "
                         "that made WingR look off-body on 2026-08-31.")
    # kp2d.npz / kp3d.npz are written AFTER reorder_detector_to_model
    # (run_bout.py:986), so their keypoint axis is cfg.model.KP_NAMES order --
    # NOT the detector's cfg.detector.kp_names order. Indexing them with the
    # DETECTOR order silently reads other body parts: on 2026-08-31 the detector
    # "WingR_base/V12/V13" indices 28/29/30 landed on model T2L_TiTa/TaT1/TaT3,
    # i.e. the middle-LEFT LEG, and produced a confident-looking "collapsed
    # right wing vein" that was really a tarsal segment. Default to the anatomy
    # config, and refuse a config that has no model.KP_NAMES.
    ap.add_argument("--kp-config", default="configs/anatomy/v1.yaml",
                    help="config carrying model.KP_NAMES (the order the "
                         "artifacts are stored in). NOT the detector config.")
    ap.add_argument("--max-frame", type=int, default=1500)
    ap.add_argument("--frames", default="225,750,1275")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from omegaconf import OmegaConf

    # Both index spaces come from viz.core.bout_artifacts, which takes names
    # (not integers) and refuses a detector config -- see its docstring and
    # tests/test_bout_artifacts_order.py for the 2026-08-31 failure this
    # prevents.
    from viz.core.bout_artifacts import load_bout_kp, model_kp_names
    kp = model_kp_names(args.kp_config)
    W = {s: {p: kp.index(f"Wing{s}_{p}") for p in ("base", "V12", "V13")}
         for s in ("L", "R")}
    T = args.max_frame
    picks = [int(x) for x in args.frames.split(",")]
    cams = None
    cen = None
    rt = None
    canon = ([c.strip() for c in args.cameras.split(",")] if args.cameras else None)
    if args.calib_dir:
        from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
        rt = ReprojectionTool(args.calib_dir)
    if args.masks:
        if canon is None:
            raise SystemExit("--masks requires --cameras (canonical order); see --help")
        z = np.load(args.masks, allow_pickle=True)
        stored = [str(c) for c in z["cameras"]]
        missing = [c for c in canon if c not in stored]
        if missing:
            raise SystemExit(f"cameras absent from the npz: {missing}")
        perm = [stored.index(c) for c in canon]     # npz order -> canonical order
        cams = list(canon)
        cen = z["centroids"][:, perm]              # (2,C,T,2) reindexed BY NAME
        if perm != sorted(perm):
            print(f"[camera-order] npz stored {stored}\n"
                  f"               canonical  {canon}\n"
                  f"               reindexed by name, perm={perm}")

    fig = plt.figure(figsize=(15, 3.1 * 2 * (1 + len(picks) // 3 + 1)))
    gs = fig.add_gridspec(2, 1 + len(picks), width_ratios=[1.5] + [1] * len(picks))
    report = {}
    for row, (fly, who) in enumerate(((1, "male"), (0, "FEMALE"))):
        bk = load_bout_kp(args.bout_dir, fly, anatomy_cfg=args.kp_config,
                          cameras=canon)
        k2, c2 = bk._kp2d[:T], bk._conf2d[:T]
        k3, c3 = bk._kp3d[:T], bk._conf3d[:T]

        ax = fig.add_subplot(gs[row, 0])
        for s, col in (("L", "tab:blue"), ("R", "tab:red")):
            L = np.linalg.norm(k3[:, W[s]["V12"], :] - k3[:, W[s]["V13"], :], axis=-1)
            ax.plot(L, lw=0.7, color=col, label=f"Wing{s} V12-V13")
            fin = L[np.isfinite(L)]
            report[f"fly{fly}|Wing{s}"] = dict(mean=float(fin.mean()),
                cv=float(fin.std() / fin.mean()), p5=float(np.percentile(fin, 5)),
                p95=float(np.percentile(fin, 95)))
        ax.axhline(21.5, ls="--", lw=1.0, color="k",
                   label="21.5u (extended-wing vein)")
        for p in picks:
            ax.axvline(p, color="0.7", lw=0.8, zorder=0)
        ax.set_title(f"fly{fly} ({who}): RIGID vein length must be a FLAT LINE",
                     fontsize=9)
        ax.set_xlabel("bout frame (0-%d)" % (T - 1)); ax.set_ylabel("length (world u)")
        ax.legend(fontsize=7)

        capv = cv2.VideoCapture(os.path.join(args.video_dir, f"{args.camera}.mp4"))
        for ci, p in enumerate(picks):
            capv.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame + p))
            ok, im = capv.read()
            if not ok:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            axi = fig.add_subplot(gs[row, 1 + ci])
            axi.imshow(im)
            cidx = cams.index(args.camera) if cams else 0
            for s, col in (("L", "cyan"), ("R", "red")):
                pts = np.array([k2[p, cidx, W[s][q]] for q in ("base", "V12", "V13")])
                cf = np.array([c2[p, cidx, W[s][q]] for q in ("base", "V12", "V13")])
                axi.plot(pts[:, 0], pts[:, 1], "-o", color=col, ms=5, lw=1.4,
                         label=f"Wing{s} (conf {cf.min():.2f})")
                for q, xy in zip(("b", "12", "13"), pts):
                    axi.annotate(q, xy, color=col, fontsize=6,
                                 xytext=(3, 3), textcoords="offset points")
                d = float(np.linalg.norm(pts[1] - pts[2]))
                report[f"fly{fly}|{args.camera}|{args.start_frame+p}|Wing{s}"] = \
                    dict(v12_v13_px=d, min_conf=float(cf.min()))
            if rt is not None:
                P = np.asarray(rt.camera_matrices, np.float32)[cidx]     # (4,3)
                for s2, mk in (("L", "s"), ("R", "D")):
                    for q in ("base", "V12", "V13"):
                        X = k3[p, W[s2][q]]
                        if not np.isfinite(X).all():
                            continue
                        uvw = np.concatenate([X, [1.0]]) @ P
                        axi.plot(uvw[0] / uvw[2], uvw[1] / uvw[2], mk,
                                 color="lime", ms=6, mew=1.4, mfc="none", zorder=5)
            if cen is not None:
                cx, cy = cen[fly, cidx, p]
                axi.plot(cx, cy, "o", mfc="none", mec="white", ms=14, mew=1.4)
                # FULL frame height (the frame is 1936x448) so nothing is cut
                # vertically -- an earlier +-115px window hid the WingR markers
                # and made them look off-body when they were not.
                axi.set_xlim(cx - 260, cx + 260)
                axi.set_ylim(im.shape[0], 0)
            L3 = np.linalg.norm(k3[p, W["L"]["V12"]] - k3[p, W["L"]["V13"]])
            R3 = np.linalg.norm(k3[p, W["R"]["V12"]] - k3[p, W["R"]["V13"]])
            axi.set_title(f"fly{fly} frame {args.start_frame+p}\n"
                          f"3D vein L {L3:.1f}u  R {R3:.1f}u", fontsize=8)
            axi.set_xticks([]); axi.set_yticks([])
            axi.legend(fontsize=6, loc="lower left")
        capv.release()
    fig.suptitle("Wing vein landmarks -- cyan = WingL 2-D, red = WingR 2-D, "
                 "GREEN open markers = the triangulated 3-D reprojected back "
                 "(square=L, diamond=R).\nGreen far from its 2-D marker means the "
                 "7 views disagreed and triangulation could not resolve that point.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=130)
    json.dump(report, open(f"{args.out}.json", "w"), indent=2)
    print("wrote", f"{args.out}.png")
    for k, v in report.items():
        if "cv" in v:
            print(f"  {k}: mean {v['mean']:.2f}u CV {v['cv']*100:.1f}%")


if __name__ == "__main__":
    main()
