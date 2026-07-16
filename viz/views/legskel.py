"""legskel view: per-leg proximal->distal joint CHAINS (detector-2D vs
fitted-3D vs an optional compare run's fitted-3D) reprojected onto each
camera's raw frame for one (bout, fly, frame), montaged into a single PNG.

Behavior ported from docs/plans/viz-reference/{viz_legskel,viz_legcompare}.py
(their draw_chains / draw_chains_2d helpers) onto viz.core (reproject/
overlays/io/layout/colors) instead of their inline cv2/project_points/
hardcoded-path code. Chain membership (per-leg proximal->distal index lists)
comes from core.colors.leg_chains(kp_names) instead of the reference
scripts' hand-rolled `segs`/`legs` dicts.

Colour choice for --compare: detector uses PALETTE["detector"] (cyan) and
this run's own fit uses PALETTE["fit"] (green); the compare run's fitted
chains use PALETTE["fly1"] (orange) -- distinct from both cyan and green
(and from PALETTE["head"]/red, reserved by the overlay view's axis/head
markers) so three chain layers stay legible on one tile.

DictConfig note / lazy scripts.run_bout import: identical
rationale to viz/views/overlay.py -- see that module's docstring. Duplicated
here (rather than imported from overlay.py) so this view has no dependency
on its sibling view module.
"""
import os
import sys

import cv2
import numpy as np
from hydra import initialize_config_dir, compose

from viz.core import colors as vcolors
from viz.core import io as vio
from viz.core import layout
from viz.core import overlays
from viz.core import reproject
from viz.config import courtship_recording, _CFG_DIR

_NODATA_COLOR = (0, 0, 255)
_COMPARE_COLOR_NAME = "fly1"  # orange; distinct from detector(cyan)/fit(green)


def _compose_cfg():
    """Re-compose the raw `pipeline` DictConfig, needed only for
    scripts.run_bout.bout_start_frame. Reuses viz.config._CFG_DIR
    (single source of truth for the configs/ path)."""
    os.environ.setdefault("USER", "eabe")
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(_CFG_DIR)):
        return compose(config_name="pipeline")


def _draw_leg_chains(bgr, chains, points_by_idx, color, project_fn=None):
    """Draw each leg's chain of 2D points (or 3D points reprojected via
    project_fn) in `color`. Returns the finite 2D points actually drawn
    (for crop-bounds accumulation) and whether anything was drawn at all."""
    drawn_pts = []
    drawn_any = False
    for idxs in chains.values():
        pts = points_by_idx[idxs]
        uv = project_fn(pts) if project_fn is not None else pts
        if np.isfinite(uv).any():
            overlays.draw_chain(bgr, uv, color)
            finite = np.asarray(uv, float)
            finite = finite[np.isfinite(finite).all(-1)]
            if len(finite):
                drawn_pts.append(finite)
            drawn_any = True
    return drawn_pts, drawn_any


def run(args):
    # cwd-independence: repo_root is 3 levels up from this file
    # (viz/views/legskel.py -> viz/views -> viz -> repo root).
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    rec = courtship_recording()
    all_cameras = list(rec["cameras"])
    bout, fly, fr = int(args.bout), int(args.fly), int(args.frame)

    cfg = _compose_cfg()
    from scripts.run_bout import bout_start_frame  # lazy: pulls in jax/mujoco/egl
    start = bout_start_frame(cfg, bout)
    frame_idx = start + fr

    run_root = args.run
    outputs_path = os.path.join(vio.fly_dir(run_root, bout, fly), "outputs.h5")
    if not os.path.exists(outputs_path):
        raise FileNotFoundError(f"missing outputs.h5 for bout={bout} fly={fly}: {outputs_path}")
    outs = vio.load_outputs(run_root, bout, fly)
    kp3d_mm, kp_names = outs["kp3d_mm"], outs["kp_names"]
    T = kp3d_mm.shape[0]
    frame_ok = 0 <= fr < T
    if not frame_ok:
        print(f"[legskel] warning: frame {fr} out of range for bout {bout} fly {fly} "
              f"(T={T}); rendering 'no data' tiles instead of aborting")

    try:
        kp2d, _conf2d = vio.load_kp2d(run_root, bout, fly)
    except (FileNotFoundError, OSError) as e:
        print(f"[legskel] warning: kp2d.npz unavailable ({e}); detector chain overlay disabled")
        kp2d = None

    compare_kp3d = None
    if args.compare:
        try:
            compare_kp3d = vio.load_outputs(args.compare, bout, fly)["kp3d_mm"]
        except Exception as e:
            print(f"[legskel] warning: --compare run unavailable ({e}); skipping compare overlay")

    chains = vcolors.leg_chains(kp_names)
    if not chains:
        print(f"[legskel] warning: no leg chains found in kp_names ({outputs_path}); "
              "all tiles will show 'no data'")

    cam_mats, cam_names = reproject.camera_matrices(rec["calib_dir"])
    cam_mat_idx = {n: i for i, n in enumerate(cam_names)}
    want_cams = set(args.cams) if args.cams else None

    tiles = []
    for ci, cam in enumerate(all_cameras):
        if want_cams is not None and cam not in want_cams:
            continue
        cmi = cam_mat_idx.get(cam)
        if cmi is None:
            print(f"[legskel] warning: camera {cam} missing from calibration; skipping tile")
            continue

        video_path = os.path.join(rec["session_dir"], f"{cam}.mp4")
        if frame_ok:
            try:
                bgr = vio.read_frame(video_path, frame_idx)
            except Exception as e:
                print(f"[legskel] warning: cannot read frame {frame_idx} for {cam} ({e}); skipping tile")
                continue
        else:
            # fr is out of range for this bout -- frame_idx (start + fr) is not a
            # meaningful video frame either, so don't attempt to decode it (it may
            # be far beyond the video's length or, for negative fr, silently wrap).
            # Use a blank placeholder sized from the container's metadata (no
            # frame decode needed) so the "no data" note still has a tile to sit on.
            cap = cv2.VideoCapture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            cap.release()
            bgr = np.zeros((h, w, 3), dtype=np.uint8)

        cam_mat = cam_mats[cmi]
        pts_for_crop = []
        notes = []

        if frame_ok:
            try:
                if kp2d is not None and kp2d.shape[-2] > 0:
                    kp2d_cam = kp2d[fr, ci]
                    pts, drawn_any = _draw_leg_chains(bgr, chains, kp2d_cam, vcolors.PALETTE["detector"])
                    pts_for_crop.extend(pts)
                else:
                    drawn_any = False
                if not drawn_any:
                    notes.append(("no detector kp data", vcolors.PALETTE["detector"]))
            except Exception as e:
                print(f"[legskel] warning: detector chain overlay failed for {cam}: {e}")
                notes.append(("detector kp error", vcolors.PALETTE["detector"]))

        if frame_ok:
            try:
                kp3d_fr = kp3d_mm[fr]
                proj = lambda p: reproject.project(cam_mat, p)  # noqa: E731
                pts, drawn_any = _draw_leg_chains(bgr, chains, kp3d_fr, vcolors.PALETTE["fit"], project_fn=proj)
                pts_for_crop.extend(pts)
                if not drawn_any:
                    notes.append(("no fit kp data", vcolors.PALETTE["fit"]))
            except Exception as e:
                print(f"[legskel] warning: fit chain overlay failed for {cam}: {e}")
                notes.append(("fit kp error", vcolors.PALETTE["fit"]))

        if args.compare and frame_ok:
            try:
                if compare_kp3d is not None and fr < compare_kp3d.shape[0]:
                    cmp_fr = compare_kp3d[fr]
                    proj = lambda p: reproject.project(cam_mat, p)  # noqa: E731
                    pts, drawn_any = _draw_leg_chains(
                        bgr, chains, cmp_fr, vcolors.PALETTE[_COMPARE_COLOR_NAME], project_fn=proj)
                    pts_for_crop.extend(pts)
                else:
                    drawn_any = False
                if not drawn_any:
                    notes.append(("no compare data", vcolors.PALETTE[_COMPARE_COLOR_NAME]))
            except Exception as e:
                print(f"[legskel] warning: compare chain overlay failed for {cam}: {e}")
                notes.append(("compare error", vcolors.PALETTE[_COMPARE_COLOR_NAME]))

        if pts_for_crop:
            allpts = np.concatenate(pts_for_crop, axis=0)
        else:
            allpts = np.array([[bgr.shape[1] / 2.0, bgr.shape[0] / 2.0]])
            note_text = "no data (frame out of range)" if not frame_ok else "no data"
            notes.append((note_text, _NODATA_COLOR))

        crop, _ = layout.crop_to_points(bgr, allpts)
        overlays.legend(crop, [(cam, (0, 255, 255))] + notes)
        tiles.append(crop)

    if not tiles:
        raise RuntimeError(
            f"no camera tiles rendered for run={run_root} bout={bout} fly={fly} frame={fr} "
            f"(all camera videos missing/unreadable under {rec['session_dir']})")

    mont = layout.montage(tiles, cols=min(2, len(tiles)))

    legend_items = [(f"bout{bout} fly{fly} frame{fr} (abs {frame_idx})", (255, 255, 255)),
                    ("detector=leg chains", vcolors.PALETTE["detector"]),
                    ("fit=leg chains", vcolors.PALETTE["fit"])]
    if args.compare:
        legend_items.append((f"compare={args.compare}", vcolors.PALETTE[_COMPARE_COLOR_NAME]))

    band = layout.banner(mont.shape[1], legend_items)
    out_img = np.vstack([mont, band])

    out_path = args.out or f"legskel_bout{bout}_fly{fly}_f{fr}.png"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, out_img)
    return 0
