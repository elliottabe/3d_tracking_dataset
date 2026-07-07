"""overlay view: mesh/kp/mask/axis reprojected onto each camera's raw frame
for one (bout, fly, frame), montaged into a single PNG.

Behavior ported from docs/plans/viz-reference/{viz_both_flies,viz_mask_orient,
viz_bodyalign}.py onto viz.core (reproject/overlays/io/layout/colors) instead
of their inline cv2/project_points/hardcoded-path code.

DictConfig note: viz.core.config.courtship_recording() returns a plain dict
(calib_dir/session_dir/predictions_dir/cameras/kp_names) used for path
lookups here. scripts.run_courtship_bout.bout_start_frame additionally needs
the *raw* composed Hydra DictConfig -- it reads cfg.recording.bouts_csv /
cfg.recording.session_dir directly (via jarvis_jax.predict.sam3_driver.
parse_bouts). Rather than re-implement that bouts_csv parsing here,
_compose_cfg() below re-composes `courtship_pipeline` a second time (a cheap
in-memory YAML merge, no extra artifact I/O) reusing core.config._CFG_DIR so
the configs/ directory path stays defined in exactly one place. Composing
twice per invocation is deliberate: it keeps core.config's public API (Step 1
of the task brief) untouched and avoids duplicating bout_start_frame's logic.

scripts.run_courtship_bout pulls in jax/mujoco/egl at import time (heavy), so
it -- and only it -- is imported lazily inside run(), not at module scope.
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
from viz.core.config import courtship_recording, _CFG_DIR

_AXIS_COLOR = (0, 255, 255)  # yellow tail->head arrow; view-local (not in the shared PALETTE)
_NODATA_COLOR = (0, 0, 255)


def umeyama(X, Y):
    """Similarity transform (s, R, t) such that Y ~= s * (R @ X) + t, least
    squares. Ported verbatim from docs/plans/viz-reference/viz_bodyalign.py."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc = X - muX, Y - muY
    U, S, Vt = np.linalg.svd((Yc.T @ Xc) / len(X))
    d = np.sign(np.linalg.det(U @ Vt))
    Dg = np.diag([1, 1, d])
    R = U @ Dg @ Vt
    s = (S * np.array([1, 1, d])).sum() / (Xc ** 2).sum() * len(X)
    t = muY - s * R @ muX
    return s, R, t


def _compose_cfg():
    """Re-compose the raw `courtship_pipeline` DictConfig, needed only for
    scripts.run_courtship_bout.bout_start_frame. Reuses core.config._CFG_DIR
    (single source of truth for the configs/ path)."""
    os.environ.setdefault("USER", "eabe")
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(_CFG_DIR)):
        return compose(config_name="courtship_pipeline")


def run(args):
    # cwd-independence: repo_root is 3 levels up from this file
    # (viz/views/overlay.py -> viz/views -> viz -> repo root).
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    rec = courtship_recording()
    all_cameras = list(rec["cameras"])
    kp_names = list(rec["kp_names"])
    bout, fly, fr = int(args.bout), int(args.fly), int(args.frame)

    cfg = _compose_cfg()
    from scripts.run_courtship_bout import bout_start_frame  # lazy: pulls in jax/mujoco/egl
    start = bout_start_frame(cfg, bout)
    frame_idx = start + fr

    run_root = args.run
    outputs_path = os.path.join(vio.fly_dir(run_root, bout, fly), "outputs.h5")
    if not os.path.exists(outputs_path):
        raise FileNotFoundError(f"missing outputs.h5 for bout={bout} fly={fly}: {outputs_path}")
    outs = vio.load_outputs(run_root, bout, fly)
    mesh_mm, kp3d_mm = outs["mesh_mm"], outs["kp3d_mm"]
    T = mesh_mm.shape[0]
    if not (0 <= fr < T):
        raise IndexError(f"frame {fr} out of range for bout {bout} fly {fly} (T={T})")

    try:
        kp2d, _conf2d = vio.load_kp2d(run_root, bout, fly)
    except (FileNotFoundError, OSError) as e:
        print(f"[overlay] warning: kp2d.npz unavailable ({e}); 'kp' overlay disabled")
        kp2d = None

    show = {tok.strip() for tok in (args.show or "").split(",") if tok.strip()}

    masks = None
    if "mask" in show:
        try:
            masks = vio.load_masks(rec["predictions_dir"], bout, fly, all_cameras)
        except Exception as e:
            print(f"[overlay] warning: sam3_masks.npz unavailable ({e}); 'mask' overlay disabled")

    compare_mesh = None
    if args.compare:
        try:
            compare_mesh = vio.load_outputs(args.compare, bout, fly)["mesh_mm"]
        except Exception as e:
            print(f"[overlay] warning: --compare run unavailable ({e}); skipping compare overlay")

    groups = vcolors.keypoint_groups(kp_names)
    head_idx = groups["head"]
    body_idx = groups["head"] + groups["thorax"] + groups["abdomen"]
    tail_idx = kp_names.index("Abd_tip") if "Abd_tip" in kp_names else None

    mm_al, bodyalign_note = None, None
    if args.bodyalign:
        try:
            kp3d_det, _ = vio.load_kp3d(run_root, bout, fly)
            mm_fr, det_fr = kp3d_mm[fr], kp3d_det[fr]
            ok = np.isfinite(mm_fr).all(-1) & np.isfinite(det_fr).all(-1)
            b = [i for i in body_idx if i < len(ok) and ok[i]]
            if len(b) >= 3:
                s, R, t = umeyama(mm_fr[b], det_fr[b])
                mm_al = s * (R @ mm_fr.T).T + t
                bodyalign_note = f"bodyalign s={s:.3f} |t|={np.linalg.norm(t):.1f} n={len(b)}"
            else:
                print(f"[overlay] warning: only {len(b)} valid body sites (<3); --bodyalign skipped")
                bodyalign_note = "bodyalign: insufficient body sites"
        except (FileNotFoundError, OSError) as e:
            print(f"[overlay] warning: kp3d.npz unavailable for --bodyalign ({e}); skipping")
            bodyalign_note = "bodyalign: kp3d.npz missing"

    cam_mats, cam_names = reproject.camera_matrices(rec["calib_dir"])
    cam_mat_idx = {n: i for i, n in enumerate(cam_names)}
    want_cams = set(args.cams) if args.cams else None

    tiles = []
    for ci, cam in enumerate(all_cameras):
        if want_cams is not None and cam not in want_cams:
            continue
        cmi = cam_mat_idx.get(cam)
        if cmi is None:
            print(f"[overlay] warning: camera {cam} missing from calibration; skipping tile")
            continue

        video_path = os.path.join(rec["session_dir"], f"{cam}.mp4")
        try:
            bgr = vio.read_frame(video_path, frame_idx)
        except Exception as e:
            print(f"[overlay] warning: cannot read frame {frame_idx} for {cam} ({e}); skipping tile")
            continue

        cam_mat = cam_mats[cmi]
        pts_for_crop = []
        notes = []

        if "mesh" in show:
            try:
                m = mesh_mm[fr]
                m = m[np.isfinite(m).all(-1)]
                if len(m):
                    uv = reproject.project(cam_mat, m)
                    overlays.draw_cloud(bgr, uv, vcolors.PALETTE[f"fly{fly}"])
                    pts_for_crop.append(uv)
            except Exception as e:
                print(f"[overlay] warning: mesh overlay failed for {cam}: {e}")
                notes.append(("mesh error", vcolors.PALETTE[f"fly{fly}"]))

        if "mask" in show:
            try:
                if masks is not None and masks["valid"][fr, ci] and masks["masks"][fr, ci].any():
                    overlays.draw_mask(bgr, masks["masks"][fr, ci], vcolors.PALETTE["mask"])
                elif masks is None:
                    notes.append(("no mask data", vcolors.PALETTE["mask"]))
            except Exception as e:
                print(f"[overlay] warning: mask overlay failed for {cam}: {e}")
                notes.append(("mask error", vcolors.PALETTE["mask"]))

        if "kp" in show:
            try:
                if kp2d is not None and kp2d.shape[-2] > 0:
                    uv = kp2d[fr, ci]
                    if np.isfinite(uv).any():
                        overlays.draw_points(bgr, uv, vcolors.PALETTE["detector"])
                        pts_for_crop.append(uv)
                    else:
                        notes.append(("no kp data", vcolors.PALETTE["detector"]))
                else:
                    notes.append(("no kp data", vcolors.PALETTE["detector"]))
            except Exception as e:
                print(f"[overlay] warning: kp overlay failed for {cam}: {e}")
                notes.append(("kp error", vcolors.PALETTE["detector"]))

        if "axis" in show:
            try:
                hh = kp3d_mm[fr][head_idx]
                hh = hh[np.isfinite(hh).all(-1)]
                tt = kp3d_mm[fr][tail_idx] if tail_idx is not None else None
                if len(hh) and tt is not None and np.isfinite(tt).all():
                    huv = reproject.project(cam_mat, hh)
                    hc = huv.mean(0)
                    tuv = reproject.project(cam_mat, tt[None])[0]
                    overlays.draw_points(bgr, huv, vcolors.PALETTE["head"])
                    overlays.draw_points(bgr, tuv[None], vcolors.PALETTE["tail"])
                    overlays.draw_axis(bgr, tuv, hc, _AXIS_COLOR)
                    pts_for_crop.append(np.vstack([huv, tuv[None]]))
                else:
                    notes.append(("no axis data", _AXIS_COLOR))
            except Exception as e:
                print(f"[overlay] warning: axis overlay failed for {cam}: {e}")
                notes.append(("axis error", _AXIS_COLOR))

        if compare_mesh is not None:
            try:
                cm = compare_mesh[fr] if fr < compare_mesh.shape[0] else np.zeros((0, 3))
                cm = cm[np.isfinite(cm).all(-1)]
                if len(cm):
                    uv = reproject.project(cam_mat, cm)
                    overlays.draw_cloud(bgr, uv, vcolors.PALETTE["fit"])
                    pts_for_crop.append(uv)
            except Exception as e:
                print(f"[overlay] warning: compare overlay failed for {cam}: {e}")
                notes.append(("compare error", vcolors.PALETTE["fit"]))

        if args.bodyalign and mm_al is not None:
            try:
                uv = reproject.project(cam_mat, mm_al)
                overlays.draw_points(bgr, uv, vcolors.PALETTE["fit"], radius=2)
                pts_for_crop.append(uv)
            except Exception as e:
                print(f"[overlay] warning: bodyalign overlay failed for {cam}: {e}")
                notes.append(("bodyalign error", vcolors.PALETTE["fit"]))

        if pts_for_crop:
            allpts = np.concatenate(pts_for_crop, axis=0)
        else:
            allpts = np.array([[bgr.shape[1] / 2.0, bgr.shape[0] / 2.0]])
            notes.append(("no data", _NODATA_COLOR))

        crop, _ = layout.crop_to_points(bgr, allpts)
        overlays.legend(crop, [(cam, (0, 255, 255))] + notes)
        tiles.append(crop)

    if not tiles:
        raise RuntimeError(
            f"no camera tiles rendered for run={run_root} bout={bout} fly={fly} frame={fr} "
            f"(all camera videos missing/unreadable under {rec['session_dir']})")

    mont = layout.montage(tiles, cols=min(2, len(tiles)))

    legend_items = [(f"bout{bout} fly{fly} frame{fr} (abs {frame_idx})", (255, 255, 255)),
                    (f"show={args.show}", (200, 200, 200))]
    if "mesh" in show:
        legend_items.append((f"mesh=fly{fly}", vcolors.PALETTE[f"fly{fly}"]))
    if "kp" in show:
        legend_items.append(("kp=detector", vcolors.PALETTE["detector"]))
    if "mask" in show:
        legend_items.append(("mask", vcolors.PALETTE["mask"]))
    if "axis" in show:
        legend_items.append(("axis=tail->head", _AXIS_COLOR))
    if compare_mesh is not None:
        legend_items.append(("compare=fit", vcolors.PALETTE["fit"]))
    if args.bodyalign:
        legend_items.append((bodyalign_note or "bodyalign", vcolors.PALETTE["fit"]))

    band = layout.banner(mont.shape[1], legend_items)
    out_img = np.vstack([mont, band])

    out_path = args.out or f"overlay_bout{bout}_fly{fly}_f{fr}.png"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, out_img)
    return 0
