"""sidebyside view: a side-by-side QC video for one (bout, fly).

LEFT : raw camera video + SAM mask overlay + ViTPose 2D keypoint SKELETON.
RIGHT: the fitted body rendered in MuJoCo through a camera built from the LEFT
       camera's own calibration (`--right rigcam`, the default), so the two
       panels show the fly from the same viewpoint at the same pixels. The rig
       uses TELECENTRIC lenses, hence an affine calibration and an orthographic
       MuJoCo camera -- see viz.core.mjcam.

`--right reproj` draws the fitted mesh/sites reprojected as points instead of
rendering them (no MuJoCo needed, useful when outputs.h5 is absent).

`--right mujoco` restores the old panel: a MuJoCo render from `track1`, a
model-space camera (`<camera name="track1" ... mode="trackcom">` in the body
XML) whose extrinsics have NOTHING to do with the left camera. Those two views
are unrelated, so a fly can appear rotated or mirrored between them for purely
geometric reasons -- which read as the IK "facing the wrong way" when the fit
was in fact correct (measured on Session1/2026_04_02_14_54_28 bout_00018 fly0:
fitted vs observed body axis 1.3 deg, wings 0.8-2.4 deg, 0% anti-aligned, and
2.8-13.4 deg wing agreement reprojected into all 7 real cameras).

Same frames on both panels, concatenated into one mp4 (plus a representative
still PNG) so the 2D evidence (left) and the IK fit (right) can be compared
directly. A FIXED crop window is computed once over the whole segment so the
left panel keeps a constant size for every frame -- a per-frame crop would
change the panel dimensions and break the video encoder.

Ported from a working scratchpad script (male_sidebyside.py) onto viz.core
(reproject/overlays/colors/io/layout) and viz.config (recording paths) instead
of that script's hardcoded RUN/SESS/XML/CAM paths. The left camera is
auto-picked as the one with the highest median 2D confidence over the segment
(as in the reference). The 2D skeleton edges are built from kp_names
(core.colors.leg_chains + the body/wing name pairs) rather than a hand-rolled
index list.

Env / import ordering (load-bearing, matches viz/views/fit_check.py): MUJOCO_GL
and PYOPENGL_PLATFORM must be "egl" BEFORE mujoco is imported anywhere, so they
are set at module import time here. The heavy render path
(stac_mjx.stac.Stac -> jax + mujoco) and scripts.run_bout
(bout_start_frame, which also pulls in jax/mujoco/egl) are imported lazily
inside run(), keeping `import viz.views.sidebyside` itself cheap under
JAX_PLATFORMS=cpu (same lazy-import convention as the sibling views).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys

import cv2
import h5py
import numpy as np
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

from viz.core import colors as vcolors
from viz.core import io as vio
from viz.core import layout
from viz.core import overlays
from viz.core import reproject
from viz.core import rigviews
from viz.config import courtship_recording, resolve_body_model_xml, _CFG_DIR

# BGR per-group keypoint colours (matches the reference script's palette;
# distinct enough that legs/head/abdomen/thorax stay legible on the raw frame).
_GROUP_COLORS = {
    "legs": (0, 215, 255),     # amber
    "head": (0, 0, 255),       # red
    "abdomen": (255, 0, 0),    # blue
    "thorax": (0, 255, 0),     # green (thorax + wings)
}
_SKELETON_COLOR = (200, 200, 200)  # grey chain lines
_MASK_FILL = (180, 120, 60)        # BGR fill for the SAM mask overlay
# Repo visual language (viz/core/colors.py): grey = mesh, green = fit.
_MESH_COLOR = (170, 170, 170)      # BGR: reprojected fitted mesh cloud
_FIT_COLOR = (90, 220, 90)         # BGR: fitted 3D sites + their chains

# Measured-vs-fitted marker colors, used by the multi-view draw_right committed
# in 9a13b23. They were only ever defined in an uncommitted working-tree hunk,
# so HEAD raised NameError on any --views render; defining them here makes the
# committed multi-view path self-contained.
_MEAS_WING = (255, 0, 255)         # magenta: measured wing keypoint
_FIT_WING = (0, 255, 255)          # yellow:  fitted   wing marker
_MEAS_OTHER = (200, 120, 60)       # dim blue-grey: measured, non-wing
_FIT_OTHER = (120, 200, 120)       # dim green:     fitted,   non-wing

# --verify overlay on the LEFT (video) panel: a three-level check in one frame.
# 2D identity/anatomy (are the named landmarks on the right body parts, and is
# left/right cleanly separated?) plus 3D consistency (does triangulated kp3d
# reproject onto its own detections?). BGR, so these really are blue/orange.
_VER_L = (255, 120, 0)             # BLUE   : a LEFT-side keypoint
_VER_R = (0, 180, 255)             # ORANGE : a RIGHT-side keypoint
_VER_MID = (220, 220, 220)         # grey   : midline / unsided
_VER_KP3D = (255, 0, 255)          # magenta cross: measured kp3d reprojected
# Sparse labels: the landmarks that would expose a keypoint-ORDER scramble
# (eyes on legs) or a LEFT/RIGHT swap. Labelling all 50 is unreadable.
_VER_LABELS = {"Antenna_Base": "ANT", "EyeL": "EyeL", "EyeR": "EyeR",
               "WingL_base": "WgL", "WingR_base": "WgR", "Abd_tip": "ABDtip",
               "T1L_TaTip": "T1L", "T1R_TaTip": "T1R",
               "T3L_TaTip": "T3L", "T3R_TaTip": "T3R"}


def _ver_side_color(name):
    if name.startswith(("T1L", "T2L", "T3L", "WingL", "EyeL")):
        return _VER_L
    if name.startswith(("T1R", "T2R", "T3R", "WingR", "EyeR")):
        return _VER_R
    return _VER_MID


def _draw_verify(bgr, uv, meas_uv, kp_names):
    """LEFT-panel verification overlay: L/R-coloured detections, magenta
    crosses for reprojected measured kp3d, sparse anatomy labels.

    The gap between a coloured dot and its magenta cross is the TRIANGULATION
    residual for that keypoint in this camera (male median ~2 px). Blue and
    orange interleaved on one flank means a left/right swap; a label sitting on
    the wrong body part means the detector keypoint ORDER is wrong. Neither is
    visible to any metric in qc.json -- reorder_detector_to_model depends on the
    checkpoint's training order, so swapping checkpoints can silently scramble
    anatomy, and this overlay is the gate for that.
    """
    for i, n in enumerate(kp_names):
        if not np.isfinite(uv[i]).all():
            continue
        col = _ver_side_color(n)
        cv2.circle(bgr, (int(uv[i][0]), int(uv[i][1])), 4, col, -1)
        if meas_uv is not None and np.isfinite(meas_uv[i]).all():
            cv2.drawMarker(bgr, (int(meas_uv[i][0]), int(meas_uv[i][1])),
                           _VER_KP3D, cv2.MARKER_CROSS, 9, 1)
        lab = _VER_LABELS.get(n)
        if lab:
            cv2.putText(bgr, lab, (int(uv[i][0]) + 6, int(uv[i][1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)





def _compose_cfg():
    """Re-compose the raw `pipeline` DictConfig, needed only for
    scripts.run_bout.bout_start_frame. Reuses viz.config._CFG_DIR
    (single source of truth for the configs/ path); mirrors viz/views/legskel.py.
    """
    os.environ.setdefault("USER", "eabe")
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(_CFG_DIR)):
        return compose(config_name="pipeline")


def _skeleton_edges(kp_names):
    """Index pairs for the 2D skeleton: fixed body/wing chain + per-leg
    proximal->distal chains (from core.colors.leg_chains), each rooted at the
    Scutellum. Missing names are silently dropped so this works for any
    keypoint ordering."""
    idx = {n: i for i, n in enumerate(kp_names)}

    def E(a, b):
        return (idx[a], idx[b]) if a in idx and b in idx else None

    edges = [E("EyeL", "Antenna_Base"), E("EyeR", "Antenna_Base"),
             E("Antenna_Base", "Scutellum"),
             E("Scutellum", "WingL_base"), E("WingL_base", "WingL_V12"), E("WingL_V12", "WingL_V13"),
             E("Scutellum", "WingR_base"), E("WingR_base", "WingR_V12"), E("WingR_V12", "WingR_V13"),
             E("Scutellum", "Abd_A4"), E("Abd_A4", "Abd_tip")]
    for chain in vcolors.leg_chains(kp_names).values():
        if "Scutellum" in idx:
            edges.append((idx["Scutellum"], chain[0]))
        edges += [(chain[k], chain[k + 1]) for k in range(len(chain) - 1)]
    return [e for e in edges if e is not None]


def _band(width, text):
    """A titled black bar (reuses core.layout.banner + overlays.legend)."""
    return layout.banner(width, [(text, (255, 255, 255))])


def run(args):
    # Opt-in multi-view mode (--views left,top,right ...): a completely
    # separate code path below (_run_multiview), so the single-view flow that
    # follows is untouched byte-for-byte when --views is absent.
    views_arg = getattr(args, "views", None)
    if views_arg:
        return _run_multiview(args, views_arg)
    # cwd-independence: repo_root is 3 levels up from this file
    # (viz/views/sidebyside.py -> viz/views -> viz -> repo root); needed so the
    # lazy `scripts.run_bout` import below resolves.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    rec = courtship_recording()
    all_cameras = list(rec["cameras"])
    # Recording overrides: without these the view uses the DEFAULT recording
    # (Session0) -> wrong videos/masks/frames for any other recording.
    session_dir = getattr(args, "session_dir", None) or rec["session_dir"]
    predictions_dir = getattr(args, "predictions_dir", None) or rec["predictions_dir"]
    # The calibration MUST follow the session override too. Taking it from
    # `rec` (the DEFAULT recording) silently rendered the right panel from
    # Session0's camera poses for a Session1 bout -- the panels then showed the
    # fly from two different viewpoints again, which is the whole failure this
    # view exists to remove. Default to the overridden session's own dir.
    calib_dir = getattr(args, "calib_dir", None)
    if calib_dir is None:
        calib_dir = (os.path.join(session_dir, "calibration")
                     if getattr(args, "session_dir", None) else rec["calib_dir"])
    if not os.path.isdir(calib_dir):
        raise FileNotFoundError(f"calibration dir not found: {calib_dir}")
    bout, fly = int(args.bout), int(args.fly)
    T0 = int(args.start)
    conf_thr = float(getattr(args, "conf", 0.3) or 0.3)
    panel_h = int(getattr(args, "panel_h", 480) or 480)
    render_cam = args.camera if getattr(args, "camera", None) is not None else "track1"

    # --- 2D detector keypoints (T,C,K,2) + confidences (T,C,K) ---
    kp2d, conf2d = vio.load_kp2d(args.run, bout, fly)
    T2, C = kp2d.shape[0], kp2d.shape[1]

    # --- fixed STAC/IK fit (qpos etc.) from the bout's stac_ik.h5 ---
    stac_h5 = os.path.join(vio.fly_dir(args.run, bout, fly), "stac_ik.h5")
    if not os.path.exists(stac_h5):
        raise FileNotFoundError(f"missing stac_ik.h5 for bout={bout} fly={fly}: {stac_h5}")
    with h5py.File(stac_h5, "r") as f:
        cfg = OmegaConf.create(f["config"][()].decode())
        kp_names = [s.decode() for s in f["kp_names"][:]]
        qpos = np.asarray(f["qpos"][:])
        kp_data = np.asarray(f["kp_data"][:])
        offsets = np.asarray(f["offsets"][:])

    # Clamp the segment to what both the detector array and the IK fit cover.
    T = min(T2, qpos.shape[0], kp_data.shape[0])
    if T0 < 0 or T0 >= T:
        raise ValueError(f"start {T0} out of range for bout {bout} fly {fly} (usable T={T})")
    N = int(args.n) if getattr(args, "n", None) else (T - T0)
    N = min(N, T - T0)
    if N <= 0:
        raise ValueError(f"no frames to render (start={T0}, n={N}, usable T={T})")

    edges = _skeleton_edges(kp_names)
    groups = vcolors.keypoint_groups(kp_names)

    # --- pick the LEFT camera: highest median 2D confidence over the segment ---
    want = set(args.cams) if getattr(args, "cams", None) else None
    cand = [i for i, c in enumerate(all_cameras) if want is None or c in want]
    if not cand:
        raise ValueError(f"no requested cameras {sorted(want)} in recording cameras {all_cameras}")
    seg = slice(T0, T0 + N)
    scores = {i: float(np.median(conf2d[seg, i].mean(-1))) for i in cand}
    lc = max(scores, key=scores.get)
    left_cam = all_cameras[lc]
    print(f"[sidebyside] left cam {left_cam} (conf {scores[lc]:.2f}); segment t[{T0}:{T0 + N}]")

    # --- SAM masks (T,C,H,W) for this fly, indexed against the full camera list ---
    try:
        masks = vio.load_masks(predictions_dir, bout, fly, all_cameras)
        mk, mv, H, W = masks["masks"], masks["valid"], masks["H"], masks["W"]
    except Exception as e:
        print(f"[sidebyside] warning: masks unavailable ({e}); rendering without SAM overlay")
        mk = mv = None
        cap = cv2.VideoCapture(os.path.join(session_dir, f"{left_cam}.mp4"))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        cap.release()

    # --- absolute start frame of this bout (to seek the raw video) ---
    if getattr(args, "start_frame", None) is not None:
        start_abs = int(args.start_frame)
    else:
        from scripts.run_bout import bout_start_frame  # lazy: pulls in jax/mujoco/egl
        start_abs = bout_start_frame(_compose_cfg(), bout)

    # --- RIGHT panel ---
    right_mode = getattr(args, "right", None) or "rigcam"
    out_path = args.out or f"sidebyside_bout{bout}_fly{fly}.mp4"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    mesh_mm = fitted_mm = None
    mj_render_path = None
    rframes = None
    rig = None
    if right_mode == "rigcam":
        outs_path = os.path.join(vio.fly_dir(args.run, bout, fly), "outputs.h5")
        qref_path = os.path.join(vio.fly_dir(args.run, bout, fly), "qpos_refined.npz")
        if not (os.path.exists(outs_path) and os.path.exists(qref_path)):
            print(f"[sidebyside] --right rigcam needs outputs.h5 + qpos_refined.npz; "
                  f"falling back to --right reproj")
            right_mode = "reproj"
        else:
            import mujoco
            from viz.core.mjcam import (mujoco_camera_from_affine,
                                        similarity_from_points)
            _outs = vio.load_outputs(args.run, bout, fly)
            rig_world = np.asarray(_outs["kp3d_mm"], float)
            rig_qpos = np.asarray(np.load(qref_path)["qpos"], float)
            _cam_mats, _cam_names = reproject.camera_matrices(calib_dir)
            _cam_names = list(_cam_names)
            if left_cam not in _cam_names:
                raise ValueError(f"left camera {left_cam} not in calibration {_cam_names}")
            _spec = mujoco.MjSpec.from_file(resolve_body_model_xml(cfg.model.MJCF_PATH))
            _spec.visual.global_.offwidth = int(W)
            _spec.visual.global_.offheight = int(H)
            _c = _spec.worldbody.add_camera()
            _c.name = "rigcam"
            _c.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
            _c.fovy = 1.0
            _c.pos = [0.0, 0.0, 1.0]
            _mj = _spec.compile()
            _dat = mujoco.MjData(_mj)
            _sn = [mujoco.mj_id2name(_mj, mujoco.mjtObj.mjOBJ_SITE, i)
                   for i in range(_mj.nsite)]
            _pairs = [(_sn.index(f"tracking[{n}]"), i) for i, n in enumerate(kp_names)
                      if f"tracking[{n}]" in _sn]
            rig = dict(
                mj=_mj, dat=_dat, mujoco=mujoco,
                cid=mujoco.mj_name2id(_mj, mujoco.mjtObj.mjOBJ_CAMERA, "rigcam"),
                sidx=[a for a, _ in _pairs], kidx=[b for _, b in _pairs],
                cam_mat=_cam_mats[_cam_names.index(left_cam)],
                qpos=rig_qpos, world=rig_world,
                renderer=mujoco.Renderer(_mj, height=int(H), width=int(W)),
                cam_from_affine=mujoco_camera_from_affine,
                similarity=similarity_from_points)
            print(f"[sidebyside] right panel: MuJoCo through a camera built from "
                  f"{left_cam}'s calibration (orthographic / telecentric)", flush=True)
    if right_mode == "reproj":
        # View-matched: the fitted mesh through the LEFT camera's own DLT
        # matrix. Same pixels as the left panel, so mask-vs-mesh orientation is
        # a real comparison rather than two unrelated viewpoints.
        outs_path = os.path.join(vio.fly_dir(args.run, bout, fly), "outputs.h5")
        if not os.path.exists(outs_path):
            raise FileNotFoundError(
                f"--right reproj needs outputs.h5 (the fitted mesh) for bout={bout} "
                f"fly={fly}: {outs_path}. Use --right mujoco to render from the "
                f"model-space camera instead (not view-matched).")
        _outs = vio.load_outputs(args.run, bout, fly)
        mesh_mm, fitted_mm = _outs["mesh_mm"], _outs["kp3d_mm"]
        _cam_mats, _cam_names = reproject.camera_matrices(calib_dir)
        if left_cam not in list(_cam_names):
            raise ValueError(f"left camera {left_cam} not in calibration {list(_cam_names)}")
        left_cam_mat = _cam_mats[list(_cam_names).index(left_cam)]
        print(f"[sidebyside] right panel: fitted mesh reprojected into {left_cam} "
              f"(view-matched)", flush=True)
    elif right_mode == "mujoco":
        from stac_mjx.stac import Stac  # heavy (jax + mujoco); lazy on purpose
        xml_path = resolve_body_model_xml(cfg.model.MJCF_PATH)
        stac = Stac(str(xml_path), cfg, kp_names)
        print("[sidebyside] rendering MuJoCo IK (NOT view-matched) ...", flush=True)
    # stac.render REQUIRES a save_path (imageio sniffs the .mp4 extension), but
    # we only consume the returned frames (`rframes`) for compositing -- so this
    # is a throwaway intermediate; keep a clean name (no double .mp4) and delete
    # it after the final side-by-side is written.
    if right_mode == "mujoco":
        mj_render_path = (out_path[:-4] if out_path.endswith(".mp4") else out_path) + ".mjrender.mp4"
        rframes = list(stac.render(
            qpos, kp_data, offsets, n_frames=N,
            save_path=mj_render_path,
            start_frame=T0, camera=render_cam,
            height=panel_h, width=int(panel_h * 1.33), show_marker_error=False))

    # --- FIXED crop window over the whole segment (constant left-panel size) ---
    bx0, by0, bx1, by1 = W, H, 0, 0
    for t in range(T0, T0 + N):
        if mk is not None and mv[t, lc] and mk[t, lc].any():
            ys, xs = np.where(mk[t, lc])
            bx0, bx1 = min(bx0, xs.min()), max(bx1, xs.max())
            by0, by1 = min(by0, ys.min()), max(by1, ys.max())
        vis = conf2d[t, lc] >= conf_thr
        p = kp2d[t, lc][vis]
        if len(p):
            bx0, bx1 = min(bx0, p[:, 0].min()), max(bx1, p[:, 0].max())
            by0, by1 = min(by0, p[:, 1].min()), max(by1, p[:, 1].max())
    if bx1 <= bx0 or by1 <= by0:  # no mask + no visible kp anywhere in segment
        bx0, by0, bx1, by1 = 0, 0, W, H
    pad = 45
    cx0, cy0 = int(max(bx0 - pad, 0)), int(max(by0 - pad, 0))
    cx1, cy1 = int(min(bx1 + pad, W)), int(min(by1 + pad, H))
    left_w = int(round((cx1 - cx0) / (cy1 - cy0) * panel_h))
    print(f"[sidebyside] left crop x[{cx0}:{cx1}] y[{cy0}:{cy1}] -> {left_w}x{panel_h}")

    def _draw_left(bgr, t):
        if mk is not None and mv[t, lc]:
            overlays.draw_mask(bgr, mk[t, lc], _MASK_FILL)
        uv = np.asarray(kp2d[t, lc], float).copy()
        uv[conf2d[t, lc] < conf_thr] = np.nan  # hide low-conf kp from chain + dots
        for a, b in edges:
            if np.isfinite(uv[a]).all() and np.isfinite(uv[b]).all():
                overlays.draw_chain(bgr, [uv[a], uv[b]], _SKELETON_COLOR)
        for grp, color in _GROUP_COLORS.items():
            if groups[grp]:
                overlays.draw_points(bgr, uv[groups[grp]], color, radius=2)
        if verify and kp_names is not None:
            meas_uv = None
            if rig.get("meas") is not None and t < len(rig["meas"]):
                meas_uv = reproject.project(rig["cam_mat"],
                                            np.nan_to_num(rig["meas"][t]))
            _draw_verify(bgr, uv, meas_uv, kp_names)
        crop = bgr[cy0:cy1, cx0:cx1]
        return cv2.resize(crop, (left_w, panel_h))

    def _draw_right_rigcam(t):
        """MuJoCo rendered through the LEFT camera's own calibration, cropped
        identically to the left panel so the two are the same pixels."""
        mj, dat, mj_mod = rig["mj"], rig["dat"], rig["mujoco"]
        q = rig["qpos"]
        if t >= len(q) or not np.isfinite(q[t]).all():
            return np.zeros((panel_h, left_w, 3), np.uint8)
        dat.qpos[:] = q[t]
        mj_mod.mj_forward(mj, dat)
        Xm = dat.site_xpos[rig["sidx"]]
        Xw = rig["world"][t][rig["kidx"]]
        ok = np.isfinite(Xw).all(axis=1)
        if ok.sum() < 4:
            return np.zeros((panel_h, left_w, 3), np.uint8)
        sc, R, tr = rig["similarity"](Xm[ok], Xw[ok])
        pos, quat, fovy = rig["cam_from_affine"](
            rig["cam_mat"], (W, H), sc, R, tr, Xm[ok].mean(0), back_off=2.0)
        mj.cam_pos[rig["cid"]] = pos
        mj.cam_quat[rig["cid"]] = quat
        mj.cam_fovy[rig["cid"]] = fovy
        # cam_xpos/cam_xmat are derived in mj_forward: re-run kinematics AFTER
        # editing the camera or the scene keeps the previous pose.
        mj_mod.mj_forward(mj, dat)
        rig["renderer"].update_scene(dat, camera="rigcam")
        bgr = cv2.cvtColor(rig["renderer"].render(), cv2.COLOR_RGB2BGR)
        if mk is not None and mv[t, lc]:
            overlays.draw_mask(bgr, mk[t, lc], _MASK_FILL, alpha=0.0)
        crop = bgr[cy0:cy1, cx0:cx1]
        return cv2.resize(crop, (left_w, panel_h))

    def _draw_right_reproj(bgr, t):
        """Fitted mesh + fitted sites in the LEFT camera, same crop as _draw_left."""
        if t < len(mesh_mm) and np.isfinite(mesh_mm[t]).all():
            overlays.draw_cloud(bgr, reproject.project(left_cam_mat, mesh_mm[t]),
                                _MESH_COLOR)
        if t < len(fitted_mm):
            fm = np.asarray(fitted_mm[t], float)
            if np.isfinite(fm).any():
                uv = reproject.project(left_cam_mat, np.nan_to_num(fm))
                uv[~np.isfinite(fm).all(axis=1)] = np.nan
                for a, b in edges:
                    if np.isfinite(uv[a]).all() and np.isfinite(uv[b]).all():
                        overlays.draw_chain(bgr, [uv[a], uv[b]], _FIT_COLOR)
                overlays.draw_points(bgr, uv, _FIT_COLOR, radius=2)
        crop = bgr[cy0:cy1, cx0:cx1]
        return cv2.resize(crop, (left_w, panel_h))

    # --- composite left+right per frame, collect BGR frames, write mp4 + still ---
    left_title = f"{left_cam} video + SAM mask + ViTPose 2D skeleton"
    right_title = {
        "rigcam": f"MuJoCo IK @ {left_cam} rig camera (same view) + SAM outline",
        "reproj": f"fitted mesh + sites reprojected into {left_cam} (same view)",
    }.get(right_mode, f"MuJoCo IK render ({render_cam}) -- NOT view-matched")
    frames_out = []
    # Sync-aware, like maskvid: cameras drop frames independently, so a
    # positional read shows the mask/keypoint overlay on the WRONG frame for
    # any camera that dropped one earlier in the recording. The pipeline's own
    # frame reads (run_bout) are already sync-aware, so a positional read here
    # disagreed with the very data it draws -- and this is the video the
    # fly-ID review GUI shows, so a reviewer saw masks that appeared out of
    # sync with the video. 25 courtship bouts start after a recorded drop.
    # Falls back to positional where no sync plan exists (all of Session0).
    left_stream = vio.read_frames_synced(session_dir, [left_cam], start_abs + T0, N)
    for k, imgs in enumerate(left_stream):
        rgb = imgs[0]
        bgr = (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb is not None
               else np.zeros((H, W, 3), np.uint8))
        L = _draw_left(bgr.copy(), T0 + k)
        if right_mode == "rigcam":
            R = _draw_right_rigcam(T0 + k)
        elif right_mode == "reproj":
            R = _draw_right_reproj(bgr.copy(), T0 + k)
        else:
            R = cv2.cvtColor(np.asarray(rframes[k]), cv2.COLOR_RGB2BGR)
            if R.shape[0] != panel_h:
                _s = panel_h / R.shape[0]
                R = cv2.resize(R, (int(R.shape[1] * _s), panel_h))
        Lh = np.vstack([_band(L.shape[1], left_title), L])
        Rh = np.vstack([_band(R.shape[1], right_title), R])
        h = max(Lh.shape[0], Rh.shape[0])

        def _padh(im):
            if im.shape[0] < h:
                return np.vstack([im, np.zeros((h - im.shape[0], im.shape[1], 3), np.uint8)])
            return im

        sep = np.full((h, 3, 3), 60, np.uint8)
        frames_out.append(np.hstack([_padh(Lh), sep, _padh(Rh)]))

    if not frames_out:
        raise RuntimeError(f"no frames rendered for bout {bout} fly {fly} (left cam {left_cam})")

    vio.write_video(out_path, frames_out, fps=int(getattr(args, "fps", 30) or 30))
    if mj_render_path:
        try:
            os.remove(mj_render_path)   # throwaway MuJoCo intermediate
        except OSError:
            pass
    still_path = os.path.splitext(out_path)[0] + "_still.png"
    cv2.imwrite(still_path, frames_out[len(frames_out) // 2])
    print(f"[sidebyside] wrote {out_path} ({len(frames_out)} frames) + {still_path}")
    return 0


def _resolve_multiview_cameras(calib_dir, all_cameras, views_arg):
    """Turn `--views` (role names and/or explicit camera names, comma/space
    separated) into an ordered [(camera_name, role_label, RigView), ...],
    using rigviews.classify_views to derive top/left/right from the
    calibration geometry -- never a hardcoded camera name."""
    triple = rigviews.classify_views(calib_dir)  # {"top"/"left"/"right": RigView}
    by_name = {rv.name: (role, rv) for role, rv in triple.items()}
    tokens = [t for t in str(views_arg).replace(",", " ").split() if t]
    if not tokens:
        raise ValueError(f"--views got no usable camera/role tokens from {views_arg!r}")
    resolved = []
    for tok in tokens:
        low = tok.lower()
        if low in triple:
            role, rv = low, triple[low]
        elif tok in by_name:
            role, rv = by_name[tok]
        elif tok in all_cameras:
            vdirs = rigviews.view_directions(calib_dir)
            if tok not in vdirs:
                raise ValueError(f"--views camera {tok!r} not found in calibration {calib_dir}")
            v = vdirs[tok]
            elev = float(np.degrees(np.arcsin(np.clip(abs(v[2]), -1.0, 1.0))))
            role, rv = "view", rigviews.RigView(tok, v, elev)
        else:
            raise ValueError(
                f"--views token {tok!r} is neither a role ({sorted(triple)}) nor a "
                f"camera in this recording ({all_cameras})")
        resolved.append((rv.name, role, rv))
    return resolved


def _build_multiview_rig(cam_name, cam_mats, cam_names, cfg, kp_names, wing_idx,
                         rig_world, rig_meas, rig_qpos, W, H):
    """Per-camera MuJoCo rigcam state, mirroring the single-view --right rigcam
    setup above (run()) but parameterized so it can be built once per row of a
    multi-view render -- each row's camera gets its OWN `cam_mat`, so its
    render is view-matched to THAT row's own calibration, never a reused one.
    """
    import mujoco
    from viz.core.mjcam import mujoco_camera_from_affine, similarity_from_points
    if cam_name not in cam_names:
        raise ValueError(f"camera {cam_name} not in calibration {cam_names}")
    spec = mujoco.MjSpec.from_file(resolve_body_model_xml(cfg.model.MJCF_PATH))
    spec.visual.global_.offwidth = int(W)
    spec.visual.global_.offheight = int(H)
    c = spec.worldbody.add_camera()
    c.name = "rigcam"
    c.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
    c.fovy = 1.0
    c.pos = [0.0, 0.0, 1.0]
    mj = spec.compile()
    dat = mujoco.MjData(mj)
    sn = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(mj.nsite)]
    pairs = [(sn.index(f"tracking[{n}]"), i) for i, n in enumerate(kp_names)
              if f"tracking[{n}]" in sn]
    return dict(
        mj=mj, dat=dat, mujoco=mujoco,
        cid=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_CAMERA, "rigcam"),
        sidx=[a for a, _ in pairs], kidx=[b for _, b in pairs],
        cam_mat=cam_mats[cam_names.index(cam_name)],
        qpos=rig_qpos, world=rig_world, meas=rig_meas, wing=wing_idx,
        renderer=mujoco.Renderer(mj, height=int(H), width=int(W)),
        cam_from_affine=mujoco_camera_from_affine,
        similarity=similarity_from_points)


def _make_multiview_drawers(cam_idx, rig, mk, mv, kp2d, conf2d, edges, groups,
                            conf_thr, panel_h, W, H, T0, N,
                            kp_names=None, verify=False):
    """Crop window + (draw_left, draw_right) closures for ONE row/camera of a
    multi-view render. Same crop-window and drawing logic as run()'s
    _draw_left / _draw_right_rigcam, generalized over an explicit camera
    index instead of the single auto-picked `lc` -- so every row is cropped
    and drawn from ITS OWN camera's mask/keypoints, not the left camera's.

    The measured-vs-fitted wing overlay (magenta=measured kp3d.npz,
    yellow=fitted outputs.h5 kp3d_mm, dim colors for non-wing, a line joining
    each measured/fitted pair) is reproduced here using the SAME module-level
    colors (_MEAS_WING/_FIT_WING/_MEAS_OTHER/_FIT_OTHER) the single-view path
    uses, so it renders identically in every row.
    """
    bx0, by0, bx1, by1 = W, H, 0, 0
    for t in range(T0, T0 + N):
        if mk is not None and mv[t, cam_idx] and mk[t, cam_idx].any():
            ys, xs = np.where(mk[t, cam_idx])
            bx0, bx1 = min(bx0, xs.min()), max(bx1, xs.max())
            by0, by1 = min(by0, ys.min()), max(by1, ys.max())
        vis = conf2d[t, cam_idx] >= conf_thr
        p = kp2d[t, cam_idx][vis]
        if len(p):
            bx0, bx1 = min(bx0, p[:, 0].min()), max(bx1, p[:, 0].max())
            by0, by1 = min(by0, p[:, 1].min()), max(by1, p[:, 1].max())
    if bx1 <= bx0 or by1 <= by0:
        bx0, by0, bx1, by1 = 0, 0, W, H
    pad = 45
    cx0, cy0 = int(max(bx0 - pad, 0)), int(max(by0 - pad, 0))
    cx1, cy1 = int(min(bx1 + pad, W)), int(min(by1 + pad, H))
    left_w = int(round((cx1 - cx0) / (cy1 - cy0) * panel_h))

    def draw_left(bgr, t):
        if mk is not None and mv[t, cam_idx]:
            overlays.draw_mask(bgr, mk[t, cam_idx], _MASK_FILL)
        uv = np.asarray(kp2d[t, cam_idx], float).copy()
        uv[conf2d[t, cam_idx] < conf_thr] = np.nan
        for a, b in edges:
            if np.isfinite(uv[a]).all() and np.isfinite(uv[b]).all():
                overlays.draw_chain(bgr, [uv[a], uv[b]], _SKELETON_COLOR)
        for grp, color in _GROUP_COLORS.items():
            if groups[grp]:
                overlays.draw_points(bgr, uv[groups[grp]], color, radius=2)
        crop = bgr[cy0:cy1, cx0:cx1]
        return cv2.resize(crop, (left_w, panel_h))

    def draw_right(t):
        mj, dat, mj_mod = rig["mj"], rig["dat"], rig["mujoco"]
        q = rig["qpos"]
        if t >= len(q) or not np.isfinite(q[t]).all():
            return np.zeros((panel_h, left_w, 3), np.uint8)
        dat.qpos[:] = q[t]
        mj_mod.mj_forward(mj, dat)
        Xm = dat.site_xpos[rig["sidx"]]
        Xw = rig["world"][t][rig["kidx"]]
        ok = np.isfinite(Xw).all(axis=1)
        if ok.sum() < 4:
            return np.zeros((panel_h, left_w, 3), np.uint8)
        sc, R, tr = rig["similarity"](Xm[ok], Xw[ok])
        pos, quat, fovy = rig["cam_from_affine"](
            rig["cam_mat"], (W, H), sc, R, tr, Xm[ok].mean(0), back_off=2.0)
        mj.cam_pos[rig["cid"]] = pos
        mj.cam_quat[rig["cid"]] = quat
        mj.cam_fovy[rig["cid"]] = fovy
        mj_mod.mj_forward(mj, dat)
        rig["renderer"].update_scene(dat, camera="rigcam")
        bgr = cv2.cvtColor(rig["renderer"].render(), cv2.COLOR_RGB2BGR)
        if mk is not None and mv[t, cam_idx]:
            overlays.draw_mask(bgr, mk[t, cam_idx], _MASK_FILL, alpha=0.0)
        wing = set(rig.get("wing") or [])
        cm = rig["cam_mat"]
        fit = rig["world"][t] if t < len(rig["world"]) else None
        meas = (rig["meas"][t] if rig.get("meas") is not None
                and t < len(rig["meas"]) else None)
        for arr, wcol, ocol, rad in ((meas, _MEAS_WING, _MEAS_OTHER, 4),
                                     (fit, _FIT_WING, _FIT_OTHER, 3)):
            if arr is None:
                continue
            good = np.isfinite(arr).all(axis=1)
            if not good.any():
                continue
            uv = reproject.project(cm, np.nan_to_num(arr))
            for i, p in enumerate(uv):
                if not good[i]:
                    continue
                overlays.draw_points(bgr, p[None], wcol if i in wing else ocol,
                                     radius=rad)
        if fit is not None and meas is not None:
            uvf = reproject.project(cm, np.nan_to_num(fit))
            uvm = reproject.project(cm, np.nan_to_num(meas))
            for i in wing:
                if np.isfinite(fit[i]).all() and np.isfinite(meas[i]).all():
                    overlays.draw_chain(bgr, [uvm[i], uvf[i]], _MEAS_WING)
        crop = bgr[cy0:cy1, cx0:cx1]
        return cv2.resize(crop, (left_w, panel_h))

    return draw_left, draw_right, left_w


def _run_multiview(args, views_arg):
    """Multi-view sidebyside: one ROW per requested camera role/name, each row
    [that camera's own video+SAM+2D | that SAME camera's view-matched MuJoCo
    rigcam], stacked vertically with viz.core.layout.montage.

    Expectation to check before trusting the render (CLAUDE.md "visualize
    what you change"): in every row's RIGHT panel, the rendered fly should
    sit INSIDE the SAM mask contour drawn on top of it (alpha=0, outline
    only) -- that outline comes from THAT row's own camera, so a correct
    per-row view match puts the render inside its own outline. A render that
    drifts outside its own row's outline means that row was built from the
    wrong camera's calibration -- exactly the bug `--right rigcam` (singular)
    already fixed for the 2-panel case; this must hold independently in every
    row of the 3-view case too.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    right_mode = getattr(args, "right", None) or "rigcam"
    if right_mode != "rigcam":
        raise ValueError(
            f"--views requires --right rigcam (the only view-matched multi-camera "
            f"mode -- 'reproj' has no MuJoCo render and 'mujoco' is NOT view-matched, "
            f"so a per-camera row would be misleading); got --right {right_mode}. "
            f"Drop --views or switch --right to rigcam.")

    rec = courtship_recording()
    all_cameras = list(rec["cameras"])
    session_dir = getattr(args, "session_dir", None) or rec["session_dir"]
    predictions_dir = getattr(args, "predictions_dir", None) or rec["predictions_dir"]
    calib_dir = getattr(args, "calib_dir", None)
    if calib_dir is None:
        calib_dir = (os.path.join(session_dir, "calibration")
                     if getattr(args, "session_dir", None) else rec["calib_dir"])
    if not os.path.isdir(calib_dir):
        raise FileNotFoundError(f"calibration dir not found: {calib_dir}")

    bout, fly = int(args.bout), int(args.fly)
    T0 = int(args.start)
    conf_thr = float(getattr(args, "conf", 0.3) or 0.3)
    panel_h = int(getattr(args, "panel_h", 480) or 480)

    resolved = _resolve_multiview_cameras(calib_dir, all_cameras, views_arg)
    print("[sidebyside] multi-view rows: " +
          ", ".join(rigviews.format_view_label(role, rv) for _, role, rv in resolved))

    kp2d, conf2d = vio.load_kp2d(args.run, bout, fly)
    T2 = kp2d.shape[0]

    stac_h5 = os.path.join(vio.fly_dir(args.run, bout, fly), "stac_ik.h5")
    if not os.path.exists(stac_h5):
        raise FileNotFoundError(f"missing stac_ik.h5 for bout={bout} fly={fly}: {stac_h5}")
    with h5py.File(stac_h5, "r") as f:
        cfg = OmegaConf.create(f["config"][()].decode())
        kp_names = [s.decode() for s in f["kp_names"][:]]
        qpos_shape0 = f["qpos"].shape[0]
        kp_data_shape0 = f["kp_data"].shape[0]

    T = min(T2, qpos_shape0, kp_data_shape0)
    if T0 < 0 or T0 >= T:
        raise ValueError(f"start {T0} out of range for bout {bout} fly {fly} (usable T={T})")
    N = int(args.n) if getattr(args, "n", None) else (T - T0)
    N = min(N, T - T0)
    if N <= 0:
        raise ValueError(f"no frames to render (start={T0}, n={N}, usable T={T})")

    edges = _skeleton_edges(kp_names)
    groups = vcolors.keypoint_groups(kp_names)
    wing_idx = [i for i, n in enumerate(kp_names) if "Wing" in n]

    outs_path = os.path.join(vio.fly_dir(args.run, bout, fly), "outputs.h5")
    qref_path = os.path.join(vio.fly_dir(args.run, bout, fly), "qpos_refined.npz")
    if not (os.path.exists(outs_path) and os.path.exists(qref_path)):
        raise FileNotFoundError(
            f"--views needs outputs.h5 + qpos_refined.npz for bout={bout} fly={fly} "
            f"(--right rigcam has no fallback in multi-view mode)")
    _outs = vio.load_outputs(args.run, bout, fly)
    rig_world = np.asarray(_outs["kp3d_mm"], float)      # FITTED sites
    _kp3d_p = os.path.join(vio.fly_dir(args.run, bout, fly), "kp3d.npz")
    rig_meas = (np.asarray(np.load(_kp3d_p)["kp3d"], float)
                if os.path.exists(_kp3d_p) else None)    # MEASURED kp
    rig_qpos = np.asarray(np.load(qref_path)["qpos"], float)

    cam_mats, cam_names = reproject.camera_matrices(calib_dir)
    cam_names = list(cam_names)

    try:
        masks = vio.load_masks(predictions_dir, bout, fly, all_cameras)
        mk, mv, H, W = masks["masks"], masks["valid"], masks["H"], masks["W"]
    except Exception as e:
        print(f"[sidebyside] warning: masks unavailable ({e}); rendering without SAM overlay")
        mk = mv = None
        cap = cv2.VideoCapture(os.path.join(session_dir, f"{resolved[0][0]}.mp4"))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        cap.release()

    if getattr(args, "start_frame", None) is not None:
        start_abs = int(args.start_frame)
    else:
        from scripts.run_bout import bout_start_frame  # lazy: pulls in jax/mujoco/egl
        start_abs = bout_start_frame(_compose_cfg(), bout)

    row_names = [name for name, _, _ in resolved]
    print(f"[sidebyside] multi-view rows (cameras): {row_names}; segment t[{T0}:{T0 + N}]")

    drawers = []
    for cam_name, role, rv in resolved:
        if cam_name not in all_cameras:
            raise ValueError(f"resolved camera {cam_name} not in recording cameras {all_cameras}")
        cam_idx = all_cameras.index(cam_name)
        rig = _build_multiview_rig(cam_name, cam_mats, cam_names, cfg, kp_names, wing_idx,
                                   rig_world, rig_meas, rig_qpos, W, H)
        drawers.append(_make_multiview_drawers(cam_idx, rig, mk, mv, kp2d, conf2d, edges,
                                               groups, conf_thr, panel_h, W, H, T0, N,
                                               kp_names=kp_names,
                                               verify=bool(getattr(args, "verify", False))))
        print(f"[sidebyside] row {role}: {rigviews.format_view_label(role, rv)}")

    out_path = args.out or f"sidebyside_mv_bout{bout}_fly{fly}.mp4"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    frames_out = []
    stream = vio.read_frames_synced(session_dir, row_names, start_abs + T0, N)
    for k, imgs in enumerate(stream):
        t = T0 + k
        row_imgs = []
        for (cam_name, role, rv), rgb, (draw_left, draw_right, left_w) in \
                zip(resolved, imgs, drawers):
            bgr = (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb is not None
                   else np.zeros((H, W, 3), np.uint8))
            L = draw_left(bgr.copy(), t)
            R = draw_right(t)
            label = rigviews.format_view_label(role, rv)
            left_title = f"{label} video + SAM mask + ViTPose 2D skeleton"
            right_title = (f"MuJoCo IK @ {label} rig cam | wings: measured=MAGENTA "
                           f"fitted=YELLOW (line = residual)")
            Lh = np.vstack([_band(L.shape[1], left_title), L])
            Rh = np.vstack([_band(R.shape[1], right_title), R])
            h = max(Lh.shape[0], Rh.shape[0])

            def _padh(im, h=h):
                if im.shape[0] < h:
                    return np.vstack([im, np.zeros((h - im.shape[0], im.shape[1], 3), np.uint8)])
                return im

            sep = np.full((h, 3, 3), 60, np.uint8)
            row_imgs.append(np.hstack([_padh(Lh), sep, _padh(Rh)]))
        frames_out.append(layout.montage(row_imgs, cols=1))

    if not frames_out:
        raise RuntimeError(f"no frames rendered for bout {bout} fly {fly} (views {row_names})")

    vio.write_video(out_path, frames_out, fps=int(getattr(args, "fps", 30) or 30))
    still_path = os.path.splitext(out_path)[0] + "_still.png"
    cv2.imwrite(still_path, frames_out[len(frames_out) // 2])
    print(f"[sidebyside] wrote {out_path} ({len(frames_out)} frames) + {still_path}")
    return 0
