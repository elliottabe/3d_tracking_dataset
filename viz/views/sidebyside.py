"""sidebyside view: a side-by-side QC video for one (bout, fly).

LEFT : raw camera video + SAM mask overlay + ViTPose 2D keypoint SKELETON.
RIGHT: MuJoCo render of the STAC/IK pose + fitted 3D keypoint sites.

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
from viz.config import courtship_recording, _CFG_DIR

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

    # --- RIGHT panel: MuJoCo render of the IK pose ---
    from stac_mjx.stac import Stac  # heavy (jax + mujoco); lazy on purpose
    xml_path = cfg.model.MJCF_PATH
    stac = Stac(str(xml_path), cfg, kp_names)
    out_path = args.out or f"sidebyside_bout{bout}_fly{fly}.mp4"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print("[sidebyside] rendering MuJoCo IK ...", flush=True)
    # stac.render REQUIRES a save_path (imageio sniffs the .mp4 extension), but
    # we only consume the returned frames (`rframes`) for compositing -- so this
    # is a throwaway intermediate; keep a clean name (no double .mp4) and delete
    # it after the final side-by-side is written.
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
        crop = bgr[cy0:cy1, cx0:cx1]
        return cv2.resize(crop, (left_w, panel_h))

    # --- composite left+right per frame, collect BGR frames, write mp4 + still ---
    left_title = f"{left_cam} video + SAM mask + ViTPose 2D skeleton"
    right_title = f"MuJoCo IK render ({render_cam}) + 3D sites"
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
        L = _draw_left(bgr, T0 + k)
        R = cv2.cvtColor(np.asarray(rframes[k]), cv2.COLOR_RGB2BGR)
        if R.shape[0] != panel_h:
            s = panel_h / R.shape[0]
            R = cv2.resize(R, (int(R.shape[1] * s), panel_h))
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
    try:
        os.remove(mj_render_path)   # throwaway MuJoCo intermediate (frames already composited)
    except OSError:
        pass
    still_path = os.path.splitext(out_path)[0] + "_still.png"
    cv2.imwrite(still_path, frames_out[len(frames_out) // 2])
    print(f"[sidebyside] wrote {out_path} ({len(frames_out)} frames) + {still_path}")
    return 0
