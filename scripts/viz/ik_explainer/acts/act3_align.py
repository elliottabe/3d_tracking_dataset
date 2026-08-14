#!/usr/bin/env python3
"""Act 3 -- body scale + root alignment, as one combined step (the solver's
first two real stages).

This act renders SNAPSHOTS `stage_ik.py` recorded from an actual STAC solve
(`06_stages.npz`) -- it does not invent motion. Two measured facts drive the
whole timeline (do not restate the pre-measurement storyboard):

1. **Body scale is applied to the KEYPOINTS, not the mesh.** Scale comes from
   `preprocess_keypoints_for_ik.compute_shared_scale` (Umeyama trunk scale;
   `shared_scale = 0.1261` for this clip). The model geometry is never
   touched -- `qpos_scaled` is bit-identical to `qpos_default` in
   `06_stages.npz`. So this act animates the KEYPOINT CLOUD shrinking onto a
   fixed-size mesh, never the mesh growing/shrinking.
2. **`root_optimization` does NOT rotate.** `configs/anatomy/v1.yaml` sets
   `TRUNK_OPTIMIZATION_KEYPOINTS: {}`, so `root_optimization`'s keypoint-fit
   loss is uniformly zero and only the manual root-translation it also
   performs has any effect. Measured on `06_stages.npz`: `qpos_root[3:7]`
   (the free-joint quaternion) is bit-identical to `qpos_default[3:7]`
   (`[1,0,0,0]` both), and `qpos_root[7:]` (all joint DOF) is bit-identical
   to `qpos_default[7:]` too. Only `qpos_root[:3]` (the free-joint
   translation) differs. So this act is scale + TRANSLATION only -- no
   rotation. Orienting is Act 4's `pose_optimization` (34.73 deg measured
   there), not this one.

Because those two stages are genuinely translation-only here, `_slerp_qpos`
SLERPs a quaternion that never actually changes over this act. It is used
anyway (rather than a plain lerp of `qpos[:3]`/`qpos[7:]` only) so the same
helper is correct for Act 4, where `pose_optimization` produces a real
34.73 deg rotation and a lerped quaternion would denormalise and visibly
tumble.

WORLD-COORDINATE STORY (measured, not staged): the raw triangulated keypoints
(`04_kp3d_filt.npz`) sit in an arena-relative mm frame far from the model's
own origin (frame 450 mean ~[13.15, 1.99, 1.19]), while the model's rest qpos
places its root at the world origin. So at f=0 the mesh and the (unscaled)
cloud are ALREADY far apart -- that is the real triangulated position, not a
staged offset. Scaling the cloud by `shared_scale` (~0.126) shrinks it toward
the origin (frame-450 scaled mean ~[1.66, 0.25, 0.15]) which happens to land
close to where `root_optimization` then translates the mesh
(`qpos_root[:3]` = [1.65, 0.25, 0.26]) -- because `root_optimization` sets
the root translation to the (scaled) root keypoint itself. The timeline
below follows this real geometry: cloud shrinks toward the mesh first
(rescale), then the mesh translates the rest of the way to meet it
(root_optimization).

CAMERA (task-14 round 3 -- ACT 3'S CAMERA IS COMPLETELY FIXED, no zoom, no
dolly, no orbit): two earlier cuts of this act both routed the story
through a MOVING camera -- first a fully live-distance camera that
CANCELLED the cloud's own shrink (task-11), then a frozen-distance camera
with an added animated dolly-in (task-14 round 2) that fixed the framing but
cropped keypoints and read as distracting camera motion. Both attempts hit
the same lesson twice: a moving camera in this act keeps fighting the thing
the act is supposed to show. This cut retires camera motion entirely.

`ACT3_CAM_DISTANCE`/`ACT3_CAM_AZIMUTH`/`ACT3_CAM_ELEV` are TRUE CONSTANTS,
set once and never touched inside the render loop -- no per-frame
`cam.distance` or `cam.azimuth` derivation of any kind. `ACT3_CAM_AZIMUTH`/
`ACT3_CAM_ELEV` are now the SAME values as the shared `AZ_START`/`ELEV` Act 4
uses (task-14 round 4 fix: an earlier cut of this redesign used a DIFFERENT
azimuth here, 40 deg, chosen only so the raw incoming skeleton faced the
camera at f=0 -- that broke the camera-angle continuity Act 3 and Act 4 are
supposed to share across their cut, and the mesh visibly "snapped" to face a
different way at the Act3->Act4 transition. Continuity across the cut
matters more than the incoming skeleton's own entry angle, so this now
matches Act 4 exactly; `ACT3_CAM_DISTANCE` stays Act-3-local since a cut is
allowed to change shot DISTANCE, just not shot ANGLE.) `cam.lookat` is the
one quantity that still updates every frame, set to the mesh's own
`mesh_ctr` (forward-kinematics, cheap) -- this is NOT zoom/dolly/orbit (none
of `distance`/`azimuth`/`elevation` change), it is the minimum "frame it on
the body model" requires given a real, measured fact: `root_optimization`
translates the mesh by ~1.69 model-units (`qpos_root[:3] - qpos_default[:3]`)
-- about 5.8x the model's own body length -- so ANY single fixed lookat
point would either clip the mesh out of frame for most of the act, or force
the camera wide enough that the mesh reads far too small. Tracking
`mesh_ctr` keeps the mesh centred and the SAME apparent size at every frame
(size depends only on `cam.distance`, which never changes) while still
showing the real translation as the mesh visibly slides during the ALIGN
phase. `ACT3_CAM_DISTANCE=1.0` was tuned against a REAL rendered-pixel
measurement (not an analytic FOV estimate -- see Change 3's history above):
near-white/low-saturation pixel count (`mx>60 & (mx-mn)<40`, excluding the
top-270px caption band) measures ~64,000 px at this distance, matching Act
4's own ~54,000-73,000 px framing.

Keypoint SKELETON, not a loose cloud (task-14 round 3): keypoints are now
connected by thin capsule "bones" (`stage_ik._add_bone`, MuJoCo's
`mjv_connector`) using `data/fly50.json`'s `edges` (44 index pairs, mapped
onto THIS array's own keypoint order by NAME --
`kp_colors.jarvis_skeleton_edges`), coloured per JARVIS's own bone-colour
convention (`colors[line[1]]`, the STOP node's colour -- verified against
all six of JARVIS's own skeleton-drawing call sites). Both marker and bone
radius scale with the cloud's own current linear scale factor (`factor`,
1.0 raw -> `shared_scale` 0.1261), so the incoming skeleton is drawn
genuinely OVERSIZED while raw and shrinks to its final, proportionate size
together with the cloud's own real shrink -- not a separately-invented
animation. The same skeleton (bones + colours) is now ALSO drawn in Act 4
(task-14 round 4) -- see `act4_solve.py`'s module docstring.

The oversized, raw-scale skeleton at f=0 sits ~13 model-units from the
fixed camera's lookat (the real triangulated position -- see the
WORLD-COORDINATE STORY below). With the camera now matched to Act 4's angle
(see above), the raw skeleton's direction from the lookat is NOT guaranteed
to be in front of the camera the way the earlier (40 deg) angle was -- it
may only swing into view partway through the ALIGN phase, once the mesh's
own translation has carried the lookat closer to the skeleton's real
position. This is an accepted trade-off: a skeleton that is briefly out of
view at f=0 is preferable to a camera that visibly reorients the fly at the
Act3->Act4 cut.

Keypoint colours (Change 2, task-14): each limb chain gets its own colour
from the JARVIS scheme (`kp_colors.jarvis_kp_colors_rgb01`, ported by calling
`third_party/JARVIS-HybridNet`'s own `get_skeleton`), replacing the previous
4-group (head/thorax/abdomen/legs) scheme.

Wing visibility (Change 1, task-14): `set_mesh_rgba`/`capture_geom_alpha`
(stage_ik.py) preserve the model's originally-invisible geoms (the wings'
`*_inertial` boxes, alpha=0 by design) across every `geom_rgba` mutation this
act performs -- see their docstrings for why a blanket alpha assignment used
to turn them into opaque white rectangles.

SCALE + ALIGN, now a SINGLE combined step (task-14 round 4): the original
design ran `rescale` (120-299) then `root_optimization` (300-539) as two
sequential beats. The user asked for scale and alignment to read as one
motion, so both now interpolate from the SAME progress variable `p` across
ONE combined window (`ALIGN_START`-`ALIGN_END`, still 120-539, i.e. the
total duration is unchanged -- only the internal SEQUENCING merged): the
keypoint scale factor (1.0 -> `shared_scale`) and the qpos SLERP
(`qpos_default` -> `qpos_root`) now animate together, reaching their targets
at the same frame, instead of the scale finishing first and the translation
only then starting. Residual is a single monotonic interpolation from
`residual_default` (13.404 mm) to `residual_root` (0.118 mm) across the
combined window -- still real, measured stage residuals, just no longer
displaying the intermediate `residual_scaled` checkpoint as a distinct beat.

Timeline (600 frames, 30 fps):
  f   0-119  mesh fades in at qpos_default (alpha ramp), skeleton already
             fully visible at its RAW (unscaled) position/size -- mostly
             off-canvas (see CAMERA above).
  f 120-539  SINGLE combined scale+align step: keypoint scale factor
             interpolates 1.0 -> `shared_scale` (0.1261) WHILE qpos
             SIMULTANEOUSLY interpolates qpos_default -> qpos_root
             (`_slerp_qpos`, translation only, no rotation). Residual ticks
             down 13.404 -> 0.118 mm over the same window. Mesh never
             changes size; the camera never moves.
  f 540-599  hold: mesh and skeleton co-located, residual 0.118 mm.

EXPECTATION: f=0 skeleton oversized and mostly off-canvas; by f=539 skeleton
and mesh are co-located, residual has fallen 13.404 -> 0.118 mm, mesh reads
at the same on-screen size it always has, and the camera's ANGLE matches
Act 4's exactly (no visible reorientation at the cut). The mesh does NOT
change size and does NOT rotate at any point; the camera does NOT zoom,
dolly, or orbit at any point.
FALSIFICATION: a visibly rotating OR resizing mesh means the act is
animating something the solver did not do; a visible "snap" in the fly's
apparent facing direction across the Act3->Act4 cut means the camera angles
have drifted apart again.

Colours come from `kp_colors.jarvis_kp_colors_rgb01` (JARVIS per-limb-chain
scheme) -- never invented here.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))
sys.path.insert(0, str(_REPO / "stac-mjx"))

from scripts.viz.ik_explainer import clip_io, draw               # noqa: E402
from scripts.viz.ik_explainer.stage_ik import (                  # noqa: E402
    _add_sphere, _add_bone, _tracking_site_map, capture_geom_alpha, set_mesh_rgba,
)
from scripts.viz.ik_explainer.kp_colors import (                 # noqa: E402
    jarvis_kp_colors_rgb01, jarvis_skeleton_edges,
)
from viz.core.colors import PALETTE                                # noqa: E402

# --- canvas / timeline ------------------------------------------------------
CANVAS_W, CANVAS_H = 1920, 1080
N_OUT = 600
FADE_START, FADE_END = 0, 119            # mesh alpha ramp (inclusive)
# Combined scale+align step (task-14 round 4): scale and root_optimization's
# translation now animate TOGETHER from one progress variable, across the
# SAME total window the two sequential stages used to share (120-539).
ALIGN_START, ALIGN_END = 120, 539
HOLD_START = 540

XML_PATH = _REPO / "models" / "fruitfly_v1" / "fruitfly_v1_free.xml"

# --- shared "wide diagnostic camera" building blocks -------------------------
# `AZ_START`/`ELEV`, `_cloud_mesh_spread`, and `_wide_camera` below are kept
# for ACT 4 ONLY (imported from there: `_wide_camera_for_aspect` builds on
# `_wide_camera` with a LIVE per-frame distance, appropriate for Act 4's real
# 921-frame playback). As of task-14 round 3, Act 3 no longer uses ANY of
# these -- its own camera is the fully independent, fully fixed
# `ACT3_CAM_DISTANCE`/`ACT3_CAM_AZIMUTH`/`ACT3_CAM_ELEV` below. Do not repoint
# Act 3 back at `AZ_START`/`ELEV`/`_wide_camera`: changing THIS block changes
# Act 4's camera too (see the CAMERA section of the module docstring).
AZ_START = 120.0
ELEV = -20.0

# Mesh fade-in floor: f=0 must already show a (translucent) mesh, per the
# task's own falsification check ("f=0: mesh AND cloud both visible") -- a
# fade that starts at literal alpha=0 would make f=0 mesh-invisible.
MESH_ALPHA_MIN = 0.45

# --- Act 3's OWN fixed camera (task-14 round 3) -----------------------------
# TRUE CONSTANTS -- never read into a per-frame formula, unlike `AZ_START`/
# `ELEV` above (which Act 4 imports for its own, deliberately LIVE camera).
# `ACT3_CAM_DISTANCE` was solved against a REAL rendered-pixel measurement
# (see `_mesh_px` below and the module docstring): at distance=1.0, the
# mesh's own near-white/low-saturation pixel count is ~64,000 px, inside the
# requested ~50,000-75,000 px band (matching Act 4's own ~54,000-73,000 px
# framing). `ACT3_CAM_AZIMUTH`/`ACT3_CAM_ELEV` were chosen (over the shared
# `AZ_START`/`ELEV`). Round-4 fix: `ACT3_CAM_AZIMUTH`/`ACT3_CAM_ELEV` are now
# set EQUAL to the shared `AZ_START`/`ELEV` (both 120/-20) so Act 3's camera
# ANGLE matches Act 4's exactly -- a previous cut used a different azimuth
# (40 deg) to keep the raw incoming skeleton facing the camera at f=0, which
# broke continuity and made the fly visibly "snap" to a different apparent
# facing direction at the Act3->Act4 cut. Distance stays Act-3-local (a cut
# is allowed to change shot distance, just not shot angle).
ACT3_CAM_DISTANCE = 1.0
ACT3_CAM_AZIMUTH = AZ_START
ACT3_CAM_ELEV = ELEV

# Marker/bone radius at the FINAL (shared_scale) size; scaled by (factor /
# shared_scale) in the render loop so the incoming, raw-scale skeleton is
# drawn genuinely bigger while it is genuinely bigger, not resized to a
# constant on-screen marker size independent of the real geometry.
BASE_MARKER_R_AT_FINAL_SCALE = 0.012   # model units; ~ a leg-segment's width at Act 4's framing
BONE_RADIUS_FRACTION = 0.4             # bone capsule radius, as a fraction of marker radius

# Real-pixel mesh-size acceptance test (Change-3 round 2/3): near-white,
# low-saturation pixels, excluding the top-270px caption band -- keypoints
# are drawn as SATURATED colours so they never count as "mesh" here.
_CAPTION_BAND_PX = 270


def _mesh_px(canvas_bgr) -> int:
    b = canvas_bgr[..., 0].astype(np.int32)
    g = canvas_bgr[..., 1].astype(np.int32)
    r = canvas_bgr[..., 2].astype(np.int32)
    mx = np.maximum(np.maximum(b, g), r)
    mn = np.minimum(np.minimum(b, g), r)
    mesh = (mx > 60) & ((mx - mn) < 40)
    mesh[:_CAPTION_BAND_PX, :] = False
    return int(mesh.sum())


def _smoothstep(p):
    p = np.clip(p, 0.0, 1.0)
    return 3 * p ** 2 - 2 * p ** 3


def _slerp_qpos(qa, qb, t):
    """Interpolate a MuJoCo free-joint qpos: `qpos[:3]` (translation) lerp,
    `qpos[3:7]` (w,x,y,z quaternion) SLERP, `qpos[7:]` (hinge/joint DOF) lerp.

    A plain lerp of the quaternion denormalises and visibly tumbles; SLERP
    keeps it a unit quaternion throughout. In THIS act the quaternion never
    actually changes (qa[3:7] == qb[3:7] == identity, per the module
    docstring's measured fact), so SLERP is a no-op here -- it is used anyway
    so this helper is directly reusable for Act 4's real 34.73 deg rotation.
    """
    qa, qb = np.asarray(qa, np.float64), np.asarray(qb, np.float64)
    pos = (1 - t) * qa[:3] + t * qb[:3]
    quat_a_xyzw = np.roll(qa[3:7], -1)     # (w,x,y,z) -> (x,y,z,w) for scipy
    quat_b_xyzw = np.roll(qb[3:7], -1)
    rots = Rotation.from_quat(np.stack([quat_a_xyzw, quat_b_xyzw]))
    slerp = Slerp([0.0, 1.0], rots)
    quat_t_xyzw = slerp(t).as_quat()
    quat_t_wxyz = np.roll(quat_t_xyzw, 1)  # back to (w,x,y,z)
    joints = (1 - t) * qa[7:] + t * qb[7:]
    return np.concatenate([pos, quat_t_wxyz, joints])


def _cloud_mesh_spread(mesh_ctr, cloud_pts, lookat, model_extent):
    """Radius (world units) that a camera centred at `lookat` must span to
    keep both `mesh_ctr` and every finite point of `cloud_pts` in frame --
    the same formula stage_ik.py's qc_stages() uses for its wide-camera row.
    Pure geometry, no camera object built here (used both for the one-time
    FIXED `cam_spread` in `render_act3` and, live, for marker sizing only --
    see task-11 review "Important 1": these two uses must NOT share a value
    that also drives `cam.distance` every frame, or the camera zooms in step
    with the shrinking cloud).
    """
    finite = cloud_pts[np.all(np.isfinite(cloud_pts), axis=-1)]
    return max(
        float(np.max(np.linalg.norm(finite - lookat, axis=-1))),
        float(np.linalg.norm(mesh_ctr - lookat)),
        model_extent * 0.5,
    )


def _wide_camera(mesh_ctr, cloud_pts, model_extent, azimuth, cam_spread, zoom=1.0):
    """The wide diagnostic camera: same lookat construction stage_ik.py's
    qc_stages() uses for its wide-camera row (never the model's `hero`
    camera, which frames the mesh only). `lookat` is recomputed every frame
    (mesh_ctr and the cloud both move over the act) so both stay centred.

    `cam_spread` -- hence the BASE `cam.distance` -- is NOT derived from the
    live cloud here. It is a FIXED value the caller computes ONCE for the
    whole act (see `render_act3`'s dry pass), so the camera cannot zoom in
    lock-step with the shrinking cloud (task-11 review, "Important 1": the
    previous per-frame version did exactly that, cancelling the cloud's
    on-screen shrink).

    `zoom` (default 1.0, i.e. no change) is a SEPARATE, optional dolly-in
    multiplier -- `cam.distance = cam_spread * 2.6 / zoom` -- added for
    Change 3 (task-14): an animated zoom-in over the act so the final
    mesh+cloud pair fills more of the frame, deliberately layered UNDER the
    task-11 fix rather than replacing it (`zoom` is caller-controlled per
    frame; `cam_spread` stays the one-time worst-case value, so f=0 still
    frames the full, still-8x-oversized raw cloud). Act 4's calls
    (`_wide_camera_for_aspect`) never pass `zoom`, so they are unaffected.

    Returns `(cam, live_cloud_spread)` where `live_cloud_spread` is the
    cloud's OWN current extent from this frame's lookat -- used only to size
    the drawn keypoint markers (see `marker_r` in `render_act3`), never the
    camera.
    """
    finite = cloud_pts[np.all(np.isfinite(cloud_pts), axis=-1)]
    cloud_ctr = finite.mean(axis=0)
    lookat = (mesh_ctr + cloud_ctr) / 2.0
    live_cloud_spread = _cloud_mesh_spread(mesh_ctr, cloud_pts, lookat, model_extent)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = cam_spread * 2.6 / zoom
    cam.azimuth, cam.elevation = azimuth, ELEV
    return cam, live_cloud_spread


def _stage_at(f, residual_default, residual_scaled, residual_root,
              qpos_default, qpos_root, shared_scale):
    """Return (stage_label, residual_mm, scale_factor, qpos, mesh_alpha) for
    output frame `f`. Pure function of the recorded stage snapshots -- no
    numbers invented here, only interpolated between measured stage values.
    """
    if f <= FADE_END:
        p = _smoothstep((f - FADE_START) / float(FADE_END - FADE_START))
        # Floor at MESH_ALPHA_MIN rather than 0: the self-review requires
        # f=0 to already show the mesh (translucent is fine, invisible is
        # not), so this is a build from translucent -> fully opaque, not
        # literally transparent -> opaque.
        alpha = MESH_ALPHA_MIN + (1.0 - MESH_ALPHA_MIN) * p
        return "default (rest pose)", residual_default, 1.0, qpos_default, alpha
    if f <= ALIGN_END:
        # Combined scale+align (task-14 round 4): ONE progress variable `p`
        # drives BOTH the keypoint scale factor and the qpos SLERP, so they
        # reach their targets together instead of scale finishing first.
        p = _smoothstep((f - ALIGN_START) / float(ALIGN_END - ALIGN_START))
        resid = (1 - p) * residual_default + p * residual_root
        factor = (1 - p) * 1.0 + p * shared_scale
        qpos = _slerp_qpos(qpos_default, qpos_root, p)
        return "rescale + root_optimization", resid, factor, qpos, 1.0
    return "rescale + root_optimization (held)", residual_root, shared_scale, qpos_root, 1.0


def render_act3(clip: str = clip_io.CLIP_DEFAULT) -> Path:
    dirs = clip_io.out_dirs(clip)

    with np.load(dirs["predictions"] / "06_stages.npz", allow_pickle=True) as z:
        qpos_default = np.asarray(z["qpos_default"], np.float64)
        qpos_root = np.asarray(z["qpos_root"], np.float64)
        residual_mm = np.asarray(z["residual_mm"], np.float64)
        stage_names = [str(s) for s in z["stage_names"]]
        kp_names = [str(n) for n in z["kp_names"]]
        frame_for_stills = int(z["frame_for_stills"])
        shared_scale = float(z["shared_scale"])

    assert stage_names[:3] == ["default", "scaled", "root"], (
        f"unexpected 06_stages.npz stage_names order: {stage_names}")
    residual_default, residual_scaled, residual_root = (
        float(residual_mm[0]), float(residual_mm[1]), float(residual_mm[2]))

    # Measured-fact guard: root_optimization must not have rotated or moved
    # any joint DOF (module docstring point 2). If a future re-run of
    # stage_ik.py ever produces a rotating root_optimization, this act's
    # entire "scale + translation only" premise would be wrong -- fail loudly
    # rather than silently render the old (incorrect) story.
    if not np.allclose(qpos_default[3:], qpos_root[3:], atol=1e-9):
        raise ValueError(
            "qpos_default[3:] != qpos_root[3:] -- root_optimization rotated "
            "and/or moved joint DOF in this solve. Act 3's premise (scale + "
            "translation only, no rotation) no longer matches the recorded "
            "snapshots; refusing to render a mesh-rotation animation that "
            "contradicts the measured facts.")

    with np.load(dirs["predictions"] / "04_kp3d_filt.npz", allow_pickle=True) as z:
        kp3d = np.asarray(z["kp3d"], np.float64)
        kp3d_names = [str(n) for n in z["kp_names"]]
    if kp3d_names != kp_names:
        raise ValueError(
            "04_kp3d_filt.npz kp_names != 06_stages.npz kp_names -- refusing "
            "to mix keypoint orders (CLAUDE.md's keypoint-order bug class).")

    kp3d_raw_frame = kp3d[frame_for_stills]   # (50,3) mm, MODEL order, RAW (unscaled)
    print(f"[act3] frame_for_stills={frame_for_stills}, shared_scale={shared_scale:.4f}")
    print(f"[act3] residuals default/scaled/root = "
          f"{residual_default:.3f}/{residual_scaled:.3f}/{residual_root:.3f} mm")
    print(f"[act3] qpos_root - qpos_default (translation only, mm): "
          f"{(qpos_root[:3] - qpos_default[:3])}")

    mj_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    orig_alpha = capture_geom_alpha(mj_model)   # BEFORE any geom_rgba mutation
    grey = PALETTE["mesh"][0] / 255.0   # PALETTE["mesh"] is (200,200,200): BGR==RGB here
    set_mesh_rgba(mj_model, orig_alpha, rgb=grey)   # alpha animated per-frame below

    site_map = _tracking_site_map(mj_model, kp_names)
    missing = [n for n in kp_names if n not in site_map]
    if missing:
        raise ValueError(f"tracking[...] site missing for keypoints: {missing}")
    body_site_idxs = np.asarray([site_map[n] for n in kp_names])
    kp_rgb01 = jarvis_kp_colors_rgb01(kp_names)
    kp_rgb01_by_idx = [kp_rgb01[n] for n in kp_names]
    # Skeleton bones (task-14 round 3): (idx_a, idx_b, BGR) mapped onto THIS
    # array's own kp_names order by name; converted to RGB/0-1 once, here,
    # not per frame.
    skeleton_edges = [
        (a, b, (bg[2] / 255.0, bg[1] / 255.0, bg[0] / 255.0))
        for a, b, bg in jarvis_skeleton_edges(kp_names)
    ]

    out_dir = dirs["frames"] / "act3_align"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Act 3's fixed camera, resolved ONCE against the real (start, end)
    # mesh positions (task-14 round 3; see module docstring's CAMERA section).
    # No cam_spread dry pass any more -- the camera no longer needs to
    # discover a worst-case distance across the act, since it never zooms.
    print(f"[act3] fixed camera: distance={ACT3_CAM_DISTANCE} azimuth="
          f"{ACT3_CAM_AZIMUTH} elevation={ACT3_CAM_ELEV} (never animated); "
          f"lookat tracks mesh_ctr each frame (translation magnitude "
          f"{np.linalg.norm(qpos_root[:3] - qpos_default[:3]):.3f} model-units "
          f"over the ROOT phase)")

    t0 = time.time()
    # Persistent renderer, created ONCE and reused for every frame (matches
    # Act 4's fix; see CLAUDE.md/module constraints -- per-frame `with
    # mujoco.Renderer(...)` construction is the prime suspect for Act 4's
    # mid-render EGL resource-leak crash at ~700/900 frames. Act 3's 600
    # frames never hit that failure, but this loop is touched here anyway,
    # so it is fixed too rather than left on the known-bad pattern.
    with mujoco.Renderer(mj_model, height=CANVAS_H, width=CANVAS_W) as renderer:
        for f in range(N_OUT):
            stage_label, resid, factor, qpos, mesh_alpha = _stage_at(
                f, residual_default, residual_scaled, residual_root,
                qpos_default, qpos_root, shared_scale,
            )
            set_mesh_rgba(mj_model, orig_alpha, alpha=mesh_alpha)

            d = mujoco.MjData(mj_model)
            d.qpos[:] = qpos
            mujoco.mj_forward(mj_model, d)

            # `mesh_ctr` is the ONLY per-frame camera quantity, and it is not
            # zoom/dolly/orbit -- see the module docstring's CAMERA section.
            # Identical to `mesh_ctr_default` through the FADE phase (qpos is
            # qpos_default there); starts changing once the combined
            # scale+align step begins (f > FADE_END), as the mesh genuinely
            # translates.
            mesh_ctr = np.asarray(d.site_xpos[body_site_idxs]).mean(axis=0)
            cloud_pts = kp3d_raw_frame * factor   # linear scale about world origin

            cam = mujoco.MjvCamera()
            cam.lookat[:] = mesh_ctr
            cam.distance = ACT3_CAM_DISTANCE
            cam.azimuth, cam.elevation = ACT3_CAM_AZIMUTH, ACT3_CAM_ELEV

            # Marker/bone radius scales with the cloud's OWN current linear
            # scale factor (raw 1.0 -> shared_scale), so the incoming
            # skeleton is drawn genuinely oversized while it genuinely is.
            marker_r = BASE_MARKER_R_AT_FINAL_SCALE * (factor / shared_scale)
            bone_r = marker_r * BONE_RADIUS_FRACTION

            renderer.update_scene(d, camera=cam)
            scn = renderer.scene
            for a, b, rgb in skeleton_edges:
                pa, pb = cloud_pts[a], cloud_pts[b]
                if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                    _add_bone(scn, pa, pb, np.array((*rgb, 1.0), np.float32), bone_r)
            for i, p3 in enumerate(cloud_pts):
                if np.all(np.isfinite(p3)):
                    rgb = kp_rgb01_by_idx[i]
                    _add_sphere(scn, p3, np.array((*rgb, 1.0), np.float32), marker_r)
            img_rgb = np.ascontiguousarray(renderer.render())

            canvas = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            canvas = draw.stage_title(canvas, "ACT 3", "Body scale + root alignment")
            canvas = draw.label(canvas, f"stage: {stage_label}", (48, 150),
                                 scale=0.6, color=(255, 255, 255))
            canvas = draw.label(
                canvas, f"residual: {resid:.3f} mm (mean over 50 sites; "
                        f"legs/wings lag until Act 4 solves the joints)",
                (48, 182), scale=0.55, color=(190, 255, 190))
            canvas = draw.label(
                canvas, f"keypoint scale factor: {factor:.4f} "
                        f"(Umeyama trunk fit, target shared_scale={shared_scale:.4f})",
                (48, 210), scale=0.5, color=(190, 190, 190))
            canvas = draw.label(
                canvas, "mesh size is fixed -- only the keypoint cloud is rescaled; "
                        "root_optimization translates the mesh, it does not rotate it",
                (48, 238), scale=0.45, color=(150, 150, 150))
            canvas = draw.label(canvas, f"frame {f + 1}/{N_OUT}", (48, CANVAS_H - 24),
                                 scale=0.45, color=(150, 150, 150))

            if f in (0, 150, 299, 450, 539):
                mpx = _mesh_px(canvas)
                print(f"[act3] fixed-camera acceptance: f={f} mesh_px={mpx} "
                      f"(target ~50,000-75,000, near-constant across the act)")

            cv2.imwrite(str(out_dir / f"f{f:05d}.png"), canvas)

    dt = time.time() - t0
    print(f"[act3] wrote {N_OUT} frames to {out_dir} in {dt:.1f} s "
          f"({dt / N_OUT:.3f} s/frame)")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    args = ap.parse_args()
    render_act3(args.clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
