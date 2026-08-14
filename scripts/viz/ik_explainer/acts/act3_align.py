#!/usr/bin/env python3
"""Act 3 -- root alignment ONLY, keypoints assumed already scaled (task-15 v2,
a direct mid-task user pivot away from the earlier "merged scale+align" cut --
see the TASK-15 PIVOT section below for why).

This act renders SNAPSHOTS `stage_ik.py` recorded from an actual STAC solve
(`06_stages.npz`) -- it does not invent motion. Two measured facts underlie
the story (do not restate the pre-measurement storyboard):

1. **Body scale is applied to the KEYPOINTS, not the mesh, and is NOT
   animated in this act.** Scale comes from
   `preprocess_keypoints_for_ik.compute_shared_scale` (Umeyama trunk scale;
   `shared_scale = 0.1261` for this clip). The model geometry is never
   touched. This act draws the keypoint cloud ALREADY at `shared_scale` --
   fixed-size for the whole act -- and animates only its POSITION (the
   `root_optimization` translation). The rescale step itself (1.0 ->
   `shared_scale`) is not depicted here; it is assumed done (see TASK-15
   PIVOT below for why).
2. **`root_optimization` does NOT rotate.** `configs/anatomy/v1.yaml` sets
   `TRUNK_OPTIMIZATION_KEYPOINTS: {}`, so `root_optimization`'s keypoint-fit
   loss is uniformly zero and only the manual root-translation it also
   performs has any effect. Measured on `06_stages.npz`: `qpos_root[3:7]`
   (the free-joint quaternion) is bit-identical to `qpos_default[3:7]`
   (`[1,0,0,0]` both), and `qpos_root[7:]` (all joint DOF) is bit-identical
   to `qpos_default[7:]` too. Only `qpos_root[:3]` (the free-joint
   translation) differs. So this act is TRANSLATION ONLY -- no rotation, no
   scale change. Orienting is Act 4's `pose_optimization` (34.73 deg
   measured there), not this one.

Because `root_optimization` is genuinely translation-only here, `_slerp_qpos`
SLERPs a quaternion that never actually changes over this act. It is used
anyway (rather than a plain lerp of `qpos[:3]`/`qpos[7:]` only) so the same
helper is correct for Act 4, where `pose_optimization` produces a real
34.73 deg rotation and a lerped quaternion would denormalise and visibly
tumble.

TASK-15 PIVOT (read this before touching the STAGING math below): task-15
originally asked for the EXISTING merged "rescale + root_optimization"
single-step design (scale 1.0 -> `shared_scale` WHILE qpos SLERPs
`qpos_default` -> `qpos_root`) to additionally stay centred on the model
throughout, rather than flying in from off-screen. That version was
implemented, rendered, and read frame-by-frame -- and while f=0/f=299 looked
right in isolation, reading the IN-BETWEEN frames (f=30/60/90/150/250)
showed the oversized cloud visibly SHRINKING AWAY FROM the model, exiting
the frame entirely by roughly f=90, staying off-screen through the middle,
then flying BACK IN near the end to land correctly -- i.e. the "flies in
from off-screen" problem the user asked to remove was still there, just
moved to the back half of the act instead of the front. Root cause (kept
here as a record, not a live design): with scale ALSO animating, the cloud's
own true (measured, unstaged) centroid trajectory `factor(f)*raw_mean`
travels along a completely different path than the mesh's own translation
`mesh_ctr(f)` -- the two only coincide at the very end (by construction of
`root_optimization`), so any staging pivot that blends toward that true
trajectory necessarily drags the displayed cloud through its huge
intermediate values (multiple model-units away from the camera's lookat)
somewhere in the middle. The user, live, asked to sidestep this rather than
patch it further: **"the scaling doesn't look good... just have it do the
root alignment and assume it is already scaled... and can we have it
shorter."** This is a direct user pivot, not a further tuning of the merged
design -- so the merged "rescale + root_optimization" single step (task-14
round 4 / task-15 v1) is RETIRED here, not layered under a bigger fix.

With scale fixed at `shared_scale` for the WHOLE act, the problem above goes
away almost entirely: the cloud's own true (fixed) position `C_true =
shared_scale * raw_mean` is now a SINGLE POINT, not a moving trajectory, and
(by construction of `root_optimization`) it sits close to `mesh_ctr_final`
-- the mesh's own real translation target already nearly arrives there
without any staging help. So the ONLY staging needed is a plain, one-shot
blend of the cloud's centroid from the mesh's own LIVE tracked centroid
(`mesh_ctr(f)`, which the camera already follows) to that single fixed true
point `C_true`, using the SAME progress `p` that drives `root_optimization`'s
own qpos SLERP:

    C_true             = shared_scale * raw_mean            (fixed, measured)
    scaled_centered     = shared_scale * (kp3d_raw_frame - raw_mean)  (fixed shape)
    staged_centroid(f)  = (1 - p) * mesh_ctr(f)  +  p * C_true
    cloud_pts(f)         = staged_centroid(f) + scaled_centered

At `p=0` (`f=0`): `staged_centroid = mesh_ctr(0)` exactly, i.e. the
(already correctly-sized) skeleton is centred on the model's REST position.
At `p=1` (`f=ALIGN_END`, and every hold frame): `staged_centroid = C_true`
exactly, so `cloud_pts = shared_scale*raw_mean + shared_scale*(kp3d_raw_frame
- raw_mean) = shared_scale * kp3d_raw_frame` -- the SOLVER's true scaled
keypoint position, unmodified, bit-for-bit the same value every earlier cut
used. Nothing about the measured endpoint changes; only the in-between
staged position of the (now fixed-size) cloud's centroid does -- captioned
on screen as a presentation choice, in the same spirit as Act 2's "panel
DISTANCE... is staging only" caveat: the pivot used for the animation's
IN-BETWEEN frames is a staging decision, the endpoints are measured.

WORLD-COORDINATE STORY (measured, not staged): the raw triangulated keypoints
(`04_kp3d_filt.npz`) sit in an arena-relative mm frame far from the model's
own origin (frame 450 mean ~[13.15, 1.99, 1.19] model-units), while the
model's rest qpos places its root at the world origin. `C_true` above
(`shared_scale * raw_mean` ~= [1.66, 0.25, 0.15]) is close to where
`root_optimization` independently translates the mesh (`qpos_root[:3]` =
[1.65, 0.25, 0.26]) -- because `root_optimization` sets the root translation
to the (already-scaled) root keypoint itself. That near-coincidence is why
this act's staged blend converges cleanly: the mesh's OWN real translation
target and the cloud's OWN true fixed position are, by the solver's
construction, already close together.

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
`ACT3_CAM_ELEV` are the SAME values as the shared `AZ_START`/`ELEV` Act 4
uses (task-14 round 4 fix: an earlier cut of this redesign used a DIFFERENT
azimuth here, 40 deg, chosen only so the raw incoming skeleton faced the
camera at f=0 -- that broke the camera-angle continuity Act 3 and Act 4 are
supposed to share across their cut, and the mesh visibly "snapped" to face a
different way at the Act3->Act4 transition. Continuity across the cut
matters more than the incoming skeleton's own entry angle, so this now
matches Act 4 exactly; `ACT3_CAM_DISTANCE` stays Act-3-local since a cut is
allowed to change shot DISTANCE, just not shot ANGLE.) `cam.lookat` is the
one quantity that still updates every frame, set to the mesh's own REAL
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
phase. `ACT3_CAM_DISTANCE` was tuned against a REAL rendered-pixel
measurement: near-white/low-saturation pixel count (`mx>60 & (mx-mn)<40`,
excluding the top-270px caption band). At `distance=1.0` (an earlier cut's
value), `f=0` measured 47,648 px -- just under the requested 50,000-75,000
px band -- because task-15 v2's skeleton sits DIRECTLY ON the mesh at f=0
(it no longer arrives from off-scale), so the saturated keypoint markers
occlude slightly more of the mesh's own near-white silhouette than they used
to. Retuned to `ACT3_CAM_DISTANCE=0.93` (closer -> larger apparent mesh),
which measures 55,312 / 70,625 / 63,323 / 63,338 px at f=0/60/119/179 --
comfortably inside the band at every checked frame and matching Act 4's own
~54,000-73,000 px framing. The 55k-71k spread (vs the ~4% spread task-14
measured on the old design) is largely f=60's skeleton sitting BESIDE the
mesh rather than overlapping it, so less of the mesh silhouette is occluded
there -- i.e. it is an OCCLUSION artefact of the pixel-counting proxy, not
the mesh itself changing size (the mesh geometry and camera distance are
identical at every frame; see the `qpos_default[3:]==qpos_root[3:]` guard).

Keypoint SKELETON, not a loose cloud (task-14 round 3): keypoints are
connected by thin capsule "bones" (`stage_ik._add_bone`, MuJoCo's
`mjv_connector`) using `data/fly50.json`'s `edges` (44 index pairs, mapped
onto THIS array's own keypoint order by NAME --
`kp_colors.jarvis_skeleton_edges`), coloured per JARVIS's own bone-colour
convention (`colors[line[1]]`, the STOP node's colour -- verified against
all six of JARVIS's own skeleton-drawing call sites). Marker/bone radius are
now FIXED (`BASE_MARKER_R_AT_FINAL_SCALE`, `BONE_RADIUS_FRACTION`) -- task-15
v1's factor-linked sizing (oversized while raw, shrinking with the cloud) no
longer applies now that the cloud is drawn at `shared_scale` for the whole
act, never resized. The same skeleton (bones + colours) is also drawn in
Act 4 (task-14 round 4) -- see `act4_solve.py`'s module docstring.

Keypoint colours (Change 2, task-14): each limb chain gets its own colour
from the JARVIS scheme (`kp_colors.jarvis_kp_colors_rgb01`, ported by calling
`third_party/JARVIS-HybridNet`'s own `get_skeleton`), replacing the previous
4-group (head/thorax/abdomen/legs) scheme.

Wing visibility (Change 1, task-14): `set_mesh_rgba`/`capture_geom_alpha`
(stage_ik.py) preserve the model's originally-invisible geoms (the wings'
`*_inertial` boxes, alpha=0 by design) across every `geom_rgba` mutation this
act performs -- see their docstrings for why a blanket alpha assignment used
to turn them into opaque white rectangles.

Timeline (180 frames, 6 s @ 30 fps -- task-15 v2: shortened again from the
360-frame/12s cut per the same live user request, "and can we have it
shorter" -- root alignment alone is a simpler story than the retired
merged scale+align one and reads fine in less time; the rest-pose fade-in
phase from before task-15 stays REMOVED, the act opens already inside the
translation):
  f   0-119  root_optimization: qpos SLERPs `qpos_default` -> `qpos_root`
             (translation only, no rotation, via `_slerp_qpos`); the
             ALREADY fixed-size (`shared_scale`) keypoint skeleton's
             STAGED centroid blends `mesh_ctr(f)` -> `C_true` (see the
             TASK-15 PIVOT section) with the SAME progress `p`. Residual
             ticks down `residual_scaled` (1.710 mm) -> `residual_root`
             (0.118 mm) over the same window. Mesh never changes size; the
             camera never moves. At `f=0` the skeleton is drawn CENTRED ON
             the model's rest position rather than off in a corner.
  f 120-179  hold: mesh and skeleton co-located at the solver's true final
             position, residual 0.118 mm.

EXPECTATION: at f=0 the (fixed-size, already-scaled) skeleton is centred on
/ surrounding the mesh, not off-screen or entering from a corner; it stays
near the mesh and visibly TRANSLATES with it (not through a huge detour)
over the align window; by f=119 (and through the hold) skeleton and mesh
are co-located at the solver's true position, residual has fallen 1.710 ->
0.118 mm, mesh reads at the same on-screen size it always has, and the
camera's ANGLE matches Act 4's exactly (no visible reorientation at the
cut). The mesh does NOT change size and does NOT rotate at any point; the
camera does NOT zoom, dolly, or orbit at any point.
FALSIFICATION: a skeleton that is NOT roughly centred on the mesh at f=0, or
that visibly leaves the frame and re-enters (see the TASK-15 PIVOT section's
account of why the earlier merged-scale cut did exactly that), means the
staging blend above is not being applied correctly; a skeleton that has not
reached the mesh's exact position by f=119/the hold means the staged-to-true
blend does not reach `p=1`, i.e. the "END state unchanged" guarantee is
broken. A visibly rotating OR resizing mesh means the act is animating
something the solver did not do; a visible "snap" in the fly's apparent
facing direction across the Act3->Act4 cut means the camera angles have
drifted apart again.

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
N_OUT = 180
# root_optimization (translation only) covers the WHOLE act (task-15 v2: no
# rescale step is depicted -- see module docstring's TASK-15 PIVOT section).
ALIGN_START, ALIGN_END = 0, 119
HOLD_START = 120

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

# --- Act 3's OWN fixed camera (task-14 round 3) -----------------------------
# TRUE CONSTANTS -- never read into a per-frame formula, unlike `AZ_START`/
# `ELEV` above (which Act 4 imports for its own, deliberately LIVE camera).
# `ACT3_CAM_DISTANCE` was solved (in an earlier cut) against a REAL
# rendered-pixel measurement (see `_mesh_px` below and the module docstring).
# `ACT3_CAM_AZIMUTH`/`ACT3_CAM_ELEV` are set EQUAL to the shared
# `AZ_START`/`ELEV` (both 120/-20) so Act 3's camera ANGLE matches Act 4's
# exactly -- a previous cut used a different azimuth (40 deg), which broke
# continuity and made the fly visibly "snap" to a different apparent facing
# direction at the Act3->Act4 cut. Distance stays Act-3-local (a cut is
# allowed to change shot distance, just not shot angle).
ACT3_CAM_DISTANCE = 0.93
ACT3_CAM_AZIMUTH = AZ_START
ACT3_CAM_ELEV = ELEV

# Marker/bone radius: FIXED for the whole act (task-15 v2) -- the keypoint
# cloud is drawn at `shared_scale` throughout, never resized, so there is no
# "factor" left to scale these with (contrast task-15 v1 / task-14 round 3,
# where markers scaled with the cloud's own shrink).
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
    multiplier -- `cam.distance = cam_spread * 2.6 / zoom` -- kept for Act 4
    (`_wide_camera_for_aspect`), which never passes it either, so it is
    unaffected here.

    Returns `(cam, live_cloud_spread)` where `live_cloud_spread` is the
    cloud's OWN current extent from this frame's lookat -- used only to size
    the drawn keypoint markers, never the camera.
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


def _progress(f):
    """Progress `p` in [0, 1] for `root_optimization`'s translation,
    smoothstep-eased over `ALIGN_START`-`ALIGN_END`, clamped at 1.0 for the
    hold (`f >= ALIGN_END`). The SAME `p` drives the qpos SLERP, the
    residual interpolation, AND (task-15) the keypoint-cloud STAGING
    centroid blend in `render_act3` -- one schedule, not several
    independently-timed ones.
    """
    if f >= ALIGN_END:
        return 1.0
    return _smoothstep((f - ALIGN_START) / float(ALIGN_END - ALIGN_START))


def _stage_at(f, residual_scaled, residual_root, qpos_default, qpos_root):
    """Return (stage_label, residual_mm, qpos, progress) for output frame
    `f`. Pure function of the recorded stage snapshots and `f` -- no numbers
    invented here, only interpolated between measured stage values. Scale is
    NOT part of this act any more (task-15 v2: keypoints are assumed already
    at `shared_scale` throughout -- see module docstring's TASK-15 PIVOT
    section), so the residual interpolated here starts from
    `residual_scaled` (the measured residual for `qpos_default` fit against
    the ALREADY-scaled keypoints), not `residual_default` (which was
    measured against the unscaled, raw keypoints and no longer applies to
    anything this act draws).
    """
    p = _progress(f)
    stage_label = "root_optimization" if f <= ALIGN_END else "root_optimization (held)"
    resid = (1 - p) * residual_scaled + p * residual_root
    qpos = _slerp_qpos(qpos_default, qpos_root, p)
    return stage_label, resid, qpos, p


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
    residual_scaled, residual_root = float(residual_mm[1]), float(residual_mm[2])

    # Measured-fact guard: root_optimization must not have rotated or moved
    # any joint DOF (module docstring point 2). If a future re-run of
    # stage_ik.py ever produces a rotating root_optimization, this act's
    # entire "translation only" premise would be wrong -- fail loudly rather
    # than silently render the old (incorrect) story.
    if not np.allclose(qpos_default[3:], qpos_root[3:], atol=1e-9):
        raise ValueError(
            "qpos_default[3:] != qpos_root[3:] -- root_optimization rotated "
            "and/or moved joint DOF in this solve. Act 3's premise "
            "(translation only, no rotation, no scale change) no longer "
            "matches the recorded snapshots; refusing to render a "
            "mesh-rotation animation that contradicts the measured facts.")

    with np.load(dirs["predictions"] / "04_kp3d_filt.npz", allow_pickle=True) as z:
        kp3d = np.asarray(z["kp3d"], np.float64)
        kp3d_names = [str(n) for n in z["kp_names"]]
    if kp3d_names != kp_names:
        raise ValueError(
            "04_kp3d_filt.npz kp_names != 06_stages.npz kp_names -- refusing "
            "to mix keypoint orders (CLAUDE.md's keypoint-order bug class).")

    kp3d_raw_frame = kp3d[frame_for_stills]   # (50,3) mm, MODEL order, RAW (unscaled)
    print(f"[act3] frame_for_stills={frame_for_stills}, shared_scale={shared_scale:.4f} "
          f"(FIXED for this act -- root_optimization only, task-15 v2)")
    print(f"[act3] residuals scaled/root = {residual_scaled:.3f}/{residual_root:.3f} mm")
    print(f"[act3] qpos_root - qpos_default (translation only, mm): "
          f"{(qpos_root[:3] - qpos_default[:3])}")

    mj_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    orig_alpha = capture_geom_alpha(mj_model)   # BEFORE any geom_rgba mutation
    grey = PALETTE["mesh"][0] / 255.0   # PALETTE["mesh"] is (200,200,200): BGR==RGB here
    # alpha=1.0 fixed for the whole act (task-15 dropped the rest-pose fade,
    # so there is no longer any alpha ramp to animate per frame).
    set_mesh_rgba(mj_model, orig_alpha, rgb=grey, alpha=1.0)

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

    # --- Cloud, ALREADY scaled to shared_scale (task-15 v2) -- fixed shape
    # for the whole act, no scale animation. `raw_mean`/`scaled_centered`
    # decompose it into a rigid centroid + a zero-mean shape (already scaled);
    # `C_true` is the SINGLE fixed point the staged centroid converges to --
    # see module docstring's TASK-15 PIVOT section for the algebra.
    finite_raw = np.all(np.isfinite(kp3d_raw_frame), axis=-1)
    raw_mean = kp3d_raw_frame[finite_raw].mean(axis=0)
    scaled_centered = shared_scale * (kp3d_raw_frame - raw_mean)
    C_true = shared_scale * raw_mean
    print(f"[act3] staging: C_true (fixed, already-scaled cloud centroid) = "
          f"{C_true} -- staged centroid blends mesh_ctr(f) -> C_true across "
          f"the align window, landing exactly on C_true (the solver's real "
          f"scaled keypoint position) by f=ALIGN_END/through the hold.")

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
    # mid-render EGL resource-leak crash at ~700/900 frames). Act 3's frame
    # count never hit that failure, but this loop is touched here anyway, so
    # it is fixed too rather than left on the known-bad pattern.
    with mujoco.Renderer(mj_model, height=CANVAS_H, width=CANVAS_W) as renderer:
        for f in range(N_OUT):
            stage_label, resid, qpos, p = _stage_at(
                f, residual_scaled, residual_root, qpos_default, qpos_root)

            d = mujoco.MjData(mj_model)
            d.qpos[:] = qpos
            mujoco.mj_forward(mj_model, d)

            # `mesh_ctr` is the ONLY per-frame camera quantity, and it is not
            # zoom/dolly/orbit -- see the module docstring's CAMERA section.
            # This is the mesh's REAL centroid; the camera tracks the true,
            # measured translation, and the staged cloud centroid (below)
            # starts glued to this same point.
            mesh_ctr = np.asarray(d.site_xpos[body_site_idxs]).mean(axis=0)

            # STAGING (task-15 v2): the ALREADY fixed-size cloud's centroid
            # blends from the mesh's own LIVE centroid (`mesh_ctr`, p=0) to
            # the cloud's single true fixed position (`C_true`, p=1) -- see
            # module docstring's TASK-15 PIVOT section for why this is now a
            # well-behaved one-shot blend rather than chasing a moving
            # target.
            staged_centroid = (1.0 - p) * mesh_ctr + p * C_true
            cloud_pts = staged_centroid + scaled_centered

            cam = mujoco.MjvCamera()
            cam.lookat[:] = mesh_ctr
            cam.distance = ACT3_CAM_DISTANCE
            cam.azimuth, cam.elevation = ACT3_CAM_AZIMUTH, ACT3_CAM_ELEV

            renderer.update_scene(d, camera=cam)
            scn = renderer.scene
            for a, b, rgb in skeleton_edges:
                pa, pb = cloud_pts[a], cloud_pts[b]
                if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                    _add_bone(scn, pa, pb, np.array((*rgb, 1.0), np.float32),
                              BASE_MARKER_R_AT_FINAL_SCALE * BONE_RADIUS_FRACTION)
            for i, p3 in enumerate(cloud_pts):
                if np.all(np.isfinite(p3)):
                    rgb = kp_rgb01_by_idx[i]
                    _add_sphere(scn, p3, np.array((*rgb, 1.0), np.float32),
                                BASE_MARKER_R_AT_FINAL_SCALE)
            img_rgb = np.ascontiguousarray(renderer.render())

            canvas = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            canvas = draw.stage_title(canvas, "ACT 3", "Root alignment")
            canvas = draw.label(canvas, f"stage: {stage_label}", (48, 150),
                                 scale=0.6, color=(255, 255, 255))
            canvas = draw.label(
                canvas, f"residual: {resid:.3f} mm (mean over 50 sites; "
                        f"legs/wings lag until Act 4 solves the joints)",
                (48, 182), scale=0.55, color=(190, 255, 190))
            canvas = draw.label(
                canvas, f"keypoints pre-scaled to shared_scale={shared_scale:.4f} "
                        f"(Umeyama trunk fit) -- fixed size; only position animates",
                (48, 210), scale=0.5, color=(190, 190, 190))
            canvas = draw.label(
                canvas, "mesh size is fixed and never rotates; "
                        "root_optimization translates the mesh only",
                (48, 238), scale=0.45, color=(150, 150, 150))
            canvas = draw.label(
                canvas, "keypoints are drawn centred on the model so the alignment "
                        "reads; their final position is the solver's",
                (48, 266), scale=0.45, color=(150, 150, 150))
            canvas = draw.label(canvas, f"frame {f + 1}/{N_OUT}", (48, CANVAS_H - 24),
                                 scale=0.45, color=(150, 150, 150))

            if f in (0, 60, 119, 179):
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
