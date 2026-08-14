#!/usr/bin/env python3
"""Act 4 -- the joints solve, then the fit plays back in motion.

This act carries the ORIENTING beat of the whole explainer. Act 3 covered
scale + translation only: `root_optimization` measured 0.0000 deg of
rotation (`TRUNK_OPTIMIZATION_KEYPOINTS` is empty in `configs/anatomy/v1.yaml`,
so its keypoint-fit loss is uniformly zero; see `act3_align.py`'s docstring).
ALL the rotation happens here, in `pose_optimization`. Measured on
`06_stages.npz` at `frame_for_stills=450` (asserted at runtime, not just
claimed): the quaternion rotates **34.73 deg** from `qpos_root` to
`qpos_pose`, and `qpos[7:]` (58 joint DOF) moves an L2 norm of **2.74** --
simultaneously, from one jaxls trajectory solve
(`compute_stac.pose_optimization`'s `JAXLS_ORIENTATION_KEYPOINTS` warm-start
fits heading at the same time the per-frame joint angles are fit). The first
240 frames of this act are where the fly visibly swings into its true
heading AND its limbs snap onto the keypoints, together -- not two separate
beats.

The rotation is a real, per-frame solve, not a staged pose: root heading
varies over the 921-frame sequence (`qpos_seq`), so Phase B's playback shows
genuinely solved per-frame rotation.

THREE PHASES (900 frames, 30 fps -> 30 s):
  0-239   (Phase A) qpos interpolates qpos_root -> qpos_pose at the single
          `frame_for_stills` (450), quaternion SLERPed (`_slerp_qpos`,
          reused unmodified from `act3_align.py` -- it was written there
          specifically so this act's real 34.73 deg rotation would have a
          SLERP, not a lerp, ready for it). The keypoint-cloud TARGET is
          held fixed at `kp3d_scaled[frame_for_stills]` (the same target
          `stage_ik.py` fit against); residual is the REAL residual of the
          interpolated qpos against that fixed target
          (`stage_ik.marker_residual_mm`, recomputed every frame, not
          lerped between the two recorded endpoints) -- it lands on 0.118 mm
          at f=0 and 0.017 mm at f=239 because those ARE `qpos_root` and
          `qpos_pose`, not because a number was faked to match.
  240-779 (Phase B) `qpos_seq` (921 frames) plays back, continuously mapped
          onto 540 output frames (same `t = round(rel * (T-1) / REL_MAX)`
          style Act 1 uses to span a longer source clip onto fewer output
          frames). The keypoint cloud (`kp3d[t] * shared_scale`, coloured per
          limb chain, JARVIS scheme -- `kp_colors.jarvis_kp_colors_rgb01`,
          shared with Act 3) advances in lock-step with the mesh, and
          residual is recomputed every frame the same way as Phase A.
  780-899 (Phase C) the closing 2-up, continuing the SAME t-mapping from
          Phase B (no jump): left is the mesh+cloud render (narrower, 960 px
          wide); right is one real camera's video, cropped and centred the
          same way Act 1 crops (`act1_views._smoothed_crop_x0`, reused), with
          the FK'd tracking sites reprojected through that camera's DLT
          (green, `PALETTE["fit"]`) over the observed 2D keypoints (cyan,
          `PALETTE["fly0"]`).

COORDINATE-FRAME CAVEAT that made the 2-up correctness non-obvious (measured,
not assumed): `pose_optimization` fits the model against `kp3d * shared_scale`
(see `stage_ik.py`), so the model's own `site_xpos` after forward kinematics
live in that SCALED coordinate frame, not the raw arena-mm frame the clip's
DLTs were calibrated in. Reprojecting `site_xpos` through `clip_io.project`
directly would be reprojecting the wrong points. Verified once, empirically,
before writing the render loop: dividing `site_xpos` by `shared_scale` before
projecting reproduces the OBSERVED 2D keypoints (`02_kp2d.npz`) to a mean
~8.5 px, max ~24 px (comparable to Act 2's own 28.5 px worst-case reprojection
tolerance against the same DLTs) -- confirming `site_xpos / shared_scale` is
the right quantity to feed `clip_io.project`, not `site_xpos` itself.

CAMERA (mesh panels): reuses `act3_align._wide_camera` (never the model's
`hero` camera, which frames the mesh only and drops the far-away keypoint
cloud out of view) for lookat/distance construction; azimuth/elevation are
FIXED at `act3_align.AZ_START`/`ELEV` (120/-20 deg) for the whole act,
continuing Act 3's own fixed values so the two acts don't visually "jump"
cameras at the cut, and -- as in Act 3 -- so a moving VIEWER azimuth is never
misread as the mesh itself rotating, which matters even more here since the
mesh really DOES rotate 34.73 deg on its own.

Distance is LIVE here (recomputed every frame), a DELIBERATE difference from
Act 3's post-review fix (checked explicitly, not assumed -- see the task-11
review this act inherited: a live per-frame distance there was CANCELLING an
intentional cloud-size reveal, since the camera zoomed in exactly as fast as
the cloud shrank). Act 4 has no analogous reveal: Phase A is a pure rotation
(mesh/cloud stay near-coincident throughout -- residual 0.118->0.017 mm --
so live vs. frozen spread is nearly identical there anyway), and Phases B/C
play back 921 frames of REAL fly motion, where adaptive per-frame framing is
the wide camera's actual job (a single frozen distance sized for one part of
the clip could crop a wide leg swing elsewhere, or leave a compact pose
looking tiny). `_wide_camera_for_aspect` gets this live value by calling
`_wide_camera` (its `cam_spread` argument is now a required, externally-
supplied value per the task-11 fix, not internally derived) twice per frame:
once to read back the live spread, once more to build the camera from it --
both calls are pure geometry, no rendering, so free.

TASK-17 ("seamless Act3->Act4 cut"): measuring Act3's last frame
(`act3_align/f00089.png`) against Act4's first (`act4_solve/f00000.png`)
directly found a real, on-screen jump -- mesh centre shifted (-24,-44) px,
mesh area 1.22x bigger, skeleton centre shifted (-11,-27) px. Root cause,
verified before touching anything: Act 3's camera is fully FROZEN
(`act3_align.ACT3_CAM_DISTANCE`/`_AZIMUTH`/`_ELEV`, `lookat=mesh_ctr_final`),
while this act's Phase A camera (above) was ALREADY live (same
`_wide_camera_for_aspect` call Phase B uses) -- so the two acts simply
disagreed about camera distance/lookat at the exact frame they're supposed
to hand off at.

Fix: Phase A (0-239) now renders with `act3_align.act3_frozen_camera`, the
SAME function Act 3 itself calls, given THIS act's own `mesh_ctr_final` --
computed identically to Act 3's own (FK against `qpos_root`, same
`body_site_idxs`), so the two acts share the identical camera number rather
than each re-deriving an approximation of it. This is a legitimate staging
choice, not just a patch: Phase A is a frozen-time interpolation (the same
single `frame_for_stills` target the whole beat), so a fixed camera suits it
exactly as it suits Act 3. `_wide_camera_for_aspect` is still called every
Phase-A/B frame for its OTHER output, `spread` (marker/bone sizing only,
left live and unchanged) -- only the `cam` it returns is overridden for
Phase A.

Phase B (>=240) resumes this SAME live/adaptive `_wide_camera_for_aspect`
camera UNMODIFIED -- not eased in from the frozen value. An eased hand-off
(smoothstep-blending `lookat`/`distance` from the frozen value back to live
over the first ~30 frames of Phase B) was tried FIRST, per this task's own
"hold through Phase A, ease over the first second of Phase B" suggestion,
and rejected after measuring and LOOKING at the result, not on suspicion:
`qpos_pose` (frame_for_stills=450, what f=239 renders) and `qpos_seq[0]`
(playback's own first real frame, what f=240 renders) differ by a REAL 1.06
model-unit root translation and ~25 deg of rotation -- Phase A solves toward
one specific mid-clip frame while Phase B immediately starts playing back
the trajectory from t=0, a genuine content discontinuity in the DATA, not an
artifact of this fix. A camera that is still mostly at the FROZEN value
(as any smooth ease necessarily is for its first several frames, by
construction -- `e` starts at 0 at f=240 no matter how short the window) has
to view that already-relocated content through the OLD, un-adapted
lookat/distance -- measured directly: f=240 under the 30-frame ease had its
mesh mask centred at px (119, 410), a 858 px / 154 px shift off Act 3's
frozen framing and the mesh clipped hard against the canvas's LEFT edge
(confirmed by opening the rendered PNG, not just the mask numbers) for a
visible stretch of Phase B's opening second -- exactly the "fly walks out of
a frozen frame" failure this task's own "When You're in Over Your Head"
section named as the trigger to stop and report a trade-off instead of
silently shipping either extreme.

Resuming the live camera on Phase B's very first frame (no easing at all)
is measurably the better of the two options: it matches the ORIGINAL,
already-shown-to-work design (a live camera re-centres on WHATEVER content
is on screen every frame, so it was already seamless at this exact seam
before Phase A was ever frozen), and it keeps the mesh framed correctly for
every one of Phase B's 660 frames. The cost is a bounded, small mismatch
right at the f=239/f=240 cut itself (Phase A's frozen f=239 no longer
perfectly tracks its own content the way a live f=239 would have, so it
doesn't line up quite as tightly with Phase B's now-fully-adaptive f=240) --
see the task-17 report for the exact measured numbers. This residual is far
smaller than either the original Act3/Act4 jump this task fixes or the
858 px/clipped-frame failure the slow-ease alternative produced, but it is
not literally zero; reported as the deliberate trade-off it is, not hidden.

Phase C's left panel is rendered at a narrower (960x1080, not 1920x1080)
aspect than Acts 3/A/B; `_wide_camera_for_aspect` scales the distance by
`1/aspect` for aspect < 1 so the same mesh+cloud content that fit the wide
16:9 frame doesn't get clipped left/right in the narrower panel (derived once
from the pinhole geometry, not tuned by eye: at the base 2.6x-spread
distance, the binding constraint is vertical for aspect >= 1 -- horizontal
FOV is always the wider of the two there -- but flips to horizontal once
aspect < 1, requiring distance to grow by 1/aspect to keep the same
real-world width in frame).

2-UP MARKER COLOURS: `draw.draw_keypoints`/`draw.draw_leg_chains` hard-code
PER-GROUP anatomical colours (Acts 1-2's convention). The 2-up wants a
different, two-colour contrast instead -- ALL observed points cyan, ALL fit
points green, so the fit-vs-observed comparison reads at a glance rather than
needing per-keypoint anatomy recall. `_dots` and `_chain` below are small
single-colour analogues of those helpers (same primitives, same
`leg_chains`/iteration pattern, imported not reimplemented), parameterised by
colour instead of group, only used in this one panel.

EXPECTATION: by f=239 the mesh's limbs lie along the keypoint chains and the
body has visibly rotated from Act 3's ending heading; residual reads 0.017 mm.
Through 240-779, tarsal tips TRACK the observed keypoints through leg swing
-- the mesh foot stays ON the marker as the leg moves, not merely near it.
In the closing 2-up, the green reprojected fit overlies the fly in the real
video, cyan observed keypoints alongside.
FALSIFICATION: tips detaching from markers during swing means `pose_
optimization` did not converge, or the wrong qpos frame is being drawn; a
tumbling/flipping body during 0-239 means the quaternion was lerped instead
of SLERPed; the 2-up markers sitting off the fly means the shared_scale
coordinate-frame division above is missing or inverted.

Colours come from `kp_colors.jarvis_kp_colors_rgb01` (JARVIS per-limb-chain
scheme, Change 2) for the mesh+cloud panels; `viz.core.colors.PALETTE`
(fit=green, observed=cyan) is UNCHANGED for the closing 2-up (see "2-UP
MARKER COLOURS" above). Wing visibility (Change 1): `set_mesh_rgba`/
`capture_geom_alpha` (stage_ik.py) keep originally-invisible geoms (the
wings' `*_inertial` boxes) at alpha=0 across every `geom_rgba` mutation.

SKELETON (task-14 round 4): the cloud is now drawn as bones, not loose
spheres, here too -- same `kp_colors.jarvis_skeleton_edges`/`stage_ik._add_bone`
construction Act 3 uses (`data/fly50.json`'s 44 edges, mapped by NAME,
coloured per JARVIS's own stop-node convention), in BOTH the Phase A/B full
mesh+cloud panel and Phase C's narrower left panel. The closing 2-up's
RIGHT panel (real video, `_chain`/`_dots`) is unaffected -- it already drew
leg-chain lines and keeps the PALETTE fit/observed convention untouched.
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
from scipy.spatial.transform import Rotation

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))
sys.path.insert(0, str(_REPO / "stac-mjx"))

from scripts.viz.ik_explainer import clip_io, draw                    # noqa: E402
from scripts.viz.ik_explainer.stage_ik import (                       # noqa: E402
    _add_sphere, _add_bone, _tracking_site_map, marker_residual_mm,
    capture_geom_alpha, set_mesh_rgba,
)
from scripts.viz.ik_explainer.kp_colors import (                      # noqa: E402
    jarvis_kp_colors_rgb01, jarvis_skeleton_edges,
)
from scripts.viz.ik_explainer.acts.act3_align import (                 # noqa: E402
    _slerp_qpos, _wide_camera, _smoothstep, AZ_START, ELEV, BONE_RADIUS_FRACTION,
    act3_frozen_camera, ACT3_CAM_DISTANCE,
)
from scripts.viz.ik_explainer.acts.act1_views import _smoothed_crop_x0  # noqa: E402
from viz.core.colors import PALETTE, leg_chains                       # noqa: E402

# --- canvas / timeline ------------------------------------------------------
CANVAS_W, CANVAS_H = 1920, 1080
N_OUT = 900
PHASE_A_END = 239                 # inclusive: qpos_root -> qpos_pose, one frame
PHASE_B_END = 779                 # inclusive: qpos_seq playback, full width
# Phase C: 780..899 inclusive -- closing 2-up

XML_PATH = _REPO / "models" / "fruitfly_v1" / "fruitfly_v1_free.xml"

# Continuous t-mapping shared by Phase B and Phase C so the cut into the 2-up
# does not skip or repeat time: rel=0 at f=240 (start of Phase B) maps to
# source frame 0, rel=REL_MAX at f=899 (end of Phase C) maps to the clip's
# last frame.
REL_MAX = (N_OUT - 1) - (PHASE_A_END + 1)   # 899 - 240 = 659

# Chosen for the closing 2-up: brief names Cam2012862/Cam2012630 as reading
# clearly in earlier acts (Cam2012857 explicitly called out as the hardest/
# least legible). Verified directly on this clip before picking: reprojecting
# the pose-stage FK sites (frame 450) through every camera's DLT and
# comparing to that camera's OWN observed 2D keypoints gives comparable
# accuracy across all 7 (8.4-9.4 px mean) -- so the choice is about legibility
# of the crop, not reprojection accuracy. Cam2012862 has a fly-keypoint pixel
# bbox of y in [134,361] (227 px span, mid-pack) vs. Cam2012857's [115,248]
# (compressed near-vertical view called out as hardest to read).
CAM_2UP = "Cam2012862"
SRC_FRAME_W = 1936                # native video width (matches act1_views.py)
CROP_W = 430                      # same fly-centred crop window as Act 1
FRAME_H = 448                     # full native strip height (no vertical pan)


def _quat_angle_deg(qa_wxyz, qb_wxyz) -> float:
    """Angle (deg) of the rotation taking quaternion a to quaternion b."""
    qa = np.roll(np.asarray(qa_wxyz, np.float64), -1)   # wxyz -> xyzw
    qb = np.roll(np.asarray(qb_wxyz, np.float64), -1)
    rel = Rotation.from_quat(qb) * Rotation.from_quat(qa).inv()
    return float(np.degrees(rel.magnitude()))


def _wide_camera_for_aspect(mesh_ctr, cloud_pts, model_extent, azimuth, aspect):
    """`act3_align._wide_camera`, with a LIVE (not frozen) per-frame distance,
    corrected for non-16:9 panels.

    Task-11 review froze Act 3's `cam_spread` for the whole act because a
    LIVE per-frame distance there exactly cancelled an intentional cloud-SIZE
    reveal (the camera zoomed in as fast as the cloud shrank, so the shrink
    never became visible on screen). Act 4 has no analogous reveal to hide:
    Phase A is a pure ROTATION (residual stays tiny -- 0.118->0.017 mm --
    throughout, so mesh/cloud spread is already near-constant; a live-vs-
    frozen distance makes no visible difference for a rotation), and Phases
    B/C play back the fly's REAL trajectory over 921 frames, where adaptive
    framing is the wide camera's actual job (a single frozen distance sized
    for one part of the clip could crop a wide leg-swing elsewhere, or leave
    a compact pose looking tiny). So this deliberately keeps `_wide_camera`'s
    `cam_spread` LIVE per frame instead of freezing it -- reusing the same
    function (and its lookat/lighting/vertical-FOV formula) rather than
    reimplementing the camera, just supplying a live-computed `cam_spread`
    instead of a pre-computed dry-pass one. (`_wide_camera` needs `cam_spread`
    to build `cam`, but only EXPOSES the live spread as part of its return
    value -- computing it requires the same `lookat` the function derives
    internally -- so this calls it twice: once with a placeholder distance
    purely to read back `live_cloud_spread`, then again with that value to
    build the actually-used camera. Both calls are pure geometry, no
    rendering, so the extra call is free.)

    Aspect correction (unchanged from the first cut): at the base `distance =
    spread * 2.6`, vertical FOV is the binding constraint for aspect >= 1
    (Acts 3/4's 1920x1080 panels, unaffected). Once aspect < 1 (Phase C's
    960-wide left panel), horizontal binds instead and distance must grow by
    1/aspect to keep the same real-world width in frame (pinhole relation
    tan(horiz_half) = aspect * tan(vert_half), not tuned by eye).
    """
    _, live_spread = _wide_camera(mesh_ctr, cloud_pts, model_extent, azimuth, cam_spread=1.0)
    cam, live_spread = _wide_camera(mesh_ctr, cloud_pts, model_extent, azimuth, cam_spread=live_spread)
    if aspect < 1.0:
        cam.distance = cam.distance / aspect
    return cam, live_spread


def _dots(img, uv, color, radius=3):
    """Single-colour analogue of `draw.draw_keypoints` (no per-group colour).

    Same primitive (`cv2.circle`, NaN-skipped) as draw.py; only the colour
    argument differs, so the 2-up's cyan-observed/green-fit contrast doesn't
    have to fight `draw.draw_keypoints`'s baked-in anatomical palette.
    """
    out = np.asarray(img).copy()
    for p in np.asarray(uv):
        if np.all(np.isfinite(p)):
            cv2.circle(out, tuple(np.round(p).astype(int)), radius, color, -1, cv2.LINE_AA)
    return out


def _chain(img, uv, kp_names, color, thickness=1):
    """Single-colour analogue of `draw.draw_leg_chains`; reuses `leg_chains`."""
    out = np.asarray(img).copy()
    for _leg, chain in leg_chains(list(kp_names)).items():
        pts = np.asarray(uv)[chain]
        for a, b in zip(pts[:-1], pts[1:]):
            if np.all(np.isfinite([a, b])):
                cv2.line(out, tuple(np.round(a).astype(int)),
                          tuple(np.round(b).astype(int)), color, thickness, cv2.LINE_AA)
    return out


def render_act4(clip: str = clip_io.CLIP_DEFAULT, start_frame: int = 0) -> Path:
    """`start_frame` resumes a render that already wrote frames 0..start_frame-1
    (e.g. after an interrupted run) without recomputing or re-rendering them --
    all setup above this point is cheap (numpy/IO), only the per-frame MuJoCo
    render loop is skipped ahead."""
    dirs = clip_io.out_dirs(clip)

    with np.load(dirs["predictions"] / "06_stages.npz", allow_pickle=True) as z:
        qpos_root = np.asarray(z["qpos_root"], np.float64)
        qpos_pose = np.asarray(z["qpos_pose"], np.float64)
        qpos_seq = np.asarray(z["qpos_seq"], np.float64)
        residual_mm = np.asarray(z["residual_mm"], np.float64)
        stage_names = [str(s) for s in z["stage_names"]]
        kp_names = [str(n) for n in z["kp_names"]]
        frame_for_stills = int(z["frame_for_stills"])
        shared_scale = float(z["shared_scale"])

    assert stage_names == ["default", "scaled", "root", "pose"], (
        f"unexpected 06_stages.npz stage_names order: {stage_names}")
    residual_root, residual_pose = float(residual_mm[2]), float(residual_mm[3])
    T = qpos_seq.shape[0]
    assert np.allclose(qpos_seq[frame_for_stills], qpos_pose), (
        "qpos_seq[frame_for_stills] != qpos_pose -- frame_for_stills is not "
        "actually the trajectory frame the still-fit stage recorded")

    # Measured-fact guard: pose_optimization must have rotated non-trivially
    # (module docstring's whole premise). If a future re-solve makes this
    # stage stop rotating (e.g. TRUNK_OPTIMIZATION_KEYPOINTS becomes
    # non-empty and root_optimization starts doing the orienting instead),
    # this act's "orienting beat lives here" story would be wrong -- fail
    # loudly rather than silently render a stale claim.
    root_pose_angle_deg = _quat_angle_deg(qpos_root[3:7], qpos_pose[3:7])
    if root_pose_angle_deg < 1.0:
        raise ValueError(
            f"root->pose rotation is only {root_pose_angle_deg:.3f} deg -- "
            "Act 4's premise (pose_optimization performs the real orienting) "
            "no longer matches the recorded snapshots; refusing to render."
        )
    joint_dof_delta = float(np.linalg.norm(qpos_pose[7:] - qpos_root[7:]))
    print(f"[act4] root->pose rotation: {root_pose_angle_deg:.2f} deg, "
          f"joint-DOF L2 delta: {joint_dof_delta:.2f}")

    with np.load(dirs["predictions"] / "04_kp3d_filt.npz", allow_pickle=True) as z:
        kp3d = np.asarray(z["kp3d"], np.float64)          # (T,50,3) mm, MODEL order, RAW
        kp3d_names = [str(n) for n in z["kp_names"]]
    if kp3d_names != kp_names:
        raise ValueError(
            "04_kp3d_filt.npz kp_names != 06_stages.npz kp_names -- refusing "
            "to mix keypoint orders (CLAUDE.md's keypoint-order bug class).")
    if kp3d.shape[0] != T:
        raise ValueError(f"kp3d has {kp3d.shape[0]} frames, qpos_seq has {T}")

    with np.load(dirs["predictions"] / "02_kp2d.npz", allow_pickle=True) as z:
        kp2d = np.asarray(z["kp2d"], np.float64)          # (T,C,50,2) full-frame px
        cam_names = [str(c) for c in z["cam_names"]]
        kp2d_names = [str(n) for n in z["kp_names"]]
    if kp2d_names != kp_names:
        raise ValueError("02_kp2d.npz kp_names != 06_stages.npz kp_names.")
    if CAM_2UP not in cam_names:
        raise ValueError(f"{CAM_2UP} not in 02_kp2d.npz cam_names {cam_names}")
    cam2up_idx = cam_names.index(CAM_2UP)

    cam_mats, dlt_names = clip_io.load_dlt(str(Path(clip) / "calibration"))
    if dlt_names != cam_names:
        raise ValueError(
            f"DLT camera order {dlt_names} != 02_kp2d.npz cam_names {cam_names} "
            "-- refusing to project with a mismatched camera/matrix pairing.")

    mj_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    orig_alpha = capture_geom_alpha(mj_model)   # BEFORE any geom_rgba mutation
    grey = PALETTE["mesh"][0] / 255.0
    # Change 1 (task-14): set_mesh_rgba keeps originally-invisible geoms
    # (the wings' *_inertial boxes, alpha=0 by design) at alpha=0 -- a blanket
    # `geom_rgba[:, 3] = 1.0` assignment used to turn them into opaque white
    # rectangles. See stage_ik.py's `set_mesh_rgba` docstring.
    set_mesh_rgba(mj_model, orig_alpha, rgb=grey, alpha=1.0)

    site_map = _tracking_site_map(mj_model, kp_names)
    missing = [n for n in kp_names if n not in site_map]
    if missing:
        raise ValueError(f"tracking[...] site missing for keypoints: {missing}")
    body_site_idxs = np.asarray([site_map[n] for n in kp_names])
    # Change 2 (task-14): each limb chain gets its own JARVIS colour instead
    # of the previous 4-group (head/thorax/abdomen/legs) scheme.
    kp_rgb01 = jarvis_kp_colors_rgb01(kp_names)
    kp_rgb01_by_idx = [kp_rgb01[n] for n in kp_names]
    # Skeleton bones (task-14 round 4, "skeleton on Act 4 too"): same
    # construction as act3_align.py -- data/fly50.json's 44 edges mapped onto
    # THIS array's own kp_names order by name, coloured per JARVIS's own
    # `colors[line[1]]` (stop-node) convention.
    skeleton_edges = [
        (a, b, (bg[2] / 255.0, bg[1] / 255.0, bg[0] / 255.0))
        for a, b, bg in jarvis_skeleton_edges(kp_names)
    ]

    kp3d_scaled = kp3d * shared_scale                     # same target pose_optimization fit

    # Phase A's fixed target cloud (frame_for_stills, scaled -- the SAME
    # target `stage_ik.py` measured residual_root/residual_pose against).
    target_a = kp3d_scaled[frame_for_stills]

    # Task-17 ("seamless Act3->Act4 cut"): the mesh's own final centroid,
    # computed EXACTLY the way act3_align.py computes `mesh_ctr_final` --
    # FK against the SAME `qpos_root`, averaged over the SAME
    # `body_site_idxs` -- so Phase A's camera lookat (below) is the
    # IDENTICAL number Act 3's last frame used, not a re-derived
    # approximation of it. See module docstring's CAMERA section.
    d_root_ref = mujoco.MjData(mj_model)
    d_root_ref.qpos[:] = qpos_root
    mujoco.mj_forward(mj_model, d_root_ref)
    mesh_ctr_final = np.asarray(d_root_ref.site_xpos[body_site_idxs]).mean(axis=0)
    print(f"[act4] task-17: Phase A frozen camera distance={ACT3_CAM_DISTANCE} "
          f"azimuth/elevation={AZ_START}/{ELEV} lookat=mesh_ctr_final={mesh_ctr_final} "
          f"(identical formula to act3_align.py's own mesh_ctr_final -- shared "
          f"via act3_frozen_camera); Phase B resumes the live camera "
          f"unmodified at f={PHASE_A_END + 1} (see module docstring's CAMERA "
          f"section for why an eased hand-off was tried and reverted)")

    # --- Phase C prerequisites: smoothed crop centring + real video frames -
    x0_full = _smoothed_crop_x0(kp2d[:, cam2up_idx], SRC_FRAME_W)   # (T,)

    def t_for_output_frame(f: int) -> int:
        rel = f - (PHASE_A_END + 1)
        return int(round(rel * (T - 1) / REL_MAX))

    t_playback = np.array([t_for_output_frame(f) for f in range(PHASE_A_END + 1, N_OUT)])
    assert np.all(np.diff(t_playback) >= 0), "playback time mapping must be non-decreasing"
    print(f"[act4] Phase B/C playback spans source frames "
          f"{t_playback.min()}-{t_playback.max()} of {T} over "
          f"{len(t_playback)} output frames")

    phase_c_t = t_playback[(PHASE_B_END + 1) - (PHASE_A_END + 1):]
    video_frames = clip_io.read_frames(clip_io.video_path(clip, CAM_2UP), phase_c_t)
    t_to_video_row = {}
    for row, t in enumerate(phase_c_t):
        t_to_video_row.setdefault(int(t), row)

    out_dir = dirs["frames"] / "act4_solve"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = mujoco.MjData(mj_model)
    t0 = time.time()
    left_w = CANVAS_W // 2

    # --- Two PERSISTENT renderers, created ONCE and reused for every frame --
    # (matching the repo's own established multi-frame idiom --
    # `scripts/viz/render_reference_clips.py:render_clip` creates one
    # `mujoco.Renderer` and calls `update_scene`/`render` per frame inside it
    # -- rather than `act3_align.py`/`stage_ik.py`'s per-frame `with
    # mujoco.Renderer(...) as renderer:`. That per-frame-construction pattern
    # is the prime suspect for this act's mid-render crash: it worked for
    # Act 3's 600 frames but this act creates up to 900+120 fresh EGL/GL
    # contexts, well past that, and died silently (no Python traceback, no
    # OOM, no dmesg segfault entry -- consistent with an accumulating
    # GPU/EGL resource leak from repeated context creation, worsened by a
    # concurrent, unrelated job contending for the GPU at the time). Two
    # renderers (one per output resolution needed: full 1920x1080 for
    # Phases A/B, 960x1080 for Phase C's left panel) removes ~900 of those
    # constructions regardless of the exact leak mechanism.
    with mujoco.Renderer(mj_model, height=CANVAS_H, width=CANVAS_W) as renderer_full, \
         mujoco.Renderer(mj_model, height=CANVAS_H, width=left_w) as renderer_left:

        for f in range(start_frame, N_OUT):
            if f <= PHASE_A_END:
                phase = "A"
                p = _smoothstep(f / float(PHASE_A_END))
                qpos = _slerp_qpos(qpos_root, qpos_pose, p)
                cloud_pts = target_a
                t_src = frame_for_stills
            else:
                phase = "B" if f <= PHASE_B_END else "C"
                t_src = int(t_for_output_frame(f))
                qpos = qpos_seq[t_src]
                cloud_pts = kp3d_scaled[t_src]

            d.qpos[:] = qpos
            mujoco.mj_forward(mj_model, d)
            sites = np.asarray(d.site_xpos)[body_site_idxs]
            resid = marker_residual_mm(mj_model, d, cloud_pts, body_site_idxs)
            mesh_ctr = sites.mean(axis=0)

            if phase in ("A", "B"):
                aspect = CANVAS_W / CANVAS_H
                cam, spread = _wide_camera_for_aspect(
                    mesh_ctr, cloud_pts, mj_model.stat.extent, AZ_START, aspect)
                # Task-17: Phase A no longer uses the live camera computed
                # above for FRAMING (only `spread`, still live, for marker/
                # bone sizing) -- it renders with the EXACT frozen camera
                # Act 3's last frame used, so the Act3->Act4 cut does not
                # jump. Phase B resumes this SAME live `cam` unmodified --
                # see module docstring's CAMERA section for why an eased
                # (rather than immediate) hand-off was tried and reverted:
                # measured directly, it left the mesh clipped against the
                # frame's left edge for a visible stretch of Phase B.
                if phase == "A":
                    cam = act3_frozen_camera(mesh_ctr_final)
                marker_r = max(spread * 0.03, mj_model.stat.extent * 0.006)
                bone_r = marker_r * BONE_RADIUS_FRACTION

                renderer_full.update_scene(d, camera=cam)
                scn = renderer_full.scene
                for a, b, rgb in skeleton_edges:
                    pa, pb = cloud_pts[a], cloud_pts[b]
                    if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                        _add_bone(scn, pa, pb, np.array((*rgb, 1.0), np.float32), bone_r)
                for i, p3 in enumerate(cloud_pts):
                    if np.all(np.isfinite(p3)):
                        rgb = kp_rgb01_by_idx[i]
                        _add_sphere(scn, p3, np.array((*rgb, 1.0), np.float32), marker_r)
                canvas = cv2.cvtColor(np.ascontiguousarray(renderer_full.render()), cv2.COLOR_RGB2BGR)

                canvas = draw.stage_title(canvas, "ACT 4", "Joint solve, then motion")
                if phase == "A":
                    stage_label = "pose_optimization -- orientation + joints, together"
                    extra = (f"rotated so far: {root_pose_angle_deg * p:.1f} / "
                              f"{root_pose_angle_deg:.2f} deg    "
                              f"joint-DOF moved: {joint_dof_delta * p:.2f} / {joint_dof_delta:.2f}")
                else:
                    stage_label = "pose_optimization (playback)"
                    extra = f"source frame {t_src}/{T - 1}"
                canvas = draw.label(canvas, f"stage: {stage_label}", (48, 150),
                                     scale=0.6, color=(255, 255, 255))
                canvas = draw.label(canvas, f"residual: {resid:.3f} mm", (48, 182),
                                     scale=0.6, color=(190, 255, 190))
                canvas = draw.label(canvas, extra, (48, 210), scale=0.5, color=(190, 190, 190))
                canvas = draw.label(
                    canvas, "root_optimization did not rotate the body (Act 3); "
                            "pose_optimization does, and bends every joint at once",
                    (48, 238), scale=0.45, color=(150, 150, 150))
                canvas = draw.label(canvas, f"frame {f + 1}/{N_OUT}", (48, CANVAS_H - 24),
                                     scale=0.45, color=(150, 150, 150))

            else:  # Phase C: closing 2-up
                aspect = left_w / CANVAS_H
                cam, spread = _wide_camera_for_aspect(
                    mesh_ctr, cloud_pts, mj_model.stat.extent, AZ_START, aspect)
                marker_r = max(spread * 0.03, mj_model.stat.extent * 0.006)
                bone_r = marker_r * BONE_RADIUS_FRACTION

                renderer_left.update_scene(d, camera=cam)
                scn = renderer_left.scene
                for a, b, rgb in skeleton_edges:
                    pa, pb = cloud_pts[a], cloud_pts[b]
                    if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                        _add_bone(scn, pa, pb, np.array((*rgb, 1.0), np.float32), bone_r)
                for i, p3 in enumerate(cloud_pts):
                    if np.all(np.isfinite(p3)):
                        rgb = kp_rgb01_by_idx[i]
                        _add_sphere(scn, p3, np.array((*rgb, 1.0), np.float32), marker_r)
                left = cv2.cvtColor(np.ascontiguousarray(renderer_left.render()), cv2.COLOR_RGB2BGR)

                left = draw.stage_title(left, "ACT 4", "MuJoCo fit")
                left = draw.label(left, f"residual: {resid:.3f} mm", (48, 150),
                                   scale=0.55, color=(190, 255, 190))
                left = draw.label(left, f"source frame {t_src}/{T - 1}", (48, 178),
                                   scale=0.5, color=(190, 190, 190))

                # --- right panel: real video, FK fit reprojected in green,
                # observed 2D in cyan ---
                row = t_to_video_row[t_src]
                native = video_frames[row].copy()               # (FRAME_H, SRC_FRAME_W, 3)
                x0 = float(x0_full[t_src])

                sites_raw = sites / shared_scale                 # back to raw arena-mm frame
                proj_all = clip_io.project(cam_mats[cam2up_idx:cam2up_idx + 1], sites_raw)
                fit_uv = proj_all[0].copy()
                fit_uv[:, 0] -= x0
                obs_uv = kp2d[t_src, cam2up_idx].copy()
                obs_uv[:, 0] -= x0

                panel_native = native[:FRAME_H, int(round(x0)):int(round(x0)) + CROP_W]
                panel_native = _chain(panel_native, obs_uv, kp_names, PALETTE["fly0"], thickness=1)
                panel_native = _dots(panel_native, obs_uv, PALETTE["fly0"], radius=3)
                panel_native = _chain(panel_native, fit_uv, kp_names, PALETTE["fit"], thickness=1)
                panel_native = _dots(panel_native, fit_uv, PALETTE["fit"], radius=3)
                right = cv2.resize(panel_native, (CANVAS_W - left_w, CANVAS_H),
                                    interpolation=cv2.INTER_LINEAR)
                right = draw.label(right, f"{CAM_2UP} -- real video", (24, 40),
                                    scale=0.6, color=(255, 255, 255))
                right = draw.label(right, "cyan: observed 2D   green: FK fit reprojected",
                                    (24, 68), scale=0.5, color=(200, 200, 200))

                canvas = np.concatenate([left, right], axis=1)
                canvas = draw.label(canvas, f"frame {f + 1}/{N_OUT}", (48, CANVAS_H - 24),
                                     scale=0.45, color=(150, 150, 150))

            cv2.imwrite(str(out_dir / f"f{f:05d}.png"), canvas)
            if (f + 1) % 25 == 0 or f == start_frame:
                print(f"[act4] wrote frame {f} ({phase}) t_src={t_src} "
                      f"({time.time() - t0:.1f}s elapsed)", flush=True)

    dt = time.time() - t0
    n_written = N_OUT - start_frame
    print(f"[act4] wrote {n_written} frames (start_frame={start_frame}) to "
          f"{out_dir} in {dt:.1f} s ({dt / max(n_written, 1):.3f} s/frame)")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    ap.add_argument("--start-frame", type=int, default=0,
                     help="resume rendering from this output frame (0..899); "
                          "frames before it are assumed already written")
    args = ap.parse_args()
    render_act4(args.clip, start_frame=args.start_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
