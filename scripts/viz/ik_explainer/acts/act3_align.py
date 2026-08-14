#!/usr/bin/env python3
"""Act 3 -- root alignment ONLY, keypoints assumed already scaled (task-15 v2,
a direct mid-task user pivot away from the earlier "merged scale+align" cut),
now trimmed to JUST the swing-in-and-align beat (task-16, a second direct
user pivot -- see the TASK-16 PIVOT section below for why the mesh itself no
longer visibly translates in this cut).

This act renders SNAPSHOTS `stage_ik.py` recorded from an actual STAC solve
(`06_stages.npz`) -- it does not invent motion. Two measured facts underlie
the story (do not restate the pre-measurement storyboard):

1. **Body scale is applied to the KEYPOINTS, not the mesh, and is NOT
   animated in this act.** Scale comes from
   `preprocess_keypoints_for_ik.compute_shared_scale` (Umeyama trunk scale;
   `shared_scale = 0.1261` for this clip). The model geometry is never
   touched. This act draws the keypoint cloud ALREADY at `shared_scale` --
   fixed-size for the whole act -- and animates only its POSITION. The
   rescale step itself (1.0 -> `shared_scale`) is not depicted here; it is
   assumed done (see the TASK-15 PIVOT section below for why).
2. **`root_optimization` does NOT rotate.** `configs/anatomy/v1.yaml` sets
   `TRUNK_OPTIMIZATION_KEYPOINTS: {}`, so `root_optimization`'s keypoint-fit
   loss is uniformly zero and only the manual root-translation it also
   performs has any effect. Measured on `06_stages.npz`: `qpos_root[3:7]`
   (the free-joint quaternion) is bit-identical to `qpos_default[3:7]`
   (`[1,0,0,0]` both), and `qpos_root[7:]` (all joint DOF) is bit-identical
   to `qpos_default[7:]` too. Only `qpos_root[:3]` (the free-joint
   translation) differs -- so root_optimization is TRANSLATION ONLY, no
   rotation, no scale change. Orienting is Act 4's `pose_optimization`
   (34.73 deg measured there), not this one. The runtime guard below still
   fails loudly if a future re-run of `stage_ik.py` ever contradicts this.

TASK-16 PIVOT (read this before touching the STAGING math below -- this is
the CURRENT design; the TASK-15 PIVOT section further down is kept only as
a record of the design this one replaces): after task-15 v2 shipped and was
assembled into the full video, the user watched it and reported it still
"looks off": *"can we have so it is just the second half of that were it
just swings in and aligns? from around 18s to 20s in"* -- i.e. keep only the
skeleton's swing-onto-the-model-and-settle beat, and drop the outward
excursion entirely. Measuring the task-15 v2 cut directly (near-white
mesh-mask pixel count is a bad probe for this -- it stayed steady; the
SKELETON's own saturated-pixel bounding box is what shows it) confirmed a
real, reproducible excursion:

    f000: skeleton_px=46860  span=536x433
    f060: skeleton_px=93688  span=709x541   <- swings OUT and gets BIGGER
    f119: skeleton_px=46517  span=533x460
    f179: skeleton_px=46517  span=533x460   (held)

Root cause (kept as a record, not a live design): task-15 v2's staged
centroid blended the ALREADY fixed-size cloud from the mesh's own LIVE,
MOVING centroid `mesh_ctr(f)` at `p=0` to the single fixed point `C_true` at
`p=1`, i.e. `staged_centroid(f) = (1-p)*mesh_ctr(f) + p*C_true`. Camera
`lookat` also tracked that same moving `mesh_ctr(f)` every frame (needed at
the time because the mesh's own real translation was ~5.8x its body length
across the WHOLE act). Substituting `mesh_ctr(f)` ~= `(1-p)*M0 + p*C_true`
(M0 = the mesh's rest position, since `root_optimization`'s translation is
itself linear in `p`) shows the camera-to-skeleton distance carries an
added term proportional to `p*(1-p)*(M0 - C_true)` -- a hump that is exactly
ZERO at `p=0` and `p=1` but PEAKS at `p=0.5` (old `f=60`, precisely the
frame the measurement above flags). That hump moves the skeleton measurably
CLOSER to the camera mid-act, which is what reads on screen as the skeleton
swelling outward and back -- a real geometric consequence of chasing a
MOVING mesh target with a camera that also chases it, not a bug in any
single number.

The fix (this cut): stop moving the things that create the hump.
`root_optimization`'s own translation is no longer depicted at all -- the
mesh is drawn STATIC at `qpos_root` (its final, true, measured position)
for the ENTIRE act, and the camera's `lookat` is likewise a SINGLE fixed
point (`mesh_ctr_final`, computed once). This is the same kind of staging
choice task-15 already made for the rescale step ("assumed done"): the
translation happened (it is real, measured, unretouched in `06_stages.npz`)
but this act no longer re-enacts it, choosing instead to show only the
skeleton's own approach onto an already-placed model -- which is exactly
what the user asked for ("it just swings in and aligns"). With BOTH the
mesh and the camera fixed, the ONLY moving thing left is the skeleton's own
centroid, blended along a straight line between two FIXED points:

    mesh_ctr_default    = FK(qpos_default)                    (fixed, measured)
    mesh_ctr_final       = FK(qpos_root)                       (fixed, measured)
    C_true               = shared_scale * raw_mean             (fixed, measured)
    START_CENTROID       = (1 - P_REF) * mesh_ctr_default + P_REF * C_true
    staged_centroid(f)   = (1 - p) * START_CENTROID + p * C_true
    cloud_pts(f)          = staged_centroid(f) + scaled_centered

`P_REF` (`0.65`) picks a point along the ORIGINAL (task-15 v2) rest->C_true
convergence line as the new, "modestly offset" starting position -- not the
full rest-position gap (which measured ~1.71 model-units and clips off
screen at this act's zoomed-in `ACT3_CAM_DISTANCE`; verified directly:
`P_REF=0.5`'s start point clips the skeleton's own bounding box against the
frame edge), and not a point so close that no "approach" is visible either.
`P_REF=0.65` was checked by rendering a probe frame and confirming the
skeleton's own saturated-pixel bounding box sits fully inside the canvas
with margin (no edge touch) while still being clearly separated from the
mesh. Because BOTH the mesh and the camera are now genuinely static, this
is a straight-line blend between two fixed points under a fixed camera --
there is no moving target left to create the old hump, so the skeleton's
on-screen distance from the mesh shrinks monotonically with no mid-act
excursion (verified numerically in `render_act3` below and reported by the
`--clip` caller).

Because the mesh is drawn at its OWN true final position for the whole act,
the residual caption can no longer start at `residual_scaled` (which is the
residual for the FAR REST position, `qpos_default`, against the scaled
keypoints) without overstating how far apart things look on screen at
`f=0` -- the skeleton is only `1 - P_REF` of the way out, not the whole
distance. The caption instead interpolates from a `START_RESID` scaled by
that same fraction: `START_RESID = residual_root + (1 - P_REF) *
(residual_scaled - residual_root)`, ending exactly at `residual_root`
(0.118 mm, unchanged, still the solver's real measured number) by `f =
ALIGN_END` and through the hold -- nothing here is invented, only the
SAME two measured residual numbers, blended along the SAME progress `p`
this act already used for the residual caption in every prior cut.

TASK-15 PIVOT (kept as a record of the design task-16 replaces; the
staging math it describes -- centroid blending toward a LIVE `mesh_ctr(f)`
-- is NOT what `render_act3` does any more): task-15 originally asked for
the EXISTING merged "rescale + root_optimization" single-step design (scale
1.0 -> `shared_scale` WHILE qpos SLERPs `qpos_default` -> `qpos_root`) to
additionally stay centred on the model throughout, rather than flying in
from off-screen. That version was implemented, rendered, and read
frame-by-frame -- and while f=0/f=299 looked right in isolation, reading the
IN-BETWEEN frames showed the oversized cloud visibly SHRINKING AWAY FROM
the model, exiting the frame entirely, then flying BACK IN near the end.
The user, live, asked to sidestep this rather than patch it further: "the
scaling doesn't look good... just have it do the root alignment and assume
it is already scaled... and can we have it shorter." That pivot fixed the
SCALE-related excursion; task-16 (above) is a second, independent pivot
that fixes a DIFFERENT, translation-related excursion that survived it.

WORLD-COORDINATE STORY (measured, not staged): the raw triangulated keypoints
(`04_kp3d_filt.npz`) sit in an arena-relative mm frame far from the model's
own origin, while the model's rest qpos places its root at the world
origin. `C_true` (`shared_scale * raw_mean` ~= [1.66, 0.25, 0.15]) is close
to where `root_optimization` independently translates the mesh
(`qpos_root[:3]` = [1.65, 0.25, 0.26]) -- because `root_optimization` sets
the root translation to the (already-scaled) root keypoint itself. That
near-coincidence (not exact -- the two points differ by ~0.05 model-units,
which is why even the fully-converged hold frames show a small, real,
residual on-screen gap between the skeleton's centroid and the mesh's own
centroid, matching `residual_root` = 0.118 mm) is why this act's staged
blend converges cleanly onto the model.

CAMERA (task-14 round 3, tightened further by task-16 -- ACT 3'S CAMERA IS
COMPLETELY FIXED, no zoom, no dolly, no orbit, and -- as of task-16 -- no
lookat tracking either): earlier cuts of this act routed the story through
a MOVING camera in one form or another (task-11's fully live-distance
camera; task-14 round 2's frozen-distance-but-dollying-in camera; every cut
through task-15 v2's live `mesh_ctr(f)`-tracking `lookat`). Each attempt hit
some version of the same lesson: a moving camera in this act keeps fighting
the thing the act is supposed to show, whether by directly cancelling the
cloud's shrink (task-11) or by creating the p*(1-p) hump described in the
TASK-16 PIVOT section above. This cut retires camera motion entirely,
INCLUDING `lookat`: `ACT3_CAM_DISTANCE`/`ACT3_CAM_AZIMUTH`/`ACT3_CAM_ELEV`
remain TRUE CONSTANTS (unchanged from task-14 round 3, and still equal to
Act 4's own `AZ_START`/`ELEV` so the camera ANGLE matches exactly across the
Act3->Act4 cut -- continuity across the cut matters more than any single
act's own framing preference), and `cam.lookat` is now ALSO a constant,
`mesh_ctr_final` (the mesh's own real, final centroid, computed ONCE via
forward kinematics against the static `qpos_root` this act now renders
throughout) -- not zoom/dolly/orbit (none of `distance`/`azimuth`/
`elevation` change, and neither does `lookat` any more), just a frame that
does not move at all, because the mesh it is framing does not move at all
either. `ACT3_CAM_DISTANCE=0.93` (unchanged from task-14 round 3) still
lands the near-white/low-saturation mesh-pixel count (`mx>60 & (mx-mn)<40`,
excluding the top-270px caption band) inside the requested 50,000-75,000 px
band -- re-measured directly for this cut at f=0/59/89 by `render_act3`
below (values reported by the CLI at render time, not restated here as a
number that would go stale the next time `P_REF` or the clip changes).

Keypoint SKELETON, not a loose cloud (task-14 round 3): keypoints are
connected by thin capsule "bones" (`stage_ik._add_bone`, MuJoCo's
`mjv_connector`) using `data/fly50.json`'s `edges` (44 index pairs, mapped
onto THIS array's own keypoint order by NAME --
`kp_colors.jarvis_skeleton_edges`), coloured per JARVIS's own bone-colour
convention (`colors[line[1]]`, the STOP node's colour -- verified against
all six of JARVIS's own skeleton-drawing call sites). Marker/bone radius are
FIXED (`BASE_MARKER_R_AT_FINAL_SCALE`, `BONE_RADIUS_FRACTION`) -- the cloud
is drawn at `shared_scale` for the whole act, never resized, so there is no
"factor" left to scale these with. The same skeleton (bones + colours) is
also drawn in Act 4 (task-14 round 4) -- see `act4_solve.py`'s module
docstring.

Keypoint colours (Change 2, task-14): each limb chain gets its own colour
from the JARVIS scheme (`kp_colors.jarvis_kp_colors_rgb01`, ported by calling
`third_party/JARVIS-HybridNet`'s own `get_skeleton`), replacing the previous
4-group (head/thorax/abdomen/legs) scheme.

Wing visibility (Change 1, task-14): `set_mesh_rgba`/`capture_geom_alpha`
(stage_ik.py) preserve the model's originally-invisible geoms (the wings'
`*_inertial` boxes, alpha=0 by design) across every `geom_rgba` mutation this
act performs -- see their docstrings for why a blanket alpha assignment used
to turn them into opaque white rectangles.

TASK-18 MIXTURE (deliberately disclosed on screen, not hidden): the user
supplied a full production STAC solve (`ik_production/stac_ik_full.h5`) that,
unlike this repo's own `06_stages.npz`, actually ran `offset_optimization`
(fitted marker offsets, mean |0.015| max |0.26| mm -- our own staged run
never fit these). Act 4 now plays that production fit back; this act's
keypoint SKELETON is switched to match -- it now swings onto the production
solve's own `kp_data` (converted back to raw mm by dividing out
`shared_scale`, so the existing centroid-blend math below runs unchanged)
instead of this repo's own re-triangulated `04_kp3d_filt.npz`. This act's
MESH, however, still renders `qpos_root` from `06_stages.npz` -- the staged,
no-offset root-alignment solve -- because that is a genuine intermediate
stage of a real solve and this act's whole point (see the module's opening
paragraph). The result is a deliberate, disclosed mixture of two solves
(captioned on screen, not just here): staged mesh position, production
keypoint target. Measured, not assumed: the production `kp_data` and this
repo's own `04_kp3d_filt.npz * shared_scale` agree to a mean 0.0016 mm (max
0.0026 mm) at this clip's keypoint scale -- i.e. the production run's target
IS (to floating-point noise) this same repository's own filtered
triangulation, just carried through a solve that also fit offsets -- so this
swap changes PROVENANCE (what array is read) without changing what is drawn
in any visually meaningful way. It does guarantee exact equality with Act 4's
own opening frame (both now read the identical `kp_data[frame_for_stills]`),
which the previous `04_kp3d_filt.npz`-sourced cloud only matched to that same
~0.002 mm noise floor. **STALE as of TASK-24, RESOLVED by TASK-25**: TASK-24
moved Act 4's opening frame off `kp_data[frame_for_stills]` to
`kp_data[0]`, and this act's own TASK-24 section (below) explains why it
deliberately did NOT follow at the time; TASK-25 (further below) closes
that gap by re-anchoring this act's own target to `kp_data[0]` too, once a
mesh solved AT frame 0 (`qpos_root_f0`) existed to pair it with.

TASK-24 (`act4_solve.py` fixed a wrap/replayed-tail bug the user reported in
Act 4's playback -- read `act4_solve.py`'s own TASK-24 section first for the
full bug report; this section covers ONLY its consequence for Act 3).
**SUPERSEDED BY TASK-25 immediately below** -- kept as the historical record
of why a naive re-anchor was rejected, NOT a description of current
behaviour: the fix re-anchored Act 4's Phase A target and Phase B playback
start from `frame_for_stills` (450) to source frame 0
(`act4_solve.PLAYBACK_ANCHOR`). Re-pointing only this act's SKELETON target
to `kp_data_prod[0]` while leaving the OLD `qpos_root` (`06_stages.npz`'s
own snapshot, a fit SPECIFICALLY to frame 450's keypoints via
`root_optimization(..., frame=frame_for_stills)`) as-is was tried and
rendered before deciding against it, not assumed bad: `mesh_ctr_final` (FK
of that old `qpos_root`) sat at `[1.634, 0.250, 0.198]`; `C_true` computed
from `kp_data_prod[0]` instead of `kp_data_prod[450]` sat at
`[0.598, 0.235, 0.146]` -- **1.038 model units** away (roughly 3.6x the
model's own ~0.289-unit body length) -- rendered directly (`f=89`, the
hold): the skeleton sat in the frame's far corner, clipped, never touching
the mesh. TASK-24's decision was correct GIVEN its own stated constraint
("out of scope for a playback bugfix" to re-run the actual STAC solve) --
TASK-25 below lifts exactly that constraint instead of accepting the
trade-off.

TASK-25 (closes the regression TASK-24 left open, by removing the blocker
TASK-24 itself named): a hand-translated `qpos_root` was never on the
table (that would not be a real solver output, just an invented
approximation of one -- exactly what the task-25 brief forbids). Instead,
`stage_ik.py` gained `run_root_f0()`, which calls the exact SAME
`compute_stac.root_optimization` (same `Stac`/rest-snapshot construction)
`run()` already calls for its own `qpos_root`, just at frame 0 of the
PRODUCTION `kp_data` instead of frame `frame_for_stills` (450) of this
repo's own rescaled keypoints. The result, `qpos_root_f0`, is persisted to
a NEW file, `predictions/06_stages_f0.npz`, living BESIDE (never
overwriting) `06_stages.npz` -- that file's own `qpos_root`/
`frame_for_stills`/etc. are untouched, still the correct provenance for the
frame-450 stage figure `stage_ik.qc_stages` produces.

This act now draws `qpos_root_f0` (not `06_stages.npz`'s `qpos_root`) as its
static mesh and reads `kp_data_prod[0]` (not `kp_data_prod[frame_for_stills]`)
as its skeleton target -- both consistently anchored at source frame 0, the
SAME frame `act4_solve.PLAYBACK_ANCHOR` plays back from. Because
`qpos_root_f0` was solved specifically against frame 0's own keypoints,
`mesh_ctr_final` (now FK of `qpos_root_f0`) is once again near-coincident
with `C_true` (now computed from `kp_data_prod[0]`) -- measured directly:
residual_root_f0 = 0.099 mm (down from a pre-solve 0.695 mm), the same
small-residual signature the WORLD-COORDINATE STORY section above describes
for the frame-450 solve, just re-measured at a different frame -- so this
act's own convergence premise (skeleton and mesh co-located by f=59/the
hold) holds exactly as it always did; see the render-time acceptance print
for the exact numbers. The measured-fact guard below (translation-only
`root_optimization`) is checked against `qpos_root_f0`, not the old
`qpos_root`. Consequence, and the whole point of this task: since both this
act's mesh AND Act 4's Phase A opening pose (`act4_solve.py`'s own TASK-25
section) are now the identical `qpos_root_f0`, and both acts' skeleton
target is now the identical `kp_data_prod[0]`, the Act3(f=89)->Act4(f=0)
handoff is a real, measured, near-zero jump on BOTH the mesh and the
keypoint cloud -- not merely the mesh, as TASK-17/24 could only guarantee.

Timeline (90 frames, 3 s @ 30 fps -- task-16: cut down from the 180-frame/
6s task-15 v2 cut per the user's direct "just the second half... swings in
and aligns" request; the mesh's own translation and the pre-task-15
rest-pose fade-in phase both stay REMOVED, not merely shortened -- the mesh
is drawn at its final `qpos_root` position for the WHOLE act, never at
`qpos_default`):
  f   0-59  the ALREADY fixed-size (`shared_scale`) keypoint skeleton's
            STAGED centroid blends `START_CENTROID` (a fixed, modestly-
            offset point -- see TASK-16 PIVOT) -> `C_true` (the solver's
            true scaled keypoint centroid), smoothstep-eased. Residual
            ticks down `START_RESID` -> `residual_root` (0.118 mm) over the
            same window. Mesh and camera are both static throughout; only
            the skeleton moves.
  f  60-89  hold: skeleton and mesh co-located at the solver's true final
            position, residual 0.118 mm, exactly as `render_act3` renders
            through the hold in every earlier cut.

EXPECTATION: at f=0 the (fixed-size, already-scaled) skeleton sits fully
on screen, modestly offset from the static mesh -- clearly two separate
things, not yet aligned, but with the approach clearly readable; it moves
monotonically toward the mesh with NO backward/outward step and NO
mid-act growth in its own on-screen size (contrast the task-15 v2 measured
excursion in the TASK-16 PIVOT section above); by f=59 (and through the
hold) skeleton and mesh are co-located at the solver's true position,
residual has fallen to 0.118 mm, mesh reads at the same on-screen size it
always has (it does not move OR resize OR rotate at any point), and the
camera does not move at all (matching Act 4's angle exactly, no visible
snap at the cut).
FALSIFICATION: a skeleton that grows then shrinks in on-screen size or
distance between f=0 and f=59 means the old hump is back (check that BOTH
the mesh qpos and `cam.lookat` are genuinely constant this frame, not
re-introducing a moving target); a skeleton that has not reached the mesh's
exact position by f=59/the hold means the blend does not reach `p=1`; a
visibly moving OR resizing OR rotating mesh means the act is animating
something task-16 deliberately stopped depicting.

Colours come from `kp_colors.jarvis_kp_colors_rgb01` (JARVIS per-limb-chain
scheme) -- never invented here.

TASK-19 (presentation-only; read before touching the render loop's drawing
calls below): on-screen title changes "ACT 3" / "Root alignment" (the old
title/subtitle pair) -> a single title, "Root alignment". ALL other caption
text is removed from the frame: the "stage: root_optimization (held)" label,
the residual-mm line, the "keypoints pre-scaled to shared_scale=..." line,
the "mesh is static..." line, the "keypoints start modestly offset..." line,
and the TASK-18 "mixture disclosed" line are all gone from the rendered PNG
per the user's explicit request. NONE of the facts they described changed --
every number and claim above (translation-only root_optimization, residual
ticking start_resid -> 0.118 mm, shared_scale=0.1261, the mesh never
resizing/rotating, the task-18 mesh/keypoint provenance mixture) is still
computed exactly as before and printed to the console (`_stage_at`'s
`resid`/`stage_label` are computed and logged, just not drawn) -- only the
on-screen captions are gone. The small bottom-left frame counter is the only
other on-screen text kept. Typography (the title) now comes from `draw.py`'s
shared `TITLE_SCALE`/`CAPTION_SCALE`/`SMALL_SCALE`.

Because the on-screen text block shrank to just one title line, `_mesh_mask`'s
`_CAPTION_BAND_PX` (the near-white-text exclusion band the mesh-pixel
acceptance test uses) was re-measured directly on a rendered frame and
lowered accordingly -- see `_CAPTION_BAND_PX`'s own comment below for the
current value and how it was checked.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import sys
import time
from pathlib import Path

import cv2
import h5py
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
N_OUT = 90
# Skeleton align-onto-model window (task-16: cut to the swing-in-and-align
# beat only). The mesh itself is static at qpos_root for the WHOLE act (see
# module docstring's TASK-16 PIVOT section) -- only the skeleton's staged
# centroid animates over this window.
ALIGN_START, ALIGN_END = 0, 59
HOLD_START = 60

XML_PATH = _REPO / "models" / "fruitfly_v1" / "fruitfly_v1_free.xml"

# --- shared "wide diagnostic camera" building blocks -------------------------
# `AZ_START`/`ELEV`, `_cloud_mesh_spread`, and `_wide_camera` below are kept
# for ACT 4 ONLY (imported from there: `_wide_camera_for_aspect` builds on
# `_wide_camera` with a LIVE per-frame distance, appropriate for Act 4's real
# 921-frame playback). Act 3 no longer uses ANY of these -- its own camera is
# the fully independent, fully fixed `ACT3_CAM_DISTANCE`/`ACT3_CAM_AZIMUTH`/
# `ACT3_CAM_ELEV` below. Do not repoint Act 3 back at
# `AZ_START`/`ELEV`/`_wide_camera`: changing THIS block changes Act 4's
# camera too (see the CAMERA section of the module docstring).
AZ_START = 120.0
ELEV = -20.0


def _cloud_mesh_spread(mesh_ctr, cloud_pts, lookat, model_extent):
    """Radius (world units) that a camera centred at `lookat` must span to
    keep both `mesh_ctr` and every finite point of `cloud_pts` in frame --
    the same formula stage_ik.py's qc_stages() uses for its wide-camera row.
    Pure geometry, no camera object built here. Kept for Act 4's import only
    (see the block comment above); Act 3's own camera stopped using this at
    task-14 round 3."""
    finite = cloud_pts[np.all(np.isfinite(cloud_pts), axis=-1)]
    return max(
        float(np.max(np.linalg.norm(finite - lookat, axis=-1))),
        float(np.linalg.norm(mesh_ctr - lookat)),
        model_extent * 0.5,
    )


def _wide_camera(mesh_ctr, cloud_pts, model_extent, azimuth, cam_spread, zoom=1.0):
    """The wide diagnostic camera: same lookat construction stage_ik.py's
    qc_stages() uses for its wide-camera row (never the model's `hero`
    camera, which frames the mesh only). Kept for Act 4's import only (see
    the block comment above); Act 3's own camera stopped using this at
    task-14 round 3.

    `cam_spread` -- hence the BASE `cam.distance` -- is NOT derived from the
    live cloud here. It is a FIXED value the caller computes ONCE for the
    whole act, so the camera cannot zoom in lock-step with the shrinking
    cloud (task-11 review, "Important 1").

    `zoom` (default 1.0, i.e. no change) is a SEPARATE, optional dolly-in
    multiplier -- `cam.distance = cam_spread * 2.6 / zoom` -- kept for Act 4
    (`_wide_camera_for_aspect`).

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


# --- Act 3's OWN fixed camera (task-14 round 3; lookat also frozen task-16) -
# TRUE CONSTANTS -- never read into a per-frame formula, unlike `AZ_START`/
# `ELEV` above (which Act 4 imports for its own, deliberately LIVE camera).
# `ACT3_CAM_DISTANCE` was solved (in an earlier cut) against a REAL
# rendered-pixel measurement (see `_mesh_mask`/`_mesh_px` below and the module
# docstring). `ACT3_CAM_AZIMUTH`/`ACT3_CAM_ELEV` are set EQUAL to the shared
# `AZ_START`/`ELEV` (both 120/-20) so Act 3's camera ANGLE matches Act 4's
# exactly.
ACT3_CAM_DISTANCE = 0.93
ACT3_CAM_AZIMUTH = AZ_START
ACT3_CAM_ELEV = ELEV


def act3_frozen_camera(lookat):
    """The EXACT fixed camera Act 3 renders every frame with: constant
    `ACT3_CAM_DISTANCE`/`ACT3_CAM_AZIMUTH`/`ACT3_CAM_ELEV`, orbiting the
    caller-supplied `lookat`. Factored out here (task-17, "make the Act3/
    Act4 cut seamless") so Act 4's Phase A can build the IDENTICAL camera
    Act 3's own last frame renders with, by calling this with its own
    `mesh_ctr_final` (computed the same way Act 3 does: FK against
    `qpos_root`, averaged over the same `body_site_idxs`) -- a shared
    function, not a re-derived approximation of the same numbers. `lookat`
    is the only thing that varies between callers; the three camera
    constants never do."""
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = ACT3_CAM_DISTANCE
    cam.azimuth, cam.elevation = ACT3_CAM_AZIMUTH, ACT3_CAM_ELEV
    return cam

# Skeleton staging (task-16): fraction of the ORIGINAL (task-15 v2) rest ->
# C_true convergence line used as the new, fixed START_CENTROID -- see
# module docstring's TASK-16 PIVOT section for why 0.65 (checked directly: a
# smaller fraction, e.g. 0.5, clips the skeleton's own bounding box against
# the frame edge at this act's zoomed-in ACT3_CAM_DISTANCE).
P_REF = 0.65

# Marker/bone radius: FIXED for the whole act -- the keypoint cloud is drawn
# at `shared_scale` throughout, never resized, so there is no "factor" left
# to scale these with.
BASE_MARKER_R_AT_FINAL_SCALE = 0.012   # model units; ~ a leg-segment's width at Act 4's framing
BONE_RADIUS_FRACTION = 0.4             # bone capsule radius, as a fraction of marker radius

# Real-pixel mesh-size acceptance test: near-white, low-saturation pixels,
# excluding the top caption band -- keypoints are drawn as SATURATED colours
# so they never count as "mesh" here (and vice versa for the
# skeleton-centroid probe below). TASK-19: lowered from 300 to 130 -- the
# on-screen text block shrank to a single title line ("Root alignment", no
# subtitle, no stage/residual/staging captions -- see module docstring's
# TASK-19 section), so the near-white title text no longer reaches anywhere
# near row 300. Measured directly on a re-rendered frame before picking 130
# (not guessed): with `TITLE_SCALE`/`TITLE_THICKNESS`, the title's tallest
# ascender/descender pixels stay within row ~110, so 130 leaves a safety
# margin without re-admitting title text into the mesh mask.
_CAPTION_BAND_PX = 130


def _mesh_mask(canvas_bgr):
    b = canvas_bgr[..., 0].astype(np.int32)
    g = canvas_bgr[..., 1].astype(np.int32)
    r = canvas_bgr[..., 2].astype(np.int32)
    mx = np.maximum(np.maximum(b, g), r)
    mn = np.minimum(np.minimum(b, g), r)
    mesh = (mx > 60) & ((mx - mn) < 40)
    mesh[:_CAPTION_BAND_PX, :] = False
    return mesh


def _mesh_px(canvas_bgr) -> int:
    return int(_mesh_mask(canvas_bgr).sum())


def _smoothstep(p):
    p = np.clip(p, 0.0, 1.0)
    return 3 * p ** 2 - 2 * p ** 3


def _slerp_qpos(qa, qb, t):
    """Interpolate a MuJoCo free-joint qpos: `qpos[:3]` (translation) lerp,
    `qpos[3:7]` (w,x,y,z quaternion) SLERP, `qpos[7:]` (hinge/joint DOF) lerp.

    Not called by `render_act3` any more (task-16: the mesh is drawn static
    at `qpos_root` for the whole act, so there is no qpos to interpolate) --
    kept here because `act4_solve.py` imports this exact helper for its own,
    real 34.73 deg rotation, and a plain lerp of the quaternion there would
    denormalise and visibly tumble.
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


def _progress(f):
    """Progress `p` in [0, 1] for the skeleton's staged-centroid blend,
    smoothstep-eased over `ALIGN_START`-`ALIGN_END`, clamped at 1.0 for the
    hold (`f >= ALIGN_END`). The SAME `p` drives the residual interpolation
    in `_stage_at` -- one schedule, not several independently-timed ones."""
    if f >= ALIGN_END:
        return 1.0
    return _smoothstep((f - ALIGN_START) / float(ALIGN_END - ALIGN_START))


def _stage_at(f, start_resid, residual_root):
    """Return (stage_label, residual_mm, progress) for output frame `f`.
    Pure function of `f` and the two measured residual endpoints -- no
    numbers invented here, only interpolated between them (task-16:
    `start_resid` replaces the old `residual_scaled` as the f=0 endpoint,
    scaled by how much of the original gap `START_CENTROID` actually spans --
    see module docstring's TASK-16 PIVOT section)."""
    p = _progress(f)
    stage_label = "root_optimization" if f < HOLD_START else "root_optimization (held)"
    resid = (1 - p) * start_resid + p * residual_root
    return stage_label, resid, p


def _camera_project(scn, fovy_deg, width, height, pt):
    """Project a world-space point `pt` into this frame's pixel coordinates,
    using the EXACT camera pose `renderer.update_scene` just resolved
    (`scn.camera[0]`'s `pos`/`forward`/`up`) rather than re-deriving a camera
    matrix from `azimuth`/`elevation`/`distance` by hand -- this guarantees
    the projection matches what was actually rendered. Used only for the
    acceptance-test centroid-distance probe below; never drives anything
    drawn on screen."""
    pos = np.asarray(scn.camera[0].pos, np.float64)
    forward = np.asarray(scn.camera[0].forward, np.float64)
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(scn.camera[0].up, np.float64)
    up = up / np.linalg.norm(up)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    rel = np.asarray(pt, np.float64) - pos
    depth = rel @ forward
    tan_half = np.tan(np.radians(fovy_deg / 2.0))
    x_ndc = (rel @ right) / (depth * tan_half * (width / height))
    y_ndc = (rel @ up) / (depth * tan_half)
    return np.array([(x_ndc * 0.5 + 0.5) * width,
                      (1.0 - (y_ndc * 0.5 + 0.5)) * height])


def render_act3(clip: str = clip_io.CLIP_DEFAULT) -> Path:
    dirs = clip_io.out_dirs(clip)

    with np.load(dirs["predictions"] / "06_stages.npz", allow_pickle=True) as z:
        qpos_default = np.asarray(z["qpos_default"], np.float64)
        stage_names = [str(s) for s in z["stage_names"]]
        kp_names = [str(n) for n in z["kp_names"]]
        frame_for_stills = int(z["frame_for_stills"])
        shared_scale = float(z["shared_scale"])

    assert stage_names[:3] == ["default", "scaled", "root"], (
        f"unexpected 06_stages.npz stage_names order: {stage_names}")

    # TASK-25: the mesh's static pose and the residual endpoints below now
    # come from `06_stages_f0.npz` (`stage_ik.run_root_f0`), a root_optimization
    # solved AT source frame 0 against the PRODUCTION kp_data -- NOT
    # `06_stages.npz`'s own `qpos_root`/`residual_mm`, which are frame-450
    # snapshots. See module docstring's TASK-25 section for why (closes the
    # Act3->Act4 cut regression TASK-24 left open). `06_stages.npz` itself is
    # untouched -- still loaded above only for `qpos_default`/`kp_names`/
    # `frame_for_stills`/`shared_scale`, all of which are frame-independent.
    with np.load(dirs["predictions"] / "06_stages_f0.npz", allow_pickle=True) as zf0:
        qpos_root_f0 = np.asarray(zf0["qpos_root_f0"], np.float64)
        f0_frame = int(zf0["frame"])
        residual_scaled_f0 = float(zf0["residual_scaled_f0"])
        residual_root_f0 = float(zf0["residual_root_f0"])
        kp_names_f0 = [str(n) for n in zf0["kp_names"]]
    if f0_frame != 0:
        raise ValueError(
            f"06_stages_f0.npz frame={f0_frame} -- expected 0 (this act's "
            "skeleton target and Act 4's PLAYBACK_ANCHOR are both hardcoded "
            "to source frame 0; a mismatched snapshot would reintroduce the "
            "Act3->Act4 cut regression TASK-25 fixes).")
    if kp_names_f0 != kp_names:
        raise ValueError(
            "06_stages_f0.npz kp_names != 06_stages.npz kp_names -- refusing "
            "to mix keypoint orders (CLAUDE.md's keypoint-order bug class).")

    # Measured-fact guard: root_optimization must not have rotated or moved
    # any joint DOF (module docstring point 2), now checked against
    # `qpos_root_f0` (the pose this act actually draws) rather than the old
    # frame-450 `qpos_root`. If a future re-run of stage_ik.py ever produces
    # a rotating root_optimization, this act's entire "translation only"
    # premise would be wrong -- fail loudly rather than silently render the
    # old (incorrect) story.
    if not np.allclose(qpos_default[3:], qpos_root_f0[3:], atol=1e-9):
        raise ValueError(
            "qpos_default[3:] != qpos_root_f0[3:] -- root_optimization "
            "rotated and/or moved joint DOF in this solve. Act 3's premise "
            "(translation only, no rotation, no scale change) no longer "
            "matches the recorded snapshots; refusing to render a "
            "mesh-rotation animation that contradicts the measured facts.")

    # TASK-18: the keypoint skeleton this act swings in now comes from the
    # PRODUCTION solve's own `kp_data` (`ik_production/stac_ik_full.h5`), not
    # our own re-triangulated `04_kp3d_filt.npz` -- see module docstring's
    # TASK-18 MIXTURE section. `kp_data` is stored in MODEL order, already
    # multiplied by `shared_scale`; dividing back out here lets the rest of
    # this function's existing raw-mm staging math (`raw_mean`/`scaled_centered`
    # below) run completely unchanged.
    h5_path = Path(clip) / "ik_production" / "stac_ik_full.h5"
    with h5py.File(h5_path, "r") as hf:
        kp_names_prod = [n.decode() if isinstance(n, bytes) else str(n)
                          for n in hf["kp_names"][:]]
        kp_data_prod = np.asarray(hf["kp_data"][:], np.float64).reshape(-1, 50, 3)
    expected_model_names = clip_io.model_kp_names()
    if kp_names_prod != expected_model_names:
        raise ValueError(
            f"{h5_path} kp_names != clip_io.model_kp_names() -- refusing to "
            "mix keypoint orders (CLAUDE.md's keypoint-order bug class). "
            f"First mismatch: {next((a, b) for a, b in zip(kp_names_prod, expected_model_names) if a != b)}")
    if kp_names_prod != kp_names:
        raise ValueError(
            f"{h5_path} kp_names != 06_stages.npz kp_names -- refusing to mix "
            "keypoint orders (CLAUDE.md's keypoint-order bug class).")

    # TASK-25: this act's skeleton target is now source frame 0 (matching
    # `act4_solve.PLAYBACK_ANCHOR` and `06_stages_f0.npz`'s own `frame`), NOT
    # `frame_for_stills` (450) -- see module docstring's TASK-25 section.
    # `frame_for_stills` is still loaded/printed above only as provenance for
    # `06_stages.npz`'s OWN frame-450 solve (unrelated to what this act draws
    # now).
    SKELETON_TARGET_FRAME = 0  # must match act4_solve.PLAYBACK_ANCHOR
    kp3d_raw_frame = kp_data_prod[SKELETON_TARGET_FRAME] / shared_scale  # (50,3) mm, MODEL order, RAW (unscaled)
    print(f"[act3] TASK-25: skeleton target frame={SKELETON_TARGET_FRAME}, "
          f"shared_scale={shared_scale:.4f} (06_stages.npz's own "
          f"frame_for_stills={frame_for_stills} is UNRELATED to what this "
          f"act draws, kept only as 06_stages.npz's own provenance)")
    print(f"[act3] TASK-18/25: keypoint skeleton sourced from the PRODUCTION "
          f"fit's kp_data ({h5_path}), not 04_kp3d_filt.npz -- mesh below is "
          f"qpos_root_f0 (06_stages_f0.npz), a root_optimization solved AT "
          f"frame 0 against this SAME kp_data, not 06_stages.npz's "
          f"frame-450 qpos_root.")
    print(f"[act3] residuals scaled_f0/root_f0 = {residual_scaled_f0:.3f}/{residual_root_f0:.3f} mm")
    print(f"[act3] qpos_root_f0 - qpos_default (translation only, mm, NOT "
          f"depicted this act -- task-16): {(qpos_root_f0[:3] - qpos_default[:3])}")

    mj_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    orig_alpha = capture_geom_alpha(mj_model)   # BEFORE any geom_rgba mutation
    grey = PALETTE["mesh"][0] / 255.0   # PALETTE["mesh"] is (200,200,200): BGR==RGB here
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

    # --- Cloud, ALREADY scaled to shared_scale -- fixed shape for the whole
    # act, no scale animation. `raw_mean`/`scaled_centered` decompose it into
    # a rigid centroid + a zero-mean shape (already scaled); `C_true` is the
    # SINGLE fixed point the staged centroid converges to.
    finite_raw = np.all(np.isfinite(kp3d_raw_frame), axis=-1)
    raw_mean = kp3d_raw_frame[finite_raw].mean(axis=0)
    scaled_centered = shared_scale * (kp3d_raw_frame - raw_mean)
    C_true = shared_scale * raw_mean

    # --- Mesh, static for the WHOLE act (task-16): forward-kinematics both
    # `qpos_default` (only used to derive the skeleton's fixed START_CENTROID
    # reference) and `qpos_root_f0` (the mesh's real, final position -- TASK-25:
    # a root_optimization solved AT frame 0, replacing 06_stages.npz's
    # frame-450 `qpos_root` -- drawn for every frame) ONCE, outside the render
    # loop -- see module docstring's TASK-16 PIVOT and TASK-25 sections.
    d_default = mujoco.MjData(mj_model)
    d_default.qpos[:] = qpos_default
    mujoco.mj_forward(mj_model, d_default)
    mesh_ctr_default = np.asarray(d_default.site_xpos[body_site_idxs]).mean(axis=0)

    d = mujoco.MjData(mj_model)
    d.qpos[:] = qpos_root_f0
    mujoco.mj_forward(mj_model, d)
    mesh_ctr_final = np.asarray(d.site_xpos[body_site_idxs]).mean(axis=0)

    start_centroid = (1.0 - P_REF) * mesh_ctr_default + P_REF * C_true
    start_resid = residual_root_f0 + (1.0 - P_REF) * (residual_scaled_f0 - residual_root_f0)
    print(f"[act3] staging (task-16/25): P_REF={P_REF} start_centroid={start_centroid} "
          f"-> C_true={C_true} (mesh_ctr_final={mesh_ctr_final}); "
          f"start_resid={start_resid:.3f} -> residual_root_f0={residual_root_f0:.3f} mm; "
          f"mesh is STATIC at qpos_root_f0 for the whole act, camera lookat is "
          f"the fixed mesh_ctr_final -- only the skeleton's centroid moves.")

    out_dir = dirs["frames"] / "act3_align"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[act3] fixed camera: distance={ACT3_CAM_DISTANCE} azimuth="
          f"{ACT3_CAM_AZIMUTH} elevation={ACT3_CAM_ELEV} lookat={mesh_ctr_final} "
          f"(all constants, never animated -- task-16)")

    t0 = time.time()
    fovy_deg = float(mj_model.vis.global_.fovy)
    # Persistent renderer, created ONCE and reused for every frame (matches
    # Act 4's fix; see CLAUDE.md/module constraints -- per-frame `with
    # mujoco.Renderer(...)` construction is the prime suspect for Act 4's
    # mid-render EGL resource-leak crash at ~700/900 frames).
    with mujoco.Renderer(mj_model, height=CANVAS_H, width=CANVAS_W) as renderer:
        cam = act3_frozen_camera(mesh_ctr_final)

        for f in range(N_OUT):
            stage_label, resid, p = _stage_at(f, start_resid, residual_root_f0)

            # STAGING (task-16): the ALREADY fixed-size cloud's centroid
            # blends from the fixed `start_centroid` (p=0) to the cloud's
            # single true fixed position (`C_true`, p=1) -- a straight line
            # between two points that never move, under a camera that never
            # moves either. See module docstring's TASK-16 PIVOT section for
            # why this has no mid-act excursion, unlike the design it
            # replaces.
            staged_centroid = (1.0 - p) * start_centroid + p * C_true
            cloud_pts = staged_centroid + scaled_centered

            # `d`/`cam` are UNCHANGED every frame (mesh and camera are both
            # static this act); update_scene is still called per frame
            # because the skeleton geoms below are added fresh each time.
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

            # TASK-19: every caption below the title is removed from the
            # frame per the user's explicit request -- stage_label/resid are
            # still computed above (and logged to the console just below)
            # so the facts they represent are not lost, only their on-screen
            # display. See module docstring's TASK-19 section.
            canvas = draw.stage_title(canvas, "Root alignment")
            canvas = draw.label(canvas, f"frame {f + 1}/{N_OUT}", (48, CANVAS_H - 24),
                                 scale=draw.SMALL_SCALE, color=(150, 150, 150))
            if f in (0, 20, 40, 59, 89):
                print(f"[act3] f={f} stage={stage_label} residual={resid:.3f} mm "
                      f"(not shown on screen since task-19; see module docstring)")
                mpx = _mesh_px(canvas)
                sk_px2d = _camera_project(scn, fovy_deg, CANVAS_W, CANVAS_H, staged_centroid)
                mesh_px2d = _camera_project(scn, fovy_deg, CANVAS_W, CANVAS_H, mesh_ctr_final)
                centroid_dist_px = float(np.linalg.norm(sk_px2d - mesh_px2d))
                print(f"[act3] acceptance: f={f} mesh_px={mpx} (target ~50,000-75,000) "
                      f"centroid_dist_px={centroid_dist_px:.1f} (must decrease "
                      f"monotonically f=0->59, ~0 by f=59)")

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
