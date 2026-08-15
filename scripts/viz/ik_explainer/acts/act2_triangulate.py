#!/usr/bin/env python3
"""Act 2 -- seven camera views collapse into one 3D keypoint cloud.

RIG GEOMETRY (measured, not invented): factoring the clip's DLTs gives seven
optical axes forming a 180 deg arc at 30 deg spacing, all perpendicular to the
arena's long axis (every camera's recovered right-axis R0 is ~world +X to
within 1.2 deg -- see the assertion in `_load_rig`). The rig is TELECENTRIC
(every DLT has L9=L10=L11=0, so `clip_io.project`'s perspective divide is a
no-op): triangulation rays are PARALLEL along each camera's view direction,
never converging to a point. Panel *direction* (`view_dir`, from
`clip_io.camera_view_dirs`) is real rig geometry; panel *distance* `R_STAGE`
from the cloud is an arbitrary staging choice with no physical meaning (the
cameras have no finite centre of projection) -- captioned on screen.

GEOMETRIC CONSTRUCTION (exact, not approximate):
  For a single representative frame `T0`, every keypoint's true 3D position
  `kp3d[T0,k]` already lies, by construction of a telecentric camera, on the
  line `{ kp3d[T0,k] + t * view_dir[c] : t in R }` for EVERY camera `c` (that
  line is exactly camera c's triangulation ray for that keypoint). So:
    - a keypoint's position on camera c's panel (radius `R_STAGE` along
      `-view_dir[c]`) is the orthogonal rejection of `kp3d[T0,k]` onto that
      plane: no separate 2D-pixel inversion needed, and by construction every
      ray in camera c's bundle shares the exact direction `view_dir[c]` --
      parallel is not an approximation here, it is the parametrisation.
    - the ray's far end is `kp3d[T0,k]` itself, so "the cloud sits exactly at
      the intersection" is likewise true by construction, not by luck.
  Sanity check (`_check_geometry`, run at startup, values printed and
  asserted): reprojecting each camera's panel-plane points back through the
  REAL rig DLT (`clip_io.project`) reproduces that camera's actually detected
  2D keypoints to within a few px (mean ~2-4 px, worst case ~28 px on one
  outlier leg-tip/camera pair) -- i.e. this construction agrees with the real
  detector data, it does not just look plausible.

  Panel rectangles (for the textured video quad) are placed with a SEPARATE
  exact inversion (`_panel_plane_point`): given camera c's real affine DLT
  matrix and a desired pixel (u,v), solve the 3x3 linear system pairing the
  camera's two (affine, telecentric) projection rows with the panel-plane
  constraint `X . view_dir[c] = -R_STAGE`. This recovers the crop corners'
  true 3D positions on the panel without ever assuming a right/up axis
  convention; the panel's actual right/up/normal are then measured FROM those
  corners (`right = TR-TL`, `up = TL-BL`) and verified orthonormal
  (`right.up ~ 0`) with the expected physical size (~5.3x5.5 mm, matching
  `CROP_W/PX_PER_MM x FRAME_H/PX_PER_MM`).

PRESENTATION CAMERA (viewer's camera, not a rig camera): a weak-perspective
pinhole (narrow FOV, ~2 deg half-angle) placed off to the side so the arc's
Y-Z sweep (world X is the shared "perpendicular to the long axis" direction,
confirmed above) reads as a wide fan on screen. Weak perspective is a
deliberate choice, not an oversight: with scene depth variation small
relative to camera distance, PARALLEL 3D lines project to (very nearly)
parallel 2D lines regardless of viewing angle -- `_check_geometry` measures
the worst-case pairwise angular spread within any one camera's ray bundle
after this presentation projection and asserts it stays under 1 deg (measured
~0.1 deg), so the viewer's perspective cannot be mistaken for rig
convergence. The camera's distance and image-plane centring are CALIBRATED
(`_calibrate_scene_radius`, then a principal-point recentring offset -- see
render_act2), not assumed: a task-10 composition review found the original
fixed distance/FOV filled only ~40% of the canvas, off-centre. Both
calibration steps are pure changes to the VIEWER's camera (distance and
image-plane translation) -- neither can affect measured ray parallelism,
which depends only on 3D ray directions and viewing distance, not on where
the image plane's origin sits or (per the re-measurement above) on this
narrow an FOV.

TASK-29 (user: shorten the whole explainer to ~30 s total; Act 2's budget
halved 300 -> 150 frames): all four phases below (fly-out, ray-extend, cloud-
materialise, fade-out) were scaled by the SAME 150/300 = 0.5 factor, so the
sequence kept the same proportions and read at a uniformly faster pace
rather than truncating any one phase.

TASK-31 (user: "the rays combining are a slightly too fast, so much so that
we can't see the 3D keypoints before the transition... keep the points on
the screen a little longer" -- and shorten the crossfades to compensate):
Act 2 GROWS 150 -> 240 frames, but NOT by uniform rescaling this time -- the
extra 90 frames are weighted toward the END so the held-cloud beat (the one
the user says they can't currently see) gets real dwell time, while the
earlier phases keep roughly their TASK-29 pace rather than being stretched
thin themselves:
  fly-out       60 -> 80 frames (+20)
  ray-extend    40 -> 50 frames (+10) -- still reads as a comfortable
                extend, not stretched thin
  cloud-materialise 30 -> 50 frames (+20)
  fade-out (cloud held alone) 20 -> 60 frames (+40) -- most of the growth
                lands here, on purpose.
The easing (`_smoothstep`, applied via each phase's own progress fraction)
is untouched -- it is already parametrised on the phase-boundary constants
below, so moving the boundaries alone reshapes timing without changing the
shape of any individual ramp (nothing snaps).

Timeline (240 frames, 30 fps; TASK-31: was 150 frames):
  f   0- 79  panels detach from the Act 1 grid (same row/col cell Act 1 used)
             and fly out (ease-in-out position + orientation slerp) to their
             true rig pose; 2D keypoint constellations fade in EARLY here
             (task-14 round 4: "keypoints projected on the frames from the
             start of Act 2", not held back until the panels finish moving)
             -- video is still fully opaque, so keypoints are drawn directly
             on the real video texture, same as Act 1's own overlay.
  f  80-129  video fades out to a dim translucent plate (keypoints, already
             visible, persist through the crossfade); parallel rays extend
             inward from the panel toward the (not-yet-visible) cloud.
  f 130-179  the 3D keypoint cloud materialises (alpha ramp) exactly where the
             ray bundles meet.
  f 180-239  panels + rays fade out; cloud remains alone for a full 60-frame
             (2 s) hold -- TASK-31's whole point: give the viewer time to
             actually read the cloud as a fly before the act ends.

EXPECTATION: f=0 shows the Act 1 grid tiles, video only, no keypoints yet
(alpha still ramping from 0); f=60 shows keypoint constellations fully
visible directly on the video, mid-fly-out; f=100 shows panels on the true
180 deg arc, translucent, with keypoint constellations and PARALLEL rays
partway extended (TASK-31 EXPECTATION: comfortably paced, not a blur);
f=170 shows the cloud mid-materialisation, panels/rays still present;
f=235 shows the fly-shaped cloud clearly legible, panels/rays faded to a
faint (~7%-alpha) residual that does not obscure it -- full fade-out
completes at the act's actual last frame (239).
FALSIFICATION: rays fanning to a shared point (not staying parallel within a
camera's own bundle) means a perspective model leaked into the rig geometry;
a cloud not sitting where the rays end means a wrong transform; a cloud that
is still hard to read as a fly at f=235 means TASK-31's dwell-time request
was not actually satisfied.

Colours: per-keypoint, JARVIS per-limb-chain scheme (`kp_colors.jarvis_kp_colors`),
same as Act 1. Task-33: the ray-bundle lines are ALSO per-keypoint JARVIS
colours now (previously a flat `PALETTE["fit"]` green) -- see the TASK-33
section below.

TASK-18 DECISION (kept on `03_kp3d.npz`, deliberately NOT switched to the
user-supplied `data3D.csv` or the production solve's `kp_data`): this act's
entire claim is that its cloud IS the triangulation of the 2D shown in Act 1
-- enforced at runtime by the `kp2d/kp3d keypoint order mismatch` assert
below and by `_check_geometry`'s reprojection sanity check, which reprojects
this act's OWN panel-plane construction back through the real rig DLTs and
compares it to `02_kp2d.npz`'s own `kp2d_t0` (the same array Act 1 draws).
Measured before deciding (not assumed): reprojecting `data3D.csv` (converted
to mm, MODEL order) through the same DLTs at T0=525 gives a mean 10.3 px /
max 38.3 px error against `02_kp2d.npz`'s OWN observed 2D -- worse than this
act's existing `03_kp3d.npz`-based construction (mean 3.1 px / max 28.5 px
against the same 2D, i.e. what the module docstring above already reports),
and closer to failing its own <40 px construction-reprojection assert.
`data3D.csv` is a separate, independently-produced triangulation, not derived
from this repo's own re-run 2D detector -- swapping the cloud to it while
Act 1 still shows OUR OWN 2D keypoints would sever the "same points, by
construction" claim this act exists to make. (The production solve's own
`kp_data`, by contrast, reprojects to 3.1 px / 28.9 px -- essentially
identical to `03_kp3d.npz` -- because it turns out to BE this repo's own
filtered triangulation, `04_kp3d_filt.npz`, carried through the STAC solve;
see act3_align.py's TASK-18 MIXTURE section for that comparison. Even so,
Act 2 stays on `03_kp3d.npz` directly rather than `kp_data`/`shared_scale`,
since `03_kp3d.npz` is this act's own pre-existing, already-verified,
un-filtered source and introducing a second production-solve dependency here
would not change what is shown.) Act 1 and Act 2 are therefore UNCHANGED by
task-18.

TASK-19 (presentation-only): on-screen title changes "ACT 2" -> "3D
triangulation", and the title card's text is stripped to ONLY the title and
the "7 cameras, 180 deg arc, 30 deg spacing (elev ... deg)" fact line -- the
two "panel DIRECTION = measured optical axis" / "panel DISTANCE ... staging
only" caption lines are removed from the FRAME per the user's explicit
request. Both facts they described remain true and are documented in this
module's RIG GEOMETRY section above (nothing about the geometric
construction changed) -- only their on-screen captions are gone. Per-panel
camera labels remain UNCHANGED IN KIND (per-panel annotations, not part of
the removed caption block) -- task-33 changes their CONTENT from "Camera N
elev <measured> deg" to "Camera N  <arc> deg" (see TASK-33 section below) and
renumbers/reorders panels to follow the arc, matching Act 1. Typography now
comes from `draw.py`'s shared `TITLE_SCALE`/`CAPTION_SCALE`/`SMALL_SCALE`
instead of this module's own ad hoc scale values.

TASK-33 (user: clean 0-180 deg arc labels + renumber cameras along the arc;
colour rays by keypoint): panel labels/numbering now come from
`clip_io.display_names`/`clip_io.arc_order` (same arc-derived numbering Act 1
uses -- see `act1_views.py`'s TASK-33 section), so "Camera N" and its
on-screen deg value both increase monotonically 0->180 across the arc
instead of folding at the arc's 90 deg mid-point. `_grid_start_frame`'s
per-camera Act-1-style "start" grid cell is keyed off this SAME arc order so
the fly-out still detaches from the cell Act 1 actually drew each camera in.
Ray-bundle lines (`_draw_rays`) are recoloured from a flat `PALETTE["fit"]`
green to per-keypoint JARVIS colours, looked up BY NAME
(`kp_colors[kp_names[k]]`) -- ray GEOMETRY (parallel-within-a-bundle
construction) is untouched; only line colour changed. See `_draw_rays`'s own
docstring for the colour-by-name rationale and `_RAY_ALPHA_SCALE` below for
the legibility tuning this required.

TASK-33 FOLLOW-UP (header/panel-label mismatch): the panel-label switch
above left the fact-line header (TASK-19's "7 cameras, 180 deg arc, 30 deg
spacing (elev ... deg)") un-updated, so it still advertised the OLD raw-
elevation convention directly under panel labels now reading the NEW
arc-position convention ("Camera 1  0 deg" ... "Camera 7  180 deg") --
"0 to 180" on the panels contradicted "-89.6 to +0.6" in the header. Fixed
by dropping the stale elevation parenthetical; the header is now just
"7 cameras, 180 deg arc, 30 deg spacing", consistent with the panel labels.
`elev_deg` (only ever computed to feed that parenthetical) is removed as
dead code along with it.

TASK-34 (user: make the Act1->Act2 cut a continuation, not a jump -- panel
fly-out should start FROM the Act 1 grid, not somewhere already different):
`_grid_start_frame` always claimed to place frame-0 panels "at Act 1's
(row,col) cell", but its formula never actually inverted Act 1's own pixel
rectangle -- it placed a flat panel at `pres_cam.forward * pres_cam.dist`
(measured from the scene origin/fly centroid) and sized it assuming that
point's on-screen depth (`zc`, the `rel@forward` `PresentationCamera.project`
divides by) equals `pres_cam.dist`. It does not: `PresentationCamera.project`
computes `rel = X_local - eye` with `eye = eye_dir*dist = -forward*dist`, so
a point at `forward*dist` has `zc = rel@forward = 2*dist`, not `dist` --
twice the assumed depth, so every start panel actually rendered at roughly
HALF its intended on-screen size, offset from Act 1's cell. Fixed by
inverting Act 1's OWN pixel rectangle (`draw.panel_cell_rect`, now shared
with `act1_views.py` instead of a second hardcoded grid-pitch copy) directly
through the camera's projection algebra at a fixed depth -- see
`_grid_start_frame`'s own docstring for the exact derivation. Measured at
render time (task-34 report): per-camera centroid shift and area ratio
between `act1_views/f00149.png`'s grid and `act2_triangulate/f00000.png`'s
opening pose. Ray geometry, panel arc pose/orientation, and every phase
boundary below are UNCHANGED -- only the fly-out's STARTING pose moved.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))

from scripts.viz.ik_explainer import clip_io, draw          # noqa: E402
from scripts.viz.ik_explainer.kp_colors import jarvis_kp_colors  # noqa: E402

# --- canvas / timeline ----------------------------------------------------
CANVAS_W, CANVAS_H = 1920, 1080
N_OUT = 240                          # TASK-31: was 150 -- NOT a uniform
                                      # rescale; see the TASK-31 module-
                                      # docstring section (extra time weighted
                                      # toward the final held-cloud phase).
FLY_START, FLY_END = 0, 79           # 0-79 inclusive: detach + fly out (TASK-31: was 0-59; TASK-29: was 0-119)
KP_EARLY_END = 59                    # task-14 round 4: keypoints fade in EARLY
                                      # (independent of the panel fly-out), so
                                      # Act 2 shows "keypoints projected on the
                                      # frames" from near the start, not only
                                      # once panels reach their arc pose.
                                      # TASK-31: was 44, scaled with the
                                      # fly-out phase's own growth (60->80
                                      # frames), keeping the same ~74%
                                      # fraction and ~21-frame margin before
                                      # RAY_START; TASK-29: was 89.
RAY_START, RAY_END = 80, 129         # 80-129 inclusive: video->constellation, rays extend (TASK-31: was 60-99; TASK-29: was 120-199)
CLOUD_START, CLOUD_END = 130, 179    # 130-179 inclusive: cloud alpha ramp (TASK-31: was 100-129; TASK-29: was 200-259)
FADE_START, FADE_END = 180, 239      # 180-239 inclusive: panels+rays fade out, cloud held alone (TASK-31: was 130-149 -- widened 20->60 frames; TASK-29: was 260-299)

# --- rig / crop constants (match Act 1's conventions) ----------------------
CROP_W, FRAME_H = 430, 448
PX_PER_MM = 80.7
T0 = 525                             # representative frame: see _pick_frame note below.
# Chosen by scanning mean 2D+3D confidence over the clip's middle third
# (scripts run by hand, not committed): frame 525 has mean detector conf 0.955,
# mean 3D conf 0.955, worst-keypoint 3D conf 0.90, and a 3.5x2.6x1.5 mm
# keypoint bbox consistent with a walking fly with legs splayed (not folded),
# i.e. a frame where "recognisably fly-shaped, six legs" is a fair ask.

R_STAGE = 35.0        # mm; ARBITRARY panel distance -- staging only, captioned on screen.
# TASK-34: Act 1's cell grid (cell 7 was the title card, left empty here) is
# now `draw.panel_cell_rect` (shared with act1_views.py), not a second
# hardcoded copy -- see `_grid_start_frame` below.

# --- presentation (viewer) camera ------------------------------------------
_EYE_DIR = np.array([0.60, 0.25, 0.65])
_EYE_DIR = _EYE_DIR / np.linalg.norm(_EYE_DIR)
# Chosen (searched over candidates, script run by hand) to keep the arc's
# ~760px on-screen spread while avoiding an eye_dir aligned with the shared
# camera "right" axis (world +X, ~identical for all 7 rig cameras -- see
# `_load_rig`'s assertion): looking too close to along +X foreshortens every
# panel's own width down to a near-invisible sliver. This direction keeps the
# worst-case panel width/height ratio at ~0.70 (vs ~0.40 for a more X-aligned
# choice), so every panel stays legibly rectangular.
_HALF_FOV_DEG = 2.0                  # narrow FOV -> weak perspective, parallel lines stay ~parallel.
# Narrower than the original 6 deg: composition review (task-10 fix) needed the
# presentation camera to sit noticeably closer to fill the frame (see
# `_calibrate_scene_radius` below), and moving closer at a fixed FOV increases
# scene-depth-variation/distance -- i.e. more perspective creep, the one thing
# that must not happen here. Narrowing the FOV compensates: for the same
# on-screen framing, `dist = f_px * r / target_px` grows with `f_px` (which
# grows as FOV shrinks), keeping depth-variation/distance small again. Refit
# and re-measured by `_check_geometry` below, not assumed.
_WORLD_UP_REF = np.array([0.0, 0.0, 1.0])
_SCENE_RADIUS_MM = R_STAGE + 4.0     # nominal panel extent; the ACTUAL wide-shot
# radius fed to the camera is `_calibrate_scene_radius`'s output (see
# render_act2) -- this constant is only the search upper bound.
_CLOUD_RADIUS_MM = 3.5               # fly-cloud half-diagonal (~2.3 mm) + margin
_FILL_FRACTION = 0.85                # fraction of half-canvas-height the scene should span

# --- composition target (task-10 fix): where the arc/rays/cloud should sit,
# leaving the top-left caption block and the bottom-left frame counter clear.
_CAPTION_CLEAR_Y = 280.0   # px; stay clear of the 3-line on-screen caveat + camera labels
_BOTTOM_CLEAR_Y = 1010.0   # px; stay clear of the "frame N/240" label (TASK-31: was N/150; TASK-29: was N/300)
_SIDE_MARGIN_PX = 90.0
_TARGET_CX = CANVAS_W / 2.0
_TARGET_CY = (_CAPTION_CLEAR_Y + _BOTTOM_CLEAR_Y) / 2.0
_TARGET_HALF_W = (CANVAS_W - 2 * _SIDE_MARGIN_PX) / 2.0
_TARGET_HALF_H = (_BOTTOM_CLEAR_Y - _CAPTION_CLEAR_Y) / 2.0


def _smoothstep(p):
    p = np.clip(p, 0.0, 1.0)
    return 3 * p ** 2 - 2 * p ** 3


class PresentationCamera:
    """Hand-rolled weak-perspective pinhole -- the VIEWER's camera, not a rig
    camera. Operates entirely in a scene-local frame (world minus the fly
    centroid); callers must subtract the centroid before calling `project`.
    """

    def __init__(self, eye_dir, half_fov_deg, scene_radius_mm, fill_fraction,
                 canvas_w, canvas_h, world_up_ref=_WORLD_UP_REF):
        self.forward = -eye_dir
        self.right = np.cross(self.forward, world_up_ref)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.forward)
        self.up /= np.linalg.norm(self.up)
        self.f_px = (canvas_h / 2.0) / np.tan(np.radians(half_fov_deg))
        self.target_px = fill_fraction * (canvas_h / 2.0)
        self.eye_dir = eye_dir
        self.dist = self.f_px * scene_radius_mm / self.target_px
        self.eye = eye_dir * self.dist
        self.canvas_w, self.canvas_h = canvas_w, canvas_h
        # Principal-point OFFSET from true canvas centre: a pure image-plane
        # translation used only to recentre the arc's off-axis geometry (the
        # fly centroid, X_local=(0,0,0), always projects to exactly
        # (ppx, ppy) regardless of `dist` -- see `project`). This cannot
        # change ray angles (it is added identically to every projected
        # point), so it cannot affect the parallel-ray check.
        self.ppx, self.ppy = self.canvas_w / 2.0, self.canvas_h / 2.0

    def set_scene_radius(self, scene_radius_mm):
        """Dolly the eye along its fixed viewing direction so `scene_radius_mm`
        fills the same on-screen fraction. FOV/right/up/forward are untouched,
        so this is a true dolly (move closer), not a focal-length zoom -- it
        does not change how parallel the rig rays look (that only depends on
        scene-depth-variation / distance, both of which shrink together)."""
        self.dist = self.f_px * scene_radius_mm / self.target_px
        self.eye = self.eye_dir * self.dist

    def set_principal_point_offset(self, delta_u, delta_v, t):
        """Blend the principal-point offset in by fraction `t` in [0,1] (t=0:
        untouched canvas-centre projection, matching the Act-1-style grid
        pose; t=1: full recentring offset). A pure translation, independent
        of `dist`/FOV, so it cannot introduce or hide ray convergence."""
        self.ppx = self.canvas_w / 2.0 + t * delta_u
        self.ppy = self.canvas_h / 2.0 + t * delta_v

    def project(self, X_local):
        """(...,3) scene-local points -> ((...,2) pixel coords, (...) depth)."""
        rel = np.asarray(X_local, np.float64) - self.eye
        xc = rel @ self.right
        yc = rel @ self.up
        zc = rel @ self.forward
        u = self.ppx + self.f_px * xc / zc
        v = self.ppy - self.f_px * yc / zc
        return np.stack([u, v], axis=-1), zc


def _panel_plane_point(M, view_dir, centroid, uv):
    """Exact inversion: real affine DLT row + panel-plane constraint -> world pt.

    M is clip_io.load_dlt's (4,3) matrix (ph @ M = [u,v,1]); telecentric means
    u,v are affine in (x,y,z) with zero component along view_dir, so the 3x3
    system [row_u; row_v; view_dir] is generically full rank and gives the
    UNIQUE point on the panel plane (X . view_dir = -R_STAGE, expressed in
    absolute world coords) that camera `M` would image at pixel `uv`.
    """
    a, bconst = M[:3, 0], M[3, 0]
    c, dconst = M[:3, 1], M[3, 1]
    A = np.stack([a, c, view_dir], axis=0)
    rhs = np.array([uv[0] - bconst, uv[1] - dconst, -R_STAGE + view_dir @ centroid])
    return np.linalg.solve(A, rhs)


def _load_rig(clip):
    cam_mats, names = clip_io.load_dlt(str(Path(clip) / "calibration"))
    view_dirs = clip_io.camera_view_dirs(cam_mats)
    # Confirms the brief's claim in the data: every camera's own right-axis is
    # ~world +X (perpendicular to the arena's long axis), independent of the
    # camera_view_dirs() call above (uses the same factor_affine under the
    # hood, cross-checked in _check_geometry).
    return cam_mats, names, view_dirs


def _panel_end_frame(cam_mats, names, view_dirs, centroid, kp2d_t0):
    """Per camera: (right, up, normal, center_local, width_mm, height_mm)."""
    out = {}
    for ci, cam in enumerate(names):
        M, vd = cam_mats[ci], view_dirs[ci]
        cx = float(np.nanmean(kp2d_t0[ci, :, 0]))
        x0 = cx - CROP_W / 2.0
        corners_px = {"TL": (x0, 0.0), "TR": (x0 + CROP_W, 0.0),
                      "BR": (x0 + CROP_W, FRAME_H), "BL": (x0, FRAME_H)}
        W = {k: _panel_plane_point(M, vd, centroid, uv) for k, uv in corners_px.items()}
        right = W["TR"] - W["TL"]
        up = W["TL"] - W["BL"]
        width_mm, height_mm = np.linalg.norm(right), np.linalg.norm(up)
        right, up = right / width_mm, up / height_mm
        normal = np.cross(right, up)
        center_world = np.mean(list(W.values()), axis=0)
        out[cam] = dict(right=right, up=up, normal=normal,
                         center_local=center_world - centroid,
                         width_mm=width_mm, height_mm=height_mm, x0=x0)
    return out


def _check_geometry(cam_mats, names, view_dirs, centroid, kp3d_t0, kp2d_t0,
                     end_frames, pres_cam):
    """Runtime defusing checks -- printed and asserted, not just claimed."""
    kp3d_c = kp3d_t0 - centroid
    max_reproj_err = 0.0
    for ci, cam in enumerate(names):
        vd = view_dirs[ci]
        dot = kp3d_c @ vd
        panel_pts_world = (kp3d_c - np.outer(dot, vd)) - R_STAGE * vd + centroid
        proj = clip_io.project(cam_mats[ci:ci + 1], panel_pts_world)[0]
        err = np.linalg.norm(proj - kp2d_t0[ci], axis=1)
        max_reproj_err = max(max_reproj_err, float(err.max()))
        ef = end_frames[cam]
        assert abs(ef["right"] @ ef["up"]) < 0.02, f"{cam}: panel right/up not orthogonal"
        assert 4.5 < ef["width_mm"] < 6.5 and 4.5 < ef["height_mm"] < 6.5, \
            f"{cam}: panel size {ef['width_mm']:.2f}x{ef['height_mm']:.2f} mm off expected ~5.3x5.5"
    print(f"[act2] panel-plane reprojection vs real 2D detections: "
          f"max {max_reproj_err:.1f} px (T0={T0})")
    assert max_reproj_err < 40.0, "panel-plane construction disagrees with real detections"

    # Ray-bundle parallelism under the PRESENTATION camera.
    max_dev_all = 0.0
    for ci, cam in enumerate(names):
        vd = view_dirs[ci]
        dot = kp3d_c @ vd
        starts = (kp3d_c - np.outer(dot, vd)) - R_STAGE * vd
        ends = kp3d_c
        uv0, _ = pres_cam.project(starts)
        uv1, _ = pres_cam.project(ends)
        d = uv1 - uv0
        n = np.linalg.norm(d, axis=1)
        dn = d[n > 1e-9] / n[n > 1e-9, None]
        mean_dir = dn.mean(axis=0)
        mean_dir /= np.linalg.norm(mean_dir)
        ang = np.degrees(np.arccos(np.clip(dn @ mean_dir, -1.0, 1.0)))
        max_dev_all = max(max_dev_all, float(ang.max()))
    print(f"[act2] max pairwise ray-direction spread within any one camera's "
          f"bundle, under the presentation camera: {max_dev_all:.2f} deg")
    assert max_dev_all < 1.0, "presentation perspective is making rays look non-parallel"


def _grid_start_frame(pres_cam, panel_idx):
    """Start pose for ARC-order panel `panel_idx` (0-6): an EXACT inversion of
    `draw.panel_cell_rect(panel_idx)` -- Act 1's own on-screen pixel rectangle
    for that panel -- through this weak-perspective camera's projection, flat
    and facing the viewer.

    TASK-34 (was: an approximate hand-placed grid -- flat panels arranged
    around `pres_cam.forward*depth` by a fixed pitch, with `depth =
    pres_cam.dist` fed into `width_mm = PANEL_W/f_px*depth` as if that were
    the point's on-screen depth `zc`. It is not: a point at
    `pres_cam.forward*depth` (measured from the scene origin, i.e. the fly
    centroid) is at `zc = 2*depth` once `PresentationCamera.project` subtracts
    `pres_cam.eye = pres_cam.eye_dir*dist`, since `forward = -eye_dir` -- so
    every start panel actually rendered at roughly HALF `PANEL_W/H`'s
    intended screen size, off-position, which is exactly why Act 2 used to
    open somewhere other than Act 1's grid):

    Fix each corner's desired SCREEN pixel (from `draw.panel_cell_rect`) and
    solve the camera's own (invertible, affine-in-u/v at fixed depth `Z`)
    projection backwards for the 3D point that lands there. `project`'s
    forward map is `rel = X_local - eye; u = ppx + f_px*(rel@right)/(rel@
    forward); v = ppy - f_px*(rel@up)/(rel@forward)`; fixing `rel@forward =
    Z` (any positive `Z` -- weak perspective, so the choice does not change
    the answer) makes this a LINEAR bijection between `(u, v)` and
    `(rel@right, rel@up)`, inverted directly below. This is exact (to float
    precision) given `pres_cam` is in the SAME state (dist/f_px/right/up/
    forward/ppx/ppy) it will be in at output frame 0 -- true here: this is
    called right after `_calibrate_scene_radius` sets `pres_cam.dist` to
    `wide_radius`, and frame 0's own `_scene_radius(0, wide_radius) ==
    wide_radius` / `set_principal_point_offset(..., t=0)` (see render_act2)
    reproduce that identical state, so the two evaluations of `project`
    agree exactly."""
    px, py, w, h = draw.panel_cell_rect(panel_idx)
    Z = pres_cam.dist   # any positive depth works (weak perspective); see docstring
    corners_px = {"TL": (px, py), "TR": (px + w, py),
                  "BR": (px + w, py + h), "BL": (px, py + h)}

    def _invert(u, v):
        xc = (u - pres_cam.ppx) * Z / pres_cam.f_px
        yc = -(v - pres_cam.ppy) * Z / pres_cam.f_px
        return pres_cam.eye + xc * pres_cam.right + yc * pres_cam.up + Z * pres_cam.forward

    W = {k: _invert(*uv) for k, uv in corners_px.items()}
    right = W["TR"] - W["TL"]
    up = W["TL"] - W["BL"]
    width_mm, height_mm = np.linalg.norm(right), np.linalg.norm(up)
    right, up = right / width_mm, up / height_mm
    normal = np.cross(right, up)
    center_local = np.mean(list(W.values()), axis=0)
    return dict(right=right, up=up, normal=normal, center_local=center_local,
                width_mm=width_mm, height_mm=height_mm)


def _basis_rotation(frame):
    return Rotation.from_matrix(np.column_stack([frame["right"], frame["up"], frame["normal"]]))


def _interp_frame(start, end, p):
    p = float(_smoothstep(p))
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([_basis_rotation(start), _basis_rotation(end)]))
    R = slerp(p).as_matrix()
    center = (1 - p) * start["center_local"] + p * end["center_local"]
    w = (1 - p) * start["width_mm"] + p * end["width_mm"]
    h = (1 - p) * start["height_mm"] + p * end["height_mm"]
    return dict(right=R[:, 0], up=R[:, 1], normal=R[:, 2], center_local=center,
                width_mm=w, height_mm=h)


def _quad_corners(frame):
    c, r, u = frame["center_local"], frame["right"], frame["up"]
    hw, hh = frame["width_mm"] / 2.0, frame["height_mm"] / 2.0
    return {"TL": c - hw * r + hh * u, "TR": c + hw * r + hh * u,
            "BR": c + hw * r - hh * u, "BL": c - hw * r - hh * u}


def _video_alpha(f):
    if f <= FLY_END:
        return 1.0
    if f >= RAY_END:
        return 0.0
    return 1.0 - (f - RAY_START) / float(RAY_END - RAY_START)


def _kp_alpha(f):
    """Keypoint-constellation opacity on the video texture. Task-14 round 4:
    fades in EARLY (0-KP_EARLY_END), independent of the panel fly-out/ray
    timing below -- the user asked for keypoints visible on the frames from
    the start of Act 2, not held back until panels reach their true arc
    pose. Reaches 1.0 well before `RAY_START` and stays there (the video->
    constellation crossfade and ray extension are separate visual layers,
    unaffected by this)."""
    return float(_smoothstep(f / float(KP_EARLY_END)))


def _ray_progress(f):
    if f <= FLY_END:
        return 0.0
    if f >= RAY_END:
        return 1.0
    return (f - RAY_START) / float(RAY_END - RAY_START)


def _cloud_alpha(f):
    if f < CLOUD_START:
        return 0.0
    if f >= CLOUD_END:
        return 1.0
    return (f - CLOUD_START) / float(CLOUD_END - CLOUD_START)


def _panel_overall_alpha(f):
    if f < FADE_START:
        return 1.0
    if f >= FADE_END:
        return 0.0
    return 1.0 - (f - FADE_START) / float(FADE_END - FADE_START)


def _scene_radius(f, wide_radius):
    """Wide arc framing (`wide_radius`, calibrated by `_calibrate_scene_radius`
    to fill most of the canvas -- see render_act2) through the panel/ray
    phases; dolly IN (not a focal zoom -- see
    `PresentationCamera.set_scene_radius`) toward a tight cloud-only framing
    only once panels/rays are already fading out (180-239, TASK-31: was
    130-149; TASK-29: was 260-299), so the fly-shaped
    cloud is actually legible in the final frames instead of a several-px blob
    at the wide, whole-arc scale. `_CLOUD_RADIUS_MM` (the dolly's endpoint) is
    untouched by the wide-shot recalibration: it only depends on `target_px`/
    `f_px`, neither of which the calibration changes."""
    if f < FADE_START:
        return wide_radius
    p = _smoothstep((f - FADE_START) / float(FADE_END - FADE_START))
    return (1 - p) * wide_radius + p * _CLOUD_RADIUS_MM


def _arc_bbox(pres_cam, end_frames, names):
    """Pixel-space bounding box of every panel-quad corner in its TRUE (arc)
    pose, under `pres_cam`'s current dist/principal-point. This is the extent
    that must fit inside the composition target during the panel/ray phases."""
    us, vs = [], []
    for cam in names:
        for X in _quad_corners(end_frames[cam]).values():
            uv, _ = pres_cam.project(X[None, :])
            us.append(uv[0, 0])
            vs.append(uv[0, 1])
    us, vs = np.asarray(us), np.asarray(vs)
    return float(us.min()), float(us.max()), float(vs.min()), float(vs.max())


def _calibrate_scene_radius(pres_cam, end_frames, names, r_hi):
    """Bisect the wide-shot `scene_radius_mm` so the arc's panel-quad bbox
    exactly fills the composition target (`_TARGET_HALF_W/H`) on its binding
    axis -- i.e. find the biggest on-screen arc that still fits inside the
    margin, rather than assuming a fill fraction that (as shipped) undershot
    by ~2x. `r_hi` (the un-recalibrated `_SCENE_RADIUS_MM`) is known to be
    an UNDER-fill (bigger r => smaller image), so it's a safe search upper
    bound; the search is monotonic (bigger r -> smaller on-screen extent)."""
    lo, hi = 0.5, r_hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        pres_cam.set_scene_radius(mid)
        u0, u1, v0, v1 = _arc_bbox(pres_cam, end_frames, names)
        ratio = max((u1 - u0) / 2.0 / _TARGET_HALF_W, (v1 - v0) / 2.0 / _TARGET_HALF_H)
        if ratio > 1.0:
            lo = mid   # overflowed the target -> need a LARGER radius (further away, smaller image)
        else:
            hi = mid
    pres_cam.set_scene_radius(hi)
    return hi


_DARK_PLATE = np.full((FRAME_H, CROP_W, 3), 22, np.uint8)
_BACKDROP_OPACITY = 0.55   # translucent plate: dims but doesn't blank the video base

# --- TASK-33 ray colouring: legibility tuning ------------------------------
# Up to 7 cameras x 50 keypoints = 350 ray segments on screen at once. Thin
# lines (thickness 1, the pre-existing value) keep each ray a hairline rather
# than a wide coloured band; `_RAY_ALPHA_SCALE` additionally dims the ray
# layer's own blend-in below the caller's `alpha` (`panel_a`) -- rendered and
# read at 1.0 (full `panel_a`) first: with 50 distinct hues at full strength
# the bundle read as a busy rainbow haze rather than "structure", especially
# where many chains' rays cross near the panels. At 0.55 the per-chain HUE
# (which is the entire point of Change 2 -- same-colour rays sharing a limb
# chain, converging on that chain's keypoints) still reads clearly, while the
# overall bundle recedes enough to look like translucent converging
# structure instead of clutter. Thickness is left at 1 (already the minimum
# useful value); alpha, not thickness, is what was tuned.
_RAY_THICKNESS = 1
_RAY_ALPHA_SCALE = 0.55


def _build_texture(video_crop, kp_local_uv, kp_names, video_alpha, kp_alpha, kp_colors):
    # crossfade video -> a dark plate, then dim the whole thing further the
    # deeper into "constellation mode" we are, so the panel genuinely reads
    # as translucent rather than just a slightly-faded photo.
    base = draw.fade(_DARK_PLATE, video_crop, video_alpha)
    dim_factor = 1.0 - (1 - video_alpha) * (1 - _BACKDROP_OPACITY)
    tex = (base.astype(np.float32) * dim_factor).astype(np.uint8)
    if kp_alpha > 0.0:
        tex = draw.draw_leg_chains(tex, kp_local_uv, kp_names, alpha=kp_alpha,
                                    thickness=1, kp_colors=kp_colors)
        tex = draw.draw_keypoints(tex, kp_local_uv, kp_names, alpha=kp_alpha,
                                   radius=4, kp_colors=kp_colors)
    return tex


def _warp_panel(canvas, texture, frame, pres_cam, alpha):
    corners = _quad_corners(frame)
    uv = {}
    for k, X in corners.items():
        p, _ = pres_cam.project(X[None, :])
        uv[k] = p[0]
    src = np.array([[0, 0], [CROP_W, 0], [CROP_W, FRAME_H], [0, FRAME_H]], np.float32)
    dst = np.array([uv["TL"], uv["TR"], uv["BR"], uv["BL"]], np.float32)
    if not np.all(np.isfinite(dst)):
        return canvas
    Mtx = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(texture, Mtx, (CANVAS_W, CANVAS_H),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    mask = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
    mask_f = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    return (canvas.astype(np.float32) * (1 - mask_f) + warped.astype(np.float32) * mask_f).astype(np.uint8)


def _draw_rays(canvas, panel_pts_local, kp3d_local, kp_names, progress, pres_cam,
                alpha, kp_colors):
    """Parallel-by-construction ray segments, one per keypoint, sharing camera
    c's exact `view_dir[c]`: see the module docstring's geometric construction.

    TASK-33 (user: colour each ray by the keypoint it belongs to, using the
    same JARVIS per-limb-chain colours as the keypoints/skeleton elsewhere):
    each of the K rays in this one camera's bundle is coloured by
    `kp_colors[kp_names[k]]` -- looked up BY NAME (never by the row's plain
    index into `panel_pts_local`/`kp3d_local`, which are in whatever order the
    caller's `kp_names` says -- see kp_colors.py's own module docstring on why
    this project keys colour lookups by name, not position). This recolours
    the lines only: `panel_pts_local`/`kp3d_local`/`progress` (the ray
    geometry) are untouched, so every ray in a camera's bundle still shares
    that camera's exact `view_dir` and stays parallel -- colour is drawn last,
    after the same endpoints already computed for the green version."""
    if progress <= 0.0 or alpha <= 0.0:
        return canvas
    out = canvas.copy()
    ends = panel_pts_local + progress * (kp3d_local - panel_pts_local)
    uv0, _ = pres_cam.project(panel_pts_local)
    uv1, _ = pres_cam.project(ends)
    for p0, p1, name in zip(uv0, uv1, kp_names):
        if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
            cv2.line(out, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                      kp_colors[name], _RAY_THICKNESS, cv2.LINE_AA)
    return draw.fade(canvas, out, alpha * _RAY_ALPHA_SCALE)


def _draw_cloud(canvas, kp3d_local, kp_names, pres_cam, alpha, kp_colors):
    """The triangulated 3D cloud, in the SAME per-limb-chain JARVIS colours
    Act 1 uses for observed keypoints (`kp_colors.jarvis_kp_colors`) -- this
    is the triangulation result, not the STAC/IK fit, so it is not
    PALETTE["fit"] green (that convention starts at Act 3)."""
    if alpha <= 0.0:
        return canvas
    uv, _ = pres_cam.project(kp3d_local)
    out = draw.draw_leg_chains(canvas, uv, kp_names, alpha=1.0, thickness=2,
                                kp_colors=kp_colors)
    out = draw.draw_keypoints(out, uv, kp_names, alpha=1.0, radius=4,
                               kp_colors=kp_colors)
    return draw.fade(canvas, out, alpha)


def render_act2(clip: str = clip_io.CLIP_DEFAULT) -> Path:
    d = clip_io.out_dirs(clip)
    z2d = np.load(d["predictions"] / "02_kp2d.npz", allow_pickle=True)
    kp2d, cam_names = z2d["kp2d"], [str(c) for c in z2d["cam_names"]]
    kp_names = [str(n) for n in z2d["kp_names"]]
    z3d = np.load(d["predictions"] / "03_kp3d.npz", allow_pickle=True)
    kp3d = z3d["kp3d"]
    assert [str(n) for n in z3d["kp_names"]] == kp_names, "kp2d/kp3d keypoint order mismatch"
    kp_colors = jarvis_kp_colors(kp_names)

    cam_mats, names, view_dirs = _load_rig(clip)
    assert list(names) == cam_names, (
        f"load_dlt order {names} != 02_kp2d.npz cam_names {cam_names}")
    # DISPLAY ONLY (task-33): "Camera N  <arc> deg" panel labels, numbered
    # along the rig's measured arc (`clip_io.display_names`) -- same numbering
    # Act 1 uses. `names`/`cam_names` (the real Cam20128xx strings) remain
    # what every DLT/kp2d/kp3d lookup below uses -- `disp_name`/`panel_order`
    # are only read when drawing/placing the on-screen panels.
    disp_name = clip_io.display_names(names, clip)
    # ARC order, same as Act 1's now-arc-ordered grid: each camera's Act-1-
    # style "start" grid cell (`_grid_start_frame`) must match the cell Act 1
    # actually drew it in, so the "detach from the Act 1 grid" fly-out reads
    # as continuous rather than jump-cutting cameras between cells.
    panel_order = clip_io.arc_order(names, clip)
    panel_idx_by_name = {cam: i for i, cam in enumerate(panel_order)}
    # Confirms the brief's "perpendicular to the arena's long axis" claim in
    # THIS clip's own data, not assumed from the spec table.
    from jarvis_jax.tracking.affine_camera import factor_affine
    right_axes = np.stack([factor_affine(M.T)[1][0] for M in cam_mats])
    max_right_dev = float(np.degrees(np.arccos(np.clip(np.abs(right_axes @ [1, 0, 0]), -1, 1))).max())
    print(f"[act2] max deviation of any camera's right-axis from world +X: {max_right_dev:.2f} deg")
    assert max_right_dev < 5.0, "cameras are not all perpendicular to a shared long axis"

    centroid = kp3d[T0].mean(axis=0)
    kp3d_local = kp3d[T0] - centroid
    kp2d_t0 = kp2d[T0]                                     # (C,K,2)

    end_frames = _panel_end_frame(cam_mats, names, view_dirs, centroid, kp2d_t0)

    pres_cam = PresentationCamera(_EYE_DIR, _HALF_FOV_DEG, _SCENE_RADIUS_MM,
                                   _FILL_FRACTION, CANVAS_W, CANVAS_H)

    # --- task-10 composition fix: rescale + recentre the wide (arc) shot ---
    # so panels+rays fill most of the canvas instead of ~40% of it, off
    # centre. `_calibrate_scene_radius` finds the biggest wide-shot radius
    # that keeps every panel-quad corner inside the composition target;
    # `delta_u/delta_v` (a pure image-plane translation -- see
    # `PresentationCamera.set_principal_point_offset`) then centres that
    # bbox in the target region. Neither can affect ray parallelism: the
    # radius change is exactly the pre-existing "dolly" mechanism (already
    # verified below by `_check_geometry`), and a translation cancels out of
    # any direction/angle computed from two projected points.
    wide_radius = _calibrate_scene_radius(pres_cam, end_frames, names, _SCENE_RADIUS_MM)
    u0, u1, v0, v1 = _arc_bbox(pres_cam, end_frames, names)
    delta_u = _TARGET_CX - (u0 + u1) / 2.0
    delta_v = _TARGET_CY - (v0 + v1) / 2.0
    print(f"[act2] wide-shot calibration: radius {wide_radius:.2f} mm "
          f"(was {_SCENE_RADIUS_MM:.2f} mm), bbox {u1 - u0:.0f}x{v1 - v0:.0f} px, "
          f"recentring offset ({delta_u:+.0f}, {delta_v:+.0f}) px")

    _check_geometry(cam_mats, names, view_dirs, centroid, kp3d[T0], kp2d_t0,
                     end_frames, pres_cam)

    # per-camera panel keypoint plane-points (local, world-orientation), and
    # their local-2D texture pixel coords, computed ONCE (T0 is static).
    panel_pts_local = {}         # cam -> (K,3) local coords, on the panel plane
    tex_uv = {}                  # cam -> (K,2) texture pixel coords
    for ci, cam in enumerate(names):
        vd = view_dirs[ci]
        dot = kp3d_local @ vd
        pp = (kp3d_local - np.outer(dot, vd)) - R_STAGE * vd
        panel_pts_local[cam] = pp
        ef = end_frames[cam]
        rel = pp - ef["center_local"]
        u_mm = rel @ ef["right"]
        v_mm = rel @ ef["up"]
        tex_uv[cam] = np.stack([CROP_W / 2.0 + u_mm * PX_PER_MM,
                                 FRAME_H / 2.0 - v_mm * PX_PER_MM], axis=-1)

    # video crops for T0 (static across the act; only their blend weight animates).
    video_crop = {}
    for ci, cam in enumerate(names):
        frame = clip_io.read_frames(clip_io.video_path(clip, cam), [T0])[0]
        fh, fw = frame.shape[:2]
        x0 = int(round(np.clip(end_frames[cam]["x0"], 0, fw - CROP_W)))
        video_crop[cam] = frame[:FRAME_H, x0:x0 + CROP_W].copy()

    # Task-33: grid start pose keyed off ARC order (matches Act 1's now-arc-
    # ordered panel placement), not `names`' sorted-by-serial order.
    start_frames = {cam: _grid_start_frame(pres_cam, panel_idx_by_name[cam])
                     for cam in names}

    out_dir = d["frames"] / "act2_triangulate"
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in range(N_OUT):
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
        pres_cam.set_scene_radius(_scene_radius(f, wide_radius))

        fly_p = (f - FLY_START) / float(FLY_END - FLY_START) if f <= FLY_END else 1.0
        # Recentring offset ramps IN with the same progress used to fly the
        # panels to their true pose (0 at the grid start, matching the
        # already-correct Act-1-style grid framing; full by f=FLY_END, held
        # through the ray-extend/cloud-materialise phases so the arc+rays
        # never drift once settled), then ramps back OUT in lockstep with
        # `_panel_overall_alpha`'s existing 180-239 fade/dolly (TASK-31: was
        # 130-149; TASK-29: was 260-299): the arc's
        # bbox centre is not where the cloud (= the fly centroid, which always
        # projects to the principal point, at any dist) naturally sits, so
        # holding the arc's offset through the pure-cloud ending would drag
        # the already-correctly-centred cloud off centre for no reason. Both
        # camera moves (recentre-out, dolly-in) share one 180-239 window and
        # ease together as a single final camera move.
        panel_a = _panel_overall_alpha(f)
        t_center = _smoothstep(fly_p) * _smoothstep(panel_a)
        pres_cam.set_principal_point_offset(delta_u, delta_v, t_center)
        v_a, k_a = _video_alpha(f), _kp_alpha(f)
        ray_p = _ray_progress(f)
        cloud_a = _cloud_alpha(f)

        frames_now = {}
        for cam in names:
            frames_now[cam] = _interp_frame(start_frames[cam], end_frames[cam], fly_p)

        # back-to-front by depth under the presentation camera.
        depths = {}
        for cam, fr in frames_now.items():
            _, zc = pres_cam.project(fr["center_local"][None, :])
            depths[cam] = float(zc[0])
        order = sorted(names, key=lambda c: -depths[c])

        if panel_a > 0.0:
            for cam in order:
                tex = _build_texture(video_crop[cam], tex_uv[cam], kp_names, v_a, k_a, kp_colors)
                canvas = _warp_panel(canvas, tex, frames_now[cam], pres_cam, panel_a)
                # Camera label ("Camera N  <arc> deg", task-33) just above the
                # panel's SCREEN-SPACE topmost corner (not the world "+up" edge --
                # once a panel is tilted well off the viewer's own up axis,
                # the world-top edge can project to the middle of the quad's
                # screen footprint, sitting the label over the panel instead
                # of clear of it; task-10 fix). Horizontally centred on the
                # quad's mean projected corner, not raw mm (panel size in mm
                # means nothing in screen pixels once perspective is applied).
                fr = frames_now[cam]
                corners_uv = np.stack(
                    [pres_cam.project(X[None, :])[0][0] for X in _quad_corners(fr).values()])
                if np.all(np.isfinite(corners_uv)):
                    cx_ = float(corners_uv[:, 0].mean())
                    top_v = float(corners_uv[:, 1].min())
                    text = disp_name[cam]
                    (tw, th), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, draw.SMALL_SCALE, 1)
                    canvas = draw.label(canvas, text,
                                         (cx_ - tw / 2.0, top_v - th - 8),
                                         scale=draw.SMALL_SCALE, color=(200, 200, 200))

            if ray_p > 0.0:
                for cam in names:
                    canvas = _draw_rays(canvas, panel_pts_local[cam], kp3d_local, kp_names,
                                         ray_p, pres_cam, panel_a, kp_colors)

        canvas = _draw_cloud(canvas, kp3d_local, kp_names, pres_cam, cloud_a, kp_colors)

        canvas = draw.stage_title(canvas, "3D triangulation")
        canvas = draw.label(
            canvas, "7 cameras, 180 deg arc, 30 deg spacing",
            (48, 150), scale=draw.CAPTION_SCALE, color=(190, 190, 190))
        canvas = draw.label(canvas, f"frame {f + 1}/{N_OUT}", (48, CANVAS_H - 24),
                             scale=draw.SMALL_SCALE, color=(150, 150, 150))

        # Task-32: PNG_COMPRESSION 1 (vs cv2's default 3) -- lossless, faster
        # zlib pass; decoded pixels are bit-identical (verified in the task-32
        # report). This is a build-speed change only.
        cv2.imwrite(str(out_dir / f"f{f:05d}.png"), canvas,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])

    print(f"wrote {N_OUT} frames to {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    a = ap.parse_args()
    render_act2(a.clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
