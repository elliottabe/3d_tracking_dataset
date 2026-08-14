#!/usr/bin/env python3
"""Act 4 -- the joints solve, then the fit plays back in motion.

TASK-18 (read this before touching this act's data-loading section below):
playback, the keypoint cloud/skeleton, and the on-screen residual now all
come from the PRODUCTION solve, `ik_production/stac_ik_full.h5` -- a full
STAC run the user supplied that, unlike this repo's own `stage_ik.py`/
`06_stages.npz`, actually ran `offset_optimization` (per-marker offsets on
top of the joint-angle/root solve). `stage_ik.py`'s own module docstring says
so explicitly: it mirrors `Stac.fit_offsets` only up through one
`root_optimization` + `pose_optimization` pair and DELIBERATELY never calls
`offset_optimization` -- so this repo's OWN staged `qpos_seq` (what this act
used before task-18) reflects a fit with the model's stock, un-fitted marker
placement, while the production `qpos` reflects a fit that also corrected
those marker positions (`offsets`, mean |0.015| max |0.26| mm -- real and
fitted, confirmed by reading them from the h5, not assumed). `06_stages.npz`
is still read for `frame_for_stills`/`shared_scale` (both unchanged between
the two solves, since they describe the SAME clip); Phase A's starting
mesh pose is `qpos_root_f0` (TASK-25: `predictions/06_stages_f0.npz`, a
root_optimization solved AT source frame 0 -- was `06_stages.npz`'s own
frame-450 `qpos_root` pre-TASK-25; see the TASK-25 section below), still
the right thing for Phase A to interpolate FROM: a genuine earlier stage of
a real solve. See PLAYBACK START and the EXPECTATION/FALSIFICATION sections
below for the specific numbers this swap changes (and the ones it,
reassuringly, does not).

TASK-19 (presentation changes -- read this before touching the render loop's
drawing calls or the Phase B/C split below; none of these alter what is
SOLVED or measured above, only what is DRAWN and how much of the playback is
side-by-side):
1. **On-screen title renamed** "ACT 4" -> "Solve joint angles" (matches the
   other three acts' renaming: "2D Keypoint tracking" / "3D triangulation" /
   "Root alignment"). Applied via `draw.stage_title`, using the ONE shared
   type scale all four acts now import from `draw.py`
   (`TITLE_SCALE`/`CAPTION_SCALE`/`SMALL_SCALE`) instead of this module's own
   ad hoc scale values.
2. **Every caption line except the title is removed from the frame**: the
   stage label, the residual-mm line, the rotated-so-far/joint-DOF-moved
   numbers, the root-vs-pose explanatory line, and (in the old closing 2-up)
   the per-camera label and the "cyan: observed / green: FK fit" legend are
   all gone from the rendered PNG. This is presentation-only: every number
   this act's docstring measures above (34.79 deg rotation, 3.95 DOF L2,
   `resid_prod_mm`'s 0.014-0.041 mm range, etc.) is UNCHANGED, still computed
   every frame, and still printed to the console (every 25 frames) -- only
   removed from what is drawn. The small bottom-left frame counter is the
   only other on-screen text kept.
3. **The side-by-side now spans the WHOLE rest of the act (240-899), not
   just the old closing 120-frame "Phase C"**: the old three-way split
   (0-239 solve-only; 240-779 full-width playback; 780-899 side-by-side)
   becomes TWO phases -- Phase A (0-239, unchanged: 3D-only, the
   `qpos_root -> qpos_prod[anchor]` solve) and a merged Phase B/C
   (240-899: side-by-side for all 660 remaining frames, left = the MuJoCo IK
   render, right = the real camera frame). UNCHANGED by this merge: the
   frame count (900 total, 660 in the merged phase) and the seam at f=240 --
   only the panel LAYOUT during 240-899 changes (every frame is now
   side-by-side, not just the last 120), never the frame's own source-index
   mapping. **STALE as of TASK-24**: the playback wrap/hold this point
   originally described (at the `frame_for_stills -> T-1 -> 0` seam, "see
   PLAYBACK START below") no longer exists -- TASK-24 removed it entirely.
   See the TASK-24 section below.
4. **The right panel no longer reprojects the FK'd marker sites** (the old
   green dots, `clip_io.project` through the DLT). It now draws the
   DETECTOR's own 2D keypoints (`02_kp2d.npz`, already loaded as `kp2d`
   below for the crop-centring smoothing -- MODEL order, matching `kp_names`,
   verified by the existing `kp2d_names != kp_names` guard) directly,
   coloured with the SAME per-limb-chain JARVIS scheme
   (`kp_colors.jarvis_kp_colors`, keyed by keypoint NAME) the left panel's
   mesh/cloud already uses (`kp_colors.jarvis_kp_colors_rgb01`, same
   underlying colour table) -- the two panels are colour-MATCHED rather than
   contrasted (see "COLOUR-MATCHED 2D KEYPOINTS" below; the old cyan-
   observed/green-fit `PALETTE` convention is retired for this panel). Drawn
   with `draw.draw_keypoints`/`draw.draw_leg_chains` (the shared helpers
   every other act already uses), not hand-rolled: `_dots`/`_chain` (the old
   single-colour analogues, and the FK-reprojection/`cam_mats`/DLT-loading
   code that was their only caller) are removed as dead code.
   Frame-index agreement between the two panels is enforced by
   CONSTRUCTION, not by a separate check: a single `t_src` value per output
   frame drives BOTH `qpos_prod[t_src]`/`kp_data_prod[t_src]` (left panel)
   AND `video_frames[t_to_video_row[t_src]]`/`kp2d[t_src, cam2up_idx]`
   (right panel) -- there is no second index variable that could drift from
   the first, through the wrap/hold included (the wrap only changes what
   `t_src` computes to, and both panels always read that same value).
   **SUPERSEDED by TASK-20 immediately below** -- point 4's own
   `02_kp2d.npz`/detector-2D design is what TASK-20 reverts, for the reason
   given there; points 1-3 above are unaffected and still describe the
   current behaviour.

TASK-20 (fixes a regression TASK-19's point 4 introduced -- the user
reported Act 4's right-panel keypoints had gone "jumpy" versus what the
original production showed): raw detector 2D (`02_kp2d.npz`) is genuinely
jittery frame-to-frame (measured 0.9-1.5 px depending on camera elevation,
worse on the near-horizontal `Cam2012857`/`Cam2012861` where the six leg
chains project nearly on top of each other) -- drawing it directly, as
TASK-19 point 4 did, is what the user saw. Fix: the right panel now
reprojects the PRODUCTION solve's own fit target (`kp_data_prod[t_src]`,
already the exact `cloud_pts` the left panel is rendering that same frame)
through `CAM_2UP`'s DLT, instead of drawing `02_kp2d.npz` directly --
restoring what the panel showed before TASK-19's regression. `02_kp2d.npz`
(`kp2d`) is still loaded and still used for `_smoothed_crop_x0`'s
crop-centring smoothing; it is simply no longer what gets DRAWN. This
reinstates the `cam_mats`/DLT loading TASK-19 point 4 removed (now indexed
by camera NAME, `dlt_cam_idx`, not position, and asserted present rather
than assumed) and the `site_xpos`-through-DLT reprojection idea from before
TASK-18/19 -- but reprojects `kp_data_prod` (the recorded fit target)
rather than a live FK `site_xpos`, since `kp_data_prod[t_src]` is already
exactly what the left panel's `cloud_pts` is for the SAME `t_src`, so no
extra FK evaluation is needed. Per the module's own COORDINATE-FRAME
CAVEAT below, `kp_data_prod` is divided by `shared_scale` before
projecting (verified empirically there: reproduces the observed 2D to a
mean ~8.5 px / max ~24 px). Colours/helpers are UNCHANGED from TASK-19
(`kp_colors.jarvis_kp_colors`, `draw.draw_keypoints`/`draw.draw_leg_chains`,
no `conf` dimming) -- only the SOURCE of `obs_uv` changes.

TASK-24 (fixes the wrap/replayed-tail bug the user reported -- "why is there
an added 10 seconds almost at the end of act 4? there is something that
jumps around 41 seconds and it seems like the replay restarts?"; read this
before touching PLAYBACK_ANCHOR, `t_for_output_frame`, or the "TWO PHASES" /
"PLAYBACK START" sections below, both of which this task supersedes):

MEASURED (in the fully assembled 1530-frame video, before this fix): Act 4
begins at output frame 630 (21.0 s); Phase B/C playback begins at output
frame 870 (29.0 s); the wrap `qpos_prod[920] -> qpos_prod[0]` fired at
output frame 1208 (40.3 s) -- a REAL 2.021 model-unit root translation and
39.3 deg heading change between consecutive rendered frames (typical
frame-to-frame translation elsewhere in the same sequence: ~0.00273, so the
wrap was ~740x a normal step). Everything after the wrap (10.7 s) was
`qpos_prod[0:frame_for_stills]` playing a SECOND time, on top of frames the
video had already shown once during Phase A/early Phase B -- this is the
"added 10 seconds" and "replay restarts" the user saw.

ROOT CAUSE: Phase A always interpolated `qpos_root -> qpos_prod[anchor]`
where `anchor` was `06_stages.npz`'s own `frame_for_stills` (450). To keep
the Phase A -> B boundary (f=239/240) continuous, the pre-TASK-24 playback
was made to *start* at that same anchor (450) -- which forced it to WRAP
`450 -> 920 -> 0 -> 449` to cover the whole 921-frame clip in one pass,
producing both the 39.3 deg jump (at the wrap) and the replayed tail (the
`0..449` stretch playing a second time, after already appearing once via the
wrap's continuation).

FIX: set the playback anchor to source frame 0 instead (`PLAYBACK_ANCHOR`
below) -- NOT by editing `06_stages.npz`'s stored `frame_for_stills` (still
450 there, left untouched; that field is a genuine, separate provenance
fact -- see stage_ik.py's own `frame=frame_for_stills` root_optimization call
-- not a free knob for this act to repoint). With `PLAYBACK_ANCHOR=0`:
  - Phase A now interpolates `qpos_root -> qpos_prod[0]` (not `qpos_prod[450]`).
    Measured (replacing the old 34.79 deg / 3.95 DOF-L2 numbers below, which
    described the anchor=450 pair): `qpos_root -> qpos_prod[0]` rotates
    **11.49 deg** with a **1.026** model-unit translation and a **3.77**
    joint-DOF L2 delta -- still comfortably above the `root_pose_angle_deg
    < 1.0` guard's threshold, so Act 4's "the orienting beat lives here"
    premise still holds (measured, not assumed -- the guard asserts this at
    runtime).
  - Phase B/C playback runs source frames `0 -> 920` MONOTONICALLY (see
    `t_for_output_frame` below) -- no wrap, no hold, no replayed tail. The
    old `WRAP_HOLD_FRAMES`/wrap-arithmetic machinery is DELETED, not left
    unreachable.
  - Phase A's end (f=239, `qpos_prod[0]`/`kp_data_prod[0]`) and Phase B's
    start (f=240, source frame 0) now render the IDENTICAL recorded qpos and
    cloud target -- a real zero-gap match, since both sides are now anchored
    at the same frame, by construction, not by coincidence.

SANITY-CHECKED before rendering (per this task's own "measure before
assuming" mandate): `qpos_prod[0]`'s pose was rendered and read directly --
a normally postured, in-frame standing fly (all six legs articulated, wings
folded, head/antennae markers in place; see the task-24 report for the
image) -- and its production fit residual (0.0230 mm) sits inside the
first-100-frame range (mean 0.0252 mm, 0.0200-0.0340 mm), i.e. an ordinary
fit quality, not an outlier frame that would have been a reason to pick a
different anchor.

ALSO MEASURED, two further consequences of moving Phase A's target from
frame 450 to frame 0 -- one required an actual fix (a), the other is
reported as a known, out-of-scope side effect (b):
  (a) Phase A is no longer close to a pure rotation. `qpos_root` (the OLD,
      pre-TASK-25 snapshot this paragraph originally measured) is a
      translation snapshot solved AT frame 450 (see stage_ik.py's
      `root_optimization(..., frame=frame_for_stills)` call), so it sat
      near `qpos_prod[450]`'s position by construction; it does NOT sit
      near `qpos_prod[0]`'s position (a real ~1.03-model-unit gap -- the fly
      walks during the clip). Phase A's live-recomputed residual ran
      1.041 -> 0.012 mm (was 0.118 -> 0.011 mm pre-TASK-24) -- a real,
      substantially bigger swing, because Phase A now closes a genuine
      translation gap in addition to the rotation it always closed.
      **UPDATED BY TASK-25** (see the TASK-25 section below, which replaces
      this paragraph's `qpos_root` with `qpos_root_f0`, a root_optimization
      solved AT frame 0 instead of 450): the ~1.03-model-unit translation
      gap this paragraph measured shrinks to **0.061 model units**
      (`qpos_root_f0[:3]` vs `qpos_anchor[:3]`) -- rotation (11.49 deg) and
      joint-DOF delta (3.77) are UNCHANGED (both depend only on
      `qpos_anchor` vs. the rest pose's own rotation/DOF, which no root
      solve at any frame ever touches -- see the translation-only guard).
      Phase A's live-recomputed residual now starts at `residual_root_f0`
      (0.099 mm, `06_stages_f0.npz`) rather than 1.041 mm, since the mesh
      once again starts near-coincident with its keypoint target instead of
      ~1 model unit away.
      TASK-24 CLIPPING FIX (found by rendering and looking, not assumed;
      kept unmodified by TASK-25 -- see the TASK-25 section below for
      whether it is still load-bearing):
      the FIRST version of this fix kept Phase A's camera lookat FROZEN at
      `mesh_ctr_final` (`qpos_root`'s own FK centroid) for the whole act,
      unchanged from pre-TASK-24 -- this was safe before because the OLD
      anchor (450) kept the mesh's actual end-of-Phase-A position
      near-coincident with `mesh_ctr_final` by construction. With the NEW
      anchor (0), it is NOT: rendering and reading the frames directly
      showed the mesh mask touching the canvas's LEFT edge from f~=200
      onward, shrinking to a bounding box of x:[0,289] (out of 1920) by
      f=239 -- exactly the "fly walks out of a frozen frame" failure this
      module's own docstring already names as the trigger to stop and fix
      rather than ship (see the CAMERA section and the task-24 report for
      the frame that first surfaced this). FIX: `lookat` now eases
      `mesh_ctr_final` (f=0) -> `mesh_ctr_anchor` (FK of `qpos_anchor`, f=239)
      using the SAME smoothstep progress `p` already driving the qpos SLERP
      -- not a separate schedule, so camera and content never disagree on
      how far along Phase A is. This reproduces `mesh_ctr_final` EXACTLY at
      p=0 (preserving the Act3->Act4 seam, unaffected) while properly
      framing the mesh's true end position by p=1. Re-rendered and
      re-checked after the fix: the mesh mask stays fully inside the canvas
      at every frame sampled (f=0/50/100/150/200/220/230/239), never
      touching an edge. See `_wide_camera_for_aspect`'s own docstring
      ("Distance is LIVE") for the same residual numbers in that context.
  (b) The Phase A->B camera-FORMULA gap (frozen `act3_frozen_camera` vs. the
      live `_wide_camera_for_aspect`, previously measured pre-TASK-24 at
      ~76 px mesh-centre shift / 1.18x area ratio at f=239/240, called out in
      this task's own verification checklist as a "known separate issue")
      changed after the clipping fix above: re-measured post-fix
      (normalizing for task-19's side-by-side panel width so the f=239
      full-1920 canvas and f=240 960-wide left panel are compared on equal
      terms) at ~2 px mesh-centre shift / 1.93x area ratio -- the centre
      shift actually IMPROVED (the eased lookat, ending exactly on the
      mesh's true position, hands off to the live camera from a
      well-centred point), while the area ratio is somewhat higher than the
      old ~1.18x, attributable to the frozen camera's FIXED distance
      (`ACT3_CAM_DISTANCE`=0.93) vs. the live camera's ADAPTIVE
      spread-based distance -- these two distance FORMULAS simply don't
      agree in general, independent of lookat. This remaining gap is
      content-CONTINUOUS (f=239/f=240 render the identical qpos/cloud,
      TASK-24's whole point) but camera-FORMULA-discontinuous, exactly the
      pre-existing, explicitly out-of-scope category this task asked to be
      reported on, not fixed.

TASK-25 (closes the regression the "KNOWN, DELIBERATE LIMITATION" this
section used to describe -- read `act3_align.py`'s own TASK-25 section
first for the full story; this section covers ONLY Act 4's side of it):
previously (pre-TASK-25), Act 3 did NOT make the matching anchor change --
its static mesh, `06_stages.npz`'s `qpos_root`, was a snapshot baked by
`root_optimization(frame=450)` and could not be moved to frame 0 without
re-running that solve, which TASK-24 (a playback bugfix) treated as out of
scope; measured and rendered directly at the time, re-anchoring Act 3's
skeleton target alone to frame 0 left the skeleton ~1.04 model units from
the static mesh at the "converged" hold, visibly never touching it -- worse
than the seam it would have fixed. Act 3 kept reading
`kp_data_prod[frame_for_stills]` (450), and the Act3(f89)->Act4(f0)
KEYPOINT CLOUD/skeleton overlay showed a real, measured ~1.04-model-unit
jump at that exact cut (the mesh handoff was already exactly continuous,
since both acts drew the same frame-450 `qpos_root` regardless of anchor).

TASK-25 removes that limitation by re-running the actual solve instead of
accepting the trade-off: `stage_ik.run_root_f0()` calls the SAME
`compute_stac.root_optimization` `06_stages.npz`'s own `qpos_root` came
from, at frame 0 of the PRODUCTION `kp_data` instead of frame
`frame_for_stills` (450), producing `qpos_root_f0`
(`predictions/06_stages_f0.npz`, a NEW file -- `06_stages.npz` and its own
`qpos_root` are untouched). Act 4's Phase A now interpolates
`qpos_root_f0 -> qpos_prod[anchor]` (this section's own preceding TASK-24
paragraphs, and `_slerp_qpos`'s call site below, both now read
`qpos_root_f0`); Act 3 now draws `qpos_root_f0` as its static mesh and reads
`kp_data_prod[0]` (not `kp_data_prod[frame_for_stills]`) as its skeleton
target -- see act3_align.py's own TASK-25 section. Both acts are once again
anchored at the identical frame (0), so BOTH the mesh AND the keypoint
cloud/skeleton are continuous across the Act3(f89)->Act4(f0) cut, not just
the mesh as before.

This act carries the ORIENTING beat of the whole explainer. Act 3 covered
scale + translation only: `root_optimization` measured 0.0000 deg of
rotation (`TRUNK_OPTIMIZATION_KEYPOINTS` is empty in `configs/anatomy/v1.yaml`,
so its keypoint-fit loss is uniformly zero; see `act3_align.py`'s docstring).
ALL the rotation happens here, in `pose_optimization`. Measured on the
PRODUCTION solve at the playback anchor (TASK-24: `PLAYBACK_ANCHOR=0`, source
frame 0 -- previously `frame_for_stills=450`; asserted at runtime, not just
claimed): the quaternion rotates **11.49 deg** from `qpos_root` to
`qpos_prod[anchor]`, and `qpos[7:]` (joint DOF) moves an L2 norm of **3.77**
(TASK-24: see the TASK-24 section above for how this replaces the pre-TASK-24
34.79 deg / 3.95 measurement taken at the OLD anchor, 450 -- both numbers
describe the same real solve, just at different frames; the SMALLER
1.0595-model-unit / 26.02-deg gap the old "PLAYBACK START" section measured
between `qpos_prod[0]` and `qpos_prod[anchor=450]` is now, with anchor=0,
EXACTLY ZERO by construction -- see TASK-24 above). The first 240 frames of
this act are where the fly visibly swings into its true heading AND its
limbs snap onto the keypoints, together -- not two separate beats.

The rotation is a real, per-frame solve, not a staged pose: root heading
varies over the 921-frame production sequence (`qpos_prod`), so Phase B's
playback shows genuinely solved per-frame rotation.

TWO PHASES (900 frames, 30 fps -> 30 s; TASK-19 merges the old Phase B/Phase
C split into one side-by-side phase spanning the entire playback -- see the
TASK-19 section above):
  0-239   (Phase A) qpos interpolates qpos_root -> qpos_prod[anchor] (the
          production fit at the single `PLAYBACK_ANCHOR` frame -- TASK-24:
          source frame 0, previously `frame_for_stills`=450; see the TASK-24
          section above),
          quaternion SLERPed (`_slerp_qpos`, reused unmodified from
          `act3_align.py` -- it was written there specifically so this act's
          real rotation would have a SLERP, not a lerp, ready for it). The
          keypoint-cloud TARGET is held fixed at `kp_data_prod[anchor]` (the
          production solve's own fit target, already in the scaled/model
          coordinate frame -- TASK-18: replaces the old
          `kp3d_scaled[frame_for_stills]`, this repo's own re-triangulated
          `04_kp3d_filt.npz * shared_scale`); residual is a LIVE recompute
          (`stage_ik.marker_residual_mm`, recomputed every frame, not lerped
          between two endpoints) of the interpolated qpos against that fixed
          target -- since no recorded production frame exists for an
          INTERPOLATED pose, this is the best-available real, per-frame
          number, but it is NOT offset-adjusted (this act never applies the
          production `offsets` to the rendered model's site positions), so it
          is a different quantity from the merged phase's precomputed
          `resid_prod_mm` below -- see TASK-18 note above and the render
          loop's own comments. Rendered full-width (1920x1080), 3D only, no
          real-camera panel -- unchanged by task-19.
  240-899 (Phase B/C, merged by TASK-19) side-by-side for the ENTIRE rest of
          the act, 660 frames: `qpos_prod` (921 frames, the production
          solve) plays back, continuously mapped onto these 660 output
          frames (same `t = round(rel * (T-1) / REL_MAX)` style Act 1 uses
          to span a longer source clip onto fewer output frames), starting
          at `PLAYBACK_ANCHOR` (source frame 0) and running MONOTONICALLY
          through to source frame T-1 -- TASK-24 removed the wrap/hold this
          paragraph previously described (`frame_for_stills -> T-1 -> 0 ->
          frame_for_stills-1`, see the now-superseded "PLAYBACK START"
          section below) entirely: with the anchor at 0, playback already
          starts where the clip itself starts, so there is nothing left to
          wrap around. UNCHANGED by task-19: the frame count (660) and the
          seam at f=240 (now a true zero-gap match, not merely
          camera-matched -- see TASK-24 above). CHANGED by task-19
          (presentation only): every one of these
          660 frames, not just the closing 120, is now drawn side-by-side --
          left is the mesh+cloud render (narrower, 960 px wide, the SAME
          `_wide_camera_for_aspect` live camera the old Phase B rendered at
          full width, coloured per limb chain, JARVIS scheme --
          `kp_colors.jarvis_kp_colors_rgb01`, shared with Act 3); right is
          one real camera's video, cropped and centred the same way Act 1
          crops (`act1_views._smoothed_crop_x0`, reused), overlaid with
          `kp_data_prod[t_src]` (the SAME cloud the left panel just
          rendered) reprojected through this camera's DLT
          (`kp_colors.jarvis_kp_colors`, colour-matched to the left panel;
          TASK-20 reverted TASK-19's brief detour through raw detector 2D,
          which was jittery -- see "COLOUR-MATCHED 2D KEYPOINTS" below). Both
          panels are indexed by the identical `t_src`, never independently --
          see TASK-19 note 4/TASK-20 above for how this is verified. Residual (not
          shown on screen since task-19, still printed) is `resid_prod_mm[t]`
          -- the REAL, offset-adjusted production fit quality for that exact
          recorded frame (`|marker_sites - kp_data|` from the h5, mean over
          50 sites, converted to mm via `/ shared_scale`), not a live
          recompute -- see TASK-18 note above.

COORDINATE-FRAME CAVEAT that made the 2-up correctness non-obvious (measured,
not assumed): `pose_optimization` fits the model against scaled keypoints
(`kp_data_prod`, TASK-18 -- the production solve's own fit target, already in
this frame; see `stage_ik.py` for the analogous, un-offset staged version),
so the model's own `site_xpos` after forward kinematics
live in that SCALED coordinate frame, not the raw arena-mm frame the clip's
DLTs were calibrated in. Reprojecting `site_xpos` through `clip_io.project`
directly would be reprojecting the wrong points. Verified once, empirically,
before writing the render loop: dividing `site_xpos` by `shared_scale` before
projecting reproduces the OBSERVED 2D keypoints (`02_kp2d.npz`) to a mean
~8.5 px, max ~24 px (comparable to Act 2's own 28.5 px worst-case reprojection
tolerance against the same DLTs) -- confirming `site_xpos / shared_scale` is
the right quantity to feed `clip_io.project`, not `site_xpos` itself.

PLAYBACK START (task-17 follow-up -- NOT a camera fix; TASK-18 re-measured
this on the production arrays, see below) -- **SUPERSEDED BY TASK-24**: this
whole section (through "Two honest options" below) is kept as a historical
record of the wrap/hold design TASK-24 REMOVES, not a description of current
behaviour -- `WRAP_HOLD_FRAMES`, the wrap arithmetic, and the "moves the
discontinuity" analysis below no longer exist in the code (see the TASK-24
section above for why: anchoring Phase A's target AND Phase B's playback
start at the SAME frame, 0, removes the original Phase A->B gap this section
describes INSTEAD of moving it elsewhere, so there is no discontinuity left
to wrap around or hold through). Read this only to understand why the wrap
existed in the first place: the first attempt at a seamless
Act3/Act4 cut matched the CAMERA at f=0-239 but left the Phase A->B seam
(f=239/f=240) mismatched, because Phase B started playback at source frame 0
while Phase A always ends its interpolation at `qpos_prod[frame_for_stills]`
(450, the anchor) -- a REAL, measured 1.0595 model-unit translation and
26.02 deg rotation gap between those two states that no camera change can
remove, because it isn't a camera problem: the mesh's actual pose genuinely
differs (TASK-18: this number is measured directly on the PRODUCTION
`qpos_prod`, and is essentially unchanged from the pre-task-18
`06_stages.npz`-based measurement of 1.06 model units / ~25 deg -- both
describe the same clip). Fix: Phase B/C's source-frame mapping now starts at
`frame_for_stills` and WRAPS through the rest of the clip --
`frame_for_stills -> T-1 -> 0 -> frame_for_stills-1` -- so its very first
frame (output f=240) IS `qpos_prod[frame_for_stills]`, bit-identical to what
Phase A just rendered at f=239. All 921 source frames still play, in the
same relative spacing/compression the old `t = round(rel*(T-1)/REL_MAX)`
mapping already used (Act 1's own style for spanning a longer clip onto
fewer output frames) -- only the START OFFSET is new, applied by wrapping
with `% T` after computing the same linear index. The keypoint cloud is
indexed by the identical wrapped `t`, never independently, so it cannot
desynchronise from the mesh.

This moves the discontinuity, it does not delete it: `qpos_prod[T-1]` and
`qpos_prod[0]` (the NEW wrap point, roughly 2/3 of the way through Phase B, at
output frame ~577/578) differ by a REAL 2.02 model-unit translation and
39.3 deg rotation (TASK-18: re-measured on `qpos_prod`, essentially unchanged
from the pre-task-18 2.02/38.5 measurement) -- LARGER than the gap this fix
removes. Measured and LOOKED AT directly before deciding what to do about it
(not assumed small): opening `f00577.png`/`f00578.png` shows the mesh stays
correctly framed on BOTH sides -- because Phase B's camera is the
live/adaptive one on both sides of this internal seam (unlike the
frozen-vs-live mismatch the Act3/Act4 cut had), `_wide_camera_for_aspect`
re-centres on whatever content is on screen every frame regardless of how far
the mesh actually moved, so the wrap does NOT reproduce the earlier
clipped-frame failure (that came from a camera CEASING to track, not from a
large qpos jump per se) -- but it IS a real, visible, sudden reorientation
(the fly is shown almost top-down at f=577, side-on at f=578).

Two honest options were weighed: (a) shorten playback to end at source frame
T-1 rather than wrap, which would drop source frames 0..frame_for_stills-1
(nearly HALF of the real 921-frame trajectory) from the video entirely; (b)
hold briefly on source frame T-1 before continuing from source frame 0.
Chose (b), `WRAP_HOLD_FRAMES=15` (0.5 s @ 30 fps): it keeps ALL 921 source
frames somewhere in the output (consistent with "Phases B/C play back 921
frames of REAL fly motion" being the whole point of this section), invents
nothing (the held frames are `qpos_prod[T-1]` itself, already-real recorded
data, just shown for longer than its one-slot "fair share" under the linear
compression), and gives the eye a beat to register a deliberate pause-then-
resume rather than reading the snap as a glitch mid-motion. The render loop
below implements this by inserting `WRAP_HOLD_FRAMES` extra logical slots at
the wrap point in the same rel->logical->source-frame mapping, rather than
changing the total frame budget, spacing scheme, or anything upstream of it.

--- END of the TASK-24-superseded "PLAYBACK START" record above. ---

CAMERA (mesh panels): reuses `act3_align._wide_camera` (never the model's
`hero` camera, which frames the mesh only and drops the far-away keypoint
cloud out of view) for lookat/distance construction; azimuth/elevation are
FIXED at `act3_align.AZ_START`/`ELEV` (120/-20 deg) for the whole act,
continuing Act 3's own fixed values so the two acts don't visually "jump"
cameras at the cut, and -- as in Act 3 -- so a moving VIEWER azimuth is never
misread as the mesh itself rotating, which matters even more here since the
mesh really DOES rotate 11.49 deg on its own (TASK-24: measured `qpos_root`
-> `qpos_prod[anchor]` at the current `PLAYBACK_ANCHOR=0`; the pre-TASK-24
anchor, 450, measured 34.79 deg for the same `qpos_root -> qpos_prod[anchor]`
pair -- both are real measurements of the same solve, just at different
anchor frames; see the TASK-24 section above).

Distance is LIVE here (recomputed every frame), a DELIBERATE difference from
Act 3's post-review fix (checked explicitly, not assumed -- see the task-11
review this act inherited: a live per-frame distance there was CANCELLING an
intentional cloud-size reveal, since the camera zoomed in exactly as fast as
the cloud shrank). Act 4 has no analogous reveal: Phase A was, pre-TASK-24,
close to a pure rotation (mesh/cloud stayed near-coincident throughout --
live-recomputed residual 0.118->0.011 mm at the old anchor, 450 -- so live
vs. frozen spread was nearly identical there anyway). **TASK-24 UPDATE**:
with `PLAYBACK_ANCHOR=0`, Phase A now ALSO closes a real ~1.03-model-unit
translation gap (`qpos_root`, a snapshot solved at frame 450, does not sit
near `qpos_prod[0]`'s position the way it sat near `qpos_prod[450]`'s) --
measured live-recomputed residual is now **1.041 -> 0.012 mm** over the same
240 frames, a much bigger swing than the pre-TASK-24 number, because Phase A
is no longer purely rotational: it now visibly repositions the mesh too.
**TASK-25 UPDATE**: this ~1.03-model-unit gap (and the 1.041 mm residual
start) is measured against the OLD, pre-TASK-25 `qpos_root` (frame 450);
with `qpos_root_f0` (frame 0), the gap shrinks to 0.061 model units and the
residual starts at 0.099 mm -- see the TASK-25 section above for the full
number set. The clipping fix below is kept regardless (harmless when the
gap it was sized for shrinks; see that section for whether it remains
load-bearing at the new, smaller gap).
Rendered and checked directly (not assumed from the residual alone): the
FIRST version of this fix kept Phase A's camera `lookat` frozen at
`mesh_ctr_final` for the whole act (unchanged from pre-TASK-24) and this
~1-unit world-space displacement turned out NOT to run harmlessly along the
viewing axis -- it left the mesh mask touching the canvas's LEFT edge from
f~=200 onward, clipped to a x:[0,289] bounding box by f=239. See the
"TASK-24 CLIPPING FIX" paragraph above (and `_wide_camera_for_aspect`'s own
docstring, below) for the fix: `lookat` now eases `mesh_ctr_final` (f=0) ->
`mesh_ctr_anchor` (f=239) via the qpos SLERP's own progress `p`. Re-checked
after the fix: the mesh mask stays fully inside the canvas at every frame
sampled (f=0/50/100/150/200/220/230/239), never touching an edge. Phases B/C
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
at the time (pre-TASK-24, anchor=`frame_for_stills`=450), `qpos_prod[anchor]`
(what f=239 rendered) and `qpos_prod[0]` (playback's own first real frame,
what f=240 rendered) differed by a REAL 1.0595 model-unit root translation
and 26.02 deg of rotation (TASK-18: measured on `qpos_prod`) -- Phase A
solved toward one specific mid-clip frame while Phase B immediately started
playing back the trajectory from t=0, a genuine content discontinuity in the
DATA, not an artifact of this fix.

**TASK-24 UPDATE**: with `PLAYBACK_ANCHOR=0`, `qpos_prod[anchor]` IS
`qpos_prod[0]` -- the content discontinuity this paragraph measured is now
EXACTLY ZERO, not merely smaller (see the TASK-24 section above). This does
NOT reopen the eased-camera question, though: the decision below (resume the
live camera unmodified, no easing) is kept as-is, because a SEPARATE,
camera-FORMULA discontinuity remains at this exact cut regardless of content
-- Phase A renders with the FROZEN `act3_frozen_camera` (matching Act 3's own
camera exactly) while Phase B renders with the LIVE `_wide_camera_for_aspect`
formula, and those two formulas do not, in general, agree even when pointed
at the identical mesh/cloud. This frozen-vs-live camera FORMULA gap
(previously measured, pre-TASK-24, at ~76 px mesh-centre shift / 1.18x area
ratio at f=239/240) is a KNOWN, SEPARATE issue, explicitly out of scope for
this task -- re-measured after TASK-24 to confirm it is still just that (see
the task-24 report). The rest of this section's reasoning (why an eased
hand-off was rejected in favour of an abrupt live-camera resume) is otherwise
UNCHANGED: even with zero content discontinuity, an ease would still spend
its first several frames looking at Phase B's already-adaptive content
through Phase A's frozen lookat/distance, which is exactly what previously
produced the clipped-frame failure below. A camera that is still mostly at
the FROZEN value
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

The side-by-side's left panel (240-899, task-19 -- previously only the
closing 120-frame Phase C) is rendered at a narrower (960x1080, not
1920x1080) aspect than Acts 3/A; `_wide_camera_for_aspect` scales the
distance by `1/aspect` for aspect < 1 so the same mesh+cloud content that fit
the wide 16:9 frame doesn't get clipped left/right in the narrower panel
(derived once from the pinhole geometry, not tuned by eye: at the base
2.6x-spread distance, the binding constraint is vertical for aspect >= 1 --
horizontal FOV is always the wider of the two there -- but flips to
horizontal once aspect < 1, requiring distance to grow by 1/aspect to keep
the same real-world width in frame).

COLOUR-MATCHED 2D KEYPOINTS (task-19, replaces the old "2-UP MARKER COLOURS"
cyan-observed/green-fit design; TASK-20 changes the SOURCE of the right
panel's points back to a reprojection -- see below -- but keeps this
colour-matching design): the right panel does not draw a flat
fit-vs-observed two-colour contrast. It draws `kp_data_prod[t_src]`
reprojected through `CAM_2UP`'s DLT (TASK-20; see the render loop's own
comments) with the SAME per-limb-chain JARVIS colours
(`kp_colors.jarvis_kp_colors`, BGR-by-NAME) the left panel's mesh/cloud uses
(`kp_colors.jarvis_kp_colors_rgb01`, RGB/0-1-by-NAME, same underlying colour
table) -- both panels are keyed by keypoint NAME, so they agree regardless
of which array order either side happens to be in (CLAUDE.md's
keypoint-order bug class). Drawn with the shared
`draw.draw_keypoints`/`draw.draw_leg_chains` helpers every other act already
uses -- the old `_dots`/`_chain` single-colour analogues from before
task-19 are not reinstated (task-19's colour-matched design is kept;
only the point SOURCE reverts).

EXPECTATION: by f=239 the mesh's limbs lie along the keypoint chains and the
body has visibly rotated (and, since TASK-24, repositioned) from Act 3's
ending heading/position; residual (live-recomputed, no offset applied to
the rendered sites, printed to the console -- not shown on screen since
task-19) reads ~0.012 mm (TASK-24: starting from ~1.041 mm at f=0, not the
pre-TASK-24 ~0.118 mm -- see the "Distance is LIVE" section above for why
the starting gap grew). Through 240-899 (all
side-by-side since task-19), `resid_prod_mm` (the real, offset-adjusted
production fit quality, still printed) stays in its measured ~0.014-0.041 mm
range, tarsal tips TRACK the observed keypoints through leg swing -- the mesh
foot stays ON the marker as the leg moves, not merely near it -- and the
right panel's reprojected production keypoints (TASK-20) move through the
SAME leg swing, in the SAME per-limb colours as the left panel's mesh/cloud,
with one frame index (`t_src`) driving both panels and both drawing the
SAME underlying `kp_data_prod[t_src]` (left in 3D, right reprojected to 2D)
-- so the two panels cannot show genuinely different content, only a
different projection of the identical fit.
FALSIFICATION: tips detaching from markers during swing means `pose_
optimization` did not converge, or the wrong qpos frame is being drawn; a
tumbling/flipping body during 0-239 means the quaternion was lerped instead
of SLERPed; the right panel showing the fly at a visibly DIFFERENT moment
than the left panel (e.g. legs in a different swing phase) means `t_src`
desynced between the two panels -- see TASK-19 note 4 above for the
single-variable construction that rules this out; the reprojected points
landing off the fly (rather than merely off the exact marker, which the
~8.5 px mean/~24 px max reprojection error already explains) would mean the
TASK-20 `/ shared_scale` coordinate-frame conversion is missing or wrong.

Colours come from `kp_colors.jarvis_kp_colors_rgb01` (JARVIS per-limb-chain
scheme, Change 2) for the mesh+cloud panels, and `kp_colors.jarvis_kp_colors`
(the same underlying colour table, BGR/0-255) for the right panel's
reprojected keypoints since task-19 (source changed by TASK-20, colours
unchanged) -- see "COLOUR-MATCHED 2D KEYPOINTS" above.
`viz.core.colors.PALETTE` is still used for the mesh's own grey
(`PALETTE["mesh"]`); its fit=green/observed=cyan convention is RETIRED for
this act's right panel (task-19). Wing visibility (Change 1):
`set_mesh_rgba`/`capture_geom_alpha` (stage_ik.py) keep originally-invisible
geoms (the wings' `*_inertial` boxes) at alpha=0 across every `geom_rgba`
mutation.

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
import h5py
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
    jarvis_kp_colors, jarvis_kp_colors_rgb01, jarvis_skeleton_edges,
)
from scripts.viz.ik_explainer.acts.act3_align import (                 # noqa: E402
    _slerp_qpos, _wide_camera, _smoothstep, AZ_START, ELEV, BONE_RADIUS_FRACTION,
    act3_frozen_camera, ACT3_CAM_DISTANCE,
)
from scripts.viz.ik_explainer.acts.act1_views import _smoothed_crop_x0  # noqa: E402
from viz.core.colors import PALETTE                                    # noqa: E402

# --- canvas / timeline ------------------------------------------------------
CANVAS_W, CANVAS_H = 1920, 1080
N_OUT = 900
PHASE_A_END = 239                 # inclusive: qpos_root -> qpos_prod[anchor], one frame
# TASK-19: the old PHASE_B_END=779 split (full-width playback 240-779, then a
# closing 780-899 2-up) is retired -- 240..899 inclusive is now ONE merged,
# side-by-side phase (see module docstring's TASK-19 section); there is no
# longer an internal boundary within it to name.

# TASK-24 (see module docstring's TASK-24 section for the full story): the
# playback ANCHOR -- the single production-solve frame Phase A interpolates
# TO, and the frame Phase B/C playback STARTS from -- is now hardcoded to
# source frame 0, deliberately overriding `06_stages.npz`'s own
# `frame_for_stills` (450, left un-edited in that file -- see below and the
# TASK-24 docstring section for why). With the anchor at 0, playback's
# `rel -> source index` mapping (below) runs 0 -> T-1 monotonically with NO
# wrap and NO hold: the two are no longer needed at all, and are removed
# (not left unreachable) from this module.
PLAYBACK_ANCHOR = 0

XML_PATH = _REPO / "models" / "fruitfly_v1" / "fruitfly_v1_free.xml"

# Continuous t-mapping for the whole merged 240-899 side-by-side phase, so it
# never skips or repeats time: rel=0 at f=240 maps to source frame
# PLAYBACK_ANCHOR (0), rel=REL_MAX at f=899 maps to the clip's last frame
# (T-1). TASK-24: since the anchor IS 0, this mapping needs no offset/wrap
# any more -- rel maps directly onto the source index.
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
    Phase A was, pre-TASK-24, close to a pure ROTATION (residual stayed tiny
    -- 0.118->0.011 mm, live-recomputed, TASK-18 -- throughout, so mesh/cloud
    spread was already near-constant); TASK-24 makes Phase A close a real
    ~1.03-model-unit translation too (residual now 1.041->0.012 mm -- see the
    module docstring's "Distance is LIVE" section), so `cam_spread` is no
    longer near-constant across Phase A the way it was pre-TASK-24 -- this
    function's LIVE-recompute design (never frozen) already accommodates that
    correctly; nothing here needed to change, only this comment. Live-vs-
    frozen distance still makes little visible difference within any single
    frame (both would size markers similarly for that instant); it is the
    across-frame LIVE recompute, already present, that keeps up with the
    now-larger swing, and Phases
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
    (Acts 3/4's 1920x1080 panels, unaffected). Once aspect < 1 (the
    side-by-side's 960-wide left panel, task-19: now used for the WHOLE
    240-899 range, not just the old closing Phase C), horizontal binds
    instead and distance must grow by 1/aspect to keep the same real-world
    width in frame (pinhole relation tan(horiz_half) = aspect * tan(vert_half),
    not tuned by eye).
    """
    _, live_spread = _wide_camera(mesh_ctr, cloud_pts, model_extent, azimuth, cam_spread=1.0)
    cam, live_spread = _wide_camera(mesh_ctr, cloud_pts, model_extent, azimuth, cam_spread=live_spread)
    if aspect < 1.0:
        cam.distance = cam.distance / aspect
    return cam, live_spread


def render_act4(clip: str = clip_io.CLIP_DEFAULT, start_frame: int = 0) -> Path:
    """`start_frame` resumes a render that already wrote frames 0..start_frame-1
    (e.g. after an interrupted run) without recomputing or re-rendering them --
    all setup above this point is cheap (numpy/IO), only the per-frame MuJoCo
    render loop is skipped ahead."""
    dirs = clip_io.out_dirs(clip)

    with np.load(dirs["predictions"] / "06_stages.npz", allow_pickle=True) as z:
        stage_names = [str(s) for s in z["stage_names"]]
        kp_names = [str(n) for n in z["kp_names"]]
        frame_for_stills = int(z["frame_for_stills"])
        shared_scale = float(z["shared_scale"])

    # TASK-25: Phase A's starting pose now comes from `06_stages_f0.npz`
    # (`stage_ik.run_root_f0`) -- a root_optimization solved AT source frame
    # 0 against the PRODUCTION kp_data, NOT `06_stages.npz`'s own `qpos_root`
    # (a frame-450 snapshot). See module docstring's TASK-25 section (and
    # act3_align.py's own TASK-25 section) for why: this closes the Act3->
    # Act4 keypoint-cloud jump TASK-24's "KNOWN, DELIBERATE LIMITATION"
    # section reported but did not fix. `06_stages.npz` itself is untouched.
    with np.load(dirs["predictions"] / "06_stages_f0.npz", allow_pickle=True) as zf0:
        qpos_root_f0 = np.asarray(zf0["qpos_root_f0"], np.float64)
        f0_frame = int(zf0["frame"])
        kp_names_f0 = [str(n) for n in zf0["kp_names"]]
    if f0_frame != 0:
        raise ValueError(
            f"06_stages_f0.npz frame={f0_frame} -- expected 0 (PLAYBACK_ANCHOR "
            "and this snapshot must agree, or Phase A's starting pose no "
            "longer matches what Act 3's own mesh is drawn at).")
    if kp_names_f0 != kp_names:
        raise ValueError(
            "06_stages_f0.npz kp_names != 06_stages.npz kp_names -- refusing "
            "to mix keypoint orders (CLAUDE.md's keypoint-order bug class).")

    assert stage_names == ["default", "scaled", "root", "pose"], (
        f"unexpected 06_stages.npz stage_names order: {stage_names}")

    # TASK-18: playback, the keypoint cloud/skeleton, and the on-screen
    # residual now all come from the PRODUCTION solve
    # (`ik_production/stac_ik_full.h5`) -- the full staged run this repo's own
    # `06_stages.npz`/`stage_ik.py` deliberately never ran `offset_optimization`
    # for (see stage_ik.py's module docstring); the production h5 did, so its
    # `offsets` (marker offsets, mean |0.015| max |0.26| mm) are real and
    # fitted, unlike anything in `06_stages.npz`. See module docstring's
    # TASK-18 section.
    h5_path = Path(clip) / "ik_production" / "stac_ik_full.h5"
    with h5py.File(h5_path, "r") as hf:
        qpos_prod = np.asarray(hf["qpos"][:], np.float64)              # (T,93)
        kp_data_prod = np.asarray(hf["kp_data"][:], np.float64).reshape(-1, 50, 3)
        marker_sites_prod = np.asarray(hf["marker_sites"][:], np.float64)  # (T,50,3)
        offsets_prod = np.asarray(hf["offsets"][:], np.float64)        # (50,3)
        kp_names_prod = [n.decode() if isinstance(n, bytes) else str(n)
                          for n in hf["kp_names"][:]]

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

    T = qpos_prod.shape[0]
    # TASK-24: the playback anchor is now PLAYBACK_ANCHOR (source frame 0),
    # NOT 06_stages.npz's own `frame_for_stills` (450, read above and still
    # printed below for provenance -- that file is deliberately left
    # unedited). Anchoring Phase A's interpolation target AND Phase B/C's
    # playback start at the SAME frame 0 is what removes the wrap: see
    # module docstring's TASK-24 section.
    anchor = PLAYBACK_ANCHOR
    if not (0 <= anchor < T):
        raise ValueError(f"PLAYBACK_ANCHOR={anchor} out of range for production T={T}")
    qpos_anchor = qpos_prod[anchor]
    print(f"[act4] TASK-24: playback anchor = PLAYBACK_ANCHOR={anchor} (source frame 0) -- "
          f"06_stages.npz's own frame_for_stills={frame_for_stills} is UNRELATED to playback "
          f"since TASK-24 (kept only as Act 3's mesh-solve provenance; see module docstring)")

    # Per-frame production fit residual (mm): |marker_sites - kp_data|,
    # mean over the 50 sites -- the REAL, offset-adjusted fit quality the
    # production solve achieved, replacing this act's old `residual_root`/
    # `residual_pose` (06_stages.npz, never offset-fit) as the on-screen
    # number for Phase B/C. See module docstring's TASK-18 section.
    resid_prod_mm = (np.linalg.norm(marker_sites_prod - kp_data_prod, axis=-1).mean(axis=-1)
                      / shared_scale)                                  # (T,)
    print(f"[act4] TASK-18: production fit ({h5_path}) per-frame residual (mm): "
          f"min={resid_prod_mm.min():.4f} max={resid_prod_mm.max():.4f} "
          f"mean={resid_prod_mm.mean():.4f} -- offsets mean|abs|={np.abs(offsets_prod).mean():.4f} "
          f"max|abs|={np.abs(offsets_prod).max():.4f}")

    # Measured-fact guard: pose_optimization must have rotated non-trivially
    # (module docstring's whole premise). If a future re-solve makes this
    # stage stop rotating (e.g. TRUNK_OPTIMIZATION_KEYPOINTS becomes
    # non-empty and root_optimization starts doing the orienting instead),
    # this act's "orienting beat lives here" story would be wrong -- fail
    # loudly rather than silently render a stale claim.
    root_pose_angle_deg = _quat_angle_deg(qpos_root_f0[3:7], qpos_anchor[3:7])
    if root_pose_angle_deg < 1.0:
        raise ValueError(
            f"root->anchor rotation is only {root_pose_angle_deg:.3f} deg -- "
            "Act 4's premise (pose_optimization performs the real orienting) "
            "no longer matches the recorded snapshots; refusing to render."
        )
    joint_dof_delta = float(np.linalg.norm(qpos_anchor[7:] - qpos_root_f0[7:]))
    print(f"[act4] root->anchor rotation: {root_pose_angle_deg:.2f} deg, "
          f"joint-DOF L2 delta: {joint_dof_delta:.2f}")

    with np.load(dirs["predictions"] / "02_kp2d.npz", allow_pickle=True) as z:
        kp2d = np.asarray(z["kp2d"], np.float64)          # (T,C,50,2) full-frame px
        cam_names = [str(c) for c in z["cam_names"]]
        kp2d_names = [str(n) for n in z["kp_names"]]
    if kp2d_names != kp_names:
        raise ValueError("02_kp2d.npz kp_names != 06_stages.npz kp_names.")
    if kp2d.shape[0] != T:
        raise ValueError(f"02_kp2d.npz has {kp2d.shape[0]} frames, production qpos has {T}")
    if CAM_2UP not in cam_names:
        raise ValueError(f"{CAM_2UP} not in 02_kp2d.npz cam_names {cam_names}")
    cam2up_idx = cam_names.index(CAM_2UP)

    # TASK-20 (fixes the task-19 regression, see module docstring's TASK-20
    # section): the right panel no longer draws the detector's raw, jittery
    # `02_kp2d.npz` points -- it reprojects the PRODUCTION solve's own fit
    # target (`kp_data_prod`, already loaded above) through this camera's DLT
    # instead, restoring what the panel showed before task-19. `kp2d` itself
    # is still loaded and used above/below for the crop-centring smoothing
    # only (`_smoothed_crop_x0`) -- it is no longer what gets DRAWN.
    cam_mats, dlt_cam_names = clip_io.load_dlt(str(Path(clip) / "calibration"))
    if CAM_2UP not in dlt_cam_names:
        raise ValueError(f"{CAM_2UP} not in calibration cam_names {dlt_cam_names}")
    dlt_cam_idx = dlt_cam_names.index(CAM_2UP)
    cam_mat_2up = cam_mats[dlt_cam_idx:dlt_cam_idx + 1]   # (1,4,3), keep the leading cam axis

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
    # TASK-19 (colours unchanged by TASK-20's source swap): the right
    # panel's 2D overlay uses the SAME underlying colour table, in cv2's
    # BGR/0-255 convention (`draw.draw_keypoints`/`draw.draw_leg_chains`'s
    # `kp_colors` argument), keyed by NAME so it matches the left panel's
    # per-limb colours regardless of array order.
    kp_colors_bgr = jarvis_kp_colors(kp_names)
    # Skeleton bones (task-14 round 4, "skeleton on Act 4 too"): same
    # construction as act3_align.py -- data/fly50.json's 44 edges mapped onto
    # THIS array's own kp_names order by name, coloured per JARVIS's own
    # `colors[line[1]]` (stop-node) convention.
    skeleton_edges = [
        (a, b, (bg[2] / 255.0, bg[1] / 255.0, bg[0] / 255.0))
        for a, b, bg in jarvis_skeleton_edges(kp_names)
    ]

    # TASK-18: Phase A's fixed target cloud is the production fit's own
    # `kp_data` at the anchor frame (already in the scaled/model coordinate
    # frame the production solve fit against) -- replaces the old
    # `kp3d_scaled[frame_for_stills]` (04_kp3d_filt.npz * shared_scale).
    target_a = kp_data_prod[anchor]

    # Task-17 ("seamless Act3->Act4 cut"): the mesh's own final centroid,
    # computed EXACTLY the way act3_align.py computes `mesh_ctr_final` --
    # FK against the SAME `qpos_root_f0` (TASK-25: was the old `qpos_root`),
    # averaged over the SAME `body_site_idxs` -- so Phase A's camera lookat
    # (below) is the IDENTICAL number Act 3's last frame used, not a
    # re-derived approximation of it. See module docstring's CAMERA and
    # TASK-25 sections.
    d_root_ref = mujoco.MjData(mj_model)
    d_root_ref.qpos[:] = qpos_root_f0
    mujoco.mj_forward(mj_model, d_root_ref)
    mesh_ctr_final = np.asarray(d_root_ref.site_xpos[body_site_idxs]).mean(axis=0)
    print(f"[act4] task-17: Phase A frozen camera distance={ACT3_CAM_DISTANCE} "
          f"azimuth/elevation={AZ_START}/{ELEV} lookat=mesh_ctr_final={mesh_ctr_final} "
          f"(identical formula to act3_align.py's own mesh_ctr_final -- shared "
          f"via act3_frozen_camera); Phase B resumes the live camera "
          f"unmodified at f={PHASE_A_END + 1} (see module docstring's CAMERA "
          f"section for why an eased hand-off was tried and reverted)")

    # TASK-24 (see module docstring's TASK-24 CLIPPING FIX section): with
    # PLAYBACK_ANCHOR=0, Phase A's mesh travels ~1.03 model units from
    # qpos_root's position toward qpos_anchor's -- a REAL, measured,
    # RENDERED consequence (not assumed): keeping Phase A's camera lookat
    # frozen at `mesh_ctr_final` (qpos_root's own centroid) for the WHOLE
    # act, as task-17 originally set it up, leaves the mesh clipped against
    # the canvas's LEFT edge for the closing ~40 frames (measured: mesh mask
    # touches x=0 from f~=200 onward, bounding box shrinking to x:[0,289] by
    # f=239) -- the exact "fly walks out of a frozen frame" failure this
    # module's own docstring already names as the trigger to stop and fix
    # rather than ship. `mesh_ctr_anchor` (FK of `qpos_anchor`, the mesh's
    # OWN true position at Phase A's end) is computed here so the render
    # loop can EASE the lookat from `mesh_ctr_final` (f=0, matching Act 3
    # exactly -- preserves the Act3->Act4 seam) to `mesh_ctr_anchor` (f=239,
    # properly centred on where the mesh actually ends up) using the SAME
    # smoothstep progress `p` that already drives the qpos interpolation --
    # not a separately-tuned ease curve, so the camera and the mesh always
    # agree on "how far along" Phase A is.
    d_anchor_ref = mujoco.MjData(mj_model)
    d_anchor_ref.qpos[:] = qpos_anchor
    mujoco.mj_forward(mj_model, d_anchor_ref)
    mesh_ctr_anchor = np.asarray(d_anchor_ref.site_xpos[body_site_idxs]).mean(axis=0)
    print(f"[act4] TASK-24: Phase A lookat eases mesh_ctr_final={mesh_ctr_final} "
          f"(f=0, matches Act 3 exactly) -> mesh_ctr_anchor={mesh_ctr_anchor} "
          f"(f={PHASE_A_END}, the mesh's own FK centroid at qpos_anchor) via the "
          f"same progress `p` as the qpos SLERP -- distance/azimuth/elevation "
          f"stay the fixed {ACT3_CAM_DISTANCE}/{AZ_START}/{ELEV} throughout, "
          f"only lookat eases; gap={float(np.linalg.norm(mesh_ctr_anchor - mesh_ctr_final)):.4f} "
          f"model units (this is why the frozen lookat alone clipped the mesh)")

    # --- side-by-side prerequisites: smoothed crop centring + real video ---
    # frames (task-19: needed for the WHOLE 240-899 merged phase now, not
    # just the old closing Phase C).
    x0_full = _smoothed_crop_x0(kp2d[:, cam2up_idx], SRC_FRAME_W)   # (T,)

    # TASK-24 (replaces the old "Task-17 follow-up" wrap/hold entirely -- see
    # module docstring's TASK-24 section for the bug report and root cause):
    # playback now starts at PLAYBACK_ANCHOR (source frame 0) and runs
    # MONOTONICALLY through to source frame T-1, with NO wrap and NO hold.
    # This is possible only because Phase A's interpolation target is ALSO
    # PLAYBACK_ANCHOR (`qpos_anchor = qpos_prod[PLAYBACK_ANCHOR]` above) --
    # Phase A's last frame (f=239) and Phase B's first frame (f=240) now
    # render the IDENTICAL recorded qpos/cloud (`qpos_prod[0]`/
    # `kp_data_prod[0]`), a real zero-gap match, not just a close one:
    # previously (anchor=frame_for_stills=450) this was a REAL, measured
    # 1.0595 model-unit / 26.02 deg gap that motivated the old wrap in the
    # first place, and wrapping around to close THAT gap created a WORSE one
    # (2.02 model units / 39.3 deg) where source frame T-1 met source frame 0
    # roughly 2/3 of the way through playback -- this was the jump/"restart"
    # the user reported. Anchoring both ends at 0 removes the original gap
    # by construction instead of moving it elsewhere, so there is no
    # remaining discontinuity to wrap around or hold through.
    def t_for_output_frame(f: int) -> int:
        """Direct (unwrapped) source-frame index for output frame `f` in the
        merged 240-899 side-by-side phase: source frame PLAYBACK_ANCHOR (0)
        at f=240, source frame T-1 at f=899, monotonically increasing in
        between -- the same `rel -> round(rel * (T-1) / REL_MAX)` style
        Act 1 uses for compressing a longer source clip onto fewer output
        frames. All 921 source frames still appear somewhere in the output
        (nearby output frames may round to the same source index under this
        660-into-921 compression, but the index never repeats a frame it
        already skipped PAST, and never goes backward)."""
        rel = f - (PHASE_A_END + 1)
        return int(round(rel * (T - 1) / REL_MAX))

    t_playback = np.array([t_for_output_frame(f) for f in range(PHASE_A_END + 1, N_OUT)])
    assert np.all(np.diff(t_playback) >= 0), (
        "playback source-frame index must be non-decreasing -- any decrease "
        "would mean a wrap has crept back in (TASK-24 removed it deliberately)")
    assert t_playback[0] == PLAYBACK_ANCHOR == 0, (
        f"Phase B must start exactly at PLAYBACK_ANCHOR (0) so it picks up "
        f"exactly where Phase A's qpos_prod[anchor] left off; got {t_playback[0]}")
    assert t_playback[-1] == T - 1, (
        f"Phase B/C must play through to the clip's last source frame "
        f"(T-1={T - 1}); got {t_playback[-1]}")
    print(f"[act4] TASK-24: Phase B/C playback runs source frame "
          f"{t_playback[0]} -> {t_playback[-1]} monotonically over "
          f"{len(t_playback)} output frames -- no wrap, no hold, no replayed "
          f"tail; Phase A's last frame and Phase B's first frame render the "
          f"identical qpos_prod[0]/kp_data_prod[0] (zero-gap handoff)")

    # TASK-19: video is now preloaded for the ENTIRE merged side-by-side phase
    # (all 660 frames of `t_playback`, f=240..899) rather than just the old
    # closing 120-frame Phase C -- the side-by-side spans the whole rest of
    # the act now, so its right panel needs a real frame for every one of
    # those output frames.
    # DISPLAY SOURCE: `clip_io.video_path` defaults to the brightness/
    # contrast-lifted `<clip>/enhanced/` copy (presentation-only -- the
    # overlaid keypoints are `kp_data_prod` reprojected, from the production
    # solve fit against the RAW videos; nothing is re-derived here; see
    # clip_io.py's module docstring).
    video_frames = clip_io.read_frames(clip_io.video_path(clip, CAM_2UP), t_playback)
    t_to_video_row = {}
    for row, t in enumerate(t_playback):
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
                qpos = _slerp_qpos(qpos_root_f0, qpos_anchor, p)
                cloud_pts = target_a
                t_src = anchor
            else:
                # TASK-19: 240-899 is now ONE merged, side-by-side phase
                # ("BC") -- see module docstring's TASK-19 section. `t_src`
                # is the SAME single value that drives both panels below
                # (left: qpos/cloud_pts; right: video row/observed 2D), so
                # the two panels cannot desynchronise.
                phase = "BC"
                t_src = int(t_for_output_frame(f))
                qpos = qpos_prod[t_src]
                cloud_pts = kp_data_prod[t_src]

            d.qpos[:] = qpos
            mujoco.mj_forward(mj_model, d)
            sites = np.asarray(d.site_xpos)[body_site_idxs]
            mesh_ctr = sites.mean(axis=0)
            if phase == "A":
                # No recorded production frame exists for an INTERPOLATED
                # pose, so this stays a live FK recompute against the fixed
                # target (unchanged mechanism from before task-18, target
                # swapped to `target_a` == `kp_data_prod[anchor]`) -- a real,
                # per-frame number, not lerped between two endpoints, but NOT
                # offset-adjusted (this act never applies `offsets_prod` to
                # the rendered model's site positions -- see module
                # docstring's TASK-18 section), so it is not directly
                # comparable to the merged phase's precomputed
                # `resid_prod_mm` below.
                resid = marker_residual_mm(mj_model, d, cloud_pts, body_site_idxs)
            else:
                # TASK-18: the REAL, offset-adjusted production residual for
                # this exact recorded frame, not a live recompute -- see
                # module docstring's TASK-18 section.
                resid = float(resid_prod_mm[t_src])
            # TASK-19: `resid` (and, for phase A, the rotation/joint-DOF
            # progress) is no longer drawn on screen -- printed to the
            # console instead so the numbers this act measures are not lost,
            # only their on-screen caption. See module docstring's TASK-19
            # section.
            if (f + 1) % 25 == 0 or f == start_frame:
                # TASK-24: t_src now truthfully counts the SOURCE frame
                # (0..T-1==920, monotonic, no wrap) -- print it against T-1
                # so this progress line doubles as the "source frame N/920"
                # counter (no wrap/hold artifacts left to describe).
                print(f"[act4] wrote frame {f} ({phase}) t_src={t_src}/{T - 1} "
                      f"residual={resid:.4f} mm ({time.time() - t0:.1f}s elapsed)",
                      flush=True)

            if phase == "A":
                aspect = CANVAS_W / CANVAS_H
                cam, spread = _wide_camera_for_aspect(
                    mesh_ctr, cloud_pts, mj_model.stat.extent, AZ_START, aspect)
                # Task-17: Phase A no longer uses the live camera computed
                # above for FRAMING (only `spread`, still live, for marker/
                # bone sizing) -- it renders with `act3_frozen_camera`, the
                # SAME distance/azimuth/elevation Act 3's last frame used, so
                # the Act3->Act4 cut does not jump. The merged BC phase
                # resumes the live `cam` (computed above) unmodified -- see
                # module docstring's CAMERA section for why an eased (rather
                # than immediate) hand-off was tried and reverted AT THAT
                # SEAM: measured directly, it left the mesh clipped against
                # the frame's left edge for a visible stretch of playback.
                #
                # TASK-24 CLIPPING FIX (a DIFFERENT eased hand-off than the
                # one above -- this one is INTERNAL to Phase A, not at the
                # Phase A/B seam): `lookat` now eases mesh_ctr_final (f=0) ->
                # mesh_ctr_anchor (f=239) via the SAME `p` already driving
                # the qpos SLERP, rather than staying frozen at
                # `mesh_ctr_final` for all 240 frames. Measured, rendered,
                # and required: with PLAYBACK_ANCHOR=0, the mesh's own true
                # position at Phase A's end (`qpos_anchor`) is ~1.03 model
                # units from `mesh_ctr_final` (unlike the pre-TASK-24 anchor,
                # 450, which was near-coincident with it) -- a frozen lookat
                # left the mesh mask touching the canvas's left edge from
                # f~=200 onward, clipped to bbox x:[0,289] by f=239 (see
                # module docstring's TASK-24 CLIPPING FIX section). Easing
                # with the qpos's OWN progress `p` (not a separate schedule)
                # keeps camera and content in lockstep throughout, and still
                # reproduces `mesh_ctr_final` EXACTLY at p=0 (f=0), so the
                # Act3->Act4 seam this task's own verification requires is
                # unaffected. distance/azimuth/elevation are UNCHANGED
                # (still the fixed ACT3_CAM_DISTANCE/AZ_START/ELEV).
                lookat_a = (1.0 - p) * mesh_ctr_final + p * mesh_ctr_anchor
                cam = act3_frozen_camera(lookat_a)
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

                # TASK-19: every caption below the title is removed from the
                # frame -- stage_label/resid/rotation-progress are still
                # computed and printed above, just not drawn. Only the title
                # and the frame counter remain on screen.
                canvas = draw.stage_title(canvas, "Solve joint angles")
                canvas = draw.label(canvas, f"frame {f + 1}/{N_OUT}", (48, CANVAS_H - 24),
                                     scale=draw.SMALL_SCALE, color=(150, 150, 150))

            else:  # phase == "BC": side-by-side for the whole rest of the act
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

                # TASK-19: title only, no other caption on the left panel.
                left = draw.stage_title(left, "Solve joint angles")

                # --- right panel (TASK-20, fixing the task-19 regression):
                # real video, overlaid with the REPROJECTED PRODUCTION
                # keypoints (`kp_data_prod`, the same fit target driving the
                # left panel this very frame via `cloud_pts` above), not the
                # detector's raw 2D -- coloured per-limb-chain to match the
                # left panel -- see module docstring's "COLOUR-MATCHED 2D
                # KEYPOINTS" / TASK-20 sections. `row`/`native`/`obs_uv` are
                # all indexed by the SAME t_src the left panel's qpos/
                # cloud_pts just used above (see the phase=="BC" branch at
                # the top of this loop) -- there is no second frame index
                # here that could disagree with the left panel's.
                row = t_to_video_row[t_src]
                native = video_frames[row].copy()               # (FRAME_H, SRC_FRAME_W, 3)
                x0 = float(x0_full[t_src])

                # kp_data_prod is in the SCALED/model coordinate frame
                # pose_optimization fit against (see module docstring's
                # COORDINATE-FRAME CAVEAT) -- divide by shared_scale to get
                # back to the raw arena-mm frame the DLTs were calibrated in
                # before projecting (verified once, empirically: this
                # reproduces the observed 2D to a mean ~8.5 px / max ~24 px,
                # see the CAVEAT section).
                obs_uv = clip_io.project(cam_mat_2up, cloud_pts / shared_scale)[0].copy()
                obs_uv[:, 0] -= x0

                panel_native = native[:FRAME_H, int(round(x0)):int(round(x0)) + CROP_W]
                panel_native = draw.draw_leg_chains(panel_native, obs_uv, kp_names,
                                                     alpha=1.0, thickness=1,
                                                     kp_colors=kp_colors_bgr)
                panel_native = draw.draw_keypoints(panel_native, obs_uv, kp_names,
                                                    alpha=1.0, radius=4,
                                                    kp_colors=kp_colors_bgr)
                right = cv2.resize(panel_native, (CANVAS_W - left_w, CANVAS_H),
                                    interpolation=cv2.INTER_LINEAR)

                canvas = np.concatenate([left, right], axis=1)
                canvas = draw.label(canvas, f"frame {f + 1}/{N_OUT}", (48, CANVAS_H - 24),
                                     scale=draw.SMALL_SCALE, color=(150, 150, 150))

            cv2.imwrite(str(out_dir / f"f{f:05d}.png"), canvas)

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
