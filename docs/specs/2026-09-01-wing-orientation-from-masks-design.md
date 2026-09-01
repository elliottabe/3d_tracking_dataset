# Wing orientation from SAM masks (wing-pitch refinement)

*2026-09-01. Session0 bout 28 as the working case. Supersedes the wing-pitch
rest-prior attempt of the same day, which is documented here as rejected.*

## 1. The defect

In the IK the wings should lie flat along the abdomen. They are rotated inward
and pass through it. It is clearest on the female but present on the male's
**non-singing** wing, and the SAM mask visibly disagrees with the rendered wing
orientation.

Measured, on bout 28:

* wing/abdomen geoms interpenetrate on essentially every frame, median depth
  ~0.043 model units (~20% of a wing length) against the model's own rest pose,
  which merely grazes at -0.0013. **The geometry is fine; the fit is not.**
* the folded wing sits ~58 deg off its rest pitch (roll ~31 deg).

## 2. Why it happens (root cause, measured)

Each wing body carries only **two** markers. `KEYPOINT_MODEL_PAIRS` maps
`WingX_base -> thorax`, and only `WingX_V12`/`WingX_V13 -> wing_left/right`.
Two points on a rigid body define an **axis, not an orientation**, so rotation
about the V12-V13 line is unconstrained.

The Jacobian `d[V12;V13]/d[yaw,roll,pitch]` at the rest pose, with the run's own
fitted offsets:

| quantity | yaw | roll | pitch |
|---|---|---|---|
| per-column sensitivity | 0.271 | **0.344** | **0.042** |
| SVD null direction | 0.035 | 0.088 | **-0.996** |

Condition number 13. **Pitch is 99.6% of the null direction and 8.2x the
weakest column**; roll is the *strongest*-observed wing DOF. One degree along
the null direction moves the two markers 0.46 units against 6.04 for the best
direction.

The marker offsets are NOT the problem: all four wing offsets sit on the wing
mesh, within 4-10% of the blade's thickness of a mesh vertex (checked against
the 20,184-vertex mesh, not the smaller collision ellipsoids).

## 3. Rejected alternatives (do not re-attempt without new evidence)

1. **Re-map `WingX_base` onto the wing body** (at the hinge, local `(0,0,0)`).
   Provably useless: adding it left the singular values **bit-identical**,
   because a marker at the rotation centre has an identically-zero rotational
   Jacobian. Even at its actual fitted offset (a 5-14% lever arm) the condition
   number moves 13.4 -> 12.8. It would also *remove* a thorax constraint.
2. **Rigid-length projection** (`rigid_lengths.enforce_bone_lengths`). Fails
   leave-one-camera-out on 4/6 wing keypoints; V12-V13 is not rigid anyway
   (CV 30% male, 80-120% female). The error is angular, so setting the correct
   length slides the point along a wrong direction.
3. **Cross-view L/R swap search** (`wing_lr_assign.resolve_wing_lr`). Fires on
   2/1500 male frames. A *collapse* is not a *swap*: permuting two labels that
   occupy the same point is a no-op.
4. **Wing-pitch rest prior** (`JAXLS_Q_REG_TO_REST` on `wing_pitch_*`). It
   fixes the geometry -- left-wing penetration -0.030 -> +0.000, wings visibly
   flat, `|yawL-yawR|` unchanged at 38 -> 39 deg -- but it **removes 75% of the
   EXTENDED wing's pitch dynamics** (hp_rms 0.3615 -> 0.0916 at w=0.01, 0.0153
   at w=0.1) and makes the folded wing's yaw spiky (kurtosis 4.6 -> 145). A
   yaw-only song guard is BLIND to this; that is how it nearly passed.
   It also aims at the wrong value -- see 4.1.

## 4. The design

**A post-STAC, wings-only, mask-driven refinement of wing PITCH.** Not a term in
the main marker solve, and not a prior.

### 4.1 Why the mask, and why it is better than rest

Pitch barely moves the markers but changes the blade's projected silhouette
enormously (face-on ellipse vs edge-on sliver). Sweeping `wing_pitch_*` at real
frames and counting projected wing-mesh vertices falling OUTSIDE the mask:

| | fitted (control) | rest | **mask-optimal** |
|---|---|---|---|
| right wing (folded) | -9.7 / -1.4 / -1.4 deg | -57.3 deg | **-40 / -30 / -20 deg** |
| left wing (extended) | -1.9 / -0.8 / -7.9 deg | -57.3 deg | **-10 / -10 / -20 deg** |

*(frames 400/750/1100; spill contrast 0.11-0.18, so a real minimum exists.)*

Three consequences:

* the mask constrains pitch where markers cannot;
* **rest is the wrong target** -- the mask wants the folded wing 20-37 deg away
  from rest, which is what the rejected prior would have overshot;
* the mask **already agrees with the fit on the extended wing** (-10 vs -1..-8)
  and disagrees only on the folded one (-20..-40 vs -1..-10). The term is
  therefore *self-targeting*: it corrects the wing that is wrong and leaves the
  singer where it already is. This is the property the rest prior lacked and the
  reason this design can satisfy "accommodate singing and non-singing" without a
  hand-tuned gate.

Prerequisite verified: SAM masks DO contain wing pixels. Wing landmarks project
inside the fly's mask on 75-91% of frames, comparable to `Abd_tip` at 78%
(`Scutellum` 100%). Camera `Cam2012631` is the weakest (43-46% for the right
wing, `Abd_tip` 20%) and must be down-weighted or excluded per-frame by the
existing mask-validity flags.

### 4.2 Reuse, do not rewrite

The needed machinery was deleted in `0bc36fe` (the silhouette-polish removal)
and is recoverable from `0bc36fe^`:

* `silhouette_sdf.py` -- offline cropped signed distance field per (frame,
  camera): negative inside, positive outside, rescaled to original pixels.
  Pure NumPy/SciPy/cv2.
* `silhouette_containment.py` -- differentiable JAX residual sampling that SDF
  bilinearly at projected vertices.
* `silhouette_dof.py` -- appendage DOF and mesh-vertex selection by name.
* `silhouette_refine.py` -- Adam refinement of appendage DOFs with the root
  frozen. **Use Adam, not jaxls GN**: the recorded reason is that jaxls uses
  non-scale-invariant `lambda*I` damping and the pixel-scale silhouette
  Jacobian (`|diag(JtJ)| ~ 1e6`) makes every GN step overshoot and be rejected.

Restore them under non-silhouette names (`mask_sdf.py`, `mask_containment.py`,
`appendage_dof.py`, `wing_mask_refine.py`) with their tests.

### 4.2b Every recovered file is re-verified, not trusted

Docstrings in this repo have been unreliable: tonight three separate ones were
contradicted by measurement (the roll prior calling roll the weak wing DOF when
it is the strongest; "a SAM3 mask is the BODY silhouette ... wings are NOT
filled" when wing landmarks sit inside the mask on 75-91% of frames;
`silhouette_iou` named for a polish it has nothing to do with). So each
recovered file's load-bearing claim was re-measured before this spec relied on
it. Results:

| file | claim | verdict |
|---|---|---|
| `silhouette_sdf` | signed, negative inside / positive outside | **PASS** (-25.3 inside, +27.8 outside) |
| `silhouette_sdf` | "rescaled to ORIGINAL-image pixels" | **APPROXIMATE ONLY -- see below** |
| `silhouette_containment` | `r = conf * relu(d + margin)`, one-sided, `present`-gated | **PASS** |
| `silhouette_dof` | `abdomen` pattern does not catch the leg `coxa_abduct` | **PASS** (no 'coxa' joint selected) |
| `silhouette_dof` | `include=("wing",)` selects exactly the wing DOFs | **PASS** (the 6 wing joints) |
| `silhouette_dof` | wing vertex selection is wing-only | **PASS** (100 verts, `wing_left`+`wing_right`) |
| `silhouette_refine` | "All other DOFs stay exactly at q_init" via `opt_mask` | **PASS** (opt_mask gates the update and the limit rows) |

**The one real defect: the SDF's original-pixel scaling is anisotropic.** The
distance transform runs on the crop AFTER an anisotropic resize to a square grid
and is then rescaled by a single factor, so it cannot be right in both axes.
Measured on a 40x60 box: `grid_scale` (0.593, 0.889), a 1.50x anisotropy, and the
SDF at the box centre reads -25.3 px against a true inradius of 20.0 px -- a
**27% error**, direction-dependent. For a containment penalty this mis-weights
the pull by up to the anisotropy factor depending on which way the vertex is
outside. Fix before use: either resize isotropically (letterbox the crop) or run
the distance transform at the crop's native resolution.

Also note three of the "failures" in the first verification pass were bugs in
the VERIFICATION, not the code: grepping for `abduct` (but `abdomen_abduct_*`
are legitimately abdomen joints), grepping for a parameter name I guessed
(`sil_qs`; it is `opt_mask`), and indexing `seg_names` with `vertex_segment`
values (`seg_ids` are values 1..67, `seg_names` is positional 0..66 -- an
off-by-one, the third index-space error of the session). Verify by reading the
code and measuring, never by pattern-matching assumed names.

### 4.3 The cost, and the trap in the recovered code

`silhouette_containment` is **one-sided**: `r = conf * relu(d + margin)`, so
vertices *inside* the mask cost nothing. Minimising it alone would tuck the wing
INSIDE the body silhouette -- which is the bug we are fixing. Containment alone
is therefore not sufficient here.

The cost is two terms on the WING vertices only:

1. **Containment** (recovered): penalise wing vertices outside the fly mask.
2. **Wing coverage**: the mask region NOT explained by the projected BODY must be
   explained by the projected WINGS. Computed as the unexplained-residual area
   after rasterising the body, so the wing is pulled out to the blade region
   rather than hidden in the torso.

The historical note that coverage "can fight containment" applied to the global
appendage fit (wings + legs + abdomen). Here both terms act on wing vertices
only, and the acceptance tests below are what decide the weights.

### 4.4 Scope and staging

* DOFs optimised: `wing_pitch_left`, `wing_pitch_right`. **Roll is OUT OF
  SCOPE** (user decision, 2026-09-01), not merely defaulted off: it is the
  strongest-observed wing DOF (per-column 0.344, vs pitch 0.042), so the markers
  already constrain it and a mask term there would fight real signal -- which is
  exactly how the earlier roll smoothness prior destroyed the song.
* Everything else frozen: root, thorax, abdomen, legs, and **wing yaw** (yaw
  carries the song; it is well observed and must not be touched).
* Runs as an opt-in stage after STAC and before the bridge, writing
  `qpos_wingfit.npz`; `config: wing_mask_fit.enabled` (default false), and it
  participates in the Stage-B/C gate-signature provenance so a resumed run
  cannot silently skip it.
* Per-frame, per-camera masking uses the existing validity flags; a frame with
  fewer than 3 valid wing-visible cameras is left at its STAC pose.

## 5. Acceptance tests (fixed BEFORE implementation)

The song guard comes first, and it is **per-DOF**, because the yaw-only guard is
what let the rest prior through.

1. **Extended wing pitch dynamics preserved.** `hp_rms` of `wing_pitch_*` on the
   SINGING wing must stay within 20% of control (control 0.3615). The rest prior
   scored 0.0916 -- a 75% loss -- and that is the failure to detect.
2. **Song unchanged.** `|yawL-yawR|` mean and its `pulse_stats` (hp_rms,
   kurtosis, peak count) within noise of control.
3. **Penetration reduced.** `mj_geomDistance(wing, abdomen)` median from -0.043
   toward the model's -0.0013 grazing value, on BOTH wings.
4. **Marker fit not degraded.** Wing-keypoint residual rise <= 20%.
5. **Pitch lands near the mask optimum**, not at rest: folded wing within ~10 deg
   of the measured -20..-40 deg, not -57 deg.
6. **Figure**, per CLAUDE.md, with the expectation written first: the 3-view
   sidebyside plus renders from **all 7 rig cameras**, on the FEMALE and on the
   male's non-singing wing, control vs treatment, mask outline overlaid. A
   still of the singer alone would hide a regression.
7. **Blade-normal is NOT an acceptance metric.** It is a hypersensitive function
   of the null direction -- a 12% marker-residual change swung it 59 deg -- so it
   cannot adjudicate a change to that direction. Report it, do not gate on it.

## 6. Risks

* **The mask minimum is broad** (~±10-15 deg), so this pins pitch loosely. That
  is still far better than the markers' ~nothing, but do not expect degree-level
  accuracy.
* **Contrast is modest** (spill 0.11-0.18), so mask noise matters; hence the
  per-camera validity gating and the exclusion of weak cameras.
* **Cost/benefit**: an SDF precompute per (frame, camera) plus an Adam loop.
  Budget it against the QC/render optimisation already in flight; the wing fit
  must not become the new dominant stage.
* **This is a narrow resurrection of deleted code.** The silhouette polish was
  removed because it was globally disabled and could not fix the legs. Nothing
  here re-enables it for legs, the abdomen, or the root.

## 7. Explicitly out of scope

* A third wing landmark (e.g. trailing edge) would close the null space
  properly, but needs new annotations and a detector retrain.
* Leg or abdomen silhouette terms.
* Any change to wing yaw, wing ROLL, or the marker offsets (all measured
  correct or well-constrained already).
