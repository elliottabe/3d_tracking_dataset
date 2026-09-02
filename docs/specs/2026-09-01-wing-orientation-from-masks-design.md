# Wing orientation from SAM masks (wing-pitch refinement)

*2026-09-01. Session0 bout 28 as the working case. Supersedes the wing-pitch
rest-prior attempt of the same day, which is documented here as rejected.*

> **STATUS (2026-09-01, after measurement): built, measured, and left OFF.**
> `wing_mask_fit.enabled` stays `false`. The stage gets wing PLACEMENT right
> (into the measured -20..-40 deg band, penetration reduced on both wings of
> both flies, marker fit not degraded) and wing FINE MOTION wrong: it destroys
> the bilaterally phase-locked courtship song in wing pitch, at every
> `smooth_weight` measured. **Acceptance criterion 1 was blind to that** and has
> been replaced. Read 8 for the outcome and 5.0/5.1'/5.3' for the amended
> criteria before re-attempting anything here.

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
5. **Tuning `smooth_weight` to protect the song** (shipped 0.005; swept 0.5, 5,
   10, 20, 30, 50 on both flies, 2026-09-01). The temporal term is
   `smooth_weight**2 * sum(dq)^2` with `dq` in RADIANS against a raw-pixel mask
   cost, so 0.005 is a coefficient of 2.5e-5: effectively no coupling, and the
   fit is independent per frame. Raising it buys the amplitude test (fly1 epoch
   1 `hp_rms` 3.77x -> 0.96x at weight 30) at no cost to any mask metric
   (`inside%` 92.5, `explained%` 47.7, penetration and residual flat to four
   decimals across 0.005 -> 50) and restores the control's own step
   distribution (median/p99/max |dpitch| per frame 1.057/11.50/27.9 deg ->
   0.338/1.63/3.8, control 0.358/2.33/3.2). **It does not fix the defect.**
   Both ends destroy the bilateral phase lock -- coherence 0.038/0.070 at 0.005
   and 0.008/0.139 at 30, against a control of 0.418/0.852 -- and weight 30
   additionally removes 37% of epoch 2's real dynamics and all 29 of its song
   pulses. Amplitude is not the axis; frequency content is. See 5.1' and 8.

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
  hand-tuned gate. **Measured 2026-09-01 and CONFIRMED, but only per epoch:**
  the folded wing moves 26-30 deg while the extended wing moves 10-12 deg and
  lands at -12.5 / -19.3 deg -- inside this table's own measured extended-wing
  optimum of -10..-20 deg. Scored across the whole bout with a single
  "singing wing" label the property appears to fail (the left wing's bout
  median moves -5.7 -> -19.1 deg); that reading was an artifact of fly1
  swapping its extended wing at frame 869. The property holds; the whole-bout
  label does not.

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

## 5. Acceptance tests (fixed BEFORE implementation; 1 and 3 AMENDED after it)

The song guard comes first, and it is **per-DOF**, because the yaw-only guard is
what let the rest prior through. **Criteria 1 and 3 were amended on 2026-09-01,
after the measurement in 8** -- criterion 1 replaced outright, criterion 3
tightened. Both originals are kept beneath their replacements, because *why*
they failed is the durable part.

### 5.0 Three scoring rules, each forced by a measurement

* **Gate on ratios to the run's OWN control, never on an absolute number.** The
  "control 0.3615" this spec quoted for criterion 1 is not a number this
  pipeline produces. It came from the weight-0 arm of
  `scripts/analysis/wing_pitch_rest_prior_ab.py`, which re-solves STAC with
  `JaxlsBatchSolver(n_iter=50, smooth_weight=0.1)` -- 20x the pipeline's
  `cfg.ik.smooth_weight = 0.005`. Re-run at weight 0 on the working case it
  reproduces its own family (`wing_pitch_left.hp_rms` 0.3437, `|yawL-yawR|`
  40.7 deg against this spec's "38 -> 39"), while the shipped
  `qpos_refined.npz` over the same frames 0-599 gives **0.7260**. Gating
  absolutely on 0.3615 would have scored the **unmodified control** a 100%
  failure. Ratios to the run's own control are the falsifiable form.
* **Score per extended-wing epoch, never per bout.** "The singing wing" is not a
  property of a bout-fly. On the working case fly1 **swaps extended wing at
  frame 869**: left-extended fraction 1.00 over frames 0-799 and 0.00 over
  900-1499, with a 101-frame majority filter placing the switch at 869, and the
  transition is smooth in every arm (|delta| at the boundary 0.07-0.69 deg
  against each arm's own p99 step of 1.47-11.50 deg, so it is behaviour, not an
  artifact). Scoring criterion 1 with one whole-bout singer label **inverted the
  verdict on which configuration passes**; it is the defect that hid the failure
  in 8.
* **Check an invariant this stage could actually violate.** The `WingX_V12`-
  `WingX_V13` FK separation is NOT one: both sites sit on the same wing body, so
  their separation is a model constant under any pose and its CV is machine
  epsilon (measured 2.8e-15). Reporting it as a passed invariant is a tautology
  and false comfort -- do not take it as evidence of anything. The genuine
  invariant is the **data-side** one, on triangulated `kp3d` indexed BY NAME out
  of `cfg.model.KP_NAMES`: `WingL_V12-V13` 5.016 mm CV 9.1% / `WingR` 5.038 mm
  CV 5.9% (fly0) and 4.907 mm CV 5.6% / 5.065 mm CV 2.5% (fly1), against the
  known-rigid `EyeL-EyeR` at CV 10.2% / 2.7% on the same frames.

### 5.1' Criterion 1, REPLACED: the song survives, tested by bilateral phase lock

*What it measures.* During song both wings are driven by the same bilateral
motor program; the non-extended wing is pinned against extension but still
oscillates (measured: the folded wing's pitch carries a peak at the EXTENDED
wing's f0 with peak/floor 34.6 on epoch 1 and 84.2 on epoch 2, while its yaw sits
exactly on the +85.94 deg joint stop). The song's signature in `wing_pitch_*` is
therefore not an amplitude, it is a **phase relationship between the two wings at
the extended wing's own song frequency** -- something per-frame-independent mask
noise cannot fake and smoothing cannot manufacture. The test is **pairwise**, so
it needs the epoch split only to choose `f0`; it never has to name "the singing
wing", which is exactly where the old criterion broke.

**1'a Applicability.** Scored per epoch with wing extension (`|yawL-yawR| >= 10
deg` under a 101-frame majority filter), and only where the control's own
coherence is BOTH significant AND periodic (1'e). A fly with no such epoch is
N/A for 1'a-1'e and is scored by 1'f instead.

**1'b PRIMARY GATE -- bilateral coherence.** Magnitude-squared coherence
`MSC(hp(wing_pitch_left), hp(wing_pitch_right))` at `f0` (taken from the extended
wing's YAW spectrum) must satisfy
`MSC_treat >= max(0.8 * MSC_control, its own 95% significance level)`.

| fly1 epoch | 95% sig | control | shipped sw 0.005 | sw 30 | gate = 0.8x ctrl |
|---|---|---|---|---|---|
| E1, frames 0-869 | 0.24 | **0.418** @ -155 deg | 0.038 | 0.008 | 0.334 |
| E2, frames 869-1500 | 0.35 | **0.852** @ +162 deg | 0.070 | 0.139 | 0.682 |

The control passes trivially; both measured arms fall **5x-42x below the gate**
on both epochs (shipped 0.038 vs 0.334 and 0.070 vs 0.682; sw 30 0.008 and
0.139). Report at `nperseg` 64/128/256 and require the verdict to hold at all
three (E2 control 0.79/0.85/0.92; both arms 0.04-0.18 throughout) -- a coherence
that depends on the window length is not a phase lock. The measured phase is
~+-160 deg, **not** 0: `wing_pitch_left/right` are hinges about each body's local
+y and the two wing bodies are mirror-placed (quats `[0,-0.4031,0,-0.9152]` and
`[0,0.9152,0,-0.4031]`), so bilaterally symmetric motion appears near 180 deg in
these joint coordinates. Read phase against the model's axes or in-phase song
will look like anti-phase.

**1'c SUPPORTING -- song-band SNR.** `peak/floor` of the pitch PSD at the yaw's
`f0` must be >= 0.5x control. Control 59.4 (E1) / 156.6 (E2); shipped 23.4 / 9.2;
sw 30 98.1 / 26.3. **Never promote this to primary**: it is a ratio, so smoothing
raises it while deleting the song -- sw 30 scores 98.1 on E1 (1.65x control) with
a bilateral coherence of 0.008, and on the non-singing female it turns control's
6.4 into 46.6, i.e. it manufactures a spectral peak where the video has none.

**1'd SUPPORTING -- content retention.** `corr(hp(pitch)_treat,
hp(pitch)_control)` >= 0.5, per epoch. Shipped 0.358 (E1) / **0.017** (E2);
sw 30 0.328 / 0.012. The E2 number is the whole finding in one line: `hp_rms` is
unchanged at 1.00x while the correlation with the content it replaced is 0.017 --
the high-frequency content is not modified, it is *substituted*. *Escape hatch,
stated so this cannot become a preservation-only gate:* a treatment that fails
1'd while RAISING 1'b above the control has recovered song the marker solve
missed; adjudicate that case on the pulse train and the figure rather than
auto-failing it. Neither measured arm is that case (both LOWER 1'b, by 6x-52x).

**1'e SUPPORTING -- pulse train, and the specificity check.** At one common
threshold (2x the control's own sigma, so counts are comparable) the treatment's
pulse count must be within 0.5x-2x control and the median matched-pulse offset
within +-1 frame. Control 43 (E1) / 29 (E2) / 16 (fly0); shipped 168 / 24 / 425;
sw 30 16 / 4 / 98. Pulses are never *delayed* -- median offset is +0.0 frames in
every arm -- they are **buried** (shipped E1 detects 168 where 43 are real,
chance match 97%) or **removed** (sw 30 keeps 0 of E2's 29). The two wings'
`hp(pitch)` cross-correlation must also be visibly PERIODIC at the `1/f0` period;
that periodicity, not the coherence value alone, is what makes 1'b specific.
fly0's control also reaches `MSC ~ 0.8` at high frequencies -- two wings on one
body share pose noise -- but its cross-correlation stays within +-0.2 and is
aperiodic, against a clean +-0.6 oscillation at the 6.4-frame period on fly1;
and the f0 the test picks on her is 43.7 Hz (an 18.3-frame period), not the
song's. Read back from `dofs/bilateral_phase_lock.png` under
`figures/2026-09-01-wing-mask-fit/`.

**1'f NEGATIVE CONTROL -- do not manufacture song.** On a fly with no extension
epoch the fit must not invent oscillation: pitch `peak/floor` must not rise above
control's, and `hp_rms` must stay <= 2x control. Measured on fly0 (female, does
not sing): peak/floor 6.4 -> **1.0** shipped (a flat broadband spectrum) and
**46.6** at sw 30, with `hp_rms` 0.279 -> 2.579 (9.3x) / 0.811 (2.9x). **Both
arms fail** -- shipped on amplitude alone (x9.3 of pure broadband), sw 30 on
both halves (x2.9 *and* a manufactured peak 7x the control's). State plainly that this threshold is *uncalibrated*: no measured arm
has ever passed it, so 2x is a target, not a bracketed corner.

**1'g REPORTED, NEVER GATED.** The `hp_rms` ratio -- the number that was blind --
and `lp_std`, the <40 Hz std that the rejected rest prior collapses (fly1 E1:
5.14 control -> 11.40 shipped -> 11.56 at sw 30; the slow structure is retained
by both arms, which is why placement survives). Per epoch, `hp_rms` of the
epoch's extended wing:

| epoch | wing | control | shipped 0.005 | sw 30 |
|---|---|---|---|---|
| E1 0-869 | `wing_pitch_left` | 0.7197 | 2.7121 (3.77x) | 0.6891 (0.96x) |
| E2 869-1500 | `wing_pitch_right` | 0.8054 | 0.8019 (**1.00x**) | 0.5045 (0.63x) |
| E2b 1000-1500 | `wing_pitch_right` | 0.8223 | 0.8090 (0.98x) | 0.5160 (0.63x) |

**Why the amplitude test had to be retired.** Read the E2 row: the shipped config
scores **1.00x -- a clean PASS -- on the epoch where 1'b, 1'c, 1'd and 1'e all
show the song was destroyed.** An RMS cannot distinguish preserved song from
substituted noise of equal RMS, and it cannot separate "jitter added" from "song
energy moved into this DOF by a changed wing orientation" either; that second
ambiguity is what made the first reading of this measurement wrong. Its failure
is also mutual across epochs: sw 30 buys E1's amplitude (0.96x) by taking 37% of
E2's real dynamics -- the rejected rest prior's own failure mode -- so no single
smoothing weight satisfies both epochs, and an amplitude gate would have shipped
whichever one the analyst happened to score.

*Superseded original, kept for the record:* "**1. Extended wing pitch dynamics
preserved.** `hp_rms` of `wing_pitch_*` on the SINGING wing must stay within 20%
of control (control 0.3615). The rest prior scored 0.0916 -- a 75% loss -- and
that is the failure to detect." It failed three ways at once: an absolute control
value this pipeline does not produce (5.0), one singing wing per bout (5.0), and
an amplitude statistic blind to substitution (above).

### 5.2 Criterion 2 (unchanged)

2. **Song unchanged.** `|yawL-yawR|` mean and its `pulse_stats` (hp_rms,
   kurtosis, peak count) within noise of control.
   *Measured: PASSES exactly, not approximately -- `max |delta yaw| = 0.0 rad` on
   both flies, every arm, because `opt_mask` freezes every DOF but the two
   `wing_pitch_*`. Note what that means for its power as a guard: criterion 2 is
   satisfied by construction and can never detect a change to pitch. It is a
   check on the implementation, not on the song.*

### 5.3' Criterion 3, AMENDED: penetration, scored as a gap-closure FRACTION

3'. **Penetration reduced.** `mj_geomDistance(wing, abdomen)` median must move
    from -0.043 toward the model's own -0.0013 grazing value on BOTH wings, and
    is scored as `(control - treatment) / (control - (-0.0013))`, gated at
    **>= 50% on both wings of both flies**. **A sign test is not acceptable**:
    the criterion was implemented as `a1 > a0`, so a 1e-9 improvement passes it.
    Measured (coverage 0.3):

| fly / wing | control | treatment | gap closed |
|---|---|---|---|
| fly0 (female) left | -0.04442 | -0.03154 | **30%** |
| fly0 (female) right | -0.04515 | -0.02746 | **40%** |
| fly1 (male) left | -0.02301 | -0.00690 | **74%** |
| fly1 (male) right | -0.03324 | -0.01235 | **65%** |

  Verdict on the measured arms: **direction met, target not met** -- not a pass.
  fly1 clears 50% on both wings; fly0 does not, and fly0's treatment is still
  ~20x the -0.0013 target: the wings stop cutting deeply through the abdomen but
  they do not stop touching it. fly0's control (-0.0444 / -0.0452) reproduces
  this spec's own measured "median ~0.043 model units" exactly, so the defect is
  confirmed on the female. The 50% line is a judgement and it is **not bracketed
  by a passing measurement on the female** -- 40% is the best any measured arm
  achieves there. Pitch alone may not be able to close it; roll is out of scope
  (4.4).

### 5.4 Criteria 4-7 (unchanged)

4. **Marker fit not degraded.** Wing-keypoint residual rise <= 20%.
   *Measured: PASS. fly1 0.011062 -> 0.011181 (+1.1%); fly0 0.014668 ->
   0.011362, a 22.5% IMPROVEMENT.*
5. **Pitch lands near the mask optimum**, not at rest: folded wing within ~10 deg
   of the measured -20..-40 deg, not -57 deg.
   *Measured: PASS on both flies. Folded wing -11.1 -> -32.0 deg (fly1) and
   -7.8 -> -35.2 deg (fly0); neither arm goes near the springref -57.3 deg, so
   the stage is not reproducing the rest prior's error.*
6. **Figure**, per CLAUDE.md, with the expectation written first: the 3-view
   sidebyside plus renders from **all 7 rig cameras**, on the FEMALE and on the
   male's non-singing wing, control vs treatment, mask outline overlaid. A
   still of the singer alone would hide a regression.
   *Measured: PASS, with one gap this criterion should name explicitly in
   future -- **which hard case was actually exercised**. On this bout every
   frame has 7/7 valid cameras for both flies, so the "worst mask coverage"
   picker degenerated to frame 0 and NO mask-dropout frame exists in the
   evidence. The wall / near-darkness hard case is present (fly0 on
   `Cam2012857`, `Cam2012861`).*
7. **Blade-normal is NOT an acceptance metric.** It is a hypersensitive function
   of the null direction -- a 12% marker-residual change swung it 59 deg -- so it
   cannot adjudicate a change to that direction. Report it, do not gate on it.
   *Reported: fly1 L 12.2 -> 7.7 deg, R 15.8 -> 2.2 deg; fly0 L 34.4 -> 10.4
   deg, R 30.7 -> 5.9 deg.*

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

## 8. Measured outcome (2026-09-01): the stage stays OFF by default

**Decision: `wing_mask_fit.enabled` stays `false`.** The stage is implemented,
tested, wired in with provenance, and it produces a real
`qpos_wingfit.npz` -- and it is not shippable, at any weight measured. The one
sentence version: **it gets PLACEMENT right and FINE MOTION wrong.**

Evidence: Session0 `2025_10_20_13_20_04` bout 28, frames 0-1499 of 2007, both
flies, control (marker-only STAC, read-only original tree) vs treatment
(the production stage worker). `docs/benchmark/2026-09-01-wing-mask-fit/notes.md`
+ `scorecard.json`; figures under `figures/2026-09-01-wing-mask-fit/`
(gitignored, regeneration commands in the notes).

### 8.1 Placement is genuinely good (criteria 2, 4, 5, 6 pass; 3' direction met)

* Both flies move out of the fitted ~-8 deg toward the measured -20..-40 deg
  optimum band: fly0 -8.4/-7.8 -> -32.4/-35.2 deg, fly1 -5.7/-11.1 ->
  -19.1/-32.0 deg (bout medians, coverage 0.3). fly1's left wing sits at -19.1
  deg only because the bout median mixes its two song epochs; per epoch it lands
  on the -10..-20 deg EXTENDED-wing optimum, which is the next bullet.
* Penetration improves on **both wings of both flies** (3'), 30-74% of the gap
  to the model's own grazing value.
* Wing-marker residual is not degraded (+1.1% on fly1; -22.5%, i.e. improved, on
  fly0), and every DOF except the two `wing_pitch_*` is bit-identical
  (`max |delta| = 0.0`, 2 of 93 qpos addresses touched).
* Per epoch the stage **is self-targeting**, as 4.1 claims: the folded wing
  moves 26-30 deg while the extended wing moves 10-12 deg and lands at
  -12.5 / -19.3 deg -- on the measured extended-wing optimum of -10..-20 deg.
* The 7-camera figure shows the female's splayed-V wings closing into the real
  folded silhouette and the male's folded blade going from
  face-on-and-outside-the-mask to edge-on-and-inside.

### 8.2 Fine motion is destroyed (criterion 1' fails on both arms, both epochs)

The control genuinely carries **bilaterally phase-locked courtship song in wing
pitch**, and the fit replaces it:

| quantity, fly1 | E1 (0-869) | E2 (869-1500) |
|---|---|---|
| MSC(pitch L, pitch R) at f0, control (95% sig) | **0.418** (0.24) | **0.852** (0.35) |
| ... shipped `smooth_weight` 0.005 | 0.038 | 0.070 |
| ... `smooth_weight` 30 | 0.008 | 0.139 |
| pitch PSD peak/floor, control -> shipped -> sw 30 | 59.4 -> 23.4 -> 98.1 | 156.6 -> 9.2 -> 26.3 |
| corr with control's own hp(pitch), shipped / sw 30 | 0.358 / 0.328 | **0.017** / 0.012 |
| song pulses at a common threshold, ctrl -> shipped -> sw 30 | 43 -> 168 -> 16 | 29 -> 24 -> 4 |

The control's cross-correlation is a clean +-0.6 oscillation at the 6.4-frame
song period; both arms' are +-0.15 and aperiodic. Pulses are never *delayed*
(median matched offset +0.0 frames) -- at 0.005 they are buried under ~4x as many
threshold crossings, at 30 they are removed outright (0 of E2's 29 kept). On the
female, who does not sing, the fit **adds** oscillation the video does not have
(`hp_rms` x9.3 at peak/floor 1.0). The negative control behaves: fly0's control
trace is flat with no periodic cross-correlation, so the test does not fire on
everything.

**And the old criterion 1 could not see any of it**: on E2 the shipped config
reads `hp_rms` 1.00x, a clean pass, in exactly the frames where the song was
destroyed. That is why 5.1' replaces the amplitude test with the phase-lock test.

### 8.3 `smooth_weight` is the wrong axis; the recommended direction

Neither shipped 0.005 nor the measured 30 is shippable. 30 was proposed because
it restores the control's step distribution and buys E1's amplitude at zero cost
to any mask metric -- but it fails the phase test on both epochs, removes 37% of
E2's dynamics and all of its pulses, and manufactures a spectral peak on the
female. Both ends of the sweep destroy the phase lock (see 3, item 5).

**Recommended direction (NOT implemented; out of scope for this spec's plan):**
stop the stage carrying high-frequency content at all -- have it supply a
slowly-varying pitch correction (per-epoch constant, or a low-passed
per-frame offset) and leave the song to the marker solve, e.g. by adding the
control's own high-frequency pitch back after the correction. Placement and fine
motion are separable here: the fit moves the wing to the right *place* and gives
it the wrong *texture*, and `lp_std` (5.14 -> 11.40 -> 11.56) shows the slow
structure survives in every arm. That is a design change, not a config change,
and it must be re-measured against 5.1' before anything is enabled.

### 8.4 `coverage_weight` is UNDECIDED (not "measured better")

On the headline metrics 0.0 and 0.3 agree to three significant figures on both
flies, but the term is **not inert**, and every real-data difference it makes on
the female is in the wrong direction:

* fly0 `hp_rms(wing_pitch_left)` 2.5188 (cov 0.0) -> 2.5785 (cov 0.3), +2.4%;
  fly0 right-wing penetration -0.026717 -> -0.027461, 2.8% **worse**.
* At cov 1.0 the female's added jitter nearly doubles (`hp_rms` 2.58 -> 5.09) and
  L/R symmetry breaks (-36.4/-30.7 against -32.4/-35.5). Do not go there.
* In the synthetic ground truth its only win is at the **REST** attitude
  (containment-only 2.49 deg -> 1.14 deg with coverage, at `huber_delta` 8) and
  it is offset by an almost equal loss at the **EXTENDED** attitude
  (containment-only 0.53 deg -> 1.63 deg) -- and the extended attitude is the one
  criterion 1' exists to protect. Alone, coverage is catastrophic at rest
  (42.75-43.49 deg, i.e. 18 deg WORSE than doing nothing), so it is a complement
  at best; the "anti-degeneracy insurance" argument is unsupported (real-data
  `explained%` RISES under containment-only, 38.6 -> 45.7 on fly0).

**Verdict: UNDECIDED.** If a tie-break is forced the evidence tips to **0.0** --
its only clear win is in the regime the song criterion does not defend, and its
only measurable real-data effects are on the hard fly and adverse. The value left
on disk (0.3) is inherited, not chosen; do not read it as a measured decision.
`huber_delta` MUST stay > 0 whatever the weight (at 0 the chamfer is an
unrobustified L2 and SAM halo dominates it; 8.0 is the measured corner).

### 8.5 Known limits of this evidence

* **The acceptance window hides the worst quarter on the HARD fly.** Frames
  0-1499 of 2007 were scored. fly0's frames 1500-2006 have **54% of frames
  moving pitch > 45 deg, max 98 deg**, against max 42 deg and 0% inside the
  window. Female-specific; `smooth_weight = 30` does not help it. Any future
  acceptance run must score the WHOLE bout.
* **Performance misses the budget: 74-97 s per bout-fly against 60 s** (74.3 s
  for the isolated refinement in Task 5, 85-97 s for the stage end to end on one
  L40S, T=2007, 7 cameras). The measured, bit-identical fix is threading
  `sdf_stack_from_masks` (34.4 s -> 4.8 s at 16 threads, projecting ~51 s),
  deferred to Phase 2 of `docs/plans/2026-09-01-wing-orientation-from-masks.md`.
* **One bout, one recording, two flies** that share a calibration, an
  `offsets.h5` and a `kp3d`. fly0's `kp3d.npz` is pre-gate and required
  `pipeline.allow_stale_kp3d=true` (correct for an A/B, but it means fly0's
  numbers ride on a triangulation that never saw the current Stage-B gates).
* **No `sex.json` on this recording.** male = fly1 is the pipeline's
  canonicalisation plus a behavioural check (`|yawL-yawR|` 28 deg mean vs
  3.5 deg), not a human-reviewed label.
* **No mask-dropout hard case is in this evidence**: every frame in the window
  has 7/7 valid cameras for both flies.
* **`fs = 800 Hz` is assumed, not read from config** (`cfg.recording` carries no
  fps key). The fps-independent form of the song claim is the **6.4-frame
  period**; `f0 = 125 Hz` holds only at the assumed fs. Ratios are unaffected
  because control and treatment share the constant.

### 8.6 What would change the decision

A variant that passes **1'b on both epochs of a singing fly** while keeping
criterion 5's placement and at least the measured 3' gap closure, scored over the
WHOLE bout, on at least one more recording -- preferably one with a reviewed
`sex.json`. Re-derive `smooth_weight` and `coverage_weight` from scratch if the
mask cost normalisation ever changes: the temporal term is
`smooth_weight**2 * sum(dq)^2` with `dq` in radians against a raw-pixel cost, so
those numbers are specific to this normalisation and must not be copied forward.
