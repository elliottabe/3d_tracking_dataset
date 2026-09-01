# Wing-pitch refinement against the SAM masks — acceptance measurement

*2026-09-01. Session0 `2025_10_20_13_20_04`, bout 28, **frames 0–1499**, both
flies. Spec: `docs/specs/2026-09-01-wing-orientation-from-masks-design.md`.
Scorecard: `scorecard.json` beside this file. Figures (gitignored):
`figures/2026-09-01-wing-mask-fit/` — regeneration commands at the end.*

**The headline: acceptance criterion 1 FAILS, and the geometry criteria all
pass.** Criteria 2–5 pass on both flies at every coverage weight; criterion 1
(per-DOF song guard) fails on fly1 with `hp_rms` of `wing_pitch_left` going
0.600 → 2.31, a **3.9× RISE**. That is the opposite sign from the failure the
criterion was written to catch, and it is *fully curable by the temporal term*:
**`smooth_weight = 30` (shipped: 0.005) passes all six criteria on both flies**
and leaves every other acceptance number unchanged — on the female it is
marginally better. See §5. Setting the default is Task 8's call; nothing was
changed here.

The measurement the sweep was for: **`coverage_weight` is inert from 0.0 to 0.3
on both flies** — the same pose to three significant figures, not a tie inside
a noise floor — and mildly harmful at 1.0. See §4.

---

## 1. How the arms were built without touching the user's data

`Session0/2025_10_20_13_20_04` bout 28 already had complete artifacts for both
flies (`DONE`, `stac_ik.h5`, `qpos_refined.npz`, `outputs.h5`, `qc.json`,
overlays, `sidebyside.mp4`). **That tree is the CONTROL arm and nothing was
written to it.** Two arms were built instead:

* **Control** — read-only from the original tree. `qpos_refined.npz`'s `qpos`
  was verified `max|Δ| == 0.0` against `stac_ik.h5`'s on both flies, so the
  control really is the plain STAC pose the refinement starts from. Its
  `outputs.h5` carries no `pose_source` key, which the stamp readers correctly
  interpret as `"none"`.
* **Treatment** — a scratch run root at
  `/gscratch/portia/eabe/data/Johnson_lab/scratch/wingfit_task7/treat/pose`,
  built by **copying** (never symlinking, so no write could reach the original
  through a link) `scale.json`, `offsets.h5`, `coverage.json` and, per fly,
  `kp2d.npz`, `kp3d.npz`, `kp3d_filt.npz`, `stac_ik.h5`, `qpos_refined.npz`.
  `DONE`, `outputs.h5`, `qc.json`, `qc_perframe.npz` and the videos were
  deliberately NOT copied, so `process_bout_fly` resumes rather than
  early-returning on `DONE`, and every pose-derived artifact is rebuilt.

**Verified, not assumed:** an md5 manifest of all 44 files under the original
`pose/` was taken before any job started and re-checked at the end —
`md5sum -c` reports OK on all 44 and the file count is unchanged.

**One flag was required and it is the right one.** The run refused with
`kp3d.npz carries no gate signature (written before this check existed)` on
fly0 — the known pre-gate `kp3d.npz` that `configs/pipeline.yaml` documents for
exactly this bout. `pipeline.allow_stale_kp3d=true` was passed. Reusing that
`kp3d.npz` is not a workaround here, it is a *requirement*: recomputing it
would change the input to the treatment arm and the control's `outputs.h5`
would no longer be a valid comparison.

The **coverage sweep** does not go through `run_bout.py` at all. It calls
`scripts.run_bout.wing_mask_fit_bout` — the production stage worker, the same
function Stage D2 calls — directly from `scripts/analysis/wing_mask_fit_ab.py`,
with the control `qpos_refined.npz` as `q_init` and the bout's own masks and
bridges. Cross-check that the two paths agree: the pipeline's own log line and
the A/B script's fly0 `cov0.3` arm both report **median wing_pitch change
L −24.51° / R −27.26°**, identical to two decimals.

## 2. The first real `[wing-mask-fit]` log line

No `qpos_wingfit.npz` had ever existed before this. Both flies ran end to end:

```
[wing-mask-fit] bout 28 fly0: fit; refined 1755/2007 frames; 252 left at the STAC pose
(252 unsolved by STAC, 0 seen by fewer than 3 mask cameras, 0 non-finite pose);
median wing_pitch change L -24.51 deg / R -27.26 deg
[wing-mask-fit] bout 28 fly1: fit; refined 2007/2007 frames; 0 left at the STAC pose
(0 unsolved by STAC, 0 seen by fewer than 3 mask cameras, 0 non-finite pose);
median wing_pitch change L -12.85 deg / R -20.29 deg
```

The three skip buckets sum to `n_skipped` as designed, and fly0's 252 are
exactly its `bridge_ok == False` frames. Task 6's provenance chain works for
real: `qpos_wingfit.npz` carries `wing_mask_fit_sig`, and that same signature
string is stamped into `outputs.h5:pose_source`, `qc.json`, `qc_perframe.npz`
and `sidebyside.pose_source.json`. Stage F rendered with `--pose wingfit`.

Wall clock, measured: the fit itself is **85–97 s** per bout-fly (T=2007,
7 cameras, one L40S) — above Task 5's 74 s because the SDF precompute is still
serial, and above the 60 s budget.

## 3. Acceptance, all six criteria

Gates are **ratios to this run's own control**, not to an absolute number — see
the provenance note in §7 for why the spec's "control 0.3615" is not the number
this pipeline produces.

### fly1 (male, singer = LEFT/extended, folded = RIGHT), coverage_weight 0.3

| # | criterion | control | treatment | verdict |
|---|---|---|---|---|
| 1 | `hp_rms(wing_pitch_left)`, singing wing, within 20% | 0.6002 | **2.3133** (ratio **3.85**) | **FAIL** |
| 2 | `|yawL−yawR|` mean + pulse stats within noise | 28.07°, hp_rms 2.1188, kurt 3.05, 45 peaks | **bit-identical** (max \|Δyaw\| = 0.0 rad) | PASS |
| 3 | penetration → −0.0013 on BOTH wings | L −0.02301 / R −0.03324 | L **−0.00690** / R **−0.01235** | PASS |
| 4 | wing-keypoint residual rise ≤ 20% | 0.011062 | 0.011181 (**+1.1%**) | PASS |
| 5 | folded-wing pitch near −20…−40°, not −57.3° | −11.1° | **−32.0°** | PASS |
| 6 | 7-camera figure, female + male's non-singing wing | — | see §6 | PASS |
| 7 | blade normal — REPORTED, NEVER GATED | L 12.2° / R 15.8° | L 7.7° / R 2.2° | (reported) |

### fly0 (FEMALE — the hard fly; not singing, both wings folded), coverage_weight 0.3

| # | criterion | control | treatment | verdict |
|---|---|---|---|---|
| 1 | per-DOF song guard | — | — | **N/A** (`\|yawL−yawR\|` = 3.52°: no wing extension) |
| 2 | song unchanged | 3.52° | bit-identical | PASS |
| 3 | penetration | L **−0.04442** / R **−0.04515** | L −0.03142 / R −0.02672 | PASS |
| 4 | wing residual | 0.014668 | **0.011362** (−22.5%, i.e. *improved*) | PASS |
| 5 | folded pitch | −7.8° | **−35.2°** | PASS |
| 7 | blade normal (reported) | L 34.4° / R 30.7° | L 10.4° / R 5.9° | (reported) |

fly0's control penetration of **−0.0444 / −0.0452** reproduces the spec's
measured "median ~0.043 model units" exactly, so the defect this stage exists
to fix is confirmed on the female, and the fit removes about a third of it
(not the whole of it — the treatment is still ~20× the model's own −0.0013
grazing value).

### Invariants — checked before believing any of the above

* `WingX_V12`–`WingX_V13` both sit on the `wing_left`/`wing_right` body, so
  their FK separation is a **rigid length**. Drift between control and
  treatment: **exactly 0.0** on both wings, both flies, every arm.
* Every qpos address except `wing_pitch_left` (9) and `wing_pitch_right` (12),
  both resolved BY JOINT NAME: **max \|Δ\| = 0.0**. `opt_mask` holds on real data.
* The same invariant in the DATA, on the triangulated `kp3d` (frames 0–1499,
  keypoints indexed by name out of `cfg.model.KP_NAMES`): `WingL_V12–V13`
  5.016 mm CV 9.1% / `WingR` 5.038 mm CV 5.9% (fly0); 4.907 mm CV 5.6% /
  5.065 mm CV 2.5% (fly1) — against the known-rigid `EyeL–EyeR` at CV 10.2% /
  2.7% on the same frames. So on **this** bout the wing vein is measured about
  as rigidly as head width, and left/right agree to 1–3%: the wing keypoints
  here are not scrambled. (The spec's "CV 30% male, 80–120% female" is not what
  this bout-fly pair measures over frames 0–1499.)

## 4. Coverage weight, on BOTH flies — the open question

24 sampled frames × 7 cameras, body basis held fixed at stride 8 so rows stay
comparable. `inside%` = projected wing vertices inside the SAM mask (should stay
high); `explained%` = mask area the body does not explain that a wing vertex is
within ~6 px of (the coverage term's own goal).

**fly0 (female)**

| arm | inside% | expl% | pitch median L / R | pen L / R | resid |
|---|---|---|---|---|---|
| control | 86.4 | 38.6 | −8.4 / −7.8 | −0.04442 / −0.04515 | 0.01467 |
| cov 0.0 | **92.3** | 45.7 | −32.4 / −35.5 | −0.03142 / −0.02672 | 0.01151 |
| cov 0.1 | **92.3** | 45.7 | −32.3 / −35.4 | −0.03134 / −0.02681 | 0.01151 |
| cov 0.3 | **92.3** | **45.8** | −32.4 / −35.2 | −0.03154 / −0.02746 | 0.01136 |
| cov 1.0 | 91.6 | 46.4 | −36.4 / **−30.7** | −0.02777 / −0.03095 | 0.01158 |

**fly1 (male)**

| arm | inside% | expl% | pitch median L / R | pen L / R | resid |
|---|---|---|---|---|---|
| control | 84.1 | 43.8 | −5.7 / −11.1 | −0.02301 / −0.03324 | 0.01106 |
| cov 0.0 | 92.4 | 47.6 | −18.9 / −32.0 | −0.00695 / −0.01236 | 0.01118 |
| cov 0.1 | 92.4 | 47.6 | −19.0 / −32.0 | −0.00695 / −0.01253 | 0.01118 |
| cov 0.3 | **92.5** | 47.7 | −19.1 / −32.0 | −0.00690 / −0.01235 | 0.01118 |
| cov 1.0 | **92.5** | **48.0** | −19.6 / −31.9 | −0.00694 / −0.01298 | 0.01118 |

**On real data the coverage term is INERT over 0.0 → 0.3.** Every metric agrees
to 3 significant figures on both flies, and the pipeline's own `dpitch` medians
differ by ≤ 0.4°. It is not "a tie inside the noise floor"; the two arms are
the *same pose*. At 1.0 it is no longer inert and it is not obviously better:
on fly0 it breaks the left/right symmetry (−36.4 / −30.7 against −32.4 / −35.5)
and gives back 0.7 pt of `inside%` and 0.004 of right-wing penetration for
0.6 pt of `explained%`; on fly1 it costs 0.0006 of right-wing penetration and
raises `hp_rms(wing_pitch_right)` 0.752 → 0.984.

### The synthetic ground-truth cells, including the one Task 5 left unpinned

25° perturbation; the error after refinement, and the ratio the 0.5 threshold
applies to. (`figures/2026-09-01-wing-mask-fit/diag_restcell.py`.)

| configuration | EXTENDED (`qpos0`) | REST (springref, where the defect lives) |
|---|---|---|
| both terms, huber 0 | 1.88° (0.075) | 1.63° (0.065) |
| **both terms, huber 8** (shipped) | **1.63° (0.065)** | **1.14° (0.046)** |
| containment only (huber irrelevant) | **0.53° (0.021)** | 2.49° (0.100) |
| coverage only, huber 0 | 6.62° (0.265) | **43.49° (1.740) FAIL** |
| coverage only, huber 8 | 7.45° (0.298) | **42.75° (1.710) FAIL** |

The unpinned cell now has a number, and it is worse than Task 5's pre-fix
15.12°: **coverage-only at the rest attitude moves the wing 18° FURTHER from
truth than doing nothing.** The per-target normalisation that repaired the
combined objective did not repair the term in isolation.

**Does the coverage term earn its place?** Split verdict, stated plainly:

* **On real data, no.** It changes nothing measurable between 0.0 and 0.3 on
  either fly, and is mildly harmful at 1.0.
* **In the synthetic fixture, yes, but only as a complement.** Combined with
  containment it halves the rest-attitude error (2.49° → 1.14° at huber 8) and
  cuts the extended-attitude error by 0.25° relative to *both*-at-huber-0 —
  though containment ALONE is the best extended-attitude configuration (0.53°).
  Alone it is actively harmful in the rest regime.
* The "anti-degeneracy insurance" argument that kept it is **not supported**:
  containment-only shows no degeneracy in either synthetic regime (0.53° /
  2.49°, both passing), and on the real bout `explained%` *rises* under
  containment-only (38.6 → 45.7 on fly0), which is the opposite of a wing
  hiding inside the body blob.

**Recommendation for Task 8: keep `coverage_weight: 0.3` with
`huber_delta: 8.0`, and cap it there.** It is free on real data (measured
inert on both flies) and it is the better half of the only ground-truth
comparison available. Do not raise it toward 1.0 — that is measurably worse on
both flies. If Task 8 prefers the simpler objective, setting it to 0.0 loses
nothing measurable on real data and only the synthetic 1.14° → 2.49°; that is a
defensible choice, but on the evidence 0.3 is the marginally better one and
costs nothing. **The coverage weight is not the reason criterion 1 fails** — it
fails identically at 0.0, 0.1, 0.3 and 1.0.

## 5. Why criterion 1 fails, and the lever that fixes it

The failure is a *rise*, not the rest prior's *loss*. Separating the two:
`hp_rms` of the CHANGE the fit applies is 2.164 for fly1 `wing_pitch_left`,
against a treatment total of 2.313 — i.e. essentially all the added
high-frequency content is the fit's own per-frame noise, not a reshaping of the
control's dynamics. Frame-to-frame step sizes say the same thing:

| | median \|Δpitch\| per frame | p99 | max |
|---|---|---|---|
| fly1 `wing_pitch_left` control | 0.358° | 2.33° | 3.2° |
| fly1 `wing_pitch_left` cov 0.3 | 1.057° | 11.50° | 27.9° |
| fly0 `wing_pitch_left` control | 0.048° | 1.39° | 7.9° |
| fly0 `wing_pitch_left` cov 0.3 | 0.995° | 23.72° | 36.3° |

Mechanism: the mask minimum is broad (spec risk 1, ±10–15°) and the temporal
term in `_refine_chunk` is `smooth_weight**2 * Σ(Δq)²` with `Δq` in **radians**
— at the shipped `smooth_weight = 0.005` that is a coefficient of 2.5e−5
against a pixel-scale mask cost, i.e. effectively no temporal coupling at all,
so the fit is independent per frame.

**Probe (fly1, coverage 0.3, only `smooth_weight` varied):**

| smooth_weight | pitchL hp_rms | pitchR hp_rms | pen L / R | resid | pitch med L / R | inside% | expl% | crit 1 |
|---|---|---|---|---|---|---|---|---|
| control | 0.6002 | 0.5580 | −0.02301 / −0.03324 | 0.01106 | −5.7 / −11.1 | 84.1 | 43.8 | — |
| 0.005 (shipped) | 2.3138 | 0.7516 | −0.00689 / −0.01235 | 0.01118 | −19.1 / −32.0 | 92.5 | 47.7 | FAIL (3.85×) |
| 0.5 | 2.3143 | 0.7515 | −0.00689 / −0.01235 | 0.01118 | −19.1 / −32.0 | 92.5 | 47.7 | FAIL |
| 5 | 2.0265 | 0.7355 | −0.00709 / −0.01235 | 0.01118 | −19.0 / −32.0 | 92.5 | 47.7 | FAIL (3.38×) |
| 50 | **0.3685** | 0.3372 | −0.00666 / −0.01241 | 0.01110 | −18.7 / −31.3 | 92.5 | 47.7 | FAIL (0.61×) |

The criterion-1 window is **bracketed**: ≤5 fails high (jitter added), 50 fails
low (real dynamics smoothed away). And crucially **every other acceptance
number is flat across the entire 0.005 → 50 range** — `inside%` 92.5,
`explained%` 47.7, penetration and residual unchanged to 4 decimals. So the
temporal term buys criterion 1 at literally zero cost to the mask fit.

**Fine sweep, 10 / 20 / 30, on BOTH flies — and 30 passes everything.**

fly1 (male, the fly criterion 1 applies to):

| smooth_weight | pitchL hp_rms | ratio | pen L / R | resid | pitch med L / R | inside% | expl% | crit 1 |
|---|---|---|---|---|---|---|---|---|
| control | 0.6002 | — | −0.02301 / −0.03324 | 0.01106 | −5.7 / −11.1 | 84.1 | 43.8 | — |
| 10 | 1.5759 | 2.63× | −0.00685 / −0.01234 | 0.01118 | −19.0 / −32.0 | 92.5 | 47.7 | FAIL |
| 20 | 0.9015 | 1.50× | −0.00661 / −0.01222 | 0.01113 | −18.9 / −31.9 | 92.5 | 47.7 | FAIL |
| **30** | **0.6088** | **1.014×** | −0.00652 / −0.01245 | 0.01112 | −18.8 / −31.7 | **92.5** | **47.7** | **PASS** |

fly0 (female; criterion 1 is N/A, but the jitter is worse there and the mask
metrics are what matter):

| smooth_weight | pitchL / pitchR hp_rms | pen L / R | resid | pitch med L / R | inside% | expl% |
|---|---|---|---|---|---|---|
| control | 0.279 / 0.151 | −0.04442 / −0.04515 | 0.01467 | −8.4 / −7.8 | 86.4 | 38.6 |
| 0.005 (shipped) | 2.579 / 2.880 | −0.03154 / −0.02746 | 0.01136 | −32.4 / −35.2 | 92.3 | 45.8 |
| 10 | 1.598 / 1.223 | −0.03143 / −0.02703 | 0.01133 | −32.4 / −35.4 | 92.6 | 46.1 |
| 20 | 1.158 / 0.592 | −0.03133 / −0.02717 | 0.01132 | −32.6 / −35.3 | 92.7 | 46.2 |
| **30** | **0.811 / 0.429** | −0.03137 / −0.02726 | 0.01130 | −32.7 / −35.3 | **92.7** | **46.2** |

**`smooth_weight = 30` with `coverage_weight = 0.3`, `huber_delta = 8.0` passes
all six acceptance criteria on both flies**, cuts the female's added jitter by
3×, and is *marginally better* than the shipped `smooth_weight` on every mask
metric on fly0 (`inside%` 92.3 → 92.7, `explained%` 45.8 → 46.2). Nothing is
traded away for it.

**Setting the default is Task 8's, not Task 7's.** Recorded here so the FAIL is
not read as "the mask objective is wrong": the objective is right, the temporal
regularisation is three to four orders of magnitude too weak. Note the value is
scale-dependent — the term is `smooth_weight**2 * Σ(Δq)²` with `Δq` in radians
against a raw-pixel mask cost — so 30 is specific to this cost normalisation and
should be re-derived, not copied, if the mask cost is ever rescaled.

## 6. What the figures show, read back

Expectations were written into `scripts/viz/wing_fit_7cam.py`'s docstring
before anything was generated; each PNG was then opened and read.

**Stated:** (a) on fly0 both blades should lie *flatter* along the abdomen in
the treatment column and fill the mask sliver without spilling outside the
outline; (b) fly1's non-singing wing likewise; (c) **fly1's singing wing should
look essentially unchanged**; (d) in every row the render must sit inside *that
row's own* mask outline; (e) each panel must name the pose file it draws.

**Read back — agrees, with one qualification.**

* (e) holds: the columns report `qpos_refined.npz` vs `qpos_wingfit.npz[qpos]`,
  and the script printed `control and treatment qpos identical: False`. The
  silent failure mode (drawing the control twice) did not occur.
* (a) `wingfit_7cam_bout28_fly0_t750.png`, `Cam2012630` and `Cam2012853`: the
  control renders the female's two wings **splayed into a wide V** with the
  abdomen visible between them and the blades poking outside the outline; the
  treatment closes them into a single narrow overlapping sheet that matches the
  real frame's folded-wing silhouette and sits inside the outline. This is the
  clearest confirmation in the set.
* (b) `wingfit_7cam_bout28_fly1_t467.png` (a genuinely unilateral singing
  frame: `wing_yaw_left` 39.3° extended, `wing_yaw_right` 85.9° = at the stop),
  `Cam2012853`: the folded right wing stands up off the thorax and outside the
  outline in the control and lies flat against it in the treatment.
  `Cam2012857`, a near-lateral view, is the strongest single panel — the folded
  blade goes from face-on-and-outside-the-mask to edge-on-and-inside.
  `Cam2012861` on fly1 t=750 shows the same thing on the extended wing seen
  edge-on: the real wing is a razor-thin line, the control renders a
  visible-width blade, the treatment reproduces the line.
* (c) **holds on unilateral frames, not on the bout median.** At t=467/707 the
  singing wing moves −0.6 → −6.9° and −0.4 → −10.3° while the folded wing moves
  −17.2 → −42.2° and −17.9 → −40.9°: self-targeting, as the spec claims. But
  the *median over frames 0–1499* moves the left wing −5.7 → −19.1°, because
  most frames are not cleanly unilateral (mean `|yawL−yawR|` is 28°, and at
  t=750 it is 23.6° with both wings part-extended). The spec's "leaves the
  singer where it already is" is true of the song itself and an overstatement
  of the bout as a whole.
* (d) holds in all seven rows on both flies; no row shows the render drifting
  outside its own outline, which is what a crossed camera axis would produce.

**Two honest caveats on "check the hard case".** (i) The script's "worst mask
coverage" frame picker degenerated on this bout: **every frame in 0–1499 has
7/7 valid cameras for both flies**, so `argmin` returned frame 0. There is no
mask-dropout frame here to look at. The wall / dark-background hard case is
nonetheless present *visually* — `Cam2012857` and `Cam2012861` show the female
pressed against a wall in near-darkness, and those are the rows where the
control's blade most clearly juts outside the outline. (ii) Fly identity is
taken from the pipeline's canonicalisation (male = fly1) and confirmed
behaviourally — fly1 shows unilateral wing extension (`|yawL−yawR|` 28° mean,
46° on the frames used, with the folded wing pinned at the joint stop) and fly0
does not (3.5°). **This recording has no `sex.json`**, so that is an inference
from behaviour, not a reviewed label.

**The figure self-checks before drawing.** FK at the control qpos, pushed
through the stored bridge, must reproduce `outputs.h5:kp3d_mm` — the array the
rigcam similarity is fitted against — or the script refuses. Measured **0.2318
mm (0.10% of the 235.87 mm span)** on fly0 and **0.2263 mm (0.11%)** on fly1.
That check caught a real bug in the first version: the rig model was built from
the raw XML, so the `tracking[...]` sites sat at NOMINAL offsets while `kp3d_mm`
was built at the run's FITTED ones, and the similarity was fitted across the
mismatch. Fixed by applying `stac_ik.h5:offsets` to the sites by name. The
rigcam pose is also recovered ONCE, from the control arm, and reused for both
columns — otherwise the viewpoint moves between them (the wing markers enter the
Umeyama fit) and a difference could be where the reader is standing.

`pitch_traces.png` was generated to test one specific claim — that the
criterion-1 rise is added jitter rather than lost dynamics. **Stated
expectation:** the treatment trace should be the control shifted down ~25° plus
frame-to-frame hash, with the control visibly smoother. **Read back:** exactly
that. Grey (control) is smooth in all four panels; green (treatment) tracks the
same slow structure — including the step at frame ~720 in fly1's right wing —
but carries dense per-frame wobble plus occasional 30–40° single-frame
excursions on fly0. Neither arm goes anywhere near the springref line at
−57.3°, and the treatment sits mostly inside the −20…−40° measured mask
optimum: the stage is not reproducing the rest prior's error.

## 7. Provenance of the spec's "control 0.3615"

That number is **not** what this pipeline's pose gives. It comes from
`scripts/analysis/wing_pitch_rest_prior_ab.py`'s own weight-0 arm, which
re-solves STAC with `JaxlsBatchSolver(n_iter=50, smooth_weight=0.1)` — twenty
times the pipeline's `cfg.ik.smooth_weight = 0.005`. Re-running that script at
weight 0 on this bout-fly reproduces its family of numbers
(`wing_pitch_left.hp_rms = 0.3437`, `|yawL−yawR|` 40.7° against the spec's
"38 → 39"), while the shipped `qpos_refined.npz` over the same frames 0–599
gives **0.726**. Gating criterion 1 on the absolute 0.3615 would therefore have
scored the *unmodified control* as a 100% failure. All gates above are ratios to
this run's own control, which is the falsifiable form of the criterion.

## 8. Regenerating

GPU node, `unset LD_LIBRARY_PATH` first. Everything writes to
`figures/2026-09-01-wing-mask-fit/`; nothing writes to the processed tree.

```bash
BOUT=/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/pose/bouts/bout_00028

# coverage sweep + full acceptance table, per fly (~9 min each on one L40S)
python scripts/analysis/wing_mask_fit_ab.py --bout-dir $BOUT --flies 0 \
    --coverage-weights 0.0,0.1,0.3,1.0 --t0 0 --nt 1500 --mask-frames 24 \
    --out figures/2026-09-01-wing-mask-fit/sweep_fly0
python scripts/analysis/wing_mask_fit_ab.py --bout-dir $BOUT --flies 1 \
    --coverage-weights 0.0,0.1,0.3,1.0 --t0 0 --nt 1500 --mask-frames 24 \
    --out figures/2026-09-01-wing-mask-fit/sweep_fly1

# any other knob (this is the smooth_weight probe of §5)
python scripts/analysis/wing_mask_fit_ab.py --bout-dir $BOUT --flies 1 \
    --arm 'sm0.005=wing_mask_fit.smooth_weight=0.005' \
    --arm 'sm5=wing_mask_fit.smooth_weight=5.0' \
    --arm 'sm50=wing_mask_fit.smooth_weight=50.0' \
    --t0 0 --nt 1500 --mask-frames 24 \
    --out figures/2026-09-01-wing-mask-fit/smoothprobe

# re-score saved poses with a new metric, without re-running the fits
python scripts/analysis/wing_mask_fit_ab.py --bout-dir $BOUT --flies 0,1 \
    --from-arms 'figures/2026-09-01-wing-mask-fit/sweep_fly{fly}/arms_fly{fly}.npz' \
    --t0 0 --nt 1500 --mask-frames 24 \
    --out figures/2026-09-01-wing-mask-fit/rescore

# the 7-camera acceptance figure (MUJOCO_GL=egl is set by the script)
python scripts/viz/wing_fit_7cam.py \
    --run /gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/pose \
    --bout 28 --fly 0 --frames auto \
    --treat figures/2026-09-01-wing-mask-fit/sweep_fly0/arms_fly0.npz --treat-key q_cov0.3 \
    --treat-label "wing-mask fit (cov 0.3, shipped)" \
    --out figures/2026-09-01-wing-mask-fit
python scripts/viz/wing_fit_7cam.py ... --fly 1 --frames 467,1085 \
    --treat figures/2026-09-01-wing-mask-fit/sweep_fly1/arms_fly1.npz --treat-key q_cov0.3 ...

# the full pipeline stage, on a SCRATCH run root copied from the original tree
python scripts/run_bout.py ++bout_ids=28 ++wing_mask_fit.enabled=true \
    ++pipeline.allow_stale_kp3d=true ++outputs.overlay=false \
    ++outputs.out=<scratch>/pose ++run_id=task7_wingfit
```

The one-off diagnostics (`diag_restcell.py`, `diag_pitch_trace.py`) live beside
their outputs in `figures/2026-09-01-wing-mask-fit/`.
