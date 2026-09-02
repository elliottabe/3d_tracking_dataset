# Wing-pitch refinement against the SAM masks — acceptance measurement

*2026-09-01. Session0 `2025_10_20_13_20_04`, bout 28, **frames 0–1499**, both
flies. Spec: `docs/specs/2026-09-01-wing-orientation-from-masks-design.md`.
Scorecard: `scorecard.json` beside this file. Figures (gitignored):
`figures/2026-09-01-wing-mask-fit/` — regeneration commands at the end.*

**DECISION (Task 8): `wing_mask_fit.enabled` stays `false`.** The stage gets
wing PLACEMENT right and wing FINE MOTION wrong. It moves both flies' wings from
~−8° into the measured −20…−40° optimum band, reduces wing/abdomen penetration
on both wings of both flies, does not degrade the marker fit, and per epoch it is
self-targeting — and it **destroys the bilaterally phase-locked courtship song in
wing pitch, at every `smooth_weight` measured**. §9 is the decision and the
evidence that forced it; **it supersedes the recommendations in §4, §5 and
§6(c) below.**

The spec's acceptance criterion 1 was **blind** to that failure — it scored the
shipped config a clean `hp_rms` 1.00× PASS on the very epoch where the song was
destroyed — and has been replaced by a bilateral phase-lock / coherence test:
`docs/specs/2026-09-01-wing-orientation-from-masks-design.md` §5.0, §5.1′,
§5.3′ and §8.

*§1–§8 are Task 7's measurement, kept as written except for arithmetic
corrections (the residual column in §4, the fly0 penetration row in §3) and the
supersession banners. What each of them got wrong, and why, is §9.*

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
| 1 | `hp_rms(wing_pitch_left)`, singing wing, within 20% | 0.6002 | **2.3133** (ratio **3.85**) | **FAIL** — and this row uses ONE whole-bout singer label; per epoch it is 3.77× / **1.00×** (§9.1), and the criterion itself is retired (§9.2) |
| 2 | `\|yawL−yawR\|` mean + pulse stats within noise | 28.07°, hp_rms 2.1188, kurt 3.05, 45 peaks | **bit-identical** (max \|Δyaw\| = 0.0 rad) | PASS |
| 3 | penetration → −0.0013 on BOTH wings | L −0.02301 / R −0.03324 | L **−0.00690** / R **−0.01235** | direction met, **74% / 65%** of the gap (§9.3) |
| 4 | wing-keypoint residual rise ≤ 20% | 0.011062 | 0.011181 (**+1.1%**) | PASS |
| 5 | folded-wing pitch near −20…−40°, not −57.3° | −11.1° | **−32.0°** | PASS |
| 6 | 7-camera figure, female + male's non-singing wing | — | see §6 | PASS |
| 7 | blade normal — REPORTED, NEVER GATED | L 12.2° / R 15.8° | L 7.7° / R 2.2° | (reported) |

### fly0 (FEMALE — the hard fly; not singing, both wings folded), coverage_weight 0.3

| # | criterion | control | treatment | verdict |
|---|---|---|---|---|
| 1 | per-DOF song guard | — | — | **N/A** (`\|yawL−yawR\|` = 3.52°: no wing extension) |
| 2 | song unchanged | 3.52° | bit-identical | PASS |
| 3 | penetration | L **−0.04442** / R **−0.04515** | L **−0.03154** / R **−0.02746** | direction met, **30% / 40%** of the gap (§9.3) |
| 4 | wing residual | 0.014668 | **0.011362** (−22.5%, i.e. *improved*) | PASS |
| 5 | folded pitch | −7.8° | **−35.2°** | PASS |
| 7 | blade normal (reported) | L 34.4° / R 30.7° | L 10.4° / R 5.9° | (reported) |

fly0's control penetration of **−0.0444 / −0.0452** reproduces the spec's
measured "median ~0.043 model units" exactly, so the defect this stage exists
to fix is confirmed on the female, and the fit removes about a third of it
(not the whole of it — the treatment is still ~20× the model's own −0.0013
grazing value).

**Corrected 2026-09-01 (Task 8).** The fly0 criterion-3 row printed
`L −0.03142 / R −0.02672`, which is the **cov 0.0** arm, under a
`coverage_weight 0.3` header; the cov 0.3 values are `−0.03154 / −0.02746` and
are what the row now carries. The residual and folded-pitch cells in that same
row were already cov 0.3. Both flies' criterion-3 verdicts also now carry the
gap-closure fraction: the criterion was **implemented as a sign test**
(`a1 > a0`), which a 1e−9 improvement passes, so "PASS" overstated it — see
§9.3.

### Invariants — checked before believing any of the above

* `WingX_V12`–`WingX_V13` both sit on the `wing_left`/`wing_right` body, so
  their FK separation is a **rigid length**. Drift between control and
  treatment: **exactly 0.0** on both wings, both flies, every arm.
  **This is a TAUTOLOGY, not an invariant this stage could violate** (noted by
  Task 8): both sites are on the same body, so their FK separation is a model
  constant under *any* pose — the measured CV is 2.8e−15, machine epsilon. Take
  no comfort from it. The genuine invariant is the data-side `kp3d` one, two
  bullets down, which was done correctly.
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

> **SUPERSEDED IN PART (Task 8, §9.4): `coverage_weight` is recorded UNDECIDED,
> not "inert then keep 0.3".** The term is not inert — on the female it shifts
> `hp_rms` 2.4% and right-wing penetration 2.8% (the wrong way) between cov 0.0
> and cov 0.3, and at cov 1.0 the female's added jitter nearly doubles, which
> this section omitted. Its only synthetic win is at the REST attitude and is
> offset by an almost equal loss at the EXTENDED attitude. If a tie-break is
> forced the evidence tips to **0.0**. `huber_delta` must stay > 0 either way.
>
> **Arithmetic corrected below.** The `resid` column was transcribed wrong in
> both tables: fly0 cov 0.0 / cov 0.1 / cov 1.0 read `0.01151 / 0.01151 /
> 0.01158` where `scorecard.json` has `0.011494 / 0.011462 / 0.011624`, and
> every fly1 row read `0.01118` where the values are `0.011192 / 0.011193 /
> 0.011181 / 0.011194`. The conclusions in this section do not turn on them
> (the spread is < 0.3%), but the printed digits were not the measured ones.

24 sampled frames × 7 cameras, body basis held fixed at stride 8 so rows stay
comparable. `inside%` = projected wing vertices inside the SAM mask (should stay
high); `explained%` = mask area the body does not explain that a wing vertex is
within ~6 px of (the coverage term's own goal).

**fly0 (female)**

| arm | inside% | expl% | pitch median L / R | pen L / R | resid |
|---|---|---|---|---|---|
| control | 86.4 | 38.6 | −8.4 / −7.8 | −0.04442 / −0.04515 | 0.01467 |
| cov 0.0 | **92.3** | 45.7 | −32.4 / −35.5 | −0.03142 / −0.02672 | 0.011494 |
| cov 0.1 | **92.3** | 45.7 | −32.3 / −35.4 | −0.03134 / −0.02681 | 0.011462 |
| cov 0.3 | **92.3** | **45.8** | −32.4 / −35.2 | −0.03154 / −0.02746 | 0.011362 |
| cov 1.0 | 91.6 | 46.4 | −36.4 / **−30.7** | −0.02777 / −0.03095 | 0.011624 |

**fly1 (male)**

| arm | inside% | expl% | pitch median L / R | pen L / R | resid |
|---|---|---|---|---|---|
| control | 84.1 | 43.8 | −5.7 / −11.1 | −0.02301 / −0.03324 | 0.01106 |
| cov 0.0 | 92.4 | 47.6 | −18.9 / −32.0 | −0.00695 / −0.01236 | 0.011192 |
| cov 0.1 | 92.4 | 47.6 | −19.0 / −32.0 | −0.00695 / −0.01253 | 0.011193 |
| cov 0.3 | **92.5** | 47.7 | −19.1 / −32.0 | −0.00690 / −0.01235 | 0.011181 |
| cov 1.0 | **92.5** | **48.0** | −19.6 / −31.9 | −0.00694 / −0.01298 | 0.011194 |

**On real data the coverage term is INERT over 0.0 → 0.3** on the headline
metrics: every metric in these tables agrees to 3 significant figures on both
flies, and the pipeline's own `dpitch` medians differ by ≤ 0.4°.
**Not inert everywhere, though** (Task 8): on the female `hp_rms(wing_pitch_left)`
goes 2.5188 → 2.5785 (+2.4%) and right-wing penetration −0.026717 → −0.027461
(2.8% *worse*) between cov 0.0 and cov 0.3, and at cov 1.0 the female's
`hp_rms` nearly doubles (2.58 → 5.09). Every real-data difference the term makes
on the hard fly is adverse. At 1.0 it is no longer inert and it is not obviously better:
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

~~**Recommendation for Task 8: keep `coverage_weight: 0.3`.**~~ **WITHDRAWN by
Task 8 — recorded UNDECIDED, leaning 0.0; see §9.4.** What survives from this
section: do **not** raise it toward 1.0 (measurably worse on both flies, and it
nearly doubles the female's jitter), `huber_delta` must stay **> 0** whatever
the weight, and **the coverage weight is not the reason criterion 1 fails** — it
fails identically at 0.0, 0.1, 0.3 and 1.0.

## 5. Why criterion 1 fails, and the lever that ~~fixes~~ does not fix it

> **SUPERSEDED (Task 8, §9.1–§9.2). `smooth_weight = 30` is NOT the fix and was
> not adopted.** Two things below are wrong. (i) The "singing wing" is scored
> with one label for the whole bout, but fly1 **swaps extended wing at frame
> 869**; per epoch, weight 30 trades epoch 1's failure for a **37% loss of
> dynamics on epoch 2** — the rejected rest prior's own failure mode. (ii) The
> `hp_rms` rise is not simply "the fit's jitter": measured against bilateral
> phase lock, the fit **replaces** a real, song-frequency, phase-locked
> oscillation that the control already carried, and it does so at *both*
> weights. Everything below about the temporal term's magnitude (2.5e−5, no
> effective coupling) and about placement being unaffected across 0.005 → 50
> still stands; the recommendation does not.

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

~~**`smooth_weight = 30` passes all six acceptance criteria on both flies.**~~
**WITHDRAWN.** It passes them only when criterion 1 is scored with a single
whole-bout "singing wing" label. Scored per epoch it **fails** — 0.63× on epoch
2, i.e. 37% of the extended wing's real dynamics removed — and it fails the
replacement phase-lock criterion on **both** epochs (coherence 0.008 / 0.139
against a control of 0.418 / 0.852). It does cut the female's added jitter 3×
and is marginally better than the shipped weight on every fly0 mask metric
(`inside%` 92.3 → 92.7, `explained%` 45.8 → 46.2); that is a true statement
about >40 Hz *amplitude* and it is not a fix. See §9.

What still stands: the FAIL is not "the mask objective is wrong" — placement is
unaffected across the whole 0.005 → 50 range, so the objective puts the wing in
the right place and the temporal term is not the axis that protects the song.
Note the value is scale-dependent — the term is `smooth_weight**2 * Σ(Δq)²` with `Δq` in radians
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
  **RETRACTED (Task 8).** The overstatement was mine, not the spec's: the
  bout median mixes the two halves of a wing swap at frame 869. Scored per
  epoch the self-targeting property holds cleanly — the folded wing moves
  26–30° while the extended wing moves only 10–12° and lands at −12.5° / −19.3°,
  which is the spec's own measured extended-wing optimum of −10…−20°. See §9.1.
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

```bash
# the §9 song figures: re-scores saved poses only -- no GPU, no new fits.
# --plot takes ONE of all|pitch|alldof|delta|crit1|bilateral|song|pulses.
python scripts/analysis/wing_mask_fit_ab.py --bout-dir $BOUT --flies 0,1 \
    --from-arms 'figures/2026-09-01-wing-mask-fit/sweep_fly{fly}/arms_fly{fly}.npz' \
    --plot-arm 'sm30=figures/2026-09-01-wing-mask-fit/smoothfine_fly{fly}/arms_fly{fly}.npz:q_sm30' \
    --plot all --out figures/2026-09-01-wing-mask-fit/dofs
```

The one-off diagnostics (`diag_restcell.py`, `diag_pitch_trace.py`) live beside
their outputs in `figures/2026-09-01-wing-mask-fit/`.

---

## 9. Task 8 — the decision, and the evidence that forced it

**`wing_mask_fit.enabled` stays `false`.** `configs/pipeline.yaml` is unchanged
except for its comments, which now say what was measured.

Figures for this section (gitignored):
`figures/2026-09-01-wing-mask-fit/dofs/` — `bilateral_phase_lock.png`,
`song_or_noise.png`, `pulse_trains.png`, `pitch_traces_shipped_sw0.005.png`,
`pitch_traces_fix_sw30.png`, `crit1_jitter.png`, `alldof_traces_fly{0,1}.png`,
`delta_qpos_spectrum.png`. Numbers: `dofs/trace_stats.json`. All of it re-scores
the saved `arms_*.npz` — no GPU, no new fits, and the 44 control files still
`md5sum -c` OK.

### 9.1 The "singing wing" is not static: fly1 swaps at frame 869

| frames | yawL med | yawR med | \|L−R\| | left-extended fraction |
|---|---|---|---|---|
| 0–799 | 40–50° | 68–86° | 27–42° | **1.00** |
| 800–899 | 54.5° | 60.9° | 7.3° | 0.69 |
| 900–1499 | 63–77° | 48–58° | 6.6 → 27° | **0.00** |

A 101-frame majority filter puts the switch at **869**, and the transition is
smooth in every arm (|Δ| at the boundary 0.07–0.69° against each arm's own p99
step of 1.47–11.50°), so it is behaviour and no arm introduces a jump at it.
fly0 has no epoch with `|yawL−yawR| ≥ 10°` and is the **negative control**.

Everything below is scored per epoch. **This single defect inverted the earlier
conclusion about which configuration passes** — §5's `smooth_weight = 30` passes
criterion 1 only under a whole-bout singer label:

| epoch | extended wing | control | shipped 0.005 | sw 30 |
|---|---|---|---|---|
| E1 0–869 | `wing_pitch_left` | 0.7197 | 2.7121 (**3.77×**) | 0.6891 (0.96×) |
| E2 869–1500 | `wing_pitch_right` | 0.8054 | 0.8019 (**1.00×**) | 0.5045 (**0.63×**) |
| E2b 1000–1500 | `wing_pitch_right` | 0.8223 | 0.8090 (0.98×) | 0.5160 (0.63×) |

(`hp_rms`, fs 800 Hz assumed, 40 Hz high-pass. E2's first ~100 frames have
`|L−R|` 6.6° — the fly pauses between song bouts — so the 1000–1500 variant is
reported; it changes nothing.)

### 9.2 The control carries real song in wing pitch. The fit replaces it.

Four measurements, all pointing the same way.

**(a) Both wings oscillate during song.** The FOLDED wing's pitch carries a peak
at the *extended* wing's f0 in the control: peak/floor **34.6** (E1, folded
right, whose yaw sits exactly on the +85.94° joint stop) and **84.2** (E2, folded
left). One wing is pinned against extension, not silent.

**(b) The two wings are phase-locked — the decisive test.** Magnitude-squared
coherence of `hp(wing_pitch_left)` with `hp(wing_pitch_right)` at f0,
`nperseg` 128:

| epoch | 95% sig | control | shipped 0.005 | sw 30 |
|---|---|---|---|---|
| E1 | 0.24 | **0.418** @ −155° | 0.038 | 0.008 |
| E2 | 0.35 | **0.852** @ +162° | 0.070 | 0.139 |
| fly0 (does not sing) | 0.13 | 0.395 | 0.056 | 0.001 |

Robust across `nperseg` 64/128/256 (E2 control 0.79/0.85/0.92; both arms
0.04–0.18 throughout). The control's cross-correlation is a clean ±0.6
oscillation at the **6.4-frame** song period; both arms' stay within ±0.2 and
are aperiodic. fly0's control is aperiodic too (within ±0.2), and the f0 the
test picks on her is 43.7 Hz — an 18.3-frame period, not the song's.
*(Read back from
`dofs/bilateral_phase_lock.png` by Task 8: every number in this table is on the
figure, the control's sharp f0 peak and periodic cross-correlation are visible,
and the female's control pitch trace is flat.)*
The ~±160° phase is the *in-phase* signature here: both `wing_pitch_*` joints
hinge about their own body's local +y and the two wing bodies are mirror-placed
(quats `[0,−0.4031,0,−0.9152]` and `[0,0.9152,0,−0.4031]`), so bilaterally
symmetric motion appears near 180° in these joint coordinates.

**(c) The added content is not song.**

| epoch | metric | control | shipped 0.005 | sw 30 |
|---|---|---|---|---|
| E1 | P(f0) | 2.25e−2 | 1.53e−1 (**×6.8**) | 1.34e−2 |
| E1 | broadband floor | 3.79e−4 | 6.53e−3 (**×17**) | 1.37e−4 |
| E1 | peak/floor | 59.4 | **23.4** | 98.1 |
| E1 | MSC with the wing's own yaw | 0.835 | 0.511 | 0.412 |
| E1 | corr with control's hp(pitch) | 1.000 | 0.358 | 0.328 |
| E2 | P(f0) | 4.58e−2 | 9.16e−3 (**÷5**) | 3.60e−3 |
| E2 | peak/floor | **156.6** | **9.2** | 26.3 |
| E2 | MSC with the wing's own yaw | 0.968 | 0.290 | 0.312 |
| E2 | corr with control's hp(pitch) | 1.000 | **0.017** | **0.012** |
| fly0 | peak/floor | 6.4 | **1.0** (flat noise) | 46.6 |
| fly0 | hp_rms | 0.279 | 2.579 (×9.3) | 0.811 (×2.9) |

On **E1** song-band power at f0 really does rise ×6.8 under the fit — but the
floor rises ×17 with it, so peak/floor *falls*, coherence with the wing's own yaw
falls 0.84 → 0.51, and bilateral coherence collapses 0.42 → 0.04. More song-band
power, worse song. On **E2** there is nothing to defend: `hp_rms` is unchanged
(1.00×) while the song peak drops ×5, peak/floor goes 157 → 9, and the
correlation with the content it replaced is **0.017** — substitution, not
modification. **The female settles it:** with no song present the shipped fit
produces `hp_rms` ×9.3 at peak/floor **1.0**, a flat broadband spectrum — that is
what this stage does to the pitch DOF when there is nothing to recover, and it is
what it adds to fly1 on top of the song. At sw 30 the female instead gets
peak/floor 46.6: a manufactured spectral peak where the video has none.

**(d) Where are the pulses? Not delayed — buried, or removed.** One common
threshold (2× the control's own σ) so the counts compare:

| epoch | control | shipped 0.005 | sw 30 |
|---|---|---|---|
| E1 | 43 | **168** (chance match 97%) | 16 (9/43 kept) |
| E2 | 29 | 24 (10/29 kept, chance 19%) | **4** (0/29 kept) |
| fly0 | 16 | **425** (chance 100%) | 98 |

Median offset of every matched pulse is **+0.0 frames**: pulses are not shifted
later, they are drowned or deleted.

**And the criterion could not see any of it.** On E2 the shipped config scores
`hp_rms` **1.00× — a clean PASS** — in exactly the frames where (b), (c) and (d)
show the song destroyed. An amplitude test cannot distinguish preserved song from
substituted noise of equal RMS.

### 9.3 What the spec's criteria become

Amended in `docs/specs/2026-09-01-wing-orientation-from-masks-design.md`:

* **Criterion 1 → §5.1′, a bilateral phase-lock test**, scored per extended-wing
  epoch. Primary gate: `MSC(hp(pitch_L), hp(pitch_R))` at f0 ≥
  `max(0.8 × control, its own 95% significance level)`, reported at `nperseg`
  64/128/256. Supporting: song-band `peak/floor` ≥ 0.5× control (never primary —
  smoothing inflates it), content retention `corr(hp_treat, hp_control)` ≥ 0.5,
  and a pulse-count window of 0.5–2× control with |median offset| ≤ 1 frame.
  Negative control on a non-singing fly: no new spectral peak and `hp_rms` ≤ 2×
  (uncalibrated — no measured arm passes it). `hp_rms` becomes **reported, not
  gated**. Specificity comes from f0 *plus* a periodic cross-correlation: fly0's
  control also reaches MSC ~0.8 at high frequencies (two wings, one body, shared
  pose noise) but its cross-correlation stays within ±0.2 and is aperiodic.
* **Gates are ratios to the run's own control** (§7), never the absolute 0.3615.
* **Criterion 3 → §5.3′, a gap-closure fraction**, `(control − treatment) /
  (control − (−0.0013))`, gated at ≥ 50% on both wings of both flies. It was
  implemented as a sign test (`a1 > a0`), which a 1e−9 improvement passes.
  Measured at cov 0.3: fly0 **30%** (L) / **40%** (R), fly1 **74%** / **65%** —
  **direction met, target not met**; the 50% line is a judgement and is not
  bracketed by a passing measurement on the female.
* **The `v12v13` FK check is a tautology** (§3), not an invariant this stage
  could violate.
* **Criterion 2 passes by construction** (`opt_mask` freezes yaw exactly), so it
  is a check on the implementation, not on the song.

### 9.4 `coverage_weight`: UNDECIDED, leaning 0.0

Not "measured better", and not inert: on the female it moves `hp_rms` +2.4% and
right-wing penetration 2.8% the wrong way between 0.0 and 0.3, and at 1.0 the
female's jitter nearly doubles. Its only ground-truth win is at the **REST**
attitude (containment-only 2.49° → 1.14° with coverage at `huber_delta` 8) and
is offset by an almost equal loss at the **EXTENDED** attitude (0.53° → 1.63°) —
the attitude criterion 1′ exists to protect. Alone it is catastrophic at rest
(42.75–43.49°, 18° worse than doing nothing). Left at 0.3 on disk only because
the stage is off; that is inheritance, not a decision. `huber_delta` **must stay
> 0**.

### 9.5 Known limits of this evidence

* **The acceptance window hides the worst quarter on the hard fly.** Frames
  0–1499 of 2007 were scored. fly0's frames **1500–2006** have **54% of frames
  moving pitch > 45°, max 98°**, against max 42° and 0% inside the window.
  Female-specific; `smooth_weight = 30` does not help. Future acceptance runs
  must score the whole bout.
* **Performance: 74–97 s per bout-fly against a 60 s budget** (74.3 s for the
  isolated refinement, 85–97 s for the stage end to end). The measured,
  bit-identical fix — threading `sdf_stack_from_masks`, 34.4 s → 4.8 s at 16
  threads, projecting ~51 s — is deferred to Phase 2 of
  `docs/plans/2026-09-01-wing-orientation-from-masks.md`.
* **One bout, one recording, two flies** sharing a calibration, an `offsets.h5`
  and a `kp3d`; fly0's `kp3d.npz` is pre-gate (`allow_stale_kp3d=true`).
* **No `sex.json` on this recording** — male = fly1 is canonicalisation plus a
  behavioural check, not a reviewed label.
* **No mask-dropout hard case exists here**: every frame in the window has 7/7
  valid cameras for both flies.
* **`fs = 800 Hz` is assumed, not read from config** (`cfg.recording` has no fps
  key). The fps-independent form of the song claim is the **6.4-frame period**;
  f0 = 125 Hz only at the assumed fs. Ratios are unaffected — control and
  treatment share the constant.

### 9.6 The decision, and what would reverse it

Keep the stage OFF. Ship neither `smooth_weight` 0.005 nor 30: both destroy the
phase lock, and the temporal weight is the wrong axis — it controls *amplitude*
of high-frequency content, and the defect is its *content*. Placement and fine
motion are separable (placement is identical at 0.005 and 30, and `lp_std` 5.14
→ 11.40 → 11.56 shows the slow structure survives in every arm), so the
recommended direction is to **stop the stage carrying high-frequency content at
all**: supply a slowly-varying pitch correction and leave the song to the marker
solve. **Not implemented — out of scope here**, and it must be measured against
the amended criterion 1′, over the whole bout, on more than one recording,
before anything is enabled.
