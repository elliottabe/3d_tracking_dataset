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

---

## 10. Task 9 — the redesign: a band-limited correction (`param_mode`)

§9 left the stage OFF because it gets **placement right and fine motion wrong**,
and recorded the direction: stop the stage carrying high-frequency content at
all. That is now implemented as a new `wing_mask_fit.param_mode`, measured, and
this section is the result. **`param_mode: free` remains the default and the
stage remains `enabled: false`** — nothing about the shipped configuration
changed, so §1–§9 stay reproducible.

### 10.1 What was built

| mode | what the optimiser varies | how the song survives |
|---|---|---|
| `free` (default) | one `Δpitch` per frame per DOF | it does not — §9 |
| `spline` | a KNOT VECTOR, `knot_spacing` frames apart; `Δpitch = knot_basis(F,K) @ θ`, piecewise linear | **by construction**: at K = 32 the basis is an order of magnitude below the 6.4-FRAME song period, so no θ can put power in the song band |
| `lowpass` | solve as `free`, then keep only `lowpass(q_fit − q_init)` (zero-phase 4th-order Butterworth at `1/(2K)` cycles/frame) | **by construction**: `hp(q_out) = hp(q_init)` up to the filter's stop-band leak |

`lowpass` is the **honest control for `spline`**: it preserves the song equally
well but spends the whole solve on content it then discards. If the two land in
the same place, `spline`'s claim to be *better* rather than merely *quieter* is
unsupported.

Chunk boundaries are part of the spline, not an afterthought: three chunks
splined independently and stitched have a STEP at every boundary, and a step is
broadband. Each chunk's first knot is clamped to the previous chunk's last, and
`knot_basis` puts its last knot at local position F (== the next chunk's local
0), so the whole trajectory is ONE globally continuous piecewise-linear
function. `frame_chunk` must therefore be a multiple of `knot_spacing` in this
mode, which keeps the global knot grid uniform and `frame_chunk` a pure
performance knob. Pinned by
`test_spline_mode_confines_the_correction_to_a_GLOBAL_knot_basis` (which fails
at 2.81e−03 of the correction's amplitude when the anchor is removed).

### 10.2 The correction really is band-limited, measured on real data

`hp_rms` of the CORRECTION (`q_arm − q_control`), scorer's own 40 Hz high pass,
degrees, over the longest contiguous FINITE run of each window. The control's
own song amplitude is beside it for scale.

| fly / window / DOF | control | free | spline32 | spline64 | lowpass32 |
|---|---|---|---|---|---|
| fly1 0–1500 `wing_pitch_left` | 0.600 | 2.166 | **0.050** | 0.020 | **0.003** |
| fly1 0–1500 `wing_pitch_right` | 0.558 | 0.894 | 0.023 | 0.013 | 0.002 |
| fly0 0–1500 `wing_pitch_left` | 0.279 | 2.579 | 0.074 | 0.030 | 0.012 |
| fly0 1500–1710 `wing_pitch_left` | 1.993 | 6.550 | 0.583 | 0.241 | 0.072 |
| fly0 1500–1710 `wing_pitch_right` | 1.172 | 15.870 | 0.975 | 0.471 | 0.099 |

Max |2nd difference| of the correction: free 5.5–234 deg, spline32 0.17–6.8 deg,
lowpass32 0.004–0.19 deg. So the residual high-frequency content of `spline` is
its own piecewise-linear KINKS — a C0 basis has a comb at the knot rate and its
harmonics — and `lowpass`, which has no kinks, is 5–10× cleaner.

**But the clamp is NOT inactive, and that makes the guarantee conditional.**
Measured in 11.2 with a global-span projection: `spline64` hits the −72.77 deg
stop on fly0's left wing for **17 consecutive frames (1720–1736)** and `free`
hits it on 12, and those 17 frames take the `spline64` correction **11.45 deg —
0.146 of its own amplitude — OUT of the knot span it lies in to 4.5e−08
everywhere else**. The clip is a per-frame nonlinearity applied AFTER the basis,
so a clamped frame has not been shown band-limited. `wing_mask_fit_bout` now
counts these into `n_clamp_hits` and the stage log prints them. `spline32` and
`lowpass32` hit no stop on this bout.

**A trap this measurement fell into first, recorded so it is not repeated.**
fly0 has 252 non-finite qpos rows, which `refine_wing_pitch` hands back
untouched, so the correction steps from tens of degrees to zero and back at every
one of them. Filling them with 0 and high-passing measures that step train, not
the correction: the first run of `diag_correction_spectrum.py` reported a 3.2 deg
high-frequency residual for the LOWPASS arm, whose correction is a 4th-order
Butterworth output and cannot have one. Score a contiguous finite run.

### 10.3 Criterion 1' — the song survives, on every epoch of the singing fly

Scored per extended-wing epoch with `criterion1prime()` in
`scripts/analysis/wing_mask_fit_ab.py`. **The unmodified control was scored as
its own arm** (`control_copy`), which is the only way to know the gate is
falsifiable rather than unpassable.

1'b PRIMARY, `MSC(hp(pitch_L), hp(pitch_R))` at f0, whole bout, fly1:

| epoch | window | control | gate | free | spline32 | lowpass32 |
|---|---|---|---|---|---|---|
| E1 0–869 | n64 | 0.430 | 0.344 | 0.038 | **0.431** | 0.430 |
| E1 0–869 | n128 | 0.418 | 0.334 | 0.039 | **0.417** | 0.418 |
| E1 0–869 | n256 | 0.317 | — | 0.147 | 0.317 | 0.317 |
| E2 869–1515 | n64 | 0.783 | 0.627 | 0.087 | **0.785** | 0.783 |
| E2 869–1515 | n128 | 0.861 | 0.689 | 0.111 | **0.861** | 0.861 |
| E2 869–1515 | n256 | 0.944 | 0.755 | 0.275 | **0.945** | 0.944 |
| E3 1515–1943 | n128 | 0.712 | 0.570 | 0.043 | **0.712** | 0.712 |

Verdict: `control_copy` **PASS**, `free` **FAIL**, `spline32` / `spline64` /
`lowpass32` **PASS**, on all three epochs. Supporting tests agree: 1'd
`corr(hp_treat, hp_ctrl)` **0.996–1.000** for the band-limited arms against
0.017–0.370 for free; 1'e pulse counts 30/30, 45/43 (spline32; lowpass32 43/43), 10/10 against free's 168/43
and 66/10; xcorr periodicity 0.89–1.00 against free's 0.19–0.74.

**This is a trivial pass and must be read as one.** `corr = 0.999` says the
band-limited arms' high-frequency pitch *is* the control's, so criterion 1'
cannot be evidence that the fit is GOOD — only that it is not destructive. The
question the redesign has to answer is criteria 2–6, in 10.5.

**A DEFECT IN THE CRITERION, found by scoring the control against itself.**
§5.1'b says report at `nperseg` 64/128/256 and require the verdict at all three;
§5.1'a says score only where the CONTROL's own coherence is significant. On fly1
E1 those conflict: `nperseg` 256 over 869 frames leaves K = 5 segments and a 0.53
significance floor, and **the control's own MSC there is 0.317 — below its own
floor**. Requiring that window fails the *unmodified control*, exactly the
pathology §5.0 forbids. Ruling: 1'a applies PER WINDOW; a window in which the
control is not significant is reported and excluded, never silently passed.
`criterion1prime()` implements that and prints `n.a.(ctrl 0.317 < sig 0.53)`.

Second scorer defect fixed: a window containing non-finite qpos makes `filtfilt`
return an all-NaN trace, every `>=` against NaN is False, and the criterion
printed a confident FAIL for every arm INCLUDING `control_copy`. It now prints
`UNDEFINED (non-finite qpos in this window)`. fly0 has exactly one long finite
run, frames 0–1710, so every fly0 song number here is scored on that.

### 10.4 Criterion 1'f, the non-singing female — where the two modes differ

Ratios to fly0's own control, on her finite run:

| window | quantity | control_copy | free | spline32 | spline64 | lowpass32 |
|---|---|---|---|---|---|---|
| 0–1500 | `peak/floor` ratio | 1.000 | 0.154 | **1.561** | 1.110 | **1.000** |
| 0–1500 | `hp_rms` ratio | 1.000 | **9.25×** | 1.022 | 1.002 | 1.000 |
| 0–1710 | `peak/floor` ratio | 1.000 | 0.413 | 1.128 | 1.057 | 1.000 |
| 0–1710 | `hp_rms` ratio | 1.000 | **4.69×** | 1.045 | 1.007 | 1.001 |
| 1500–1710 | `peak/floor` ratio | 1.000 | 0.566 | 1.296 | 1.096 | 1.000 |
| 1500–1710 | `hp_rms` ratio | 1.000 | **3.63×** | 1.049 | 1.008 | 1.001 |

`free` fails on amplitude (9.25×), exactly as §9 recorded. `spline32` fixes the
amplitude (1.02–1.05×) but raises `peak/floor` 13–56%: its knot comb lands near
the 43.7 Hz "f0" the test picks on a fly that does not sing. `spline64` halves
that (6–11%). `lowpass32` does not do it at all.

**As literally written — "peak/floor must not rise above control's" — 1'f had no
tolerance band and only the identity transformation could pass it**: `lowpass32`
scores 3.9402 against 3.9392 at 0–1710 and was recorded FAIL by a 0.025% excess.
A criterion that fails a correct fit is worse than no criterion — the §5.0
pathology in a different clause. **11.1 calibrates the band by measurement and
re-scores; with it, every band-limited arm passes 1'f on all three windows and
the arms that should fail still do.**

### 10.5 Does it still fix the wings? Criteria 3'–5, every window

Criterion 3' is the gap-closure fraction `(control − treatment)/(control −
(−0.0013))`, gated at ≥ 50% on both wings. Full table in
`scorecard_redesign.json`.

| window | fly | arm | gap L | gap R | resid | pitch L/R | inside% | expl% |
|---|---|---|---|---|---|---|---|---|
| 0–1500 | fly0 | control | — | — | 0.0147 | −8.4 / −7.8 | 86.4 | 38.6 |
| 0–1500 | fly0 | free | 29.9% | 40.3% | 0.0114 | −32.4 / −35.2 | 92.3 | 45.8 |
| 0–1500 | fly0 | **spline32** | **31.1%** | **41.1%** | **0.0113** | −33.2 / −35.4 | **92.8** | **46.5** |
| 0–1500 | fly0 | spline64 | 31.9% | 41.9% | 0.0113 | −33.5 / −35.5 | 92.9 | 46.4 |
| 0–1500 | fly0 | lowpass32 | 29.4% | 41.0% | 0.0112 | −32.7 / −35.4 | 92.5 | 46.0 |
| 0–1500 | fly1 | control | — | — | 0.0111 | −5.7 / −11.1 | 84.1 | 43.8 |
| 0–1500 | fly1 | free | 74.2% | 65.4% | 0.0112 | −19.1 / −32.0 | 92.5 | 47.7 |
| 0–1500 | fly1 | spline32 | 73.9% | 61.2% | 0.0111 | −19.3 / −31.0 | 92.4 | 47.6 |
| 0–1500 | fly1 | lowpass32 | 74.8% | 63.6% | 0.0111 | −19.0 / −31.4 | 92.3 | 47.6 |
| 0–2007 | fly1 | free | 61.5% | 59.4% | 0.0097 | −17.8 / −30.7 | 92.0 | 47.4 |
| 0–2007 | fly1 | spline32 | 57.0% | 58.3% | 0.0096 | −17.7 / −30.5 | 92.0 | 47.4 |
| 0–2007 | fly0 | free | 30.4% | 41.8% | 0.0124 | −32.6 / −35.0 | 87.4 | 45.7 |
| 0–2007 | fly0 | spline32 | 32.2% | 42.1% | 0.0122 | −33.3 / −35.3 | 87.7 | 45.9 |

**Placement survives, in full.** Criterion 5 (folded wing in the measured
−20…−40 deg band, not the −57.3 deg springref): every arm lands there, and the
spline's bout medians differ from free's by ≤ 1.0 deg. Criterion 4 (residual):
+0.7% on fly1 and −22.8% (i.e. improved) on fly0, both better than free's +1.1%
/ −22.5%. Criterion 2: `max |Δ|` over every non-pitch qpos address is **exactly
0.0** in every arm and every mode — the `opt_mask` contract holds under the
reparameterisation. Criterion 3' is unchanged in verdict: fly1 clears 50% on both
wings, fly0 does not (31%/41%), so **"direction met, target not met" on the
female still stands**. The redesign was never going to fix that: it is a
pitch-only limitation and roll is out of scope (spec §4.4).

**Headline figures are the WHOLE-BOUT row**, not 0–1500 — 8.5 already ruled that
window insufficient, so quoting it as the result would repeat the mistake. Whole
bout: criterion 4 is **+9.2%** on fly1 (not the +0.7% of 0–1500) and −24.1% on
fly0; criterion 3' is **57.0%/58.3%** on fly1 (not 73.9/61.2) and 32.2%/42.1% on
fly0. Every one of those still passes its gate except fly0's 3'.

**Is `spline` better, or only quieter?** On the male: neither — the three arms
agree to 0.1 pt on `inside%`/`explained%` and to ~3 pt of gap closure. On the
female's clean stretch `spline32` beats both free and lowpass on `inside%` (92.8
vs 92.3 / 92.5), `explained%` (46.5 vs 45.8 / 46.0) and both penetrations —
**but NOT on the residual**: at 0–1500 the ranking is lowpass32 −23.33% <
spline64 −23.25% < spline32 −22.85% < free −22.54%, i.e. `spline32` is the WORST
of the three band-limited arms there. (Over the whole bout it is the best,
−24.11%; the metric flips with the window, which is itself a reason not to lean
on it.) **Option A's advantage over Option B is real but marginal and
axis-dependent, and it exists only on the hard fly.**

**And `spline64`, not `spline32`, is the better-measured spline on that fly** —
see 11.3; it was not recommended in the first pass and should have been.

### 10.6 The hard case: fly0 frames 1500–2006 is a SAM MASK failure

§8.5 recorded this quarter as the worst behaviour measured anywhere (54% of
frames moving pitch > 45 deg, max 98 deg) and never scored it. Scored now, and
the cause is upstream of everything this spec is about:

| fly / frames | valid | median SAM mask area | valid but < 25% of that camera's median |
|---|---|---|---|
| fly0 0–1500 | 100.0% | 15 862 px | 0.0% |
| fly0 1500–1710 | 90.7% | **4 882 px** | **40.2%** |
| fly0 1710–2007 | 58.0% | **2 834 px** | 29.4% |
| fly1 (all three) | 100.0% | 20 838 – 22 014 px | 0.0% |

fly0's mask collapses to **a third of its own area** and two fifths of the
surviving masks are slivers. `min_present_cameras: 3` cannot catch this: the
masks are *present*, they are just tiny and truncated (the female pressed
against the wall). The control's own numbers there say the same — `inside%` 40.2
against 86.4 on the clean stretch, wing residual 0.0617 against 0.0147 — so the
STAC pose is bad there too. fly0's only long finite run ends at frame 1710;
frames 1710–2006 are 58% valid and largely STAC-unsolved.

`pitch_traces_spline32.png` (read back, `figures/.../redesign/traces/`) shows
what the redesign does to that stretch: fly0's right wing ramps from −55 deg to
**+77 deg** at frame 1664 and back to −45 deg by 1695, as one clean ~60-frame
triangle with straight sides — the signature of a piecewise-linear basis. `free`
does the same excursion as a jagged square wave reaching +95 deg. **The spline
makes the error smooth; it does not make it correct, and a smooth wrong answer
is the more dangerous of the two because it looks plausible.** Any future
acceptance run must gate on mask AREA, not just on camera count.

### 10.7 Performance: the redesign does NOT fix it

| arm | wall, one bout-fly, T=2007, 7 cameras, one L40S |
|---|---|
| free (shipped) | 93.6 s (fly1) / 85.3 s (fly0) |
| spline32 | 94.5 s / 85.9 s |
| spline64 | 93.7 s / 86.1 s |
| lowpass32 | 93.7 s |
| spline32, `n_steps` 300 → 150 | 93.4 s → **91.0 s** |
| spline32, `n_steps` 300 → 100 | **91.7 s** |

Cutting the parameter count 28-fold and the Adam steps 3-fold buys **2.6%**, and
every placement metric is identical to three significant figures at `n_steps`
100. The Adam loop is not the cost: `prefetch=True` overlaps it with host-side
coverage-target sampling and the serial `sdf_stack_from_masks`, so wall time is
the HOST side. That reproduces §8.5's diagnosis and leaves its fix — threading
the SDF stack, measured 34.4 s → 4.8 s at 16 threads — as the only lever. The
budget is still 60 s and the stage is still 85–95 s.

### 10.8 Regenerating

```bash
BOUT=/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/pose/bouts/bout_00028
A='figures/2026-09-01-wing-mask-fit/redesign/fly{fly}/arms_fly{fly}.npz'

# the arms (~7 min per fly on one L40S); READ-ONLY on the processed tree
for F in 0 1; do python scripts/analysis/wing_mask_fit_ab.py --bout-dir $BOUT --flies $F \
  --arm 'free=wing_mask_fit.param_mode=free' \
  --arm 'spline32=wing_mask_fit.param_mode=spline;wing_mask_fit.knot_spacing=32' \
  --arm 'spline64=wing_mask_fit.param_mode=spline;wing_mask_fit.knot_spacing=64' \
  --arm 'lowpass32=wing_mask_fit.param_mode=lowpass;wing_mask_fit.knot_spacing=32' \
  --t0 0 --nt 1500 --mask-frames 24 \
  --out figures/2026-09-01-wing-mask-fit/redesign/fly$F; done

# criteria 2-6 on the never-scored quarter and the whole bout
python scripts/analysis/wing_mask_fit_ab.py --bout-dir $BOUT --flies 0,1 --from-arms "$A" \
  --t0 1500 --nt 507 --mask-frames 24 --out figures/2026-09-01-wing-mask-fit/redesign/rescore_1500-2007

# criterion 1', with the CONTROL scored as its own arm (no GPU, no fits)
JAX_PLATFORMS=cpu python scripts/analysis/wing_mask_fit_ab.py --bout-dir $BOUT --flies 1 \
  --plot-arm "control_copy=$A:q_control" --plot-arm "free=$A:q_free" \
  --plot-arm "spline32=$A:q_spline32" --plot-arm "lowpass32=$A:q_lowpass32" \
  --plot bilateral --t0 0 --nt 2007 --out figures/2026-09-01-wing-mask-fit/redesign/crit1_fly1_0-2007

# the band-limitedness of the correction itself, and the scorecard
python figures/2026-09-01-wing-mask-fit/redesign/diag_correction_spectrum.py
python figures/2026-09-01-wing-mask-fit/redesign/make_scorecard.py
```

### 10.9 Verdict

`spline` and `lowpass` both do what they were designed to do: **the song is
preserved by construction, and placement survives intact.** That removes the
single reason §9 gave for keeping the stage off on the SINGING fly. It does not
make the stage shippable, for three reasons that are all outside the
parameterisation:

1. fly0's criterion 3' is still 31%/41% against a 50% gate. **This was written
   here as "a pitch-only limit"; that was an inference and 11.4 REFUTES it** —
   swept against `mj_geomDistance`, pitch alone reaches POSITIVE clearance on
   100% of sampled frames on both flies. The DOF is not the limit.
2. fly0's frames 1500–2006 are a SAM mask failure (10.6) that no `param_mode`
   can fix, and EVERY band-limited arm renders that failure as a *smooth* wrong
   answer (spline32 +76.6 deg, lowpass32 +62.5 deg, spline64 +47.4 deg peak
   pitch; free +97.4 deg) — see 11.5, which also corrects the remedy design.
3. The stage costs 85–95 s against a 60 s budget and the redesign does not
   change that (10.7).

If it is enabled, `spline` at K = 32–64 is the mode to enable, with the
reservation in 10.4 that a C0 basis puts a small comb at the knot rate on a fly
that does not sing (`lowpass` does not), and with a mask-AREA gate added first.
**Nothing has been enabled: `param_mode: free` and `enabled: false` are
unchanged on disk.**

---

## 11. Fix round 1

Six defects, two of them false statements in §10 (corrected in place above, with
pointers here). **The headline: a band-limited arm now passes criterion 1' on
BOTH flies** — `spline32`, `spline64` and `lowpass32` PASS 1'b–1'e on all three
of fly1's song epochs and PASS 1'f on all three of fly0's finite windows, while
`free` FAILS on both flies and the rejected `smooth_weight` 30 arm still FAILS.

### 11.1 Criterion 1'f now has a MEASURED tolerance band (was: unpassable)

The clause "peak/floor must not rise above control's" has no tolerance, so only
the identity passes it. Rather than pick a number off the bracket, the band is
calibrated against two nulls (`crit1f_null.json`, 400 draws per wing, fly0's
finite run 0–1710):

* **Null A — the basis itself.** Take the arm's own measured knot vector,
  randomly PERMUTE it and add it back: same basis, same amplitude, same kink
  statistics, zero relationship to this fly. Whatever ratio that produces is the
  floor below which "manufactured a peak" cannot be claimed.
* **Null B — the statistic's own sampling variability**, from re-computing the
  control's `peak/floor` on random half-windows, i.e. with no change at all.

| wing | Null A `spline32` p50 / p95 | Null A `spline64` p50 / p95 | Null B p95 / max | MEASURED spline32 |
|---|---|---|---|---|
| `wing_pitch_left` | 1.51 / 1.80 | 1.13 / 1.20 | 1.58 / 2.06 | **1.13** |
| `wing_pitch_right` | 4.20 / 5.12 | 1.68 / 2.41 | 1.50 / 2.07 | **0.34** |

**Expectation, written before running:** if `spline32`'s 1.13–1.56× is the
generic comb of its basis rather than a manufactured feature, Null A's p95 at
K=32 sits at or above 1.5. **Read back: it does — and more, the measured arm
falls BELOW the MEDIAN of its own null on both wings (1.13 vs 1.51, 0.34 vs
4.20).** `spline32` does not manufacture a peak; it produces less comb than a
random correction in the same basis.

`CRIT1F_PEAK_TOL = 2.0`, set from **Null B** because that null is arm-independent
and therefore usable as a fixed criterion: the control's own `peak/floor` moves
up to 2.07× across sub-windows of the same recording with no change whatsoever.
Re-scored, with the control as its own arm:

| window | control_copy | free | spline32 | spline64 | lowpass32 | sw30 |
|---|---|---|---|---|---|---|
| fly0 0–1500 | 1.00 PASS | 0.15 (amp 9.25× **FAIL**) | 1.56 **PASS** | 1.11 **PASS** | 1.00 **PASS** | 7.28 **FAIL** |
| fly0 0–1710 | 1.00 PASS | 0.41 (amp 4.69× **FAIL**) | 1.13 **PASS** | 1.06 **PASS** | 1.00 **PASS** | — |
| fly0 1500–1710 | 1.00 PASS | 0.57 (amp 3.63× **FAIL**) | 1.30 **PASS** | 1.10 **PASS** | 1.00 **PASS** | — |

Falsification check: the band still rejects what it was written to reject —
`smooth_weight` 30 manufactures a peak at 7.28× and `free` fails on amplitude.
The AMPLITUDE half of 1'f stays **uncalibrated** (no measured arm sits between
lowpass32's 1.00× and free's 3.63×) and says so in the code.

Also fixed: `xcorr_periodicity` returned **1.0** for an all-NaN curve, because
`min(1.0, nan)` is 1.0 in Python — "maximally periodic" is the worst possible
default for a metric whose whole job is to refuse to fire on a fly that does not
sing. It returns 0.0 now.

### 11.2 The knot-span guarantee on real data, and the clamp that voids it

`diag_knot_span_and_clamp.py` projects each arm's correction onto the basis the
PRODUCTION run actually spans (T=2007, 8 chunks of 256, knots every K):

| fly | arm | wing | clamp frames | out-of-span | excl. clamped | suppression vs `free` |
|---|---|---|---|---|---|---|
| fly0 | spline32 | left / right | 0 / 0 | 4.9e−08 / 4.5e−08 | — | 15.6× / 17.8× |
| fly0 | spline64 | left | **17** | **1.5e−01 (11.45 deg)** | **4.5e−08** | 37.6× |
| fly0 | spline64 | right | 0 | 6.3e−08 | — | 37.3× |
| fly0 | lowpass32 | left / right | 0 / 0 | — | — | 129× / 173× |
| fly1 | spline32 | left / right | 0 / 0 | 9.6e−08 / 7.6e−08 | — | 43.7× / 42.6× |
| fly1 | lowpass32 | left / right | 0 / 0 | — | — | 673× / 583× |

Two things follow, and both were overstated in §10.

1. **The guarantee holds to float32 roundoff across all 8 chunk boundaries** —
   4.5e−08 to 9.6e−08 of the correction's own amplitude — which is the strongest
   form of the claim and is now measured on production data, not only on a
   24-frame fixture.
2. **It is CONDITIONAL on the joint clamp not firing.** `spline64` clamps for 17
   frames on fly0 and that alone takes it 11.45 deg out of span; dropping those
   frames restores 4.5e−08. The clip is a per-frame nonlinearity applied AFTER
   the basis. `n_clamp_hits` is now in the stage stats and the log.

**The claim is a SUPPRESSION FACTOR, not an impossibility.** "No knot vector can
put power in the song band" was wrong: a C0 basis carries 1/f² kink energy at
every frequency, the unit test only ever asserted < 1e−3 of variance, and §10.4's
comb *is* that leak. Measured minima across both flies and both wings:
**≥ 15× for `spline32`, ≥ 37× for `spline64`, ≥ 129× for `lowpass32`**, each
conditional on zero clamp hits. The module docstring now says this.

### 11.3 `spline64` is the better-measured arm on the hard fly

It was measured in the first pass and never recommended. On fly0 it dominates
`spline32` on every axis measured:

| | inside% | gap L / R | crit-4 resid | 1'f comb | peak pitch, 1500–1750 | suppression |
|---|---|---|---|---|---|---|
| spline32 | 92.80 | 31.1% / 41.1% | −22.85% | 1.13–1.56× | +76.6 deg | 15.6× |
| **spline64** | **92.89** | **31.9% / 41.9%** | **−23.25%** | **1.06–1.11×** | **+47.4 deg** | **37.6×** |

On fly1 the two are a wash (0.1 pt of `inside%`, ~2 pt of gap closure). So
**raising `knot_spacing` is already measured and already better, and should be
tried before anyone writes a C2 stitcher** — §10 presented a cubic basis as the
only fix for the comb, which was the wrong ordering. `spline64`'s one
disadvantage is that it is the only band-limited arm that hit a joint stop
(11.2), on 17 frames inside the mask-failure stretch that 11.5's gate should be
removing anyway. **Candidate default if the stage is ever enabled:
`param_mode: spline`, `knot_spacing: 64`**; `lowpass32` is the alternative if the
comb matters more than the mask metrics.

### 11.4 "Pitch alone cannot close the female's penetration gap" is REFUTED

§10.9 asserted a structural ceiling. Nobody had ever swept pitch against
`mj_geomDistance` — §4.1 swept it against mask SPILL — so it was an inference,
and there was a counter-datum in the scorecard (on 1500–2007 `spline32` reaches
56.2%/50.2% from an essentially identical control penetration). Swept now, 50
frames per fly, 241 pitch values across the model's own joint range, everything
else held at the control pose (`pitch_penetration_ceiling.json`):

| fly | wing | control median | BEST achievable | gap closed | best pitch (p5..p95) | frames reaching ≥ −0.005 |
|---|---|---|---|---|---|---|
| fly0 | left | −0.04440 | **+0.00813** | 121.9% | +78.8 deg (54..115) | **100%** |
| fly0 | right | −0.04526 | **+0.00642** | 117.6% | +78.8 deg (54..117) | **100%** |
| fly1 | left | −0.02563 | **+0.03289** | 240.5% | +96.8 deg (70..113) | **100%** |
| fly1 | right | −0.03677 | **+0.01954** | 158.8% | +97.8 deg (67..114) | **100%** |

**Expectation, written before running:** ≥ −0.005 means the ceiling is not real;
≤ −0.025 means it is. **Read back: POSITIVE clearance on every wing of both
flies, on 100% of sampled frames.** Pitch alone can lift the blade entirely off
the abdomen. **The DOF is not the limit, so the 31%/41% shortfall is not evidence
for roll and must not be used to reopen §4.4's scope decision.**

What it *is* evidence for needs care, because the best-achievable pitch (+79 deg
on fly0, +97 deg on fly1) is nowhere near the measured mask optimum of −20…−40
deg — it is the blade swung out and up, which is not what a folded wing does.
So the honest reading is that **criterion 3' is measuring a disagreement between
the mask objective and the collision metric, not a DOF limit**, and there are two
live explanations: either the mask optimum genuinely leaves the wing resting on
the abdomen (real fly wings do), in which case 3''s −0.0013 target is wrong; or
the mask cost is choosing a wrong optimum. Two caveats the sweep cannot remove,
recorded in the script: `mj_geomDistance` uses MuJoCo COLLISION geoms, which
§4.2b flags as coarser than the 20 184-vertex mesh; and fly0's control
penetration is ~2× fly1's while her best achievable clearance is 4× worse
(+0.008 vs +0.033), which points at an ABDOMEN-pose or BODY-SCALE error on her
rather than a wing error.

### 11.5 The mask-evidence remedy, redesigned — the first version could not work

§10.6 proposed "gate on mask AREA". In `free` mode zeroing `present` freezes the
frame; **in `spline`/`lowpass` it does not** — the knots interpolate across the
gated frame and the filter smears across it — so an area gate would only decay
the correction toward STAC over roughly one knot spacing while the frames at the
edge of the gap stay fitted to slivers. And the framing was wrong in a second
way: **every band-limited arm renders the mask failure as a smooth wrong answer**,
not just the spline —

| arm | peak `wing_pitch_right`, fly0 1500–1750 | swing |
|---|---|---|
| free | +97.4 deg | 153.2 deg |
| spline32 | +76.6 deg | 133.5 deg |
| lowpass32 | +62.5 deg | 115.8 deg |
| spline64 | **+47.4 deg** | 107.3 deg |

so "lowpass is cleaner" does not hold here either. The remedy is three parts:

1. **A POSE-AWARE validity gate, not area-only.** The control's own `inside%`
   drops 86.4 → 40.2 on that stretch and is already computed by
   `mask_metrics` — a stronger signal than raw area, and an area gate cannot
   catch a wrong-fly or merged-fly mask, which has perfectly normal area.
2. **A hard physiological bound on |Δpitch|**, so that no amount of missing
   evidence can produce a 130 deg swing. The model's own −72.8…+167.3 deg joint
   limits are not a constraint — 11.4 shows the optimiser can travel most of
   that range and stay legal.
3. **The telemetry from 11.6**, without which the gate's effect is invisible: a
   gated frame that still moved now shows up as `n_interpolated`.

None of the three is implemented; this is the design, recorded so the next
attempt does not build the version that cannot work.

### 11.6 The stage's frame accounting was a predicate, not a measurement

`moved = present.any(1) & finite_pose` is only equivalent to "this frame changed"
in `free` mode. In a band-limited mode a gated frame is interpolated and
genuinely moves, so the committed log line "*N* left at the STAC pose (*M* seen
by fewer than 3 mask cameras)" was **false**, and `dpitch_left/right_deg` were
medians over a subset that excluded frames which had changed. `moved` is now
computed from the actual pose delta, and the stats carry `n_interpolated`
(no-evidence frames that moved anyway), `n_no_evidence_frames` and
`n_clamp_hits`. The stub in `tests/test_run_bout_pipeline_structure.py` ignored
`present` unconditionally, so it was faking neither refiner; it now takes
`honour_present` and clips to the passed limits, and two tests pin the two modes.

### 11.7 Minor corrections

* `spline32`'s E1 pulse count is **45/43**, not 43/43 (§10.3 said 43/43; it still
  passes the 0.5–2× gate).
* fly1's **E3 clears its applicability gate narrowly** — xcorr periodicity 0.388
  against `min_periodicity` 0.35. Nudging the threshold to 0.40 would reclassify
  E3 as a negative control and score it by 1'f instead. Every arm's verdict on E3
  is the same as on E1/E2, so nothing here turns on it, but the margin is thin
  and the threshold is a judgement, not a measurement.
* the mask-area table of §10.6 existed only as prose; it is now in
  `scorecard_redesign.json` under `mask_area`, with the script that produces it.

### 11.8 Regenerating the fix-round measurements

```bash
D=figures/2026-09-01-wing-mask-fit/redesign
python $D/diag_knot_span_and_clamp.py        # 11.2, CPU
python $D/diag_crit1f_null.py                # 11.1, CPU, ~400 draws/wing
python $D/diag_pitch_penetration_sweep.py    # 11.4, CPU, ~20 min
python $D/make_scorecard.py                  # folds all of the above in
```

---

## 12. Task 10 — tracking quality, the validity gate, and more bouts

*The user's scope ruling, in their own words: "our focus for correcting is when
the flies are well tracked, so if the female is not well tracked it is ok to
skip it." That turns §11.5's open blocker into a feature. This section builds
the gate, measures what it does on bout 28, and takes the stage to bouts it has
never seen.*

**Nothing under the processed data root was written.** md5 manifest of all 44
files under `.../2025_10_20_13_20_04/pose/` taken before any job and re-checked
at the end: `md5sum -c` OK on 44/44. New bouts were written to a scratch run
root, `/gscratch/portia/eabe/data/Johnson_lab/scratch/wingfit_task10/pose`.

### 12.1 A mask-only tracking-quality score, calibrated on bout 28

`jarvis_jax.tracking.mask_quality` + `scripts/analysis/mask_tracking_quality.py`.
CPU only, ~10 s per bout, no pose and no GPU: the masks already exist for all 30
Session0 bouts while pose artifacts existed for exactly one.

**Calibration first, because a score that cannot reproduce a known split must
not be used to pick anything.** `--calibrate 28` re-derives the three windows
whose truth is recorded in §10.6 and prints PASS/FAIL. It reproduces §10.6
exactly, from the packed masks, with no shared code:

| fly / frames | valid | median area | slivers | usable cams | frac frames ok | verdict |
|---|---|---|---|---|---|---|
| fly0 0–1500 | 1.000 | 15 862 px | 0.001 | 7.00 | 1.000 | GOOD (truth GOOD) |
| fly0 1500–1710 | 0.907 | **4 882 px** | **0.473** | **3.35** | 0.519 | BAD (truth BAD) |
| fly0 1710–2007 | 0.580 | **2 834 px** | **0.507** | **2.00** | 0.000 | BAD (truth BAD) |
| fly1 (all three) | 1.000 | 20 838–22 014 px | 0.000 | 7.00 | 1.000 | GOOD ×3 |

**Verdict: PASS.** The 15 862 / 4 882 / 2 834 px and 100% / 90.7% / 58.0% agree
with §10.6 to the digit.

**One thing this exposed, and it changed the score.** Scored on
`frac_frames_ok` ALONE at `min_cameras = 3` — the stage's own bar — fly0's
1500–1710 window reads **0.519**, i.e. "MID", because with a mean of 3.35 usable
cameras about half of those frames still scrape three. Every other signal calls
it loudly (47% slivers against 0.1%, 3.35 cameras against 7.00, 4 882 px against
15 862). A one-number rule that lands mid-scale on a case this extreme would not
separate a marginal bout at all, so the verdict is taken on the SIGNAL VECTOR:
BAD when any signal screams, GOOD only when they all agree. The bands (sliver
≤ 0.05 / ≥ 0.25, usable cameras ≥ 0.9C / ≤ 0.6C) are far from every measured
value on both sides.

**Two candidate signals measured and REJECTED, both reported and never scored:**

* **centroid jump** does not separate the known split — fly0 p95 1.19 px on her
  clean stretch and 2.04 px on the collapsed one, against fly1's 3.73 and 5.66
  px while he is fine throughout. It ranks MOTION, not quality.
* **mask-area p90/p50**, the obvious wing-extension (song) proxy, is **1.119 on
  bout 28's SINGING male and 1.215 on its non-singing female**. It measures
  occlusion, not song. **There is therefore no mask-only way to pre-select for
  song**, and bouts had to be chosen on quality and length alone, with the song
  epochs measured after STAC.

### 12.2 The ranking: all 30 bouts × 2 flies

`figures/2026-09-01-wing-mask-fit/tracking_quality/tracking_quality.{json,md}`.
`score = frac_frames_ok × worst_window_frac_ok × mean_cams_ok/C`. The third
factor is a tie-break and it is not cosmetic: **all 30 males and 20 of 30
females saturate the first two factors at 1.000**, so without it the ranking
cannot choose; with it, 12 of 30 females score exactly 1.000.

The headline finding is the same asymmetry the rest of this pipeline shows:
**every one of the 30 males scores 1.000 with 7.00/7 usable cameras and zero
sliver camera-frames; the females span 1.000 down to 0.000.** Twelve females
score 1.000; **eight score below 0.5 — bouts 2, 13, 15, 22, 23, 26, 27 and 28**
(bout 22 sits at 2.00 usable cameras for its whole 1393 frames; bouts 13, 15,
22, 23, 27 and 28 all score 0.000 on the worst-window factor). **Bout 28 — the
only bout that had ever been processed — has its female in that bottom group.**

Chosen: **bouts 10 (T = 1741), 30 (T = 1298), 20 (T = 1160)** — the three
longest with `min(fly0, fly1) score == 1.000`, 7.00/7 usable cameras and 0%
slivers on BOTH flies. Length matters because criterion 1'b is scored at
`nperseg` 64/128/256.

**Figure, expectation written first** (the docstring of `plot_timelines`): on
bout 28 the male should be a flat line at 7 usable cameras with the area ratio
pinned near 1.0 for all 2007 frames, and the female identical to him until
~1500 and then falling off a cliff; on a bout chosen as well tracked both flies
should look like bout 28's male throughout. **Read back
(`timeline_bout00028.png`, `timeline_bout00010.png`, `timeline_bout00030.png`):
exactly that.** fly1 flat at 7 cameras and 0.95–1.05 area ratio end to end;
fly0 at 7 cameras until ~1500, then a staircase 7→6→5→4→3→2 over 1500–1600 and
flat at 2 — below the 3-camera line — to the end, with her area ratio at 0.07 by
1650. Bouts 10 and 30: both flies flat at 7 cameras and 0.78–1.05 area ratio for
every frame, never within 3× of either gate line.

**And one thing the figure shows that the window table did not: the female's
decline starts at ~frame 1050, not 1500.** Her area ratio slides 1.0 → 0.5 over
1050–1500 before the camera count moves at all. §10.6's 1500 boundary is where
it became catastrophic, not where it began — which is why the gate below starts
holding frames at 1430.

### 12.3 The gate, measured on bout 28: it removes the failure and leaves the rest alone

`param_mode: spline`, `knot_spacing: 64`, whole bout, both flies, gate off vs on.
Read-only on the processed tree; arms in
`figures/2026-09-01-wing-mask-fit/gate/fly{0,1}/`.

**fly0, the female (the hard fly):**

| arm | pen L | pen R | wing resid | pitch L / R med | inside% | expl% | hp(Δpitch) L / R |
|---|---|---|---|---|---|---|---|
| control | −0.04442 | −0.04504 | 0.0161 | −8.2 / −7.7 | 80.9 | 37.7 | — |
| gate OFF | −0.03026 | −0.02673 | 0.0123 | −33.7 / −35.3 | 87.7 | 45.7 | 0.170 / 0.172 |
| **gate ON** | −0.03228 | −0.02765 | 0.0123 | −31.4 / −35.0 | 86.1 | 44.1 | **0.034 / 0.029** |

**fly1, the male, well tracked throughout — the control that says the gate is
not just "do less":**

| arm | pen L | pen R | wing resid | pitch L / R med | inside% | expl% |
|---|---|---|---|---|---|---|
| control | −0.02620 | −0.03620 | 0.0088 | −5.4 / −10.4 | 84.4 | 43.8 |
| gate OFF | −0.01219 | −0.01593 | 0.0097 | −17.9 / −30.7 | 91.9 | 47.3 |
| **gate ON** | **−0.01218** | **−0.01593** | **0.0097** | **−17.9 / −30.7** | **91.9** | **47.3** |

**On the well-tracked fly the gate is a measured no-op**: 0 gated frames, 0
sliver camera-frames, 0 pose-rejected camera-frames, 2007/2007 refined, median
Δpitch −12.91 vs −12.92 deg, every metric identical to 4 s.f.

**Telemetry, which is where the gate's effect is actually visible:**

| | fly0 gate OFF | fly0 gate ON | fly1 gate OFF | fly1 gate ON |
|---|---|---|---|---|
| refined / T | 1755 / 2007 | **1431 / 2007** | 2007 / 2007 | 2007 / 2007 |
| `n_gated_frames` | 0 | **324** | 0 | **0** |
| `n_sliver_camera_frames` | 0 | 681 | 0 | 0 |
| `n_pose_reject_camera_frames` | 0 | 711 | 0 | 0 |
| `n_at_dpitch_bound` | **34** | **0** | 0 | 0 |
| `n_interpolated` | 0 | 0 | 0 | 0 |
| wall (s) | 92.3 | 104.1 | 108.0 | 118.5 |

The gate skipped **324 of fly0's 2007 frames (16.1%)** in three runs —
1430–1436, 1438–1709, 1711–1750 — i.e. the collapsed stretch, starting 70 frames
before §10.6's hand-drawn 1500 boundary and consistent with the area decline the
timeline figure shows. `n_at_dpitch_bound` going 34 → 0 is the cleanest single
statement of what happened: **with the gate on, no frame needs the bound,
because the evidence that was driving the correction into it is gone.**

**Peak excursion, the number §11.5 tabulated:**

| arm | max \|Δpitch\| 0–1500 (L / R) | max \|Δpitch\| 1500–1750 (L / R) | peak `wing_pitch_right` 1500–1750 |
|---|---|---|---|
| gate OFF | 33.5 / 39.1 deg | **50.0 / 50.0 deg** (at the bound) | **+45.9 deg** |
| **gate ON** | **33.5 / 39.1 deg** | **0.00 / 0.00 deg** | **+5.6 deg** |

Identical on the well-tracked stretch, exactly zero on the collapsed one.

**Figure, expectation written first:** the gate-OFF arm's green trace should
show the large positive excursion on fly0's right wing in 1550–1780; the gate-ON
arm's green trace should lie EXACTLY on the grey control there, and be
indistinguishable from gate-OFF on 0–1500 and on both of fly1's rows.
**Read back (`gate/traces/pitch_traces_sp64_{nogate,gate}.png`): exactly that.**
Gate OFF, the fly0 right-wing zoom shows one clean triangle from −57 deg at
frame 1600 to **+46 deg** at 1670 and back to −55 deg by 1730 while the control
sits flat at −5 deg. Gate ON, the grey control trace is not visible in either of
fly0's zoom panels **because the green fit is drawn exactly on top of it** — the
measured Δ is 0.00 deg. fly1's two rows are unchanged between the two figures.

**Criterion 1' (scorer validated against spec §8.2 before use: control MSC
0.418 / 0.861 / 0.712 reproduced to three decimals):** on fly1's three song
epochs, `control_copy`, gate OFF and gate ON all **PASS** 1'b–1'e (MSC 0.418 /
0.861 / 0.710, 1'd corr +0.999–1.000, pulses 42/43, 30/30, 10/10). On fly0's
finite run 0–1710, 1'f **PASSES** for both arms (peak/floor 1.03× off, 1.02× on,
against the calibrated 2.0× tolerance). Criteria 2, 3', 4, 5 and the invariants
PASS in every arm; `max |Δ|` over every non-pitch qpos address is exactly 0.0.

**What the gate costs, honestly:**

1. **`inside%` 87.7 → 86.1 and `explained%` 45.7 → 44.1 on fly0.** The skipped
   frames revert to the STAC pose, which is worse on those metrics — that is the
   point, and the price.
2. **Criterion 3' gap closure falls slightly on the female**, 32.8% / 42.0% →
   28.2% / 39.8%. Still "direction met, target not met" against the 50% gate;
   the verdict does not change.
3. **`lp_std` on fly0 rises 2.29× → 4.76× (left) and 5.87× → 6.28× (right).**
   The gate puts a real low-frequency step into her trajectory: a corrected
   stretch and an uncorrected one, joined over one 64-frame knot spacing. That
   is the honest consequence of skipping rather than guessing, and it is visible
   in the figure. `hp_rms` is unaffected (1.00× with the gate against 0.99× /
   1.07× without), so nothing is added in the song band.
4. **+11 s per bout-fly** (92.3 → 104.1, 108.0 → 118.5) for the body projection
   and the inside-fraction count, against a budget already missed by 50%.

### 12.4 The escape hatch, and a default that changed

**The shipped default now gates and bounds.** Every number in §§1–11 predates
both. `gate_enabled: false` ALONE does not reproduce them — the |Δpitch| bound
fires on 34 frames of bout 28 fly0 and moves her left wing by up to 28.2 deg.
The reproducing config is

```
wing_mask_fit.gate_enabled=false wing_mask_fit.max_dpitch_deg=null
```

Measured against the saved Task-9 `q_spline64` arm on bout 28 fly0, whole bout:
**max |Δ| 0.028 deg, 0 frames over 0.1 deg** — and two runs of that same config
differ from each other by **0.014 deg**, so the residue is the solver's own
float32 non-determinism to within a factor of two, not a change of behaviour.
Bit-for-bit is not available on a GPU Adam solve and claiming it would be false;
what IS bit-exact is that the new code does not run — `apply_gate_and_bound`
returns the same object when both switches are off, pinned by a pure-numpy test.

### 12.5 Where the |Δpitch| bound of 50 deg comes from

Measured, not chosen. Across all four `param_mode` arms of §10, on every
WELL-TRACKED frame (0–1500) of both flies of bout 28, the largest correction any
arm produces is **42.2 deg** (`free`, fly0 right wing); `spline64` reaches 39.1.
On the collapsed stretch the same arms reach **71–98 deg**. 50 deg is also about
the distance from the fitted pose (~−8 deg) to the model's own springref rest
(−57.3 deg), so no legitimate placement correction can need more. The model's
own joint range (−72.8…+167.3 deg) is not a constraint at all.

### 12.6 Three bouts the stage had never seen

Chosen by 12.2: **bouts 10 (T = 1741), 30 (T = 1298), 20 (T = 1160)**, the three
longest with `min(fly0, fly1) score == 1.000`. Each was run end to end into
`/gscratch/portia/eabe/data/Johnson_lab/scratch/wingfit_task10/pose` — 2D
(ViTPose) → triangulate → Stage-B2 filter → STAC → bridges — reusing the
recording's own `scale.json` (which records `identity: canonical`, so the
per-fly scale check passes) and `offsets.h5`, with bout 28's own overrides
`wing_collapse.enabled=true rigid_repair.enabled=true`. Wall clock ~28 min per
bout-fly end to end on one L40S with three bouts sharing the node.

**IDENTITY CHECKED, not assumed.** The SAM3 `sex_meta` says `male_slot: 1` on
all four bouts, but its mask-area vote is weak on two of them — camera agreement
**0.429 on bout 10 and 0.143 on bout 20**, against 1.0 on bouts 28 and 30. The
behavioural invariant settles it: mean `|yawL−yawR|` on the STAC pose is

| bout | fly0 | fly1 | ratio |
|---|---|---|---|
| 28 (reference) | 4.78° | **25.35°** | 5.3× |
| 10 | 1.42° | **14.74°** | 10.4× |
| 20 | 8.14° | **45.80°** | 5.6× |
| 30 | 0.20° | **41.07°** | 205× |

so fly1 is the wing-extending fly on every bout measured and `male = fly1`
holds, weak mask-area vote notwithstanding.

#### The FEMALE — the hard fly, where she is well tracked

| bout-fly | control pen L/R | fitted pen L/R | gap L / R | wing resid | pitch L/R | inside% | expl% |
|---|---|---|---|---|---|---|---|
| 28 fly0 (worst-tracked) | −0.0444 / −0.0450 | −0.0323 / −0.0277 | 28.1% / 39.8% | 0.0161 → 0.0123 (−24%) | −31.4 / −35.0 | 80.9 → 86.1 | 37.7 → 44.1 |
| **10 fly0** | −0.0458 / −0.0444 | −0.0246 / −0.0291 | **47.7%** / 35.5% | 0.0134 → 0.0113 (−16%) | −37.7 / −32.7 | 88.9 → 92.9 | 37.8 → 44.7 |
| **20 fly0** | −0.0439 / −0.0451 | −0.0262 / −0.0281 | 41.6% / 38.9% | 0.0201 → 0.0149 (−26%) | −36.2 / −30.9 | 81.9 → 88.3 | 33.6 → 39.9 |
| **30 fly0** | −0.0446 / −0.0423 | −0.0266 / −0.0217 | 41.5% / **50.3%** | 0.0100 → 0.0089 (−11%) | −34.5 / −39.0 | 87.9 → 94.0 | 41.7 → 48.3 |

**The defect and the fix both generalise.** Every female's control penetration
is −0.042 to −0.046 and her control wing pitch −6 to −12° — §1's measured defect,
reproduced on three bouts chosen without reference to it. Every one moves into
the −20…−40° band on both wings, penetration closes 35–50% of the gap, the
wing-marker residual IMPROVES 11–26%, `inside%` rises 4–6 points and
`explained%` 6–7. Criteria 2, 3′, 4, 5 and the rigid invariants PASS on all
three, with `max |Δ|` over every non-pitch qpos address exactly 0.0.

**Criterion 3′ clears on a female for the first time**: bout 30's right wing at
**50.3%**. Across the eight female wings measured here the range is 28–50% and
the mean ~41%, against bout 28's 28/40. Seven of eight are still short, so the
verdict stands — but a good part of the female's historic shortfall was her
TRACKING, not the DOF.

**The gate skipped nothing on any of them.** `n_gated_frames = 0`,
`n_sliver_camera_frames = 0` on all six new bout-flies, and
`n_pose_reject_camera_frames` 0 except **2 camera-frames of 6 960** on bout 20
fly0. Gate-on and gate-off agree to two decimals on every median. Measured
straight off the masks, the minimum usable-camera count over all three bouts and
both flies is **7 of 7** and the median area ratio never falls below **0.592**
(bout 20 fly0's mild end-of-bout occlusion) against the 0.25 line. **The ranking
predicted the gate would be inert here, and it was** — which is the whole point
of calibrating it first.

One exception worth naming: on bout 10 fly0 the |Δpitch| bound fires on **1
frame of 1741** (0.06%), in both arms. At 50° it is not perfectly inert even on
a clean bout.

#### The MALE, and the song

| bout-fly | extension | epochs found | criterion 1' | Δpitch extended / folded | folded gap | inside% | expl% |
|---|---|---|---|---|---|---|---|
| 28 fly1 | 25.4° | E1 L 0–869, E2 R 869–1515, E3 L 1515–1943 | **PASS ×3** | — | 56.3 / 58.1% | 84.4 → 91.9 | 43.8 → 47.3 |
| **10 fly1** | 14.7° | E1 R 123–870, E2 L 933–1378 (both under the applicability gate → 1'f), E3 R 1378–1741 | **PASS ×3** | −8.28° / −15.61° | 50.7 / 62.0% | 83.0 → 90.0 | 44.0 → 47.2 |
| **20 fly1** | 45.8° | one, L 0–1160 | **PASS** | **+0.46° / −24.66°** | **55.3%** | 84.3 → 90.8 | 44.2 → 48.0 |
| **30 fly1** | 41.1° | E1 L 0–925 (not periodic → 1'f), E2 R 925–1298 | **PASS ×2** | **+4.55° / −22.56°** | **62.5%** | 83.5 → 90.6 | 42.3 → 46.4 |

**Criterion 1' passes on every epoch of every bout, gate on and gate off.** The
control's own coherence is reproduced to three decimals in every arm — bout 20
E1 `MSC` 0.802 → 0.803 with 1'd corr +0.997 and pulses 32/33; bout 30 E2
0.722 → 0.723, corr +1.000, 25/25; bout 10 E3 0.834 → 0.834, corr +1.000, 18/18.
The epoch finder independently rediscovered the extended-wing swap on bouts 30
(frame 925) and 10 (frames 870 and 1378), so 5.0's "the singing wing is not a
property of a bout-fly" is confirmed on new data.

Note what the criterion does with the epochs where there is no measurable phase
lock: **5 of the 9 male epochs here have a control MSC below their own 95%
significance floor and are reported NOT APPLICABLE and scored by 1'f instead.**
That is the gate refusing to fire where there is nothing to protect, rather than
passing them by default — the behaviour 9.2's second defect fix put in.

**Spec 4.1's SELF-TARGETING property, confirmed cleanly on new data.** On the two
strongly-extending males the median Δpitch is **+0.46° on the EXTENDED wing and
−24.66° on the FOLDED one** (bout 20) and **+4.55 / −22.56°** (bout 30): the
stage corrects the wing that is wrong and leaves the singer where it already is.
It is weaker where extension is weaker (bout 10, 14.7°: −8.28 extended /
−15.61 folded), and on bout 30's first epoch the extended wing moves ~+15°
against the control — so this is a strong tendency, not a guarantee.

Every male's FOLDED wing clears criterion 3''s 50% gate: **50.7–62.5%.**

#### What the gate skipped, and the one place it did not fire but arguably should

| bout-fly | `n_gated_frames` | sliver cam-frames | pose-reject cam-frames | `n_at_dpitch_bound` |
|---|---|---|---|---|
| 28 fly0 | **324 / 2007 (16.1%)** | 681 | 711 | 0 (**34** with the gate off) |
| 28 fly1 | 0 | 0 | 0 | 0 |
| 10 fly0 | 0 | 0 | 0 | **1** |
| 20 fly0 | 0 | 0 | **2** | 0 |
| 30 fly0 | 0 | 0 | 0 | 0 |
| 10 fly1 | 0 | 0 | **1** | 0 |
| 20 fly1 | 0 | 0 | 0 | 0 |
| 30 fly1 | 0 | 0 | **39** | **12** |

**The gate's thresholds are calibrated on a catastrophe and miss a mild one.**
On bout 10 fly0 the correction exceeds 45° on **32 frames of 1741**, and on bout
20 fly0 on **43 of 1160** — both in the second half, where the female's median
mask-area ratio has drifted to 0.88–0.95 of her own p75. That is far above the
0.25 sliver line, so the gate does not fire, and only the 50° bound contains it.
Read off `traces_bout20/pitch_traces_sp64_gate.png`: fly0's right-wing fit makes
+42° excursions over frames 800–1050 while the control sits at −5°.

**Whether those are real behaviour or fit error is NOT settled here** — the
CONTROL pose moves in the same stretch (its own trace goes −6 → +5°), so the fly
is doing something. What can be said is that the current thresholds do not treat
a 40% area decline as a reason to stop fitting, and that a reader looking for the
next weakness should look there.

**Read back, `traces_bout30/pitch_traces_sp64_gate.png`** (stated first: on the
best-tracked bout the female's green trace should sit smoothly inside the
−20…−40° band for all 1298 frames with no excursion, and the male's extended
wing should track the control): the female's two rows are exactly that — a
smooth green trace in the band, control flat at −11°, no spikes. The male's
folded wing shows the expected −20° offset; during his SECOND (right-extended)
epoch the fitted right wing oscillates ±25° around −35° where the control
oscillates ±10° around 0, i.e. the fit amplifies the slow swing while `hp_rms`
stays 1.00× — visible, and consistent with `lp_std` 1.43×.

### 12.8 Verdict

**The bout-28 result generalises.** On three bouts the stage had never seen,
picked by a score calibrated on bout 28 and nothing else:

* the DEFECT is the same everywhere — female control penetration −0.042…−0.046,
  control wing pitch −6…−12°;
* placement is fixed the same way — into the −20…−40° band on every wing,
  35–50% gap closure on the female and 50–63% on the male's folded wing, wing
  residual improved 11–26% on the female, `inside%` +4…+7 points;
* **criterion 1' PASSES on every epoch of every fly**, and the self-targeting
  property is confirmed on two strongly-singing males;
* **the gate is inert where the ranking says it should be** (0 frames skipped on
  all six new bout-flies) and removes exactly the stretch it was built for on
  bout 28 (324 frames, `n_at_dpitch_bound` 34 → 0).

What has NOT changed: criterion 3' still fails on 7 of the 8 female wings
measured (though one clears at 50.3%, the first ever), the stage still costs
92–119 s against a 60 s budget, and it is still one recording with no
human-reviewed `sex.json`. `enabled: false` is unchanged on disk.

### 12.7 Regenerating

CPU-only unless marked GPU. `unset LD_LIBRARY_PATH` first; nothing below writes
to the processed tree.

```bash
M=/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/sam3_masks
B28=/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/pose/bouts/bout_00028
S=/gscratch/portia/eabe/data/Johnson_lab/scratch/wingfit_task10/pose

# 12.1-12.2 the ranking + the calibration verdict + the timeline figures (~5 min)
python scripts/analysis/mask_tracking_quality.py --masks-dir $M --calibrate 28 \
  --plot-bouts 10,20,28,30 --out figures/2026-09-01-wing-mask-fit/tracking_quality

# 12.3 the gate on bout 28, both flies (GPU, ~4 min per fly)
for F in 0 1; do /path/to/gpu-env python scripts/analysis/wing_mask_fit_ab.py \
  --bout-dir $B28 --flies $F \
  --arm 'sp64_nogate=wing_mask_fit.param_mode=spline;wing_mask_fit.knot_spacing=64;wing_mask_fit.gate_enabled=false' \
  --arm 'sp64_gate=wing_mask_fit.param_mode=spline;wing_mask_fit.knot_spacing=64' \
  --t0 0 --nt 2007 --mask-frames 24 \
  --out figures/2026-09-01-wing-mask-fit/gate/fly$F; done

# 12.3 the decisive figure + criterion 1' (no GPU, no fits)
A='figures/2026-09-01-wing-mask-fit/gate/fly{fly}/arms_fly{fly}.npz'
JAX_PLATFORMS=cpu python scripts/analysis/wing_mask_fit_ab.py --bout-dir $B28 --flies 0,1 \
  --plot-arm "control_copy=$A:q_control" --plot-arm "sp64_nogate=$A:q_sp64_nogate" \
  --plot-arm "sp64_gate=$A:q_sp64_gate" --plot pitch --t0 0 --nt 2007 --zoom 1550,1780 \
  --out figures/2026-09-01-wing-mask-fit/gate/traces
JAX_PLATFORMS=cpu python scripts/analysis/wing_mask_fit_ab.py --bout-dir $B28 --flies 0 \
  --plot-arm "control_copy=$A:q_control" --plot-arm "sp64_nogate=$A:q_sp64_nogate" \
  --plot-arm "sp64_gate=$A:q_sp64_gate" --plot bilateral --t0 0 --nt 1710 \
  --out figures/2026-09-01-wing-mask-fit/gate/crit1f_fly0_0-1710

# 12.4 the escape hatch, on real data (GPU, ~2 min)
python scripts/analysis/wing_mask_fit_ab.py --bout-dir $B28 --flies 0 \
  --arm 'sp64_task9=wing_mask_fit.param_mode=spline;wing_mask_fit.knot_spacing=64;wing_mask_fit.gate_enabled=false;wing_mask_fit.max_dpitch_deg=null' \
  --t0 0 --nt 2007 --mask-frames 8 --out figures/2026-09-01-wing-mask-fit/gate/repro_fly0

# 12.6 the new bouts: FULL PIPELINE into a scratch root (GPU, ~28 min per bout-fly).
# scale.json + offsets.h5 are COPIED from the recording's own pose/ first.
for B in 10 20 30; do python scripts/run_bout.py paths=hyak ++bout_ids=$B \
  ++wing_collapse.enabled=true ++rigid_repair.enabled=true \
  ++outputs.out=$S ++run_id=task10_b$B ++outputs.overlay=false ++outputs.sidebyside=false; done
# then the A/B per bout-fly, and the traces, exactly as above with --bout-dir
# $S/bouts/bout_000NN and --nt 1741 / 1160 / 1298.
```
