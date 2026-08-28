# Wing blade-roll prior — implemented, do NOT enable

Date: 2026-08-28.  Figures: `figures/2026-08-28-wingroll/`.
Data: `roll_noise_survey.json`, `roll_ab.json` (summary copies here).

## What was built

`JAXLS_SMOOTH_Q_MULT`, a per-DOF multiplier on the temporal smoothness cost, so
one weakly-observable joint can be damped without raising `smooth_weight`
globally and blunting every DOF.

- `stac_core_jaxls.py` — `JaxlsBatchSolver(smooth_q_mult=...)`, applied in both
  the SE3 and plain-q builders and folded into the problem cache key.
- `stac_core.py` — `jaxls_smooth_q_mult=`.
- `stac.py` — `_resolve_smooth_q_mult()` maps `{joint_name: multiplier}` to a
  qpos-layout array, raising on an unknown name so a typo cannot become a
  silent no-op.

Default `None` (uniform): behaviour is unchanged unless configured.

## Verdict: leave it off

On the **singing male** it destroys the only wing signal that distinguishes an
extended wing, displaces the motion into other DOFs, and makes the wing fit
worse. Details below.

## Reading the model's wing convention first

`fruitfly_v1` puts the wing spring rest at **`wing_yaw` = +85.94°, which is
exactly its upper joint stop** (roll +40.11°, pitch −57.30° = its lower stop).
That pose is the wings **folded back and flat** — a fly at rest sits *on* the
stop by construction.

Two consequences:

- Joint-limit saturation is **useless as a QC signal for wing yaw**. A resting
  wing always reads as "at its limit". An earlier reading of this note claimed
  a yaw-saturation defect (fly0 55.6% of frames) and inferred the ±85.9° limit
  was too tight for a ~90° extension. Both were wrong: fly0's wings are simply
  folded, correctly, and the fly that actually extends (fly1) saturates 3.4%.
- Departure from 85.9° *is* wing extension, and is how to score it.

## The premise, and why it failed

A survey of 316 bout-flies (`wing_roll_noise_survey.py`) refutes the claims the
prior rested on:

| Claim | Measured | Verdict |
|---|---|---|
| roll wanders on noise; yaw/pitch track real motion | lag-1 autocorrelation **roll 0.836** vs **yaw/pitch 0.807**; only 30% of bout-flies have roll less autocorrelated | refuted — roll is the *smoother* DOF |
| roll takes larger frame-to-frame steps | med \|Δ\| **roll 0.429°** vs **0.367°** | weak; larger steps *with* higher autocorrelation means larger real amplitude |
| roll parks against its joint stop | roll saturation **0.0% / 0.0%** for both flies | refuted |

The earlier supporting number (roll autocorrelation 0.34–0.37 vs 0.55–0.78) was
from one clip and does not replicate.

## What the prior does on the singing male

`wing_roll_prior_ab.py` on Session0/2025_10_20_13_20_04 bout_00003 **fly1**,
1024 frames @ 800 Hz, multipliers 1 / 5 / 20 (`roll_prior_fly1.png`):

- Extended-wing roll hp-RMS **1.90° → 0.50° → 0.05°** — the prior erases it.
- The motion is **displaced, not removed**: `pitch_right` hp-RMS
  **0.45 → 1.27 → 1.59**, `yaw_left` **0.55 → 0.86 → 0.95**. Same trajectory,
  re-expressed through different joints.
- Wing-keypoint residual gets **worse: +2.84%** (×5), +2.62% (×20).

## The signal it would destroy

Control over a breakdown-free window (0.2–1.1 s), `wing_roll` hp-RMS:

| | value |
|---|---|
| fly1 singer, **extended** wing | **1.87°** |
| fly1 singer, folded wing | 0.38° |
| fly0 non-singer, folded wing | 0.46° |

The elevation is specific to the *extended* wing — not to the fly, and not to
the solver. Its spectrum has a narrowband peak at **≈117–120 Hz** (with
secondary energy near 240 Hz) roughly two orders of magnitude above both folded
wings, which show no such peak. Whether that maps to the pulse-song carrier is
**not established here** — fs = 800 Hz, and 120 Hz could be a subharmonic of a
~240 Hz carrier or the wing's mechanical flap rate. What is established is that
it is wing-driven, extension-specific, and the prior removes it.

Do not score this over the whole bout: a late-bout tracking breakdown (fly0,
1.15–1.3 s, ±20°) dominates the statistic and inverts the answer — scored that
way the non-singer looks *noisier* than the singer (hp-RMS 2.48° vs 1.90°) and
the conclusion flips. Sparse peak-detection is also mis-tuned for a sustained
oscillation, since the oscillation inflates its own MAD threshold.

## Method notes worth keeping

- **Read wing DOFs from each file's own `names_qpos`.** The IK model is
  fruitfly_v1 (nq=93, wings at columns **7–12**); `fruitfly_v2_3_ik` has nq=101
  and the same wing names land on **leg** columns. An intermediate version of
  the survey made exactly that mistake and produced a confident, entirely wrong
  answer about wing amplitude.
- **Run wing/song analyses on the singer.** In bout_00003 that is **fly1**,
  confirmed by unilateral extension (|yawL−yawR| median 24°, 25% of frames
  >30°, vs 5.4° for fly0). The same A/B on fly0 makes the prior look harmless
  (+0.6% residual, wing residual flat) because that fly's wings never leave
  rest. This recording has **no `sex.json`** — identity was confirmed from
  kinematics, not labels.
- Narrowband power at a sine-song carrier does not measure pulse song; that is
  why `wing_roll_prior_ab.py` reports hp-RMS, kurtosis, transient count and
  amplitude alongside band power.

## Regenerate

```
python scripts/analysis/wing_roll_noise_survey.py --out figures/2026-08-28-wingroll
scripts/slurm/submit_task.sh --time 2:00:00 rollab1 \
  "python scripts/analysis/wing_roll_prior_ab.py --dir <...bout_00003/fly1> \
     --t0 0 --nt 1024 --mults 1,5,20 --fs 800 --out figures/2026-08-28-wingroll/fly1"
python scripts/analysis/wing_roll_prior_figure.py \
  --npz <out>/roll_ab.npz --json <out>/roll_ab.json --h5 <...fly1/stac_ik.h5> \
  --control-h5 <...fly0/stac_ik.h5> --win 0.2,1.1 --out <out>
```
