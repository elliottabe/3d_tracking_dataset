# Track 1 — scale / segment-calibration 2×2: results and verdict

**Date:** 2026-08-08
**Benchmark:** `configs/benchmark/bouts.yaml` (13 bouts → 21 bout-flies per variant, 84 runs total)
**Artifacts:** `scorecard_<variant>.json/.md`, `input_divergence.json`, this file.

## Verdict, up front

**The 2×2 does not select a scale estimator, and no default should be changed on
its basis.** It instead surfaced a more consequential defect in the pipeline:

> The per-recording body scale is estimated from **one arbitrary bout-fly** —
> literally "whichever bout/fly gets there first" (`scripts/run_bout.py:509-511`)
> — and then applied to every bout of that recording. That estimate is not
> robust: on this benchmark one bout-fly produced a scale **38× too small**,
> which corrupted every downstream fit for that recording in all four variants.

Fixing that is a prerequisite to answering the trunk-vs-all question, and is a
strong candidate explanation for the original symptom that started this work
(body-model meshes looking visibly wrong).

## Evidence

### 1. Scale estimated per bout-fly is unstable

`compute_trunk_scale` run independently on each benchmark bout-fly's frozen kp3d:

| bout-fly | trunk / umeyama | all / norm_ratio |
|---|---|---|
| **S0 b22 fly0** | **0.000338** (38× too small) | **0.001819** (6× too small) |
| S0 b22 fly1 | 0.012993 | 0.012646 |
| S0 b26 fly0 | 0.009829 | 0.013668 |
| S0 b26 fly1 | 0.013033 | 0.012291 |
| S1 14_54_28 b12 fly0 | 0.012600 | 0.012828 |
| S1 14_54_28 b17 fly0 | 0.011650 | 0.012638 |

Production values for the same recordings: **S0 = 0.010994**, **S1 14_54_28 = 0.010821**.

Two things follow. First, a single pathological bout-fly can produce a
catastrophic estimate, and whether it is used is decided by processing order.
Second, even on healthy bouts both estimators land ~7–18% above the production
value, because production's scale came from a different first bout — i.e. the
recording-level scale is not reproducible run-to-run.

`all + norm_ratio` degraded less on the pathological case (6× vs 38× error), which
is weak evidence for its robustness, but far too little to justify a default change.

### 2. Why the headline numbers moved vs the baseline

Whole-benchmark medians came out much worse than the 2026-08-07 baseline for
courtship (male 7.2 → ~20 px) while free-running reproduced almost exactly
(4.13 → 4.12 px). Config diff between the source run and a variant run shows
**no meaningful difference** (only the new `scaling.scale_keypoints: trunk` key,
which preserves default behaviour). The degradation is entirely explained by the
scale defect above: the benchmark re-runs re-fit the shared scale from a
different (and for S0, pathological) first bout.

### 3. Input-integrity audit

`python -m scripts.benchmark.input_divergence` over all 84 runs:

- **15/21 bout-flies CLEAN** — byte-identical frozen inputs across all four variants.
- **2/21 near-identical** — S1 17_52_50 b1; regenerated but variants agree within ~1 px.
- **4/21 CONFOUNDED** — S1 15_25_51 b8/b29; variants differ **from each other** by up
  to 428 px, so these are not a controlled comparison.
- 0 incomplete, 0 shape mismatch.

Cause: `scripts/run_bout.py:425-434` deletes `kp2d/kp3d/kp3d_filt` (and all
downstream artifacts) whenever `masks_are_stale()` fires — recordings whose
`sync_plan` status is `trim`/`reindex`. The freeze mechanism only pre-seeds those
files, so it cannot survive a stage that deletes them.

## Restricted comparison

Excluding the scale-poisoned recording (S0) and the 2D-confounded one
(S1 15_25_51) leaves 13 uncontaminated bout-flies:

| variant | free-running reproj | male reproj | female reproj | male IoU | female IoU |
|---|---|---|---|---|---|
| trunk_umeyama_segcal | 4.115 | 9.762 | 9.482 | 0.1496 | 0.1465 |
| **trunk_umeyama_nosegcal** | **3.778** | **9.393** | 9.205 | 0.1505 | 0.1469 |
| all_norm_segcal | 4.294 | 10.698 | 9.565 | 0.1500 | 0.1473 |
| all_norm_nosegcal | 3.903 | 10.256 | **9.167** | 0.1505 | 0.1475 |

Read cautiously — n = 4–5 bout-flies per cohort:

- **Segment calibration OFF is better or equal on every cohort**, for both scale
  estimators (e.g. male 9.76 → 9.39 for trunk; 10.70 → 10.26 for all+norm). This is
  the most consistent signal in the table and matches the original observation that
  per-segment calibration makes the meshes look wrong.
- **trunk vs all+norm_ratio is a wash**: trunk wins on reprojection, all+norm wins
  marginally on silhouette IoU and on female reprojection. No winner.
- Differences of ~0.3–0.5 px on n=4 are not separable from noise here.

## Recommended next steps

1. **Fix scale estimation (highest value).** Estimate the recording's body scale
   from an aggregate over bouts/flies — e.g. the median of per-bout-fly estimates
   with outlier rejection — instead of the first bout-fly. This removes both the
   38× failure mode and the run-to-run irreproducibility. Track 1's original
   question should be re-asked only after this lands.
2. **Make freezing survive invalidation.** Add a flag (e.g.
   `sync.invalidate_stale=false`) so benchmark variants keep pre-seeded inputs,
   then re-run the 6 affected bout-flies so all 21 are controlled.
3. **Re-run the 2×2** once (1) and (2) are in, and only then decide the default.
   Carry the segment-calibration-off signal into that run as the leading hypothesis.

## Caveats

- Courtship male/female cohort labels for 3 recordings are provisional
  (`sex.json` absent; `_male_fly()` falls back to fly1=male).
- `docs/benchmark/2026-08-07-baseline-scorecard.json` remains the reference for
  production-configuration numbers; the variant scorecards here are **not**
  comparable to it in absolute terms, for the reason in §2.
