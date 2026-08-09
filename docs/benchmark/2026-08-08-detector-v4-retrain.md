# Detector retrain on red_data_unified_V4 — results

**Date:** 2026-08-08
**Run:** `v4_8gpu_20260808` (ViTPose, 30k steps, batch 32, 8x L40S, MAE-pretrained
backbone, `sampling=balanced` + `aug=heavy`)
**Motivation:** audit recommendation #4 — the female's 2D confidence is the
dominant correlate of bad 3D (r = -0.85), and V4 adds 5,054 previously-excluded
labelled images (headless, amputated, female-on-wall).

## Headline

Evaluated head-to-head on the **same** val set (V4 val, n=2927) with
`jarvis_jax.scripts.eval_keypoints_2d`:

| checkpoint | overall MPJPE | female 11_56_05 |
|---|---|---|
| `v4_kp_maskaware_fc` (production) | 10.090px | 29.104px |
| `vitpose_hardmine_ft` | 10.226px | 30.285px |
| **`v4_8gpu_20260808` (new)** | **8.554px** | **16.523px** |

Overall **-15.2%**; on the hard female courtship recording **-43.2%**.

## Per-recording, and why three "regressions" are not regressions

| recording | category | in V3 | old | new | |
|---|---|---|---|---|---|
| 2026_05_27_11_56_05 | courtship_11_50_female | val | 29.104 | **16.523** | -43% |
| 2026_05_27_11_57_05 | courtship_11_50_male | val | 8.143 | **6.259** | -23% |
| 2026_02_13_13_44_49 | amputated (S8) | absent | 7.854 | **5.432** | -31% |
| 2026_06_09_15_38_35 | headless_56_42 | absent | 14.237 | **9.791** | -31% |
| 2026_06_09_15_46_55 | headless_22_50 | absent | 11.545 | **7.589** | -34% |
| 2026_06_10_15_05_02 | headless_56_42_1 | absent | 12.426 | **8.955** | -28% |
| 2026_04_01_16_23_08 | courtship_V4 | **train** | 5.343 | 7.551 | +41% |
| 2026_06_18_19_23_03 | courtship_25_51_male | **train** | 18.243 | 23.132 | +27% |
| 2026_06_19_11_09_36 | courtship_25_51_female | **train** | 14.914 | 16.444 | +10% |

The three recordings where the new model scores worse are **exactly** the three
that were in V3's *training* split. The old model's numbers there are
train-set scores; the new model's are held-out. That is not a like-for-like
comparison and should not be read as a regression. (It is also mild: on
2026_06_18_19_23_03 the old model manages 18.2px on data it trained on, versus
23.1px held-out for the new one.)

Splitting the val set by what the comparison can actually support:

- **Strictly fair** — held out for *both* models (n=208): **18.523 -> 11.342px,
  38.8% better.** This is the only unambiguous number in the table, and it
  includes the female recording that motivated the whole exercise.
- **New categories** the old model never trained on (n=1873): 9.320 -> 6.444px,
  30.9% better. Expected rather than surprising — this is the V4 data paying
  off — but it is what makes headless/amputated recordings processable at all.
- **Old model's training data** (n=846): not comparable, see above.

## Throughput

30,000 steps in **1h46m** on 8x L40S = 212ms/step at batch 32. Recorded because
audit finding 4 noted speed was entirely unmeasured; this is a training number,
not an inference one, so it does not close that gap.

## Caveats

- MPJPE is measured on annotated 2D val frames. It does not follow that 3D and
  IK improve proportionally — the frozen 13-bout benchmark is the test that
  matters, and it has not been re-run yet.
- `wall` (33 annotations) and `grooming`/`female_general` are train-only
  categories, so val cannot measure them at all. The female-on-wall case — the
  hard case from the audit — is invisible to every number above.
- The balanced sampler down-weights `courtship_other` (0.76x) and `grooming`
  (0.73x) relative to uniform. If a later benchmark shows courtship_other
  degrading, `sampling.balance_alpha` is the knob.

## Next

1. Re-run the frozen 13-bout benchmark against this checkpoint (`detector.ckpt`
   override) — confirm the 2D gain propagates to 3D/IK.
2. Only then promote it in `configs/detector/vitpose_v3.yaml`, which currently
   points at `v4_kp_maskaware_fc/final`.

Note: `configs/paths/hyak.yaml:vitpose_ckpt` points at
`v3_8gpu_20260620/final`, which **does not exist** on disk. The pipeline reads
`configs/detector/vitpose_v3.yaml` instead, so this is a stale path rather than
a live break, but it is misleading and should be corrected when the new
checkpoint is promoted.
