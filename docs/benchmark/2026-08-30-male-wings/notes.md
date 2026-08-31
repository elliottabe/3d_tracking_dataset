# Does 2D wing L/R mirror confusion corrupt the MALE's 3D wing identity? (bout 28)

Diagnosis only, no pipeline changes. Full methodology, per-task numbers, and
figure-by-figure readback are in the (untracked) task report; this is the
committed summary + pointer.

Data: Session0 `2025_10_20_13_20_04` bout 28 (2007 frames, 7 cams), archived
arrays at `figures/2026-08-29-c2f-3d/phase0-baseline/arrays/bout_00028/fly1/`
(fly1 = male), produced with detector `v4_8gpu_20260808`. Figures
(gitignored, regenerate via the scratch scripts listed in the full report):
`figures/2026-08-30-male-wings/`.

## Headline numbers

- **View survival** (post `conf_thresh`+`view_conf_thresh`+consensus-gate
  views/7 cams, male): median is **7/7 for every wing keypoint** — the
  worry that wings "routinely triangulate from 2-3 views" does not hold.
  The only dip is `WingL_V12`/`WingL_V13`'s 25th percentile (5/7, mean
  6.3), specifically while that wing is actively extended.
- **Precision**: reprojection residual for all 6 wing keypoints (1.9-2.6 px
  median) sits in the same band as `Scutellum`/`T2L_FeTi` controls
  (1.8-2.1 px) — not a residual outlier. Apparent wing-tip "jitter"
  (2.3-4.0x higher acceleration than control) tracks independently-measured
  wing-tip activity; the wing-BASE points (near the hinge) match the
  control's flat ~1.3-1.4x ratio exactly — noise-floor, not swap damage.
- **Identity**: a geometric check (`WingL_base` vs `WingR_base` lateral
  position relative to an `EyeL-EyeR` body axis, validated against a
  known-correct leg-pair reference) finds **0/2007 frames (0.0%) with wing
  L/R swapped**, on the raw triangulated (pre-IK) 3D. Confirmed visually by
  reprojecting into 2 cameras at each of two clear single-wing-extension
  moments (one L-extended, one R-extended) — the label lands on the
  visibly-extended wing both times.
- **Mid-bout stability**: which wing is extended DOES change during the
  bout (genuine alternating unilateral extension — 2 long runs, 737 and 924
  frames, covering 82% of the bout, joined by a smooth multi-frame
  crossover, not a jump), but the identity LABEL never flips. The two are
  cleanly separable phenomena here.
- **New v5 detector** (`v5_s70_bal_augdef_full`, val MPJPE 5.9 vs v4's
  8.6 px): re-ran 2D+triangulation on 600 representative frames (3
  windows spanning L-extended / crossover / R-dominant). Wing mirror rate
  and view survival are statistically indistinguishable from v4 on the
  same frames (pooled 97.7% -> 99.1%, within noise; view survival within
  0.1 views of each other everywhere) — **the better detector does not
  change this pattern**, consistent with it being a viewing-geometry/motion
  effect (an actively swept wing tip at grazing camera angles) rather than
  a fixable detector-quality problem. 0/600 identity swaps with v5 too.

## Context found in repo history

Commit `021ece1` (2026-08-28, "protect the target's wings from the
distractor fill") had already measured, on this exact bout: "wing L/R
lateral order flipped on 3.33% of temporally-adjacent frames [2D level]
... every one of the worst cases was fly0 [female] — consistent with the
male's wing tips never entering the fill in this bout (0.0% at both
radii)." This diagnosis's finding (male wing identity is stable and
correct in the final 3D) is consistent with, and extends, that earlier,
narrower measurement.

## Verdict

Multi-view consensus is doing its job for the male: 2D wing keypoints are
frequently mirror-confused (as the prior mirror-in-3d diagnosis found,
94.4% of large 2D errors), but that confusion is either gated out before it
reaches the 3D solve, or does not survive as a coherent L/R swap in the
final result — **zero measured identity errors, in any of 2607 male frames
checked across two detector versions**. This is a genuine negative result:
no fix is needed for the male's wing identity in this bout. The one real,
reproducible-but-benign effect is that the actively-extended wing's tip
veins (`WingL/R_V12`, `WingL/R_V13`) see somewhat reduced camera coverage
(~5/7 vs 7/7) while swept through its stroke — a precision/coverage note,
not an identity-correctness problem, and not something the new v5 detector
changes.

Full report: `.superpowers/sdd/2026-08-29-coarse-to-fine-3d/male-wing-mirror.md`
(untracked/gitignored — this file is the durable, committed pointer).
