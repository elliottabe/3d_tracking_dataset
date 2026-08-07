# Tracking & IK Upgrade Program — Design

**Date:** 2026-08-07
**Status:** Approved by user (brainstorming session)
**Repos:** `3d_tracking_dataset` (walking/preprocess pipeline, benchmarks) and
`third_party/jarvis_jax` (courtship + free-running inference pipeline)

## Problem

The fly-ID review GUI made the underlying quality problem visible: courtship
tracking and IK are not good enough, even for the male. With per-segment
calibration (`segment_calibration: true`, v1 model) the body-model meshes look
visibly wrong. Beyond scale, the known failure modes are leg posture
(leg-curl — STAC leg-chain local minima), female quality (walls/OOD poses,
occlusion near the male), and temporal jitter.

**Program goal:** courtship keypoints and IK work as well as free-running,
with multi-animal understanding. Free-running is the quality reference; the
courtship male and female cohorts should close the gap to it, and the
remaining delta must be attributable only to genuine multi-animal hardness.
One stack, one detector, one body-model configuration serves both assays —
multi-animal understanding is added capability, never a pipeline fork.

## Prior findings this design builds on

- IK is the accuracy bottleneck; ViTPose→DLT stays (HybridNet dropped).
- Root-orientation/legs are ill-conditioned in the IK (cond ~5e7); anatomy
  morphing gave only −4.5% residual.
- Silhouette polish cannot fix leg-curl — the defect is upstream in STAC.
- 2D shimmer was diffuse-heatmap decoding; decode_sharpen 3.0 shipped.
- Pre-STAC keypoint filtering (7× less jitter) and bridge_mode=keypoint shipped.
- The v2_3 scale estimator (`all` markers + `norm_ratio`) fixed
  markers-floating-off-mesh in the walking pipeline (commit a2f9f4a).
  `all` + umeyama is provably wrong (Umeyama scale ≤ norm ratio; collapses
  −34%) — the correct combination is `all` + `norm_ratio`.
- Courtship runs used the v1 model with trunk-umeyama global scale
  (`jarvis_jax/tracking/scale.py`) + per-segment calibration
  (`segment_fit.py` → `segment_scales.json`).
- A 387-clip walking reference dataset exists
  (`Fruitfly_v2_3_walk_1000hz_interp_padded.h5`).
- Keypoint-gated 3D mask fusion (recovers SAM leg dropouts near the other
  fly) was prototyped during silhouette-polish work.
- Curated female detector training data exists (`red_data_unified_V3`).

## Success criteria

- Benchmark scorecard reports each courtship metric absolutely AND as a
  gap-to-free-running ratio; the headline number is that ratio approaching 1.0.
- An upgrade lands only if (a) no cohort regresses (free-running, courtship
  male, courtship female) and (b) its targeted metric improves.
- "Better" is defined by the suite, not by any single bout looking nicer.

## Program structure

Parallel tracks on separate branches, each landing on the benchmark
independently. Merge order on conflict-free wins: Track 1 → 4 → 2 → 3
(cheap/deterministic before learned). Final combined benchmark run, then full
re-processing of both assays and a human-eyes pass over the review videos.

## Track 0 — Benchmark + metric suite (foundation)

- ~12–15 frozen bouts across three cohorts: free-running (single fly),
  courtship male, courtship female; each spanning easy + hard (wall poses,
  close interaction/occlusion, wing extension).
- 2D inputs locked on disk; experiments re-run only 3D/IK stages.
- Courtship bouts get a per-frame partner-proximity annotation (inter-fly
  centroid distance from existing masks/keypoints); the scorecard splits
  courtship metrics into "flies apart" vs "flies close".
  - If courtship-apart already matches free-running, the remaining gap is
    interaction handling (Tracks 2/3 aim there).
  - If courtship-apart is also worse, something assay-level (arena, cameras,
    scale) is broken — surfaced immediately.
- One eval runner emits a per-cohort scorecard:
  - per-keypoint-group reprojection error (median + tail percentiles)
  - silhouette IoU per view
  - joint-angle jitter (high-frequency power above the behavior band)
  - joint-limit-violation rate
  - fixed render grid (same frames every run) for visual comparison
- Baseline scorecard = current pipeline as shipped.

## Track 1 — Scale/anatomy (work item #1)

- Port `all` + `norm_ratio` into `jarvis_jax/tracking/scale.py` as a
  config-selectable estimator (trunk-umeyama stays the default until the A/B
  decides), honoring that module's "keep in sync" docstring note.
- Run the 2×2 on benchmark bouts from BOTH assays (all on the v1 model —
  the walking-pipeline v2.3 finding does not automatically transfer):
  {trunk-umeyama, all+norm_ratio} × {segment_calibration on, off}.
- Deliverables: a chosen default for both assays; quantified split of how
  much mesh weirdness was scale vs IK (informs Track 2 priorities).
- Out of scope, parked: migrating courtship to the v2.3 model (the
  silhouette stack is v1-hardcoded).

## Track 2 — Leg posture & temporal stability

Approach: two new cost terms inside the existing jaxls STAC, both
off-by-default config options. The frozen solver machinery is untouched.

- **Leg pose prior:** Gaussian/PCA energy on leg joint angles fit from the
  387-clip walking reference dataset; penalizes improbable chain
  configurations (curled-leg minima). Shared across assays; wing DOFs
  excluded so courtship wing extension is unaffected. Start with a weak
  weight; upgradeable to a richer density model later if the benchmark
  demands it.
- **Windowed IK:** solve short overlapping windows with a qpos-smoothness
  term instead of independent frames, so well-observed frames pull ambiguous
  neighbors out of bad minima. Window size, overlap, and weight are
  benchmark-tuned; the per-frame path remains available.

Contingency (trigger-gated, not scheduled): silhouette-in-the-loop IK
(`silhouette_ik*.py` exists) only if the benchmark still shows
silhouette-visible leg residuals after both terms land. Parked: learned 3D
lifter (qpos regression network) — biggest ceiling, biggest lift, weakest
OOD trust.

Rejected for now: silhouette-first IK as the primary fix (polish already
proved the defect is upstream of the silhouette term; jaxls LM struggled
with px-scale silhouette costs).

## Track 3 — General detector quality + multi-animal understanding

- One general checkpoint: retrain on `red_data_unified_V3` PLUS the
  free-running training data combined; gated on all three cohorts (male and
  free-running must not regress while female improves).
- One hard-mining round (scoped to a single iteration to avoid pseudolabel
  drift): harvest failures flagged by courtship QC and free-running QC via
  the existing pseudolabel-finetune loop (`courtship_pseudolabel.py`,
  `run_pseudolabel_finetune.py`), targeting wall poses and near-partner
  occlusion.
- Land keypoint-gated 3D mask fusion (SAM leg-dropout recovery near the
  other fly; inert for single-fly).
- These multi-animal items are the explicit mechanism for closing the
  "flies close" gap in the scorecard.
- Out of scope: identity assignment (fly-ID GUI + sexing canonicalization
  already cover it).

## Track 4 — Confidence-weighted triangulation

Down-weight per-camera detections by heatmap confidence in the DLT solve
(currently unweighted). Small, independent, identical for both assays,
benchmark-gated like everything else.

## Error handling / attribution discipline

- Every experiment runs against the frozen benchmark inputs; no experiment
  mutates shared session outputs.
- Each track lands alone before merging; the combined run re-verifies no
  interaction regressions.
- Scorecards are committed artifacts (JSON + render grid) so history is
  auditable.

## Risks & mitigations

- **Pose prior too strong** → grooming/courtship-specific leg poses
  penalized. Mitigation: weak default weight, legs-only coverage, benchmark
  trade-off curve (jitter vs reprojection) before adoption.
- **Windowed IK wall-clock cost** → per-bout runtime rises. Mitigation:
  batch windows in jaxls; measure on benchmark before adopting.
- **Detector retrain regresses male/free-running** → per-cohort gate exists
  precisely for this; single hard-mining iteration bounds drift.
- **Scale change interacts with Track 2 tuning** → merge order (1 before 2)
  plus combined final run.

## Out of scope (parked)

- v2.3 body-model migration for courtship.
- Learned 3D lifter.
- Identity/sexing changes.
- HybridNet revival.

## Sub-project decomposition

Each track is its own spec → plan → implementation cycle; this document is
the program umbrella. First cycle to plan: **Track 0 (benchmark suite)**,
since every other track gates on it; Track 1 can be planned in the same
session as it is small and depends only on Track 0's bout selection.
