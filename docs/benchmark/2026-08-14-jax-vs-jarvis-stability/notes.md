# jax_vitpose vs JARVIS fly50_V6: where the stability gap actually is

**Question** (raised by the IK-explainer review): the shipped JARVIS
`data3D.csv` (fly50_V6, after removing its rigid cropping bias) looks far more
stable than our jax_vitpose pipeline's 3D on the Session6 clip — is our 2D
detector worse than the original JARVIS implementation?

**Answer: no.** The 2D detectors are comparable (ours is slightly *smoother*
at the noise floor and slightly worse on catastrophic tarsal-tip swaps).
The stability gap is in the **3D lifting**: JARVIS's HybridNet fuses
per-camera heatmaps in a 3D grid, which out-votes a single camera whose 2D
swapped to the wrong leg; our confidence-gated plain DLT
(`jarvis_jax/tracking/triangulate.py`) has no reprojection-residual outlier
rejection, so one swapped view at conf ≥ 0.3 drags the whole 3D point.

## Setup

- Clip: `/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/Clip/Session6/2025_10_12_15_06_46`
  (7 cameras, 921 frames @ 800 fps, single fly, easy open-arena case).
- Ours: `ik_explainer/predictions/02_kp2d.npz` (jax_vitpose v4, SAM3 4-ch,
  decode_sharpen 3.0) → `03_kp3d.npz` (plain DLT) → `04_kp3d_filt.npz`
  (temporal filter).
- JARVIS: shipped `data3D.csv` (full fly50_V6 stack, CenterDetect +
  KeypointDetect + HybridNet fusion), plus a fresh `predict2D` run of the SAME
  videos with the SAME weights (KeypointDetect `Run_20260811-105944`,
  project copy `third_party/JARVIS-HybridNet/projects/fly50_V6_infer` —
  config minus the two augmentation keys this checkout's yacs schema lacks).
- Metric: magnitude of the 2nd temporal difference (accel, mm/frame²);
  translation-invariant, so the rigid data3D bias
  (−0.112, −0.082, +0.072) mm ≈ 12.6 px cannot affect it.
- Spike = (frame, keypoint) with accel > 0.1 mm/frame² (~8 px equivalent).

## The 2×2 (+2) result — n = 921 frames × 50 keypoints

| variant                                | med accel | p99 accel | spikes>0.1 |
|----------------------------------------|-----------|-----------|------------|
| JARVIS 2D + HybridNet fusion (shipped)  | 0.0105    | 0.0738    | **156**    |
| jax_vitpose 2D + plain DLT (03_kp3d)    | 0.0106    | 0.1441    | **834**    |
| JARVIS 2D + plain DLT                   | 0.0171    | 0.1303    | **849**    |
| jax_vitpose 2D + robust DLT (12 px LOO) | 0.0106    | 0.0908    | **340**    |
| JARVIS 2D + robust DLT                  | 0.0171    | 0.0958    | **404**    |
| jax_vitpose + plain DLT + temporal filter (04) | 0.0046 | 0.0421 | 6      |

Reading the table: hold the lifting fixed and swap the detector — spike
counts barely move (834↔849 plain, 340↔404 robust; ours slightly better both
times). Hold the detector fixed and swap the lifting — spikes drop 5×.
The detector is not the problem; the lifting is.

Raw 2D head-to-head (same frames, per camera): median accel ours 1.10 px vs
JARVIS 2.00 px (their CSV is integer-quantized); >20 px jumps ours 1788 vs
JARVIS 1973; >50 px tarsal swaps ours 693 vs JARVIS 368. So jax_vitpose has
a somewhat higher *catastrophic* swap rate at leg tips but is otherwise equal
or better — and through identical lifting it produces slightly *fewer* 3D
spikes than JARVIS's own 2D.

## Spike anatomy

- 75% of our 834 spike events (630) have exactly ONE camera with a >20 px 2D
  jump; 121 more have two. They are single-view left/right or
  neighboring-tarsal swaps, concentrated on `*_TaTip`/`*_TaT3`/`*_TaT1`.
- Confidence cannot gate them: median conf3d at spike events is 0.889
  (overall 0.945); the swapped 2D peaks carry conf 0.4–0.8.
  (Same lesson as the keypoint-order bug: confidence/QC metrics are blind to
  "wrong place, confidently".)
- Visual confirmation: `figures/2026-08-14-jax-vs-jarvis-stability/`
  `swap_event_f441_cam853.png` — T1R_TaTip on Cam2012853 flips to the raised
  contralateral foreleg tip at f=441/443 and back at f=442/444, while the
  debiased data3D reprojection stays put.
- Robust re-triangulation (iterative leave-worst-view-out when reprojection
  residual > 12 px, keep ≥ 2 views) discards only 1.8% of view-points and
  removes 60% of spike events with zero added NaNs.

## Notes and caveats

- The rigid data3D bias matches the user's cropping-artifact explanation:
  pure translation (body-length ratio 0.9971, lag scan flat), so it drops out
  of every jitter number here.
- The earlier HybridNet-vs-DLT A/B (frozen 13-bout benchmark) scored the two
  a wash and dropped HybridNet — those metrics looked at accuracy-style
  medians, which this analysis confirms are identical. The fused lifting's
  advantage lives entirely in the p99 tail / spike rate, which the benchmark
  scorecard did not measure. Consider adding a spike-rate metric to the
  benchmark before revisiting that decision.
- Our standard temporal filter (Stage B2) already suppresses the spikes
  numerically (6 events), but a filter can only smooth *through* a swap, not
  reject it; fixing at triangulation is upstream and principled. The two
  compose.
- This is the EASY clip (single fly, open arena). The female-fly hard cases
  have fewer good views per keypoint, where dropping a swapped view matters
  even more — but verify there before generalizing (per CLAUDE.md).

## Recommendation

Add reprojection-residual outlier rejection to
`jarvis_jax/tracking/triangulate.py::triangulate_keypoints` (iterative
leave-worst-view-out, threshold ~12 px, only when ≥ 3 valid views), then
re-run the frozen 13-bout benchmark with a new spike-rate column plus the
standard scorecard to check nothing regresses.

## Follow-up (same day): robust triangulation implemented + benchmark A/B

`triangulate_keypoints` gained `reproj_resid_px` (default null = byte-identical
no-op; tests: `third_party/jarvis_jax/tests/test_triangulate_outlier_reject.py`).
Two design lessons, both measured, both now encoded as tests:

1. **Greedy worst-view dropping diverges.** With 2 of 6 views swapped, the
   dragged least-squares solution hangs its worst residual on a GOOD view;
   greedy removal walked the error 248 -> 534 mm. Replaced with deterministic
   consensus: triangulate from every valid view pair, keep the largest view
   set that reprojects within the band of that pair's solution.
2. **Margin-free consensus churns on diffusely-bad 2D.** Full-benchmark A/B
   of the margin-free version (band=trigger=10 px, min 2 views; variant
   `dlt_robust`, frozen kp2d, Stage B onward re-run) improved every cohort
   aggregate (spike rate: free-running -66%, male -19%, female -18%; female
   soft-IoU +28%, female close-frame reproj -47%) but REGRESSED three
   bout-flies: freerun_13_25_13 b7 (+64% spikes), male b12 fly1 (+80%),
   female b8 fly0 (+25%). Diagnosis (figures `b12fly1_worst_point.png`,
   sweep scripts beside the figures): those bouts' 2D is DIFFUSELY
   inconsistent across cameras (b12 fly1: median max-residual 35.6 px, 59% of
   points > 30 px, male climbing the wall edge with legs occluded -- vs 5.7%
   on a healthy male bout; NOT identity confusion, the flies' 2D centroids
   are ~285 px apart in every camera). When most frames trigger, the winning
   view-subset changes frame to frame and the subset-to-subset solution jump
   IS the new jitter. A 2-view consensus can also lock onto one of two
   clusters arbitrarily (b8).

**Final parameters** (kp3d-level sweep over all 21 frozen bout-flies, scripts
`consensus_min3_sweep.py` / `consensus_trigger_sweep.py`): inlier band 10 px,
intervene only when some view exceeds 3x the band (TRIGGER_FACTOR -- real
swaps are 100-400 px, diffuse noise is 10-50 px), consensus must keep >= 3
views (MIN_CONSENSUS_VIEWS). Effect per cohort (spike-rate mean,
plain -> margin-free -> final):

| cohort        | plain  | margin-free | final  |
|---------------|--------|-------------|--------|
| free-running  | 0.0563 | 0.0144      | 0.0146 |
| courtship male| 0.0633 | 0.0569      | 0.0556 |
| courtship female | 0.3131 | 0.2774   | 0.2737 |

Regressions: freerun_13_25_13 FIXED (0.0104, better than plain), female b8
FIXED (0.5474 vs plain 0.5647), male b12 halved but still above plain (0.155
vs 0.0997) -- that bout-fly's 2D is broken upstream (wall-edge OOD male;
same class as the female wall problem) and is a detector/training issue, not
a lifting issue. Session6 clip with final params: 834 -> 515 spikes (the
margin sacrifices sub-30-px swaps; Stage B2 temporal filtering still runs
downstream). Per-bout figure: `benchmark_perbout_spikes.png`.

**STAC-level scorecard, final parameters** (variant `dlt_robust_t30` vs
`dlt_plain`, both from identical frozen kp2d, full Stage B -> STAC -> QC):

| cohort | metric | plain | final | delta |
|---|---|---|---|---|
| free_running (n=5) | kp3d_spike_rate | 0.0325 | 0.0142 | **-56%** |
| free_running | soft_iou_median | 0.140 | 0.140 | +0.4% |
| courtship_male (n=8) | kp3d_spike_rate | 0.0443 | 0.0341 | **-23%** |
| courtship_male | reproj_px_close | 13.60 | 13.48 | -0.9% (min2 blowup +317% GONE) |
| courtship_male | soft_iou_median | 0.130 | 0.129 | -1.1% |
| courtship_female (n=8) | kp3d_spike_rate | 0.2807 | 0.2426 | **-14%** |
| courtship_female | soft_iou_median | 0.0787 | 0.1151 | **+46%** |
| courtship_female | hard_iou_median | 0.0141 | 0.0195 | **+38%** |
| courtship_female | reproj_px_close | 79.10 | 48.90 | **-38%** |
| courtship_female | jitter_median | 0.0025 | 0.0021 | -14% |

3/21 bout-flies show any >5% regression: b12 fly1 spikes (broken wall-edge 2D,
IoU unchanged), b6 fly0 soft-IoU -5%, b8 fly0 soft-IoU -5% at the 0.009
catastrophic-bout floor. Biggest single mover: female b14 fly0 soft-IoU
0.065 -> 0.146 (still comparison: `b14fly0_still_plain_vs_t30.png`). Male
jitter_median rose 0.0020 -> 0.0028 rad/frame^2 (absolute ~0.05 deg -- noted,
not actioned). `detector.reproj_resid_px: 10.0` is now the default in
`configs/detector/vitpose_v3.yaml`; null restores byte-identical old
behaviour.

## Session6 clip regenerated with the fix (the original goal)

The clip's own pipeline artifacts were rebuilt with `reproj_resid_px=10`
(`scripts/viz/ik_explainer/triangulate3d.py` now passes it; plain versions
kept beside as `*_plain.npz`): 03_kp3d spikes 834 -> 515 (p99 accel 0.144 ->
0.106 mm; JARVIS fused = 156/0.074). The motivating T1R_TaTip swap
(f441-443, Cam2012853) is eliminated: 3D accel 1.1-1.6 -> 0.03-0.11 mm
(`swap_event_f441_cam853_fixed.png` -- raw 2D still jumps, the 3D no longer
follows it). The production STAC solve was regenerated on Hyak from the new
04_kp3d_filt (`ik_production/stac_ik_full.h5`, via the same
fit_offsets_once + ik_only_bout path run_bout uses; the workstation original
was never synced), stage residuals unchanged (13.404/1.710/0.118/0.017 mm),
and all four acts + `ik_explainer.mp4` (975 frames, 32.5 s) re-rendered and
re-assembled from the improved data on the `enhanced/` display videos.
Gotcha for future re-renders: the acts' frame dirs held STALE frames from
the longer pre-TASK-29/30 edits (act1 330 vs 150 expected, act4 900 vs 540)
-- assemble.py's count check caught it; delete indices >= the expected count
before assembling.

Figures + generating scripts + `.npz`/`.json` live in
`figures/2026-08-14-jax-vs-jarvis-stability/` (scripts copied beside the
outputs: `jitter_compare.py`, `spike_analysis.py`,
`robust_triangulate_test.py`, `compare_2d.py`, `cross_triangulate.py`,
`swap_frames_fig.py`, `final_summary_fig.py`). JARVIS 2D was produced with:

```bash
cd third_party/JARVIS-HybridNet
python -c "
from jarvis.utils.paramClasses import Predict2DParams
from jarvis.prediction.predict2D import predict2D
p = Predict2DParams('fly50_V6_infer',
    '/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/Clip/Session6/2025_10_12_15_06_46/videos/')
predict2D(p)"
```

(`videos/` is a symlink dir mapping `CamXXXXXXX.mp4` → the clip's
`CamXXXXXXX_frames_1161383_1162303.mp4`.)
