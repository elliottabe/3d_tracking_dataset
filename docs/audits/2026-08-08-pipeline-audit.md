# Pipeline audit: JARVIS vs JAX rewrite, and is SAM3 earning its place?

**Date:** 2026-08-08
**Question:** the JAX rewrite was meant to be faster and at least as accurate as
JARVIS-HybridNet; SAM3 segmentation was added on the theory it would help
tracking. Are either of those true, and what should we do given a large backlog
of recordings to process?

## Method

Both pipelines have outputs for the *same* recording and bouts
(Session0/2025_10_20_13_20_04): the original JARVIS runs survive at
`Video_recordings/.../Predictions_3D_34662304/bout_*/fly{0,1}.csv` (3D x,y,z,conf
per keypoint, fly50 order), and the current pipeline's `kp3d.npz` sits under
`processed/courtship/.../pose/bouts/`.

Accuracy is scored with **within-bone CV**: the coefficient of variation, across
frames, of the distance between two keypoints on the same rigid segment. A tibia
cannot change length, so any variation is error. This needs no ground truth and
is pose-invariant. Fly indices are NOT comparable across pipelines (sex
canonicalisation reindexed the processed data, and bout 9 shows the JARVIS dirs
carry their own ordering), so bouts are compared best-fly vs best-fly and
worst-fly vs worst-fly.

## Finding 1 — the JAX rewrite is not an accuracy regression

| bout | JARVIS best / worst | JAX best / worst |
|---|---|---|
| 1 | 3.2% / 33.9% | 3.2% / 28.4% |
| 3 | 4.8% / 22.5% | 4.5% / 29.7% |
| 6 | 4.8% / 34.5% | 3.5% / 35.0% |
| 9 | 4.2% / 23.8% | 3.9% / 24.6% |
| 14 | 7.1% / 45.4% | 3.7% / 62.0% |
| 22 | 11.0% / 48.8% | 4.9% / **418.8%** |
| **median** | **4.8% / 34.2%** | **3.8% / 32.4%** |

On the well-tracked fly the JAX pipeline is **better** (median 3.8% vs 4.8%), and
its advantage grows on the harder bouts (14: 7.1→3.7; 22: 11.0→4.9). On the
poorly-tracked fly the two are equivalent in the median (32.4% vs 34.2%).

## Finding 2 — the female problem predates the rewrite and predates SAM3

JARVIS's worst fly also sits at a 34% median within-bone CV. Whatever makes the
second fly hard, it is **not** caused by the JAX port or by introducing SAM3. It
is a long-standing property of tracking the female in this assay. Corroborating
evidence from the current pipeline: her 2D keypoint confidence is 0.70–0.89 vs
the male's 0.96–0.97, and across 28 bout-flies, 2D confidence correlates with
bad 3D at r = −0.85 (camera coverage only −0.63).

> **CORRECTION (2026-08-08, later the same day).** Finding 3 below identifies a
> real symptom but MISATTRIBUTES its cause, and its recommendation (#2) was
> unsafe as written. Measurements across all 28 Session0 bouts:
>
> - SAM3 segments the fly in **93.5%** of (fly, camera, frame) views, and
>   **85.2%** of fly-frames have all 7 cameras. Coverage is good.
> - Of the missing views, **5.1% are physically OUTSIDE the camera's field of
>   view** and only **1.4%** are in frame and unsegmented. In bout 22 the female
>   is out of frame ~89% of the time in the four "failing" cameras — she is at
>   the end of a long narrow arena that those cameras do not cover. SAM3
>   reporting "not found" there is CORRECT.
> - The bad 3D is therefore a **camera-coverage** problem (rig geometry), not a
>   segmentation one. `masks.min_views: 4`, shipped before this audit was
>   written, already gates those frames.
> - The 1.4% that IS fixable has a specific cause found later: SAM3 prompts at
>   `frame_index 0` and propagates forward only, so a fly entering a camera
>   mid-bout is never picked up there (bout 22: she enters 4 cameras at t≈1200
>   of 1393). Fixed by `repair_missing_cameras` (commit f0077ec).
> - Recommendation #2 ("fall back to a CenterDetect-style centroid so a crop
>   always exists") is **unsafe ungated**: on a verified-empty crop the detector
>   still reports median confidence 0.332 with 72% of keypoints above
>   `conf_thresh=0.3`. A misplaced crop yields confident garbage, and garbage
>   that triangulates is worse than a dropout that yields NaN. See
>   `docs/benchmark/2026-08-08-detector-v4-retrain.md` and the per-view gate
>   (commit e1a4ae4).
>
> Retained below unedited, since the reasoning it records is what the later
> measurements were built to test.

## Finding 3 — SAM3 adds a catastrophic tail

The medians are equivalent, but the tails are not. Bout 22: JARVIS 48.8%, JAX
**418.8%**. Bout 14: 45.4% vs 62.0%. The mechanism is established:
`sam3_masks.npz` shows the female validly segmented in only **3 of 7 cameras** in
bout 22 (4/7 in bout 26, 5/7 in bout 13), while the male is 7/7 in every bout.
SAM3 correctly reports "not found" rather than mislabelling — but triangulating
from three views is ill-conditioned, and the resulting 3D collapses.

JARVIS's CenterDetect degrades gracefully instead: it always emits a centre per
animal, so the downstream crop exists even when it is poor. SAM3 substitutes a
hard dropout for a soft error. That is the wrong trade when the consequence is
unusable 3D rather than merely noisy 3D.

SAM3 does earn its place elsewhere: distractor gray-fill for the 2D detector, and
the silhouette term used by polish/QC. This finding is about its role in
*localisation*, not about removing it.

## Finding 4 — speed is unmeasured

No throughput comparison exists. No SAM3 timing logs survive under
`processed/courtship`, and the original JARVIS runs left no timing artefacts.
Measured JAX-side numbers: benchmark tasks ran 4,282 fly-frames in a median
92.6 min (full pipeline incl. STAC, polish, FK, overlays, video). Bouts occupy
only **2.6%** of a recording (21,518 of 821,128 frames), so whole-recording
processing is ~38x today's work: roughly 595 GPU-hr/recording for the full
pipeline, or ~150–180 GPU-hr with `pipeline.stop_after=triangulate` (~20 h on
8 GPUs). SAM3 over 500–800k frames x 7 cameras is likely the dominant term and
is entirely unquantified.

**The claim that the rewrite did not deliver speed is therefore untested.** Its
real, structural win is distributed execution: JARVIS could not use multiple
GPUs, and the current stack runs as SLURM arrays across 8+.

## Recommendations

1. **Keep the JAX pipeline.** It is measurably more accurate on the well-tracked
   fly and equivalent in the median on the hard one, and it is the only path that
   parallelises across GPUs — which is the binding constraint on a large backlog.
2. **Fix SAM3's dropout, don't revert to CenterDetect.** Two concrete options,
   cheapest first: (a) fall back to a CenterDetect-style centroid for any
   (fly, camera, frame) where SAM3 has no mask, so the crop still exists; (b)
   recover the missing views geometrically — reproject the 3D from the cameras
   that did see her (this is the Track 5 multi-view/visual-hull idea, and this
   audit is its strongest justification). The min-views gate landed today stops
   the bad frames from being emitted as confident output in the meantime.
3. **Measure before optimising.** Instrument per-stage wall-clock (SAM3, 2D,
   triangulation) on one recording. Speed is currently the only claim in this
   audit with no evidence behind it, and whole-recording processing is the
   largest cost we will pay.
4. **Retrain the detector before any large processing run.** The female's 2D
   confidence is the dominant correlate of bad 3D (r = −0.85). The V4 dataset
   (built 2026-08-08) adds 5,054 previously-excluded labelled images — headless,
   amputated, and female-on-wall — that the old curated set omitted because their
   47/44-length keypoint arrays did not match the 50-node schema. Processing a
   large backlog with the current detector means paying the cost twice.
5. **Sequence the backlog:** detector retrain -> re-run the frozen benchmark to
   confirm the gain -> instrument timings on one recording -> then full-recording
   extraction with `stop_after=triangulate`.

## Caveats

- Six bouts from one recording; the comparison should be repeated across
  recordings before treating the medians as settled.
- `Predictions_3D_34662304` is one of four JARVIS prediction dirs for this
  recording and may not be the best JARVIS run available.
- Within-bone CV measures 3D self-consistency, not correctness against ground
  truth: a pipeline could be self-consistent and systematically wrong. It is the
  best available proxy in the absence of labelled 3D.
