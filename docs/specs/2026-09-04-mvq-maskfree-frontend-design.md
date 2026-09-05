# mvq mask-free front end: raw video to per-bout 3D keypoints with identity

Successor to `docs/specs/2026-09-03-mvq-dinov3-query-decoder-design.md` §8 (pipeline
integration) and `docs/specs/2026-09-04-mvq-p3a-identity-existence-design.md`. Approved in
outline by the user 2026-09-04 (approach A, sections 1-7 plus the learned bout detector).
Where this document and the earlier ones disagree, this one wins.

## 1. Why, and what "done" means

The P3a lifter (`mvq_t1_b16_p3a_20260904`, spec §8 acceptance passed) identifies female and
male by typed slots without a mask prompt, at 0.092 mm oracle / 0.094 mm policy error on
val versus 0.107 mm for ViTPose+DLT, and on bout 28 it held identity through the whole bout
while the female's IK fit came out seven times smoother than the ViTPose+DLT fit. The
pipeline's dominant cost is no longer the lifter but the SAM3 stages that feed it: a
recording is ~498,000 frames at 800 fps, 1936x448, seven cameras (~10 min of video), and
today's coarse pass is SAM3 at stride 16 over all of it, followed by per-bout SAM3 masks,
ViTPose and DLT -- of the order of hundreds of GPU-hours per recording, never precisely
measured (`docs/audits/2026-08-08-pipeline-audit.md` finding 4).

**Goal.** Raw synced video in; per-bout `kp2d.npz`/`kp3d.npz` with fly identity out, in the
pipeline's own format, so filter, STAC, bridge, outputs and QC run unchanged; no SAM3
anywhere on the new-recording path. Unit of work: a whole recording; bouts derived from
the tracks. Assays: courtship pairs (one female, one male) and single-fly recordings.

**Acceptance.**
- Session0 `2025_10_20_13_20_04` (30 human-reviewed bouts): bout recall and boundary
  offsets at least equal to the SAM3 coarse pass's calibration on the same recording
  (`docs/specs/2026-08-31-pipeline-schematic-and-inventory-design.md`), measured with
  that design's recall / precision / boundary-offset protocol.
- 13-bout frozen benchmark (`scripts/benchmark/`): IK reprojection and fitted-keypoint
  jitter no worse than the current pipeline on any cohort, zero identity swaps.
- Wall clock under 3 GPU-hours per recording on one L40S, measured.
- Every figure gate in §8 read and recorded.

**Decisions (user, 2026-09-04).** Mask-free end to end, not "lifter behind SAM first".
Whole recordings, bouts derived afterwards (bout mode kept for re-running existing sessions).
Courtship pairs and single-fly recordings only; same-sex pairs deferred. Approach A:
CenterDetect coarse pass, mvq tracker inside bouts, learned bout detector on top of the
coarse tracks with the existing hand gates as baseline and fallback.

## 2. Facts that shape the design

- **CenterDetect exists**: a JAX/NNX port of JARVIS CenterDetect (`jarvis_jax/scripts/
  train_centerdetect.py`, EfficientTrack-medium, RGB only, 320 px input) with checkpoints
  under `/gscratch/portia/eabe/data/Johnson_lab/jax_centerdetect_runs/*`, top-2 peak decode
  in `jarvis_jax/eval/centerdetect_decode.py`. Known limit: on real courtship frames it
  collapses to ONE peak when the flies touch (~20 % of frames after fine-tuning). For this
  design that is harmless: one window containing both flies is exactly what the typed
  slots handle. What the coarse pass needs from CenterDetect is "a fly is near here", to
  within the 5.6 mm window.
- **mvq cost** (bout-28 script, unbatched pairs, two passes): 0.18 s per window. Batched
  32-wide, one pass, bf16: budget 40 ms per window (to be measured in P4a).
- **Motion at 800 fps**: a walking fly moves ~0.02 mm per frame, ~0.6 mm between two coarse
  frames 16 apart; a jump can exceed the window.
- **Training centre jitter was 3 units (0.3 mm)**; robustness to off-centre windows is
  unknown -- §6 measures it first.
- **Masks downstream are optional already** except for Stage A's crop and gray-fill (which
  mvq does not use): bridge `bridge_mode: keypoint` (default) uses no masks; coverage's
  `min_views` gate is config-gated; the wing-mask fit (D2) is opt-in; sexing operates on 3D
  keypoints; the side-by-side render uses masks only as an overlay.
- **Bout tables** for existing sessions come from the legacy JARVIS tracking pass
  (`Predictions_3D_*`), normalised by `jarvis_jax/predict/bouts_resolve.py`; the SAM3
  coarse pass (`scripts/coarse_pass.py`, gates in `scripts/coarse_pass_gates.py`) is
  designed and implemented but has no recorded end-to-end validation.
- **Keypoint order**: mvq emits the detector order (`configs/detector/vitpose_v3.yaml
  kp_names`, identical to v12 `keypoint_names`); the pipeline's files are in model order
  (`configs/anatomy/v1.yaml model.KP_NAMES`). Permute BY NAME (`detector_to_model_perm`);
  never by position.
- **Confidence scale**: mvq's `conf3d` head is the D4RT confidence (~0.02-0.2), not a
  probability; the pipeline thresholds conf at 0.3-0.5. The pipeline-facing confidences are
  the view-visibility sigmoid (2D) and its mean over cameras (3D); the raw head value is
  stored beside them as `conf3d_mvq_raw`.
- **Synced reader** (`jarvis_jax/predict/synced_reader.py::read_window`) streams any frame
  range sequentially, including a whole recording; Session0 has no sync plan (identity).

## 3. Architecture

```
recording (7 mp4 + calibration)
  |  stride 16
  v
[A] CenterDetect per camera -> top-2 peaks -> DLT -> candidate 3D centres -> windows
  v
[B] mvq coarse pass at those windows -> 50 fps tracks: kp3d, sex, existence per fly
  v
[C] bout segmentation: hand gates (baseline) | learned detector (§5) -> bout table
  v  per bout
[D] mvq fine pass: windows at the coarse track interpolated to 800 fps, batched;
    re-seed pass for low-existence frames -> per-fly kp2d/kp3d (model order) + sex.json
  v
[E] existing stages unchanged: filter -> STAC -> bridge -> outputs/QC -> renders
```

Every stage writes to the recording's processed tree; stage boundaries are files, so a
stage re-runs only when its output is missing (the pipeline's existing convention).

## 4. Components

### 4.1 Coarse localiser (`jarvis_jax/tracking/coarse_centres.py`)

Input: the recording's cameras, a stride (default 16), a CenterDetect checkpoint.
Per sampled frame and camera: 320-px inference, top-2 peaks with scores above
`centerdetect.min_score`. Lift to 3D: for every combination of one peak per camera that
is geometrically consistent (DLT residual below `centres.max_resid_px`, at least
`centres.min_views` cameras), keep the centre; cluster centres closer than 1.5 mm; keep at
most `recording.num_animals` clusters by summed peak score. Output `coarse_centres.npz`:
`frame (N,)`, `centres (N, A, 3)` NaN-padded, `n_views (N, A)`, `score (N, A)`. Window
placement for [B]: one window per cluster, merged into one at the midpoint when two
clusters are within `mvq.merge_dist_units` (30 units = 3 mm).

### 4.2 mvq runner (`jarvis_jax/tracking/lift_mvq.py`)

The one place that builds mvq inputs from frames and reads its outputs, shared by the
coarse and fine passes and by `scripts/viz/mvq_bout_video.py` (which is refactored to call
it). `MVQRunner(checkpoint, cameras, calib_dir, batch=32, exist_thresh=0.5)` with:

- `windows(frames (C,H,W,3), centres (B,3)) -> crops, cam_valid, M, t_local, origin` --
  the exact `V12WindowDataset._build` geometry (`crop_origin`, `t_local = M c + t - o`).
- `infer(batch) -> kp3d_world (B, I, K, 3), kp2d_full (B, I, C, K, 2), vis (B, I, C, K),
  exist (B, I), sex_prob (B, I)` -- unprompted only; `assemble()` under the hood.
- `read_typed(out, want_sex) -> per-fly result or None` -- slot 1 for female, slot 2 for
  male; None when that slot's existence is below threshold. Single-fly recordings take
  whichever typed slot exists (the sex is reported, not assumed).
- `to_pipeline(kp3d, kp2d, vis, kp_names_mvq, model_names)` -- permutation by name,
  visibility confidences, `conf3d_mvq_raw`, and the Stage-B `gates` signature with
  `lifter: mvq`, checkpoint path and hash, so `run_bout.py`'s stale-artifact check works
  without `allow_stale_kp3d`.

### 4.3 Coarse pass (`jarvis_jax/tracking/coarse_track.py`)

For each sampled frame: windows from §4.1, `infer`, `read_typed` for female and male.
Output `coarse_tracks.npz`: `frame (N,)`, per fly `kp3d (N, K, 3)`, `centroid (N, 3)`,
`exist (N,)`, `sex_prob (N,)`, `slot (N,)`; plus derived per-frame features for §5:
inter-fly distance, male-to-female heading angle, per-fly speed, wing angles (from the
wing keypoints), height above the floor plane (fit once per recording from the tracks),
trackable flags. Schema documented in the module docstring; the SAM3 coarse pass's
`coarse_tracks.npz` fields (`centroid`, `in_frame`, `X3d`) are emitted under the same names
so `scripts/coarse_pass_gates.py` reads either.

### 4.4 Fine pass (`jarvis_jax/tracking/fine_track.py`)

Per bout: interpolate each fly's coarse centroid to every frame (linear; the coarse track
is at 50 fps). Windows at the interpolated centres, merged when within
`mvq.merge_dist_units`; batched `infer`; `read_typed`. Frames whose typed slot is missing
or below threshold are collected and re-run once with windows centred on the nearest
accepted frames' mvq centroids (before and after, averaged). Frames still missing after the
re-seed pass are NaN, and a run of misses longer than `mvq.max_gap_frames` (default 40 =
50 ms) is left NaN rather than interpolated. Output per fly: `kp2d.npz`, `kp3d.npz`
(pipeline format via `to_pipeline`), `mvq_meta.json` (per-frame slot, existence, sex
probabilities, re-seeded flags, checkpoint), and `sex.json` at the bout level
(`male_fly: 1`, `method: mvq_typed_slots`, per-fly mean sex probability).

### 4.5 Pipeline integration (`scripts/run_bout.py`, `configs/pipeline.yaml`)

Two switches, both default to today's behaviour:
- `pipeline.front_end: sam3 | mvq` -- where the bout table and the per-frame placement
  come from. `mvq` reads `coarse_tracks.npz` + the derived bout table; `sam3` reads
  `recording.predictions_dir` masks as now.
- `pipeline.lifter: dlt | mvq` -- `mvq` replaces Stages A and B with §4.4 (allowed with
  `front_end: sam3` too, where windows are centred on the SAM centroid DLT: that is the
  bout-mode path for re-running existing sessions and the benchmark).
With `lifter: mvq`, mask-dependent stages degrade explicitly: coverage.json is computed
from mvq existence (`source: mvq_exist`), the wing-mask fit is skipped with a logged
reason unless masks exist, the side-by-side render draws no mask layer. `bout_start_frame`
reads the derived bout table when `front_end: mvq`. Config block `mvq:` carries
`checkpoint`, `coarse_stride`, `merge_dist_units`, `exist_thresh`, `batch`,
`max_gap_frames`; `centerdetect:` carries `checkpoint`, `min_score`; `centres:` carries
`max_resid_px`, `min_views`.

### 4.6 Whole-recording driver (`scripts/run_recording_maskfree.py`)

One Hydra entry point: `[A] -> [B] -> [C] -> per bout [D]`, then hands the bout table and
fly directories to `run_bout.py`'s downstream stages (invoked per bout as today, or via
`scripts/slurm_bout_array.py`). Resumable at each file boundary. Emits `timing.json` with
per-stage wall clock for the 3 GPU-hour acceptance.

## 5. Bout detection: hand gates, then a learned detector

**5.1 Baseline.** `scripts/coarse_pass_gates.py` fed the mvq coarse tracks through the
shared schema (§4.3) -- but its area-ratio trackability gate can never pass on an mvq
file (`area`/`area_med` are all-NaN: there are no masks), which the original plan here
did not anticipate. Fixed in P4b task 5: the gate logic moved to
`jarvis_jax/tracking/bout_gates.py` (the script is now a thin re-exporting CLI wrapper)
and dispatches on the file's own schema (`is_mvq_schema`: meta `source == "mvq"` or an
`exist` array). On an mvq file, `trackability_ok` drops the area/border gate for
`per_fly_trackable = (exist >= 0.5) & (n_valid_cams >= MIN_CAMS)`, and `behaviour_ok`
becomes `(wing_angle_deg[male] >= --wing-angle-min, default 30 deg) | (sep3d <=
--proximity-max-units, default 30 world units)` in place of the SAM3 area-ratio /
sep2d_med gate -- the mvq coarse pass has no masks but does have the model's own wing
angle and a true 3D (not reprojected-2D) separation. `separable` and `behaviour_ok` both
fall back gracefully (bypassed to "always true") for single-fly (F=1) recordings, which
have no second fly to gate proximity or wing angle against. The SAM3 path (no `exist`
array) is unchanged and regression-tested byte-identical to the pre-task-5 script. Its
thresholds were tuned against 20_04's reviewed bouts; they are the first thing validated
in §8.

**5.2 Learned detector (`jarvis_jax/tracking/bout_detector.py`).** A per-coarse-frame
classifier over a temporal window (default 32 coarse frames = 0.64 s) of the §4.3
features, trained on recordings with reviewed bout tables: positives = coarse frames inside
a reviewed bout, negatives = the rest, with a margin of one coarse frame around each
boundary excluded from the loss. Model: gradient-boosted trees on windowed feature
statistics (mean, min, max, slope per feature) as the first candidate, a one-layer
temporal convolution as the second; both are a few hundred parameters. Segments from
per-frame probabilities by hysteresis (`on` 0.7 / `off` 0.3) and minimum duration
(`min_frames` = 25 coarse frames = 0.5 s). Config switch `bouts.detector: gates | learned`.

**Data and protocol.** Training recordings: every Session0/Session1 recording with a bout
table whose review status is recorded; evaluation is leave-one-recording-out, reporting
per-bout recall, precision and boundary offset exactly as the SAM3 coarse-pass design
defines them, against the gates on the same folds. Adopted only if it beats the gates on
recall at equal or better precision AND its boundary offsets are within the reviewers'
tolerance (measured in P4d on a handful of bouts by reading the coarse features at the
reviewed boundaries: is the label "courtship behaviour" or "trackable stretch with the
flies close"? the answer is recorded before training). Label provenance caveat: the tables
partly encode the legacy tracker's behaviour; the held-out-recording protocol is what
keeps that honest.

## 6. Prerequisite: window-centre robustness (P4a spike, conditional retrain)

On the P3a final checkpoint, val split, unprompted policy: shift every window centre by
0, 0.5, 1, 2 and 3 mm in a random floor-plane direction and record MPJPE and miss fraction
per shift (`scripts/benchmark/mvq_centre_shift.py`, figure + JSON). Decision rule: if the
policy error at 1 mm is within 10 % of the unshifted value and misses stay under 2 %, the
interpolated coarse centres (worst case ~0.6 mm between coarse frames, plus re-seed) are
safe and no retrain happens. Otherwise this phase includes ONE 10k-step fine-tune from the
P3a final with `jitter_units 10` (1 mm) and the prompt disabled (`prompt_p_start =
prompt_p_end = 0`; slot 0 then never exists and the mask-free route needs no prompt),
same evaluation as P3a §8 plus the shift curve; that checkpoint becomes the shipped one.

## 7. Throughput budget (measured in P4a, recorded in the notes)

| stage | per recording | basis |
|---|---|---|
| CenterDetect, 31k frames x 7 cams | ~10 min | ms-scale 320-px inference |
| coarse mvq, 31k frames x 1-2 windows | 20-40 min | 40 ms/window batched |
| fine mvq, ~30 bouts x 2000 frames x 1-2 windows | ~60-90 min | 40 ms/window batched |
| bout detection, filter, STAC, bridge, outputs | as today (~12 min STAC per bout-fly) | unchanged |

If the measured mvq window cost is above 60 ms the fine pass runs at stride 2 with linear
interpolation of the 3D output (0.04 mm of motion between frames at 800 fps), and this is
recorded as a deviation.

## 8. Figure gates (expectation in each script's docstring; PNG read before any claim)

1. **Coarse centres** (`scripts/viz/coarse_centres_check.py`): 12 sampled frames across
   the recording, two cameras, CenterDetect peaks and the DLT centres reprojected.
   Expectation: every visible fly has a centre within its body in both cameras; when the
   flies touch, one centre between them is acceptable; no centre on the wall or reflection.
2. **Coarse tracks** (`scripts/viz/coarse_tracks_check.py`): inter-fly distance, wing
   angles and speeds over the whole recording with the reviewed bouts shaded. Expectation:
   reviewed bouts coincide with close-distance, wing-extension episodes; a bout that does
   not is recorded as such (it decides what §5 learns).
3. **Bout table** vs reviewed: recall / precision / boundary-offset table plus the
   per-bout timeline figure, gates and learned detector side by side.
4. **Per-bout composite video** (`scripts/viz/mvq_bout_video.py` via `MVQRunner`) for
   bout 28 and two others of different kinds (a wall bout, a mounting bout). Expectation
   unchanged from P3a: female slot on the female, male slot on the male, no swaps.
5. **IK comparison** (`scripts/viz/compare_stac_fits.py`) mask-free mvq vs current
   pipeline on the same bouts, plus the frozen-benchmark scorecard delta.
6. **Centre-shift curve** (§6).

## 9. Testing (TDD, CPU)

- `test_coarse_centres.py`: peaks-to-3D on synthetic affine cameras (known centres
  recovered; inconsistent peaks rejected; two flies within 1.5 mm cluster to one; merge rule).
- `test_lift_mvq.py`: `windows()` reproduces `V12WindowDataset._build` geometry on the
  fixture (same `origin`, `t_local`); `to_pipeline()` permutes by name (eye-spacing
  invariant unchanged; names equal `model.KP_NAMES`), confidence fields and the `gates`
  signature present and accepted by `run_bout.py`'s check.
- `test_fine_track.py`: interpolation to 800 fps, window merge, re-seed selection,
  `max_gap_frames` NaN rule, `sex.json` contents; end-to-end on the synthetic fixture with
  the tiny model through a frame-provider interface (frames from the fixture JPEGs).
- `test_bout_detector.py`: segments from a synthetic probability trace (hysteresis, min
  duration), feature windowing shapes, leave-one-recording-out split logic.
- `test_run_bout_mvq.py`: `pipeline.lifter: mvq` skips Stages A/B, produces the fly
  directories, and the downstream filter stage consumes them; masks absent -> wing-fit
  skipped with the logged reason, coverage from `mvq_exist`.

## 10. Code placement

`jarvis_jax/tracking/{coarse_centres,coarse_track,fine_track,lift_mvq,bout_detector}.py`;
`scripts/run_recording_maskfree.py`; `scripts/run_bout.py` (two switches, stage bypass);
`configs/pipeline.yaml` (`pipeline.front_end`, `pipeline.lifter`, `mvq:`, `centerdetect:`,
`centres:`, `bouts.detector`); `scripts/viz/{coarse_centres_check,coarse_tracks_check}.py`;
`scripts/benchmark/mvq_centre_shift.py`; `scripts/viz/mvq_bout_video.py` refactored onto
`MVQRunner`; notes in `docs/benchmark/2026-09-mvq/p4-maskfree-notes.md`.

## 11. Risks

- **Off-centre windows.** Unmeasured; §6 measures it first and the retrain is budgeted.
- **CenterDetect misses both flies** on dark side views or at the arena edge. Mitigation:
  DLT needs only `centres.min_views` cameras (default 3 of 7); frames with no centre fall
  back to the previous coarse frame's centres; a run of coarse misses longer than 1 s is
  reported per recording. The re-seed pass in §4.4 covers the fine level.
- **Female at the arena edge** (bout 28, frames 447954-448078): existence drops when she is
  clipped in most cameras. The re-seed pass helps only if some frames around it succeed;
  otherwise NaN, which is the correct output (the current pipeline emits junk there).
- **Sex convention on single-fly recordings**: the model reports the sex; the pipeline's
  `male_fly` field is derived from it, not assumed.
- **Bout label provenance** (§5): mitigated by the held-out-recording protocol and the
  pre-training boundary read.
- **Throughput**: the 3 GPU-hour target rests on the 40 ms/window budget; §7 has the
  stride-2 fallback.

## 12. Phases (for the implementation plan)

- **P4a** centre-shift spike (+ conditional retrain); `MVQRunner` with the bout-video
  script refactored onto it; measured window cost.
- **P4b** coarse localiser + coarse pass + gate adapter; figure gates 1-2 on 20_04.
- **P4c** fine pass + `run_bout.py` switches + whole-recording driver; bout 28 through IK;
  benchmark scorecard.
- **P4d** learned bout detector; gate 3; adopt-or-not decision.
- **P4e** whole-recording run on 20_04 and one Session1 recording; timing; acceptance table.

Out of scope: same-sex pairs, T > 1, per-camera copy-paste layering, the 20_04 export
label fix, the prompted path (retired for the mask-free route if the P4a retrain happens).
