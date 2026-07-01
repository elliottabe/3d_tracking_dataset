# Design: Parallel V3-masked 3D prediction on courtship Session1 (SLURM array)

**Date:** 2026-07-01
**Status:** Design — awaiting review before implementation plan

## Goal

Run full 3D pose predictions on the courtship **Session1** dataset using the
original JARVIS-HybridNet SAM3 shard pipeline, parallelized across recordings
with a **SLURM array job** so the whole session completes quickly and is easy to
manage/requeue on the preemptible `ckpt-g2` partition.

## Background / current state

- **Data:** `/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1`
  contains 14 timestamped recordings, each a 7-camera calibrated set
  (`CamXXXXXXX.mp4` + `calibration/`).
- **Bouts** (courtship interaction frame-ranges) are the sub-unit of prediction.
  Bout definitions live *outside* the video dirs, in
  `/gscratch/portia/eabe/data/Johnson_lab/courtship/Session1_bouts_04172026/<rec>/Predictions_3D_*/courtship_bouts_unified_summary.csv`.
  **Only 10 of the 14 recordings have bout CSVs** — those 10 are in scope.
  (The other 4 — `11_52_43`, `13_03_01`, `15_12_14`, `17_13_56` — have no bout
  definitions and are explicitly out of scope for this run.)
- **Existing infrastructure already does most of this:**
  - `third_party/JARVIS-HybridNet/tools/predict3D_multianimal_shard.py` — the
    proven courtship SAM3 pipeline. It is a *dispatcher*: it spawns
    `num_gpus/2` shard subprocesses (each shard = 2 GPUs, JARVIS + SAM3),
    distributes the recording's bouts across shards, and writes per-fly 3D CSVs.
    Uses SAM3 video propagation to keep the two flies from collapsing onto one.
  - `third_party/JARVIS-HybridNet/submit_courtship_repredict.sh` — submits **one
    independent `sbatch` job per recording** using that shard script.
  - The new work converts this **loop of independent jobs into a true SLURM
    `--array` job**. This does not change raw parallelism materially, but gives a
    single job ID, a concurrency throttle (`%K`), and clean requeue behavior.
- **Model:** `unified_V3_masked` — a 4-channel masked model
  (`KEYPOINTDETECT.INSTANCE_MASK_INPUT: true`, `MODEL_SIZE: large`,
  `NUM_JOINTS: 50`, `NUM_CAMERAS: 7`). This project exists **only in the repo
  copy** (`third_party/JARVIS-HybridNet/projects/unified_V3_masked`), not in the
  Github runtime root, so the array job must run from the repo copy.

## Decisions (locked)

| Decision | Choice |
|---|---|
| Parallelization | True SLURM `--array` job, one task per recording |
| Scope | The **10** Session1 recordings that have bout CSVs |
| Model | `unified_V3_masked` (repo copy) |
| Pipeline script | `tools/predict3D_multianimal_shard.py` (original JARVIS-HybridNet), 2 animals |
| CenterDetect | Point `--center-weights` at existing `red_data_unified/phase4_center_ft` (zero code change) |
| HybridNet weights | `unified_V3_masked/models/HybridNet/Run_20260620-173554/HybridNet-large_final.pth` (newer run) |
| GPUs / concurrency | `--num_gpus 4` (2 shards/recording), `--array=1-10%5` (≤5 recordings, ≤20 GPUs in flight) |
| JARVIS root | Repo copy: `third_party/JARVIS-HybridNet` |
| Conda env | `jarvis` (the proven runtime env) |

### CenterDetect note

The masked worker (`tools/predict3D_multianimal.py`) computes crop centers from
**SAM3 mask centroids triangulated to 3D** (`reconstructPoint → reprojectPoint`),
*not* from CenterDetect — CenterDetect output is unused in this path. However the
worker still *constructs and loads* a CenterDetect model at startup, and
`unified_V3_masked` has no CenterDetect weights. To avoid a missing-file crash
with zero code change, we pass the existing compatible weights (medium/320,
matching the V3 config):

```
/gscratch/portia/eabe/Research/Github/JARVIS-HybridNet/projects/red_data_unified/models/CenterDetect/phase4_center_ft/EfficientTrack-medium_final.pth
```

(Future cleanup, out of scope: patch the worker to skip CenterDetect construction
when SAM3 masks are provided.)

## Architecture

### Component 1 — Manifest (stable index → recording mapping)

A generated `session1_manifest.tsv` with one line per in-scope recording:

```
<recording_dir>\t<bouts_csv>
```

- Built by scanning `Session1_bouts_04172026/*/Predictions_3D_*/courtship_bouts_unified_summary.csv`,
  mapping each CSV's grandparent dir name to the video recording under
  `.../Video_recordings/courtship/Session1/`, and sorting deterministically.
- **Purpose:** `$SLURM_ARRAY_TASK_ID` indexes a fixed line, so the same task
  always maps to the same recording — safe across requeue/preemption.
- 10 lines. All 10 recordings verified to have `calibration/` and 7 `.mp4`s.

Manifest contents (verified 2026-07-01):

| idx | recording | bouts_csv (under Session1_bouts_04172026/) |
|----|-----------|--------------------------------------------|
| 1 | 2026_04_02_12_11_50 | .../Predictions_3D_34662586/courtship_bouts_unified_summary.csv |
| 2 | 2026_04_02_14_54_28 | .../Predictions_3D_34662588/... |
| 3 | 2026_04_02_15_25_51 | .../Predictions_3D_34662589/... |
| 4 | 2026_04_02_15_44_42 | .../Predictions_3D_34662590/... |
| 5 | 2026_04_02_16_03_48 | .../Predictions_3D_34662591/... |
| 6 | 2026_04_02_16_21_32 | .../Predictions_3D_34662592/... |
| 7 | 2026_04_02_16_39_56 | .../Predictions_3D_34662593/... |
| 8 | 2026_04_02_16_56_37 | .../Predictions_3D_34662594/... |
| 9 | 2026_04_02_17_28_34 | .../Predictions_3D_34662595/... |
| 10 | 2026_04_02_17_52_50 | .../Predictions_3D_34662596/... |

### Component 2 — Array sbatch script

`submit_session1_array.sh` (adapted from `submit_courtship_repredict.sh`). Key
elements:

- SBATCH directives: `--array=1-10%5`, `--partition=ckpt-g2`, `--account=portia`,
  `--nodes=1`, `--gpus=4`, `--cpus-per-task=16`, `--mem=128G`, `--requeue`,
  `--open-mode=append`, per-task log `slurm-session1-%A_%a.out`, nodelist matching
  the existing script.
- Body per task:
  1. `module load cuda/12.9.1 gcc/12`; `source ~/.bashrc`; `micromamba activate jarvis`; `unset LD_LIBRARY_PATH`.
  2. Read line `$SLURM_ARRAY_TASK_ID` from the manifest → `rec_dir`, `bouts_csv`.
  3. `cd` to repo `JARVIS_ROOT`; run:
     ```
     python -u tools/predict3D_multianimal_shard.py \
       --project unified_V3_masked \
       --video_folder <rec_dir> \
       --calib_folder <rec_dir>/calibration \
       --num_animals 2 \
       --num_gpus 4 \
       --bouts_csv <bouts_csv> \
       --output_name ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
       --center-weights <red_data_unified phase4 CenterDetect> \
       --kp-weights <V3 KeypointDetect Run_20260619-185121/EfficientTrack-large_final.pth> \
       --hybridnet-weights <V3 HybridNet Run_20260620-173554/HybridNet-large_final.pth> \
       --save-masks --save-clips
     ```
- `DRY_RUN` support (echo the sbatch/array plan without submitting), matching the
  existing script's convention.

### Component 3 — Outputs

Each recording gets a `Predictions_3D_<arrayjob>_<task>/` directory inside its own
video folder, containing per bout: `bout_NNNNN/fly0.csv`, `fly1.csv`,
`sam3_masks.npz`, and `clips/` (per-camera clip videos). This matches the layout
of the one previously-completed run (`Predictions_3D_34662593`, 18 bouts).

## Data flow

```
Session1_bouts_04172026/*/…/courtship_bouts_unified_summary.csv
        │  (manifest generation, one-time)
        ▼
session1_manifest.tsv  (10 lines: rec_dir  bouts_csv)
        │  SLURM --array=1-10%5, task i → line i
        ▼
predict3D_multianimal_shard.py  (per recording task, 4 GPUs)
        │  splits bouts across num_gpus/2 = 2 shards
        ▼
Predictions_3D_<job>_<task>/bout_*/{fly0.csv, fly1.csv, sam3_masks.npz, clips/}
```

## Error handling & idempotence

- **Requeue/preemption:** `--requeue` + stable manifest index; the shard
  script's `--reuse-masks` avoids recomputing SAM3 masks for finished bouts.
- **Fresh output dirs:** `--output_name` uses the array job/task IDs, so runs
  never overwrite prior `Predictions_3D_*` dirs (they accumulate; cleanup of old
  empty dirs is a separate, optional housekeeping step).
- **Per-task isolation:** a failure in one recording's task does not affect the
  others; failed indices can be re-submitted individually
  (`--array=<idx>`).
- **Missing inputs:** manifest generation asserts each recording dir has
  `calibration/` and ≥1 `.mp4` before writing the line (all 10 pass today).

## Testing / validation

1. **Dry run:** `DRY_RUN=1` prints the array plan and the resolved per-task
   command; eyeball the 10 mappings and weight paths.
2. **Single-recording smoke test:** submit `--array=7` (the recording that
   previously completed with 18 bouts) and confirm it produces per-fly CSVs +
   masks without a CenterDetect load error.
3. **Full submit:** `--array=1-10%5`; monitor `squeue -u $USER`; spot-check a
   couple of `fly0.csv`/`fly1.csv` for sane, non-NaN 3D tracks and confirm both
   flies stay separate (identity not collapsed).

## Out of scope

- The 4 recordings without bout CSVs (would need an upstream bout-building pass).
- Patching the worker to drop the unused CenterDetect load.
- Cleaning up prior empty `Predictions_3D_*` dirs.
- Any downstream analysis/staging of the produced CSVs.
