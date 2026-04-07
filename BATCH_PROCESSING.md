# Batch Processing Pipeline for Free Walking Predictions

This guide explains how to process all Predictions_3D folders in the free_walking dataset using the batch processing scripts.

## Quick Start - Full Pipeline

The easiest way to run the complete pipeline is using the master orchestration script:

```bash
# Run the entire pipeline (with confirmation prompt)
python scripts/run_full_pipeline.py --anatomy v1

# Dry run to see what would happen
python scripts/run_full_pipeline.py --anatomy v1 --dry-run

# Force reprocessing everything
python scripts/run_full_pipeline.py --anatomy v1 --force

# Run specific steps only
python scripts/run_full_pipeline.py --anatomy v1 --steps preprocess,stac

# Custom STAC settings
python scripts/run_full_pipeline.py --anatomy v1 \
    --stac-overrides "dataset.stac.n_fit_frames=401"
```

The full pipeline script:
- Runs all steps in sequence with proper error handling
- Confirms before execution (unless `--dry-run`)
- Stops if any step fails
- Reports timing and success/failure for each step
- Creates comprehensive logs

See [Full Pipeline Orchestration](#full-pipeline-orchestration) section below for details.

## Overview

The complete pipeline consists of four steps:
1. **Preprocessing** - Extract and align keypoints from raw data
2. **STAC IK** - Run inverse kinematics solver
3. **Postprocessing** - Compute velocities, egocentric positions, reorganize data
4. **Combine** - Merge all processed folders into a single file

You can run each step individually (detailed below) or use the full pipeline script (recommended).

## Directory Structure

```
/data2/users/eabe/datasets/Johnson_lab/free_walking/
├── Predictions_3D_20260202-171900/
│   ├── data3D.csv                    # Raw 3D keypoint data
│   ├── free_walking_bouts_summary.csv     # Bout info with fly_id ({dataset}_bouts_summary.csv)
│   ├── preprocessed_bout_v1.h5       # Output from step 1
│   ├── Fruitfly_ik_v1_free.h5        # Output from step 2 (STAC)
│   └── ik_output_v1.h5               # Output from step 3
├── Predictions_3D_20260203-103416/
│   └── ...
└── ... (more folders)
```

## Step 1: Batch Preprocessing

Process all folders to extract and align keypoints.

### Basic Usage

```bash
# Process all folders with default settings
python scripts/batch_process_predictions.py

# Process with specific anatomy version
python scripts/batch_process_predictions.py --anatomy v1

# Use different paths config
python scripts/batch_process_predictions.py --paths workstation
```

### Advanced Options

```bash
# Dry run - show what would be processed without actually running
python scripts/batch_process_predictions.py --dry-run

# Force reprocess even if outputs already exist
python scripts/batch_process_predictions.py --force

# Custom base directory
python scripts/batch_process_predictions.py --base-dir /path/to/data

# Custom log file
python scripts/batch_process_predictions.py --log-file my_preprocess.log
```

### What It Does

For each `Predictions_3D_*` folder:
- Checks for required input files (`data3D.csv`, `{dataset}_bouts_summary.csv`)
- Skips if output already exists (unless `--force`)
- Runs `preprocess_keypoints_for_ik.py` with appropriate paths
- Logs all results to timestamped log file

### Output

Each folder will have:
- `preprocessed_bout_{anatomy}.h5` - Aligned keypoints ready for STAC IK
- Contains `info['fly_ids']` list tracking fly identity for each bout

### Requirements

CSV files must have a `fly_id` column (required as of the recent update). If missing, preprocessing will fail with a clear error message.

## Step 2: STAC IK Solver (Batch Processing)

Run inverse kinematics on all preprocessed folders using the STAC solver.

### Prerequisite: Initialize Submodule

The STAC IK solver is available as a git submodule. If you haven't initialized it yet:

```bash
# Initialize submodule (first time only)
git submodule update --init --recursive
```

See [SUBMODULES.md](SUBMODULES.md) for more information about working with git submodules.

### Basic Usage

```bash
# Process all folders with default settings
python scripts/batch_run_stac.py --anatomy v1

# Dry run - show what would be processed
python scripts/batch_run_stac.py --anatomy v1 --dry-run

# Force reprocess even if outputs exist
python scripts/batch_run_stac.py --anatomy v1 --force
```

### Advanced Options

```bash
# Adjust GPU memory allocation (default 0.9)
python scripts/batch_run_stac.py --anatomy v1 --gpu-mem-fraction 0.8

# Pass custom STAC configuration overrides
python scripts/batch_run_stac.py --anatomy v1 \
    --stac-overrides "dataset.stac.n_fit_frames=401"

# Skip fit_offsets stage (only run ik_only)
python scripts/batch_run_stac.py --anatomy v1 \
    --stac-overrides "dataset.stac.skip_fit_offsets=True"

# Custom paths
python scripts/batch_run_stac.py --anatomy v1 \
    --base-dir /path/to/data \
    --stac-dir /path/to/stac-mjx
```

### What It Does

For each `Predictions_3D_*` folder with preprocessed data:
- Checks for `preprocessed_bout_{anatomy}.h5`
- Skips if STAC outputs already exist (unless `--force`)
- Sets up proper environment for headless GPU rendering:
  - `MUJOCO_GL='egl'` - Headless rendering (no display needed)
  - `PYOPENGL_PLATFORM='egl'` - PyOpenGL backend
  - `XLA_PYTHON_CLIENT_MEM_FRACTION` - GPU memory allocation
  - `JAX_COMPILATION_CACHE_DIR` - Speed up repeated runs
  - `XLA_FLAGS` - GPU optimizations
- Runs STAC IK solver using `run_stac_fly_model.py` which:
  - Handles multi-bout format (concatenates all bouts)
  - Automatically pads clips to max_length (if `enable_padding=True`)
  - Applies MOCAP_SCALE_FACTOR to keypoint data
  - Uses Hydra config overrides:
    - `paths=workstation`
    - `dataset=free_walking`
    - `anatomy={anatomy}`
    - `version={folder_name}` - Dynamically set for each folder
- Creates two output files per folder:
  - `Fruitfly_fit_{anatomy}_free.h5` - fit_offsets stage
  - `Fruitfly_ik_{anatomy}_free.h5` - ik_only stage
- Logs all results to timestamped log file

### Environment Setup

The script automatically configures the environment based on `stac-mjx/demos/run_stac_fly_model.py`:
- **Headless Rendering**: Critical for cluster/server execution without displays
- **GPU Memory**: Configurable fraction (default 0.9 = 90% of GPU memory)
- **JAX Caching**: Persistent compilation cache in `/tmp/jax_cache`
- **GPU Optimizations**: XLA flags for Triton GEMM and per-fusion autotune

### Configuration Flexibility

The batch script uses the stac-mjx config system. The key overrides are:
- `version={folder_name}` - Tells Hydra which folder to process
- `paths.data_dir` is automatically set based on `version`
- Additional overrides can be passed via `--stac-overrides`

Example configuration flow:
```yaml
# stac-mjx/configs/config.yaml sets defaults
defaults:
  - anatomy: v1
  - paths: workstation
  - dataset: free_walking

# Batch script overrides version for each folder
version: Predictions_3D_20260202-171900  # Set dynamically

# This makes paths.data_dir resolve to:
# /data2/users/eabe/datasets/Johnson_lab/free_walking/Predictions_3D_20260202-171900

# Which makes dataset.stac.data_path resolve to:
# {data_dir}/preprocessed_bout_v1.h5
```

### Output

Each folder will have:
- `Fruitfly_fit_{anatomy}_free.h5` - Fitted body offsets (fit_offsets stage)
- `Fruitfly_ik_{anatomy}_free.h5` - IK solution with qpos/qvel (ik_only stage)
- Both preserve `info['fly_ids']` from preprocessing

### Manual Processing (Single Folder)

If you need to process individual folders manually:

```bash
cd stac-mjx
python run_stac_fly_model.py paths=workstation dataset=free_walking anatomy=v1 \
    version=Predictions_3D_20260202-171900
```

The `run_stac_fly_model.py` script handles multi-bout concatenation and padding automatically.

## Step 3: Batch Postprocessing

Process STAC IK outputs to compute velocities, egocentric positions, etc.

### Basic Usage

```bash
# Postprocess all folders with STAC outputs
python scripts/batch_postprocess_predictions.py

# Process with specific anatomy version
python scripts/batch_postprocess_predictions.py --anatomy v1
```

### Advanced Options

```bash
# Dry run
python scripts/batch_postprocess_predictions.py --dry-run

# Force reprocess
python scripts/batch_postprocess_predictions.py --force

# Custom paths config
python scripts/batch_postprocess_predictions.py --paths workstation

# Custom log file
python scripts/batch_postprocess_predictions.py --log-file my_postprocess.log
```

### What It Does

For each `Predictions_3D_*` folder:
- Checks for STAC output (`Fruitfly_ik_{anatomy}_free.h5`)
- Skips if already postprocessed (unless `--force`)
- Runs `postprocess_stac_data.py`:
  - Computes velocities via MJX
  - Adjusts floor alignment
  - Computes egocentric site positions
  - Reorganizes data by bouts
- Preserves `fly_ids` through the pipeline
- Logs all results

### Output

Each folder will have:
- `ik_output_{anatomy}.h5` - Complete postprocessed data with velocities, egocentric positions
- Contains `info['fly_ids']` preserved from preprocessing

## Step 4: Combine All Results

Merge all postprocessed outputs into a single file.

```bash
# Combine all ik_output files
python scripts/combine_data.py paths=workstation dataset=free_walking anatomy=v1
```

This will:
- Find all `ik_output_{anatomy}.h5` files
- Concatenate all bouts sequentially
- Preserve `fly_ids` in combined output
- Create two versions:
  - `ik_output_combined_{anatomy}.h5` - Non-interpolated (original frame rates)
  - `ik_output_combined_interp_{anatomy}.h5` - Interpolated to target Hz

### Output

In the base directory (`/data2/users/eabe/datasets/Johnson_lab/free_walking/Data_analysis/Testing/`):
- `ik_output_combined_{anatomy}.h5`
- `ik_output_combined_interp_{anatomy}.h5`

Both files contain:
```python
{
    'info': {
        'fly_ids': [...],          # Preserved from all folders
        'clip_lengths': [...],
        'names_qpos': [...],
        # ... other metadata
    },
    'bout_000': {...},
    'bout_001': {...},
    # ... all bouts from all folders
}
```

## Complete Workflow Example

### Recommended: Full Pipeline Script

The easiest way to run everything:

```bash
# Preview with dry-run
python scripts/run_full_pipeline.py --anatomy v1 --dry-run

# Run complete pipeline (with confirmation)
python scripts/run_full_pipeline.py --anatomy v1

# Or force reprocessing everything
python scripts/run_full_pipeline.py --anatomy v1 --force
```

### Alternative: Individual Scripts

Run each step manually for more control:

```bash
# 1. Preprocess all folders
python scripts/batch_process_predictions.py --anatomy v1 --paths workstation

# 2. Run STAC IK on all folders
python scripts/batch_run_stac.py --anatomy v1

# 3. Postprocess all STAC outputs
python scripts/batch_postprocess_predictions.py --anatomy v1 --paths workstation

# 4. Combine everything
python scripts/combine_data.py paths=workstation dataset=free_walking anatomy=v1
```

## Monitoring Progress

### Check Logs

Batch scripts create timestamped log files:
- `batch_process_YYYYMMDD_HHMMSS.log` - Preprocessing
- `logs/batch_stac_YYYYMMDD_HHMMSS.log` - STAC IK
- `batch_postprocess_YYYYMMDD_HHMMSS.log` - Postprocessing

### Check Individual Folder Status

```bash
# List preprocessing outputs
ls -lh /data2/users/eabe/datasets/Johnson_lab/free_walking/Predictions_*/preprocessed_bout_*.h5

# List STAC outputs
ls -lh /data2/users/eabe/datasets/Johnson_lab/free_walking/Predictions_*/Fruitfly_ik_*.h5

# List postprocessing outputs
ls -lh /data2/users/eabe/datasets/Johnson_lab/free_walking/Predictions_*/ik_output_*.h5
```

## Troubleshooting

### Missing fly_id Column

If you get an error about missing `fly_id` column:
```
ValueError: CSV missing required columns: ['fly_id']
Available columns: ['bout_idx', 'start_frame', 'end_frame', ...]
The 'fly_id' column is required for tracking bout sources.
```

**Solution:** Ensure your `{dataset}_bouts_summary.csv` (e.g. `free_walking_bouts_summary.csv`) has a `fly_id` column. It should look like:
```csv
fly_id,bout_idx,start_frame,end_frame,n_frames,...
Session6/2025_10_12_15_06_46,1,13258,13491,234,...
```

### Timeout Errors

If processing times out (default 10 minutes):
- Large datasets may need longer
- Edit the `timeout=600` parameter in the batch scripts
- Or process individual folders manually

### Force Reprocessing

If you need to reprocess everything (e.g., after fixing a bug):
```bash
python scripts/batch_process_predictions.py --force
python scripts/batch_postprocess_predictions.py --force
```

### Selective Reprocessing

To process only specific folders:
1. Move processed outputs out of the way temporarily
2. Run batch script (it will skip folders with outputs)
3. Move outputs back

Or process individual folders manually:
```bash
python scripts/preprocess_keypoints_for_ik.py \
    paths=workstation dataset=free_walking anatomy=v1 \
    paths.data_dir=/data2/users/eabe/datasets/Johnson_lab/free_walking/Predictions_3D_20260202-171900
```

## Anatomy Versions

Support for multiple anatomy models:
- `v1` - Original fruitfly_v1 model
- `v2_muscles` - Fruitfly v2.1 with muscles

Process different versions separately:
```bash
# Process all with v1
python scripts/batch_process_predictions.py --anatomy v1
python scripts/batch_postprocess_predictions.py --anatomy v1

# Then process with v2_muscles
python scripts/batch_process_predictions.py --anatomy v2_muscles
python scripts/batch_postprocess_predictions.py --anatomy v2_muscles
```

## Data Tracking

The updated pipeline tracks `fly_id` through all stages:
- **Preprocessing:** Extracts from CSV, stores in `info['fly_ids']`
- **STAC:** Preserves through reorganization
- **Postprocessing:** Maintains in processed outputs
- **Combine:** Concatenates fly_ids in correct order

Access fly_id for any bout:
```python
import utils.io_dict_to_hdf5 as ioh5

data = ioh5.load('ik_output_combined_v1.h5')
bout_idx = 42
fly_id = data['info']['fly_ids'][bout_idx]
print(f"Bout {bout_idx} is from fly: {fly_id}")
```

## Full Pipeline Orchestration

The `run_full_pipeline.py` script orchestrates all steps of the pipeline in sequence, providing a single command to run everything from raw data to final combined output.

### Features

- **Sequential Execution**: Runs all steps in order: preprocess → STAC → postprocess → combine
- **Error Handling**: Stops if any step fails, preventing downstream errors
- **Progress Tracking**: Reports timing and status for each step
- **Selective Execution**: Run only specific steps (e.g., just preprocess and STAC)
- **Confirmation Prompt**: Asks for confirmation before running (unless `--dry-run`)
- **Comprehensive Logging**: Shows all output from each batch script
- **Flexible Configuration**: Pass through options to individual steps

### Basic Usage

```bash
# Run complete pipeline (asks for confirmation)
python scripts/run_full_pipeline.py --anatomy v1

# Dry run to preview what will happen
python scripts/run_full_pipeline.py --anatomy v1 --dry-run

# Force reprocessing all steps
python scripts/run_full_pipeline.py --anatomy v1 --force

# Different anatomy version
python scripts/run_full_pipeline.py --anatomy v2_muscles
```

### Selective Step Execution

Run only specific steps by passing a comma-separated list:

```bash
# Only preprocessing and STAC
python scripts/run_full_pipeline.py --anatomy v1 --steps preprocess,stac

# Only postprocessing and combine (after STAC is done)
python scripts/run_full_pipeline.py --anatomy v1 --steps postprocess,combine

# Just combine (if everything else is complete)
python scripts/run_full_pipeline.py --anatomy v1 --steps combine
```

Available steps:
- `preprocess` - Run batch_process_predictions.py
- `stac` - Run batch_run_stac.py
- `postprocess` - Run batch_postprocess_predictions.py
- `combine` - Run combine_data.py

### Advanced Options

```bash
# Custom STAC configuration
python scripts/run_full_pipeline.py --anatomy v1 \
    --stac-overrides "dataset.stac.n_fit_frames=401 dataset.stac.enable_padding=False"

# Adjust GPU memory for STAC
python scripts/run_full_pipeline.py --anatomy v1 --gpu-mem-fraction 0.8

# Different dataset and paths config
python scripts/run_full_pipeline.py --anatomy v1 \
    --dataset free_walking \
    --paths workstation \
    --base-dir /path/to/data

# See all options
python scripts/run_full_pipeline.py --help
```

### Example Output

```bash
$ python scripts/run_full_pipeline.py --anatomy v1 --dry-run

================================================================================
                           3D TRACKING DATA PIPELINE                            
================================================================================

Configuration:
  Anatomy: v1
  Dataset: free_walking
  Base directory: /data2/users/eabe/datasets/Johnson_lab/free_walking
  Paths config: workstation
  Steps to run: preprocess, stac, postprocess, combine
  Force reprocessing: False
  Dry run: True

--------------------------------------------------------------------------------
                            STEP 1/4: PREPROCESSING                             
--------------------------------------------------------------------------------

[DRY RUN] Would execute: python batch_process_predictions.py --anatomy=v1 ...
✅ Preprocessing complete: Dry run

--------------------------------------------------------------------------------
                            STEP 2/4: STAC IK SOLVER                            
--------------------------------------------------------------------------------

[DRY RUN] Would execute: python batch_run_stac.py --anatomy=v1 ...
✅ STAC IK complete: Dry run

--------------------------------------------------------------------------------
                            STEP 3/4: POSTPROCESSING                            
--------------------------------------------------------------------------------

[DRY RUN] Would execute: python batch_postprocess_predictions.py --anatomy=v1 ...
✅ Postprocessing complete: Dry run

--------------------------------------------------------------------------------
                             STEP 4/4: COMBINE DATA                             
--------------------------------------------------------------------------------

[DRY RUN] Would execute: python combine_data.py paths=workstation ...
✅ Combine complete: Dry run

================================================================================
SUMMARY
================================================================================
✅ PREPROCESS (0.0s): Dry run
✅ STAC (0.0s): Dry run
✅ POSTPROCESS (0.0s): Dry run
✅ COMBINE (0.0s): Dry run
================================================================================
```

### Error Handling

If any step fails, the pipeline stops and shows:
```bash
❌ Step 'stac' failed. Stopping pipeline.

================================================================================
SUMMARY
================================================================================
✅ PREPROCESS (213.5s): Success
❌ STAC (145.2s): Failed with return code 1
   (subsequent steps not attempted)
================================================================================
```

You can then:
1. Fix the issue that caused the failure
2. Rerun from the failed step: `--steps stac,postprocess,combine`
3. Or rerun everything with `--force`

### Integration with Individual Scripts

The full pipeline script calls the individual batch scripts with appropriate arguments:
- It passes `--dry-run` and `--force` flags through to each step
- You can still run individual scripts manually for more control
- All scripts use the same configuration system

### Recommended Workflow

For production runs:
```bash
# 1. Dry run to verify configuration
python scripts/run_full_pipeline.py --anatomy v1 --dry-run

# 2. Run the pipeline (with confirmation)
python scripts/run_full_pipeline.py --anatomy v1

# 3. If any step fails, debug and rerun from that step
python scripts/run_full_pipeline.py --anatomy v1 --steps stac,postprocess,combine
```

For development/testing:
```bash
# Test just one step
python scripts/run_full_pipeline.py --anatomy v1 --steps preprocess --dry-run

# Run with force to overwrite test outputs
python scripts/run_full_pipeline.py --anatomy v1 --force
```

