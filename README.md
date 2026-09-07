# 3D Tracking Dataset

A comprehensive pipeline for processing 3D keypoint tracking data of fruit fly (Drosophila) behavior, with Procrustes alignment and inverse kinematics.

## Quick Start

```bash
# Clone with submodules
git clone --recurse-submodules <repository-url>
cd 3d_tracking_dataset

# Or if already cloned, initialize submodules
git submodule update --init --recursive

# Setup environment
conda env create -f environment.yml
conda activate 3d_tracking
uv pip install -r requirements.txt
uv pip install -e third_party/jarvis_jax
```

This is the minimal env setup for the batch (`v1`/`v2_3`) pipeline below. The
full SAM3 → ViTPose → triangulation → STAC/IK courtship/free-running
pipeline and mvq training need a sibling body-model clone, SAM3 weights, and
cluster-specific paths/Slurm config — see **"Install and run on a new
cluster"** in [docs/running_the_pipeline.md](docs/running_the_pipeline.md)
for the full sequence and exact commands, and
[docs/portability-checklist.md](docs/portability-checklist.md) for what is
Hyak-specific if you're moving to a different cluster.

## Repository Structure

- **`scripts/`** - Processing pipeline scripts
  - `preprocess_keypoints_for_ik.py` - Extract and align keypoints
  - `postprocess_stac_data.py` - Compute velocities and egocentric positions
  - `combine_data.py` - Merge processed data
  - `batch_process_predictions.py` - Batch preprocessing
  - `batch_postprocess_predictions.py` - Batch postprocessing
- **`utils/`** - Utility functions and data processing tools
- **`configs/`** - Hydra configuration files
- **`stac-mjx/`** - STAC inverse kinematics solver (git submodule)

## Documentation

- **[BATCH_PROCESSING.md](BATCH_PROCESSING.md)** - Complete guide for batch processing all prediction folders
- **[SUBMODULES.md](SUBMODULES.md)** - Working with git submodules
- **[PREPROCESSING_README.md](PREPROCESSING_README.md)** - Detailed preprocessing documentation
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - Project architecture and conventions

## Batch Processing Pipeline

Process all `Predictions_3D_*` folders:

```bash
# 1. Preprocess all folders
python scripts/batch_process_predictions.py --anatomy v1

# 2. Run STAC IK (see BATCH_PROCESSING.md)
cd stac-mjx
python run_stac.py ...

# 3. Postprocess all outputs
python scripts/batch_postprocess_predictions.py --anatomy v1

# 4. Combine into single file
python scripts/combine_data.py paths=workstation dataset= anatomy=v1
```

See [BATCH_PROCESSING.md](BATCH_PROCESSING.md) for complete documentation.

## Key Features

- **Procrustes Alignment** - Align keypoints to reference pose with optional scaling
- **Ground Contact Alignment** - Rotate and translate to align ground plane
- **Keypoint Exclusion** - Exclude noisy keypoints (antenna, wings) from alignment computation
- **Fly ID Tracking** - Track fly identity (fly_id) through entire pipeline
- **JAX Acceleration** - Hardware-accelerated processing with JIT compilation
- **Batch Processing** - Efficient processing of multiple prediction folders
- **MuJoCo Integration** - Forward kinematics and physics simulation

## Dependencies

- JAX (with GPU support recommended)
- MuJoCo and mujoco-mjx
- Hydra for configuration management
- HDF5 for data storage

See `environment.yml` for complete dependency list.

## Git Submodules

This repository uses git submodules for dependencies:
- **stac-mjx** - STAC inverse kinematics solver
- **third_party/JARVIS-HybridNet** - calibration/project-config plumbing for the SAM3/pose pipeline

`third_party/jarvis_jax` (the SAM3/ViTPose/mvq courtship pipeline) is vendored
in-tree, not a submodule — it travels with a normal clone but needs its own
`pip install -e third_party/jarvis_jax` (see Quick Start above).

See [SUBMODULES.md](SUBMODULES.md) for working with submodules.
