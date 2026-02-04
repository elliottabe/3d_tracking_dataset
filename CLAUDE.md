# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D keypoint preprocessing pipeline for fruit fly motion capture data. Converts raw 3D keypoint tracking data (CSV) into the format required by the STAC (Scalable and Trackable Articulated Motion) IK solver for biomechanical analysis.

## Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml -n 3d_tracking
conda activate 3d_tracking

# Mac users
conda env create -f environment_MAC.yml -n 3d_tracking

# If LD_LIBRARY_PATH issues occur
bash setup_conda_unset_ld_library_path.sh
```

Key dependencies: Python 3.12, JAX (GPU-accelerated alignment), MuJoCo/MuJoCo-MJX (physics models), Hydra (configuration), H5PY (data I/O), RAPIDS (CUDA acceleration).

## Running the Preprocessing Pipeline

```bash
# Basic preprocessing
python preprocess_keypoints_for_ik.py \
    --csv_path /path/to/data3D.csv \
    --skeleton_path data/fly50.json \
    --xml_path assets/fruitfly_v1/fruitfly_v1_free.xml \
    --output_dir output/ \
    --bout_name example_bout

# With Procrustes alignment
python preprocess_keypoints_for_ik.py \
    --csv_path /path/to/data3D.csv \
    --skeleton_path data/fly50.json \
    --xml_path assets/fruitfly_v1/fruitfly_v1_free.xml \
    --output_dir output/ \
    --bout_name example_bout \
    --apply_alignment --apply_scaling \
    --exclude_antenna --exclude_wings \
    --frame_start 1000 --frame_end 2000
```

## Testing

```bash
python test_configs.py  # Test Hydra configuration system
```

## Architecture

### Core Pipeline (`preprocess_keypoints_for_ik.py`)
1. Load CSV with multi-level headers (node_name, coordinate)
2. Fuzzy match CSV keypoint names to skeleton nodes
3. Filter skeleton to matched nodes only
4. **Critical**: Reorder keypoints to match MuJoCo XML site order (STAC requirement)
5. Optional Procrustes alignment (JAX-accelerated)
6. Export to HDF5

### Key Modules (`utils/`)
- `io.py` - CSV/skeleton loading, keypoint matching, STAC config parsing
- `io_dict_to_hdf5.py` - HDF5 serialization with nested dict support
- `optimized_floor_alignment.py` - JAX/JIT Procrustes alignment with exclusion support
- `kp_viz.py` - Visualization with ground plane and skeleton edges
- `add_aligned_keypoint_sites.py` - Add visualization sites to MuJoCo XML
- `path_utils.py` - Hydra resolver configuration, cross-environment paths

### Configuration (`configs/`)
- Hydra-based system with environment-specific path configs
- `paths/`: workstation.yaml, mbook.yaml, hyak.yaml, desktop.yaml
- `dataset/`: courtship.yaml (default dataset)

### Data Files
- `data/fly50.json` - 50-node skeleton definition
- `assets/fruitfly_v1/` - MuJoCo fruit fly physics models

## Output Format (HDF5)

```python
{
    'keypoints': (T, N, 3),      # Preprocessed keypoints in XML site order
    'orig_keypoints': (T, N, 3), # Before alignment
    'kp_names': list,            # Node names matching STAC KP_NAMES
    'skeleton_edges': (E, 2),    # Skeleton connectivity
    'alignment_info': dict       # Optional: rotation, translation, scale
}
```

Load with: `utils.io_dict_to_hdf5.load('file.h5', enable_jax=True)`

## Critical Implementation Details

- **XML Site Order**: Keypoints MUST be reordered to match MuJoCo XML tracking site order for STAC compatibility
- **Fuzzy Matching**: CSV column names matched to skeleton using normalized string distance
- **Exclusion Indices**: Antenna/wings can be excluded from alignment computation while still being transformed
- **Memory**: Use `--frame_start`/`--frame_end` for large datasets
