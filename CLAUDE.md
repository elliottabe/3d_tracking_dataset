# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D keypoint preprocessing pipeline for fruit fly motion capture data. Converts raw 3D keypoint tracking data (CSV) into the format required by the STAC IK solver, then postprocesses STAC output into egocentric reference-clip data for downstream RL/simulation.

## Environment Setup

```bash
conda env create -f environment.yml -n 3d_tracking
conda activate 3d_tracking

# If LD_LIBRARY_PATH issues occur
bash setup_conda_unset_ld_library_path.sh
```

Key dependencies: Python 3.12, JAX (GPU-accelerated alignment), MuJoCo/MuJoCo-MJX, Hydra, H5PY.

## Commands

```bash
# Preprocessing: CSV keypoints → HDF5 for STAC
python scripts/preprocess_keypoints_for_ik.py paths=workstation dataset=free_walking
python scripts/preprocess_keypoints_for_ik.py paths=hyak dataset=courtship preprocessing.apply_alignment=false
python scripts/preprocess_keypoints_for_ik.py paths=workstation preprocessing.frame_start=100 preprocessing.frame_end=300

# Postprocessing: STAC IK output → egocentric clips
python scripts/postprocess_stac_data.py paths=workstation dataset=free_walking
python scripts/postprocess_stac_data.py paths=hyak dataset=courtship

# Combine multiple processed bouts
python scripts/combine_data.py paths=workstation dataset=free_walking

# Test Hydra config system
python test_configs.py
```

## Full Pipeline

```
data3D.csv
  → scripts/preprocess_keypoints_for_ik.py  →  preprocessed_bout_<anatomy>.h5
  → stac-mjx (external)                     →  Fruitfly_ik_<anatomy>_free.h5
  → scripts/postprocess_stac_data.py        →  ik_output_<anatomy>.h5
  → scripts/combine_data.py                 →  ik_output_combined.h5
```

## Architecture

### Configuration (`configs/`)

Hydra-based with three config groups composed together:
- `paths/`: environment-specific directories (`workstation`, `mbook`, `hyak`, `desktop`, `hyak_scrubbed`). Each defines `user`, `base_dir`, `data_dir`, `body_model_dir`, `project_dir`.
- `dataset/`: dataset params + includes `preprocessing.yaml` and `postprocessing.yaml` sub-configs (`free_walking`, `courtship`)
- `anatomy/`: MuJoCo model variant (`v1`, `v2`, `v2_muscles`) — controls `mjcf_path`, joint/body/end-effector name lists

Default: `paths=workstation`, `dataset=free_walking`, `anatomy=v1`. Override on the command line.

### Key Utilities (`utils/`)

- `io.py` — CSV loading, fuzzy keypoint name matching (`match_csv_to_skeleton`), skeleton reordering (`reorder_keypoints_array`), STAC output loading; also defines `ModelConfig`, `StacConfig`, `StacData` dataclasses
- `io_dict_to_hdf5.py` — nested dict ↔ HDF5 serialization; `load(path, enable_jax=True)`
- `optimized_floor_alignment.py` — JAX/JIT Procrustes alignment with per-keypoint exclusion (`jit_vectorized_procrustes_with_scaling`)
- `stac_data_utils.py` — reorganize concatenated/padded STAC output into per-bout dicts (`reorganize_stac_by_bouts`); also `interpolate_trajectory`, `adjust_root_z_for_floor`
- `mjx_preprocess.py` — `process_clip()` / `ReferenceClip` dataclass: converts qpos → positions, velocities, quaternions for MJX
- `geometric_angles.py` — anipose-style flex/dihedral angles from raw keypoints (model-free); used to validate ROM limits and tune XML joint ranges via `suggest_xml_ranges` / `update_xml_ranges`
- `walking_cycle_filter.py` — swing-phase detection with body-displacement validation (`detect_swing_phases_with_displacement`); used to generate `walking_bouts_summary.csv` for multi-bout preprocessing
- `transformations.py` — 3D rotation/quaternion math helpers
- `plot_utils.py` — MuJoCo renderer video generation (uses Ray for parallelism)
- `kp_viz.py` — matplotlib visualization of Procrustes alignment results (`visualize_alignment`)
- `add_wing_tracking_sites.py` — one-off script to programmatically add wing tracking sites to MuJoCo XML via spec API
- `add_aligned_keypoint_sites.py` — adds worldbody sites to MuJoCo XML for overlaying aligned keypoint trajectories during rendering

### Postprocessing Pipeline Stages

`postprocess_stac_data.py` runs the following stages in sequence, each configurable under `postprocessing.*`:
1. **Interpolation** (`postprocessing.interpolation`) — resample from capture Hz (e.g. 800) to target Hz (e.g. 1000) using scipy
2. **Floor alignment** (`postprocessing.floor_alignment`) — shift root Z so claw end-effectors touch `target_z` at the 5th percentile
3. **MJX processing** (`postprocessing.mjx_processing`) — run `process_clip()` to compute velocities and `ReferenceClip` fields
4. **Egocentric sites** (`postprocessing.egocentric_sites`) — compute marker site positions relative to the thorax body frame

### Critical Implementation Details

- **XML Site Order**: Keypoints MUST be reordered to match MuJoCo XML `tracking[NodeName]` site order before passing to STAC. `utils/io.py::reorder_keypoints_array` handles this.
- **Fuzzy Matching**: CSV column names are fuzzy-matched to skeleton node names — always check console output for low-confidence matches.
- **STAC Output Formats**: STAC can emit either concatenated (total frames = sum of clip lengths) or padded (total frames = max_clip × n_bouts) arrays. `reorganize_stac_by_bouts()` auto-detects which format and splits back into per-bout dicts.
- **Anatomy Versions**: v1 has `femur_twist` joints that v2 lacks; `anatomy` config controls which joint/body name lists are used throughout.
- **Multi-bout preprocessing**: When `preprocessing.bouts_csv` points to a `walking_bouts_summary.csv` (with `bout_idx`, `start_frame`, `end_frame` columns), the preprocessor loops over bouts. Single-bout mode uses `frame_start`/`frame_end`.
- **fly50 node ordering**: `geometric_angles.py` hard-codes the 50-node skeleton order as `FLY50_NODE_NAMES`. CSV keypoints use this ordering; XML site order is different and determined by `reorder_keypoints_array`.

### HDF5 Data Format

Preprocessed keypoints (input to STAC), one group per bout:
```python
{
    'bout_0': {
        'keypoints':      (T, N, 3),   # XML-ordered, optionally aligned
        'orig_keypoints': (T, N, 3),   # Before alignment
        'kp_names':       list,        # Names in XML site order
        'skeleton_edges': (E, 2),
        'alignment_info': dict,        # scales, rotation, translation, exclude_indices
    },
    ...
}
```

Postprocessed IK output (per-bout dict):
```python
{
    'info': { 'names_qpos', 'kp_names', 'offsets', 'egocentric_site_names', ... },
    'bout_0': { 'qpos', 'qvel', 'xpos', 'xquat', 'marker_sites', 'kp_data',
                'egocentric_site_pos',  # body-frame site positions
                'position', 'quaternion', 'joints', 'body_positions',  # ReferenceClip fields
                'velocity', 'joints_velocity', 'angular_velocity', ... },
    'bout_1': { ... },
    ...
}
```

### Notebooks (`notebooks/`)

Jupyter notebooks for exploratory analysis — not part of the automated pipeline:
- `IK_preprocessing.ipynb` — interactive version of the preprocessing script; useful for debugging alignment
- `Walking_Bout_Detection_V2.ipynb` — generates `walking_bouts_summary.csv` using `walking_cycle_filter`
- `Joint_Kinematics_Analysis.ipynb` — analyzes STAC joint angles across conditions (amputation, speed)
- `Amputation_Coordination.ipynb` / `Amputation_Longitudinal.ipynb` — inter-leg coordination and longitudinal analysis for amputee flies
- `Scutellum_Height_Running.ipynb` — body height analysis during locomotion
- `Hilbert_Phase_Tester.ipynb` — Hilbert transform-based phase estimation for gait analysis
- `Verify_Data.ipynb` / `Courtship_viz.ipynb` — data quality checks and courtship visualization
- `Keypoint_MoSeq_Analysis.ipynb` — MoSeq-style unsupervised behavior segmentation from keypoints
