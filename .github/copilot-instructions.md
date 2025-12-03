# Copilot Instructions for 3D Tracking Dataset

## Project Overview

This is a 3D keypoint tracking data processing pipeline for fruit fly (Drosophila) behavioral analysis. The core functionality performs **Procrustes alignment** and **ground contact alignment** on 3D keypoint trajectories using JAX for hardware acceleration.

**Key Dataset**: The primary dataset is `courtship` with 13 keypoints per frame including body parts, legs, wings, and antenna.

## Architecture & Data Flow

### 1. Configuration System (Hydra)

- Uses **Hydra** with OmegaConf for hierarchical configuration management
- Config structure: `configs/config.yaml` (main) → `configs/dataset/*.yaml` → `configs/paths/*.yaml`
- **Multi-environment support**: Path configurations switch between `workstation`, `hyak` (cluster), `desktop`, and `mbook`
- Run scripts with overrides: `python script.py paths=workstation dataset=courtship version=analysis run_id=MyRun`

**Critical path functions** in `utils/path_utils.py`:
- `load_config_with_path_template()` - Load config with specific machine paths
- `override_config_paths()` - Switch paths after loading (e.g., moving between workstation and cluster)
- `convert_dict_to_path()` - Convert path strings to Path objects and create directories

### 2. Core Alignment Pipeline

**File**: `utils/optimized_floor_alignment.py` - All functions are **JIT-compiled with JAX** for performance.

**Two-stage pipeline**:
1. **Procrustes Alignment** → aligns keypoints to reference pose (rotation, translation, optional scaling)
2. **Ground Contact Alignment** → rotates to make ground plane horizontal, then translates vertically so leg tips touch floor

**Key functions**:
- `complete_alignment_pipeline_with_ground_contact()` - Full two-stage pipeline
- `jit_vectorized_procrustes_with_scaling()` - Batch Procrustes with scaling
- `jit_vectorized_procrustes_no_scaling()` - Batch Procrustes without scaling
- `batch_process_with_ground_contact()` - Process entire dictionaries of walking bouts

### 3. Keypoint Exclusion Feature

**See**: `KEYPOINT_EXCLUSION_FEATURE.md` for complete documentation.

**Pattern**: Exclude noisy/variable keypoints (antenna, wings) from computing the Procrustes transformation, but still apply that transformation to them.

```python
# Courtship dataset indices (13 keypoints)
ANTENNA_INDICES = jnp.array([0])
WING_INDICES = jnp.array([6, 7, 8, 12])
LEG_TIP_INDICES = jnp.array([9, 10, 11])  # For ground contact

# Usage: Exclude antenna from alignment computation
aligned_kp, info = jit_vectorized_procrustes_with_scaling(
    kp_data, ref_pose, 
    use_clip_average=True,
    exclude_indices=ANTENNA_INDICES  # Still transformed, not used in computing alignment
)
```

All Procrustes functions accept `exclude_indices` parameter. Test suite: `test_exclude_indices.py`.

### 4. Data I/O

**File**: `utils/io_dict_to_hdf5.py` - HDF5 utilities with JAX support.

- `save()` - Save nested dictionaries to HDF5 with compression
- `load(filename, enable_jax=True)` - Load HDF5, auto-convert to JAX arrays when `enable_jax=True`
- Handles nested dicts, arrays, and automatically converts between dict/list representations

**Typical data structure** (loaded from HDF5):
```python
bout_dict = {
    'bout_0': {
        'orig_kp': jnp.array,  # Shape: (T, N, 3) - T frames, N keypoints, 3D coords
        'aligned_kp': jnp.array,
        'pipeline_info': {...}
    },
    'bout_1': {...}
}
```

### 5. MuJoCo Integration

**Assets**: `fruitfly_v1/` contains MuJoCo XML models of fruit fly anatomy.

**Utilities**:
- `utils/add_wing_tracking_sites.py` - Add tracking sites to MuJoCo XML for wing keypoints
- `utils/add_aligned_keypoint_sites.py` - Add sites for aligned keypoint visualization

Pattern: Programmatically modify MuJoCo XML specs to add/update tracking sites matching dataset keypoints.

## Development Workflows

### Running Tests

```bash
# Test configuration loading with path overrides
python test_configs.py paths=workstation dataset=courtship

# Test keypoint exclusion feature
python3 test_exclude_indices.py

# Example usage of exclusion
python3 example_exclude_keypoints.py
```

### Processing Data

Typical workflow for batch processing:
1. Load config with appropriate paths: `cfg = load_config_with_path_template(config_path, "workstation")`
2. Load data: `bout_dict = io_dict_to_hdf5.load("data.h5", enable_jax=True)`
3. Define reference pose and indices
4. Process: `processed, summary = batch_process_with_ground_contact(bout_dict, ref_pose, end_eff_indices=LEG_TIP_INDICES)`
5. Save: `io_dict_to_hdf5.save("output.h5", processed)`

### Path Management

**When switching machines**:
```python
# Option 1: Load with specific template
cfg = load_config_with_path_template("config.yaml", paths_template="hyak")

# Option 2: Override existing config
cfg = override_config_paths(cfg, "workstation")
```

**Machine-specific paths** are in `configs/paths/{workstation,hyak,desktop,mbook}.yaml`. These use Hydra interpolations like `${paths.user}` and `${dataset.name}`.

## Project-Specific Conventions

### JAX Patterns

- **Always use `jax.numpy` (jnp) not `numpy`** for functions that will be JIT-compiled
- Use `jnp.setdiff1d()` for integer indexing (JIT-compatible), NOT boolean masks
- Shapes follow convention: `(T, N, 3)` = T frames, N keypoints, 3D coordinates
- Pre-compile with `jax.jit()` for performance: `jit_func = jax.jit(my_function)`

### Configuration Patterns

- All configs use Hydra defaults and overrides
- **Custom resolvers** registered in `utils/path_utils.py`:
  - `${multirun_save_dir:...}` - Handles both single and multi-run output directories
  - `${eq:x,y}`, `${divide:x,y}`, `${contains:x,y}` - Custom comparisons
- Dataset-specific configs inherit from base: `defaults: [_self_, dataset: courtship, paths: workstation]`

### Alignment Pipeline Conventions

- **Ground contact indices**: Always use leg tips for `end_eff_indices` (e.g., `[9, 10, 11]` for courtship)
- **Percentile filtering**: Use `percentile=10` to identify ground contact points (lowest 10% of leg positions)
- **Target z-coordinate**: Default `target_z=-0.125` aligns floor to this height
- **Use clip average**: Set `use_clip_average=True` for temporal consistency (computes one transformation from average pose, applies to all frames)

### Error Handling

- Procrustes alignment checks for singular matrices (when points are colinear) and falls back to identity
- Ground plane fitting requires ≥3 points; falls back to identity rotation if insufficient data
- SVD determinant correction ensures proper rotations (det(R) = 1), not reflections

## Key Files Reference

- `utils/optimized_floor_alignment.py` - Core alignment algorithms (JAX/JIT)
- `utils/path_utils.py` - Hydra configuration and path management
- `utils/io_dict_to_hdf5.py` - HDF5 I/O with nested dicts and JAX arrays
- `utils/io.py` - Additional I/O utilities, dataclasses for configs
- `configs/config.yaml` - Main Hydra config entry point
- `configs/dataset/courtship.yaml` - Courtship dataset parameters
- `KEYPOINT_EXCLUSION_FEATURE.md` - Complete documentation of exclusion feature

## Common Gotchas

1. **Path resolution**: Always call `convert_dict_to_path()` after loading config to create directories and convert strings to Path objects
2. **JAX device placement**: Data automatically goes to GPU if available; use `.block_until_ready()` for timing
3. **Keypoint indices**: Courtship has 13 keypoints (0-12); indices are **zero-based** and dataset-specific
4. **HDF5 compression**: Default `compression='gzip', compression_opts=5` balances speed/size
5. **Multirun mode**: Hydra changes output directory structure; use `${multirun_save_dir:...}` resolver to handle both single/multi-run
