# Keypoint Preprocessing for STAC IK

This script preprocesses 3D keypoint data from CSV format into the correct format required by the STAC IK solver.

## What It Does

1. **Loads CSV data** with multi-level headers (node_name, coordinate)
2. **Matches keypoints** to skeleton nodes (using fuzzy matching)
3. **Filters skeleton** to only matched nodes
4. **Reorders keypoints** to match MuJoCo XML site order (required by STAC)
5. **Optionally aligns** using Procrustes transformation
6. **Saves to HDF5** in format compatible with STAC IK solver

## Critical Feature: XML Site Order

**IMPORTANT**: The STAC IK solver expects keypoints in the same order as defined in its config file (`fly_free.yaml` → `KP_NAMES`). This script automatically reorders keypoints by MuJoCo XML site index to match this requirement.

## Usage

### Basic (No Alignment)
```bash
python preprocess_keypoints_for_ik.py \
    --csv_path /path/to/data3D.csv \
    --skeleton_path data/fly50.json \
    --xml_path assets/fruitfly_v1/fruitfly_v1_free.xml \
    --output_dir /path/to/output \
    --bout_name my_bout \
    --frame_start 1000 \
    --frame_end 2000
```

### With Procrustes Alignment
```bash
python preprocess_keypoints_for_ik.py \
    --csv_path /path/to/data3D.csv \
    --skeleton_path data/fly50.json \
    --xml_path assets/fruitfly_v1/fruitfly_v1_free.xml \
    --output_dir /path/to/output \
    --bout_name my_bout_aligned \
    --frame_start 1000 \
    --frame_end 2000 \
    --apply_alignment \
    --apply_scaling \
    --exclude_antenna \
    --exclude_wings
```

## Arguments

### Required
- `--csv_path`: Path to CSV file with keypoint data (multi-level header)
- `--skeleton_path`: Path to skeleton JSON (e.g., `data/fly50.json`)
- `--xml_path`: Path to MuJoCo XML model (e.g., `assets/fruitfly_v1/fruitfly_v1_free.xml`)
- `--output_dir`: Directory to save preprocessed HDF5 files

### Optional
- `--bout_name`: Output filename (default: `preprocessed_bout`)
- `--frame_start`, `--frame_end`: Extract specific frame range (default: all frames)
- `--apply_alignment`: Apply Procrustes alignment
- `--apply_scaling`: Apply scaling during alignment
- `--exclude_antenna`: Exclude antenna from alignment computation (but still transform it)
- `--exclude_wings`: Exclude wings from alignment computation (but still transform them)

## Output Format

The script saves an HDF5 file containing:
```python
{
    'keypoints': np.ndarray,        # (T, N, 3) - preprocessed keypoints in XML order
    'orig_keypoints': np.ndarray,   # (T, N, 3) - original keypoints before alignment
    'kp_names': list,               # Node names in XML order (matches STAC KP_NAMES)
    'skeleton_edges': np.ndarray,   # (E, 2) - skeleton connectivity
    'alignment_info': dict,         # Optional: alignment parameters if --apply_alignment
}
```

### Alignment Info (if enabled)
```python
{
    'scales': float,                # Scale factor applied
    'rotation': np.ndarray,         # (3, 3) rotation matrix
    'translation': np.ndarray,      # (3,) translation vector
    'exclude_indices': list,        # Keypoint indices excluded from alignment
}
```

## Loading Preprocessed Data

```python
import utils.io_dict_to_hdf5 as ioh5

# Load preprocessed data
data = ioh5.load('output/my_bout.h5', enable_jax=True)

keypoints = data['keypoints']        # Shape: (T, 50, 3)
kp_names = data['kp_names']          # List of 50 names in STAC order
edges = data['skeleton_edges']       # Skeleton connectivity

# Verify order matches STAC config
print("Keypoint names (first 10):")
for i, name in enumerate(kp_names[:10]):
    print(f"  {i}: {name}")
```

## Keypoint Order Validation

The script automatically validates that the final keypoint order matches the STAC config `KP_NAMES`. The expected order is:

```
0: Scutellum
1: WingL_base
2: WingR_base
3: Antenna_Base
4: EyeL
5: EyeR
6: WingL_V12
7: WingL_V13
8: WingR_V12
9: WingR_V13
10: Abd_A4
11: Abd_tip
12-49: Leg keypoints (T1L, T1R, T2L, T2R, T3L, T3R)
```

## Common Issues

### 1. CSV Format
CSV must have multi-level headers: `(node_name, coordinate)`. Example:
```
Scutellum,Scutellum,Scutellum,WingL_base,WingL_base,...
x,y,z,x,y,z,...
```

### 2. Skeleton Mismatch
If CSV keypoint names don't match skeleton nodes, the script uses fuzzy matching. Check console output for match quality.

### 3. MuJoCo Site Names
Ensure the XML file has tracking sites named like `tracking[NodeName]`. The script automatically strips this prefix.

## Example Workflow

```bash
# 1. Preprocess multiple bouts
for start in 1000 2000 3000; do
    end=$((start + 1000))
    python preprocess_keypoints_for_ik.py \
        --csv_path data/data3D.csv \
        --skeleton_path data/fly50.json \
        --xml_path assets/fruitfly_v1/fruitfly_v1_free.xml \
        --output_dir output/bouts \
        --bout_name bout_${start} \
        --frame_start $start \
        --frame_end $end \
        --apply_alignment \
        --apply_scaling
done

# 2. Verify output
python -c "
import utils.io_dict_to_hdf5 as ioh5
data = ioh5.load('output/bouts/bout_1000.h5')
print(f'Shape: {data[\"keypoints\"].shape}')
print(f'Keypoints: {len(data[\"kp_names\"])}')
print(f'First 5 names: {data[\"kp_names\"][:5]}')
"

# 3. Run STAC IK (use your STAC workflow)
# The preprocessed HDF5 files are now ready for IK!
```

## Notes

- **JAX Arrays**: The script uses JAX for alignment but saves as NumPy arrays. When loading with `enable_jax=True`, data is converted back to JAX.
- **Scaling**: When `--apply_scaling` is used, the scale factor is saved in `alignment_info` for reference.
- **Exclusion Indices**: Antenna and wings can be excluded from computing the Procrustes transformation, but they are still transformed using the computed rotation/translation/scale.
- **Memory**: For large datasets, consider processing in chunks using `--frame_start` and `--frame_end`.

## Related Files

- `utils/kp_viz.py`: Keypoint matching and reordering functions
- `utils/optimized_floor_alignment.py`: Procrustes alignment (JAX/JIT)
- `utils/io_dict_to_hdf5.py`: HDF5 I/O with nested dictionaries
- `fix_stac_config.py`: Script to fix STAC config offsets (if needed)
