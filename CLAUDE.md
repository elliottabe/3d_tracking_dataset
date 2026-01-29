# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D motion capture and biomechanical analysis for fruit fly behavior. Processes JARVIS 3D keypoint tracking data through MuJoCo inverse kinematics to analyze walking bouts, gait patterns, and joint coordination.

**Key characteristics:**
- 50 keypoints tracked per frame (JARVIS): body, wings, 6 legs with multiple joints
- Frame rate: 800 Hz (walking videos), 500 Hz (courtship videos)
- Primary output: Analysis PDFs for walking bouts, gait phases, joint kinematics

## Commands

```bash
# Run config validation (tests Hydra config loading)
python test_configs.py paths=workstation

# Run Jupyter notebooks (primary analysis interface)
jupyter lab notebooks/

# Switch environment paths
python test_configs.py paths=hyak       # For cluster
python test_configs.py paths=workstation  # For local workstation
python test_configs.py paths=desktop    # For desktop
```

## Data Locations

**Sample IK output:**
```
/home/user/src/JARVIS-HybridNet/projects/fly50_V4/predictions/predictions3D/Predictions_3D_20260121-104455/
├── data3D.csv              # JARVIS 3D keypoints (50 keypoints, x/y/z/confidence)
├── Fruitfly_ik_V1_free.h5  # Inverse kinematics output
└── info.yaml               # Metadata
```

**MuJoCo model:**
`/home/user/src/Fly_tracking/assets/fruitfly_v1/fruitfly_v1_free.xml`

## IK Data Format (H5 files)

```python
import h5py
with h5py.File('Fruitfly_ik_V1_free.h5', 'r') as f:
    qpos = f['qpos'][:]           # (N, 93) joint angles
    qvel = f['qvel'][:]           # (N, 92) joint velocities
    xpos = f['xpos'][:]           # (N, 68, 3) body segment positions
    xquat = f['xquat'][:]         # (N, 68, 4) body segment orientations
    marker_sites = f['marker_sites'][:]  # (N, 50, 3) keypoint positions
    names_qpos = [n.decode() for n in f['names_qpos'][:]]
```

**Leg joint naming (11 DoFs per leg):**
- `coxa_flexion`, `coxa_twist`, `coxa` (abduction)
- `femur_twist`, `femur` (flexion)
- `tibia`, `tarsus`, `tarsus2-5`

**Leg naming:** T1=front, T2=mid, T3=hind; `_left`/`_right` suffix
Example: `tibia_T1_left` = front left tibia flexion angle

## Architecture

### Configuration System (Hydra + OmegaConf)
- `configs/config.yaml` - Main config with dataset/run parameters
- `configs/dataset/courtship.yaml` - Dataset-specific settings
- `configs/paths/*.yaml` - Environment-specific paths (workstation, hyak, desktop, mbook)
- `utils/path_utils.py` - Custom OmegaConf resolvers for dynamic path resolution

**Loading config in notebooks:**
```python
from utils.path_utils import load_config_and_override_paths
cfg = load_config_and_override_paths('path/to/saved_config.yaml', 'workstation')
```

### Core Utilities (`utils/`)
| File | Purpose |
|------|---------|
| `path_utils.py` | Hydra config loading, path management, custom resolvers |
| `io.py` | Data loading/saving for HDF5, MATLAB .mat, NWB; keypoint-to-skeleton matching |
| `optimized_floor_alignment.py` | JAX-compiled Procrustes alignment with floor contact preservation |
| `kp_viz.py` | 3D visualization with skeleton rendering and ground plane |
| `add_aligned_keypoint_sites.py` | MuJoCo model site manipulation for frame-by-frame overlay |

### Primary Notebooks
| Notebook | Purpose |
|----------|---------|
| `Sandbox_Strict.ipynb` | Walking bout detection with strict filters (confidence, upright, floor) |
| `Joint_Kinematics_Analysis.ipynb` | Dimensionality reduction (PCA/UMAP) for joint coordination |
| `Walking_Joint_Animation.ipynb` | MuJoCo visualization with frame scrubbing |

### Data Flow
```
JARVIS 3D tracking → data3D.csv (50 keypoints)
         ↓
MuJoCo inverse kinematics → .h5 (93 joint angles)
         ↓
Walking bout detection (Sandbox_Strict)
         ↓
Joint kinematics analysis → PDF reports
```

## Analysis Patterns

**Joint sets for analysis:**
```python
JOINT_SETS = {
    'core': ['coxa_flexion', 'coxa', 'femur', 'tibia'],  # 4×6 = 24 joints
    'main': ['coxa_flexion', 'coxa_twist', 'coxa', 'femur_twist',
             'femur', 'tibia', 'tarsus'],  # 7×6 = 42 joints
    'full': [...all 11 joints per leg...]  # 11×6 = 66 joints
}
```

**Step phase computation:**
```python
from scipy import signal
# Bandpass filter 5-50 Hz for 800 fps data
sos = signal.butter(1, [5/400, 50/400], 'bandpass', output='sos')
ang_filt = signal.sosfiltfilt(sos, angle_normalized)
phase = np.angle(signal.hilbert(ang_filt))  # phase >= 0 = swing
```

**Tripod coordination:**
- Left tripod: L1-R2-L3 (T1_left, T2_right, T3_left)
- Right tripod: R1-L2-R3 (T1_right, T2_left, T3_right)

## Key Dependencies
- **Hydra/OmegaConf** - Configuration management
- **JAX** - Performance-critical numerical operations
- **MuJoCo** - Physics simulation and model visualization
- **h5py** - HDF5 I/O
- **scipy** - Signal processing, MATLAB .mat file support

## Reference Literature

- **Anipose**: https://github.com/lambdaloop/anipose - 3D pose estimation
- **Pratt et al., 2024**: https://github.com/Prattbuw/Treadmill_Paper - Fly walking kinematics
- **Grant's analysis**: `sphere:/home/tuthill/grant/fly_walking_analysis/` (SSH)
  - Key notebook: `joint_kinematics_speed_and_heading.ipynb`
  - Key module: `postproc.py` with `compute_phases()`, `compute_derivs()`, `adjust_rot_angles()`

## Gait Metrics Reference

**Temporal:** Step frequency, stance/swing duration, duty factor
**Spatial:** Step length, swing height, swing linearity (normalized to body length)
**Coordination:** Inter-leg phase, TCS (tripod coordination strength), n_legs_stance (1-6)
**Posture:** Body pitch, body height, support polygon area
