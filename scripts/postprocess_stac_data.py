"""
Postprocess STAC output data:
1. Load keypoint data and extract clip_lengths
2. Load STAC IK output
3. Compute egocentric site positions (JAX/MJX vectorized)
4. Reorganize data by bouts
5. Save processed output

Usage:
    python postprocess_stac_data.py paths=workstation dataset=
    python postprocess_stac_data.py paths=hyak dataset=courtship
"""


import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

import jax
# Persistent compilation cache dir; override with JAX_COMPILATION_CACHE_DIR
# (e.g. point at scratch on clusters where /tmp is node-local/ephemeral).
jax.config.update("jax_compilation_cache_dir",
                  os.environ.get("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_cache"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# Note: jax_persistent_cache_enable_xla_caches may not be available in all JAX versions
try:
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
except AttributeError:
    pass  # Skip if not available in this JAX version

import sys
import re
from pathlib import Path
import time
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import io_dict_to_hdf5 as ioh5
from utils.stac_data_utils import (
    reorganize_stac_by_bouts, 
    print_bout_dict_structure,
    interpolate_trajectory,
    adjust_root_z_for_floor
)
from utils.path_utils import load_config_with_path_template, convert_dict_to_path, convert_dict_to_string
from utils.io import load_stac_data
from utils.mjx_preprocess import process_clip, ReferenceClip
from utils.geometric_angles import compute_geometric_angles_all_legs


def filter_end_effectors_to_tracked(end_effector_names, kp_names, verbose=False):
    """Filter floor-alignment end effectors to legs actually tracked in the data.

    A claw is a valid ground-contact reference only if its leg was constrained
    during IK. An untracked leg (e.g. an amputated leg with no distal keypoints)
    sits at the model rest pose and would bias floor-height estimation.

    A configured end effector is kept when its leg tag — parsed from names like
    ``claw_T1_left`` -> ``T1L`` — has a tarsal keypoint (``T1L_Ta*``) in
    ``kp_names``. End effectors whose name can't be parsed are kept (conservative),
    and if ``kp_names`` is unavailable or filtering would drop everything, the
    original list is returned unchanged.

    Args:
        end_effector_names: Configured end-effector (claw) body-name fragments.
        kp_names: Keypoint names present in the data (from STAC ``info``).
        verbose: Print which end effectors were dropped.

    Returns:
        Filtered list of end-effector names.
    """
    names = list(end_effector_names)
    if not kp_names:
        return names
    kp_set = set(kp_names)

    def leg_tag(eff):
        m = re.search(r'T([123])_(left|right)', eff)
        if not m:
            return None
        return f"T{m.group(1)}{'L' if m.group(2) == 'left' else 'R'}"

    def is_tracked(tag):
        # Leg is tracked distally if it has any tarsal keypoint (e.g. *_TaTip).
        return any(k.startswith(f"{tag}_Ta") for k in kp_set)

    kept, dropped = [], []
    for eff in names:
        tag = leg_tag(eff)
        if tag is None or is_tracked(tag):
            kept.append(eff)
        else:
            dropped.append(eff)
    if dropped and verbose:
        print(f"  Floor alignment: dropping untracked end effectors {dropped} "
              f"(no tarsal keypoints in data)")
    return kept if kept else names  # never return empty


def load_segment_scales(preprocessed_path: Path):
    """Read per-segment calibration scales from the preprocessed h5 (robust)."""
    from utils.segment_calibration import read_segment_scales
    try:
        return read_segment_scales(preprocessed_path)
    except Exception as exc:
        print(f"  [calibration] could not read segment_scales: {exc}")
        return None


def load_clip_lengths(data_path: Path, filename: str) -> tuple:
    """
    Load clip_lengths and fly_ids from preprocessed keypoint data.

    Args:
        data_path: Path to directory containing preprocessed_bout.h5

    Returns:
        clip_lengths: List of frame counts for each bout
        fly_ids: List of fly identifiers for each bout (or None if not present)
    """
    print("=" * 80)
    print("LOADING CLIP LENGTHS")
    print("=" * 80)

    preprocessed_path = data_path / filename
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_path}")

    print(f"Loading: {preprocessed_path}")
    data_dict = ioh5.load(preprocessed_path, enable_jax=False)

    clip_lengths = [
        data_dict[key]['keypoints'].shape[0]
        for key in data_dict
        if 'keypoints' in data_dict[key]
    ]

    # Extract fly_ids from info if present
    fly_ids = None
    if 'info' in data_dict and 'fly_ids' in data_dict['info']:
        fly_ids = list(data_dict['info']['fly_ids'])
        print(f"✓ Found fly_ids: {fly_ids}")

    # Extract source_flies (which fly originally detected each bout)
    source_flies = None
    if 'info' in data_dict and 'source_flies' in data_dict['info']:
        source_flies = list(data_dict['info']['source_flies'])
        print(f"✓ Found source_flies: {source_flies}")

    # Extract bucket (validity bucket per bout: fly0_only/fly1_only/both)
    buckets = None
    if 'info' in data_dict and 'bucket' in data_dict['info']:
        buckets = list(data_dict['info']['bucket'])
        print(f"✓ Found bucket tags: {buckets}")

    # Extract original raw-video frame ranges per bout
    start_frames = None
    end_frames = None
    if 'info' in data_dict and 'start_frames' in data_dict['info']:
        start_frames = [int(x) for x in data_dict['info']['start_frames']]
    if 'info' in data_dict and 'end_frames' in data_dict['info']:
        end_frames = [int(x) for x in data_dict['info']['end_frames']]

    print(f"✓ Found {len(clip_lengths)} bouts")
    print(f"  Clip lengths: {clip_lengths}")
    print(f"  Total frames: {sum(clip_lengths)}")
    print()

    return clip_lengths, fly_ids, source_flies, buckets, start_frames, end_frames


def load_stac_output(stac_path: Path):
    """
    Load STAC IK output data.
    
    Args:
        stac_path: Path to STAC output HDF5 file
        
    Returns:
        cfg_d: STAC config dictionary
        d: STAC data object
        stac_data: STAC data as dictionary
    """
    print("=" * 80)
    print("LOADING STAC OUTPUT")
    print("=" * 80)
    
    if not stac_path.exists():
        raise FileNotFoundError(f"STAC output not found: {stac_path}")
    
    print(f"Loading: {stac_path}")
    cfg_d, d = load_stac_data(stac_path.as_posix())
    stac_data = d.as_dict()
    
    print(f"✓ Loaded STAC data")
    print(f"  qpos shape: {d.qpos.shape}")
    print(f"  xpos shape: {d.xpos.shape}")
    print(f"  Keypoint names: {d.kp_names}")
    
    # Print model info from config if available
    if 'model' in cfg_d:
        print(f"  STAC config model info:")
        if 'model_path' in cfg_d.model:
            print(f"    model_path: {cfg_d.model.model_path}")
        if 'nq' in cfg_d.model:
            print(f"    nq (DoFs): {cfg_d.model.nq}")
    print()
    
    return cfg_d, d, stac_data


def trim_bouts_to_valid_span(
    bout_dict: Dict,
    bout_keys: List[str],
    verbose: bool = False,
) -> Dict:
    """Trim leading/trailing all-NaN frame runs from each bout, in place.

    `filtering.interpolation.max_edge_extrap_frames` (preprocessing) caps
    edge gap-filling; frames beyond that cap stay NaN. For single-fly
    `free_running` data `pair_validity.enabled` is False, so nothing drops
    those NaN frames before they reach STAC and then
    `interpolate_trajectory`'s `scipy.interp1d(kind='cubic')` -- a *global*
    solve where a single leftover NaN poisons the entire interpolated array.

    For each bout, a frame is BAD if any coordinate is NaN in `kp_data` or
    `qpos` (checked independently and OR'd -- the two can differ). The bout
    is then sliced to `[first_valid, last_valid]` across every per-bout array
    whose leading dimension equals the bout's original frame count (e.g.
    `qpos, qvel, xpos, xquat, kp_data, marker_sites, site_xpos`); arrays with
    a different leading dim (e.g. `offsets`) are left alone.

    A bout with no NaN is returned untouched (true no-op: no reassignment).
    A bout with a NaN gap *inside* `[first_valid, last_valid]` is not
    edge-damaged -- trimming cannot save it -- so it is left completely
    untouched and reported in `interior_nan_bouts` instead of being silently
    passed through half-fixed. A wholly-NaN bout (no valid frame at all) is
    likewise left untouched and reported (never turned into a zero-length
    bout).

    Args:
        bout_dict: Dict with 'info' and per-bout data (mutated in place).
        bout_keys: Bout keys to consider (excluding 'info').
        verbose: Print a one-line summary and loudly warn about any
            interior/all-NaN bouts.

    Returns:
        Summary dict: {'bouts_trimmed': int, 'frames_dropped': int,
        'interior_nan_bouts': list[str]}.
    """
    summary = {
        'bouts_trimmed': 0,
        'frames_dropped': 0,
        'interior_nan_bouts': [],
    }

    for bout_key in bout_keys:
        bout = bout_dict[bout_key]

        # Build the per-frame BAD mask from whichever of kp_data/qpos are
        # present, OR'd together (either one going NaN marks the frame bad).
        bad_mask = None
        T = None
        for check_key in ('kp_data', 'qpos'):
            arr = bout.get(check_key)
            if arr is None:
                continue
            arr_np = np.asarray(arr)
            if T is None:
                T = arr_np.shape[0]
            frame_bad = np.isnan(arr_np.reshape(arr_np.shape[0], -1)).any(axis=1)
            bad_mask = frame_bad if bad_mask is None else (bad_mask | frame_bad)

        if bad_mask is None or not bad_mask.any():
            # Nothing to check against, or a fully clean bout: true no-op.
            continue

        valid_idx = np.flatnonzero(~bad_mask)
        if valid_idx.size == 0:
            # Entirely NaN -- leave untouched, do not fabricate a zero-length bout.
            summary['interior_nan_bouts'].append(bout_key)
            if verbose:
                print(f"  [trim_nan_edges] WARNING: bout '{bout_key}' is entirely "
                      f"NaN ({T} frames) -- left untouched")
            continue

        first_valid = int(valid_idx[0])
        last_valid = int(valid_idx[-1])

        # Interior NaN (a gap inside the valid span) is not fixable by
        # trimming edges -- fail loudly instead of emitting a half-fixed bout.
        if bad_mask[first_valid:last_valid + 1].any():
            summary['interior_nan_bouts'].append(bout_key)
            if verbose:
                n_interior = int(bad_mask[first_valid:last_valid + 1].sum())
                print(f"  [trim_nan_edges] WARNING: bout '{bout_key}' has "
                      f"{n_interior} interior NaN frame(s) inside its valid "
                      f"span -- trimming cannot fix this; left untouched")
            continue

        frames_dropped = T - (last_valid - first_valid + 1)

        for key, val in list(bout.items()):
            if (isinstance(val, (np.ndarray, jnp.ndarray))
                    and val.ndim >= 1 and val.shape[0] == T):
                bout[key] = val[first_valid:last_valid + 1]

        summary['bouts_trimmed'] += 1
        summary['frames_dropped'] += frames_dropped

    if verbose:
        if summary['interior_nan_bouts']:
            print(f"  [trim_nan_edges] {'!' * 60}")
            print(f"  [trim_nan_edges] {len(summary['interior_nan_bouts'])} bout(s) "
                  f"have UNFIXABLE interior/all-NaN gaps: {summary['interior_nan_bouts']}")
            print(f"  [trim_nan_edges] {'!' * 60}")
        print(f"  [trim_nan_edges] Trimmed {summary['bouts_trimmed']} bouts, "
              f"dropped {summary['frames_dropped']} frames")

    return summary


def reorganize_and_save(
    stac_data: dict,
    clip_lengths: list,
    output_path: Path,
    verbose: bool = True
):
    """
    Reorganize STAC data by bouts and save to HDF5.
    
    Args:
        stac_data: Dictionary of STAC output data
        clip_lengths: List of frame counts for each bout
        output_path: Path to save reorganized data
        verbose: Print structure information
    """
    print("=" * 80)
    print("REORGANIZING BY BOUTS")
    print("=" * 80)
    
    # Reorganize
    bout_dict = reorganize_stac_by_bouts(
        stac_data=stac_data,
        clip_lengths=clip_lengths,
    )
    
    # Print structure
    if verbose:
        print()
        print_bout_dict_structure(bout_dict, show_values=False)
    
    # Save
    print("\n" + "=" * 80)
    print("SAVING OUTPUT")
    print("=" * 80)
    print(f"Saving to: {output_path}")
    ioh5.save(output_path, bout_dict)
    print(f"✓ Saved successfully")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    
    return bout_dict


def process_bout_with_mjx(
    bout_qpos: np.ndarray,
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    dt: float,
    max_qvel: float = 20.0,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Process a single bout with MJX to compute velocities and updated body positions.
    
    Args:
        bout_qpos: Joint positions (T, nq)
        mjx_model: MJX model
        mjx_data: MJX data
        dt: Timestep in seconds
        max_qvel: Maximum velocity (not currently used)
        verbose: Print progress
        
    Returns:
        Dictionary with qpos, qvel, xpos, xquat arrays
    """
    if verbose:
        print(f"    Processing with MJX (dt={dt:.4f}s)...")
    
    # Convert to JAX array
    bout_qpos_jax = jnp.array(bout_qpos)
    
    # Process with MJX
    ref_clip = process_clip(bout_qpos_jax, mjx_model, mjx_data, 
                           max_qvel=max_qvel, dt=dt)
    
    # Reconstruct full qpos from components
    qpos_full = jnp.concatenate([
        ref_clip.position,      # (T, 3)
        ref_clip.quaternion,    # (T, 4)
        ref_clip.joints         # (T, n_joints)
    ], axis=1)
    
    # Reconstruct full qvel from components
    qvel_full = jnp.concatenate([
        ref_clip.velocity,          # (T, 3)
        ref_clip.angular_velocity,  # (T, 3)
        ref_clip.joints_velocity    # (T, n_joints)
    ], axis=1)
    
    return {
        'qpos': np.array(qpos_full),
        'qvel': np.array(qvel_full),
        'xpos': np.array(ref_clip.body_positions),
        'xquat': np.array(ref_clip.body_quaternions)
    }


def process_bouts_batched(
    qpos_batch: jnp.ndarray,
    xpos_batch: jnp.ndarray,
    end_eff_indices: List[int],
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    dt: float,
    site_indices: jnp.ndarray = None,
    thorax_body_idx: int = None,
    percentile: float = 5.0,
    target_z: float = -0.125,
    max_qvel: float = 20.0
) -> Dict[str, jnp.ndarray]:
    """
    Process all bouts in parallel using vmap for floor adjustment, MJX, and egocentric sites.
    
    Args:
        qpos_batch: Batched qpos (n_bouts, max_T, nq)
        xpos_batch: Batched xpos (n_bouts, max_T, nbodies, 3)
        end_eff_indices: Indices of end effector bodies
        mjx_model: MJX model
        mjx_data: MJX data
        dt: Timestep
        site_indices: Indices of tracking sites (optional, for egocentric computation)
        thorax_body_idx: Index of thorax body (optional, for egocentric computation)
        percentile: Ground contact percentile
        target_z: Target floor height
        max_qvel: Max velocity
        
    Returns:
        Dictionary with batched qpos, qvel, xpos, xquat, and optionally xpos_egocentric
    """
    # Define single-bout floor adjustment
    def adjust_single_bout(qpos, xpos):
        # Skip floor adjustment if no end effectors provided
        if len(end_eff_indices) == 0:
            return qpos
        
        # Extract end effector z-positions
        end_eff_z = xpos[:, end_eff_indices, 2]  # (T, n_end_eff)
        
        # Compute floor as mean of lowest percentile
        floor_z = jnp.mean(jnp.quantile(end_eff_z, percentile / 100.0, axis=0))
        
        # Compute offset
        z_offset = target_z - floor_z
        
        # Apply to root z-position
        qpos_adjusted = qpos.at[:, 2].add(z_offset)
        
        return qpos_adjusted
    
    # Define single-bout MJX processing
    def process_single_bout(qpos):
        ref_clip = process_clip(qpos, mjx_model, mjx_data, max_qvel=max_qvel, dt=dt)
        
        # Reconstruct qpos and qvel
        qpos_full = jnp.concatenate([
            ref_clip.position,
            ref_clip.quaternion,
            ref_clip.joints
        ], axis=1)
        
        qvel_full = jnp.concatenate([
            ref_clip.velocity,
            ref_clip.angular_velocity,
            ref_clip.joints_velocity
        ], axis=1)
        
        return qpos_full, qvel_full, ref_clip.body_positions, ref_clip.body_quaternions
    
    # Define single-timestep egocentric computation
    def compute_egocentric_sites_single(qpos):
        mjx_data_temp = mjx.make_data(mjx_model)
        mjx_data_temp = mjx_data_temp.replace(qpos=qpos)
        mjx_data_temp = mjx.forward(mjx_model, mjx_data_temp)
        
        thorax_xpos = mjx_data_temp.xpos[thorax_body_idx]
        thorax_xmat = mjx_data_temp.xmat[thorax_body_idx].reshape(3, 3)
        site_xpos = mjx_data_temp.site_xpos[site_indices]
        
        relative_pos = site_xpos - thorax_xpos[None, :]
        egocentric_pos = jnp.dot(relative_pos, thorax_xmat)

        return site_xpos, egocentric_pos
    
    # Vmap floor adjustment
    qpos_adjusted = jax.vmap(adjust_single_bout)(qpos_batch, xpos_batch)    
    
    # Vmap MJX processing
    qpos_out, qvel_out, xpos_out, xquat_out = jax.vmap(process_single_bout)(qpos_adjusted)
    
    result = {
        'qpos': qpos_out,
        'qvel': qvel_out,
        'xpos': xpos_out,
        'xquat': xquat_out
    }
    
    # Optionally compute egocentric positions
    if site_indices is not None and thorax_body_idx is not None:
        # Vmap over timesteps then bouts
        compute_egocentric_bout = jax.vmap(compute_egocentric_sites_single)
        compute_egocentric_batched = jax.vmap(compute_egocentric_bout)
        site_xpos, xpos_egocentric = compute_egocentric_batched(qpos_out)
        result['site_xpos'] = site_xpos
        result['xpos_egocentric'] = xpos_egocentric
    
    return result


def process_all_bouts(
    bout_dict: Dict,
    cfg: DictConfig,
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    site_indices: jnp.ndarray = None,
    thorax_body_idx: int = None,
    site_names: List[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Process all bouts with interpolation, floor adjustment, and MJX forward kinematics.
    
    Pipeline per bout:
    1. Interpolate qpos, xpos, xquat, kp_data (if enabled)
    2. Adjust root z-position for floor contact (if enabled)
    3. Run MJX forward kinematics to compute qvel and updated xpos/xquat (if enabled)
    4. Compute egocentric site positions (if site_indices provided)
    5. Update bout dict with processed data
    
    Original data gets suffix '_stac', final processed data has clean names.
    
    Args:
        bout_dict: Dictionary with 'info' and bout data
        cfg: Hydra configuration
        mjx_model: Compiled MJX model
        mjx_data: MJX data structure
        site_indices: Indices of tracking sites (optional, for egocentric)
        thorax_body_idx: Index of thorax body (optional, for egocentric)
        site_names: Names of tracking sites (optional, for egocentric)
        verbose: Print progress
        
    Returns:
        Updated bout_dict with processed data
    """
    print("=" * 80)
    print("PROCESSING BOUTS (BATCHED)")
    print("=" * 80)
    
    interp_cfg = cfg.postprocessing.interpolation
    floor_cfg = cfg.postprocessing.floor_alignment
    mjx_cfg = cfg.postprocessing.mjx_processing
    
    bout_keys = sorted([k for k in bout_dict.keys() if k != 'info'])

    # Trim leading/trailing all-NaN frame runs BEFORE anything else touches
    # the bout arrays, so clip_lengths_original (below) reflects the trimmed
    # length and the '_stac'-suffixed data / interpolation never sees the
    # edge NaNs that would otherwise poison interp1d(kind='cubic')'s global
    # solve. See configs/postprocessing/default.yaml: trim_nan_edges.
    trim_cfg = cfg.postprocessing.get('trim_nan_edges', {})
    if trim_cfg.get('enabled', False):
        trim_summary = trim_bouts_to_valid_span(bout_dict, bout_keys, verbose=verbose)
        if trim_summary['interior_nan_bouts']:
            print(f"  ⚠ WARNING: {len(trim_summary['interior_nan_bouts'])} bout(s) have "
                  f"unfixable interior/all-NaN gaps (trimming only fixes edges): "
                  f"{trim_summary['interior_nan_bouts']}")

    # Get original clip lengths (computed AFTER trimming so downstream
    # lengths -- clip_lengths_interp_unpadded, combine_data.py's unpadding --
    # stay consistent with the trimmed data).
    clip_lengths_original = [
        bout_dict[key]['qpos'].shape[0]
        for key in bout_keys
    ]
    
    # Step 1: Rename original STAC data with suffix
    if verbose:
        print("\nRenaming original STAC data with '_stac' suffix...")
    for bout_key in bout_keys:
        bout = bout_dict[bout_key]
        for key in ['qpos', 'xpos', 'xquat', 'kp_data']:
            if key in bout:
                bout[f'{key}_stac'] = bout[key]
    
    # Step 2: Interpolation (if enabled)
    clip_lengths_interp = []
    if interp_cfg.enabled:
        if verbose:
            print(f"\nInterpolating {interp_cfg.source_hz}Hz → {interp_cfg.target_hz}Hz...")
        
        # First pass: interpolate without padding to get actual lengths
        for bout_key in bout_keys:
            bout = bout_dict[bout_key]
            for key in ['qpos', 'xpos', 'xquat', 'kp_data']:
                stac_key = f'{key}_stac' if key in ['qpos', 'xpos', 'xquat', 'kp_data'] else key
                if stac_key in bout:
                    bout[key] = interpolate_trajectory(
                        bout[stac_key],
                        source_hz=interp_cfg.source_hz,
                        target_hz=interp_cfg.target_hz,
                        method=interp_cfg.method,
                        pad_to_length=None  # No padding yet
                    )
            clip_lengths_interp.append(bout['qpos'].shape[0])
        
        # Find max length after interpolation
        max_clip_length = max(clip_lengths_interp)
        
        if verbose:
            print(f"  Interpolated lengths: min={min(clip_lengths_interp)}, max={max_clip_length}")
            print(f"  Padding all bouts to max length: {max_clip_length}...")
        
        # Second pass: pad all bouts to max length
        for bout_idx, bout_key in enumerate(bout_keys):
            bout = bout_dict[bout_key]
            current_length = clip_lengths_interp[bout_idx]
            
            if current_length < max_clip_length:
                for key in ['qpos', 'xpos', 'xquat', 'kp_data']:
                    if key in bout:
                        # Pad by repeating last frame
                        n_pad = max_clip_length - current_length
                        padding = np.tile(bout[key][-1:], (n_pad,) + (1,) * (len(bout[key].shape) - 1))
                        bout[key] = np.concatenate([bout[key], padding], axis=0)
        
        clip_lengths_new = [max_clip_length] * len(bout_keys)
    else:
        # No interpolation - copy data
        for bout_key in bout_keys:
            bout = bout_dict[bout_key]
            for key in ['qpos', 'xpos', 'xquat']:
                if f'{key}_stac' in bout:
                    bout[key] = bout[f'{key}_stac']
        clip_lengths_new = clip_lengths_original
        
        # If floor/MJX processing will be done, pad to max length for batching
        if floor_cfg.enabled or mjx_cfg.enabled:
            max_clip_length = max(clip_lengths_original)
            if verbose:
                print(f"  Padding all bouts to max length for batched processing: {max_clip_length}...")
            
            for bout_idx, bout_key in enumerate(bout_keys):
                bout = bout_dict[bout_key]
                current_length = clip_lengths_original[bout_idx]
                
                if current_length < max_clip_length:
                    for key in ['qpos', 'xpos', 'xquat', 'kp_data']:
                        if key in bout:
                            # Pad by repeating last frame
                            n_pad = max_clip_length - current_length
                            padding = np.tile(bout[key][-1:], (n_pad,) + (1,) * (len(bout[key].shape) - 1))
                            bout[key] = np.concatenate([bout[key], padding], axis=0)
            
            clip_lengths_new = [max_clip_length] * len(bout_keys)
    
    # Step 3 & 4: Batched floor adjustment and MJX processing (if enabled)
    if floor_cfg.enabled or mjx_cfg.enabled:
        if verbose:
            print(f"\nStacking {len(bout_keys)} bouts for batched processing...")
        
        # Stack all bouts into batched arrays
        qpos_batch = np.stack([bout_dict[key]['qpos'] for key in bout_keys], axis=0)
        xpos_batch = np.stack([bout_dict[key]['xpos'] for key in bout_keys], axis=0)
        
        # Convert to JAX arrays
        qpos_batch = jnp.array(qpos_batch)
        xpos_batch = jnp.array(xpos_batch)
        
        if verbose:
            print(f"  Batch shape: qpos={qpos_batch.shape}, xpos={xpos_batch.shape}")
        
        # Find end effector indices.
        # Auto-filter the configured end effectors to legs actually tracked in
        # the data: an untracked leg (e.g. amputated) has no tarsal keypoints and
        # its claw rests at the model default pose, which would bias floor
        # detection. This replaces any need for per-dataset end_effector overrides.
        names_xpos = bout_dict['info']['names_xpos']
        end_effector_names = filter_end_effectors_to_tracked(
            floor_cfg.end_effector_names,
            bout_dict['info'].get('kp_names', []),
            verbose=verbose,
        )
        end_eff_indices = [
            i for i, name in enumerate(names_xpos)
            if any(eff_name in name for eff_name in end_effector_names)
        ]
        
        if len(end_eff_indices) == 0:
            if floor_cfg.enabled:
                print(f"  ⚠ ERROR: No end effectors found matching {end_effector_names}")
                print(f"  Available body names: {names_xpos}")
                print(f"  Floor alignment will be DISABLED due to missing end effectors!")
                # Disable floor alignment if end effectors not found
                floor_cfg_enabled = False
            else:
                floor_cfg_enabled = False
        else:
            if verbose:
                print(f"  Found {len(end_eff_indices)} end effectors: {[names_xpos[i] for i in end_eff_indices]}")
            floor_cfg_enabled = floor_cfg.enabled
        
        # Check if egocentric computation will be included
        if site_indices is not None and thorax_body_idx is not None:
            if verbose:
                print(f"  Including egocentric computation with {len(site_names)} sites")
        
        # Compute timestep
        dt = 1.0 / interp_cfg.target_hz if interp_cfg.enabled else 1.0 / interp_cfg.source_hz
        
        if verbose:
            processing_stages = []
            if floor_cfg_enabled and len(end_eff_indices) > 0:
                processing_stages.append("floor")
            if mjx_cfg.enabled:
                processing_stages.append("MJX")
            if site_indices is not None and thorax_body_idx is not None:
                processing_stages.append("egocentric")
            print(f"  JIT compiling batched processing ({' + '.join(processing_stages)}, dt={dt:.5f}s)...")
        
        # JIT compile the batched processing
        process_batched_jit = jax.jit(
            lambda qpos, xpos: process_bouts_batched(
                qpos, xpos, end_eff_indices, mjx_model, mjx_data,
                dt, site_indices, thorax_body_idx,
                floor_cfg.percentile, floor_cfg.target_z, mjx_cfg.max_qvel
            )
        )
        
        # Warm-up compilation
        _ = process_batched_jit(qpos_batch[:1, :10], xpos_batch[:1, :10])
        _['qpos'].block_until_ready()
        
        if verbose:
            print(f"  Running batched processing on GPU...")
        
        # Process all bouts
        import time
        start = time.time()
        processed = process_batched_jit(qpos_batch, xpos_batch)
        processed['qpos'].block_until_ready()
        elapsed = time.time() - start
        
        if verbose:
            print(f"  ✓ Processed {len(bout_keys)} bouts in {elapsed:.2f}s ({elapsed/len(bout_keys):.3f}s per bout)")
            
            # Show floor alignment verification if enabled
            if floor_cfg_enabled and len(end_eff_indices) > 0:
                # Check z-offset applied
                z_before = float(jnp.mean(qpos_batch[:, :, 2]))
                z_after = float(jnp.mean(processed['qpos'][:, :, 2]))
                z_offset = z_after - z_before
                print(f"  Floor alignment z-offset applied: {z_offset:.4f} (mean root z: {z_before:.4f} → {z_after:.4f})")
        
        # Unpack results back into bout_dict
        for bout_idx, bout_key in enumerate(bout_keys):
            bout = bout_dict[bout_key]
            bout['qpos'] = np.array(processed['qpos'][bout_idx])
            bout['qvel'] = np.array(processed['qvel'][bout_idx])
            bout['xpos'] = np.array(processed['xpos'][bout_idx])
            bout['xquat'] = np.array(processed['xquat'][bout_idx])
            
            # Unpack site positions if computed
            if 'site_xpos' in processed:
                bout['site_xpos'] = np.array(processed['site_xpos'][bout_idx])
            if 'xpos_egocentric' in processed:
                bout['xpos_egocentric'] = np.array(processed['xpos_egocentric'][bout_idx])
        
        # Store site names if egocentric was computed
        if site_names is not None:
            bout_dict['info']['site_names_egocentric'] = site_names
    
    # Update info with both old and new clip lengths
    bout_dict['info']['clip_lengths_original'] = clip_lengths_original
    bout_dict['info']['clip_lengths'] = clip_lengths_new
    if interp_cfg.enabled:
        bout_dict['info']['clip_lengths_interp_unpadded'] = clip_lengths_interp
    
    if verbose:
        print(f"\n✓ Processed {len(bout_keys)} bouts")
        print(f"  Original frames per bout: {clip_lengths_original}")
        print(f"  Final frames per bout (padded): {clip_lengths_new[0] if clip_lengths_new else 0}")
        print(f"  Total frames: {sum(clip_lengths_original)} → {sum(clip_lengths_new)}")
        if interp_cfg.enabled:
            print(f"  Unpadded interpolated lengths: {clip_lengths_interp}")
    print()
    
    return bout_dict


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main postprocessing pipeline using Hydra configuration."""
    
    print("\n" + "=" * 80)
    print("STAC DATA POSTPROCESSING PIPELINE")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print()
    
    # Convert path strings to Path objects
    cfg.paths = convert_dict_to_path(cfg.paths)
    
    # Resolve paths
    data_path = cfg.paths.data_dir
    flybody_path = Path(cfg.anatomy.mjcf_path)
    floor_path = Path(cfg.anatomy.arena_path)
    
    print(f"Resolved paths:")
    print(f"  Data directory: {data_path}")
    print(f"  Flybody model: {flybody_path}")
    print(f"  Floor model: {floor_path}")
    print()
    
    # Define file paths from config
    stac_filename = cfg.postprocessing.stac_output_file
    preprocessed_filename = cfg.postprocessing.preprocessed_file
    output_filename = cfg.postprocessing.output_file
    
    stac_path = data_path / stac_filename
    preprocessed_path = data_path / preprocessed_filename
    output_path = data_path / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"File paths:")
    print(f"  STAC output: {stac_path}")
    print(f"  Preprocessed data: {preprocessed_path}")
    print(f"  Output: {output_path}")
    print()
    
    # Step 1: Load clip lengths and fly_ids
    clip_lengths, fly_ids, source_flies, buckets, start_frames, end_frames = load_clip_lengths(data_path, cfg.postprocessing.preprocessed_file)
    
    # Step 2: Load STAC output
    cfg_d, d, stac_data = load_stac_output(stac_path)
    
    # Step 3: Reorganize by bouts (before processing)
    print("=" * 80)
    print("REORGANIZING BY BOUTS")
    print("=" * 80)
    
    bout_dict = reorganize_stac_by_bouts(
        stac_data=stac_data,
        clip_lengths=clip_lengths,
    )

    # Inject fly_ids from preprocessing (STAC solver doesn't preserve them)
    if fly_ids is not None:
        bout_dict['info']['fly_ids'] = fly_ids
    if source_flies is not None:
        bout_dict['info']['source_flies'] = source_flies
    if buckets is not None:
        bout_dict['info']['bucket'] = buckets
    if start_frames is not None:
        bout_dict['info']['start_frames'] = start_frames
    if end_frames is not None:
        bout_dict['info']['end_frames'] = end_frames

    # if cfg.postprocessing.verbose:
    #     print()
    #     print_bout_dict_structure(bout_dict, show_values=False)
    print()
    
    # Step 4: Compile MJX model for processing (if enabled)
    mjx_model = None
    mjx_data = None
    site_indices = None
    thorax_body_idx = None
    site_names = None
    
    if (cfg.postprocessing.interpolation.enabled or 
        cfg.postprocessing.floor_alignment.enabled or
        cfg.postprocessing.mjx_processing.enabled):
        
        print("=" * 80)
        print("COMPILING MJX MODEL")
        print("=" * 80)
        print("Compiling MuJoCo models...")
        
        # Compile models
        spec = mujoco.MjSpec().from_file(flybody_path.as_posix())
        floor_spec = mujoco.MjSpec().from_file(floor_path.as_posix())
        
        # Use STAC model path if available
        if cfg_d and 'model' in cfg_d and 'model_path' in cfg_d.model:
            stac_model_path = Path(cfg_d.model.model_path)
            if stac_model_path.exists():
                print(f"  Using STAC model: {stac_model_path}")
                spec = mujoco.MjSpec().from_file(stac_model_path.as_posix())

        # Apply the SAME subject-specific per-segment morph STAC used, so FK on the
        # solved qpos (xpos, floor alignment, velocities) is consistent with the
        # model the poses were solved on. Scales read from the preprocessed h5.
        seg_scales = load_segment_scales(preprocessed_path)
        if seg_scales:
            sys.path.insert(0, str(project_root / "stac-mjx"))
            from stac_mjx.rescale import rescale_per_segment
            rescale_per_segment(spec, seg_scales)
            print(f"  [calibration] morphed {len(seg_scales)} body segments for FK consistency")

        spawn_frame = floor_spec.worldbody.add_frame(
            pos=[0, 0, -.125],
            quat=[1, 0, 0, 0],
        )
        spawn_body = spawn_frame.attach_body(spec.body("thorax"), "", suffix='_fly')
        
        # Get standard MuJoCo model  
        mj_model = floor_spec.compile()
        
        # Compile to MJX model for fast batched forward kinematics
        mjx_model = mjx.put_model(mj_model)
        mjx_data = mjx.make_data(mjx_model)
        
        print(f"✓ Models compiled")
        print(f"  nq (DoFs): {mj_model.nq}")
        
        # Extract site information for egocentric computation (if enabled)
        if cfg.postprocessing.get('egocentric_sites', {}).get('enabled', True):
            site_names = [site.name for site in floor_spec.sites if 'tracking' in site.name]
            _suffix = '_fly'
            thorax_body_idx = mj_model.body(f"thorax{_suffix}").id
            site_indices = jnp.array([floor_spec.site(site_name).id for site_name in site_names])
            print(f"  Extracted {len(site_names)} tracking sites for egocentric computation")
        
        print()
    
    # Step 5: Process all bouts (interpolation, floor adjustment, MJX)
    if (cfg.postprocessing.interpolation.enabled or 
        cfg.postprocessing.floor_alignment.enabled or
        cfg.postprocessing.mjx_processing.enabled):
        
        bout_dict = process_all_bouts(
            bout_dict,
            cfg,
            mjx_model,
            mjx_data,
            site_indices=site_indices,
            thorax_body_idx=thorax_body_idx,
            site_names=site_names,
            verbose=cfg.postprocessing.verbose
        )
    
    # Step 6: Compute geometric joint angles (anipose-compatible)
    # Concatenate all bouts and compute once for efficiency
    print("=" * 80)
    print("COMPUTING GEOMETRIC ANGLES")
    print("=" * 80)
    kp_names = bout_dict['info'].get('kp_names', [])
    bout_keys = sorted([k for k in bout_dict.keys() if k != 'info'])
    n_computed = 0

    # Collect marker data from all bouts
    all_marker_data = []
    bout_lengths = []
    valid_bout_keys = []
    for bout_key in bout_keys:
        bout = bout_dict[bout_key]
        marker_data = bout.get('marker_sites', bout.get('kp_data'))
        if marker_data is not None and len(kp_names) > 0:
            md = np.asarray(marker_data)
            all_marker_data.append(md)
            bout_lengths.append(md.shape[0])
            valid_bout_keys.append(bout_key)

    if all_marker_data:
        # Concatenate along time axis and compute angles once
        concat_markers = np.concatenate(all_marker_data, axis=0)
        geo_angles_all = compute_geometric_angles_all_legs(concat_markers, kp_names)

        if geo_angles_all:
            # Split results back into per-bout dicts
            for bout_key, length in zip(valid_bout_keys, bout_lengths):
                bout_angles = {}
                for leg_name, leg_angles in geo_angles_all.items():
                    bout_leg = {}
                    for angle_name, angle_arr in leg_angles.items():
                        bout_leg[angle_name] = angle_arr[:length]
                    bout_angles[leg_name] = bout_leg
                bout_dict[bout_key]['geometric_angles'] = bout_angles
                n_computed += 1
                # Trim arrays for next bout
                geo_angles_all = {
                    leg_name: {
                        angle_name: angle_arr[length:]
                        for angle_name, angle_arr in leg_angles.items()
                    }
                    for leg_name, leg_angles in geo_angles_all.items()
                }

    print(f"  Computed geometric angles for {n_computed}/{len(bout_keys)} bouts")
    if n_computed > 0:
        example = bout_dict[bout_keys[0]].get('geometric_angles', {})
        print(f"  Legs computed: {list(example.keys())}")
        if example:
            first_leg = list(example.keys())[0]
            print(f"  Angles for {first_leg}: {list(example[first_leg].keys())}")
    print()

    # Step 7: Save output
    print("=" * 80)
    print("SAVING OUTPUT")
    print("=" * 80)
    print(f"Saving to: {output_path}")
    ioh5.save(output_path, bout_dict)
    print(f"✓ Saved successfully")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Print summary of data keys
    if cfg.postprocessing.verbose and bout_dict:
        example_bout = bout_dict[sorted([k for k in bout_dict.keys() if k != 'info'])[0]]
        print(f"\n  Data keys per bout: {list(example_bout.keys())}")
    print()
    
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    # Get final clip lengths
    final_clip_lengths = bout_dict['info'].get('clip_lengths', clip_lengths)
    original_clip_lengths = bout_dict['info'].get('clip_lengths_original', clip_lengths)
    
    print(f"✓ Processed {len(final_clip_lengths)} bouts")
    
    if 'clip_lengths_original' in bout_dict['info']:
        print(f"  Original frames: {sum(original_clip_lengths)}")
        print(f"  Final frames: {sum(final_clip_lengths)}")
        upsampling = sum(final_clip_lengths) / sum(original_clip_lengths)
        print(f"  Upsampling: {upsampling:.2f}x")
    
    print(f"✓ Output saved to: {output_path}")
    cfg_temp = cfg.copy()
    cfg_temp.paths = convert_dict_to_string(cfg_temp.paths)
    OmegaConf.save(cfg_temp, cfg.paths.log_dir / "postprocess_config.yaml")
    print()


if __name__ == "__main__":
    main()
