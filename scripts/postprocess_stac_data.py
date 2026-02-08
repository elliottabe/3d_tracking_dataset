"""
Postprocess STAC output data:
1. Load keypoint data and extract clip_lengths
2. Load STAC IK output
3. Compute egocentric site positions (JAX/MJX vectorized)
4. Reorganize data by bouts
5. Save processed output

Usage:
    python postprocess_stac_data.py paths=workstation dataset=free_walking
    python postprocess_stac_data.py paths=hyak dataset=courtship
"""


import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# Note: jax_persistent_cache_enable_xla_caches may not be available in all JAX versions
try:
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
except AttributeError:
    pass  # Skip if not available in this JAX version

import sys
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
from utils.path_utils import load_config_with_path_template, convert_dict_to_path
from utils.io import load_stac_data
from utils.mjx_preprocess import process_clip, ReferenceClip


def load_clip_lengths(data_path: Path, filename: str) -> list:
    """
    Load clip_lengths from preprocessed keypoint data.
    
    Args:
        data_path: Path to directory containing preprocessed_bout.h5
        
    Returns:
        clip_lengths: List of frame counts for each bout
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
    
    print(f"✓ Found {len(clip_lengths)} bouts")
    print(f"  Clip lengths: {clip_lengths}")
    print(f"  Total frames: {sum(clip_lengths)}")
    print()
    
    return clip_lengths


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


def compute_egocentric_site_positions(
    mocap_qpos,
    flybody_path: Path,
    floor_path: Path,
    cfg_d=None,
    verbose: bool = True
):
    """
    Compute egocentric site positions using JAX/MJX vectorized forward kinematics.
    
    Args:
        mocap_qpos: Joint position trajectory (n_timesteps, n_qpos)
        flybody_path: Path to fruitfly MuJoCo XML
        floor_path: Path to floor MuJoCo XML
        cfg_d: STAC config dictionary (to extract model path if available)
        verbose: Print progress messages
        
    Returns:
        site_pos: Egocentric site positions (n_timesteps, n_sites, 3)
        site_names: List of site names
    """
    print("=" * 80)
    print("COMPUTING EGOCENTRIC SITE POSITIONS")
    print("=" * 80)
    
    # Compile models - use STAC model path if available
    if verbose:
        print("Compiling MuJoCo models...")
    
    # Try to use the model path from STAC config
    if cfg_d and 'model' in cfg_d and 'model_path' in cfg_d.model:
        stac_model_path = Path(cfg_d.model.model_path)
        if stac_model_path.exists():
            if verbose:
                print(f"  Using STAC model: {stac_model_path}")
            flybody_path = stac_model_path
        else:
            if verbose:
                print(f"  STAC model not found, using default: {flybody_path}")
    
    spec = mujoco.MjSpec().from_file(flybody_path.as_posix())
    floor_spec = mujoco.MjSpec().from_file(floor_path.as_posix())
    spawn_frame = floor_spec.worldbody.add_frame(
        pos=[0, 0, -.125],
        quat=[1, 0, 0, 0],
    )
    spawn_body = spawn_frame.attach_body(spec.body("thorax"), "", suffix='_fly')
    
    # Get standard MuJoCo model for extracting site info
    mj_model = floor_spec.compile()
    
    # Validate qpos dimensions match
    expected_nq = mj_model.nq
    actual_nq = mocap_qpos.shape[1]
    if expected_nq != actual_nq:
        raise ValueError(
            f"qpos dimension mismatch!\n"
            f"  Model expects: {expected_nq} DoFs\n"
            f"  STAC data has: {actual_nq} DoFs\n"
            f"  Model file: {flybody_path}\n"
            f"  Hint: Make sure you're using the same model that STAC used for IK.\n"
            f"  Check cfg_d.model.model_path in the STAC output."
        )
    
    # Compile to MJX model for fast batched forward kinematics
    mjx_model = mjx.put_model(mj_model)
    
    # Get site names and indices
    site_names = [site.name for site in floor_spec.sites if 'tracking' in site.name]
    _suffix = '_fly'
    
    # Find thorax body index
    thorax_body_idx = mj_model.body(f"thorax{_suffix}").id
    
    # Find site indices
    site_indices = jnp.array([floor_spec.site(site_name).id for site_name in site_names])
    
    if verbose:
        print(f"✓ Models compiled")
        print(f"  Processing {len(mocap_qpos)} timesteps with {len(site_names)} sites")
        print(f"  Thorax body index: {thorax_body_idx}")
        print(f"  Site indices: {site_indices}")
    
    # Define vectorized forward kinematics function
    def compute_egocentric_sites(qpos):
        """
        Compute egocentric site positions for a single timestep.
        
        Args:
            qpos: Joint positions for one timestep (n_qpos,)
            
        Returns:
            egocentric_positions: Site positions in thorax frame (n_sites, 3)
        """
        # Create mjx data and set qpos
        mjx_data = mjx.make_data(mjx_model)
        mjx_data = mjx_data.replace(qpos=qpos)
        
        # Forward kinematics
        mjx_data = mjx.forward(mjx_model, mjx_data)
        
        # Get thorax position and orientation
        thorax_xpos = mjx_data.xpos[thorax_body_idx]  # (3,)
        thorax_xmat = mjx_data.xmat[thorax_body_idx].reshape(3, 3)  # (3, 3)
        
        # Get site positions (global frame)
        site_xpos = mjx_data.site_xpos[site_indices]  # (n_sites, 3)
        
        # Transform to egocentric (thorax-centered) coordinates
        relative_pos = site_xpos - thorax_xpos[None, :]  # (n_sites, 3)
        egocentric_pos = jnp.dot(relative_pos, thorax_xmat)  # (n_sites, 3)
        
        return egocentric_pos
    
    # Vectorize over time dimension and JIT compile
    compute_egocentric_sites_vmap = jax.vmap(compute_egocentric_sites)
    compute_egocentric_sites_jit = jax.jit(compute_egocentric_sites_vmap)
    
    # Convert qpos to JAX array if not already
    qpos_traj = jnp.asarray(mocap_qpos)
    
    # Warm-up JIT compilation
    if verbose:
        print("\nJIT compiling...")
    start = time.time()
    _ = compute_egocentric_sites_jit(qpos_traj[:2])
    _ = _.block_until_ready()
    compile_time = time.time() - start
    if verbose:
        print(f"✓ Compilation time: {compile_time:.2f}s")
    
    # Run batched computation
    if verbose:
        print("\nComputing egocentric positions...")
    start = time.time()
    site_pos = compute_egocentric_sites_jit(qpos_traj)
    site_pos = site_pos.block_until_ready()  # Wait for GPU computation
    compute_time = time.time() - start
    
    if verbose:
        print(f"✓ Computation time: {compute_time:.2f}s")
        print(f"  Output shape: {site_pos.shape}")
        print(f"  Estimated speedup vs loop: ~{len(qpos_traj) * 0.01 / compute_time:.1f}x")
        print(f"  Sites: {site_names}")
    print()
    
    return site_pos, site_names


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
    percentile: float = 5.0,
    target_z: float = -0.125,
    max_qvel: float = 20.0
) -> Dict[str, jnp.ndarray]:
    """
    Process all bouts in parallel using vmap for floor adjustment and MJX.
    
    Args:
        qpos_batch: Batched qpos (n_bouts, max_T, nq)
        xpos_batch: Batched xpos (n_bouts, max_T, nbodies, 3)
        end_eff_indices: Indices of end effector bodies
        mjx_model: MJX model
        mjx_data: MJX data
        dt: Timestep
        percentile: Ground contact percentile
        target_z: Target floor height
        max_qvel: Max velocity
        
    Returns:
        Dictionary with batched qpos, qvel, xpos, xquat
    """
    # Define single-bout floor adjustment
    def adjust_single_bout(qpos, xpos):
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
    
    # Vmap floor adjustment
    qpos_adjusted = jax.vmap(adjust_single_bout)(qpos_batch, xpos_batch)
    
    # Vmap MJX processing
    qpos_out, qvel_out, xpos_out, xquat_out = jax.vmap(process_single_bout)(qpos_adjusted)
    
    return {
        'qpos': qpos_out,
        'qvel': qvel_out,
        'xpos': xpos_out,
        'xquat': xquat_out
    }


def process_all_bouts(
    bout_dict: Dict,
    cfg: DictConfig,
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    verbose: bool = True
) -> Dict:
    """
    Process all bouts with interpolation, floor adjustment, and MJX forward kinematics.
    
    Pipeline per bout:
    1. Interpolate qpos, xpos, xquat, kp_data (if enabled)
    2. Adjust root z-position for floor contact (if enabled)
    3. Run MJX forward kinematics to compute qvel and updated xpos/xquat (if enabled)
    4. Update bout dict with processed data
    
    Original data gets suffix '_stac', final processed data has clean names.
    
    Args:
        bout_dict: Dictionary with 'info' and bout data
        cfg: Hydra configuration
        mjx_model: Compiled MJX model
        mjx_data: MJX data structure
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
    
    # Get original clip lengths
    clip_lengths_original = [
        bout_dict[key]['qpos'].shape[0] 
        for key in bout_keys
    ]
    
    # Step 1: Rename original STAC data with suffix
    if verbose:
        print("\nRenaming original STAC data with '_stac' suffix...")
    for bout_key in bout_keys:
        bout = bout_dict[bout_key]
        for key in ['qpos', 'xpos', 'xquat']:
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
                stac_key = f'{key}_stac' if key in ['qpos', 'xpos', 'xquat'] else key
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
    
    # Step 3 & 4: Batched floor adjustment and MJX processing (if enabled)
    if (floor_cfg.enabled or mjx_cfg.enabled) and interp_cfg.enabled:
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
        
        # Find end effector indices
        names_xpos = bout_dict['info']['names_xpos']
        end_eff_indices = [
            i for i, name in enumerate(names_xpos)
            if any(eff_name in name for eff_name in floor_cfg.end_effector_names)
        ]
        
        if len(end_eff_indices) == 0:
            print(f"  ⚠ Warning: No end effectors found matching {floor_cfg.end_effector_names}")
            print(f"  Available names: {names_xpos[:10]}...")
        else:
            if verbose:
                print(f"  Found {len(end_eff_indices)} end effectors")
        
        # Compute timestep
        dt = 1.0 / interp_cfg.target_hz if interp_cfg.enabled else 1.0 / interp_cfg.source_hz
        
        if verbose:
            print(f"  JIT compiling batched processing (floor + MJX, dt={dt:.5f}s)...")
        
        # JIT compile the batched processing
        process_batched_jit = jax.jit(
            lambda qpos, xpos: process_bouts_batched(
                qpos, xpos, end_eff_indices, mjx_model, mjx_data,
                dt, floor_cfg.percentile, floor_cfg.target_z, mjx_cfg.max_qvel
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
        
        # Unpack results back into bout_dict
        for bout_idx, bout_key in enumerate(bout_keys):
            bout = bout_dict[bout_key]
            bout['qpos'] = np.array(processed['qpos'][bout_idx])
            bout['qvel'] = np.array(processed['qvel'][bout_idx])
            bout['xpos'] = np.array(processed['xpos'][bout_idx])
            bout['xquat'] = np.array(processed['xquat'][bout_idx])
    
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
    
    print(f"File paths:")
    print(f"  STAC output: {stac_path}")
    print(f"  Preprocessed data: {preprocessed_path}")
    print(f"  Output: {output_path}")
    print()
    
    # Step 1: Load clip lengths
    clip_lengths = load_clip_lengths(data_path, cfg.postprocessing.preprocessed_file)
    
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
    
    if cfg.postprocessing.verbose:
        print()
        print_bout_dict_structure(bout_dict, show_values=False)
    print()
    
    # Step 4: Compile MJX model for processing (if enabled)
    mjx_model = None
    mjx_data = None
    
    if (cfg.postprocessing.interpolation.enabled or 
        cfg.postprocessing.floor_alignment.enabled or
        cfg.postprocessing.mjx_processing.enabled):
        
        print("=" * 80)
        print("COMPILING MJX MODEL")
        print("=" * 80)
        print("Compiling MuJoCo models...")
        
        # Compile models (same pattern as compute_egocentric_site_positions)
        spec = mujoco.MjSpec().from_file(flybody_path.as_posix())
        floor_spec = mujoco.MjSpec().from_file(floor_path.as_posix())
        
        # Use STAC model path if available
        if cfg_d and 'model' in cfg_d and 'model_path' in cfg_d.model:
            stac_model_path = Path(cfg_d.model.model_path)
            if stac_model_path.exists():
                print(f"  Using STAC model: {stac_model_path}")
                spec = mujoco.MjSpec().from_file(stac_model_path.as_posix())
        
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
            verbose=cfg.postprocessing.verbose
        )
    
    # Step 6: Save output
    print("=" * 80)
    print("SAVING OUTPUT")
    print("=" * 80)
    print(f"Saving to: {output_path}")
    ioh5.save(output_path, bout_dict)
    print(f"✓ Saved successfully")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
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
    print()


if __name__ == "__main__":
    main()
