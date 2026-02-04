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

import sys
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import io_dict_to_hdf5 as ioh5
from utils.stac_data_utils import reorganize_stac_by_bouts, print_bout_dict_structure
from utils.path_utils import load_config_with_path_template, convert_dict_to_path
from utils.io import load_stac_data


def load_clip_lengths(data_path: Path) -> list:
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
    
    preprocessed_path = data_path / 'preprocessed_bout.h5'
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
    clip_lengths = load_clip_lengths(data_path)
    
    # Step 2: Load STAC output
    cfg_d, d, stac_data = load_stac_output(stac_path)
    
    # Step 3: Compute egocentric site positions
    site_pos, site_names = compute_egocentric_site_positions(
        mocap_qpos=d.qpos,
        flybody_path=flybody_path,
        floor_path=floor_path,
        cfg_d=cfg_d,
        verbose=cfg.postprocessing.verbose
    )
    
    # Add to stac_data
    stac_data['egocentric_site_pos'] = site_pos
    stac_data['egocentric_site_names'] = site_names
    
    # Step 4: Reorganize by bouts and save
    bout_dict = reorganize_and_save(
        stac_data=stac_data,
        clip_lengths=clip_lengths,
        output_path=output_path,
        verbose=cfg.postprocessing.verbose
    )
    
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"✓ Processed {len(clip_lengths)} bouts")
    print(f"✓ Output saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
