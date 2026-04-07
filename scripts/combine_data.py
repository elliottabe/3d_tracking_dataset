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
from utils.path_utils import load_config_with_path_template, convert_dict_to_path, convert_dict_to_string
from utils.io import load_stac_data
from utils.mjx_preprocess import process_clip, ReferenceClip
from utils.stac_data_utils import concatenate_bout_dicts



@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main postprocessing pipeline using Hydra configuration."""
    
    # Convert path strings to Path objects
    cfg.paths = convert_dict_to_path(cfg.paths)

    # Create anatomy-specific output directory
    anatomy_save_dir = cfg.paths.save_dir / cfg.anatomy.name
    anatomy_save_dir.mkdir(parents=True, exist_ok=True)
    anatomy_log_dir = anatomy_save_dir / "logs"
    anatomy_log_dir.mkdir(parents=True, exist_ok=True)

    # Resolve paths
    base_dir = cfg.paths.data_dir.parent

    # Get concatenation config (with defaults)
    input_pattern = cfg.dataset.get('concat', {}).get('input_pattern', 'ik_output_*')
    output_file = anatomy_save_dir / cfg.dataset.get('concat', {}).get('output_file', 'ik_output_combined')
    output_file_interp = anatomy_save_dir / cfg.dataset.get('concat', {}).get('output_file_interpolated',
                                                           output_file.name.replace('.h5', '_interpolated.h5'))
    enable_jax = cfg.dataset.get('concat', {}).get('enable_jax', True)

    print(f"Data directory: {base_dir}")
    print(f"Input pattern: {input_pattern}.h5")
    print(f"Output file: {output_file}.h5")
    print()

    # Find all matching files
    file_paths = [fp for fp in sorted(base_dir.rglob(f"{input_pattern}")) if 'combined' not in fp.name]
    # file_paths = [Path('/data2/users/eabe/datasets/Johnson_lab/free_walking/Predictions_3D_20260202-171900/ik_output_v2_muscles_stationary_free.h5')]


    print(f"Found {len(file_paths)} files:")
    for i, fp in enumerate(file_paths):
        print(f"  [{i+1}] {fp.name}")
    print()

    # Concatenate
    combined_dict = concatenate_bout_dicts(
        file_paths=[str(fp) for fp in file_paths],
        enable_jax=enable_jax,
        verbose=True
    )

    # Print structure
    # print_bout_dict_structure(combined_dict, show_values=False, max_depth=2)

    # Create non-interpolated version by renaming _stac keys to clean keys
    print("\n" + "=" * 80)
    print("CREATING NON-INTERPOLATED VERSION")
    print("=" * 80)
    
    # Get clip_lengths_original for unpadding
    clip_lengths_original = combined_dict.get('info', {}).get('clip_lengths_original', [])
    bout_keys = sorted([k for k in combined_dict.keys() if k != 'info'])
    
    if clip_lengths_original is not None and len(clip_lengths_original) > 0:
        print(f"Unpadding {len(bout_keys)} bouts to original lengths...")
    
    non_interp_dict = {}
    for bout_idx, bout_key in enumerate(bout_keys):
        bout_data = combined_dict[bout_key]
        non_interp_dict[bout_key] = {}
        
        # Get original length for this bout (for unpadding)
        orig_length = clip_lengths_original[bout_idx] if bout_idx < len(clip_lengths_original) else None
        
        for key, value in bout_data.items():
            if key.endswith('_stac'):
                # Rename _stac keys to clean keys (already at original length)
                clean_key = key[:-5]  # Remove '_stac' suffix
                non_interp_dict[bout_key][clean_key] = value
            elif not any(f"{key}_stac" in bout_data for k in [key]):
                # For keys without _stac counterpart, check if they need unpadding
                if not any(f"{k}_stac" == key for k in bout_data.keys()):
                    # If this is a trajectory array and we have original length, unpad it
                    if orig_length is not None and isinstance(value, (np.ndarray, jnp.ndarray)) and len(value.shape) > 0 and value.shape[0] > orig_length:
                        # Unpad by taking only the first orig_length frames
                        if enable_jax and isinstance(value, jnp.ndarray):
                            non_interp_dict[bout_key][key] = np.array(value[:orig_length])
                        else:
                            non_interp_dict[bout_key][key] = value[:orig_length]
                    else:
                        non_interp_dict[bout_key][key] = value
    
    # Handle info specially - only include clip_lengths_original as clip_lengths
    # Important: preserve fly_ids
    non_interp_dict['info'] = {}
    for key, value in combined_dict['info'].items():
        if key == 'clip_lengths_original':
            non_interp_dict['info']['clip_lengths'] = value
        elif key not in ['clip_lengths', 'clip_lengths_interp_unpadded']:
            non_interp_dict['info'][key] = value
    
    # Save non-interpolated version (base file)
    print(f"Saving non-interpolated data to: {output_file}")
    if 'info' in non_interp_dict and 'clip_lengths' in non_interp_dict['info']:
        print(f"  Clip lengths (original, unpadded): {non_interp_dict['info']['clip_lengths']}")
    ioh5.save(output_file, non_interp_dict)
    print(f"✓ Saved successfully")
    print(f"  File size: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")
    print()

    # Save interpolated version (clean keys contain interpolated data)
    print("=" * 80)
    print("SAVING COMBINED OUTPUT - INTERPOLATED")
    print("=" * 80)
    
    # Clean up interpolated dict to remove clip_lengths_original
    # and optionally unpad to clip_lengths_interp_unpadded
    clip_lengths_interp_unpadded = combined_dict.get('info', {}).get('clip_lengths_interp_unpadded', [])
    
    if clip_lengths_interp_unpadded is not None and len(clip_lengths_interp_unpadded) > 0:
        print(f"Unpadding {len(bout_keys)} bouts to interpolated unpadded lengths...")
    
    interp_dict = {}
    for bout_idx, bout_key in enumerate(bout_keys):
        bout_data = combined_dict[bout_key]
        interp_dict[bout_key] = {}
        
        # Get unpadded length for this bout (if available)
        unpadded_length = clip_lengths_interp_unpadded[bout_idx] if bout_idx < len(clip_lengths_interp_unpadded) else None
        
        for key, value in bout_data.items():
            # Skip _stac keys in interpolated version
            if key.endswith('_stac'):
                continue
                
            # Unpad trajectory arrays to their unpadded interpolated length
            if unpadded_length is not None and isinstance(value, (np.ndarray, jnp.ndarray)) and len(value.shape) > 0 and value.shape[0] > unpadded_length:
                if enable_jax and isinstance(value, jnp.ndarray):
                    interp_dict[bout_key][key] = np.array(value[:unpadded_length])
                else:
                    interp_dict[bout_key][key] = value[:unpadded_length]
            else:
                interp_dict[bout_key][key] = value
    
    # Handle info specially - keep clip_lengths_interp_unpadded as clip_lengths, remove original
    # Important: preserve fly_ids
    interp_dict['info'] = {}
    for key, value in combined_dict['info'].items():
        if key == 'clip_lengths_interp_unpadded':
            interp_dict['info']['clip_lengths'] = value
        elif key not in ['clip_lengths_original', 'clip_lengths']:
            interp_dict['info'][key] = value
    
    print(f"Saving interpolated data to: {output_file_interp}")
    if 'info' in interp_dict:
        if 'clip_lengths' in interp_dict['info']:
            print(f"  Clip lengths (interpolated, unpadded): {interp_dict['info']['clip_lengths']}")
    ioh5.save(output_file_interp, interp_dict)
    print(f"✓ Saved successfully")
    print(f"  File size: {Path(output_file_interp).stat().st_size / 1024 / 1024:.2f} MB")
    print()

    cfg_temp = cfg.copy()
    cfg_temp.paths = convert_dict_to_string(cfg_temp.paths)
    OmegaConf.save(cfg_temp, anatomy_log_dir / "combined_config.yaml")
    # print(OmegaConf.to_yaml(cfg_temp, resolve=True))
    print("=" * 80)
    print("CONCATENATION PIPELINE COMPLETE")
    print("=" * 80)



if __name__ == "__main__":
    main()
