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

    # Resolve paths
    base_dir = cfg.paths.data_dir.parent

    # Get concatenation config (with defaults)
    input_pattern = cfg.dataset.get('concat', {}).get('input_pattern', 'ik_output_*.h5')
    output_file = cfg.dataset.get('concat', {}).get('output_file', 'ik_output_combined.h5')
    enable_jax = cfg.dataset.get('concat', {}).get('enable_jax', True)

    print(f"Data directory: {base_dir}")
    print(f"Input pattern: {input_pattern}")
    print(f"Output file: {output_file}")
    print()

    # Find all matching files
    file_paths = [fp for fp in sorted(base_dir.rglob(input_pattern)) if 'combined' not in fp.name]


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

    # Save
    output_path = cfg.paths.save_dir / output_file
    print("\n" + "=" * 80)
    print("SAVING COMBINED OUTPUT")
    print("=" * 80)
    print(f"Saving to: {output_path}")
    ioh5.save(output_path, combined_dict)
    print(f"✓ Saved successfully")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    cfg_temp = cfg.copy()
    cfg_temp.paths = convert_dict_to_string(cfg_temp.paths)
    OmegaConf.save(cfg_temp, cfg.paths.log_dir / "run_config.yaml")
    # print(OmegaConf.to_yaml(cfg_temp, resolve=True))
    print("=" * 80)
    print("CONCATENATION PIPELINE COMPLETE")
    print("=" * 80)



if __name__ == "__main__":
    main()
