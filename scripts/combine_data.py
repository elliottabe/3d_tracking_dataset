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

    # Resolve paths — prefer a CLI-supplied base_dir so the combine step only
    # aggregates predictions from the current run's base dir rather than every
    # Predictions_3D_* under the dataset-level parent.
    override = cfg.get('base_dir', None)
    if override:
        base_dir = Path(override)
        print(f"Using base_dir override: {base_dir}")
    else:
        base_dir = cfg.paths.data_dir.parent
        print(f"Using default base_dir from config: {base_dir}")

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
    full_dict = concatenate_bout_dicts(
        file_paths=[str(fp) for fp in file_paths],
        enable_jax=enable_jax,
        verbose=True
    )

    # Partition by validity bucket if present (multianimal pipeline emits a
    # parallel `info['bucket']` list of fly0_only / fly1_only / both tags).
    # For single-animal datasets there is no bucket info, so we keep a single
    # combined output named `ik_output_combined.h5` (legacy behaviour).
    bucket_list = []
    info_buckets = full_dict.get('info', {}).get('bucket', None)
    if info_buckets is not None:
        bucket_list = list(np.asarray(info_buckets).tolist()) \
            if not isinstance(info_buckets, list) else list(info_buckets)

    def _filter_dict_by_bucket(src: dict, keep_indices: list[int]) -> dict:
        """Build a sub combined_dict containing only bouts at the given indices,
        renumbered sequentially. Concatenated info fields are sliced to match."""
        all_keys = sorted([k for k in src.keys() if k != 'info'])
        out = {}
        for new_idx, old_idx in enumerate(keep_indices):
            new_key = f'bout_{new_idx:03d}'
            out[new_key] = src[all_keys[old_idx]]
        out['info'] = {}
        concat_fields = {'clip_lengths', 'clip_lengths_original',
                         'clip_lengths_interp_unpadded', 'fly_ids',
                         'source_flies', 'start_frames', 'end_frames',
                         'bucket'}
        for k, v in src.get('info', {}).items():
            if k in concat_fields:
                try:
                    arr = np.asarray(v).tolist() if not isinstance(v, list) else v
                    out['info'][k] = [arr[i] for i in keep_indices]
                except Exception:
                    out['info'][k] = v
            else:
                out['info'][k] = v
        return out

    if bucket_list:
        present = sorted(set(bucket_list), key=lambda x: ['fly0_only', 'both', 'fly1_only'].index(x)
                         if x in ('fly0_only', 'both', 'fly1_only') else 99)
        partitions = []
        for bucket in present:
            keep = [i for i, b in enumerate(bucket_list) if b == bucket]
            if not keep:
                continue
            sub = _filter_dict_by_bucket(full_dict, keep)
            partitions.append((bucket, sub))
        if not partitions:
            print("⚠ No bouts in any bucket — nothing to write")
            return
        print(f"\nPartitioned into {len(partitions)} validity bucket(s): "
              f"{[b for b, _ in partitions]}")
    else:
        partitions = [(None, full_dict)]

    def _save_one(combined_dict: dict, out_base: Path, out_interp: Path):
        # Create non-interpolated version by renaming _stac keys to clean keys
        print("\n" + "=" * 80)
        print(f"CREATING NON-INTERPOLATED VERSION → {out_base.name}")
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

            orig_length = clip_lengths_original[bout_idx] if bout_idx < len(clip_lengths_original) else None

            for key, value in bout_data.items():
                if key.endswith('_stac'):
                    clean_key = key[:-5]
                    non_interp_dict[bout_key][clean_key] = value
                elif not any(f"{k}_stac" == key for k in bout_data.keys()):
                    if orig_length is not None and isinstance(value, (np.ndarray, jnp.ndarray)) and len(value.shape) > 0 and value.shape[0] > orig_length:
                        if enable_jax and isinstance(value, jnp.ndarray):
                            non_interp_dict[bout_key][key] = np.array(value[:orig_length])
                        else:
                            non_interp_dict[bout_key][key] = value[:orig_length]
                    else:
                        non_interp_dict[bout_key][key] = value

        non_interp_dict['info'] = {}
        for key, value in combined_dict['info'].items():
            if key == 'clip_lengths_original':
                non_interp_dict['info']['clip_lengths'] = value
            elif key not in ['clip_lengths', 'clip_lengths_interp_unpadded']:
                non_interp_dict['info'][key] = value

        print(f"Saving non-interpolated data to: {out_base}")
        if 'info' in non_interp_dict and 'clip_lengths' in non_interp_dict['info']:
            print(f"  Clip lengths (original, unpadded): {non_interp_dict['info']['clip_lengths']}")
        ioh5.save(out_base, non_interp_dict)
        print(f"✓ Saved successfully")
        print(f"  File size: {Path(out_base).stat().st_size / 1024 / 1024:.2f} MB")
        print()

        print("=" * 80)
        print(f"SAVING COMBINED OUTPUT - INTERPOLATED → {out_interp.name}")
        print("=" * 80)

        clip_lengths_interp_unpadded = combined_dict.get('info', {}).get('clip_lengths_interp_unpadded', [])

        if clip_lengths_interp_unpadded is not None and len(clip_lengths_interp_unpadded) > 0:
            print(f"Unpadding {len(bout_keys)} bouts to interpolated unpadded lengths...")

        interp_dict = {}
        for bout_idx, bout_key in enumerate(bout_keys):
            bout_data = combined_dict[bout_key]
            interp_dict[bout_key] = {}

            unpadded_length = clip_lengths_interp_unpadded[bout_idx] if bout_idx < len(clip_lengths_interp_unpadded) else None

            for key, value in bout_data.items():
                if key.endswith('_stac'):
                    continue

                if unpadded_length is not None and isinstance(value, (np.ndarray, jnp.ndarray)) and len(value.shape) > 0 and value.shape[0] > unpadded_length:
                    if enable_jax and isinstance(value, jnp.ndarray):
                        interp_dict[bout_key][key] = np.array(value[:unpadded_length])
                    else:
                        interp_dict[bout_key][key] = value[:unpadded_length]
                else:
                    interp_dict[bout_key][key] = value

        interp_dict['info'] = {}
        for key, value in combined_dict['info'].items():
            if key == 'clip_lengths_interp_unpadded':
                interp_dict['info']['clip_lengths'] = value
            elif key not in ['clip_lengths_original', 'clip_lengths']:
                interp_dict['info'][key] = value

        print(f"Saving interpolated data to: {out_interp}")
        if 'info' in interp_dict and 'clip_lengths' in interp_dict['info']:
            print(f"  Clip lengths (interpolated, unpadded): {interp_dict['info']['clip_lengths']}")
        ioh5.save(out_interp, interp_dict)
        print(f"✓ Saved successfully")
        print(f"  File size: {Path(out_interp).stat().st_size / 1024 / 1024:.2f} MB")
        print()

    # Run save for each partition (one path for single-animal, three for multianimal)
    for bucket, sub in partitions:
        if bucket is None:
            base_out = output_file
            interp_out = output_file_interp
        else:
            stem = output_file.name[:-3] if output_file.name.endswith('.h5') else output_file.name
            base_out = output_file.with_name(f"{stem}_{bucket}.h5")
            interp_out = output_file.with_name(f"{stem}_{bucket}_interpolated.h5")
        _save_one(sub, base_out, interp_out)

    cfg_temp = cfg.copy()
    cfg_temp.paths = convert_dict_to_string(cfg_temp.paths)
    OmegaConf.save(cfg_temp, anatomy_log_dir / "combined_config.yaml")
    # print(OmegaConf.to_yaml(cfg_temp, resolve=True))
    print("=" * 80)
    print("CONCATENATION PIPELINE COMPLETE")
    print("=" * 80)



if __name__ == "__main__":
    main()
