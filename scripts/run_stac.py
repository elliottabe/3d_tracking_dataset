#!/usr/bin/env python3
"""
Run the STAC IK solver from the MAIN repo's single config tree.

This replaces stac-mjx/run_stac_fly_model.py: it composes the unified
`configs/` tree (anatomy.model + the `stac` config group) and calls stac-mjx as
a LIBRARY (stac_mjx.run_stac / viz_stac) — no duplicated config tree, no
cwd=stac-mjx subprocess.

Usage:
    python scripts/run_stac.py paths=hyak dataset=amputation anatomy=v1
    python scripts/run_stac.py paths=hyak dataset=free_walking \
        paths.data_dir=/path/to/Predictions_3D_XXXX
"""

import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
try:
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
except AttributeError:
    pass

import sys
from pathlib import Path
from jax import numpy as jp
import hydra
from omegaconf import DictConfig, OmegaConf

# Make both the main repo and the stac-mjx submodule importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "stac-mjx"))

import stac_mjx
from stac_mjx.keypoint_prune import prune_model_to_available
from utils import io_dict_to_hdf5 as ioh5
from utils.path_utils import convert_dict_to_path, register_custom_resolvers

register_custom_resolvers()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("=" * 80)
    print("STAC IK PIPELINE (library call, unified configs)")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg.stac))

    cfg.paths = convert_dict_to_path(cfg.paths)
    stac_cfg = cfg.stac
    model_cfg = cfg.model
    data_path = Path(stac_cfg.data_path)
    save_path = Path(stac_cfg.save_path)
    base_path = Path(stac_cfg.xml_dir)
    print(f"  Data: {data_path}\n  Save: {save_path}\n  XML dir: {base_path}\n")

    # --- Load keypoint data (single-bout or multi-bout h5) ---
    bout_dict = ioh5.load(data_path)
    if 'kp_names' in bout_dict:
        sorted_kp_names = bout_dict['kp_names']
        kp_data = bout_dict['keypoints'].reshape(bout_dict['keypoints'].shape[0], -1)
        kp_data = kp_data * model_cfg['MOCAP_SCALE_FACTOR']
        print(f"Loaded single bout: {kp_data.shape[0]} frames, {len(sorted_kp_names)} keypoints")
    elif any(k.startswith('bout_') for k in bout_dict.keys()):
        bout_keys = [k for k in bout_dict.keys() if k.startswith('bout_')]
        print(f"Detected multi-bout format with {len(bout_keys)} bouts")
        first_key = bout_keys[0]
        info_dict = bout_dict.get('info', {}) if isinstance(bout_dict.get('info', {}), dict) else {}
        if 'kp_names' in info_dict and len(info_dict['kp_names']) > 0:
            sorted_kp_names = info_dict['kp_names']
        elif 'kp_names' in bout_dict and not str(first_key).startswith('bout_'):
            sorted_kp_names = bout_dict['kp_names']
        elif 'kp_names' in bout_dict[first_key]:
            sorted_kp_names = bout_dict[first_key]['kp_names']
        else:
            raise KeyError("Could not locate 'kp_names' in multi-bout file.")

        clips = []
        max_length = 0
        for key in sorted(bout_keys):
            bout_data = bout_dict[key]['keypoints']
            bout_data = bout_data.reshape(bout_data.shape[0], -1)
            clips.append(bout_data)
            max_length = max(max_length, bout_data.shape[0])
            print(f"  {key}: {bout_data.shape[0]} frames")

        if stac_cfg.enable_padding:
            print(f"\nPadding all clips to {max_length} frames")
            cfg.stac.n_frames_per_clip = max_length
            padded = []
            for bout_data in clips:
                if bout_data.shape[0] < max_length:
                    last = bout_data[-1:, :]
                    pad = jp.repeat(last, max_length - bout_data.shape[0], axis=0)
                    bout_data = jp.concatenate([bout_data, pad], axis=0)
                padded.append(bout_data)
            clips = padded

        kp_data = jp.concatenate(clips, axis=0)
        kp_data = kp_data * model_cfg['MOCAP_SCALE_FACTOR']
        print(f"\nConcatenated data: {kp_data.shape[0]} total frames")
    else:
        raise ValueError("Unknown data format: expected 'kp_names' or 'bout_XXX' keys")

    print(f"Keypoint data shape: {kp_data.shape}")
    print(f"Frames per clip: {cfg.stac.n_frames_per_clip}\n")

    # Auto-detect missing keypoints (e.g. amputated leg): prune model config +
    # reorder data columns to match. No-op when every model keypoint is present.
    kp_data, sorted_kp_names = prune_model_to_available(cfg, sorted_kp_names, kp_data)

    # Subject-specific per-segment calibration scales (from preprocessing) ->
    # cfg.model.SEGMENT_SCALES, applied as a model morph in Stac._create_body_sites.
    info = bout_dict.get('info', {}) if isinstance(bout_dict.get('info', {}), dict) else {}
    seg_scales = info.get('segment_scales', None)
    if seg_scales:
        seg = {str(k): {'geom_body': str(v['geom_body']),
                        'length_body': str(v['length_body']),
                        'scale': float(v['scale'])} for k, v in seg_scales.items()}
        OmegaConf.set_struct(cfg, False)
        cfg.model.SEGMENT_SCALES = seg
        OmegaConf.set_struct(cfg, True)
        print(f"[calibration] {len(seg)} segment scales loaded from preprocessing")

    # --- Run STAC IK solver (library) ---
    print("=" * 80, "\nRUNNING STAC IK SOLVER\n", "=" * 80)
    fit_path, transform_path = stac_mjx.run_stac(
        cfg, kp_data, sorted_kp_names, base_path=base_path, save_path=save_path
    )
    print(f"\n✓ STAC IK complete!\n  Fit: {fit_path}\n  Transform: {transform_path}")

    # --- Render visualization video ---
    data_path = save_path / cfg.stac["ik_only_path"]
    n_frames = min(1000, kp_data.shape[0])
    video_dir = data_path.parent / f"{cfg.dataset.name}_{cfg.anatomy.name}.mp4"
    stac_mjx.viz_stac(
        data_path, cfg, n_frames, video_dir,
        start_frame=0, camera='track1', height=544, width=832, base_path=base_path,
    )
    print(f"✓ Video saved to: {video_dir}\nDONE!")


if __name__ == "__main__":
    main()
