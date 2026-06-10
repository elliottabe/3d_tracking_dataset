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
# Persistent compilation cache dir; override with JAX_COMPILATION_CACHE_DIR
# (e.g. point at scratch on clusters where /tmp is node-local/ephemeral).
jax.config.update("jax_compilation_cache_dir",
                  os.environ.get("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_cache"))
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


def _bucketed_run_stac(cfg, kp_data, bout_real_lens, kp_names, base_path, save_path):
    """Length-bucketed IK orchestration (accuracy-identical to the plain path).

    The default pipeline pads every bout to the global max clip length, so a short
    bout is solved with up to ~90% replicated padding frames — wasted compute.
    This groups bouts into length buckets, pads each bout only to its bucket's max,
    and runs ``ik_only`` per bucket by reusing ``stac_mjx.run_stac`` with
    ``skip_fit_offsets`` (offsets are fit once up front and shared). Per-bout
    results are merged back in the original order and re-padded to the global max,
    so the saved h5 is byte-for-byte the same *format* as the non-bucketed run.

    Accuracy: the replicated-frame padding tail is flat under the smoothness cost
    (q[t]==q[t-1] there), so the amount of trailing padding does not change the
    real frames' solution. A round-trip assertion on ``kp_data`` guards the
    bout->output ordering. Guarded by ``cfg.stac.bucketed_ik`` (default off).
    """
    import os as _os
    import numpy as np
    from stac_mjx import io as _sio

    kp_data = np.asarray(kp_data)
    n_bouts = len(bout_real_lens)
    global_max = max(bout_real_lens)

    # Re-split the global-padded array back into per-bout real-frame clips.
    bouts = [kp_data[j * global_max: j * global_max + bout_real_lens[j]]
             for j in range(n_bouts)]

    # Assign bouts to length buckets by their real length.
    edges = sorted(cfg.stac.get('bucket_edges', [1000, 2000, 3500]))
    def _bucket(L):
        for bi, e in enumerate(edges):
            if L <= e:
                return bi
        return len(edges)
    buckets = {}
    for j, L in enumerate(bout_real_lens):
        buckets.setdefault(_bucket(L), []).append(j)
    bucket_max = {bid: max(bout_real_lens[j] for j in js) for bid, js in buckets.items()}
    saved_frames = sum(bucket_max[bid] * len(js) for bid, js in buckets.items())
    print(f"[bucketed-ik] {n_bouts} bouts -> {len(buckets)} buckets (global_max={global_max}); "
          f"bucket maxima {bucket_max}; solve frames {saved_frames} vs {n_bouts * global_max} padded")

    OmegaConf.set_struct(cfg, False)
    base_ik_path = cfg.stac.ik_only_path
    orig = (cfg.stac.skip_fit_offsets, cfg.stac.skip_ik_only, cfg.stac.n_frames_per_clip)
    fit_path = save_path / cfg.stac.fit_offsets_path
    bucket_files = []

    # 1. Fit offsets ONCE (skip ik). fit_offsets() uses the first n_fit_frames of
    #    bout 0; pass the (already scaled+pruned) bout 0 real frames.
    cfg.stac.skip_fit_offsets = False
    cfg.stac.skip_ik_only = 1
    cfg.stac.n_frames_per_clip = int(bout_real_lens[0])
    stac_mjx.run_stac(cfg, bouts[0], kp_names, base_path=base_path, save_path=save_path)

    # 2. ik_only per bucket (clips are uniform within a bucket).
    cfg.stac.skip_fit_offsets = 1
    cfg.stac.skip_ik_only = 0
    per_bout, ref = {}, None
    for bid, js in sorted(buckets.items()):
        bmax = bucket_max[bid]
        padded = []
        for j in js:
            c = bouts[j]
            if c.shape[0] < bmax:
                fr = np.isfinite(c).all(axis=1)
                li = int(np.nonzero(fr)[0][-1]) if fr.any() else c.shape[0] - 1
                c = np.concatenate([c, np.repeat(c[li:li + 1], bmax - c.shape[0], 0)], 0)
            padded.append(c)
        cfg.stac.n_frames_per_clip = int(bmax)
        cfg.stac.ik_only_path = base_ik_path.replace('.h5', f'_bucket{bid}.h5')
        bucket_files.append(save_path / cfg.stac.ik_only_path)
        print(f"[bucketed-ik] bucket {bid}: {len(js)} bouts x {bmax} frames")
        stac_mjx.run_stac(cfg, np.concatenate(padded, 0), kp_names,
                          base_path=base_path, save_path=save_path)
        _, sd = _sio.load_stac_data(str(save_path / cfg.stac.ik_only_path))
        ref = sd
        for k, j in enumerate(js):
            s, L = k * bmax, bout_real_lens[j]
            per_bout[j] = {f: np.asarray(getattr(sd, f))[s:s + L]
                           for f in ('qpos', 'xpos', 'xquat', 'marker_sites', 'kp_data', 'qvel')}

    # 3. Merge in original bout order, re-pad each bout back to global_max.
    def _repad(a):
        return (np.concatenate([a, np.repeat(a[-1:], global_max - a.shape[0], 0)], 0)
                if a.shape[0] < global_max else a)
    fields = ('qpos', 'xpos', 'xquat', 'marker_sites', 'kp_data', 'qvel')
    merged = {f: np.concatenate([_repad(per_bout[j][f]) for j in range(n_bouts)], 0)
              for f in fields}

    # Ordering guard: round-tripped kp_data must equal the input keypoints.
    for j in range(n_bouts):
        L = bout_real_lens[j]
        got = merged['kp_data'][j * global_max: j * global_max + L]
        if not np.allclose(np.nan_to_num(got), np.nan_to_num(bouts[j][:L]), atol=1e-4):
            raise RuntimeError(f"[bucketed-ik] bout {j} kp_data mismatch after merge — ordering bug")

    # 4. Save merged output (format-identical to the non-bucketed run) + cleanup.
    cfg.stac.n_frames_per_clip = int(global_max)
    cfg.stac.ik_only_path = base_ik_path
    _sio.save_data_to_h5(
        config=cfg, file_path=str(save_path / base_ik_path),
        kp_names=ref.kp_names, names_qpos=ref.names_qpos, names_xpos=ref.names_xpos,
        offsets=np.asarray(ref.offsets), **merged,
    )
    (cfg.stac.skip_fit_offsets, cfg.stac.skip_ik_only, cfg.stac.n_frames_per_clip) = orig
    OmegaConf.set_struct(cfg, True)
    for bf in bucket_files:
        try:
            _os.remove(bf)
        except OSError:
            pass
    print(f"[bucketed-ik] merged {n_bouts} bouts -> {merged['qpos'].shape[0]} frames "
          f"saved to {save_path / base_ik_path}")
    return fit_path, save_path / base_ik_path


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
    bout_real_lens = None  # per-bout unpadded lengths (set in multi-bout mode; enables bucketed IK)
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
        bout_real_lens = [int(c.shape[0]) for c in clips]  # before padding

        if stac_cfg.enable_padding:
            print(f"\nPadding all clips to {max_length} frames")
            cfg.stac.n_frames_per_clip = max_length
            padded = []
            for bout_data in clips:
                if bout_data.shape[0] < max_length:
                    # Pad by replicating the last *fully-finite* frame, not just the
                    # last frame. If a clip ends in a NaN run (e.g. a bone-length-
                    # masked, unfillable trailing edge), repeating the NaN last frame
                    # smears NaN across the whole padded clip and freezes the STAC
                    # solve. Fall back to zeros only if no finite frame exists.
                    finite_rows = jp.isfinite(bout_data).all(axis=1)
                    if bool(finite_rows.any()):
                        li = int(jp.nonzero(finite_rows)[0][-1])
                        last = bout_data[li:li + 1, :]
                    else:
                        last = jp.zeros((1, bout_data.shape[1]), bout_data.dtype)
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
                        'scale': float(v['scale']),
                        'scale_sites_on_body': str(v.get('scale_sites_on_body', ''))}
               for k, v in seg_scales.items()}
        OmegaConf.set_struct(cfg, False)
        cfg.model.SEGMENT_SCALES = seg
        OmegaConf.set_struct(cfg, True)
        print(f"[calibration] {len(seg)} segment scales loaded from preprocessing")

    # --- Run STAC IK solver (library) ---
    print("=" * 80, "\nRUNNING STAC IK SOLVER\n", "=" * 80)
    if (cfg.stac.get('bucketed_ik', False) and bout_real_lens is not None
            and len(bout_real_lens) > 1):
        # Length-bucketed IK: pad each bout only to its bucket max (not the global
        # max), cutting the replicated-padding compute. Accuracy-identical on real
        # frames; output is re-padded to global max so the h5 format is unchanged.
        fit_path, transform_path = _bucketed_run_stac(
            cfg, kp_data, bout_real_lens, sorted_kp_names, base_path, save_path
        )
    else:
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
