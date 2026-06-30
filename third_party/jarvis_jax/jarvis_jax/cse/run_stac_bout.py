"""Minimal STAC IK driver for a single CSE bout (GPU; run via sbatch).

Reads a ``preprocessed_bout``-format HDF5 (kp_names + keypoints, model order,
scaled to model cm by cse_labels.build_bout), runs fit_offsets + ik_only with the
stac-mjx v1 config, and writes ``Fruitfly_ik_*.h5`` (with per-frameset qpos).

No visualisation (unlike run_stac_fly_model.py) and paths are explicit, so it is
decoupled from the analysis-pipeline directory templates.

Usage (inside an sbatch GPU job, 3d_tracking env):
    python -m jarvis_jax.cse.run_stac_bout \
        --bout   <work>/<rec>_bout.h5 \
        --out    <work>/<rec>/Fruitfly_ik_v1_cse.h5 \
        --stac-config-dir /.../stac-mjx/configs \
        --overrides paths=hyak anatomy=v1 dataset=free_walking
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def run(bout_h5, out_h5, stac_config_dir, overrides):
    import numpy as np
    import hydra
    from omegaconf import OmegaConf
    import stac_mjx
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from stac_mjx.keypoint_prune import prune_model_to_available
    from stac_mjx.path_utils import convert_dict_to_path, register_custom_resolvers

    register_custom_resolvers()

    with hydra.initialize_config_dir(config_dir=str(stac_config_dir), version_base=None):
        cfg = hydra.compose(
            config_name="config",
            overrides=list(overrides) + [
                "hydra/job_logging=disabled", "hydra/hydra_logging=disabled",
            ],
        )
    cfg.paths = convert_dict_to_path(cfg.paths)

    bout = ioh5.load(bout_h5)
    kp_names = [n.decode() if isinstance(n, bytes) else n for n in bout["kp_names"]]
    kp = np.asarray(bout["keypoints"])                       # (T, K, 3) model cm
    T = kp.shape[0]
    kp_data = kp.reshape(T, -1) * cfg.model["MOCAP_SCALE_FACTOR"]

    # one clip = whole bout; per-frame IK independent of clip layout
    cfg.stac.n_frames_per_clip = T
    cfg.stac.n_fit_frames = min(int(cfg.stac.get("n_fit_frames", T)), T)
    cfg.stac.continuous = False
    cfg.stac.infer_qvels = False
    cfg.stac.skip_fit_offsets = False
    cfg.stac.skip_ik_only = False

    kp_data, kp_names = prune_model_to_available(cfg, kp_names, kp_data)

    base = Path(__file__).resolve()
    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    # run_stac writes cfg.stac.fit_offsets_path / ik_only_path under save_path
    cfg.stac.fit_offsets_path = out_h5.stem + "_fit.h5"
    cfg.stac.ik_only_path = out_h5.name
    fit_p, ik_p = stac_mjx.run_stac(
        cfg, kp_data, kp_names, base_path=Path.cwd(), save_path=out_h5.parent)
    print(f"[run_stac_bout] wrote {ik_p}")
    return ik_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stac-config-dir", required=True)
    ap.add_argument("--overrides", nargs="*", default=["paths=hyak", "anatomy=v1", "dataset=free_walking"])
    a = ap.parse_args()
    run(a.bout, a.out, a.stac_config_dir, a.overrides)


if __name__ == "__main__":
    main()
