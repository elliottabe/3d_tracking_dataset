"""STAC articulated fit for courtship bouts (fit offsets once, ik_only per bout).

Wraps stac_mjx.run_stac. Offsets are fit ONCE on a high-confidence keypoint
sample and shared; each bout then runs ik_only reusing those offsets. Output h5
is the schema build_solver_inputs consumes (qpos, kp_data, offsets, kp_names,
names_qpos, config).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import stac_mjx


def _flat_scaled(kp3d, scale):
    """(T,K,3) mm -> (T,K*3) scaled by MOCAP_SCALE_FACTOR (NaNs preserved)."""
    T, K, _ = kp3d.shape
    return (np.asarray(kp3d, np.float32).reshape(T, K * 3) * float(scale))


def fit_offsets_once(cfg, kp3d_sample, kp_names, *, offsets_path, save_path, scale: float = 1.0):
    """Fit marker offsets on a sample (skip ik). Writes <save_path>/<offsets_path>."""
    kp3d_sample = np.asarray(kp3d_sample) * float(scale)
    save_path = Path(save_path)
    cfg.stac.fit_offsets_path = offsets_path
    cfg.stac.skip_fit_offsets = False
    cfg.stac.skip_ik_only = 1
    T = kp3d_sample.shape[0]
    cfg.stac.n_fit_frames = min(int(cfg.stac.get("n_fit_frames", T)), T)
    cfg.stac.n_frames_per_clip = T
    kp_flat = _flat_scaled(kp3d_sample, cfg.model["MOCAP_SCALE_FACTOR"])
    stac_mjx.run_stac(cfg, kp_flat, list(kp_names), save_path=save_path)
    return os.path.join(str(save_path), offsets_path)


def ik_only_bout(cfg, kp3d, kp_names, *, offsets_path, out_h5, save_path, scale: float = 1.0):
    """ik_only for one bout reusing fitted offsets. Writes <save_path>/<out_h5>."""
    kp3d = np.asarray(kp3d) * float(scale)
    save_path = Path(save_path)
    cfg.stac.fit_offsets_path = offsets_path      # run_stac reloads offsets from here
    cfg.stac.ik_only_path = out_h5
    cfg.stac.skip_fit_offsets = 1
    cfg.stac.skip_ik_only = 0
    cfg.stac.n_frames_per_clip = int(kp3d.shape[0])
    kp_flat = _flat_scaled(kp3d, cfg.model["MOCAP_SCALE_FACTOR"])
    stac_mjx.run_stac(cfg, kp_flat, list(kp_names), save_path=save_path)
    return os.path.join(str(save_path), out_h5)
