import os

import numpy as np
import mujoco
import pytest
from hydra import initialize_config_dir, compose

from jarvis_jax.tracking.scale import (
    DEFAULT_TRUNK_KEYPOINTS, compute_trunk_scale)

CFG_DIR = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"


def _cfg():
    os.environ.setdefault("USER", "eabe")
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        return compose(config_name="courtship_pipeline")


def _tracking_site_positions(model_xml):
    """name (without 'tracking[]') -> rest-pose world site position."""
    mj = mujoco.MjModel.from_xml_path(model_xml)
    d = mujoco.MjData(mj)
    mujoco.mj_forward(mj, d)
    out = {}
    for i in range(mj.nsite):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SITE, i)
        if name and name.startswith("tracking[") and name.endswith("]"):
            out[name[len("tracking["):-1]] = np.array(d.site_xpos[i])
    return out


@pytest.mark.parametrize("estimator", ["norm_ratio", "umeyama"])
def test_compute_trunk_scale_recovers_known_scale(estimator):
    cfg = _cfg()
    model_xml = str(cfg.silhouette.xml)
    kp_names = list(cfg.model.KP_NAMES)
    T, K = 20, len(kp_names)
    factor = 50.0

    rest = _tracking_site_positions(model_xml)
    kp3d = np.zeros((T, K, 3), dtype=np.float64)
    for name in DEFAULT_TRUNK_KEYPOINTS:
        idx = kp_names.index(name)
        kp3d[:, idx, :] = rest[name] / factor

    scale = compute_trunk_scale(kp3d, kp_names, model_xml, estimator=estimator)
    assert abs(scale - factor) / factor < 0.05


def test_compute_trunk_scale_too_few_markers_raises():
    cfg = _cfg()
    model_xml = str(cfg.silhouette.xml)
    # Only 2 of the 5 default trunk markers present -> must raise.
    kp_names = ["Scutellum", "WingL_base", "foo", "bar"]
    kp3d = np.zeros((5, len(kp_names), 3), dtype=np.float64)
    with pytest.raises(ValueError):
        compute_trunk_scale(kp3d, kp_names, model_xml)
