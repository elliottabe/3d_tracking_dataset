import numpy as np
from omegaconf import OmegaConf
import jarvis_jax.cse.courtship_stac as cst


def _cfg():
    return OmegaConf.create({"model": {"MOCAP_SCALE_FACTOR": 1000.0, "MJCF_PATH": "x.xml"},
                             "stac": {"fit_offsets_path": "off.h5", "ik_only_path": "ik.h5",
                                      "skip_fit_offsets": False, "skip_ik_only": 0,
                                      "n_fit_frames": 5, "n_frames_per_clip": 5}})


def test_fit_offsets_once_sets_skip_ik(monkeypatch, tmp_path):
    seen = {}
    def fake_run_stac(cfg, kp_flat, kp_names, base_path=None, save_path=None):
        _ = save_path / cfg.stac.fit_offsets_path  # simulate real run_stac's Path use; crashes if save_path is a str
        seen.update(skip_ik=int(cfg.stac.skip_ik_only), skip_fit=int(cfg.stac.skip_fit_offsets),
                    shape=kp_flat.shape); return (str(tmp_path / "off.h5"), None)
    monkeypatch.setattr(cst.stac_mjx, "run_stac", fake_run_stac)
    cst.fit_offsets_once(_cfg(), np.zeros((5, 3, 3), np.float32), ["a", "b", "c"],
                         offsets_path="off.h5", save_path=str(tmp_path))
    assert seen["skip_ik"] == 1 and seen["skip_fit"] == 0 and seen["shape"] == (5, 9)


def test_ik_only_bout_reuses_offsets(monkeypatch, tmp_path):
    seen = {}
    def fake_run_stac(cfg, kp_flat, kp_names, base_path=None, save_path=None):
        _ = save_path / cfg.stac.ik_only_path  # simulate real run_stac's Path use; crashes if save_path is a str
        seen.update(skip_ik=int(cfg.stac.skip_ik_only), skip_fit=int(cfg.stac.skip_fit_offsets),
                    nfpc=int(cfg.stac.n_frames_per_clip), ik=cfg.stac.ik_only_path)
        return (str(tmp_path / "off.h5"), str(tmp_path / "ik.h5"))
    monkeypatch.setattr(cst.stac_mjx, "run_stac", fake_run_stac)
    cst.ik_only_bout(_cfg(), np.zeros((7, 3, 3), np.float32), ["a", "b", "c"],
                     offsets_path="off.h5", out_h5="bout1_ik.h5", save_path=str(tmp_path))
    assert seen["skip_fit"] == 1 and seen["skip_ik"] == 0 and seen["nfpc"] == 7
    assert seen["ik"] == "bout1_ik.h5"


def test_ik_only_bout_applies_scale_before_flattening(monkeypatch, tmp_path):
    seen = {}
    def fake_run_stac(cfg, kp_flat, kp_names, base_path=None, save_path=None):
        seen["kp_flat"] = kp_flat
        return (str(tmp_path / "off.h5"), str(tmp_path / "ik.h5"))
    monkeypatch.setattr(cst.stac_mjx, "run_stac", fake_run_stac)
    rng = np.random.default_rng(0)
    kp3d = rng.normal(size=(7, 3, 3)).astype(np.float32)
    scale = 0.0198
    cst.ik_only_bout(_cfg(), kp3d, ["a", "b", "c"],
                     offsets_path="off.h5", out_h5="bout1_ik.h5", save_path=str(tmp_path),
                     scale=scale)
    expected = cst._flat_scaled(kp3d * scale, _cfg().model["MOCAP_SCALE_FACTOR"])
    np.testing.assert_allclose(seen["kp_flat"], expected)


def test_fit_offsets_once_applies_scale_before_flattening(monkeypatch, tmp_path):
    seen = {}
    def fake_run_stac(cfg, kp_flat, kp_names, base_path=None, save_path=None):
        seen["kp_flat"] = kp_flat
        return (str(tmp_path / "off.h5"), None)
    monkeypatch.setattr(cst.stac_mjx, "run_stac", fake_run_stac)
    rng = np.random.default_rng(1)
    kp3d = rng.normal(size=(5, 3, 3)).astype(np.float32)
    scale = 0.0198
    cst.fit_offsets_once(_cfg(), kp3d, ["a", "b", "c"],
                         offsets_path="off.h5", save_path=str(tmp_path), scale=scale)
    expected = cst._flat_scaled(kp3d * scale, _cfg().model["MOCAP_SCALE_FACTOR"])
    np.testing.assert_allclose(seen["kp_flat"], expected)
