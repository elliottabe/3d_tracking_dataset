import numpy as np
import jax.numpy as jnp
import pytest
from jarvis_jax.tracking.predict_2d import (
    peaks_and_conf, detector_to_model_perm, reorder_detector_to_model)
from jarvis_jax.geometry.center3d import centroids_to_fullpx


def test_peaks_and_conf_locates_peak_and_reports_value():
    # one 224x224 heatmap, K=2, sharp peaks at known locations
    hm = np.full((1, 224, 224, 2), -5.0, np.float32)
    hm[0, 50, 30, 0] = 9.0     # (row=50,col=30) -> in_size=448 => (x=60,y=100)
    hm[0, 100, 200, 1] = 4.0
    kp, conf = peaks_and_conf(jnp.asarray(hm))
    kp = np.asarray(kp); conf = np.asarray(conf)
    assert abs(kp[0, 0, 0] - 60) < 2 and abs(kp[0, 0, 1] - 100) < 2   # x=col*2, y=row*2
    assert conf[0, 0] > conf[0, 1] > 0                                # peak values, relu'd


def test_crop_to_fullframe_mapping():
    # centerHM = crop center in full px; kp at crop center -> full == centerHM
    kp_crop = np.array([[[224.0, 224.0]]])          # (nc=1,K=1,2)
    centerHM = np.array([[900.0, 300.0]])           # (nc=1,2)
    full = np.asarray(centroids_to_fullpx(kp_crop, centerHM, 448))
    assert np.allclose(full[0, 0], [900.0, 300.0])


# --- detector-order -> model-order reorder (the keypoint-order fix) --------------

def test_detector_to_model_perm_maps_by_name():
    det = ["Antenna_Base", "EyeL", "EyeR", "Scutellum"]   # detector/tracking order
    mod = ["Scutellum", "EyeL", "EyeR", "Antenna_Base"]   # model/XML order
    perm = detector_to_model_perm(det, mod)
    # kp_model[k] = kp_detector[perm[k]]  -> perm names the detector index for each model slot
    assert [det[p] for p in perm] == mod
    assert perm == [3, 1, 2, 0]


def test_reorder_moves_channels_to_model_slots():
    det = ["Antenna_Base", "EyeL", "EyeR", "Scutellum"]
    mod = ["Scutellum", "EyeL", "EyeR", "Antenna_Base"]
    # tag each detector channel k with value k so we can trace where it lands
    K = 4
    kp2d = np.tile(np.arange(K, dtype=np.float32)[None, None, :, None], (2, 3, 1, 2))  # (T,C,K,2)
    conf = np.tile(np.arange(K, dtype=np.float32)[None, None, :], (2, 3, 1))            # (T,C,K)
    kp_m, cf_m = reorder_detector_to_model(kp2d, conf, det, mod)
    # model slot k must now hold the detector channel named mod[k]
    for k, name in enumerate(mod):
        assert np.all(kp_m[..., k, :] == det.index(name))
        assert np.all(cf_m[..., k] == det.index(name))


def test_reorder_fails_loud_on_name_mismatch():
    with pytest.raises(ValueError):
        detector_to_model_perm(["A", "B", "C"], ["A", "B", "X"])   # different sets


def test_reorder_passthrough_empty_bout():
    kp2d = np.zeros((5, 7, 0, 2), np.float32); conf = np.zeros((5, 7, 0), np.float32)
    kp_m, cf_m = reorder_detector_to_model(kp2d, conf, ["A", "B"], ["B", "A"])
    assert kp_m.shape == kp2d.shape and cf_m.shape == conf.shape


def test_config_detector_order_reorders_to_correct_anatomy():
    """The real cfg.detector.kp_names (order O) must be the same set as
    cfg.model.KP_NAMES, and the reorder must place the head landmarks (EyeL/EyeR/
    Antenna_Base) and Abd_tip into their XML-order slots. This pins the fix that
    the red_data GT anatomy check verified visually."""
    from hydra import initialize_config_dir, compose
    cfg_dir = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="pipeline", overrides=["paths=hyak"])
    det = list(cfg.detector.kp_names); mod = list(cfg.model.KP_NAMES)
    assert len(det) == 50 and set(det) == set(mod)
    perm = detector_to_model_perm(det, mod)
    # after reorder, model slot for each name pulls the correctly-named detector channel
    for name in ("EyeL", "EyeR", "Antenna_Base", "Abd_tip", "Scutellum", "WingL_base"):
        assert det[perm[mod.index(name)]] == name
    # and the detector's order really is the tracking order (head-first, not XML)
    assert det[:3] == ["Antenna_Base", "EyeL", "EyeR"]
    assert mod[:3] == ["Scutellum", "WingL_base", "WingR_base"]
