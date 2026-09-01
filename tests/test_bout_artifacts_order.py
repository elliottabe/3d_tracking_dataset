"""Pin the two order traps that produced four rounds of wrong wing numbers on
2026-08-31. These are not hypothetical: both were fallen into in one sitting,
and every jitter/confidence metric rated the wrong keypoints as the BEST ones.
"""
import numpy as np
import pytest
from omegaconf import OmegaConf

from viz.core.bout_artifacts import (load_bout_kp, centroids_canonical,
                                     KeypointOrderError, CameraOrderError,
                                     BoutKeypoints)

DET = "configs/detector/vitpose_v3.yaml"
ANA = "configs/anatomy/v1.yaml"


def test_detector_and_model_keypoint_order_really_do_differ():
    """The whole trap rests on this. If these ever become equal the guards are
    harmless, but the test documents WHY they exist."""
    det = list(OmegaConf.load(DET).kp_names)
    mdl = list(OmegaConf.load(ANA).model.KP_NAMES)
    assert set(det) == set(mdl), "same keypoints, different order -- that is the trap"
    assert det != mdl, "orders are equal; the 2026-08-31 bug would be impossible"


def test_the_exact_mislabelling_that_happened():
    """Detector WingR_base/V12/V13 indices land on LEG keypoints in model order.
    This is the specific wrong reading that produced a 'collapsed right wing
    vein of 2.53u' which was really a tarsal segment."""
    det = list(OmegaConf.load(DET).kp_names)
    mdl = list(OmegaConf.load(ANA).model.KP_NAMES)
    for name in ("WingR_base", "WingR_V12", "WingR_V13"):
        i = det.index(name)
        assert mdl[i].startswith("T2L_"), (
            f"detector index {i} ({name}) maps to model {mdl[i]}; the recorded "
            f"failure was T2L_*, so this test needs updating with the new mapping")


def _bout(tmp_path, kp_names, cameras, T=5):
    d = tmp_path / "fly1"; d.mkdir(parents=True)
    K, C = len(kp_names), len(cameras)
    np.savez(d / "kp2d.npz", kp2d=np.zeros((T, C, K, 2), np.float32),
             conf=np.ones((T, C, K), np.float32))
    np.savez(d / "kp3d.npz", kp3d=np.zeros((T, K, 3), np.float32),
             conf3d=np.ones((T, K), np.float32))
    return tmp_path


def test_detector_config_is_refused():
    """Passing the DETECTOR config is the bug; it must fail loudly, not silently
    read other body parts."""
    with pytest.raises(KeypointOrderError, match="model.KP_NAMES"):
        load_bout_kp("/nonexistent", 1, anatomy_cfg=DET, cameras=["c"])


def test_named_access_and_unknown_name_raises(tmp_path):
    kp_names = list(OmegaConf.load(ANA).model.KP_NAMES)
    cams = list(OmegaConf.load("configs/recording/session0.yaml").cameras)
    b = load_bout_kp(str(_bout(tmp_path, kp_names, cams)), 1,
                     anatomy_cfg=ANA, cameras=cams)
    assert b.kp3d("WingL_V12").shape == (5, 3)
    assert b.kp2d("WingL_V12", cams[0]).shape == (5, 2)
    with pytest.raises(KeypointOrderError, match="MODEL order"):
        b.kp3d("NotAKeypoint")
    with pytest.raises(CameraOrderError):
        b.kp2d("WingL_V12", "CamDoesNotExist")


def test_camera_count_mismatch_is_refused(tmp_path):
    """Handing the wrong-length camera list must raise, not index off the end."""
    kp_names = list(OmegaConf.load(ANA).model.KP_NAMES)
    cams = list(OmegaConf.load("configs/recording/session0.yaml").cameras)
    root = _bout(tmp_path, kp_names, cams)
    with pytest.raises(CameraOrderError):
        load_bout_kp(str(root), 1, anatomy_cfg=ANA, cameras=cams[:3])


def test_centroids_are_reindexed_by_name_not_position(tmp_path):
    """The mask npz stores its own camera order. centroids_canonical must
    permute BY NAME -- taking the npz's index against kp2d was the camera half
    of the 2026-08-31 bug."""
    stored = ["camC", "camA", "camB"]
    canon = ["camA", "camB", "camC"]
    T = 4
    cen = np.zeros((2, 3, T, 2), np.float32)
    for i in range(3):
        cen[:, i] = i                      # tag each stored slot with its index
    p = tmp_path / "m.npz"
    np.savez(p, cameras=np.array(stored), centroids=cen,
             valid=np.ones((2, 3, T), bool))
    out, val, perm = centroids_canonical(str(p), canon)
    assert perm == [1, 2, 0]
    # canonical slot 0 is camA, which was stored at index 1 -> tagged 1
    assert out[0, 0, 0, 0] == 1
    assert out[0, 1, 0, 0] == 2
    assert out[0, 2, 0, 0] == 0
    with pytest.raises(CameraOrderError):
        centroids_canonical(str(p), ["camA", "camMissing"])


def test_segment_length_is_the_check_that_catches_an_order_bug(tmp_path):
    """A RIGID pair's length must be constant; that invariant is what exposed
    the bug when jitter and confidence all looked excellent."""
    kp_names = list(OmegaConf.load(ANA).model.KP_NAMES)
    cams = list(OmegaConf.load("configs/recording/session0.yaml").cameras)
    root = _bout(tmp_path, kp_names, cams)
    b = load_bout_kp(str(root), 1, anatomy_cfg=ANA, cameras=cams)
    L = b.segment_length("WingL_V12", "WingL_V13")
    assert L.shape == (5,) and np.allclose(L, 0.0)
    with pytest.raises(ValueError, match="3-D only"):
        b.segment_length("WingL_V12", "WingL_V13", use_3d=False)
