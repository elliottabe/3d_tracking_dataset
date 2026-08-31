"""The detector keypoint-order guard: cfg.detector.kp_names must actually
describe the checkpoint it is used with.

Motivation: reorder_detector_to_model is NAME-based, so a mislabelled channel
list permutes every keypoint into the wrong slot while residuals, NaN counts
and silhouette IoU all stay plausible -- the same failure class as the
historical keypoint-order bug that LOO/IoU was fully blind to.
"""
import json
import warnings

import pytest

from jarvis_jax.tracking.predict_2d import (
    DetectorOrderMismatch, verify_detector_kp_order)

ORDER = ["Antenna_Base", "EyeL", "EyeR", "Scutellum", "Abd_A4", "Abd_tip"]


def _fake_run(tmp_path, training_order):
    """Build <run>/final + <run>/.hydra/overrides.yaml + <data>/annotations."""
    data = tmp_path / "dataset"
    (data / "annotations").mkdir(parents=True)
    if training_order is not None:
        (data / "annotations" / "keypoint_names.json").write_text(
            json.dumps(training_order))
    run = tmp_path / "run"
    (run / ".hydra").mkdir(parents=True)
    (run / ".hydra" / "overrides.yaml").write_text(
        f"- run_id=whatever\n- paths.data_root={data}\n")
    ckpt = run / "final"
    ckpt.mkdir()
    return ckpt


def test_matching_order_passes(tmp_path):
    ckpt = _fake_run(tmp_path, ORDER)
    assert verify_detector_kp_order(ckpt, ORDER) == ORDER


def test_permuted_order_raises(tmp_path):
    """Same landmarks, wrong order -- the dangerous case, must STOP the run."""
    ckpt = _fake_run(tmp_path, ORDER)
    declared = [ORDER[3], ORDER[1], ORDER[2], ORDER[0], ORDER[4], ORDER[5]]
    with pytest.raises(DetectorOrderMismatch) as e:
        verify_detector_kp_order(ckpt, declared)
    assert "PERMUTED" in str(e.value)
    assert "channel 0" in str(e.value)


def test_different_landmark_set_raises(tmp_path):
    ckpt = _fake_run(tmp_path, ORDER)
    declared = list(ORDER[:-1]) + ["NotAKeypoint"]
    with pytest.raises(DetectorOrderMismatch) as e:
        verify_detector_kp_order(ckpt, declared)
    assert "different landmark sets" in str(e.value)


def test_unverifiable_checkpoint_warns_but_does_not_raise(tmp_path):
    """Older runs (e.g. red_data_unified_V4) carry no keypoint_names.json.
    Refusing to run on a known-good old checkpoint would be worse than the
    risk, so this warns."""
    ckpt = _fake_run(tmp_path, None)
    with pytest.warns(UserWarning, match="UNVERIFIED"):
        assert verify_detector_kp_order(ckpt, ORDER) is None


def test_strict_false_downgrades_mismatch_to_warning(tmp_path):
    ckpt = _fake_run(tmp_path, ORDER)
    declared = [ORDER[3], ORDER[1], ORDER[2], ORDER[0], ORDER[4], ORDER[5]]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert verify_detector_kp_order(ckpt, declared, strict=False) == ORDER
    assert any("MISMATCH" in str(x.message) for x in w)
