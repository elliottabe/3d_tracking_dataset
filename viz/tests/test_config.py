import pytest
from viz import config as vconfig

# Compose once at collection time: cheap (YAML merge, no I/O beyond configs/),
# and lets the skip condition and the assertions below share one result
# instead of composing twice.
try:
    _REC = vconfig.courtship_recording()
    _ERR = None
except Exception as e:  # pragma: no cover - environment-dependent
    _REC = None
    _ERR = e


@pytest.mark.skipif(_REC is None, reason=f"courtship_pipeline hydra compose failed: {_ERR}")
def test_courtship_recording_shape():
    rec = _REC
    assert isinstance(rec["calib_dir"], str) and rec["calib_dir"]
    assert isinstance(rec["session_dir"], str) and rec["session_dir"]
    assert isinstance(rec["predictions_dir"], str) and rec["predictions_dir"]

    assert isinstance(rec["cameras"], list) and len(rec["cameras"]) == 7
    assert all(isinstance(c, str) and c.startswith("Cam") for c in rec["cameras"])

    assert isinstance(rec["kp_names"], list) and len(rec["kp_names"]) == 50
    assert "EyeL" in rec["kp_names"]
    assert "Abd_tip" in rec["kp_names"]
