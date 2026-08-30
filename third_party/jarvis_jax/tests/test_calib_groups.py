import os
import pytest
from jarvis_jax.data.calib_groups import calib_fingerprint, group_calibrations

CAMS = ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855",
        "Cam2012857", "Cam2012861", "Cam2012862"]

def _write_calib(tmp_path, name, first_coeff, *, pretty=False):
    d = tmp_path / name
    d.mkdir(parents=True)
    for c in CAMS:
        data = [first_coeff, 0.0074869, -0.031773, -2.828,
                0.0093308, -8.0788, -0.17912, 462.78, 0.0, 0.0, 0.0, 1.0]
        if pretty:
            body = ",\n       ".join(f"{v}" for v in data)
        else:
            body = ", ".join(f"{v:.16g}" for v in data)
        (d / f"{c}.yaml").write_text(
            "%YAML:1.0\n---\nimage_width: 1936\nimage_height: 448\n"
            "projectionMatrix: !!opencv-matrix\n   rows: 3\n   cols: 4\n   dt: d\n"
            f"   data: [ {body} ]\nscale: 10\n")
    return str(d)

def test_fingerprint_ignores_formatting(tmp_path):
    """Formatting-only differences must NOT create a new calibration group.
    Real case: 20_04_female_climbing is numerically identical to Session0 but
    written with fewer decimal places."""
    a = _write_calib(tmp_path, "terse", 8.1001)
    b = _write_calib(tmp_path, "pretty", 8.1001, pretty=True)
    assert calib_fingerprint(a) == calib_fingerprint(b)

def test_fingerprint_separates_real_differences(tmp_path):
    a = _write_calib(tmp_path, "grpA", 8.1001)
    b = _write_calib(tmp_path, "grpB", 8.1333)
    assert calib_fingerprint(a) != calib_fingerprint(b)

def test_groups_labelled_by_descending_size(tmp_path):
    dirs = {
        "rec_b1": _write_calib(tmp_path, "b1", 8.1333),
        "rec_b2": _write_calib(tmp_path, "b2", 8.1333),
        "rec_b3": _write_calib(tmp_path, "b3", 8.1333),
        "rec_a1": _write_calib(tmp_path, "a1", 8.1001),
        "rec_a2": _write_calib(tmp_path, "a2", 8.1001),
        "rec_c1": _write_calib(tmp_path, "c1", 8.0898),
    }
    g = group_calibrations(dirs)
    assert g["rec_b1"] == g["rec_b2"] == g["rec_b3"] == "A"   # biggest group
    assert g["rec_a1"] == g["rec_a2"] == "B"
    assert g["rec_c1"] == "C"

def test_missing_camera_raises(tmp_path):
    d = _write_calib(tmp_path, "short", 8.1001)
    os.remove(os.path.join(d, "Cam2012862.yaml"))
    with pytest.raises(ValueError, match="expected 7 cameras"):
        calib_fingerprint(d)
