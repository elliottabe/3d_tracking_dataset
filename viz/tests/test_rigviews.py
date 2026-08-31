import os

import numpy as np
import pytest

from viz.core import reproject, rigviews

CALIB_DIR = ("/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/"
             "courtship/Session0/2025_10_20_13_20_04/calibration")


@pytest.mark.skipif(not os.path.isdir(CALIB_DIR), reason="Session0 calibration dir not present")
def test_classify_views_session0_triple():
    # Documented rig geometry (see rigviews module docstring): a ~180 deg arc
    # in the Y-Z plane. Structural rule -> top is the straight-down camera;
    # left/right are the ~30 deg-elevated obliques (elevation_target=0.5), one
    # per side of the dominant (Y) axis -- NOT the exactly-horizontal pair
    # (Cam2012857/Cam2012861), which sees the fly edge-on.
    got = rigviews.classify_views(CALIB_DIR)
    assert set(got) == {"top", "left", "right"}
    assert got["top"].name == "Cam2012630"
    assert got["left"].name == "Cam2012855"
    assert got["right"].name == "Cam2012631"
    # Sanity on the numbers backing the choice: top is near-vertical, the
    # laterals sit close to the 0.5 (~30 deg) elevation target and are on
    # opposite sides of the dominant horizontal axis.
    assert abs(got["top"].direction[2]) > 0.99
    assert 20 < got["left"].elevation_deg < 40
    assert 20 < got["right"].elevation_deg < 40
    assert np.sign(got["left"].direction[1]) != np.sign(got["right"].direction[1])


def _fake_rt(cam_mats_by_name):
    class FakeCamera:
        def __init__(self, name):
            self.name = name

    class FakeRT:
        def __init__(self, calib_dir):
            pass

        @property
        def camera_matrices(self):
            return np.stack(list(cam_mats_by_name.values()))

        @property
        def _camera_list(self):
            return [FakeCamera(n) for n in cam_mats_by_name]

    return FakeRT


def _affine_mat(m0, m1, o0=0.0, o1=0.0):
    """Build a (4,3) `ph @ M` camera matrix with image axes m0, m1 (world
    units) and the rig's affine third column [0,0,0,1]."""
    M = np.zeros((4, 3))
    M[:3, 0], M[3, 0] = m0, o0
    M[:3, 1], M[3, 1] = m1, o1
    M[3, 2] = 1.0
    return M


def test_classify_views_two_camera_rig_raises_clearly(monkeypatch):
    # Only 2 cameras -> no top/left/right triple possible; must fail with a
    # clear, typed error rather than crashing on an index/key lookup.
    mats = {
        "camA": _affine_mat([1, 0, 0], [0, 1, 0]),
        "camB": _affine_mat([0, 1, 0], [1, 0, 0]),
    }
    monkeypatch.setattr(reproject, "ReprojectionTool", _fake_rt(mats))
    with pytest.raises(ValueError, match="need >=3 cameras"):
        rigviews.classify_views("ignored")


def test_classify_views_no_camera_on_one_side_raises_clearly(monkeypatch):
    # 3 cameras, but two share the same side of the dominant horizontal axis
    # and none is on the other side -> no left/right pair; degrade with a
    # clear error instead of silently picking an arbitrary camera. Image-axis
    # rows (m0, m1) are an orthonormal basis chosen so cross(m0, m1) gives the
    # intended view direction exactly (right-hand rule: m0, m1, direction).
    mats = {
        "top":    _affine_mat([1, 0, 0], [0, -1, 0]),        # view (0, 0, -1)
        "side_a": _affine_mat([1, 0, 0], [0, 0, -1]),        # view (0, 1, 0)
        "side_b": _affine_mat([0.9988, -0.0499, 0], [0, 0, -1]),  # view ~(0.05, 1, 0)
    }
    monkeypatch.setattr(reproject, "ReprojectionTool", _fake_rt(mats))
    with pytest.raises(ValueError, match="no camera on one side"):
        rigviews.classify_views("ignored")


def test_format_view_label_names_role_and_direction():
    rv = rigviews.RigView("Cam2012855", np.array([-0.01, -0.85, -0.52]), 31.0)
    label = rigviews.format_view_label("left", rv)
    assert "Cam2012855" in label and "left" in label and "31" in label
