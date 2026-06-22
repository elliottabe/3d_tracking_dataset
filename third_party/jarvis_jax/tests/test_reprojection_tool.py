"""Tests for the NumPy ReprojectionTool (Task 3 – geometry/reprojection_tool.py)."""
import os
import numpy as np
import pytest
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

CALIB = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3/calib_params/2026_05_27_11_56_05"
skip = pytest.mark.skipif(not os.path.isdir(CALIB), reason="calibration absent")


@skip
def test_triangulate_roundtrip():
    rt = ReprojectionTool(CALIB)
    assert rt.num_cameras >= 2
    p3d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    pts2d = rt.reproject_point(p3d)                 # (num_cam, 2)
    rec = rt.reconstruct_point(pts2d)               # (3,)
    assert np.linalg.norm(rec - p3d) < 1e-3, f"round-trip error: {rec}"


@skip
def test_camera_matrices_shape():
    rt = ReprojectionTool(CALIB)
    assert rt.camera_matrices.shape == (rt.num_cameras, 4, 3)
    assert rt.camera_matrices.dtype == np.float32


@skip
def test_reproject_shape():
    rt = ReprojectionTool(CALIB)
    p3d = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    pts2d = rt.reproject_point(p3d)
    assert pts2d.shape == (rt.num_cameras, 2)


@skip
def test_cams_to_use():
    rt = ReprojectionTool(CALIB)
    assert rt.num_cameras >= 4
    p3d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    pts2d = rt.reproject_point(p3d)
    # Use only first 4 cameras
    rec = rt.reconstruct_point(pts2d, cams_to_use=list(range(4)))
    assert np.linalg.norm(rec - p3d) < 1e-3, f"partial-cam round-trip error: {rec}"
