import numpy as np
from viz.core import reproject

def test_project_matches_ph_at_M_convention():
    # M (4,3): pick a simple orthographic-ish matrix; verify uv = (ph@M)[:2]/[2]
    M = np.array([[2.,0.,0.],[0.,3.,0.],[0.,0.,0.],[10.,20.,1.]], np.float32)  # (4,3)
    pts = np.array([[1.,1.,5.],[0.,0.,0.]], np.float64)
    ph = np.concatenate([pts, np.ones((2,1))],1); proj = ph @ M
    expect = (proj[:,:2]/proj[:,2:3])
    got = reproject.project(M, pts)
    assert np.allclose(got, expect, atol=1e-4) and got.shape == (2,2)

def test_project_empty_safe():
    M = np.eye(4,3, dtype=np.float32)
    assert reproject.project(M, np.zeros((0,3))).shape == (0,2)

def test_reproject_all_stacks_per_camera():
    M = np.array([[2.,0.,0.],[0.,3.,0.],[0.,0.,0.],[10.,20.,1.]], np.float32)
    cam_mats = np.stack([M, M*1.0])
    out = reproject.reproject_all(cam_mats, np.array([[1.,1.,5.]]))
    assert out.shape == (2,1,2)

def test_camera_matrices_reuses_reprojection_tool(monkeypatch):
    # Fake ReprojectionTool with minimal interface
    class FakeCamera:
        def __init__(self, name):
            self.name = name

    class FakeRT:
        def __init__(self, calib_dir):
            pass  # ignore calib_dir

        @property
        def camera_matrices(self):
            # Return float64 array so we can test dtype conversion
            return np.arange(24, dtype=np.float64).reshape(2, 4, 3)

        @property
        def _camera_list(self):
            return [FakeCamera("cam0"), FakeCamera("cam1")]

    monkeypatch.setattr(reproject, "ReprojectionTool", FakeRT)
    cam_mats, names = reproject.camera_matrices("ignored")

    # Assert shape and dtype conversion
    assert cam_mats.shape == (2, 4, 3)
    assert cam_mats.dtype == np.float32
    # Assert names extracted in order
    assert names == ["cam0", "cam1"]
    # Assert values preserved through conversion
    assert np.allclose(cam_mats, np.arange(24, dtype=np.float64).reshape(2, 4, 3))
