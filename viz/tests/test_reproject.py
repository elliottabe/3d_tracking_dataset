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
