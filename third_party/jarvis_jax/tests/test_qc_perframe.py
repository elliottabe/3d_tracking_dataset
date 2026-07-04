# tests/test_qc_perframe.py
import numpy as np, pytest
from jarvis_jax.cse.qc_perframe import per_frame_qc


class FakeRT:
    num_cameras = 2
    def reproject_point(self, X):            # identity-ish 2 cams
        return np.array([[X[0], X[1]], [X[0]+1, X[1]]], float)


def test_per_frame_qc_shapes_and_nan():
    T = 3
    mesh = [np.zeros((4,3)) for _ in range(T)]
    kp3d = [np.zeros((2,3)) for _ in range(T)]
    kp2d = [{0: np.zeros((2,2)), 1: np.zeros((2,2))} for _ in range(T)]
    vis  = [{0: np.ones(2,bool), 1: np.ones(2,bool)} for _ in range(T)]
    masks = [{0: np.ones((5,5),bool), 1: np.ones((5,5),bool)} for _ in range(T)]
    masks[1] = {0: None, 1: None}            # frame 1 has no masks -> NaN row
    out = per_frame_qc(FakeRT(), mesh_by_frame=mesh, kp3d_by_frame=kp3d,
                       kp2d_by_frame=kp2d, vis_by_frame=vis, masks_by_frame=masks)
    for k in ("soft_iou","hard_iou","reproj_px","n_cams"):
        assert out[k].shape == (T,)
    assert np.isnan(out["soft_iou"][1])      # no-mask frame -> NaN
    assert out["n_cams"][0] == 2
