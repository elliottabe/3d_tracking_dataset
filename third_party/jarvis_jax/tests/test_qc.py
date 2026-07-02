# tests/test_qc.py
import numpy as np
import pytest


class _FakeRT:
    """Minimal ReprojectionTool stand-in with real DLT-ish 3x4 matrices.

    Two orthographic-ish cameras looking down different axes so a 3-D point is
    recoverable and reprojects back exactly (consistent cameras -> LOO ~ 0)."""
    def __init__(self):
        # cam0: image = (x, y); cam1: image = (x, z). Both affine (3rd row homog).
        P0 = np.array([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        P1 = np.array([[1., 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self._P = [P0, P1]
        self.num_cameras = 2

        class _C:
            def __init__(s, M): s.cameraMatrix = M
        self._camera_list = [_C(P0), _C(P1)]

    def reproject_point(self, X):
        X = np.asarray(X, float)
        Xh = np.concatenate([X, [1.0]])
        out = np.zeros((self.num_cameras, 2))
        for i, P in enumerate(self._P):
            p = P @ Xh
            out[i] = (p / p[2])[:2]
        return out

    def reconstruct_point(self, points2d, cams_to_use=None):
        cams = cams_to_use if cams_to_use is not None else list(range(self.num_cameras))
        A = []
        for c in cams:
            P = self._P[c]; u, v = points2d[c]
            A.append(u * P[2] - P[0]); A.append(v * P[2] - P[1])
        A = np.asarray(A)
        _, _, Vh = np.linalg.svd(A)
        Xh = Vh[-1] / Vh[-1][3]
        return Xh[:3]


def test_per_camera_reproj_error_zero_on_consistent():
    from jarvis_jax.cse.qc import per_camera_reproj_error
    rt = _FakeRT()
    kp3d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    kp2d = {0: np.array([[1., 2.], [4., 5.]]),   # (x,y)
            1: np.array([[1., 3.], [4., 6.]])}   # (x,z)
    vis = {0: np.array([True, True]), 1: np.array([True, True])}
    err = per_camera_reproj_error(rt, kp3d, kp2d, vis)
    assert set(err) == {0, 1}
    assert err[0] < 1e-9 and err[1] < 1e-9


def test_per_camera_reproj_error_offset():
    from jarvis_jax.cse.qc import per_camera_reproj_error
    rt = _FakeRT()
    kp3d = np.array([[0.0, 0.0, 0.0]])
    kp2d = {0: np.array([[3.0, 4.0]])}   # true proj is (0,0); err = 5
    vis = {0: np.array([True])}
    err = per_camera_reproj_error(rt, kp3d, kp2d, vis)
    assert abs(err[0] - 5.0) < 1e-9


def test_loo_reproj_near_zero_when_consistent():
    from jarvis_jax.cse.qc import loo_reproj
    rt = _FakeRT()
    # one keypoint at (1,2,3); its two 2-D obs are the exact projections.
    kp2d = {0: np.array([[1., 2.]]), 1: np.array([[1., 3.]])}
    vis = {0: np.array([True]), 1: np.array([True])}
    rep = loo_reproj(rt, kp2d, vis)
    # only 2 cams -> each held-out leaves 1 other (<2) -> no LOO error terms.
    assert rep["n"] == 0
    assert np.isnan(rep["median"])


def test_loo_reproj_three_cams_consistent():
    from jarvis_jax.cse.qc import loo_reproj

    class _RT3(_FakeRT):
        def __init__(s):
            P0 = np.array([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            P1 = np.array([[1., 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            P2 = np.array([[0., 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            s._P = [P0, P1, P2]; s.num_cameras = 3

            class _C:
                def __init__(c, M): c.cameraMatrix = M
            s._camera_list = [_C(P0), _C(P1), _C(P2)]

    rt = _RT3()
    X = np.array([1.0, 2.0, 3.0])
    proj = rt.reproject_point(X)
    kp2d = {c: proj[c][None] for c in range(3)}
    vis = {c: np.array([True]) for c in range(3)}
    rep = loo_reproj(rt, kp2d, vis)
    assert rep["n"] == 3          # each of 3 cams held out, 2 others triangulate
    assert rep["median"] < 1e-6   # consistent cams -> ~0 held-out error


def test_qc_report_bundles_keys():
    from jarvis_jax.cse.qc import qc_report
    rt = _FakeRT()
    kp3d_by_frame = [np.array([[1., 2., 3.]])]
    mesh_by_frame = [np.zeros((0, 3))]              # no mesh -> empty iou
    kp2d_by_frame = [{0: np.array([[1., 2.]]), 1: np.array([[1., 3.]])}]
    vis_by_frame = [{0: np.array([True]), 1: np.array([True])}]
    masks_by_frame = [{}]                           # no masks
    rep = qc_report(rt, kp3d_by_frame=kp3d_by_frame, mesh_by_frame=mesh_by_frame,
                    kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
                    masks_by_frame=masks_by_frame, out_json=None)
    for k in ("per_camera_reproj_px", "loo_reproj_px", "silhouette_iou", "n_frames"):
        assert k in rep
    assert rep["n_frames"] == 1
