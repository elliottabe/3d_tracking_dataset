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
    from jarvis_jax.tracking.qc import per_camera_reproj_error
    rt = _FakeRT()
    kp3d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    kp2d = {0: np.array([[1., 2.], [4., 5.]]),   # (x,y)
            1: np.array([[1., 3.], [4., 6.]])}   # (x,z)
    vis = {0: np.array([True, True]), 1: np.array([True, True])}
    err = per_camera_reproj_error(rt, kp3d, kp2d, vis)
    assert set(err) == {0, 1}
    assert err[0] < 1e-9 and err[1] < 1e-9


def test_per_camera_reproj_error_offset():
    from jarvis_jax.tracking.qc import per_camera_reproj_error
    rt = _FakeRT()
    kp3d = np.array([[0.0, 0.0, 0.0]])
    kp2d = {0: np.array([[3.0, 4.0]])}   # true proj is (0,0); err = 5
    vis = {0: np.array([True])}
    err = per_camera_reproj_error(rt, kp3d, kp2d, vis)
    assert abs(err[0] - 5.0) < 1e-9


def test_loo_reproj_near_zero_when_consistent():
    from jarvis_jax.tracking.qc import loo_reproj
    rt = _FakeRT()
    # one keypoint at (1,2,3); its two 2-D obs are the exact projections.
    kp2d = {0: np.array([[1., 2.]]), 1: np.array([[1., 3.]])}
    vis = {0: np.array([True]), 1: np.array([True])}
    rep = loo_reproj(rt, kp2d, vis)
    # only 2 cams -> each held-out leaves 1 other (<2) -> no LOO error terms.
    assert rep["n"] == 0
    assert np.isnan(rep["median"])


def test_loo_reproj_three_cams_consistent():
    from jarvis_jax.tracking.qc import loo_reproj

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
    from jarvis_jax.tracking.qc import qc_report
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
    # additive-only: kp3d_measured_by_frame/kp_names/group_defs all omitted
    # -> no "ik_reproj" key, existing keys untouched.
    assert "ik_reproj" not in rep


def test_per_camera_reproj_error_skips_nan_kp3d_instead_of_poisoning_median():
    """Regression for the measured real-run bug: a frame whose kp3d_mm is
    NaN (bridge failed) but whose detector confidence is still `vis=True`
    must not poison this camera's median to NaN -- the NaN sample is simply
    dropped, and the median is over the remaining finite ones."""
    from jarvis_jax.tracking.qc import per_camera_reproj_error
    rt = _FakeRT()
    kp3d = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]])
    kp2d = {0: np.array([[1., 2.], [5., 5.]])}   # kp1's obs is irrelevant -- kp1's 3D is NaN
    vis = {0: np.array([True, True])}
    err = per_camera_reproj_error(rt, kp3d, kp2d, vis)
    assert np.isfinite(err[0])
    assert err[0] < 1e-9  # only kp0 (exact, finite) contributes


def test_qc_report_per_camera_reproj_px_not_poisoned_by_one_nan_frame():
    """Regression: qc_report's aggregate `per_camera_reproj_px` must not
    go NaN just because ONE frame's kp3d is entirely NaN, even though other
    frames are fine (this is exactly what was observed on real fly0 data:
    483/2007 NaN frames poisoned n=12232 samples to a NaN median)."""
    from jarvis_jax.tracking.qc import qc_report
    rt = _FakeRT()
    good_frame = {0: np.array([[1., 2.]]), 1: np.array([[1., 3.]])}
    vis_frame = {0: np.array([True]), 1: np.array([True])}
    kp3d_by_frame = [np.array([[1., 2., 3.]]), np.array([[np.nan, np.nan, np.nan]])]
    kp2d_by_frame = [good_frame, good_frame]
    vis_by_frame = [vis_frame, vis_frame]
    mesh_by_frame = [np.zeros((0, 3)), np.zeros((0, 3))]
    masks_by_frame = [{}, {}]
    rep = qc_report(rt, kp3d_by_frame=kp3d_by_frame, mesh_by_frame=mesh_by_frame,
                    kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
                    masks_by_frame=masks_by_frame, out_json=None)
    assert np.isfinite(rep["per_camera_reproj_px"]["median"])
    assert rep["per_camera_reproj_px"]["median"] < 1e-6


# ---------------------------------------------------------------------------
# ik_reproj_report / qc_report(..., kp3d_measured_by_frame=..., kp_names=...)
# ---------------------------------------------------------------------------

def test_ik_reproj_report_overall_and_ratio():
    from jarvis_jax.tracking.qc import ik_reproj_report
    rt = _FakeRT()
    # kp0's true projection is (1,2)/(1,3). Fitted is exact (err=0); measured
    # is offset by (3,4) in cam0 -> err=5 in cam0, 0 in cam1.
    kp3d_fitted = [np.array([[1.0, 2.0, 3.0]])]
    kp3d_measured = [np.array([[1.0, 2.0, 3.0]]) + np.array([[0.0, 0.0, 0.0]])]
    kp2d_by_frame = [{0: np.array([[4.0, 6.0]]), 1: np.array([[1.0, 3.0]])}]
    vis_by_frame = [{0: np.array([True]), 1: np.array([True])}]

    rep = ik_reproj_report(rt, kp3d_fitted_by_frame=kp3d_fitted,
                           kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
                           kp3d_measured_by_frame=kp3d_measured)
    # both sources are the SAME 3-D point here -- fitted == measured.
    assert rep["fitted_median_px"] == pytest.approx(rep["measured_median_px"])
    assert rep["ratio_fitted_over_measured"] == pytest.approx(1.0)
    assert rep["n_fitted"] == 2 and rep["n_measured"] == 2


def test_ik_reproj_report_no_measured_source_leaves_measured_nan():
    from jarvis_jax.tracking.qc import ik_reproj_report
    rt = _FakeRT()
    kp3d_fitted = [np.array([[1.0, 2.0, 3.0]])]
    kp2d_by_frame = [{0: np.array([[1., 2.]]), 1: np.array([[1., 3.]])}]
    vis_by_frame = [{0: np.array([True]), 1: np.array([True])}]
    rep = ik_reproj_report(rt, kp3d_fitted_by_frame=kp3d_fitted,
                           kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame)
    assert np.isfinite(rep["fitted_median_px"])
    assert np.isnan(rep["measured_median_px"])
    assert np.isnan(rep["ratio_fitted_over_measured"])
    assert rep["n_measured"] == 0


def test_ik_reproj_report_per_keypoint_and_per_group_use_real_names():
    from jarvis_jax.tracking.qc import ik_reproj_report
    rt = _FakeRT()
    # kp0 exact (err 0 both cams); kp1 fitted exact, measured offset in cam0.
    kp3d_fitted = [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])]
    kp3d_measured = [np.array([[1.0, 2.0, 3.0], [7.0, 9.0, 6.0]])]
    kp2d_by_frame = [{0: np.array([[1., 2.], [4., 5.]]),
                      1: np.array([[1., 3.], [4., 6.]])}]
    vis_by_frame = [{0: np.array([True, True]), 1: np.array([True, True])}]
    kp_names = ["Scutellum", "Abd_tip"]
    group_defs = {"trunk": [0], "abdomen": [1]}

    rep = ik_reproj_report(rt, kp3d_fitted_by_frame=kp3d_fitted,
                           kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
                           kp3d_measured_by_frame=kp3d_measured,
                           kp_names=kp_names, group_defs=group_defs)

    assert set(rep["per_keypoint"]) == {"Scutellum", "Abd_tip"}
    assert rep["per_keypoint"]["Scutellum"]["fitted_median_px"] == pytest.approx(0.0, abs=1e-9)
    assert rep["per_keypoint"]["Scutellum"]["measured_median_px"] == pytest.approx(0.0, abs=1e-9)
    assert rep["per_keypoint"]["Abd_tip"]["measured_median_px"] > 0.0

    assert set(rep["per_group"]) == {"trunk", "abdomen"}
    assert rep["per_group"]["trunk"]["fitted_median_px"] == pytest.approx(0.0, abs=1e-9)
    assert rep["per_group"]["abdomen"]["measured_median_px"] > 0.0


def test_qc_report_ik_reproj_key_is_additive_and_carries_measured_and_group():
    from jarvis_jax.tracking.qc import qc_report
    rt = _FakeRT()
    kp3d_by_frame = [np.array([[1., 2., 3.]])]                # "fitted": exact
    kp3d_measured_by_frame = [np.array([[1.0, 2.0, 3.0 + 3.0]])]  # "measured": offset in cam1 only
    mesh_by_frame = [np.zeros((0, 3))]
    kp2d_by_frame = [{0: np.array([[1., 2.]]), 1: np.array([[1., 3.]])}]
    vis_by_frame = [{0: np.array([True]), 1: np.array([True])}]
    masks_by_frame = [{}]
    kp_names = ["Scutellum"]
    group_defs = {"trunk": [0]}

    rep = qc_report(rt, kp3d_by_frame=kp3d_by_frame, mesh_by_frame=mesh_by_frame,
                    kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
                    masks_by_frame=masks_by_frame, out_json=None,
                    kp3d_measured_by_frame=kp3d_measured_by_frame,
                    kp_names=kp_names, group_defs=group_defs)

    for k in ("per_camera_reproj_px", "loo_reproj_px", "silhouette_iou", "n_frames"):
        assert k in rep  # existing keys untouched
    assert "ik_reproj" in rep
    assert "per_keypoint" in rep["ik_reproj"] and "Scutellum" in rep["ik_reproj"]["per_keypoint"]
    assert "per_group" in rep["ik_reproj"] and "trunk" in rep["ik_reproj"]["per_group"]
    # fitted is exact (median 0); measured has a nonzero offset -> ratio is 0,
    # not NaN, and the top-level/per-keypoint/per-group numbers all agree.
    assert rep["ik_reproj"]["fitted_median_px"] == pytest.approx(0.0, abs=1e-9)
    assert rep["ik_reproj"]["measured_median_px"] > 0.0
    assert rep["ik_reproj"]["ratio_fitted_over_measured"] == pytest.approx(0.0, abs=1e-9)
    assert rep["ik_reproj"]["per_keypoint"]["Scutellum"] == {
        k: rep["ik_reproj"][k] for k in
        ("fitted_median_px", "measured_median_px", "ratio_fitted_over_measured", "n_fitted", "n_measured")}
    assert rep["ik_reproj"]["per_group"]["trunk"]["measured_median_px"] > 0.0
