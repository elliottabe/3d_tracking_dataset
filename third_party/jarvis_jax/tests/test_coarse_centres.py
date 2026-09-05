"""`jarvis_jax.tracking.coarse_centres` -- CenterDetect peaks to 3D centres
and window plans (spec 2026-09-04-mvq-maskfree-frontend-design.md §4.1).

Expectations these tests encode (per CLAUDE.md -- state the expectation,
then check invariants, not just smoothness):

  * `lift_peaks_to_centres` must recover BOTH flies' true 3D positions from
    noisy multi-camera peaks even when some cameras are missing a peak or
    report the two peaks in swapped order -- a real per-camera failure mode
    (`extract_top_k_peaks` has no cross-camera identity), and even when one
    camera's peak is a geometric outlier (it must be excluded as a
    non-inlier, not silently averaged in and bias the centre).
  * `cluster_centres`/`plan_windows` merge only centres closer than their
    stated physical threshold (1.5 mm / 3 mm respectively, in 0.1 mm world
    units) and NaN-pad rather than silently drop, so downstream code sees a
    fixed-shape array with an explicit "nothing here" marker instead of a
    length that varies with how many animals were found.
  * `_peaks_from_heatmap` is tested against a hand-built heatmap (never the
    real checkpoint -- this module's tests must run on CPU with no GPU and
    no network) with a KNOWN anisotropic x/y scale, so a bug that conflated
    the two scale factors (or dropped the sub-`min_score` NaN-out) would
    show up as a wrong peak position, not just a wrong shape.
  * `_centerdetect_preprocess` is tested for shape/dtype and for actually
    normalising (a mid-grey frame lands near zero) -- it does not re-derive
    "PIL BILINEAR vs cv2 INTER_LINEAR differ by ~11/255" itself (that would
    need a real frame and the training dataset's own resize path); it just
    guards the resize call and constant that make the two match.
  * `lift_peaks_to_centres` raises on a camera-COUNT mismatch between
    `peaks` and `cam_mats` -- the one camera-order error it can detect
    (module docstring's camera-order contract).
"""
import numpy as np
import pytest

from jarvis_jax.tracking.coarse_centres import (
    _centerdetect_preprocess,
    _peaks_from_heatmap,
    lift_peaks_to_centres,
    cluster_centres,
    plan_windows,
)


# ---------------------------------------------------------------------------
# A non-degenerate 7-camera AFFINE rig for the lift_peaks_to_centres tests,
# built inline the way `tests/test_triangulate_refine.py::_rig` is (no
# calibration YAMLs / filesystem fixture needed for pure geometry).
#
# `tests/mvq_fixtures.py::make_v12_root`'s 7 cameras rotate ONE base affine
# camera's projection submatrix about the Z axis ONLY (`cam_P`). Rz's third
# column is [0,0,1], so every one of those 7 "different" cameras keeps the
# IDENTICAL world-Z image-plane sensitivity -- world Z sits along a
# near-null direction of the assembled multi-view DLT system (SVD:
# [21.4, 21.4, 0.48, 0], ~44x condition number), exactly the "camera centers
# share the rotation axis -> depth along it is unrecoverable" defect
# `test_triangulate_refine.py::_rig`'s own docstring documents (and that
# fixture works around with a REAL camera ring instead of a shared axis).
# Rotating about a MIX of x and y (at different rates per camera, so no
# axis is shared) instead keeps row 3 = [0,0,0,1] (still affine, matching
# the brief) while making all three world axes well observed: SVD of the
# same assembled system is [18.7, 18.4, 15.2, 0] here -- condition number
# ~1.2x, not ~44x -- confirmed to recover both flies' X/Y/Z to <0.2 units
# under 1px peak noise across 20 random seeds before picking seed 0 below.
# ---------------------------------------------------------------------------

_P_REAL = np.array([[8.1001, 0.0074869, -0.031773, 900.0],
                    [0.0093308, -8.0788, -0.17912, 300.0],
                    [0.0, 0.0, 0.0, 1.0]], np.float64)


def _affine_rig(n_cam=7):
    """(n_cam, 4, 3) DLT matrices (`P.T`, `p_h @ M` convention, matching
    `ReprojectionTool.camera_matrices`), row 3 = [0,0,0,1] (affine)."""
    cams = []
    for i in range(n_cam):
        ax = 2 * np.pi * i / n_cam           # rotation about x
        ay = 4 * np.pi * i / n_cam           # rotation about y, different rate
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(ax), -np.sin(ax)],
                       [0, np.sin(ax), np.cos(ax)]])
        Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                       [0, 1, 0],
                       [-np.sin(ay), 0, np.cos(ay)]])
        P = _P_REAL.copy()
        P[:2, :3] = P[:2, :3] @ (Ry @ Rx)
        cams.append(P.T.astype(np.float32))
    return np.stack(cams)


def _project_all(cam_mats, X):
    """(3,) world point -> (C, 2) full-image px on every camera (same
    `p_h @ M`, perspective-divide convention as `ReprojectionTool.reproject_point`)."""
    h = np.append(np.asarray(X, np.float64), 1.0)
    out = [ph[:2] / ph[2] for ph in (h @ M for M in cam_mats)]
    return np.asarray(out, np.float32)


# ---------------------------------------------------------------------------
# lift_peaks_to_centres
# ---------------------------------------------------------------------------

def test_lift_two_flies_from_perfect_peaks():
    cam_mats = _affine_rig()
    X = np.array([[0.0, 0.0, 0.0], [40.0, 5.0, 0.0]])
    peaks = np.stack([_project_all(cam_mats, x) for x in X], axis=1)  # (C, fly, xy)
    peaks = (peaks.astype(np.float32)
             + np.random.default_rng(0).normal(scale=1.0, size=peaks.shape).astype(np.float32))
    scores = np.ones((7, 2), np.float32)

    c, nv, sc = lift_peaks_to_centres(peaks, scores, cam_mats)

    assert c.shape == (2, 3)
    order = np.argsort(c[:, 0])
    np.testing.assert_allclose(c[order], X, atol=1.0)
    assert (nv[order] == 7).all()
    assert np.all(sc[order] > 0)


def test_lift_tolerates_missing_and_swapped_peaks():
    cam_mats = _affine_rig()
    X = np.array([[0.0, 0.0, 0.0], [40.0, 5.0, 0.0]])
    peaks = np.stack([_project_all(cam_mats, x) for x in X], axis=1).astype(np.float32)

    # Drop fly 1's peak in 3 cameras (NaN).
    dropped_cams = [0, 2, 4]
    for c_idx in dropped_cams:
        peaks[c_idx, 1] = np.nan
    # Swap the peak order in 2 (different, still-present) cameras.
    for c_idx in [1, 3]:
        peaks[c_idx, [0, 1]] = peaks[c_idx, [1, 0]]

    scores = np.ones((7, 2), np.float32)
    for c_idx in dropped_cams:
        scores[c_idx, 1] = np.nan

    c, nv, sc = lift_peaks_to_centres(peaks, scores, cam_mats)

    assert c.shape == (2, 3)
    order = np.argsort(c[:, 0])
    np.testing.assert_allclose(c[order], X, atol=1.0)
    assert sorted(nv[order].tolist()) == [4, 7]


def test_lift_rejects_inconsistent_peaks():
    cam_mats = _affine_rig()
    X = np.array([0.0, 0.0, 0.0])
    peaks_single = _project_all(cam_mats, X)  # (C, 2) -- all cams, one point
    # One camera's peak moved by 80 px -- a geometric outlier.
    bad_cam = 3
    peaks_single[bad_cam] += np.array([80.0, 0.0], np.float32)

    # Feed as the "first" peak slot; second slot is empty (no second fly).
    peaks = np.stack([peaks_single, np.full_like(peaks_single, np.nan)], axis=1)
    scores = np.stack([np.ones(7, np.float32), np.full(7, np.nan, np.float32)], axis=1)

    c, nv, sc = lift_peaks_to_centres(peaks, scores, cam_mats)

    assert c.shape == (2, 3)
    np.testing.assert_allclose(c[0], X, atol=1.0)
    assert nv[0] == 6
    assert np.all(np.isnan(c[1]))


def test_single_fly_and_min_views():
    cam_mats = _affine_rig()
    X = np.array([0.0, 0.0, 0.0])
    peaks_single = _project_all(cam_mats, X)  # (C, 2) -- all cams, one point
    peaks = np.stack([peaks_single, np.full_like(peaks_single, np.nan)], axis=1)
    scores = np.stack([np.ones(7, np.float32), np.full(7, np.nan, np.float32)], axis=1)

    c, nv, sc = lift_peaks_to_centres(peaks, scores, cam_mats)
    assert c.shape == (2, 3)
    np.testing.assert_allclose(c[0], X, atol=1.0)
    assert np.all(np.isnan(c[1]))

    # Only 2 cameras have a peak at all -> below min_views=3 -> no centre.
    peaks_sparse = np.full_like(peaks, np.nan)
    scores_sparse = np.full_like(scores, np.nan)
    peaks_sparse[[0, 1], 0] = peaks_single[[0, 1]]
    scores_sparse[[0, 1], 0] = 1.0

    c2, nv2, sc2 = lift_peaks_to_centres(peaks_sparse, scores_sparse, cam_mats)
    assert c2.shape == (2, 3)
    assert np.all(np.isnan(c2))


def test_lift_peaks_to_centres_rejects_camera_axis_mismatch():
    """`peaks`/`cam_mats` disagreeing on camera COUNT is the one camera-order
    mismatch this function can catch cheaply (module docstring's contract)."""
    cam_mats = _affine_rig(n_cam=7)
    peaks = np.zeros((5, 2, 2), np.float32)   # wrong camera axis length
    scores = np.ones((5, 2), np.float32)
    with pytest.raises(ValueError):
        lift_peaks_to_centres(peaks, scores, cam_mats)


# ---------------------------------------------------------------------------
# cluster_centres / plan_windows
# ---------------------------------------------------------------------------

def test_cluster_and_plan_windows():
    c = np.array([[0, 0, 0], [8.0, 0, 0]], np.float32)  # 0.8 mm apart -> one cluster
    cc = cluster_centres(c)
    assert cc.shape == (2, 3)
    assert np.isnan(cc[1]).all()
    assert np.allclose(cc[0], [4, 0, 0])

    w, a = plan_windows(np.array([[0, 0, 0], [20.0, 0, 0]], np.float32))  # 2mm -> shared window
    assert w.shape == (1, 3)
    assert np.allclose(w[0], [10, 0, 0])
    assert a.tolist() == [0, 0]

    w, a = plan_windows(np.array([[0, 0, 0], [50.0, 0, 0]], np.float32))  # 5mm -> two windows
    assert w.shape == (2, 3)
    assert a.tolist() == [0, 1]


def test_cluster_centres_no_merge_when_far_apart():
    c = np.array([[0, 0, 0], [200.0, 0, 0]], np.float32)  # 20 mm apart
    cc = cluster_centres(c)
    assert cc.shape == (2, 3)
    np.testing.assert_allclose(cc, c)


# ---------------------------------------------------------------------------
# _peaks_from_heatmap (pure post-processing; no checkpoint/GPU involved)
# ---------------------------------------------------------------------------

def _gaussian_heatmap(h, w, centres_xy, sigma=3.0, amp=1.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    hm = np.zeros((h, w), np.float32)
    for (cx, cy), a in zip(centres_xy, amp if hasattr(amp, "__len__") else [amp] * len(centres_xy)):
        hm += a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return hm


def test_peaks_from_heatmap_two_blobs_anisotropic_scale():
    h = w = 160
    img_w, img_h = 1936, 448  # strongly anisotropic full-frame size
    centres = [(40.0, 60.0), (110.0, 90.0)]
    hm = np.stack([_gaussian_heatmap(h, w, centres, sigma=3.0, amp=[1.0, 0.9])])

    peaks, scores = _peaks_from_heatmap(hm, img_w, img_h, min_score=0.2)

    assert peaks.shape == (1, 2, 2)
    assert scores.shape == (1, 2)
    sx, sy = img_w / w, img_h / h
    expected = np.array([[centres[0][0] * sx, centres[0][1] * sy],
                         [centres[1][0] * sx, centres[1][1] * sy]])
    np.testing.assert_allclose(peaks[0], expected, atol=2.0)
    assert np.all(scores[0] > 0.2)


def test_peaks_from_heatmap_below_min_score_is_nan():
    h = w = 160
    img_w, img_h = 1936, 448
    # Only ONE real blob; the second "peak" returned by top-k is background,
    # deliberately kept dim (amp well under min_score).
    hm = np.stack([_gaussian_heatmap(h, w, [(80.0, 80.0)], sigma=3.0, amp=[1.0])])
    hm += 0.05  # uniform low background so the 2nd NMS peak has low but nonzero score

    peaks, scores = _peaks_from_heatmap(hm, img_w, img_h, min_score=0.5)

    assert scores[0, 0] > 0.5
    assert np.all(np.isfinite(peaks[0, 0]))
    assert scores[0, 1] < 0.5
    assert np.all(np.isnan(peaks[0, 1]))


# ---------------------------------------------------------------------------
# _centerdetect_preprocess (PIL BILINEAR resize + ImageNet normalise)
# ---------------------------------------------------------------------------

def test_centerdetect_preprocess_shape_dtype_and_normalises():
    # 115/255 is close to the ImageNet per-channel mean (0.406-0.485), so a
    # flat frame at this value should normalise to ~0 -- if the mean/std
    # constants or their broadcasting were wrong (e.g. applied per PIXEL
    # instead of per CHANNEL), this would be the cheapest way to notice.
    frame = np.full((448, 1936, 3), 115, np.uint8)
    out = _centerdetect_preprocess(frame)
    assert out.shape == (320, 320, 3)
    assert out.dtype == np.float32
    assert abs(float(out.mean())) < 0.1
