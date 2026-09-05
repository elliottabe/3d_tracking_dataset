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
"""
import numpy as np
import pytest

from mvq_fixtures import make_v12_root

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.tracking.coarse_centres import (
    _peaks_from_heatmap,
    lift_peaks_to_centres,
    cluster_centres,
    plan_windows,
)


# ---------------------------------------------------------------------------
# lift_peaks_to_centres
# ---------------------------------------------------------------------------

def test_lift_two_flies_from_perfect_peaks(tmp_path):
    root = make_v12_root(tmp_path)
    rt = ReprojectionTool(f"{root}/calibrations/A")
    X = np.array([[0.0, 0.0, 0.0], [40.0, 5.0, 0.0]])
    # rt.reproject_point(x) is (num_cam, 2) -- ALL cameras for one 3D point;
    # stacking over the 2 flies on axis=1 gives (C, 2, 2) = (camera, fly, xy).
    peaks = np.stack([rt.reproject_point(x) for x in X], axis=1)
    peaks = (peaks.astype(np.float32)
             + np.random.default_rng(0).normal(scale=1.0, size=peaks.shape).astype(np.float32))
    scores = np.ones((7, 2), np.float32)

    c, nv, sc = lift_peaks_to_centres(peaks, scores, rt.camera_matrices)

    assert c.shape == (2, 3)
    order = np.argsort(c[:, 0])
    # `make_v12_root`'s 7 cameras are ALL the same base affine camera rotated
    # by Rz only (see cam_P): Rz's third column is [0,0,1], so every rotated
    # camera keeps the SAME world-Z coefficient -- world Z is (near-)shared
    # across all 7 views' image-plane sensitivity, exactly the "camera
    # centers share the rotation axis -> depth along it is unrecoverable"
    # defect `tests/test_triangulate_refine.py::_rig`'s docstring documents
    # for this same fixture family (confirmed here: SVD of the assembled DLT
    # system has singular values [21.4, 21.4, 0.48, 0] -- condition number
    # ~44x on the weak direction, which world Z lies along since the truth
    # points are Z=0). So 1px image noise recovers world X/Y (the
    # well-conditioned plane) to ~0.1 units but can throw world Z off by
    # 10s-100s of units -- a real fixture limitation, not an algorithm bug
    # (confirmed: even the raw multi-view `triangulate_dlt_batched` call,
    # bypassing this module entirely, reproduces the same Z blowup on
    # noiseless-vs-noisy inputs). Check X/Y tightly and Z only loosely.
    np.testing.assert_allclose(c[order][:, :2], X[:, :2], atol=1.0)
    assert np.abs(c[order][:, 2]).max() < 100.0
    assert (nv[order] >= 6).all()
    assert np.all(sc[order] > 0)


def test_lift_tolerates_missing_and_swapped_peaks(tmp_path):
    root = make_v12_root(tmp_path)
    rt = ReprojectionTool(f"{root}/calibrations/A")
    X = np.array([[0.0, 0.0, 0.0], [40.0, 5.0, 0.0]])
    peaks = np.stack([rt.reproject_point(x) for x in X], axis=1).astype(np.float32)

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

    c, nv, sc = lift_peaks_to_centres(peaks, scores, rt.camera_matrices)

    assert c.shape == (2, 3)
    order = np.argsort(c[:, 0])
    np.testing.assert_allclose(c[order], X, atol=1.5)
    assert sorted(nv[order].tolist()) == [4, 7]


def test_lift_rejects_inconsistent_peaks(tmp_path):
    root = make_v12_root(tmp_path)
    rt = ReprojectionTool(f"{root}/calibrations/A")
    X = np.array([0.0, 0.0, 0.0])
    peaks_single = rt.reproject_point(X).astype(np.float32)  # (C, 2) -- all cams, one point
    # One camera's peak moved by 80 px -- a geometric outlier.
    bad_cam = 3
    peaks_single[bad_cam] += np.array([80.0, 0.0], np.float32)

    # Feed as the "first" peak slot; second slot is empty (no second fly).
    peaks = np.stack([peaks_single, np.full_like(peaks_single, np.nan)], axis=1)
    scores = np.stack([np.ones(7, np.float32), np.full(7, np.nan, np.float32)], axis=1)

    c, nv, sc = lift_peaks_to_centres(peaks, scores, rt.camera_matrices)

    assert c.shape == (2, 3)
    np.testing.assert_allclose(c[0], X, atol=1.0)
    assert nv[0] == 6
    assert np.all(np.isnan(c[1]))


def test_single_fly_and_min_views(tmp_path):
    root = make_v12_root(tmp_path)
    rt = ReprojectionTool(f"{root}/calibrations/A")
    X = np.array([0.0, 0.0, 0.0])
    peaks_single = rt.reproject_point(X).astype(np.float32)  # (C, 2) -- all cams, one point
    peaks = np.stack([peaks_single, np.full_like(peaks_single, np.nan)], axis=1)
    scores = np.stack([np.ones(7, np.float32), np.full(7, np.nan, np.float32)], axis=1)

    c, nv, sc = lift_peaks_to_centres(peaks, scores, rt.camera_matrices)
    assert c.shape == (2, 3)
    np.testing.assert_allclose(c[0], X, atol=1.0)
    assert np.all(np.isnan(c[1]))

    # Only 2 cameras have a peak at all -> below min_views=3 -> no centre.
    peaks_sparse = np.full_like(peaks, np.nan)
    scores_sparse = np.full_like(scores, np.nan)
    peaks_sparse[[0, 1], 0] = peaks_single[[0, 1]]
    scores_sparse[[0, 1], 0] = 1.0

    c2, nv2, sc2 = lift_peaks_to_centres(peaks_sparse, scores_sparse, rt.camera_matrices)
    assert c2.shape == (2, 3)
    assert np.all(np.isnan(c2))


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
