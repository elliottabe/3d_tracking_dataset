"""Tests for the per-view confidence gate in triangulate_keypoints.

The detector fills all K keypoint channels for any crop it is handed, including
one containing no fly, and reports high PER-KEYPOINT confidence while doing so.
So `conf_thresh` alone cannot keep a misplaced crop out of the DLT, where it
produces a confident but WRONG 3D point -- worse than dropping the view, since
a NaN is visibly missing downstream and a plausible wrong point is not.
`view_conf_thresh` gates whole (frame, camera) views on their median keypoint
confidence. See docs/benchmark/2026-08-08-detector-v4-retrain.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis_jax.tracking.triangulate import triangulate_keypoints, view_median_conf


def make_cams(n=4, radius=500.0, height=120.0):
    """n cameras ringed around the origin, as (n,4,3) so `p_h @ M` -> image coords.

    Each camera gets a DISTINCT centre via a look-at at the origin. (Rotating a
    fixed translation about z instead would put every camera at the same centre
    -- C = -R^T t is invariant when t lies on the rotation axis -- which is
    degenerate: nothing triangulates and every assertion below would compare
    two meaningless numbers.)
    """
    rng = np.random.default_rng(0)
    K = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]])
    mats = []
    for i in range(n):
        ang = 2 * np.pi * i / n
        C = np.array([radius * np.cos(ang), radius * np.sin(ang), height])
        f = -C / np.linalg.norm(C)                      # look at the origin
        right = np.cross(f, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        R = np.stack([right, up, f])                    # world -> camera
        P = np.hstack([R, (-R @ C)[:, None]])           # (3,4)
        mats.append((K @ P).T.astype(np.float32))       # (4,3)
    return np.stack(mats), rng


def project(cam_mats, X):
    """X (3,) -> (C,2) pixel coords under the (C,4,3) matrices."""
    Xh = np.concatenate([np.asarray(X, np.float64), [1.0]])
    out = []
    for M in cam_mats:
        p = Xh @ np.asarray(M, np.float64)
        out.append(p[:2] / p[2])
    return np.stack(out)


# ---------------------------------------------------------------------------
# view_median_conf
# ---------------------------------------------------------------------------

def test_view_median_conf_shape_and_value():
    conf = np.zeros((2, 3, 5), np.float32)
    conf[0, 0] = [0.1, 0.2, 0.3, 0.9, 0.9]      # median 0.3
    conf[1, 2] = [1.0, 1.0, 1.0, 1.0, 1.0]      # median 1.0
    out = view_median_conf(conf)
    assert out.shape == (2, 3)
    assert np.isclose(out[0, 0], 0.3)
    assert np.isclose(out[1, 2], 1.0)


def test_view_median_conf_is_robust_to_a_few_occluded_keypoints():
    # Median, not mean: a handful of genuinely occluded keypoints must not
    # drag an otherwise-confident view below the gate.
    conf = np.full((1, 1, 50), 0.95, np.float32)
    conf[0, 0, :8] = 0.0
    assert view_median_conf(conf)[0, 0] >= 0.9


def test_view_median_conf_rejects_wrong_rank():
    with pytest.raises(ValueError, match="T,C,K"):
        view_median_conf(np.zeros((3, 4), np.float32))


# ---------------------------------------------------------------------------
# gating behaviour
# ---------------------------------------------------------------------------

def test_gate_disabled_by_default_reproduces_old_behaviour():
    cam_mats, _ = make_cams(4)
    X = np.array([10.0, 5.0, 20.0])
    pts = project(cam_mats, X)
    kp2d = pts.reshape(1, 4, 1, 2).astype(np.float32)
    conf = np.full((1, 4, 1), 0.4, np.float32)      # above conf_thresh, below 0.6
    a, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3)
    b, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3,
                                 view_conf_thresh=None)
    assert np.allclose(a, b, equal_nan=True)
    assert np.isfinite(a).all()


def test_gate_drops_a_low_confidence_view():
    # 3 good views agreeing on X, 1 view with a WRONG point but per-keypoint
    # confidence above conf_thresh. Ungated it corrupts the solution; gated it
    # is excluded and the remaining 3 recover X.
    cam_mats, _ = make_cams(4)
    X = np.array([10.0, 5.0, 20.0])
    pts = project(cam_mats, X)
    kp2d = pts.reshape(1, 4, 1, 2).astype(np.float32).copy()
    kp2d[0, 3, 0] += np.array([120.0, -90.0], np.float32)    # bogus observation
    conf = np.full((1, 4, 1), 0.95, np.float32)
    conf[0, 3, 0] = 0.35                     # passes conf_thresh=0.3, fails 0.6

    ungated, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3)
    gated, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3,
                                     view_conf_thresh=0.6)
    err_ungated = np.linalg.norm(ungated[0, 0] - X)
    err_gated = np.linalg.norm(gated[0, 0] - X)
    assert err_gated < err_ungated
    assert err_gated < 1e-2


def test_gate_uses_the_view_median_not_individual_keypoints():
    # A view whose median is high keeps ALL its keypoints, including ones
    # individually below the view threshold (they still face conf_thresh).
    cam_mats, _ = make_cams(3)
    X = np.array([4.0, -3.0, 12.0])
    pts = project(cam_mats, X)
    kp2d = np.repeat(pts[None, :, None, :], 3, axis=2).astype(np.float32)  # K=3
    conf = np.full((1, 3, 3), 0.95, np.float32)
    conf[0, :, 1] = 0.45          # one keypoint low everywhere; median stays 0.95
    out, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3,
                                   view_conf_thresh=0.6)
    assert np.isfinite(out[0]).all(), "high-median views must keep all keypoints"


def test_gating_below_two_views_yields_nan_not_a_wrong_point():
    # The whole point of the gate: prefer a visible NaN to a confident wrong 3D.
    cam_mats, _ = make_cams(4)
    X = np.array([1.0, 2.0, 15.0])
    pts = project(cam_mats, X)
    kp2d = pts.reshape(1, 4, 1, 2).astype(np.float32)
    conf = np.full((1, 4, 1), 0.35, np.float32)     # every view below the gate
    conf[0, 0, 0] = 0.95                            # only one survives
    out, conf3d = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3,
                                        view_conf_thresh=0.6)
    assert np.isnan(out[0, 0]).all()
    assert conf3d[0, 0] == 0.0


def test_gate_of_zero_keeps_everything():
    cam_mats, _ = make_cams(3)
    X = np.array([0.0, 0.0, 10.0])
    pts = project(cam_mats, X)
    kp2d = pts.reshape(1, 3, 1, 2).astype(np.float32)
    conf = np.full((1, 3, 1), 0.31, np.float32)
    out, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3,
                                   view_conf_thresh=0.0)
    assert np.isfinite(out).all()


def test_calibrated_threshold_separates_measured_distributions():
    # The measured per-view medians (scripts/calibrate_view_gate.py, n=273 per
    # class): fly present ~0.963, empty ~0.379. The shipped 0.6 must sit
    # between them.
    cam_mats, _ = make_cams(4)
    X = np.array([6.0, 6.0, 18.0])
    pts = project(cam_mats, X)
    K = 50
    kp2d = np.repeat(pts[None, :, None, :], K, axis=2).astype(np.float32)
    conf = np.empty((1, 4, K), np.float32)
    conf[0, :2] = 0.963            # fly present
    conf[0, 2:] = 0.379            # empty crop
    med = view_median_conf(conf)[0]
    assert (med[:2] >= 0.6).all() and (med[2:] < 0.6).all()
    out, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3,
                                   view_conf_thresh=0.6)
    assert np.isfinite(out).all()  # the 2 surviving views still triangulate


def test_camera_fixture_is_not_degenerate():
    # Guards the trap above: with a common camera centre, DLT triangulation is
    # impossible and every assertion in this file would silently compare noise.
    cam_mats, _ = make_cams(4)
    centres = []
    for M in cam_mats:
        P = np.asarray(M, np.float64).T                 # (3,4)
        # C is the null space of P: P @ [C,1] = 0  ->  C = -inv(P[:,:3]) P[:,3]
        centres.append(-np.linalg.solve(P[:, :3], P[:, 3]))
    centres = np.stack(centres)
    dists = [np.linalg.norm(a - b) for i, a in enumerate(centres)
             for b in centres[i + 1:]]
    assert min(dists) > 1.0, f"cameras share a centre: {centres}"
