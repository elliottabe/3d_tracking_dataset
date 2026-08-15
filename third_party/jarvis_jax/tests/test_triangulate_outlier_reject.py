"""Tests for reprojection-residual outlier rejection in triangulate_keypoints.

Motivation (docs/benchmark/2026-08-14-jax-vs-jarvis-stability/notes.md in the
parent repo): 75% of the 3D jitter spikes on the Session6 clip are a SINGLE
camera whose 2D swapped to the wrong leg at conf 0.4-0.8 -- high enough to
pass both confidence gates, so plain DLT drags the point. Confidence cannot
catch these ("wrong place, confidently"); the reprojection residual against
the other views can. `reproj_resid_px` iteratively drops the worst view while
its residual exceeds the threshold and >= 3 valid views remain (never below
2 views, which would trade a fixable point for a NaN).
"""
from __future__ import annotations

import numpy as np

from jarvis_jax.tracking.triangulate import triangulate_keypoints

from test_view_conf_gate import make_cams, project


def _one_point(cam_mats, pts_px, confs):
    """Wrap per-camera 2D of a single (frame, keypoint) into (1,C,1,*) arrays."""
    C = len(confs)
    kp2d = np.asarray(pts_px, np.float32).reshape(1, C, 1, 2)
    conf = np.asarray(confs, np.float32).reshape(1, C, 1)
    return kp2d, conf


def test_none_threshold_is_exact_noop():
    cam_mats, _ = make_cams(5)
    X = np.array([10.0, 5.0, 20.0])
    pts = project(cam_mats, X)
    pts[2] += 300.0                      # an outlier that rejection WOULD drop
    kp2d, conf = _one_point(cam_mats, pts, [0.9] * 5)
    plain, cp = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3)
    off, co = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3,
                                    reproj_resid_px=None)
    assert np.array_equal(plain, off, equal_nan=True)
    assert np.array_equal(cp, co)


def test_single_swapped_view_is_rejected():
    # 5 cameras, 4 agree on X, 1 swapped 300 px at HIGH confidence (the real
    # failure mode: confidence gates cannot catch it). Plain DLT is dragged;
    # rejection recovers X to well under the drag error.
    cam_mats, _ = make_cams(5)
    X = np.array([10.0, 5.0, 20.0])
    pts = project(cam_mats, X)
    pts[2] += 300.0
    kp2d, conf = _one_point(cam_mats, pts, [0.9] * 5)
    plain, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3)
    rej, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3,
                                   reproj_resid_px=12.0)
    err_plain = np.linalg.norm(plain[0, 0] - X)
    err_rej = np.linalg.norm(rej[0, 0] - X)
    assert err_plain > 1.0, "outlier should visibly drag the plain solution"
    assert err_rej < 0.1
    assert err_rej < err_plain / 10.0


def test_conf3d_averages_only_surviving_views():
    # The swapped view carries conf 0.2 (vs 0.9 for the good views, all above
    # conf_thresh). Once rejected it must not contaminate conf3d either.
    cam_mats, _ = make_cams(5)
    X = np.array([10.0, 5.0, 20.0])
    pts = project(cam_mats, X)
    pts[2] += 300.0
    kp2d, conf = _one_point(cam_mats, pts, [0.9, 0.9, 0.2, 0.9, 0.9])
    _, c3d = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.1,
                                   reproj_resid_px=12.0)
    assert np.isclose(c3d[0, 0], 0.9, atol=1e-6)


def test_two_outlier_views_both_rejected_iteratively():
    cam_mats, _ = make_cams(6)
    X = np.array([-8.0, 12.0, 15.0])
    pts = project(cam_mats, X)
    pts[1] += 250.0
    pts[4] -= 180.0
    kp2d, conf = _one_point(cam_mats, pts, [0.9] * 6)
    rej, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3,
                                   reproj_resid_px=12.0)
    assert np.linalg.norm(rej[0, 0] - X) < 0.1


def test_never_drops_below_two_views():
    # Exactly 2 valid views that disagree wildly: rejection must NOT run (it
    # cannot tell which is wrong) and the point stays finite, identical to
    # the plain solve.
    cam_mats, _ = make_cams(4)
    X = np.array([3.0, -4.0, 10.0])
    pts = project(cam_mats, X)
    pts[1] += 400.0
    kp2d, conf = _one_point(cam_mats, pts, [0.9, 0.9, 0.0, 0.0])
    plain, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3)
    rej, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3,
                                   reproj_resid_px=12.0)
    assert np.isfinite(rej[0, 0]).all()
    assert np.allclose(rej, plain, equal_nan=True)


def test_consistent_views_untouched():
    # All views agree: residuals sit at the numerical floor, far below the
    # threshold, so rejection must return exactly the plain solution.
    cam_mats, _ = make_cams(5)
    X = np.array([1.0, 2.0, 30.0])
    pts = project(cam_mats, X)
    kp2d, conf = _one_point(cam_mats, pts, [0.9] * 5)
    plain, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3)
    rej, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3,
                                   reproj_resid_px=12.0)
    assert np.allclose(rej, plain, atol=1e-5)


def test_moderate_inconsistency_below_trigger_left_alone():
    # Anti-churn guarantee (benchmark b12-fly1 regression, notes.md): diffuse
    # 10-30 px disagreement -- detector noise on hard poses, not a swap --
    # must NOT trigger intervention. Rejection engages only when some view is
    # grossly (3x band) inconsistent; otherwise the result is exactly plain.
    cam_mats, _ = make_cams(5)
    X = np.array([10.0, 5.0, 20.0])
    pts = project(cam_mats, X)
    pts += np.array([[8.0, -6.0], [-7.0, 8.0], [9.0, 5.0], [-5.0, -9.0],
                     [6.0, 7.0]])                 # ~10 px diffuse noise
    kp2d, conf = _one_point(cam_mats, pts, [0.9] * 5)
    plain, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3)
    rej, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3,
                                   reproj_resid_px=12.0)
    assert np.allclose(rej, plain, equal_nan=True)


def test_two_view_consensus_refused():
    # 4 views in two 2-2 clusters far apart: no 3-view consensus exists, and a
    # 2-view consensus would LOCK ONTO one cluster arbitrarily (measured on
    # benchmark female b8: min-2 consensus made spikes 25% WORSE). With no
    # >= 3-view consensus the point keeps all views -- same as plain.
    cam_mats, _ = make_cams(4)
    X = np.array([3.0, -4.0, 10.0])
    pts = project(cam_mats, X)
    pts[2] += 200.0
    pts[3] += 200.0
    kp2d, conf = _one_point(cam_mats, pts, [0.9] * 4)
    plain, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3)
    rej, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3,
                                   reproj_resid_px=12.0)
    assert np.allclose(rej, plain, equal_nan=True)


def test_batched_frames_and_keypoints_reject_independently():
    # (T=2, K=2): only frame 0 / keypoint 1 has the swapped view; every other
    # (frame, keypoint) must be byte-identical to the plain solve.
    cam_mats, _ = make_cams(5)
    Xs = {(0, 0): [10.0, 5.0, 20.0], (0, 1): [-6.0, 7.0, 18.0],
          (1, 0): [2.0, -9.0, 25.0], (1, 1): [0.0, 0.0, 12.0]}
    kp2d = np.zeros((2, 5, 2, 2), np.float32)
    for (t, k), X in Xs.items():
        kp2d[t, :, k] = project(cam_mats, X)
    kp2d[0, 3, 1] += 300.0                       # the one swapped view
    conf = np.full((2, 5, 2), 0.9, np.float32)
    plain, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3)
    rej, _ = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3,
                                   reproj_resid_px=12.0)
    assert np.linalg.norm(rej[0, 1] - Xs[(0, 1)]) < 0.1
    for (t, k) in [(0, 0), (1, 0), (1, 1)]:
        assert np.allclose(rej[t, k], plain[t, k], atol=1e-5), (t, k)
