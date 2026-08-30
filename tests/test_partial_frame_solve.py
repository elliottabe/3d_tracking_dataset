import numpy as np
import pytest

from scripts.run_bout import finite_frame_mask, contiguous_segments, NAN_SOLVE_MIN_SEG


def _kp(T=100, K=50, missing=()):
    kp = np.random.default_rng(0).normal(size=(T, K, 3)).astype(np.float32)
    for t, joints in missing:
        kp[t, list(joints)] = np.nan
    return kp


def test_default_is_byte_identical_to_all_or_nothing():
    """The shipped behaviour must not move unless explicitly opted in."""
    kp = _kp(missing=[(5, [0]), (6, [1, 2])])
    np.testing.assert_array_equal(finite_frame_mask(kp),
                                  finite_frame_mask(kp, min_keypoints=None))
    assert not finite_frame_mask(kp)[5]


def test_min_keypoints_accepts_a_partial_frame():
    """A frame missing 3 of 50 keypoints is usable; today it is discarded."""
    kp = _kp(missing=[(5, [0, 1, 2])])
    assert finite_frame_mask(kp, min_keypoints=40)[5]
    assert not finite_frame_mask(kp, min_keypoints=50)[5]


def test_min_keypoints_still_rejects_a_mostly_empty_frame():
    kp = _kp(missing=[(5, range(45))])
    assert not finite_frame_mask(kp, min_keypoints=40)[5]


def test_partial_frames_can_form_a_solvable_segment():
    """The bout-28 failure in miniature: scattered complete frames never reach
    NAN_SOLVE_MIN_SEG, but the same frames counted partially do."""
    kp = _kp(T=100)
    for t in range(100):
        if t % 3:                       # 2 of every 3 frames lose 3 keypoints
            kp[t, [0, 1, 2]] = np.nan
    assert contiguous_segments(finite_frame_mask(kp), NAN_SOLVE_MIN_SEG) == []
    segs = contiguous_segments(finite_frame_mask(kp, min_keypoints=40),
                               NAN_SOLVE_MIN_SEG)
    assert segs and (segs[0][1] - segs[0][0]) >= NAN_SOLVE_MIN_SEG


def test_marker_mask_matches_the_accepted_frames():
    """Every keypoint the solver is told to ignore must be exactly the NaN ones --
    an off-by-one here silently drops real observations."""
    from scripts.run_bout import marker_validity_mask
    kp = _kp(missing=[(5, [0, 1, 2]), (7, [9])])
    m = marker_validity_mask(kp)
    assert m.shape == kp.shape[:2]
    assert not m[5, 0] and not m[5, 1] and not m[5, 2] and m[5, 3]
    assert not m[7, 9] and m[7, 8]
    np.testing.assert_array_equal(m, np.isfinite(kp).all(axis=-1))
