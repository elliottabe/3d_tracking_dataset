"""Tests for SAM3 visibility bookkeeping + mid-bout gap repair.

`valid=False` conflates "outside this camera's field of view" (nothing to fix)
with "in frame and SAM3 missed it" (fixable). `in_frame_codes` separates them;
`find_gap_cameras` picks the fixable ones worth a re-segmentation pass;
`_merge_fill_camera` fills the holes WITHOUT discarding masks SAM3 already got
right (unlike the outlier repair, which replaces a camera wholesale because its
masks are wrong).

Root cause being repaired: SAM3VideoTracker prompts at frame_index 0 and
propagates 'forward' only, so a fly entering a camera's view mid-bout is never
picked up there. See docs and configs/sam3/default.yaml.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis_jax.predict.sam3_driver import (
    IN_FRAME_NO,
    IN_FRAME_UNKNOWN,
    IN_FRAME_YES,
    _merge_fill_camera,
    find_gap_cameras,
    in_frame_codes,
)


class FakeRepro:
    """Reprojection stub: '3D' is the mean of the given 2D observations, and
    every camera sees it unchanged. Enough to exercise the bounds logic without
    a calibration."""

    def __init__(self, n_cam, offsets=None):
        self.n_cam = n_cam
        self.offsets = offsets if offsets is not None else np.zeros((n_cam, 2))

    def reconstruct_point(self, points2d, cams_to_use=None):
        cams = list(range(self.n_cam)) if cams_to_use is None else list(cams_to_use)
        pts = np.asarray(points2d, float)[cams]
        m = pts.mean(axis=0)
        return np.array([m[0], m[1], 0.0])

    def reproject_point(self, p3d):
        p = np.asarray(p3d, float)[:2]
        return np.stack([p + self.offsets[k] for k in range(self.n_cam)])


# ---------------------------------------------------------------------------
# in_frame_codes
# ---------------------------------------------------------------------------

def test_valid_views_are_in_frame_by_definition():
    A, C, T = 1, 3, 4
    val = np.ones((A, C, T), bool)
    cent = np.full((A, C, T, 2), 50.0)
    codes = in_frame_codes(cent, val, FakeRepro(C), W=100, H=100)
    assert (codes == IN_FRAME_YES).all()


def test_missing_view_inside_bounds_is_a_recoverable_miss():
    A, C, T = 1, 3, 1
    val = np.ones((A, C, T), bool)
    val[0, 2, 0] = False                       # cam2 missing
    cent = np.full((A, C, T, 2), 50.0)         # ...but the fly is at (50,50)
    codes = in_frame_codes(cent, val, FakeRepro(C), W=100, H=100)
    assert codes[0, 2, 0] == IN_FRAME_YES


def test_missing_view_outside_bounds_is_out_of_fov():
    A, C, T = 1, 3, 1
    val = np.ones((A, C, T), bool)
    val[0, 2, 0] = False
    cent = np.full((A, C, T, 2), 50.0)
    # cam2 is offset far away, so the fly reprojects off its sensor.
    off = np.zeros((C, 2)); off[2] = [500.0, 0.0]
    codes = in_frame_codes(cent, val, FakeRepro(C, off), W=100, H=100)
    assert codes[0, 2, 0] == IN_FRAME_NO


def test_fewer_than_two_other_cameras_is_unknown_not_a_guess():
    A, C, T = 1, 3, 1
    val = np.zeros((A, C, T), bool)
    val[0, 0, 0] = True                        # only ONE valid camera
    cent = np.full((A, C, T, 2), 50.0)
    codes = in_frame_codes(cent, val, FakeRepro(C), W=100, H=100)
    assert codes[0, 0, 0] == IN_FRAME_YES      # the valid one
    assert codes[0, 1, 0] == IN_FRAME_UNKNOWN
    assert codes[0, 2, 0] == IN_FRAME_UNKNOWN


def test_a_valid_camera_is_excluded_from_its_own_reprojection():
    # cam0 valid -> YES without consulting others; the point is that the other
    # cameras' codes are computed from sources EXCLUDING themselves.
    A, C, T = 1, 4, 1
    val = np.ones((A, C, T), bool); val[0, 3, 0] = False
    cent = np.full((A, C, T, 2), 10.0)
    cent[0, 3, 0] = [999.0, 999.0]             # garbage in the invalid view
    codes = in_frame_codes(cent, val, FakeRepro(C), W=100, H=100)
    # cam3's own (garbage) centroid must not be used -> still resolves in-frame
    assert codes[0, 3, 0] == IN_FRAME_YES


def test_in_frame_codes_rejects_wrong_shapes():
    with pytest.raises(ValueError, match=r"\(A,C,T,2\)"):
        in_frame_codes(np.zeros((2, 3, 4)), np.zeros((2, 3, 4), bool),
                       FakeRepro(3), 10, 10)


def test_bout22_shape_three_cameras_see_her_four_do_not():
    # The real Session0 bout 22 geometry: 3 cameras have her, 4 are offset far
    # enough that she is off-sensor -> those must read out-of-FOV, not "missed".
    A, C, T = 1, 7, 5
    val = np.zeros((A, C, T), bool); val[0, :3] = True
    cent = np.full((A, C, T, 2), 40.0)
    off = np.zeros((C, 2)); off[3:] = [900.0, 0.0]
    codes = in_frame_codes(cent, val, FakeRepro(C, off), W=200, H=100)
    assert (codes[0, :3] == IN_FRAME_YES).all()
    assert (codes[0, 3:] == IN_FRAME_NO).all()


# ---------------------------------------------------------------------------
# find_gap_cameras
# ---------------------------------------------------------------------------

def test_gap_found_when_in_frame_but_invalid():
    A, C, T = 1, 3, 100
    val = np.ones((A, C, T), bool); val[0, 1, :50] = False
    codes = np.full((A, C, T), IN_FRAME_YES, np.int8)
    gaps = find_gap_cameras(val, codes, min_frames=30, min_frac=0.02)
    assert gaps == [(0, 1, 50)]


def test_out_of_fov_frames_are_not_a_gap():
    # This is the distinction that matters: an unobservable fly must never
    # trigger a (pointless, GPU-expensive) re-segmentation pass.
    A, C, T = 1, 3, 100
    val = np.ones((A, C, T), bool); val[0, 1, :50] = False
    codes = np.full((A, C, T), IN_FRAME_YES, np.int8)
    codes[0, 1, :50] = IN_FRAME_NO
    assert find_gap_cameras(val, codes, min_frames=10, min_frac=0.0) == []


def test_unknown_frames_are_not_a_gap():
    A, C, T = 1, 3, 100
    val = np.ones((A, C, T), bool); val[0, 1, :50] = False
    codes = np.full((A, C, T), IN_FRAME_YES, np.int8)
    codes[0, 1, :50] = IN_FRAME_UNKNOWN
    assert find_gap_cameras(val, codes, min_frames=10, min_frac=0.0) == []


def test_small_gaps_are_ignored_by_both_thresholds():
    A, C, T = 1, 2, 1000
    val = np.ones((A, C, T), bool)
    val[0, 1, :20] = False                     # 20 frames: below min_frames=30
    codes = np.full((A, C, T), IN_FRAME_YES, np.int8)
    assert find_gap_cameras(val, codes, min_frames=30, min_frac=0.0) == []
    val2 = np.ones((A, C, T), bool)
    val2[0, 1, :15] = False                    # 1.5% of the bout: below min_frac
    assert find_gap_cameras(val2, codes, min_frames=10, min_frac=0.02) == []


def test_gaps_sorted_largest_first():
    A, C, T = 2, 3, 200
    val = np.ones((A, C, T), bool)
    val[0, 1, :40] = False
    val[1, 2, :90] = False
    codes = np.full((A, C, T), IN_FRAME_YES, np.int8)
    gaps = find_gap_cameras(val, codes, min_frames=30, min_frac=0.0)
    assert [g[2] for g in gaps] == [90, 40]
    assert gaps[0][:2] == (1, 2)


def test_find_gap_cameras_shape_mismatch_raises():
    with pytest.raises(ValueError, match="!="):
        find_gap_cameras(np.ones((1, 2, 3), bool), np.ones((1, 2, 4), np.int8))


# ---------------------------------------------------------------------------
# _merge_fill_camera
# ---------------------------------------------------------------------------

class FakeBM:
    def __init__(self, n_cam, T, masks, identity_map):
        self.num_cameras = n_cam
        self.num_frames = T
        self.masks = masks
        self.identity_map = identity_map


def _m(val=True):
    a = np.zeros((4, 4), bool)
    if val:
        a[1:3, 1:3] = True
    return a


def test_merge_fill_preserves_existing_masks():
    # The critical difference from repair_outlier_cameras, which replaces a
    # camera wholesale: a gap fill must not discard frames SAM3 got right.
    T = 4
    original = _m()
    masks = [[{7: {"mask": original, "centroid": np.array([1.5, 1.5]),
                   "score": 0.9}} if t < 2 else {} for t in range(T)]]
    bm = FakeBM(1, T, masks, [{7: 0}])
    filled = {0: [_m() for _ in range(T)]}
    n_added = _merge_fill_camera(bm, 0, num_animals=1, T=T, filled=filled)
    assert n_added == 2                        # only the two empty frames
    for t in range(2):
        assert bm.masks[0][t][0]["mask"] is original     # untouched
        assert bm.masks[0][t][0]["score"] == 0.9
    for t in (2, 3):
        assert 0 in bm.masks[0][t]


def test_merge_fill_rekeys_identity_map_to_fly_index():
    T = 2
    masks = [[{7: {"mask": _m(), "centroid": np.array([1.5, 1.5]), "score": 1.0}}
              for _ in range(T)]]
    bm = FakeBM(1, T, masks, [{7: 0}])
    _merge_fill_camera(bm, 0, num_animals=1, T=T, filled={})
    assert bm.identity_map[0] == {0: 0}
    assert 0 in bm.masks[0][0]                 # re-keyed from obj_id 7 to fly 0


def test_merge_fill_ignores_empty_candidate_masks():
    T = 3
    bm = FakeBM(1, T, [[{} for _ in range(T)]], [{}])
    filled = {0: [np.zeros((4, 4), bool), None, _m()]}
    n_added = _merge_fill_camera(bm, 0, num_animals=1, T=T, filled=filled)
    assert n_added == 1                        # only the non-empty one
    assert bm.masks[0][2][0]["mask"].any()


def test_merge_fill_drops_flies_beyond_num_animals():
    T = 1
    masks = [[{7: {"mask": _m(), "centroid": np.array([1.5, 1.5]), "score": 1.0},
               9: {"mask": _m(), "centroid": np.array([2.5, 2.5]), "score": 1.0}}]]
    bm = FakeBM(1, T, masks, [{7: 0, 9: 5}])   # obj 9 maps outside num_animals
    _merge_fill_camera(bm, 0, num_animals=2, T=T, filled={})
    assert set(bm.masks[0][0]) == {0}
