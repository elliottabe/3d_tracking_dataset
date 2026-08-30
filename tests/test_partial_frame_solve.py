import numpy as np
import pytest
from omegaconf import OmegaConf

from scripts.run_bout import finite_frame_mask, contiguous_segments, NAN_SOLVE_MIN_SEG


def _kp(T=100, K=50, missing=()):
    kp = np.random.default_rng(0).normal(size=(T, K, 3)).astype(np.float32)
    for t, joints in missing:
        kp[t, list(joints)] = np.nan
    return kp


# Fix round 1: the warm-start gap. finite_frame_mask/marker_validity_mask
# protect the SOLVER RESIDUAL (stac_core_jaxls.py's marker_cost is already
# NaN-safe per marker per frame), but stac_mjx/compute_stac.py's per-frame
# warm-start reads the root keypoint (line ~468) and the four orientation
# keypoints (_estimate_orientation_from_keypoints) directly out of kp_flat
# with NO finite check. A partial frame missing exactly the root or an
# orientation keypoint gets a NaN initial guess -- these tests pin down the
# additional gate that prevents that: `required_indices`.
_KP_NAMES = ["Head", "Scutellum", "WingL_base", "WingR_base", "Antenna_Base",
             "T1L_FeTi", "T1R_FeTi", "T2L_FeTi", "T2R_FeTi", "T3L_FeTi"]
_ROOT_IDX = _KP_NAMES.index("Scutellum")       # 1
_REAR_IDX = _KP_NAMES.index("Scutellum")       # 1 (rear == root here, as in v1.yaml)
_LEFT_IDX = _KP_NAMES.index("WingL_base")      # 2
_RIGHT_IDX = _KP_NAMES.index("WingR_base")     # 3
_FRONT_IDX = _KP_NAMES.index("Antenna_Base")   # 4


def _cfg():
    """Mirrors the real anatomy config keys stac_mjx.stac.Stac resolves its
    per-frame warm-start from (ROOT_OPTIMIZATION_KEYPOINT,
    JAXLS_ORIENTATION_KEYPOINTS), e.g. configs/anatomy/v1.yaml."""
    return OmegaConf.create({
        "model": {
            "KP_NAMES": list(_KP_NAMES),
            "ROOT_OPTIMIZATION_KEYPOINT": "Scutellum",
            "JAXLS_ORIENTATION_KEYPOINTS": {
                "rear": "Scutellum", "left": "WingL_base",
                "right": "WingR_base", "front": "Antenna_Base",
            },
        }
    })


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


def test_required_keypoint_indices_mirror_the_solver_warm_start_config():
    """required_keypoint_indices must resolve the SAME two anatomy-config keys
    (ROOT_OPTIMIZATION_KEYPOINT, JAXLS_ORIENTATION_KEYPOINTS) that
    stac_mjx.stac.Stac.__init__ resolves its warm-start indices from -- a
    hardcoded keypoint name here would silently diverge from the solver the
    moment the anatomy config changes which keypoint plays which role."""
    from scripts.run_bout import required_keypoint_indices
    idx = required_keypoint_indices(_cfg(), _KP_NAMES)
    assert idx == sorted({_ROOT_IDX, _REAR_IDX, _LEFT_IDX, _RIGHT_IDX, _FRONT_IDX})


def test_partial_frame_missing_root_keypoint_is_rejected():
    """A frame clearing min_keypoints but missing the ROOT keypoint must be
    rejected: compute_stac.py's per-frame root warm-start
    (`kp_root_xyz = kp_flat[:, root_kp_idx*3:...]`) has no finite check, so a
    NaN root here hands the solver a NaN initial guess."""
    from scripts.run_bout import required_keypoint_indices
    kp = _kp(T=10, K=len(_KP_NAMES),
              missing=[(5, [_ROOT_IDX, 8, 9])])   # 7/10 finite, root among the missing
    req = required_keypoint_indices(_cfg(), _KP_NAMES)
    assert not finite_frame_mask(kp, min_keypoints=7, required_indices=req)[5]
    # sanity: min_keypoints alone (no required-set gate) WOULD have accepted it
    assert finite_frame_mask(kp, min_keypoints=7)[5]


def test_partial_frame_missing_orientation_keypoint_is_rejected():
    """A frame clearing min_keypoints but missing ONE orientation keypoint
    (e.g. WingR_base) must be rejected: _estimate_orientation_from_keypoints
    reads all four orientation keypoints with no finite check."""
    from scripts.run_bout import required_keypoint_indices
    kp = _kp(T=10, K=len(_KP_NAMES),
              missing=[(5, [_RIGHT_IDX, 8, 9])])  # 7/10 finite, right-orientation missing
    req = required_keypoint_indices(_cfg(), _KP_NAMES)
    assert not finite_frame_mask(kp, min_keypoints=7, required_indices=req)[5]
    assert finite_frame_mask(kp, min_keypoints=7)[5]


def test_partial_frame_with_root_and_orientation_present_is_accepted():
    """A frame clearing min_keypoints WITH root + all orientation keypoints
    finite must be accepted -- the whole point of Task 20 is that these frames
    are usable, and the new gate must not become a second all-or-nothing
    check in disguise."""
    from scripts.run_bout import required_keypoint_indices
    # missing only non-required keypoints (T1L/T1R/T2L_FeTi): root+orientation intact
    kp = _kp(T=10, K=len(_KP_NAMES), missing=[(5, [5, 6, 7])])
    req = required_keypoint_indices(_cfg(), _KP_NAMES)
    assert finite_frame_mask(kp, min_keypoints=7, required_indices=req)[5]


def test_min_keypoints_none_ignores_required_indices():
    """The min_keypoints=None default must stay byte-identical to today's
    all-or-nothing behaviour even when a required-indices set is passed --
    this is the task's safety rail and must not move."""
    from scripts.run_bout import required_keypoint_indices
    kp = _kp(T=10, K=len(_KP_NAMES), missing=[(5, [_ROOT_IDX])])
    req = required_keypoint_indices(_cfg(), _KP_NAMES)
    np.testing.assert_array_equal(
        finite_frame_mask(kp, min_keypoints=None, required_indices=req),
        finite_frame_mask(kp))
    np.testing.assert_array_equal(
        finite_frame_mask(kp, min_keypoints=None, required_indices=req),
        finite_frame_mask(kp, min_keypoints=None))
