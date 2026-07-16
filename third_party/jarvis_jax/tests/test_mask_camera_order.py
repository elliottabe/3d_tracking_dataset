"""Tests for the courtship SAM3 mask camera-identity fix.

Background: `sam3_masks.npz` packs masks as (A,C,T,H,Wp) with NO identifying
metadata on the camera (C) axis. A prior run wrote cameras in a SCRAMBLED
order vs the calibration/videos, silently corrupting the whole courtship 3D
pipeline (triangulating stored per-camera mask centroids under the identity
mapping gave 46px reproj residual; the correct permutation gave 6.8px).

This module tests:
  - `detect_camera_order`: brute-force geometric permutation search that
    recovers the true camera order from centroids + calibration.
  - `verify_mask_camera_order`: the loud-fail guard built on top of it.
  - `load_bout_masks(..., expected_cameras=...)`: safe, exact NAME-based
    reordering for npz files that DO carry a `cameras` array.

Uses the real Session0 calibration (7 cameras) so the geometry is exactly
the DLT triangulation/reprojection the production pipeline uses -- CPU-only,
pure numpy (no jax/torch).
"""
import os

import numpy as np
import pytest

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.tracking.bout_masks import (
    load_bout_masks, detect_camera_order, verify_mask_camera_order,
    detect_camera_order_robust, check_bout_camera_order,
)

CALIB_DIR = ("/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/"
             "courtship/Session0/2025_10_20_13_20_04/calibration")

needs_calib = pytest.mark.skipif(
    not os.path.isdir(CALIB_DIR), reason="real Session0 calibration not available")


@pytest.fixture(scope="module")
def rt():
    return ReprojectionTool(CALIB_DIR)


def _synthetic_track(rt, n_frames=6, seed=0):
    """(T,C,2) ground-truth per-camera centroid track (CORRECT / unscrambled
    camera order, i.e. mask-index == calibration-camera index) for
    `n_frames` random 3D points projected through the real calibration, plus
    an all-True (T,C) valid mask."""
    rng = np.random.default_rng(seed)
    pts3d = rng.uniform(low=[-15, -15, -5], high=[15, 15, 15], size=(n_frames, 3))
    C = rt.num_cameras
    centroids = np.zeros((n_frames, C, 2), np.float64)
    for t, p in enumerate(pts3d):
        centroids[t] = rt.reproject_point(p)
    valid = np.ones((n_frames, C), bool)
    return centroids, valid


# ---------------------------------------------------------------------------
# detect_camera_order
# ---------------------------------------------------------------------------

@needs_calib
def test_detect_camera_order_identity_is_low_residual_when_aligned(rt):
    centroids, valid = _synthetic_track(rt, n_frames=6)
    det = detect_camera_order(centroids, valid, rt, n_frames=6)
    assert det["identity_resid"] < 1.0
    assert det["best_perm"] == tuple(range(rt.num_cameras))
    assert det["best_resid"] < 1.0


@needs_calib
def test_detect_camera_order_recovers_known_permutation(rt):
    """The core regression test: a known camera-axis scramble must be
    recovered EXACTLY, with a clearly-bad identity residual and a
    near-perfect best-permutation residual (mirrors the real 46px -> 6.8px
    finding that uncovered this bug)."""
    C = rt.num_cameras
    centroids, valid = _synthetic_track(rt, n_frames=6)
    true_perm = tuple(np.roll(np.arange(C), 2).tolist())  # non-trivial cyclic scramble
    assert true_perm != tuple(range(C))                    # sanity: not accidentally identity
    scrambled = centroids[:, list(true_perm)]

    det = detect_camera_order(scrambled, valid, rt, n_frames=6)
    assert det["identity_resid"] > 20.0      # identity mapping is clearly wrong
    assert det["best_perm"] == true_perm     # recovered EXACTLY
    assert det["best_resid"] < 1.0           # near-perfect (noiseless synthetic data)


@needs_calib
def test_detect_camera_order_rejects_mismatched_camera_count(rt):
    centroids = np.zeros((3, rt.num_cameras + 1, 2))
    valid = np.ones((3, rt.num_cameras + 1), bool)
    with pytest.raises(ValueError, match="num_cameras"):
        detect_camera_order(centroids, valid, rt, n_frames=3)


def test_detect_camera_order_rejects_too_many_cameras():
    class _FakeRT:
        num_cameras = 9

    centroids = np.zeros((3, 9, 2))
    valid = np.ones((3, 9), bool)
    with pytest.raises(ValueError, match="8|many"):
        detect_camera_order(centroids, valid, _FakeRT(), n_frames=3)


@needs_calib
def test_detect_camera_order_raises_without_any_all_valid_frame(rt):
    centroids, valid = _synthetic_track(rt, n_frames=4)
    valid[:, 0] = False   # camera 0 never valid -> no frame has every camera valid
    with pytest.raises(ValueError, match="valid"):
        detect_camera_order(centroids, valid, rt, n_frames=4)


# ---------------------------------------------------------------------------
# verify_mask_camera_order
# ---------------------------------------------------------------------------

@needs_calib
def test_verify_mask_camera_order_passes_when_aligned(rt):
    centroids, valid = _synthetic_track(rt, n_frames=6)
    det = verify_mask_camera_order(centroids, valid, rt, n_frames=6)
    assert det["identity_resid"] < 1.0


@needs_calib
def test_verify_mask_camera_order_raises_on_scramble(rt):
    C = rt.num_cameras
    centroids, valid = _synthetic_track(rt, n_frames=6)
    true_perm = tuple(np.roll(np.arange(C), 3).tolist())
    scrambled = centroids[:, list(true_perm)]

    with pytest.raises(RuntimeError, match="[Cc]amera order"):
        verify_mask_camera_order(scrambled, valid, rt, n_frames=6)


# ---------------------------------------------------------------------------
# load_bout_masks: name-based reorder (safe/exact -- no geometry involved)
# ---------------------------------------------------------------------------

def _make_npz(tmp_path, *, cameras=None, C=3, T=2, H=4, W=5):
    """Tiny synthetic sam3_masks.npz. Each camera c gets a DISTINCTIVE mask
    pixel at column c and a distinctive centroid x = c+1, so a C-axis
    permutation is easy to verify by content."""
    A = 1
    full = np.zeros((A, C, T, H, W), np.uint8)
    valid = np.zeros((A, C, T), bool)
    centroids = np.zeros((A, C, T, 2), np.float32)
    for c in range(C):
        full[0, c, :, 0, c] = 1
        valid[0, c, :] = True
        centroids[0, c, :, 0] = c + 1
        centroids[0, c, :, 1] = (c + 1) * 10
    packed = np.packbits(full, axis=-1)
    kwargs = dict(packed=packed, valid=valid, centroids=centroids,
                  shape=np.array([H, W], np.int32))
    if cameras is not None:
        kwargs["cameras"] = np.array(cameras)
    p = tmp_path / "sam3_masks.npz"
    np.savez(p, **kwargs)
    return str(p)


def test_load_bout_masks_no_cameras_field_is_legacy_unreordered(tmp_path):
    p = _make_npz(tmp_path, cameras=None, C=3)
    out = load_bout_masks(p, fly=0, expected_cameras=["CamA", "CamB", "CamC"])
    assert "cameras" not in out
    # No metadata to reorder by -> stored order preserved unchanged.
    assert out["centroids"][0, 0, 0] == 1.0
    assert out["masks"][0, 0, 0, 0] == True


def test_load_bout_masks_reorders_by_name(tmp_path):
    stored_order = ["CamB", "CamC", "CamA"]      # index0=CamB, index1=CamC, index2=CamA
    expected = ["CamA", "CamB", "CamC"]
    p = _make_npz(tmp_path, cameras=stored_order, C=3)
    out = load_bout_masks(p, fly=0, expected_cameras=expected)

    assert out["cameras"] == expected
    assert out["C"] == 3
    # CamA was stored at index 2 (pixel col=2, centroid x=3) -> now output index 0.
    assert out["centroids"][0, 0, 0] == 3.0
    assert out["masks"][0, 0, 0, 2] == True
    # CamB was stored at index 0 (pixel col=0, centroid x=1) -> now output index 1.
    assert out["centroids"][0, 1, 0] == 1.0
    assert out["masks"][0, 1, 0, 0] == True
    # CamC was stored at index 1 (pixel col=1, centroid x=2) -> now output index 2.
    assert out["centroids"][0, 2, 0] == 2.0
    assert out["masks"][0, 2, 0, 1] == True


def test_load_bout_masks_cameras_present_no_expected_keeps_stored_order(tmp_path):
    stored_order = ["CamB", "CamC", "CamA"]
    p = _make_npz(tmp_path, cameras=stored_order, C=3)
    out = load_bout_masks(p, fly=0)     # no expected_cameras -> stored order, as-is
    assert out["cameras"] == stored_order
    assert out["centroids"][0, 0, 0] == 1.0    # CamB's original data, unmoved


def test_load_bout_masks_missing_expected_camera_raises(tmp_path):
    p = _make_npz(tmp_path, cameras=["CamA", "CamB", "CamC"], C=3)
    with pytest.raises(ValueError, match="CamZ"):
        load_bout_masks(p, fly=0, expected_cameras=["CamA", "CamZ"])


def test_load_bout_masks_backward_compatible_no_kwargs(tmp_path):
    """Existing callers (no expected_cameras at all) must be unaffected,
    whether or not the npz happens to carry a `cameras` array."""
    p = _make_npz(tmp_path, cameras=None, C=2, T=3, H=8, W=13)
    out = load_bout_masks(p, fly=0)
    assert out["T"] == 3 and out["C"] == 2 and out["H"] == 8 and out["W"] == 13
    assert "cameras" not in out


# ---------------------------------------------------------------------------
# detect_camera_order_robust: partial-visibility-tolerant detection
# ---------------------------------------------------------------------------

def _partial_valid(valid, rng, n_invalid):
    """Knock out `n_invalid` random cameras per frame (in-place), leaving the
    rest valid -- simulates a bout where the fly is never visible in every
    camera at once."""
    T, C = valid.shape
    for t in range(T):
        for c in rng.choice(C, size=n_invalid, replace=False):
            valid[t, c] = False
    return valid


@needs_calib
def test_detect_robust_recovers_permutation_with_partial_visibility(rt):
    """A scramble must still be recovered when NO frame has every camera valid
    -- the case the strict detect_camera_order raises on."""
    C = rt.num_cameras
    centroids, valid = _synthetic_track(rt, n_frames=12, seed=3)
    true_perm = tuple(np.roll(np.arange(C), 2).tolist())
    scrambled = centroids[:, list(true_perm)]
    rng = np.random.default_rng(0)
    valid = _partial_valid(valid.copy(), rng, n_invalid=2)   # 5/7 valid per frame
    assert not valid.all(axis=1).any()                       # no all-valid frame

    det = detect_camera_order_robust(scrambled, valid, rt, n_frames=12, min_valid_cams=4)
    assert det["n_frames_used"] > 0
    assert det["best_perm"] == true_perm
    assert det["identity_resid"] > 20.0
    assert det["best_resid"] < 1.0


@needs_calib
def test_detect_robust_unverifiable_when_too_sparse(rt):
    """< 3 valid cameras in every frame -> cannot triangulate -> report
    n_frames_used==0 (NOT raise)."""
    centroids, valid = _synthetic_track(rt, n_frames=6)
    valid[:] = False
    valid[:, :2] = True                                      # only 2 cameras ever valid
    det = detect_camera_order_robust(centroids, valid, rt, n_frames=6, min_valid_cams=4)
    assert det["n_frames_used"] == 0
    assert not np.isfinite(det["identity_resid"])


# ---------------------------------------------------------------------------
# check_bout_camera_order: NON-MUTATING QC (never remaps, never raises on data)
# ---------------------------------------------------------------------------

def _make_geo_npz(tmp_path, rt, *, scramble_perm=None, n=12, n_invalid=0,
                  bad_cam=None, cameras=None, seed=7, name="sam3_masks.npz"):
    """Geometrically-consistent sam3_masks.npz: `n` random 3D points projected
    through the real calibration give the per-camera centroids (CORRECT camera
    order == calibration order). `scramble_perm` permutes the stored C axis;
    `bad_cam` corrupts ONE camera's centroids (simulating a SAM3 mis-track);
    `n_invalid` knocks out cameras per frame. Native (A,C,T,...) layout."""
    C = rt.num_cameras
    rng = np.random.default_rng(seed)
    pts3d = rng.uniform(low=[-15, -15, -5], high=[15, 15, 15], size=(n, 3))
    cent_corr = np.zeros((C, n, 2), np.float32)              # (C,T,2) correct order
    for t, p in enumerate(pts3d):
        cent_corr[:, t] = rt.reproject_point(p)              # (C,2)
    if bad_cam is not None:
        cent_corr[bad_cam] += 250.0                          # gross per-camera error
    valid_corr = np.ones((C, n), bool)                       # (C,T)
    if n_invalid:
        valid_corr = _partial_valid(valid_corr.T.copy(), rng, n_invalid).T
    packed_corr = np.zeros((C, n, 1, 1), np.uint8)
    for c in range(C):
        packed_corr[c, :, 0, 0] = c

    if scramble_perm is not None:
        sp = list(scramble_perm)
        cent_corr, valid_corr, packed_corr = cent_corr[sp], valid_corr[sp], packed_corr[sp]

    kwargs = dict(packed=packed_corr[None], valid=valid_corr[None],
                  centroids=cent_corr[None], shape=np.array([1, 8], np.int32))
    if cameras is not None:
        kwargs["cameras"] = np.array(cameras)
    p = tmp_path / name
    np.savez(p, **kwargs)
    return str(p)


def _unchanged(p):
    """True if the npz on disk is byte-identical to when check was called
    (check_bout_camera_order must never mutate)."""
    return not os.path.exists(p + ".scrambled.bak")


@needs_calib
def test_check_aligned_when_geometry_good(tmp_path, rt):
    expected = list(rt.cameras)
    p = _make_geo_npz(tmp_path, rt)                           # clean, correct order
    st = check_bout_camera_order(p, 0, expected, rt)
    assert st["status"] == "aligned"
    assert st["worst_cam"] is None and _unchanged(p)


@needs_calib
def test_check_correctly_named_nonidentity_is_aligned(tmp_path, rt):
    """Non-identity stored order with a truthful `cameras` array: the pipeline's
    name reorder yields calibration order, so the check sees aligned geometry."""
    expected = list(rt.cameras)
    C = rt.num_cameras
    scrambled = list(np.roll(np.arange(C), 1))
    p = _make_geo_npz(tmp_path, rt, scramble_perm=tuple(scrambled),
                      cameras=[expected[i] for i in scrambled])
    st = check_bout_camera_order(p, 0, expected, rt)
    assert st["status"] == "aligned" and _unchanged(p)


@needs_calib
def test_check_flags_bad_camera_never_mutates(tmp_path, rt):
    """One mis-tracked camera -> 'suspect' naming that camera, file UNTOUCHED.
    This is the real failure mode (not a scramble) and must NOT be remapped."""
    expected = list(rt.cameras)
    p = _make_geo_npz(tmp_path, rt, bad_cam=4)
    st = check_bout_camera_order(p, 0, expected, rt)
    assert st["status"] == "suspect"
    assert st["worst_cam"] == 4                               # names the bad camera
    assert _unchanged(p)                                      # never mutated


@needs_calib
def test_check_bad_data_never_raises_and_never_mutates(tmp_path, rt):
    """Even a fully scrambled-looking file is only flagged, never remapped or
    raised on -- one odd bout cannot block a session."""
    expected = list(rt.cameras)
    C = rt.num_cameras
    p = _make_geo_npz(tmp_path, rt, scramble_perm=tuple(np.roll(np.arange(C), 2)),
                      cameras=expected)                       # data != label
    st = check_bout_camera_order(p, 0, expected, rt)          # must not raise
    assert st["status"] in ("suspect", "aligned")
    assert _unchanged(p)


@needs_calib
def test_check_unverified_when_too_sparse(tmp_path, rt):
    expected = list(rt.cameras)
    C = rt.num_cameras
    p = _make_geo_npz(tmp_path, rt, n_invalid=C - 2)          # only 2 cams/frame
    st = check_bout_camera_order(p, 0, expected, rt)
    assert st["status"] == "unverified" and _unchanged(p)
