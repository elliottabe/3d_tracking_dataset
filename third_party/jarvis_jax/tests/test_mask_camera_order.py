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
from jarvis_jax.cse.courtship_bout_masks import (
    load_bout_masks, detect_camera_order, verify_mask_camera_order,
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
