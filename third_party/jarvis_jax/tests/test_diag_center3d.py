"""CPU, no-checkpoint unit tests for diag_center3d_sensitivity.py.

Covers the two pure-python building blocks (``keypoint_spread``,
``perturb_center3d``) directly, plus the full sweep plumbing
(``sweep_center3d_sensitivity``) against a tiny stub model that mimics
HybridNet3D's ``__call__`` contract (``model(crops, center3D, centerHM,
cameraMatrices, use_running_average=True) -> (vol, points3D, conf)``) without
needing a real front-end/V2VNet or a checkpoint.
"""
import numpy as np
import pytest

from jarvis_jax.scripts.diag_center3d_sensitivity import (
    keypoint_spread,
    perturb_center3d,
    mpjpe,
    sweep_center3d_sensitivity,
)


# ---------------------------------------------------------------------------
# keypoint_spread
# ---------------------------------------------------------------------------

def test_keypoint_spread_bbox_diagonal():
    pts = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    assert keypoint_spread(pts) == pytest.approx(5.0)


def test_keypoint_spread_shift_invariant():
    rng = np.random.RandomState(0)
    pts = rng.randn(10, 3)
    shifted = pts + np.array([100.0, -50.0, 7.0])
    assert keypoint_spread(pts) == pytest.approx(keypoint_spread(shifted), abs=1e-9)


def test_keypoint_spread_single_point_is_zero():
    pts = np.array([[1.0, 2.0, 3.0]])
    assert keypoint_spread(pts) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# perturb_center3d
# ---------------------------------------------------------------------------

def test_perturb_center3d_applies_offset_along_direction():
    c = np.array([1.0, 2.0, 3.0])
    out = perturb_center3d(c, 5.0, direction=(1.0, 0.0, 0.0))
    assert np.allclose(out, c + np.array([5.0, 0.0, 0.0]))


def test_perturb_center3d_zero_delta_is_noop():
    c = np.array([1.0, 2.0, 3.0])
    out = perturb_center3d(c, 0.0)
    assert np.allclose(out, c)


def test_perturb_center3d_normalizes_direction():
    c = np.zeros(3)
    out = perturb_center3d(c, 10.0, direction=(3.0, 4.0, 0.0))  # norm 5
    assert np.allclose(out, np.array([6.0, 8.0, 0.0]))


def test_perturb_center3d_zero_direction_is_noop():
    c = np.array([1.0, 2.0, 3.0])
    out = perturb_center3d(c, 5.0, direction=(0.0, 0.0, 0.0))
    assert np.allclose(out, c)


# ---------------------------------------------------------------------------
# mpjpe (masked mean 3-D error)
# ---------------------------------------------------------------------------

def test_mpjpe_masks_invisible_joints():
    pred = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    gt = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    vis = np.array([True, False])
    # Only joint 0 counted: ||[0,0,0]-[3,4,0]|| = 5.
    assert mpjpe(pred, gt, vis) == pytest.approx(5.0)


def test_mpjpe_nan_when_nothing_visible():
    pred = np.zeros((2, 3))
    gt = np.zeros((2, 3))
    vis = np.array([False, False])
    assert np.isnan(mpjpe(pred, gt, vis))


# ---------------------------------------------------------------------------
# sweep_center3d_sensitivity against a stub model
# ---------------------------------------------------------------------------

class _StubModel:
    """Mimics HybridNet3D.__call__'s contract with a FIXED local shape:
    regardless of the crops it's given, it always predicts
    ``local_points + center3D`` (i.e. it re-centers the same rigid shape at
    whatever center3D it's told to use). This isolates the
    perturbation/metric plumbing from any real front-end/V2VNet forward.
    """
    _frontend_channels = None

    def __init__(self, local_points):
        self.local_points = np.asarray(local_points, dtype=np.float64)  # (J, 3)

    def __call__(self, crops, center3D, centerHM, cameraMatrices, masks=None,
                 use_running_average=False):
        center3D = np.asarray(center3D)                       # (B, 3)
        B = center3D.shape[0]
        J = self.local_points.shape[0]
        pts = self.local_points[None] + center3D[:, None, :]  # (B, J, 3)
        conf = np.ones((B, J), dtype=np.float32)
        vol = np.zeros((B, J, 2, 2, 2), dtype=np.float32)
        return vol, pts, conf


class _FakeFramesetDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def _make_sample(local_points, gt_center3D, num_cam=2, crop=8):
    gt_kp3d = (local_points + gt_center3D).astype(np.float32)
    J = local_points.shape[0]
    return {
        "crops4": np.zeros((num_cam, crop, crop, 4), dtype=np.uint8),
        "centerHM": np.zeros((num_cam, 2), dtype=np.float32),
        "center3D": gt_center3D.astype(np.float32),
        "cameraMatrices": np.zeros((num_cam, 4, 3), dtype=np.float32),
        "kp3d": gt_kp3d,
        "vis": np.ones(J, dtype=bool),
    }


def test_sweep_shift_invariant_spread_and_local_mpjpe():
    """The stub predicts the SAME local shape (matching GT's local shape)
    re-centered at whatever center3D it's given. So:
      * spread must be IDENTICAL across delta (rigid shift doesn't change bbox
        diagonal) -- this is exactly the "center3D is NOT the cause" case.
      * mpjpe_local must be ~0 for every delta (shift-corrected, and the
        stub's local shape exactly matches GT's local shape).
      * mpjpe_raw must grow LINEARLY with delta (pure unrecovered rigid shift,
        since direction is a unit vector).
    """
    local_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    gt_center3D = np.array([10.0, 20.0, 30.0])
    sample = _make_sample(local_points, gt_center3D)
    ds = _FakeFramesetDataset([sample, sample])
    model = _StubModel(local_points)

    deltas = (0.0, 5.0, 10.0)
    result = sweep_center3d_sensitivity(
        model, ds, n=2, deltas=deltas, direction=(1.0, 0.0, 0.0), quiet=True)
    summary = result["summary"]

    assert result["n"] == 2
    assert set(summary.keys()) == set(deltas)

    spreads = [summary[d]["spread_mean"] for d in deltas]
    assert spreads[0] == pytest.approx(spreads[1], abs=1e-6)
    assert spreads[0] == pytest.approx(spreads[2], abs=1e-6)

    for d in deltas:
        assert summary[d]["mpjpe_local_mean"] == pytest.approx(0.0, abs=1e-6)

    assert summary[0.0]["mpjpe_raw_mean"] == pytest.approx(0.0, abs=1e-6)
    assert summary[5.0]["mpjpe_raw_mean"] == pytest.approx(5.0, abs=1e-4)
    assert summary[10.0]["mpjpe_raw_mean"] == pytest.approx(10.0, abs=1e-4)


def test_sweep_center3d_discrepancy_skipped_when_masks_empty():
    """crops4's mask channel (index 3) is all-zero in the fake samples above
    -> estimate_center3d_from_masks sees 0 valid cameras -> the discrepancy
    list stays empty (the "SKIP" path noted in the module docstring)."""
    local_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    gt_center3D = np.array([1.0, 2.0, 3.0])
    sample = _make_sample(local_points, gt_center3D)
    ds = _FakeFramesetDataset([sample])
    model = _StubModel(local_points)

    result = sweep_center3d_sensitivity(
        model, ds, n=1, deltas=(0.0,), quiet=True)
    assert result["center3d_discrepancy"] == []


def test_sweep_clamps_n_to_dataset_length():
    local_points = np.array([[0.0, 0.0, 0.0]])
    gt_center3D = np.zeros(3)
    sample = _make_sample(local_points, gt_center3D)
    ds = _FakeFramesetDataset([sample])
    model = _StubModel(local_points)

    result = sweep_center3d_sensitivity(model, ds, n=100, deltas=(0.0,), quiet=True)
    assert result["n"] == 1
