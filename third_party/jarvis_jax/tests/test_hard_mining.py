"""CPU-fast unit tests for error-based hard-example mining -- no real
checkpoint, no GPU, no V3 data root.

  1. ``error_weights`` (jarvis_jax.scripts.train_keypoints) -- the pure
     weight-building function `configs/sampling/hard_error.yaml` drives:
     monotonic in error, every weight > 0 (no forgetting), the cap actually
     clips the tail, NaN errors are treated as neutral (median-like), and
     the result is sum-normalised.
  2. ``mine_errors`` (jarvis_jax.scripts.mine_hard_frames) -- wired against a
     real (tiny) ViTPose forward pass + a minimal V3-like dataset stand-in:
     one finite error per sample, in dataset order, and a deliberately
     mismatched GT/prediction sample scores higher error than a sample whose
     GT is set to exactly match the model's own (real) prediction.
"""
import numpy as np
import jax.numpy as jnp
import pytest
from flax import nnx

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.data.device import normalize_image
from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.train.train import _eval_forward
from jarvis_jax.scripts.train_keypoints import error_weights
from jarvis_jax.scripts.mine_hard_frames import mine_errors


# ---------------------------------------------------------------------------
# error_weights
# ---------------------------------------------------------------------------

def test_error_weights_monotonic_in_error():
    error = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    w = error_weights(error, alpha=4.0, cap=8.0)
    assert np.all(np.diff(w) >= 0)   # non-decreasing as error increases


def test_error_weights_all_positive_no_forgetting():
    # Include a zero-error (perfect) sample -- it must still get weight > 0.
    error = np.array([0.0, 1.0, 1.0, 1.0, 100.0], dtype=np.float32)
    w = error_weights(error, alpha=4.0, cap=8.0)
    assert np.all(w > 0.0)


def test_error_weights_sum_normalised():
    rng = np.random.default_rng(0)
    error = rng.exponential(scale=3.0, size=50).astype(np.float32)
    w = error_weights(error, alpha=4.0, cap=8.0)
    assert w.shape == error.shape
    assert abs(w.sum() - 1.0) < 1e-9


def test_error_weights_cap_clips_the_tail():
    # 9 typical (error=1) samples + one huge outlier: with an odd n=10,
    # median = mean of the 5th/6th order statistics = 1 regardless of the
    # outlier's magnitude (it's always the max). So two runs that only differ
    # in HOW large the outlier is (but both >> cap*median) must produce
    # identical weight vectors once the outlier's r is clipped to `cap`.
    base = [1.0] * 9
    w_100 = error_weights(np.array(base + [100.0], dtype=np.float32), alpha=4.0, cap=8.0)
    w_100000 = error_weights(np.array(base + [100000.0], dtype=np.float32), alpha=4.0, cap=8.0)
    np.testing.assert_allclose(w_100, w_100000, rtol=1e-6)

    # And the capped outlier's raw (pre-normalisation) weight is exactly
    # 1 + alpha*cap relative to a median (r=1) sample's 1 + alpha.
    ratio = w_100[-1] / w_100[0]
    expected_ratio = (1.0 + 4.0 * 8.0) / (1.0 + 4.0 * 1.0)
    assert abs(ratio - expected_ratio) < 1e-6


def test_error_weights_nan_is_neutral():
    # A NaN-error (no visible joints) sample must be treated exactly like a
    # median-error sample (r=1), not like an error of 0 or of the cap.
    error = np.array([1.0, 1.0, 1.0, np.nan], dtype=np.float32)
    w = error_weights(error, alpha=4.0, cap=8.0)
    assert np.isfinite(w).all()
    np.testing.assert_allclose(w[3], w[0], rtol=1e-6)


def test_error_weights_requires_some_finite_value():
    with pytest.raises(ValueError):
        error_weights(np.array([np.nan, np.nan], dtype=np.float32))


# ---------------------------------------------------------------------------
# mine_errors
# ---------------------------------------------------------------------------

IMG_SIZE = 64
HEATMAP_SIZE = 32   # ClassicDecoder fixed 8x upsample: (64/16)**2 tokens -> 4*8=32
NUM_KEYPOINTS = 5


class _TinyMineDS:
    """Minimal V3Dataset stand-in: exposes exactly what `batches`/`mine_errors`
    read -- heatmap_size, __len__, __getitem__ -> (img4_u8, kp_xy, vis)."""

    def __init__(self, imgs, kps, viss, heatmap_size):
        self.imgs, self.kps, self.viss = imgs, kps, viss
        self.heatmap_size = heatmap_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.kps[i], self.viss[i]


def _tiny_vitpose_cfg():
    return ViTPoseConfig(img_size=IMG_SIZE, patch=16, embed_dim=48, depth=2,
                         num_heads=4, num_keypoints=NUM_KEYPOINTS,
                         heatmap_size=HEATMAP_SIZE)


def test_mine_errors_dataset_order_and_wrong_pred_scores_higher():
    cfg = _tiny_vitpose_cfg()
    model = ViTPose(cfg, rngs=nnx.Rngs(0))
    model.eval()

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(IMG_SIZE, IMG_SIZE, 4), dtype=np.uint8)

    # Get the model's OWN real prediction for this image (same primitives
    # mine_errors uses), so we can construct a "correct" GT that matches it
    # exactly, without needing a trained model.
    img_n = normalize_image(jnp.asarray(img[None]))
    pred = _eval_forward(model, img_n)
    pk = heatmaps_to_keypoints(pred, in_size=IMG_SIZE)   # (1,K,2) pixel coords
    scale = IMG_SIZE / HEATMAP_SIZE
    kp_correct = np.asarray(pk[0]) / scale                # heatmap-coord space
    kp_wrong = kp_correct + 1000.0                          # deliberately way off

    vis = np.ones(NUM_KEYPOINTS, dtype=np.float32)
    imgs = np.stack([img, img])
    kps = np.stack([kp_correct, kp_wrong]).astype(np.float32)
    viss = np.stack([vis, vis])
    ds = _TinyMineDS(imgs, kps, viss, HEATMAP_SIZE)

    error = mine_errors(model, ds, batch_size=2, in_size=IMG_SIZE)

    assert error.shape == (2,)
    assert np.all(np.isfinite(error))
    assert error[0] < 1e-2         # "correct" sample: GT == model's own prediction
    assert error[1] > error[0]     # deliberately-wrong sample scores higher
    assert error[1] > 100.0        # and by a lot (shifted by 1000px in heatmap space)


def test_mine_errors_nan_when_no_visible_joints():
    cfg = _tiny_vitpose_cfg()
    model = ViTPose(cfg, rngs=nnx.Rngs(0))

    rng = np.random.default_rng(1)
    imgs = rng.integers(0, 256, size=(3, IMG_SIZE, IMG_SIZE, 4), dtype=np.uint8)
    kps = rng.uniform(0, HEATMAP_SIZE, size=(3, NUM_KEYPOINTS, 2)).astype(np.float32)
    viss = np.ones((3, NUM_KEYPOINTS), dtype=np.float32)
    viss[1] = 0.0   # middle sample has no visible joints at all

    ds = _TinyMineDS(imgs, kps, viss, HEATMAP_SIZE)
    error = mine_errors(model, ds, batch_size=2, in_size=IMG_SIZE)

    assert error.shape == (3,)
    assert np.isfinite(error[0])
    assert np.isnan(error[1])
    assert np.isfinite(error[2])
