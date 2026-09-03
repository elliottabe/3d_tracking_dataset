import jax.numpy as jnp
import numpy as np
from jarvis_jax.train.losses import heatmap_mse, mask_containment


def test_mse_zero_when_equal():
    g = jnp.asarray(np.random.RandomState(0).rand(2, 8, 8, 3).astype("float32"))
    vis = jnp.ones((2, 3), dtype=bool)
    assert float(heatmap_mse(g, g, vis)) == 0.0


def test_mse_positive_and_ignores_invisible():
    pred = jnp.zeros((1, 4, 4, 2), dtype=jnp.float32)
    gt = jnp.zeros((1, 4, 4, 2), dtype=jnp.float32).at[0, 0, 0, 0].set(1.0)
    # keypoint 0 differs but is invisible -> loss 0; keypoint 1 identical
    vis = jnp.asarray([[False, True]])
    assert float(heatmap_mse(pred, gt, vis)) == 0.0
    # now mark keypoint 0 visible -> positive loss
    vis2 = jnp.asarray([[True, True]])
    assert float(heatmap_mse(pred, gt, vis2)) > 0.0


def test_mse_penalizes_missed_foreground_peak():
    # gt has a foreground bump (value 1); pred is all zeros -> the loss must be
    # dominated by the missed peak (~1.0), not diluted toward 0 as plain
    # mean-MSE would be.
    gt = jnp.zeros((1, 8, 8, 1), dtype=jnp.float32).at[0, 3:5, 3:5, 0].set(1.0)
    pred = jnp.zeros((1, 8, 8, 1), dtype=jnp.float32)
    vis = jnp.ones((1, 1), dtype=bool)
    assert float(heatmap_mse(pred, gt, vis)) > 0.5


def test_mask_containment_zero_inside_one_outside():
    pred = jnp.zeros((1, 4, 4, 1), dtype=jnp.float32).at[0, 1, 1, 0].set(1.0)
    inside = jnp.zeros((1, 4, 4), dtype=jnp.float32).at[0, 1, 1].set(1.0)
    outside = jnp.ones((1, 4, 4), dtype=jnp.float32) - inside
    assert float(mask_containment(pred, inside)) == 0.0
    assert abs(float(mask_containment(pred, outside)) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 2026-09-03: hard-negative background + repulsion footprint
# ---------------------------------------------------------------------------
def _peak(h, w, cx, cy, sigma=3.0):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype("float32")


def test_hardneg_makes_a_spurious_blob_expensive():
    # gt: one peak at (20,20), pred: the SAME peak plus a spurious full-height
    # blob at (60,60). Plain loss dilutes the blob over ~all background pixels;
    # hard-negative top-k charges its worst pixels at full strength.
    h = w = 96
    gt = _peak(h, w, 20, 20)[None, :, :, None]
    pred = gt + _peak(h, w, 60, 60)[None, :, :, None]
    vis = jnp.ones((1, 1), bool)
    plain = float(heatmap_mse(jnp.asarray(pred), jnp.asarray(gt), vis))
    hard = float(heatmap_mse(jnp.asarray(pred), jnp.asarray(gt), vis,
                             hardneg_k=64, hardneg_weight=1.0))
    assert plain < 0.02                       # the blob is nearly free today
    assert hard > 0.3 and hard > 20 * plain   # ~mean of the blob's 64 worst pixels
    # off by default: identical to before
    assert float(heatmap_mse(jnp.asarray(pred), jnp.asarray(gt), vis, hardneg_k=64)) == plain


def test_repulsion_penalises_only_inside_footprint_and_outside_own_fg():
    h = w = 96
    gt = _peak(h, w, 20, 20)[None, :, :, None]
    vis = jnp.ones((1, 1), bool)
    blob_far = _peak(h, w, 60, 60)[None, :, :, None]
    fp_far = _peak(h, w, 60, 60, sigma=5.0)[None, :, :, None]
    # pred fires on the distractor's part -> repulsion adds a large term
    pred = gt + blob_far
    plain = float(heatmap_mse(jnp.asarray(pred), jnp.asarray(gt), vis))
    rep = float(heatmap_mse(jnp.asarray(pred), jnp.asarray(gt), vis,
                            rep_fp=jnp.asarray(fp_far), rep_weight=1.0))
    assert rep > plain + 0.1        # footprint-weighted MSE of a sigma-3 blob under a sigma-5 footprint
    # pred quiet under the footprint -> repulsion adds nothing
    same = float(heatmap_mse(jnp.asarray(gt), jnp.asarray(gt), vis,
                             rep_fp=jnp.asarray(fp_far), rep_weight=1.0))
    assert same == 0.0
    # footprint ON the target's own peak, pred all zero: the missed peak is
    # charged by the fg term exactly as before, the repulsion term excludes it
    fp_on_gt = _peak(h, w, 20, 20, sigma=5.0)[None, :, :, None]
    zeros = jnp.zeros_like(jnp.asarray(gt))
    base = float(heatmap_mse(zeros, jnp.asarray(gt), vis))
    with_fp = float(heatmap_mse(zeros, jnp.asarray(gt), vis,
                                rep_fp=jnp.asarray(fp_on_gt), rep_weight=1.0))
    assert abs(with_fp - base) < 1e-3
    # empty footprint contributes exactly 0
    empty = float(heatmap_mse(jnp.asarray(pred), jnp.asarray(gt), vis,
                              rep_fp=jnp.zeros_like(jnp.asarray(gt)), rep_weight=1.0))
    assert empty == plain
