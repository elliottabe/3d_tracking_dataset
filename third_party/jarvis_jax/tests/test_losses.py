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
