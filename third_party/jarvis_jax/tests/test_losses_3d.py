"""Tests for 3D losses: graph_laplacian, heatmap3d_mse."""
import jax.numpy as jnp
import numpy as np
from jarvis_jax.train.losses_3d import graph_laplacian, build_skeleton_edges, heatmap3d_mse


def test_graph_laplacian_zero_when_equal_and_ignores_invalid():
    pred = jnp.asarray(np.random.RandomState(0).rand(2, 5, 3).astype("float32"))
    valid = jnp.ones((2, 5), dtype=bool)
    ei = jnp.asarray([0, 1, 2]); ej = jnp.asarray([1, 2, 3])
    assert float(graph_laplacian(pred, pred, valid, ei, ej)) == 0.0
    gt = pred.at[0, 0].add(1.0)               # bone (0,1) differs in sample 0
    v = jnp.ones((2, 5), dtype=bool).at[0, 0].set(False)  # endpoint 0 invalid
    assert float(graph_laplacian(pred, gt, v, ei, ej)) == 0.0  # that bone masked


def test_heatmap3d_mse_in_grid_finite_and_out_of_grid_masked():
    """in-grid GT -> finite positive loss; out-of-grid GT -> masked (finite, not blow-up)."""
    B, J, G = 1, 2, 24
    grid_spacing = 1
    roi_cube = 48
    sigma = 2.0

    # Zero pred volume
    pred_vol = jnp.zeros((B, J, G, G, G), dtype=jnp.float32)

    # center3D at origin
    center3D = jnp.zeros((B, 3), dtype=jnp.float32)

    # Joint 0: in-grid GT point at world (0, 0, 0) -> grid idx ~12 (centre)
    # Joint 1: out-of-grid GT point far outside cube
    #   grid_idx = (gt_kp - center3D + roi_cube/2) / (grid_spacing*2)
    #   With gt_kp_j1 = (999, 999, 999) -> idx = (999 + 24) / 2 >> 23 (way out of grid)
    gt_kp = jnp.array([[[0.0, 0.0, 0.0], [999.0, 999.0, 999.0]]], dtype=jnp.float32)  # (1, 2, 3)
    valid = jnp.ones((B, J), dtype=bool)

    loss = heatmap3d_mse(
        pred_vol, gt_kp, valid,
        grid_spacing=grid_spacing,
        roi_cube=roi_cube,
        center3D=center3D,
        sigma=sigma,
    )

    # Must be finite
    assert jnp.isfinite(loss), f"loss is not finite: {loss}"
    # Must be clearly positive: fg-MSE dominates (missed Gaussian peak vs zero pred),
    # so loss should be well above 0.01.
    assert float(loss) > 0.01, f"expected fg-dominated positive loss > 0.01, got {loss}"

    # Now check that when both joints are out-of-grid, loss is 0 (all masked)
    gt_kp_oob = jnp.array([[[999.0, 999.0, 999.0], [999.0, 999.0, 999.0]]], dtype=jnp.float32)
    loss_oob = heatmap3d_mse(
        pred_vol, gt_kp_oob, valid,
        grid_spacing=grid_spacing,
        roi_cube=roi_cube,
        center3D=center3D,
        sigma=sigma,
    )
    assert jnp.isfinite(loss_oob), f"out-of-grid loss is not finite: {loss_oob}"
    assert float(loss_oob) == 0.0, f"expected 0 for all-OOB, got {loss_oob}"
