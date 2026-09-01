import numpy as np
import jax.numpy as jnp
import pytest
from jarvis_jax.tracking.wing_coverage import wing_target_points, coverage_residual


def test_targets_exclude_pixels_the_body_already_explains():
    mask = np.zeros((100, 100), bool)
    mask[40:60, 20:80] = True                 # a wide bar
    body_uv = np.stack(np.meshgrid(np.arange(20, 50), np.arange(40, 60)),
                       -1).reshape(-1, 2).astype(np.float32)   # covers the LEFT half
    pts = np.asarray(wing_target_points(mask, body_uv, n_points=200, dilate_px=0, seed=0))
    good = pts[np.isfinite(pts).all(1)]
    assert len(good) > 20
    assert good[:, 0].min() > 45, "targets must avoid the body-covered left half"


def test_coverage_pulls_toward_an_uncovered_target():
    tgt = jnp.asarray([[50.0, 50.0]])
    near = coverage_residual(tgt, jnp.asarray([[52.0, 50.0]]))
    far = coverage_residual(tgt, jnp.asarray([[90.0, 50.0]]))
    assert float(near[0]) < float(far[0])


def test_softmin_approaches_the_hard_min_for_large_beta():
    tgt = jnp.asarray([[0.0, 0.0]])
    verts = jnp.asarray([[3.0, 0.0], [10.0, 0.0]])
    r = float(coverage_residual(tgt, verts, beta=200.0)[0])
    assert r == pytest.approx(3.0, abs=0.1)


def test_nan_targets_contribute_zero():
    tgt = jnp.asarray([[np.nan, np.nan], [0.0, 0.0]])
    r = coverage_residual(tgt, jnp.asarray([[1.0, 0.0]]))
    assert float(r[0]) == 0.0 and float(r[1]) > 0.0


def test_gradient_flows_to_the_wing_vertices():
    """A descent step must move the vertex TOWARD the target.

    The gradient itself is POSITIVE here: the cost is the distance from the
    target to the nearest vertex, so pushing the vertex further right (away)
    raises it. Descent (`v -= lr * grad`) therefore moves the vertex left,
    toward the target at x=0. Verified: cost 4.0/5.0/6.0 at x=4/5/6, analytic
    grad +1.0, finite-difference +0.99993.
    """
    import jax
    tgt = jnp.asarray([[0.0, 0.0]])
    g = jax.grad(lambda v: coverage_residual(tgt, v)[0])(jnp.asarray([[5.0, 0.0]]))
    assert float(g[0, 0]) > 0.0, "cost must rise as the vertex moves away"
    step = 5.0 - 0.5 * float(g[0, 0])
    assert step < 5.0, "a descent step must move the vertex toward the target"
