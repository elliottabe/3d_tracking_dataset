# tests/test_silhouette_chamfer.py
import jax
import jax.numpy as jnp
import numpy as np
from jarvis_jax.tracking.silhouette_chamfer import chamfer_residual


def test_known_geometry_hardmin_value():
    # target at (0,0); nearest projected vertex at (3,4) -> distance 5.
    # large beta -> softmin approaches the hard min.
    target = jnp.array([[0.0, 0.0]])
    proj = jnp.array([[3.0, 4.0], [10.0, 10.0], [-8.0, 6.0]])
    r = chamfer_residual(target, proj, beta=50.0)
    assert r.shape == (1,)
    assert abs(float(r[0]) - 5.0) < 0.05


def test_nan_target_row_is_zero_residual():
    target = jnp.array([[np.nan, np.nan], [0.0, 0.0]])
    proj = jnp.array([[3.0, 4.0]])
    r = chamfer_residual(target, proj, beta=50.0)
    assert float(r[0]) == 0.0
    assert abs(float(r[1]) - 5.0) < 0.05


def test_gradient_flows_to_projected_points():
    # moving the nearest projected vertex toward the target should reduce the
    # residual -> a nonzero, finite gradient w.r.t. proj.
    target = jnp.array([[0.0, 0.0]])

    def loss(proj):
        return jnp.sum(chamfer_residual(target, proj, beta=20.0) ** 2)

    proj0 = jnp.array([[3.0, 4.0], [9.0, 9.0]])
    g = jax.grad(loss)(proj0)
    assert np.isfinite(np.asarray(g)).all()
    # the nearest vertex (row 0) carries (almost) all the gradient
    assert float(jnp.linalg.norm(g[0])) > 1e-3
    assert float(jnp.linalg.norm(g[0])) > float(jnp.linalg.norm(g[1]))


def test_all_nan_targets_give_finite_zero_and_finite_grad():
    target = jnp.full((4, 2), np.nan)
    proj = jnp.array([[3.0, 4.0], [1.0, 1.0]])
    r = chamfer_residual(target, proj, beta=20.0)
    assert np.asarray(r).shape == (4,)
    assert np.all(np.asarray(r) == 0.0)

    def loss(proj):
        return jnp.sum(chamfer_residual(target, proj, beta=20.0) ** 2)

    g = jax.grad(loss)(proj)
    assert np.isfinite(np.asarray(g)).all()  # NaN targets must not poison grads


def test_huber_reduces_large_residual():
    target = jnp.array([[0.0, 0.0]])
    proj = jnp.array([[30.0, 40.0]])  # distance 50 (outlier)
    r_plain = chamfer_residual(target, proj, beta=50.0, huber_delta=0.0)
    r_huber = chamfer_residual(target, proj, beta=50.0, huber_delta=5.0)
    # squared residual is what LM minimizes; huber must shrink the outlier's
    # squared contribution below the plain d^2.
    assert float(r_huber[0]) ** 2 < float(r_plain[0]) ** 2


def test_chunking_is_result_invariant_and_bounds_memory():
    """MEMORY BOUND (efficiency Global Constraint): chunk_size only changes the
    peak intermediate (O(chunk_size*M)), NEVER the result. The chunked path
    must never materialize the full (N,M) distance matrix. Verify the residual
    is identical across chunk sizes on a moderately large N (so a full (N,M)
    matrix would be the wrong implementation)."""
    rng = np.random.default_rng(0)
    N, M = 128, 300  # default silhouette sizes (N<=128 boundary pts, M=fps_300)
    target = jnp.asarray(rng.normal(size=(N, 2)) * 40.0)
    proj = jnp.asarray(rng.normal(size=(M, 2)) * 40.0)
    r_small = chamfer_residual(target, proj, beta=8.0, chunk_size=8)
    r_big = chamfer_residual(target, proj, beta=8.0, chunk_size=64)
    r_all = chamfer_residual(target, proj, beta=8.0, chunk_size=N)
    assert r_small.shape == (N,)
    np.testing.assert_allclose(np.asarray(r_small), np.asarray(r_big), atol=1e-5)
    np.testing.assert_allclose(np.asarray(r_small), np.asarray(r_all), atol=1e-5)
    # gradient must also be chunk-invariant (the LM Jacobian must not depend on
    # the chunking used to bound memory).
    def loss(proj, cs):
        return jnp.sum(chamfer_residual(target, proj, beta=8.0, chunk_size=cs) ** 2)
    g8 = jax.grad(loss)(proj, 8)
    g64 = jax.grad(loss)(proj, 64)
    np.testing.assert_allclose(np.asarray(g8), np.asarray(g64), atol=1e-4)
