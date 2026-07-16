import jax
import jax.numpy as jnp
import numpy as np
from jarvis_jax.tracking.silhouette_containment import bilinear_sample, containment_residual


def _ramp_sdf(H=20, W=20):
    # signed distance-like field: negative in the left half, positive in the right
    xs = np.arange(W)[None, :].repeat(H, 0).astype(np.float32)
    return jnp.asarray(xs - (W / 2))     # d = column - 10


def test_bilinear_matches_manual():
    img = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4))
    # point (x=1.5, y=0.5): interp of rows 0..1, cols 1..2
    val = bilinear_sample(img, jnp.asarray([[1.5, 0.5]]))
    manual = (img[0, 1] + img[0, 2] + img[1, 1] + img[1, 2]) / 4.0
    assert float(abs(val[0] - manual)) < 1e-5


def test_bilinear_clamps_out_of_bounds():
    img = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4))
    assert float(bilinear_sample(img, jnp.asarray([[-5.0, -5.0]]))[0]) == float(img[0, 0])
    assert float(bilinear_sample(img, jnp.asarray([[99.0, 99.0]]))[0]) == float(img[3, 3])


def test_inside_zero_outside_positive():
    sdf = _ramp_sdf()
    gs = jnp.array([1.0, 1.0]); go = jnp.array([0.0, 0.0])
    # vert projecting to column 3 (d=-7, inside) -> 0 ; column 17 (d=+7, outside) -> +7
    r = containment_residual(jnp.asarray([[3.0, 10.0], [17.0, 10.0]]), sdf, gs, go,
                             conf=jnp.ones(2), margin=0.0)
    assert float(r[0]) == 0.0
    assert abs(float(r[1]) - 7.0) < 1e-4


def test_confidence_scales():
    sdf = _ramp_sdf(); gs = jnp.array([1.0, 1.0]); go = jnp.array([0.0, 0.0])
    r = containment_residual(jnp.asarray([[17.0, 10.0]]), sdf, gs, go,
                             conf=jnp.array([0.5]), margin=0.0)
    assert abs(float(r[0]) - 3.5) < 1e-4


def test_present_gate_zeros():
    sdf = _ramp_sdf(); gs = jnp.array([1.0, 1.0]); go = jnp.array([0.0, 0.0])
    r = containment_residual(jnp.asarray([[17.0, 10.0]]), sdf, gs, go,
                             conf=jnp.ones(1), margin=0.0, present=False)
    assert float(r[0]) == 0.0


def test_gradient_pulls_inward_and_is_finite():
    sdf = _ramp_sdf(); gs = jnp.array([1.0, 1.0]); go = jnp.array([0.0, 0.0])
    # a vert far outside the grid (column 40) should have finite grad pointing to -x (inward)
    def loss(px):
        p = jnp.stack([px, jnp.array(10.0)])[None, :]
        return containment_residual(p, sdf, gs, go, conf=jnp.ones(1)).sum()
    g = jax.grad(loss)(jnp.array(40.0))
    assert np.isfinite(float(g))
    assert float(g) > 0    # increasing x increases residual -> descent moves x inward
