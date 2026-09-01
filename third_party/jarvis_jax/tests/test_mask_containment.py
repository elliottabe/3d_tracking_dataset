import numpy as np
import jax.numpy as jnp
import pytest
from jarvis_jax.tracking.mask_containment import bilinear_sample, containment_residual
from jarvis_jax.tracking.mask_sdf import mask_bbox, mask_to_sdf_crop


def _sdf_box(h=40, w=60):
    m = np.zeros((200, 300), bool)
    m[80:80 + h, 120:120 + w] = True
    sdf, gs, go = mask_to_sdf_crop(m, mask_bbox(m, 0.4), (64, 64))
    return jnp.asarray(sdf), jnp.asarray(gs), jnp.asarray(go), m


def test_inside_costs_nothing_outside_costs_distance():
    sdf, gs, go, _ = _sdf_box()
    pts = jnp.asarray([[150.0, 100.0],        # box centre, inside
                       [250.0, 100.0]])       # well to the right, outside
    r = containment_residual(pts, sdf, gs, go, jnp.ones(2))
    assert float(r[0]) == pytest.approx(0.0, abs=1e-6), "inside must be free"
    assert float(r[1]) > 5.0, "outside must be penalised in original px"


def test_present_false_zeroes_the_residual():
    sdf, gs, go, _ = _sdf_box()
    pts = jnp.asarray([[250.0, 100.0]])
    assert float(containment_residual(pts, sdf, gs, go, jnp.ones(1),
                                      present=False)[0]) == 0.0


def test_gradient_points_back_toward_the_mask():
    """The whole purpose: an outside vertex must feel an inward pull."""
    import jax
    sdf, gs, go, _ = _sdf_box()
    f = lambda x: float(containment_residual(
        jnp.asarray([[x, 100.0]]), sdf, gs, go, jnp.ones(1))[0])
    g = jax.grad(lambda x: containment_residual(
        jnp.stack([jnp.stack([x, jnp.asarray(100.0)])]),
        sdf, gs, go, jnp.ones(1))[0])(jnp.asarray(250.0))
    assert float(g) > 0.0, "moving further right must increase the cost"
    assert f(250.0) > f(200.0), "closer to the mask must cost less"


def test_far_outside_the_grid_keeps_a_finite_gradient():
    """The recovered code adds an `overflow` term so a vertex beyond the SDF crop
    does not sit on a clamp plateau with zero gradient."""
    sdf, gs, go, _ = _sdf_box()
    near = containment_residual(jnp.asarray([[250.0, 100.0]]), sdf, gs, go, jnp.ones(1))
    far = containment_residual(jnp.asarray([[900.0, 100.0]]), sdf, gs, go, jnp.ones(1))
    assert float(far[0]) > float(near[0]) + 50.0, "no plateau far from the crop"


def test_conf_scales_the_residual_linearly():
    sdf, gs, go, _ = _sdf_box()
    pts = jnp.asarray([[250.0, 100.0]])
    a = float(containment_residual(pts, sdf, gs, go, jnp.asarray([1.0]))[0])
    b = float(containment_residual(pts, sdf, gs, go, jnp.asarray([0.5]))[0])
    assert b == pytest.approx(0.5 * a, rel=1e-5)


def test_bilinear_sample_matches_the_array_at_integer_coords():
    img = jnp.asarray(np.arange(25, dtype=np.float32).reshape(5, 5))
    got = bilinear_sample(img, jnp.asarray([[2.0, 3.0]]))   # (x=2, y=3)
    assert float(got[0]) == pytest.approx(float(img[3, 2]))
