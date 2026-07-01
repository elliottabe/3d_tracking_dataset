import jax.numpy as jnp
import numpy as np
import pytest


def test_single_pixel_dilates_to_k_block():
    from jarvis_jax.models.dilation import dilate_mask_jax
    # a single hot pixel at the center of an 11x11 field -> a k x k block for odd k.
    for k in (3, 5, 7):
        m = jnp.zeros((11, 11)).at[5, 5].set(1.0)
        out = np.asarray(dilate_mask_jax(m, k))
        r = k // 2
        block = out[5 - r:5 + r + 1, 5 - r:5 + r + 1]
        assert block.shape == (k, k)
        assert np.all(block == 1.0), f"k={k}: interior not fully set"
        assert out.sum() == float(k * k), f"k={k}: expected exactly k*k hot pixels, got {out.sum()}"


def test_k1_and_k0_are_identity():
    from jarvis_jax.models.dilation import dilate_mask_jax
    rng = np.random.default_rng(0)
    m = jnp.asarray((rng.random((7, 9)) > 0.5).astype("float32"))
    assert np.array_equal(np.asarray(dilate_mask_jax(m, 1)), np.asarray(m))
    assert np.array_equal(np.asarray(dilate_mask_jax(m, 0)), np.asarray(m))


def test_batched_nhwc_preserves_shape_and_is_per_sample():
    from jarvis_jax.models.dilation import dilate_mask_jax
    # NHWC: (B=2, H=9, W=9, C=1); one hot pixel in each sample at different spots.
    m = jnp.zeros((2, 9, 9, 1))
    m = m.at[0, 4, 4, 0].set(1.0).at[1, 0, 0, 0].set(1.0)  # sample 1 hot at a corner
    out = np.asarray(dilate_mask_jax(m, 3))
    assert out.shape == (2, 9, 9, 1)
    assert out[0, 3:6, 3:6, 0].sum() == 9.0            # center: full 3x3
    assert out[0].sum() == 9.0
    assert out[1, 0:2, 0:2, 0].sum() == 4.0            # corner: clipped to 2x2 by SAME padding
    assert out[1].sum() == 4.0


def test_nhw_shape_supported():
    from jarvis_jax.models.dilation import dilate_mask_jax
    m = jnp.zeros((3, 8, 8)).at[:, 4, 4].set(1.0)
    out = np.asarray(dilate_mask_jax(m, 5))
    assert out.shape == (3, 8, 8)
    assert np.all(out[:, 2:7, 2:7] == 1.0)


def test_binary_stays_binary_float_stays_float():
    from jarvis_jax.models.dilation import dilate_mask_jax
    m = jnp.asarray(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype="float32"))
    out = dilate_mask_jax(m, 3)
    assert out.dtype == m.dtype
    assert set(np.unique(np.asarray(out)).tolist()) <= {0.0, 1.0}


def test_bool_and_int_masks_supported():
    # SAM3 masks are stored bool; -inf is not representable in bool/int, so the
    # impl must dilate in float and cast back, preserving the input dtype.
    from jarvis_jax.models.dilation import dilate_mask_jax
    mb = jnp.zeros((5, 5), dtype=bool).at[2, 2].set(True)
    ob = dilate_mask_jax(mb, 3)
    assert ob.dtype == jnp.bool_
    assert np.all(np.asarray(ob)[1:4, 1:4]) and np.asarray(ob).sum() == 9
    mi = jnp.zeros((5, 5), dtype=jnp.int32).at[2, 2].set(1)
    oi = dilate_mask_jax(mi, 3)
    assert oi.dtype == jnp.int32
    assert np.asarray(oi).sum() == 9
