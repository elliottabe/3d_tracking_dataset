"""Tests for 3D MPJPE metric."""
import jax.numpy as jnp
import numpy as np
from jarvis_jax.eval.mpjpe_3d import mpjpe_3d


def test_mpjpe_3d_zero_and_ignores_invalid():
    kp = jnp.asarray(np.random.RandomState(1).rand(2, 4, 3).astype("float32"))
    valid = jnp.ones((2, 4), dtype=bool)
    assert float(mpjpe_3d(kp, kp, valid)) == 0.0
    pred = kp.at[0, 0, 0].add(10.0)
    v = jnp.ones((2, 4), dtype=bool).at[0, 0].set(False)
    assert float(mpjpe_3d(pred, kp, v)) == 0.0
