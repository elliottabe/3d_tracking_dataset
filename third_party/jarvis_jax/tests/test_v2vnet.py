# tests/test_v2vnet.py
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jarvis_jax.hybridnet.v2vnet import V2VNet


def test_v2vnet_shapes_and_finite():
    m = V2VNet(50, 50, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).rand(1, 48, 48, 48, 50).astype("float32"))
    y = m(x, use_running_average=False)
    assert y.shape == (1, 24, 24, 24, 50), y.shape
    assert bool(jnp.isfinite(y).all())


def test_v2vnet_eval_mode():
    """Dropout should be disabled in eval mode (deterministic=True)."""
    m = V2VNet(50, 50, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(1).rand(1, 48, 48, 48, 50).astype("float32"))
    y = m(x, use_running_average=True)
    assert y.shape == (1, 24, 24, 24, 50), y.shape
    assert bool(jnp.isfinite(y).all())
