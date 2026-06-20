import jax.numpy as jnp
from flax import nnx
from jarvis_jax import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose

def test_vitpose_forward():
    cfg = ViTPoseConfig()
    m = ViTPose(cfg, rngs=nnx.Rngs(0))
    out = m(jnp.zeros((1, 448, 448, 4)))
    assert out.shape == (1, 224, 224, 50)
    assert jnp.isfinite(out).all()
