import jax.numpy as jnp
from flax import nnx
from jarvis_jax.models.decoder import ClassicDecoder


def test_decoder_upsamples_to_224():
    m = ClassicDecoder(embed_dim=768, num_keypoints=50, rngs=nnx.Rngs(0))
    tokens = jnp.zeros((2, 784, 768))
    out = m(tokens)
    assert out.shape == (2, 224, 224, 50)
    assert jnp.isfinite(out).all()
