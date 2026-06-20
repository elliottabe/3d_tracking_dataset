import jax, jax.numpy as jnp
from flax import nnx
from jarvis_jax.models.vit import PatchEmbed

def test_patch_embed_shapes():
    m = PatchEmbed(in_ch=4, embed_dim=768, patch=16, rngs=nnx.Rngs(0))
    x = jnp.zeros((2, 448, 448, 4))
    out = m(x)
    assert out.shape == (2, 784, 768)
