import jax, jax.numpy as jnp
from flax import nnx
from jarvis_jax.models.vit import PatchEmbed, Attention, MLP, Block

def test_patch_embed_shapes():
    m = PatchEmbed(in_ch=4, embed_dim=768, patch=16, rngs=nnx.Rngs(0))
    x = jnp.zeros((2, 448, 448, 4))
    out = m(x)
    assert out.shape == (2, 784, 768)

def test_attention_shape_and_names():
    m = Attention(dim=768, num_heads=12, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 785, 768))
    out = m(x)
    assert out.shape == (2, 785, 768)
    # fused qkv layout for clean timm porting
    assert m.qkv.kernel.value.shape == (768, 2304)
    assert m.proj.kernel.value.shape == (768, 768)

def test_mlp_and_block():
    mlp = MLP(dim=768, mlp_ratio=4, rngs=nnx.Rngs(0))
    assert mlp.fc1.kernel.value.shape == (768, 3072)
    assert mlp.fc2.kernel.value.shape == (3072, 768)
    blk = Block(dim=768, num_heads=12, mlp_ratio=4, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 785, 768))
    assert blk(x).shape == (2, 785, 768)
