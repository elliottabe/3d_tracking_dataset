import jax, jax.numpy as jnp
from flax import nnx

class PatchEmbed(nnx.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch: int, *, rngs: nnx.Rngs):
        # NHWC conv; non-overlapping patches.
        self.proj = nnx.Conv(in_ch, embed_dim, kernel_size=(patch, patch),
                             strides=(patch, patch), padding="VALID", rngs=rngs)

    def __call__(self, x):              # x: (B,H,W,in_ch)
        x = self.proj(x)                # (B, H/16, W/16, embed_dim)
        b, h, w, d = x.shape
        return x.reshape(b, h * w, d)   # (B, N, D)

class Attention(nnx.Module):
    def __init__(self, dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x):                              # (B,N,D)
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)              # (3,B,heads,N,hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, n, d)
        return self.proj(out)
