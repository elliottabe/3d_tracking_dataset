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
