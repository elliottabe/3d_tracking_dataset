"""ClassicDecoder: upsample ViT patch tokens (28×28) to heatmaps (224×224).

Three ×2 ConvTranspose stages (28→56→112→224) with BatchNorm+ReLU,
then a 1×1 Conv to num_keypoints.

Note on padding: flax 0.12.6 nnx.ConvTranspose with padding=((1,1),(1,1))
and kernel 4, stride 2 yields 54 (not 56). Using padding='SAME' with
kernel_size=(4,4) and strides=(2,2) gives exact doubling at each stage.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class _Up(nnx.Module):
    """Single ×2 upsample block: ConvTranspose → BatchNorm → ReLU."""

    def __init__(self, cin: int, cout: int, *, rngs: nnx.Rngs):
        # padding='SAME' with kernel (4,4) stride (2,2) gives exact ×2 spatial doubling
        # in flax 0.12.6 (padding=((1,1),(1,1)) gives 54 instead of 56).
        self.deconv = nnx.ConvTranspose(
            cin, cout,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            rngs=rngs,
        )
        self.bn = nnx.BatchNorm(cout, rngs=rngs)

    def __call__(self, x, *, use_running_average: bool = False):
        x = self.deconv(x)
        x = self.bn(x, use_running_average=use_running_average)
        return jax.nn.relu(x)


class ClassicDecoder(nnx.Module):
    """Decode ViT patch tokens to spatial heatmaps.

    Args:
        embed_dim: ViT token embedding dimension (e.g. 768).
        num_keypoints: Number of output heatmap channels (e.g. 50).
        rngs: Flax NNX Rngs.

    Call signature:
        tokens: (B, num_tokens, embed_dim) where num_tokens = 28*28 = 784.
        returns: (B, 224, 224, num_keypoints).
    """

    def __init__(self, embed_dim: int, num_keypoints: int, *, rngs: nnx.Rngs):
        self.up1 = _Up(embed_dim, 256, rngs=rngs)  # 28 → 56
        self.up2 = _Up(256, 256, rngs=rngs)         # 56 → 112
        self.up3 = _Up(256, 256, rngs=rngs)         # 112 → 224
        self.head = nnx.Conv(256, num_keypoints, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, tokens, *, use_running_average: bool = False):
        """Forward pass.

        Args:
            tokens: (B, num_tokens, embed_dim) — ViT patch token sequence.
            use_running_average: passed to BatchNorm layers (False = training).

        Returns:
            (B, 224, 224, num_keypoints) heatmap logits.
        """
        b, n, d = tokens.shape
        s = int(n ** 0.5)  # 28 for n=784
        x = tokens.reshape(b, s, s, d)                                    # (B,28,28,D)
        x = self.up1(x, use_running_average=use_running_average)           # (B,56,56,256)
        x = self.up2(x, use_running_average=use_running_average)           # (B,112,112,256)
        x = self.up3(x, use_running_average=use_running_average)           # (B,224,224,256)
        return self.head(x)                                                # (B,224,224,K)
