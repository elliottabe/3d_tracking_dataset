"""NNX port of the JARVIS v2vNet 3D U-Net.

Reference: JARVIS-HybridNet/jarvis/hybridnet/v2vnet.py

Architecture (channels-last, tensors are B,D,H,W,C):
  Basic3DBlock  — Conv3d(k,stride,pad=(k-1)//2) + InstanceNorm3d + ReLU + Dropout(0.2)
  Res3DBlock    — 2×Conv3d(3,1,1)+IN, residual add, ReLU + Dropout(0.2)
  Upsample3DBlock — ConvTranspose3d(k2,s2,pad0) + IN + ReLU + Dropout(0.2)
  EncoderDecorder — skip_res + Basic(stride-2 down) + mid_res + Upsample(up) + res + skip-add
  V2VNet        — front [Basic(in→2in,3,2) + Res] + EncoderDecorder + Conv(2in→out,1)

InstanceNorm3d == GroupNorm(num_groups=C) (one group per channel).
GroupNorm has no running stats, so use_running_average only affects Dropout.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class Basic3DBlock(nnx.Module):
    """Conv3d(k,stride,same-like pad) + InstanceNorm3d + ReLU + Dropout(0.2)."""

    def __init__(self, in_planes: int, out_planes: int, kernel_size: int,
                 stride: int = 1, *, rngs: nnx.Rngs):
        pad = (kernel_size - 1) // 2
        self.conv = nnx.Conv(
            in_planes, out_planes,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            strides=(stride, stride, stride),
            padding=((pad, pad), (pad, pad), (pad, pad)),
            rngs=rngs,
        )
        # InstanceNorm3d == GroupNorm with num_groups == num_features
        self.norm = nnx.GroupNorm(num_features=out_planes, num_groups=out_planes, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.2, rngs=rngs)

    def __call__(self, x, *, use_running_average: bool = False):
        x = self.conv(x)
        x = self.norm(x)
        x = jax.nn.relu(x)
        return self.dropout(x, deterministic=use_running_average)


class Res3DBlock(nnx.Module):
    """Two Conv3d(3,1,pad1)+IN layers with residual add, ReLU + Dropout(0.2).

    Matches PyTorch Res3DBlock: in_planes == out_planes required for residual.
    """

    def __init__(self, in_planes: int, out_planes: int, *, rngs: nnx.Rngs):
        # Branch: Conv(3,1,1) → IN → ReLU → Conv(3,1,1) → IN
        self.conv1 = nnx.Conv(
            in_planes, out_planes,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding=((1, 1), (1, 1), (1, 1)),
            rngs=rngs,
        )
        self.norm1 = nnx.GroupNorm(num_features=out_planes, num_groups=out_planes, rngs=rngs)
        self.conv2 = nnx.Conv(
            out_planes, out_planes,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding=((1, 1), (1, 1), (1, 1)),
            rngs=rngs,
        )
        self.norm2 = nnx.GroupNorm(num_features=out_planes, num_groups=out_planes, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.2, rngs=rngs)

    def __call__(self, x, *, use_running_average: bool = False):
        res = self.conv1(x)
        res = self.norm1(res)
        res = jax.nn.relu(res)
        res = self.conv2(res)
        res = self.norm2(res)
        out = jax.nn.relu(res + x)
        return self.dropout(out, deterministic=use_running_average)


class Upsample3DBlock(nnx.Module):
    """ConvTranspose3d(k2,s2,pad0) + InstanceNorm3d + ReLU + Dropout(0.2).

    kernel_size=2, stride=2 doubles each spatial dimension.
    """

    def __init__(self, in_planes: int, out_planes: int, kernel_size: int,
                 stride: int, *, rngs: nnx.Rngs):
        assert kernel_size == 2
        assert stride == 2
        # padding='VALID' (no padding) with kernel=2, stride=2 → exact ×2 upsample
        self.deconv = nnx.ConvTranspose(
            in_planes, out_planes,
            kernel_size=(2, 2, 2),
            strides=(2, 2, 2),
            padding='VALID',
            rngs=rngs,
        )
        self.norm = nnx.GroupNorm(num_features=out_planes, num_groups=out_planes, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.2, rngs=rngs)

    def __call__(self, x, *, use_running_average: bool = False):
        x = self.deconv(x)
        x = self.norm(x)
        x = jax.nn.relu(x)
        return self.dropout(x, deterministic=use_running_average)


class EncoderDecorder(nnx.Module):
    """Single-level U-Net encoder/decoder with one skip connection.

    Input: (B,D,H,W, input_channels*2)
    The skip_res1 preserves resolution; encoder_pool1 downsamples ×2;
    mid_res processes the bottleneck; decoder_upsample1 upsamples ×2;
    decoder_res1 refines; output is skip + decoded.
    """

    def __init__(self, input_channels: int, *, rngs: nnx.Rngs):
        c = input_channels
        self.skip_res1 = Res3DBlock(c * 2, c * 2, rngs=rngs)
        self.encoder_pool1 = Basic3DBlock(c * 2, c * 4, 2, 2, rngs=rngs)
        self.mid_res = Res3DBlock(c * 4, c * 4, rngs=rngs)
        self.decoder_upsample1 = Upsample3DBlock(c * 4, c * 2, 2, 2, rngs=rngs)
        self.decoder_res1 = Res3DBlock(c * 2, c * 2, rngs=rngs)

    def __call__(self, x, *, use_running_average: bool = False):
        ura = use_running_average
        res1 = self.skip_res1(x, use_running_average=ura)
        x = self.encoder_pool1(x, use_running_average=ura)
        x = self.mid_res(x, use_running_average=ura)
        x = self.decoder_upsample1(x, use_running_average=ura)
        x = self.decoder_res1(x, use_running_average=ura)
        return x + res1


class V2VNet(nnx.Module):
    """3D volumetric-to-volumetric U-Net (NNX port of JARVIS v2vNet).

    Args:
        input_channels:  C_in  (e.g. 50)
        output_channels: C_out (e.g. 50)
        rngs: Flax NNX Rngs

    Call signature:
        x: (B, D, H, W, input_channels)
        use_running_average: disables dropout when True (eval mode)
        returns: (B, D//2, H//2, W//2, output_channels)
            e.g. (B,48,48,48,50) → (B,24,24,24,50)
    """

    def __init__(self, input_channels: int, output_channels: int, *, rngs: nnx.Rngs):
        c = input_channels
        # front_layers: Basic(in→2in, k3, stride2) + Res(2in,2in)
        self.front_basic = Basic3DBlock(c, c * 2, 3, 2, rngs=rngs)
        self.front_res = Res3DBlock(c * 2, c * 2, rngs=rngs)
        # encoder/decoder
        self.encoder_decoder = EncoderDecorder(c, rngs=rngs)
        # output 1×1×1 conv
        self.output_layer = nnx.Conv(
            c * 2, output_channels,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='VALID',
            rngs=rngs,
        )

    def __call__(self, x, *, use_running_average: bool = False):
        """Forward pass.

        Args:
            x: (B, D, H, W, C_in) — volumetric feature map
            use_running_average: True at eval (disables dropout)

        Returns:
            (B, D//2, H//2, W//2, C_out)
        """
        ura = use_running_average
        x = self.front_basic(x, use_running_average=ura)       # (B, D/2, H/2, W/2, 2C)
        x = self.front_res(x, use_running_average=ura)         # (B, D/2, H/2, W/2, 2C)
        x = self.encoder_decoder(x, use_running_average=ura)   # (B, D/2, H/2, W/2, 2C)
        x = self.output_layer(x)                               # (B, D/2, H/2, W/2, C_out)
        return x
