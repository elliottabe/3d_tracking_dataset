"""NNX port of the JARVIS-HybridNet EfficientTrack head (BiFPN + detection head).

Reference (PyTorch, source of truth for the exact math):
  third_party/JARVIS-HybridNet/jarvis/efficienttrack/model.py
    EfficientTrackBackbone.forward (L117-133) -- top-level wiring: backbone ->
      stacked BiFPN blocks -> weighted 3-way fusion -> first_conv -> deconv1
      (res2) / final_conv1 (res1).
    BiFPN_first.forward (L449-507) -- first BiFPN cell; adds the channel
      projection convs (P3/P4/P5 -> num_channels) and builds P6/P7 from P5.
    BiFPN.forward (L304-356) -- subsequent BiFPN cells (all 5 levels already
      at num_channels).
    SeparableConvBlock (L183-235) -- depthwise -> pointwise -> InstanceNorm
      (optional) -> SiLU (optional).

Tensors are NHWC throughout (JAX convention); PyTorch checkpoints are NCHW and
are transposed on load (see ``jarvis_jax.models.efficientnet._conv_w``).

Key gotchas (see task brief):
  * BiFPN internal fusion weights (``p*_w1``/``p*_w2``) are normalized as
    ``relu(w) / (sum(relu(w)) + 1e-4)``.
  * The head's ``weights_cat`` (3-way P3/P4/P5 fusion) is normalized with
    **softplus** instead of relu: ``softplus(w) / (sum(softplus(w)) + 1e-4)``.
  * Upsample is nearest-neighbor (``jax.image.resize(..., method="nearest")``);
    downsample is ``MaxPool2d(2,2)`` (``flax.linen.max_pool`` with
    ``padding="VALID"``, which discards any odd trailing row/col exactly like
    PyTorch's default ``MaxPool2d``).
  * ``ConvTranspose`` weight conversion: PyTorch's ``ConvTranspose2d`` weight
    is laid out ``(in, out, kh, kw)`` (note: **reversed** in/out order vs a
    regular ``Conv2d``, whose weight is ``(out, in, kh, kw)``). NNX's
    ``nnx.ConvTranspose`` with ``transpose_kernel=True`` expects a kernel of
    shape ``(kh, kw, out, in)`` and internally flips the spatial axes to
    reproduce PyTorch/Keras "gradient-of-conv" semantics -- this is the
    correct mode for numerical parity (``transpose_kernel=False`` uses the raw
    un-flipped kernel and does *not* match PyTorch). Because the transpose
    ``(2, 3, 1, 0)`` maps torch's ``(in, out, kh, kw)`` to ``(kh, kw, out,
    in)`` -- the exact same axis permutation ``efficientnet._conv_w`` already
    uses for regular convs (which maps ``(out, in, kh, kw)`` -> ``(kh, kw, in,
    out)``) -- the very same ``_conv_w`` helper is reused unchanged for
    ``deconv1``.
  * The PyTorch padding of 1 (kernel=4, stride=2) does not carry over
    literally: NNX/JAX's ``padding`` argument for ``ConvTranspose`` means
    something different (see the ``nnx.ConvTranspose`` docstring: to invert a
    ``Conv`` with padding ``P`` and dilation ``D`` you pass ``D*(kernel-1) -
    P``). Verified empirically (padding=((2,2),(2,2)) is the only explicit
    integer padding, among the values tried, that reproduces the expected
    112->224 spatial doubling for kernel=4/stride=2/torch-padding=1: ``D*(k-1)
    - P = 1*3 - 1 = 2``).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as linen_fn
from flax import nnx

from jarvis_jax.models.efficientnet import (
    EfficientNetB3,
    _conv_w,
    instance_norm,
    load_backbone_from_npz,
)
from jarvis_jax.models.effnet_b3_std import (
    EfficientNetB3Std,
    load_b3std_from_npz,
)


# ---------------------------------------------------------------------------
# small free-function helpers (no params)
# ---------------------------------------------------------------------------
def _relu_norm(w: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    """BiFPN internal fusion-weight normalization: relu(w) / (sum + eps)."""
    w = jax.nn.relu(w)
    return w / (jnp.sum(w) + eps)


def _softplus_norm(w: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    """Head ``weights_cat`` normalization: softplus(w) / (sum + eps)."""
    w = jax.nn.softplus(w)
    return w / (jnp.sum(w) + eps)


def _upsample_nearest(x: jnp.ndarray, factor: int) -> jnp.ndarray:
    n, h, w, c = x.shape
    return jax.image.resize(x, (n, h * factor, w * factor, c), method="nearest")


def _upsample_to(x: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    """Nearest-neighbor resize of ``x``'s spatial dims to match ``ref``'s.

    Used ONLY for the p7_in -> p6-level upsample inside BiFPN(_first): with
    the InstanceNorm EfficientNetB3 backbone, P5's spatial size is always even
    at every level down to P7 (e.g. 28 -> p6=14 -> p7=7 -> upsample x2 -> 14,
    an exact match for p6_in, identical to ``_upsample_nearest(x, 2)``: this
    call is a byte-identical no-op for that path). The standard-stride
    EfficientNetB3Std backbone (EfficientTrackBN) has a smaller P5 (true /32
    stride, e.g. 14 for a 448 input), so p6=7 is ODD and p7=maxpool(7)=3
    (``_maxpool2``'s VALID padding floor-divides, dropping the trailing row/
    col) -- a fixed factor-2 upsample of p7 (3 -> 6) would then mismatch p6
    (7). Resizing to p6_in's actual shape (rather than assuming a clean
    doubling) fixes this the same way for both backbones, since jax.image
    .resize(..., method="nearest") to an explicit output shape that happens to
    be exactly 2x the input is identical to the fixed-factor call."""
    n, h, w, c = ref.shape
    return jax.image.resize(x, (n, h, w, c), method="nearest")


def _maxpool2(x: jnp.ndarray) -> jnp.ndarray:
    """MaxPool2d(2, 2), matching PyTorch's default (no padding, floor divide)."""
    return linen_fn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
class SeparableConvBlockJAX(nnx.Module):
    """depthwise(k3,pad1,groups=in,bias=False) -> pointwise(1x1,bias=True) ->
    optional param-free InstanceNorm -> optional SiLU."""

    def __init__(self, in_channels: int, out_channels: int | None = None,
                 norm: bool = True, activation: bool = False, *, rngs: nnx.Rngs):
        out_channels = out_channels if out_channels is not None else in_channels
        self.depthwise_conv = nnx.Conv(
            in_channels, in_channels, kernel_size=(3, 3), strides=(1, 1),
            padding=((1, 1), (1, 1)), use_bias=False,
            feature_group_count=in_channels, rngs=rngs,
        )
        self.pointwise_conv = nnx.Conv(
            in_channels, out_channels, kernel_size=(1, 1), strides=(1, 1),
            padding=((0, 0), (0, 0)), use_bias=True, rngs=rngs,
        )
        self.norm = norm
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm:
            x = instance_norm(x)
        if self.activation:
            x = nnx.silu(x)
        return x


class _ChannelProject(nnx.Module):
    """Conv2d(1x1, bias=True) -> param-free InstanceNorm2d.

    Mirrors the ``nn.Sequential(nn.Conv2d(cin, cout, 1), nn.InstanceNorm2d(cout))``
    blocks used by ``BiFPN_first`` for p3/p4/p5_down_channel(_2) and the conv
    half of p5_to_p6 (the MaxPool2d half is applied separately in forward()).
    """

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_channels, out_channels, kernel_size=(1, 1), strides=(1, 1),
            padding=((0, 0), (0, 0)), use_bias=True, rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return instance_norm(self.conv(x))


_SEP_CONV_NAMES = (
    "conv6_up", "conv5_up", "conv4_up", "conv3_up",
    "conv4_down", "conv5_down", "conv6_down", "conv7_down",
)
_W1_NAMES = ("p6_w1", "p5_w1", "p4_w1", "p3_w1")
_W2_NAMES = ("p4_w2", "p5_w2", "p6_w2", "p7_w2")


def _w2_init_len(name: str) -> int:
    """p7_w2 only ever combines 2 terms (w[0]*p7_in + w[1]*maxpool(p6_out) --
    see both ``__call__`` methods below); the PyTorch checkpoint stores it as
    a 2-element tensor (verified: ``w::bifpn.{i}.p7_w2`` shape (2,) in the
    EfficientTrack-large fixture, vs. (3,) for p4_w2/p5_w2/p6_w2). Declaring
    the un-loaded default at the wrong length (3) is harmless for the direct
    npz-loading path (``load_efficienttrack_from_npz`` overwrites ``.value``
    in place, shape and all) but breaks the Orbax eval_shape/restore round
    trip (Task 4), whose abstract target reflects this __init__-declared
    shape."""
    return 2 if name == "p7_w2" else 3


class BiFPNFirstJAX(nnx.Module):
    """First BiFPN cell: projects P3/P4/P5 to ``num_channels`` and derives P6/P7."""

    def __init__(self, num_channels: int, conv_channels: tuple[int, int, int],
                 *, rngs: nnx.Rngs):
        c3, c4, c5 = conv_channels
        for name in _SEP_CONV_NAMES:
            setattr(self, name, SeparableConvBlockJAX(num_channels, rngs=rngs))

        self.p5_down_channel = _ChannelProject(c5, num_channels, rngs=rngs)
        self.p4_down_channel = _ChannelProject(c4, num_channels, rngs=rngs)
        self.p3_down_channel = _ChannelProject(c3, num_channels, rngs=rngs)
        self.p5_to_p6_conv = _ChannelProject(c5, num_channels, rngs=rngs)
        self.p4_down_channel_2 = _ChannelProject(c4, num_channels, rngs=rngs)
        self.p5_down_channel_2 = _ChannelProject(c5, num_channels, rngs=rngs)

        for name in _W1_NAMES:
            setattr(self, name, nnx.Param(jnp.ones((2,))))
        for name in _W2_NAMES:
            setattr(self, name, nnx.Param(jnp.ones((_w2_init_len(name),))))

    def __call__(self, inputs):
        p3, p4, p5 = inputs

        p6_in = _maxpool2(self.p5_to_p6_conv(p5))
        p7_in = _maxpool2(p6_in)
        p3_in = self.p3_down_channel(p3)
        p4_in = self.p4_down_channel(p4)
        p5_in = self.p5_down_channel(p5)

        w = _relu_norm(self.p6_w1.value)
        p6_up = self.conv6_up(nnx.silu(w[0] * p6_in + w[1] * _upsample_to(p7_in, p6_in)))

        w = _relu_norm(self.p5_w1.value)
        p5_up = self.conv5_up(nnx.silu(w[0] * p5_in + w[1] * _upsample_nearest(p6_up, 2)))

        w = _relu_norm(self.p4_w1.value)
        p4_up = self.conv4_up(nnx.silu(w[0] * p4_in + w[1] * _upsample_nearest(p5_up, 2)))

        w = _relu_norm(self.p3_w1.value)
        p3_out = self.conv3_up(nnx.silu(w[0] * p3_in + w[1] * _upsample_nearest(p4_up, 2)))

        p4_in2 = self.p4_down_channel_2(p4)
        p5_in2 = self.p5_down_channel_2(p5)

        w = _relu_norm(self.p4_w2.value)
        p4_out = self.conv4_down(
            nnx.silu(w[0] * p4_in2 + w[1] * p4_up + w[2] * _maxpool2(p3_out)))

        w = _relu_norm(self.p5_w2.value)
        p5_out = self.conv5_down(
            nnx.silu(w[0] * p5_in2 + w[1] * p5_up + w[2] * _maxpool2(p4_out)))

        w = _relu_norm(self.p6_w2.value)
        p6_out = self.conv6_down(
            nnx.silu(w[0] * p6_in + w[1] * p6_up + w[2] * _maxpool2(p5_out)))

        w = _relu_norm(self.p7_w2.value)
        p7_out = self.conv7_down(nnx.silu(w[0] * p7_in + w[1] * _maxpool2(p6_out)))

        return (p3_out, p4_out, p5_out, p6_out, p7_out)


class BiFPNJAX(nnx.Module):
    """Subsequent BiFPN cells: all 5 pyramid levels already at ``num_channels``."""

    def __init__(self, num_channels: int, *, rngs: nnx.Rngs):
        for name in _SEP_CONV_NAMES:
            setattr(self, name, SeparableConvBlockJAX(num_channels, rngs=rngs))
        for name in _W1_NAMES:
            setattr(self, name, nnx.Param(jnp.ones((2,))))
        for name in _W2_NAMES:
            setattr(self, name, nnx.Param(jnp.ones((_w2_init_len(name),))))

    def __call__(self, inputs):
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        w = _relu_norm(self.p6_w1.value)
        p6_up = self.conv6_up(nnx.silu(w[0] * p6_in + w[1] * _upsample_to(p7_in, p6_in)))

        w = _relu_norm(self.p5_w1.value)
        p5_up = self.conv5_up(nnx.silu(w[0] * p5_in + w[1] * _upsample_nearest(p6_up, 2)))

        w = _relu_norm(self.p4_w1.value)
        p4_up = self.conv4_up(nnx.silu(w[0] * p4_in + w[1] * _upsample_nearest(p5_up, 2)))

        w = _relu_norm(self.p3_w1.value)
        p3_out = self.conv3_up(nnx.silu(w[0] * p3_in + w[1] * _upsample_nearest(p4_up, 2)))

        w = _relu_norm(self.p4_w2.value)
        p4_out = self.conv4_down(
            nnx.silu(w[0] * p4_in + w[1] * p4_up + w[2] * _maxpool2(p3_out)))

        w = _relu_norm(self.p5_w2.value)
        p5_out = self.conv5_down(
            nnx.silu(w[0] * p5_in + w[1] * p5_up + w[2] * _maxpool2(p4_out)))

        w = _relu_norm(self.p6_w2.value)
        p6_out = self.conv6_down(
            nnx.silu(w[0] * p6_in + w[1] * p6_up + w[2] * _maxpool2(p5_out)))

        w = _relu_norm(self.p7_w2.value)
        p7_out = self.conv7_down(nnx.silu(w[0] * p7_in + w[1] * _maxpool2(p6_out)))

        return (p3_out, p4_out, p5_out, p6_out, p7_out)


# Size table, transcribed VERBATIM from
# third_party/JARVIS-HybridNet/jarvis/efficienttrack/model.py
# (EfficientTrackBackbone.__init__'s ``if model_size == ...`` ladder) --
# fpn_num_filters/fpn_cell_repeats/final_layer_sizes/conv_channel_coef per
# named size. ``backbone_width``/``backbone_depth`` are the
# ``width_coefficient``/``depth_coefficient`` each size's PyTorch-side
# ``backbone_compound_coef`` (0/1/3) resolves to via JARVIS's OWN
# ``efficientnet_params()`` table (jarvis/efficienttrack/utils.py:150-164 --
# NOT the standard EfficientNet paper table; JARVIS's is shifted by one index,
# e.g. its "b3" = the official b2 config): b0->(0.5,0.5), b1->(1.0,1.0),
# b3->(1.1,1.2). Cross-checked independently: feeding each (backbone_width,
# backbone_depth) pair through ``_build_all_block_meta``/
# ``_compute_save_idxs_and_truncate`` reproduces the matching
# ``conv_channel_coef`` in this table exactly (24/48/120 for large, 24/40/112
# for medium, 16/24/56 for small) -- see EfficientNetB3's docstring.
_MODEL_SIZE_TABLE = {
    "small":  dict(fpn_num_filters=56,  fpn_cell_repeats=3, final_layer_sizes=64,
                   conv_channel_coef=(16, 24, 56),  backbone_width=0.5, backbone_depth=0.5),
    "medium": dict(fpn_num_filters=88,  fpn_cell_repeats=4, final_layer_sizes=88,
                   conv_channel_coef=(24, 40, 112), backbone_width=1.0, backbone_depth=1.0),
    "large":  dict(fpn_num_filters=160, fpn_cell_repeats=6, final_layer_sizes=160,
                   conv_channel_coef=(24, 48, 120), backbone_width=1.1, backbone_depth=1.2),
}


class EfficientTrack(nnx.Module):
    """EfficientTrack: EfficientNet backbone + BiFPN + detection head,
    parameterized over JARVIS's ``small``/``medium``/``large`` size table
    (see ``_MODEL_SIZE_TABLE``). ``model_size="large"`` (the default,
    preserved for backward compat) is byte-identical to the original
    large-only port: fpn_num_filters=160, fpn_cell_repeats=6 (1x BiFPN_first +
    5x BiFPN), final_layer_sizes=160, conv_channel_coef=[24, 48, 120].
    ``model_size="medium"`` (EfficientNet-b1 backbone) is CenterDetect's
    config: fpn_num_filters=88, fpn_cell_repeats=4, final_layer_sizes=88,
    conv_channel_coef=[24, 40, 112].

    ``__call__`` returns **res2** (the deconv'd, full-resolution heatmap) --
    the tensor HybridNet consumes. ``forward_both`` also exposes res1 (half
    resolution, from ``final_conv1``) for parity testing / diagnostics.
    """

    def __init__(self, *, num_joints: int = 50, in_channels: int = 4,
                 # 4 = JARVIS unified_V3_masked "large"-config convention
                 # (RGB + SAM3 mask channel); upstream PyTorch reference
                 # defaults to 3 (RGB only). CenterDetect (any size) is
                 # always in_channels=3 (no mask channel).
                 model_size: str = "large",
                 rngs: nnx.Rngs):
        if model_size not in _MODEL_SIZE_TABLE:
            raise ValueError(
                f"model_size must be one of {sorted(_MODEL_SIZE_TABLE)}, got {model_size!r}")
        size_cfg = _MODEL_SIZE_TABLE[model_size]
        self.model_size = model_size
        self.num_joints = num_joints
        self.fpn_num_filters = size_cfg["fpn_num_filters"]
        self.fpn_cell_repeats = size_cfg["fpn_cell_repeats"]
        self.final_layer_sizes = size_cfg["final_layer_sizes"]
        self.conv_channel_coef = size_cfg["conv_channel_coef"]

        self.backbone = EfficientNetB3(
            in_channels=in_channels,
            width_coefficient=size_cfg["backbone_width"],
            depth_coefficient=size_cfg["backbone_depth"],
            rngs=rngs)

        cells = [BiFPNFirstJAX(self.fpn_num_filters, self.conv_channel_coef, rngs=rngs)]
        cells += [BiFPNJAX(self.fpn_num_filters, rngs=rngs)
                  for _ in range(1, self.fpn_cell_repeats)]
        self.bifpn = nnx.List(cells)

        self.weights_cat = nnx.Param(jnp.ones((3,)))
        self.first_conv = SeparableConvBlockJAX(
            self.fpn_num_filters, self.final_layer_sizes, norm=True,
            activation=False, rngs=rngs)

        # ConvTranspose2d(final_layer_sizes -> J, k4, s2, pad1, bias=False) in
        # PyTorch. See module docstring for the transpose_kernel=True +
        # padding=((2,2),(2,2)) derivation (112 -> 224 doubling).
        self.deconv1 = nnx.ConvTranspose(
            self.final_layer_sizes, num_joints, kernel_size=(4, 4), strides=(2, 2),
            padding=((2, 2), (2, 2)), use_bias=False, transpose_kernel=True,
            rngs=rngs,
        )
        self.final_conv1 = nnx.Conv(
            self.final_layer_sizes, num_joints, kernel_size=(3, 3), strides=(1, 1),
            padding=((1, 1), (1, 1)), use_bias=False, rngs=rngs,
        )

    def forward_both(self, x: jnp.ndarray):
        features = self.backbone(x)
        for cell in self.bifpn:
            features = cell(features)

        x3 = _upsample_nearest(features[2], 4)
        x2 = _upsample_nearest(features[1], 2)
        w = _softplus_norm(self.weights_cat.value)
        x1 = w[0] * features[0] + w[1] * x2 + w[2] * x3

        pre = self.first_conv(x1)
        res2 = self.deconv1(pre)
        res1 = self.final_conv1(pre)
        return res1, res2

    def __call__(self, x: jnp.ndarray, *, use_running_average: bool = False) -> jnp.ndarray:
        """``use_running_average`` is accepted (and ignored) only so this model
        satisfies the 2D trainer's generic model contract
        (``model(img, use_running_average=...)``, see train/train.py). It is a
        no-op here: EfficientTrack has no dropout / batch-stats layers --
        InstanceNorm (see ``instance_norm``) is computed per-example from the
        current activations, so there is no running-average state to switch."""
        _, res2 = self.forward_both(x)
        return res2

    def predict_heatmaps(self, crops_nhwc: jnp.ndarray) -> jnp.ndarray:
        """Front-end contract: accepts (*lead, H, W, C) and returns
        (*lead, 224, 224, J) -- any number of leading batch dims (e.g. plain
        ``(N, H, W, C)`` or ``(B, num_cam, H, W, C)``).

        The network body (``forward_both``/``__call__``) hard-assumes exactly
        rank-4 NHWC input (``_upsample_nearest`` unpacks ``n, h, w, c =
        x.shape``; ``instance_norm`` reduces over hardcoded axes ``(1, 2)``),
        so any extra leading dims are flattened into one batch axis here and
        restored on the way out -- identical math to a caller doing this
        flatten/unflatten itself (Task 7: this lets ``HybridNet3D`` pass
        un-flattened ``(B, num_cam, H, W, C)`` crops straight through, e.g.
        for per-camera crop-level mask fusion, with zero numerical change for
        existing rank-4 callers, where this flatten/unflatten is a no-op).
        """
        *lead, H, W, C = crops_nhwc.shape
        flat = crops_nhwc.reshape((-1, H, W, C))
        out = self(flat)
        J = out.shape[-1]
        return out.reshape((*lead, out.shape[1], out.shape[2], J))


# ---------------------------------------------------------------------------
# npz loading
# ---------------------------------------------------------------------------
def _load_sep_conv(block: SeparableConvBlockJAX, z, prefix: str) -> None:
    block.depthwise_conv.kernel.value = jnp.asarray(
        _conv_w(z, prefix + ".depthwise_conv.weight"))
    block.pointwise_conv.kernel.value = jnp.asarray(
        _conv_w(z, prefix + ".pointwise_conv.weight"))
    block.pointwise_conv.bias.value = jnp.asarray(z[prefix + ".pointwise_conv.bias"])


def _load_chanproj(block: _ChannelProject, z, prefix: str) -> None:
    block.conv.kernel.value = jnp.asarray(_conv_w(z, prefix + ".weight"))
    block.conv.bias.value = jnp.asarray(z[prefix + ".bias"])


def _load_bifpn_common(cell, z, prefix: str) -> None:
    for name in _SEP_CONV_NAMES:
        _load_sep_conv(getattr(cell, name), z, f"{prefix}.{name}")
    for name in _W1_NAMES + _W2_NAMES:
        getattr(cell, name).value = jnp.asarray(z[f"{prefix}.{name}"])


def load_efficienttrack_from_npz(module: EfficientTrack, z) -> EfficientTrack:
    """Assign converted PyTorch weights (npz from Task 1's fixture export)
    into ``module`` in place and return it: backbone + BiFPN + head."""
    load_backbone_from_npz(module.backbone, z)

    for i, cell in enumerate(module.bifpn):
        prefix = f"w::bifpn.{i}"
        _load_bifpn_common(cell, z, prefix)
        if i == 0:
            _load_chanproj(cell.p5_down_channel, z, prefix + ".p5_down_channel.0")
            _load_chanproj(cell.p4_down_channel, z, prefix + ".p4_down_channel.0")
            _load_chanproj(cell.p3_down_channel, z, prefix + ".p3_down_channel.0")
            _load_chanproj(cell.p5_to_p6_conv, z, prefix + ".p5_to_p6.0")
            _load_chanproj(cell.p4_down_channel_2, z, prefix + ".p4_down_channel_2.0")
            _load_chanproj(cell.p5_down_channel_2, z, prefix + ".p5_down_channel_2.0")

    _load_sep_conv(module.first_conv, z, "w::first_conv")
    module.weights_cat.value = jnp.asarray(z["w::weights_cat"])
    module.deconv1.kernel.value = jnp.asarray(_conv_w(z, "w::deconv1.weight"))
    module.final_conv1.kernel.value = jnp.asarray(_conv_w(z, "w::final_conv1.weight"))
    return module


# ---------------------------------------------------------------------------
# EfficientTrack-BN: standard (BatchNorm) ImageNet-pretrainable EfficientNet-b3
# backbone variant, for a fair ViT-vs-EfficientNet keypoint-backbone
# comparison (unlike plain EfficientTrack above, which has no pretrained
# checkpoint in this pipeline and is always random-init).
# ---------------------------------------------------------------------------
class EfficientTrackBN(nnx.Module):
    """EfficientTrack driven by ``EfficientNetB3Std`` (the standard torchvision
    BatchNorm EfficientNet-b3, ImageNet-pretrainable) instead of the JARVIS
    ``EfficientNetB3`` (InstanceNorm, always random-init). The BiFPN(_first) +
    detection head are IDENTICAL to ``EfficientTrack`` (same InstanceNorm
    building blocks, always freshly initialized) -- only the backbone and its
    channel-projection ``conv_channel_coef`` differ:

    ================  ===================  ==========================
    variant           backbone norm         conv_channel_coef (P3,P4,P5)
    ================  ===================  ==========================
    EfficientTrack     InstanceNorm          (24, 48, 120)
    EfficientTrackBN   BatchNorm (ImageNet)  (32, 48, 136)
    ================  ===================  ==========================

    ``__call__``/``forward_both`` thread ``use_running_average`` straight into
    the BN backbone (training: batch stats + running-stat update; eval:
    running stats) -- unlike ``EfficientTrack.__call__``, where the flag is a
    accepted-but-ignored no-op (InstanceNorm has no running-average state).
    """

    def __init__(self, *, num_joints: int = 50, in_channels: int = 4,
                 rngs: nnx.Rngs):
        self.num_joints = num_joints
        self.fpn_num_filters = 160
        self.fpn_cell_repeats = 6
        self.final_layer_sizes = 160
        # EfficientNetB3Std's P3/P4/P5 tap channels at /4,/8,/16 strides (see
        # effnet_b3_std.py module docstring) -- NOT the InstanceNorm
        # backbone's (24,48,120), and NOT the /8,/16,/32 taps (48,136,384)
        # that a naive "last three stages" reading would suggest (that would
        # make P3 land at /8=112->56, breaking the head's x2 deconv to 224).
        self.conv_channel_coef = (32, 48, 136)

        self.backbone = EfficientNetB3Std(in_channels=in_channels, rngs=rngs)

        cells = [BiFPNFirstJAX(self.fpn_num_filters, self.conv_channel_coef, rngs=rngs)]
        cells += [BiFPNJAX(self.fpn_num_filters, rngs=rngs)
                  for _ in range(1, self.fpn_cell_repeats)]
        self.bifpn = nnx.List(cells)

        self.weights_cat = nnx.Param(jnp.ones((3,)))
        self.first_conv = SeparableConvBlockJAX(
            self.fpn_num_filters, self.final_layer_sizes, norm=True,
            activation=False, rngs=rngs)

        self.deconv1 = nnx.ConvTranspose(
            self.final_layer_sizes, num_joints, kernel_size=(4, 4), strides=(2, 2),
            padding=((2, 2), (2, 2)), use_bias=False, transpose_kernel=True,
            rngs=rngs,
        )
        self.final_conv1 = nnx.Conv(
            self.final_layer_sizes, num_joints, kernel_size=(3, 3), strides=(1, 1),
            padding=((1, 1), (1, 1)), use_bias=False, rngs=rngs,
        )

    def forward_both(self, x: jnp.ndarray, *, use_running_average: bool):
        features = self.backbone(x, use_running_average=use_running_average)
        for cell in self.bifpn:
            features = cell(features)

        x3 = _upsample_nearest(features[2], 4)
        x2 = _upsample_nearest(features[1], 2)
        w = _softplus_norm(self.weights_cat.value)
        x1 = w[0] * features[0] + w[1] * x2 + w[2] * x3

        pre = self.first_conv(x1)
        res2 = self.deconv1(pre)
        res1 = self.final_conv1(pre)
        return res1, res2

    def __call__(self, x: jnp.ndarray, *, use_running_average: bool = False) -> jnp.ndarray:
        """Unlike ``EfficientTrack.__call__``, ``use_running_average`` is NOT
        a no-op here: it is threaded into the BatchNorm backbone (training:
        batch stats computed + running mean/var updated in place;
        eval/inference: running stats used directly)."""
        _, res2 = self.forward_both(x, use_running_average=use_running_average)
        return res2

    def predict_heatmaps(self, crops_nhwc: jnp.ndarray,
                         *, use_running_average: bool = True) -> jnp.ndarray:
        """Front-end contract identical to ``EfficientTrack.predict_heatmaps``
        (accepts/returns arbitrary leading batch dims), but defaults to
        ``use_running_average=True`` since this is the inference entry point
        and the BN backbone's running stats -- not the current batch's -- are
        what should be used at predict time."""
        *lead, H, W, C = crops_nhwc.shape
        flat = crops_nhwc.reshape((-1, H, W, C))
        out = self(flat, use_running_average=use_running_average)
        J = out.shape[-1]
        return out.reshape((*lead, out.shape[1], out.shape[2], J))


def build_efficienttrack_bn_imagenet(num_joints: int, npz_path: str, *,
                                     in_channels: int = 4,
                                     rngs: nnx.Rngs) -> EfficientTrackBN:
    """Factory: build an ``EfficientTrackBN`` with the backbone warm-started
    from ImageNet (``load_b3std_from_npz`` on the torchvision ``efficientnet_b3``
    fixture npz at ``npz_path``) and the stem conv zero-inflated 3->``in_channels``
    (mirrors the ViT MAE 3->4 patch-embed inflation, see
    ``convert/load_weights.py``: a zero mask channel is a no-op at init). The
    BiFPN + detection head are left at their fresh ``nnx.Rngs`` init (there is
    no pretrained checkpoint for them in this pipeline -- they always train
    from scratch, same as plain ``EfficientTrack``)."""
    model = EfficientTrackBN(num_joints=num_joints, in_channels=in_channels, rngs=rngs)
    z = np.load(npz_path)
    load_b3std_from_npz(model.backbone, z)
    return model

    return module
