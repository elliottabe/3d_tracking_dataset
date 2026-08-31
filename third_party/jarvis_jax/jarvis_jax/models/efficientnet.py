"""NNX port of the JARVIS-HybridNet EfficientNet-b3 backbone.

Reference (PyTorch, source of truth for the exact math):
  third_party/JARVIS-HybridNet/jarvis/efficienttrack/efficientnet.py
    (MBConvBlock.forward L90-123, EfficientNet.forward L178-188)
  third_party/JARVIS-HybridNet/jarvis/efficienttrack/model.py
    (EfficientNet wrapper L511-552 -- the ``save_idxs``/truncation logic that
    turns the plain classifier backbone into a 3-feature-map [P3,P4,P5]
    front-end for the BiFPN)
  third_party/JARVIS-HybridNet/jarvis/efficienttrack/utils.py
    (round_filters/round_repeats, b0 block-args table, b3 = width 1.1 / depth 1.2)

Tensors are NHWC throughout (JAX convention); PyTorch checkpoints are NCHW and
are transposed on load.

JARVIS-custom MBConv quirk (see gotcha in task brief): for blocks belonging to
one of the first four *stages* (stage index < 4, where a "stage" is one entry
of the 7-entry b0 block-args table -- NOT the flattened block-list position)
the ``_expand_conv`` that PyTorch constructs is never applied in ``forward``;
instead ``_depthwise_conv`` is built as a plain (non-grouped) conv mapping
``input_filters -> input_filters*expand_ratio`` directly. Only for stage index
>= 4 does the standard expand-conv -> grouped-depthwise-conv path run, and only
there is ``_gn0``/swish absent (PyTorch has those two lines commented out, so
even in that branch there is no norm/activation between expand and depthwise).

This backbone was trained with a 4th (SAM3 mask) input channel appended to
RGB, so ``EfficientNetB3`` defaults to ``in_channels=4``.
"""
from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
import numpy as np
from flax import nnx


# ---------------------------------------------------------------------------
# round_filters / round_repeats -- transcribed from utils.py, b3 uses
# width_coefficient=1.1, depth_coefficient=1.2, depth_divisor=8, min_depth=None
# ---------------------------------------------------------------------------
def round_filters(filters: int, width_coefficient: float = 1.1,
                   depth_divisor: int = 8, min_depth: int | None = None) -> int:
    multiplier = width_coefficient
    if not multiplier:
        return filters
    divisor = depth_divisor
    min_depth = min_depth or divisor
    filters *= multiplier
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats: int, depth_coefficient: float = 1.2) -> int:
    multiplier = depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


# ---------------------------------------------------------------------------
# b0 base block-args table (efficientnet() in utils.py), decoded from the
# string notation -- kernel_size, num_repeat, input_filters, output_filters,
# expand_ratio, stride, se_ratio (id_skip is always True for every string).
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class _BaseStage:
    kernel_size: int
    num_repeat: int
    input_filters: int
    output_filters: int
    expand_ratio: int
    stride: int
    se_ratio: float


_BASE_STAGES = [
    _BaseStage(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
               expand_ratio=1, stride=1, se_ratio=0.25),
    _BaseStage(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
               expand_ratio=6, stride=2, se_ratio=0.25),
    _BaseStage(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
               expand_ratio=6, stride=2, se_ratio=0.25),
    _BaseStage(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
               expand_ratio=6, stride=2, se_ratio=0.25),
    _BaseStage(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
               expand_ratio=6, stride=1, se_ratio=0.25),
    _BaseStage(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
               expand_ratio=6, stride=2, se_ratio=0.25),
    _BaseStage(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
               expand_ratio=6, stride=1, se_ratio=0.25),
]


@dataclasses.dataclass(frozen=True)
class _BlockMeta:
    """Per (flattened) block metadata, mirroring one MBConvBlock instance."""
    stage_idx: int          # the `block_idx` passed to MBConvBlock in PyTorch
    kernel_size: int
    stride: int
    input_filters: int
    output_filters: int
    expand_ratio: int
    se_ratio: float
    id_skip: bool = True


def _build_all_block_meta(width_coefficient: float = 1.1,
                           depth_coefficient: float = 1.2) -> list[_BlockMeta]:
    """Reproduce EfficientNet.__init__'s block-list construction (full 23
    blocks for b3, before any truncation)."""
    metas: list[_BlockMeta] = []
    for stage_idx, base in enumerate(_BASE_STAGES):
        inp = round_filters(base.input_filters, width_coefficient)
        oup = round_filters(base.output_filters, width_coefficient)
        num_repeat = round_repeats(base.num_repeat, depth_coefficient)

        metas.append(_BlockMeta(
            stage_idx=stage_idx, kernel_size=base.kernel_size, stride=base.stride,
            input_filters=inp, output_filters=oup, expand_ratio=base.expand_ratio,
            se_ratio=base.se_ratio,
        ))
        for _ in range(num_repeat - 1):
            metas.append(_BlockMeta(
                stage_idx=stage_idx, kernel_size=base.kernel_size, stride=1,
                input_filters=oup, output_filters=oup, expand_ratio=base.expand_ratio,
                se_ratio=base.se_ratio,
            ))
    return metas


def _compute_save_idxs_and_truncate(metas: list[_BlockMeta]):
    """Transcribe model.py::EfficientNet.__init__'s save_idxs/last_idx loop.

    Returns (truncated_metas, save_idxs) where save_idxs has one more entry
    than truncated_metas is needed for (indices 0..len(truncated_metas)),
    matching the PyTorch ``self.save_idxs[idx+1]`` lookup in forward().
    """
    ignore_first = True
    last_idx = 0
    save_idxs: list[bool] = []
    for idx, meta in enumerate(metas):
        is_stride2 = meta.stride == 2
        if ignore_first and is_stride2:
            ignore_first = False
            save_idxs.append(False)
        else:
            save_idxs.append(is_stride2)
            if is_stride2:
                last_idx = idx - 1
    truncated = metas[: last_idx + 1]
    return truncated, save_idxs


def instance_norm(x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Param-free InstanceNorm2d (affine=False, track_running_stats=False).

    x: NHWC. Mean/var computed per-sample per-channel over the H,W axes
    (biased variance, matching PyTorch's InstanceNorm2d).
    """
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)
    var = jnp.var(x, axis=(1, 2), keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def _pad_same(k: int) -> int:
    return (k - 1) // 2


class MBConvBlockJAX(nnx.Module):
    """Mobile Inverted Residual Block (JARVIS-custom variant).

    See module docstring for the stage_idx<4 vs >=4 quirk.
    """

    def __init__(self, meta: _BlockMeta, *, rngs: nnx.Rngs):
        self.stage_idx = meta.stage_idx
        self.stride = meta.stride
        self.input_filters = meta.input_filters
        self.output_filters = meta.output_filters
        self.expand_ratio = meta.expand_ratio
        self.id_skip = meta.id_skip
        self.has_se = meta.se_ratio is not None and 0 < meta.se_ratio <= 1

        inp = meta.input_filters
        oup = inp * meta.expand_ratio
        k = meta.kernel_size
        s = meta.stride
        pad = _pad_same(k)

        if meta.stage_idx < 4:
            # JARVIS quirk: expand_conv (if any) is never applied in forward;
            # depthwise_conv maps inp -> oup directly, non-grouped.
            self.expand_conv = None
            self.depthwise_conv = nnx.Conv(
                inp, oup, kernel_size=(k, k), strides=(s, s),
                padding=((pad, pad), (pad, pad)), use_bias=False,
                feature_group_count=1, rngs=rngs,
            )
        else:
            if meta.expand_ratio != 1:
                self.expand_conv = nnx.Conv(
                    inp, oup, kernel_size=(1, 1), strides=(1, 1),
                    padding=((0, 0), (0, 0)), use_bias=False, rngs=rngs,
                )
            else:
                self.expand_conv = None
            self.depthwise_conv = nnx.Conv(
                oup, oup, kernel_size=(k, k), strides=(s, s),
                padding=((pad, pad), (pad, pad)), use_bias=False,
                feature_group_count=oup, rngs=rngs,
            )

        if self.has_se:
            num_squeezed = max(1, int(meta.input_filters * meta.se_ratio))
            self.se_reduce = nnx.Conv(
                oup, num_squeezed, kernel_size=(1, 1), strides=(1, 1),
                padding=((0, 0), (0, 0)), use_bias=True, rngs=rngs,
            )
            self.se_expand = nnx.Conv(
                num_squeezed, oup, kernel_size=(1, 1), strides=(1, 1),
                padding=((0, 0), (0, 0)), use_bias=True, rngs=rngs,
            )
        else:
            self.se_reduce = None
            self.se_expand = None

        self.project_conv = nnx.Conv(
            oup, meta.output_filters, kernel_size=(1, 1), strides=(1, 1),
            padding=((0, 0), (0, 0)), use_bias=False, rngs=rngs,
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        if self.stage_idx < 4:
            x = self.depthwise_conv(x)
        else:
            if self.expand_conv is not None:
                x = self.expand_conv(inputs)
            x = self.depthwise_conv(x)

        x = instance_norm(x)
        x = nnx.silu(x)

        if self.has_se:
            x_sq = jnp.mean(x, axis=(1, 2), keepdims=True)
            x_sq = self.se_reduce(x_sq)
            x_sq = nnx.silu(x_sq)
            x_sq = self.se_expand(x_sq)
            x = nnx.sigmoid(x_sq) * x

        x = self.project_conv(x)
        x = instance_norm(x)

        if (self.id_skip and self.stride == 1
                and self.input_filters == self.output_filters):
            x = x + inputs
        return x


class EfficientNetB3(nnx.Module):
    """EfficientNet backbone (JARVIS-custom InstanceNorm variant), truncated to
    return (P3, P4, P5) feature maps.

    Despite the name (kept for backward compat -- this class was written for
    b3/``large`` first), this is fully parameterized by ``width_coefficient``/
    ``depth_coefficient`` and works for any of JARVIS's ``efficientnet-b{N}``
    entries: everything downstream of them (``round_filters``/``round_repeats``,
    ``_build_all_block_meta``, ``_compute_save_idxs_and_truncate``) is already
    generic -- only the two coefficients (and the resulting concrete per-stage
    shapes) differ per size. Defaults are the b3 values, so existing callers
    (``EfficientTrack``'s ``large`` config) are byte-identical.

    Mirrors ``jarvis/efficienttrack/model.py::EfficientNet`` (the BiFPN
    front-end wrapper), NOT the plain classifier's ``forward`` (which returns
    only the final tensor).

    Verified (see ``jarvis_jax.models.efficienttrack._MODEL_SIZE_TABLE`` for
    provenance) P3/P4/P5 channel dims by ``(width_coefficient, depth_coefficient)``,
    matching JARVIS's own ``conv_channel_coef`` per named size exactly:
      * b3 / ``large``  (1.1, 1.2) -> (24, 48, 120)
      * b1 / ``medium`` (1.0, 1.0) -> (24, 40, 112)
      * b0 / ``small``  (0.5, 0.5) -> (16, 24, 56)
    """

    def __init__(self, *, in_channels: int = 4, width_coefficient: float = 1.1,
                 depth_coefficient: float = 1.2, rngs: nnx.Rngs):
        self.in_channels = in_channels
        stem_out = round_filters(32, width_coefficient=width_coefficient)
        self.conv_stem = nnx.Conv(
            in_channels, stem_out, kernel_size=(3, 3), strides=(2, 2),
            padding=((1, 1), (1, 1)), use_bias=False, rngs=rngs,
        )

        all_metas = _build_all_block_meta(
            width_coefficient=width_coefficient, depth_coefficient=depth_coefficient)
        truncated_metas, save_idxs = _compute_save_idxs_and_truncate(all_metas)
        self._save_idxs = save_idxs  # plain python list, static (not a param)

        self.blocks = nnx.List([MBConvBlockJAX(m, rngs=rngs) for m in truncated_metas])

    def __call__(self, x: jnp.ndarray):
        x = self.conv_stem(x)
        x = instance_norm(x)
        x = nnx.silu(x)

        feature_maps = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if self._save_idxs[idx + 1]:
                feature_maps.append(x)
        assert len(feature_maps) == 3, (
            f"expected 3 feature maps (P3,P4,P5), got {len(feature_maps)}")
        return tuple(feature_maps)


# ---------------------------------------------------------------------------
# npz loading
# ---------------------------------------------------------------------------
def _conv_w(z, key) -> np.ndarray:
    """(out,in,kh,kw) -> (kh,kw,in,out). Also correct for a grouped/depthwise
    conv, where PyTorch's `in` dim is in_channels/groups (e.g. 1 for a full
    depthwise conv) -- the transpose is identical, only the interpretation of
    the resulting HWIO tensor (via feature_group_count) differs."""
    return np.transpose(z[key], (2, 3, 1, 0))


def load_backbone_from_npz(module: EfficientNetB3, z) -> EfficientNetB3:
    """Assign converted PyTorch weights (npz produced by Task 1's fixture
    export) into ``module`` in place and return it.

    Keys are prefixed ``w::backbone_net.model...`` in the fixture; only the
    weights actually used by the JARVIS-custom forward path are assigned
    (block stages < 4 have an unused ``_expand_conv``/``_gn0`` in the PyTorch
    checkpoint that this port intentionally never allocates/loads, matching
    forward() which never calls them).
    """
    prefix = "w::backbone_net.model."

    module.conv_stem.kernel.value = jnp.asarray(
        _conv_w(z, prefix + "_conv_stem.weight"))

    for i, block in enumerate(module.blocks):
        bp = f"{prefix}_blocks.{i}."
        if block.stage_idx >= 4 and block.expand_conv is not None:
            block.expand_conv.kernel.value = jnp.asarray(
                _conv_w(z, bp + "_expand_conv.weight"))
        block.depthwise_conv.kernel.value = jnp.asarray(
            _conv_w(z, bp + "_depthwise_conv.weight"))
        if block.has_se:
            block.se_reduce.kernel.value = jnp.asarray(
                _conv_w(z, bp + "_se_reduce.weight"))
            block.se_reduce.bias.value = jnp.asarray(z[bp + "_se_reduce.bias"])
            block.se_expand.kernel.value = jnp.asarray(
                _conv_w(z, bp + "_se_expand.weight"))
            block.se_expand.bias.value = jnp.asarray(z[bp + "_se_expand.bias"])
        block.project_conv.kernel.value = jnp.asarray(
            _conv_w(z, bp + "_project_conv.weight"))

    return module
