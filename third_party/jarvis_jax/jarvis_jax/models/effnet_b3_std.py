"""JAX/NNX port of the STANDARD torchvision EfficientNet-b3 (BatchNorm).

Reference (PyTorch, source of truth): ``torchvision.models.efficientnet``
(``efficientnet_b3`` / ``MBConv`` / ``SqueezeExcitation`` / ``Conv2dNormActivation``).
This is a *different* backbone from ``jarvis_jax.models.efficientnet.EfficientNetB3``
(the JARVIS-HybridNet custom InstanceNorm variant with an idiosyncratic forward
quirk) -- this module is the plain, faithful torchvision port used to
warm-start from ``EfficientNet_B3_Weights.IMAGENET1K_V1`` for a fair
ViT-vs-EfficientNet keypoint-backbone comparison.

Architecture, transcribed from inspecting the fixture's ``w::features.*``
state_dict keys (torchvision groups the 7 MBConv stages + stem + head into
``net.features``, a 9-entry ``nn.Sequential``; each of the 7 stage entries is
itself an ``nn.Sequential`` of repeated ``MBConv`` blocks). b3 uses
``width_mult=1.2``, ``depth_mult=1.4`` (rounded via the standard
round_filters/round_repeats, divisor=8) applied to the b0 base config
``(expand_ratio, kernel, stride, in_ch, out_ch, num_layers)``:

    (1, 3, 1,  32,  16, 1)
    (6, 3, 2,  16,  24, 2)
    (6, 5, 2,  24,  40, 2)
    (6, 3, 2,  40,  80, 3)
    (6, 5, 1,  80, 112, 3)
    (6, 5, 2, 112, 192, 4)
    (6, 3, 1, 192, 320, 1)

which for b3 resolves to the concrete per-stage config baked into
``_STAGE_CFG`` below (stem out=40; stage out channels 24,32,48,96,136,232,384;
block counts 2,3,3,5,5,6,2). ``features[i]`` for ``i`` in 1..7 is stage ``i``;
``features[0]`` is the stem, ``features[8]`` is the final 1x1 head conv
(1536-ch) -- NOT used here (we need the /4, /8, /16 taps for the EfficientTrack
FPN -- matching the InstanceNorm backbone's tap strides -- which land at the
*end* of stages 2, 3, 5 respectively: P3=32ch/112x112 (/4), P4=48ch/56x56
(/8), P5=136ch/28x28 (/16), for a 448x448 input. NOT the /8,/16,/32 taps
(stages 3,5,7 -> 48,136,384) that a naive "last three stages" reading would
suggest: the EfficientTrack head deconvs x2 from P3, so P3 must be /4 (112)
to reach the required 224x224 heatmap output, not /8 (56).

Per-block structure (``MBConv.block`` in torchvision, a plain
``nn.Sequential``):
  - if ``expand_ratio != 1``: ``block.0`` = expand 1x1 Conv2dNormActivation
    (conv -> BN -> SiLU), else omitted (stage 1 has expand_ratio=1 for ALL its
    blocks, not just the first -- confirmed from the fixture: both
    ``features.1.0`` and ``features.1.1`` lack an expand conv).
  - depthwise kxk Conv2dNormActivation (conv -> BN -> SiLU), grouped conv with
    ``groups == expanded_channels``.
  - ``SqueezeExcitation``: global-avgpool -> ``fc1`` (1x1 conv, reduce) ->
    SiLU -> ``fc2`` (1x1 conv, expand) -> sigmoid -> scale. Squeeze channels
    = ``max(1, in_channels_of_this_block // 4)`` where ``in_channels_of_this_
    block`` is THIS block's own (pre-expansion) input channel count -- verified
    against every fc1 shape in the fixture (e.g. stage5 block0: in=96 ->
    squeeze=24; stage5 block1..4: in=136 -> squeeze=34).
  - project 1x1 Conv2dNormActivation WITHOUT activation (conv -> BN only).
  - residual add iff ``stride == 1 and in_channels == out_channels`` (only the
    first block of a stage can change stride/channels; stochastic depth is
    identity at eval, so this is unconditional at eval time).

Tensors are NHWC throughout (JAX convention); PyTorch (NCHW) checkpoints are
transposed on load in ``load_b3std_from_npz``.
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
from flax import nnx

# BatchNorm defaults matching torch.nn.BatchNorm2d(eps=1e-5, momentum=0.1).
# NOTE: flax's `momentum` is the *retention* factor for the running average
# (new = momentum*old + (1-momentum)*batch), i.e. the complement of torch's
# momentum (torch: new = (1-momentum)*old + momentum*batch). torch default
# momentum=0.1 <=> flax momentum=0.9. This only matters if training
# (use_running_average=False, updating the running stats); for parity we
# always run eval with use_running_average=True using the loaded running
# stats directly, so this value is inert here but kept correct for
# completeness / future fine-tuning.
_BN_EPS = 1e-5
_BN_MOMENTUM = 0.9


@dataclasses.dataclass(frozen=True)
class _StageCfg:
    kernel_size: int
    stride: int          # stride of the FIRST block in the stage
    expand_ratio: int
    in_channels: int      # input channels of the FIRST block
    out_channels: int
    num_blocks: int


# Concrete b3 stage config (width_mult=1.2, depth_mult=1.4), transcribed from
# the fixture's `w::features.<i>.*` shapes -- see module docstring.
_STAGE_CFG = [
    _StageCfg(kernel_size=3, stride=1, expand_ratio=1, in_channels=40, out_channels=24, num_blocks=2),
    _StageCfg(kernel_size=3, stride=2, expand_ratio=6, in_channels=24, out_channels=32, num_blocks=3),
    _StageCfg(kernel_size=5, stride=2, expand_ratio=6, in_channels=32, out_channels=48, num_blocks=3),
    _StageCfg(kernel_size=3, stride=2, expand_ratio=6, in_channels=48, out_channels=96, num_blocks=5),
    _StageCfg(kernel_size=5, stride=1, expand_ratio=6, in_channels=96, out_channels=136, num_blocks=5),
    _StageCfg(kernel_size=5, stride=2, expand_ratio=6, in_channels=136, out_channels=232, num_blocks=6),
    _StageCfg(kernel_size=3, stride=1, expand_ratio=6, in_channels=232, out_channels=384, num_blocks=2),
]

_STEM_OUT = 40
# 1-based stage index (matching torchvision's `features.<i>`) whose LAST
# block's output is a P3/P4/P5 FPN tap. P3/P4/P5 = /4,/8,/16 strides (stages
# 2,3,5 -> 32,48,136 channels), matching the InstanceNorm EfficientTrack
# backbone's tap strides so the head's x2 deconv from P3 (/4) lands on
# 224x224 for a 448x448 input -- NOT stages 3,5,7 (/8,/16,/32).
_TAP_STAGE = {2: "p3", 3: "p4", 5: "p5"}


def _pad_same(k: int) -> int:
    return (k - 1) // 2


def _bn(num_features: int, *, rngs: nnx.Rngs) -> nnx.BatchNorm:
    return nnx.BatchNorm(
        num_features, use_running_average=False, epsilon=_BN_EPS,
        momentum=_BN_MOMENTUM, rngs=rngs,
    )


class SqueezeExciteStd(nnx.Module):
    """torchvision SqueezeExcitation: avgpool -> fc1 -> SiLU -> fc2 -> sigmoid -> scale."""

    def __init__(self, channels: int, squeeze_channels: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Conv(
            channels, squeeze_channels, kernel_size=(1, 1), strides=(1, 1),
            padding=((0, 0), (0, 0)), use_bias=True, rngs=rngs,
        )
        self.fc2 = nnx.Conv(
            squeeze_channels, channels, kernel_size=(1, 1), strides=(1, 1),
            padding=((0, 0), (0, 0)), use_bias=True, rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        s = jnp.mean(x, axis=(1, 2), keepdims=True)
        s = self.fc1(s)
        s = nnx.silu(s)
        s = self.fc2(s)
        s = nnx.sigmoid(s)
        return x * s


class MBConvStd(nnx.Module):
    """Standard (torchvision) MBConv block: optional expand -> depthwise ->
    SE -> project, with a residual add when stride==1 and in==out channels."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, expand_ratio: int, *, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        expanded = in_channels * expand_ratio
        pad = _pad_same(kernel_size)

        if expand_ratio != 1:
            self.expand_conv = nnx.Conv(
                in_channels, expanded, kernel_size=(1, 1), strides=(1, 1),
                padding=((0, 0), (0, 0)), use_bias=False, rngs=rngs,
            )
            self.expand_bn = _bn(expanded, rngs=rngs)
        else:
            self.expand_conv = None
            self.expand_bn = None

        self.depthwise_conv = nnx.Conv(
            expanded, expanded, kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride), padding=((pad, pad), (pad, pad)),
            use_bias=False, feature_group_count=expanded, rngs=rngs,
        )
        self.depthwise_bn = _bn(expanded, rngs=rngs)

        squeeze_channels = max(1, in_channels // 4)
        self.se = SqueezeExciteStd(expanded, squeeze_channels, rngs=rngs)

        self.project_conv = nnx.Conv(
            expanded, out_channels, kernel_size=(1, 1), strides=(1, 1),
            padding=((0, 0), (0, 0)), use_bias=False, rngs=rngs,
        )
        self.project_bn = _bn(out_channels, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, use_running_average: bool) -> jnp.ndarray:
        inputs = x
        if self.expand_conv is not None:
            x = self.expand_conv(x)
            x = self.expand_bn(x, use_running_average=use_running_average)
            x = nnx.silu(x)
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x, use_running_average=use_running_average)
        x = nnx.silu(x)
        x = self.se(x)
        x = self.project_conv(x)
        x = self.project_bn(x, use_running_average=use_running_average)
        if self.use_residual:
            x = x + inputs
        return x


class EfficientNetB3Std(nnx.Module):
    """Standard (torchvision) EfficientNet-b3 backbone, truncated to the
    (P3, P4, P5) FPN taps (/4, /8, /16 strides; channels 32, 48, 136).

    Supports ``in_channels`` != 3 (e.g. 4, for a later SAM3-mask-aware
    variant) by simply sizing the stem conv's input dimension accordingly;
    the ImageNet-pretrained parity test always uses the default 3.
    """

    def __init__(self, *, in_channels: int = 3, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.stem_conv = nnx.Conv(
            in_channels, _STEM_OUT, kernel_size=(3, 3), strides=(2, 2),
            padding=((1, 1), (1, 1)), use_bias=False, rngs=rngs,
        )
        self.stem_bn = _bn(_STEM_OUT, rngs=rngs)

        blocks: list[MBConvStd] = []
        stage_end_idx: dict[int, int] = {}  # 1-based stage -> flat block idx of its last block
        flat = 0
        for stage_i, cfg in enumerate(_STAGE_CFG, start=1):
            for j in range(cfg.num_blocks):
                stride = cfg.stride if j == 0 else 1
                bin_ch = cfg.in_channels if j == 0 else cfg.out_channels
                blocks.append(MBConvStd(
                    bin_ch, cfg.out_channels, cfg.kernel_size, stride,
                    cfg.expand_ratio, rngs=rngs,
                ))
                flat += 1
            stage_end_idx[stage_i] = flat - 1
        self.blocks = nnx.List(blocks)
        # flat block idx (0-based) -> tap name, for the last block of stages 3/5/7
        self._tap_at_idx = {stage_end_idx[s]: name for s, name in _TAP_STAGE.items()}

    def __call__(self, x_nhwc: jnp.ndarray, *, use_running_average: bool = True):
        x = self.stem_conv(x_nhwc)
        x = self.stem_bn(x, use_running_average=use_running_average)
        x = nnx.silu(x)

        taps: dict[str, jnp.ndarray] = {}
        for idx, block in enumerate(self.blocks):
            x = block(x, use_running_average=use_running_average)
            tap_name = self._tap_at_idx.get(idx)
            if tap_name is not None:
                taps[tap_name] = x

        assert set(taps) == {"p3", "p4", "p5"}, f"missing taps: {taps.keys()}"
        return taps["p3"], taps["p4"], taps["p5"]


# ---------------------------------------------------------------------------
# npz loading (torchvision state_dict -> NNX params)
# ---------------------------------------------------------------------------
def _conv_w(z, key: str) -> np.ndarray:
    """(out,in,kh,kw) -> (kh,kw,in,out). For a grouped/depthwise conv,
    PyTorch's `in` dim is in_channels/groups (1 for a full depthwise conv);
    the transpose is identical, only feature_group_count changes how the
    resulting HWIO tensor is interpreted."""
    return np.transpose(z[key], (2, 3, 1, 0))


def _assign_bn(bn: nnx.BatchNorm, z, prefix: str, assigned: set[str]) -> None:
    bn.scale.value = jnp.asarray(z[prefix + ".weight"])
    bn.bias.value = jnp.asarray(z[prefix + ".bias"])
    bn.mean.value = jnp.asarray(z[prefix + ".running_mean"])
    bn.var.value = jnp.asarray(z[prefix + ".running_var"])
    assigned.update({
        prefix + ".weight", prefix + ".bias",
        prefix + ".running_mean", prefix + ".running_var",
    })


def load_b3std_from_npz(module: EfficientNetB3Std, z) -> EfficientNetB3Std:
    """Assign torchvision ``efficientnet_b3`` state_dict weights (npz keys
    prefixed ``w::``) into ``module`` in place. Asserts every expected weight
    key (stem + all 7 stages' blocks; excludes the unused ``features.8.*``
    head conv and the ``classifier.*`` FC head, which this truncated backbone
    never allocates) was assigned, to catch silent naming-mismatch misses."""
    assigned: set[str] = set()

    # Stem conv: torchvision kernel is (3,3,3,stem_out). ``module`` may have
    # been sized for a different ``in_channels`` (e.g. 4, RGB + SAM3 mask
    # channel) -- any extra input channel(s) are zero-initialized so a zero
    # 4th channel is exactly a no-op at init, mirroring the ViT MAE 3->4
    # patch-embed inflation (see convert/load_weights.py). No-op when
    # ``in_channels == 3`` (the parity-test default): the branch is skipped.
    w = _conv_w(z, "w::features.0.0.weight")  # (3,3,3,stem_out)
    want_in = module.stem_conv.kernel.value.shape[2]
    if want_in != w.shape[2]:
        w_inflated = np.zeros((w.shape[0], w.shape[1], want_in, w.shape[3]), dtype=w.dtype)
        w_inflated[:, :, : w.shape[2], :] = w
        w = w_inflated
    module.stem_conv.kernel.value = jnp.asarray(w)
    assigned.add("w::features.0.0.weight")
    _assign_bn(module.stem_bn, z, "w::features.0.1", assigned)

    flat = 0
    for stage_i, cfg in enumerate(_STAGE_CFG, start=1):
        for j in range(cfg.num_blocks):
            block = module.blocks[flat]
            bp = f"w::features.{stage_i}.{j}.block."
            sub = 0
            if block.expand_conv is not None:
                block.expand_conv.kernel.value = jnp.asarray(_conv_w(z, bp + f"{sub}.0.weight"))
                assigned.add(bp + f"{sub}.0.weight")
                _assign_bn(block.expand_bn, z, bp + f"{sub}.1", assigned)
                sub += 1
            block.depthwise_conv.kernel.value = jnp.asarray(_conv_w(z, bp + f"{sub}.0.weight"))
            assigned.add(bp + f"{sub}.0.weight")
            _assign_bn(block.depthwise_bn, z, bp + f"{sub}.1", assigned)
            sub += 1
            block.se.fc1.kernel.value = jnp.asarray(_conv_w(z, bp + f"{sub}.fc1.weight"))
            block.se.fc1.bias.value = jnp.asarray(z[bp + f"{sub}.fc1.bias"])
            block.se.fc2.kernel.value = jnp.asarray(_conv_w(z, bp + f"{sub}.fc2.weight"))
            block.se.fc2.bias.value = jnp.asarray(z[bp + f"{sub}.fc2.bias"])
            assigned.update({
                bp + f"{sub}.fc1.weight", bp + f"{sub}.fc1.bias",
                bp + f"{sub}.fc2.weight", bp + f"{sub}.fc2.bias",
            })
            sub += 1
            block.project_conv.kernel.value = jnp.asarray(_conv_w(z, bp + f"{sub}.0.weight"))
            assigned.add(bp + f"{sub}.0.weight")
            _assign_bn(block.project_bn, z, bp + f"{sub}.1", assigned)
            flat += 1
    assert flat == len(module.blocks)

    # Every w:: key that belongs to the stem + the 7 stages (i.e. not the
    # unused features.8 head conv / classifier) must have been consumed.
    expected = {
        k for k in z.files
        if k.startswith("w::features.") and not k.startswith("w::features.8.")
        and not k.endswith(".num_batches_tracked")
    }
    missing = expected - assigned
    extra = assigned - expected
    assert not missing, f"load_b3std_from_npz missed {len(missing)} keys: {sorted(missing)[:10]}"
    assert not extra, f"load_b3std_from_npz assigned unexpected keys: {sorted(extra)[:10]}"

    return module
