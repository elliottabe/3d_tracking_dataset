"""Convert the ``v2vNet.*`` submodule of a PyTorch ``HybridNet-*.pth``
checkpoint into an Orbax-backed NNX ``V2VNet``.

Reference (PyTorch, source of truth): ``jarvis/hybridnet/v2vnet.py``. State-dict
key layout (verified directly against ``HybridNet-large_final.pth``)::

    v2vNet.front_layers.0.block.0.{weight,bias}                  Basic3DBlock conv
    v2vNet.front_layers.1.res_branch.{0,3}.{weight,bias}          Res3DBlock convs
    v2vNet.encoder_decoder.skip_res1.res_branch.{0,3}.{weight,bias}
    v2vNet.encoder_decoder.encoder_pool1.block.0.{weight,bias}    Basic3DBlock conv
    v2vNet.encoder_decoder.mid_res.res_branch.{0,3}.{weight,bias}
    v2vNet.encoder_decoder.decoder_upsample1.block.0.{weight,bias}  ConvTranspose3d
    v2vNet.encoder_decoder.decoder_res1.res_branch.{0,3}.{weight,bias}
    v2vNet.output_layer.{weight,bias}                             1x1x1 Conv3d

``nn.InstanceNorm3d`` in the PyTorch model is constructed with its default
``affine=False`` (no ``weight``/``bias`` tensors at all -- confirmed: none of
the 24 ``v2vNet.*`` state_dict keys reference a norm layer). The NNX port's
``nnx.GroupNorm(num_groups=C)`` stand-in, by contrast, is affine by default
(``use_bias=True, use_scale=True``, ``scale_init=ones``, ``bias_init=zeros``).
Since ones/zeros is exactly the identity affine transform, and this converter
never assigns anything into ``.norm.scale``/``.norm.bias``, the freshly
constructed module's default-initialized norm params already reproduce
``affine=False`` -- no explicit action is required beyond *not* loading them.

Weight-layout conversions (verified with a synthetic forward-parity probe,
see Task-4 report -- max abs diff 0.0 against ``torch.nn.ConvTranspose3d``):
  * ``Conv3d`` weight ``(out, in, kD, kH, kW)`` -> NNX ``(kD, kH, kW, in, out)``
    via ``transpose(2, 3, 4, 1, 0)``.
  * ``ConvTranspose3d`` weight ``(in, out, kD, kH, kW)`` -> NNX
    ``(kD, kH, kW, out, in)`` -- via the SAME ``transpose(2, 3, 4, 1, 0)``
    permutation (only the semantic labels of axes 0/1 differ between the two
    weight layouts, not the required permutation), used together with
    ``nnx.ConvTranspose(..., transpose_kernel=True)`` in
    ``V2VNet.Upsample3DBlock`` (see fix + rationale in
    ``jarvis_jax/hybridnet/v2vnet.py``): the JAX default
    ``transpose_kernel=False`` reproduces a *different* op (the "true"
    mathematical transpose of cross-correlation) than PyTorch's
    ``ConvTranspose3d`` (adjoint of a flipped-kernel convolution).
"""
from __future__ import annotations

import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.hybridnet.v2vnet import V2VNet


def _conv3d_w(state_dict: dict, key: str) -> np.ndarray:
    """Permute a Conv3d/ConvTranspose3d weight for NNX.

    Works for both layouts because the permutation ``(2,3,4,1,0)`` moves the
    3 spatial axes to the front and swaps the remaining two -- exactly what's
    needed whether axes (0,1) are (out,in) [Conv3d] or (in,out)
    [ConvTranspose3d]; see module docstring.
    """
    w = state_dict[key]
    w = w.detach().cpu().numpy() if hasattr(w, "detach") else np.asarray(w)
    return np.transpose(w, (2, 3, 4, 1, 0))


def _bias(state_dict: dict, key: str) -> np.ndarray:
    b = state_dict[key]
    return b.detach().cpu().numpy() if hasattr(b, "detach") else np.asarray(b)


def _load_basic3d(block, state_dict: dict, prefix: str) -> None:
    block.conv.kernel.value = _conv3d_w(state_dict, prefix + ".block.0.weight")
    block.conv.bias.value = _bias(state_dict, prefix + ".block.0.bias")


def _load_res3d(block, state_dict: dict, prefix: str) -> None:
    block.conv1.kernel.value = _conv3d_w(state_dict, prefix + ".res_branch.0.weight")
    block.conv1.bias.value = _bias(state_dict, prefix + ".res_branch.0.bias")
    block.conv2.kernel.value = _conv3d_w(state_dict, prefix + ".res_branch.3.weight")
    block.conv2.bias.value = _bias(state_dict, prefix + ".res_branch.3.bias")


def _load_upsample3d(block, state_dict: dict, prefix: str) -> None:
    block.deconv.kernel.value = _conv3d_w(state_dict, prefix + ".block.0.weight")
    block.deconv.bias.value = _bias(state_dict, prefix + ".block.0.bias")


def load_v2vnet_from_state_dict(module: V2VNet, state_dict: dict, *,
                                 strip_prefix: str = "v2vNet.") -> V2VNet:
    """Assign converted PyTorch ``v2vNet.*`` weights into ``module`` in place."""
    if strip_prefix:
        sd = {k[len(strip_prefix):]: v for k, v in state_dict.items()
              if k.startswith(strip_prefix)}
    else:
        sd = dict(state_dict)

    _load_basic3d(module.front_basic, sd, "front_layers.0")
    _load_res3d(module.front_res, sd, "front_layers.1")

    ed = module.encoder_decoder
    _load_res3d(ed.skip_res1, sd, "encoder_decoder.skip_res1")
    _load_basic3d(ed.encoder_pool1, sd, "encoder_decoder.encoder_pool1")
    _load_res3d(ed.mid_res, sd, "encoder_decoder.mid_res")
    _load_upsample3d(ed.decoder_upsample1, sd, "encoder_decoder.decoder_upsample1")
    _load_res3d(ed.decoder_res1, sd, "encoder_decoder.decoder_res1")

    module.output_layer.kernel.value = _conv3d_w(sd, "output_layer.weight")
    module.output_layer.bias.value = _bias(sd, "output_layer.bias")

    return module


def convert_v2vnet_pth(pth_path: str, in_ch: int, out_ch: int, out_dir: str, *,
                        strip_prefix: str = "v2vNet.") -> V2VNet:
    """Load *pth_path*, build+load a ``V2VNet``, save an Orbax checkpoint.

    Args:
        pth_path: path to a ``HybridNet-*.pth`` state_dict.
        in_ch: V2VNet input channels (e.g. 50).
        out_ch: V2VNet output channels (e.g. 50).
        out_dir: Orbax checkpoint output directory.
        strip_prefix: key prefix selecting the v2vNet submodule (default
            ``"v2vNet."``, matching HybridNet's namespacing).

    Returns:
        The loaded ``V2VNet`` NNX module (also saved to ``out_dir``).
    """
    import torch  # local import: keep torch optional for pure-JAX call sites

    state_dict = torch.load(pth_path, map_location="cpu")
    model = V2VNet(in_ch, out_ch, rngs=nnx.Rngs(0))
    load_v2vnet_from_state_dict(model, state_dict, strip_prefix=strip_prefix)

    _, state = nnx.split(model)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(out_dir, state)
    ckptr.wait_until_finished()
    return model


def load_v2vnet_ckpt(out_dir: str, in_ch: int, out_ch: int) -> V2VNet:
    """Restore a ``V2VNet`` from an Orbax checkpoint directory.

    Mirrors ``build_checkpoint.load_vitpose``: build an abstract (shape-only)
    target via ``nnx.eval_shape``, restore into it, merge back with the
    GraphDef -- avoids needing PyTorch at load time.
    """
    m_abstract = nnx.eval_shape(lambda: V2VNet(in_ch, out_ch, rngs=nnx.Rngs(0)))
    gdef, abstract_state = nnx.split(m_abstract)

    ckptr = ocp.StandardCheckpointer()
    restored_state = ckptr.restore(out_dir, target=abstract_state)
    return nnx.merge(gdef, restored_state)
