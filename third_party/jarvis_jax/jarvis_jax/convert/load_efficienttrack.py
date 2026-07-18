"""Convert a PyTorch ``EfficientTrack-*.pth`` checkpoint into an Orbax-backed
NNX ``EfficientTrack`` module.

Reads the ``.pth`` state_dict with ``torch`` (CPU), builds a ``z``-like dict
keyed ``w::<torch_name>`` (exactly the naming convention produced by
``export_efficienttrack_fixture.py`` and consumed by
``jarvis_jax.models.efficienttrack.load_efficienttrack_from_npz``), assigns
the weights into a freshly-constructed NNX ``EfficientTrack``, and saves an
Orbax ``StandardCheckpointer`` checkpoint of ``nnx.state(model)``.

Two checkpoints share this converter:
  * ``EfficientTrack-large_final.pth`` (standalone KeypointDetect run) has
    **no** key prefix and 4 input channels (RGB + SAM3 mask).
  * ``HybridNet-large_final.pth`` namespaces the same submodule under
    ``effTrack.*`` (Task 5) and uses 3 input channels (RGB only) for that run.
    Pass ``strip_prefix="effTrack."`` to select just those keys.

``in_channels`` is inferred from the ``_conv_stem.weight`` tensor shape
(NCHW: (out, in, kh, kw)) rather than hardcoded, since it differs between the
two checkpoints above.
"""
from __future__ import annotations

import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.models.efficienttrack import EfficientTrack, load_efficienttrack_from_npz


def _state_dict_to_z(state_dict: dict, *, strip_prefix: str = "") -> dict:
    """Build a ``w::<name>``-keyed dict from a torch state_dict.

    If ``strip_prefix`` is given, only keys starting with it are kept, with
    the prefix stripped (e.g. ``effTrack.bifpn.0...`` -> ``bifpn.0...``),
    matching how ``load_efficienttrack_from_npz`` looks up keys.
    """
    z = {}
    for k, v in state_dict.items():
        if strip_prefix:
            if not k.startswith(strip_prefix):
                continue
            name = k[len(strip_prefix):]
        else:
            name = k
        z[f"w::{name}"] = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v)
    return z


def convert_efficienttrack_pth(pth_path: str, num_joints: int, out_dir: str, *,
                                strip_prefix: str = "") -> EfficientTrack:
    """Load *pth_path*, build+load an ``EfficientTrack``, save an Orbax ckpt.

    Args:
        pth_path: path to a ``.pth`` state_dict (``EfficientTrack-*.pth`` or
            ``HybridNet-*.pth`` with ``strip_prefix="effTrack."``).
        num_joints: number of output joints/keypoints.
        out_dir: Orbax checkpoint output directory.
        strip_prefix: optional key prefix to select+strip (submodule namespacing).

    Returns:
        The loaded ``EfficientTrack`` NNX module (also saved to ``out_dir``).
    """
    import torch  # local import: keep torch optional for pure-JAX call sites

    state_dict = torch.load(pth_path, map_location="cpu")
    z = _state_dict_to_z(state_dict, strip_prefix=strip_prefix)

    stem_key = "w::backbone_net.model._conv_stem.weight"
    in_channels = int(z[stem_key].shape[1])  # NCHW (out, in, kh, kw)
    z["num_joints"] = np.int64(num_joints)
    z["in_channels"] = np.int64(in_channels)

    model = EfficientTrack(num_joints=num_joints, in_channels=in_channels, rngs=nnx.Rngs(0))
    load_efficienttrack_from_npz(model, z)

    _, state = nnx.split(model)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(out_dir, state)
    ckptr.wait_until_finished()
    return model


def load_efficienttrack_ckpt(out_dir: str, num_joints: int, *,
                              in_channels: int = 4) -> EfficientTrack:
    """Restore an ``EfficientTrack`` from an Orbax checkpoint directory.

    Mirrors ``build_checkpoint.load_vitpose``: build an abstract (shape-only)
    target via ``nnx.eval_shape``, restore into it, merge back with the
    GraphDef -- avoids needing PyTorch at load time.
    """
    m_abstract = nnx.eval_shape(
        lambda: EfficientTrack(num_joints=num_joints, in_channels=in_channels,
                                rngs=nnx.Rngs(0)))
    gdef, abstract_state = nnx.split(m_abstract)

    ckptr = ocp.StandardCheckpointer()
    restored_state = ckptr.restore(out_dir, target=abstract_state)
    return nnx.merge(gdef, restored_state)
