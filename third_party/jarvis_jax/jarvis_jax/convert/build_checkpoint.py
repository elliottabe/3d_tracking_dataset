"""Build an Orbax checkpoint from a MAE-initialised ViTPose.

CLI (Hydra):
    python -m jarvis_jax.convert.build_checkpoint \\
        paths=hyak model=vitpose +convert.out=/path/to/ckpt_dir

The ViT backbone weights are loaded from the NPZ (Task 8 export); the
ClassicDecoder stays randomly initialised.  The Orbax checkpoint stores
``nnx.state(model)`` so it can be restored without PyTorch.

Load path (JAX-only)::

    from jarvis_jax.convert.build_checkpoint import load_vitpose
    model = load_vitpose(ckpt_dir, ViTPoseConfig())
"""

import hydra
import jax
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jarvis_jax import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.convert.load_weights import load_vit_from_npz
from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, build_dataclass

register_resolvers()


def build(npz: str, cfg: ViTPoseConfig) -> ViTPose:
    """Create a ViTPose, load backbone weights from *npz*, return the model."""
    m = ViTPose(cfg, rngs=nnx.Rngs(0))
    load_vit_from_npz(m.backbone, npz, cfg)
    return m


def load_vitpose(ckpt_dir: str, cfg: ViTPoseConfig) -> ViTPose:
    """Restore a ViTPose from an Orbax checkpoint directory.

    Uses ``nnx.eval_shape`` to build an abstract target so the restore does
    not require the original state object, then merges the restored state with
    the GraphDef to produce a live model.

    The restore target is pinned to the CURRENT devices (replicated), so a
    checkpoint saved on a different number of devices — e.g. trained on 8 GPUs —
    loads on any topology (1/2/4 GPUs) without an Orbax topology mismatch.
    Replicated rather than single-device, so sharded-batch inference also works.
    """
    # Build an abstract (shape-only) model to use as the restore target.
    m_abstract = nnx.eval_shape(lambda: ViTPose(cfg, rngs=nnx.Rngs(0)))
    gdef, abstract_state = nnx.split(m_abstract)

    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())
    target = jax.tree_util.tree_map(
        lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=repl),
        abstract_state)

    ckptr = ocp.StandardCheckpointer()
    restored_state = ckptr.restore(ckpt_dir, target=target)
    return nnx.merge(gdef, restored_state)


def run_build(*, npz, out, vitpose_cfg=None):
    """Build and save a ViTPose checkpoint from MAE-initialized weights.

    Args:
        npz: Path to mae_vitb.npz weight file.
        out: Output checkpoint directory.
        vitpose_cfg: Optional ViTPoseConfig; defaults to ViTPoseConfig().
    """
    cfg = vitpose_cfg or ViTPoseConfig()
    m = build(npz, cfg)

    _, state = nnx.split(m)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(out, state)
    ckptr.wait_until_finished()
    print("saved checkpoint to", out)


def main_from_cfg(cfg):
    """Map a composed Hydra config into run_build."""
    vitpose_cfg = build_dataclass(ViTPoseConfig, cfg.model)
    return run_build(npz=cfg.paths.mae_npz, out=cfg.convert.out, vitpose_cfg=vitpose_cfg)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
