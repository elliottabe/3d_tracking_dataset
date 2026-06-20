"""Build an Orbax checkpoint from a MAE-initialised ViTPose.

CLI:
    python -m jarvis_jax.convert.build_checkpoint \\
        --npz /tmp/mae_vitb.npz --out /path/to/ckpt_dir

The ViT backbone weights are loaded from the NPZ (Task 8 export); the
ClassicDecoder stays randomly initialised.  The Orbax checkpoint stores
``nnx.state(model)`` so it can be restored without PyTorch.

Load path (JAX-only)::

    from jarvis_jax.convert.build_checkpoint import load_vitpose
    model = load_vitpose(ckpt_dir, ViTPoseConfig())
"""

import argparse

import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.convert.load_weights import load_vit_from_npz


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
    """
    # Build an abstract (shape-only) model to use as the restore target.
    m_abstract = nnx.eval_shape(lambda: ViTPose(cfg, rngs=nnx.Rngs(0)))
    gdef, abstract_state = nnx.split(m_abstract)

    ckptr = ocp.StandardCheckpointer()
    restored_state = ckptr.restore(ckpt_dir, target=abstract_state)
    return nnx.merge(gdef, restored_state)


def main():
    ap = argparse.ArgumentParser(
        description="Build an Orbax checkpoint from a MAE-init ViTPose."
    )
    ap.add_argument("--npz", required=True, help="Path to mae_vitb.npz")
    ap.add_argument("--out", required=True, help="Output checkpoint directory")
    args = ap.parse_args()

    cfg = ViTPoseConfig()
    m = build(args.npz, cfg)

    _, state = nnx.split(m)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(args.out, state)
    ckptr.wait_until_finished()
    print("saved checkpoint to", args.out)


if __name__ == "__main__":
    main()
