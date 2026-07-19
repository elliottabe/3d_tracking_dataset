"""Standalone 2D keypoint-detector eval: ViTPose-vs-EfficientTrack MPJPE
head-to-head on the SAME val set.

Reuses (does not reinvent) the training pipeline's machinery:
  * ``jarvis_jax.train.train.eval_mpjpe`` (already jitted, see
    tests/test_eval_mpjpe_jit.py) for the metric itself.
  * ``jarvis_jax.data.v3.V3Dataset`` for the val split + per-recording subsets
    -- the exact same subsetting approach ``train_keypoints.run_training``
    uses for its full-val (``val_ds_all``) + female-courtship-recording
    (``val_ds``) reporting.
  * The Orbax ``eval_shape`` + ``nnx.merge`` restore pattern from
    ``jarvis_jax.convert.build_checkpoint.load_vitpose``, generalised here to
    also cover EfficientTrack (no MAE-pretrained checkpoint exists for it --
    see ``train_keypoints.py`` -- but the trained "final" Orbax dir has the
    same ``StandardCheckpointer.save(out_dir, nnx.split(model)[1])`` layout
    regardless of arch, so the same restore shape works).

CLI (Hydra; mirrors train_keypoints.py's config composition). The checkpoint
dir isn't a registered config group (same pattern as
``convert.out`` in build_checkpoint.py) -- add it ad hoc with ``+eval.*``::

    python -m jarvis_jax.scripts.eval_keypoints_2d \\
        paths=hyak model=vitpose \\
        +eval.ckpt_dir=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v4_kp_maskaware_fc/final

    python -m jarvis_jax.scripts.eval_keypoints_2d \\
        paths=hyak model=efficienttrack \\
        +eval.ckpt_dir=/gscratch/portia/eabe/data/Johnson_lab/jax_efficienttrack_runs/<run>/final

Optional overrides (each is a NEW key under the ad-hoc ``eval.*`` node, so it
also needs the ``+`` prefix, e.g. ``+eval.batch_size=8``):
``eval.batch_size`` (default 16), ``eval.data_root`` (default
``paths.data_root``), ``eval.val_recording`` (default
``2026_05_27_11_56_05``, the training loop's female-courtship recording),
``eval.per_recording`` (default true -- also report each val recording
separately). ``model.num_keypoints``/``model.in_ch`` etc. come from the
``model=vitpose``/``model=efficienttrack`` config groups (override with plain
``model.num_keypoints=50`` -- that key already exists in the composed
config, no ``+`` needed).
"""
import hydra
import jax
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, build_dataclass

register_resolvers()

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.models.efficienttrack import EfficientTrack
from jarvis_jax.data.v3 import V3Dataset
from jarvis_jax.train.train import eval_mpjpe

DEFAULT_VAL_RECORDING = "2026_05_27_11_56_05"   # matches train_keypoints.py's default


def _build_abstract(arch, cfg):
    """Construct the (arch-specific) model as a constructor closure, matching
    the metadata `cfg` (ViTPoseConfig doubles as the generic num_keypoints/
    in_ch/heatmap_size container for both arches -- see train_keypoints.py)."""
    if arch == "vitpose":
        return lambda: ViTPose(cfg, rngs=nnx.Rngs(0))
    if arch == "efficienttrack":
        return lambda: EfficientTrack(
            num_joints=cfg.num_keypoints, in_channels=cfg.in_ch, rngs=nnx.Rngs(0))
    raise ValueError(f"unknown arch {arch!r} (expected 'vitpose' or 'efficienttrack')")


def restore_model(ckpt_dir, arch, cfg):
    """Restore a trained model of the given `arch` from an Orbax "final"
    checkpoint dir (``StandardCheckpointer.save(out_dir, nnx.split(model)[1])``
    -- see ``train_keypoints.run_training``'s save at the end of training).

    Same eval_shape + replicated-sharding + StandardCheckpointer.restore +
    nnx.merge pattern as ``jarvis_jax.convert.build_checkpoint.load_vitpose``,
    generalised over `arch` so it also restores EfficientTrack. Replicated
    (not single-device) target sharding, so the checkpoint loads on any device
    topology and a sharded-batch eval also works unchanged.
    """
    ctor = _build_abstract(arch, cfg)
    m_abstract = nnx.eval_shape(ctor)
    gdef, abstract_state = nnx.split(m_abstract)

    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())
    target = jax.tree_util.tree_map(
        lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=repl),
        abstract_state)

    ckptr = ocp.StandardCheckpointer()
    restored_state = ckptr.restore(ckpt_dir, target=target)
    return nnx.merge(gdef, restored_state)


def evaluate(ckpt_dir, arch, root, *, vitpose_cfg=None, batch_size=16,
            val_recording=DEFAULT_VAL_RECORDING, per_recording=True,
            dataset_cls=V3Dataset, in_size=448, quiet=False):
    """Restore `arch`'s checkpoint at `ckpt_dir`, evaluate MPJPE on the `root`
    val split -- overall, the `val_recording` female-courtship subset (if
    given), and (if `per_recording`) every recording present in val -- and
    print a summary. Returns a dict with the same numbers.

    `dataset_cls` defaults to ``V3Dataset`` (``dataset_cls(root, split,
    recordings=...)``); tests inject a tiny stand-in so this whole path is
    unit-testable without a real 30k-step checkpoint or GPU.
    """
    cfg = vitpose_cfg if vitpose_cfg is not None else ViTPoseConfig()
    model = restore_model(ckpt_dir, arch, cfg)

    val_ds_all = dataset_cls(root, "val")
    overall = eval_mpjpe(model, val_ds_all, batch_size, in_size=in_size)

    fem = None
    if val_recording is not None:
        fem_ds = dataset_cls(root, "val", recordings=[val_recording])
        if len(fem_ds) > 0:
            fem = eval_mpjpe(model, fem_ds, batch_size, in_size=in_size)

    per_rec = {}
    if per_recording:
        recs = sorted({fn.split("/")[0] for fn in val_ds_all.file_names})
        for r in recs:
            ds_r = dataset_cls(root, "val", recordings=[r])
            if len(ds_r) == 0:
                continue
            per_rec[r] = eval_mpjpe(model, ds_r, batch_size, in_size=in_size)

    if not quiet:
        print(f"checkpoint: {ckpt_dir}")
        print(f"arch:       {arch}")
        print(f"overall val MPJPE: {overall:.3f}px  (n={len(val_ds_all)})")
        if fem is not None:
            print(f"  {val_recording} (female/courtship): {fem:.3f}px")
        if per_recording:
            print("per-recording val MPJPE:")
            for r, v in per_rec.items():
                print(f"  {r}: {v:.3f}px")

    return {
        "ckpt_dir": ckpt_dir, "arch": arch, "overall": overall,
        "val_recording": val_recording, "female": fem,
        "per_recording": per_rec,
    }


def main_from_cfg(cfg):
    model_node = cfg.model.get("vitpose", cfg.model)
    arch = model_node.get("arch", "vitpose")
    vitpose_cfg = build_dataclass(ViTPoseConfig, model_node)
    root = cfg.eval.get("data_root", cfg.paths.data_root)
    return evaluate(
        cfg.eval.ckpt_dir, arch, root,
        vitpose_cfg=vitpose_cfg,
        batch_size=cfg.eval.get("batch_size", 16),
        val_recording=cfg.eval.get("val_recording", DEFAULT_VAL_RECORDING),
        per_recording=bool(cfg.eval.get("per_recording", True)),
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
