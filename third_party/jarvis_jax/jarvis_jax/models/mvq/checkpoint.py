"""Shared mvq checkpoint loader.

`scripts/viz/mvq_overlay.py` and `scripts/benchmark/mvq_val_baselines.py` each
carried their own copy of this restore logic (device-count-independent
replicated sharding so a 4-GPU training run can be restored on 1 GPU or on
CPU without Orbax's "Topology mismatch detected"). This module is the single
place it lives now.
"""
from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jarvis_jax.models.mvq.model import MVQConfig, MVQModel


def _replicated_target(state):
    """Abstract restore target: same pytree as `state`, REPLICATED over every
    currently-visible device -- NOT the checkpoint's own stored sharding,
    which raises "Topology mismatch detected" the moment the current
    process's device count differs from the training run's (e.g. restoring
    on 1 GPU, or on CPU, after a 4-GPU training job)."""
    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())
    return jax.tree_util.tree_map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=repl), state)


def load_mvq_model(run_dir_or_final, *, step=None, attn_impl=None):
    """Load an mvq checkpoint into a fresh `MVQModel` -- returns `(model, meta)`.

    `step is None` (default): `run_dir_or_final` IS a `final/` directory --
    the EMA `train_mvq.run_training` already debiased before saving it there
    with `ocp.StandardCheckpointer`, so this restores it directly, no further
    debiasing needed.

    `step` given (an int step, or `"latest"`): `run_dir_or_final` is the RUN
    directory instead (the parent of `final/` and `ckpt/`, e.g. what
    `scripts/train_mvq.py`'s `run_dir_for` returns) -- restores the RAW EMA
    running sum plus the live model's non-Param state from
    `<run_dir>/ckpt/<step>` (the `orbax.CheckpointManager` with items
    `model`/`opt`/`ema`/`ema_meta`, see `train_mvq._make_manager`/
    `_save_step`), then debiases the EMA by `1 - decay**ema_updates` exactly
    as `train_mvq._with_ema` does during training. `decay` is read from
    the run's `mvq_run.json`'s `train.ema` when that file and key exist
    (an older run's json might predate this field); it defaults to 0.999
    (`MVQTrainConfig`'s own default) otherwise.

    `mvq_run.json` is REQUIRED either way for the model architecture
    (`MVQConfig`) -- a bare `ckpt/<step>` has no config of its own
    (`_save_step` never writes one). `train_mvq.run_training` writes it
    directly under `<run_dir>/mvq_run.json` BEFORE step 0 (so a mid-run
    checkpoint has one even with no `final/` yet) and again at the very
    end under `<run_dir>/final/mvq_run.json` (this time with the real
    `val` numbers) once the run completes -- when `step` is given, this
    loader looks in `<run_dir>/mvq_run.json` FIRST and falls back to
    `<run_dir>/final/mvq_run.json` for an older run directory that only
    ever wrote the `final/` copy.

    `attn_impl`: override the config's own `attn_impl` after loading (e.g. a
    "cudnn" GPU training run evaluated on a CPU host, which has no cuDNN
    flash-attention kernel -- pass `attn_impl="xla"`).
    """
    final_dir = run_dir_or_final if step is None else os.path.join(run_dir_or_final, "final")
    if step is None:
        meta_path = os.path.join(final_dir, "mvq_run.json")
    else:
        run_dir_meta = os.path.join(run_dir_or_final, "mvq_run.json")
        meta_path = run_dir_meta if os.path.exists(run_dir_meta) else os.path.join(final_dir, "mvq_run.json")
    meta = json.load(open(meta_path))
    cfg_kwargs = dict(meta["model"])
    if attn_impl is not None:
        cfg_kwargs["attn_impl"] = attn_impl
    cfg = MVQConfig(**cfg_kwargs)
    abstract_model = nnx.eval_shape(lambda: MVQModel(cfg, rngs=nnx.Rngs(0)))
    gdef, state = nnx.split(abstract_model)

    if step is None:
        target = _replicated_target(state)
        restored = ocp.StandardCheckpointer().restore(final_dir, target=target)
        model = nnx.merge(gdef, restored)
        model.eval()
        return model, meta

    ckpt_dir = os.path.join(run_dir_or_final, "ckpt")
    ema_abstract = jax.tree_util.tree_map(jnp.zeros_like, nnx.state(abstract_model, nnx.Param))
    model_target, ema_target = _replicated_target(state), _replicated_target(ema_abstract)
    mngr = ocp.CheckpointManager(os.path.abspath(ckpt_dir),
                                 options=ocp.CheckpointManagerOptions(read_only=True),
                                 item_names=("model", "opt", "ema", "ema_meta"))
    use_step = mngr.latest_step() if step == "latest" else step
    r = mngr.restore(use_step, args=ocp.args.Composite(
        model=ocp.args.StandardRestore(model_target),
        ema=ocp.args.StandardRestore(ema_target),
        ema_meta=ocp.args.JsonRestore()))
    model = nnx.merge(gdef, r["model"])
    decay = meta.get("train", {}).get("ema", 0.999)
    t = int(r["ema_meta"]["ema_updates"])
    correction = 1.0 - decay ** t if t > 0 else 1.0
    ema_params = jax.tree_util.tree_map(lambda e: e / correction, r["ema"])
    nnx.update(model, ema_params)
    model.eval()
    return model, meta
