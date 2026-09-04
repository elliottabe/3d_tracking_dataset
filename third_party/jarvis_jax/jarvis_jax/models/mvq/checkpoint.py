"""Shared mvq checkpoint loader.

`scripts/viz/mvq_overlay.py` and `scripts/benchmark/mvq_val_baselines.py` each
carried their own copy of this restore logic (device-count-independent
replicated sharding so a 4-GPU training run can be restored on 1 GPU or on
CPU without Orbax's "Topology mismatch detected"). This module is the single
place it lives now; the path/shape-tolerant merge itself is shared with
`train/checkpoint.py::warm_start_partial` (see `merge_state_by_path`).
"""
from __future__ import annotations

import json
import os

import jax
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.models.mvq.model import MVQConfig, MVQModel
from jarvis_jax.train.checkpoint import (merge_state_by_path, replicated_abstract_tree,
                                         restore_own_tree)

# Orbax needs an explicit handler per item to answer `item_metadata(step)`
# (without one it warns "could not be restored" and returns None for every
# item) -- see `load_mvq_model`'s `ckpt/` branch, which reads the
# checkpoint's OWN tree shapes from that metadata.
_CKPT_ITEM_HANDLERS = {"model": ocp.StandardCheckpointHandler(),
                       "ema": ocp.StandardCheckpointHandler(),
                       "ema_meta": ocp.JsonCheckpointHandler()}


def _report_unrestored(what, skipped):
    if skipped:
        print(f"[mvq] {what}: {len(skipped)} leaf/leaves NOT restored (kept at fresh init): "
              f"{sorted(skipped)}", flush=True)
    return skipped


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

    TOLERANT restore (both branches, 2026-09-04): the restore target is built
    from the CHECKPOINT's own stored tree and merged onto a freshly built
    model by path and shape (`train/checkpoint.py::merge_state_by_path`, the
    same helper `warm_start_partial` uses, including its `e_inst`
    leading-rows rule). Building the target from the fresh model instead --
    the previous behaviour -- raised on every pre-P3a checkpoint: the real
    30k-step run `mvq_t1_b16_local8_20260904` has 606 leaves and 3 instance
    slots, today's model has 608 (the sex head) and 4, so a strict restore
    could not even open it and no figure/benchmark script could measure the
    baseline it is the baseline FOR. Leaves that could not be restored keep
    their fresh init, are printed, and are returned in
    `meta["_unrestored_leaves"]` so a caller can refuse to report a metric
    that depends on an unrestored head (e.g. `mvq_overlay.py`'s `sex_prob`).
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
    # A REAL (not eval_shape) init: `merge_state_by_path` keeps this model's own
    # values for any leaf the checkpoint does not carry, so they have to exist.
    model = MVQModel(cfg, rngs=nnx.Rngs(0))

    if step is None:
        model, skipped = merge_state_by_path(model, restore_own_tree(final_dir))
        meta["_unrestored_leaves"] = _report_unrestored(f"load {final_dir}", skipped)
        model.eval()
        return model, meta

    ckpt_dir = os.path.join(run_dir_or_final, "ckpt")
    mngr = ocp.CheckpointManager(os.path.abspath(ckpt_dir),
                                 options=ocp.CheckpointManagerOptions(read_only=True),
                                 item_names=("model", "opt", "ema", "ema_meta"),
                                 item_handlers=_CKPT_ITEM_HANDLERS)
    use_step = mngr.latest_step() if step == "latest" else step
    im = mngr.item_metadata(use_step)
    tree_of = lambda x: getattr(x, "tree", x)
    r = mngr.restore(use_step, args=ocp.args.Composite(
        model=ocp.args.StandardRestore(replicated_abstract_tree(tree_of(im["model"]))),
        ema=ocp.args.StandardRestore(replicated_abstract_tree(tree_of(im["ema"]))),
        ema_meta=ocp.args.JsonRestore()))
    model, skipped = merge_state_by_path(model, r["model"])
    decay = meta.get("train", {}).get("ema", 0.999)
    t = int(r["ema_meta"]["ema_updates"])
    correction = 1.0 - decay ** t if t > 0 else 1.0
    ema_params = jax.tree_util.tree_map(lambda e: e / correction, r["ema"])
    # The debiased EMA overwrites the params it carries; every other leaf
    # (non-Param state, and anything the EMA predates) keeps what the `model`
    # item just restored -- the tolerant generalisation of `nnx.update`.
    model, skipped_ema = merge_state_by_path(model, ema_params)
    meta["_unrestored_leaves"] = _report_unrestored(f"load {ckpt_dir}/{use_step} (model)", skipped)
    _report_unrestored(f"load {ckpt_dir}/{use_step} (ema, non-Param leaves expected)",
                       [s for s in skipped_ema if s not in skipped])
    model.eval()
    return model, meta
