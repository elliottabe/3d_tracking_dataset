"""Finetune the ViTPose detector from a v3 checkpoint on a real+pseudo mix.

Continue-trains from the existing v3 checkpoint (READ-ONLY -- never
overwritten), mixes real (curated) V3-format labels with silhouette
pseudo-labels (Task 3's ``build_pseudolabel_dataset`` output) via
``ConcatV3``, early-stops on held-out courtship MPJPE (the female-fly val
recordings), and saves the BEST model state as a brand-new (v4) checkpoint in
``out_dir``.

Mirrors ``jarvis_jax.scripts.train_keypoints.run_training``'s loop (same
``make_optimizer``/``make_train_step``/``eval_mpjpe`` wiring and the same
``ocp.StandardCheckpointer().save(..., force=True)`` save pattern), but adds
the real+pseudo mix, best-checkpoint tracking, and early stopping.
"""
import os

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.convert.build_checkpoint import load_vitpose
from jarvis_jax.data.concat import ConcatV3
from jarvis_jax.data.v3 import V3Dataset, batches
from jarvis_jax.train.train import (
    TrainConfig, make_optimizer, make_train_step, eval_mpjpe,
)


def _epochs(ds, batch_size, base_seed):
    """Infinite stream of batches, reshuffled each epoch (mirrors
    ``scripts/train_keypoints.py``'s ``_epochs``)."""
    epoch = 0
    while True:
        yield from batches(ds, batch_size, shuffle=True, seed=base_seed + epoch)
        epoch += 1


def _copy_state(state):
    """Snapshot a live ``nnx`` state so later training steps can't mutate the
    saved-best checkpoint. ``nnx.split(model)[1]`` is cheap but its leaves are
    the model's live arrays; a plain ``jnp.array(x, copy=True)`` over every
    leaf forces a fresh buffer per array, matching flax/orbax's own state
    pytree structure (``nnx.merge(gdef, copied_state)`` reconstructs a valid
    live model, verified against this checkpoint's actual state tree)."""
    return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), state)


def finetune(*, v3_ckpt, real_root, pseudo_root, out_dir, val_recordings,
             tcfg=None, pseudo_weight=1, total_steps=4000, eval_every=250,
             patience=6, batch_size=None, lr=2e-5, seed=0):
    """Continue-train the v3 ViTPose detector on a real+pseudo mix.

    Args:
        v3_ckpt: Orbax checkpoint dir for the existing v3 model. READ-ONLY;
            this function never writes to it.
        real_root: V3-format dataset root of curated real labels; needs
            "train" and "val" splits.
        pseudo_root: V3-format dataset root of silhouette pseudo-labels
            (``build_pseudolabel_dataset.write_pseudolabel_coco`` output);
            needs a "train" split.
        out_dir: NEW checkpoint directory for the finetuned (v4) model. Must
            differ from ``v3_ckpt``.
        val_recordings: recording name(s) held out as the early-stopping
            signal (e.g. courtship recordings with female-fly annotations) --
            passed to ``V3Dataset(real_root, "val", recordings=...)``.
        tcfg: optional ``TrainConfig``; if omitted, one is built from `lr`/
            `total_steps`/`batch_size`/`seed`.
        pseudo_weight: integer oversampling weight for the pseudo dataset in
            ``ConcatV3`` (the real dataset's weight is fixed at 1).
        total_steps, eval_every, patience: training-loop control. Evaluation
            (and early-stop bookkeeping) happens every `eval_every` steps;
            training stops early after `patience` non-improving evals.
        batch_size, lr, seed: shorthand for building `tcfg` when one isn't
            passed explicitly.

    Returns:
        dict with "best_step", "best_female_mpjpe", "full_val_mpjpe" -- all
        computed against the checkpoint actually written to `out_dir`.
    """
    if os.path.abspath(out_dir) == os.path.abspath(v3_ckpt):
        raise ValueError(
            "out_dir must not be v3_ckpt -- finetune() must never overwrite "
            f"the source v3 checkpoint (got out_dir == v3_ckpt == {v3_ckpt!r})")

    cfg = ViTPoseConfig()
    model = load_vitpose(v3_ckpt, cfg)

    tcfg = tcfg or TrainConfig(lr=lr, total_steps=total_steps,
                               batch_size=batch_size or 8, seed=seed)
    opt = make_optimizer(model, tcfg)

    train_ds = ConcatV3(
        [V3Dataset(real_root, "train"), V3Dataset(pseudo_root, "train")],
        [1, pseudo_weight])
    fem_ds = V3Dataset(real_root, "val", recordings=list(val_recordings))
    full_ds = V3Dataset(real_root, "val")

    step = make_train_step(tcfg.mask_weight, None, None,
                           heatmap_size=cfg.heatmap_size,
                           mask_dilate=tcfg.mask_dilate)

    key = jax.random.PRNGKey(tcfg.seed)
    train_stream = _epochs(train_ds, tcfg.batch_size, tcfg.seed)

    best = {"best_step": -1, "best_female_mpjpe": float("inf"),
            "full_val_mpjpe": float("nan")}
    best_state = None
    bad_evals = 0
    for i in range(total_steps):
        img4_u8, kp_xy, vis = next(train_stream)
        step(model, opt, jax.random.fold_in(key, i), img4_u8, kp_xy, vis)

        if (i + 1) % eval_every == 0:
            fem_mpjpe = float(eval_mpjpe(model, fem_ds, tcfg.batch_size))
            if fem_mpjpe < best["best_female_mpjpe"]:
                best["best_step"] = i + 1
                best["best_female_mpjpe"] = fem_mpjpe
                best_state = _copy_state(nnx.split(model)[1])
                bad_evals = 0
            else:
                bad_evals += 1
                if bad_evals >= patience:
                    break

    gdef, _ = nnx.split(model)
    if best_state is None:
        # Never improved (or total_steps < eval_every): fall back to the
        # final trained state so a usable checkpoint still gets written.
        best_state = nnx.split(model)[1]
    else:
        # Restore the BEST state before the final full-val eval, so the
        # returned metrics describe the exact checkpoint written to out_dir
        # (not whatever the training loop happened to end on).
        model = nnx.merge(gdef, best_state)

    best["full_val_mpjpe"] = float(eval_mpjpe(model, full_ds, tcfg.batch_size))

    ckptr = ocp.StandardCheckpointer()
    ckptr.save(out_dir, best_state, force=True)
    ckptr.wait_until_finished()
    return best
