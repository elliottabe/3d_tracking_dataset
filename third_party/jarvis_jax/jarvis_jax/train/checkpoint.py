"""Checkpoint save/auto-resume for preemptible training (Orbax CheckpointManager).
Saves the model state and the optimizer state (which carries the optax step
count, so the LR schedule resumes correctly). On restart, restore_latest picks
up the most recent step automatically — survives SLURM preempt+requeue."""
import os

import jax
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def make_manager(ckpt_dir, *, max_to_keep=3):
    os.makedirs(ckpt_dir, exist_ok=True)
    opts = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep,
                                        save_interval_steps=1)
    return ocp.CheckpointManager(
        os.path.abspath(ckpt_dir), options=opts, item_names=("model", "opt"))


def save_step(mngr, steps_done, model, optimizer):
    """Checkpoint at key=steps_done (number of completed optimizer steps)."""
    mngr.save(steps_done, args=ocp.args.Composite(
        model=ocp.args.StandardSave(nnx.split(model)[1]),
        opt=ocp.args.StandardSave(nnx.split(optimizer)[1])))


def restore_latest(mngr, model, optimizer):
    """If a checkpoint exists, return (model, optimizer, start_step) restored
    from the latest step (start_step = completed steps = next loop index).
    Otherwise return (model, optimizer, 0) unchanged."""
    latest = mngr.latest_step()
    if latest is None:
        return model, optimizer, 0
    gm, am = nnx.split(model)
    go, ao = nnx.split(optimizer)
    r = mngr.restore(latest, args=ocp.args.Composite(
        model=ocp.args.StandardRestore(am),
        opt=ocp.args.StandardRestore(ao)))
    return nnx.merge(gm, r["model"]), nnx.merge(go, r["opt"]), latest


def warm_start_restore(model, final_dir):
    """WARM-START (not resume): overwrite `model`'s weights in place from a
    finished run's ``final/`` Orbax dir -- i.e. the
    ``StandardCheckpointer.save(out_dir, nnx.split(model)[1])`` a training run
    writes at the end (see ``train_keypoints.run_training``'s final save; same
    layout ``eval_keypoints_2d.restore_model`` reads for eval-only restores).

    `model` must already be built for the SAME arch/shape (e.g. the normal
    per-arch construction in ``run_training`` -- for ``efficienttrack_bn``
    that includes its ImageNet-warm-started init, which this fully overwrites:
    backbone conv/BN running-stats, BiFPN, and head, since `nnx.split` captures
    every Param/BatchStat/etc. leaf). Returns a NEW model object merging the
    restored state back onto `model`'s GraphDef.

    Deliberately does NOT touch any optimizer or step counter -- unlike
    ``restore_latest`` (SAME-run resume: model + optax state + step, so the LR
    schedule continues), this restores ONLY the weights so the caller can
    build a FRESH optimizer and start at step 0 for a NEW fine-tune run. If
    that new run's OWN `ckpt_dir` already has snapshots (e.g. a SLURM
    preempt+requeue of this fine-tune), call `restore_latest` on the result
    afterwards as usual -- it will override these warm-started weights (and
    the optimizer/step) with the fine-tune's own progress, which is the
    correct precedence: resume beats warm_start on requeue; warm_start only
    ever seeds the very first launch.
    """
    gdef, state = nnx.split(model)
    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())
    target = jax.tree_util.tree_map(
        lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=repl), state)
    ckptr = ocp.StandardCheckpointer()
    restored_state = ckptr.restore(final_dir, target=target)
    return nnx.merge(gdef, restored_state)
