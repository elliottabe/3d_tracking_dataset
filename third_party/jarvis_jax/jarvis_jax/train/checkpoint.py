"""Checkpoint save/auto-resume for preemptible training (Orbax CheckpointManager).
Saves the model state and the optimizer state (which carries the optax step
count, so the LR schedule resumes correctly). On restart, restore_latest picks
up the most recent step automatically — survives SLURM preempt+requeue."""
import os

import orbax.checkpoint as ocp
from flax import nnx


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
