"""Checkpoint save/auto-resume for preemptible training (Orbax CheckpointManager).
Saves the model state and the optimizer state (which carries the optax step
count, so the LR schedule resumes correctly). On restart, restore_latest picks
up the most recent step automatically — survives SLURM preempt+requeue."""
import os

import jax
import jax.numpy as jnp
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


def _keypath_to_name(path_or_key):
    """Normalise a `jax.tree_util.keystr(...)` path (or a raw path tuple) to
    plain `a/b/c` form, e.g. `\"['decoder']['e_inst']\"` -> `\"decoder/e_inst\"`.
    Handles BOTH dict keys (`['layer']`, quoted) and list/sequence indices
    (`[0]`, unquoted) -- `nnx.to_pure_dict`'s pure-dict tree renders a
    `nnx.List`-backed module list as plain `[0]` while the raw restored
    `nnx.split(...)` State (see `warm_start_partial`) renders the very same
    position as the quoted dict key `['0']`; stripping every `'` before
    splitting on brackets normalises both to the same `layer/0` segment."""
    key = path_or_key if isinstance(path_or_key, str) else jax.tree_util.keystr(path_or_key)
    key = key.replace("'", "").replace("][", "/").replace("[", "/").replace("]", "")
    return key.strip("/")


def warm_start_partial(model, src_dir):
    """Shape-tolerant warm start (P3a spec §8): restore every leaf of `src_dir`
    (a `final/` written by StandardCheckpointer) whose path AND shape match
    `model`'s state; for `decoder/e_inst` with fewer source rows copy the
    leading rows; leave everything else at its fresh init. Returns the new
    model and the list of leaves NOT fully restored (human-readable paths).

    The two trees being compared flatten to DIFFERENT key spellings: `pure`
    (below, from `nnx.to_pure_dict` on the live `model`'s split state) has
    paths like `['decoder']['e_inst']`, while `src` (the raw `nnx.split(...)`
    State restored from `src_dir`, i.e. what `warm_start_restore`/training
    actually saves -- each leaf is a `VariableState` with its own `.value`)
    flattens with a trailing `['value']`: `['decoder']['e_inst']['value']`.
    Both are normalised to plain `a/b/c` form here, and that trailing
    `/value` segment (present only on the `src` side) is stripped before the
    two are matched by name, or every lookup below would silently miss and
    every source leaf would report itself skipped.

    `decoder/heads/sex/{kernel,bias}` is ALWAYS left fresh, even when
    `src_dir` happens to carry a leaf at that same path and shape (as a
    same-arch smoke-test source does): this warm start exists specifically
    to seed a fine-tune from the P3a-era 30k-step run whose model predates
    the sex head (P3a spec §5/§8), so the sex head must always train from
    scratch, not inherit an unrelated run's coincidentally-shaped weights."""
    ALWAYS_FRESH = {"decoder/heads/sex/kernel", "decoder/heads/sex/bias"}
    gdef, state = nnx.split(model)
    pure = nnx.to_pure_dict(state)
    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())
    ck = ocp.StandardCheckpointer()
    meta = ck.metadata(os.path.abspath(src_dir)).item_metadata.tree
    is_arr = lambda x: hasattr(x, "shape") and hasattr(x, "dtype")
    target = jax.tree_util.tree_map(
        lambda m: jax.ShapeDtypeStruct(tuple(m.shape), m.dtype, sharding=repl) if is_arr(m) else m, meta, is_leaf=is_arr)
    src = ck.restore(os.path.abspath(src_dir), target=target)
    src_flat = {}
    for k, v in jax.tree_util.tree_flatten_with_path(src)[0]:
        name = _keypath_to_name(k)
        if name.endswith("/value"):
            name = name[: -len("/value")]
        src_flat[name] = v
    skipped = []

    def merge(path, v):
        name = _keypath_to_name(path)
        if name in ALWAYS_FRESH:
            skipped.append(name); return v
        s = src_flat.get(name)
        if s is None or not hasattr(v, "shape"):
            skipped.append(name); return v
        if tuple(s.shape) == tuple(v.shape):
            return jnp.asarray(s, v.dtype)
        if name.endswith("e_inst") and s.shape[1:] == v.shape[1:] and s.shape[0] < v.shape[0]:
            skipped.append(f"{name} (partial rows 0:{s.shape[0]})")
            return v.at[: s.shape[0]].set(jnp.asarray(s, v.dtype))
        skipped.append(name); return v

    new_pure = jax.tree_util.tree_map_with_path(merge, pure)
    nnx.replace_by_pure_dict(state, new_pure)
    return nnx.merge(gdef, state), skipped
