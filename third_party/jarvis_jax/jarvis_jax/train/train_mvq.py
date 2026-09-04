"""Training loop for the multi-view query lifter (mvq)."""
from __future__ import annotations

import dataclasses
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.data.augment import build_lr_swap
from jarvis_jax.data.distractor import build_part_index
from jarvis_jax.data.mv_augment import MVAugParams, augment_window
from jarvis_jax.data.prefetch import prefetch
from jarvis_jax.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from jarvis_jax.data.v12_windows import V12WindowDataset, window_batches, WINDOW_KEYS
from jarvis_jax.models.dinov3 import HF_REPOS, dinov3_snapshot, load_dinov3_safetensors
from jarvis_jax.models.mvq import MVQConfig, MVQModel
from jarvis_jax.sharding import data_parallel_mesh, replicate
from jarvis_jax.train.losses_mvq import LossWeights, mvq_loss

MM_PER_UNIT = 0.1


@dataclasses.dataclass
class MVQTrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    total_steps: int = 30000
    batch_size: int = 32
    backbone_lr_mult: float = 0.1
    grad_clip: float = 1.0
    ema: float = 0.999
    seed: int = 0
    window_lengths: tuple = (1,)
    prompt_p_start: float = 1.0
    prompt_p_end: float = 0.5
    prompt_anneal_steps: int = 2000
    female_weight: float = 1.0
    balance_alpha: float = 0.5
    log_every: int = 50
    eval_every: int = 2000
    save_every: int = 1000
    num_workers: int = 16
    pretrained: bool = True
    smoke: bool = False
    val_cohorts: tuple = ("female", "two_fly")


_MEAN = jnp.asarray(IMAGENET_MEAN); _STD = jnp.asarray(IMAGENET_STD)


def normalize_crops(u8):
    return (u8.astype(jnp.float32) / 255.0 - _MEAN) / _STD


def _labels(params):
    return jax.tree_util.tree_map_with_path(
        lambda path, _: "backbone" if "backbone" in jax.tree_util.keystr(path) else "head", params)


def make_optimizer(model, tcfg):
    """AdamW, two LR groups (backbone vs head), ONE global gradient clip applied
    BEFORE the group split -- clipping per-group (inside each multi_transform
    branch) would clip the backbone's and the head's grad norms independently,
    which is not what `grad_clip` is meant to bound."""
    decay = max(tcfg.total_steps, tcfg.warmup_steps + 1)
    sched = lambda peak: optax.warmup_cosine_decay_schedule(0.0, peak, tcfg.warmup_steps, decay, 0.0)
    def group(peak):
        return optax.adamw(sched(peak), weight_decay=tcfg.weight_decay)
    tx = optax.chain(
        optax.clip_by_global_norm(tcfg.grad_clip),
        optax.multi_transform({"backbone": group(tcfg.lr * tcfg.backbone_lr_mult),
                               "head": group(tcfg.lr)}, _labels))
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def _batch_to_model(batch):
    return dict(crops=normalize_crops(batch["crops"]), cam_valid=batch["cam_valid"], M=batch["M"],
                t_local=batch["t_local"], prompt_mask=batch["prompt_mask"])


def make_train_step(aug: MVAugParams, lr_swap, part_of_k, weights: LossWeights, ema_decay: float):
    swap = jnp.asarray(lr_swap); pok = np.asarray(part_of_k)

    def loss_fn(model, batch):
        out = model(**_batch_to_model(batch), prompt_on=batch["prompt_on"])
        return mvq_loss(out, batch, weights, pok)

    @nnx.jit
    def step(model, optimizer, ema, key, batch, prompt_p):
        k_aug, k_p = jax.random.split(key)
        batch = augment_window(k_aug, batch, aug, swap)
        has_mask = batch["prompt_mask"].reshape(batch["prompt_mask"].shape[0], -1).any(-1)
        batch["prompt_on"] = jax.random.bernoulli(k_p, prompt_p, has_mask.shape) & has_mask
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
        optimizer.update(model, grads)
        params = nnx.state(model, nnx.Param)
        ema = jax.tree_util.tree_map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema, params)
        return loss, metrics, ema
    return step


@nnx.jit
def _fwd(model, crops, cam_valid, M, t_local, prompt_mask, prompt_on):
    return model(crops, cam_valid, M, t_local, prompt_mask, prompt_on=prompt_on)


def evaluate(model, ds, batch_size, *, cohorts: dict, part_of_k, weights, num_workers=8):
    """One pass over `ds` (JPEGs decoded ONCE per window via `window_batches`,
    not once per prompted/unprompted mode) running BOTH the prompted and the
    unprompted forward on every decoded batch. `cohorts`: name -> bool array
    over ds indices (by dataset index, recovered from `window_batches`'
    shuffle=False, drop_last=False order: batches are yielded in dataset-index
    order, so the sample seen at running position `offset+bi` IS ds index
    `offset+bi`). Returns {"prompted": {...}, "unprompted": {...}}."""
    names = ds.keypoint_names
    ii = lambda n: names.index(n)
    seg_names = [(a, c) for a, c in
                 [("EyeL", "EyeR")] + [(f"T{i}{s}_Tro", f"T{i}{s}_FeTi") for i in (1, 2, 3) for s in "LR"]
                 if a in names and c in names]
    seg_idx = [(ii(a), ii(c)) for a, c in seg_names]           # kept in lock-step with seg_names -- never re-filter one alone
    # per_sample[mode] rows: (mpjpe_units, n_joints_with_gt, match_reproj_px,
    # uv2d_px, head_vs_reproj_px, n_exist_pred, n_flies_true, seg_lengths|None,
    # ds_index, is_two_fly_window)
    per_sample = {"prompted": [], "unprompted": []}
    offset = 0
    for b in window_batches(ds, batch_size, shuffle=False, drop_last=False, num_workers=num_workers):
        B = b["crops"].shape[0]
        jb = {k: jnp.asarray(v) for k, v in b.items()}
        has_mask = jb["prompt_mask"].reshape(B, -1).any(-1)
        for mode, prompted in (("prompted", True), ("unprompted", False)):
            on = jnp.full((B,), prompted) & has_mask
            out = _fwd(model, normalize_crops(jb["crops"]), jb["cam_valid"], jb["M"], jb["t_local"],
                       jb["prompt_mask"], on)
            jb_m = dict(jb); jb_m["prompt_on"] = on
            _, m = mvq_loss(out, jb_m, weights, part_of_k)      # reuses the matching
            xyz = np.asarray(out["xyz"])                          # (B,I,T,K,3)
            uv2d_px, head_vs_reproj_px = float(m["uv2d_px"]), float(m["head_vs_reproj_px"])
            # per-sample numbers from the matched instance (fly 0 = host): redo the cheap host match
            for bi in range(B):
                i_ds = offset + bi
                gt = b["kp3d_local"][bi, 0]; has = b["has3d"][bi, 0]
                d = np.linalg.norm(xyz[bi] - gt[None], axis=-1)      # (I,T,K)
                inst = int(np.argmin(np.where(has[None], d, 0).sum((1, 2)) / max(has.sum(), 1)))
                e = d[inst][has]
                L = ([np.linalg.norm(xyz[bi, inst, 0, a] - xyz[bi, inst, 0, c]) for a, c in seg_idx]
                     if has.sum() > 0 else None)
                exist = 1 / (1 + np.exp(-np.asarray(out["exist_logit"][bi]))) > 0.5
                two_fly = ds.n_flies(i_ds) > 1
                per_sample[mode].append((float(e.mean()) if e.size else np.nan, int(e.size),
                                         float(m["match_reproj_px"]), uv2d_px, head_vs_reproj_px,
                                         int(exist.sum()), int(b["fly_valid"][bi].sum()), L, i_ds, two_fly))
        offset += B

    def _finish(mode):
        samples = per_sample[mode]
        mp = np.array([p[0] for p in samples]); n = np.array([p[1] for p in samples])
        ok = np.isfinite(mp)
        res = {"mpjpe3d_units": float(np.average(mp[ok], weights=n[ok])) if ok.any() else float("nan"),
               "reproj_px": float(np.mean([p[2] for p in samples])),
               "uv2d_px": float(np.mean([p[3] for p in samples])),
               "head_vs_reproj_px": float(np.mean([p[4] for p in samples]))}
        res["mpjpe3d_mm"] = res["mpjpe3d_units"] * MM_PER_UNIT
        # exist precision/recall (spec §7): two-fly windows only -- a single-fly
        # crop has nothing for a 2nd/3rd instance to correctly NOT exist against.
        two_fly = np.array([p[9] for p in samples])
        if two_fly.any():
            pred = np.array([p[5] for p in samples])[two_fly]
            true = np.array([p[6] for p in samples])[two_fly]
            res["exist_prec"] = float(np.sum(np.minimum(pred, true)) / max(np.sum(pred), 1))
            res["exist_rec"] = float(np.sum(np.minimum(pred, true)) / max(np.sum(true), 1))
        else:
            res["exist_prec"] = float("nan"); res["exist_rec"] = float("nan")
        Ls = [p[7] for p in samples if p[7] is not None]        # skip samples with no 3D-labelled joint
        if Ls:
            Ls = np.array(Ls)
            for (a, c), col in zip(seg_names, Ls.T):
                res[f"rigid_spread_mm/{a}-{c}"] = float(np.std(col) * MM_PER_UNIT)
        for name, mask in cohorts.items():
            sel = np.array([mask[p[8]] for p in samples]) & ok
            res[f"cohort_{name}"] = float(np.average(mp[sel], weights=n[sel])) if sel.any() else float("nan")
        return res

    return {"prompted": _finish("prompted"), "unprompted": _finish("unprompted")}


def _cohorts(ds):
    n = len(ds)
    c = {"female": np.array([ds.is_female(i) for i in range(n)]),
         "two_fly": np.array([ds.n_flies(i) > 1 for i in range(n)])}
    for g in sorted({ds.calib_group(i) for i in range(n)}):
        c[f"group_{g}"] = np.array([ds.calib_group(i) == g for i in range(n)])
    return c


def _balanced_weights(ds, alpha, female_weight):
    cats = [f"{ds.manifest[ds.windows[i][0]].get('behavior', 'unknown')}_{'female' if ds.is_female(i) else 'other'}"
            for i in range(len(ds))]
    counts = {c: cats.count(c) for c in set(cats)}
    w = np.array([(1.0 / counts[c]) ** alpha for c in cats])
    w *= np.where([ds.is_female(i) for i in range(len(ds))], female_weight, 1.0)
    return w / w.sum()


def _make_manager(ckpt_dir, *, max_to_keep=3):
    """mvq-local checkpoint manager: model + optimizer + EMA (a 3rd Orbax item
    the shared `train/checkpoint.py` doesn't know about -- kept local here
    rather than widening that shared module for one caller)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    opts = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, save_interval_steps=1)
    return ocp.CheckpointManager(os.path.abspath(ckpt_dir), options=opts,
                                 item_names=("model", "opt", "ema"))


def _save_step(mngr, steps_done, model, optimizer, ema):
    mngr.save(steps_done, args=ocp.args.Composite(
        model=ocp.args.StandardSave(nnx.split(model)[1]),
        opt=ocp.args.StandardSave(nnx.split(optimizer)[1]),
        ema=ocp.args.StandardSave(ema)))


def _restore_latest(mngr, model, optimizer, ema):
    """(model, optimizer, ema, start_step); ema restored using the freshly-built
    `ema` pytree as the abstract target, same pattern train/checkpoint.py's
    restore_latest uses for model/optimizer. (model, optimizer, ema, 0)
    unchanged if no checkpoint exists yet."""
    latest = mngr.latest_step()
    if latest is None:
        return model, optimizer, ema, 0
    gm, am = nnx.split(model)
    go, ao = nnx.split(optimizer)
    r = mngr.restore(latest, args=ocp.args.Composite(
        model=ocp.args.StandardRestore(am),
        opt=ocp.args.StandardRestore(ao),
        ema=ocp.args.StandardRestore(ema)))
    return nnx.merge(gm, r["model"]), nnx.merge(go, r["opt"]), r["ema"], latest


def run_training(root, *, out_dir, ckpt_dir, mcfg: MVQConfig, tcfg: MVQTrainConfig,
                 aug: MVAugParams, weights: LossWeights):
    n_dev = len(jax.devices())
    if tcfg.batch_size % n_dev:
        raise ValueError(f"batch_size {tcfg.batch_size} not divisible by {n_dev} devices")

    # Dataset load + cohort validation BEFORE any GPU-heavy work (model build,
    # pretrained-weight download, device replication): a misconfigured/empty
    # val cohort should fail fast, not after minutes of backbone init.
    train_sets = {T: V12WindowDataset(root, "train", T=T, train=True, seed=tcfg.seed) for T in tcfg.window_lengths}
    val_ds = V12WindowDataset(root, "val", T=1, train=False)
    names = train_sets[tcfg.window_lengths[0]].keypoint_names
    lr_swap = build_lr_swap(names); part_of_k, _ = build_part_index(names)
    cohorts = _cohorts(val_ds)
    for c in tcfg.val_cohorts:
        if not cohorts.get(c, np.zeros(1, bool)).any():
            raise ValueError(f"val cohort '{c}' is empty on {root}")

    model = MVQModel(mcfg, rngs=nnx.Rngs(tcfg.seed))
    if tcfg.pretrained:
        model.backbone = load_dinov3_safetensors(model.backbone, dinov3_snapshot(HF_REPOS[mcfg.backbone]))
        print(f"[mvq] loaded {HF_REPOS[mcfg.backbone]}")
    opt = make_optimizer(model, tcfg)
    ema = jax.tree_util.tree_map(lambda p: p, nnx.state(model, nnx.Param))
    mngr = _make_manager(ckpt_dir) if ckpt_dir else None
    start = 0
    if mngr is not None:
        model, opt, ema, start = _restore_latest(mngr, model, opt, ema)
        if start:
            print(f"resume @ {start}", flush=True)
    mesh = data_parallel_mesh()
    gm, sm = nnx.split(model); model = nnx.merge(gm, replicate(sm, mesh))
    go, so = nnx.split(opt); opt = nnx.merge(go, replicate(so, mesh))
    ema = replicate(ema, mesh)

    step_fns = {T: make_train_step(aug, lr_swap, part_of_k, weights, tcfg.ema) for T in tcfg.window_lengths}
    streams = {}
    for T, ds in train_sets.items():
        w = _balanced_weights(ds, tcfg.balance_alpha, tcfg.female_weight)
        def epochs(ds=ds, w=w, T=T):
            e = 0
            while True:
                yield from window_batches(ds, tcfg.batch_size, shuffle=True, seed=tcfg.seed + 1000 * e + T,
                                          weights=w, num_workers=tcfg.num_workers)
                e += 1
        streams[T] = prefetch((tuple(bt[k] for k in WINDOW_KEYS) for bt in epochs()), mesh, depth=2)
    key = jax.random.PRNGKey(tcfg.seed)
    Ts = list(tcfg.window_lengths)
    loss = float("nan"); t0 = time.time()
    for i in range(start, tcfg.total_steps):
        T = Ts[i % len(Ts)]
        batch = dict(zip(WINDOW_KEYS, next(streams[T])))
        pp = tcfg.prompt_p_start + (tcfg.prompt_p_end - tcfg.prompt_p_start) * min(i / max(tcfg.prompt_anneal_steps, 1), 1.0)
        loss, metrics, ema = step_fns[T](model, opt, ema, jax.random.fold_in(key, i), batch, jnp.float32(pp))
        loss = float(loss)
        if (i + 1) % tcfg.log_every == 0:
            ms = " ".join(f"{k}={float(v):.4f}" for k, v in metrics.items() if k != "total")
            print(f"step {i+1}/{tcfg.total_steps} T={T} loss {loss:.4f} {ms} ({time.time()-t0:.0f}s)", flush=True)
        if (i + 1) % tcfg.eval_every == 0 or i + 1 == tcfg.total_steps:
            em = _with_ema(model, ema)
            val = evaluate(em, val_ds, min(tcfg.batch_size, 8), cohorts=cohorts, part_of_k=part_of_k,
                           weights=weights, num_workers=tcfg.num_workers)
            for mode, r in val.items():
                print(f"  val[{mode}] " + " ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)
        if mngr is not None and (i + 1) % tcfg.save_every == 0:
            _save_step(mngr, i + 1, model, opt, ema)
    if mngr is not None:
        _save_step(mngr, tcfg.total_steps, model, opt, ema); mngr.wait_until_finished()
    em = _with_ema(model, ema)
    val = evaluate(em, val_ds, min(tcfg.batch_size, 8), cohorts=cohorts, part_of_k=part_of_k,
                   weights=weights, num_workers=tcfg.num_workers)
    os.makedirs(out_dir, exist_ok=True)
    ckptr = ocp.StandardCheckpointer(); ckptr.save(out_dir, nnx.split(em)[1], force=True); ckptr.wait_until_finished()
    with open(os.path.join(out_dir, "mvq_run.json"), "w") as f:
        json.dump({"model": dataclasses.asdict(mcfg), "train": dataclasses.asdict(tcfg), "val": val,
                  "keypoint_names": names}, f, indent=1)
    return {"final_loss": loss, "val": val, "steps": tcfg.total_steps, "resumed_from": start}


def _with_ema(model, ema):
    """A NEW module holding the EMA weights, independent of `model`'s own
    Variable objects. `nnx.split`/`nnx.merge` on `model` itself would return a
    module aliasing the SAME Variables as `model` (flax 0.12.8), so
    `nnx.update(em, ema)` would silently overwrite the training model's live
    params too -- `nnx.clone` makes a real copy first."""
    em = nnx.clone(model)
    nnx.update(em, ema)
    em.eval()
    return em
