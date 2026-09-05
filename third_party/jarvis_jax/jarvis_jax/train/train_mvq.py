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
from jarvis_jax.data.mv_copy_paste import CopyPasteParams
from jarvis_jax.data.prefetch import prefetch
from jarvis_jax.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from jarvis_jax.data.v12_windows import V12WindowDataset, window_batches, WINDOW_KEYS
from jarvis_jax.models.dinov3 import HF_REPOS, dinov3_snapshot, load_dinov3_safetensors
from jarvis_jax.models.mvq import MVQConfig, MVQModel
from jarvis_jax.models.mvq.policy import EXIST_THRESH, mask_containment, policy_instance
from jarvis_jax.sharding import data_parallel_mesh, replicate, shard_batch
from jarvis_jax.train.checkpoint import warm_start_partial
from jarvis_jax.train.losses_mvq import LossWeights, mvq_loss

MM_PER_UNIT = 0.1
CONTACT_UNITS = 30.0  # 3 mm; real mounting pairs have centroid gaps of ~24-30 units, see p3a-notes.md


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
    copy_paste_p: float = 0.0
    copy_paste_opposite_sex_p: float = 0.7
    copy_paste_contact_p: float = 0.3
    warm_start: str | None = None
    jitter_units: float = 3.0  # train-time window-centre jitter, world units (0.1 mm); P4 §6 raises it to 10 for the mask-free route


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
        # A window only "has" a usable prompt if the (post-augmentation) mask
        # has a labelled pixel in a view that is ALSO a valid camera -- camera
        # dropout (augment_window, above) can invalidate the only camera the
        # mask lived in, and _prompt (model.py) already gates its own mean by
        # cam_valid, so scoring has_mask on the mask alone would turn prompt_on
        # on for a window whose prompt token is actually an all-invalid,
        # zeroed-out mean.
        has_mask = (batch["prompt_mask"].any((3, 4)) & batch["cam_valid"]).any((1, 2))
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


def evaluate(model, ds, batch_size, *, cohorts: dict, part_of_k, weights, mesh, num_workers=8):
    """One pass over `ds` (JPEGs decoded ONCE per window via `window_batches`,
    not once per prompted/unprompted mode) running BOTH the prompted and the
    unprompted forward on every decoded batch. `batch_size` should be
    `tcfg.batch_size` (the SAME as training), so eval's per-device batch
    matches training's -- a smaller eval batch changes both the sharding
    layout and the batch-composition of the reproj/uv2d/head_vs_reproj
    aggregates below. `cohorts`: name -> bool array over ds indices (by
    dataset index, recovered from `window_batches`' shuffle=False,
    drop_last=False order: batches are yielded in dataset-index order, so the
    sample seen at running position `offset+bi` IS ds index `offset+bi`).
    Returns {"prompted": {...}, "unprompted": {...}}.

    Batches are sharded across `mesh` (`shard_batch`, same as the training
    step) rather than left as plain `jnp.asarray` -- unsharded eval arrays
    place the whole batch on device 0 only, which fragments its BFC pool and
    OOMs the *next* training step's ~19GB arena (measured 2026-09-03: eval
    succeeds, the very next step's allocation then fails and the other ranks
    hang on the NCCL clique rendezvous). The ragged last batch is padded (by
    repeating its final row) up to `batch_size` so every batch shards evenly
    across the mesh; the per-sample loop below still only reads the first
    `B0` (real) rows, so padding never contaminates a per-sample statistic.

    `reproj_px`/`uv2d_px`/`head_vs_reproj_px` are themselves BATCH-level means
    (from `mvq_loss`, over that batch's valid (vis2d & fly_valid) entries) --
    a plain mean-of-batches would let how samples happen to fall into batches
    change the reported number (a batch with more occluded/invisible entries
    contributes the SAME weight as a fully-visible one). Each batch's triple
    is instead weighted by its OWN valid-entry count when combining across
    batches, so the result is batch-grouping-independent. That count is
    computed from only the `B0` REAL rows (`b["vis2d"][:B0]`), not the padded
    `batch_size`-shaped array `mvq_loss` itself saw -- a ragged last batch's
    OWN reported value can still carry a small bias from its duplicated pad
    row(s) (mvq_loss averaged over them too), but weighting the CROSS-BATCH
    combination by real-only counts stops that one batch's padding from also
    distorting every OTHER batch's contribution."""
    names = ds.keypoint_names
    ii = lambda n: names.index(n)
    seg_names = [(a, c) for a, c in
                 [("EyeL", "EyeR")] + [(f"T{i}{s}_Tro", f"T{i}{s}_FeTi") for i in (1, 2, 3) for s in "LR"]
                 if a in names and c in names]
    seg_idx = [(ii(a), ii(c)) for a, c in seg_names]           # kept in lock-step with seg_names -- never re-filter one alone
    # per_sample[mode] rows: (mpjpe_units, n_joints_with_gt, n_exist_pred,
    # n_flies_true, seg_lengths_pred|None, seg_lengths_gt|None, ds_index,
    # is_two_fly_window, mpjpe_policy_units|nan, is_policy_miss,
    # mask_containment_frac|nan -- the policy instance's reprojection inside the host mask)
    per_sample = {"prompted": [], "unprompted": []}
    # batch_stats[mode] rows: (reproj_px, uv2d_px, head_vs_reproj_px, valid_entry_count)
    batch_stats = {"prompted": [], "unprompted": []}
    # slot_counts[mode]: (I,3) int TP/FP/FN of per-slot existence, over non-ignored
    # slots (spec §7); lazily sized to I on the first batch/mode that runs.
    slot_counts = {}
    sex_hits = {"prompted": [], "unprompted": []}
    offset = 0
    for b in window_batches(ds, batch_size, shuffle=False, drop_last=False, num_workers=num_workers):
        B0 = b["crops"].shape[0]
        if B0 < batch_size:
            pad = batch_size - B0
            b = {k: np.concatenate([v, np.repeat(v[-1:], pad, axis=0)], axis=0) for k, v in b.items()}
        B = batch_size
        jb = {k: shard_batch(jnp.asarray(v), mesh) for k, v in b.items()}
        has_mask = jb["prompt_mask"].reshape(B, -1).any(-1)
        # weight from the REAL rows only ([:B0]) -- not the padded batch_size-shaped
        # arrays -- so a ragged last batch's padding inflates neither its own nor any
        # other batch's contribution to the cross-batch weighted mean below.
        weight = int((b["vis2d"][:B0] & b["fly_valid"][:B0, :, None, None, None]).sum())
        for mode, prompted in (("prompted", True), ("unprompted", False)):
            on = jnp.full((B,), prompted) & has_mask
            out = _fwd(model, normalize_crops(jb["crops"]), jb["cam_valid"], jb["M"], jb["t_local"],
                       jb["prompt_mask"], on)
            jb_m = dict(jb); jb_m["prompt_on"] = on
            _, m = mvq_loss(out, jb_m, weights, part_of_k)      # reuses the matching
            xyz = np.asarray(out["xyz"])                          # (B,I,T,K,3)
            I = xyz.shape[1]
            if mode not in slot_counts:
                slot_counts[mode] = np.zeros((I, 3), int)
            # label-driven typed-slot targets (P3a §3-4): a slot's existence/sex target is a
            # deterministic function of the labels + this mode's prompt flag, never of the
            # prediction -- computed on the host from the SAME `on` this mode's forward used.
            from jarvis_jax.train.matching import assign_slots, slot_ignore, SLOT_OTHER
            has_f = b["has3d"].astype(np.float32)
            cen = (b["kp3d_local"] * has_f[..., None]).sum((2, 3)) / np.maximum(has_f.sum((2, 3)), 1.0)[..., None]
            dist = np.linalg.norm(cen, axis=-1)
            assign, slot_t = assign_slots(jnp.asarray(b["fly_sex"]), jnp.asarray(b["fly_valid"]), on, jnp.asarray(dist), I)
            assign, slot_t = np.asarray(assign), np.asarray(slot_t)
            # `& ~slot_t`: a slot holding a LABELLED fly certainly exists, so it is never
            # ignored even when the unlabelled animal's sex points at the same slot --
            # the identical correction losses_mvq.mvq_loss applies, so eval's per-slot
            # precision/recall counts exactly the slots the loss supervised.
            ignore = np.asarray(slot_ignore(jnp.asarray(b["unlabelled_sex"]), I)) & ~slot_t
            sex_logit = np.asarray(out["sex_logit"])
            batch_stats[mode].append((float(m["match_reproj_px"]), float(m["uv2d_px"]),
                                      float(m["head_vs_reproj_px"]), weight))
            # per-sample numbers from the matched instance (fly 0 = host): redo the cheap host match.
            # This is the ORACLE instance choice (nearest to GT, which a real deployment does not have
            # access to) -- `mpjpe3d_units` below stays this oracle number; `mpjpe3d_policy_units`
            # (below) uses the POLICY a real inference call would actually use to pick an instance.
            for bi in range(B0):        # only the REAL rows -- padding never enters a per-sample stat
                i_ds = offset + bi
                gt = b["kp3d_local"][bi, 0]; has = b["has3d"][bi, 0]
                d = np.linalg.norm(xyz[bi] - gt[None], axis=-1)      # (I,T,K)
                inst_oracle = int(np.argmin(np.where(has[None], d, 0).sum((1, 2)) / max(has.sum(), 1)))
                e = d[inst_oracle][has]
                L_pred = ([np.linalg.norm(xyz[bi, inst_oracle, 0, a] - xyz[bi, inst_oracle, 0, c])
                          for a, c in seg_idx] if has.sum() > 0 else None)
                # GT segment length per-segment gated on BOTH endpoints having GT (NaN otherwise,
                # so a segment missing one endpoint doesn't silently pull rigid_len_ratio toward 0).
                # gt/has are (T,K,.)/(T,K) -- index frame 0 explicitly (T==1 here), matching L_pred's
                # own xyz[bi, inst_oracle, 0, a] indexing just above.
                L_gt = ([np.linalg.norm(gt[0, a] - gt[0, c]) if has[0, a] and has[0, c] else np.nan
                        for a, c in seg_idx] if has.sum() > 0 else None)
                exist_probs = 1 / (1 + np.exp(-np.asarray(out["exist_logit"][bi])))    # (I,)
                exist = exist_probs >= EXIST_THRESH
                two_fly = ds.n_flies(i_ds) > 1
                # POLICY (P3a §7) -- the ONE shared implementation (models/mvq/policy.py), so the
                # figure scripts and this metric cannot drift apart.
                inst_policy = policy_instance(exist_probs, xyz[bi], prompted=prompted,
                                              has_mask=bool(np.asarray(has_mask)[bi]))
                is_miss = inst_policy is None
                if is_miss:
                    mpjpe_policy = np.nan
                else:
                    e_pol = d[inst_policy][has]
                    mpjpe_policy = float(e_pol.mean()) if e_pol.size else np.nan
                # per-slot existence bookkeeping (non-ignored slots), sex accuracy on assigned known-sex slots
                for s in range(I):
                    if ignore[bi, s]:
                        continue
                    slot_counts[mode][s] += np.array([exist[s] and slot_t[bi, s], exist[s] and not slot_t[bi, s],
                                                      (not exist[s]) and slot_t[bi, s]], int)     # TP, FP, FN
                for f in range(b["fly_valid"].shape[1]):
                    if b["fly_valid"][bi, f] and b["fly_sex"][bi, f] >= 0 and assign[bi, f] >= 0:
                        sex_hits[mode].append((i_ds, int((sex_logit[bi, assign[bi, f]] > 0)
                                                         == (b["fly_sex"][bi, f] == 0))))
                # mask containment of the policy instance's reprojection inside the HOST mask
                # (shared definition, models/mvq/policy.py -- the figure scripts call the same one)
                contain = (mask_containment(xyz[bi, inst_policy, 0], b["M"][bi], b["t_local"][bi, 0],
                                            b["prompt_mask"][bi, 0], b["cam_valid"][bi, 0],
                                            b["vis2d"][bi, 0, 0])
                           if inst_policy is not None else np.nan)
                per_sample[mode].append((float(e.mean()) if e.size else np.nan, int(e.size),
                                         int(exist.sum()), int(b["fly_valid"][bi].sum()),
                                         L_pred, L_gt, i_ds, two_fly, mpjpe_policy, is_miss, contain))
        offset += B0

    def _finish(mode):
        samples = per_sample[mode]
        mp = np.array([p[0] for p in samples]); n = np.array([p[1] for p in samples])
        ok = np.isfinite(mp)
        # np.zeros((0, 4)), not np.array([]), when there were no batches at all (e.g. an
        # empty ds) -- keeps the [:, k] column slices below 1-D-empty instead of IndexError.
        bstats = np.array(batch_stats[mode]) if batch_stats[mode] else np.zeros((0, 4))
        w = bstats[:, 3]
        def _wmean(col):
            if col.size == 0:
                return float("nan")
            return float(np.average(col, weights=w)) if w.sum() > 0 else float(np.mean(col))
        # mpjpe3d_units/_mm are the ORACLE number (ground-truth-nearest instance, see the
        # per-sample loop above) -- mpjpe3d_policy_units/_mm is what a real inference call
        # (no GT to match against) would actually report, and policy_miss_frac is the
        # fraction of windows the policy could not even name an instance for: unprompted
        # windows (or prompted windows that fell back to the unprompted rule because this
        # window had no usable mask) where no TYPED slot (1-3) clears the 0.5 threshold.
        res = {"mpjpe3d_units": float(np.average(mp[ok], weights=n[ok])) if ok.any() else float("nan"),
               "reproj_px": _wmean(bstats[:, 0]), "uv2d_px": _wmean(bstats[:, 1]),
               "head_vs_reproj_px": _wmean(bstats[:, 2])}
        res["mpjpe3d_mm"] = res["mpjpe3d_units"] * MM_PER_UNIT
        mp_policy = np.array([p[8] for p in samples]); ok_policy = np.isfinite(mp_policy)
        miss = np.array([p[9] for p in samples], bool)
        res["mpjpe3d_policy_units"] = (float(np.average(mp_policy[ok_policy], weights=n[ok_policy]))
                                       if ok_policy.any() else float("nan"))
        res["mpjpe3d_policy_mm"] = res["mpjpe3d_policy_units"] * MM_PER_UNIT
        res["policy_miss_frac"] = float(miss.mean()) if len(samples) else 0.0
        # per-slot existence precision/recall (spec §7), from the label-driven TP/FP/FN
        # counted in the per-sample loop above (non-ignored slots only -- an unlabelled
        # fly legitimately present gets no existence loss/credit for its would-be slot).
        sc = slot_counts.get(mode, np.zeros((0, 3), int))
        for s in range(sc.shape[0]):
            tp, fp, fn = (int(x) for x in sc[s])
            res[f"exist_prec_slot{s}"] = float(tp) / max(tp + fp, 1)
            res[f"exist_rec_slot{s}"] = (float(tp) / (tp + fn)) if (tp + fn) > 0 else float("nan")
        # legacy exist_prec/exist_rec keys (logged historically): same per-slot TP/FP/FN,
        # summed over the TYPED slots (1..3) only -- slot 0 (prompted) has no existence
        # target of its own to score (the prompt always targets it directly).
        tp = int(sc[1:, 0].sum()); fp = int(sc[1:, 1].sum()); fn = int(sc[1:, 2].sum())
        res["exist_prec"] = float(tp) / max(tp + fp, 1)
        res["exist_rec"] = float(tp) / max(tp + fn, 1)
        # sex_hits rows are (ds_index, hit) so the same hits can be sliced per cohort
        # below -- a flat list of hits could not be (a two-fly window contributes two).
        hits = sex_hits.get(mode, [])
        res["sex_acc"] = float(np.mean([h for _, h in hits])) if hits else float("nan")
        contain = np.array([p[10] for p in samples], float) if samples else np.zeros(0)
        res["mask_containment"] = float(np.nanmean(contain)) if np.isfinite(contain).any() else float("nan")
        miss_arr = np.array([p[9] for p in samples], bool) if samples else np.zeros(0, bool)
        Ls = [p[4] for p in samples if p[4] is not None]        # skip samples with no 3D-labelled joint
        Ls_gt = [p[5] for p in samples if p[5] is not None]     # same gate as Ls -- see the shared `has.sum()>0` above
        if Ls:
            Ls = np.array(Ls); Ls_gt = np.array(Ls_gt)
            for j, (a, c) in enumerate(seg_names):
                res[f"rigid_spread_mm/{a}-{c}"] = float(np.std(Ls[:, j]) * MM_PER_UNIT)
                gt_col = Ls_gt[:, j]; gt_ok = np.isfinite(gt_col)
                # mean predicted / mean GT length, over samples where THIS segment's
                # GT is available -- a collapsed pair (predicted length -> 0 while GT
                # stays real) shows up here as a ratio near 0, which rigid_spread_mm
                # alone cannot: a collapsed-but-STEADY pair is smoother, not noisier
                # (CLAUDE.md's "check a rigid invariant" history).
                res[f"rigid_len_ratio/{a}-{c}"] = (
                    float(np.mean(Ls[gt_ok, j]) / np.mean(gt_col[gt_ok])) if gt_ok.any() else float("nan"))
        for name, mask in cohorts.items():
            in_cohort = np.array([mask[p[6]] for p in samples], bool) if samples else np.zeros(0, bool)
            sel = in_cohort & ok
            res[f"cohort_{name}"] = float(np.average(mp[sel], weights=n[sel])) if sel.any() else float("nan")
            # The three acceptance metrics PER COHORT (spec §8 grades sex_acc on group C,
            # mask_containment on contact_pair and the policy miss fraction on single-fly
            # windows -- none of which the aggregate numbers above can answer).
            ch = contain[in_cohort] if in_cohort.any() else np.zeros(0)
            res[f"mask_containment_{name}"] = (float(np.nanmean(ch))
                                               if ch.size and np.isfinite(ch).any() else float("nan"))
            res[f"policy_miss_frac_{name}"] = float(miss_arr[in_cohort].mean()) if in_cohort.any() else float("nan")
            ch_hits = [h for i_ds, h in hits if mask[i_ds]]
            res[f"sex_acc_{name}"] = float(np.mean(ch_hits)) if ch_hits else float("nan")
        return res

    return {"prompted": _finish("prompted"), "unprompted": _finish("unprompted")}


def _cohorts(ds):
    n = len(ds)
    c = {"female": np.array([ds.is_female(i) for i in range(n)]),
         "two_fly": np.array([ds.n_flies(i) > 1 for i in range(n)]),
         # single_fly: where the unprompted policy misses hardest (99 % at step 14000,
         # spec §1) and where §8's "miss fraction under 5 %" acceptance target applies.
         "single_fly": np.array([ds.n_flies(i) == 1 for i in range(n)])}
    for g in sorted({ds.calib_group(i) for i in range(n)}):
        c[f"group_{g}"] = np.array([ds.calib_group(i) == g for i in range(n)])

    def _contact(i):
        if ds.n_flies(i) < 2:
            return False
        c = ds.fly_centroids(i)
        return bool(np.isfinite(c).all() and np.linalg.norm(c[0] - c[1]) < CONTACT_UNITS)
    c["contact_pair"] = np.array([_contact(i) for i in range(n)])
    return c


def _balanced_weights(ds, alpha, female_weight):
    cats = [f"{ds.manifest[ds.windows[i][0]].get('behavior', 'unknown')}_{'female' if ds.is_female(i) else 'other'}"
            for i in range(len(ds))]
    counts = {c: cats.count(c) for c in set(cats)}
    w = np.array([(1.0 / counts[c]) ** alpha for c in cats])
    w *= np.where([ds.is_female(i) for i in range(len(ds))], female_weight, 1.0)
    return w / w.sum()


def _make_manager(ckpt_dir, *, max_to_keep=3):
    """mvq-local checkpoint manager: model + optimizer + EMA + ema_meta (Orbax
    items the shared `train/checkpoint.py` doesn't know about -- kept local
    here rather than widening that shared module for one caller). `ema_meta`
    is a small JSON dict ({"ema_updates": int, "ema_zero_seeded": True}); see
    `_restore_latest` for why it must be explicit rather than inferred."""
    os.makedirs(ckpt_dir, exist_ok=True)
    opts = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, save_interval_steps=1)
    return ocp.CheckpointManager(os.path.abspath(ckpt_dir), options=opts,
                                 item_names=("model", "opt", "ema", "ema_meta"))


def _save_step(mngr, steps_done, model, optimizer, ema, ema_updates: int):
    mngr.save(steps_done, args=ocp.args.Composite(
        model=ocp.args.StandardSave(nnx.split(model)[1]),
        opt=ocp.args.StandardSave(nnx.split(optimizer)[1]),
        ema=ocp.args.StandardSave(ema),
        ema_meta=ocp.args.JsonSave({"ema_updates": int(ema_updates), "ema_zero_seeded": True})))


def _replicated_abstract(tree):
    """Abstract (ShapeDtypeStruct) twin of `tree` with a replicated sharding
    over all current devices, for topology-independent Orbax restores."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())
    return jax.tree_util.tree_map(
        lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=repl)
        if hasattr(v, "shape") and hasattr(v, "dtype") else v, tree)


def _restore_latest(mngr, model, optimizer, ema):
    """(model, optimizer, ema, ema_updates, start_step); ema restored using the
    freshly-built `ema` pytree as the abstract target, same pattern
    train/checkpoint.py's restore_latest uses for model/optimizer.
    (model, optimizer, ema, 0, 0) unchanged if no checkpoint exists yet.

    `_with_ema` divides by `(1 - decay**t)` unconditionally to debias a
    zero-seeded EMA -- but a PRE-fix checkpoint (EMA seeded from live params,
    not zero; every checkpoint before 2026-09-04) would be silently rescaled
    by that same divisor as though it needed the same correction, and there
    is no way to tell the two apart from the `ema` pytree's contents alone.
    `ema_meta` (saved alongside `ema` from this fix onward) makes the
    provenance explicit instead of inferred: its `ema_updates` becomes `t`,
    and a checkpoint missing this item (KeyError from Orbax, since `mngr`'s
    `item_names` declares it mandatory) or whose `ema_zero_seeded` is not
    True is REFUSED (ValueError) rather than guessed at -- resume with a
    fresh run instead of trying to convert it.
    """
    latest = mngr.latest_step()
    if latest is None:
        return model, optimizer, ema, 0, 0
    gm, am = nnx.split(model)
    go, ao = nnx.split(optimizer)
    # Restore targets carry an explicit REPLICATED sharding over the current
    # devices (the same pattern as train/checkpoint.py::warm_start_restore).
    # Without it Orbax infers the sharding from the checkpoint's own metadata
    # and refuses with "Topology mismatch" whenever the device count differs
    # from the one that saved it (8-GPU save -> 4-GPU or CPU resume). The
    # arrays are re-replicated by `replicate(..., mesh)` right after restore.
    am, ao, ema_t = (_replicated_abstract(t) for t in (am, ao, ema))
    try:
        r = mngr.restore(latest, args=ocp.args.Composite(
            model=ocp.args.StandardRestore(am),
            opt=ocp.args.StandardRestore(ao),
            ema=ocp.args.StandardRestore(ema_t),
            ema_meta=ocp.args.JsonRestore()))
    except KeyError as e:
        raise ValueError(
            f"checkpoint at {mngr.directory} (step {latest}) has no 'ema_meta' item -- "
            f"it predates the zero-seeded, debiased EMA (fix round 2, 2026-09-04) and "
            f"cannot be resumed: its EMA was seeded from live params, not zero, so "
            f"_with_ema's /(1-decay**t) debiasing would silently rescale it as though "
            f"it needed the same correction, which it does not. Start a fresh run "
            f"instead -- do not attempt to resume or convert this checkpoint.") from e
    meta = r["ema_meta"]
    if not meta.get("ema_zero_seeded", False):
        raise ValueError(
            f"checkpoint at {mngr.directory} (step {latest}) has ema_meta={meta!r} -- "
            f"'ema_zero_seeded' is not True, so its EMA cannot be resumed by this code. "
            f"Start a fresh run instead.")
    return (nnx.merge(gm, r["model"]), nnx.merge(go, r["opt"]), r["ema"],
           int(meta["ema_updates"]), latest)


def run_training(root, *, out_dir, ckpt_dir, mcfg: MVQConfig, tcfg: MVQTrainConfig,
                 aug: MVAugParams, weights: LossWeights):
    n_dev = len(jax.devices())
    if tcfg.batch_size % n_dev:
        raise ValueError(f"batch_size {tcfg.batch_size} not divisible by {n_dev} devices")

    # Dataset load + cohort validation BEFORE any GPU-heavy work (model build,
    # pretrained-weight download, device replication): a misconfigured/empty
    # val cohort should fail fast, not after minutes of backbone init.
    copy_paste = (CopyPasteParams(p=tcfg.copy_paste_p, opposite_sex_p=tcfg.copy_paste_opposite_sex_p,
                                  contact_p=tcfg.copy_paste_contact_p) if tcfg.copy_paste_p > 0 else None)
    train_sets = {T: V12WindowDataset(root, "train", T=T, train=True, seed=tcfg.seed, copy_paste=copy_paste,
                                      jitter_units=tcfg.jitter_units)
                 for T in tcfg.window_lengths}
    val_ds = V12WindowDataset(root, "val", T=1, train=False)
    names = train_sets[tcfg.window_lengths[0]].keypoint_names
    lr_swap = build_lr_swap(names); part_of_k, _ = build_part_index(names)
    cohorts = _cohorts(val_ds)
    for c in tcfg.val_cohorts:
        if not cohorts.get(c, np.zeros(1, bool)).any():
            raise ValueError(f"val cohort '{c}' is empty on {root}")

    # Written BEFORE the model build (below) so a failed model build (bad
    # backbone name, OOM, etc.) still leaves a usable config on disk -- see
    # `models/mvq/checkpoint.py::load_mvq_model`, which now reads this run-dir
    # copy first and falls back to `final/mvq_run.json` for older layouts.
    # `val` is None here (not yet computed) and overwritten with the real
    # numbers by the final write at the end of this function.
    run_dir = os.path.dirname(os.path.abspath(out_dir))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "mvq_run.json"), "w") as f:
        json.dump({"model": dataclasses.asdict(mcfg), "train": dataclasses.asdict(tcfg), "val": None,
                  "keypoint_names": names}, f, indent=1)

    model = MVQModel(mcfg, rngs=nnx.Rngs(tcfg.seed))
    if tcfg.pretrained:
        model.backbone = load_dinov3_safetensors(model.backbone, dinov3_snapshot(HF_REPOS[mcfg.backbone]))
        print(f"[mvq] loaded {HF_REPOS[mcfg.backbone]}")
    # `mngr` must exist BEFORE the warm start so a requeue can skip it: resume beats
    # warm start (warm_start_restore's docstring rules this), so reading the source
    # checkpoint only to have `_restore_latest` overwrite it below is pure waste --
    # tens of GB of Orbax reads on every preemption of a long fine-tune.
    mngr = _make_manager(ckpt_dir) if ckpt_dir else None
    resuming = mngr is not None and mngr.latest_step() is not None
    if tcfg.warm_start and resuming:
        print(f"[mvq] warm start from {tcfg.warm_start} SKIPPED: this run's own ckpt/ is at step "
             f"{mngr.latest_step()}, and resume beats warm start", flush=True)
    elif tcfg.warm_start:
        model, skipped = warm_start_partial(model, tcfg.warm_start)
        total = len(jax.tree_util.tree_leaves(nnx.state(model)))
        print(f"[mvq] warm start from {tcfg.warm_start}: restored {total - len(skipped)}/{total} leaves; "
             f"not restored: {skipped}", flush=True)
    opt = make_optimizer(model, tcfg)
    # EMA seeded at ZERO (not the step-0 params): a running sum, debiased by
    # `_with_ema`'s `/(1-decay**t)` divisor at read time (t = the number of
    # updates so far, which is always exactly the current absolute step count
    # `i+1` since the EMA is seeded once at true step 0 -- including across a
    # resume, since `ema` is restored from the checkpoint below along with
    # `start` -- and updated exactly once per training step; no separate
    # counter needs to be saved/restored, the step count already is one).
    # Seeding from params (the old behaviour) meant the near-zero init xyz
    # head still carried 0.999**2000 = 13.5% weight at step 2000 -- measured
    # as a uniform SHRINK of every predicted joint toward the crop centre,
    # worse for distal legs (radial dist/GT ratio 0.68-0.83) than proximal
    # body joints (0.86-0.93), since distal joints sit furthest from centre.
    ema = jax.tree_util.tree_map(jnp.zeros_like, nnx.state(model, nnx.Param))
    start = 0
    ema_updates = 0     # tracked explicitly (not inferred from the step count -- see
                        # _restore_latest); incremented once per training step below,
                        # exactly like `start`/`i+1`, and persisted alongside `ema`.
    if mngr is not None:
        model, opt, ema, ema_updates, start = _restore_latest(mngr, model, opt, ema)
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
        loss = float(loss); ema_updates += 1
        if (i + 1) % tcfg.log_every == 0:
            ms = " ".join(f"{k}={float(v):.4f}" for k, v in metrics.items() if k != "total")
            print(f"step {i+1}/{tcfg.total_steps} T={T} loss {loss:.4f} {ms} ({time.time()-t0:.0f}s)", flush=True)
        if (i + 1) % tcfg.eval_every == 0 or i + 1 == tcfg.total_steps:
            em = _with_ema(model, ema, tcfg.ema, ema_updates)
            val = evaluate(em, val_ds, tcfg.batch_size, cohorts=cohorts, part_of_k=part_of_k,
                           weights=weights, mesh=mesh, num_workers=tcfg.num_workers)
            for mode, r in val.items():
                print(f"  val[{mode}] " + " ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)
            # `del em` frees nothing by itself -- em's arrays alias the SAME
            # underlying device buffers as `ema` (nnx.update wrote references,
            # not copies), which are still live via the `ema` pytree in this
            # scope. The actual relief is `jax.clear_caches()`, which discards
            # eval's own compiled `_fwd` executable and its cached buffers;
            # WITHOUT it those otherwise linger and fragment GPU 0's BFC pool
            # enough that the VERY NEXT training step's ~19GB arena OOMs, and
            # the other 3 ranks hang on the NCCL clique rendezvous (measured
            # 2026-09-03). Cost: the training step recompiles on the next
            # call after every eval (a multi-second one-time hit each time,
            # not per-step).
            del em
            jax.clear_caches()
        if mngr is not None and (i + 1) % tcfg.save_every == 0:
            _save_step(mngr, i + 1, model, opt, ema, ema_updates)
    if mngr is not None:
        _save_step(mngr, tcfg.total_steps, model, opt, ema, ema_updates); mngr.wait_until_finished()
    em = _with_ema(model, ema, tcfg.ema, ema_updates)
    val = evaluate(em, val_ds, tcfg.batch_size, cohorts=cohorts, part_of_k=part_of_k,
                   weights=weights, mesh=mesh, num_workers=tcfg.num_workers)
    os.makedirs(out_dir, exist_ok=True)
    ckptr = ocp.StandardCheckpointer(); ckptr.save(out_dir, nnx.split(em)[1], force=True); ckptr.wait_until_finished()
    with open(os.path.join(out_dir, "mvq_run.json"), "w") as f:
        json.dump({"model": dataclasses.asdict(mcfg), "train": dataclasses.asdict(tcfg), "val": val,
                  "keypoint_names": names}, f, indent=1)
    return {"final_loss": loss, "val": val, "steps": tcfg.total_steps, "resumed_from": start,
           "ema_updates": ema_updates}


def _with_ema(model, ema, decay: float, t: int):
    """A NEW module holding the DEBIASED EMA weights, independent of `model`'s
    own Variable objects. `nnx.split`/`nnx.merge` on `model` itself would
    return a module aliasing the SAME Variables as `model` (flax 0.12.8), so
    `nnx.update(em, ema)` would silently overwrite the training model's live
    params too -- `nnx.clone` makes a real copy first.

    `ema` is a raw running sum seeded from ZERO (see run_training), so after
    `t` updates it still carries a `decay**t` shortfall relative to the true
    average -- the same bias Adam corrects for its moment estimates. `t=0`
    (no updates yet) returns the live params undebiased, since 0/0 is
    undefined and this case should not arise in normal use (eval/save always
    happen at t>=1)."""
    em = nnx.clone(model)
    if t > 0:
        correction = 1.0 - decay ** t
        ema = jax.tree_util.tree_map(lambda e: e / correction, ema)
        nnx.update(em, ema)
    em.eval()
    return em
