"""Training loop for the multi-view query lifter (mvq)."""
from __future__ import annotations

import collections
import dataclasses
import json
import os
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.data.augment import assert_lr_swap_covers, build_lr_swap
from jarvis_jax.data.concat_windows import ConcatWindowDataset
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
from jarvis_jax.train.losses_mvq import LossWeights, mvq_loss, wing_kp_weight

MM_PER_UNIT = 0.1
CONTACT_UNITS = 30.0  # 3 mm; real mounting pairs have centroid gaps of ~24-30 units, see p3a-notes.md
_RATIO_BATCHES = 200  # batches over which run_training reports the realised host-sex sampling ratio
# v2 (spec 2026-09-05 §4): the extra TRAIN roots concatenated onto the human v12
# root, in the order they are stacked. "negatives" is special-cased by
# `_mix_weights` (it is the one root whose share is pinned, not proportional to
# its window count); the others split the remaining mass by window count.
_NEGATIVES_NAME = "negatives"


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
    # P3b: the (lo, hi) centroid separation, in world units (0.1 mm), a CONTACT paste
    # is drawn from. `CopyPasteParams.contact_sep`'s own default (8, 30) brackets the
    # real 24-30-unit mounting pairs; P3b tightens it to (4, 25) to spend the pastes
    # on heavier overlap, which is where the slots actually mix.
    copy_paste_contact_sep: tuple = (8.0, 30.0)
    # P3b: multiplier on FEMALE-HOST windows' sampling weight, applied AFTER
    # `_balanced_weights`' behaviour-category normalisation (unlike `female_weight`,
    # which is folded into that normalisation and whose effect on the realised ratio
    # is therefore data-dependent and opaque). 1.0 = unchanged. `run_training` prints
    # both the resulting weight mass and the realised host-sex ratio of the first
    # `_RATIO_BATCHES` batches, so the number is verifiable in the run log.
    female_host_weight: float = 1.0
    # P3b: {recording: {fly_index: "female"|"male"}} forced onto the window loader's
    # sex-resolution chain, above the annotation and the manifest. Empty by default.
    sex_label_overrides: dict = dataclasses.field(default_factory=dict)
    warm_start: str | None = None
    jitter_units: float = 3.0  # train-time window-centre jitter, world units (0.1 mm); P4 §6 raises it to 10 for the mask-free route
    # ---- mvq v2 (spec 2026-09-05 §2 decision 3, §4) ---------------------------
    # Frame spacings a T > 1 window may span, in VIDEO frames: {1, 4, 16} is
    # 1.25-20 ms at 800 fps. Ignored at T=1. `(1,)` is the pre-v2 behaviour
    # (consecutive pairs only), so every P3a/P3b config is unchanged.
    pair_deltas: tuple = (1,)
    # Extra TRAIN roots concatenated onto the human v12 root (`ConcatWindowDataset`);
    # the VAL split stays the human root alone (spec §7, "validation on human labels
    # only"). `pseudo_weight` is the per-sample loss weight those roots are EXPECTED
    # to carry -- the actual number reaching `sample_weight` comes from each root's
    # own manifest (`V12WindowDataset.weight`), and `run_training` warns if the two
    # disagree rather than silently training at the export's value.
    pseudo_root: str | None = None
    pseudo_weight: float = 0.3
    singlefly_root: str | None = None
    # Empty-window negatives (spec §3.5): `negatives_frac` of the SAMPLED MASS,
    # pinned rather than proportional to the root's window count -- how many
    # negatives happen to have been exported must not decide how often the
    # existence head sees one.
    negatives_root: str | None = None
    negatives_frac: float = 0.05
    # v2: the female-host mass to SOLVE for, PER ROOT (`_balanced_weights`). Each
    # root has its own census, so the one hand-solved `female_host_weight` above
    # is only right for the root it was solved on -- applied to all four and
    # mass-averaged it lands wherever they happen to average out to (measured
    # ~0.75 female for 4.27). None = fall back to `female_host_weight` (the P3b
    # path, so `mvq.yaml` runs are unchanged).
    female_host_target: float | None = None
    # Per-keypoint loss multiplier on the WING landmarks (spec §4: wing keypoint
    # AND wing visibility weight x2). Built BY NAME (`losses_mvq.wing_kp_weight`),
    # never by index -- see CLAUDE.md's keypoint-order history.
    wing_kp_mult: float = 1.0


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


def make_train_step(aug: MVAugParams, lr_swap, part_of_k, weights: LossWeights, ema_decay: float,
                    kp_weight=None):
    """`kp_weight`: the (K,) per-keypoint loss multiplier (`losses_mvq.wing_kp_weight`,
    built BY NAME), or None for the uniform pre-v2 behaviour."""
    swap = jnp.asarray(lr_swap); pok = np.asarray(part_of_k)
    kpw = None if kp_weight is None else jnp.asarray(kp_weight)

    def loss_fn(model, batch):
        out = model(**_batch_to_model(batch), prompt_on=batch["prompt_on"])
        return mvq_loss(out, batch, weights, pok, kpw)

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


def evaluate(model, ds, batch_size, *, cohorts: dict, part_of_k, weights, mesh, num_workers=8,
             kp_weight=None):
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

    `kp_weight` (v2) is passed straight through to `mvq_loss` so eval scores the
    SAME objective training optimises. It moves no number reported here: the
    three px metrics and every per-sample statistic below are built from
    unweighted L2 distances, and eval discards `mvq_loss`'s scalar loss.

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
    # mask_containment_frac|nan -- the policy instance's reprojection inside the host mask,
    # cross_fly_frac|nan -- P3b: on a window with TWO labelled flies, the fraction of an
    #   ASSIGNED slot's predicted 3D keypoints that sit nearer the OTHER fly's GT centroid
    #   than their own, averaged over the labelled flies. This is the identity-mixing
    #   failure measured directly: mask_containment says "inside the host mask" (generous
    #   for a 50-keypoint skeleton and blind to the second fly), this says "on the wrong
    #   animal". NaN on single-fly windows.
    per_sample = {"prompted": [], "unprompted": []}
    # batch_stats[mode] rows: (reproj_px, uv2d_px, head_vs_reproj_px, valid_entry_count)
    batch_stats = {"prompted": [], "unprompted": []}
    # slot_counts[mode]: (I,3) int TP/FP/FN of per-slot existence, over non-ignored
    # slots (spec §7); lazily sized to I on the first batch/mode that runs.
    slot_counts = {}
    # cross_counts[mode]: (I,2) int [n_keypoints_on_the_other_fly, n_keypoints_scored]
    # per SLOT, over two-labelled-fly windows -- the per-typed-slot view of cross_fly_frac.
    cross_counts = {}
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
            _, m = mvq_loss(out, jb_m, weights, part_of_k, kp_weight)   # reuses the matching
            xyz = np.asarray(out["xyz"])                          # (B,I,T,K,3)
            I = xyz.shape[1]
            if mode not in slot_counts:
                slot_counts[mode] = np.zeros((I, 3), int)
                cross_counts[mode] = np.zeros((I, 2), int)
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
                # cross_fly_frac (P3b): is this slot's prediction on the right ANIMAL?
                # `cen` (above) is each labelled fly's GT 3D centroid; a predicted keypoint
                # nearer the OTHER fly's centroid than its own has crossed the two
                # centroids' midplane. Only scored where BOTH flies are labelled, assigned
                # to a slot, and have a 3D centroid at all.
                fl = [f for f in range(b["fly_valid"].shape[1])
                      if b["fly_valid"][bi, f] and assign[bi, f] >= 0 and has_f[bi, f].sum() > 0]
                cross = np.nan
                if len(fl) >= 2:
                    vals = []
                    for f in fl:
                        sl = int(assign[bi, f])
                        d_own = np.linalg.norm(xyz[bi, sl] - cen[bi, f], axis=-1)          # (T,K)
                        d_oth = np.min([np.linalg.norm(xyz[bi, sl] - cen[bi, o], axis=-1)
                                        for o in fl if o != f], axis=0)
                        wrong = d_own > d_oth
                        cross_counts[mode][sl] += np.array([int(wrong.sum()), int(wrong.size)], int)
                        vals.append(float(wrong.mean()))
                    cross = float(np.mean(vals))
                per_sample[mode].append((float(e.mean()) if e.size else np.nan, int(e.size),
                                         int(exist.sum()), int(b["fly_valid"][bi].sum()),
                                         L_pred, L_gt, i_ds, two_fly, mpjpe_policy, is_miss, contain,
                                         cross))
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
        cross = np.array([p[11] for p in samples], float) if samples else np.zeros(0)
        res["cross_fly_frac"] = float(np.nanmean(cross)) if np.isfinite(cross).any() else float("nan")
        cc = cross_counts.get(mode, np.zeros((0, 2), int))
        for s_ in range(cc.shape[0]):
            res[f"cross_fly_frac_slot{s_}"] = (float(cc[s_, 0]) / cc[s_, 1]) if cc[s_, 1] else float("nan")
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
            cx = cross[in_cohort] if in_cohort.any() else np.zeros(0)
            res[f"cross_fly_frac_{name}"] = (float(np.nanmean(cx))
                                             if cx.size and np.isfinite(cx).any() else float("nan"))
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


def _balanced_weights(ds, alpha, female_weight, female_host_weight=1.0,
                      female_host_target=None, label=None):
    """Per-window sampling weights, normalised to sum 1.

    `female_weight` (P2) multiplies female-host windows INSIDE the
    behaviour-category balance, so how much of the sampled mass it actually buys
    depends on how the categories fall out. `female_host_weight` (P3b) multiplies
    them again on the FINAL normalised weights, so the mass ratio it produces is
    exactly `female_host_weight x (mass_F / mass_M)`.

    `female_host_target` (v2) SOLVES that multiplier instead of taking it on
    faith: given this dataset's own post-balance female mass `f`, the multiplier
    that lands the female-host mass exactly on `target` is
    `target*(1-f) / ((1-target)*f)`. A hand-tuned `female_host_weight` is only
    correct for the ONE census it was solved on -- 4.27 was solved for
    `red_data_3d_v12_export0902` train (202 female-host windows of 2661) -- and
    v2 applies the balance PER ROOT, where each root has its own census, so a
    single hand number stacks into whatever the roots happen to average out to.
    When `female_host_target` is set it wins and `female_host_weight` is unused.
    A root with no female host (or no male/other host) cannot reach any interior
    target at all: the multiplier stays 1.0 and a note is printed naming `label`.
    """
    is_f = np.array([ds.is_female(i) for i in range(len(ds))], bool)
    cats = [f"{ds.manifest[ds.windows[i][0]].get('behavior', 'unknown')}_{'female' if is_f[i] else 'other'}"
            for i in range(len(ds))]
    counts = {c: cats.count(c) for c in set(cats)}
    w = np.array([(1.0 / counts[c]) ** alpha for c in cats])
    w *= np.where(is_f, female_weight, 1.0)
    w = w / w.sum()
    mult = float(female_host_weight)
    if female_host_target is not None:
        t = float(female_host_target)
        if not 0.0 < t < 1.0:
            raise ValueError(f"female_host_target must be strictly between 0 and 1, got {t}")
        f = float(w[is_f].sum())
        who = f"[mvq] {label or getattr(ds, 'root', 'train set')}"
        # One-sided by COUNT first (exact), then by mass with a tolerance: an
        # all-female root's `f` comes back as 0.9999999999999998, not 1.0, and a
        # bare `f >= 1.0` would sail past it and "solve" a multiplier of 4e-16 --
        # which reads as a legitimate number in the log and would zero out the
        # female mass of any root that is merely NEARLY one-sided.
        n_f = int(is_f.sum())
        one_sided = n_f == 0 or n_f == len(is_f) or f <= 1e-9 or f >= 1.0 - 1e-9
        if one_sided:
            mult = 1.0
            side = "no female-host window" if n_f == 0 or f <= 1e-9 else "no male/other-host window"
            print(f"{who}: female_host_target={t:.3f} UNATTAINABLE -- this root has {side} "
                  f"(post-balance female mass {f:.4f}); multiplier left at 1.0 and the root's "
                  f"sampling weights are unchanged", flush=True)
        else:
            mult = t * (1.0 - f) / ((1.0 - t) * f)
            print(f"{who}: female_host_target={t:.3f}, post-balance female mass {f:.4f} "
                  f"-> solved female-host multiplier {mult:.4f} (train.female_host_weight="
                  f"{female_host_weight} is unused while a target is set)", flush=True)
    w = w * np.where(is_f, mult, 1.0)
    return w / w.sum()


def _mix_weights(ds, tcfg):
    """(weights, mass) for ONE train dataset: per-window sampling weights summing
    to 1, and the per-root mass table behind them.

    A plain `V12WindowDataset` is just `_balanced_weights` (unchanged). A
    `ConcatWindowDataset` gets `_balanced_weights` run SEPARATELY on each root,
    so each root's behaviour balance and host-sex ratio are restored INSIDE it
    -- which is why the pseudo export's manifest `balance.female_host_weight`
    is information to log, not a second multiplier to apply here (applying it
    again would over-sample female hosts by ~2x), and why the female-host
    multiplier is SOLVED per root from `female_host_target` rather than taken
    from a hand number solved on one census (which, applied to every root and
    then mass-averaged, lands wherever the roots happen to average out to --
    measured ~0.75 female for 4.27 across these four). The roots' TOTAL masses are
    then set so the negatives take exactly `negatives_frac` and the remaining
    mass splits by window count: how many negatives the exporter happened to
    write must not decide how often the existence head sees an empty window,
    whereas the real/pseudo/single-fly split IS meant to follow how much data
    each root actually holds (their per-sample loss weight, not their sampling
    rate, is what marks a pseudo label as less trustworthy).
    """
    frac = float(tcfg.negatives_frac)
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"negatives_frac must be in [0, 1), got {frac} -- at 1.0 the sampler would "
                         f"draw nothing but empty windows")
    subs = list(getattr(ds, "datasets", [ds]))
    names_ = list(getattr(ds, "names", ["real"]))
    ws = []
    for d, nm in zip(subs, names_):
        if len(d) == 0:
            raise ValueError(
                f"train root '{nm}' ({getattr(d, 'root', '?')}) contributes 0 windows at T={ds.T} "
                f"-- a root with no window at this length would silently drop out of the mix "
                f"(check pair_deltas: a Delta no pair of labelled frames spans yields nothing)")
        ws.append(_balanced_weights(d, tcfg.balance_alpha, tcfg.female_weight, tcfg.female_host_weight,
                                    female_host_target=tcfg.female_host_target,
                                    label=f"T={ds.T} root '{nm}'"))
    n_win = np.array([float(len(d)) for d in subs])
    is_neg = np.array([nm == _NEGATIVES_NAME for nm in names_], bool)
    if is_neg.any():
        pos = np.where(is_neg, 0.0, n_win)
        mass = np.where(is_neg, frac / max(int(is_neg.sum()), 1),
                        pos / max(pos.sum(), 1e-12) * (1.0 - frac))
    else:
        mass = n_win / max(n_win.sum(), 1e-12)
    w = np.concatenate([wi * m for wi, m in zip(ws, mass)])
    return w / w.sum(), dict(zip(names_, (float(m) for m in mass)))


class _DatasetView:
    """Transparent wrapper over a window dataset: everything the loader,
    sampler and cohort code asks (`is_female`, `n_flies`, `manifest`,
    `windows`, `delta`, ...) delegates to the wrapped dataset, so a view can
    be handed to `ConcatWindowDataset`/`window_batches` in place of the real
    thing. `epoch` needs an explicit property because it is WRITTEN (once per
    epoch, by `window_batches`) and the write has to reach the real dataset."""

    def __init__(self, ds):
        self._ds = ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        return self._ds[i]

    @property
    def epoch(self):
        return self._ds.epoch

    @epoch.setter
    def epoch(self, value):
        self._ds.epoch = int(value)       # must reach every sub-dataset (ConcatWindowDataset fans out)

    def __getattr__(self, k):
        if k == "_ds":                    # before __init__, or after a failed unpickle
            raise AttributeError(k)
        return getattr(self._ds, k)


class _ForceSampleWeight(_DatasetView):
    """A view whose every window reports and carries `sample_weight = value`.

    Used for the empty-window NEGATIVES root (spec §3.5). The export ships a
    whole-manifest `weight` of 0.3 like the rest of the pseudo-labelled data,
    but `sample_weight` multiplies EVERY loss term per sample -- including the
    existence BCE that is the only reason a negative exists -- so training a
    negative at 0.3 would down-weight the one target it carries. The sampling
    RATE of negatives is set separately and explicitly by `negatives_frac`."""

    def __init__(self, ds, value=1.0):
        super().__init__(ds)
        self._value = np.float32(value)

    def weight(self, i):
        return float(self._value)

    def __getitem__(self, i):
        s = dict(self._ds[i])
        s["sample_weight"] = np.float32(self._value)
        return s


class _MixCounter(_DatasetView):
    """A view that COUNTS which root each drawn window came from.

    The realised real/pseudo/negative mix has to be counted, not inferred: two
    roots can carry the same `sample_weight` (the batch's only per-sample
    provenance signal), and reproducing the sampler's own RNG draw here would
    silently drift the moment `window_batches` changed how it draws. Counting
    at `__getitem__` is exact for whatever the sampler actually asked for.

    `window_batches` fetches a batch from a ThreadPoolExecutor, so the counter
    is locked; `prefetch` runs up to `depth` batches ahead, so the report is
    keyed on the counter's OWN total rather than assuming it is in lock-step
    with the training step.
    """

    def __init__(self, ds):
        super().__init__(ds)
        self._lock = threading.Lock()
        self.counts = collections.Counter()

    def __getitem__(self, i):
        nm = self._ds.name(i)
        with self._lock:
            self.counts[nm] += 1
        return self._ds[i]

    def realised(self):
        with self._lock:
            c = dict(self.counts)
        n = max(sum(c.values()), 1)
        return c, n


# `mvq_loss` metric name -> the LossWeights field that multiplies it into `total`.
# `conf` is absent on purpose: it enters `total` at weight 1 (LossWeights.conf is
# lambda INSIDE the term, not an outer weight -- see losses_mvq).
_TERM_WEIGHTS = {"reproj": "reproj", "l3d": "l3d", "uv2d": "uv2d", "vis": "vis",
                 "exist": "exist", "sex": "sex", "rep": "rep",
                 "other_rep": "other_fly_repulsion", "persist": "persist"}
# the only metrics `_loss_shares` reads -- accumulating the rest would force a
# host sync on per-batch diagnostics (exist_acc, n_negative, ...) nothing reports.
_SHARE_KEYS = tuple(_TERM_WEIGHTS) + ("conf", "total")


def _loss_shares(sums, n, weights: LossWeights):
    """Each loss term's WEIGHTED share of `total`, averaged over `n` steps.

    `mvq_loss` reports the deep-supervision terms (pass1 + aux layers) only
    inside `total`, so their contribution is reported as the remainder
    `deep_supervision` rather than left as an unexplained gap -- the shares
    then sum to 1 and a term's share can be read as a fraction of the whole.
    """
    total = sums["total"] / n
    terms, named = {}, 0.0
    for name, attr in _TERM_WEIGHTS.items():
        w = float(getattr(weights, attr))
        mean = sums.get(name, 0.0) / n
        terms[name] = {"weight": w, "mean": mean, "weighted": w * mean}
        named += w * mean
    terms["conf"] = {"weight": 1.0, "mean": sums.get("conf", 0.0) / n,
                     "weighted": sums.get("conf", 0.0) / n}
    named += terms["conf"]["weighted"]
    terms["deep_supervision"] = {"weight": float(weights.pass1), "mean": float("nan"),
                                 "weighted": total - named}
    for t in terms.values():
        t["share"] = t["weighted"] / total if total else float("nan")
    return {"steps": int(n), "total": total, "terms": terms}


def _print_loss_shares(sh):
    print(f"[mvq] loss shares over {sh['steps']} steps (mean total {sh['total']:.4f}):", flush=True)
    for name, t in sorted(sh["terms"].items(), key=lambda kv: -kv[1]["weighted"]):
        print(f"[mvq]   {name:<17s} w={t['weight']:<6.3g} mean={t['mean']:<10.4g} "
              f"weighted={t['weighted']:<10.4g} share={100 * t['share']:6.2f}%", flush=True)
    s = 100 * sh["terms"]["other_rep"]["share"]
    w = sh["terms"]["other_rep"]["weight"]
    if w <= 0:
        verdict = "other_fly_repulsion is OFF (weight 0)"
    elif s > 50:
        verdict = f"DOMINANT (> 50 %): halve the weight to {w / 2:g} and re-measure"
    elif s < 2:
        verdict = f"INERT (< 2 %): raise the weight to {2 * w:g} and re-measure"
    elif 5 <= s <= 30:
        verdict = "in the 5-30 % launch band"
    else:
        verdict = "borderline (outside 5-30 %, inside 2-50 %): judge against p3b-notes.md"
    print(f"[mvq] other_fly_repulsion (weight {w:g}) share = {s:.2f}% -- {verdict}", flush=True)


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
                 aug: MVAugParams, weights: LossWeights, share_check_steps: int = 0):
    """`share_check_steps > 0` (v2 Step 6): run that many steps of the REAL
    pipeline -- same concatenated roots, sampler, augmentation and T
    alternation the full run will use -- print each loss term's weighted share
    of `total`, and return `{"loss_shares": ..., ...}` WITHOUT evaluating or
    writing a final checkpoint. Reusing `run_training` rather than a separate
    harness is the point of the check: a standalone loop would measure a
    different loss balance than the one the 40k launch actually optimises.
    """
    n_dev = len(jax.devices())
    if tcfg.batch_size % n_dev:
        raise ValueError(f"batch_size {tcfg.batch_size} not divisible by {n_dev} devices")
    n_share = int(share_check_steps or 0)
    if n_share > 0:
        tcfg = dataclasses.replace(tcfg, total_steps=n_share,
                                   log_every=max(1, min(tcfg.log_every, n_share // 4 or 1)))
        ckpt_dir = None

    # Dataset load + cohort validation BEFORE any GPU-heavy work (model build,
    # pretrained-weight download, device replication): a misconfigured/empty
    # val cohort should fail fast, not after minutes of backbone init.
    copy_paste = (CopyPasteParams(p=tcfg.copy_paste_p, opposite_sex_p=tcfg.copy_paste_opposite_sex_p,
                                  contact_p=tcfg.copy_paste_contact_p,
                                  contact_sep=tuple(tcfg.copy_paste_contact_sep)) if tcfg.copy_paste_p > 0 else None)
    ov = dict(tcfg.sex_label_overrides or {})
    pair_deltas = tuple(int(d) for d in (tcfg.pair_deltas or (1,)))
    train_sets = {T: V12WindowDataset(root, "train", T=T, train=True, seed=tcfg.seed, copy_paste=copy_paste,
                                      jitter_units=tcfg.jitter_units, sex_overrides=ov,
                                      pair_deltas=pair_deltas)
                 for T in tcfg.window_lengths}
    # v2 (spec §4): the pseudo-label, single-fly and empty-window-negative
    # exports are SEPARATE v12 roots (own calibrations, images, annotation ids),
    # so they are stacked index-wise onto the human root rather than merged.
    # TRAIN ONLY -- `val_ds` below stays the human root alone (spec §7,
    # "validation on human labels only"): a pseudo frameset in the val set
    # would move the very number the run is judged by.
    extra = [(tcfg.pseudo_root, "pseudo", copy_paste), (tcfg.singlefly_root, "singlefly", copy_paste),
             (tcfg.negatives_root, _NEGATIVES_NAME, None)]   # a negative has no host to paste onto
    for T in list(train_sets):
        parts, part_names = [train_sets[T]], ["real"]
        for path, nm, cp in extra:
            if not path:
                continue
            d = V12WindowDataset(path, "train", T=T, train=True, seed=tcfg.seed, copy_paste=cp,
                                 jitter_units=tcfg.jitter_units, sex_overrides=ov,
                                 pair_deltas=pair_deltas)
            if nm == _NEGATIVES_NAME:
                # The export ships the pseudo-label weight (0.3); a negative must
                # train at 1.0 -- see `_ForceSampleWeight`.
                exported = float(d.weight(0)) if len(d) else float("nan")
                print(f"[mvq] T={T} negatives root {path}: export sample_weight {exported:g} "
                      f"OVERRIDDEN to 1.0 -- sample_weight multiplies every loss term, and the "
                      f"existence target is the only thing an empty window carries (spec §3.5); "
                      f"how OFTEN a negative is drawn is set by negatives_frac="
                      f"{tcfg.negatives_frac}, not by its loss weight", flush=True)
                d = _ForceSampleWeight(d, 1.0)
            parts.append(d)
            part_names.append(nm)
        if len(parts) > 1:
            train_sets[T] = ConcatWindowDataset(parts, names=part_names)
    val_ds = V12WindowDataset(root, "val", T=1, train=False, sex_overrides=ov)
    names = train_sets[tcfg.window_lengths[0]].keypoint_names
    lr_swap = build_lr_swap(names); part_of_k, _ = build_part_index(names)
    # The horizontal flip relabels left/right BY NAME; an unpaired landmark would
    # be mirrored in pixels and NOT in labels, training the two sides to average.
    # Checked against the keypoint names the MODEL emits, before step 0.
    assert_lr_swap_covers(names, required=names)
    # ONE place for the wing multiplier: `train.wing_kp_mult`. `mvq_loss` takes the
    # already-built (K,) vector, so there is no second knob inside `LossWeights`.
    kp_weight = wing_kp_weight(names, tcfg.wing_kp_mult)
    n_wing = int((np.asarray(kp_weight) != 1.0).sum())
    if tcfg.wing_kp_mult != 1.0:
        print(f"[mvq] wing keypoint/visibility weight x{tcfg.wing_kp_mult} on {n_wing} of {len(names)} "
              f"landmarks: {[n for n, w in zip(names, kp_weight) if w != 1.0]}", flush=True)
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
    # (a share check writes nothing: it is a 200-step probe, and leaving its
    # `total_steps` behind in a real run dir would misdescribe the run.)
    if not n_share:
        run_dir = os.path.dirname(os.path.abspath(out_dir))
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "mvq_run.json"), "w") as f:
            json.dump({"model": dataclasses.asdict(mcfg), "train": dataclasses.asdict(tcfg),
                      "loss": dataclasses.asdict(weights), "aug": dataclasses.asdict(aug),
                      "val": None, "keypoint_names": names}, f, indent=1)

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

    step_fns = {T: make_train_step(aug, lr_swap, part_of_k, weights, tcfg.ema, kp_weight)
               for T in tcfg.window_lengths}
    streams = {}
    counters = {}
    for T, ds in train_sets.items():
        w, mass = _mix_weights(ds, tcfg)
        is_f = np.array([ds.is_female(i) for i in range(len(ds))], bool)
        mf, mm = float(w[is_f].sum()), float(w[~is_f].sum())
        # NOTE the aggregate is over ALL roots, so it only equals `female_host_target`
        # when every root could reach it: a one-sided root (an all-female single-fly
        # export, the all-"other" negatives) is left unchanged by design and pulls the
        # aggregate off the target by its own mass. The per-root lines above say which.
        knob = (f"female_host_target={tcfg.female_host_target}" if tcfg.female_host_target is not None
                else f"female_host_weight={tcfg.female_host_weight}")
        print(f"[mvq] T={T} sampler: {int(is_f.sum())}/{len(ds)} female-host windows, "
              f"{knob} -> weight mass female {mf:.4f} "
              f"male/other {mm:.4f} (F/M {mf / max(mm, 1e-12):.3f})", flush=True)
        subs = list(getattr(ds, "datasets", [ds]))
        sub_names = list(getattr(ds, "names", ["real"]))
        if len(subs) > 1:
            print(f"[mvq] T={T} sampler mix: "
                  + "  ".join(f"{nm} {len(d)} windows -> mass {mass[nm]:.4f}"
                              for nm, d in zip(sub_names, subs)), flush=True)
            for nm, d in zip(sub_names, subs):
                # The pseudo export's manifest carries `balance.female_host_weight`
                # (n_male/n_female) as INFORMATION: `_mix_weights` already ran
                # `_balanced_weights` inside this root, which restores the 0.5
                # host-sex ratio, so applying the manifest number again would
                # over-sample female hosts by that factor.
                bal = (getattr(d, "manifest_root", None) or {}).get("balance") or {}
                if bal.get("female_host_weight") is not None:
                    print(f"[mvq]   {nm}: manifest balance.female_host_weight="
                          f"{bal['female_host_weight']} (INFORMATION -- not applied again; the "
                          f"per-root balance above already restores the host-sex ratio)", flush=True)
                # `sample_weight` comes from each ROOT's manifest, not from tcfg, so a
                # mislabelled pseudo export would train at a weight nobody configured.
                # The negatives root is exempt: its weight is FORCED to 1.0 above
                # (`_ForceSampleWeight`), and the override was already announced.
                mw = float(np.mean([d.weight(k) for k in range(len(d))]))
                flag = ""
                if nm in ("pseudo", "singlefly") and abs(mw - float(tcfg.pseudo_weight)) > 1e-6:
                    flag = f"  <-- WARNING: expected train.pseudo_weight={tcfg.pseudo_weight:g}"
                print(f"[mvq]   {nm}: source={d.source(0)!r} role={d.role(0)!r} "
                      f"mean sample_weight={mw:.4f}{flag}", flush=True)
            ds = _MixCounter(ds)
            counters[T] = ds
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
    # Realised host-sex ratio and root mix of what the sampler ACTUALLY draws,
    # over the first `_RATIO_BATCHES` batches (or the whole run, if shorter --
    # a 20-step smoke run must print it too, or the plumbing is unverified).
    # `female_host_weight` and the mass table above set a weight mass; these are
    # the numbers that say whether the draws came out where they were aimed
    # (`is_female` is the HOST's sex and survives copy-paste, which only adds a
    # donor). NEGATIVES have no host sex at all, so the ratio is reported both
    # over everything drawn and over the non-negative windows alone.
    n_ratio = max(1, min(_RATIO_BATCHES, tcfg.total_steps - start))
    seen_f = seen_n = seen_neg = 0
    share_sums = collections.defaultdict(float)
    for i in range(start, tcfg.total_steps):
        T = Ts[i % len(Ts)]
        batch = dict(zip(WINDOW_KEYS, next(streams[T])))
        if i - start < n_ratio:
            is_f = np.asarray(batch["is_female"]); neg = np.asarray(batch["is_negative"])
            seen_f += int(is_f.sum()); seen_n += int(is_f.shape[0]); seen_neg += int(neg.sum())
            if i - start == n_ratio - 1:
                pos = max(seen_n - seen_neg, 1)
                print(f"[mvq] realised host-sex ratio over the first {n_ratio} batches: "
                      f"female {seen_f}/{seen_n} = {seen_f / max(seen_n, 1):.3f} "
                      f"(of the {pos} non-negative windows: {seen_f / pos:.3f}); "
                      f"negatives {seen_neg}/{seen_n} = {seen_neg / max(seen_n, 1):.3f}", flush=True)
                for T_, cnt in sorted(counters.items()):
                    c, n_c = cnt.realised()
                    print(f"[mvq] T={T_} realised mix over {n_c} windows drawn: "
                          + " ".join(f"{nm}={c.get(nm, 0) / n_c:.3f}" for nm in cnt.names), flush=True)
        pp = tcfg.prompt_p_start + (tcfg.prompt_p_end - tcfg.prompt_p_start) * min(i / max(tcfg.prompt_anneal_steps, 1), 1.0)
        loss, metrics, ema = step_fns[T](model, opt, ema, jax.random.fold_in(key, i), batch, jnp.float32(pp))
        loss = float(loss); ema_updates += 1
        if n_share:
            for k_ in _SHARE_KEYS:
                if k_ in metrics:
                    share_sums[k_] += float(metrics[k_])
        if (i + 1) % tcfg.log_every == 0:
            ms = " ".join(f"{k}={float(v):.4f}" for k, v in metrics.items() if k != "total")
            print(f"step {i+1}/{tcfg.total_steps} T={T} loss {loss:.4f} {ms} ({time.time()-t0:.0f}s)", flush=True)
        if not n_share and ((i + 1) % tcfg.eval_every == 0 or i + 1 == tcfg.total_steps):
            em = _with_ema(model, ema, tcfg.ema, ema_updates)
            val = evaluate(em, val_ds, tcfg.batch_size, cohorts=cohorts, part_of_k=part_of_k,
                           weights=weights, mesh=mesh, num_workers=tcfg.num_workers,
                           kp_weight=kp_weight)
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
    if n_share:
        sh = _loss_shares(share_sums, n_share, weights)
        _print_loss_shares(sh)
        return {"loss_shares": sh, "final_loss": loss, "steps": n_share}
    if mngr is not None:
        _save_step(mngr, tcfg.total_steps, model, opt, ema, ema_updates); mngr.wait_until_finished()
    em = _with_ema(model, ema, tcfg.ema, ema_updates)
    val = evaluate(em, val_ds, tcfg.batch_size, cohorts=cohorts, part_of_k=part_of_k,
                   weights=weights, mesh=mesh, num_workers=tcfg.num_workers, kp_weight=kp_weight)
    os.makedirs(out_dir, exist_ok=True)
    ckptr = ocp.StandardCheckpointer(); ckptr.save(out_dir, nnx.split(em)[1], force=True); ckptr.wait_until_finished()
    with open(os.path.join(out_dir, "mvq_run.json"), "w") as f:
        json.dump({"model": dataclasses.asdict(mcfg), "train": dataclasses.asdict(tcfg),
                  "loss": dataclasses.asdict(weights), "aug": dataclasses.asdict(aug), "val": val,
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
