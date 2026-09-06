"""Loss for the multi-view query lifter (spec 2026-09-03 §5). All terms in px."""
from __future__ import annotations

import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np

from jarvis_jax.models.mvq.geometry import project_local
from jarvis_jax.train.matching import assign_slots, slot_ignore, SEX_FEMALE, N_SLOTS

WING_RE = re.compile(r"^Wing")


def wing_kp_weight(kp_names, mult=2.0):
    """(K,) per-keypoint loss weight: `mult` on every wing landmark, 1.0
    elsewhere. BY NAME -- the wing landmarks are the weakest in the model
    (spec §1) and their indices differ between the detector and model orders."""
    return np.asarray([mult if WING_RE.match(str(n)) else 1.0 for n in kp_names], np.float32)


@dataclasses.dataclass(frozen=True)
class LossWeights:
    reproj: float = 1.0
    l3d: float = 0.5
    uv2d: float = 0.5
    vis: float = 0.1
    # NOT an outer weight on term 5 -- it is lambda INSIDE that term's own
    # formula (c*err - lambda*log(c), see mvq_loss's term-5 comment below):
    # the confidence-calibration term always contributes at weight 1, this
    # just trades off how hard `c` is penalised for saturating toward 0.
    conf: float = 0.2
    exist: float = 1.0
    sex: float = 0.5
    rep: float = 0.5
    # P3b: cross-fly repulsion. 0 = OFF (the P3a default, so every earlier run's
    # loss is unchanged). Penalises a slot's predicted 3D keypoints for sitting
    # nearer the OTHER labelled fly's GT centroid than their own -- the training
    # -side answer to a slot absorbing the other animal's head/T1 legs/wing base
    # on contact frames (docs/benchmark/2026-09-mvq/p3b-notes.md).
    other_fly_repulsion: float = 0.0
    pass1: float = 0.5
    aux: float = 0.3
    huber_px: float = 8.0
    rep_px: float = 20.0
    rep_units: float = 2.5
    # v2 T=2 (spec §4): identity-persistence hinge on a slot's per-frame centroid
    # displacement beyond what the GT fly actually moved. 0 = OFF (T=1 batches,
    # and every P3a/P3b run's loss, are unchanged).
    persist: float = 0.5
    persist_margin_units: float = 2.0
    # per-keypoint loss weight multiplier applied to wing landmarks (see
    # `wing_kp_weight`); this is only the DEFAULT multiplier baked for callers
    # that build the weight vector from it -- `mvq_loss` itself takes the
    # already-built (K,) vector as `kp_weight`.
    wing_kp_mult: float = 1.0


def _huber(d, delta):
    a = jnp.abs(d)
    return jnp.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


def _mmean(x, m, sw=None):
    """Masked mean; `sw` (B,) is a PER-SAMPLE weight folded into the mask, so a
    pseudo-label sample (0.3) contributes 0.3 of a real one's entries."""
    m = jnp.broadcast_to(m.astype(x.dtype), x.shape)
    if sw is not None:
        m = m * sw.reshape((sw.shape[0],) + (1,) * (x.ndim - 1)).astype(x.dtype)
    return (x * m).sum() / jnp.maximum(m.sum(), 1.0)


def _reproject(xyz, M, t_local):
    """xyz (B,N,T,K,3) -> (B,N,T,C,K,2)."""
    uv = jax.vmap(lambda X, Mb, tl: jax.vmap(lambda Xt, tlt: project_local(Xt, Mb, tlt), in_axes=(1, 0), out_axes=1)(X, tl))(
        xyz, M, t_local)                                                        # (B,N,T,K,C,2)
    return jnp.moveaxis(uv, 4, 3)


def _gather_inst(x, assign):
    """x (B,I,...) , assign (B,F) -> (B,F,...) (index clamped; invalid flies masked by caller)."""
    idx = jnp.clip(assign, 0, x.shape[1] - 1)
    return jnp.take_along_axis(x, idx.reshape(idx.shape + (1,) * (x.ndim - 2)), axis=1)


def _geo_terms(pred, batch, w, valid_f, sw=None, kp_weight=None):
    """Terms 1-3 for a readout `pred` already gathered per fly: dict with xyz (B,F,T,K,3), uv (B,F,T,C,K,2).
    `kp_weight` (K,) reweights every term on the keypoint axis BEFORE the mean
    (wing landmarks at 2x by default, spec §4); `sw` (B,) is the per-sample weight."""
    s = batch["px_scale"][:, None, None, None]
    uv_re = _reproject(pred["xyz"], batch["M"], batch["t_local"])
    e2 = _huber(uv_re - batch["kp2d"], w.huber_px).mean(-1)                    # (B,F,T,C,K)
    if kp_weight is not None:
        e2 = e2 * kp_weight[None, None, None, None, :]
    m2 = batch["vis2d"] & valid_f[:, :, None, None, None]
    reproj = _mmean(e2, m2, sw)
    m3 = batch["has3d"] & valid_f[:, :, None, None]
    l3d_term = jnp.abs(pred["xyz"] - batch["kp3d_local"]).mean(-1)             # (B,F,T,K)
    if kp_weight is not None:
        l3d_term = l3d_term * kp_weight[None, None, None, :]
    l3d = _mmean(s * l3d_term, m3, sw)
    uv2d_term = _huber(pred["uv"] - batch["kp2d"], w.huber_px).mean(-1)        # (B,F,T,C,K)
    if kp_weight is not None:
        uv2d_term = uv2d_term * kp_weight[None, None, None, None, :]
    uv2d = _mmean(uv2d_term, m2, sw)
    per_kp = (e2 * m2).sum(3) / jnp.maximum(m2.sum(3), 1.0)                     # (B,F,T,K) mean over views
    return reproj, l3d, uv2d, per_kp, (m2.sum(3) > 0)


def mvq_loss(out, batch, w: LossWeights, part_of_k, kp_weight=None):
    B, I = out["xyz"].shape[:2]; F = batch["kp3d_local"].shape[1]; T = batch["kp3d_local"].shape[2]
    fv = batch["fly_valid"]
    sw = batch.get("sample_weight", jnp.ones((B,), jnp.float32))
    if I != N_SLOTS:
        raise ValueError(f"mvq_loss expects n_instances == {N_SLOTS} (slots prompted/female/male/other), got {I}")
    # ---------------- label-driven slot assignment (P3a §3): no prediction enters
    has_f = batch["has3d"].astype(jnp.float32)                                          # (B,F,T,K)
    cen = (batch["kp3d_local"] * has_f[..., None]).sum((2, 3)) / jnp.maximum(has_f.sum((2, 3)), 1.0)[..., None]
    # v2 T=2 (spec §4): the slot assignment is made on FRAME 0 ONLY and reused on
    # frame 1 -- `cen` above (all frames) still feeds term 7b, but `dist` (what
    # decides the slot) must not let frame 1 outvote frame 0.
    has0 = batch["has3d"][:, :, 0].astype(jnp.float32)                                   # (B,F,K)
    cen0 = (batch["kp3d_local"][:, :, 0] * has0[..., None]).sum(2) / jnp.maximum(has0.sum(2), 1.0)[..., None]
    dist = jnp.linalg.norm(cen0, axis=-1)                                                # (B,F) frame 0 decides the slot
    assign, slot_target = assign_slots(batch["fly_sex"], fv, batch["prompt_on"], dist, I)
    ignore = slot_ignore(batch["unlabelled_sex"], I)                                     # (B,I)
    # A slot that HOLDS a labelled fly certainly exists, so it is supervised even when
    # the unlabelled animal's sex points at the same slot. Without this, a window with a
    # female host in slot 1 and an unlabelled female present had slot 1's existence
    # ignored -- the one slot whose answer is known for certain got no gradient.
    ignore = ignore & ~slot_target
    inst_matched = slot_target
    fv_eff = fv & (assign >= 0)      # a dropped fly (assign=-1) is treated like an unlabelled one
    # ---------------- gather per fly
    g = lambda d: {k: _gather_inst(d[k], assign) for k in ("xyz", "uv", "conf_logit", "vis_logit")}
    pf = g(out)
    reproj, l3d, uv2d, per_kp, per_kp_m = _geo_terms(pf, batch, w, fv_eff, sw, kp_weight)
    # term 4 visibility BCE (per view query, target v flag; absent cams masked)
    vis_t = batch["vis2d"].astype(jnp.float32)
    bce_v = jnp.maximum(pf["vis_logit"], 0) - pf["vis_logit"] * vis_t + jnp.log1p(jnp.exp(-jnp.abs(pf["vis_logit"])))
    if kp_weight is not None:
        bce_v = bce_v * kp_weight[None, None, None, None, :]
    vis = _mmean(bce_v, batch["cam_valid"][:, None, :, :, None] & fv_eff[:, :, None, None, None], sw)
    # term 5 confidence (D4RT): c*err - lambda*log c (log_sigmoid keeps the gradient on a saturated head)
    log_c = jax.nn.log_sigmoid(pf["conf_logit"])
    c = jnp.exp(log_c)
    conf = _mmean(c * per_kp - w.conf * log_c, per_kp_m & fv_eff[:, :, None, None], sw)
    # term 6 existence -- masked mean over slots NOT ignored (an unlabelled fly could occupy an ignored slot)
    tgt = inst_matched.astype(jnp.float32)
    bce_e = jnp.maximum(out["exist_logit"], 0) - out["exist_logit"] * tgt + jnp.log1p(jnp.exp(-jnp.abs(out["exist_logit"])))
    keep = ~ignore
    exist = _mmean(bce_e, keep, sw)
    exist_acc = _mmean(((out["exist_logit"] > 0) == inst_matched).astype(jnp.float32), keep, sw)
    # term 8 sex (P3a §4): BCE female=1 on assigned slots whose fly has a known sex
    oh = ((assign[:, :, None] == jnp.arange(I)[None, None, :]) & fv[:, :, None]).astype(jnp.int32)   # (B,F,I)
    slot_sex = (oh * (batch["fly_sex"].astype(jnp.int32) + 1)[:, :, None]).sum(1) - 1               # (B,I) -1 = none/unknown
    m_sex = slot_sex >= 0
    sex_t = (slot_sex == SEX_FEMALE).astype(jnp.float32)
    bce_s = jnp.maximum(out["sex_logit"], 0) - out["sex_logit"] * sex_t + jnp.log1p(jnp.exp(-jnp.abs(out["sex_logit"])))
    sex = _mmean(bce_s, m_sex, sw)
    sex_acc = _mmean(((out["sex_logit"] > 0) == (slot_sex == SEX_FEMALE)).astype(jnp.float32), m_sex, sw)
    # term 7 repulsion against OTHER flies' same-part labels
    pok = jnp.asarray(np.asarray(part_of_k))
    same_part = (pok[:, None] == pok[None, :]).astype(jnp.float32)              # (K,K')
    same_part = same_part / jnp.maximum(same_part.sum(1, keepdims=True), 1.0)     # per-row average, not sum, over a part's members
    rep = jnp.zeros(())
    if F > 1:
        for f in range(F):
            for o in range(F):
                if o == f:
                    continue
                ok = fv_eff[:, f] & fv_eff[:, o]
                d2 = jnp.linalg.norm(pf["uv"][:, f][..., :, None, :] - batch["kp2d"][:, o][..., None, :, :], axis=-1)  # (B,T,C,K,K')
                h2 = jax.nn.relu(w.rep_px - d2) * same_part * batch["vis2d"][:, o][..., None, :]
                d3 = jnp.linalg.norm(pf["xyz"][:, f][..., :, None, :] - batch["kp3d_local"][:, o][..., None, :, :], axis=-1)  # (B,T,K,K')
                h3 = batch["px_scale"][:, None, None, None] * jax.nn.relu(w.rep_units - d3) * same_part * batch["has3d"][:, o][..., None, :]
                rep = rep + (h2.sum((1, 2, 3, 4)) * ok).sum() / jnp.maximum(ok.sum() * h2.shape[1] * h2.shape[2] * h2.shape[3], 1.0) \
                          + (h3.sum((1, 2, 3)) * ok).sum() / jnp.maximum(ok.sum() * h3.shape[1] * h3.shape[2], 1.0)
    # term 7b cross-fly repulsion (P3b) -- a CENTROID-level version of term 7.
    # Term 7 pushes a predicted keypoint off the other fly's same-part LABEL, so it
    # can only see the keypoints that fly actually has labels for and only within
    # rep_px/rep_units. This one asks the coarser question the contact failure is
    # about: is this point on the right ANIMAL at all? `cen` (computed above for the
    # slot assignment) is each labelled fly's GT 3D centroid, so
    # relu(d_own - d_other) is exactly zero as soon as the point sits on its own
    # side of the two centroids' midplane and grows linearly (in units, 0.1 mm)
    # once it crosses. Only windows with TWO labelled flies contribute (real two-fly
    # windows and copy-paste composites alike -- `fv_eff` covers both), and only
    # flies that HAVE a 3D centroid to be measured against.
    other_rep = jnp.zeros(())
    if F > 1 and w.other_fly_repulsion > 0:
        has_cen = has_f.sum((2, 3)) > 0                                          # (B,F)
        n_pairs = 0
        for f in range(F):
            for o in range(F):
                if o == f:
                    continue
                ok = (fv_eff[:, f] & fv_eff[:, o] & has_cen[:, f] & has_cen[:, o])[:, None, None]
                d_own = jnp.linalg.norm(pf["xyz"][:, f] - cen[:, f][:, None, None, :], axis=-1)   # (B,T,K)
                d_oth = jnp.linalg.norm(pf["xyz"][:, f] - cen[:, o][:, None, None, :], axis=-1)
                other_rep = other_rep + _mmean(jax.nn.relu(d_own - d_oth), ok)
                n_pairs += 1
        other_rep = other_rep / max(n_pairs, 1)
    total = (w.reproj * reproj + w.l3d * l3d + w.uv2d * uv2d + w.vis * vis + conf
             + w.exist * exist + w.sex * sex + w.rep * rep + w.other_fly_repulsion * other_rep)
    # term 9 identity persistence (spec §4): the slot assignment is made on frame 0
    # and reused on frame 1, so a slot that swaps animals between the two frames shows
    # up as a per-slot centroid displacement far beyond what the GT fly actually moved.
    persist = jnp.zeros(())
    if T > 1 and w.persist > 0:
        pc = pf["xyz"].mean(3)                                                   # (B,F,T,3) predicted centroid
        d_pred = jnp.linalg.norm(pc[:, :, 1] - pc[:, :, 0], axis=-1)             # (B,F)
        hasT = batch["has3d"].astype(jnp.float32)
        gc = ((batch["kp3d_local"] * hasT[..., None]).sum(3)
              / jnp.maximum(hasT.sum(3), 1.0)[..., None])                        # (B,F,T,3)
        d_gt = jnp.linalg.norm(gc[:, :, 1] - gc[:, :, 0], axis=-1)
        ok = fv_eff & (hasT[:, :, 0].sum(-1) > 0) & (hasT[:, :, 1].sum(-1) > 0)
        persist = _mmean(jax.nn.relu(d_pred - d_gt - w.persist_margin_units), ok, sw)
    total = total + w.persist * persist
    # deep supervision
    if out.get("aux_pass1") is not None:
        r1, l1, u1, _, _ = _geo_terms(g(out["aux_pass1"]), batch, w, fv_eff, sw, kp_weight)
        total = total + w.pass1 * (w.reproj * r1 + w.l3d * l1 + w.uv2d * u1)
    for a in out.get("aux_layers", []):                 # intermediate 3D-only readouts: terms 1-2
        xyz_f = _gather_inst(a["xyz"], assign)
        uv_re = _reproject(xyz_f, batch["M"], batch["t_local"])
        r_ = _mmean(_huber(uv_re - batch["kp2d"], w.huber_px).mean(-1), batch["vis2d"] & fv_eff[:, :, None, None, None], sw)
        l_ = _mmean(batch["px_scale"][:, None, None, None] * jnp.abs(xyz_f - batch["kp3d_local"]).mean(-1),
                    batch["has3d"] & fv_eff[:, :, None, None], sw)
        total = total + w.aux * (w.reproj * r_ + w.l3d * l_)
    mp = _mmean(jnp.linalg.norm(pf["xyz"] - batch["kp3d_local"], axis=-1), batch["has3d"] & fv_eff[:, :, None, None], sw)
    pf_reproj = _reproject(pf["xyz"], batch["M"], batch["t_local"])
    m2_full = batch["vis2d"] & fv_eff[:, :, None, None, None]
    px = _mmean(jnp.linalg.norm(pf_reproj - batch["kp2d"], axis=-1), m2_full, sw)
    # raw (non-Huber) L2 metrics in px, reported alongside the Huber training terms:
    # uv2d_px = 2D-head prediction vs GT label; head_vs_reproj_px = 2D-head vs the
    # SAME instance's reprojected 3D estimate (head/3D-branch self-consistency).
    uv2d_px = _mmean(jnp.linalg.norm(pf["uv"] - batch["kp2d"], axis=-1), m2_full, sw)
    head_vs_reproj_px = _mmean(jnp.linalg.norm(pf["uv"] - pf_reproj, axis=-1), m2_full, sw)
    metrics = {"total": total, "reproj": reproj, "l3d": l3d, "uv2d": uv2d, "vis": vis, "conf": conf,
               "exist": exist, "rep": rep, "other_rep": other_rep, "exist_acc": exist_acc,
               "sex": sex, "sex_acc": sex_acc, "persist": persist,
               "n_negative": batch.get("is_negative", jnp.zeros((B,), bool)).sum(),
               "match_reproj_px": px, "mpjpe3d_units": mp, "uv2d_px": uv2d_px, "head_vs_reproj_px": head_vs_reproj_px}
    return total, metrics
