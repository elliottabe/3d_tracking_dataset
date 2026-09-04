"""Loss for the multi-view query lifter (spec 2026-09-03 §5). All terms in px."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from jarvis_jax.models.mvq.geometry import project_local
from jarvis_jax.train.matching import match


@dataclasses.dataclass(frozen=True)
class LossWeights:
    reproj: float = 1.0
    l3d: float = 0.5
    uv2d: float = 0.5
    vis: float = 0.1
    conf: float = 0.2
    exist: float = 1.0
    rep: float = 0.5
    pass1: float = 0.5
    aux: float = 0.3
    huber_px: float = 8.0
    rep_px: float = 20.0
    rep_units: float = 2.5


def _huber(d, delta):
    a = jnp.abs(d)
    return jnp.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


def _mmean(x, m):
    m = jnp.broadcast_to(m.astype(x.dtype), x.shape)
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


def _geo_terms(pred, batch, w, valid_f):
    """Terms 1-3 for a readout `pred` already gathered per fly: dict with xyz (B,F,T,K,3), uv (B,F,T,C,K,2)."""
    s = batch["px_scale"][:, None, None, None]
    uv_re = _reproject(pred["xyz"], batch["M"], batch["t_local"])
    e2 = _huber(uv_re - batch["kp2d"], w.huber_px).mean(-1)                    # (B,F,T,C,K)
    m2 = batch["vis2d"] & valid_f[:, :, None, None, None]
    reproj = _mmean(e2, m2)
    m3 = batch["has3d"] & valid_f[:, :, None, None]
    l3d = _mmean(s * jnp.abs(pred["xyz"] - batch["kp3d_local"]).mean(-1), m3)
    uv2d = _mmean(_huber(pred["uv"] - batch["kp2d"], w.huber_px).mean(-1), m2)
    per_kp = (e2 * m2).sum(3) / jnp.maximum(m2.sum(3), 1.0)                     # (B,F,T,K) mean over views
    return reproj, l3d, uv2d, per_kp, (m2.sum(3) > 0)


def mvq_loss(out, batch, w: LossWeights, part_of_k):
    B, I = out["xyz"].shape[:2]; F = batch["kp3d_local"].shape[1]
    fv = batch["fly_valid"]
    # ---------------- matching on the final pass (no gradient)
    xyz_sg = jax.lax.stop_gradient(out["xyz"])
    uv_all = _reproject(xyz_sg, batch["M"], batch["t_local"])                   # (B,I,T,C,K,2)
    e2 = _huber(uv_all[:, :, None] - batch["kp2d"][:, None], w.huber_px).mean(-1)   # (B,I,F,T,C,K)
    m2 = batch["vis2d"][:, None]
    c2 = (e2 * m2).sum((3, 4, 5)) / jnp.maximum(m2.sum((3, 4, 5)), 1.0)
    e3 = batch["px_scale"][:, None, None, None, None] * jnp.abs(xyz_sg[:, :, None] - batch["kp3d_local"][:, None]).mean(-1)
    m3 = batch["has3d"][:, None]
    c3 = (e3 * m3).sum((3, 4)) / jnp.maximum(m3.sum((3, 4)), 1.0)
    cost = jnp.where(fv[:, None, :], c2 + c3, 1e6)
    assign, inst_matched = match(cost, fv, batch["prompt_on"])
    # ---------------- gather per fly
    g = lambda d: {k: _gather_inst(d[k], assign) for k in ("xyz", "uv", "conf_logit", "vis_logit")}
    pf = g(out)
    reproj, l3d, uv2d, per_kp, per_kp_m = _geo_terms(pf, batch, w, fv)
    # term 4 visibility BCE (per view query, target v flag; absent cams masked)
    vis_t = batch["vis2d"].astype(jnp.float32)
    bce_v = jnp.maximum(pf["vis_logit"], 0) - pf["vis_logit"] * vis_t + jnp.log1p(jnp.exp(-jnp.abs(pf["vis_logit"])))
    vis = _mmean(bce_v, batch["cam_valid"][:, None, :, :, None] & fv[:, :, None, None, None])
    # term 5 confidence (D4RT): c*err - lambda*log c (log_sigmoid keeps the gradient on a saturated head)
    log_c = jax.nn.log_sigmoid(pf["conf_logit"])
    c = jnp.exp(log_c)
    conf = _mmean(c * per_kp - w.conf * log_c, per_kp_m & fv[:, :, None, None])
    # term 6 existence
    tgt = inst_matched.astype(jnp.float32)
    bce_e = jnp.maximum(out["exist_logit"], 0) - out["exist_logit"] * tgt + jnp.log1p(jnp.exp(-jnp.abs(out["exist_logit"])))
    exist = bce_e.mean()
    exist_acc = ((out["exist_logit"] > 0) == inst_matched).astype(jnp.float32).mean()
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
                ok = fv[:, f] & fv[:, o]
                d2 = jnp.linalg.norm(pf["uv"][:, f][..., :, None, :] - batch["kp2d"][:, o][..., None, :, :], axis=-1)  # (B,T,C,K,K')
                h2 = jax.nn.relu(w.rep_px - d2) * same_part * batch["vis2d"][:, o][..., None, :]
                d3 = jnp.linalg.norm(pf["xyz"][:, f][..., :, None, :] - batch["kp3d_local"][:, o][..., None, :, :], axis=-1)  # (B,T,K,K')
                h3 = batch["px_scale"][:, None, None, None] * jax.nn.relu(w.rep_units - d3) * same_part * batch["has3d"][:, o][..., None, :]
                rep = rep + (h2.sum((1, 2, 3, 4)) * ok).sum() / jnp.maximum(ok.sum() * h2.shape[1] * h2.shape[2] * h2.shape[3], 1.0) \
                          + (h3.sum((1, 2, 3)) * ok).sum() / jnp.maximum(ok.sum() * h3.shape[1] * h3.shape[2], 1.0)
    total = (w.reproj * reproj + w.l3d * l3d + w.uv2d * uv2d + w.vis * vis + conf
             + w.exist * exist + w.rep * rep)
    # deep supervision
    if out.get("aux_pass1") is not None:
        r1, l1, u1, _, _ = _geo_terms(g(out["aux_pass1"]), batch, w, fv)
        total = total + w.pass1 * (w.reproj * r1 + w.l3d * l1 + w.uv2d * u1)
    for a in out.get("aux_layers", []):                 # intermediate 3D-only readouts: terms 1-2
        xyz_f = _gather_inst(a["xyz"], assign)
        uv_re = _reproject(xyz_f, batch["M"], batch["t_local"])
        r_ = _mmean(_huber(uv_re - batch["kp2d"], w.huber_px).mean(-1), batch["vis2d"] & fv[:, :, None, None, None])
        l_ = _mmean(batch["px_scale"][:, None, None, None] * jnp.abs(xyz_f - batch["kp3d_local"]).mean(-1),
                    batch["has3d"] & fv[:, :, None, None])
        total = total + w.aux * (w.reproj * r_ + w.l3d * l_)
    mp = _mmean(jnp.linalg.norm(pf["xyz"] - batch["kp3d_local"], axis=-1), batch["has3d"] & fv[:, :, None, None])
    px = _mmean(jnp.linalg.norm(_reproject(pf["xyz"], batch["M"], batch["t_local"]) - batch["kp2d"], axis=-1),
                batch["vis2d"] & fv[:, :, None, None, None])
    metrics = {"total": total, "reproj": reproj, "l3d": l3d, "uv2d": uv2d, "vis": vis, "conf": conf,
               "exist": exist, "rep": rep, "exist_acc": exist_acc, "match_reproj_px": px,
               "mpjpe3d_units": mp}
    return total, metrics
