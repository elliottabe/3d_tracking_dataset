"""Phase-1 per-segment scale optimization (adjustabodies-style, ported to our
fly stac-mjx stack).

The M-step: given fixed joint angles `qpos` and target keypoints, optimize a
small vector of per-segment scale factors (one per L/R mirror group) via Adam on
MJX differentiable forward kinematics, minimizing the site<->keypoint distance
plus (a) a regularizer pulling scales to 1 and (b) a soft target pulling
measured segments toward their data/model bone-length ratio (Phase-0).

Kinematics of a segment scale `s` (mirrors stac_mjx.rescale.rescale_per_segment):
  - non-wing: multiply `body_pos[length_body]` by s  (moves the distal joint out
    -> lengthens the segment; the keypoint site on that body rides along).
  - wing:     multiply `site_pos[sites on scale_sites_on_body]` by s  (the tip is
    a site offset on the fixed-hinge wing body).

The Q-step (IK) is our existing jaxls STAC (run with SEGMENT_SCALES set), so the
frozen solver is untouched; this module only adds the differentiable scale fit.
"""
from __future__ import annotations

import numpy as np


def build_scale_index(mj_model, segment_map, kp_names):
    """Static index arrays mapping a per-mirror-group scale vector onto MJX model
    fields + the tracking-site ids for each keypoint.

    Returns dict with:
      groups            : list[str]         unique mirror-group keys (param order)
      seg_group         : (S,) int          group index per segment
      length_body_id    : (S,) int          body id whose body_pos scales (or -1)
      wing_site_ids     : list[np.ndarray]  site ids to scale per segment (wings)
      site_id_for_kp    : (K,) int          tracking-site id per kp_name (or -1)
      seg_names         : list[str]
    """
    import mujoco
    def bid(name):
        return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name) if name else -1
    def sid(name):
        return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)

    groups, gidx = [], {}
    def group_of(seg):
        key = seg.get("mirror") or seg["name"]
        if key not in gidx:
            gidx[key] = len(groups); groups.append(key)
        return gidx[key]

    seg_group, length_body_id, wing_site_ids, seg_names = [], [], [], []
    for seg in segment_map:
        seg_group.append(group_of(seg))
        length_body_id.append(bid(seg.get("length_body") or ""))
        seg_names.append(seg["name"])
        sb = seg.get("scale_sites_on_body") or ""
        if sb:
            bodyid = bid(sb)
            ids = [i for i in range(mj_model.nsite) if mj_model.site_bodyid[i] == bodyid]
            wing_site_ids.append(np.array(ids, dtype=int))
        else:
            wing_site_ids.append(np.array([], dtype=int))

    site_id_for_kp = np.array([sid(f"tracking[{k}]") for k in kp_names], dtype=int)
    return dict(groups=groups, seg_group=np.array(seg_group, int),
                length_body_id=np.array(length_body_id, int),
                wing_site_ids=wing_site_ids, site_id_for_kp=site_id_for_kp,
                seg_names=seg_names)


def optimize_segment_scales(mj_model, segment_map, kp_names, qpos, kp_model_units,
                            *, phase0_targets=None, lam_reg=0.001, lam_target=0.05,
                            lr=0.003, iters=400, clamp=(0.7, 1.4), verbose=True):
    """M-step. Returns dict{segment_name: scale}. `qpos` (T,nq), `kp_model_units`
    (T,K,3) in the SAME model units mjx FK produces (NaNs allowed = missing).
    `phase0_targets` optional dict{group_or_seg: target_abs_scale} soft prior."""
    import jax, jax.numpy as jnp
    import mujoco.mjx as mjx
    import optax

    idx = build_scale_index(mj_model, segment_map, kp_names)
    G = len(idx["groups"])
    seg_group = jnp.asarray(idx["seg_group"])
    length_body_id = idx["length_body_id"]
    site_id_for_kp = idx["site_id_for_kp"]

    m0 = mjx.put_model(mj_model)
    base_body_pos = m0.body_pos
    base_site_pos = m0.site_pos
    d0 = mjx.make_data(m0)

    # per-segment -> body/site scatter (static)
    lb_mask = length_body_id >= 0
    lb_ids = jnp.asarray(length_body_id[lb_mask])
    lb_seg = jnp.asarray(np.nonzero(lb_mask)[0])
    wing = [(s, jnp.asarray(w)) for s, w in enumerate(idx["wing_site_ids"]) if w.size]

    kp = jnp.asarray(kp_model_units, jnp.float32)          # (T,K,3)
    qpos = jnp.asarray(qpos, jnp.float32)                  # (T,nq)
    valid = jnp.isfinite(kp).all(-1)                       # (T,K)
    kp_id = jnp.asarray(site_id_for_kp)
    kp_ok_col = kp_id >= 0                                 # keypoints with a site

    # soft target vector (per group), default 1.0
    tgt = np.ones(G, np.float32)
    if phase0_targets:
        for gi, g in enumerate(idx["groups"]):
            if g in phase0_targets:
                tgt[gi] = float(phase0_targets[g])
    tgt = jnp.asarray(tgt)

    def scaled_model(gp):                                  # gp: (G,) group scales
        seg_scale = gp[seg_group]                          # (S,)
        body_scale = jnp.ones(base_body_pos.shape[0]).at[lb_ids].set(seg_scale[lb_seg])
        body_pos = base_body_pos * body_scale[:, None]
        site_scale = jnp.ones(base_site_pos.shape[0])
        for s, wids in wing:
            site_scale = site_scale.at[wids].set(gp[seg_group[s]])
        site_pos = base_site_pos * site_scale[:, None]
        return m0.replace(body_pos=body_pos, site_pos=site_pos)

    def fk_sites(gp):                                      # (T,K,3) model-space
        m = scaled_model(gp)
        def one(q):
            d = mjx.kinematics(m, d0.replace(qpos=q))
            return d.site_xpos
        allsites = jax.vmap(one)(qpos)                     # (T,nsite,3)
        return allsites[:, jnp.where(kp_ok_col, kp_id, 0)]  # (T,K,3)

    def loss(raw):
        gp = jnp.clip(raw, clamp[0], clamp[1])
        pred = fk_sites(gp)
        d = jnp.linalg.norm(pred - kp, axis=-1)            # (T,K)
        m = valid & kp_ok_col[None, :]
        ik = jnp.sum(jnp.where(m, d, 0.0)) / jnp.maximum(jnp.sum(m), 1)
        reg = lam_reg * jnp.sum((gp - 1.0) ** 2)
        soft = lam_target * jnp.sum((gp - tgt) ** 2)
        return ik + reg + soft, ik

    raw = jnp.ones(G)
    opt = optax.adam(lr)
    st = opt.init(raw)
    gl = jax.jit(jax.value_and_grad(lambda r: loss(r)[0]))
    ik0 = float(loss(raw)[1])
    for it in range(iters):
        v, g = gl(raw)
        upd, st = opt.update(g, st)
        raw = optax.apply_updates(raw, upd)
    gp = np.clip(np.asarray(raw), clamp[0], clamp[1])
    ik1 = float(loss(jnp.asarray(gp))[1])
    if verbose:
        print(f"[mstep] IK site-error {ik0:.4f} -> {ik1:.4f} model-units "
              f"({G} group scales, {iters} iters)")
    # expand groups -> per-segment dict
    out = {}
    for si, seg in enumerate(segment_map):
        out[seg["name"]] = float(gp[idx["seg_group"][si]])
    return out, dict(ik_before=ik0, ik_after=ik1, group_scales=dict(zip(idx["groups"], gp.tolist())))
