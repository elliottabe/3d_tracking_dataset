"""Gradient-based (Adam) silhouette refinement of the APPENDAGE pose.

Replaces the jaxls Gauss-Newton/LM solve for the silhouette terms, which is
ill-suited to the highly nonlinear softmin-Chamfer / SDF-relu costs: jaxls uses
non-scale-invariant Levenberg damping (lambda*I), so the pixel-scale silhouette
Jacobian (|diag(JtJ)| ~ 1e6) drives every GN step to overshoot and be rejected
-- verified: a plain gradient step cuts the cost ~45% while LM rejects all steps.

This module instead runs first-order optimization (optax.adam) on the appendage
hinge DOFs only, starting from the STAC/keypoint fit `q_init`. Root + body +
non-appendage joints stay FROZEN at q_init (which is the keypoint-anchored pose;
the silhouette gradient is anatomically restricted to appendages anyway). The
objective is the SAME silhouette (coverage-Chamfer) + containment (SDF-relu)
terms as the jaxls path, plus temporal smoothness on the appendage joints, an
anchor to q_init, and a soft joint-limit barrier.

The objective and its autodiff gradient are identical to what the jaxls factors
computed (that gradient was verified correct); only the optimizer changes.
"""
from __future__ import annotations
import functools
import jax
import jax.numpy as jnp
import optax

from jarvis_jax.cse.silhouette_chamfer import chamfer_residual
from jarvis_jax.cse.silhouette_containment import containment_residual

_FREE_JOINT_NDOF = 7  # 3 translation + 4 quaternion, matches stac_core_jaxls


def _frame_cost(q_t, br_s, br_R, br_t, sdf_c, gsc_c, goff_c, pres_c, bnd_c, cfp_c,
                *, fk_repose, cov_idx, cont_idx, cam_Ms, cam_ts, conf_v,
                silhouette_weight, containment_weight, beta, huber_delta, margin,
                chunk_size):
    """Silhouette (coverage-Chamfer) + containment cost for ONE frame.

    verts are FK'd in model frame then mapped to mm via the per-frame bridge
    (s,R,t) BEFORE the affine (mm->px) projection, matching make_silhouette_cost.
    Cameras iterated with jax.lax.scan (FK once per frame).
    """
    verts_cont = fk_repose(q_t, 1.0, cont_idx)               # (Mc,3) model frame
    vmm_cont = br_s * (verts_cont @ br_R.T) + br_t           # -> mm
    verts_cov = fk_repose(q_t, 1.0, cov_idx)                 # (Mv,3)
    vmm_cov = br_s * (verts_cov @ br_R.T) + br_t

    def scan_body(carry, cam):
        M, tt, sdf_i, gs_i, go_i, pr_i, bnd_i, cf_i = cam
        proj_cont = vmm_cont @ M.T + tt                      # (Mc,2)
        r_cont = containment_residual(proj_cont, sdf_i, gs_i, go_i, conf_v,
                                      margin=margin, present=pr_i)          # (Mc,)
        proj_cov = vmm_cov @ M.T + tt                        # (Mv,2)
        r_cov = chamfer_residual(bnd_i, proj_cov, beta=beta,
                                 huber_delta=huber_delta, chunk_size=chunk_size) * cf_i
        c = (jnp.sum((containment_weight * r_cont) ** 2)
             + jnp.sum((silhouette_weight * r_cov) ** 2))
        return carry + c, None
    total, _ = jax.lax.scan(scan_body, 0.0,
                            (cam_Ms, cam_ts, sdf_c, gsc_c, goff_c, pres_c, bnd_c, cfp_c))
    return total


def refine_appendages_adam(
    q_init, *,
    fk_repose, cov_vert_indices, cont_vert_indices, cam_Ms, cam_ts,
    boundary_all, conf_p_all, sdf_all, grid_scale_all, grid_offset_all, present_all,
    conf_v, bridge_s_all, bridge_R_all, bridge_t_all,
    opt_mask, lb, ub,
    silhouette_weight=0.3, containment_weight=0.3, smooth_weight=0.005,
    anchor_weight=0.0, limit_weight=10.0, beta=8.0, huber_delta=0.0, margin=0.0,
    n_steps=300, lr=1e-2, chunk_size=32, return_history=False,
):
    """Adam-refine the appendage hinge DOFs of a trajectory toward the silhouette.

    Args:
        q_init: (T, nq) starting pose (STAC/keypoint fit); root+body kept frozen.
        opt_mask: (nq,) bool -- DOFs to optimize (appendage hinges). All other
            DOFs stay exactly at q_init.
        lb, ub: (nq,) joint limits; a soft relu barrier keeps opt DOFs inside,
            and the returned qpos is hard-clamped on the opt DOFs.
        anchor_weight: L2 pull of the optimized DOFs back to q_init (0 disables;
            the frozen root/body already anchor global placement).
        n_steps, lr: Adam schedule.
        return_history: if True, also return the (n_steps+1,) objective history.

    Returns:
        q_refined (T, nq) float32, or (q_refined, history) if return_history.
    """
    q0 = jnp.asarray(q_init)
    T, nq = q0.shape
    opt_mask = jnp.asarray(opt_mask).astype(bool)
    lb = jnp.asarray(lb); ub = jnp.asarray(ub)
    cov_idx = jnp.asarray(cov_vert_indices); cont_idx = jnp.asarray(cont_vert_indices)
    cam_Ms = jnp.asarray(cam_Ms); cam_ts = jnp.asarray(cam_ts); conf_v = jnp.asarray(conf_v)

    fc = functools.partial(
        _frame_cost, fk_repose=fk_repose, cov_idx=cov_idx, cont_idx=cont_idx,
        cam_Ms=cam_Ms, cam_ts=cam_ts, conf_v=conf_v,
        silhouette_weight=silhouette_weight, containment_weight=containment_weight,
        beta=beta, huber_delta=huber_delta, margin=margin, chunk_size=chunk_size)

    lb_row = jnp.where(opt_mask, lb, -jnp.inf)
    ub_row = jnp.where(opt_mask, ub, jnp.inf)

    def make_q(dq):
        return q0 + jnp.where(opt_mask[None, :], dq, 0.0)

    def objective(dq):
        q = make_q(dq)
        # silhouette + containment over frames
        sil = jnp.sum(jax.vmap(fc)(q, bridge_s_all, bridge_R_all, bridge_t_all,
                                   sdf_all, grid_scale_all, grid_offset_all, present_all,
                                   boundary_all, conf_p_all))
        # temporal smoothness on the optimized joints (frame-to-frame diff)
        if smooth_weight > 0.0 and T > 1:
            dj = (q[1:] - q[:-1]) * opt_mask[None, :]
            smooth = (smooth_weight ** 2) * jnp.sum(dj ** 2)
        else:
            smooth = 0.0
        # anchor the optimized DOFs to q_init
        anchor = (anchor_weight ** 2) * jnp.sum((dq * opt_mask[None, :]) ** 2)
        # soft joint-limit barrier on the optimized DOFs
        over = jax.nn.relu(q - ub_row[None, :]); under = jax.nn.relu(lb_row[None, :] - q)
        limit = (limit_weight ** 2) * jnp.sum(over ** 2 + under ** 2)
        return sil + smooth + anchor + limit

    opt = optax.adam(lr)
    dq0 = jnp.zeros_like(q0)
    state0 = opt.init(dq0)
    val_and_grad = jax.value_and_grad(objective)

    def step(carry, _):
        dq, st = carry
        v, g = val_and_grad(dq)
        g = g * opt_mask[None, :]                     # never move frozen DOFs
        updates, st = opt.update(g, st, dq)
        dq = optax.apply_updates(dq, updates)
        return (dq, st), v

    (dq_final, _), hist = jax.lax.scan(step, (dq0, state0), None, length=n_steps)
    v_final = objective(dq_final)
    hist = jnp.concatenate([hist, v_final[None]])

    q_ref = make_q(dq_final)
    # hard-clamp the optimized DOFs to limits for a valid returned pose
    q_ref = jnp.where(opt_mask[None, :], jnp.clip(q_ref, lb_row[None, :], ub_row[None, :]), q_ref)
    if return_history:
        return q_ref, hist
    return q_ref
