import os
import numpy as np
import jax.numpy as jnp
import pytest

from jarvis_jax.cse.silhouette_refine import refine_appendages_adam

XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_visual_canonical_wings.npz"
skip = pytest.mark.skipif(not (os.path.exists(XML) and os.path.exists(MESH)), reason="assets absent")


@skip
def test_refine_reduces_objective_moves_only_appendages_respects_limits():
    """Adam refinement must (1) reduce the silhouette objective, (2) move ONLY the
    opt_mask (appendage) DOFs -- root/body exactly unchanged, and (3) keep the
    optimized DOFs within their joint limits. Uses an all-'outside' SDF so the
    containment term has a real inward gradient."""
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.cse.silhouette_dof import build_appendage_dof_mask, appendage_vertex_indices
    from jarvis_jax.cse.silhouette_targets import silhouette_fk_indices
    import mujoco
    anat = load_anatomy(XML, MESH); fk = make_fk_repose(anat)
    m = mujoco.MjModel.from_xml_path(XML)
    nq = m.nq
    cov_idx = silhouette_fk_indices(MESH, subset="fps_300")
    cont_idx = appendage_vertex_indices(MESH, subset="fps_300")
    sil_qs = build_appendage_dof_mask(m)
    T, n_cam, n_pts, H, W = 3, 2, 8, 32, 32
    q_init = jnp.tile(jnp.asarray(anat["qpos0"])[None], (T, 1))
    cam_Ms = jnp.tile(jnp.eye(2, 3)[None], (n_cam, 1, 1)); cam_ts = jnp.zeros((n_cam, 2))
    # all-'outside' SDF -> containment pulls verts in (nonzero inward gradient)
    sdf_all = jnp.full((T, n_cam, H, W), 5.0)
    gs = jnp.ones((T, n_cam, 2)); go = jnp.zeros((T, n_cam, 2)); pres = jnp.ones((T, n_cam), bool)
    boundary = jnp.zeros((T, n_cam, n_pts, 2)); conf_p = jnp.ones((T, n_cam, n_pts))
    bridge_s = jnp.ones((T,)); bridge_R = jnp.broadcast_to(jnp.eye(3), (T, 3, 3)); bridge_t = jnp.zeros((T, 3))
    lb = jnp.full((nq,), -2.0); ub = jnp.full((nq,), 2.0)

    q_ref, hist = refine_appendages_adam(
        q_init, fk_repose=fk, cov_vert_indices=cov_idx, cont_vert_indices=cont_idx,
        cam_Ms=cam_Ms, cam_ts=cam_ts, boundary_all=boundary, conf_p_all=conf_p,
        sdf_all=sdf_all, grid_scale_all=gs, grid_offset_all=go, present_all=pres,
        conf_v=jnp.ones((len(cont_idx),)), bridge_s_all=bridge_s, bridge_R_all=bridge_R,
        bridge_t_all=bridge_t, opt_mask=sil_qs, lb=lb, ub=ub,
        silhouette_weight=0.0, containment_weight=1.0, smooth_weight=0.0,
        anchor_weight=0.0, n_steps=30, lr=1e-2, return_history=True)
    q_ref = np.asarray(q_ref); hist = np.asarray(hist); q0 = np.asarray(q_init)
    sil_qs = np.asarray(sil_qs)

    # (1) objective decreased
    assert hist[-1] < hist[0]
    # (2) ONLY appendage DOFs moved; root/body exactly frozen
    assert np.allclose(q_ref[:, ~sil_qs], q0[:, ~sil_qs], atol=0.0)
    assert np.abs(q_ref[:, sil_qs] - q0[:, sil_qs]).max() > 1e-4
    # (3) optimized DOFs within limits
    assert (q_ref[:, sil_qs] >= np.asarray(lb)[sil_qs] - 1e-4).all()
    assert (q_ref[:, sil_qs] <= np.asarray(ub)[sil_qs] + 1e-4).all()
    assert np.isfinite(q_ref).all()
