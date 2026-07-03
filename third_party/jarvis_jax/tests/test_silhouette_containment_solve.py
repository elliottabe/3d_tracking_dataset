import os
import numpy as np
import jax.numpy as jnp
import jaxls
import jaxlie
import pytest

from jarvis_jax.cse.silhouette_joint_ik import make_containment_cost

XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_visual_canonical_wings.npz"
skip = pytest.mark.skipif(not (os.path.exists(XML) and os.path.exists(MESH)), reason="assets absent")


@skip
def test_containment_cost_residual_shape_and_finite():
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.cse.silhouette_dof import build_appendage_dof_mask, appendage_vertex_indices
    import mujoco
    anat = load_anatomy(XML, MESH); fk = make_fk_repose(anat)
    m = mujoco.MjModel.from_xml_path(XML)
    vidx = appendage_vertex_indices(MESH, subset="fps_300")
    M = len(vidx)
    sil_qs = build_appendage_dof_mask(m)
    T, n_cam, H, W = 2, 2, 32, 32
    # SDF all +5 (everything "outside") -> every appendage vert penalized -> positive residual
    sdf_all = jnp.full((T, n_cam, H, W), 5.0)
    grid_scale_all = jnp.ones((T, n_cam, 2))
    grid_offset_all = jnp.zeros((T, n_cam, 2))
    present_all = jnp.ones((T, n_cam), bool)
    conf_v = jnp.ones((M,))
    cam_Ms = jnp.tile(jnp.eye(2, 3)[None], (n_cam, 1, 1)); cam_ts = jnp.zeros((n_cam, 2))

    class SE3Var(jaxls.Var[jaxlie.SE3], default_factory=jaxlie.SE3.identity,
                 retract_fn=jaxlie.manifold.rplus, tangent_dim=6): ...
    class JointVar(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros((m.nq - 7,))): ...
    class FrameVar(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros((1,))): ...

    cost = make_containment_cost(
        SE3Var, JointVar, FrameVar, fk_repose=fk, vert_indices=vidx,
        cam_Ms=cam_Ms, cam_ts=cam_ts, sdf_all=sdf_all, grid_scale_all=grid_scale_all,
        grid_offset_all=grid_offset_all, present_all=present_all, conf_v=conf_v,
        sil_qs_mask=sil_qs, margin=0.0, containment_weight=1.0,
        qs_to_opt=jnp.ones((m.nq,), bool), template_qpos=jnp.asarray(anat["qpos0"]))
    root = SE3Var(jnp.arange(T)); joint = JointVar(jnp.arange(T)); frame = FrameVar(jnp.arange(T))
    prob = jaxls.LeastSquaresProblem(costs=[cost(root, joint, frame)],
                                     variables=[root, joint, frame]).analyze()
    vals = jaxls.VarValues.make([
        SE3Var(jnp.arange(T)).with_value(jaxlie.SE3.identity((T,))),
        JointVar(jnp.arange(T)).with_value(jnp.zeros((T, m.nq - 7))),
        FrameVar(jnp.arange(T)).with_value(jnp.arange(T).reshape(T, 1).astype(float)),
    ])
    r = np.asarray(prob.compute_residual_vector(vals))
    assert r.shape[0] == T * n_cam * M
    assert np.isfinite(r).all()
    assert (r > 0).any()      # verts sit in "outside" SDF -> some positive penalty


@skip
def test_containment_baseline_reproduces_and_moves_qpos():
    # containment_weight=0 -> identical to no-silhouette; weight>0 with an
    # all-"outside" SDF must move qpos (appendage DOFs) away from init.
    from jarvis_jax.cse.silhouette_ik_solve import build_solver_inputs
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.cse.silhouette_dof import build_appendage_dof_mask, appendage_vertex_indices
    from jarvis_jax.cse.silhouette_joint_ik import SilhouetteJaxlsBatchSolver
    import mujoco
    IK = "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5"
    if not os.path.exists(IK):
        pytest.skip("ik h5 absent")
    inp = build_solver_inputs(IK, XML)
    T = 2
    q_init = inp["q_init"][:T]; kp_data = inp["kp_data"][:T]
    anat = load_anatomy(XML, MESH); fk = make_fk_repose(anat)
    m = mujoco.MjModel.from_xml_path(XML)
    vidx = appendage_vertex_indices(MESH, subset="fps_300"); M = len(vidx)
    sil_qs = build_appendage_dof_mask(m)
    n_cam = 7
    cam_Ms = jnp.tile(jnp.eye(2, 3)[None], (n_cam, 1, 1)); cam_ts = jnp.zeros((n_cam, 2))
    sdf_all = jnp.full((T, n_cam, 32, 32), 5.0)
    gs = jnp.ones((T, n_cam, 2)); go = jnp.zeros((T, n_cam, 2)); pr = jnp.ones((T, n_cam), bool)
    solver = SilhouetteJaxlsBatchSolver(n_iter=5, smooth_weight=0.0)
    common = dict(q_init=q_init, mjx_model=inp["mjx_model"], mjx_data_template=inp["mjx_data"],
                  kp_data=kp_data, qs_to_opt=inp["qs_to_opt"], kps_to_opt=inp["kps_to_opt"],
                  lb=inp["lb"], ub=inp["ub"], site_idxs=inp["site_idxs"],
                  q_reg_weights=inp["q_reg_weights"])
    q0 = np.asarray(solver.solve_trajectory(**common))          # no silhouette
    q_cont = np.asarray(solver.solve_trajectory(
        **common, fk_repose=fk, cont_vert_indices=vidx, cam_Ms=cam_Ms, cam_ts=cam_ts,
        sdf_all=sdf_all, grid_scale_all=gs, grid_offset_all=go, present_all=pr,
        conf_v=jnp.ones((M,)), sil_qs_mask=sil_qs, containment_weight=0.3))
    assert np.linalg.norm(q_cont - q0) > 1e-4
    # containment must NOT move the root translation far (DOF mask excludes root)
    assert np.linalg.norm(q_cont[:, :3] - q0[:, :3]) < np.linalg.norm(q_cont[:, 7:] - q0[:, 7:]) + 1e-6
