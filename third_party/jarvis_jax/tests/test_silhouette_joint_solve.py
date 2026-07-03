import os
import numpy as np
import jax.numpy as jnp
import pytest

XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"
IK = "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5"


def test_stac_core_jaxls_is_byte_identical():
    """The shared solver file must remain untouched by Phase 6 (Phases 1-4
    depend on it). Assert the git working tree has no changes to it."""
    import subprocess
    repo = "/mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset"
    out = subprocess.run(
        ["git", "diff", "--", "stac-mjx/stac_mjx/stac_core_jaxls.py"],
        cwd=repo, capture_output=True, text=True,
    )
    assert out.stdout.strip() == "", (
        "stac_core_jaxls.py was modified; Phase 6 must add a sibling module, "
        "not touch the shared solver."
    )


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present")
def test_no_silhouette_matches_jaxls_batch_solver():
    """SilhouetteJaxlsBatchSolver with no silhouette/containment kwargs (all
    defaults) must reproduce JaxlsBatchSolver's trajectory to tight tolerance;
    no coverage cost and no FrameVar should be added."""
    from jarvis_jax.cse.silhouette_ik_solve import build_solver_inputs
    from stac_mjx.stac_core_jaxls import JaxlsBatchSolver
    from jarvis_jax.cse.silhouette_joint_ik import SilhouetteJaxlsBatchSolver

    inp = build_solver_inputs(IK, XML)
    sl = slice(0, 6)
    common = dict(
        q_init=inp["q_init"][sl],
        mjx_model=inp["mjx_model"],
        mjx_data_template=inp["mjx_data"],
        kp_data=inp["kp_data"][sl],
        qs_to_opt=inp["qs_to_opt"],
        kps_to_opt=inp["kps_to_opt"],
        lb=inp["lb"], ub=inp["ub"],
        site_idxs=inp["site_idxs"],
        q_reg_weights=inp["q_reg_weights"],
    )
    q_ref = np.asarray(JaxlsBatchSolver(n_iter=30, smooth_weight=0.1).solve_trajectory(**common))
    q_sib = np.asarray(
        SilhouetteJaxlsBatchSolver(n_iter=30, smooth_weight=0.1).solve_trajectory(**common)
    )
    assert q_sib.shape == q_ref.shape
    np.testing.assert_allclose(q_sib, q_ref, atol=1e-5)


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present")
def test_silhouette_weight_moves_qpos_toward_boundary():
    """A coverage target far outside the projected mesh, with weight>0, must
    change qpos relative to the weight=0 solve (the coverage factor is live)."""
    from jarvis_jax.cse.silhouette_ik_solve import build_solver_inputs
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.cse.silhouette_dof import build_appendage_dof_mask
    from jarvis_jax.cse.silhouette_targets import silhouette_fk_indices
    from jarvis_jax.cse.silhouette_joint_ik import SilhouetteJaxlsBatchSolver
    import mujoco

    inp = build_solver_inputs(IK, XML)
    sl = slice(0, 4)
    anat = load_anatomy(XML, MESH)
    fk = make_fk_repose(anat)
    vidx = silhouette_fk_indices(MESH, subset="fps_300")
    m = mujoco.MjModel.from_xml_path(XML)
    sil_qs = build_appendage_dof_mask(m)

    q_init = inp["q_init"][sl]
    mjx_model = inp["mjx_model"]
    mjx_data = inp["mjx_data"]
    kp_data = inp["kp_data"][sl]
    qs_to_opt = inp["qs_to_opt"]
    kps_to_opt = inp["kps_to_opt"]
    lb = inp["lb"]; ub = inp["ub"]
    site_idxs = inp["site_idxs"]
    q_reg_weights = inp["q_reg_weights"]

    n_cam = 2
    cam_Ms = jnp.tile(jnp.eye(2, 3)[None], (n_cam, 1, 1))
    cam_ts = jnp.zeros((n_cam, 2))

    solver = SilhouetteJaxlsBatchSolver(n_iter=25, smooth_weight=0.0, beta=8.0)
    q0 = np.asarray(solver.solve_trajectory(
        q_init=q_init, mjx_model=mjx_model, mjx_data_template=mjx_data,
        kp_data=kp_data, qs_to_opt=qs_to_opt, kps_to_opt=kps_to_opt, lb=lb, ub=ub,
        site_idxs=site_idxs, q_reg_weights=q_reg_weights,
    ))

    # far-away boundary target so the coverage term must move qpos
    T = q_init.shape[0]
    n_cam, n_pts = cam_Ms.shape[0], 8
    boundary_all = jnp.full((T, n_cam, n_pts, 2), 1e4)         # unreachable -> nonzero grad
    conf_p_all = jnp.ones((T, n_cam, n_pts))
    q_w = solver.solve_trajectory(
        q_init=q_init, mjx_model=mjx_model, mjx_data_template=mjx_data,
        kp_data=kp_data, qs_to_opt=qs_to_opt, kps_to_opt=kps_to_opt, lb=lb, ub=ub,
        site_idxs=site_idxs, q_reg_weights=q_reg_weights,
        fk_repose=fk, cov_vert_indices=vidx, cam_Ms=cam_Ms, cam_ts=cam_ts,
        boundary_all=boundary_all, conf_p_all=conf_p_all, sil_qs_mask=sil_qs,
        silhouette_weight=0.3)
    assert q_w.shape == q0.shape
    assert np.isfinite(np.asarray(q_w)).all()
    assert float(jnp.linalg.norm(q_w - q0)) > 1e-4
