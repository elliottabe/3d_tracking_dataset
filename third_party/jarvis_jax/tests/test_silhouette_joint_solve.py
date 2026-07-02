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
    """SilhouetteJaxlsBatchSolver with silhouette_weight=0 (no sil_data) must
    reproduce JaxlsBatchSolver's trajectory to tight tolerance."""
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
        SilhouetteJaxlsBatchSolver(n_iter=30, smooth_weight=0.1).solve_trajectory(
            **common, silhouette_weight=0.0, sil_data=None,
        )
    )
    assert q_sib.shape == q_ref.shape
    np.testing.assert_allclose(q_sib, q_ref, atol=1e-5)


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present")
def test_silhouette_weight_moves_qpos_toward_boundary():
    """A silhouette target pulling a wing vertex outward, with weight>0, must
    change qpos relative to the weight=0 solve (the joint factor is live)."""
    from jarvis_jax.cse.silhouette_ik_solve import build_solver_inputs
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.cse.silhouette_joint_ik import (
        SilhouetteJaxlsBatchSolver, pack_sil_value, SIL_PER_CAM,
    )

    inp = build_solver_inputs(IK, XML)
    sl = slice(0, 4)
    anat = load_anatomy(XML, MESH)
    fk = make_fk_repose(anat)
    vert_indices = np.asarray(anat["fps"][300], dtype=np.int32)
    n_pts, n_cam = 12, 2
    T = 4

    # affine cams; boundary points placed FAR outside the projected mesh so the
    # Chamfer residual is large and pulls qpos. (Values are arbitrary but fixed.)
    P0 = np.array([[8.0, 0, 0, -2.0], [0, -8.0, 0, 460.0], [0, 0, 0, 1.0]])
    P1 = np.array([[0, 8.0, 0, -2.0], [0, 0, -8.0, 460.0], [0, 0, 0, 1.0]])
    boundary = np.full((n_cam, n_pts, 2), 500.0)  # far away -> big pull
    sil_row = pack_sil_value([P0, P1], boundary)
    sil_data = np.tile(sil_row[None], (T, 1))

    common = dict(
        q_init=inp["q_init"][sl], mjx_model=inp["mjx_model"],
        mjx_data_template=inp["mjx_data"], kp_data=inp["kp_data"][sl],
        qs_to_opt=inp["qs_to_opt"], kps_to_opt=inp["kps_to_opt"],
        lb=inp["lb"], ub=inp["ub"], site_idxs=inp["site_idxs"],
        q_reg_weights=inp["q_reg_weights"],
    )
    solver = SilhouetteJaxlsBatchSolver(n_iter=25, smooth_weight=0.0, beta=8.0)
    q0 = np.asarray(solver.solve_trajectory(**common, silhouette_weight=0.0, sil_data=None))
    q1 = np.asarray(solver.solve_trajectory(
        **common, silhouette_weight=0.5, sil_data=sil_data,
        fk_repose=fk, vert_indices=vert_indices, n_pts=n_pts,
    ))
    assert q1.shape == q0.shape
    assert np.isfinite(q1).all()
    # the silhouette term must have moved the solution
    assert np.linalg.norm(q1 - q0) > 1e-4
