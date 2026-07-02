import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pytest

XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"


@pytest.mark.skipif(not __import__("os").path.exists(XML), reason="fly model not present")
def test_silhouette_cost_residual_shape_and_finite():
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.cse.silhouette_joint_ik import (
        make_silhouette_cost, SIL_PER_CAM, pack_sil_value,
    )

    anat = load_anatomy(XML, MESH)
    fk = make_fk_repose(anat)
    nq = anat["nq"]
    n_hinges = nq - 7
    vert_indices = np.asarray(anat["fps"][300], dtype=np.int32)  # (300,) full-array idx
    n_pts, n_cam = 16, 2
    T = 3

    # --- Var classes matching the solver's ---
    dummy_joints = jnp.zeros((n_hinges,))
    dummy_sil = jnp.zeros((n_cam * SIL_PER_CAM(n_pts),))

    class SE3Var(jaxls.Var[jaxlie.SE3], default_factory=jaxlie.SE3.identity,
                 retract_fn=jaxlie.manifold.rplus, tangent_dim=6): ...
    class JointVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_joints): ...
    class SilVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_sil): ...

    qs_to_opt = jnp.ones(nq, dtype=bool)
    template_qpos = jnp.asarray(anat["qpos0"])

    cost = make_silhouette_cost(
        SE3Var, JointVar, SilVar,
        fk_repose=fk, vert_indices=vert_indices, n_cam=n_cam, n_pts=n_pts,
        beta=8.0, huber_delta=0.0, silhouette_weight=1.0,
        qs_to_opt=qs_to_opt, template_qpos=template_qpos, scale=1.0,
    )

    # build a value: identity roots, zero joints, and a packed SilVar per frame.
    # affine cam matrices: simple orthographic-ish 3x4 DLT with 3rd row [0,0,0,1].
    P0 = np.array([[8.0, 0, 0, -2.0], [0, -8.0, 0, 460.0], [0, 0, 0, 1.0]])
    P1 = np.array([[0, 8.0, 0, -2.0], [0, 0, -8.0, 460.0], [0, 0, 0, 1.0]])
    boundary = np.zeros((n_cam, n_pts, 2), dtype=np.float64)  # dummy target pts
    sil_row = pack_sil_value([P0, P1], boundary)  # (n_cam*SIL_PER_CAM,)
    sil_batch = jnp.tile(jnp.asarray(sil_row)[None], (T, 1))

    problem = jaxls.LeastSquaresProblem(
        costs=[cost(SE3Var(jnp.arange(T)), JointVar(jnp.arange(T)), SilVar(jnp.arange(T)))],
        variables=[SE3Var(jnp.arange(T)), JointVar(jnp.arange(T)), SilVar(jnp.arange(T))],
    ).analyze()

    vals = jaxls.VarValues.make([
        SE3Var(jnp.arange(T)).with_value(jaxlie.SE3.identity((T,))),
        JointVar(jnp.arange(T)).with_value(jnp.zeros((T, n_hinges))),
        SilVar(jnp.arange(T)).with_value(sil_batch),
    ])
    residuals = problem.compute_residual_vector(vals)
    r = np.asarray(residuals)
    assert np.isfinite(r).all()
    assert r.size == T * n_cam * n_pts


@pytest.mark.skipif(not __import__("os").path.exists(XML), reason="fly model not present")
def test_pack_sil_value_layout_roundtrip():
    from jarvis_jax.cse.silhouette_joint_ik import (
        pack_sil_value, unpack_sil_value, SIL_PER_CAM,
    )
    n_cam, n_pts = 2, 5
    P0 = np.array([[8.0, 0, 0, -2.0], [0, -8.0, 0, 460.0], [0, 0, 0, 1.0]])
    P1 = np.array([[0, 8.0, 0, -3.0], [0, 0, -8.0, 461.0], [0, 0, 0, 1.0]])
    bnd = np.arange(n_cam * n_pts * 2, dtype=np.float64).reshape(n_cam, n_pts, 2)
    packed = pack_sil_value([P0, P1], bnd)
    assert packed.shape == (n_cam * SIL_PER_CAM(n_pts),)
    Ms, ts, pts = unpack_sil_value(jnp.asarray(packed), n_cam, n_pts)
    np.testing.assert_allclose(np.asarray(Ms[0]), P0[:2, :3])
    np.testing.assert_allclose(np.asarray(ts[1]), P1[:2, 3])
    np.testing.assert_allclose(np.asarray(pts[1]), bnd[1])
