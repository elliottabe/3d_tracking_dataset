import os
import numpy as np
import jax.numpy as jnp
import jaxls
import jaxlie
import pytest

from jarvis_jax.cse.silhouette_joint_ik import make_silhouette_cost

XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_visual_canonical_wings.npz"
skip = pytest.mark.skipif(not (os.path.exists(XML) and os.path.exists(MESH)), reason="assets absent")


@skip
def test_silhouette_cost_builds_and_residual_shape():
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.cse.silhouette_dof import build_appendage_dof_mask
    from jarvis_jax.cse.silhouette_targets import silhouette_fk_indices
    import mujoco
    anat = load_anatomy(XML, MESH); fk = make_fk_repose(anat)
    m = mujoco.MjModel.from_xml_path(XML)
    vidx = silhouette_fk_indices(MESH, subset="fps_300")
    sil_qs = build_appendage_dof_mask(m)
    T, n_cam, n_pts = 3, 2, 16
    cam_Ms = jnp.tile(jnp.eye(2, 3)[None], (n_cam, 1, 1))
    cam_ts = jnp.zeros((n_cam, 2))
    boundary_all = jnp.zeros((T, n_cam, n_pts, 2))
    conf_p_all = jnp.ones((T, n_cam, n_pts))
    qs_to_opt = jnp.ones((m.nq,), bool)

    class SE3Var(jaxls.Var[jaxlie.SE3], default_factory=jaxlie.SE3.identity,
                 retract_fn=jaxlie.manifold.rplus, tangent_dim=6): ...
    class JointVar(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros((m.nq - 7,))): ...
    class FrameVar(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros((1,))): ...

    cost = make_silhouette_cost(
        SE3Var, JointVar, FrameVar, fk_repose=fk, vert_indices=vidx,
        cam_Ms=cam_Ms, cam_ts=cam_ts, boundary_all=boundary_all, conf_p_all=conf_p_all,
        sil_qs_mask=sil_qs, beta=8.0, huber_delta=0.0, silhouette_weight=1.0,
        qs_to_opt=qs_to_opt, template_qpos=jnp.asarray(anat["qpos0"]))
    root = SE3Var(jnp.arange(T)); joint = JointVar(jnp.arange(T)); frame = FrameVar(jnp.arange(T))
    prob = jaxls.LeastSquaresProblem(costs=[cost(root, joint, frame)],
                                     variables=[root, joint, frame]).analyze()
    vals = jaxls.VarValues.make([
        SE3Var(jnp.arange(T)).with_value(jaxlie.SE3.identity((T,))),
        JointVar(jnp.arange(T)).with_value(jnp.zeros((T, m.nq - 7))),
        FrameVar(jnp.arange(T)).with_value(jnp.arange(T).reshape(T, 1).astype(float)),
    ])
    r = prob.compute_residual_vector(vals)
    assert np.isfinite(np.asarray(r)).all()
    assert r.shape[0] == T * n_cam * n_pts
