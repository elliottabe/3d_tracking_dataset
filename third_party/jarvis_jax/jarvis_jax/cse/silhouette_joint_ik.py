"""Sibling joint 2D+3D silhouette IK solver (Phase 6).

Assembles its OWN jaxls LeastSquaresProblem reusing the SAME cost formulas as
stac_core_jaxls.JaxlsBatchSolver._build_se3 (marker / smoothness / limit / reg)
PLUS a differentiable silhouette boundary-Chamfer cost, over the SAME SE3Var +
JointVar. stac_core_jaxls.py is left byte-identical (Phases 1-4 invariant); a
test proves silhouette_weight=0 reproduces JaxlsBatchSolver's output.

The silhouette cost factory closes over per-frame constants (cam matrices,
boundary points, per-point confidence) and selects the active frame via a
tiny per-frame FrameVar (int index), rather than packing per-frame data into
a jaxls Var (the retired SilVar/pack_sil_value/unpack_sil_value scheme).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import jaxls
import jaxlie

from stac_mjx import utils

from jarvis_jax.cse.silhouette_chamfer import chamfer_residual

# Number of free-joint DOFs in MuJoCo (3 translation + 4 quaternion), matching
# stac_core_jaxls._FREE_JOINT_NDOF.
_FREE_JOINT_NDOF = 7


def make_silhouette_cost(
    SE3Var, JointVar, FrameVar, *,
    fk_repose, vert_indices, cam_Ms, cam_ts, boundary_all, conf_p_all,
    sil_qs_mask, beta: float, huber_delta: float, silhouette_weight: float,
    qs_to_opt, template_qpos, scale: float = 1.0, chunk_size: int = 32,
):
    """Coverage cost: SAM(eroded)-boundary -> nearest projected mesh vertex.

    Per-frame data (boundary points, per-point confidence) is closed over as
    constants and selected by the integer FrameVar; cameras are iterated with
    jax.lax.scan (FK once per frame, no (C,N,M) tensor). The silhouette gradient
    is restricted to appendage DOFs via where(sil_qs_mask, full_q, stop_grad).
    """
    vert_indices = jnp.asarray(vert_indices)
    template_qpos = jnp.asarray(template_qpos)
    qs_to_opt = jnp.asarray(qs_to_opt)
    sil_qs_mask = jnp.asarray(sil_qs_mask)
    cam_Ms = jnp.asarray(cam_Ms); cam_ts = jnp.asarray(cam_ts)
    boundary_all = jnp.asarray(boundary_all); conf_p_all = jnp.asarray(conf_p_all)

    @jaxls.Cost.factory
    def silhouette_cost(var_values, root_var: SE3Var, joint_var: JointVar,
                        frame_var: FrameVar) -> jnp.ndarray:
        T_root = var_values[root_var]
        joints = var_values[joint_var]
        t = jax.lax.stop_gradient(var_values[frame_var])[0].astype(jnp.int32)
        xyz = T_root.translation(); wxyz = T_root.rotation().wxyz
        q = jnp.concatenate([xyz, wxyz, joints])
        full_q = jnp.where(qs_to_opt, q, template_qpos)
        sil_q = jnp.where(sil_qs_mask, full_q, jax.lax.stop_gradient(full_q))
        verts3d = fk_repose(sil_q, scale, vert_indices)     # (M,3) FK once per frame

        tgt_all = boundary_all[t]; conf_all = conf_p_all[t]   # (C,n_pts,2),(C,n_pts)

        def scan_body(carry, cam):
            M, tt, tgt, cf = cam
            proj = verts3d @ M.T + tt
            r = chamfer_residual(tgt, proj, beta=beta, huber_delta=huber_delta,
                                 chunk_size=chunk_size)      # (n_pts,)
            return carry, r * cf
        _, res = jax.lax.scan(scan_body, None, (cam_Ms, cam_ts, tgt_all, conf_all))
        return (res * silhouette_weight).reshape(-1)         # (C*n_pts,)

    return silhouette_cost


class SilhouetteJaxlsBatchSolver:
    """Joint 2D+3D IK solver: 3-D marker cost (verbatim from
    stac_core_jaxls._build_se3) + smoothness/limit/reg + an additive
    silhouette boundary-Chamfer cost, over the SAME SE3Var + JointVar.

    With silhouette_weight=0 (or sil_data=None) the silhouette cost is NOT
    added and the solve reproduces JaxlsBatchSolver's output (to atol=1e-5):
    the marker/reg/limit/smoothness cost formulas, the [root, joint, kp]
    variable ordering, the SE3-from-normalized-quat initial values, and the
    solve config (auto linear solver, TrustRegionConfig(lambda_initial),
    TerminationConfig(max_iterations)) all match _solve_se3 exactly.
    """

    # Threshold below which dense_cholesky is faster than conjugate_gradient
    # (matches stac_core_jaxls.JaxlsBatchSolver._DENSE_THRESHOLD).
    _DENSE_THRESHOLD = 5000

    def __init__(self, n_iter=50, linear_solver="auto", lambda_initial=1.0,
                 smooth_weight=0.0, use_se3_root=True, beta=8.0, huber_delta=0.0):
        assert use_se3_root, "Phase 6 sibling supports SE3-root mode only"
        self.n_iter = n_iter
        self.linear_solver = linear_solver
        self.lambda_initial = lambda_initial
        self.smooth_weight = smooth_weight
        self.use_se3_root = use_se3_root
        self.beta = beta
        self.huber_delta = huber_delta

    def _pick_linear_solver(self, T, tangent_dim):
        if self.linear_solver != "auto":
            return self.linear_solver
        return "dense_cholesky" if T * tangent_dim < self._DENSE_THRESHOLD else "conjugate_gradient"

    def solve_trajectory(
        self, q_init, mjx_model, mjx_data_template, kp_data, qs_to_opt,
        kps_to_opt, lb, ub, site_idxs, q_reg_weights, *,
        fk_repose=None, cov_vert_indices=None, cont_vert_indices=None,
        cam_Ms=None, cam_ts=None, boundary_all=None, conf_p_all=None,
        sil_qs_mask=None, silhouette_weight=0.0,
        sdf_all=None, grid_scale_all=None, grid_offset_all=None, present_all=None,
        conf_v=None, containment_weight=0.0, margin=0.0,
    ):
        if kp_data.ndim == 3:
            kp_data = kp_data.reshape(kp_data.shape[0], -1)
        q_init = jnp.asarray(q_init); kp_data = jnp.asarray(kp_data)
        qs_to_opt = jnp.asarray(qs_to_opt); kps_to_opt = jnp.asarray(kps_to_opt)
        lb = jnp.asarray(lb); ub = jnp.asarray(ub)
        q_reg_weights = jnp.asarray(q_reg_weights)

        T = q_init.shape[0]
        nq = int(mjx_model.nq)
        n_kp_dim = int(kp_data.shape[-1])
        n_hinges = nq - _FREE_JOINT_NDOF
        smooth_weight = self.smooth_weight

        use_cov = (silhouette_weight != 0.0) and (boundary_all is not None)
        use_cont = (containment_weight != 0.0) and (sdf_all is not None)
        use_sil = use_cov or use_cont
        if sil_qs_mask is None:
            sil_qs_mask = jnp.ones((nq,), bool)

        dummy_joints = jnp.zeros((n_hinges,))
        dummy_kp = jnp.zeros((n_kp_dim,))

        class SE3Var(jaxls.Var[jaxlie.SE3], default_factory=jaxlie.SE3.identity,
                     retract_fn=jaxlie.manifold.rplus, tangent_dim=6): ...
        class JointVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_joints): ...
        class KpVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_kp): ...
        class FrameVar(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros((1,))): ...

        root_all = SE3Var(jnp.arange(T))
        joint_all = JointVar(jnp.arange(T))
        kp_all = KpVar(jnp.arange(T))

        costs: list[jaxls.Cost] = []

        @jaxls.Cost.factory
        def marker_cost(var_values, root_var: SE3Var, joint_var: JointVar, kp_var: KpVar):
            T_root = var_values[root_var]; joints = var_values[joint_var]
            kp = jax.lax.stop_gradient(var_values[kp_var])
            xyz = T_root.translation(); wxyz = T_root.rotation().wxyz
            q = jnp.concatenate([xyz, wxyz, joints])
            full_q = jnp.where(qs_to_opt, q, mjx_data_template.qpos)
            data = mjx_data_template.replace(qpos=full_q)
            data = utils.kinematics(mjx_model, data)
            data = utils.com_pos(mjx_model, data)
            markers = utils.get_site_xpos(data, site_idxs).flatten()
            finite = jnp.isfinite(kp)
            kp_clean = jnp.where(finite, kp, 0.0)
            return (kp_clean - markers) * kps_to_opt * finite
        costs.append(marker_cost(root_all, joint_all, kp_all))

        if jnp.any(q_reg_weights[_FREE_JOINT_NDOF:] > 0):
            hinge_regs = q_reg_weights[_FREE_JOINT_NDOF:]
            hinge_opt = qs_to_opt[_FREE_JOINT_NDOF:]

            @jaxls.Cost.factory
            def reg_cost(var_values, joint_var: JointVar):
                j = var_values[joint_var]
                return jnp.sqrt(hinge_regs * hinge_opt) * j
            costs.append(reg_cost(joint_all))

        hinge_lb = lb[_FREE_JOINT_NDOF:]; hinge_ub = ub[_FREE_JOINT_NDOF:]

        @jaxls.Cost.factory(kind="constraint_leq_zero")
        def limit_cost(var_values, joint_var: JointVar):
            j = var_values[joint_var]
            return jnp.concatenate([hinge_lb - j, j - hinge_ub])
        costs.append(limit_cost(joint_all))

        if smooth_weight > 0.0 and T > 1:
            @jaxls.Cost.factory
            def smoothness_cost(var_values, root_curr: SE3Var, root_prev: SE3Var,
                                joint_curr: JointVar, joint_prev: JointVar):
                root_diff = (var_values[root_prev].inverse() @ var_values[root_curr]).log()
                joint_diff = var_values[joint_curr] - var_values[joint_prev]
                return jnp.concatenate([root_diff, joint_diff]) * smooth_weight
            costs.append(smoothness_cost(
                SE3Var(jnp.arange(1, T)), SE3Var(jnp.arange(0, T - 1)),
                JointVar(jnp.arange(1, T)), JointVar(jnp.arange(0, T - 1))))

        variables = [root_all, joint_all, kp_all]

        frame_all = None
        if use_sil:
            frame_all = FrameVar(jnp.arange(T))
            variables.append(frame_all)

        if use_cov:
            cov_cost = make_silhouette_cost(
                SE3Var, JointVar, FrameVar,
                fk_repose=fk_repose, vert_indices=cov_vert_indices,
                cam_Ms=cam_Ms, cam_ts=cam_ts, boundary_all=boundary_all,
                conf_p_all=conf_p_all, sil_qs_mask=sil_qs_mask, beta=self.beta,
                huber_delta=self.huber_delta, silhouette_weight=silhouette_weight,
                qs_to_opt=qs_to_opt, template_qpos=mjx_data_template.qpos, scale=1.0)
            costs.append(cov_cost(root_all, joint_all, frame_all))

        # containment cost is appended in Task 6 (guarded by use_cont)
        _ = (cont_vert_indices, sdf_all, grid_scale_all, grid_offset_all,
             present_all, conf_v, containment_weight, margin, use_cont)

        analyzed = jaxls.LeastSquaresProblem(costs=costs, variables=variables).analyze()

        xyz_init = q_init[:, :3]; wxyz_init = q_init[:, 3:7]
        hinges_init = q_init[:, _FREE_JOINT_NDOF:]
        qn = jnp.linalg.norm(wxyz_init, axis=-1, keepdims=True)
        wxyz_init = wxyz_init / jnp.where(qn > 0, qn, 1.0)
        roots_init = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(wxyz=wxyz_init), xyz_init)
        init_list = [
            SE3Var(jnp.arange(T)).with_value(roots_init),
            JointVar(jnp.arange(T)).with_value(hinges_init),
            KpVar(jnp.arange(T)).with_value(kp_data),
        ]
        if use_sil:
            init_list.append(FrameVar(jnp.arange(T)).with_value(
                jnp.arange(T).reshape(T, 1).astype(jnp.float32)))

        tangent_dim = 6 + n_hinges
        linear_solver = self._pick_linear_solver(T, tangent_dim)
        if use_sil and self.linear_solver == "auto":
            linear_solver = "conjugate_gradient"

        sol = analyzed.solve(
            verbose=False, linear_solver=linear_solver,
            trust_region=jaxls.TrustRegionConfig(lambda_initial=self.lambda_initial),
            termination=jaxls.TerminationConfig(max_iterations=self.n_iter),
            initial_vals=jaxls.VarValues.make(init_list))
        sol_roots = sol[SE3Var(jnp.arange(T))]
        sol_joints = sol[JointVar(jnp.arange(T))]
        xyz_sol = sol_roots.translation(); wxyz_sol = sol_roots.rotation().wxyz
        return jnp.concatenate([xyz_sol, wxyz_sol, sol_joints], axis=-1)
