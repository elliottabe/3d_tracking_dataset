"""Sibling joint 2D+3D silhouette IK solver (Phase 6).

Assembles its OWN jaxls LeastSquaresProblem reusing the SAME cost formulas as
stac_core_jaxls.JaxlsBatchSolver._build_se3 (marker / smoothness / limit / reg)
PLUS a differentiable silhouette boundary-Chamfer cost, over the SAME SE3Var +
JointVar. stac_core_jaxls.py is left byte-identical (Phases 1-4 invariant); a
test proves silhouette_weight=0 reproduces JaxlsBatchSolver's output.

Two additive silhouette cost factories share the same pattern: each takes the
frame-invariant camera matrices as closed-over constants, and receives its
per-frame data (coverage boundary points/confidence or containment SDF
crops/confidence, plus the model->mm bridge) as BATCHED factory arguments so
jaxls vectorizes one factor definition over all T frames (the pyroki/jaxls
idiom), rather than packing per-frame data into a jaxls Var or closing full
(T,...) trajectories over a scan. `make_silhouette_cost` is the coverage
(SAM-boundary Chamfer) term; `make_containment_cost` is the mesh->mask
containment (SDF relu) term; both are gated independently in `solve_trajectory`
by their own weight/data kwargs.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import jaxls
import jaxlie

from stac_mjx import utils

from jarvis_jax.tracking.silhouette_chamfer import chamfer_residual
from jarvis_jax.tracking.silhouette_containment import containment_residual

# Number of free-joint DOFs in MuJoCo (3 translation + 4 quaternion), matching
# stac_core_jaxls._FREE_JOINT_NDOF.
_FREE_JOINT_NDOF = 7


def _identity_bridges(T):
    """(s,R,t) per-frame identity model->mm bridges: s=1, R=I, t=0."""
    return (jnp.ones((T,)), jnp.broadcast_to(jnp.eye(3), (T, 3, 3)), jnp.zeros((T, 3)))


def make_silhouette_cost(
    SE3Var, JointVar, *,
    fk_repose, vert_indices, cam_Ms, cam_ts,
    sil_qs_mask, beta: float, huber_delta: float, silhouette_weight: float,
    qs_to_opt, template_qpos, scale: float = 1.0, chunk_size: int = 32,
):
    """Coverage cost: SAM(eroded)-boundary -> nearest projected mesh vertex.

    Per-frame data (boundary points, per-point confidence, model->mm bridge) is
    passed as BATCHED factory arguments, so jaxls vectorizes this one factor over
    T frames; cameras are iterated with jax.lax.scan (FK once per frame, no
    (C,N,M) tensor). The silhouette gradient is restricted to appendage DOFs via
    where(sil_qs_mask, full_q, stop_grad).

    CRITICAL: fk_repose returns MODEL-frame verts, but the affine cameras map
    mm-world -> px. The per-frame model->mm bridge (s,R,t) (from
    build_model_to_mm_bridges, same Umeyama fit run_polish/_metrics use) is
    applied BEFORE projection: verts_mm = s*(verts3d @ R.T) + t. Without it the
    mesh projects to garbage px and the fit is destroyed. Defaults to identity
    only for synthetic-camera unit tests; real callers MUST pass bridges.
    """
    vert_indices = jnp.asarray(vert_indices)
    template_qpos = jnp.asarray(template_qpos)
    qs_to_opt = jnp.asarray(qs_to_opt)
    sil_qs_mask = jnp.asarray(sil_qs_mask)
    cam_Ms = jnp.asarray(cam_Ms); cam_ts = jnp.asarray(cam_ts)

    @jaxls.Cost.factory
    def silhouette_cost(var_values, root_var: SE3Var, joint_var: JointVar,
                        boundary, conf_p, bridge_s, bridge_R, bridge_t) -> jnp.ndarray:
        # per-frame (jaxls-sliced) args: boundary (C,n_pts,2), conf_p (C,n_pts),
        # bridge_s (), bridge_R (3,3), bridge_t (3,)
        T_root = var_values[root_var]
        joints = var_values[joint_var]
        xyz = T_root.translation(); wxyz = T_root.rotation().wxyz
        q = jnp.concatenate([xyz, wxyz, joints])
        full_q = jnp.where(qs_to_opt, q, template_qpos)
        sil_q = jnp.where(sil_qs_mask, full_q, jax.lax.stop_gradient(full_q))
        verts3d = fk_repose(sil_q, scale, vert_indices)     # (M,3) FK once per frame
        verts_mm = bridge_s * (verts3d @ bridge_R.T) + bridge_t

        def scan_body(carry, cam):
            M, tt, tgt, cf = cam
            proj = verts_mm @ M.T + tt
            r = chamfer_residual(tgt, proj, beta=beta, huber_delta=huber_delta,
                                 chunk_size=chunk_size)      # (n_pts,)
            return carry, r * cf
        _, res = jax.lax.scan(scan_body, None, (cam_Ms, cam_ts, boundary, conf_p))
        return (res * silhouette_weight).reshape(-1)         # (C*n_pts,)

    return silhouette_cost


def make_containment_cost(
    SE3Var, JointVar, *,
    fk_repose, vert_indices, cam_Ms, cam_ts, conf_v, sil_qs_mask, margin: float,
    containment_weight: float, qs_to_opt, template_qpos, scale: float = 1.0,
):
    """Containment cost: appendage verts outside the mask -> relu(SDF) penalty.

    Per-frame SDF crops + transforms are passed as BATCHED factory arguments, so
    jaxls vectorizes this one factor over T frames; cameras iterated with
    jax.lax.scan (FK once/frame). Gradient restricted to appendage DOFs via
    where(sil_qs_mask, full_q, stop_grad).

    CRITICAL: verts are FK'd in MODEL frame, then mapped to mm via the per-frame
    model->mm bridge (s,R,t) BEFORE the affine (mm->px) projection:
    verts_mm = s*(verts3d @ R.T) + t. Identity default is for synthetic-camera
    unit tests only; real callers MUST pass bridges (see make_silhouette_cost).
    """
    vert_indices = jnp.asarray(vert_indices)
    template_qpos = jnp.asarray(template_qpos)
    qs_to_opt = jnp.asarray(qs_to_opt); sil_qs_mask = jnp.asarray(sil_qs_mask)
    cam_Ms = jnp.asarray(cam_Ms); cam_ts = jnp.asarray(cam_ts)
    conf_v = jnp.asarray(conf_v)

    @jaxls.Cost.factory
    def containment_cost(var_values, root_var: SE3Var, joint_var: JointVar,
                         sdf, grid_scale, grid_offset, present,
                         bridge_s, bridge_R, bridge_t) -> jnp.ndarray:
        # per-frame (jaxls-sliced) args: sdf (C,H,W), grid_scale (C,2),
        # grid_offset (C,2), present (C,), bridge_s (), bridge_R (3,3), bridge_t (3,)
        T_root = var_values[root_var]; joints = var_values[joint_var]
        xyz = T_root.translation(); wxyz = T_root.rotation().wxyz
        q = jnp.concatenate([xyz, wxyz, joints])
        full_q = jnp.where(qs_to_opt, q, template_qpos)
        sil_q = jnp.where(sil_qs_mask, full_q, jax.lax.stop_gradient(full_q))
        verts3d = fk_repose(sil_q, scale, vert_indices)     # (M,3) model frame
        verts_mm = bridge_s * (verts3d @ bridge_R.T) + bridge_t

        def scan_body(carry, cam):
            M, tt, sdf_c, gs, go, pres = cam
            proj = verts_mm @ M.T + tt                       # (M,2)
            r = containment_residual(proj, sdf_c, gs, go, conf_v,
                                     margin=margin, present=pres)
            return carry, r                                  # (M,)
        _, res = jax.lax.scan(scan_body, None,
                              (cam_Ms, cam_ts, sdf, grid_scale, grid_offset, present))
        return (res * containment_weight).reshape(-1)        # (C*M,)

    return containment_cost


class SilhouetteJaxlsBatchSolver:
    """Joint 2D+3D IK solver: 3-D marker cost (verbatim from
    stac_core_jaxls._build_se3) + smoothness/limit/reg + additive
    silhouette coverage (boundary-Chamfer) and/or containment (SDF-relu)
    costs, over the SAME SE3Var + JointVar.

    `silhouette_weight` and `containment_weight` independently gate their
    respective additive costs (each also requires its data kwarg -
    `boundary_all` / `sdf_all` - to be non-None); there is no `sil_data`
    kwarg. With both weights 0 (the default), neither silhouette cost is
    added, and the solve reproduces JaxlsBatchSolver's output
    (to atol=1e-5): the marker/reg/limit/smoothness cost formulas, the
    [root, joint, kp] variable ordering, the SE3-from-normalized-quat
    initial values, and the solve config (auto linear solver,
    TrustRegionConfig(lambda_initial), TerminationConfig(max_iterations))
    all match _solve_se3 exactly.
    """

    # Threshold below which dense_cholesky is faster than conjugate_gradient
    # (matches stac_core_jaxls.JaxlsBatchSolver._DENSE_THRESHOLD).
    _DENSE_THRESHOLD = 5000

    def __init__(self, n_iter=50, linear_solver="auto", lambda_initial=1.0,
                 smooth_weight=0.0, use_se3_root=True, beta=8.0, huber_delta=0.0,
                 cg_tolerance_max=1e-2, cg_tolerance_min=1e-7, verbose=False):
        assert use_se3_root, "Phase 6 sibling supports SE3-root mode only"
        self.n_iter = n_iter
        self.linear_solver = linear_solver
        self.lambda_initial = lambda_initial
        self.smooth_weight = smooth_weight
        self.use_se3_root = use_se3_root
        self.beta = beta
        self.huber_delta = huber_delta
        self.verbose = verbose
        # Conjugate-gradient inexact-Newton tolerance (Eisenstat-Walker). jaxls
        # caps CG at maxiter=len(x)=T*(6+n_hinges) (~44k for a 513-frame bout)
        # and tightens toward tolerance_min; with the ill-conditioned silhouette
        # normal equations that means thousands of matvecs per LM step. For a
        # REFINEMENT polish an inexact linear solve is enough, so loosen these to
        # trade a little linear-solve accuracy for a large wall-time win.
        self.cg_tolerance_max = cg_tolerance_max
        self.cg_tolerance_min = cg_tolerance_min

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
        bridge_s_all=None, bridge_R_all=None, bridge_t_all=None,
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
        # per-frame model->mm bridge (fixed); identity if not supplied (unit tests
        # with synthetic cameras). Real fits MUST pass bridges or the mesh projects
        # in model frame through mm cameras and the fit is destroyed.
        if use_sil and bridge_s_all is None:
            bridge_s_all, bridge_R_all, bridge_t_all = _identity_bridges(T)

        dummy_joints = jnp.zeros((n_hinges,))
        dummy_kp = jnp.zeros((n_kp_dim,))

        class SE3Var(jaxls.Var[jaxlie.SE3], default_factory=jaxlie.SE3.identity,
                     retract_fn=jaxlie.manifold.rplus, tangent_dim=6): ...
        class JointVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_joints): ...
        class KpVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_kp): ...

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

        if use_cov:
            cov_cost = make_silhouette_cost(
                SE3Var, JointVar,
                fk_repose=fk_repose, vert_indices=cov_vert_indices,
                cam_Ms=cam_Ms, cam_ts=cam_ts, sil_qs_mask=sil_qs_mask, beta=self.beta,
                huber_delta=self.huber_delta, silhouette_weight=silhouette_weight,
                qs_to_opt=qs_to_opt, template_qpos=mjx_data_template.qpos, scale=1.0)
            # per-frame data as BATCHED factory args (jaxls vectorizes over T)
            costs.append(cov_cost(root_all, joint_all, boundary_all, conf_p_all,
                                  bridge_s_all, bridge_R_all, bridge_t_all))

        if use_cont:
            cont_cost = make_containment_cost(
                SE3Var, JointVar,
                fk_repose=fk_repose, vert_indices=cont_vert_indices,
                cam_Ms=cam_Ms, cam_ts=cam_ts, conf_v=conf_v, sil_qs_mask=sil_qs_mask,
                margin=margin, containment_weight=containment_weight,
                qs_to_opt=qs_to_opt, template_qpos=mjx_data_template.qpos, scale=1.0)
            costs.append(cont_cost(root_all, joint_all, sdf_all, grid_scale_all,
                                   grid_offset_all, present_all,
                                   bridge_s_all, bridge_R_all, bridge_t_all))

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

        tangent_dim = 6 + n_hinges
        linear_solver = self._pick_linear_solver(T, tangent_dim)
        # The silhouette costs pass their per-frame data (boundary/SDF crops,
        # confidences, bridge transforms) as BATCHED factory arguments, so
        # jaxls vectorizes one factor definition over T frames -- the optimized
        # tangent is (6+n_hinges) per frame, exactly what the threshold above
        # counts. Forcing conjugate_gradient while the silhouette factors are
        # active keeps the linear solve memory-bounded on long bouts where the
        # dense normal-equation factorization would blow up.
        if use_sil and self.linear_solver == "auto":
            linear_solver = "conjugate_gradient"

        # jaxls carries the CG tolerance by passing a ConjugateGradientConfig
        # *as* the linear_solver argument (it extracts the config and sets the
        # solver to "conjugate_gradient" internally).
        linear_solver_arg = linear_solver
        if linear_solver == "conjugate_gradient":
            linear_solver_arg = jaxls.ConjugateGradientConfig(
                tolerance_max=self.cg_tolerance_max,
                tolerance_min=self.cg_tolerance_min)

        sol = analyzed.solve(
            verbose=self.verbose, linear_solver=linear_solver_arg,
            trust_region=jaxls.TrustRegionConfig(lambda_initial=self.lambda_initial),
            termination=jaxls.TerminationConfig(max_iterations=self.n_iter),
            initial_vals=jaxls.VarValues.make(init_list))
        sol_roots = sol[SE3Var(jnp.arange(T))]
        sol_joints = sol[JointVar(jnp.arange(T))]
        xyz_sol = sol_roots.translation(); wxyz_sol = sol_roots.rotation().wxyz
        return jnp.concatenate([xyz_sol, wxyz_sol, sol_joints], axis=-1)
