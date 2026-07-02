"""Sibling joint 2D+3D silhouette IK solver (Phase 6).

Assembles its OWN jaxls LeastSquaresProblem reusing the SAME cost formulas as
stac_core_jaxls.JaxlsBatchSolver._build_se3 (marker / smoothness / limit / reg)
PLUS a differentiable silhouette boundary-Chamfer cost, over the SAME SE3Var +
JointVar. stac_core_jaxls.py is left byte-identical (Phases 1-4 invariant); a
test proves silhouette_weight=0 reproduces JaxlsBatchSolver's output.

This first section provides the silhouette cost factory + the SilVar packing
helpers. The SilhouetteJaxlsBatchSolver class is added in Task 4.
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


def SIL_PER_CAM(n_pts: int) -> int:
    """Flat SilVar length per camera: M(2x3=6) + t(2) + boundary(n_pts*2)."""
    return 6 + 2 + n_pts * 2


def pack_sil_value(cam_mats, boundary_pts):
    """Pack per-frame silhouette data into the flat SilVar layout.

    Args:
        cam_mats: length-n_cam list of (3,4) affine DLT matrices.
        boundary_pts: (n_cam, n_pts, 2) sampled SAM boundary points (NaN rows
            allowed for cameras that don't see this fly this frame).

    Returns:
        np.ndarray (n_cam * SIL_PER_CAM(n_pts),): per camera
        [M.flatten()(6), t(2), boundary.flatten()(n_pts*2)].
    """
    import numpy as np
    boundary_pts = np.asarray(boundary_pts, dtype=np.float64)
    n_cam, n_pts, _ = boundary_pts.shape
    rows = []
    for c in range(n_cam):
        P = np.asarray(cam_mats[c], dtype=np.float64)
        M = P[:2, :3].reshape(-1)   # (6,)
        t = P[:2, 3].reshape(-1)    # (2,)
        rows.append(np.concatenate([M, t, boundary_pts[c].reshape(-1)]))
    return np.concatenate(rows).astype(np.float64)


def unpack_sil_value(sil_flat, n_cam: int, n_pts: int):
    """Inverse of pack_sil_value for one frame's flat SilVar value.

    Returns (Ms, ts, pts): Ms is (n_cam,2,3), ts is (n_cam,2), pts is
    (n_cam,n_pts,2). Works on jnp arrays (traceable).
    """
    per = SIL_PER_CAM(n_pts)
    blocks = sil_flat.reshape(n_cam, per)
    Ms = blocks[:, :6].reshape(n_cam, 2, 3)
    ts = blocks[:, 6:8]
    pts = blocks[:, 8:].reshape(n_cam, n_pts, 2)
    return Ms, ts, pts


def make_silhouette_cost(
    SE3Var, JointVar, SilVar, *,
    fk_repose, vert_indices, n_cam: int, n_pts: int,
    beta: float, huber_delta: float, silhouette_weight: float,
    qs_to_opt, template_qpos, scale: float = 1.0, chunk_size: int = 32,
):
    """Return a jaxls Cost.factory fn: FK verts -> project_affine -> Chamfer.

    Per frame residual shape: (n_cam * n_pts,), weighted by silhouette_weight.
    Reconstructs full_q exactly as stac_core_jaxls._build_se3.marker_cost.
    vert_indices are FULL-vertex-array indices (see Task 6 index-space audit).

    MEMORY BOUND (efficiency Global Constraint): the C cameras are iterated with
    jax.lax.scan (each camera projects the M verts and calls chunked
    chamfer_residual, appending its (n_pts,) residual to the scan output), NOT by
    jax.vmap-stacking a (C, N, M) tensor. jaxls vmaps this whole factor over T
    frames, so a (C, N, M) intermediate would peak at O(T * C * N * M); the scan
    + chunked Chamfer keeps the peak at O(chunk_size * M) per camera step. FK
    runs ONCE per frame (verts3d is closed over, reused across the camera scan).
    Fully jit-fusable; no host callbacks in the traced step.
    """
    vert_indices = jnp.asarray(vert_indices)
    template_qpos = jnp.asarray(template_qpos)
    qs_to_opt = jnp.asarray(qs_to_opt)

    @jaxls.Cost.factory
    def silhouette_cost(
        var_values: jaxls.VarValues,
        root_var: SE3Var,
        joint_var: JointVar,
        sil_var: SilVar,
    ) -> jnp.ndarray:
        T_root = var_values[root_var]                    # jaxlie.SE3
        joints = var_values[joint_var]                   # (n_hinges,)
        sil = jax.lax.stop_gradient(var_values[sil_var])  # (n_cam*SIL_PER_CAM,)

        xyz = T_root.translation()
        wxyz = T_root.rotation().wxyz
        q = jnp.concatenate([xyz, wxyz, joints])         # (nq,)
        full_q = jnp.where(qs_to_opt, q, template_qpos)

        verts3d = fk_repose(full_q, scale, vert_indices)  # (M,3) FK once per frame

        Ms, ts, pts = unpack_sil_value(sil, n_cam, n_pts)  # (C,2,3),(C,2),(C,n_pts,2)

        # Scan over the C cameras; each step projects verts (O(M)) and runs the
        # chunked Chamfer (O(chunk_size*M)). No (C,N,M) tensor is ever built.
        def scan_body(carry, cam):
            M, t, tgt = cam                               # (2,3),(2,),(n_pts,2)
            proj = verts3d @ M.T + t                      # (M_verts, 2) affine project
            r = chamfer_residual(tgt, proj, beta=beta, huber_delta=huber_delta,
                                 chunk_size=chunk_size)   # (n_pts,)
            return carry, r

        _, res = jax.lax.scan(scan_body, None, (Ms, ts, pts))  # res: (n_cam, n_pts)
        return (res * silhouette_weight).reshape(-1)      # (n_cam*n_pts,)

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
        fk_repose=None, vert_indices=None, sil_data=None, n_pts=0,
        silhouette_weight=0.0,
    ):
        if kp_data.ndim == 3:
            kp_data = kp_data.reshape(kp_data.shape[0], -1)
        q_init = jnp.asarray(q_init)
        kp_data = jnp.asarray(kp_data)
        qs_to_opt = jnp.asarray(qs_to_opt)
        kps_to_opt = jnp.asarray(kps_to_opt)
        lb = jnp.asarray(lb); ub = jnp.asarray(ub)
        q_reg_weights = jnp.asarray(q_reg_weights)

        T = q_init.shape[0]
        nq = int(mjx_model.nq)
        n_kp_dim = int(kp_data.shape[-1])
        n_hinges = nq - _FREE_JOINT_NDOF
        smooth_weight = self.smooth_weight

        use_sil = (silhouette_weight != 0.0) and (sil_data is not None)

        dummy_joints = jnp.zeros((n_hinges,))
        dummy_kp = jnp.zeros((n_kp_dim,))

        # ---- Variable classes (identical to _build_se3) ----
        class SE3Var(
            jaxls.Var[jaxlie.SE3],
            default_factory=jaxlie.SE3.identity,
            retract_fn=jaxlie.manifold.rplus,
            tangent_dim=6,
        ): ...
        class JointVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_joints): ...
        class KpVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_kp): ...

        root_all = SE3Var(jnp.arange(T))
        joint_all = JointVar(jnp.arange(T))
        kp_all = KpVar(jnp.arange(T))

        costs: list[jaxls.Cost] = []

        # ---- marker_cost (VERBATIM from stac_core_jaxls._build_se3) ----
        @jaxls.Cost.factory
        def marker_cost(var_values, root_var: SE3Var, joint_var: JointVar, kp_var: KpVar):
            T_root = var_values[root_var]
            joints = var_values[joint_var]
            kp = jax.lax.stop_gradient(var_values[kp_var])
            xyz = T_root.translation()
            wxyz = T_root.rotation().wxyz
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

        # ---- reg_cost (VERBATIM) ----
        if jnp.any(q_reg_weights[_FREE_JOINT_NDOF:] > 0):
            hinge_regs = q_reg_weights[_FREE_JOINT_NDOF:]
            hinge_opt = qs_to_opt[_FREE_JOINT_NDOF:]

            @jaxls.Cost.factory
            def reg_cost(var_values, joint_var: JointVar):
                j = var_values[joint_var]
                return jnp.sqrt(hinge_regs * hinge_opt) * j

            costs.append(reg_cost(joint_all))

        # ---- limit_cost (VERBATIM) ----
        hinge_lb = lb[_FREE_JOINT_NDOF:]
        hinge_ub = ub[_FREE_JOINT_NDOF:]

        @jaxls.Cost.factory(kind="constraint_leq_zero")
        def limit_cost(var_values, joint_var: JointVar):
            j = var_values[joint_var]
            return jnp.concatenate([hinge_lb - j, j - hinge_ub])

        costs.append(limit_cost(joint_all))

        # ---- smoothness_cost (VERBATIM) ----
        if smooth_weight > 0.0 and T > 1:
            @jaxls.Cost.factory
            def smoothness_cost(var_values, root_curr: SE3Var, root_prev: SE3Var,
                                joint_curr: JointVar, joint_prev: JointVar):
                root_diff = (var_values[root_prev].inverse() @ var_values[root_curr]).log()
                joint_diff = var_values[joint_curr] - var_values[joint_prev]
                return jnp.concatenate([root_diff, joint_diff]) * smooth_weight

            costs.append(smoothness_cost(
                SE3Var(jnp.arange(1, T)), SE3Var(jnp.arange(0, T - 1)),
                JointVar(jnp.arange(1, T)), JointVar(jnp.arange(0, T - 1)),
            ))

        variables = [root_all, joint_all, kp_all]

        # ---- ADDITIVE silhouette cost (only when weight!=0 and data present) ----
        sil_all = None
        if use_sil:
            sil_data = jnp.asarray(sil_data)
            per_cam = SIL_PER_CAM(n_pts)
            n_cam = sil_data.shape[-1] // per_cam
            dummy_sil = jnp.zeros((sil_data.shape[-1],))

            class SilVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_sil): ...

            sil_all = SilVar(jnp.arange(T))
            sil_cost = make_silhouette_cost(
                SE3Var, JointVar, SilVar,
                fk_repose=fk_repose, vert_indices=vert_indices,
                n_cam=n_cam, n_pts=n_pts, beta=self.beta,
                huber_delta=self.huber_delta, silhouette_weight=silhouette_weight,
                qs_to_opt=qs_to_opt, template_qpos=mjx_data_template.qpos, scale=1.0,
            )
            costs.append(sil_cost(root_all, joint_all, sil_all))
            variables.append(sil_all)

        analyzed = jaxls.LeastSquaresProblem(costs=costs, variables=variables).analyze()

        # ---- initial values (root/joint/kp exactly like _solve_se3) ----
        xyz_init = q_init[:, :3]
        wxyz_init = q_init[:, 3:7]
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
            init_list.append(sil_all.with_value(sil_data))

        # SE3 path tangent_dim = 6 (root) + n_hinges per frame (matches _solve_se3).
        tangent_dim = 6 + n_hinges
        linear_solver = self._pick_linear_solver(T, tangent_dim)

        sol = analyzed.solve(
            verbose=False,
            linear_solver=linear_solver,
            trust_region=jaxls.TrustRegionConfig(lambda_initial=self.lambda_initial),
            termination=jaxls.TerminationConfig(max_iterations=self.n_iter),
            initial_vals=jaxls.VarValues.make(init_list),
        )
        sol_roots = sol[SE3Var(jnp.arange(T))]
        sol_joints = sol[JointVar(jnp.arange(T))]
        xyz_sol = sol_roots.translation()
        wxyz_sol = sol_roots.rotation().wxyz
        return jnp.concatenate([xyz_sol, wxyz_sol, sol_joints], axis=-1)
