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

from jarvis_jax.cse.silhouette_chamfer import chamfer_residual


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
