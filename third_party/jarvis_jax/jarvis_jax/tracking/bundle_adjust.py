"""Per-recording affine-camera bundle adjustment (jaxls LM).

Factors each camera into (K2, R, t) (affine_camera), then jointly optimizes
per-camera (R, t) [optionally K2] and one 3-D point per (frame, keypoint) to
minimize robust reprojection error with a soft prior to the factory cameras.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Observations:
    cam_idx: np.ndarray       # (n_obs,) int32
    point_idx: np.ndarray     # (n_obs,) int32
    uv: np.ndarray            # (n_obs, 2) float64
    n_cams: int
    n_points: int


def assemble_observations(kp2d: np.ndarray, min_cams: int = 2) -> Observations:
    """kp2d[f,c,k] = (x, y, visible). One point per (f,k) seen by >= min_cams."""
    kp2d = np.asarray(kp2d)
    F, C, K, _ = kp2d.shape
    cam_idx, point_idx, uv = [], [], []
    pid = 0
    for f in range(F):
        for k in range(K):
            seen = [c for c in range(C) if kp2d[f, c, k, 2] > 0]
            if len(seen) < min_cams:
                continue
            for c in seen:
                cam_idx.append(c); point_idx.append(pid); uv.append(kp2d[f, c, k, :2])
            pid += 1
    return Observations(
        cam_idx=np.asarray(cam_idx, np.int32),
        point_idx=np.asarray(point_idx, np.int32),
        uv=np.asarray(uv, np.float64).reshape(-1, 2),
        n_cams=C, n_points=pid,
    )


def initial_points(obs: Observations, cam_mats) -> np.ndarray:
    """DLT-triangulate each point from the cameras that observe it."""
    X = np.zeros((obs.n_points, 3), np.float64)
    for pid in range(obs.n_points):
        m = obs.point_idx == pid
        cams = obs.cam_idx[m]; uvs = obs.uv[m]
        # Build (2n x 4) DLT system: uv * P[2] - P[0:2].
        A = np.zeros((2 * len(cams), 4))
        for i, (c, uvp) in enumerate(zip(cams, uvs)):
            P = np.asarray(cam_mats[c])
            A[2 * i:2 * i + 2] = uvp.reshape(2, 1) * P[2].reshape(1, 4) - P[0:2]
        _, _, Vh = np.linalg.svd(A)
        Xh = Vh[-1]
        X[pid] = (Xh / Xh[3])[:3]
    return X


def mean_reproj_error(obs: Observations, cam_mats, points) -> float:
    """Mean L2 pixel error of `points` reprojected into their observing cameras."""
    from jarvis_jax.tracking.affine_camera import project_affine
    points = np.asarray(points)
    errs = []
    for c, pid, uv in zip(obs.cam_idx, obs.point_idx, obs.uv):
        pred = project_affine(cam_mats[c], points[pid])
        errs.append(np.linalg.norm(pred - uv))
    return float(np.mean(errs)) if errs else 0.0


def solve_bundle_adjust(obs: Observations, cam_mats, *, refine="pose",
                         huber_delta=4.0, rot_prior=50.0, trans_prior=5.0,
                         k_prior=1e4, n_iter=80):
    """Affine-camera bundle adjustment via jaxls Levenberg-Marquardt.

    Factors each camera into (K2, R, t) (see `affine_camera.factor_affine`) and
    jointly refines per-camera (R, t) [and, if `refine` requests it, K2] plus
    one 3-D point per observed (frame, keypoint), minimizing a robust (Huber)
    reprojection residual with a soft prior pulling every camera back toward
    its factory (input) calibration. `refine` in {"pose", "pose+K", "full"};
    "full" == "pose+K" for an affine (telecentric) camera model.

    NOTE on jaxls API: this repo's installed `jaxls` is 0.0.0, which ships a
    ready-made `jaxls.SO3Var` (Var[jaxlie.SO3] with the rplus retraction) and
    expects `.analyze()` + `.solve(initial_vals=..., linear_solver=...,
    trust_region=jaxls.TrustRegionConfig(...), termination=
    jaxls.TerminationConfig(max_iterations=...))`, matching the pattern used in
    `stac_mjx/stac_core_jaxls.py`. We use `jaxls.SO3Var` directly instead of
    hand-rolling a `CamRot` variable class, and skip the separate "prior" var
    ids (`rot0`/`trans0`/`kk0`) from the brief's sketch: since the *only*
    per-camera unknowns are exactly `CamRot`/`CamTrans`/`CamK`, the prior can
    just close over the fixed initial numpy/jnp arrays instead of declaring
    duplicate jaxls variables for them (fewer moving parts, same effect: the
    prior target is a constant, not touched by autodiff).
    """
    import jax, jax.numpy as jnp, jaxls, jaxlie
    from jarvis_jax.tracking.affine_camera import factor_affine, reconstruct_affine, project_from_params

    C, n_pts = obs.n_cams, obs.n_points
    K2_0 = np.zeros((C, 2, 2)); R_0 = np.zeros((C, 3, 3)); t_0 = np.zeros((C, 2))
    for c in range(C):
        K2_0[c], R_0[c], t_0[c] = factor_affine(cam_mats[c])
    X0 = initial_points(obs, cam_mats)
    refine_K = refine in ("pose+K", "full")

    CamRot = jaxls.SO3Var

    class CamTrans(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros(2)): ...

    class CamK(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros(3)): ...  # [k00,k01,k11]

    class Point(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros(3)): ...

    # Observed pixel coordinates, held fixed via stop_gradient inside the cost.
    class Obs(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros(2)): ...

    # Constant (non-optimized) targets for the camera prior: factory calibration.
    rot0_const = jaxlie.SO3(jnp.asarray(np.stack([
        # R_0[c] -> quaternion via jaxlie, batched over cameras
        np.asarray(jaxlie.SO3.from_matrix(jnp.asarray(R_0[c])).wxyz) for c in range(C)
    ])))
    trans0_const = jnp.asarray(t_0)
    kvec0_const = jnp.asarray([[K2_0[c, 0, 0], K2_0[c, 0, 1], K2_0[c, 1, 1]] for c in range(C)])

    def k_vec_to_mat(kv):
        return jnp.array([[kv[0], kv[1]], [0.0, kv[2]]])

    @jaxls.Cost.factory
    def reproj_cost(vals: jaxls.VarValues, rot: CamRot, trans: CamTrans,
                     kk: CamK, pt: Point, ob: Obs) -> jnp.ndarray:
        R = vals[rot].as_matrix()                        # (3,3)
        kv = vals[kk] if refine_K else jax.lax.stop_gradient(vals[kk])
        pred = project_from_params(k_vec_to_mat(kv), R, vals[trans], vals[pt])  # (2,)
        r = pred - jax.lax.stop_gradient(vals[ob])
        # IRLS Huber weighting: weight computed on stopped residual so the
        # Jacobian sees a fixed scalar (standard IRLS linearization).
        w = jnp.minimum(1.0, huber_delta / (jnp.linalg.norm(r) + 1e-9))
        return jnp.sqrt(jax.lax.stop_gradient(w)) * r

    @jaxls.Cost.factory
    def cam_prior_cost(vals: jaxls.VarValues, rot: CamRot, trans: CamTrans,
                        kk: CamK, cam_id: jnp.ndarray) -> jnp.ndarray:
        rot0_c = jax.tree.map(lambda x: x[cam_id], rot0_const)
        trans0_c = trans0_const[cam_id]
        kk0_c = kvec0_const[cam_id]
        dR = (rot0_c.inverse() @ vals[rot]).log()          # (3,)
        dt = vals[trans] - trans0_c
        dk = (vals[kk] - kk0_c) if refine_K else jnp.zeros(3)
        return jnp.concatenate([rot_prior * dR, trans_prior * dt, k_prior * dk])

    cam_ids = jnp.arange(C)
    rotC, transC, kkC = CamRot(cam_ids), CamTrans(cam_ids), CamK(cam_ids)
    pt_all = Point(jnp.arange(n_pts))
    ob_all = Obs(jnp.arange(len(obs.uv)))

    costs = [
        reproj_cost(CamRot(jnp.asarray(obs.cam_idx)), CamTrans(jnp.asarray(obs.cam_idx)),
                    CamK(jnp.asarray(obs.cam_idx)), Point(jnp.asarray(obs.point_idx)), ob_all),
        cam_prior_cost(rotC, transC, kkC, cam_ids),
    ]
    variables = [rotC, transC, kkC, pt_all, ob_all]
    problem = jaxls.LeastSquaresProblem(costs=costs, variables=variables).analyze()

    so3_0 = jaxlie.SO3(jnp.asarray(np.stack([
        np.asarray(jaxlie.SO3.from_matrix(jnp.asarray(R_0[c])).wxyz) for c in range(C)
    ])))
    kvec_0 = jnp.asarray([[K2_0[c, 0, 0], K2_0[c, 0, 1], K2_0[c, 1, 1]] for c in range(C)])

    init = jaxls.VarValues.make([
        CamRot(cam_ids).with_value(so3_0),
        CamTrans(cam_ids).with_value(jnp.asarray(t_0)),
        CamK(cam_ids).with_value(kvec_0),
        pt_all.with_value(jnp.asarray(X0)),
        ob_all.with_value(jnp.asarray(obs.uv)),
    ])
    sol = problem.solve(
        initial_vals=init,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(max_iterations=n_iter),
        verbose=False,
    )

    out = []
    for c in range(C):
        R = np.asarray(sol[CamRot(jnp.array(c))].as_matrix())
        tt = np.asarray(sol[CamTrans(jnp.array(c))])
        kv = np.asarray(sol[CamK(jnp.array(c))])
        K2 = np.array([[kv[0], kv[1]], [0.0, kv[2]]])
        out.append(reconstruct_affine(K2, R, tt))
    return out


def refine_calibration(obs: Observations, cam_mats, *, improve_tol=0.99, **kw):
    """Run BA; revert to factory cameras unless mean reproj error improves."""
    err_before = mean_reproj_error(obs, cam_mats, initial_points(obs, cam_mats))
    refined = solve_bundle_adjust(obs, cam_mats, **kw)
    err_after = mean_reproj_error(obs, refined, initial_points(obs, refined))
    improved = err_after < improve_tol * err_before
    report = dict(err_before=err_before, err_after=err_after, improved=bool(improved),
                  n_points=int(obs.n_points), n_obs=int(len(obs.uv)),
                  refine=kw.get("refine", "pose"))
    return (refined if improved else [np.asarray(P) for P in cam_mats]), report
