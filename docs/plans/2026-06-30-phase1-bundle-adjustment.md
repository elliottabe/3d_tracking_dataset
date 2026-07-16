# Phase 1: Per-Recording Bundle Adjustment (Affine Camera Refinement) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine each recording's per-camera affine (telecentric) DLT calibration by bundle-adjusting it against multi-view keypoint tracks, improving multi-view reprojection accuracy for every downstream stage.

**Architecture:** The cameras are affine/orthographic (3×4 DLT, 3rd row `[0,0,0,1]`), so a camera factorizes as `P = [[K2·R[:2], t],[0,0,0,1]]` with `K2` a 2×2 upper-triangular shape matrix (magnification/shear/aspect), `R ∈ SO(3)` orientation, `t` a 2-D image translation. A jaxls Levenberg–Marquardt least-squares problem jointly optimizes per-camera `(R, t)` (configurably also `K2`) and one 3-D point per (keypoint, frame), minimizing robust reprojection error with a soft prior to the factory calibration (which also fixes the gauge). A guard reverts to factory calibration if mean reprojection error does not improve.

**Tech Stack:** Python, JAX, `jaxls` (LM solver, `SO3Var`), `jaxlie` (SO3 manifold), NumPy, SciPy (RQ decomposition), `cv2` (OpenCV YAML calib I/O), pytest.

## Global Constraints

- **Camera model is affine:** projection is `uv = K2·R[:2]·X + t` (NO perspective divide); 3×4 DLT matrix always has 3rd row exactly `[0,0,0,1]`. Verified on real calibration.
- **Refine-not-replace:** always keep a soft prior to the factory calibration; never optimize cameras free of that anchor (it also fixes the affine gauge).
- **Configurable refine-set**, default `pose` (R + t only; hold K2). Options: `pose`, `pose+K` (also K2), `full` (same as `pose+K` for affine).
- **Guard:** never emit a refined calibration whose mean reprojection error exceeds the factory calibration's — revert to factory in that case.
- **Environment:** runs on the `gpu-l40s` compute node; `micromamba activate 3d_tracking`; `unset LD_LIBRARY_PATH`. Tests are CPU-only (`JAX_PLATFORMS=cpu`).
- **Reuse, don't fork:** camera I/O and triangulation come from `jarvis_jax.geometry.reprojection_tool`; jaxls usage follows `stac-mjx/stac_core_jaxls.py` patterns (fixed observations via `stop_gradient`).
- New code lives under `third_party/jarvis_jax/jarvis_jax/cse/`; tests under `third_party/jarvis_jax/tests/`. All paths below are relative to repo root `3d_tracking_dataset/`.

---

## File structure

- `third_party/jarvis_jax/jarvis_jax/cse/affine_camera.py` **(new)** — pure affine-camera math: factor a 3×4 DLT matrix into `(K2, R, t)`, reconstruct it back, and project points (NumPy + a JAX projection helper). No jaxls.
- `third_party/jarvis_jax/jarvis_jax/cse/bundle_adjust.py` **(new)** — jaxls BA: variable classes, robust reprojection cost, camera prior cost, `solve_bundle_adjust(...)`, and the reprojection-error guard. Depends on `affine_camera`.
- `third_party/jarvis_jax/jarvis_jax/cse/run_bundle_adjust.py` **(new)** — CLI driver: load coco keypoints + factory calib for a recording, assemble observations, triangulate initial points, call `solve_bundle_adjust`, apply the guard, write refined `Cam*.yaml` + a JSON report.
- `third_party/jarvis_jax/tests/test_affine_camera.py` **(new)** — round-trip + projection-equivalence tests.
- `third_party/jarvis_jax/tests/test_bundle_adjust.py` **(new)** — synthetic recover-perturbed-cameras test + guard test.

Run all commands from `third_party/jarvis_jax/` unless noted. Test prefix for every pytest command:
`cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest`

---

## Task 1: Affine camera factor / reconstruct

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/affine_camera.py`
- Test: `third_party/jarvis_jax/tests/test_affine_camera.py`

**Interfaces:**
- Produces:
  - `factor_affine(P: np.ndarray(3,4)) -> (K2: np.ndarray(2,2), R: np.ndarray(3,3), t: np.ndarray(2,))` — `K2` upper-triangular with positive diagonal, `R ∈ SO(3)` (det +1), such that `P[:2,:3] ≈ K2 @ R[:2,:]` and `t = P[:2,3]`.
  - `reconstruct_affine(K2: np.ndarray(2,2), R: np.ndarray(3,3), t: np.ndarray(2,)) -> np.ndarray(3,4)` — inverse of `factor_affine`, 3rd row `[0,0,0,1]`.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_affine_camera.py`:

```python
import numpy as np
from jarvis_jax.tracking.affine_camera import factor_affine, reconstruct_affine

# A real telecentric calibration matrix (Cam2012630, 2026_03_18_15_31_22).
P_REAL = np.array([
    [8.1001000000000012, 0.0074869000000000012, -0.031773000000000003, -2.8279999999999998],
    [0.0093308000000000002, -8.0787999999999993, -0.17912, 462.77999999999997],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def test_factor_reconstruct_roundtrip():
    K2, R, t = factor_affine(P_REAL)
    P2 = reconstruct_affine(K2, R, t)
    assert np.allclose(P2, P_REAL, atol=1e-9), f"roundtrip mismatch:\n{P2}\nvs\n{P_REAL}"


def test_factor_R_is_rotation():
    K2, R, t = factor_affine(P_REAL)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "R not orthonormal"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9), "det(R) != 1"


def test_factor_K2_upper_triangular_positive_diag():
    K2, R, t = factor_affine(P_REAL)
    assert abs(K2[1, 0]) < 1e-12, "K2 not upper triangular"
    assert K2[0, 0] > 0 and K2[1, 1] > 0, "K2 diagonal not positive"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_affine_camera.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.tracking.affine_camera'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/affine_camera.py`:

```python
"""Affine (telecentric / orthographic) camera math.

A telecentric camera's 3x4 DLT matrix has 3rd row [0,0,0,1], so projection is
affine:  uv = M @ X + t,  M = P[:2,:3] (2x3), t = P[:2,3] (2,).
We factor M = K2 @ R[:2,:] where K2 is 2x2 upper-triangular (magnification /
shear / aspect) and R in SO(3) is the camera orientation. This separates the
"intrinsic" shape (K2) from the pose (R, t) for bundle adjustment.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import rq


def factor_affine(P: np.ndarray):
    """Factor a 3x4 affine DLT matrix into (K2, R, t). See module docstring."""
    P = np.asarray(P, dtype=np.float64)
    assert P.shape == (3, 4)
    M = P[:2, :3]                      # (2,3)
    t = P[:2, 3].copy()               # (2,)
    K2, Q = rq(M)                     # M = K2 @ Q ; K2 (2,2) upper-tri, Q (2,3) orthonormal rows
    # Normalize signs so K2 has a positive diagonal (absorb sign flips into Q).
    S = np.diag(np.sign(np.diag(K2)))
    S[S == 0] = 1.0
    K2 = K2 @ S
    Q = S @ Q
    # Extend the two orthonormal rows to a full right-handed rotation.
    r2 = np.cross(Q[0], Q[1])
    R = np.vstack([Q, r2])
    if np.linalg.det(R) < 0:         # enforce det(R) = +1
        R[2] = -R[2]
    return K2, R, t


def reconstruct_affine(K2: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Inverse of factor_affine: build the 3x4 affine DLT matrix."""
    M = np.asarray(K2, np.float64) @ np.asarray(R, np.float64)[:2, :]   # (2,3)
    P = np.zeros((3, 4), dtype=np.float64)
    P[:2, :3] = M
    P[:2, 3] = np.asarray(t, np.float64)
    P[2, 3] = 1.0
    return P
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_affine_camera.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/affine_camera.py third_party/jarvis_jax/tests/test_affine_camera.py
git commit -m "feat(cse): affine camera factor/reconstruct for bundle adjustment"
```

---

## Task 2: Affine projection (NumPy + JAX) and ReprojectionTool equivalence

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/affine_camera.py`
- Test: `third_party/jarvis_jax/tests/test_affine_camera.py`

**Interfaces:**
- Consumes: `factor_affine`, `reconstruct_affine` (Task 1).
- Produces:
  - `project_affine(P: np.ndarray(3,4), X: np.ndarray(...,3)) -> np.ndarray(...,2)` — affine projection (no perspective divide), batched over leading dims.
  - `project_from_params(K2, R_mat, t, X)` — JAX-friendly: `R_mat` a (3,3) rotation matrix, returns `(...,2)`; differentiable. Used by the BA cost.

- [ ] **Step 1: Write the failing test**

Append to `third_party/jarvis_jax/tests/test_affine_camera.py`:

```python
def test_project_affine_matches_reprojection_tool(tmp_path):
    import cv2
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.tracking.affine_camera import project_affine
    # Write P_REAL as a one-camera calib dir and compare projections.
    d = tmp_path / "calib"; d.mkdir()
    fs = cv2.FileStorage(str(d / "Cam0001.yaml"), cv2.FILE_STORAGE_WRITE)
    fs.write("projectionMatrix", P_REAL); fs.release()
    rt = ReprojectionTool(str(d))
    X = np.array([1.5, -0.7, 12.0])
    uv_tool = rt.reproject_point(X)[0]            # (2,)
    uv_aff = project_affine(P_REAL, X)            # (2,)
    assert np.allclose(uv_tool, uv_aff, atol=1e-9), f"{uv_tool} vs {uv_aff}"


def test_project_from_params_matches_project_affine():
    import jax.numpy as jnp
    from jarvis_jax.tracking.affine_camera import factor_affine, project_affine, project_from_params
    K2, R, t = factor_affine(P_REAL)
    X = np.array([[1.5, -0.7, 12.0], [0.2, 0.3, 9.0]])
    uv_np = project_affine(P_REAL, X)                                   # (2,2)
    uv_jx = np.asarray(project_from_params(jnp.asarray(K2), jnp.asarray(R),
                                           jnp.asarray(t), jnp.asarray(X)))
    assert np.allclose(uv_np, uv_jx, atol=1e-6), f"{uv_np} vs {uv_jx}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_affine_camera.py -k project -v`
Expected: FAIL with `ImportError: cannot import name 'project_affine'`.

- [ ] **Step 3: Write minimal implementation**

Append to `third_party/jarvis_jax/jarvis_jax/cse/affine_camera.py`:

```python
def project_affine(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Affine projection uv = P[:2,:3] @ X + P[:2,3] (no perspective divide)."""
    P = np.asarray(P, np.float64); X = np.asarray(X, np.float64)
    return X @ P[:2, :3].T + P[:2, 3]


def project_from_params(K2, R_mat, t, X):
    """JAX-friendly affine projection from (K2 (2,2), R_mat (3,3), t (2,), X (...,3))."""
    import jax.numpy as jnp
    M = K2 @ R_mat[:2, :]                      # (2,3)
    return X @ M.T + t                         # (...,2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_affine_camera.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/affine_camera.py third_party/jarvis_jax/tests/test_affine_camera.py
git commit -m "feat(cse): affine projection (numpy + jax) matching ReprojectionTool"
```

---

## Task 3: Observation assembly + initial triangulation

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/bundle_adjust.py`
- Test: `third_party/jarvis_jax/tests/test_bundle_adjust.py`

**Interfaces:**
- Consumes: `affine_camera.factor_affine`, `ReprojectionTool.reconstruct_point`.
- Produces:
  - `Observations` dataclass with fields: `cam_idx: np.int32(n_obs)`, `point_idx: np.int32(n_obs)`, `uv: np.float64(n_obs,2)`, `n_cams: int`, `n_points: int`.
  - `assemble_observations(kp2d: np.ndarray(F,C,K,3), min_cams: int = 2) -> Observations` — `kp2d[f,c,k] = (x,y,visible)`; one BA point per `(frame f, keypoint k)` that ≥`min_cams` cameras see; one observation per visible `(point, camera)`.
  - `initial_points(obs: Observations, cam_mats: list[np.ndarray(3,4)]) -> np.ndarray(n_points,3)` — DLT-triangulate each point from its observing cameras.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_bundle_adjust.py`:

```python
import numpy as np
from jarvis_jax.tracking.affine_camera import reconstruct_affine, factor_affine, project_affine
from jarvis_jax.tracking.bundle_adjust import assemble_observations, initial_points

P_REAL = np.array([
    [8.1001, 0.0074869, -0.031773, -2.828],
    [0.0093308, -8.0788, -0.17912, 462.78],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def _two_cam_rig():
    # camera 0 = P_REAL; camera 1 = P_REAL with a 25-degree yaw applied to R.
    K2, R, t = factor_affine(P_REAL)
    th = np.deg2rad(25.0)
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    return [P_REAL, reconstruct_affine(K2, Ry @ R, t)]


def test_assemble_and_triangulate_recovers_points():
    cams = _two_cam_rig()
    rng = np.random.default_rng(0)
    pts = rng.uniform([-3, -3, 8], [3, 3, 14], size=(5, 3))   # 5 world points
    F, C, K = 1, 2, 5
    kp2d = np.zeros((F, C, K, 3))
    for c, P in enumerate(cams):
        uv = project_affine(P, pts)                            # (5,2)
        kp2d[0, c, :, :2] = uv; kp2d[0, c, :, 2] = 1.0
    obs = assemble_observations(kp2d, min_cams=2)
    assert obs.n_points == 5 and obs.n_cams == 2 and obs.uv.shape[0] == 10
    X0 = initial_points(obs, cams)
    assert np.allclose(X0, pts, atol=1e-6), f"triangulation off:\n{X0}\nvs\n{pts}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_bundle_adjust.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.tracking.bundle_adjust'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/bundle_adjust.py`:

```python
"""Per-recording affine-camera bundle adjustment (jaxls LM).

Factors each camera into (K2, R, t) (affine_camera), then jointly optimizes
per-camera (R, t) [optionally K2] and one 3-D point per (frame, keypoint) to
minimize robust reprojection error with a soft prior to the factory cameras.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool  # noqa: F401  (used by callers)


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_bundle_adjust.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/bundle_adjust.py third_party/jarvis_jax/tests/test_bundle_adjust.py
git commit -m "feat(cse): BA observation assembly + initial triangulation"
```

---

## Task 4: jaxls bundle-adjustment solve (recover perturbed cameras)

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/bundle_adjust.py`
- Test: `third_party/jarvis_jax/tests/test_bundle_adjust.py`

**Interfaces:**
- Consumes: `Observations`, `initial_points` (Task 3); `affine_camera.factor_affine`, `reconstruct_affine`, `project_from_params` (Tasks 1–2).
- Produces:
  - `solve_bundle_adjust(obs, cam_mats, *, refine="pose", huber_delta=4.0, rot_prior=50.0, trans_prior=5.0, k_prior=1e4, n_iter=80) -> list[np.ndarray(3,4)]` — returns refined 3×4 matrices (same order/length as `cam_mats`). `refine` in {`"pose"`, `"pose+K"`, `"full"`} (`"full"` == `"pose+K"` for affine).
  - `mean_reproj_error(obs, cam_mats, points) -> float` — mean L2 pixel error of `points` reprojected into observing cameras.

- [ ] **Step 1: Write the failing test**

Append to `third_party/jarvis_jax/tests/test_bundle_adjust.py`:

```python
def test_solve_recovers_perturbed_cameras():
    from jarvis_jax.tracking.bundle_adjust import solve_bundle_adjust, mean_reproj_error, initial_points
    true_cams = _two_cam_rig()
    rng = np.random.default_rng(1)
    pts = rng.uniform([-3, -3, 8], [3, 3, 14], size=(40, 3))
    F, C, K = 1, 2, 40
    kp2d = np.zeros((F, C, K, 3))
    for c, P in enumerate(true_cams):
        kp2d[0, c, :, :2] = project_affine(P, pts); kp2d[0, c, :, 2] = 1.0
    obs = assemble_observations(kp2d, min_cams=2)

    # Perturb camera 1's orientation by ~3 degrees -> reprojection error appears.
    K2, R, t = factor_affine(true_cams[1])
    th = np.deg2rad(3.0)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    init_cams = [true_cams[0], reconstruct_affine(K2, Rz @ R, t)]

    err_before = mean_reproj_error(obs, init_cams, initial_points(obs, init_cams))
    refined = solve_bundle_adjust(obs, init_cams, refine="pose", n_iter=120)
    err_after = mean_reproj_error(obs, refined, initial_points(obs, refined))
    assert err_before > 1.0, f"expected a real perturbation, got {err_before}"
    assert err_after < 0.2 * err_before, f"BA did not reduce error: {err_before} -> {err_after}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_bundle_adjust.py -k solve -v`
Expected: FAIL with `ImportError: cannot import name 'solve_bundle_adjust'`.

- [ ] **Step 3: Write minimal implementation**

Append to `third_party/jarvis_jax/jarvis_jax/cse/bundle_adjust.py`:

```python
def mean_reproj_error(obs: Observations, cam_mats, points) -> float:
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
    """Affine-camera bundle adjustment via jaxls LM. See module docstring."""
    import jax, jax.numpy as jnp, jaxls, jaxlie
    from jarvis_jax.tracking.affine_camera import factor_affine, reconstruct_affine, project_from_params

    C, n_pts = obs.n_cams, obs.n_points
    K2_0 = np.zeros((C, 2, 2)); R_0 = np.zeros((C, 3, 3)); t_0 = np.zeros((C, 2))
    for c in range(C):
        K2_0[c], R_0[c], t_0[c] = factor_affine(cam_mats[c])
    X0 = initial_points(obs, cam_mats)
    refine_K = refine in ("pose+K", "full")

    class CamRot(jaxls.Var[jaxlie.SO3], default_factory=lambda: jaxlie.SO3.identity()): ...
    class CamTrans(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros(2)): ...
    class CamK(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros(3)): ...   # [k00,k01,k11]
    class Point(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros(3)): ...
    class Obs(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros(2)): ...

    def k_vec_to_mat(kv):
        return jnp.array([[kv[0], kv[1]], [0.0, kv[2]]])

    @jaxls.Cost.factory
    def reproj_cost(vals, rot: CamRot, trans: CamTrans, kk: CamK, pt: Point, ob: Obs):
        R = vals[rot].as_matrix()                      # (3,3)
        kv = vals[kk] if refine_K else jax.lax.stop_gradient(vals[kk])
        pred = project_from_params(k_vec_to_mat(kv), R, vals[trans],
                                   vals[pt])            # (2,)
        r = pred - jax.lax.stop_gradient(vals[ob])
        w = jnp.minimum(1.0, huber_delta / (jnp.linalg.norm(r) + 1e-9))  # IRLS Huber
        return jnp.sqrt(jax.lax.stop_gradient(w)) * r

    @jaxls.Cost.factory
    def cam_prior_cost(vals, rot: CamRot, trans: CamTrans, kk: CamK,
                       rot0: CamRot, trans0: CamTrans, kk0: CamK):
        dR = (jax.lax.stop_gradient(vals[rot0]).inverse() @ vals[rot]).log()  # (3,)
        dt = vals[trans] - jax.lax.stop_gradient(vals[trans0])
        dk = vals[kk] - jax.lax.stop_gradient(vals[kk0]) if refine_K else jnp.zeros(3)
        return jnp.concatenate([rot_prior * dR, trans_prior * dt, k_prior * dk])

    so3_0 = [jaxlie.SO3.from_matrix(jnp.asarray(R_0[c])) for c in range(C)]
    kvec_0 = jnp.asarray([[K2_0[c, 0, 0], K2_0[c, 0, 1], K2_0[c, 1, 1]] for c in range(C)])

    rot = CamRot(jnp.asarray(obs.cam_idx)); trans = CamTrans(jnp.asarray(obs.cam_idx))
    kk = CamK(jnp.asarray(obs.cam_idx)); pt = Point(jnp.asarray(obs.point_idx))
    ob = Obs(jnp.arange(len(obs.uv)))
    rotC = CamRot(jnp.arange(C)); transC = CamTrans(jnp.arange(C)); kkC = CamK(jnp.arange(C))
    # separate "factory" var ids for the prior (held fixed via stop_gradient)
    rot0 = CamRot(jnp.arange(C)); trans0 = CamTrans(jnp.arange(C)); kk0 = CamK(jnp.arange(C))

    costs = [reproj_cost(rot, trans, kk, pt, ob),
             cam_prior_cost(rotC, transC, kkC, rot0, trans0, kk0)]
    variables = [rotC, transC, kkC, Point(jnp.arange(n_pts)), Obs(jnp.arange(len(obs.uv)))]
    problem = jaxls.LeastSquaresProblem(costs=costs, variables=variables).analyze()

    init = jaxls.VarValues.make([
        CamRot(jnp.arange(C)).with_value(jaxlie.SO3(jnp.stack([s.wxyz for s in so3_0]))),
        CamTrans(jnp.arange(C)).with_value(jnp.asarray(t_0)),
        CamK(jnp.arange(C)).with_value(kvec_0),
        Point(jnp.arange(n_pts)).with_value(jnp.asarray(X0)),
        Obs(jnp.arange(len(obs.uv))).with_value(jnp.asarray(obs.uv)),
    ])
    sol = problem.solve(initial_vals=init, linear_solver="dense_cholesky",
                        trust_region=jaxls.TrustRegionConfig(),
                        termination=jaxls.TerminationConfig(max_iterations=n_iter),
                        verbose=False)

    out = []
    for c in range(C):
        R = np.asarray(sol[CamRot(jnp.array(c))].as_matrix())
        tt = np.asarray(sol[CamTrans(jnp.array(c))])
        kv = np.asarray(sol[CamK(jnp.array(c))])
        K2 = np.array([[kv[0], kv[1]], [0.0, kv[2]]])
        out.append(reconstruct_affine(K2, R, tt))
    return out
```

> Note for the implementer: `jaxlie.SO3` stores `wxyz`; `SO3(jnp.stack([s.wxyz ...]))` builds the batched value. If the installed jaxls/jaxlie version rejects a batched `with_value`, fall back to `jaxls.VarValues.make` over per-camera singletons. Verify the API in Step 4; the test is the gate.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_bundle_adjust.py -k solve -v`
Expected: PASS. If it fails on a jaxls API mismatch (batched `with_value`, `trust_region`/`termination` kwarg names), adjust to the installed `jaxls` API per `stac-mjx/stac_core_jaxls.py` (which calls `.solve(...)` the same way) and re-run until PASS.

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/bundle_adjust.py third_party/jarvis_jax/tests/test_bundle_adjust.py
git commit -m "feat(cse): jaxls affine bundle adjustment (recover perturbed cameras)"
```

---

## Task 5: Guarded refine (revert on regression)

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/bundle_adjust.py`
- Test: `third_party/jarvis_jax/tests/test_bundle_adjust.py`

**Interfaces:**
- Consumes: `solve_bundle_adjust`, `mean_reproj_error`, `initial_points`.
- Produces:
  - `refine_calibration(obs, cam_mats, **kw) -> (refined: list[np.ndarray(3,4)], report: dict)` — runs BA, compares mean reprojection error before/after (each side re-triangulated with its own cameras), and **returns the factory cameras unchanged if error did not improve**. `report` has keys `err_before`, `err_after`, `improved` (bool), `n_points`, `n_obs`, `refine`.

- [ ] **Step 1: Write the failing test**

Append to `third_party/jarvis_jax/tests/test_bundle_adjust.py`:

```python
def test_refine_reverts_when_no_improvement():
    from jarvis_jax.tracking.bundle_adjust import refine_calibration
    # Perfect data + perfect cameras: BA cannot improve -> must return factory cams.
    cams = _two_cam_rig()
    rng = np.random.default_rng(2)
    pts = rng.uniform([-3, -3, 8], [3, 3, 14], size=(30, 3))
    kp2d = np.zeros((1, 2, 30, 3))
    for c, P in enumerate(cams):
        kp2d[0, c, :, :2] = project_affine(P, pts); kp2d[0, c, :, 2] = 1.0
    obs = assemble_observations(kp2d, min_cams=2)
    refined, report = refine_calibration(obs, cams, refine="pose")
    assert report["err_before"] < 1e-4
    assert not report["improved"]
    for a, b in zip(refined, cams):
        assert np.allclose(a, b), "should revert to factory cameras"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_bundle_adjust.py -k revert -v`
Expected: FAIL with `ImportError: cannot import name 'refine_calibration'`.

- [ ] **Step 3: Write minimal implementation**

Append to `third_party/jarvis_jax/jarvis_jax/cse/bundle_adjust.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_bundle_adjust.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/bundle_adjust.py third_party/jarvis_jax/tests/test_bundle_adjust.py
git commit -m "feat(cse): guarded calibration refine (revert on regression)"
```

---

## Task 6: CLI driver + real-recording validation

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/run_bundle_adjust.py`
- Test: `third_party/jarvis_jax/tests/test_bundle_adjust.py`

**Interfaces:**
- Consumes: `ReprojectionTool`, `assemble_observations`, `refine_calibration`.
- Produces:
  - `load_kp2d_from_coco(root, split, recording, kp_names_out=None, max_frames=0) -> (kp2d: np.ndarray(F,C,K,3), cam_names: list[str])` — read coco framesets for `recording`, build the `(F,C,K,3)` keypoint array in `ReprojectionTool` camera order (matched by `Cam<id>` in the file path); `max_frames>0` evenly subsamples framesets.
  - `write_calibration(out_dir, cam_names, mats)` — write `Cam<name>.yaml` with `projectionMatrix` (OpenCV FileStorage).
  - `main()` — Hydra-free argparse CLI: `--root --recording --split --calib-out --report-out --refine --max-frames`.

- [ ] **Step 1: Write the failing test**

Append this complete test to `third_party/jarvis_jax/tests/test_bundle_adjust.py`:

```python
def test_load_kp2d_orders_cameras_like_reprojection_tool(tmp_path):
    import json, cv2
    from jarvis_jax.tracking.run_bundle_adjust import load_kp2d_from_coco
    root = tmp_path; rec = "RECX"
    (root / "annotations").mkdir(parents=True)
    (root / "calib_params" / rec).mkdir(parents=True)
    for nm in ("Cam0002", "Cam0001"):                   # intentionally unsorted on disk
        fs = cv2.FileStorage(str(root / "calib_params" / rec / f"{nm}.yaml"), cv2.FILE_STORAGE_WRITE)
        fs.write("projectionMatrix", P_REAL); fs.release()
    # 2 cameras of the same frame; each annotation carries 2 keypoints (x,y,vis).
    images = [
        {"id": 1, "file_name": f"{rec}/Cam0001/F0.jpg"},
        {"id": 2, "file_name": f"{rec}/Cam0002/F0.jpg"},
    ]
    anns = [
        {"image_id": 1, "keypoints": [10, 20, 2, 30, 40, 2]},
        {"image_id": 2, "keypoints": [11, 21, 2, 31, 41, 2]},
    ]
    coco = {"keypoint_names": ["a", "b"], "images": images, "annotations": anns,
            "framesets": {f"{rec}/F0": {"datasetName": rec, "frames": [1, 2]}}}
    (root / "annotations" / "instances_val.json").write_text(json.dumps(coco))
    kp2d, cam_names = load_kp2d_from_coco(str(root), "val", rec)
    assert cam_names == ["Cam0001", "Cam0002"]          # ReprojectionTool sorted order
    assert kp2d.shape == (1, 2, 2, 3)
    assert list(kp2d[0, 0, 0, :2]) == [10, 20]          # cam0 = Cam0001
    assert list(kp2d[0, 1, 0, :2]) == [11, 21]          # cam1 = Cam0002
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_bundle_adjust.py -k orders_cameras -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.tracking.run_bundle_adjust'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/run_bundle_adjust.py`:

```python
"""CLI: per-recording affine-camera bundle adjustment from coco keypoints.

    python -m jarvis_jax.tracking.run_bundle_adjust \
        --root /.../red_data_unified_V3 --recording 2026_03_18_15_31_22 --split val \
        --calib-out /.../calib_refined/2026_03_18_15_31_22 \
        --report-out /.../calib_refined/2026_03_18_15_31_22/ba_report.json \
        --refine pose --max-frames 300
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import cv2
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.tracking.bundle_adjust import assemble_observations, refine_calibration


def load_kp2d_from_coco(root, split, recording, max_frames=0):
    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann = {an["image_id"]: an for an in coco["annotations"]}
    K = len(coco["keypoint_names"])
    rt = ReprojectionTool(os.path.join(root, "calib_params", recording))
    cam_names = list(rt.cameras.keys())
    fs_items = [(k, v) for k, v in coco["framesets"].items()
                if v.get("datasetName") == recording]
    if max_frames and len(fs_items) > max_frames:
        idx = np.linspace(0, len(fs_items) - 1, max_frames).astype(int)
        fs_items = [fs_items[i] for i in idx]
    F, C = len(fs_items), len(cam_names)
    kp2d = np.zeros((F, C, K, 3), np.float64)
    for f, (_, fv) in enumerate(fs_items):
        for iid in fv["frames"]:
            fn = id2file.get(int(iid), "")
            cam = fn.split("/")[1] if "/" in fn else ""
            if cam not in cam_names:
                continue
            ann = id2ann.get(int(iid))
            if ann is None:
                continue
            kp = np.asarray(ann["keypoints"], float).reshape(-1, 3)
            kp2d[f, cam_names.index(cam)] = kp
    return kp2d, cam_names


def write_calibration(out_dir, cam_names, mats):
    os.makedirs(out_dir, exist_ok=True)
    for nm, P in zip(cam_names, mats):
        fs = cv2.FileStorage(os.path.join(out_dir, f"{nm}.yaml"), cv2.FILE_STORAGE_WRITE)
        fs.write("projectionMatrix", np.asarray(P, np.float64)); fs.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--recording", required=True)
    ap.add_argument("--split", default="val"); ap.add_argument("--calib-out", required=True)
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--refine", default="pose", choices=["pose", "pose+K", "full"])
    ap.add_argument("--max-frames", type=int, default=300)
    a = ap.parse_args()
    rt = ReprojectionTool(os.path.join(a.root, "calib_params", a.recording))
    cam_mats = [c.cameraMatrix for c in rt._camera_list]
    kp2d, cam_names = load_kp2d_from_coco(a.root, a.split, a.recording, a.max_frames)
    obs = assemble_observations(kp2d, min_cams=2)
    refined, report = refine_calibration(obs, cam_mats, refine=a.refine)
    write_calibration(a.calib_out, cam_names, refined)
    report["recording"] = a.recording
    print(f"BA {a.recording}: err {report['err_before']:.3f} -> {report['err_after']:.3f} px "
          f"(improved={report['improved']}, n_pts={report['n_points']})")
    if a.report_out:
        os.makedirs(os.path.dirname(a.report_out), exist_ok=True)
        json.dump(report, open(a.report_out, "w"), indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_bundle_adjust.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Real-recording validation (gpu-l40s compute node)**

Run (after `micromamba activate 3d_tracking && unset LD_LIBRARY_PATH`):

```bash
cd third_party/jarvis_jax
python -m jarvis_jax.tracking.run_bundle_adjust \
  --root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3 \
  --recording 2026_03_18_15_31_22 --split val \
  --calib-out /gscratch/portia/eabe/data/Johnson_lab/cse_work/calib_refined/2026_03_18_15_31_22 \
  --report-out /gscratch/portia/eabe/data/Johnson_lab/cse_work/calib_refined/2026_03_18_15_31_22/ba_report.json \
  --refine pose --max-frames 300
```

Expected: prints `BA 2026_03_18_15_31_22: err X.XXX -> Y.YYY px (improved=True, ...)` with `Y < X` (reprojection error drops; this is the Phase-1 success criterion). If `improved=False`, the guard kept factory calibration — inspect `ba_report.json` and the keypoint visibility before tuning priors (`rot_prior`/`trans_prior`) or `--refine pose+K`.

- [ ] **Step 6: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/run_bundle_adjust.py third_party/jarvis_jax/tests/test_bundle_adjust.py
git commit -m "feat(cse): bundle-adjustment CLI + coco loader; validate on val recording"
```

---

## Self-review notes (spec coverage)

- Affine DLT model (3×4, `[0,0,0,1]`) — Tasks 1–2. Robust (Huber) reprojection — Task 4. Soft prior to factory + gauge — Task 4 (`cam_prior_cost`). Configurable refine-set (default `pose`) — Task 4 (`refine`). Reproject-before/after guard with revert — Task 5. Inputs from coco keypoints + factory `ReprojectionTool` — Tasks 3, 6. Validation on `2026_03_18_15_31_22` by reprojection drop — Task 6 Step 5. Reuse of `reprojection_tool` + stac jaxls patterns — throughout.
- Out of Phase-1 scope (later phases): using ViTPose *predicted* keypoints (here we use coco GT keypoints — same array shape, swappable), and consuming the refined calibration in the IK.
