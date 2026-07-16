# Phase 6 — Dense-Silhouette Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an additive, off-by-default 2-D silhouette boundary-Chamfer factor to the fruit-fly multi-view IK so the collision-mesh outline is pulled out to meet the SAM mask boundary (recovers wing extent), solved JOINTLY with the existing 3-D marker cost over the same SE3/joint variables.

**Architecture:** A NEW sibling joint-solver module `stac-mjx/stac_mjx/stac_silhouette_jaxls.py` builds its own `jaxls.LeastSquaresProblem` that reuses the SAME cost formulas (marker/smoothness/limit/reg) as `stac_core_jaxls.JaxlsBatchSolver._build_se3` PLUS a new silhouette-Chamfer cost over the same `SE3Var`+`JointVar`. `stac_core_jaxls.py` is left byte-identical, and a dedicated test proves the no-silhouette path is unchanged. The silhouette residual is a one-directional sparse Chamfer (SAM-boundary point → nearest projected mesh vertex), differentiable via soft-min, with the mesh vertices FK'd by the vertex-level `silhouette_ik.make_fk_repose` FK (the same qpos reconstruction `marker_cost` uses).

**Tech Stack:** JAX / jaxls (Levenberg-Marquardt) / jaxlie SE3 / MuJoCo MJX FK; NumPy + scipy.ndimage/cv2 for offline mask-boundary extraction; pytest (CPU-only for unit tests).

## Global Constraints

- Env: gpu-l40s compute node; `source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH`; GPU on-node MEM_FRACTION=0.9; coordinator runs GPU work (subagents orphan on long GPU jobs).
- Unit tests CPU-only: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest`.
- New code under `third_party/jarvis_jax/jarvis_jax/cse/` (and geometry/ if a projection helper); tests under `third_party/jarvis_jax/tests/`. jaxls solver code lives in `stac-mjx/stac_mjx/`.
- TDD: failing test w/ REAL asserts -> run(fail) -> minimal impl -> run(pass) -> commit per task by EXPLICIT path (never git add -A). Commits end "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>".
- Do NOT commit checkpoints, label npz, sam masks, or any data outputs.
- Branch elliottabe/paper_update_062026 (rolling multi-phase; work directly on it).
- INVARIANT: stac_core_jaxls's existing behavior with no silhouette data must be byte-identical (Phases 1-4 rely on it). A test must assert this.
- Write things in JAX when possible for speed, and prioritize computational efficiency in BOTH speed and memory.

---

## Architecture decision (justification)

The brief offers two realizations: (a) a guarded optional factor inside `stac_core_jaxls._build_se3` (cache key extended with a `has_silhouette` flag); or (b) a NEW sibling joint-solver module that reuses the same cost formulas plus the silhouette factor over the same Vars, leaving `stac_core_jaxls` untouched.

**We choose (b): a new sibling module `stac-mjx/stac_mjx/stac_silhouette_jaxls.py`.** Rationale:

1. **Preserves the byte-identical invariant with zero risk.** `stac_core_jaxls.py` is the shared solver that Phases 1-4 (and the existing `silhouette_ik_solve.solve_ik`) depend on. Option (a) mutates `_build_se3` and the `_get_analyzed` cache key. Even a "guarded" edit changes the file that produces today's results — the surest proof that the no-silhouette path is unchanged is to *not touch that file at all*. Option (b) makes the invariant test a trivial `git diff --exit-code` on `stac_core_jaxls.py` plus a numerical equality of the two solvers' no-silhouette output.
2. **The threading problem is cleaner in a sibling.** The silhouette factor needs per-frame per-camera targets (boundary points + camera matrices) threaded as jaxls Vars (the `KpVar` stop-gradient pattern), plus a *different* FK (the vertex-level `silhouette_ik` FK, not stac's site FK). Adding a second FK path and two more stop-gradient Var classes inside `_build_se3` would bloat the shared solver's cache key and code with Phase-6-only concerns.
3. **Spec fidelity is retained.** The sibling still performs the spec's "joint 2D+3D refinement": it builds ONE `LeastSquaresProblem` with the 3-D `marker_cost` + smoothness + limits + reg (copied verbatim from `_build_se3`) AND the silhouette cost, over the SAME `SE3Var`+`JointVar`. It is literally "the existing IK plus a factor," just assembled in a new file so the shared one stays frozen.

The cost is one modest code duplication (the four existing cost formulas are repeated in the sibling). This is the accepted tradeoff for a frozen shared solver; a Task-4 test asserts the sibling with `silhouette_weight=0.0` reproduces `JaxlsBatchSolver`'s output to tight tolerance, so the duplication cannot silently drift.

---

## File structure

- `third_party/jarvis_jax/jarvis_jax/cse/silhouette_boundary.py` **(N)** — offline NumPy mask-boundary extraction + uniform boundary-point sampling (Task 1). Pure NumPy/scipy; no JAX, no qpos.
- `third_party/jarvis_jax/jarvis_jax/cse/silhouette_chamfer.py` **(N)** — the pure-JAX differentiable sparse-Chamfer residual (Task 2). No mesh/FK knowledge; operates on already-projected 2-D points.
- `stac-mjx/stac_mjx/stac_silhouette_jaxls.py` **(N)** — the silhouette IK factory + the sibling joint solver `SilhouetteJaxlsBatchSolver` (Tasks 3 & 4).
- `third_party/jarvis_jax/jarvis_jax/cse/silhouette_targets.py` **(N)** — assembler: SAM masks + refined calib -> per-(frame,camera) sampled boundary points + camera matrices, packed into the arrays the solver threads (Task 5). Also the fps-vs-full-array index helper (`silhouette_fk_indices`, Task 6).
- `third_party/jarvis_jax/jarvis_jax/cse/run_silhouette_polish.py` **(N)** — the committed validation *driver* (Task 7), coordinator-run on GPU.
- Tests (all **N**): `tests/test_silhouette_boundary.py`, `tests/test_silhouette_chamfer.py`, `tests/test_silhouette_factor.py`, `tests/test_silhouette_joint_solve.py`, `tests/test_silhouette_targets.py`, `tests/test_silhouette_index_space.py`, `tests/test_run_silhouette_polish.py`.
- `stac-mjx/stac_mjx/stac_core_jaxls.py` **(UNCHANGED — frozen).**

## Shared constants (referenced by several tasks)

```
XML  = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"
ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
IK   = "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5"
RECORDING = "2026_03_18_15_31_22"
```

Mesh npz keys (verified): `vertices_local`(61666,3), `vertex_geom`(61666,), `fps_300`(300,) — **`fps_300` holds indices INTO the full 61666-vertex array (values 0..61665)**, so it is already full-array index space and can be passed straight to `make_fk_repose(indices=...)`. Model: nq=93, free-joint layout `[x,y,z, qw,qx,qy,qz, hinges(86)...]`, n_hinges=86.

---

### Task 1: Offline SAM-mask boundary extraction + uniform boundary-point sampling

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_boundary.py`
- Test: `third_party/jarvis_jax/tests/test_silhouette_boundary.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Pure NumPy in.
- Produces:
  - `mask_boundary_pixels(mask: np.ndarray) -> np.ndarray` — input `mask` is a 2-D bool/`{0,1}` array `(H,W)`; returns `(K,2)` float array of boundary pixel `(x,y)` coordinates (column, row order — matches `affine_camera.project_affine`'s uv convention). A boundary pixel = a foreground pixel with at least one 4-connected background neighbor (binary erosion XOR).
  - `sample_boundary_points(mask: np.ndarray, n_points: int, *, seed: int = 0) -> np.ndarray` — returns exactly `(n_points, 2)` float `(x,y)` points sampled uniformly (evenly spaced by arc order) along the mask boundary. If the mask is empty (no foreground) returns `np.full((n_points, 2), np.nan)`. If fewer than `n_points` boundary pixels exist, samples WITH replacement (deterministic given `seed`) so the shape is always `(n_points, 2)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_silhouette_boundary.py
import numpy as np
from jarvis_jax.tracking.silhouette_boundary import mask_boundary_pixels, sample_boundary_points


def test_boundary_pixels_of_a_solid_square():
    # 10x10 foreground square inside a 20x20 image -> boundary is its perimeter.
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    bp = mask_boundary_pixels(mask)
    # coordinates are (x, y) = (col, row).
    xs, ys = bp[:, 0], bp[:, 1]
    # every boundary pixel is foreground
    assert mask[ys.astype(int), xs.astype(int)].all()
    # boundary is the outer ring only: the 8x8 interior [6..13]^2 is NOT boundary.
    interior = (xs >= 6) & (xs <= 13) & (ys >= 6) & (ys <= 13)
    assert not interior.any()
    # perimeter of a 10x10 filled square (4-connectivity erosion) = 36 pixels.
    assert bp.shape[0] == 36


def test_sample_boundary_points_shape_and_on_boundary():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    pts = sample_boundary_points(mask, n_points=64, seed=0)
    assert pts.shape == (64, 2)
    assert np.isfinite(pts).all()
    # every sampled point coincides with a real boundary pixel
    bp = mask_boundary_pixels(mask)
    bset = {(float(x), float(y)) for x, y in bp}
    for x, y in pts:
        assert (float(x), float(y)) in bset


def test_sample_boundary_points_empty_mask_is_nan():
    mask = np.zeros((20, 20), dtype=bool)
    pts = sample_boundary_points(mask, n_points=32, seed=0)
    assert pts.shape == (32, 2)
    assert np.isnan(pts).all()


def test_sample_boundary_points_deterministic():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    a = sample_boundary_points(mask, n_points=40, seed=7)
    b = sample_boundary_points(mask, n_points=40, seed=7)
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_boundary.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_jax.tracking.silhouette_boundary'`.

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis_jax/cse/silhouette_boundary.py
"""Offline SAM-mask boundary extraction + uniform boundary-point sampling.

Pure NumPy/scipy (NO JAX, NO qpos): produces the Phase-6 silhouette *target*.
The boundary is the set of foreground pixels adjacent to background (a binary
erosion XOR); sampled points are evenly spaced along the boundary's arc order.
Coordinates are (x, y) = (column, row) to match affine_camera.project_affine's
uv convention. Fixed per (frame, camera); not a function of qpos.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage


def mask_boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Return (K,2) float (x,y) boundary-pixel coordinates of a 2-D mask.

    A boundary pixel is a foreground pixel with >=1 4-connected background
    neighbor (foreground minus its 4-connected erosion).
    """
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return np.zeros((0, 2), dtype=np.float64)
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    eroded = ndimage.binary_erosion(m, structure=struct, border_value=0)
    boundary = m & ~eroded
    ys, xs = np.where(boundary)  # row, col
    return np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)


def _order_boundary(bp: np.ndarray) -> np.ndarray:
    """Order boundary pixels by a nearest-neighbor walk (approx arc order).

    Greedy chain from an arbitrary start; good enough for even *arc-order*
    subsampling of a single closed contour (the fly silhouette).
    """
    n = bp.shape[0]
    if n <= 2:
        return bp
    remaining = list(range(n))
    order = [remaining.pop(0)]
    while remaining:
        last = bp[order[-1]]
        d = np.linalg.norm(bp[remaining] - last, axis=1)
        k = int(np.argmin(d))
        order.append(remaining.pop(k))
    return bp[np.asarray(order)]


def sample_boundary_points(mask: np.ndarray, n_points: int, *, seed: int = 0) -> np.ndarray:
    """Return exactly (n_points, 2) float (x,y) points along the mask boundary.

    Empty mask -> all-NaN (n_points,2). Fewer boundary pixels than n_points ->
    deterministic sampling WITH replacement so the shape is always fixed.
    """
    bp = mask_boundary_pixels(mask)
    if bp.shape[0] == 0:
        return np.full((n_points, 2), np.nan, dtype=np.float64)
    ordered = _order_boundary(bp)
    n = ordered.shape[0]
    if n >= n_points:
        # even arc-order subsample
        idx = np.linspace(0, n - 1, n_points).round().astype(int)
        idx = np.clip(idx, 0, n - 1)
        return ordered[idx]
    # too few pixels: deterministic sample with replacement
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=n_points)
    return ordered[idx]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_boundary.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/silhouette_boundary.py third_party/jarvis_jax/tests/test_silhouette_boundary.py
git commit -m "feat(cse): SAM-mask boundary extraction + uniform boundary-point sampling (Phase 6 Task 1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Differentiable sparse-Chamfer residual (pure JAX)

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_chamfer.py`
- Test: `third_party/jarvis_jax/tests/test_silhouette_chamfer.py`

**Memory bound (Global Constraint — efficiency):** The residual must NOT materialize the full `(N,M,2)` diff / `(N,M)` distance matrix, because the silhouette factor calls this once per camera (Task 3 scans C cameras) and jaxls vmaps the whole factor over T frames — a naive `(N,M)` matrix balloons to a `(T,C,N,M)` peak intermediate. Instead compute the per-target softmin over the M verts by CHUNKING the N target points via `jax.lax.map` (a `chunk_size` batches N), so the peak intermediate is `O(chunk_size · M)`, never `O(N·M)` (let alone `O(T·C·N·M)`). Defaults stay modest: N ≤ 128 boundary points, M = fps_300 subset (300). All existing semantics (softmin / NaN-safe / Huber) unchanged; a test documents the `O(chunk_size·M)` bound.

**Interfaces:**
- Consumes: nothing (operates on already-projected 2-D points).
- Produces:
  - `chamfer_residual(target_pts, proj_pts, *, beta=8.0, huber_delta=0.0, chunk_size=32) -> jnp.ndarray` — one-directional sparse Chamfer, SAM-boundary → nearest projected mesh vertex. `target_pts` is `(N,2)` (sampled SAM boundary points, may contain NaN rows for missing cameras/frames), `proj_pts` is `(M,2)` (projected mesh vertices). Returns `(N,)` per-target residual `r_p = softmin_v ||p - proj_v||` where softmin is `-1/beta * logsumexp(-beta * d)` (a smooth lower bound on the hard min; as `beta→∞` it → hard min). NaN target rows produce a `0.0` residual (NaN-safe, mirrors `marker_cost`). If `huber_delta > 0`, the per-target distance is passed through a Huber-sqrt so the squared LM residual behaves like Huber (robust); `huber_delta=0.0` disables it (plain distance). The softmin over M is computed per target point inside a `jax.lax.map`, so the peak intermediate is `O(chunk_size · M)` — never the full `(N,M)` matrix.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_silhouette_chamfer.py
import jax
import jax.numpy as jnp
import numpy as np
from jarvis_jax.tracking.silhouette_chamfer import chamfer_residual


def test_known_geometry_hardmin_value():
    # target at (0,0); nearest projected vertex at (3,4) -> distance 5.
    # large beta -> softmin approaches the hard min.
    target = jnp.array([[0.0, 0.0]])
    proj = jnp.array([[3.0, 4.0], [10.0, 10.0], [-8.0, 6.0]])
    r = chamfer_residual(target, proj, beta=50.0)
    assert r.shape == (1,)
    assert abs(float(r[0]) - 5.0) < 0.05


def test_nan_target_row_is_zero_residual():
    target = jnp.array([[np.nan, np.nan], [0.0, 0.0]])
    proj = jnp.array([[3.0, 4.0]])
    r = chamfer_residual(target, proj, beta=50.0)
    assert float(r[0]) == 0.0
    assert abs(float(r[1]) - 5.0) < 0.05


def test_gradient_flows_to_projected_points():
    # moving the nearest projected vertex toward the target should reduce the
    # residual -> a nonzero, finite gradient w.r.t. proj.
    target = jnp.array([[0.0, 0.0]])

    def loss(proj):
        return jnp.sum(chamfer_residual(target, proj, beta=20.0) ** 2)

    proj0 = jnp.array([[3.0, 4.0], [9.0, 9.0]])
    g = jax.grad(loss)(proj0)
    assert np.isfinite(np.asarray(g)).all()
    # the nearest vertex (row 0) carries (almost) all the gradient
    assert float(jnp.linalg.norm(g[0])) > 1e-3
    assert float(jnp.linalg.norm(g[0])) > float(jnp.linalg.norm(g[1]))


def test_all_nan_targets_give_finite_zero_and_finite_grad():
    target = jnp.full((4, 2), np.nan)
    proj = jnp.array([[3.0, 4.0], [1.0, 1.0]])
    r = chamfer_residual(target, proj, beta=20.0)
    assert np.asarray(r).shape == (4,)
    assert np.all(np.asarray(r) == 0.0)

    def loss(proj):
        return jnp.sum(chamfer_residual(target, proj, beta=20.0) ** 2)

    g = jax.grad(loss)(proj)
    assert np.isfinite(np.asarray(g)).all()  # NaN targets must not poison grads


def test_huber_reduces_large_residual():
    target = jnp.array([[0.0, 0.0]])
    proj = jnp.array([[30.0, 40.0]])  # distance 50 (outlier)
    r_plain = chamfer_residual(target, proj, beta=50.0, huber_delta=0.0)
    r_huber = chamfer_residual(target, proj, beta=50.0, huber_delta=5.0)
    # squared residual is what LM minimizes; huber must shrink the outlier's
    # squared contribution below the plain d^2.
    assert float(r_huber[0]) ** 2 < float(r_plain[0]) ** 2


def test_chunking_is_result_invariant_and_bounds_memory():
    """MEMORY BOUND (efficiency Global Constraint): chunk_size only changes the
    peak intermediate (O(chunk_size*M)), NEVER the result. The chunked path
    must never materialize the full (N,M) distance matrix. Verify the residual
    is identical across chunk sizes on a moderately large N (so a full (N,M)
    matrix would be the wrong implementation)."""
    rng = np.random.default_rng(0)
    N, M = 128, 300  # default silhouette sizes (N<=128 boundary pts, M=fps_300)
    target = jnp.asarray(rng.normal(size=(N, 2)) * 40.0)
    proj = jnp.asarray(rng.normal(size=(M, 2)) * 40.0)
    r_small = chamfer_residual(target, proj, beta=8.0, chunk_size=8)
    r_big = chamfer_residual(target, proj, beta=8.0, chunk_size=64)
    r_all = chamfer_residual(target, proj, beta=8.0, chunk_size=N)
    assert r_small.shape == (N,)
    np.testing.assert_allclose(np.asarray(r_small), np.asarray(r_big), atol=1e-5)
    np.testing.assert_allclose(np.asarray(r_small), np.asarray(r_all), atol=1e-5)
    # gradient must also be chunk-invariant (the LM Jacobian must not depend on
    # the chunking used to bound memory).
    def loss(proj, cs):
        return jnp.sum(chamfer_residual(target, proj, beta=8.0, chunk_size=cs) ** 2)
    g8 = jax.grad(loss)(proj, 8)
    g64 = jax.grad(loss)(proj, 64)
    np.testing.assert_allclose(np.asarray(g8), np.asarray(g64), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_chamfer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_jax.tracking.silhouette_chamfer'`.

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis_jax/cse/silhouette_chamfer.py
"""Differentiable one-directional sparse Chamfer residual (pure JAX).

For each SAM-boundary point p, r_p = softmin_v ||p - proj_v|| over the
projected mesh vertices v. Softmin = -1/beta * logsumexp(-beta * d) is a smooth
lower bound on the hard min (beta -> inf recovers the hard min); the nearest
vertex receives (almost) all the gradient, so this pulls the mesh outline OUT to
the mask boundary WITHOUT needing an explicit differentiable mesh contour.

NaN-safe (NaN target rows -> 0 residual, no gradient), mirroring stac
marker_cost's finite-mask convention. Optional Huber robustification.

MEMORY BOUND (efficiency Global Constraint): the per-target softmin over the M
verts is computed inside a jax.lax.map over CHUNKS of the N target points, so
the peak intermediate is O(chunk_size * M) -- the full (N, M) distance matrix is
NEVER materialized. This matters because the silhouette factor (Task 3) calls
this once per camera and jaxls vmaps the whole factor over T frames; a naive
(N, M) matrix would peak at O(T * C * N * M).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _huber_sqrt(d, delta):
    """sqrt of the Huber loss so that (return)^2 == huber(d).

    huber(d) = d^2 for |d|<=delta, else 2*delta*|d| - delta^2. Returning its
    sqrt lets the LM least-squares objective (sum of residual^2) behave as a
    Huber loss on the raw distance d.
    """
    quad = d
    lin = jnp.sqrt(jnp.clip(2.0 * delta * jnp.abs(d) - delta ** 2, a_min=0.0))
    return jnp.where(jnp.abs(d) <= delta, quad, lin)


def chamfer_residual(target_pts, proj_pts, *, beta: float = 8.0,
                     huber_delta: float = 0.0, chunk_size: int = 32):
    """One-directional sparse Chamfer: SAM boundary -> nearest projected vertex.

    Args:
        target_pts: (N, 2) sampled SAM boundary points (NaN rows allowed).
        proj_pts:   (M, 2) projected mesh vertices (function of qpos).
        beta:       softmin sharpness (larger -> closer to hard min).
        huber_delta: >0 enables Huber on the per-target distance; 0 disables.
            MUST be a static Python float (branched with a plain `if`, NOT
            jnp.where) -- jnp.where would trace the unused Huber branch, whose
            sqrt(clip(.,0)) has a NaN gradient at 0 and would poison grads even
            when huber_delta==0 (verified). It is closed over the inner _one, so
            it stays static under jax.lax.map.
        chunk_size: batch size for jax.lax.map over the N targets. Bounds the
            peak intermediate at O(chunk_size * M); result is chunk-invariant.

    Returns:
        (N,) per-target residual; NaN target rows -> 0.0.
    """
    target_pts = jnp.asarray(target_pts)
    proj_pts = jnp.asarray(proj_pts)
    finite = jnp.isfinite(target_pts).all(axis=-1)              # (N,)
    tgt = jnp.where(finite[:, None], target_pts, 0.0)           # sanitize NaNs

    # Per-target softmin over the M verts. Peak intermediate here is O(M) (the
    # per-vertex distance vector); jax.lax.map batches this by chunk_size, so the
    # whole call peaks at O(chunk_size * M), never the full (N, M) matrix. No
    # host callbacks -> jit-fusable. huber_delta is branched with a STATIC `if`
    # (never jnp.where) so the unused sqrt branch is not traced.
    def _one(p):
        d = jnp.sqrt(jnp.sum((proj_pts - p[None, :]) ** 2, axis=-1) + 1e-12)  # (M,)
        soft = -(1.0 / beta) * logsumexp(-beta * d)                          # scalar
        if huber_delta > 0.0:
            soft = _huber_sqrt(soft, huber_delta)
        return soft

    soft = jax.lax.map(_one, tgt, batch_size=chunk_size)        # (N,)
    return jnp.where(finite, soft, 0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_chamfer.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/silhouette_chamfer.py third_party/jarvis_jax/tests/test_silhouette_chamfer.py
git commit -m "feat(cse): differentiable sparse-Chamfer residual (softmin, NaN-safe, huber) (Phase 6 Task 2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: The silhouette IK factor (FK → project_affine → Chamfer) as a jaxls Cost factory

**Files:**
- Create: `stac-mjx/stac_mjx/stac_silhouette_jaxls.py` (the factory helper only in this task; the solver class comes in Task 4)
- Test: `third_party/jarvis_jax/tests/test_silhouette_factor.py`

**Memory bound (Global Constraint — efficiency):** FK the vertex subset ONCE per frame (`verts3d` closed over), then iterate the C cameras with `jax.lax.scan` (each step: project the M verts `O(M)`, run the chunked `chamfer_residual` `O(chunk_size·M)`, emit its `(n_pts,)` residual). Do NOT `jax.vmap` into a `(C,N,M)` tensor. Rely on jaxls's per-frame vmap for T; never build a `(T, ·)` silhouette tensor in Python. Fully jit-fusable, no host callbacks in the traced step. Peak intermediate: `O(chunk_size·M)` per camera step, NOT `O(T·C·N·M)`.

**Interfaces:**
- Consumes:
  - `chamfer_residual(target_pts, proj_pts, *, beta, huber_delta, chunk_size)` from Task 2 (chunked over N; memory-bounded).
  - `silhouette_ik.make_fk_repose(anat)` -> `fk_repose(qpos, scale=1.0, indices=None) -> (K,3)` world verts (verified signature; `indices` are FULL-vertex-array indices).
  - `silhouette_ik.load_anatomy(model_xml, mesh_npz) -> dict` with keys `m, mx, dx, vlocal, vgeom, faces, fps, nq, qpos0, ...`.
- Produces:
  - `make_silhouette_cost(SE3Var, JointVar, SilVar, *, fk_repose, vert_indices, n_cam, n_pts, beta, huber_delta, silhouette_weight, qs_to_opt, template_qpos, scale=1.0, chunk_size=32) -> jaxls.Cost.factory-decorated fn` returning a per-frame residual of shape `(n_cam * n_pts,)`. Iterates cameras via `jax.lax.scan` (no `(C,N,M)` tensor).
  - The factory reconstructs `full_q` exactly as `stac_core_jaxls._build_se3.marker_cost` does: `q = concat([T_root.translation(), T_root.rotation().wxyz, joints])`, then `full_q = jnp.where(qs_to_opt, q, template_qpos)`.
  - `SilVar` value layout per frame: a flat `(n_cam*(6 + n_pts*2),)` vector = for each camera, `[P_row0(3), P_row1(3), boundary_xy(n_pts*2)]`, where `P_row0/P_row1` are the affine DLT matrix's first two rows' `[:, :3]` and translation is folded in via a leading (see impl: we pack `M(2,3)` flat = 6 numbers, and `t(2,)` — total 8; corrected below). **Exact layout is fixed in the impl and asserted by the test.**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_silhouette_factor.py
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
    from jarvis_jax.tracking.silhouette_ik import load_anatomy, make_fk_repose
    from stac_mjx.stac_silhouette_jaxls import (
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
    from stac_mjx.stac_silhouette_jaxls import (
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_factor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stac_mjx.stac_silhouette_jaxls'`.

- [ ] **Step 3: Write minimal implementation**

```python
# stac-mjx/stac_mjx/stac_silhouette_jaxls.py
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

from jarvis_jax.tracking.silhouette_chamfer import chamfer_residual


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_factor.py -v`
Expected: PASS (2 passed). (Note: `project_affine`'s convention is `X @ P[:2,:3].T + P[:2,3]`, matched here by `verts3d @ M.T + t`. `jax.lax.scan` over `(Ms, ts, pts)` iterates the leading camera axis, so each `cam` step sees `(2,3),(2,),(n_pts,2)`.)

- [ ] **Step 5: Commit**

```bash
git add stac-mjx/stac_mjx/stac_silhouette_jaxls.py third_party/jarvis_jax/tests/test_silhouette_factor.py
git commit -m "feat(stac): silhouette IK cost factory (FK->project_affine->Chamfer) + SilVar packing (Phase 6 Task 3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Sibling joint solver + BYTE-IDENTICAL no-silhouette invariant + joint-solve smoke

**Files:**
- Modify: `stac-mjx/stac_mjx/stac_silhouette_jaxls.py` (append the `SilhouetteJaxlsBatchSolver` class)
- Test: `third_party/jarvis_jax/tests/test_silhouette_joint_solve.py`

**Interfaces:**
- Consumes:
  - `make_silhouette_cost(...)`, `SIL_PER_CAM`, `pack_sil_value` from Task 3.
  - `stac_core_jaxls.JaxlsBatchSolver` (only imported by the TEST, to compare outputs; the solver module itself is frozen).
  - `jarvis_jax.tracking.silhouette_ik.make_fk_repose` (the vertex FK for the factory).
- Produces:
  - `SilhouetteJaxlsBatchSolver(n_iter=50, linear_solver="auto", lambda_initial=1.0, smooth_weight=0.0, use_se3_root=True, beta=8.0, huber_delta=0.0)` with method
    `solve_trajectory(q_init, mjx_model, mjx_data_template, kp_data, qs_to_opt, kps_to_opt, lb, ub, site_idxs, q_reg_weights, *, fk_repose=None, vert_indices=None, sil_data=None, silhouette_weight=0.0) -> jnp.ndarray (T,nq)`.
  - When `silhouette_weight == 0.0` OR `sil_data is None`: the silhouette cost is NOT added, and the solve must reproduce `JaxlsBatchSolver.solve_trajectory(...)` to `atol=1e-5` (numerical-equality invariant; the shared solver file itself is untouched, `git diff --exit-code` clean).
  - `sil_data`: `(T, n_cam*SIL_PER_CAM(n_pts))` array (per-frame packed SilVar values from Task 5). `n_cam`, `n_pts` inferred from `sil_data.shape[-1]` and a passed `n_pts`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_silhouette_joint_solve.py
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
    from jarvis_jax.tracking.silhouette_ik_solve import build_solver_inputs
    from stac_mjx.stac_core_jaxls import JaxlsBatchSolver
    from stac_mjx.stac_silhouette_jaxls import SilhouetteJaxlsBatchSolver

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
    from jarvis_jax.tracking.silhouette_ik_solve import build_solver_inputs
    from jarvis_jax.tracking.silhouette_ik import load_anatomy, make_fk_repose
    from stac_mjx.stac_silhouette_jaxls import (
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_joint_solve.py -v`
Expected: FAIL — `ImportError: cannot import name 'SilhouetteJaxlsBatchSolver'` (the byte-identical test may pass immediately since the file is not yet modified; the two solve tests fail on import).

- [ ] **Step 3: Write minimal implementation (append to `stac_silhouette_jaxls.py`)**

```python
# ---- appended to stac-mjx/stac_mjx/stac_silhouette_jaxls.py ----
import jaxlie
from stac_mjx import utils

_FREE_JOINT_NDOF = 7


class SilhouetteJaxlsBatchSolver:
    """Joint 2D+3D IK solver: 3-D marker cost (verbatim from
    stac_core_jaxls._build_se3) + smoothness/limit/reg + an additive
    silhouette boundary-Chamfer cost, over the SAME SE3Var + JointVar.

    With silhouette_weight=0 (or sil_data=None) the silhouette cost is NOT
    added and the solve reproduces JaxlsBatchSolver's output.
    """

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

        class SE3Var(jaxls.Var[jaxlie.SE3], default_factory=jaxlie.SE3.identity,
                     retract_fn=jaxlie.manifold.rplus, tangent_dim=6): ...
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
        if bool(jnp.any(q_reg_weights[_FREE_JOINT_NDOF:] > 0)):
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

        # ---- ADDITIVE silhouette cost (only when weight>0 and data present) ----
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_joint_solve.py -v`
Expected: PASS (3 passed). If `test_no_silhouette_matches_jaxls_batch_solver` fails on tolerance, confirm the copied cost formulas match `_build_se3` verbatim (variable ordering `[root_all, joint_all, kp_all]`, `smooth_weight`>0 branch, reg branch condition on `q_reg_weights[7:]`).

- [ ] **Step 5: Commit**

```bash
git add stac-mjx/stac_mjx/stac_silhouette_jaxls.py third_party/jarvis_jax/tests/test_silhouette_joint_solve.py
git commit -m "feat(stac): SilhouetteJaxlsBatchSolver sibling joint solver + byte-identical invariant test (Phase 6 Task 4)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Silhouette-target assembler (SAM masks + refined calib -> packed per-frame SilVar data)

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_targets.py`
- Test: `third_party/jarvis_jax/tests/test_silhouette_targets.py`

**Interfaces:**
- Consumes:
  - `silhouette_boundary.sample_boundary_points(mask, n_points, seed) -> (n_points,2)` (Task 1).
  - `stac_silhouette_jaxls.pack_sil_value(cam_mats, boundary_pts) -> (n_cam*SIL_PER_CAM,)` and `SIL_PER_CAM(n_pts)` (Task 3).
  - `silhouette_ik_solve._cam2img_for_frame(fs_imgids_row, id2file, cam_names)`, `_ann_for_image(id2ann_multi, image_id, ann_id_by_image)`, `_load_sam_mask(root, split, file_name, ann_id)` (existing, verified).
  - `jarvis_jax.geometry.reprojection_tool.ReprojectionTool(calib_dir)` -> `.cameras` (dict), `._camera_list[c].cameraMatrix` (3,4), `.num_cameras`.
- Produces:
  - `build_silhouette_targets(root, split, fs_imgids, calib_dir, *, n_points=128, ann_id_by_image=None, seed=0) -> (sil_data, meta)` where `sil_data` is `(T, num_cameras*SIL_PER_CAM(n_points))` float32 and `meta` is a dict with `n_cam=num_cameras`, `n_pts=n_points`, `cam_names`. A camera with no mask for a frame gets its boundary block filled with NaN (Chamfer NaN-safe); its DLT matrix is still packed (constant). Camera-matrix blocks are the SAME for every frame (calibration is time-invariant) but repacked per frame for a uniform `(T, ...)` array.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_silhouette_targets.py
import numpy as np
from jarvis_jax.tracking.silhouette_targets import build_silhouette_targets


def test_build_targets_shapes_and_nan_for_missing(monkeypatch):
    from jarvis_jax.tracking import silhouette_targets as st
    from stac_mjx.stac_silhouette_jaxls import SIL_PER_CAM, unpack_sil_value
    import jax.numpy as jnp

    n_cam, n_points, T = 3, 32, 2
    P = [np.array([[8.0, 0, 0, -2.0], [0, -8.0, 0, 460.0], [0, 0, 0, 1.0]]) for _ in range(n_cam)]

    # --- stub the calibration + coco + mask I/O so the test is offline ---
    class _FakeCam:
        def __init__(self, P): self.cameraMatrix = P
    class _FakeRT:
        def __init__(self, calib_dir):
            self.cameras = {f"cam{i}": None for i in range(n_cam)}
            self._camera_list = [_FakeCam(P[i]) for i in range(n_cam)]
            self.num_cameras = n_cam
    monkeypatch.setattr(st, "ReprojectionTool", _FakeRT)

    # frame 0: cam0 and cam1 have a mask, cam2 does not. frame 1: only cam0.
    square = np.zeros((30, 30), dtype=bool); square[8:20, 8:20] = True
    def _fake_cam2img(row, id2file, cam_names): return dict(row)
    def _fake_ann(id2ann_multi, iid, sel): return {"id": iid}
    def _fake_mask(root, split, fn, ann_id):
        return square if ann_id in (0, 1, 10) else None
    def _fake_id2file(*a, **k): return {}
    monkeypatch.setattr(st, "_cam2img_for_frame", _fake_cam2img)
    monkeypatch.setattr(st, "_ann_for_image", _fake_ann)
    monkeypatch.setattr(st, "_load_sam_mask", _fake_mask)
    monkeypatch.setattr(st, "_load_coco_index", lambda root, split: ({}, {}, ["cam0", "cam1", "cam2"]))

    # fs_imgids: per frame, {cam_idx: image_id}. Use the ids the fake mask keys on.
    fs_imgids = [{0: 0, 1: 1, 2: 99}, {0: 10, 1: 98, 2: 97}]

    sil_data, meta = build_silhouette_targets(
        root="X", split="val", fs_imgids=fs_imgids, calib_dir="Y",
        n_points=n_points, seed=0,
    )
    assert meta["n_cam"] == n_cam and meta["n_pts"] == n_points
    assert sil_data.shape == (T, n_cam * SIL_PER_CAM(n_points))

    Ms0, ts0, pts0 = unpack_sil_value(jnp.asarray(sil_data[0]), n_cam, n_points)
    # cam0, cam1 present -> finite boundary; cam2 missing -> NaN boundary.
    assert np.isfinite(np.asarray(pts0[0])).all()
    assert np.isfinite(np.asarray(pts0[1])).all()
    assert np.isnan(np.asarray(pts0[2])).all()
    # camera matrix blocks are always finite (calibration constant).
    assert np.isfinite(np.asarray(Ms0)).all()
    # frame 1: only cam0 present.
    _, _, pts1 = unpack_sil_value(jnp.asarray(sil_data[1]), n_cam, n_points)
    assert np.isfinite(np.asarray(pts1[0])).all()
    assert np.isnan(np.asarray(pts1[1])).all()
    assert np.isnan(np.asarray(pts1[2])).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_targets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_jax.tracking.silhouette_targets'`.

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis_jax/cse/silhouette_targets.py
"""Assemble per-(frame,camera) silhouette targets for the Phase-6 IK.

For each bout frame + camera, loads the SAM mask for the selected fly, samples
n_points uniform boundary points, and packs them with the camera's affine DLT
matrix into the flat SilVar layout the sibling solver threads. Cameras with no
mask this frame get NaN boundary blocks (the Chamfer residual is NaN-safe).
"""
from __future__ import annotations
import json
import os
import numpy as np

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.tracking.silhouette_boundary import sample_boundary_points
from jarvis_jax.tracking.silhouette_ik_solve import (
    _cam2img_for_frame, _ann_for_image, _load_sam_mask,
)
from stac_mjx.stac_silhouette_jaxls import SIL_PER_CAM, pack_sil_value


def _load_coco_index(root, split):
    """Return (id2file, id2ann_multi, cam_names_placeholder). cam_names comes
    from the ReprojectionTool at call time; this returns the coco maps."""
    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann_multi = {}
    for an in coco["annotations"]:
        id2ann_multi.setdefault(an["image_id"], []).append(an)
    return id2file, id2ann_multi, None


def build_silhouette_targets(
    root, split, fs_imgids, calib_dir, *,
    n_points: int = 128, ann_id_by_image=None, seed: int = 0,
):
    """Build (sil_data, meta) for the silhouette IK.

    Args:
        root, split: dataset root + split (coco annotations + sam3_masks).
        fs_imgids: (T, n_cam) array (or list of per-frame {cam_idx:image_id}
            dicts) of coco image_ids per frame/camera.
        calib_dir: (refined) calibration dir for ReprojectionTool.
        n_points: boundary points sampled per camera per frame.
        ann_id_by_image: optional image_id->chosen ann id (multi-fly identity).
        seed: boundary-sampling seed (deterministic).

    Returns:
        sil_data: (T, num_cameras*SIL_PER_CAM(n_points)) float32.
        meta: dict(n_cam, n_pts, cam_names).
    """
    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    n_cam = rt.num_cameras
    cam_mats = [rt._camera_list[c].cameraMatrix for c in range(n_cam)]

    id2file, id2ann_multi, _ = _load_coco_index(root, split)

    fs_list = list(fs_imgids)
    T = len(fs_list)
    rows = []
    for t in range(T):
        cam2img = _cam2img_for_frame(fs_list[t], id2file, cam_names)
        boundary = np.full((n_cam, n_points, 2), np.nan, dtype=np.float64)
        for c in range(n_cam):
            iid = cam2img.get(c)
            if iid is None:
                continue
            ann = _ann_for_image(id2ann_multi, iid, ann_id_by_image)
            if ann is None:
                continue
            fn = id2file.get(int(iid))
            if fn is None:
                continue
            mask = _load_sam_mask(root, split, fn, ann["id"])
            if mask is None:
                continue
            boundary[c] = sample_boundary_points(np.asarray(mask), n_points, seed=seed)
        rows.append(pack_sil_value(cam_mats, boundary))

    sil_data = np.stack(rows, axis=0).astype(np.float32)
    meta = dict(n_cam=n_cam, n_pts=n_points, cam_names=cam_names)
    return sil_data, meta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_targets.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/silhouette_targets.py third_party/jarvis_jax/tests/test_silhouette_targets.py
git commit -m "feat(cse): silhouette-target assembler (SAM masks+calib -> packed per-frame SilVar data) (Phase 6 Task 5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: fps-vs-full-array index-space audit (the deferred carry-forward)

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_targets.py` (add `silhouette_fk_indices`)
- Test: `third_party/jarvis_jax/tests/test_silhouette_index_space.py`

**Interfaces:**
- Consumes:
  - The mesh npz `fps_300` array (values are FULL-array vertex indices 0..61665) and `vertex_geom` (61666,).
  - `silhouette_ik_solve._wing_fk_indices(mesh_npz, exclude_seg_ids)` (existing bridge — the pattern to mirror: it does `fps[idx]` to convert fps-relative → full-array).
- Produces:
  - `silhouette_fk_indices(mesh_npz, subset="fps_300", exclude_seg_ids=None) -> np.ndarray (M,)` of FULL-vertex-array indices for the silhouette factor's projected-vertex subset. For `subset="fps_300"` this is simply `z["fps_300"]` (already full-array space) with optional segment exclusion applied in full-array space. **The hazard this guards:** any code that reuses Phase-4 `wing_side_vertices` / `active_parts.excluded_fps_indices` outputs (which are fps-RELATIVE 0..299) must bridge via `fps[idx]` before passing to `make_fk_repose(indices=...)`; passing fps-relative indices straight in silently selects the WRONG vertices (e.g. thorax verts whose FK distance is pose-invariant), exactly the bug `_wing_fk_indices` documents.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_silhouette_index_space.py
import os
import numpy as np
import pytest

MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"


@pytest.mark.skipif(not os.path.exists(MESH), reason="mesh not present")
def test_silhouette_fk_indices_are_full_array_space_not_fps_relative():
    """The silhouette factor's vertex subset MUST be FULL-vertex-array indices
    (0..61665), not fps-relative (0..299). Regression guard mirroring
    _wing_fk_indices' fps[idx] bridge: fps-relative indices would (a) all be
    < 300 and (b) select the wrong vertices."""
    from jarvis_jax.tracking.silhouette_targets import silhouette_fk_indices
    z = np.load(MESH, allow_pickle=True)
    n_verts = z["vertices_local"].shape[0]      # 61666
    fps300 = z["fps_300"]

    idx = silhouette_fk_indices(MESH, subset="fps_300")
    assert idx.shape == fps300.shape
    # full-array space: exactly equals fps_300 (which itself holds full-array ids)
    np.testing.assert_array_equal(np.sort(idx), np.sort(fps300))
    # values span the FULL array, not just 0..299 (proves not fps-relative).
    assert idx.max() >= 300 and idx.max() < n_verts
    assert not (idx < 300).all()


@pytest.mark.skipif(not os.path.exists(MESH), reason="mesh not present")
def test_silhouette_fk_indices_index_geoms_correctly():
    """Indexing vertex_geom (a FULL-array (61666,) map) with the returned
    indices must be in-bounds and select real geoms -- a fps-relative index
    set would index the first 300 rows only (a different, wrong selection)."""
    from jarvis_jax.tracking.silhouette_targets import silhouette_fk_indices
    z = np.load(MESH, allow_pickle=True)
    vgeom = z["vertex_geom"]                      # (61666,)
    idx = silhouette_fk_indices(MESH, subset="fps_300")
    assert idx.max() < vgeom.shape[0]
    geoms = vgeom[idx]
    assert np.isfinite(geoms.astype(float)).all()
    # the subset must touch MORE than one geom (a whole-body silhouette subset)
    assert len(np.unique(geoms)) > 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_index_space.py -v`
Expected: FAIL — `ImportError: cannot import name 'silhouette_fk_indices'`.

- [ ] **Step 3: Write minimal implementation (append to `silhouette_targets.py`)**

```python
# ---- appended to jarvis_jax/cse/silhouette_targets.py ----

def silhouette_fk_indices(mesh_npz, subset: str = "fps_300", exclude_seg_ids=None):
    """FULL-vertex-array indices for the silhouette factor's projected subset.

    INDEX-SPACE HAZARD (Phase-4 carry-forward, mirrors
    silhouette_ik_solve._wing_fk_indices): the mesh npz's ``fps_*`` arrays hold
    indices INTO the full ``vertices``/``vertex_geom`` arrays (values 0..61665),
    so they are ALREADY full-array space and can be passed straight to
    ``silhouette_ik.make_fk_repose(indices=...)``. In contrast,
    ``silhouette_landmarks.wing_side_vertices`` /
    ``active_parts.excluded_fps_indices`` return indices INTO the fps subset
    (0..299) -- those MUST be bridged via ``fps[idx]`` before FK, or they
    silently select the wrong vertices (verified thorax-vertex bug in
    _wing_fk_indices' docstring). This helper only ever returns full-array
    indices, and applies ``exclude_seg_ids`` in full-array space.

    Args:
        mesh_npz: canonical mesh npz path.
        subset: which fps subset to use ("fps_300" default).
        exclude_seg_ids: optional list of segment ids to drop (full-array
            filtering via ``vertex_segment``), for Phase-4 active-parts.

    Returns:
        np.ndarray (M,) int32 full-vertex-array indices.
    """
    z = np.load(mesh_npz, allow_pickle=True)
    fps = np.asarray(z[subset], dtype=np.int64)   # full-array indices already
    if exclude_seg_ids:
        seg = np.asarray(z["vertex_segment"])     # (61666,) full-array
        excl = set(int(s) for s in exclude_seg_ids)
        keep = np.array([int(seg[i]) not in excl for i in fps])
        fps = fps[keep]
    return fps.astype(np.int32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_index_space.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/silhouette_targets.py third_party/jarvis_jax/tests/test_silhouette_index_space.py
git commit -m "feat(cse): silhouette_fk_indices full-array index-space audit + hazard doc (Phase 6 Task 6)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Validation driver on 2026_03_18_15_31_22 (MALE) + CPU existence/summary test

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/run_silhouette_polish.py` (coordinator-run GPU driver; NOT a pytest for the heavy solve)
- Test: `third_party/jarvis_jax/tests/test_run_silhouette_polish.py` (CPU existence/summary + tiny helpers only)

**Interfaces:**
- Consumes:
  - `silhouette_ik_solve.build_solver_inputs(ik_h5, model_xml) -> dict`.
  - `silhouette_ik_solve._umeyama`, `_model_to_mm`, `_mm_to_model`, `_triangulate_kp_mm`, `_cam2img_for_frame`, `_ann_for_image` (existing; used to build the per-frame model->mm bridge and reprojection metric, mirroring `run_single_fly`).
  - `silhouette_targets.build_silhouette_targets(...)`, `silhouette_targets.silhouette_fk_indices(...)`.
  - `stac_silhouette_jaxls.SilhouetteJaxlsBatchSolver`.
  - `silhouette_ik.load_anatomy`, `make_fk_repose`.
- Produces:
  - `iou_of_projected_verts(verts2d, mask_shape, ref_mask) -> float` — a small, TESTABLE helper: splat projected 2-D vertices to a binary mask (round to pixel, set True), then IoU vs `ref_mask`. Deterministic, pure NumPy. (Coarse HARD-IoU proxy; the point is the before/after DELTA.)
  - `soft_iou_of_verts(verts2d, mask, *, sigma=1.3, splat_k=2) -> float` — an EVAL-ONLY (non-differentiable, report-time) soft-IoU: Gaussian-splat the projected 2-D verts into a soft occupancy grid over the mask's own pixel frame, then `IoU_soft = sum(min-ish overlap)/sum(union)` exactly as `silhouette_fit.py::soft_sil` + its IoU loss do, so the reported number is comparable to the known collision-mesh baseline (0.65→0.76). This mirrors `soft_sil`'s windowed-Gaussian splat (extracted as a reusable function here, using `affine_camera.project_affine`'s affine convention — NOT the perspective divide in the demo closure — for consistency with the rest of Phase 6). **It is a METRIC ONLY, run once per frame at report time, NEVER an objective term** — the IK objective stays Chamfer (locked "no soft-IoU objective" scope preserved). Runs on GPU but not in the LM loop, so its all-verts cost is fine.
  - `run_polish(recording, *, ik_h5, model_xml, mesh_npz, root, split="val", calib_dir=None, n_points=128, silhouette_weight=0.3, beta=8.0, huber_delta=0.0, max_frames=0, smooth_weight=0.1, n_iter=50, out_dir) -> dict` reporting `iou_before`, `iou_after` (hard proxy), `soft_iou_before`, `soft_iou_after` (baseline-comparable soft-IoU), `reproj_px_before`, `reproj_px_after`, `marker_resid_before`, `marker_resid_after`, `n_frames`.
  - A `main()` argparse CLI so the coordinator can run it on GPU.

- [ ] **Step 1: Write the failing test (CPU-only helpers + existence)**

```python
# tests/test_run_silhouette_polish.py
import os
import numpy as np
import pytest


def test_iou_of_projected_verts_known_overlap():
    from jarvis_jax.tracking.run_silhouette_polish import iou_of_projected_verts
    # ref mask: filled 10x10 square in a 20x20 image.
    ref = np.zeros((20, 20), dtype=bool); ref[5:15, 5:15] = True
    # projected verts exactly filling the same square -> IoU == 1.
    yy, xx = np.mgrid[5:15, 5:15]
    verts2d = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)  # (x,y)
    iou = iou_of_projected_verts(verts2d, (20, 20), ref)
    assert abs(iou - 1.0) < 1e-9
    # a disjoint square -> IoU == 0.
    verts_off = verts2d + np.array([10.0, 0.0])  # shift x by 10 -> no overlap kept in-bounds
    verts_off = verts_off[(verts_off[:, 0] < 20)]
    iou0 = iou_of_projected_verts(verts_off, (20, 20), ref)
    assert iou0 < 0.2


def test_iou_ignores_out_of_bounds_verts():
    from jarvis_jax.tracking.run_silhouette_polish import iou_of_projected_verts
    ref = np.zeros((10, 10), dtype=bool); ref[2:8, 2:8] = True
    verts = np.array([[100.0, 100.0], [-5.0, -5.0], [4.0, 4.0]])  # 2 OOB, 1 inside
    iou = iou_of_projected_verts(verts, (10, 10), ref)
    assert 0.0 <= iou <= 1.0  # no crash on OOB; finite


def test_soft_iou_of_verts_higher_when_verts_fill_mask():
    """Eval-only soft-IoU (baseline-comparable): verts densely filling the mask
    give a higher soft-IoU than verts sitting outside it. Bounded in [0,1]."""
    from jarvis_jax.tracking.run_silhouette_polish import soft_iou_of_verts
    mask = np.zeros((30, 30), dtype=bool); mask[8:22, 8:22] = True
    yy, xx = np.mgrid[8:22, 8:22]
    inside = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)  # (x,y)
    outside = inside + np.array([25.0, 0.0])  # shifted well off the mask
    s_in = soft_iou_of_verts(inside, mask, sigma=1.3)
    s_out = soft_iou_of_verts(outside, mask, sigma=1.3)
    assert 0.0 <= s_in <= 1.0 and 0.0 <= s_out <= 1.0
    assert s_in > s_out
    # empty mask -> 0 (no crash)
    assert soft_iou_of_verts(inside, np.zeros((30, 30), dtype=bool), sigma=1.3) == 0.0


def test_run_polish_is_importable_and_has_cli():
    import jarvis_jax.tracking.run_silhouette_polish as m
    assert hasattr(m, "run_polish")
    assert hasattr(m, "main")
    assert callable(m.run_polish)


IK = "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5"


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present (GPU-only heavy run)")
def test_run_polish_report_keys_smoke(tmp_path):
    """Tiny 2-frame smoke (CPU) exercising the report assembly. The scientific
    magnitude comes from the coordinator GPU run, not this test."""
    from jarvis_jax.tracking.run_silhouette_polish import run_polish
    XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
    MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"
    ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
    rep = run_polish(
        "2026_03_18_15_31_22", ik_h5=IK, model_xml=XML, mesh_npz=MESH, root=ROOT,
        split="val", n_points=32, silhouette_weight=0.3, max_frames=2, n_iter=10,
        out_dir=str(tmp_path),
    )
    for k in ("iou_before", "iou_after", "soft_iou_before", "soft_iou_after",
              "reproj_px_before", "reproj_px_after",
              "marker_resid_before", "marker_resid_after", "n_frames"):
        assert k in rep
    assert rep["n_frames"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_run_silhouette_polish.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_jax.tracking.run_silhouette_polish'`.

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis_jax/cse/run_silhouette_polish.py
"""Phase 6 validation driver: joint 2D+3D silhouette-Chamfer refinement on the
MALE recording 2026_03_18_15_31_22, warm-started from the STAC fit.

Reports silhouette IoU before/after, multi-view reprojection before/after, and
the 3-D marker residual delta. HONEST success criterion: IoU improves (up to the
~0.76 collision-mesh cap), reprojection does NOT regress, 3-D markers not
materially worse. A neutral/small result is acceptable and must be reported
truthfully (the collision-mesh silhouette IoU cap is a known ceiling).

Coordinator-run on GPU (the heavy solve is NOT a pytest). A tiny CPU smoke lives
in tests/test_run_silhouette_polish.py.
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np


def iou_of_projected_verts(verts2d, mask_shape, ref_mask) -> float:
    """Coarse silhouette HARD-IoU: splat projected 2-D verts to a binary mask
    (round to pixel), IoU vs ref_mask. Out-of-bounds verts dropped. Pure NumPy.
    The before/after DELTA is what matters (a point-splat is sparse vs a filled
    silhouette); the baseline-comparable absolute is soft_iou_of_verts below."""
    H, W = mask_shape
    pred = np.zeros((H, W), dtype=bool)
    v = np.asarray(verts2d)
    xs = np.round(v[:, 0]).astype(int)
    ys = np.round(v[:, 1]).astype(int)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    pred[ys[ok], xs[ok]] = True
    ref = np.asarray(ref_mask, dtype=bool)
    inter = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    return float(inter / union) if union > 0 else 0.0


def soft_iou_of_verts(verts2d, mask, *, sigma: float = 1.3, splat_k: int = 2) -> float:
    """EVAL-ONLY soft-IoU (baseline-comparable). NOT an objective term.

    Windowed-Gaussian splat of the projected 2-D verts into a soft occupancy
    grid on the mask's own pixel frame, then soft-IoU vs the (binary) SAM mask,
    exactly mirroring silhouette_fit.py::soft_sil + its IoU loss (extracted here
    as a reusable, report-time-only function). Runs once per frame at report
    time -- never inside the LM loop -- so rasterizing ALL verts is fine and the
    locked "no soft-IoU objective" scope is preserved (this is a METRIC only).

    Args:
        verts2d: (V, 2) projected mesh verts in pixel (x, y) for this camera.
        mask: (H, W) SAM mask (bool / {0,1}).
        sigma: Gaussian splat sigma (matches silhouette_fit default 1.3).
        splat_k: half-window in pixels for the splat (K=2 in silhouette_fit).

    Returns:
        soft-IoU in [0, 1]; 0.0 if the mask is empty.
    """
    mask = np.asarray(mask).astype(np.float32)
    H, W = mask.shape
    if mask.sum() == 0:
        return 0.0
    v = np.asarray(verts2d, dtype=np.float64)
    grid = np.zeros((H, W), dtype=np.float64)
    bx = np.floor(v[:, 0]).astype(int)
    by = np.floor(v[:, 1]).astype(int)
    for di in range(-splat_k, splat_k + 1):
        for dj in range(-splat_k, splat_k + 1):
            cx = bx + dj
            cy = by + di
            w = np.exp(-(((cx + 0.5 - v[:, 0]) ** 2 + (cy + 0.5 - v[:, 1]) ** 2))
                       / (2 * sigma ** 2))
            valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
            np.add.at(grid, (np.clip(cy, 0, H - 1), np.clip(cx, 0, W - 1)),
                      np.where(valid, w, 0.0))
    soft = 1.0 - np.exp(-grid)                       # soft occupancy in [0,1)
    inter = float((soft * mask).sum())
    union = float((soft + mask - soft * mask).sum())
    return inter / union if union > 0 else 0.0


def run_polish(
    recording, *, ik_h5, model_xml, mesh_npz, root, split="val", calib_dir=None,
    n_points=128, silhouette_weight=0.3, beta=8.0, huber_delta=0.0,
    max_frames=0, smooth_weight=0.1, n_iter=50, out_dir,
):
    import h5py
    import jax.numpy as jnp
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.tracking.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.tracking.silhouette_ik_solve import (
        build_solver_inputs, _umeyama, _model_to_mm, _triangulate_kp_mm,
        _cam2img_for_frame, _ann_for_image, _load_sam_mask,
        _DEFAULT_REFINED_CALIB_DIR, _DEFAULT_FACTORY_CALIB_DIR,
    )
    from jarvis_jax.tracking.silhouette_targets import (
        build_silhouette_targets, silhouette_fk_indices,
    )
    from stac_mjx.stac_silhouette_jaxls import SilhouetteJaxlsBatchSolver

    if calib_dir is None:
        calib_dir = (_DEFAULT_REFINED_CALIB_DIR
                     if os.path.exists(_DEFAULT_REFINED_CALIB_DIR)
                     else _DEFAULT_FACTORY_CALIB_DIR)

    inputs = build_solver_inputs(ik_h5, model_xml)
    T_full = inputs["q_init"].shape[0]
    T = T_full if max_frames <= 0 else min(max_frames, T_full)
    q_init = inputs["q_init"][:T]
    kp_data = inputs["kp_data"][:T]

    ik_raw = ioh5.load(ik_h5)
    marker_sites = np.asarray(ik_raw["marker_sites"])[:T]

    bout_h5 = os.path.join(os.path.dirname(os.path.dirname(ik_h5)), f"{recording}_bout.h5")
    with h5py.File(bout_h5, "r") as f:
        fs_imgids = f["fs_imgids"][()][:T]

    # --- assemble silhouette targets ---
    sil_data, meta = build_silhouette_targets(
        root, split, fs_imgids, calib_dir, n_points=n_points,
    )
    anat = load_anatomy(model_xml, mesh_npz)
    fk = make_fk_repose(anat)
    vert_indices = silhouette_fk_indices(mesh_npz, subset="fps_300")

    small = dict(inputs)
    small["q_init"] = q_init
    small["kp_data"] = kp_data

    solver = SilhouetteJaxlsBatchSolver(
        n_iter=n_iter, smooth_weight=smooth_weight, beta=beta, huber_delta=huber_delta)

    def _solve(weight, sd):
        return np.asarray(solver.solve_trajectory(
            q_init=small["q_init"], mjx_model=small["mjx_model"],
            mjx_data_template=small["mjx_data"], kp_data=small["kp_data"],
            qs_to_opt=small["qs_to_opt"], kps_to_opt=small["kps_to_opt"],
            lb=small["lb"], ub=small["ub"], site_idxs=small["site_idxs"],
            q_reg_weights=small["q_reg_weights"],
            fk_repose=fk, vert_indices=vert_indices, sil_data=sd,
            n_pts=n_points, silhouette_weight=weight,
        ))

    q_before = _solve(0.0, None)                       # 3-D only (baseline)
    q_after = _solve(silhouette_weight, sil_data)      # joint 2D+3D

    # --- metrics (IoU + reproj + marker residual), before vs after ---
    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann_multi = {}
    for an in coco["annotations"]:
        id2ann_multi.setdefault(an["image_id"], []).append(an)
    coco_kpnames = coco["keypoint_names"]
    kp_names = list(inputs["kp_names"])
    name2coco = {n: i for i, n in enumerate(coco_kpnames)}

    def _metrics(qtraj):
        ious, soft_ious, reprojs, mresids = [], [], [], []
        for t in range(T):
            cam2img = _cam2img_for_frame(fs_imgids[t], id2file, cam_names)
            kp_mm, kok = _triangulate_kp_mm(rt, kp_names, coco_kpnames, cam2img, id2ann_multi)
            if kok.sum() < 3:
                continue
            s, R, tr = _umeyama(marker_sites[t][kok], kp_mm[kok])
            verts_model = np.asarray(fk(jnp.asarray(qtraj[t].astype(np.float32)), 1.0, vert_indices))
            verts_mm = _model_to_mm(verts_model, s, R, tr)
            # 3-D marker residual (FK sites vs triangulated kp) -- proxy: distance
            # of triangulated markers to their FK positions (kok subset).
            mresids.append(float(np.mean(np.linalg.norm(
                _model_to_mm(marker_sites[t][kok], s, R, tr) - kp_mm[kok], axis=1))))
            for c, iid in cam2img.items():
                ann = _ann_for_image(id2ann_multi, iid, None)
                if ann is None:
                    continue
                fn = id2file[iid]
                mask = _load_sam_mask(root, split, fn, ann["id"])
                if mask is None:
                    continue
                mask = np.asarray(mask)
                uv = np.stack([rt.reproject_point(verts_mm[k])[c] for k in range(0, len(verts_mm))], 0)
                ious.append(iou_of_projected_verts(uv, mask.shape, mask))
                # baseline-comparable soft-IoU (eval-only; not an objective term)
                soft_ious.append(soft_iou_of_verts(uv, mask))
            # reprojection of the 50 kp sites
            for j, nm in enumerate(kp_names):
                ci = name2coco.get(nm)
                if ci is None:
                    continue
                for c, iid in cam2img.items():
                    ann = _ann_for_image(id2ann_multi, iid, None)
                    if ann is None:
                        continue
                    kp2d = np.asarray(ann["keypoints"], float).reshape(-1, 3)
                    if kp2d[ci, 2] > 0:
                        uv_pred = rt.reproject_point(_model_to_mm(marker_sites[t][j][None], s, R, tr)[0])[c]
                        reprojs.append(float(np.linalg.norm(uv_pred - kp2d[ci, :2])))
        return (float(np.mean(ious)) if ious else float("nan"),
                float(np.mean(soft_ious)) if soft_ious else float("nan"),
                float(np.mean(reprojs)) if reprojs else float("nan"),
                float(np.mean(mresids)) if mresids else float("nan"))

    iou_b, soft_b, reproj_b, mres_b = _metrics(q_before)
    iou_a, soft_a, reproj_a, mres_a = _metrics(q_after)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{recording}_polish_qpos.npz"),
             q_before=q_before, q_after=q_after)

    return dict(
        iou_before=iou_b, iou_after=iou_a,
        soft_iou_before=soft_b, soft_iou_after=soft_a,
        reproj_px_before=reproj_b, reproj_px_after=reproj_a,
        marker_resid_before=mres_b, marker_resid_after=mres_a,
        n_frames=T,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="2026_03_18_15_31_22")
    ap.add_argument("--ik-h5", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--n-points", type=int, default=128)
    ap.add_argument("--silhouette-weight", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--huber-delta", type=float, default=0.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--smooth-weight", type=float, default=0.1)
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    rep = run_polish(
        a.recording, ik_h5=a.ik_h5, model_xml=a.xml, mesh_npz=a.mesh, root=a.root,
        split=a.split, calib_dir=a.calib_dir, n_points=a.n_points,
        silhouette_weight=a.silhouette_weight, beta=a.beta, huber_delta=a.huber_delta,
        max_frames=a.max_frames, smooth_weight=a.smooth_weight, n_iter=a.n_iter,
        out_dir=a.out_dir,
    )
    print("PHASE6 POLISH REPORT")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print(f"  soft-IoU delta: {rep['soft_iou_after'] - rep['soft_iou_before']:+.4f} "
          f"(baseline-comparable; cap ~0.76; neutral/small is acceptable and reported truthfully)")
    print(f"  hard-IoU delta: {rep['iou_after'] - rep['iou_before']:+.4f} (point-splat proxy)")
    print(f"  reproj delta (px): {rep['reproj_px_after'] - rep['reproj_px_before']:+.4f} "
          f"(must NOT regress)")
    print(f"  marker-resid delta (mm): {rep['marker_resid_after'] - rep['marker_resid_before']:+.4f} "
          f"(3-D markers must not be materially worse)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_run_silhouette_polish.py -v`
Expected: PASS. The 4 CPU-only tests (`iou_*`, `soft_iou_*`, importable/CLI) pass anywhere; `test_run_polish_report_keys_smoke` runs only where the STAC ik h5 exists.

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/run_silhouette_polish.py third_party/jarvis_jax/tests/test_run_silhouette_polish.py
git commit -m "feat(cse): Phase-6 silhouette-polish validation driver + IoU helper + CPU smoke (Phase 6 Task 7)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Coordinator GPU validation run (NOT a pytest)**

On a gpu-l40s node (coordinator runs this; subagents orphan on long GPU jobs):

```bash
source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m jarvis_jax.tracking.run_silhouette_polish \
  --ik-h5 /gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5 \
  --xml   /gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml \
  --mesh  /gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz \
  --root  /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3 \
  --n-points 128 --silhouette-weight 0.3 --smooth-weight 0.1 --n-iter 50 \
  --out-dir /gscratch/portia/eabe/data/Johnson_lab/cse_work/phase6_polish
```

**`silhouette_weight` / `beta` are a DOCUMENTED SWEEP, not fixed values.** `--silhouette-weight 0.3` and `--beta 8.0` are starting points only. The coordinator sweeps `silhouette_weight ∈ {0.1, 0.3, 1.0}` and `beta ∈ {8.0, 20.0}` on this GPU run (re-invoking the CLI per setting; each run reports before/after independently) and SELECTS the setting by the honest gate below. A too-high weight can trade away keypoint/reprojection accuracy — the gate catches it; expect the sweep to land on the largest weight that still passes the gate.

Expected: each run prints `PHASE6 POLISH REPORT` with `soft_iou_before/after` (baseline-comparable), `iou_before/after` (hard proxy), `reproj_px_before/after`, `marker_resid_before/after`, and the four deltas. **Honest acceptance (per setting):** `soft_iou_after >= soft_iou_before` (up to the ~0.76 collision-mesh cap), `reproj_px_after <= reproj_px_before * 1.1 + 0.5` (no material regression), `marker_resid_after <= marker_resid_before * 1.1` (3-D markers not materially worse). Report BOTH the absolute soft-IoU before/after AND the delta (the absolute is comparable to the known 0.65→0.76 baseline; the delta is the effect of the factor). A neutral/small soft-IoU gain is acceptable and MUST be reported truthfully — the collision-mesh silhouette is a known ceiling; a finer visual mesh (Phase 6b, deferred) is what raises it.

**One-directional Chamfer (SAM→mesh) is the right primary (accepted scope).** The residual only pulls the mesh outline OUT to the mask (fixes wing UNDER-extent). Over-extension is already prevented by the mask-containment training penalty (Phase 5) + the 3-D marker cost anchoring the body, so the SAM→mesh half is the correct primary; the reverse (mesh→SAM) half is a documented future extension, not needed here.

**`fs_imgids` real-data spot-check (first GPU run only):** `build_silhouette_targets` iterates `fs_imgids` via `_cam2img_for_frame` (name-matched). On the FIRST GPU run, verify `meta["n_cam"] == 7` and that `sil_data` has finite boundary blocks for a nonzero fraction of (frame,camera) pairs (i.e. masks were actually found) — a wholly-NaN `sil_data` means the coco/mask lookup mis-threaded and the silhouette factor is a silent no-op. Print `np.isfinite(sil_data).mean()` as a one-line sanity check.

Do NOT commit the output npz or any data.

---

## Self-Review

### 1. Spec-coverage table (Phase-6 requirement → task)

| Phase-6 requirement (brief / spec §6/§11/§12) | Task |
|---|---|
| SAM-mask boundary extraction (offline NumPy, binary-erosion XOR) | Task 1 |
| Uniform boundary-point sampling, N=128-256 target, fixed constant | Task 1 |
| Differentiable sparse Chamfer, SAM-boundary → nearest projected vertex (soft-min) | Task 2 |
| NaN-safety + Huber robustness (marker_cost conventions) | Task 2 |
| Known-geometry value + gradient-flow tests | Task 2 |
| Silhouette IK factor: vertex FK → project_affine → Chamfer as jaxls Cost.factory | Task 3 |
| full_q reconstruction identical to marker_cost; sparse vertex subset (fps_300) | Task 3 |
| Joint solve, additive + OFF-by-default (silhouette + 3-D marker over same SE3/JointVar) | Task 4 |
| BYTE-IDENTICAL no-silhouette invariant (stac_core_jaxls untouched + numeric equality) | Task 4 (`test_stac_core_jaxls_is_byte_identical`, `test_no_silhouette_matches_jaxls_batch_solver`) |
| Joint-solve smoke: weight>0 changes qpos in expected direction | Task 4 (`test_silhouette_weight_moves_qpos_toward_boundary`) |
| Target assembler: SAM masks + refined calib → per-(frame,cam) boundary pts + cam mats, threaded via Var | Task 5 |
| Threading shapes/NaN-for-missing test | Task 5 |
| fps-vs-full-array index-space audit (deferred carry-forward), mirror `_wing_fk_indices` | Task 6 |
| Validation on 2026_03_18_15_31_22 (MALE): IoU/reproj/marker before-after, honest criterion, GPU driver + CPU smoke | Task 7 |
| Camera = affine (no perspective divide), consistent everywhere | Tasks 3/5/7 (`project_affine` convention `X @ M.T + t`; `pack_sil_value` from DLT 3×4; `soft_iou_of_verts` uses affine reproject, not the demo's divide) |
| Efficiency Global Constraint — memory-bounded Chamfer (chunk N via `jax.lax.map`, O(chunk·M) peak) | Task 2 (`chunk_size`; `test_chunking_is_result_invariant_and_bounds_memory`) |
| Efficiency Global Constraint — factor scans C cameras (no `(C,N,M)` tensor), jaxls vmaps T | Task 3 (`jax.lax.scan` over cameras; Memory-bound note) |
| Eval-only soft-IoU (baseline-comparable 0.65→0.76), NOT an objective term | Task 7 (`soft_iou_of_verts`; `test_soft_iou_of_verts_higher_when_verts_fill_mask`) |
| Single fly, V1 anatomy; NO finer mesh / DensePose / per-pixel / soft-IoU OBJECTIVE | Whole plan (objective stays Chamfer; soft-IoU is metric-only at report time, ruled in-scope by coordinator) |

No gap: every locked-scope requirement maps to a task. Explicit non-goals (finer mesh, IUV, per-pixel CSE, soft-IoU **objective** term, >1 fly) are NOT implemented — the Task-7 soft-IoU is an eval-time metric only (coordinator ruling B1), not part of the LM objective.

### 2. Placeholder scan

- Searched for "TODO / TBD / similar to Task N / add appropriate / handle edge cases": none present. Every code step shows real code.
- The earlier `... if False else None` import guard in Task 7 was REMOVED in this revision — Task 7 Step 3 now imports `SilhouetteJaxlsBatchSolver` directly with no placeholder. No placeholder lines remain in any task.
- All commands are exact with expected output. TDD cycle (failing test → run fail → impl → run pass → commit by explicit path) is present in every task.

### 3. Type consistency across tasks

- `chamfer_residual(target_pts, proj_pts, *, beta, huber_delta, chunk_size=32)` — defined Task 2 (chunked over N via `jax.lax.map`, `huber_delta` branched with a STATIC `if` not `jnp.where` to avoid NaN grads at 0), consumed in Task 3's `make_silhouette_cost` scan body (`chamfer_residual(tgt, proj, beta=beta, huber_delta=huber_delta, chunk_size=chunk_size)`). ✓
- `SIL_PER_CAM(n_pts) -> int` and `pack_sil_value(cam_mats, boundary_pts)` / `unpack_sil_value(sil_flat, n_cam, n_pts)` — defined Task 3, consumed by Task 4 (`SIL_PER_CAM`, `pack_sil_value` in tests + solver) and Task 5 (`pack_sil_value`, `SIL_PER_CAM`) and Task 3's own roundtrip test. Layout `[M(6), t(2), boundary(n_pts*2)]` is consistent in pack/unpack. ✓
- `make_silhouette_cost(SE3Var, JointVar, SilVar, *, fk_repose, vert_indices, n_cam, n_pts, beta, huber_delta, silhouette_weight, qs_to_opt, template_qpos, scale=1.0, chunk_size=32)` — defined Task 3 (iterates cameras via `jax.lax.scan`, FK once per frame), called by Task 4 solver with matching kwargs (`template_qpos=mjx_data_template.qpos`, `scale=1.0`; `chunk_size` defaults). ✓
- `fk_repose(qpos, scale=1.0, indices=None)` — the verified `make_fk_repose` signature; Task 3 calls `fk_repose(full_q, scale, vert_indices)` (positional, matches), Task 7 calls `fk(jnp.asarray(...), 1.0, vert_indices)`. ✓
- `SilhouetteJaxlsBatchSolver.solve_trajectory(..., fk_repose, vert_indices, sil_data, n_pts, silhouette_weight)` — defined Task 4; Task 7 `run_polish._solve` passes exactly these. Task 4's own smoke test passes `fk_repose=fk, vert_indices=..., n_pts=n_pts` (consistent). ✓ (Note: Task 4 test's first no-sil call omits `n_pts`, which defaults to 0 and is unused when `use_sil=False` — consistent.)
- `build_silhouette_targets(root, split, fs_imgids, calib_dir, *, n_points, ann_id_by_image, seed) -> (sil_data, meta)` — defined Task 5, consumed by Task 7 with `n_points=n_points`. `meta["n_pts"]==n_points`, `sil_data.shape[-1]==n_cam*SIL_PER_CAM(n_points)`. Task 4 solver infers `n_cam = sil_data.shape[-1]//SIL_PER_CAM(n_pts)` — consistent iff Task 7 passes the same `n_points` as `n_pts` (it does). ✓
- `silhouette_fk_indices(mesh_npz, subset, exclude_seg_ids) -> (M,) int32 full-array` — defined Task 6, consumed by Task 7 (`silhouette_fk_indices(mesh_npz, subset="fps_300")`). ✓
- `iou_of_projected_verts(verts2d, mask_shape, ref_mask) -> float` (hard proxy) and `soft_iou_of_verts(verts2d, mask, *, sigma=1.3, splat_k=2) -> float` (eval-only soft-IoU, baseline-comparable) — defined + tested Task 7, both used inside `run_polish._metrics` (which now returns a 4-tuple `(iou, soft_iou, reproj, mres)`; report dict adds `soft_iou_before/after`). ✓
- Existing consumed signatures verified against source: `project_affine(P, X) = X @ P[:2,:3].T + P[:2,3]` (affine_camera.py:45-48); `make_fk_repose` returns `fk_repose(qpos, scale=1.0, indices=None)` (silhouette_ik.py:38-54); `_wing_fk_indices` fps[idx] bridge (silhouette_ik_solve.py:365-397); `JaxlsBatchSolver._build_se3` marker/reg/limit/smoothness formulas + `_solve_se3` init (stac_core_jaxls.py:158-296, 500-553); `ReprojectionTool._camera_list[c].cameraMatrix` (3,4) + `.reproject_point` + `.reconstruct_point` (reprojection_tool.py); `_cam2img_for_frame`/`_ann_for_image`/`_load_sam_mask`/`_umeyama`/`_model_to_mm`/`_triangulate_kp_mm` (silhouette_ik_solve.py). ✓

No inconsistencies found.
