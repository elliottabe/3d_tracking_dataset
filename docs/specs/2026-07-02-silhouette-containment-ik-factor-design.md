# Coverage-Asymmetric, Confidence-Weighted Silhouette IK Factor — Design

**Date:** 2026-07-02
**Status:** Design (approved for planning)
**Depends on:** `docs/specs/2026-06-30-multiview-silhouette-ik-pipeline-design.md` (parent pipeline), the watertight visual mesh (`fly_v1_visual_canonical_wings.npz`, commit 0cfe122), and the existing silhouette IK subsystem in `third_party/jarvis_jax/jarvis_jax/cse/`.

## 1. Motivation

The multiview fly IK must recover **female** kinematics during courtship, when she is often on the chamber walls / off the flat ground — a pose regime under-represented in training, so her 2-D keypoints are unreliable (measured: JAX detector female MPJPE 31 px vs male 9 px). The pipeline's answer is to lean on the **silhouette** (from SAM masks) where keypoints fail.

Two problems block the existing silhouette term:

1. **Wrong direction for the halo.** The current factor (`make_silhouette_cost` → `chamfer_residual`) is one-directional: it pulls each SAM-mask-boundary point to the nearest projected mesh vertex, i.e. it drags the mesh **outward to the mask boundary**. But the SAM mask on this reflective chamber floor includes a **shadow/reflection halo** (a thin inflated ring around the true fly — visible in the real-frame overlay). That term would pull the fit outward into the shadow.
2. **No confidence weighting and no DOF restriction.** The term has only a single global scalar weight. There is no way to (a) weight individual observations by reliability, or (b) restrict the silhouette's influence to the appendage DOFs the keypoints miss while keeping the reliable body core keypoint-driven.

This design adds a **coverage-asymmetric, confidence-weighted, DOF-restricted** silhouette factor and fixes the halo.

## 2. Goal & success criterion

Add a silhouette term to the existing jaxls fly IK (`SilhouetteJaxlsBatchSolver.solve_trajectory`) that:

- penalizes projected mesh verts **outside** the mask hard and inside softly (asymmetry robust to the halo),
- pulls appendages **out to the true fly boundary** (halo stripped) so wings/legs reach full extent,
- carries **per-observation confidence** weights (uniform for the male now; keypoint-confidence-driven for the female later),
- influences **only appendage DOFs** (wings + legs + abdomen) via a factor-specific DOF mask, leaving the shared keypoint DOF mask untouched.

**Success criterion (validation target: MALE recording `2026_03_18_15_31_22`, watertight visual mesh):** compare baseline (silhouette weights = 0) vs treatment.

- **PASS (primary):** mean multi-view mesh-vs-SAM **IoU** of the fitted pose increases vs the keypoint-only baseline.
- **Do-no-harm guard:** keypoint reprojection RMSE and MPJPE-vs-STAC must not worsen beyond tolerance (defaults: reproj RMSE not worse by > 0.5 px; MPJPE-vs-STAC not worse by > 2%). Tolerances are configurable.

The male has reliable keypoints + a STAC fit, so this proves the factor helps the silhouette without wrecking the good keypoint fit. IoU is the only one of these metrics that transfers to the female case (no GT there).

## 3. Architecture

Two complementary residuals, both reusing the differentiable in-loop `fk_repose` (`silhouette_ik.py:38`) + affine/telecentric projection (`proj = verts3d @ M.T + t`, no perspective divide), added as jaxls `Cost` objects in `solve_trajectory`:

### 3.1 Containment residual (NEW, mesh → mask)

For frame `t`, camera `c`, appendage vertex `v`:

```
proj      = fk_repose(full_q, scale, appendage_vert_idx) @ M_c.T + t_c     # (M,2) px
grid      = (proj - grid_offset[t,c]) * grid_scale[t,c]                    # (M,2) SDF-grid coords
d         = bilinear_sample(sdf[t,c], grid)                                # (M,) signed dist, ORIGINAL px, (- inside / + outside)
r_{tcv}   = sqrt(lambda_cont) * conf_v * present[t,c] * relu(d + margin)   # (M,)
```

- `sdf[t,c]` is the **signed distance transform** of the SAM mask, cropped to the fly window and resized to a fixed grid, with distances rescaled to **original-image pixels**.
- `relu(d + margin)` is the asymmetry: verts outside the mask (d > −margin) are penalized ∝ how far outside; verts comfortably inside contribute 0. `margin` default 0.
- Verts projecting outside the crop grid clamp to the border (a large positive `d`) → penalized, which is correct (far outside the mask).
- Missing camera (`present[t,c] == False`) or NaN rows → 0 residual (existing NaN-safe pattern).

### 3.2 Coverage residual (EXISTING Chamfer, halo-fixed)

Reuse `chamfer_residual` (`silhouette_chamfer.py:37`, one-directional SAM-boundary → nearest mesh vertex, softmin, Huber-capable, NaN-safe, chunk-invariant), with two changes:

- boundary points sampled from an **eroded** SAM mask (erode by `erode_px`, default ~8) so the pull target is the **true** fly boundary, not the inflated shadow ring;
- multiply the per-point residual by a **per-point confidence** `conf_p` (currently absent — only a global scalar) and by `sqrt(silhouette_weight)` (the existing coverage weight, kept for back-compat).

Coverage projects the existing silhouette vertex subset (`silhouette_fk_indices`, default `fps_300`) so each mask-boundary point finds its correct nearest part; the DOF mask (§3.3) ensures only appendage DOFs actually move.

### 3.3 Combined effect

Containment bounds the mesh from spilling **outside** the (raw) mask; coverage pulls it **out to** the eroded boundary. The mesh edge settles in the band between the eroded and raw boundaries — i.e. the true fly edge, with the halo ignored.

## 4. DOF restriction & confidence

### 4.1 Factor-specific DOF mask

A boolean `sil_qs_mask` (shape matching the existing q-assembly) selecting **wing + leg + abdomen** hinge DOFs, closed over the silhouette cost factory. Inside the residual: `full_q = where(sil_qs_mask, q, template_qpos)` **before FK**, so the silhouette gradient reaches only appendage DOFs. The keypoint term's shared `qs_to_opt` is untouched → keypoints keep steering root/head/thorax.

`sil_qs_mask` is built by joint-name matching against the model (`wing`, `coxa|femur|tibia|tarsus|leg`, `abdomen|abd`) mapped to their qpos DOF indices, aligned to the same q layout the existing factor uses. Root (free joint) is excluded.

### 4.2 Confidence plumbing

- `conf_v` (M,) per appendage vertex — containment.
- `conf_p` (T, n_cam, n_pts) per boundary point — coverage.
- `present` / `w_cam` (T, n_cam) per-camera gate.

All default to **uniform 1.0** (present-camera gating still applied) for the male validation — this design delivers the *mechanism*. Deriving `conf` from actual keypoint confidence is a follow-up for the female deployment (Non-goals §9).

## 5. Data feeding — memory efficiency

The current factor packs camera matrices + boundary points into a large per-frame `SilVar` (~1848 dims/frame), which forces the CG linear solver and balloons the normal equations. This design **replaces `SilVar` with a tiny per-frame `FrameVar` carrying the integer frame index `t`**. The heavy constants — `sdf[T,n_cam,H,W]`, `grid_scale`/`grid_offset`, eroded boundary points, `conf`, per-camera affine matrices — are **closed over the cost factory** and selected inside the vmapped residual by `t` (dynamic gather, `stop_gradient` on `FrameVar`).

- The jaxls optimization variable stays tiny (root SE3 + appendage hinges + a scalar frame index) — no CG blowup from variable size; the existing CG override can remain as a safety net for the added residual count.
- SDF constant memory: `(T≈136, n_cam=7, 128, 128)` f32 ≈ 1.25 GB on device (L40S 48 GB, fine). Fallbacks if tight: f16 storage and/or `96×96` grid. Aligns with the project's JAX-first + memory-efficiency constraint.

## 6. Component / file structure

All under `third_party/jarvis_jax/jarvis_jax/cse/` unless noted. `stac-mjx/stac_mjx/stac_core_jaxls.py` stays **byte-identical** (guarded by an existing `git diff` test).

**New — `silhouette_sdf.py`** (offline precompute, NumPy/cv2/scipy, no JAX):
```python
def build_sdf_stack(masks_by_frame, cam_names, *, out_hw=(128,128), bbox_margin=0.4,
                    signed=True) -> dict:
    """masks_by_frame: list[T] of dict {cam_index -> bool mask (H,W) | absent}.
    Returns dict:
      sdf         (T, n_cam, out_h, out_w) float32  signed distance in ORIGINAL px (neg inside)
      grid_scale  (T, n_cam, 2) float32   grid = (proj - grid_offset) * grid_scale
      grid_offset (T, n_cam, 2) float32
      present     (T, n_cam) bool
    Absent/empty cam -> present False, sdf filled with a large positive constant."""
```

**New — `silhouette_containment.py`** (JAX residual, CPU-testable without a model):
```python
def bilinear_sample(grid_img, xy) -> jnp.ndarray:  # grid_img (H,W), xy (M,2) -> (M,) clamped
def containment_residual(verts3d, cam_M, cam_t, sdf, grid_scale, grid_offset, conf,
                         *, margin=0.0, present=True) -> jnp.ndarray:  # -> (M,)
```

**Modify — `silhouette_joint_ik.py`:**
- add `make_containment_cost(SE3Var, JointVar, FrameVar, *, fk_repose, appendage_vert_idx, cam_Ms, cam_ts, sdf_all, grid_scale_all, grid_offset_all, present_all, conf_v, sil_qs_mask, template_qpos, margin, containment_weight, scale=1.0, chunk_size=32)` returning a `@jaxls.Cost.factory` closure `(vals, root, joint, frame) -> (n_cam*M,)`;
- introduce `FrameVar` (tiny int per-frame Var; `stop_gradient`); refactor coverage (`make_silhouette_cost`) to also read its constants via `FrameVar` + closures (retire `SilVar`), adding `conf_p` and eroded boundary;
- extend `SilhouetteJaxlsBatchSolver.solve_trajectory(...)` with `containment_weight=0.0` (new), keep `silhouette_weight` as the coverage weight (back-compat), plus `sil_qs_mask=None`, `conf_v=None`, `conf_p=None`, and the precomputed SDF/boundary constants; append both costs + the shared `FrameVar`; keep the CG override. Baseline = both weights 0 → reproduces `JaxlsBatchSolver`.

**Modify — `silhouette_targets.py`:** add `erode_px` to the boundary sampler and emit per-point `conf_p` (uniform default, present-gated).

**Modify — `run_silhouette_polish.py`:** wire `containment_weight`, `erode_px`, `sil_qs_mask`, confidence; run baseline vs treatment; report **IoU + do-no-harm guard** (keypoint reproj RMSE, MPJPE-vs-STAC) via the existing metric helpers + filled-triangle IoU (the overlay methodology: fitted qpos → full-mesh FK → affine project → `cv2.fillPoly` → IoU vs SAM).

**New — helper** `appendage_vertex_indices(anat, include=("wing","leg","abdomen")) -> (M,) int32` (full-array indices of appendage-segment verts) and `build_appendage_dof_mask(model, include=(...)) -> bool mask`. Placement: `silhouette_targets.py` or a small `silhouette_dof.py` (decided in planning).

## 7. Data flow

```
SAM masks + refined calib + STAC qpos init (ik h5)
  -> build_silhouette_targets(..., erode_px)            # eroded boundary pts + conf_p
  -> build_sdf_stack(masks, cams, out_hw)               # cropped signed-distance stack + transforms
  -> load_anatomy + make_fk_repose                      # differentiable FK
  -> appendage_vertex_indices / build_appendage_dof_mask
  -> SilhouetteJaxlsBatchSolver.solve_trajectory(
         containment_weight, silhouette_weight, sil_qs_mask, conf_v, conf_p, sdf_*, boundary_*)
       # costs: marker(3D kp) + reg + limit + smoothness + coverage + containment ; FrameVar selects per-frame data
  -> fitted qpos (T,nq)
  -> report: IoU (fillPoly vs SAM) ; keypoint reproj RMSE ; MPJPE-vs-STAC
```

## 8. Testing (CPU: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/<file> -v`)

- **`test_silhouette_sdf.py`** — signed distance on a known filled square (sign flips at boundary; interior negative, exterior positive, magnitudes ≈ pixel distance after rescale); crop-transform roundtrip (a known original-px point maps to the expected grid coord); absent camera → `present False` + large-positive fill.
- **`test_silhouette_containment.py`** — vertex projecting **outside** the mask → positive residual ∝ distance; **inside** → ~0; `jax.grad` pulls an outside vertex **inward**; NaN/absent-camera → finite 0 residual + finite grad; **chunk-invariant** value + gradient; `bilinear_sample` matches manual interpolation on a small grid and clamps out-of-bounds.
- **extend `test_silhouette_joint_solve.py`** — `FrameVar` path with `containment_weight=0, silhouette_weight=0` reproduces `JaxlsBatchSolver` to `atol=1e-5`; `sil_qs_mask` restricted to appendages moves **only** appendage DOFs (core hinges unchanged) when silhouette weights > 0; a synthetic outside-mask configuration has its outside-vertex count **reduced** after a few LM steps; `stac_core_jaxls.py` byte-identical test still passes.

Heavy real-recording tests (needing the ik h5 / mesh / masks) skip when assets are absent, matching the existing suite.

## 9. Non-goals (explicitly out of scope)

- Female deployment / tuning and deriving `conf` from real keypoint confidence (follow-up once the male validation passes).
- The dense-pose CSE 2-D factor and any per-pixel dense-silhouette term (later pipeline phase).
- Any change to `stac_core_jaxls.py` or the frozen `stac-mjx` submodule internals.
- Retraining the keypoint detector.

## 10. Constraints

- Env: gpu-l40s compute node, `micromamba activate 3d_tracking`, `unset LD_LIBRARY_PATH`; run mjx/jaxls on GPU (not CPU) for real fits; unit tests CPU-only via the pytest invocation above.
- New code under `third_party/jarvis_jax/jarvis_jax/cse/`, tests under `third_party/jarvis_jax/tests/`.
- Affine/telecentric projection convention everywhere (3×4 DLT, 3rd row `[0,0,0,1]`, no perspective divide).
- Do not commit checkpoints/data/label-npz/masks/outputs/videos; commit by explicit path.
