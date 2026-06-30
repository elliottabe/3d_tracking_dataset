# Multi-view Silhouette + Keypoint IK Pipeline — Design

- **Date:** 2026-06-30
- **Status:** Approved design (pre-implementation-plan)
- **Repo:** `3d_tracking_dataset` (JAX stack under `third_party/jarvis_jax`)
- **Anatomy at bring-up:** V1 (`fruitfly_v1/fruitfly_v1_free.xml`, nq=93) + `fly_v1_collision_canonical_wings.npz`

## 1. Goal

Produce **accurate 3-D inverse kinematics** (joint angles + root pose) for fruit flies
from 7 calibrated cameras, robust across four scenarios — **single animal, multi-animal
(courtship), amputation, headless** — and across body-model anatomies (V1 now; V2/V2.1
later). The dense surface (dense pose) falls out as a byproduct of the fit.

Design principle: **maximally leverage the 7 calibrated views** — fuse all views into one
articulated-model fit per fly so each view fills in what the others miss.

## 2. Background / findings that motivate this design

Established empirically this cycle (see `cse_work/viz/*`):

- **Wing under-spread is a 2-D image-evidence limit**, not a training defect: raw per-vertex
  predictions recover only ~0.66 of true wing length; a heatmap-σ fine-tune and a from-scratch
  argument both fail to fix it (transparent, frequently-occluded tip).
- **The fitted *mesh* has correct wings by construction** — model-fitting to keypoints already
  yields full-size wings even when raw dense points are compressed. The wing problem lives in the
  *raw points*, not in a mesh fit.
- **SAM masks carry the true wing extent** (95% of wing verts inside the mask; 100% when the wing
  is extended — the behaviorally important state). Triangulating the SAM wing-tip across cameras
  recovers wing length 0.66 → **1.09** and tip error 6.2 → **1.7 mm**.
- **The female (keypoint-hard case) is 3-D-trackable from masks alone** — SAM segments her in all
  7 cameras; her silhouette centroid triangulates leave-one-out to ~8 px.
- **Multi-view reprojection validation works**: predicted keypoints reproject at 3.5 px median
  across all 7 cameras; SAM wing-tip leave-one-out at 6 px.

Conclusion: the right architecture is **model-fitting IK driven by per-view keypoints + dense
observations + SAM silhouette**, fused across all cameras, temporally coupled.

## 3. Key decisions

1. **Scope:** full end-to-end pipeline (detection → correspondence → IK → outputs).
2. **Outputs:** qpos + root pose; posed 3-D mesh (dense pose); 3-D keypoints (world mm);
   reprojection overlays + QC metrics. All per fly per frame.
3. **Topology variants (headless / amputation):** one model per anatomy + a per-recording
   **active-parts mask** (parts on/off) that simultaneously drops those markers, excludes those
   geoms from the silhouette, and locks those joints. No per-case XMLs.
4. **Temporal:** reuse the existing **jaxls batch solver** (`stac_core_jaxls.JaxlsBatchSolver`)
   that solves all T frames jointly (SE3 root + joint hinges + smoothness + limits + reg).
5. **Silhouette integration (approach C):** start with **silhouette landmarks as extra weighted
   markers** in the existing `marker_cost` (reuses the whole temporal solver), then add a **dense
   silhouette `jaxls.Cost`** as a refinement stage.
6. **Detector:** unified **JAX multi-head** (mask + 50 keypoints + M dense verts), **dilated-mask
   gated**, **pluggable backbone** (JAX ViT-B now; optional ported SAM3 ViTDet trunk later). Dense
   pose **retained as observations** (not as final output); the canonical mesh provides the
   geometric prior the graph-Laplacian used to.
7. **Observation model:** **per-view 2-D reprojection residuals** (not pre-triangulated 3-D), so a
   point seen in even one view still constrains the fit and per-view occlusion is handled by weights.
8. **SAM3 front-end stays PyTorch:** concept-detect + propagate + identity is SOTA, stateful, and
   runs once per session as preprocessing. **Its tracker identity is reused as the per-camera
   multi-animal correspondence.** A full SAM3 JAX port is an explicit non-goal (see §10).

## 4. Architecture & data flow

```
 anatomy config: (model_xml, canonical_mesh_npz)   [V1 | V2 | V2.1]
 active-parts mask: {head, legT1L..T3R, wings} on/off   [per recording]

 raw videos (7 cams) + calibration
   │
   ▼
[0] SAM3 FRONT-END (PyTorch, keep)  concept-detect flies → propagate masks + identity
        per cam / frame / fly  →  masks + per-camera track-IDs
   │
   ▼
[1] CROSS-CAMERA IDENTITY LINK  tie per-camera track-IDs into one global fly-ID
        (epipolar / triangulation consistency; max 2 flies)
   │
   ▼
[2] JAX DETECTOR HEAD (mask + 50 kp + M dense verts, dilated-mask gated)
        per fly mask  →  (kp, dense, confidence) per view
   │
   ▼
[3] SILHOUETTE-LANDMARK EXTRACTOR  SAM mask → wing tips / body extremes → 3D landmark markers
   │
   ▼
[4] OBSERVATION ASSEMBLER  per-view weighted obs set (kp + dense + silhouette),
        weights = confidence × visibility × active-part mask
   │
   ▼
[5] TEMPORAL IK (jaxls, per fly)  all T frames jointly
        vars : SE3 root ⊕ active joint hinges
        costs: marker_cost (2D-reproj: kp + dense + silhouette landmarks)
               + smoothness (temporal) + joint limits + rest-pose reg
               [+ dense silhouette factor — phase 5]
   │
   ▼
[6] OUTPUTS + QC  qpos+root · posed mesh (FK→world) · 3D kp ·
        reprojection overlays (all cams) · reproj-err / silhouette-IoU / LOO QC
```

### Scenario handling (two config axes cover all four)

| scenario | instance count | active-parts mask | keypoint reliance |
|---|---|---|---|
| single (free-walk / stationary) | 1 | all on | normal |
| multi (courtship) | 2 + identity link | all on (per fly) | male normal; **female silhouette-driven** (low-conf kp down-weighted) |
| amputation | 1–2 | missing leg(s) **off** | missing-leg kp dropped; silhouette excludes those geoms |
| headless | 1 | head **off** | head kp dropped; silhouette excludes head mesh |

## 5. Components & interfaces

`(E)` exists · `(M)` modify · `(N)` new.

| # | Component | Responsibility | Interface (→in / ←out) |
|---|---|---|---|
| 0 | SAM3 front-end (PyTorch) **(E)** | concept-detect + propagate masks + identity | → videos, calib, concept prompt · ← per-cam/frame/fly masks + track-IDs (npz) |
| 1 | Cross-cam identity linker **(N)** | global fly-ID via epipolar/triangulation | → per-cam masks+IDs, calib · ← global fly-ID map |
| 2 | JAX detector head **(M)** (from ViTPose) | 50 kp + M dense + conf; dilated-mask gated; pluggable backbone | → masked crop · ← (kp, dense, conf) per view |
| 3 | Silhouette-landmark extractor **(N)** | SAM mask → wing tips / body extremes → 3D landmarks | → per-cam masks, calib, pred axis · ← 3D landmark markers + weights |
| 4 | Anatomy + active-parts config **(N)** | declare model+mesh + active joint/marker/geom masks | → scenario id · ← model, mesh, masks |
| 5 | Observation assembler **(N)** | per-view weighted obs (kp+dense+silhouette) | → 2,3,4 · ← per-view weighted obs set per fly |
| 6 | Temporal jaxls IK **(M)** (from `stac_core_jaxls`) | all-T solve; marker (2D-reproj) + smooth + limits + reg + dense-silhouette factor | → obs, model, masks · ← qpos+root trajectory, scale |
| 7 | Outputs + QC **(N)** | qpos/root, posed mesh, 3D kp, overlays, QC | → IK result, videos, calib · ← CSV/h5 + videos + QC report |

### Foundations already built this cycle (Phase 0)

- `jarvis_jax/cse/silhouette_ik.py` — anatomy-agnostic mjx FK + repose; verified vs MuJoCo (9e-08), differentiable.
- `jarvis_jax/cse/silhouette_render_demo.py` — qpos→FK→mm→all-camera projection, validated against SAM masks.
- `jarvis_jax/cse/silhouette_fit.py` — JAX soft-silhouette rasterizer + differentiable wing-joint recovery (IoU 0.65→0.76).
- `jarvis_jax/cse/demo_wingtip_triangulation.py`, `reproj_validate.py` — SAM wing-tip triangulation + multi-view reprojection validation.

## 6. The IK objective (jaxls)

Per fly, all T frames in one `LeastSquaresProblem`:

- **Variables:** `SE3Var` (root, tangent-6) ⊕ `JointVar` (active hinges only). Inactive joints
  (per active-parts mask) are held at rest.
- **marker_cost** (2-D reprojection): for each marker (50 kp + M dense verts + silhouette
  landmarks) and each camera that sees it, residual = `project(FK_marker(qpos), P_cam) − obs_2D`,
  weighted by `confidence × visibility × active`. NaN-safe (occluded → zero residual).
- **smoothness_cost** (T−1): `||[SE3_log_diff; joint_diff]||² · smooth_w`.
- **limit_cost**: joint limits (constraint).
- **reg_cost**: rest-pose hinge regularization.
- **dense silhouette factor** (Phase 5): render soft silhouette per camera; residual =
  Chamfer(projected mesh boundary, SAM mask boundary) or per-pixel `(render − mask)`.

Scale: per-fly model→world similarity (Umeyama from markers / mask extent), consistent with the
existing STAC preprocessing.

## 7. Detector (unified JAX multi-head)

- One JAX model: `image → {mask, 50 kp, M dense verts}`, shared backbone, multi-GPU + XLA.
- **Dilated-mask gating**: kp/dense heatmaps gated by the dilated SAM mask (inference) + a training
  penalty for mass outside — predictions must land on the fly.
- **Pluggable backbone**: Phase 4 uses the existing JAX ViT-B (reuse ViTPose weights); optional
  later port of the SAM3 **ViTDet vision trunk** to JAX + weight transfer, with PyTorch SAM3 masks
  as a **distillation** target. (SAM3 VL/concept/memory/tracker are *not* ported.)
- Dense pose is an **observation source** for the IK, never the final output.

## 8. Multi-view fusion principle

All observations enter the IK as **per-view 2-D reprojection residuals**, confidence/visibility
weighted. Consequences: single-view points still constrain the fit; occlusions handled by weights;
the known mesh + temporal smoothness fuse 7 views × T frames into one qpos trajectory per fly.
This is the mechanism by which views "fill in each other's missing information."

## 9. Outputs & QC

Per fly per frame: `qpos` + root (model frame, + scale); posed mesh vertices (world mm); 3-D
keypoints (world mm). Per bout: reprojection overlay videos (fitted mesh/keypoints on every
camera); QC metrics — per-camera reprojection error, silhouette IoU, leave-one-out reprojection —
for outlier flagging.

## 10. Non-goals / future

- **Full SAM3 JAX port** (VL backbone + concept head + memory + video tracker). Large, high-risk,
  low ROI (front-end runs once). Revisit only if zero-PyTorch dependency becomes a hard requirement.
- Real-time/online inference (this is offline analysis).
- >2 animals (dataset is ≤2 per image).

## 11. Build phases (each independently validatable)

- **Phase 0 — Foundations (done):** mjx FK+repose, render geometry, soft-silhouette fit + wing-tip
  triangulation POCs.
- **Phase 1 — Silhouette-landmark IK, single fly (V1):** extractor (#3) + silhouette markers in
  the temporal jaxls IK (#6, approach A) with the 2-D-reprojection observation model. Validate male
  vs STAC + reprojection/LOO.
- **Phase 2 — Multi-animal:** identity linker (#1) + per-fly solve; female rides on silhouette.
  Validate on courtship (both flies reproject cleanly).
- **Phase 3 — Active-parts:** config (#4) for headless + amputation; verify no phantom residuals.
- **Phase 4 — Unified JAX detector:** dense head + mask-gating (#2); feed dense observations;
  optional SAM3 ViTDet trunk port + mask distillation.
- **Phase 5 — Dense-silhouette polish (#6, approach B) + finer visual mesh** for shape fidelity.
- **Phase 6 — Outputs/QC (#7) + deploy** across all scenarios + anatomies (V2/V2.1 configs).

## 12. Risks / assumptions

- **mjx loads each anatomy** for differentiable FK — confirmed for V1 (nq=93) and v2.1_muscles.
- **SAM3 identity is reliable enough** per camera to link across views; the "largest extra"
  heuristic worked 6/7 in spot checks — the linker must handle the occasional miss (epipolar gating).
- **Silhouette-blind DOF** (wing-roll, global orientation/symmetry) are resolved by keypoints +
  temporal continuity + priors, not the silhouette alone. Frames lacking both a reliable orientation
  cue and a good neighbor remain ambiguous — temporal coupling is the mitigation.
- **Collision-mesh silhouette** caps IoU (~0.76); a finer visual mesh (Phase 5) raises the ceiling.
- Dense-silhouette factor cost (render-in-the-LM-loop) must stay tractable — sparse Chamfer over
  sampled boundary points preferred over per-pixel.
