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
7. **Observation model: 3-D.** Multi-view observations are **triangulated to 3-D** and the model
   markers are fit to those 3-D points (reuses the existing `stac_core_jaxls` marker cost + the STAC
   offset-fitting machinery unchanged). Triangulated **SAM silhouette landmarks** enter as extra 3-D
   markers. The dense-silhouette polish (Phase 6) remains a separate, optional **2-D render factor**.
8. **Camera model: affine / orthographic (telecentric lenses).** Calibration is a per-camera **3×4
   DLT matrix with 3rd row `[0,0,0,1]`** (verified) — projection is affine (no perspective divide),
   and triangulation is affine (relies on the angular diversity of the 7 views, which is ample). The
   same DLT matrices are used **consistently** everywhere (triangulation, IK, silhouette render).
9. **SAM3 front-end stays PyTorch:** concept-detect + propagate + identity is SOTA, stateful, and
   runs once per session as preprocessing. **Its tracker identity is reused as the per-camera
   multi-animal correspondence.** A full SAM3 JAX port is an explicit non-goal (see §10).
10. **Per-recording bundle adjustment (calibration refinement):** a preprocessing stage that
    refines the per-camera **affine DLT** parameters to minimize multi-view reprojection of
    keypoint tracks, robust (Huber) + soft-prior to the factory calibration + fixed gauge.
    **Configurable refine set; default = affine pose ("extrinsics") only** (rotation + in-plane
    translation), holding magnification/shear. Itself a jaxls least-squares problem.

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
[1] JAX DETECTOR HEAD (mask + 50 kp + M dense verts, dilated-mask gated)
        per fly mask  →  (kp, dense, confidence) per view
   │
   ▼
[2] CROSS-CAMERA IDENTITY LINK  tie per-camera track-IDs into one global fly-ID
        (epipolar / triangulation consistency; max 2 flies)
   │
   ▼
[3] PER-RECORDING BUNDLE ADJUSTMENT  refine affine DLT cameras from keypoint tracks
        (jaxls; robust + prior-to-factory + fixed gauge; default: affine pose only)
        →  refined per-camera matrices used by ALL stages below
   │
   ▼
[4] SILHOUETTE-LANDMARK EXTRACTOR  SAM mask → wing tips / body extremes → triangulate → 3D markers
   │
   ▼
[5] OBSERVATION ASSEMBLER + TRIANGULATION  fuse views → 3-D observations (kp + dense + silhouette);
        weights = confidence × visibility × active-part mask
   │
   ▼
[6] TEMPORAL IK (jaxls, per fly)  all T frames jointly
        vars : SE3 root ⊕ active joint hinges
        costs: marker_cost (3-D: kp + dense + silhouette landmarks)
               + smoothness (temporal) + joint limits + rest-pose reg
               [+ dense silhouette 2-D render factor — phase 5]
   │
   ▼
[7] OUTPUTS + QC  qpos+root · posed mesh (FK→world) · 3D kp ·
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
| 1 | JAX detector head **(M)** (from ViTPose) | 50 kp + M dense + conf; dilated-mask gated; pluggable backbone | → masked crop · ← (kp, dense, conf) per view |
| 2 | Cross-cam identity linker **(N)** | global fly-ID via epipolar/triangulation | → per-cam masks+IDs, calib · ← global fly-ID map |
| 3 | Per-recording bundle adjustment **(N)** | refine affine DLT cameras from kp tracks (jaxls; robust + prior + fixed gauge; configurable, default affine-pose) | → kp tracks, factory calib · ← refined per-camera matrices + reproj before/after |
| 4 | Silhouette-landmark extractor **(N)** | SAM mask → wing tips / body extremes → triangulate → 3D markers | → per-cam masks, refined calib · ← 3D landmark markers + weights |
| 5 | Anatomy + active-parts config **(N)** | declare model+mesh + active joint/marker/geom masks | → scenario id · ← model, mesh, masks |
| 6 | Observation assembler + triangulation **(N)** | fuse views → 3-D observations (kp+dense+silhouette), weighted | → 1,4,5, refined calib · ← 3-D observation set per fly |
| 7 | Temporal jaxls IK **(M)** (from `stac_core_jaxls`) | all-T solve; marker_cost (3-D) + smooth + limits + reg; + dense-silhouette 2-D factor (Ph5) | → obs, model, masks · ← qpos+root trajectory, scale |
| 8 | Outputs + QC **(N)** | qpos/root, posed mesh, 3D kp, overlays, QC | → IK result, videos, calib · ← CSV/h5 + videos + QC report |

### Foundations already built this cycle (Phase 0)

- `jarvis_jax/cse/silhouette_ik.py` — anatomy-agnostic mjx FK + repose; verified vs MuJoCo (9e-08), differentiable.
- `jarvis_jax/cse/silhouette_render_demo.py` — qpos→FK→mm→all-camera projection, validated against SAM masks.
- `jarvis_jax/cse/silhouette_fit.py` — JAX soft-silhouette rasterizer + differentiable wing-joint recovery (IoU 0.65→0.76).
- `jarvis_jax/cse/demo_wingtip_triangulation.py`, `reproj_validate.py` — SAM wing-tip triangulation + multi-view reprojection validation.

## 6. The IK objective (jaxls)

Per fly, all T frames in one `LeastSquaresProblem`:

- **Variables:** `SE3Var` (root, tangent-6) ⊕ `JointVar` (active hinges only). Inactive joints
  (per active-parts mask) are held at rest.
- **marker_cost** (3-D): for each marker (50 kp + M dense verts + triangulated silhouette
  landmarks), residual = `FK_marker(qpos) − obs_3D`, weighted by `confidence × visibility ×
  active`. NaN-safe (missing/occluded → zero residual). This is the existing `stac_core_jaxls`
  cost, unchanged; observations are triangulated upstream (component 6).
- **smoothness_cost** (T−1): `||[SE3_log_diff; joint_diff]||² · smooth_w`.
- **limit_cost**: joint limits (constraint).
- **reg_cost**: rest-pose hinge regularization.
- **dense silhouette factor** (Phase 6, optional): a separate **2-D** term — render soft silhouette
  per camera (affine projection); residual = Chamfer(projected mesh boundary, SAM mask boundary)
  over sampled boundary points. The only 2-D term; the rest of the IK stays 3-D.

Scale: per-fly model→world similarity (Umeyama from markers / mask extent), consistent with the
existing STAC preprocessing.

## 7. Detector (unified JAX multi-head)

- One JAX model: `image → {mask, 50 kp, M dense verts}`, shared backbone, multi-GPU + XLA.
- **Dilated-mask gating**: kp/dense heatmaps gated by the dilated SAM mask (inference) + a training
  penalty for mass outside — predictions must land on the fly.
- **Pluggable backbone**: Phase 5 uses the existing JAX ViT-B (reuse ViTPose weights); optional
  later port of the SAM3 **ViTDet vision trunk** to JAX + weight transfer, with PyTorch SAM3 masks
  as a **distillation** target. (SAM3 VL/concept/memory/tracker are *not* ported.)
- Dense pose is an **observation source** for the IK, never the final output.

## 8. Multi-view fusion principle

Fusion happens at **triangulation**: each observation (keypoint, dense vertex, silhouette landmark)
is combined across all cameras that see it into one 3-D point (affine DLT triangulation), then the
articulated model is fit to those 3-D points and temporally smoothed — fusing 7 views × T frames
into one qpos trajectory per fly. The per-recording bundle adjustment first tightens the cameras so
this triangulation is as accurate as possible.

Tradeoff of the 3-D model (vs per-view 2-D reprojection): a point seen in only **one** view can't be
triangulated and is dropped for that frame. With 7 affine views this is rare; the dense-silhouette
2-D factor (Phase 6) recovers extent where triangulated landmarks are sparse. Affine (telecentric)
triangulation relies on the **angular diversity** of the 7 cameras to resolve depth — amply
satisfied by the rig.

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
- **Phase 1 — Bundle adjustment (#3):** per-recording affine-camera refinement (jaxls; robust +
  prior + fixed gauge; configurable, default affine-pose). Validate by reprojection-error drop;
  fall back to factory calibration if it worsens.
- **Phase 2 — Silhouette-landmark IK, single fly (V1):** extractor (#4) + triangulated silhouette
  markers in the temporal jaxls IK (#7, approach A), 3-D observation model. Validate male vs STAC +
  reprojection/LOO.
- **Phase 3 — Multi-animal:** identity linker (#2) + per-fly solve; female rides on silhouette.
  Validate on courtship (both flies reproject cleanly).
- **Phase 4 — Active-parts:** config (#5) for headless + amputation; verify no phantom residuals.
- **Phase 5 — Unified JAX detector:** dense head + mask-gating (#1); feed dense observations;
  optional SAM3 ViTDet trunk port + mask distillation.
- **Phase 6 — Dense-silhouette polish (#7 2-D factor, approach B) + finer visual mesh** for shape
  fidelity.
- **Phase 7 — Outputs/QC (#8) + deploy** across all scenarios + anatomies (V2/V2.1 configs).

## 12. Risks / assumptions

- **mjx loads each anatomy** for differentiable FK — confirmed for V1 (nq=93) and v2.1_muscles.
- **SAM3 identity is reliable enough** per camera to link across views; the "largest extra"
  heuristic worked 6/7 in spot checks — the linker must handle the occasional miss (epipolar gating).
- **Silhouette-blind DOF** (wing-roll, global orientation/symmetry) are resolved by keypoints +
  temporal continuity + priors, not the silhouette alone. Frames lacking both a reliable orientation
  cue and a good neighbor remain ambiguous — temporal coupling is the mitigation.
- **Collision-mesh silhouette** caps IoU (~0.76); a finer visual mesh (Phase 6) raises the ceiling.
- Dense-silhouette factor cost (render-in-the-LM-loop) must stay tractable — sparse Chamfer over
  sampled boundary points preferred over per-pixel.
- **Affine/telecentric cameras** (3×4 DLT, 3rd row `[0,0,0,1]`): depth comes only from the angular
  diversity of the 7 views; near-degenerate viewing configs would weaken triangulation (not a
  concern for this rig). The projection/triangulation/silhouette code must treat the camera as
  affine (no perspective divide) consistently.
- **Bundle adjustment overfit/gauge:** refining cameras can absorb detection noise or drift the
  affine gauge. Mitigations: robust (Huber) loss, soft prior to factory calibration, fixed gauge,
  default to affine-pose-only, and a reproject-before/after guard that reverts on regression.
