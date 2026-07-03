# V2VNet-Free Mesh/Silhouette Courtship Inference Pipeline — Design

**Date:** 2026-07-03
**Status:** Design (approved for planning)
**Depends on:** the watertight visual mesh (`fly_v1_visual_canonical_wings.npz`), the silhouette-containment IK factor (`silhouette_joint_ik.py` + bridge fix, this session), the retrained mask-aware ViTPose (`jax_vitpose_runs/v3_kp_maskaware/final`), and the validated `unpackbits` courtship mask adapter.

## 1. Motivation & goal

Produce 3-D fly kinematics for the full Session0 courtship recording
(`Video_recordings/courtship/Session0/2025_10_20_13_20_04`, 7 telecentric cameras,
30 bouts × 2 flies) using a **V2VNet-free** path: the mesh/articulated-model +
multi-view silhouettes replace the learned 3-D lift. Deliver per-bout 3-D
kinematics + a whole-session QC dashboard + overlay videos. **Compute speed is a
first-class constraint** (JAX-first, memory-efficient; parallelize across bouts).

Why V2VNet-free: the existing predict path is ViTPose(2-D) → reproject-to-volume →
**V2VNet**(learned 3-D). Our pipeline gets 3-D by **DLT triangulation + articulated
STAC fit + silhouette refinement**, so the articulated model + mask evidence
provide the priors V2VNet learned — more interpretable, and it removes the
ViTPose↔V2VNet train/inference mismatch risk entirely.

## 2. Success criterion (v1, throughput-first)

All **30 bouts × 2 flies** processed end-to-end and landing in per-bout outputs +
a whole-session QC dashboard + overlay videos. Per-bout correctness is spot-checked
via QC (silhouette IoU, reproj RMSE) and the representative overlays; the hard gate
is a clean full-session run. **De-risk before the full run:** the pipeline must
first run correctly on `bout_00001` (both flies) with the silhouette-refined fit
improving multi-view IoU over the keypoint-only fit.

## 3. Architecture — five staged, batched components

Each stage compiles once and batches across cameras/frames. Data flows per
(bout, fly). Affine/telecentric projection throughout (`uv = X @ M.T + t`).

### Stage A — ViTPose 2-D inference (`courtship_predict_2d.py`)
For each camera, crop the fly using the SAM mask/centroid (reuse the crop logic in
`jarvis_jax/predict/session_frameset.py` / `session_predict.py`), run the retrained
ViTPose (`v3_kp_maskaware/final`, 4-channel RGB+mask) → per-camera 2-D keypoint
heatmaps → peak coords + peak confidence, mapped back to full-frame pixels.
Batched across (camera, frame). Per fly (mask channel = that fly's SAM mask).

### Stage B — DLT triangulation (`courtship_triangulate.py`)
Per keypoint, triangulate across the cameras where it is confidently visible
(peak conf ≥ threshold, ≥2 views) using `triangulate_dlt_batched`
(`jarvis_jax/geometry/center3d.py`) → 3-D keypoint (mm) + a 3-D confidence
(function of view count × mean peak conf). <2 confident views → NaN (down-weighted
downstream). Vectorized over (frame, keypoint).

### Stage C — STAC articulated fit (`courtship_stac.py`)
Feed the triangulated 3-D keypoints into `stac_mjx.run_stac`:
- **fit marker offsets ONCE** globally (a shared precompute over a high-confidence
  frame sample) → offsets file;
- per bout, **bucketed `ik_only`** (the fast path in `scripts/run_stac.py`,
  `skip_fit_offsets`) → per-frame qpos + the model→mm bridge inputs.
Frozen `stac-mjx/stac_mjx/stac_core_jaxls.py` is untouched.

### Stage D — Silhouette-containment IK (reuse `SilhouetteJaxlsBatchSolver`)
Refine qpos with the silhouette factor (this session): eroded-mask coverage
Chamfer + SDF containment, with the **model→mm bridge fix**, **confidence weights
from the kp conf** (down-weight weak keypoints so the silhouette drives),
appendage-DOF restriction. Masks come from the validated **`courtship_bout_masks.py`**
adapter: `np.unpackbits(packed[fly,cam,frame])[:, :W]` → full-frame mask (matches
the calibration frame), then the existing eroded-boundary + `build_sdf_stack`
helpers. Vmapped over the bout's frames.

### Stage E — Outputs + QC + overlays
- **Outputs h5** via `jarvis_jax/cse/outputs.py` (`build_fly_outputs`): qpos,
  3-D keypoints (mm), mesh verts (mm, subset), joint angles.
- **QC** per bout via `qc.py`: silhouette IoU, reproj RMSE, kp-availability %,
  n_frames, sex_swap flag.
- **Overlay videos** per bout × camera via `reproj_video.py`, rasterizing a
  **one-time decimated watertight mesh (~5k faces)** — not the 875k-vertex mesh —
  the single biggest raster speedup.

## 4. Speed strategy (first-class)

- **Parallelize across bouts with a SLURM array (1 GPU/bout)** on **gpu-l40s**
  (non-preemptible; ~20 nodes × 4 L40S). 30 bouts schedule concurrently →
  wall-clock ≈ one bout's (compile + process) time, not the sum. Array **chunk
  size** (bouts-per-job) is tunable: 1/GPU by default (max parallelism); raise it
  if per-job compile overhead dominates (amortizes compile across a group).
- **Shared precompute (once, before the array):** STAC `fit_offsets` (global) →
  offsets file; the decimated mesh. All array jobs read these read-only.
- **Within a job:** compile-once + big batches (one jitted ViTPose step over
  cam×frame batches; vmapped triangulation; vmapped silhouette solve; camera
  `lax.scan`, chunked Chamfer, cropped SDF — no `(C,N,M)` materialization).
- **Decimated mesh** for all rasterization (overlays + filled-IoU); full mesh only
  if explicitly requested for output verts.
- **Final aggregation job:** merge per-bout QC JSON → session dashboard (cheap,
  CPU).

## 5. Occlusion / confidence handling

Triangulation has no learned prior, so weak/occluded keypoints are handled by:
(a) **confidence weights** (kp conf → marker weights + silhouette conf), (b) STAC
**model priors** (joint limits, bone lengths), (c) **silhouette evidence**,
(d) the solver's **temporal smoothness** term. Deep temporal bridge/pose
propagation into fully-occluded frames is **out of scope for v1** (the next lever
if QC shows keypoint-dropout frames failing). Identity: process **both** flies per
bout; `sex_swaps` recorded in metadata but kinematics is sex-agnostic.

## 6. Component / file structure

**New** (`third_party/jarvis_jax/jarvis_jax/cse/`):
`courtship_predict_2d.py`, `courtship_triangulate.py`, `courtship_stac.py`,
`courtship_bout_masks.py`, `courtship_qc.py` (dashboard aggregation), and a
decimated-mesh builder helper (one-time, e.g. `build_decimated_mesh.py` or a
function in an existing mesh module).

**New drivers** (`scripts/`): `run_courtship_bout.py` (single bout, both flies —
the de-risk + the array-job body) and `slurm_courtship_array.py` (submits the
offset-fit precompute + the SLURM array over bouts + the aggregation job) on
gpu-l40s.

**Reused unchanged:** ViTPose model/load + crop logic, `triangulate_dlt_batched`,
`stac_mjx.run_stac` (bucketed `ik_only`), `SilhouetteJaxlsBatchSolver` (+ bridge
fix), `outputs.py`, `qc.py`, `reproj_video.py`, `build_solver_inputs`,
`build_silhouette_targets`/`build_sdf_stack` helpers.

## 7. Data flow

```
Session0 video + sam3 masks (packed) + calibration
  ├─ [once] STAC fit_offsets (global sample) -> offsets.h5 ; decimated_mesh.npz
  └─ [SLURM array, 1 GPU/bout] for bout b:
       for fly f in {0,1}:
         A ViTPose 2D (SAM-crop, per cam)      -> kp2d[cam,frame], conf
         B DLT triangulate                     -> kp3d_mm[frame], conf3d
         C STAC ik_only (shared offsets)       -> qpos_init, bridge
         D silhouette-containment IK           -> qpos_refined
         E outputs.h5 + qc.json + overlay mp4s (decimated mesh)
  └─ [aggregate] merge qc.json -> session dashboard (plots + json)
```

## 8. Testing (CPU: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/<file>`)

- `test_courtship_bout_masks.py` — `unpackbits` full-frame alignment (nonzero extent matches expected bytes×8), per-fly/per-cam selection, empty/absent handling.
- `test_courtship_triangulate.py` — known multi-view geometry triangulates to the right 3-D point; <2 views → NaN; confidence aggregation.
- `test_courtship_predict_2d.py` — crop→heatmap-peak→full-frame-px mapping on a synthetic heatmap; conf = peak value.
- `test_courtship_qc.py` — per-bout metric aggregation → dashboard JSON schema.
- Frozen-file guard (`stac_core_jaxls.py` byte-identical) stays green.
- **De-risk gate (GPU, coordinator-run):** full pipeline on `bout_00001` both flies; assert outputs written + silhouette IoU ≥ keypoint-only IoU; eyeball an overlay.

## 9. Non-goals (v1)

V2VNet / learned 3-D lift; temporal bridge/pose propagation; detector or V2VNet
retraining; multi-session generality (v1 targets this Session0, parameterized by
recording path but only run/validated here); real-time / streaming.

## 10. Constraints

- Frozen `stac-mjx/stac_mjx/stac_core_jaxls.py` (byte-identical test).
- New code under `cse/` + `scripts/`, tests under `tests/`.
- Affine/telecentric projection; masks via `unpackbits` full-frame.
- Env: `micromamba activate 3d_tracking`, `unset LD_LIBRARY_PATH`; heavy runs sbatch on **gpu-l40s** (non-preemptible); unit tests CPU-only.
- Do not commit checkpoints/data/masks/outputs/videos; commit by explicit path.
- JAX-first + memory-efficient (no large materialized pairwise tensors).
