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

## 3. Architecture — Stage 0 (masks) + five batched JAX stages

Each JAX stage compiles once and batches across cameras/frames. Data flows per
(bout, fly). Affine/telecentric projection throughout (`uv = X @ M.T + t`).
Stage 0 is a PyTorch prerequisite (different CUDA env), run only when masks are
absent.

### Stage 0 — SAM3 mask generation (new recordings; PyTorch) — reuse `scripts/sam3_masks.py`
For a recording without masks, generate per-bout `sam3_masks.npz` (bit-packed
full-frame per fly×cam×frame) via the existing `sam3_masks.py` / `sam3_driver.py`
(SAM3.1, `sam3_text=insect`, `num_animals=2`). **Skip-if-present is built in**
(`reuse_masks=true`) → Session0's existing masks are reused, new recordings
generate. Bout-restricted via the bouts CSV. **Separate CUDA env** (SAM3 is
PyTorch): `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` +
`LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib`,
`sam3_compile=false` — so Stage 0 is its OWN job(s), never in a JAX process.
For Session0 this stage is a no-op (masks exist).

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
- **Stage 0 (SAM3) parallelism** (new recordings only): PRIMARY = a SLURM array
  (1 GPU/bout, `bout_ids=$SLURM_ARRAY_TASK_ID`) — same shape as the JAX array, so
  all bouts generate masks concurrently across the cluster (the built-in
  `sam3.gpus=[...]` fan-out only parallelizes within one node, so it's the
  single-node fallback). `lowmem=true` bounds GPU memory on long bouts;
  `reuse_masks=true` skips existing. The two arrays chain by **SLURM dependency**
  (`--dependency=afterok:<sam3_array>`): masks array → precompute → JAX array →
  aggregation. Stage 0 self-skips entirely when all masks already exist (Session0).
- **Shared precompute (once, before the array):** STAC `fit_offsets` (global) →
  offsets file; the decimated mesh. All array jobs read these read-only.
- **Within a job:** compile-once + big batches (one jitted ViTPose step over
  cam×frame batches; vmapped triangulation; vmapped silhouette solve; camera
  `lax.scan`, chunked Chamfer, cropped SDF — no `(C,N,M)` materialization).
- **Decimated mesh** for all rasterization (overlays + filled-IoU); full mesh only
  if explicitly requested for output verts.
- **Final aggregation job:** merge per-bout QC JSON → session dashboard (cheap,
  CPU).

### Resumability & preemption safety (ckpt partitions)

The pipeline will run mostly on **preemptible ckpt partitions** (ckpt-g2), so
every job must be idempotent and resume correctly after a preempt+requeue.
Mechanisms:

- **`--requeue` + fixed per-bout output dir.** Each array task writes to a
  deterministic `<run_root>/bouts/bout_<idx:05d>/fly<f>/` baked from the bout index
  (NOT from `$SLURM_JOB_ID`/date), so a requeued task lands in the same dir and
  continues. (Layout in §7a.)
- **Stage-artifact checkpoints within a bout.** Each stage writes its output and,
  on (re)start, the bout driver **skips stages whose artifact already exists** and
  resumes at the first missing one: A→`kp2d.npz`, B→`kp3d.npz`, C→`stac_ik.h5`
  (STAC already checkpoints), D→`qpos_refined.npz`, E→`outputs.h5` + `qc.json` +
  per-camera `*_reproj.mp4`. So a preempted bout redoes at most the *current*
  stage, not the whole bout.
- **Atomic writes.** Every artifact is written to `*.tmp` then `os.replace`d into
  place, so a job killed mid-write never leaves a half-file that looks complete
  (orbax already does tmp+rename; our npz/json/mp4 follow the same pattern). A
  per-bout `DONE` marker (written last) lets the array skip fully-finished bouts on
  resubmit.
- **Idempotent stages + reuse-if-present everywhere.** Re-running a stage
  overwrites its own artifact cleanly; Stage 0 (`reuse_masks`) and STAC
  (`ik_only` path) already skip completed work. Overlay videos (Stage E) skip
  per-camera files that already exist.
- **Partition-agnostic.** The orchestrator takes a `--slurm` group: `ckpt_g2`
  (preemptible, `requeue=true` — the default given ckpt usage) or `gpu_l40s`
  (non-preemptible). Resumability makes ckpt safe; a resubmit of the whole array
  is a no-op for done bouts and a resume for interrupted ones.

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

**Config (extend existing repo-root `configs/`, §6a):** add `pipeline.yaml`
+ `recording/`, `detector/`, `silhouette/`, `outputs/`, `slurm/` groups; reuse the
existing `paths/`, `stac/`, `anatomy/`; generalize `paths/hyak.yaml`
(`user → ${oc.env:USER,eabe}`). Also generalize the submodule's
`third_party/jarvis_jax/configs/paths/hyak.yaml` the same way.

**New drivers** (`scripts/`, Hydra apps rooted at repo `configs/`): `run_bout.py` (single bout, both flies —
the de-risk + the array-job body; writes per-stage artifacts atomically and
skips stages whose artifact exists, so it resumes correctly after preemption) and `slurm_bout_array.py` (submits, in
order: an optional **Stage-0 SAM3 job/array** for recordings lacking masks; the
offset-fit + decimated-mesh precompute; the SLURM array over bouts; the QC
aggregation job) on gpu-l40s. The orchestrator checks for existing masks and only
schedules Stage 0 when absent (Session0 → skipped). Stage 0 runs in the PyTorch
CUDA env; Stages A–E in the JAX env.

**Reused unchanged:** `scripts/sam3_masks.py` + `sam3_driver.py` (Stage 0),
ViTPose model/load + crop logic, `triangulate_dlt_batched`, `stac_mjx.run_stac`
(bucketed `ik_only`), `SilhouetteJaxlsBatchSolver` (+ bridge fix), `outputs.py`,
`qc.py`, `reproj_video.py`, `build_solver_inputs`,
`build_silhouette_targets`/`build_sdf_stack` helpers.

## 6a. Configuration (Hydra) — EXTEND the existing repo-root `configs/`

The repo already has a Hydra tree at `3d_tracking_dataset/configs/` (`config.yaml`
with `anatomy`/`dataset`/`paths`/`stac` groups; multiple `paths/` machine profiles).
The pipeline **extends this existing tree** (it does NOT add configs to the
`jarvis_jax` submodule — the submodule stays reusable). **No absolute paths in
pipeline code**; paths are **generalized via env interpolation**.

**Add** these groups + a top-level pipeline config:
```
configs/
├── pipeline.yaml     # NEW top-level defaults list: paths, recording, detector, silhouette, outputs, slurm, + reuse anatomy/stac
├── recording/session0.yaml     # NEW: session_dir, bouts_csv, calibration, cameras, num_animals
├── detector/vitpose_v3.yaml    # NEW: ViTPose ckpt path (from config), in_ch=4, heatmap params
├── silhouette/default.yaml     # NEW: mesh npz, silhouette+containment weights, erode_px, appendage DOF set, conf source
├── outputs/default.yaml        # NEW: run-root pattern, overlay params, decimated-mesh, qc
├── slurm/{ckpt_g2,gpu_l40s}.yaml   # NEW: partition groups
└── (reuse existing) paths/, stac/, anatomy/
```

**Generalize the existing `paths/hyak.yaml`**: it currently hardcodes `user: eabe`
then interpolates the rest. Change the hardcoded root(s) to **env-overridable**:
```
user: ${oc.env:USER,eabe}
# existing interpolated paths (base_dir/data_dir/body_model_dir/...) then resolve per-user
```
Add pipeline path keys as needed (`out_root: ${paths.data_dir}/../courtship` or an
explicit `johnson_root`). The run root (§7a) = `${outputs.out}` defaulting to
`${paths.out_root}/${recording.session}_bouts_${now:%m%d%Y}`. Overriding a run is
just Hydra: `recording=sessionX paths=workstation outputs.out=/somewhere`.

**Reused jarvis_jax stages** (STAC via `run_stac`, SAM3 via `sam3_masks`) currently
read the submodule's own `configs/paths/hyak.yaml` (hardcoded). The pipeline passes
its generalized paths to those invocations as **Hydra overrides**
(`paths.data_root=…`, `sam3.session_dir=…`, etc.), so the whole pipeline is
path-generalizable without editing the submodule; additionally,
`third_party/jarvis_jax/configs/paths/hyak.yaml` is updated to the same
env-interpolation pattern for consistency (a targeted generalization of the
existing hardcoded file).

The drivers (`run_bout.py`, `slurm_bout_array.py`) are Hydra apps
with `config_path` = the repo-root `configs/`.

## 7. Data flow

```
Session video + calibration + bouts CSV
  ├─ [Stage 0, PyTorch, only if masks absent] SAM3 -> per-bout sam3_masks.npz
  │     (reuse_masks skips existing; SLURM array 1 GPU/bout or multi-GPU fan-out)
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

## 7a. Output directory layout

A single **date-stamped run root** per pipeline run (matches the existing
`courtship/Session<N>_bouts_<MMDDYYYY>` convention), self-contained and separate
from the raw video dir:

```
<data>/courtship/Session0_bouts_<MMDDYYYY>/     # run root (default; --out overridable)
├── run_manifest.json          # provenance: recording, ViTPose ckpt, git sha, params, per-bout/stage status
├── offsets.h5                 # shared STAC offset-fit (Stage C precompute, once)
├── decimated_mesh.npz         # shared decimated mesh (overlay raster, once)
├── sam3_masks/bout_<NNNNN>/sam3_masks.npz    # Stage 0 output (only if generated; else reuse recording's)
├── bouts/bout_<NNNNN>/
│   ├── DONE                    # per-bout completion marker (written last, atomically)
│   ├── fly0/
│   │   ├── kp2d.npz  kp3d.npz  stac_ik.h5  qpos_refined.npz   # resumable stage A–D artifacts
│   │   ├── outputs.h5          # Stage E: qpos, kp3d_mm, mesh_mm (subset), joint angles
│   │   └── qc.json             # Stage E: per-fly QC
│   ├── fly1/ …
│   └── overlays/Cam*_reproj.mp4   # per-camera overlay videos (skip-if-exists)
├── qc/session_qc.json + dashboard/*.png       # aggregated session QC + plots
└── logs/slurm-*.out
```

Rationale: the date-stamped root keeps each run reproducible and self-contained;
`bouts/bout_<NNNNN>/fly<f>/` IS the resumability structure (stage artifacts +
`DONE`); shared precompute (`offsets.h5`, `decimated_mesh.npz`) lives at the root,
read-only for every array task; `sam3_masks/` is populated only when Stage 0 runs
(Session0 points at the recording's existing masks, no copy); `qc/` + `logs/`
isolate session-level artifacts. The run root is a driver parameter (`--out`),
defaulting to `courtship/<session>_bouts_<date>`.

## 8. Testing (CPU: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/<file>`)

- `test_courtship_bout_masks.py` — `unpackbits` full-frame alignment (nonzero extent matches expected bytes×8), per-fly/per-cam selection, empty/absent handling.
- `test_courtship_triangulate.py` — known multi-view geometry triangulates to the right 3-D point; <2 views → NaN; confidence aggregation.
- `test_courtship_predict_2d.py` — crop→heatmap-peak→full-frame-px mapping on a synthetic heatmap; conf = peak value.
- `test_courtship_qc.py` — per-bout metric aggregation → dashboard JSON schema.
- `test_courtship_resume.py` — the bout driver's stage-skip logic: given a bout
  dir with some stage artifacts present, it resumes at the first missing stage and
  does not recompute completed ones; a half-written (`.tmp`, no `DONE`) artifact is
  treated as incomplete and redone; atomic write leaves no partial file on abort.
- `test_courtship_config.py` — the repo-root Hydra config composes; paths resolve
  from env overrides + defaults (no hardcoded absolute in the pipeline code path);
  the run-root pattern expands as expected.
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
- Env: `micromamba activate 3d_tracking`; JAX stages `unset LD_LIBRARY_PATH`, Stage 0 sets the cu13 `LD_LIBRARY_PATH` + `LD_PRELOAD`; unit tests CPU-only.
- Heavy runs sbatch, partition-selectable: **ckpt-g2** (preemptible — the default; safe via the resumability mechanisms in §4) or **gpu-l40s** (non-preemptible). All array tasks submit with `--requeue` and resume from stage artifacts.
- Do not commit checkpoints/data/masks/outputs/videos; commit by explicit path.
- JAX-first + memory-efficient (no large materialized pairwise tensors).
- **Config-driven, no hardcoded paths:** all params via the repo-root Hydra tree
  (§6a); every path is config/env-interpolated with sensible defaults so the
  pipeline generalizes across users/clusters/recordings; the reused jarvis_jax
  `paths/hyak.yaml` is generalized to the same env-interpolation pattern.
