# Silhouette-Bootstrapped Detector Finetuning — Design Spec

**Date:** 2026-07-03
**Status:** Approved (design)
**Sub-project of:** courtship 3D kinematics recovery (V2VNet-free mesh/silhouette pipeline)

## Goal

Raise the 2D keypoint accuracy and **multiview consistency** of the ViTPose detector on
courtship (especially female) flies, so that DLT triangulation → STAC → silhouette-containment
IK produce good 3D kinematics on courtship recordings.

## Background / diagnosis (established this session)

- Detector: `jax_vitpose_runs/v3_kp_maskaware` — 4-channel (RGB + SAM mask), 448 crop, 50
  keypoints (`cfg.model.KP_NAMES`). Heatmap size 224.
- Quality: full-val MPJPE **6.67px**, but **female 26px**; on the target Session0 courtship bout
  the 2D keypoints are **~40–55px multiview-inconsistent** (leave-one-out reprojection, `qc.loo_reproj`)
  even at high confidence (conf≥0.9 floor ≈ 39px).
- **Ruled out** as causes: camera-model mismatch (affine LOO 54.7px == perspective 54.8px, ratio
  1.00), crop-scale mismatch (courtship fly SAM-bbox diag ~299/323px ≈ training bbox ~331px; both
  train and infer use the same `crop_origin` fixed-448-native window), and the model→mm bridge.
- **Root cause:** detector domain precision. Training data `red_data_unified_V3` is dominated by
  ONE general male recording (~3563 imgs) with only ~1000 courtship imgs total (15–46 annotated
  frames per courtship recording, both sexes); female-courtship is a tiny fraction.
- Reliable assets: multiview calibration (7 telecentric cams), SAM3 per-fly masks, the articulated
  MuJoCo mesh + validated silhouette-containment IK, DLT triangulation, and the now-working
  courtship inference pipeline (`run_courtship_bout.py`) that writes per-frame `outputs.h5`
  (`kp3d_mm` = FK'd 50 keypoint-sites in world mm), `qc.json`, `kp2d.npz`, and per-bout SAM masks.

**Constraint:** courtship fly identity (male/female) is unreliable (sex_swaps); treat sex-agnostically.

## Core idea

The (fixed) courtship pipeline becomes a **label factory**. The posed mesh is a single rigid 3D
object per frame, so reprojecting its keypoint-sites to all 7 cameras yields 2D labels that are
**multiview-consistent by construction** (LOO ≈ 0) and **anatomically structured** (correct L/R and
skeleton topology from the model) — exactly the two properties the detector's diffuse, L/R-confused
output lacks. We gate hard on fit quality, finetune the detector on the gated pseudo-labels mixed
with the real annotations, validate on held-out data, and optionally iterate.

Circularity is manageable because: the mask-anchored model→mm bridge already makes global placement
mask-driven (not keypoint-driven); per-keypoint consensus catches where the mesh is locally wrong;
and a held-out courtship recording (never pseudo-labeled) plus mixing real annotations detect and
prevent drift/forgetting.

## Architecture / data flow

```
Run pipeline on TRAIN-split courtship recs + Session0   (val-split courtship recs held out)
  → per-frame outputs.h5 (kp3d_mm: FK'd 50 sites, world-mm) + qc(per-frame) + kp2d.npz + SAM masks
      │
      ▼  Pseudo-label extraction
  reproject kp3d_mm[t] → 2D per camera  (multiview-consistent by construction)
      │
      ▼  Quality gate
  Gate A (per-frame fit): soft-IoU ≥ τ_iou ∧ containment ≤ τ_cont ∧ marker-reproj ≤ τ_reproj ∧ bridge≠None
  Gate B (per-kp consensus): keep kp k in cam c iff ‖mesh2d[k]−det2d[k]‖ ≤ N ∧ det_conf[k] ≥ τ_conf;
                             else vis=0 (no loss)
      │
      ▼  Dataset assembly
  gated (crop4, keypoints_heatmap, visibility) in V3 COCO format  ⊕  red_data_unified_V3 real annotations
      │
      ▼  Finetune  (continue from v3 checkpoint, mixed dataset, modest LR, early-stop on held-out)
  → new checkpoint v4_kp_silbootstrap  (v3 never overwritten — reversible)
      │
      ▼  Validate
  (1) held-out courtship LOO-reproj ↓   (2) female/full val MPJPE (no regress)   (3) de-risk fit IoU ↑ / reproj ↓
      │
      └── optional 1–2 self-training rounds; held-out LOO is the drift alarm
```

## Components (files / units)

Each unit has one responsibility and a clear interface:

1. **Per-frame QC extension** — the QC path currently emits median metrics
   (`qc.qc_report` / `courtship_qc`). Extend it (or add a sibling) to also emit **per-frame**
   soft-IoU, containment residual, and marker reproj arrays (length T), which Gate A consumes.
   Consumes: `rt`, per-frame mesh/kp/masks. Produces: per-frame metric arrays (npz or in qc.json).

2. **`courtship_pseudolabel.py`** — pseudo-label extraction + gating for one bout/fly.
   - Input: bout dir (`outputs.h5` → `kp3d_mm`; `kp2d.npz` → detector `kp2d`,`conf`; per-frame QC;
     `masks_dict`), `rt`/calibration, gate thresholds.
   - Reproject `kp3d_mm[t]` → per-cam 2D (`rt.reproject_point`).
   - Apply Gate A (drop frames) + Gate B (per-kp visibility).
   - Output: per-(frame,cam) records `{crop_origin window, rgb+mask 4ch source ref, keypoints 2D (full-px),
     visibility (50,)}` for kept samples.

3. **`build_pseudolabel_dataset.py`** — assemble gated records into the V3 COCO-style dataset the
   trainer reads (mirrors `data/v3.py`: `crop_origin` 448 crop, `transform_keypoints` → heatmap
   coords, visibility flags), and **merge** with the existing `red_data_unified_V3` annotations
   (configurable pseudo:real mix). Produces a new dataset root (annotations + image/mask refs).

4. **Finetune driver** — reuse the existing ViTPose training loop; load `v3` weights, train on the
   mixed dataset, modest LR, early-stop on the held-out metric, write `v4` checkpoint (Orbax,
   model-state). Never overwrites `v3`.

5. **Validation script** — (a) LOO-reproj on a held-out courtship recording (reuse `qc.loo_reproj`
   logic on that recording's detector kp2d); (b) MPJPE on `red_data_unified_V3/val`
   (reuse `eval.mpjpe`); (c) downstream: re-run one de-risk bout with `v4` and compare `qc.json`
   to the recorded v3 baseline.

## Gating details

- **Gate A (per-frame):** thresholds τ_iou, τ_cont, τ_reproj are config-driven; a frame passing all
  three (and having a non-None bridge, i.e. ≥2 mask-triangulable views) is kept, else all its
  cameras are dropped.
- **Gate B (per-keypoint consensus):** keep keypoint k in camera c iff
  `‖mesh2d[k] − det2d[k]‖ ≤ N` px AND detector `conf[k] ≥ τ_conf`; otherwise set that keypoint's
  visibility to 0 so it contributes no loss. This propagates the consistency the detector already
  had in *some* view into the views where it failed, and stays silent where mesh/detector disagree.
- **Known limitation (accepted for v1):** a keypoint the mesh places correctly but the detector
  misses in *all* views (fully occluded) is dropped by Gate B. Acceptable — other frames where it
  is visible still teach it. Relaxing Gate B (trust the mesh on high-fit frames without consensus)
  is a tunable lever for a later round.

## Finetune

- Continue-train from `v3_kp_maskaware/final` (do not restart).
- Mixed dataset: gated pseudo-labels ⊕ real `red_data_unified_V3` annotations (mix ratio config).
- Per-keypoint visibility gates the heatmap loss (vis=0 → no loss for that keypoint).
- Modest finetune LR; early-stop / checkpoint-select on the held-out courtship LOO metric.
- New checkpoint only; `v3` preserved.

## Data scope & held-out split (resolves scope↔validation)

Pseudo-label the **train-split** courtship recordings (the courtship recs whose `split_of` is
`train` in `red_data_unified_V3/build_report.json`) **+ Session0** (transductive; Session0 has no
ground truth). Do **not** pseudo-label the **val-split** courtship recordings — they stay held-out
for validation. This keeps "all courtship recordings + Session0" (the chosen scope) while preserving
a clean held-out set. Crucially, the val-split courtship recordings (e.g. `2026_05_27_11_56_05`
female-courtship, `2026_04_07_11_33_33`) carry **ground-truth annotations**, so held-out evaluation
uses real MPJPE, not only self-consistency.

## Validation & success criteria

Adoption gate — `v4` replaces `v3` in the pipeline only if ALL hold:
1. **Held-out courtship MPJPE** on the val-split courtship recordings improves vs. the v3 baseline
   (esp. female, from ~26px), AND their **LOO-reproj** (multiview self-consistency) drops toward the
   general-fly range (~10–15px from ~40–55px).
2. **No regression** on the full `red_data_unified_V3/val`: full-val MPJPE stays ~6–7px.
3. **Downstream**: a de-risk bout re-run with `v4` shows higher silhouette soft-IoU and lower
   per-camera / LOO reproj than the recorded v3 baseline (fly0 soft-IoU 0.102, reproj 55px, LOO 46px).

## Iteration (optional, guarded)

If round 1 improves the held-out LOO, refit the pipeline with `v4` → regenerate gated pseudo-labels
→ finetune again. Stop when held-out LOO stops improving or after 2 rounds. The held-out set is the
drift alarm; the real-annotation mix is the forgetting guard.

## Compute

Pseudo-label generation *is* the deferred full-session pipeline run (Session0 30 bouts × 2 flies +
other courtship recordings), executed via the existing resumable SLURM array
(`slurm_courtship_array.py`). No new heavy infra.

## Out of scope (v1)

- Retraining from scratch or changing the ViTPose architecture / heatmap resolution.
- Manual annotation tooling (may be added later if pseudo-labels prove insufficient).
- Changing the SAM mask pipeline, calibration, or the silhouette solver itself.
- Fixing per-camera calibration differences (cams 2862/2631/2857 worse) — noted, not addressed here.

## Global constraints

- Frozen: `stac-mjx/stac_mjx/stac_core_jaxls.py` (byte-identical).
- The existing `v3` checkpoint is never overwritten; all new artifacts are additive/reversible.
- New pipeline code under `third_party/jarvis_jax/jarvis_jax/cse/`; tests under
  `third_party/jarvis_jax/tests/`; drivers + configs at repo root; extend the existing Hydra
  `configs/` (do not fork).
- Heavy work on GPU compute nodes only (sbatch to ckpt-g2 / gpu-l40s); do not OOM the login node.
- Do not commit checkpoints / data / label-npz / masks / outputs / videos. Commit by explicit path.
- Sex-agnostic (fly identity unreliable).
