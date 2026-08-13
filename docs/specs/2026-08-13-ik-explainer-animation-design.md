# IK Explainer Animation — Design

## Context

We want a short talk video that explains how inverse kinematics turns
multi-camera fly video into an articulated body-model fit. The pipeline already
produces every ingredient, but nothing renders the *stages* — existing viz
(`viz/views/fit_check.py`, `viz/views/overlay.py`,
`scripts/viz/compare_stac_fits.py`) shows finished fits, never the solve.

Source clip: `/data2/users/eabe/datasets/3d_tracking/clips/Session6/2025_10_12_15_06_46`.

The narrative arc, in four acts: seven camera views → 2D keypoints fade in →
views converge into a 3D keypoint cloud → the MuJoCo body model appears,
misaligned → the model scales, orients, and solves onto the keypoints → both
move together.

The key design insight is that **this arc is not a staged dramatisation — it is
the actual STAC algorithm.** `stac-mjx/stac_mjx/stac.py:296` (`fit_offsets`)
runs `root_optimization` (6-DOF rigid align of the free-joint root to the trunk
keypoints), then `pose_optimization` (per-joint IK), alternating with
`offset_optimization`. The "misaligned" state the video opens Act 3 with is
literally the optimiser's frame zero. So the video is built by *snapshotting the
solver*, not by hand-authoring motion.

**Every stage is re-run from raw video for this clip.** The shipped
`data3D_*.csv` is not the animation's 3D source; it is demoted to a crop/prompt
seed and an independent comparison baseline. This makes the chain internally
consistent — Act 2's triangulation rays converge on the very 3D points Act 3
then fits — and, as a side effect, removes a unit hazard (below).

## Deliverable

`<CLIP>/ik_explainer/ik_explainer.mp4` — **1920×1080, 30 fps, ~75 s, H.264,
silent**, with burned-in stage labels. The generating code is committed under
`scripts/viz/ik_explainer/`; the video and every artifact are never committed,
so the generator is the durable artifact.

### On-disk layout

All results live beside the clip they were computed from, under a single
`ik_explainer/` directory, so the clip dir stays self-describing and nothing we
generate is mixed in with the raw inputs. `<CLIP>` =
`/data2/users/eabe/datasets/3d_tracking/clips/Session6/2025_10_12_15_06_46`.

```
<CLIP>/
  calibration/  Cam*.mp4  enhanced/  preview/      <- existing raw inputs, untouched
  data3D_frames1161383-1162304.csv                 <- existing; now the BASELINE only
  ik_explainer/                                    <- everything we produce
    README.md              what this is, how it was made, how to regenerate
    manifest.json          provenance: repo commit, ckpt paths, configs, timings
    predictions/           the re-run pipeline, in stage order
      01_sam3_masks.npz    compressed (pipeline convention: 50-500 MB)
      02_kp2d.npz          (7, 921, 50, 2) + conf
      03_kp3d.npz          (921, 50, 3) mm + conf3d
      04_kp3d_filt.npz     post temporal filter
      05_stac_ik.h5        STAC output
      06_stages.npz        per-stage qpos snapshots + residuals
    qc/                    verification figures + numbers (the CLAUDE.md gate)
      01_masks.png  02_kp2d_overlay.png  03_kp3d_vs_shipped.png
      04_filter_effect.png  05_ik_stages.png  qc.json
    frames/                per-act PNG sequences, re-renderable independently
      act1_views/  act2_triangulate/  act3_align/  act4_solve/
    ik_explainer.mp4       the deliverable
```

Numeric prefixes on `predictions/` make the dependency order readable at a
glance: each stage consumes the one above it. `qc/` mirrors that order so a
failed check points at the stage that produced it.

This overrides CLAUDE.md's default of putting figures under `figures/<topic>/`
— on explicit instruction to keep all results with the clip. The earlier
calibration-verification figures remain at
`figures/2026-08-13-ik-anim-dlt-check/`.

## Scope

**In scope:**
- A committed generator under `scripts/viz/ik_explainer/` producing one mp4.
- A full re-prediction of this clip: SAM3 masks → 4-channel ViTPose 2D →
  DLT triangulation → temporal filtering → staged STAC IK.
- Instrumenting the STAC solve to snapshot intermediate stages (Acts 3–4).
- A thin adapter for this clip's nonstandard format (DLT csv, no bout summary).

**Out of scope:**
- Voiceover / narration. Burned-in stage labels only.
- Changing any pipeline stage. The adapter reads this clip; it does not alter
  how production recordings are processed.
- Generalising the generator to arbitrary recordings. It targets this clip;
  paths are arguments, but the format assumptions are this clip's.
- Silhouette polish (`run_silhouette_polish.py`). The video ends at STAC IK.

## Verified facts about the source clip

Established by inspection before design; the generator depends on all of these.

| Property | Value |
|---|---|
| Cameras | 7 (`Cam2012630/631/853/855/857/861/862`) |
| Video | 1936×448, **800 fps**, 921 frames (+ contrast-`enhanced/` copies) |
| Calibration | 7× **11-coefficient DLT** (`calibration/Cam*_dlt.csv`) |
| Shipped 3D (baseline only) | `data3D_frames1161383-1162304.csv`, 50 kp × 922 frames + confidence |
| Clip duration | 921 / 800 = **1.15 s of fly time** |

Frame height is **exactly 448**, equal to the detector's crop size — a
full-height crop, no vertical clamping.

**Frame-count mismatch:** `ffprobe` reports **921** video frames while the
shipped CSV has **922** rows. Since the re-predicted chain is driven by video,
`N = 921` throughout; the CSV is truncated to `min(n_video, n_csv)` only where
it is used as a seed or baseline, asserting the two differ by at most 1.

### Geometry: the DLT chain closes exactly

Each `Cam*_dlt.csv` gives 11 coefficients with `L9 = L10 = L11 = 0`, i.e. an
**affine (telecentric) camera**: `uv = P[:2,:3] @ X + P[:2,3]`, no perspective
divide. The 3×4 `P` transposes to the `(4,3)` matrix the repo's
`triangulate_dlt_batched` and `viz.core.reproject.project` expect.

Verified numerically over 200 frames × 50 keypoints: projecting known 3D (mm)
into all 7 views and triangulating back recovers the input to a **max error of
1.1e-5 mm**, 100% finite, with `proj[..., 2]` identically 1 (confirming the
divide is a no-op). Noise sensitivity is excellent — **1 px of 2D error costs
9.3 µm in 3D** across 7 views (0.5 px → 4.7 µm, 2 px → 18.5 µm).

**Consequence: triangulating our own 2D with these DLTs yields 3D natively in
mm.** No unit conversion enters the animation's chain.

### The unit trap (now confined to the baseline)

The shipped CSV is in units of **0.1 mm** while the DLTs expect mm. Projecting
it raw puts `u ∈ [7927, 10705]` on a 1936-px frame — **0% of keypoints in
frame**. Scaling XYZ by 0.1 puts **99.958% in frame** (134 of 322,700
projections outside), with a body length (`Antenna_Base`→`Abd_tip`) of
**2.39 mm**, correct for *Drosophila*. The 134 excursions are all on
`Cam2012630` (the vertical camera), all distal right-leg tips, leaving the
bottom edge by at most 15.7 px — real field-of-view clipping on a 448-px-tall
strip, not a calibration fault. Two independent confirmations:

- `third_party/jarvis_jax/tests/test_identity_link.py:17` holds `P_REAL`, a real
  rig DLT, at linear-term scale ~8.1 where this clip's is ~81.2 — a clean 10×.
- Factoring the DLTs gives **80.7 px/mm** magnification on every camera;
  80.7 × 2.39 mm = 193 px, matching the observed fly size.

CLAUDE.md records marker offsets silently absorbing a 38× body-scale error, with
residual and NaN checks blind to it. Re-predicting removes this hazard from the
production path entirely: the ×0.1 conversion now applies **only** where the
shipped CSV is read as a seed or baseline, and lives in one function in
`clip_io.py`. The marker residual is still displayed in mm on screen so any
residual scale error is visible rather than absorbed.

### Rig geometry (recovered, drives Act 2)

Factoring each DLT with `jarvis_jax.tracking.affine_camera.factor_affine`
recovers the optical axes:

| Camera | Azimuth | Elevation | Role |
|---|---|---|---|
| `Cam2012857` | −90.0° | −0.6° | side, one end of arc |
| `Cam2012855` | −90.0° | −30.5° | |
| `Cam2012853` | −90.4° | −60.5° | |
| `Cam2012630` | — | **−89.6°** | top / vertical |
| `Cam2012862` | +91.1° | −59.5° | |
| `Cam2012631` | +90.5° | −29.3° | |
| `Cam2012861` | +90.7° | +0.6° | side, far end of arc |

Pairwise angles come out at 30.0 / 60.0 / 90.0 / 120.2 / 150.1°, with
`857`↔`861` at 179.3°: **a 180° arc at 30° spacing**, all viewing perpendicular
to the arena's long X axis.

**Consequence for Act 2:** telecentric cameras have no finite centre of
projection — rays are *parallel* and the camera sits at infinity. Panel
*orientation* is real rig geometry; panel *distance* is an arbitrary staging
choice. Act 2 must draw parallel rays, not converging ones, and say so on screen.

### Other assumptions

- **Calibration is borrowed.** `calibration/notes.txt` reads only
  `2025_10_08_11_50_16` — a different session four days earlier. It reprojects
  cleanly (verified on real frames), so the rig evidently did not move, but this
  is assumed, not measured.
- **One fly, sex unrecorded.** A single fly in an open arena is the *easy* case.
  CLAUDE.md is explicit that the female on walls / in occlusion is where this
  pipeline fails. This video will look better than the pipeline's hard case and
  must not be cited as general QC.
- **`viz/core/reproject.py` cannot read this clip.** It calls
  `ReprojectionTool`, which requires `Cam*.yaml` and raises `FileNotFoundError`
  on this clip's `Cam*_dlt.csv`. Hence `clip_io.py`'s own DLT loader.

## Architecture

Two phases. A **prediction phase** re-runs the pipeline on this clip and writes
intermediate artifacts to disk; a **render phase** turns those artifacts into
acts. Every act renders an independent numbered PNG sequence, so any act can be
re-rendered without repaying a GPU pass.

```
scripts/viz/ik_explainer/
  clip_io.py       DLT loader (11-coef -> 3x4 P -> (4,3) P.T); video reader;
                   shipped-CSV loader (the ONLY place x0.1 is applied)
  masks.py         SAM3 -> 01_sam3_masks.npz + centroids    [GPU, torch]
  detect2d.py      4ch ViTPose -> 02_kp2d.npz               [GPU, jax]
  triangulate3d.py triangulate -> 03_kp3d.npz (mm); filter -> 04_kp3d_filt.npz
  stage_ik.py      staged STAC -> 05_stac_ik.h5, 06_stages.npz    [GPU, jax]
  acts/act1_views.py act2_triangulate.py act3_align.py act4_solve.py
  assemble.py      concat + crossfade + titles -> mp4
```

All outputs go to `<CLIP>/ik_explainer/` per the layout above. Every stage
writes its artifact to disk before the next reads it, so the pipeline is
resumable: a failed Act 4 render never re-runs SAM3. Each script takes
`--clip <CLIP>` and derives its input/output paths from that one argument.

Colours come from `viz/core/colors.py` (`PALETTE`, `keypoint_groups`,
`leg_chains`) so this video shares the repo's visual language: white/cyan =
observed/detector, green = fit, grey = mask/mesh.

### Prediction phase

| Step | Uses | Output |
|---|---|---|
| Masks | `jarvis_jax.predict.sam3_driver` | `01_sam3_masks.npz` (7, 921, 448, 1936) bool, compressed + centroids |
| 2D | `tracking.predict_2d.predict_bout_2d` | `02_kp2d.npz` (7, 921, 50, 2) + conf |
| 3D | `tracking.triangulate.triangulate_keypoints` | `03_kp3d.npz` (921, 50, 3) **mm** + conf3d |
| Filter | `tracking.filter.filter_bout_kp3d` | `04_kp3d_filt.npz` |
| IK | `stage_ik.py` (see below) | `05_stac_ik.h5`, `06_stages.npz` |

Two details that are easy to get wrong and are load-bearing:

- **Keypoint order.** `predict_2d.detector_to_model_perm` /
  `reorder_detector_to_model` must be applied to map detector channel order onto
  model `KP_NAMES`. CLAUDE.md records a keypoint-order bug that scrambled
  anatomy while LOO/IoU QC stayed entirely blind to it.
- **Filtering is not optional.** `tracking/filter.py`'s docstring records that
  feeding raw triangulated 3D to STAC makes distal keypoints (esp. `*_TaTip`)
  jump many mm frame-to-frame, and STAC then bends the leg to chase the outlier
  — measured foot-tip max acceleration 12.5 → 0.9 mm and ~7× lower IK jitter
  with the filter on. Act 4's whole claim is that tarsal tips track the
  keypoints, so this stage is a prerequisite, not a polish.

`distractor_masks` is `None` — documented as correct for single-animal assays.

### `stage_ik.py`

Replicates `Stac.fit_offsets`' control flow while snapshotting `mjx_data.qpos`
after each stage, rather than keeping only the final result:

| Snapshot | Produced by | Act |
|---|---|---|
| `qpos_default` | model rest pose | 3 (the genuine misaligned state) |
| `qpos_scaled` | `stac_mjx/rescale.py` body scale | 3 |
| `qpos_root` | `compute_stac.root_optimization` | 3 |
| `qpos_pose` | `compute_stac.pose_optimization` | 4 |
| `qpos[t]` | final per-frame sequence | 4 |

Also records mean marker residual (mm) at each snapshot, driving the on-screen
readout and doubling as the correctness gate.

Anatomy is **v1** (`configs/anatomy/v1.yaml` →
`models/fruitfly_v1/fruitfly_v1_free.xml`): 86 meshes, `nq=93`, shipping named
cameras (`hero`, `side`, `bottom`, `track1`) that frame Acts 3–4 directly.
Verified to render cleanly at rest.

Keypoints reach STAC via `scripts/preprocess_keypoints_for_ik.py`, which
reorders columns to MuJoCo site order — fed **in mm**.

## The four acts

**Act 1 — seven views (0–15 s).** 7-panel grid of fly-centred crops playing real
video; 2D keypoints fade in per panel, coloured by group with leg chains drawn.
Labels: camera name + elevation, `800 fps → 1/27 speed`, mm scale bar.
Low-confidence keypoints fade in **dimmer**, so per-view occlusion failures are
visible — the honest setup for why Act 2 needs seven cameras.

**Act 2 — triangulation (15–25 s).** Panels detach from the grid and fly to
their true rig orientations (the 30° arc above). Video fades, leaving 2D
keypoint constellations on translucent panels. **Parallel** rays cast inward
along each optical axis; the 3D cloud materialises where they meet; panels fade.
Because the cloud *is* the triangulation of the 2D shown, this is a computed
result, not a visual assertion. On-screen caveat that panel distance is staging,
direction is real.

**Act 3 — meet the model (25–45 s).** Time frozen on one pose, camera slowly
orbiting. The v1 mesh fades in at default `qpos` — wrong size, wrong place,
beside the keypoint cloud. Then two real solver beats: body scale (factor shown,
mm), then `root_optimization` rigidly translating and rotating the mesh onto the
cloud, mean marker residual ticking down in mm.

**Act 4 — the solve, then motion (45–75 s).** `pose_optimization` bends the
joints onto the keypoints, residual dropping. Then time starts: the per-frame
`qpos` sequence plays with cloud and mesh moving together. Closes on a 2-up —
the 3D fit beside one real camera view with the fit reprojected onto video,
returning to where Act 1 began.

## Verification

Per CLAUDE.md, each act states its expectation *before* rendering, is compared
against something, and the resulting figure is read back before any claim.
Figures land in `<CLIP>/ik_explainer/qc/`, numbered to match the
`predictions/` stage that produced them, with the numbers in `qc/qc.json`.

| Stage | Expectation if correct | Falsification |
|---|---|---|
| Masks | one connected fly-sized blob per view, tracking across frames | blobs on arena features, or identity flicker |
| 2D | keypoints on the fly in all 7 views; confidence drops on occluded views | offset from body, or uniformly confident (⇒ not real detector output) |
| 3D | median 3D disagreement vs the **shipped CSV baseline** within a few tens of µm, given 1 px ⇒ 9.3 µm | systematic mm-scale offset ⇒ unit or ordering error |
| Filter | `*_TaTip` max acceleration drops (docstring precedent: 12.5 → 0.9 mm) | unchanged ⇒ filter not applied |
| Act 2 | rays from all 7 panels meet within the marker residual | rays miss, or converge (⇒ perspective used on a telecentric rig) |
| Act 3 | residual **decreases monotonically** scale→root; mesh anatomically oriented | residual flat/rising, or mesh 180°-flipped with head on abdomen keypoints |
| Act 4 | tarsal tips track observed keypoints through leg swing | tips detach during swing (⇒ pose stage not converged) |

The shipped CSV, freed from being the 3D source, becomes an **independent
baseline**: a re-predicted-vs-shipped 3D comparison figure is a free QC artifact
that would catch a keypoint-order or unit error immediately.

Act 1's reprojection geometry is **already proven** on real frames of this clip
(all 7 cameras, `figures/2026-08-13-ik-anim-dlt-check/`): keypoints land on the
fly, `Antenna_Base`/`EyeL`/`EyeR` on the head, `Scutellum` on the thorax,
`Abd_A4`→`Abd_tip` marching posteriorly, `Wing*_base`→`V12`/`V13` out along the
wing blade, confidences 0.86–0.97.

Renders need `MUJOCO_GL=egl` and a GPU. Work runs directly on `glados`
(2× RTX A6000, idle) — not via sbatch, and never on a login node.

## Risks

1. **SAM3 runtime is unknown.** 7 × 921 = 6,447 frames through the pipeline's
   heaviest stage. It is installed and importable, but the cost is unmeasured;
   this is the main schedule risk. Mitigation: run masks first, standalone, and
   measure before committing to the rest.
2. **Mask quality now gates everything.** The detector reads the mask as crop
   channel 3, so a bad mask degrades 2D, hence 3D, hence IK. Mitigation: the
   mask QC row in the verification table, checked before the detector runs.
3. **Clip format is nonstandard** — no bout summary, DLT rather than
   `Cam*.yaml`. Both SAM3 and `preprocess_keypoints_for_ik.py` need a thin
   adapter. Main unknown-effort item after SAM3.
4. **Keypoint order** — see the prediction-phase note. Caught by the
   shipped-CSV comparison.
5. **1.15 s of fly time** — Act 4's walk is short. Fallbacks: hold on slower
   playback, or loop the sequence.
6. **Borrowed calibration** — assumed valid; reprojects cleanly.
7. **Easy-case fly** — the video overstates typical pipeline performance and
   should be labelled an explainer, not QC evidence.

## Open decisions (assumed, not confirmed)

- The full clip (all `N` = 921 frames) rather than a sub-range.
- No captions beyond stage labels.
