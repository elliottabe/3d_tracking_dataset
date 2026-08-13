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

## Deliverable

`figures/2026-08-13-ik-explainer/ik_explainer.mp4` — **1920×1080, 30 fps, ~75 s,
H.264, silent**, with burned-in stage labels. The generating code is committed
under `scripts/viz/ik_explainer/`; per CLAUDE.md the video itself is never
committed, so the generator is the durable artifact.

## Scope

**In scope:**
- A committed generator under `scripts/viz/ik_explainer/` producing one mp4.
- A real ViTPose 2D detector pass on this clip (Act 1).
- A real STAC IK solve on this clip, instrumented to snapshot intermediate
  stages (Acts 3–4).
- A thin adapter for this clip's nonstandard format (DLT csv, no bout summary).

**Out of scope:**
- Voiceover / narration. Burned-in stage labels only.
- Changing any pipeline stage. The adapter reads this clip; it does not alter
  how production recordings are processed.
- Generalising the generator to arbitrary recordings. It targets this clip;
  paths are arguments, but the format assumptions are this clip's.
- SAM3 masks. Crops are seeded from the existing 3D (see `detect2d.py`).

## Verified facts about the source clip

Established by inspection before design; the generator depends on all of these.

| Property | Value |
|---|---|
| Cameras | 7 (`Cam2012630/631/853/855/857/861/862`) |
| Video | 1936×448, **800 fps**, 921 frames (+ contrast-`enhanced/` copies) |
| Calibration | 7× **11-coefficient DLT** (`calibration/Cam*_dlt.csv`) |
| 3D keypoints | `data3D_frames1161383-1162304.csv`, 50 kp × 922 frames + confidence |
| Clip duration | 922 / 800 = **1.15 s of fly time** |

**Frame-count mismatch:** `ffprobe` reports **921** video frames while the 3D CSV
has **922** data rows. The adapter must reconcile this explicitly rather than
assuming alignment — an off-by-one here silently shifts every 2D overlay by one
frame, which at 800 fps is invisible to the eye but corrupts the residual. The
implementation resolves it by indexing on the video and truncating the CSV to
`min(n_video, n_csv)`, and asserting the two differ by at most 1.

### The unit trap (top correctness risk)

**The 3D CSV is in units of 0.1 mm; this clip's DLT expects mm.** Projecting the
CSV raw gives `u ∈ [7927, 10705]` on a 1936-px-wide frame — **0% of keypoints in
frame**. Scaling XYZ by 0.1 first gives **100% in frame across all 7 cameras and
all 922 frames**, and a body length (`Antenna_Base`→`Abd_tip`) of **2.39 mm**,
correct for *Drosophila*.

Two independent confirmations:
- `third_party/jarvis_jax/tests/test_identity_link.py:17` holds `P_REAL`, a real
  rig DLT, at linear-term scale ~8.1 where this clip's is ~81.2 — a clean 10×.
- Factoring the DLTs gives a magnification of **80.7 px/mm** on every camera;
  80.7 × 2.39 mm = 193 px, matching the observed fly size.

CLAUDE.md records marker offsets silently absorbing a 38× body-scale error, with
residual and NaN checks blind to it. This is the same failure mode. The scale
conversion therefore lives in exactly one place (`clip_io.py`) and the marker
residual is displayed **in mm on screen**, so a unit error becomes visible rather
than absorbed.

### Rig geometry (recovered, drives Act 2)

Factoring each DLT with
`jarvis_jax.tracking.affine_camera.factor_affine` (valid because `L9=L10=L11=0`,
i.e. the rig is telecentric/affine) recovers the optical axes:

| Camera | Azimuth | Elevation | Role |
|---|---|---|---|
| `Cam2012857` | −90.0° | −0.6° | side, one end of arc |
| `Cam2012855` | −90.0° | −30.5° | |
| `Cam2012853` | −90.4° | −60.5° | |
| `Cam2012630` | — | **−89.6°** | top / vertical |
| `Cam2012862` | +91.1° | −59.5° | |
| `Cam2012631` | +90.5° | −29.3° | |
| `Cam2012861` | +90.7° | +0.6° | side, far end of arc |

Pairwise angles between optical axes come out at 30.0 / 60.0 / 90.0 / 120.2 /
150.1°, with `857`↔`861` at 179.3°: **a 180° arc at 30° spacing**, all viewing
perpendicular to the arena's long X axis. Magnification is uniform 80.7 px/mm.

**Consequence for Act 2:** because the cameras are telecentric they have no
finite centre of projection — rays are *parallel*, and the camera is
effectively at infinity. Panel *orientation* in the 3D transition is real rig
geometry; panel *distance* is an arbitrary staging choice. Act 2 must draw
parallel rays, not converging ones, and must say so on screen.

### Gaps this design fills

The clip has **no 2D keypoints** and **no IK output**. Both are generated by
running the real pipeline stages (chosen over reprojecting/faking: reprojected
2D would point the causal arrow backwards, and a hand-authored IK finale in an
*IK explainer* would be indefensible).

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

Each act renders an independent numbered PNG sequence into its own directory; a
final assembler concatenates them with crossfades. Acts are re-renderable in
isolation — Act 1 costs a GPU detector pass and Act 4 a GPU MuJoCo pass, and
neither should have to be repaid to fix a typo in a label.

```
scripts/viz/ik_explainer/
  clip_io.py     DLT loader (11-coef -> 3x4 affine P); 3D CSV loader
                 (THE 0.1 scale lives here, once); fly-centred video crops
  detect2d.py    ViTPose -> kp2d.npz          [GPU]
  stage_ik.py    staged STAC driver -> stages.npz + stac_ik.h5   [GPU]
  acts/act1_views.py act2_triangulate.py act3_align.py act4_solve.py
  assemble.py    concat + crossfade + titles -> mp4
```

Outputs: frames and mp4 under `figures/2026-08-13-ik-explainer/` (gitignored);
the small `.npz`/`.json` stage artifacts saved beside them so the render can be
re-examined without re-solving.

Colours come from `viz/core/colors.py` (`PALETTE`, `keypoint_groups`,
`leg_chains`) so this video shares the repo's visual language: white/cyan =
observed/detector, green = fit, grey = mask/mesh.

### `detect2d.py`

Runs the ViTPose checkpoint at
`/data2/users/eabe/datasets/3d_tracking/jax_vitpose_runs/v4_8gpu_20260808/final`
over 7 cameras × `N` frames (`N = min(n_video, n_csv) = 921`, see the
frame-count mismatch above). Crop centres come from reprojecting the existing 3D
centroid through each DLT, which **bypasses SAM3 entirely** — the masks exist
only to locate the animal, and here we already know where it is. Output
`kp2d.npz`: `(7, N, 50, 3)` as `(u, v, confidence)`.

This is genuine detector output including its misses; it is not derived from the
3D beyond crop placement.

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

Also records mean marker residual (mm) at each snapshot, which drives the
on-screen readout and doubles as the correctness gate.

Anatomy is **v1** (`configs/anatomy/v1.yaml` →
`models/fruitfly_v1/fruitfly_v1_free.xml`): 86 meshes, `nq=93`, and it ships
named cameras (`hero`, `side`, `bottom`, `track1`) that frame Acts 3–4 directly.
Verified to render cleanly at rest.

Input keypoints reach STAC via `scripts/preprocess_keypoints_for_ik.py`, which
already reads a `data3D` CSV and reorders columns to MuJoCo site order — **in
mm**.

## The four acts

**Act 1 — seven views (0–15 s).** 7-panel grid of fly-centred crops playing real
video; 2D keypoints fade in per panel, coloured by group with leg chains drawn.
Labels: camera name + elevation, `800 fps → 1/27 speed`, mm scale bar.
Low-confidence keypoints fade in **dimmer**, so per-view occlusion failures are
visible — that is the honest setup for why Act 2 needs seven cameras.

**Act 2 — triangulation (15–25 s).** Panels detach from the grid and fly to
their true rig orientations (the 30° arc above). Video fades out, leaving 2D
keypoint constellations on translucent panels. **Parallel** rays cast inward
along each optical axis; the 3D cloud materialises where they meet; panels fade.
On-screen caveat that panel distance is staging, direction is real.

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

| Act | Expectation if correct | Falsification |
|---|---|---|
| 1 | Detector 2D lands on the fly in all 7 views; confidence drops on occluded views | keypoints offset from the body, or uniformly confident (⇒ not real detector output) |
| 2 | Rays from all 7 panels meet within the marker residual | rays miss, or converge (⇒ perspective used where the rig is telecentric) |
| 3 | Residual **decreases monotonically** scale→root; mesh ends anatomically oriented | residual flat/rising, or mesh 180°-flipped with head on the abdomen keypoints |
| 4 | Tarsal tips track observed keypoints through leg swing | tips detach during swing (⇒ pose stage not converged) |

Act 1's reprojection geometry is **already proven** on real frames of this clip
(all 7 cameras, `figures/2026-08-13-ik-anim-dlt-check/`): keypoints land on the
fly, `Antenna_Base`/`EyeL`/`EyeR` on the head, `Scutellum` on the thorax,
`Abd_A4`→`Abd_tip` marching posteriorly, `Wing*_base`→`V12`/`V13` out along the
wing blade, confidences 0.86–0.97.

Renders need `MUJOCO_GL=egl` and a GPU. Work runs directly on `glados`
(2× RTX A6000, idle) — not via sbatch, and never on a login node.

## Risks

1. **Clip format is nonstandard** — no bout summary, DLT rather than
   `Cam*.yaml`. Both `detect2d.py` and `preprocess_keypoints_for_ik.py` need a
   thin adapter. This is the main unknown-effort item.
2. **Units** — see the unit trap above. Mitigated by single-point conversion plus
   an on-screen mm residual.
3. **1.15 s of fly time** — Act 4's walk is short. Fallbacks: hold on slower
   playback, or loop the sequence.
4. **Borrowed calibration** — assumed valid; reprojects cleanly.
5. **Easy-case fly** — the video overstates typical pipeline performance and
   should be labelled as an explainer, not QC evidence.

## Open decisions (assumed, not confirmed)

- The full clip (all `N` = 921 usable frames) rather than a sub-range.
- No captions beyond stage labels.
