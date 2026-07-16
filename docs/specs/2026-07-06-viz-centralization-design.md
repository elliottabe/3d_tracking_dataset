# Visualization Centralization — Design

**Date:** 2026-07-06
**Status:** design approved; ready for implementation plan

## Goal

Replace the scattered, copy-paste visualization scripts with one organized,
importable `viz/` package — a shared reprojection/overlay core plus a unified
CLI — covering both the original STAC pipeline and the new courtship
mesh/silhouette pipeline, and promoting the useful one-off courtship-QC
overlays built during debugging into first-class, reusable views.

## Motivation

Visualization currently lives in ~10 places that each re-implement 3D→2D
reprojection and cv2/matplotlib overlay drawing:

- **Detector-QC figures** (matplotlib): `scripts/viz_keypoints.py`,
  `scripts/viz_courtship_mf.py` — predicted vs GT keypoints + per-keypoint error.
- **3D-reprojection overlays** (cv2 on frames/video): `scripts/viz_predictions_reproject.py`,
  `scripts/render_fit_check.py`, plus scratchpad one-offs (both-flies mesh
  overlay, leg-skeleton vs detector, mask/orientation checks, detector-vs-fit,
  body-alignment).
- **Video-clip assembly**: `scripts/viz/{cut_videos_by_frame,make_bout_clip,render_bout_clips,stack_clips}.py`.

They all duplicate logic that already exists as `project_points`
(`scripts/run_bout.py`) and `ReprojectionTool`
(`jarvis_jax.geometry.reprojection_tool`).

## Architecture

Top-level importable `viz/` package (peer to `scripts/`, `configs/`), invoked as
`python -m viz <subcommand> …`. A pipeline-agnostic **core** carries the shared
engine; **views** (one per subcommand) build pipeline-specific renders on it.
The core lives at the repo top level (not inside `jarvis_jax`) so it can import
both `jarvis_jax` and `stac_mjx`/JARVIS without making `jarvis_jax` depend on
`stac_mjx`.

```
viz/
├── __main__.py          # `python -m viz <subcommand> …` dispatcher
├── cli.py               # argparse subcommands -> view functions
├── core/
│   ├── reproject.py     # cam_matrices(calib_dir); project(cam_mat, pts3d); reproject_all(rt, pts3d) -> per-cam 2D  (wraps ReprojectionTool/project_points)
│   ├── overlays.py      # cv2 primitives: keypoints, skeleton, mesh-cloud, mask outline/fill, leg-chains, head->tail axis, legend
│   ├── io.py            # loaders: outputs.h5 (kp3d_mm/mesh_mm), kp2d/kp3d npz, SAM masks (reuse jarvis_jax.tracking.bout_masks), video frames, COCO GT, data3D CSVs
│   ├── layout.py        # per-camera montage/grid, crop-to-content, banner
│   └── colors.py        # single palette: fly0/fly1, head/tail, detector/fit, keypoint groups
└── tests/               # unit tests for the core
```

**Design-for-isolation:** each core module has one responsibility and a small
interface (reproject: 3D→2D; overlays: draw on an image; io: load an artifact;
layout: arrange tiles; colors: the palette). Views compose them and never
re-implement reprojection or drawing.

## Views (CLI subcommands)

| subcommand | replaces / promotes | renders |
|---|---|---|
| `viz overlay` | scratchpad both-flies, mask-orient, detector-vs-fit, body-align | mesh cloud + keypoints + SAM mask reprojected on frames, per-camera montage. Flags: `--show mesh,kp,mask,axis`, `--fly`, `--frame`, `--cams`, `--compare <run2>`, `--bodyalign` |
| `viz legskel` | scratchpad legskel/legcompare | leg-joint chains: detector-2D vs fitted vs triangulated, per camera |
| `viz kp-qc` | `viz_keypoints`, `viz_courtship_mf` | matplotlib: pred vs GT keypoints + per-keypoint error (single recording; `--female-vs-male` for the paired courtship view) |
| `viz fit-check` | `render_fit_check` | STAC-fit verification frames (model + markers + residual tendons; via `stac_mjx.viz`) |
| `viz reproj-video` | `viz_predictions_reproject` | predicted 3D reprojected onto a bout's raw video, reimplemented on the core (drops the JARVIS `create_multi_animal_videos3D` dependency) |
| `viz clip` | `scripts/viz/` (cut / make_bout_clip / stack / render_bout_clips) | multi-camera clip cut/stack/render for a bout frame range |

Common flags (`--run`, `--bout`, `--fly`, `--frame`, `--cams`, `--out`) are
shared and resolved to artifacts through `core.io`.

## Data flow

CLI → view → `core.io` loads (outputs.h5 / masks / frames / GT / CSV) →
`core.reproject` maps 3D→2D per camera → `core.overlays` draws → `core.layout`
tiles into a montage → save PNG (figures/overlays) or MP4 (video/clips).

## Migration

- **Port** the four repo scripts + the `scripts/viz/` dir into the views above,
  then **delete** the originals (`scripts/viz_keypoints.py`,
  `scripts/viz_courtship_mf.py`, `scripts/viz_predictions_reproject.py`,
  `scripts/render_fit_check.py`, and `scripts/viz/`).
- **Promote** the scratchpad courtship-QC overlays (both-flies, mask-orient,
  detector-headtail, legmask-overlay, legskel, legcompare, bodyalign) into
  `viz overlay` / `viz legskel`, rewritten cleanly on the core.
- **Exclude** (not promoted — one-off build-time debugging, work already
  done/abandoned): `viz_sites`, `viz_leg_sites`, `viz_wingblade`, `viz_wingmesh`.

Net effect: `scripts/` loses the four top-level viz scripts and the
`scripts/viz/` directory; all visualization lives under `viz/`.

## Testing

- **Core, unit-tested** (`viz/tests/`): reproject (project a known 3D point,
  compare to a hand-computed 2D pixel), overlay primitives (draw on a synthetic
  array; assert target pixels changed and non-target channels untouched — e.g.
  the mask channel), io loaders (tiny synthetic `outputs.h5` / npz / COCO
  fixtures), layout montage (grid dimensions), CLI (arg-parse + dispatch to a
  stubbed view).
- **Views, smoke-tested**: run each on a tiny fixture / minimal frame count to
  confirm it produces an output file without error; not pixel-compared (matches
  how the pipeline's heavy renders are handled). GPU-dependent renders
  (`fit-check` via mujoco EGL) run only where a GPU is available.

## Dependencies

`numpy`, `cv2`, `matplotlib`, `h5` (via `stac_mjx.io_dict_to_hdf5`), and
`jarvis_jax` (`ReprojectionTool`, `cse.courtship_bout_masks`, outputs loaders).
`viz fit-check` additionally imports `stac_mjx.viz`. `viz reproj-video` is
reimplemented on the core rather than calling the JARVIS monolith.

## Error handling

- Missing run/artifact → clear error naming the path.
- Missing camera video or out-of-range frame → skip that tile with a warning
  (don't abort the whole montage).
- Empty bout (K=0 keypoints / no valid views) → render the frame with a
  "no data" note rather than crashing.
- Calibration/camera-count mismatch → surfaced (reuse the pipeline's
  fail-loud checks where applicable).

## Non-goals

- No new visualization *types* beyond consolidating/promoting what exists.
- No change to pipeline outputs or the artifacts being visualized.
- Not a general plotting framework; scoped to this repo's two pipelines.
