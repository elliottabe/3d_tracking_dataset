# viz

Centralized visualization CLI for the STAC-fit and courtship (SAM3/JARVIS)
pipelines. All views are invoked as subcommands of one entry point:

```bash
python -m viz <subcommand> [options]
```

Run `python -m viz --help` or `python -m viz <subcommand> --help` for the
authoritative, up-to-date flag list — the table below is a quick-reference,
not a substitute.

## Subcommands

| Subcommand | Purpose | Example |
| --- | --- | --- |
| `overlay` | Mesh + keypoint + mask + axis reprojection onto each camera's raw frame for one (bout, fly, frame), montaged into a single PNG. | `python -m viz overlay --run /path/to/Session0_bouts_2026_05_27 --bout 3 --fly 0 --frame 120 --cams cam1 cam2 --show mesh,kp,mask,axis --out overlay_b3_f0.png` |
| `legskel` | Per-leg proximal→distal joint chains — detector-2D vs fitted-3D (and optionally a second run's fitted-3D for comparison) — reprojected per camera into a montage. | `python -m viz legskel --run /path/to/Session0_bouts_2026_05_27 --bout 3 --fly 0 --frame 120 --cams cam1 cam2 --compare /path/to/other_run --out legskel_b3_f0.png` |
| `kp-qc` | Detector pred-vs-GT keypoint QC: per-keypoint error, single recording or paired female-vs-male courtship layout. | Single: `python -m viz kp-qc --run-dir /path/to/train_run --ckpt best.ckpt --recording 2026_05_27_11_56_05 --n 8 --out kp_qc.png`<br>Paired: `python -m viz kp-qc --run-dir /path/to/train_run --ckpt best.ckpt --female-vs-male --female-rec 2026_05_27_11_56_05 --male-rec 2026_05_27_11_57_05 --data-root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3 --out kp_qc_mf.png` |
| `fit-check` | STAC-fit verification frames: mesh + data keypoints + fitted model markers + a residual-length error tendon per keypoint. | `python -m viz fit-check /path/to/fit.h5 --start 0 --n 10 --camera track1 --body-model-dir /home/eabe/Research/MyRepos/fruitfly_body_models --out fit_check.mp4` |
| `reproj-video` | Predicted 3D keypoints reprojected onto a bout's raw camera video (one mp4 per camera); JARVIS-free reimplementation on `viz.core`. | `python -m viz reproj-video --session-dir /path/to/session --pred-dir /path/to/predictions --bout 3 --cameras cam1 cam2 --with-masks --max-frames 500 --out reproj_bout3` |
| `clip` | Multi-camera clip `cut` \| `stack` \| `render` for a bout's frame range; streams frames, no ffmpeg subprocess, no JARVIS. | Cut: `python -m viz clip cut --session-dir /path/to/session --start 1000 --end 1200 --cameras cam1 cam2 --out /path/to/session/clips_1000_1200`<br>Stack: `python -m viz clip stack --session-dir /path/to/session/clips_1000_1200 --cameras cam1 cam2 --out stacked.mp4`<br>Render: `python -m viz clip render --session-dir /path/to/session --bout-dir /path/to/predictions/bout_00003 --cameras cam1 cam2 --fps 30 --out bout3_vstack.mp4` |

Notes on selected flags:
- `overlay`/`legskel` share `--run`/`--bout`/`--fly`/`--frame`/`--cams`/`--out`;
  `overlay` additionally takes `--show` (comma list, default
  `mesh,kp,mask,axis`), `--compare` (a second run dir), and `--bodyalign`
  (flag); `legskel` additionally takes `--compare`.
- `kp-qc` resolves paths via `viz.core.config.courtship_recording()`-style
  Hydra config rather than `--run`; `--n` has no default so it doesn't
  override the mode-specific defaults (8 for single, 5-per-sex for paired).
- `fit-check` takes the STAC `ik_h5` output as a positional argument; add
  `--no-error` to drop the residual tendons, `--n-stills` to control how many
  still frames are also written alongside the video.
- `clip`'s `mode` is positional (`cut`, `stack`, or `render`); `render` can
  infer `--start`/`--end` from `--bout-dir`'s `fly0.csv` if not given
  explicitly.

## Architecture

Shared engine lives in `viz/core/`:
- `reproject.py` — 3D(mm)→2D(px) camera reprojection (`ReprojectionTool`
  convention, matches `scripts/run_courtship_bout.py::project_points`).
- `overlays.py` — cv2 drawing primitives (points, chains, masks, axes) on a
  BGR uint8 image.
- `io.py` — artifact loaders, run-path resolution, frame reading/writing.
- `layout.py` — multi-camera tile arrangement/cropping for montages.
- `colors.py` — one shared BGR palette + keypoint-chain semantics, derived
  from `KP_NAMES` so it works for any keypoint ordering.
- `config.py` — resolves recording paths (`calib_dir`/`session_dir`/
  `predictions_dir`/`cameras`/`KP_NAMES`) for the courtship pipeline via
  Hydra (`courtship_recording()`).

Each `viz/views/*.py` is a thin CLI view: parse args → load via `core.io` →
reproject via `core.reproject` → draw via `core.overlays` → arrange via
`core.layout` → save. Logic worth unit-testing belongs in `core/`, not the
views.

## Tests

```bash
OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu python -m pytest viz/tests/ -q
```

The `kp-qc` and `fit-check` GPU-backed smoke tests skip under
`JAX_PLATFORMS=cpu`; everything else (core, layout, colors, reproject, cli)
runs on CPU.
