# 3D Tracking Dataset

Pipeline for processing 3D keypoint tracking of fruit fly (*Drosophila*) behavior:
Procrustes alignment, inverse kinematics via STAC-MJX, and analysis utilities for
free-running and courtship datasets.

## Quick Start

The recommended workflow is to create a minimal conda environment that
provides Python 3.12, `uv`, and the system-level libraries needed for
headless rendering, and then let `uv` install all the Python dependencies
on top of it.

```bash
# Clone with submodules (use --shallow-submodules to avoid pulling JARVIS history)
git clone --recurse-submodules --shallow-submodules <repository-url>
cd 3d_tracking_dataset

# (If already cloned without --recurse-submodules)
git submodule update --init --recursive

# 1. Minimal conda env: Python 3.12 + uv + ffmpeg + headless OpenGL
conda env create -f environment.yml
conda activate fly-3d-tracking

# 2. Install all Python deps into the active conda env via uv
uv sync --active --extra cuda12 --extra dev

# 3. Smoke test
python test_configs.py
```

To pull in the optional upstream tools (rendering bout clips with JARVIS,
regenerating SAM3 masks), add their extras:

```bash
uv sync --active --extra cuda12 --extra dev --extra jarvis --extra sam3
```

See [External tools](#external-tools) below for details on what each does
and when you actually need it.

The `--active` flag tells `uv` to install into the activated conda env
instead of creating a separate `.venv/`. `uv.lock` pins exact versions for
reproducibility. The `cuda12` extra pulls JAX with CUDA 12 wheels (drop it
on CPU-only systems); `dev` adds pytest, ruff, jupyterlab, and nbstripout.

### Pure-uv install (no conda)

If your system already has ffmpeg and headless OpenGL libraries available
(e.g. via apt: `ffmpeg libegl1 libgl1 libglfw3`), you can skip conda
entirely and let `uv` create its own venv:

```bash
uv sync --extra cuda12 --extra dev
uv run python test_configs.py
uv run pytest tests/
```

### Notebook outputs

A `nbstripout` filter is declared in [.gitattributes](.gitattributes); contributors
should run `nbstripout --install` after cloning so notebook outputs are
automatically removed from commits.

## External tools

Two upstream tools are integrated as opt-in extras. They are not required
for the core analysis pipeline; pre-computed outputs from each ship with
the dataset.

### JARVIS-HybridNet

Multi-view markerless 3D pose estimation (Hüser et al.). Pinned as a git
submodule under [third_party/JARVIS-HybridNet/](third_party/JARVIS-HybridNet/)
on the publication branch `elliottabe/multianimal-publication`.

You need JARVIS only if you want to:

- **Reproduce the upstream tracking step** — running JARVIS on raw
  multi-view video produces the per-fly `data3D.csv` files this repo
  consumes. See JARVIS's own
  [Getting Started Guide](https://jarvis-mocap.github.io/jarvis-docs/) for
  the full workflow.
- **Render bout clips** — [scripts/render_bout_clips.py](scripts/render_bout_clips.py)
  uses `jarvis.visualization.create_multi_animal_videos3D` to overlay
  3D poses on multi-camera video for QC.

Install:

```bash
uv sync --active --extra jarvis
```

The extra adds the editable JARVIS submodule plus its transitive deps
(streamlit, imgaug, etc.). It works with Python 3.12 because the
publication branch on the fork relaxes JARVIS's old version pins.

### SAM3 (Meta Segment Anything 3)

Used to (re)generate per-bout segmentation masks (`sam3_masks.npz`,
`sam3_aligned.h5`) consumed by the courtship analyses in
[utils/sam3_female_com.py](utils/sam3_female_com.py) and
[utils/sam3_aligned_bouts.py](utils/sam3_aligned_bouts.py).

You need SAM3 only if you want to **regenerate the segmentation masks
from raw video**. The repo's analysis code only reads pre-computed
masks; those mask files ship as data with the dataset.

Install (pulls SAM3 from a fork pinned for numpy 2 compatibility):

```bash
uv sync --active --extra sam3
```

SAM3 weights are auto-fetched from Hugging Face on first use; some
models are gated behind a HF account.

## Configuring data paths

This repo uses [Hydra](https://hydra.cc/) for configuration. Per-machine path
templates live in [configs/paths/](configs/paths/). To set up a new machine:

1. Copy [configs/paths/template.yaml](configs/paths/template.yaml) to a new
   file named after your machine (e.g. `mymachine.yaml`).
2. Fill in `user`, `cwd_dir` (path to this repo checkout), `base_dir`/`data_dir`
   (where your raw datasets live).
3. Select your config on the CLI: `paths=mymachine`.

Pipeline scripts also honour the `FLY3D_DATA_ROOT` environment variable as a
fallback for the dataset root, e.g.:

```bash
export FLY3D_DATA_ROOT=/path/to/Johnson_lab
uv run python scripts/run_full_pipeline.py --anatomy v1 --dataset free_running
```

## Repository structure

```
3d_tracking_dataset/
├── configs/                  Hydra configuration (anatomy, dataset, paths)
│   ├── anatomy/              v1, v2, v2_muscles model definitions
│   ├── dataset/              free_running, courtship, etc.
│   └── paths/                per-machine path templates (incl. template.yaml)
├── models/
│   └── fruitfly_v1/          vendored MuJoCo model used by anatomy=v1
├── notebooks/                analysis notebooks (figures, validation)
├── scripts/                  pipeline scripts (preprocess, IK, postprocess, combine)
├── utils/                    shared library code (IO, alignment, song analysis, ...)
├── stac-mjx/                 git submodule: STAC inverse-kinematics solver
├── third_party/
│   └── JARVIS-HybridNet/     git submodule: multi-view 3D pose estimation (optional)
├── pyproject.toml            uv-managed dependency spec
├── uv.lock                   resolved lockfile
└── environment.yml           optional conda env for system-level libs
```

## Pipeline overview

The full pipeline runs in four steps; each can also be invoked individually.

```bash
# All-in-one (after FLY3D_DATA_ROOT or --base-dir is set)
uv run python scripts/run_full_pipeline.py --anatomy v1 --dataset free_running

# Or step by step
uv run python scripts/batch_process_predictions.py     --anatomy v1   # 1. preprocess
uv run python scripts/batch_split_valid_bouts.py       --dataset courtship  # courtship only
uv run python scripts/batch_run_stac.py                --anatomy v1   # 2. STAC IK
uv run python scripts/batch_postprocess_predictions.py --anatomy v1   # 3. postprocess
uv run python scripts/combine_data.py paths=mymachine dataset=free_running anatomy=v1  # 4. combine
```

See [BATCH_PROCESSING.md](BATCH_PROCESSING.md) for full pipeline documentation.

## Notebooks

- [notebooks/Verify_Data.ipynb](notebooks/Verify_Data.ipynb) — load and validate processed data
- [notebooks/Courtship_Song_Figures2.ipynb](notebooks/Courtship_Song_Figures2.ipynb) — courtship song & rendered-bout figures
- [notebooks/Joint_Kinematics_Analysis.ipynb](notebooks/Joint_Kinematics_Analysis.ipynb) — PCA/UMAP analysis of joint kinematics during free running
- [notebooks/Sandbox_Strict.ipynb](notebooks/Sandbox_Strict.ipynb) — 2D keypoint trajectory overlays
- [notebooks/Scutellum_Height_Running.ipynb](notebooks/Scutellum_Height_Running.ipynb) — inverted-pendulum vs. spring-mass locomotion analysis

## Key features

- **Procrustes alignment** — align keypoints to reference pose with optional scaling
- **Ground-contact alignment** — rotate/translate to align ground plane
- **Configurable keypoint exclusion** — drop noisy keypoints (antenna, wings) from alignment
- **Identity tracking** — `fly_id` propagates through preprocessing → IK → analysis
- **JAX/MJX acceleration** — JIT-compiled alignment and forward kinematics
- **Batch processing** — orchestrated multi-folder runs with logging and resume

## Dependencies

Python ≥ 3.12, JAX (CUDA 12 recommended), MuJoCo & mujoco-mjx, Hydra,
PyTorch (used by SAM3 utilities), HDF5. See [pyproject.toml](pyproject.toml)
for the canonical list.

## Submodules

- [stac-mjx/](stac-mjx/) — STAC inverse kinematics solver, installed editable
  (always required).
- [third_party/JARVIS-HybridNet/](third_party/JARVIS-HybridNet/) — multi-view
  pose estimation, installed editable when `--extra jarvis` is used.

See [SUBMODULES.md](SUBMODULES.md) for submodule workflow.

## Citation

If you use this dataset or pipeline, please cite the accompanying manuscript
(see preprint link / DOI here).
