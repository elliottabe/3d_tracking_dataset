# 3D Tracking Dataset

Pipeline for processing 3D keypoint tracking of fruit fly (*Drosophila*) behavior:
Procrustes alignment, inverse kinematics via STAC-MJX, and analysis utilities for
free-walking and courtship datasets.

## Quick Start

```bash
# Clone with submodules
git clone --recurse-submodules <repository-url>
cd 3d_tracking_dataset

# (If already cloned without --recurse-submodules)
git submodule update --init --recursive

# Recommended: install via uv (Python deps in pyproject.toml + uv.lock)
uv sync --extra cuda12 --extra dev

# Run a smoke test
uv run python test_configs.py
uv run pytest tests/
```

`uv sync` creates `.venv/` with all Python dependencies including the editable
`stac-mjx` submodule. The `cuda12` extra pulls JAX with CUDA 12 wheels; drop it
on CPU-only systems. The `dev` extra adds pytest, ruff, jupyterlab, and
nbstripout.

### Optional: conda for system libraries

If your system lacks headless OpenGL or ffmpeg, [environment.yml](environment.yml)
provides a minimal conda env with those system libraries:

```bash
conda env create -f environment.yml
conda activate fly-3d-tracking
uv sync --extra cuda12 --extra dev
```

### Notebook outputs

A `nbstripout` filter is declared in [.gitattributes](.gitattributes); contributors
should run `nbstripout --install` after cloning so notebook outputs are
automatically removed from commits.

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
uv run python scripts/run_full_pipeline.py --anatomy v1 --dataset free_walking
```

## Repository structure

```
3d_tracking_dataset/
├── configs/                  Hydra configuration (anatomy, dataset, paths)
│   ├── anatomy/              v1, v2, v2_muscles model definitions
│   ├── dataset/              free_walking, courtship, etc.
│   └── paths/                per-machine path templates (incl. template.yaml)
├── models/
│   └── fruitfly_v1/          vendored MuJoCo model used by anatomy=v1
├── notebooks/                analysis notebooks (figures, validation)
├── scripts/                  pipeline scripts (preprocess, IK, postprocess, combine)
├── utils/                    shared library code (IO, alignment, song analysis, ...)
├── stac-mjx/                 git submodule: STAC inverse-kinematics solver
├── tests/                    pytest suite
├── pyproject.toml            uv-managed dependency spec
├── uv.lock                   resolved lockfile
└── environment.yml           optional conda env for system-level libs
```

## Pipeline overview

The full pipeline runs in four steps; each can also be invoked individually.

```bash
# All-in-one (after FLY3D_DATA_ROOT or --base-dir is set)
uv run python scripts/run_full_pipeline.py --anatomy v1 --dataset free_walking

# Or step by step
uv run python scripts/batch_process_predictions.py     --anatomy v1   # 1. preprocess
uv run python scripts/batch_split_valid_bouts.py       --dataset courtship  # courtship only
uv run python scripts/batch_run_stac.py                --anatomy v1   # 2. STAC IK
uv run python scripts/batch_postprocess_predictions.py --anatomy v1   # 3. postprocess
uv run python scripts/combine_data.py paths=mymachine dataset=free_walking anatomy=v1  # 4. combine
```

See [BATCH_PROCESSING.md](BATCH_PROCESSING.md) for full pipeline documentation.

## Notebooks

- [notebooks/Verify_Data.ipynb](notebooks/Verify_Data.ipynb) — load and validate processed data
- [notebooks/Courtship_Song_Figures2.ipynb](notebooks/Courtship_Song_Figures2.ipynb) — courtship song & rendered-bout figures
- [notebooks/Joint_Kinematics_Analysis.ipynb](notebooks/Joint_Kinematics_Analysis.ipynb) — PCA/UMAP analysis of joint kinematics during free walking
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

- [stac-mjx/](stac-mjx/) — STAC inverse kinematics solver, installed editable.

See [SUBMODULES.md](SUBMODULES.md) for submodule workflow.

## Citation

If you use this dataset or pipeline, please cite the accompanying manuscript
(see preprint link / DOI here).
