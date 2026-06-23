# Hydra Configuration Guide

This document describes the Hydra configuration system for `jarvis_jax` and the canonical commands to run all entrypoints. All commands require `conda activate 3d_tracking`.

## Config Tree

The configuration lives in `configs/` and is organized into the following groups:

```
configs/
├── config.yaml              # Defaults list; hydra settings
├── paths/
│   ├── hyak.yaml           # Hyak cluster paths (default)
│   └── workstation.yaml     # Local workstation paths
├── data/
│   ├── v3.yaml             # Red data V3 dataset (default)
│   └── courtship.yaml       # Courtship dataset
├── model/
│   ├── vitpose.yaml         # ViTPose 2D keypoint backbone (shared)
│   ├── v2vnet.yaml          # 3D volumetric net (v2vnet module)
│   └── hybridnet.yaml       # Composes vitpose + v2vnet + 3D params (default)
├── train/
│   ├── cached3d.yaml        # Cached v2vNet training (default)
│   ├── vit2d.yaml           # 2D ViTPose keypoint training
│   └── inline3d.yaml        # Inline 3D HybridNet training
├── cache/
│   └── default.yaml         # Precompute cache settings (default)
├── slurm/
│   ├── ckpt_g2.yaml         # ckpt-g2 partition; 8 GPUs (default)
│   └── gpu_l40s.yaml        # gpu-l40s partition; 4 GPUs
└── viz/
    └── default.yaml         # Visualization & diagnostic settings (default)
```

## Defaults List

`config.yaml` specifies default group selections:

```yaml
defaults:
  - _self_
  - paths: hyak
  - model: hybridnet
  - data: v3
  - cache: default
  - train: cached3d
  - slurm: ckpt_g2
  - viz: default
```

Each can be overridden at the command line (see Override Syntax below).

## Model Composition

The `model/` group uses two distinct strategies:

### `model=vitpose` (2D training)
Mounts ViT fields directly at `cfg.model.*`:
- `cfg.model.num_keypoints = 50`
- `cfg.model.img_size = 448`
- `cfg.model.depth = 12`
- etc.

Used by the 2D ViTPose trainer to access keypoint backbone parameters.

### `model=hybridnet` (3D training)
Composes the **same** `vitpose.yaml` file at two package levels:
- Under `cfg.model.vitpose.*` — the shared ViT definition
- Under `cfg.model.v2vnet.*` — 3D volumetric net config
- Plus 3D training params at `cfg.model.*`:
  - `roi_cube: 48` — cube size for roi-to-grid conversion
  - `grid_spacing: 1` — grid discretization
  - `num_cameras: 7` — number of cameras (matches `data.num_cameras`)
  - `sharpen: 3.0` — soft-argmax sharpening temperature

One ViT definition file (`vitpose.yaml`), used by both 2D and 3D pipelines—no duplication.

## Override Syntax

All commands use Hydra override syntax:

| Form | Meaning |
|------|---------|
| `key=value` | Set a scalar field |
| `group=option` | Select a config group (e.g., `paths=hyak`, `train=cached3d`) |
| `+key=value` | Add a new key (not in the schema) |
| `group.nested.key=value` | Deep override (e.g., `train.total_steps=30000`) |

Examples:
```bash
# Select configs
python -m jarvis_jax.train.train_3d_cached paths=hyak model=hybridnet

# Override scalars
python -m jarvis_jax.train.train_3d_cached train.total_steps=30000 train.sharpen=3

# Add non-schema keys (e.g., convert script)
python -m jarvis_jax.convert.build_checkpoint +convert.out=/path/to/ckpt
```

## Canonical Commands

All commands run from the repository root or as module invocations with `conda activate 3d_tracking`.

### Precompute Cache

Builds reprojected volume heatmaps. Run from the `third_party/jarvis_jax` directory:

```bash
cd third_party/jarvis_jax && python scripts/precompute_repro_cache.py paths=hyak cache.split=train
cd third_party/jarvis_jax && python scripts/precompute_repro_cache.py paths=hyak cache.split=val
```

The script is idempotent: if cache metadata exists and matches config, the precompute is skipped unless `cache.force=true` is given.

### Train Cached 3D (v2vNet on frozen ViTPose)

```bash
python -m jarvis_jax.train.train_3d_cached \
    run_id=myrun \
    train=cached3d \
    train.total_steps=20000 \
    train.sharpen=3 \
    train.laplacian_weight=0.05 \
    paths=hyak
```

**Resume or extend a run:** use the same `run_id` to pick up from the last checkpoint and update hyperparameters:

```bash
python -m jarvis_jax.train.train_3d_cached \
    run_id=here_run3 \
    train=cached3d \
    train.total_steps=30000 \
    train.sharpen=3 \
    paths=hyak
```

### Train 2D ViTPose

```bash
python -m jarvis_jax.scripts.train_keypoints \
    run_id=vit_run \
    model=vitpose \
    train=vit2d \
    train.total_steps=20000 \
    paths=hyak
```

### Train Inline 3D (HybridNet with ViTPose inference)

```bash
python -m jarvis_jax.train.train_3d \
    run_id=inl_run \
    train=inline3d \
    train.total_steps=20000 \
    paths=hyak
```

### Build ViTPose Checkpoint from MAE npz

```bash
python -m jarvis_jax.convert.build_checkpoint \
    paths=hyak \
    model=vitpose \
    +convert.out=/path/to/vit_ckpt
```

### Compare 3D Runs (Visualization)

Run from the `third_party/jarvis_jax` directory:

```bash
cd third_party/jarvis_jax && python scripts/viz_compare_3d_runs.py \
    paths=hyak \
    viz.run1=<runA>/final \
    viz.run2=<runB>/final \
    viz.sharpen2=3
```

### Shrinkage Diagnostic

```bash
cd third_party/jarvis_jax && python scripts/diag_shrinkage.py \
    paths=hyak \
    viz.run2=<run>/final
```

### Sharpen Sweep Diagnostic

```bash
cd third_party/jarvis_jax && python scripts/diag_sharpen_sweep.py \
    paths=hyak \
    viz.run2=<run>/final
```

### SLURM Job Submission

Submit a job to preemptible GPU cluster (auto-resumes on preempt via fixed run directory).

**Cached 3D training:**
```bash
python scripts/slurm_train_3d_cached.py \
    --run-name myrun \
    --slurm ckpt_g2 \
    train.total_steps=30000 \
    train.sharpen=3
```

**2D ViTPose training:**
```bash
python scripts/slurm_train_vit.py \
    --run-name vit_run \
    --slurm gpu_l40s \
    train.total_steps=20000
```

**Inline 3D training:**
```bash
python scripts/slurm_train_hybridnet.py \
    --run-name inl_run \
    --slurm ckpt_g2 \
    train.total_steps=20000
```

Preview the job script without submitting:
```bash
python scripts/slurm_train_3d_cached.py --run-name test --dry-run train.total_steps=30000
```

Resume a previous run by using the same `--run-name` (same checkpoint directory).

### Inspect Composed Config

Append `--cfg job` to any `-m` entrypoint to print the fully resolved config without running:

```bash
python -m jarvis_jax.train.train_3d_cached \
    run_id=test \
    train=cached3d \
    paths=hyak \
    --cfg job
```

## Run Outputs

Training runs produce artifacts in the following structure:

```
${paths.runs_root}/${run_id}/
├── final/
│   ├── {loss, metrics, per-step logs}
│   └── [step-N-loss-X checkpoint files]
├── ckpt/
│   └── [orbax checkpoint symlinks for resume]
└── run_config.json                         # Full composed config
```

Run roots depend on the training mode:
- **Cached 3D** (`train=cached3d`): `${paths.runs_root}` (e.g., `jax_cached3d_runs`)
- **2D ViTPose** (`train=vit2d`): `${paths.vit_runs_root}` (e.g., `jax_vitpose_runs`)
- **Inline 3D** (`train=inline3d`): `${paths.hybridnet_runs_root}` (e.g., `jax_hybridnet_runs`)

On Hyak, these expand to `/gscratch/portia/eabe/data/Johnson_lab/jax_*_runs/` by default.

## Exceptions (Still Argparse)

The following scripts run in the `torch` + `timm` environment (NOT `3d_tracking`) and were intentionally NOT converted to Hydra:

- `jarvis_jax/convert/export_mae_timm.py` — export MAE from timm checkpoints
- `jarvis_jax/convert/export_reproject_fixture.py` — test reprojection fixtures
- `scripts/make_ref_features.py` — reference features for shape matching

These remain argparse-based; see their `--help` for usage.

## Resolvers

The configuration uses custom Hydra resolvers for path interpolation. These are automatically registered when importing `jarvis_jax.hydra_utils` and allow references like:

```yaml
cache_dir: /gscratch/portia/${paths.user}/data/...
```

All resolvers are defined in `jarvis_jax/hydra_utils.py` and handle standard path expansions and defaults.
