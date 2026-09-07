# Portability checklist: everything Hyak-specific

Companion to the "Install and run on a new cluster" section in
`docs/running_the_pipeline.md`. Verified 2026-09-06 against the working
`3d_tracking` conda env and this checkout (branch
`elliottabe/paper_update_062026`, HEAD `e8d8446`). This file only lists what
is cluster-specific and where -- it does not rewrite any of it.

## 1. Slurm partition / account / constraint

These are UW Hyak names (`ckpt-all`, `ckpt-g2`, `gpu-l40s`, account `portia`)
and will not exist on another cluster.

- **Parameterized (edit here for a new cluster)** -- `configs/slurm/*.yaml`:
  `configs/slurm/ckpt_all.yaml`, `configs/slurm/ckpt_g2.yaml`,
  `configs/slurm/gpu_l40s.yaml` (each sets `partition`, `constraint`,
  `account`, `mem`, `time`, `conda_env`). All of `scripts/slurm_bout_array.py`,
  `scripts/slurm_run.py`, `scripts/slurm_dir_array.py`,
  `scripts/slurm_train_vit.py`, `scripts/slurm_predict_session.py`,
  `scripts/slurm_coarse_pass_array.py` read these via Hydra's `slurm` group --
  add a new `configs/slurm/<cluster>.yaml` rather than editing partition
  strings in the `.py` files.
- **Hardcoded directly in sbatch heredocs** (bypass the `slurm` group, must be
  hand-edited):
  - `scripts/slurm_recanonicalize_masks.sh:9-10` (`--partition=ckpt`,
    `--account=ckpt-portia` -- note even the partition name differs from the
    `configs/slurm` ones)
  - `scripts/slurm/submit_task.sh:35-37`
  - `scripts/analysis/stac_ik_memory_probe.sh:12-14`
  - `scripts/analysis/restage_ckpt_array.sh:15-17`
  - `scripts/data_prep/sam3_dataset_masks_array.sh:18-20`
  - `scripts/slurm/mvq_session_pipeline.sh:296-297` -- parameterized via
    `${PARTITION}`/`${ACCOUNT}` shell vars, defaults set earlier in the same
    script (grep `PARTITION=` / `ACCOUNT=` there)
  - `third_party/jarvis_jax/**/*.sh` (~30 files, mostly one-off training
    launchers under `third_party/jarvis_jax/jarvis_jax/tracking/sbatch_*.sh`
    and `third_party/jarvis_jax/submit_*.sh`): all hardcode
    `--partition=ckpt-g2` / `--account=portia` (two use `gpu-l40s`:
    `sbatch_vit350b_l40s.sh`, `sbatch_cse_v2v_350_l40s.sh`,
    `slurm_canonicalize_session_sex.sh`). These are historical one-off
    scripts, not the current entry points (`mvq_session_pipeline.sh`,
    `slurm_bout_array.py`) -- lowest priority to port.
- `scripts/slurm_run.py:224` also hardcodes `GPU_NODELISTS` (a dict of
  partition -> Hyak node names) for `--nodelist`.

GPU sizing assumption baked into `configs/slurm/ckpt_all.yaml`'s comment and
`docs/running_the_pipeline.md`: model fits on l40/l40s/a40 (48GB), a100
(40/80GB), h200 (141GB); excludes rtx6k/2080ti/p100/CPU nodes explicitly via
the constraint string. A new cluster's equivalent GPU-size filter (if its
scheduler supports a similar constraint syntax) needs re-deriving from this
list, not copied verbatim.

## 2. `module load cuda/12.9.1`

Hyak-specific environment-module name/version
(`docs/running_the_pipeline.md`, `docs/benchmark/2026-09-mvq/v2-notes.md`,
CLAUDE.md). Purpose: batch (non-interactive) nodes don't expose `libcuda.so`
to JAX without it -- **any** mechanism that puts a matching driver's
`libcuda.so` on the loader path substitutes on another cluster (their module
name will differ, e.g. `cuda/12.4`, `nvidia/cuda-12.6`, or none if the driver
is already on the default path). Confirm the training/pipeline works with
`unset JAX_PLATFORMS && python -c "import jax; print(jax.devices())"` and
check for `[gpu-check] jax sees N GPU(s)` in job logs (N=0 means this step is
still needed and mismatched).

## 3. Runtime env every GPU job sets

```bash
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
unset LD_LIBRARY_PATH JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 TF_GPU_ALLOCATOR=cuda_malloc_async
```
Not Hyak-specific in principle (any Linux cluster can hit the same libstdc++
ABI mismatch and CPU-fallback traps -- see "Install and run on a new cluster"
step 5 for why), but the exact values (`0.9` fraction, `libstdc++.so.6` path
under `$CONDA_PREFIX`) were tuned/verified only on Hyak L40S nodes and should
be re-checked (OOM vs headroom) on different GPU memory sizes.

## 4. Absolute `/gscratch` / `/mmfs1` paths outside the `paths=` Hydra group

The `paths` group (`configs/paths/*.yaml` and
`third_party/jarvis_jax/configs/paths/*.yaml`, TWO separate groups, see
"Install and run on a new cluster" step 6) is the sanctioned place for a
cluster's data root. These are places that bypass it:

- `scripts/run_full_pipeline.py:36` -- `DATA_ROOT = Path('/gscratch/portia/eabe/data/Johnson_lab')`
- `scripts/slurm_run.py:23` -- same `DATA_ROOT` literal
- `scripts/slurm_run.py:299-300` -- example `CACHE=`/`RUN=` values in a docstring (harmless, but copy-pastable as-is)
- `scripts/slurm_dir_array.py:154,217` -- `export JAX_COMPILATION_CACHE_DIR=/gscratch/portia/eabe/data/Johnson_lab/torch_cache/jax_cache_v2_3`
- `scripts/calibrate_view_gate.py:67,70` -- `--ckpt` / dataset-root argparse defaults
- `scripts/session_collect.py:44,404` -- `--processed` default / fallback
- `third_party/jarvis_jax/scripts/make_ref_features.py:17` -- example invocation naming a `/gscratch/.../envs/jarvis/bin/python` interpreter (a THIRD, older conda env, not `3d_tracking`)
- ~230 more matches across `scripts/*.py` and `third_party/jarvis_jax/**` are
  docstring/help-text example commands (`--session /gscratch/.../Session0/...`)
  rather than defaults baked into code -- copy-paste hazards for a new user,
  not code that needs editing to run elsewhere.

Full grep used: `grep -rn '/gscratch\|/mmfs1' scripts/ configs/ third_party/jarvis_jax/{configs,scripts} | grep -v '^configs/paths/'`.

## 5. Model weights and checkpoints (not in git; must be copied per fresh cluster)

| artifact | path (Hyak) | config key(s) |
|---|---|---|
| SAM3 weights (~6.5 GB) + HF cache | `$HF_HOME` = `/gscratch/portia/eabe/data/Johnson_lab/sam3` | env var `HF_HOME`; `sam3.jarvis_root` for the JARVIS project dir |
| SAM3 code | separate clone, `facebookresearch/sam3`, pip-installed editable from `/gscratch/portia/eabe/Research/Github/sam3` -- **not vendored in this repo, not documented anywhere else in this repo's docs before this file** | none (import `sam3`) |
| HF token | `$HF_HOME/token` (825 bytes, present) -- DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`, gated) needs Hugging Face access requested for the account that generated this token | `huggingface_hub` reads it automatically when `HF_TOKEN` is unset/empty |
| ViTPose 2D detector checkpoint (326 MB) | `jax_vitpose_runs/v4_8gpu_20260808/final` | `paths.vit_runs_root` (main repo, both trees) + `vitpose_ckpt` (jarvis_jax `paths` group only) |
| mvq p3a checkpoint | `jax_mvq_runs/mvq_t1_b16_p3a_20260904/final` | `configs/mvq/p3a.yaml:17` (`checkpoint:`), literal, not built from `paths.mvq_runs_root` |
| mvq p3b checkpoint | `jax_mvq_runs/mvq_t1_b16_p3b_contact_20260905/final` | `configs/mvq/p3b.yaml:24`, same pattern |
| mvq v2 checkpoint (being trained) | `jax_mvq_runs/mvq_t2_v2_20260905/final` (per `run_id` at launch) | `third_party/jarvis_jax/configs/train/mvq_v2.yaml` interpolates `paths.mvq_runs_root` |
| DINOv3 ViT-B/16 backbone | pulled from HF hub at first run, cached under `$HF_HOME` | `third_party/jarvis_jax/configs/model/mvq.yaml:27` (`backbone: dinov3_b16` -> `facebook/dinov3-vitb16-pretrain-lvd1689m` in `jarvis_jax/models/dinov3.py:22`) |

`configs/mvq/p3a.yaml` and `p3b.yaml` hold literal Hyak paths rather than
building off `paths.mvq_runs_root` -- worth fixing to an interpolation at
some point, noted here rather than changed (checklist only).

## 6. Body models and code trees outside git

- `models/` in this repo is intentionally empty. The real body model is the
  **sibling** clone `fruitfly_body_models` (pinned branch
  `elliottabe/3d_tracking_paper`) at `${paths.project_dir}/fruitfly_body_models`
  -- both `configs/paths/hyak.yaml` and `third_party/jarvis_jax/configs/paths/hyak.yaml`
  set `body_model_dir` there. Confirmed present, correct branch, HEAD
  `7d395f7`.
- `anatomy=v2_3` needs the derived, gitignored `fruitfly_v2_3_ik/` built from
  it by `scripts/models/build_v2_3_ik_model.py` -- **run once per fresh
  checkout** (idempotent) or `ParseXML` fails.
- Submodules (`.gitmodules`): `stac-mjx` and `third_party/JARVIS-HybridNet`,
  both pinned to branch `elliottabe/paper_update_062026`. `git submodule
  status` shows both initialized and checked out at the pinned commits.
  `README.md` and `SUBMODULES.md` currently mention **only** `stac-mjx` --
  wrong/stale, fixed in this pass.
- `third_party/jarvis_jax` (274 MB) is vendored in-tree (NOT a submodule) and
  must be `pip install -e third_party/jarvis_jax` separately -- it is not
  listed in `requirements.txt`'s `-e ./stac-mjx` line, and no doc anywhere
  said to install it before this pass; confirmed it IS pip-editable-installed
  in the working env (`pip show jarvis_jax` -> editable at
  `third_party/jarvis_jax`).

## 7. Data layout assumed by config

- Raw: `Video_recordings/<assay>/<Session>/<rec>/{Cam*.mp4, calibration/Cam*.yaml, Cam*_meta.csv, <dataset>_bout_summary.csv}`
- Processed: `processed/<assay>/<Session>/<rec>/{sam3_masks/, pose/}` (canonical tree; also `pose_mvq_*` run-name variants for the mvq route)
- v12 human label root: `paths.red_data_root` (main) /
  `third_party/jarvis_jax/configs/paths/hyak.yaml`'s `data_root` ->
  `red_data/red_data_3d_v12_export0902`
- mvq v2's three pseudo-label roots (all under `/gscratch/portia/eabe/data/Johnson_lab/`,
  hardcoded literals in `third_party/jarvis_jax/configs/train/mvq_v2.yaml`,
  not built from a `paths.*` key):
  `red_data_3d_v12_pseudo_p3b_20260905`,
  `red_data_3d_v12_pseudo_singlefly_20260905`,
  `red_data_3d_v12_pseudo_negatives_20260905`.

## 8. `paths=` Hydra group

Two SEPARATE groups exist (main repo `configs/paths/`, vendored
`third_party/jarvis_jax/configs/paths/`) -- a new cluster needs a new file in
**both**. `paths=hyak` sets, in the main repo: `base_dir` (single Johnson_lab
data root everything else builds from), `data_dir`, `run_root`/`save_dir`
(hydra run dirs), `out_root`, `processed_root`, `vit_runs_root`,
`red_data_root`, `cwd_dir`, `project_dir`, `body_model_dir`. The
`jarvis_jax` copy sets an overlapping but not identical key set (`data_root`,
`runs_root`, `vit_runs_root`, `processed_root`, `vitpose_ckpt`, `mae_npz`,
`body_model_dir`, `project_dir`, `general_model`, `mvq_runs_root`). Existing
non-Hyak examples to copy the shape from: `configs/paths/workstation.yaml`,
`configs/paths/desktop.yaml`, `configs/paths/mbook.yaml` and
`third_party/jarvis_jax/configs/paths/workstation.yaml` -- none of the four
non-Hyak variants is verified working end-to-end, they're just templates.

## Not verified (could not check without guessing)

- Whether another cluster's Slurm build supports the same `--constraint
  'h200|a100|l40s|l40|a40'` OR-set syntax (this is a Slurm feature, not
  Hyak-specific, but untested elsewhere here).
- Whether `mujoco-warp`/`warp-lang` 1.16.0 and `torch` 2.11.0+cu130 have
  working wheels for a non-cu13-capable driver -- only the Hyak node's driver
  was checked (`torch.version.cuda == 13.0`).
- Actual runtime/memory behavior on GPUs other than L40S/A40/A100/H200 (the
  `mem: 64` / `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` values are tuned against
  Hyak's node RAM and this GPU family only).
