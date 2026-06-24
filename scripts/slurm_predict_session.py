#!/usr/bin/env python3
"""
Submit ONE SLURM job that runs the full JAX courtship-prediction pipeline for a
session: D2 SAM3 masks (all bouts) -> D3 JAX 3D prediction (all bouts), producing
per-fly data3D_fly{0,1}.csv.

Companion to scripts/slurm_train_3d_cached.py (same launcher pattern). The job body
has TWO sequential stages on one multi-GPU node:
  1. SAM3 masks  (scripts/sam3_masks.py) — PyTorch; reuse_masks=true skips bouts
     whose sam3_masks.npz already exists, so a requeue resumes cheaply.
  2. JAX predict (scripts/predict_session.py) — reads stage-1 masks, runs the
     v2vNet+ViTPose 3D pipeline, writes Predictions_3D-style CSVs.

--run-name selects the TRAINED v2vNet run to predict with (run_id): the predict
entrypoint loads ${paths.runs_root}/${run_id}/final. Output + masks live under
--out (default ${paths.runs_root}/predict_session/<run-name>); masks go in
<out>/sam3_masks and are passed to both stages so stage 2 reads stage 1's output.

Per-stage CUDA libs: SAM3 (PyTorch) needs the cu13 wheels on LD_LIBRARY_PATH;
JAX needs its own bundled CUDA, so LD_LIBRARY_PATH is unset before stage 2
(exactly as slurm_train_3d_cached.py does for the trainer).

Preemption resume
-----------------
Single-host job (JAX sees one node's GPUs). ckpt* partitions are preemptible, so
the job is submitted --requeue with a FIXED out dir baked in at submit time. On
preempt+requeue the same script reruns: stage 1 skips finished bouts
(reuse_masks), stage 2 re-runs idempotently (overwrites CSVs). Re-submitting with
the same --run-name/--out resumes; a fresh --out starts clean.

Usage:
    # Predict all of session 0 with the run4 v2vNet on ckpt-g2 (8 GPU, requeue):
    python scripts/slurm_predict_session.py --run-name run4

    # A different session / output dir, pass Hydra overrides to both stages:
    python scripts/slurm_predict_session.py --run-name run4 \\
        --session-dir /path/to/SessionX/vid --out /path/to/out \\
        predict_session.batch=8

    # See the script without submitting:
    python scripts/slurm_predict_session.py --run-name run4 --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"


def build_script(
    *,
    job_name: str,
    partition: str,
    account: str,
    nodelist_line: str,
    exclude_line: str,
    requeue_line: str,
    gpus: int,
    cpus: int,
    mem: int,
    time_limit: str,
    conda_env: str,
    mail_user: str,
    pkg_dir: Path,
    log_dir: str,
    run_name: str,
    paths: str,
    session_dir: str,
    masks_dir: str,
    out: str,
    overrides_str: str,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus={gpus}
#SBATCH --mem={mem}G
#SBATCH --open-mode=append
#SBATCH -o {log_dir}/slurm-%j.out
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user={mail_user}
{nodelist_line}
{exclude_line}
{requeue_line}
set -x
source ~/.bashrc
micromamba activate {conda_env}
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"   # cv2 (both stages)
echo "Node: $SLURMD_NODENAME  job: $SLURM_JOB_ID"
nvidia-smi -L
cd {pkg_dir}
# --- Stage 1: SAM3 masks (PyTorch; needs cu13 CUDA libs) ---
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
python -u scripts/sam3_masks.py paths={paths} sam3=default \\
    sam3.session_dir={session_dir} sam3.out={masks_dir} sam3.sam3_compile=false{overrides_str}
# --- Stage 2: JAX 3D predict (needs JAX's bundled CUDA -> unset LD_LIBRARY_PATH) ---
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
python -u scripts/predict_session.py paths={paths} model=hybridnet predict_session=default \\
    run_id={run_name} predict_session.session_dir={session_dir} \\
    predict_session.masks_dir={masks_dir} predict_session.out={out}{overrides_str}
"""
