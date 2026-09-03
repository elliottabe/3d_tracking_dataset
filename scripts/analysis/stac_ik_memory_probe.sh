#!/bin/bash
# Peak-memory / utilisation probe for ONE STAC IK bout-fly solve, on ckpt-all.
#
# Constraint set is the pipeline's usual h200|a100|l40s|l40|a40, which spans
# 40-141 GB of VRAM and excludes the cards that cannot hold the solve
# (rtx6k 24 GB, p100 16 GB, 2080ti 11 GB). Because the answer -- how many
# concurrent solves fit -- depends on which card we land on, the probe reports
# ABSOLUTE bytes plus the device's own bytes_limit, so the concurrency figure
# can be recomputed for any card without re-running.
#
#SBATCH --job-name=stacmem
#SBATCH --partition=ckpt-all
#SBATCH --account=portia
#SBATCH --constraint=h200|a100|l40s|l40|a40
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --output=slurm_logs/stacmem-%j.out

set -uo pipefail
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset || exit 1

source /mmfs1/gscratch/portia/eabe/miniconda3/etc/profile.d/conda.sh
conda activate 3d_tracking
module load cuda 2>/dev/null

export MUJOCO_GL=egl
# NOT the usual 0.7 mem fraction: preallocation would mask the real high-water
# mark, which is the only number this job exists to produce.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH
unset JAX_PLATFORMS

echo "[stacmem] $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/analysis/stac_ik_memory_probe.py \
    --fly "${FLY:-1}" --frames ${FRAMES:-2007 1000 500} \
    --out "OutFiles/stac_ik_memory_${SLURM_JOB_ID}.json"
rc=$?
echo "[stacmem] rc=$rc $(date)"
# PROPAGATE IT. `echo` was the last command, so its 0 became the job's exit
# status and sacct reported COMPLETED 0:0 for a run whose python died at
# import (job 39501845). A failed job that reports success is worse than a
# failed job.
exit $rc
