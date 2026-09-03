#!/bin/bash
# SAM3 silhouette masks for the LABEL dataset root, as a ckpt-all array.
#
# One array task per SHARD of (recording, camera) groups. Sharding by camera
# directory means no two tasks ever write the same npz, so this is safe to
# preempt and requeue: --overwrite is OFF, so a requeued task skips every npz
# already on disk and continues where it stopped.
#
# Why this exists at all: only 2 of 17 recordings in a general_model-derived
# root carry masks (5.8% of train annotations return a non-empty channel 3),
# because red_data_unified_V3 -- which the old roots borrowed sam3_masks/ from
# -- was deleted. A mask-ON detector arm needs these.
#
# SAM3 env: needs LD_PRELOAD and the cu13 LD_LIBRARY_PATH SET (the opposite of
# every JAX stage here), and compile=False (the script hardcodes it).
#
#SBATCH --job-name=sam3ds
#SBATCH --partition=ckpt-all
#SBATCH --account=portia
#SBATCH --constraint=h200|a100|l40s|l40|a40
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --array=0-15
#SBATCH --output=slurm_logs/sam3ds-%A_%a.out

set -uo pipefail
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset || exit 1

ROOT=${ROOT:-/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902}
NSHARDS=${NSHARDS:-16}

source /mmfs1/gscratch/portia/eabe/miniconda3/etc/profile.d/conda.sh
conda activate 3d_tracking
module load cuda 2>/dev/null

export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"

echo "[shard $SLURM_ARRAY_TASK_ID] $(hostname) $(date) root=$ROOT"
python scripts/data_prep/sam3_dataset_masks.py \
    --root "$ROOT" --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$NSHARDS"
rc=$?
echo "[shard $SLURM_ARRAY_TASK_ID] rc=$rc $(date)"
# PROPAGATE IT -- see job 39501845: with `echo` last, sacct reported
# COMPLETED 0:0 for a task whose python died at import. Across a 16-shard
# array that would report 16 successes and zero masks.
exit $rc
