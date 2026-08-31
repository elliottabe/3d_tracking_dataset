#!/bin/bash
# Stage B onward re-run for the courtship recordings, as a ckpt-all array.
#
# One array task per RECORDING, not per bout: offsets.h5 / scale.json are
# per-recording artifacts fit by whichever bout runs first and reused by the
# rest, so two tasks on the same recording would race on them.
#
# Resumable by construction -- run_bout skips any bout-fly that still has its
# DONE marker -- which is what makes this safe on a preemptible partition.
# --requeue means a preempted task restarts and simply continues where it left
# off, and it is also why the local 4-way run could be stopped mid-flight and
# handed over here without losing the 63 bout-flies already finished.
#
#SBATCH --job-name=restageB
#SBATCH --partition=ckpt-all
#SBATCH --account=portia
#SBATCH --constraint=h200|a100|l40s|l40|a40
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --array=0-10
#SBATCH --output=slurm_logs/restageB-%A_%a.out

set -uo pipefail
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset || exit 1

VID=/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship
RECS=(
  "session0|$VID/Session0/2025_10_20_13_20_04"
  "session1|$VID/Session1/2026_04_02_12_11_50"
  "session1|$VID/Session1/2026_04_02_14_54_28"
  "session1|$VID/Session1/2026_04_02_15_25_51"
  "session1|$VID/Session1/2026_04_02_15_44_42"
  "session1|$VID/Session1/2026_04_02_16_03_48"
  "session1|$VID/Session1/2026_04_02_16_21_32"
  "session1|$VID/Session1/2026_04_02_16_39_56"
  "session1|$VID/Session1/2026_04_02_16_56_37"
  "session1|$VID/Session1/2026_04_02_17_28_34"
  "session1|$VID/Session1/2026_04_02_17_52_50"
)
entry="${RECS[$SLURM_ARRAY_TASK_ID]}"
cfg="${entry%%|*}"; dir="${entry##*|}"

source /mmfs1/gscratch/portia/eabe/miniconda3/etc/profile.d/conda.sh
conda activate 3d_tracking
module load cuda 2>/dev/null

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export LD_PRELOAD=/mmfs1/gscratch/portia/eabe/miniconda3/envs/3d_tracking/lib/libstdc++.so.6
unset LD_LIBRARY_PATH
unset JAX_PLATFORMS

echo "[task $SLURM_ARRAY_TASK_ID] $(basename "$dir") on $(hostname) $(date)"
python scripts/run_bout.py recording="$cfg" recording.session_dir="$dir"
echo "[task $SLURM_ARRAY_TASK_ID] rc=$? $(date)"
