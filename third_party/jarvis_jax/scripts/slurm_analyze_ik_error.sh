#!/bin/bash
# ckpt-g2 wrapper for the IK error-quantification driver
# (scripts/analyze_ik_error.py): Q1 local IK sensitivity (marker Jacobian +
# noise model + Monte-Carlo check) and Q2 morph-vs-no-morph (per-segment
# shape calibration) fit residual / qpos-delta / implied-bias, for one bout.
#
# Override any knob via env, e.g.:
#   env -u JAX_PLATFORMS BOUT=3 sbatch third_party/jarvis_jax/scripts/slurm_analyze_ik_error.sh
#SBATCH --job-name=ik-error
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --nodelist=g[3090-3137]
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH -o ./OutFiles/slurm-%A.out

module load cuda/12.9.1
set -x
source ~/.bashrc
nvidia-smi
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH

cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
python -u scripts/analyze_ik_error.py recording=session0 \
    ++bout_ids="${BOUT:-1}" \
    ++ik_error.n_frames=80 \
    ++ik_error.mc_n=20
