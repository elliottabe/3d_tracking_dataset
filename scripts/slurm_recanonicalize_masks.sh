#!/bin/bash
# CPU job for scripts/recanonicalize_masks.py: re-sex existing SAM3 mask npz files
# so male=fly1 (multi-view area-vote) IN PLACE, before a pose run reuses them.
# Heavy I/O (each packed ~1 GB) -> must run on a compute node, NOT the login node
# (a login-node run gets OOM-killed by the per-user cgroup cap). Idempotent.
#
#   SESSION=<session_dir> sbatch scripts/slurm_recanonicalize_masks.sh
#SBATCH --job-name=recanon
#SBATCH --partition=compute
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o ./OutFiles/slurm-%A.out

set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export JAX_PLATFORMS=cpu
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
SESSION=${SESSION:-/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1}
python -u scripts/recanonicalize_masks.py --session "$SESSION"
