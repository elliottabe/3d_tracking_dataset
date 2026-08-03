#!/bin/bash
# gpu-l40s wrapper for the one-off fly-sexing canonicalization driver
# (scripts/canonicalize_session_sex.py): sexes every processed bout of a session
# from the wing-song CV and physically canonicalizes male -> fly1, writing a
# per-bout sex.json. Pure numpy / CPU (JAX_PLATFORMS=cpu); the node allocation is
# only to run uninterrupted off the login node. Runs the REAL (mutating) pass,
# then a dry-run re-verify that must report swap=False everywhere (idempotent).
#
# Override the recording via env, e.g.:
#   REC=session1 sbatch third_party/jarvis_jax/scripts/slurm_canonicalize_session_sex.sh
#SBATCH --job-name=sex-canon
#SBATCH --partition=gpu-l40s
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o ./OutFiles/slurm-%A.out

set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export JAX_PLATFORMS=cpu
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset

REC=${REC:-session0}

echo "=== REAL canonicalization pass (recording=$REC) ==="
python -u scripts/canonicalize_session_sex.py recording="$REC"

echo
echo "=== dry-run re-verify (expect swap=False everywhere, idempotent) ==="
python -u scripts/canonicalize_session_sex.py recording="$REC" ++dry_run=true
