#!/bin/bash
#SBATCH --job-name=jaxvitpose_v3
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodelist=g[3090-3137]
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=eabe@uw.edu
#SBATCH -o /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v3_8gpu_20260620/slurm-%j.out
#
# Train the JAX ViTPose KeypointDetect model on red_data_unified_V3, 8x L40S,
# single-host data-parallel. ckpt-g2 is preemptible, so this relies on the
# checkpoint auto-resume: --ckpt-dir is a FIXED path and the job is --requeue,
# so a preempt+requeue reruns this script and resume_latest() picks up where it
# left off. Do NOT make CKPT_DIR depend on $SLURM_JOB_ID/date (would break resume).

set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH                       # let JAX use its bundled CUDA wheels
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
nvidia-smi -L
echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID"

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
RUN=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v3_8gpu_20260620
ROOT=/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3

cd "$PKG"
python -u -c "import jax; print('jax devices:', jax.device_count())"
python -u -m jarvis_jax.scripts.train_keypoints \
    --root "$ROOT" \
    --out "$RUN/final" \
    --ckpt-dir "$RUN/ckpt" \
    --steps 20000 \
    --batch 64 \
    --lr 1e-3 \
    --backbone-lr-mult 0.1 \
    --save-every 500
