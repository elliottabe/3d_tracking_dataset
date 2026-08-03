#!/bin/bash
#SBATCH --job-name=et2d_maskaware
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /gscratch/portia/eabe/data/Johnson_lab/jax_efficienttrack_runs/et2d_v1_maskaware/slurm-%j.out
#
# Train the JAX EfficientTrack (EfficientNet-b3 + BiFPN) 2D keypoint detector on
# red_data_unified_V3, mask-aware (4-channel RGB+SAM3), for the ViT-vs-EfficientNet
# comparison (counterpart to the ViTPose run at jax_vitpose_runs/v4_kp_maskaware_fc).
#
# Fairness note: EfficientTrack is RANDOM-INIT here (no ImageNet weights ported to
# JAX), so backbone_lr_mult=1.0 (train the backbone at full LR) — unlike ViTPose,
# whose MAE-pretrained backbone uses 0.1. Same data, same 4-channel mask-aware input.
#
# ckpt-g2 is preemptible: --requeue + a FIXED run_id (=> fixed ckpt dir) let a
# preempt+requeue resume from the latest checkpoint. Do NOT tie run_id to $SLURM_JOB_ID.

set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH                       # JAX uses its bundled CUDA wheels
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
nvidia-smi -L
echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID"

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
cd "$PKG"
python -u -c "import jax; print('jax devices:', jax.devices())"
python -u -m jarvis_jax.scripts.train_keypoints \
    run_id=et2d_v1_maskaware \
    train=vit2d \
    model=efficienttrack \
    paths=hyak \
    train.backbone_lr_mult=1.0 \
    paths.runs_root=/gscratch/portia/eabe/data/Johnson_lab/jax_efficienttrack_runs
