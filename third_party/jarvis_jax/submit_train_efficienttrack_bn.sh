#!/bin/bash
#SBATCH --job-name=etbn_ft
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
#SBATCH -o /gscratch/portia/eabe/data/Johnson_lab/jax_efficienttrack_runs/et2d_bn_imagenet/slurm-%j.out
#
# FINETUNE the ImageNet-pretrained standard EfficientNet-b3 (BatchNorm) EfficientTrack
# on red_data_unified_V3, mask-aware (4-channel), for the fair ViT-vs-EfficientNet
# comparison. Backbone starts from ImageNet weights (loaded in the model=efficienttrack_bn
# construction); vit2d's default backbone_lr_mult=0.1 finetunes the pretrained backbone
# slowly while the fresh BiFPN/head train at full LR -- the SAME recipe as ViTPose
# (jax_vitpose_runs/v4_kp_maskaware_fc). Do NOT override backbone_lr_mult here.
#
# --requeue + fixed run_id => resume from ckpt on preempt/time-limit.

set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
nvidia-smi -L
echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID"

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
cd "$PKG"
python -u -c "import jax; print('jax devices:', jax.devices())"
python -u -m jarvis_jax.scripts.train_keypoints \
    run_id=et2d_bn_imagenet \
    train=vit2d \
    model=efficienttrack_bn \
    paths=hyak \
    paths.runs_root=/gscratch/portia/eabe/data/Johnson_lab/jax_efficienttrack_runs
