#!/bin/bash
#SBATCH --job-name=vit_hardmine
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/vitpose_hardmine_ft/slurm-%j.out
#
# Error-based hard-example fine-tune of the ViTPose detector (symmetric to the
# EfficientTrack-BN arm). Warm-start from v4_kp_maskaware_fc/final; sample each
# train annotation by ViTPose's OWN per-frame error (capped); heavy aug; low LR.
# vit2d's backbone_lr_mult=0.1 finetunes the pretrained ViT-B backbone slowly.
set -ex
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
nvidia-smi -L; echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID"

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
cd "$PKG"
VITR=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs
python -u -m jarvis_jax.scripts.train_keypoints \
    run_id=vitpose_hardmine_ft \
    train=vit2d model=vitpose paths=hyak \
    paths.runs_root=$VITR \
    train.warm_start=$VITR/v4_kp_maskaware_fc/final \
    sampling=hard_error \
    sampling.weights_file=$VITR/v4_kp_maskaware_fc_train_errors.npz \
    sampling.error_alpha=4 sampling.error_cap=8 \
    aug=heavy \
    train.batch_size=8 train.lr=3e-5 train.total_steps=8000
