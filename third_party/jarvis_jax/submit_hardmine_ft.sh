#!/bin/bash
#SBATCH --job-name=et2d_hardmine
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
#SBATCH -o /gscratch/portia/eabe/data/Johnson_lab/jax_efficienttrack_runs/et2d_bn_hardmine_ft/slurm-%j.out
#
# Error-based hard-example fine-tune of the ImageNet EfficientTrack-BN detector.
# Warm-start from et2d_bn_imagenet/final; sample each train annotation by its own
# per-frame error (mined with that detector), capped so the ambiguous two-fly
# courtship-contact tail (err>~15px, likely partial mislabels) doesn't dominate;
# heaviest augmentation to generalize the repeated hard frames; low LR, short.
set -ex
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
nvidia-smi -L; echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID"

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
cd "$PKG"
ET=/gscratch/portia/eabe/data/Johnson_lab/jax_efficienttrack_runs
python -u -m jarvis_jax.scripts.train_keypoints \
    run_id=et2d_bn_hardmine_ft \
    train=vit2d model=efficienttrack_bn paths=hyak \
    paths.runs_root=$ET \
    train.warm_start=$ET/et2d_bn_imagenet/final \
    sampling=hard_error \
    sampling.weights_file=$ET/et2d_bn_imagenet_train_errors.npz \
    sampling.error_alpha=4 sampling.error_cap=8 \
    aug=heavy \
    train.lr=3e-5 train.total_steps=8000
