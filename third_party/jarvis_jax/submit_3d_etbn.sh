#!/bin/bash
#SBATCH --job-name=v2v_etbn
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/v2v_etbn/slurm-%j.out
#
# 3D arm B: ImageNet EfficientTrack-BN front-end. Build reprojected-volume cache
# (train+val) with the et2d_bn_imagenet detector (normalize_frontend=True), then
# train a V2VNet on it. Prints val 3D MPJPE. Same V2VNet recipe as the ViTPose arm.
set -ex
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
nvidia-smi -L; echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID"

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
cd "$PKG"
ETBN=/gscratch/portia/eabe/data/Johnson_lab/jax_efficienttrack_runs/et2d_bn_imagenet/final
CACHE=/gscratch/portia/eabe/data/Johnson_lab/jax_repro_cache/v3_etbn

for SPLIT in train val; do
  python -u scripts/precompute_repro_cache.py paths=hyak cache=default \
    cache.front_end=efficienttrack_bn \
    cache.frontend_ckpt="$ETBN" \
    paths.cache_dir="$CACHE" \
    cache.split=$SPLIT
done

python -u -m jarvis_jax.train.train_3d_cached \
  run_id=v2v_etbn train=cached3d \
  train.total_steps=20000 train.sharpen=3 train.laplacian_weight=0.0 \
  paths=hyak paths.cache_dir="$CACHE"
