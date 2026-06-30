#!/bin/bash
#SBATCH --job-name=cse_cache350
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eabe@uw.edu
#
# Wings v1 / stage 1: build the 350-channel reproject cache (train+val) from the
# ViTPose-350 wings checkpoint + M=300 wing labels.  Single GPU; batch 4 keeps the
# 350-channel ViTPose forward + 48^3 reproject volume under the 40 GB L40S floor
# (matches the predict batch-4 floor found at 350 channels).
set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
nvidia-smi -L

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
ROOT=/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3
WORK=/gscratch/portia/eabe/data/Johnson_lab/cse_work
MESH=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz
VIT=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit350_wings/final
CACHE=$WORK/cache_350

cd "$PKG"
python -u -c "import jax; print('jax devices:', jax.device_count())"
for SPLIT in train val; do
  python -u -m jarvis_jax.cse.build_cache_cse \
    --root "$ROOT" --split $SPLIT --aux "$WORK/cse_labels_${SPLIT}_M300.npz" \
    --vitpose-ckpt "$VIT" --mesh "$MESH" --cache-dir "$CACHE" --num-joints 350 --batch 4
done
echo "CACHE350 DONE"
