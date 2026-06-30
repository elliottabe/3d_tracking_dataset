#!/bin/bash
#SBATCH --job-name=cse_3d
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
# C3: build 250-channel reproject cache (train+val) from ViTPose-250, then train V2VNet-250.
# Requires the ViTPose-250 checkpoint (cse_vit250_full/final) to exist.
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
MESH=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical.npz
VIT=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit250_full/final
CACHE=$WORK/cache_250
RUN=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_v2v250

cd "$PKG"
for SPLIT in train val; do
  python -u -m jarvis_jax.cse.build_cache_cse \
    --root "$ROOT" --split $SPLIT --aux "$WORK/cse_labels_${SPLIT}_M200.npz" \
    --vitpose-ckpt "$VIT" --mesh "$MESH" --cache-dir "$CACHE" --num-joints 250 --batch 8
done

python -u -m jarvis_jax.cse.train_v2v_cse \
  --cache-dir "$CACHE" --out "$RUN/final" --ckpt-dir "$RUN/ckpt" \
  --steps 20000 --batch 16 --laplacian-weight 0.05 --sharpen 3.0
echo "C3 DONE"
