#!/bin/bash
#SBATCH --job-name=sharpen_sweep
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eabe@uw.edu
#
# Sweep the soft-argmax sharpen exponent on the CURRENT V2VNet-350 ckpt to see how
# much of the 3-D wing under-spread is decode center-bias (free to fix) vs model.
set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
nvidia-smi -L

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
RUNS=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs
WORK=/gscratch/portia/eabe/data/Johnson_lab/cse_work
MESH=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz

cd "$PKG"
for S in 1 3 6 10 16 24; do
  python -u -m jarvis_jax.densepose.viz_3d_cache \
    --cache-dir "$WORK/cache_350" --ckpt-dir "$RUNS/cse_v2v350/ckpt" \
    --mesh "$MESH" --num-joints 350 --n-kp 50 --n 6 --sharpen $S \
    --out "$WORK/viz/v2v350_3d_s${S}.png" 2>/dev/null | grep -E "restored|wrote"
done
echo "SHARPEN SWEEP DONE"
