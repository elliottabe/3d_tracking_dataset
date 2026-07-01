#!/bin/bash
#SBATCH --job-name=viz350b_cur
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eabe@uw.edu
#
# Visualize the CURRENT mid-training vit350b checkpoint (latest committed step, ~7000)
# vs GT, with the wing+leg spread/err breakdown.
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
ROOT=/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3

cd "$PKG"
python -u -m jarvis_jax.cse.viz_predictions \
    --root "$ROOT" --aux "$WORK/cse_labels_val_M300.npz" \
    --ckpt-dir "$RUNS/cse_vit350b_wings/ckpt" --mesh "$MESH" \
    --num-joints 350 --n-kp 50 --n 8 --show-gt \
    --out "$WORK/viz/vit350b_current.png"
echo "VIZ VIT350b CUR DONE"
