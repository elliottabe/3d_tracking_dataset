#!/bin/bash
#SBATCH --job-name=cse_vit250_pilot
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eabe@uw.edu
#
# Design-1 pilot: ViTPose with 250 outputs (50 kp + 200 vertices), single GPU.
set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
nvidia-smi -L

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
ROOT=/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3
WORK=/gscratch/portia/eabe/data/Johnson_lab/cse_work
RUN=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit250_pilot

cd "$PKG"
python -u -m jarvis_jax.cse.train_keypoints_cse \
    --root "$ROOT" \
    --aux-train "$WORK/cse_labels_train_M200.npz" \
    --aux-val   "$WORK/cse_labels_val_M200.npz" \
    --out "$RUN/final" --ckpt-dir "$RUN/ckpt" \
    --num-joints 250 --steps 2000 --batch 4 --lr 3e-4 \
    --log-every 50 --eval-every 500 --save-every 1000
echo "PILOT DONE"
