#!/bin/bash
#SBATCH --job-name=cse_vit350_wings
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodelist=g[3090-3137]
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=eabe@uw.edu
#
# Wings v1: ViTPose-350 (50 kp + 300 verts = 200 body/leg + 100 wing), 8xL40S.
# Warm-started from the body+leg cse_vit250 model: copies the 50 kp + 200 body/leg
# channels (first 250), only the 100 wing channels (250:350) train fresh.
set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
nvidia-smi -L

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
WORK=/gscratch/portia/eabe/data/Johnson_lab/cse_work
RUNS=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs
ROOT=/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3
RUN=$RUNS/cse_vit350_wings

cd "$PKG"
python -u -c "import jax; print('jax devices:', jax.device_count())"
python -u -m jarvis_jax.cse.train_keypoints_cse_full \
    --root "$ROOT" \
    --aux-train "$WORK/cse_labels_train_M300.npz" \
    --aux-val   "$WORK/cse_labels_val_M300.npz" \
    --v3-ckpt "$RUNS/cse_vit250_full/final" --warmstart-joints 250 \
    --out "$RUN/final" --ckpt-dir "$RUN/ckpt" \
    --num-joints 350 --steps 12000 --batch 24 --lr 1e-3 --backbone-lr-mult 0.1 \
    --save-every 500 --log-every 50 --eval-every 1000
echo "WINGS VIT DONE"
