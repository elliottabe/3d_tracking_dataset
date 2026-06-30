#!/bin/bash
#SBATCH --job-name=cse_vit350b_ft
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodelist=g[3090-3137]
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=eabe@uw.edu
#
# Wings+legs v2 fine-tune: warm-start from cse_vit350_wings/final and re-train with the
# thin densely-sampled structures rendered at a TIGHT Gaussian so their verts form
# resolvable peaks instead of one smeared blob (fixes inboard tip compression):
#   100 wing ch @ sigma 2.5, 4x loss   |   149 leg ch @ sigma 3.0, 2x loss
# 51 body (thorax/abdomen/head) + 50 kp channels keep sigma 7.0 / weight 1.0 (already good).
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
MESH=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz
RUN=$RUNS/cse_vit350b_wings

cd "$PKG"
python -u -c "import jax; print('jax devices:', jax.device_count())"
python -u -m jarvis_jax.cse.train_keypoints_cse_full \
    --root "$ROOT" \
    --aux-train "$WORK/cse_labels_train_M300.npz" \
    --aux-val   "$WORK/cse_labels_val_M300.npz" \
    --v3-ckpt "$RUNS/cse_vit350_wings/final" --warmstart-joints 350 \
    --mesh-npz "$MESH" --wing-sigma 2.5 --wing-weight 4.0 --leg-sigma 3.0 --leg-weight 2.0 \
    --out "$RUN/final" --ckpt-dir "$RUN/ckpt" \
    --num-joints 350 --steps 8000 --batch 24 --lr 5e-4 --backbone-lr-mult 0.1 \
    --save-every 500 --log-every 50 --eval-every 1000
echo "WING FINETUNE DONE"
