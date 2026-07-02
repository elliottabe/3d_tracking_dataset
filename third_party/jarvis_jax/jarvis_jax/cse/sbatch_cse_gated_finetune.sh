#!/bin/bash
#SBATCH --job-name=cse_vit350_gated
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
#SBATCH -o /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit350_gated/slurm-%j.out
#
# Phase-5 GATED fine-tune: warm-start cse_vit350_wings/final (350 = 50 kp + 300 verts)
# and continue training with the dilated-mask containment loss ON (mask-weight>0,
# mask-dilate=11) AND flip augmentation ON (dense (50+M) L/R swap). ckpt-g2 is
# preemptible -> FIXED --ckpt-dir + --requeue => auto-resume.
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
RUN=$RUNS/cse_vit350_gated
mkdir -p "$RUN"

cd "$PKG"
python -u -c "import jax; print('jax devices:', jax.device_count())"
python -u -m jarvis_jax.cse.train_keypoints_cse_full \
    --root "$ROOT" \
    --aux-train "$WORK/cse_labels_train_M300.npz" \
    --aux-val   "$WORK/cse_labels_val_M300.npz" \
    --v3-ckpt "$RUNS/cse_vit350_wings/final" --warmstart-joints 350 \
    --mesh-npz "$MESH" \
    --mask-weight 0.1 --mask-dilate 11 --flip-p 0.5 \
    --out "$RUN/final" --ckpt-dir "$RUN/ckpt" \
    --num-joints 350 --steps 6000 --batch 24 --lr 5e-4 --backbone-lr-mult 0.1 \
    --save-every 500 --log-every 50 --eval-every 1000
echo "GATED FINETUNE DONE"
