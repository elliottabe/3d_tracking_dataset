#!/bin/bash
#SBATCH --job-name=cse_vit250_full
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
# Design-1 FULL: ViTPose-250 (50 kp + 200 verts), v3 warm-start, 8xL40S DP.
# Preemptible: fixed CKPT dir + --requeue -> restore_latest auto-resumes.
set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
nvidia-smi -L

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
ROOT=/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3
WORK=/gscratch/portia/eabe/data/Johnson_lab/cse_work
RUN=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit250_full
V3=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v3_8gpu_20260620/final

cd "$PKG"
python -u -c "import jax; print('jax devices:', jax.device_count())"
python -u -m jarvis_jax.cse.train_keypoints_cse_full \
    --root "$ROOT" \
    --aux-train "$WORK/cse_labels_train_M200.npz" \
    --aux-val   "$WORK/cse_labels_val_M200.npz" \
    --v3-ckpt "$V3" \
    --out "$RUN/final" --ckpt-dir "$RUN/ckpt" \
    --num-joints 250 --steps 20000 --batch 32 --lr 1e-3 --backbone-lr-mult 0.1 \
    --save-every 500 --log-every 50 --eval-every 1000
echo "FULL DONE"
