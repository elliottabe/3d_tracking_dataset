#!/bin/bash
#SBATCH --job-name=validate350b
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eabe@uw.edu
#
# Before/after 2-D spread validation: run the GT-vs-pred viz on v1 (vit350) and the
# wings+legs fine-tune (vit350b). Reports body/leg/wing err + wing & leg spread ratio.
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
echo "===== V1 (vit350, sigma7 baseline) ====="
python -u -m jarvis_jax.tracking.viz_predictions \
    --root "$ROOT" --aux "$WORK/cse_labels_val_M300.npz" \
    --ckpt "$RUNS/cse_vit350_wings/final" --mesh "$MESH" \
    --num-joints 350 --n-kp 50 --n 8 --show-gt --out "$WORK/viz/vit350_val_gt.png"
echo "===== V2 (vit350b, wings sigma2.5/x4 + legs sigma3.0/x2) ====="
python -u -m jarvis_jax.tracking.viz_predictions \
    --root "$ROOT" --aux "$WORK/cse_labels_val_M300.npz" \
    --ckpt "$RUNS/cse_vit350b_wings/final" --mesh "$MESH" \
    --num-joints 350 --n-kp 50 --n 8 --show-gt --out "$WORK/viz/vit350b_val_gt.png"
echo "VALIDATE350b DONE"
