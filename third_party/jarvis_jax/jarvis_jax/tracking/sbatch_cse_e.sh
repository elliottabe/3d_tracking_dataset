#!/bin/bash
#SBATCH --job-name=cse_e_eval
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eabe@uw.edu
#
# Workstream E: full-dataset dense-pose predictions + accuracy/multi-view eval +
# REAL-WORLD (predicted-vertex) IK-robustness. Chained afterok the V2VNet-250 job.
set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
nvidia-smi -L

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
STAC_CFG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs
ROOT=/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3
MESH=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical.npz
WORK=/gscratch/portia/eabe/data/Johnson_lab/cse_work
VIT=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit250_full/final
V2V=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_v2v250/final
PRED=$WORK/predictions
REC=2026_03_18_15_31_22
DPRED=$WORK/d_experiment_pred/$REC

cd "$PKG"
# 0. Promote the early-stopped best checkpoint (step 5000, val 1.710) to <run>/final
python -u -m jarvis_jax.tracking.promote_ckpt \
    --ckpt-dir /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_v2v250/ckpt \
    --step 5000 --out "$V2V" --num-joints 250

common="--vitpose-ckpt $VIT --v2v-ckpt $V2V --num-joints 250 --batch 4 --sharpen 3.0"

# 1. Full-dataset dense-pose predictions (the deployment output)
python -u -m jarvis_jax.densepose.predict_full --root "$ROOT" --split train --aux "$WORK/cse_labels_train_M200.npz" --out "$PRED/pred_train.npz" $common
python -u -m jarvis_jax.densepose.predict_full --root "$ROOT" --split val   --aux "$WORK/cse_labels_val_M200.npz"   --out "$PRED/pred_val.npz"   $common

# 2. E metrics: 3-D accuracy + multi-view consistency (val)
python -u -m jarvis_jax.densepose.eval_e --pred "$PRED/pred_val.npz"

# 3. Real-world IK robustness: predict the robustness recording, then kp vs dense(PREDICTED)
python -u -m jarvis_jax.densepose.predict_full --root "$ROOT" --split val --aux "$WORK/cse_labels_val_M200.npz" \
    --recordings $REC --out "$PRED/pred_${REC}.npz" $common
for OCC in 0.0 0.3 0.5; do
  python -u -m jarvis_jax.densepose.stac_dense --bout "$WORK/${REC}_bout.h5" --labels "$WORK/$REC/cse_labels_M200.npz" \
    --mesh "$MESH" --stac-config-dir "$STAC_CFG" --out-dir "$DPRED" --mode kp --M 200 --occlude-frac $OCC
  python -u -m jarvis_jax.densepose.stac_dense --bout "$WORK/${REC}_bout.h5" --labels "$WORK/$REC/cse_labels_M200.npz" \
    --mesh "$MESH" --stac-config-dir "$STAC_CFG" --out-dir "$DPRED" --mode dense --M 200 --occlude-frac $OCC \
    --pred-npz "$PRED/pred_${REC}.npz"
done
echo "==== REAL-WORLD (predicted-vertex) IK robustness ===="
python -u -m jarvis_jax.tracking.analyze_d --dir "$DPRED"
echo "E DONE"
