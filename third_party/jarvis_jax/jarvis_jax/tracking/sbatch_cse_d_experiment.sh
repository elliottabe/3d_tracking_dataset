#!/bin/bash
#SBATCH --job-name=cse_d_robust
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eabe@uw.edu
#
# Workstream D robustness experiment: STAC IK with kp-only vs kp+200 GT vertices,
# under simulated keypoint occlusion. Independent of the trained model (uses GT
# vertices = upper-bound on the IK-robustness gain).
set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
nvidia-smi -L

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
STAC_CFG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs
MESH=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical.npz
WORK=/gscratch/portia/eabe/data/Johnson_lab/cse_work
REC=2026_03_18_15_31_22
OUT=$WORK/d_experiment/$REC

cd "$PKG"
for OCC in 0.0 0.3 0.5; do
  for MODE in kp dense; do
    python -u -m jarvis_jax.densepose.stac_dense \
      --bout "$WORK/${REC}_bout.h5" --labels "$WORK/$REC/cse_labels_M200.npz" \
      --mesh "$MESH" --stac-config-dir "$STAC_CFG" --out-dir "$OUT" \
      --mode $MODE --M 200 --occlude-frac $OCC
  done
done
echo "D EXPERIMENT DONE"
