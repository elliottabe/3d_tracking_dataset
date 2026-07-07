#!/bin/bash
#SBATCH --job-name=cse_stac_pilot
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eabe@uw.edu
#
# CSE auto-label pilot: STAC-fit one V3 recording's bout, then project the M
# canonical vertices into every camera -> aux label npz.
#   sbatch sbatch_cse_pilot.sh <RECORDING>
# Bout must already exist at $WORK/<REC>_bout.h5 (built on the login node).

set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH                       # JAX uses its bundled CUDA wheels
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
nvidia-smi -L

REC=${1:?usage: sbatch sbatch_cse_pilot.sh <RECORDING>}
PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
STAC_CFG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs
ROOT=/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3
XML=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml
MESH=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical.npz
WORK=/gscratch/portia/eabe/data/Johnson_lab/cse_work

cd "$PKG"
python -u -c "import jax; print('jax devices:', jax.device_count())"

# Stage 2: STAC IK (GPU)
python -u -m jarvis_jax.cse.run_stac_bout \
    --bout "$WORK/${REC}_bout.h5" \
    --out  "$WORK/${REC}/Fruitfly_ik_v1_cse.h5" \
    --stac-config-dir "$STAC_CFG" \
    --overrides paths=hyak anatomy=v1 dataset=

# Stage 3: project canonical vertices -> aux labels (CPU)
python -u -m jarvis_jax.cse.cse_labels project \
    --bout    "$WORK/${REC}_bout.h5" \
    --stac-ik "$WORK/${REC}/Fruitfly_ik_v1_cse.h5" \
    --calib-root "$ROOT/calib_params" \
    --rec "$REC" \
    --model-xml "$XML" \
    --mesh "$MESH" \
    --out "$WORK/${REC}/cse_labels_M200.npz" \
    --M 200

echo "DONE ${REC}"
