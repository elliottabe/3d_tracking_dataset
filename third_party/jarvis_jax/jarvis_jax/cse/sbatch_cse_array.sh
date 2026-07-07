#!/bin/bash
#SBATCH --job-name=cse_stac
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eabe@uw.edu
#
# Full CSE auto-label run: one array task per V3 recording (manifest line).
#   sbatch --array=1-17%6 sbatch_cse_array.sh
# Manifest line format (TSV): split <tab> rec <tab> coco_json <tab> bout_h5
set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
STAC_CFG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs
ROOT=/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3
XML=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml
MESH=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical.npz
WORK=/gscratch/portia/eabe/data/Johnson_lab/cse_work
MAN=$WORK/manifest.tsv

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MAN")
REC=$(echo "$LINE" | cut -f2)
BOUT=$(echo "$LINE" | cut -f4)
echo "task ${SLURM_ARRAY_TASK_ID}: REC=$REC"
nvidia-smi -L
cd "$PKG"

python -u -m jarvis_jax.cse.run_stac_bout \
    --bout "$BOUT" --out "$WORK/${REC}/Fruitfly_ik_v1_cse.h5" \
    --stac-config-dir "$STAC_CFG" \
    --overrides paths=hyak anatomy=v1 dataset=

python -u -m jarvis_jax.cse.cse_labels project \
    --bout "$BOUT" --stac-ik "$WORK/${REC}/Fruitfly_ik_v1_cse.h5" \
    --calib-root "$ROOT/calib_params" --rec "$REC" \
    --model-xml "$XML" --mesh "$MESH" \
    --out "$WORK/${REC}/cse_labels_M200.npz" --M 200

echo "DONE ${REC}"
