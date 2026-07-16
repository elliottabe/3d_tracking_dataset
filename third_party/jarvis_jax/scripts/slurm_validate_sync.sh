#!/bin/bash
# ckpt validation for the desync gate: regenerate 16_21_32 (reindex) bouts 1-2
# masks WITH the gate into a throwaway dir, to compare cross-camera consistency
# vs the existing (desynced) Predictions_3D_sam3 masks. SAM3 env = cu13 libs
# (mirrors slurm_bout_array.build_sam3_array_script), NOT the jax pipeline env.
#SBATCH --job-name=sync-validate
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --nodelist=g[3090-3137]
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --open-mode=append
#SBATCH -o ./OutFiles/slurm-%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=eabe@uw.edu
set -x
source ~/.bashrc
nvidia-smi
micromamba activate 3d_tracking
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REC=/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1/2026_04_02_16_21_32
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
python -u scripts/sam3_masks.py \
    sam3.session_dir="$REC" \
    sam3.out="$REC/Predictions_3D_sync_validate" \
    sam3.bout_ids=1 sam3.reuse_masks=false sam3.num_animals=2 \
    sam3.overlay=false sam3.sam3_compile=false
