#!/bin/bash
# ckpt-g2 wrapper for the 2D keypoint-wobble diagnostic (decode-comparison).
# Runs ViTPose once on one camera of one bout, decodes the identical heatmaps
# several ways, prints the per-method wobble table + soft/hard ratio verdict.
#
# The configured recording.predictions_dir (Predictions_3D_36233268_fixed) no
# longer exists on disk; Predictions_3D_sam3_all30 is the current full-session
# SAM3 run with camera-named masks (correct camera order), so we override
# recording.predictions_dir to it. Override any knob via env, e.g.:
#   env -u JAX_PLATFORMS BOUT_ID=3 CAMERA=Cam2012630 N_FRAMES=500 sbatch <this>
#SBATCH --job-name=wobble-diag
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --nodelist=g[3090-3137]
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --open-mode=append
#SBATCH -o ./OutFiles/slurm-%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=eabe@uw.edu

module load cuda/12.9.1
set -x
source ~/.bashrc
nvidia-smi
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
PRED_DIR_DEFAULT=/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04/Predictions_3D_sam3_all30
python -u third_party/jarvis_jax/scripts/diagnose_2d_wobble.py \
    +bout_id="${BOUT_ID:-3}" +fly="${FLY:-0}" \
    +camera="${CAMERA:-Cam2012630}" +n_frames="${N_FRAMES:-500}" \
    recording.predictions_dir="${PRED_DIR:-$PRED_DIR_DEFAULT}" \
    +diag_out="diagnostics/wobble_diag_bout${BOUT_ID:-3}_${CAMERA:-Cam2012630}.npz"
