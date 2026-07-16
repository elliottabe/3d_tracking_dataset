#!/bin/bash
#SBATCH --job-name=cse_predict350
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eabe@uw.edu
#
# Wings v1 / stage 3: redeploy Session0 with the 350-channel (wing) model, one bout
# per array task, then render a per-bout dense-pose-on-video overlay.
#   sbatch --array=1-24%6 sbatch_predict_session_wings.sh
# Each task: SAM3 masks + videos -> ViTPose-350 -> reproject -> V2VNet-350 ->
# per-fly 350-pt CSVs (50 kp + 300 verts = 200 body/leg + 100 wing) in
# out/bout_<idx>/fly{0,1}.csv, then viz_bout -> out/viz/bout_<idx>.png.
set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH                                  # JAX uses bundled CUDA wheels
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"   # cv2 needs the conda libstdc++
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
nvidia-smi -L

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
RUNS=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs
MESH=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz
SESS=/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04
OUT=/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/predictions_cse350_wings
BOUT=${SLURM_ARRAY_TASK_ID:-1}

cd "$PKG"
python -u scripts/predict_session.py paths=hyak model=hybridnet predict_session=default \
    run_id=cse_v2v350 \
    paths.runs_root="$RUNS" \
    paths.vitpose_ckpt="$RUNS/cse_vit350_wings/final" \
    model.sharpen=3.0 \
    predict_session.session_dir="$SESS" \
    predict_session.num_keypoints=350 \
    predict_session.num_animals=2 \
    predict_session.batch=4 \
    predict_session.out="$OUT" \
    predict_session.bout_ids="$BOUT" \
    hydra.run.dir="/tmp/hydra_${SLURM_JOB_ID}_${BOUT}"
echo "PREDICT350 DONE bout $BOUT"

# per-bout dense-pose-on-video overlay (cv2-only; body/leg=cyan, wing=red)
BDIR=$(printf "%s/bout_%05d" "$OUT" "$BOUT")
python -u -m jarvis_jax.tracking.viz_bout \
    --pred-dir "$BDIR" --session-dir "$SESS" --mesh "$MESH" \
    --num-animals 2 --n-frames 4 --n-kp 50 \
    --out "$(printf "%s/viz/bout_%05d.png" "$OUT" "$BOUT")"
echo "VIZ350 DONE bout $BOUT"
