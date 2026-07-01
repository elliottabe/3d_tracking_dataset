#!/usr/bin/env bash
# Phase-4 active-parts data-prep for ONE general_model condition/split.
# Usage: dataprep_active_parts.sh <COND_DIR> <REC> <SPLIT> <DATASET_CFG>
#   e.g. dataprep_active_parts.sh \
#        /gscratch/portia/eabe/data/Johnson_lab/red_data/general_model/S8_male_R_amp \
#        2026_02_13_13_44_49 train amputation
# headless: ... general_model/headless_22_50 2026_06_09_15_46_55 train BDN2_headless
set -euo pipefail
COND="$1"; REC="$2"; SPLIT="$3"; DSCFG="$4"
XML=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml
ANATOMY=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs/anatomy/v1.yaml
STAC_CFG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs
HN=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/JARVIS-HybridNet
WORK="$COND/cse_work"; mkdir -p "$WORK"

source ~/.bashrc; micromamba activate 3d_tracking

# --- Stage A: SAM3 masks (LD_PRELOAD + cu13 LD_LIBRARY_PATH) ---
# NOTE: the SAM3 weights were deleted from ~/.cache/huggingface. Redirect
# HF_HOME so the model download+cache lands on scratch instead of $HOME.
export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3
mkdir -p "$HF_HOME"
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib
python "$HN/tools/sam3_label_masks.py" \
  --data-root "$COND" --splits "$SPLIT" \
  --confidence 0.5 --qc-thresh 0.6 --resolution 1008
unset LD_PRELOAD

# --- Stage B: build bout (CPU; Task-3 reduced-schema patch REQUIRED) ---
unset LD_LIBRARY_PATH
python -m jarvis_jax.cse.cse_labels build-bout \
  --coco "$COND/annotations/instances_${SPLIT}.json" \
  --calib-root "$COND/calib_params" --rec "$REC" \
  --anatomy "$ANATOMY" --model-xml "$XML" \
  --out "$WORK/${REC}_bout.h5"

# --- Stage C: STAC solve (JAX; LD_LIBRARY_PATH MUST be unset) ---
python -m jarvis_jax.cse.run_stac_bout \
  --bout "$WORK/${REC}_bout.h5" \
  --out "$WORK/${REC}/Fruitfly_ik_v1_cse.h5" \
  --stac-config-dir "$STAC_CFG" \
  --overrides paths=hyak anatomy=v1 dataset="$DSCFG"
echo "[dataprep] done: $WORK/${REC}/Fruitfly_ik_v1_cse.h5"
