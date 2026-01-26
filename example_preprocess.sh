#!/bin/bash
# Example: Preprocess keypoints for STAC IK

# Example 1: Basic preprocessing (no alignment)
python preprocess_keypoints_for_ik.py \
    --csv_path /data2/users/eabe/datasets/Johnson_lab/free_walking/Predictions_3D_20260114/data3D.csv \
    --skeleton_path data/fly50.json \
    --xml_path assets/fruitfly_v1/fruitfly_v1_free.xml \
    --output_dir /data2/users/eabe/datasets/Johnson_lab/free_walking/Predictions_3D_20260114 \
    --bout_name example_bout_basic \
    --frame_start 6569 \
    --frame_end 7685

# Example 2: With Procrustes alignment and scaling
python preprocess_keypoints_for_ik.py \
    --csv_path /data2/users/eabe/datasets/Johnson_lab/free_walking/Predictions_3D_20260114/data3D.csv \
    --skeleton_path data/fly50.json \
    --xml_path assets/fruitfly_v1/fruitfly_v1_free.xml \
    --output_dir /data2/users/eabe/datasets/Johnson_lab/free_walking/Predictions_3D_20260114 \
    --bout_name example_bout_aligned \
    --frame_start 6569 \
    --frame_end 7685 \
    --apply_alignment \
    --apply_scaling \
    --exclude_antenna \
    --exclude_wings

# Example 3: Process full dataset (no frame selection)
python preprocess_keypoints_for_ik.py \
    --csv_path /data2/users/eabe/datasets/Johnson_lab/free_walking/Predictions_3D_20260114/data3D.csv \
    --skeleton_path data/fly50.json \
    --xml_path assets/fruitfly_v1/fruitfly_v1_free.xml \
    --output_dir /data2/users/eabe/datasets/Johnson_lab/free_walking/Predictions_3D_20260114 \
    --bout_name full_dataset \
    --apply_alignment \
    --apply_scaling

echo "✓ Preprocessing complete!"
