import numpy as np
from scripts.viz.voxel_resolution import segment_lengths_voxels

NAMES = ["Scutellum", "Abd_tip", "T1L_TaT3", "T1L_TaTip"]


def test_segment_lengths_in_voxels_scale_with_spacing():
    kp = np.zeros((4, 4, 3), np.float32)
    kp[:, 1] = [12.8, 0, 0]      # Scutellum -> Abd_tip = 12.8 world units
    kp[:, 3] = [1.59, 0, 0]      # tarsal segment = 1.59 world units
    kp[:, 2] = [0, 0, 0]
    a = segment_lengths_voxels(kp, NAMES, grid_spacing=1.0)
    b = segment_lengths_voxels(kp, NAMES, grid_spacing=0.25)
    assert np.isclose(a["Scutellum->Abd_tip"], 12.8, atol=0.1)
    assert np.isclose(a["T1L_TaT3->T1L_TaTip"], 1.59, atol=0.1)
    # finer spacing => MORE voxels per segment (the point of stage 2)
    assert np.isclose(b["T1L_TaT3->T1L_TaTip"] / a["T1L_TaT3->T1L_TaTip"], 4.0,
                      rtol=1e-3)


def test_nan_keypoints_are_ignored():
    kp = np.full((2, 4, 3), np.nan, np.float32)
    kp[0, 0] = [0, 0, 0]
    kp[0, 1] = [10, 0, 0]
    out = segment_lengths_voxels(kp, NAMES, grid_spacing=1.0)
    assert np.isclose(out["Scutellum->Abd_tip"], 10.0, atol=1e-3)
