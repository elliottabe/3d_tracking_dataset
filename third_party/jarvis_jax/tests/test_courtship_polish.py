import numpy as np
from jarvis_jax.tracking.polish import courtship_targets_from_masks


def test_targets_from_masks_shapes_and_present():
    T, C, H, W = 2, 2, 60, 60
    masks = np.zeros((T, C, H, W), bool)
    masks[0, 0, 20:40, 20:40] = True                 # one present cell
    present = masks.any(axis=(2, 3))
    cam_Ms = np.tile(np.eye(2, 3)[None], (C, 1, 1)).astype(np.float32)
    cam_ts = np.zeros((C, 2), np.float32)
    out = courtship_targets_from_masks(masks, present, (cam_Ms, cam_ts),
                                       erode_px=2, n_points=16, sdf_hw=(32, 32), bbox_margin=0.4)
    assert out["boundary"].shape == (T, C, 16, 2)
    assert out["sdf"].shape == (T, C, 32, 32)
    assert out["present"][0, 0] and not out["present"][1, 1]
    # present cell: some finite boundary points; absent cell: NaN
    assert np.isfinite(out["boundary"][0, 0]).any()
    assert np.isnan(out["boundary"][1, 1]).all()
