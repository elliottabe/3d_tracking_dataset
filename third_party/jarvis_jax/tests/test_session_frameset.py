import numpy as np
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.geometry.center3d import project_center_to_cameras
from jarvis_jax.predict.session_frameset import build_frameset

CALIB = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3/calib_params/2026_01_13_18_47_45"


def test_build_frameset_shapes_and_centerhm():
    rt = ReprojectionTool(CALIB)
    cm = rt.camera_matrices.astype(np.float32); nc = cm.shape[0]
    H, W = 1080, 1920
    X = np.array([2.0, 1.0, 10.0], np.float32)
    px = project_center_to_cameras(X, cm)                       # where the fly projects
    frames = np.zeros((nc, H, W, 3), np.uint8)
    masks = np.zeros((nc, H, W), bool)
    cents = np.zeros((nc, 2), np.float32)
    valid = np.ones(nc, bool)
    for c in range(nc):
        cx, cy = int(px[c, 0]), int(px[c, 1])
        if 0 <= cx < W and 0 <= cy < H:
            masks[c, max(0, cy-3):cy+3, max(0, cx-3):cx+3] = True
            cents[c] = px[c]
        else:
            valid[c] = False
    crops4, centerHM, n_valid = build_frameset(frames, masks, cents, valid, cm)
    assert n_valid >= 2
    assert crops4.shape == (nc, 448, 448, 4) and crops4.dtype == np.uint8
    # mask channel present (the blob lands in a valid camera's crop)
    assert crops4[..., 3].sum() > 0
    # centerHM is a crop center in full px near the projected point (within clamp)
    for c in range(nc):
        if valid[c]:
            assert abs(centerHM[c, 0] - px[c, 0]) < 448 and abs(centerHM[c, 1] - px[c, 1]) < 448


def test_build_frameset_too_few_valid_returns_none():
    rt = ReprojectionTool(CALIB)
    cm = rt.camera_matrices.astype(np.float32); nc = cm.shape[0]
    frames = np.zeros((nc, 100, 100, 3), np.uint8)
    masks = np.zeros((nc, 100, 100), bool)
    cents = np.zeros((nc, 2), np.float32)
    valid = np.zeros(nc, bool); valid[0] = True   # only 1 valid
    crops4, centerHM, n_valid = build_frameset(frames, masks, cents, valid, cm)
    assert crops4 is None and n_valid == 1


def test_reorder_matrices_by_name():
    import numpy as np
    from jarvis_jax.predict.session_predict import reorder_matrices_by_name
    jax_names = ["CamA", "CamB", "CamC"]
    mats = np.arange(3 * 4 * 3).reshape(3, 4, 3).astype(np.float32)
    target = ["CamC", "CamA", "CamB"]
    out = reorder_matrices_by_name(jax_names, mats, target)
    assert np.array_equal(out[0], mats[2]) and np.array_equal(out[1], mats[0])
    import pytest
    with pytest.raises(KeyError):
        reorder_matrices_by_name(jax_names, mats, ["CamA", "CamX", "CamB"])
