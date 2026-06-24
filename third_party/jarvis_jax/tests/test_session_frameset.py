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


# ---------------------------------------------------------------------------
# GPU integration test — Task 4: run_predict_session (one bout)
# ---------------------------------------------------------------------------

import os as _os
import csv as _csv
import numpy as _np
import pytest as _pytest


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


needs_cuda = _pytest.mark.skipif(not _has_cuda(), reason="needs CUDA + SAM3 masks")

_SESSION = "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04"
_VIT = "/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v3_8gpu_20260620/final"
_RUN4 = "/gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/run4/final"
_JROOT = "/gscratch/portia/eabe/Research/Github/JARVIS-HybridNet"


@needs_cuda
def test_run_predict_session_one_bout(tmp_path):
    """GPU integration: predict_session on bout 4 (394 frames, 2 flies)."""
    from jarvis_jax.predict.session_predict import run_predict_session

    masks_dir = _os.environ["D2_MASKS_DIR"]  # D2 output dir holding bout_00004/sam3_masks.npz

    res = run_predict_session(
        session_dir=_SESSION,
        masks_dir=masks_dir,
        out=str(tmp_path),
        project="red_data_unified",
        jarvis_root=_JROOT,
        v2v_final=_RUN4,
        vitpose_ckpt=_VIT,
        sharpen=3.0,
        num_animals=2,
        batch=8,
        bout_ids=[4],
    )

    f0 = tmp_path / "data3D_fly0.csv"
    f1 = tmp_path / "data3D_fly1.csv"
    assert f0.is_file() and f1.is_file(), "per-fly session CSVs must exist"

    with open(f0, newline="") as f:
        r = list(_csv.reader(f))

    # Header row 1: frame + 50*4 joint columns
    assert r[0][0] == "frame", f"first header cell should be 'frame', got {r[0][0]!r}"
    assert len(r[0]) == 1 + 50 * 4, f"expected 201 header cols, got {len(r[0])}"
    # Header row 2: frame, x, y, z, confidence repeated
    assert r[1][1:5] == ["x", "y", "z", "confidence"], f"sub-header mismatch: {r[1][1:5]}"

    # At least some predicted (non-NaN) rows
    vals = [row for row in r[2:] if row[1] != "nan"]
    assert len(vals) > 0, "no predicted (non-NaN) rows in data3D_fly0.csv"
    assert _np.isfinite(float(vals[0][1])), "first predicted x-coord is not finite"

    # Frame indices should start at 105107 (bout start)
    frame_nums = [int(row[0]) for row in r[2:]]
    assert min(frame_nums) == 105107, f"expected min frame 105107, got {min(frame_nums)}"

    print(f"\n[test] fly0: {len(vals)}/{len(r)-2} predicted, "
          f"first kp3d x={vals[0][1]}, y={vals[0][2]}, z={vals[0][3]}")
