"""Tests for V3FramesetDataset and frameset_batches (Task 6 – data/v3_3d.py)."""
import os
import numpy as np
import pytest
from jarvis_jax.data.v3_3d import V3FramesetDataset, frameset_batches
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
skip = pytest.mark.skipif(not os.path.isdir(ROOT), reason="V3 data absent")

CALIB_BASE = os.path.join(ROOT, "calib_params")


@skip
def test_frameset_shapes_and_triangulation_consistency():
    ds = V3FramesetDataset(ROOT, "val")
    assert len(ds) > 0
    s = ds[0]
    nc = s["crops4"].shape[0]

    # Shape and dtype checks
    assert s["crops4"].shape == (nc, 448, 448, 4), f"crops4 shape {s['crops4'].shape}"
    assert s["crops4"].dtype == np.uint8
    assert s["centerHM"].shape == (nc, 2), f"centerHM shape {s['centerHM'].shape}"
    assert s["centerHM"].dtype == np.float32
    assert s["center3D"].shape == (3,), f"center3D shape {s['center3D'].shape}"
    assert s["center3D"].dtype == np.float32
    assert s["cameraMatrices"].shape == (nc, 4, 3), f"cameraMatrices shape {s['cameraMatrices'].shape}"
    assert s["cameraMatrices"].dtype == np.float32
    assert s["kp3d"].shape == (50, 3), f"kp3d shape {s['kp3d'].shape}"
    assert s["kp3d"].dtype == np.float32
    assert s["vis"].shape == (50,), f"vis shape {s['vis'].shape}"
    assert s["vis"].dtype == bool

    # At least one joint is visible (triangulated from >=2 cameras)
    assert s["vis"].any(), "No visible joints in first frameset"

    # Triangulation consistency: visible 3D kp should reproject near its 2D labels
    dataset_name = ds.framesets[0]["datasetName"]
    calib_dir = os.path.join(CALIB_BASE, dataset_name)
    rt = ReprojectionTool(calib_dir)

    errors = []
    for j in range(50):
        if not s["vis"][j]:
            continue
        kp3d_j = s["kp3d"][j].astype(np.float64)
        reprojected = rt.reproject_point(kp3d_j)  # (nc, 2)
        # Get the per-camera 2D labels for this joint
        kp2d_j = ds.get_joint_2d(0, j)  # (nc, 3) [x, y, v]
        for c in range(nc):
            if kp2d_j[c, 2] > 0:
                dist = np.linalg.norm(reprojected[c] - kp2d_j[c, :2])
                errors.append(dist)

    assert len(errors) > 0, "No valid 2D observations to check triangulation"
    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))
    print(f"\nTriangulation reprojection error: mean={mean_err:.3f}px  max={max_err:.3f}px")
    # Allow up to 5 pixels mean error (DLT from 7 cameras is typically <1px)
    assert mean_err < 5.0, f"Triangulation reprojection error too high: mean={mean_err:.3f}px"


@skip
def test_vis_requires_two_cameras():
    """vis[j] is True only if joint j is visible in >=2 cameras."""
    ds = V3FramesetDataset(ROOT, "val")
    s = ds[0]
    nc = s["crops4"].shape[0]
    for j in range(50):
        if s["vis"][j]:
            kp2d_j = ds.get_joint_2d(0, j)  # (nc, 3)
            n_visible = int(np.sum(kp2d_j[:, 2] > 0))
            assert n_visible >= 2, (
                f"Joint {j} marked vis but only {n_visible} camera(s) have v>0"
            )


@skip
def test_frameset_batches_stack():
    ds = V3FramesetDataset(ROOT, "val")
    b = next(frameset_batches(ds, 2, shuffle=False))
    assert b["crops4"].shape[0] == 2
    assert b["kp3d"].shape == (2, 50, 3)
    assert b["vis"].shape == (2, 50)
    assert b["cameraMatrices"].shape[0] == 2
    assert b["centerHM"].shape[0] == 2
    assert b["center3D"].shape == (2, 3)


@skip
def test_recordings_filter():
    """recordings= kwarg restricts to matching frameset datasetNames."""
    ds_full = V3FramesetDataset(ROOT, "val")
    if len(ds_full) == 0:
        pytest.skip("No framesets in val")
    first_rec = ds_full.framesets[0]["datasetName"]
    ds_filtered = V3FramesetDataset(ROOT, "val", recordings=[first_rec])
    assert len(ds_filtered) > 0
    assert len(ds_filtered) <= len(ds_full)
    for fs in ds_filtered.framesets:
        assert fs["datasetName"] == first_rec
