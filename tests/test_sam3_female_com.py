"""Tests for utils.sam3_female_com."""
from __future__ import annotations

import numpy as np
import pytest

from utils.sam3_female_com import _triangulate_point, triangulate_sam3_female_com


def _synthetic_dlt(
    focal: float = 800.0,
    cx: float = 200.0,
    cy: float = 200.0,
    cam_xyz: tuple = (0.0, 0.0, 50.0),
) -> np.ndarray:
    """Build an 11-param DLT for a pinhole camera looking down the -z axis.

    World point (X, Y, Z) with cam at (0, 0, C) pointing at origin:
        u = focal * X / (C - Z) + cx
        v = focal * Y / (C - Z) + cy
    which matches the DLT form
        u = (L1 X + L2 Y + L3 Z + L4) / (L9 X + L10 Y + L11 Z + 1)
    with L9=L10=0, L11=-1/C, L1=focal/C, L2=0, L3=0, L4=cx, and similarly for v.
    """
    C = float(cam_xyz[2])
    L = np.zeros(11, dtype=float)
    # u = focal/C * X / (1 - Z/C) + cx
    L[0] = focal / C          # L1 (X → u)
    L[1] = 0.0
    L[2] = 0.0
    L[3] = cx                 # L4 (bias for u)
    L[4] = 0.0
    L[5] = focal / C          # L6 (Y → v)
    L[6] = 0.0
    L[7] = cy                 # L8
    L[8] = 0.0                # L9
    L[9] = 0.0                # L10
    L[10] = -1.0 / C          # L11 (perspective term in Z)
    return L


def _project(L: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    X, Y, Z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    denom = L[8] * X + L[9] * Y + L[10] * Z + 1.0
    u = (L[0] * X + L[1] * Y + L[2] * Z + L[3]) / denom
    v = (L[4] * X + L[5] * Y + L[6] * Z + L[7]) / denom
    return np.array([u, v], dtype=float)


def test_triangulate_point_recovers_synthetic_3d():
    # Two cameras along slightly different Z-heights to give a well-posed system.
    L1 = _synthetic_dlt(cam_xyz=(0.0, 0.0, 40.0))
    L2 = _synthetic_dlt(cam_xyz=(0.0, 0.0, 60.0))
    target = np.array([1.5, -0.8, 2.0])
    uv1 = _project(L1, target)
    uv2 = _project(L2, target)
    coeffs = np.stack([L1, L2], axis=0)       # (2, 11)
    uv = np.stack([uv1, uv2], axis=0)         # (2, 2)
    valid = np.array([True, True])
    xyz = _triangulate_point(coeffs, uv, valid)
    assert xyz.shape == (3,)
    np.testing.assert_allclose(xyz, target, atol=1e-6)


def test_triangulate_point_returns_nan_below_min_cams():
    L1 = _synthetic_dlt()
    coeffs = np.stack([L1, L1], axis=0)
    uv = np.zeros((2, 2))
    valid = np.array([True, False])           # only 1 valid cam
    xyz = _triangulate_point(coeffs, uv, valid, min_cams=2)
    assert xyz.shape == (3,)
    assert np.all(np.isnan(xyz))


def test_triangulate_sam3_female_com_returns_T_by_3(tmp_path):
    # Build a 2-camera synthetic npz + calib dir and check end-to-end shape.
    L1 = _synthetic_dlt(cam_xyz=(0.0, 0.0, 40.0))
    L2 = _synthetic_dlt(cam_xyz=(0.0, 0.0, 60.0))
    calib_dir = tmp_path / 'calib'
    calib_dir.mkdir()
    np.savetxt(calib_dir / 'Cam01_dlt.csv', L1)
    np.savetxt(calib_dir / 'Cam02_dlt.csv', L2)

    target = np.array([0.5, 0.3, 1.2])
    uv1 = _project(L1, target)
    uv2 = _project(L2, target)
    T = 5
    # float64 to match the atol=1e-6 assertion below; SAM3 npz files on disk
    # may use float32, but the triangulator promotes to float64 internally.
    centroids = np.zeros((1, 2, T, 2), dtype=np.float64)
    centroids[0, 0, :, :] = uv1
    centroids[0, 1, :, :] = uv2
    valid = np.ones((1, 2, T), dtype=bool)
    packed = np.zeros((1, 2, T, 4, 4), dtype=np.uint8)  # shape-only
    npz_path = tmp_path / 'sam3_masks.npz'
    np.savez(npz_path, packed=packed, valid=valid, centroids=centroids)

    com = triangulate_sam3_female_com(npz_path, calib_dir, fly_idx=0, min_cams=2)
    assert com.shape == (T, 3)
    np.testing.assert_allclose(com[0], target, atol=1e-6)


def test_triangulate_sam3_female_com_nan_where_valid_below_min_cams(tmp_path):
    L1 = _synthetic_dlt(cam_xyz=(0.0, 0.0, 40.0))
    L2 = _synthetic_dlt(cam_xyz=(0.0, 0.0, 60.0))
    calib_dir = tmp_path / 'calib'
    calib_dir.mkdir()
    np.savetxt(calib_dir / 'Cam01_dlt.csv', L1)
    np.savetxt(calib_dir / 'Cam02_dlt.csv', L2)

    T = 3
    centroids = np.zeros((1, 2, T, 2), dtype=np.float64)
    valid = np.ones((1, 2, T), dtype=bool)
    valid[0, 1, 1] = False  # frame 1 has only 1 valid cam
    packed = np.zeros((1, 2, T, 4, 4), dtype=np.uint8)
    npz_path = tmp_path / 'sam3_masks.npz'
    np.savez(npz_path, packed=packed, valid=valid, centroids=centroids)

    com = triangulate_sam3_female_com(npz_path, calib_dir, fly_idx=0, min_cams=2)
    assert np.all(np.isnan(com[1]))
    assert not np.any(np.isnan(com[0]))
