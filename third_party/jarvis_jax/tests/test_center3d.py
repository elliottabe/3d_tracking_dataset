import numpy as np
from jarvis_jax.geometry.center3d import quantize_center3d


def test_quantize_matches_v3_recipe():
    # midrange of non-zero coords, int-snapped to the grid (gs=1)
    pts = np.array([[10.4, -3.0, 6.6], [2.0, -1.0, 8.0]], dtype=np.float32)
    # axis0 nz: max10.4 min2 -> mid=6.2 -> int=6 ; axis1: max-1 min-3 -> mid=-2 -> int=-2
    # axis2: max8 min6.6 -> mid=7.3 -> int=7
    out = quantize_center3d(pts, grid_spacing=1)
    assert out.tolist() == [6.0, -2.0, 7.0]


def test_quantize_single_point_is_lattice_snap():
    p = np.array([[12.7, -4.2, 0.9]], dtype=np.float32)  # one triangulated centroid
    out = quantize_center3d(p, grid_spacing=1)            # mid = p itself -> int(p)
    assert out.tolist() == [12.0, -4.0, 0.0]


def test_quantize_empty_returns_zeros():
    out = quantize_center3d(np.zeros((0, 3), np.float32), grid_spacing=1)
    assert out.tolist() == [0.0, 0.0, 0.0]
