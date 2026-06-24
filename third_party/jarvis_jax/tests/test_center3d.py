import numpy as np
import jax.numpy as jnp
from jarvis_jax.geometry.center3d import quantize_center3d, mask_centroids, centroids_to_fullpx


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


def test_mask_centroids_single_blob():
    crops = jnp.zeros((1, 2, 448, 448, 4), jnp.uint8)
    crops = crops.at[0, 0, 100, 200, 3].set(1)   # cam0 mask pixel at (y=100,x=200)
    crops = crops.at[0, 1, 50, 60, 3].set(1)     # cam1 at (y=50,x=60)
    cent, valid = mask_centroids(crops)
    assert bool(valid[0, 0]) and bool(valid[0, 1])
    assert jnp.allclose(cent[0, 0], jnp.array([200.0, 100.0]))  # [x, y]
    assert jnp.allclose(cent[0, 1], jnp.array([60.0, 50.0]))


def test_mask_centroids_empty_marks_invalid():
    crops = jnp.zeros((1, 1, 448, 448, 4), jnp.uint8)  # all-zero mask
    cent, valid = mask_centroids(crops)
    assert not bool(valid[0, 0])


def test_centroids_to_fullpx():
    cent = jnp.array([[[224.0, 224.0]]])      # crop center
    centerHM = jnp.array([[[500.0, 300.0]]])  # crop center in full px
    full = centroids_to_fullpx(cent, centerHM, crop=448)
    assert jnp.allclose(full[0, 0], jnp.array([500.0, 300.0]))  # 224 + 500 - 224
