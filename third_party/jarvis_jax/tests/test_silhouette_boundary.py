# tests/test_silhouette_boundary.py
import numpy as np
from jarvis_jax.cse.silhouette_boundary import mask_boundary_pixels, sample_boundary_points


def test_boundary_pixels_of_a_solid_square():
    # 10x10 foreground square inside a 20x20 image -> boundary is its perimeter.
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    bp = mask_boundary_pixels(mask)
    # coordinates are (x, y) = (col, row).
    xs, ys = bp[:, 0], bp[:, 1]
    # every boundary pixel is foreground
    assert mask[ys.astype(int), xs.astype(int)].all()
    # boundary is the outer ring only: the 8x8 interior [6..13]^2 is NOT boundary.
    interior = (xs >= 6) & (xs <= 13) & (ys >= 6) & (ys <= 13)
    assert not interior.any()
    # perimeter of a 10x10 filled square (4-connectivity erosion) = 36 pixels.
    assert bp.shape[0] == 36


def test_sample_boundary_points_shape_and_on_boundary():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    pts = sample_boundary_points(mask, n_points=64, seed=0)
    assert pts.shape == (64, 2)
    assert np.isfinite(pts).all()
    # every sampled point coincides with a real boundary pixel
    bp = mask_boundary_pixels(mask)
    bset = {(float(x), float(y)) for x, y in bp}
    for x, y in pts:
        assert (float(x), float(y)) in bset


def test_sample_boundary_points_empty_mask_is_nan():
    mask = np.zeros((20, 20), dtype=bool)
    pts = sample_boundary_points(mask, n_points=32, seed=0)
    assert pts.shape == (32, 2)
    assert np.isnan(pts).all()


def test_sample_boundary_points_deterministic():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    a = sample_boundary_points(mask, n_points=40, seed=7)
    b = sample_boundary_points(mask, n_points=40, seed=7)
    np.testing.assert_array_equal(a, b)
