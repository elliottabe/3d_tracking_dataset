import numpy as np
from viz.core import layout

def test_montage_grid_dims():
    tiles = [np.zeros((10,20,3),np.uint8), np.zeros((12,18,3),np.uint8), np.zeros((8,8,3),np.uint8)]
    m = layout.montage(tiles, cols=2)
    # 3 tiles, 2 cols -> 2 rows; cell = max h(12) x max w(20)
    assert m.shape == (2*12, 2*20, 3)

def test_crop_to_points_bounds():
    img = np.zeros((100,100,3),np.uint8)
    crop, (x0,y0) = layout.crop_to_points(img, np.array([[40.,40.],[60.,60.]]), pad=5)
    assert crop.shape[0] <= 100 and x0 >= 0 and y0 >= 0 and crop.shape[0] > 0
    # exact geometry: min-pad=35, max+pad=65 -> 30x30 crop at (35,35)
    assert (x0, y0) == (35, 35)
    assert crop.shape == (30, 30, 3)

def test_crop_to_points_clips_offframe():
    img = np.zeros((100,100,3),np.uint8)
    pts = np.array([[150.,150.],[200.,200.]])
    crop, (x0,y0) = layout.crop_to_points(img, pts, pad=5)
    assert x0 == 100 and y0 == 100
    assert crop.shape[0] == 0 and crop.shape[1] == 0
    assert crop.ndim == 3

def test_crop_to_points_no_points_returns_copy():
    img_input = np.zeros((100,100,3),np.uint8)
    uv = np.full((2,2), np.nan)
    returned, (x0,y0) = layout.crop_to_points(img_input, uv)
    assert returned is not img_input
    assert (x0, y0) == (0, 0)
    returned[:] = 255
    assert not np.any(img_input == 255)

def test_banner_builds_strip_with_legend():
    out = layout.banner(120, [("hello", (0, 0, 255))])
    assert out.shape == (28, 120, 3)
    assert int((out[:, :, 2] > 0).sum()) > 0
