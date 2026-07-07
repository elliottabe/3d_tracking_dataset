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
