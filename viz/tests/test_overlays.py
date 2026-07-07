# viz/tests/test_overlays.py
import numpy as np
from viz.core import overlays

def _img(): return np.zeros((32, 32, 3), np.uint8)

def test_draw_points_marks_pixels_finite_only():
    img = _img()
    out = overlays.draw_points(img.copy(), np.array([[10.,10.],[np.nan,5.]]), (0,0,255), radius=2)
    assert out[10,10].tolist() == [0,0,255]           # drawn
    assert int((out[:,:,2] > 0).sum()) > 0            # some red
    assert out.shape == (32,32,3)

def test_draw_mask_fill_keeps_shape_and_tints():
    img = _img(); m = np.zeros((32,32), bool); m[4:8,4:8] = True
    out = overlays.draw_mask(img, m, (200,200,200), alpha=0.5, outline=True)
    assert out.shape == (32,32,3) and int(out[6,6].sum()) > 0

def test_draw_chain_connects_points():
    img = _img()
    out = overlays.draw_chain(img, [(2,2),(20,20)], (0,255,0), thickness=1)
    assert int((out[:,:,1] > 0).sum()) > 2            # a line of green
