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

def test_draw_mask_mutates_in_place():
    img = _img(); m = np.zeros((32,32), bool); m[4:8,4:8] = True
    out = overlays.draw_mask(img, m, (200,200,200), alpha=0.5, outline=True)
    assert out is img
    assert int(img[4:8,4:8].sum()) > 0                # caller's array itself was modified

def test_draw_mask_empty_is_noop():
    img = _img(); m = np.zeros((32,32), bool)
    out = overlays.draw_mask(img, m, (200,200,200), alpha=0.5, outline=True)
    assert out is img
    assert int(img.sum()) == 0

def test_draw_axis_draws_arrow():
    img = _img()
    out = overlays.draw_axis(img, (5,5), (25,25), (0,0,255))
    assert int((out[:,:,2] > 0).sum()) > 0            # arrow drawn in red

    img2 = _img()
    out2 = overlays.draw_axis(img2, (np.nan,5), (25,25), (0,0,255))
    assert int(out2.sum()) == 0                       # non-finite endpoint -> no-op, no raise

def test_legend_writes_text():
    img = _img()
    out = overlays.legend(img, [("hi",(0,0,255))])
    assert int((out[:,:,2] > 0).sum()) > 0            # some red pixels from the text

def test_draw_cloud_draws_points():
    img = _img()
    out = overlays.draw_cloud(img, np.array([[10.,10.]]), (0,255,0), radius=2)
    assert out[10,10].tolist() == [0,255,0]
    assert int((out[:,:,1] > 0).sum()) > 0

def test_extreme_and_oob_coords_safe():
    img = _img()
    uv = np.array([[10.,10.], [1e12, 1e12], [1000.,1000.]])
    out = overlays.draw_points(img, uv, (0,0,255), radius=2)   # must not raise
    assert out[10,10].tolist() == [0,0,255]           # in-frame point drawn

    ref = overlays.draw_points(_img(), np.array([[10.,10.]]), (0,0,255), radius=2)
    assert np.array_equal(out, ref)                   # off-image points drew nothing extra

    img2 = _img()
    out2 = overlays.draw_axis(img2, (5,5), (1e12,1e12), (0,255,0))  # must not raise
    assert int((out2[:,:,1] > 0).sum()) > 0
