import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.viz.ik_explainer import trace_panel as tp


def _series(n=45):
    x = np.linspace(0.0, 1.0, n)
    return [("raw",  x + 0.5 * (np.arange(n) == 20), (0, 0, 255)),
            ("filt", x + 0.02,                        (0, 255, 0)),
            ("ik",   x - 0.02,                        (255, 255, 0))]


def test_panel_has_the_requested_shape_and_dtype():
    p = tp.render_trace_panel(800, 600, _series(), np.arange(420, 465), 441,
                              ylabel="z (mm)")
    assert p.shape == (600, 800, 3)
    assert p.dtype == np.uint8


def test_each_series_colour_actually_appears():
    """A chart that silently drops a trace would still look plausible."""
    p = tp.render_trace_panel(800, 600, _series(), np.arange(420, 465), 441,
                              ylabel="z (mm)")
    for _name, _vals, col in _series():
        hit = np.all(p == np.array(col, np.uint8), axis=-1).sum()
        assert hit > 0, f"colour {col} never drawn"


def test_playhead_moves_with_the_cursor():
    a = tp.render_trace_panel(800, 600, _series(), np.arange(420, 465), 425,
                              ylabel="z (mm)")
    b = tp.render_trace_panel(800, 600, _series(), np.arange(420, 465), 460,
                              ylabel="z (mm)")
    assert not np.array_equal(a, b), "playhead did not move"


def test_flat_series_does_not_divide_by_zero():
    n = 20
    flat = [("a", np.full(n, 3.0), (255, 255, 255))]
    p = tp.render_trace_panel(400, 300, flat, np.arange(n), 5, ylabel="mm")
    assert np.isfinite(p).all() and p.shape == (300, 400, 3)


def test_returns_a_new_array_each_call():
    s, f = _series(), np.arange(420, 465)
    a = tp.render_trace_panel(400, 300, s, f, 430, ylabel="mm")
    b = tp.render_trace_panel(400, 300, s, f, 430, ylabel="mm")
    assert a is not b


def test_coincident_series_both_remain_visible():
    """filtered and IK are nearly identical in the real clip -- that agreement
    is the message, but both must still be visible."""
    n = 30
    x = np.linspace(0.0, 1.0, n)
    s = [("a", x, (0, 255, 0)), ("b", x, (255, 255, 0))]   # exactly coincident
    p = tp.render_trace_panel(600, 400, s, np.arange(n), 15, ylabel="mm")
    for _n, _v, col in s:
        assert np.all(p == np.array(col, np.uint8), axis=-1).sum() > 0, \
            f"{col} fully occluded by a coincident series"
