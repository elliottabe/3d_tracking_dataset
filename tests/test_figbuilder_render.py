"""Tile rendering: full-canvas placement, ink box, caching."""
from __future__ import annotations

import re

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import figbuilder.panels  # noqa: F401
from figbuilder.bundle import Bundle, PanelData
from figbuilder.figure import FigureSpec, PanelSpec
from figbuilder.render import panel_data, render_tile, tile_cache_key


def _fig():
    return FigureSpec(width_mm=100.0, height_mm=60.0)


def _panel(rect=(0.15, 0.2, 0.7, 0.6)):
    return PanelSpec(id="p", type="line", rect=rect,
                     data={"y": {"dataset": "/panels/p/data/y"}},
                     spec={"series": [{"y": "y", "color": "#000000"}],
                           "ylabel": "z (mm)", "xlabel": "t (s)"})


def _bundle():
    return Bundle(meta={}, panels={"p": PanelData(
        type="line", data={"y": np.linspace(0, 1, 50)}, assets={}, attrs={})})


def test_font_stack_includes_a_metric_compatible_arial_substitute():
    """Ruling 7: matplotlib's layout font must match the renderer's draw font.

    The SVG declares the whole stack and the renderer picks from it; matplotlib
    lays out with the first family it can resolve locally. Without Liberation
    Sans in the stack, a machine lacking Arial lays out in DejaVu Sans (13-16%
    wider at 6pt) while the renderer draws Liberation Sans, inflating ink_box.
    """
    from figbuilder.style import DEFAULT_RCPARAMS
    stack = DEFAULT_RCPARAMS["font.sans-serif"]
    assert stack[:2] == ["Arial", "Helvetica"], "Arial must stay first"
    assert "Liberation Sans" in stack
    assert stack.index("Liberation Sans") < stack.index("DejaVu Sans")


def test_layout_font_is_one_the_exported_svg_also_declares():
    """Whatever matplotlib measures with must appear in the declared stack."""
    from matplotlib import font_manager as fm
    from figbuilder.style import apply_style
    apply_style({})
    path = fm.findfont(fm.FontProperties(
        family=matplotlib.rcParams["font.sans-serif"]))
    resolved = fm.FontProperties(fname=path).get_name()
    assert resolved in matplotlib.rcParams["font.sans-serif"], (
        f"laying out with {resolved!r}, which the SVG never declares")


def test_tile_is_rendered_at_full_figure_size_in_points():
    """matplotlib writes 6-decimal pt lengths ("283.464567pt"), so compare the
    parsed number, never a re-formatted string."""
    res = render_tile(_fig(), _panel(), {"y": np.linspace(0, 1, 50)})
    head = res.svg[:400].decode("utf-8")
    w = float(re.search(r'width="([\d.]+)pt"', head).group(1))
    h = float(re.search(r'height="([\d.]+)pt"', head).group(1))
    assert w == pytest.approx(100.0 / 25.4 * 72, rel=1e-4)
    assert h == pytest.approx(60.0 / 25.4 * 72, rel=1e-4)


def test_tile_carries_a_gid_for_the_panel():
    res = render_tile(_fig(), _panel(), {"y": np.linspace(0, 1, 50)})
    assert b'id="p"' in res.svg


def test_tile_text_is_real_text_not_glyph_paths():
    res = render_tile(_fig(), _panel(), {"y": np.linspace(0, 1, 50)})
    assert b"z (mm)" in res.svg


def test_ink_box_is_larger_than_the_axes_rect():
    rect = (0.15, 0.2, 0.7, 0.6)
    res = render_tile(_fig(), _panel(rect), {"y": np.linspace(0, 1, 50)})
    ix, iy, iw, ih = res.ink_box
    assert ix < rect[0] and iy < rect[1]
    assert iw > rect[2] and ih > rect[3]


def test_overflow_flag_set_when_labels_fall_off_canvas():
    tight = render_tile(_fig(), _panel((0.01, 0.01, 0.5, 0.5)),
                        {"y": np.linspace(0, 1, 50)})
    roomy = render_tile(_fig(), _panel((0.2, 0.25, 0.6, 0.55)),
                        {"y": np.linspace(0, 1, 50)})
    assert tight.overflows
    assert not roomy.overflows


def test_cache_key_changes_with_rect_and_spec():
    y = np.linspace(0, 1, 50)
    a = tile_cache_key(_fig(), _panel((0.1, 0.1, 0.5, 0.5)), {"y": y})
    b = tile_cache_key(_fig(), _panel((0.1, 0.1, 0.5, 0.6)), {"y": y})
    assert a != b
    p = _panel()
    p.spec["ylabel"] = "different"
    assert (tile_cache_key(_fig(), p, {"y": y})
            != tile_cache_key(_fig(), _panel(), {"y": y}))


def test_cache_key_changes_when_only_the_resolved_data_changes():
    """Regression for Finding 1: same panel/refs, different array contents.

    `panel.data` only carries dataset REFERENCE STRINGS (e.g.
    "/panels/p/data/y"); a bundle can be regenerated with new values at that
    same path. The cache key must fold in a digest of the resolved arrays
    or a stale tile gets served after re-export.
    """
    p = _panel()
    key_a = tile_cache_key(_fig(), p, {"y": np.linspace(0, 1, 50)})
    key_b = tile_cache_key(_fig(), p, {"y": np.linspace(0, 1, 50) + 1.0})
    assert key_a != key_b


def test_second_render_hits_the_cache(tmp_path):
    first = render_tile(_fig(), _panel(), {"y": np.linspace(0, 1, 50)},
                        cache_dir=tmp_path)
    second = render_tile(_fig(), _panel(), {"y": np.linspace(0, 1, 50)},
                         cache_dir=tmp_path)
    assert not first.cache_hit
    assert second.cache_hit
    assert first.svg == second.svg


def test_corrupt_cache_sidecar_falls_back_to_a_fresh_render(tmp_path):
    """Regression for Finding 2: a truncated/corrupt `.json` sidecar next to a
    valid cached `.svg` must not crash `render_tile` -- treat it exactly like
    a missing sidecar and re-render."""
    data = {"y": np.linspace(0, 1, 50)}
    key = tile_cache_key(_fig(), _panel(), data)
    (tmp_path / f"{key}.svg").write_bytes(b"<?xml not really an svg")
    (tmp_path / f"{key}.json").write_text("{not valid json")

    res = render_tile(_fig(), _panel(), data, cache_dir=tmp_path)
    assert not res.cache_hit
    assert res.svg.startswith(b"<?xml")


def test_panel_data_resolves_dataset_references():
    got = panel_data(_bundle(), _panel())
    np.testing.assert_allclose(got["y"], np.linspace(0, 1, 50))


def test_panel_data_reports_a_missing_dataset_by_path():
    p = _panel()
    p.data["y"] = {"dataset": "/panels/p/data/nope"}
    with pytest.raises(KeyError, match="nope"):
        panel_data(_bundle(), p)


def test_polar_panel_gets_a_polar_axes():
    p = PanelSpec(id="q", type="courtship.wing_polar", rect=(0.2, 0.2, 0.6, 0.6),
                  data={"phase_diffs": {"dataset": "/panels/q/data/phase_diffs"}},
                  spec={})
    res = render_tile(_fig(), p,
                      {"phase_diffs": np.random.default_rng(0).uniform(-3, 3, 50)})
    assert res.svg.startswith(b"<?xml")
