"""Panel registry and generic panel types."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import figbuilder.panels.generic  # noqa: F401  (registers types)
from figbuilder.panels.base import get_panel_type, list_panel_types
from figbuilder.style import apply_style


def test_apply_style_sets_svg_text_and_font_size():
    apply_style({"font.size": 7.5})
    assert matplotlib.rcParams["svg.fonttype"] == "none"
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["font.size"] == 7.5


def test_unknown_panel_type_raises_with_helpful_message():
    with pytest.raises(KeyError, match="unknown panel type"):
        get_panel_type("nope.not.real")


def test_list_panel_types_includes_line_with_schema():
    entries = {e["id"]: e for e in list_panel_types()}
    assert "line" in entries
    assert "properties" in entries["line"]["schema"]


def test_line_panel_draws_series_and_labels():
    pt = get_panel_type("line")
    fig, ax = plt.subplots()
    try:
        pt.draw(ax,
                {"x": np.arange(10), "y": np.arange(10) * 2.0},
                {"series": [{"x": "x", "y": "y", "label": "wing L",
                             "color": "#38bdf8"}],
                 "xlabel": "time (s)", "ylabel": "z (mm)"})
        assert len(ax.lines) == 1
        assert ax.lines[0].get_label() == "wing L"
        assert ax.get_xlabel() == "time (s)"
    finally:
        plt.close(fig)


def test_image_panel_draws_asset_without_axes_decoration():
    pt = get_panel_type("image")
    fig, ax = plt.subplots()
    try:
        img = np.zeros((8, 6, 3), dtype=np.uint8)
        pt.draw(ax, {"img": img}, {"asset": "img"})
        assert len(ax.images) == 1
        assert ax.get_xticks().size == 0
    finally:
        plt.close(fig)


def test_blank_panel_draws_nothing_and_hides_spines():
    pt = get_panel_type("blank")
    fig, ax = plt.subplots()
    try:
        pt.draw(ax, {}, {})
        assert len(ax.lines) == 0
        assert not any(s.get_visible() for s in ax.spines.values())
    finally:
        plt.close(fig)
