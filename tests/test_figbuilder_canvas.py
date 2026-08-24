"""Canvas sizing: presets, unit conversion, and the max-width guard."""
from __future__ import annotations

import json

import pytest

from figbuilder.canvas import (
    CANVAS_PRESETS, check_canvas, from_mm, resolve_preset, to_mm,
)
from figbuilder.figure import FigureSpec, load_figure


def test_unit_conversion_roundtrips():
    assert to_mm(8.5, "in") == pytest.approx(215.9)
    assert to_mm(1.0, "cm") == pytest.approx(10.0)
    assert to_mm(72.0, "pt") == pytest.approx(25.4)
    assert from_mm(215.9, "in") == pytest.approx(8.5)


def test_unknown_unit_is_rejected():
    with pytest.raises(ValueError, match="unknown unit"):
        to_mm(1.0, "furlong")


def test_presets_include_common_journal_widths():
    assert resolve_preset("nature-double")[0] == pytest.approx(183.0)
    assert resolve_preset("nature-single")[0] == pytest.approx(89.0)
    assert resolve_preset("us-letter") == (pytest.approx(215.9), pytest.approx(279.4))


def test_unknown_preset_lists_known_ones():
    with pytest.raises(KeyError, match="known presets"):
        resolve_preset("nope")


def test_set_size_accepts_inches():
    spec = FigureSpec()
    spec.set_size(7.2, 5.5, unit="in")
    assert spec.width_mm == pytest.approx(182.88)
    assert spec.height_mm == pytest.approx(139.7)


def test_set_size_height_none_preserves_existing_height():
    spec = FigureSpec(width_mm=100.0, height_mm=77.0)
    spec.set_size(183.0)
    assert spec.width_mm == pytest.approx(183.0)
    assert spec.height_mm == pytest.approx(77.0)


def test_check_canvas_is_silent_within_max_width():
    assert check_canvas(FigureSpec(width_mm=183.0, max_width_mm=215.9)) == []


def test_check_canvas_warns_when_wider_than_max():
    warnings = check_canvas(FigureSpec(width_mm=240.0, max_width_mm=215.9))
    assert len(warnings) == 1
    assert "240" in warnings[0] and "215.9" in warnings[0]


def test_check_canvas_guard_disabled_when_max_is_none():
    assert check_canvas(FigureSpec(width_mm=999.0, max_width_mm=None)) == []


def test_check_canvas_rejects_nonpositive_size():
    warnings = check_canvas(FigureSpec(width_mm=0.0, height_mm=-3.0))
    assert any("positive" in w for w in warnings)


def test_preset_in_json_sets_size_but_explicit_width_wins(tmp_path):
    d = {"version": 1, "bundle": "b.h5",
         "figure": {"preset": "nature-double"},
         "panels": [], "groups": [], "annotations": []}
    p = tmp_path / "a.json"
    p.write_text(json.dumps(d))
    assert load_figure(p).width_mm == pytest.approx(183.0)

    d["figure"]["width_mm"] = 120.0
    p.write_text(json.dumps(d))
    assert load_figure(p).width_mm == pytest.approx(120.0)


def test_default_max_width_is_us_letter(tmp_path):
    d = {"version": 1, "bundle": "b.h5", "figure": {"width_mm": 183.0},
         "panels": [], "groups": [], "annotations": []}
    p = tmp_path / "a.json"
    p.write_text(json.dumps(d))
    assert load_figure(p).max_width_mm == pytest.approx(215.9)
