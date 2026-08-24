"""figure.json load/save/validation."""
from __future__ import annotations

import json

import pytest

from figbuilder.figure import FigureSpec, PanelSpec, load_figure, save_figure


def _minimal() -> dict:
    return {
        "version": 1,
        "bundle": "bundle.h5",
        "figure": {"width_mm": 183.0, "height_mm": 140.0, "dpi": 300,
                   "transparent": True},
        "style": {"font.size": 6.0},
        "panels": [{"id": "wing", "type": "line", "rect": [0.07, 0.6, 0.9, 0.2],
                    "data": {"y": {"dataset": "/panels/wing/data/wingL_z"}},
                    "spec": {"fs": 800.0}}],
        "groups": [],
        "annotations": [],
    }


def test_load_figure_parses_panels_and_size(tmp_path):
    p = tmp_path / "fig.json"
    p.write_text(json.dumps(_minimal()))
    spec = load_figure(p)
    assert spec.width_mm == 183.0
    assert len(spec.panels) == 1
    assert spec.panels[0].id == "wing"
    assert spec.panels[0].rect == (0.07, 0.6, 0.9, 0.2)
    assert spec.panels[0].spec["fs"] == 800.0


def test_size_inches_converts_from_mm(tmp_path):
    p = tmp_path / "fig.json"
    p.write_text(json.dumps(_minimal()))
    w, h = load_figure(p).size_inches()
    assert w == pytest.approx(183.0 / 25.4)
    assert h == pytest.approx(140.0 / 25.4)


def test_save_then_load_roundtrips(tmp_path):
    p = tmp_path / "fig.json"
    p.write_text(json.dumps(_minimal()))
    spec = load_figure(p)
    out = tmp_path / "out.json"
    save_figure(spec, out)
    again = load_figure(out)
    assert again.panels[0].rect == spec.panels[0].rect
    assert again.style == spec.style


def test_duplicate_panel_ids_are_rejected(tmp_path):
    d = _minimal()
    d["panels"].append(dict(d["panels"][0]))
    p = tmp_path / "fig.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="duplicate panel id"):
        load_figure(p)


def test_rect_must_have_four_numbers(tmp_path):
    d = _minimal()
    d["panels"][0]["rect"] = [0.1, 0.2, 0.3]
    p = tmp_path / "fig.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="rect"):
        load_figure(p)


def test_panel_referencing_unknown_group_is_rejected(tmp_path):
    d = _minimal()
    d["panels"][0]["group"] = "row9"
    p = tmp_path / "fig.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="unknown group"):
        load_figure(p)
