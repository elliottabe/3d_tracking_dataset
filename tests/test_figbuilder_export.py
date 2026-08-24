"""End-to-end export: figure.json + bundle -> svg/png/pdf."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

from figbuilder.bundle import PanelData, read_bundle, write_bundle
from figbuilder.export import export_figure
from figbuilder.figure import load_figure


@pytest.fixture()
def project(tmp_path):
    """A minimal two-panel figure with its bundle, on disk."""
    bundle_path = tmp_path / "bundle.h5"
    write_bundle(bundle_path, meta={"fig_width_mm": 100.0, "fig_height_mm": 60.0},
                 panels={
                     "a": PanelData(type="line",
                                    data={"y": np.linspace(0, 1, 40)}),
                     "b": PanelData(type="line",
                                    data={"y": np.cos(np.linspace(0, 6, 40))}),
                 })
    doc = {
        "version": 1, "bundle": "bundle.h5",
        "figure": {"width_mm": 100.0, "height_mm": 60.0, "dpi": 200},
        "style": {"font.size": 6.0},
        "panels": [
            {"id": "a", "type": "line", "rect": [0.15, 0.58, 0.78, 0.33],
             "data": {"y": {"dataset": "/panels/a/data/y"}},
             "spec": {"series": [{"y": "y", "color": "#38bdf8"}],
                      "ylabel": "wing z (mm)"}},
            {"id": "b", "type": "line", "rect": [0.15, 0.18, 0.78, 0.30],
             "data": {"y": {"dataset": "/panels/b/data/y"}},
             "spec": {"series": [{"y": "y", "color": "#000000"}],
                      "ylabel": "scut z (mm)", "xlabel": "time (s)"}},
        ],
        "groups": [],
        "annotations": [
            {"id": "letter_a", "kind": "text", "pos_mm": [2.0, 56.0],
             "text": "A", "style": {"font_size_pt": 8, "weight": "bold"}},
        ],
    }
    fig_path = tmp_path / "fig.json"
    fig_path.write_text(json.dumps(doc))
    return fig_path, bundle_path


def test_export_writes_svg_containing_both_panels(project, tmp_path):
    fig_path, bundle_path = project
    res = export_figure(load_figure(fig_path), read_bundle(bundle_path),
                        tmp_path / "out.svg", formats=("svg",))
    blob = res.paths["svg"].read_bytes()
    assert b'id="panel_a"' in blob and b'id="panel_b"' in blob


def test_exported_svg_has_real_text_labels(project, tmp_path):
    fig_path, bundle_path = project
    res = export_figure(load_figure(fig_path), read_bundle(bundle_path),
                        tmp_path / "out.svg", formats=("svg",))
    blob = res.paths["svg"].read_bytes()
    assert b"wing z (mm)" in blob and b"time (s)" in blob


def test_exported_svg_includes_the_annotation_layer(project, tmp_path):
    fig_path, bundle_path = project
    res = export_figure(load_figure(fig_path), read_bundle(bundle_path),
                        tmp_path / "out.svg", formats=("svg",))
    assert b'id="letter_a"' in res.paths["svg"].read_bytes()


def test_export_is_deterministic(project, tmp_path):
    """Same inputs -> byte-identical SVG.

    Guards Ruling 10: without a fixed `svg.hashsalt` matplotlib mints random
    marker-definition ids per save and this assertion fails intermittently.
    Note both exports here render FRESH (no cache_dir), so this really does
    exercise reproducibility rather than replaying cached bytes.
    """
    fig_path, bundle_path = project
    spec, bundle = load_figure(fig_path), read_bundle(bundle_path)
    a = export_figure(spec, bundle, tmp_path / "a.svg").paths["svg"].read_bytes()
    b = export_figure(spec, bundle, tmp_path / "b.svg").paths["svg"].read_bytes()
    assert a == b


def test_forced_rcparams_pin_the_svg_hashsalt():
    """A caller's style must not be able to switch determinism off."""
    import matplotlib

    from figbuilder.style import FORCED_RCPARAMS, apply_style
    assert FORCED_RCPARAMS["svg.hashsalt"] == "figbuilder"
    apply_style({"svg.hashsalt": "something-else"})
    assert matplotlib.rcParams["svg.hashsalt"] == "figbuilder"


@pytest.mark.skipif(shutil.which("rsvg-convert") is None,
                    reason="rsvg-convert not on PATH")
def test_export_produces_png_and_pdf(project, tmp_path):
    fig_path, bundle_path = project
    res = export_figure(load_figure(fig_path), read_bundle(bundle_path),
                        tmp_path / "out.svg", formats=("svg", "png", "pdf"))
    assert res.paths["png"].stat().st_size > 0
    assert res.paths["pdf"].read_bytes().startswith(b"%PDF")


def test_export_surfaces_canvas_and_overflow_warnings(tmp_path):
    """A too-wide canvas and a clipped panel both warn, neither raises."""
    bundle_path = tmp_path / "b.h5"
    write_bundle(bundle_path, meta={},
                 panels={"a": PanelData(type="line",
                                        data={"y": np.linspace(0, 1, 20)})})
    doc = {"version": 1, "bundle": "b.h5",
           "figure": {"width_mm": 400.0, "height_mm": 60.0},
           "panels": [{"id": "a", "type": "line", "rect": [0.01, 0.01, 0.5, 0.5],
                       "data": {"y": {"dataset": "/panels/a/data/y"}},
                       "spec": {"series": [{"y": "y"}], "ylabel": "z"}}],
           "groups": [], "annotations": []}
    fp = tmp_path / "f.json"
    fp.write_text(json.dumps(doc))
    res = export_figure(load_figure(fp), read_bundle(bundle_path),
                        tmp_path / "o.svg")
    assert any("exceeds" in w for w in res.warnings)
    assert any("overflow" in w for w in res.warnings)


def test_cli_export_runs_and_writes_the_file(project, tmp_path):
    fig_path, _ = project
    out = tmp_path / "cli.svg"
    r = subprocess.run(
        [sys.executable, "-m", "figbuilder", "export", str(fig_path),
         "-o", str(out), "--format", "svg"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists()
