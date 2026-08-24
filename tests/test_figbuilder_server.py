# tests/test_figbuilder_server.py
"""FastAPI routes."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from figbuilder.bundle import PanelData, write_bundle
from figbuilder.server import create_app


@pytest.fixture()
def client(tmp_path):
    write_bundle(tmp_path / "bundle.h5", meta={"fig_width_mm": 100.0},
                 panels={"a": PanelData(type="line",
                                        data={"y": np.linspace(0, 1, 40)})})
    doc = {"version": 1, "bundle": "bundle.h5",
           "figure": {"width_mm": 100.0, "height_mm": 60.0},
           "panels": [{"id": "a", "type": "line", "rect": [0.2, 0.2, 0.6, 0.6],
                       "data": {"y": {"dataset": "/panels/a/data/y"}},
                       "spec": {"series": [{"y": "y"}], "ylabel": "z"}}],
           "groups": [], "annotations": []}
    (tmp_path / "fig.json").write_text(json.dumps(doc))
    return fastapi_testclient.TestClient(
        create_app(tmp_path / "fig.json", tmp_path / "bundle.h5"))


def test_bundle_route_lists_panels_and_dataset_shapes(client):
    body = client.get("/api/bundle").json()
    assert body["panels"][0]["id"] == "a"
    assert body["panels"][0]["data"]["y"]["shape"] == [40]


def test_figure_route_returns_the_design_document(client):
    body = client.get("/api/figure").json()
    assert body["figure"]["width_mm"] == 100.0
    assert body["panels"][0]["id"] == "a"


def test_panel_route_returns_svg_and_ink_box(client):
    body = client.post("/api/panel", json={
        "panel": {"id": "a", "type": "line", "rect": [0.2, 0.2, 0.6, 0.6],
                  "data": {"y": {"dataset": "/panels/a/data/y"}},
                  "spec": {"series": [{"y": "y"}], "ylabel": "z"}}}).json()
    assert body["svg"].startswith("<?xml")
    assert len(body["ink_box"]) == 4
    assert "overflows" in body


def test_panel_route_second_call_reports_a_cache_hit(client):
    payload = {"panel": {"id": "a", "type": "line", "rect": [0.2, 0.2, 0.6, 0.6],
                         "data": {"y": {"dataset": "/panels/a/data/y"}},
                         "spec": {"series": [{"y": "y"}]}}}
    client.post("/api/panel", json=payload)
    assert client.post("/api/panel", json=payload).json()["cache_hit"] is True


def test_panel_types_route_exposes_schemas(client):
    body = client.get("/api/panel-types").json()
    assert any(e["id"] == "line" and "schema" in e for e in body)


def test_dataset_route_decimates_long_arrays(client):
    body = client.get("/api/dataset",
                      params={"path": "/panels/a/data/y", "max": 10}).json()
    assert len(body["values"]) <= 10


def test_unknown_dataset_returns_404(client):
    assert client.get("/api/dataset", params={"path": "/panels/a/data/no"}).status_code == 404


def test_export_route_writes_a_file_and_returns_warnings(client, tmp_path):
    body = client.post("/api/export", json={"formats": ["svg"]}).json()
    assert body["paths"]["svg"].endswith(".svg")
    assert isinstance(body["warnings"], list)
