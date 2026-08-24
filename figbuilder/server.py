"""FastAPI backend for the figbuilder editor.

The server is stateless with respect to layout: the browser owns the design
and posts whatever it wants rendered. The server owns data access, matplotlib,
and the tile cache.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import figbuilder.panels  # noqa: F401  (registers types)
from figbuilder.bundle import read_bundle
from figbuilder.export import export_figure
from figbuilder.figure import PanelSpec, load_figure
from figbuilder.panels.base import list_panel_types
from figbuilder.render import panel_data, render_tile


def _panel_from_dict(d: dict) -> PanelSpec:
    return PanelSpec(id=d["id"], type=d["type"], rect=tuple(d["rect"]),
                     data=d.get("data", {}), spec=d.get("spec", {}),
                     group=d.get("group"), z=int(d.get("z", 0)))


def create_app(figure_path: str | Path,
               bundle_path: Optional[str | Path] = None) -> FastAPI:
    figure_path = Path(figure_path)
    fig_spec = load_figure(figure_path)
    bundle_path = Path(bundle_path) if bundle_path else figure_path.parent / fig_spec.bundle
    bundle = read_bundle(bundle_path)
    cache_dir = figure_path.parent / ".figbuilder_cache"

    app = FastAPI(title="figbuilder")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/bundle")
    def get_bundle() -> Dict[str, Any]:
        return {"meta": bundle.meta, "panels": [
            {"id": pid, "type": pd.type,
             "data": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                      for k, v in sorted(pd.data.items())},
             "assets": {k: {"shape": list(v.shape)}
                        for k, v in sorted(pd.assets.items())},
             "attrs": pd.attrs}
            for pid, pd in sorted(bundle.panels.items())]}

    @app.get("/api/figure")
    def get_figure() -> Dict[str, Any]:
        import json
        return json.loads(figure_path.read_text())

    @app.get("/api/panel-types")
    def get_panel_types() -> List[Dict[str, Any]]:
        return list_panel_types()

    @app.post("/api/panel")
    def post_panel(body: Dict[str, Any]) -> Dict[str, Any]:
        panel = _panel_from_dict(body["panel"])
        spec = fig_spec
        if "figure" in body:
            spec.width_mm = float(body["figure"].get("width_mm", spec.width_mm))
            spec.height_mm = float(body["figure"].get("height_mm", spec.height_mm))
        try:
            data = panel_data(bundle, panel)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from None
        tile = render_tile(spec, panel, data, cache_dir=cache_dir)
        return {"svg": tile.svg.decode("utf-8"), "ink_box": list(tile.ink_box),
                "cache_hit": tile.cache_hit, "overflows": tile.overflows}

    @app.get("/api/dataset")
    def get_dataset(path: str, max: int = 4000) -> Dict[str, Any]:
        parts = [p for p in path.split("/") if p]
        if len(parts) != 4 or parts[0] != "panels":
            raise HTTPException(status_code=400, detail=f"malformed path {path!r}")
        _, pid, kind, key = parts
        pd = bundle.panels.get(pid)
        store = (pd.data if kind == "data" else pd.assets) if pd else {}
        if key not in store:
            raise HTTPException(status_code=404, detail=f"no dataset {path!r}")
        arr = np.asarray(store[key])
        flat = arr.reshape(-1) if arr.dtype.names is None else arr
        if arr.dtype.names is None and flat.size > max:
            idx = np.linspace(0, flat.size - 1, max).astype(int)
            flat = flat[idx]
        return {"shape": list(arr.shape), "dtype": str(arr.dtype),
                "values": np.asarray(flat).tolist()}

    @app.post("/api/export")
    def post_export(body: Dict[str, Any]) -> Dict[str, Any]:
        # The browser owns the design, so an export request may carry the
        # current in-editor document. Persist it before rendering, so the
        # exported SVG and figure.json on disk can never disagree.
        if "figure_json" in body:
            import json
            figure_path.write_text(json.dumps(body["figure_json"], indent=2) + "\n")
        spec = load_figure(figure_path)
        out = figure_path.with_suffix(".svg")
        res = export_figure(spec, bundle, out,
                            formats=tuple(body.get("formats", ["svg"])),
                            cache_dir=cache_dir)
        return {"paths": {k: str(v) for k, v in res.paths.items()},
                "warnings": res.warnings}

    return app
