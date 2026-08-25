"""FastAPI backend for the figbuilder editor.

The server is stateless with respect to layout: the browser owns the design
and posts whatever it wants rendered. The server owns data access, matplotlib,
and the tile cache.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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

    @app.put("/api/figure")
    def put_figure(body: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the browser's document.

        Validated BEFORE writing: a malformed post must not corrupt the file
        the researcher's figure is regenerated from. Writes `figure.json`
        only — never `bundle.h5`, which is pipeline-regenerated data.
        """
        import json as _json
        import tempfile

        from figbuilder.figure import load_figure as _load

        # `_load` defaults every missing top-level key (a bare {} parses to
        # a valid, empty FigureSpec), so it alone would silently accept a
        # document that isn't a figure at all and clobber the real one.
        # Require the one field every real document has and a stray blob
        # would not: a `panels` list.
        if not isinstance(body, dict) or not isinstance(body.get("panels"), list):
            raise HTTPException(
                status_code=400,
                detail="figure document must include a 'panels' list") from None

        tmp = Path(tempfile.mkstemp(suffix=".json", dir=str(figure_path.parent))[1])
        try:
            tmp.write_text(_json.dumps(body, indent=2) + "\n")
            _load(tmp)                      # raises on a malformed document
        except Exception as e:              # noqa: BLE001 - surfaced to the client
            tmp.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(e)) from None
        os.replace(tmp, figure_path)        # atomic; never a torn file
        return {"ok": True, "path": str(figure_path)}

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

    # Mount the built UI LAST, so it can never shadow an /api/* route: a
    # StaticFiles(html=True) mount at "/" only takes over once no earlier
    # route (all registered above) has matched.
    _DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
    if (_DIST / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="ui")
    else:
        @app.get("/", response_class=HTMLResponse)
        def _ui_hint() -> str:
            # Do NOT return a bare 404 here. It cost a real debugging round trip.
            return (
                "<h1>figbuilder API</h1>"
                "<p>This port serves <code>/api/*</code> only.</p>"
                "<p>The editor UI is not built. Either run "
                "<code>cd web && npm run dev</code> and open "
                "<a href='http://localhost:5173'>http://localhost:5173</a>, "
                "or run <code>cd web && npm run build</code> to have this "
                "server host it here.</p>"
            )

    return app
