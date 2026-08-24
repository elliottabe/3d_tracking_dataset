"""Render one panel to a full-canvas SVG tile.

A tile is a complete figure-sized SVG containing exactly one axes, positioned
at that panel's final rect and drawn on a transparent background. Compositing
is therefore a pure overlay: no translate, no scale, no clipping risk. The
cost is mostly-empty vector canvases, which is negligible.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figbuilder.bundle import Bundle
from figbuilder.figure import FigureSpec, PanelSpec
from figbuilder.panels.base import get_panel_type
from figbuilder.style import apply_style

Rect = Tuple[float, float, float, float]


@dataclass
class TileResult:
    svg: bytes
    ink_box: Rect
    cache_hit: bool = False
    overflows: bool = False


def panel_data(bundle: Bundle, panel: PanelSpec) -> Dict[str, np.ndarray]:
    """Resolve a panel's `data` dataset references against a bundle."""
    out: Dict[str, np.ndarray] = {}
    for name, ref in panel.data.items():
        path = ref["dataset"]
        parts = [p for p in path.split("/") if p]
        # /panels/<pid>/<data|assets>/<key>
        if len(parts) != 4 or parts[0] != "panels":
            raise KeyError(f"malformed dataset reference {path!r}")
        _, pid, kind, key = parts
        pd = bundle.panels.get(pid)
        if pd is None:
            raise KeyError(f"{path!r}: no panel {pid!r} in bundle")
        store = pd.data if kind == "data" else pd.assets
        if key not in store:
            raise KeyError(f"{path!r}: no dataset {key!r} in panel {pid!r}")
        arr = store[key]
        sl = ref.get("slice")
        if sl:
            start, _, stop = sl.partition(":")
            arr = arr[int(start or 0):int(stop) if stop else None]
        out[name] = arr
    return out


def _data_digest(data: Dict[str, Any]) -> str:
    """Digest the RESOLVED arrays, not just their dataset paths.

    A bundle can be regenerated with new values at the same dataset path while
    the on-disk tile cache survives, so hashing only the reference strings
    would serve a stale tile for changed data. See Ruling 9 in the SDD ledger.
    """
    h = hashlib.blake2b(digest_size=16)
    for name in sorted(data):
        arr = np.asarray(data[name])
        h.update(name.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def tile_cache_key(fig_spec: FigureSpec, panel: PanelSpec,
                   data: Dict[str, Any]) -> str:
    payload = {
        "size_mm": [fig_spec.width_mm, fig_spec.height_mm],
        "style": fig_spec.style,
        "transparent": fig_spec.transparent,
        "panel": {"id": panel.id, "type": panel.type, "rect": list(panel.rect),
                  "data": panel.data, "spec": panel.spec},
        "data_digest": _data_digest(data),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def _ink_box(fig, ax) -> Rect:
    fig.canvas.draw()
    bb = ax.get_tightbbox().transformed(fig.transFigure.inverted())
    return (float(bb.x0), float(bb.y0), float(bb.width), float(bb.height))


def _atomic_write(path: Path, data: bytes) -> None:
    """Write `data` to `path` atomically via a same-directory temp file + rename.

    `os.replace` is atomic within a filesystem, so a concurrent reader never
    observes a partially-written cache file (Finding 3 hardening).
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def render_tile(fig_spec: FigureSpec, panel: PanelSpec,
                data: Dict[str, Any],
                cache_dir: Optional[str | Path] = None) -> TileResult:
    """Render `panel` onto a full-size transparent canvas."""
    key = tile_cache_key(fig_spec, panel, data)
    cache_path = Path(cache_dir) / f"{key}.svg" if cache_dir else None
    meta_path = Path(cache_dir) / f"{key}.json" if cache_dir else None
    if cache_path and cache_path.exists() and meta_path and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            svg = cache_path.read_bytes()
            ink_box = tuple(meta["ink_box"])
            overflows = meta["overflows"]
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            # Corrupt or unreadable sidecar: treat exactly like a cache miss
            # and fall through to a fresh render, rather than crashing.
            pass
        else:
            return TileResult(svg=svg, ink_box=ink_box,
                              cache_hit=True, overflows=overflows)

    apply_style(fig_spec.style)
    ptype = get_panel_type(panel.type)
    fig = plt.figure(figsize=fig_spec.size_inches())
    try:
        ax = fig.add_axes(list(panel.rect), projection=ptype.projection)
        ax.set_gid(panel.id)
        ptype.draw(ax, data, panel.spec)
        ink = _ink_box(fig, ax)
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", transparent=True)
        svg = buf.getvalue()
    finally:
        plt.close(fig)

    eps = 1e-6
    overflows = (ink[0] < -eps or ink[1] < -eps
                 or ink[0] + ink[2] > 1 + eps or ink[1] + ink[3] > 1 + eps)

    if cache_path and meta_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(cache_path, svg)
        meta_bytes = json.dumps({"ink_box": list(ink),
                                 "overflows": bool(overflows)}).encode("utf-8")
        _atomic_write(meta_path, meta_bytes)
    return TileResult(svg=svg, ink_box=ink, cache_hit=False, overflows=overflows)
