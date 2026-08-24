"""figure.json + bundle.h5 -> a composed SVG (and derived PNG / PDF).

PNG and PDF are DERIVED from the SVG via `rsvg-convert`, not rendered
separately, so every output format shows exactly the same document. This
matters because the annotation layer is plain SVG that matplotlib's own PDF
backend could not reproduce.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from figbuilder.annot import emit_annotations
from figbuilder.bundle import Bundle
from figbuilder.canvas import check_canvas
from figbuilder.compose import compose
from figbuilder.figure import FigureSpec
from figbuilder.render import panel_data, render_tile


@dataclass
class ExportResult:
    paths: Dict[str, Path] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def svg_to(src: Path, dst: Path, fmt: str, dpi: int = 300) -> None:
    exe = shutil.which("rsvg-convert")
    if exe is None:
        raise RuntimeError(
            "rsvg-convert not found on PATH; it is required for PNG/PDF export")
    cmd = [exe, "-f", fmt, str(src), "-o", str(dst)]
    if fmt == "png":
        cmd[1:1] = ["-d", str(dpi), "-p", str(dpi)]
    subprocess.run(cmd, check=True, capture_output=True)


def export_figure(fig_spec: FigureSpec, bundle: Bundle, out_path: str | Path,
                  formats: Sequence[str] = ("svg",),
                  cache_dir: str | Path | None = None) -> ExportResult:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = ExportResult()
    res.warnings.extend(check_canvas(fig_spec))

    tiles = []
    for panel in sorted(fig_spec.panels, key=lambda p: p.z):
        data = panel_data(bundle, panel)
        tile = render_tile(fig_spec, panel, data, cache_dir=cache_dir)
        if tile.overflows:
            res.warnings.append(
                f"panel {panel.id!r}: labels overflow the canvas "
                f"(ink box {tuple(round(v, 3) for v in tile.ink_box)})")
        tiles.append((panel.id, tile.svg))

    doc = compose(fig_spec, tiles,
                  emit_annotations(fig_spec, fig_spec.annotations))

    svg_path = out_path.with_suffix(".svg")
    svg_path.write_bytes(doc)
    res.paths["svg"] = svg_path

    for fmt in formats:
        if fmt == "svg":
            continue
        dst = out_path.with_suffix(f".{fmt}")
        svg_to(svg_path, dst, fmt, dpi=fig_spec.dpi)
        res.paths[fmt] = dst
    return res
