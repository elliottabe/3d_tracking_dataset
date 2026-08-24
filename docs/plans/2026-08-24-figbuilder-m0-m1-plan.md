# figbuilder M0–M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the headless half of figbuilder — a frozen data bundle, a panel
registry, a matplotlib tile renderer, an SVG compositor, and an export CLI that
regenerates Figure 4 of the courtship-song paper from committed inputs — plus a
read-only browser canvas that proves tile composition works.

**Architecture:** matplotlib is the only chart renderer. Each panel is rendered
onto its own transparent, figure-sized SVG canvas with the axes already at its
final rect; compositing is then a pure overlay (no transforms), with per-panel
id namespacing to prevent collisions. Annotations are plain SVG appended after
the tiles. A FastAPI server serves tiles to a browser canvas.

**Tech Stack:** Python 3.10, matplotlib 3.10.8 (Agg), h5py 3.16, lxml 6.0,
FastAPI + uvicorn, `rsvg-convert` for PNG/PDF, pytest 9. Front-end: Vite +
React + TypeScript.

**Spec:** `docs/plans/2026-08-24-figbuilder-design.md` (commit `da2a4f2`)

## Global Constraints

- Python 3.10 syntax; every module starts `from __future__ import annotations`.
- Tests live flat in `tests/test_<topic>.py`, matching repo convention.
- Every test module that imports matplotlib does `matplotlib.use('Agg')` BEFORE
  `import matplotlib.pyplot`. See `tests/test_assemble_figure.py`.
- `utils/courtship_figure_panels.py` is **consumed unmodified**. No edits to it
  in this plan.
- rcParams for all rendering: `svg.fonttype='none'`, `pdf.fonttype=42`,
  `ps.fonttype=42`. Text must export as real `<text>`.
- All rects are `[x, y, w, h]` in **root-figure fractions**, matching
  `fig.add_axes` semantics.
- Generated data (`bundle.h5`, `.figbuilder_cache/`) lives under `figures/`,
  which is gitignored. Never commit an `.h5`, `.png`, or `.svg` output.
- `figure.json` design files ARE committed, under `figures/paper_figures/`.
- New runtime dependency: `fastapi` only. `uvicorn`, `lxml`, `h5py`,
  `matplotlib`, `Pillow`, and `rsvg-convert` are already in the env.

## Deviations from the spec (validated by probe, 2026-08-24)

These supersede the spec text where they conflict:

1. **Full-canvas tiles.** The spec described placing tiles with
   `translate()/scale()`. Instead each tile is rendered at the FULL figure size
   with its axes at the final rect, so compositing is a pure overlay. Verified:
   two overlaid tiles land correctly with no clipping.
2. **`rsvg-convert` replaces cairosvg/Inkscape** for PNG and PDF. It is already
   on PATH in the `3d_tracking` env. Verified: both formats produced.
3. **Id collisions are severe** — 35 of 39 ids collide between two independent
   tiles (`figure_1`, `axes_1`, `line2d_1`, …). Namespacing is mandatory, not
   defensive.

## Scope boundaries — deferred on purpose

These are spec requirements that belong to LATER milestones. They are listed so
their absence here is not mistaken for an oversight.

| Spec item | Deferred to | Why |
|---|---|---|
| Remaining generic panel types (`scatter`, `hist_kde`, `violin`, `box`, `strip`, `bar`, `heatmap`) | M4 | Task 3 ships `line`, `image`, `blank` — enough to prove the registry. The rest are additive, each ~20 lines with the same shape. |
| Group solver (children's rects from a gutter) | M2 | `figure.json` stores every panel's SOLVED rect, so rendering never needs the solver. It is only needed once panels can be dragged. |
| Layout math — snap, align, distribute | M2 | Pure TypeScript, browser-only, no Python counterpart. |
| Panel selection, drag, resize, numeric entry, undo | M2 | M1 is deliberately read-only. |
| Annotation EDITING (create, drag, restyle in the browser) | M3 | Task 8 + Task 14 build the emitters; authoring UI comes later. Task 11 seeds letters programmatically so the export is complete without it. |
| Data rebinding UI, panel add/delete | M4 | `/api/bundle` and `/api/panel-types` already expose what the UI will need. |
| Live pipeline mode | M5 | Optional; the bundle contract is the abstraction that makes it swappable. |

---

---

## File Structure

| File | Responsibility |
|---|---|
| `figbuilder/__init__.py` | Package marker, version |
| `figbuilder/style.py` | rcParams application; single source of render style |
| `figbuilder/bundle.py` | Read/write `bundle.h5`; array, segment-table, image-asset codecs |
| `figbuilder/figure.py` | `figure.json` dataclasses, validation, load/save |
| `figbuilder/panels/base.py` | `PanelType`, `@register`, registry lookup |
| `figbuilder/panels/generic.py` | `line`, `image`, `violin`, `blank` panel types |
| `figbuilder/panels/courtship.py` | Adapters over `utils.courtship_figure_panels` |
| `figbuilder/render.py` | Panel → full-canvas SVG tile + ink box; disk cache |
| `figbuilder/svgutil.py` | Id namespacing, SVG parsing helpers |
| `figbuilder/annot.py` | Annotation objects → SVG elements |
| `figbuilder/compose.py` | Tiles + annotations → one SVG document |
| `figbuilder/export.py` | `figure.json` → `.svg` / `.png` / `.pdf` |
| `figbuilder/server.py` | FastAPI routes |
| `figbuilder/cli.py` | `serve`, `export`, `bundle-info` subcommands |
| `scripts/figures/export_fig4_bundle.py` | Run the heavy pipeline → `bundle.h5` |
| `scripts/figures/seed_fig4_layout.py` | Current `assemble_figure` → `fig4.json` |
| `tests/conftest.py` | `synthetic_bundle` fixture |
| `web/` | Vite + React read-only canvas |

---

## Task 1: Bundle I/O

**Files:**
- Create: `figbuilder/__init__.py`, `figbuilder/bundle.py`
- Test: `tests/test_figbuilder_bundle.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `write_bundle(path, meta: dict, panels: dict[str, PanelData]) -> None`
  - `read_bundle(path) -> Bundle`
  - `Bundle` with `.meta: dict`, `.panels: dict[str, PanelData]`
  - `PanelData` with `.type: str`, `.data: dict[str, np.ndarray]`, `.assets: dict[str, np.ndarray]`, `.attrs: dict`
  - `segments_to_array(list[dict]) -> np.ndarray` (structured, fields `start:i8 end:i8 type:S8`)
  - `array_to_segments(np.ndarray) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figbuilder_bundle.py
"""Round-trip tests for the figbuilder bundle format."""
from __future__ import annotations

import numpy as np

from figbuilder.bundle import (
    PanelData, array_to_segments, read_bundle, segments_to_array, write_bundle,
)


def test_bundle_roundtrip_preserves_arrays_and_meta(tmp_path):
    p = tmp_path / "b.h5"
    panels = {
        "wing": PanelData(
            type="courtship.wing_z",
            data={"t_ms": np.linspace(0, 100, 51),
                  "wingL_z": np.random.default_rng(0).normal(size=51)},
            assets={},
            attrs={"fs": 800.0},
        ),
    }
    write_bundle(p, meta={"fig_width_mm": 183.0, "fig_height_mm": 140.0}, panels=panels)
    b = read_bundle(p)

    assert b.meta["fig_width_mm"] == 183.0
    assert set(b.panels) == {"wing"}
    assert b.panels["wing"].type == "courtship.wing_z"
    assert b.panels["wing"].attrs["fs"] == 800.0
    np.testing.assert_allclose(b.panels["wing"].data["t_ms"], panels["wing"].data["t_ms"])


def test_bundle_roundtrip_preserves_uint8_image_assets(tmp_path):
    p = tmp_path / "b.h5"
    img = np.random.default_rng(1).integers(0, 255, (12, 10, 3), dtype=np.uint8)
    panels = {"vid": PanelData(type="image", data={}, assets={"frame": img}, attrs={})}
    write_bundle(p, meta={}, panels=panels)
    b = read_bundle(p)
    got = b.panels["vid"].assets["frame"]
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, img)


def test_segment_table_roundtrip():
    segs = [{"start": 0, "end": 12, "type": "pulse"},
            {"start": 40, "end": 91, "type": "sine"}]
    arr = segments_to_array(segs)
    assert arr.dtype.names == ("start", "end", "type")
    assert array_to_segments(arr) == segs


def test_empty_segment_table_roundtrips_to_empty_list():
    arr = segments_to_array([])
    assert arr.shape == (0,)
    assert array_to_segments(arr) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figbuilder_bundle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'figbuilder'`

- [ ] **Step 3: Write minimal implementation**

```python
# figbuilder/__init__.py
"""figbuilder — interactive paper-figure composer."""
from __future__ import annotations

__version__ = "0.1.0"
```

```python
# figbuilder/bundle.py
"""Read and write the figbuilder data bundle (`bundle.h5`).

The bundle is the frozen, plot-ready output of a heavy analysis pipeline.
It holds one group per panel containing numeric arrays (`data/`) and baked
raster assets (`assets/`), so that interactive editing never has to re-run
video decoding, MuJoCo, or the song-analysis pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import h5py
import numpy as np

SEGMENT_DTYPE = np.dtype([("start", "<i8"), ("end", "<i8"), ("type", "S8")])


@dataclass
class PanelData:
    """One panel's frozen inputs."""

    type: str
    data: Dict[str, np.ndarray] = field(default_factory=dict)
    assets: Dict[str, np.ndarray] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bundle:
    meta: Dict[str, Any] = field(default_factory=dict)
    panels: Dict[str, PanelData] = field(default_factory=dict)


def segments_to_array(segments: Iterable[dict]) -> np.ndarray:
    """Convert matplotlib-panel segment dicts to a structured array."""
    segs = list(segments)
    out = np.zeros(len(segs), dtype=SEGMENT_DTYPE)
    for i, s in enumerate(segs):
        out[i] = (int(s["start"]), int(s["end"]),
                  str(s.get("type", "")).encode("utf-8"))
    return out


def array_to_segments(arr: np.ndarray) -> List[dict]:
    """Inverse of :func:`segments_to_array`."""
    return [{"start": int(r["start"]), "end": int(r["end"]),
             "type": r["type"].decode("utf-8")} for r in arr]


def _write_attrs(obj, attrs: Dict[str, Any]) -> None:
    for k, v in attrs.items():
        obj.attrs[k] = v


def write_bundle(path: str | Path, meta: Dict[str, Any],
                 panels: Dict[str, PanelData]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        _write_attrs(f.create_group("meta"), meta)
        pg = f.create_group("panels")
        for pid, pd in panels.items():
            g = pg.create_group(pid)
            g.attrs["type"] = pd.type
            _write_attrs(g, pd.attrs)
            dg = g.create_group("data")
            for k, v in pd.data.items():
                dg.create_dataset(k, data=np.asarray(v))
            ag = g.create_group("assets")
            for k, v in pd.assets.items():
                ag.create_dataset(k, data=np.asarray(v), compression="gzip")


def _read_attrs(obj) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obj.attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


def read_bundle(path: str | Path) -> Bundle:
    with h5py.File(Path(path), "r") as f:
        meta = _read_attrs(f["meta"]) if "meta" in f else {}
        panels: Dict[str, PanelData] = {}
        for pid, g in f.get("panels", {}).items():
            attrs = _read_attrs(g)
            ptype = attrs.pop("type")
            panels[pid] = PanelData(
                type=ptype,
                data={k: np.asarray(v) for k, v in g["data"].items()},
                assets={k: np.asarray(v) for k, v in g["assets"].items()},
                attrs=attrs,
            )
    return Bundle(meta=meta, panels=panels)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figbuilder_bundle.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add figbuilder/__init__.py figbuilder/bundle.py tests/test_figbuilder_bundle.py
git commit -m "feat(figbuilder): bundle.h5 read/write with segment and asset codecs"
```

---

## Task 2: figure.json schema

**Files:**
- Create: `figbuilder/figure.py`
- Test: `tests/test_figbuilder_figure_spec.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `FigureSpec` with `.width_mm: float`, `.height_mm: float`, `.dpi: int`, `.transparent: bool`, `.style: dict`, `.bundle: str`, `.panels: list[PanelSpec]`, `.groups: list[GroupSpec]`, `.annotations: list[dict]`
  - `PanelSpec` with `.id: str`, `.type: str`, `.rect: tuple[float,float,float,float]`, `.data: dict[str, dict]`, `.spec: dict`, `.group: str | None`, `.z: int`
  - `GroupSpec` with `.id: str`, `.axis: str`, `.gutter_mm: float`, `.equal: bool`, `.rect: tuple`
  - `load_figure(path) -> FigureSpec`, `save_figure(spec, path) -> None`
  - `FigureSpec.size_inches() -> tuple[float, float]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figbuilder_figure_spec.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figbuilder_figure_spec.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'figbuilder.figure'`

- [ ] **Step 3: Write minimal implementation**

```python
# figbuilder/figure.py
"""The `figure.json` design document: panels, groups, annotations, style.

`figure.json` is the committed, human-readable design artifact. It is
deliberately separate from `bundle.h5` (the data): the bundle is large and
regenerated by the pipeline, the design is small and version-controlled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MM_PER_INCH = 25.4

Rect = Tuple[float, float, float, float]


def _as_rect(value: Any, where: str) -> Rect:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{where}: rect must be 4 numbers [x, y, w, h], got {value!r}")
    return tuple(float(v) for v in value)  # type: ignore[return-value]


@dataclass
class PanelSpec:
    id: str
    type: str
    rect: Rect
    data: Dict[str, dict] = field(default_factory=dict)
    spec: Dict[str, Any] = field(default_factory=dict)
    group: Optional[str] = None
    z: int = 0


@dataclass
class GroupSpec:
    id: str
    axis: str = "x"
    gutter_mm: float = 1.0
    equal: bool = False
    rect: Rect = (0.0, 0.0, 1.0, 1.0)


@dataclass
class FigureSpec:
    width_mm: float = 183.0
    height_mm: float = 140.0
    dpi: int = 300
    transparent: bool = True
    bundle: str = "bundle.h5"
    style: Dict[str, Any] = field(default_factory=dict)
    panels: List[PanelSpec] = field(default_factory=list)
    groups: List[GroupSpec] = field(default_factory=list)
    annotations: List[dict] = field(default_factory=list)

    def size_inches(self) -> Tuple[float, float]:
        return (self.width_mm / MM_PER_INCH, self.height_mm / MM_PER_INCH)

    def panel(self, panel_id: str) -> PanelSpec:
        for p in self.panels:
            if p.id == panel_id:
                return p
        raise KeyError(panel_id)


def _parse(d: dict) -> FigureSpec:
    figd = d.get("figure", {})
    groups = [GroupSpec(id=g["id"], axis=g.get("axis", "x"),
                        gutter_mm=float(g.get("gutter_mm", 1.0)),
                        equal=bool(g.get("equal", False)),
                        rect=_as_rect(g.get("rect", [0, 0, 1, 1]), f"group {g['id']}"))
              for g in d.get("groups", [])]
    group_ids = {g.id for g in groups}

    panels: List[PanelSpec] = []
    seen: set[str] = set()
    for p in d.get("panels", []):
        pid = p["id"]
        if pid in seen:
            raise ValueError(f"duplicate panel id: {pid!r}")
        seen.add(pid)
        grp = p.get("group")
        if grp is not None and grp not in group_ids:
            raise ValueError(f"panel {pid!r} references unknown group {grp!r}")
        panels.append(PanelSpec(
            id=pid, type=p["type"], rect=_as_rect(p["rect"], f"panel {pid}"),
            data=p.get("data", {}), spec=p.get("spec", {}),
            group=grp, z=int(p.get("z", 0)),
        ))

    return FigureSpec(
        width_mm=float(figd.get("width_mm", 183.0)),
        height_mm=float(figd.get("height_mm", 140.0)),
        dpi=int(figd.get("dpi", 300)),
        transparent=bool(figd.get("transparent", True)),
        bundle=d.get("bundle", "bundle.h5"),
        style=d.get("style", {}),
        panels=panels, groups=groups,
        annotations=list(d.get("annotations", [])),
    )


def _unparse(spec: FigureSpec) -> dict:
    return {
        "version": 1,
        "bundle": spec.bundle,
        "figure": {"width_mm": spec.width_mm, "height_mm": spec.height_mm,
                   "dpi": spec.dpi, "transparent": spec.transparent},
        "style": spec.style,
        "panels": [{"id": p.id, "type": p.type, "rect": list(p.rect),
                    "data": p.data, "spec": p.spec, "group": p.group, "z": p.z}
                   for p in spec.panels],
        "groups": [{"id": g.id, "axis": g.axis, "gutter_mm": g.gutter_mm,
                    "equal": g.equal, "rect": list(g.rect)} for g in spec.groups],
        "annotations": spec.annotations,
    }


def load_figure(path: str | Path) -> FigureSpec:
    return _parse(json.loads(Path(path).read_text()))


def save_figure(spec: FigureSpec, path: str | Path) -> None:
    Path(path).write_text(json.dumps(_unparse(spec), indent=2) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figbuilder_figure_spec.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add figbuilder/figure.py tests/test_figbuilder_figure_spec.py
git commit -m "feat(figbuilder): figure.json dataclasses with validation"
```

---

## Task 3: Style + panel registry with generic types

**Files:**
- Create: `figbuilder/style.py`, `figbuilder/panels/__init__.py`, `figbuilder/panels/base.py`, `figbuilder/panels/generic.py`
- Test: `tests/test_figbuilder_panels.py`

**Interfaces:**
- Consumes: `figbuilder.bundle.PanelData` (Task 1)
- Produces:
  - `figbuilder.style.apply_style(style: dict) -> None`
  - `PanelType` base class with class attrs `id: str`, `label: str`, `needs: list[str]`, `schema: dict`, and method `draw(self, ax, data: dict, spec: dict) -> None`
  - `@register` decorator; `get_panel_type(type_id) -> PanelType`; `list_panel_types() -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figbuilder_panels.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figbuilder_panels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'figbuilder.style'`

- [ ] **Step 3: Write minimal implementation**

```python
# figbuilder/style.py
"""Single source of truth for render style.

Every renderer — tile, export, and test — routes through `apply_style`, so a
figure rendered in the browser and one exported headlessly cannot drift.
"""
from __future__ import annotations

from typing import Any, Dict

import matplotlib

#: Non-negotiable: text must export as real <text>, and PDF must embed Type-42.
FORCED_RCPARAMS: Dict[str, Any] = {
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

DEFAULT_RCPARAMS: Dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
}


def apply_style(style: Dict[str, Any] | None = None) -> None:
    """Apply default rcParams, then the figure's overrides, then the forced set."""
    matplotlib.rcParams.update(DEFAULT_RCPARAMS)
    if style:
        matplotlib.rcParams.update(style)
    matplotlib.rcParams.update(FORCED_RCPARAMS)
```

```python
# figbuilder/panels/__init__.py
"""Panel type registry. Importing this package registers all built-in types."""
from __future__ import annotations

from figbuilder.panels import courtship, generic  # noqa: F401
from figbuilder.panels.base import (  # noqa: F401
    PanelType, get_panel_type, list_panel_types, register,
)
```

> **Note for the implementer:** `figbuilder/panels/__init__.py` imports
> `courtship`, which does not exist until Task 4. Until then, create
> `figbuilder/panels/courtship.py` as an empty module with a docstring so the
> import resolves. Task 4 fills it in.

```python
# figbuilder/panels/base.py
"""Panel type registry.

A panel type owns three things: a JSON schema (which drives the front-end
properties form), a list of the bundle datasets it needs, and a `draw` method
that renders into one matplotlib axes. Adding a panel type is Python-only —
the UI is generated from the schema.
"""
from __future__ import annotations

from typing import Any, Dict, List, Type

import matplotlib.pyplot as plt


class PanelType:
    """Base class for all panel types."""

    id: str = ""
    label: str = ""
    #: Names of bundle datasets this type requires in `data`.
    needs: List[str] = []
    #: JSON Schema for the `spec` object; drives the properties UI.
    schema: Dict[str, Any] = {"type": "object", "properties": {}}

    def draw(self, ax: plt.Axes, data: Dict[str, Any], spec: Dict[str, Any]) -> None:
        raise NotImplementedError


_REGISTRY: Dict[str, PanelType] = {}


def register(cls: Type[PanelType]) -> Type[PanelType]:
    """Class decorator: add a panel type to the registry."""
    if not cls.id:
        raise ValueError(f"{cls.__name__} must set a non-empty `id`")
    _REGISTRY[cls.id] = cls()
    return cls


def get_panel_type(type_id: str) -> PanelType:
    try:
        return _REGISTRY[type_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"unknown panel type {type_id!r}; known types: {known}") from None


def list_panel_types() -> List[Dict[str, Any]]:
    return [{"id": p.id, "label": p.label, "needs": list(p.needs),
             "schema": p.schema} for p in sorted(_REGISTRY.values(), key=lambda p: p.id)]
```

```python
# figbuilder/panels/generic.py
"""Generic, data-agnostic panel types."""
from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from figbuilder.panels.base import PanelType, register


def _decorate(ax: plt.Axes, spec: Dict[str, Any]) -> None:
    if spec.get("xlabel"):
        ax.set_xlabel(spec["xlabel"])
    if spec.get("ylabel"):
        ax.set_ylabel(spec["ylabel"])
    if spec.get("title"):
        ax.set_title(spec["title"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


@register
class LinePanel(PanelType):
    id = "line"
    label = "Line / trace"
    needs = []
    schema = {
        "type": "object",
        "properties": {
            "series": {"type": "array", "title": "Series", "items": {
                "type": "object", "properties": {
                    "x": {"type": "string", "title": "x dataset"},
                    "y": {"type": "string", "title": "y dataset"},
                    "label": {"type": "string"},
                    "color": {"type": "string", "format": "color"},
                    "lw": {"type": "number", "default": 0.7},
                }}},
            "xlabel": {"type": "string"},
            "ylabel": {"type": "string"},
            "title": {"type": "string"},
            "legend": {"type": "boolean", "default": False},
        },
    }

    def draw(self, ax, data, spec):
        for s in spec.get("series", []):
            y = np.asarray(data[s["y"]], dtype=float)
            x = np.asarray(data[s["x"]], dtype=float) if s.get("x") else np.arange(y.size)
            ax.plot(x, y, label=s.get("label", ""),
                    color=s.get("color"), lw=float(s.get("lw", 0.7)))
        if spec.get("legend"):
            ax.legend()
        _decorate(ax, spec)


@register
class ImagePanel(PanelType):
    id = "image"
    label = "Image / rendered frame"
    needs = []
    schema = {
        "type": "object",
        "properties": {
            "asset": {"type": "string", "title": "Asset name"},
            "interpolation": {"type": "string", "default": "nearest"},
        },
    }

    def draw(self, ax, data, spec):
        ax.imshow(np.asarray(data[spec["asset"]]),
                  interpolation=spec.get("interpolation", "nearest"))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)


@register
class BlankPanel(PanelType):
    id = "blank"
    label = "Blank (spacer / hand graphic)"
    needs = []
    schema = {"type": "object", "properties": {}}

    def draw(self, ax, data, spec):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.patch.set_alpha(0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figbuilder_panels.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add figbuilder/style.py figbuilder/panels/ tests/test_figbuilder_panels.py
git commit -m "feat(figbuilder): style module and panel registry with generic types"
```

---

## Task 4: Canvas size — presets, units, and a soft max-width guard

Paper figures are sized to a journal column, not to taste. This task makes the
overall canvas an explicit, first-class, unit-flexible property with named
presets and a **non-blocking** guard against exceeding a page width. The guard
warns; it never refuses. `max_width_mm: null` disables it entirely.

**Files:**
- Create: `figbuilder/canvas.py`
- Modify: `figbuilder/figure.py` (add `preset`, `max_width_mm`, `set_size`)
- Test: `tests/test_figbuilder_canvas.py`

**Interfaces:**
- Consumes: `FigureSpec` (Task 2)
- Produces:
  - `figbuilder.canvas.CANVAS_PRESETS: dict[str, tuple[float, float | None]]` — width/height in mm
  - `to_mm(value: float, unit: str) -> float` (`"mm"`, `"cm"`, `"in"`, `"pt"`)
  - `from_mm(value_mm: float, unit: str) -> float`
  - `resolve_preset(name: str) -> tuple[float, float | None]`
  - `check_canvas(spec: FigureSpec) -> list[str]` — human-readable warnings
  - `FigureSpec.set_size(width, height=None, unit="mm") -> None`
  - `FigureSpec.preset: str | None`, `FigureSpec.max_width_mm: float | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figbuilder_canvas.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figbuilder_canvas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'figbuilder.canvas'`

- [ ] **Step 3: Write minimal implementation**

```python
# figbuilder/canvas.py
"""Canvas sizing for paper figures.

Journal figures are sized to a column width, so the canvas is an explicit
property rather than an afterthought. Sizes are stored in millimetres (the
unit journals specify) but may be entered in mm, cm, inches, or points.

The max-width guard is deliberately a WARNING, never an error: house styles
vary, posters and talk slides are legitimately wider, and a tool that refuses
to render is worse than one that tells you what it noticed.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

MM_PER_INCH = 25.4
MM_PER_PT = MM_PER_INCH / 72.0

#: US Letter width — the widest a figure can be and still print on a page.
DEFAULT_MAX_WIDTH_MM = 8.5 * MM_PER_INCH  # 215.9

_UNIT_TO_MM: Dict[str, float] = {
    "mm": 1.0,
    "cm": 10.0,
    "in": MM_PER_INCH,
    "pt": MM_PER_PT,
}

#: (width_mm, height_mm | None). None height means "you choose".
CANVAS_PRESETS: Dict[str, Tuple[float, Optional[float]]] = {
    "nature-single":   (89.0, None),
    "nature-double":   (183.0, None),
    "cell-1.5col":     (120.0, None),
    "elsevier-single": (90.0, None),
    "elsevier-double": (190.0, None),
    "plos-single":     (83.0, None),
    "plos-double":     (173.0, None),
    "us-letter":       (215.9, 279.4),
    "a4":              (210.0, 297.0),
}


def to_mm(value: float, unit: str = "mm") -> float:
    try:
        return float(value) * _UNIT_TO_MM[unit]
    except KeyError:
        raise ValueError(
            f"unknown unit {unit!r}; expected one of {sorted(_UNIT_TO_MM)}"
        ) from None


def from_mm(value_mm: float, unit: str = "mm") -> float:
    try:
        return float(value_mm) / _UNIT_TO_MM[unit]
    except KeyError:
        raise ValueError(
            f"unknown unit {unit!r}; expected one of {sorted(_UNIT_TO_MM)}"
        ) from None


def resolve_preset(name: str) -> Tuple[float, Optional[float]]:
    try:
        return CANVAS_PRESETS[name]
    except KeyError:
        raise KeyError(
            f"unknown canvas preset {name!r}; known presets: "
            f"{', '.join(sorted(CANVAS_PRESETS))}"
        ) from None


def check_canvas(spec) -> List[str]:
    """Return human-readable warnings about a figure's canvas size.

    Never raises for an oversized canvas — the caller decides what to do.
    """
    out: List[str] = []
    if spec.width_mm <= 0 or spec.height_mm <= 0:
        out.append(
            f"canvas size must be positive, got "
            f"{spec.width_mm:g} x {spec.height_mm:g} mm"
        )
        return out
    if spec.max_width_mm is not None and spec.width_mm > spec.max_width_mm:
        out.append(
            f"canvas width {spec.width_mm:g} mm "
            f"({from_mm(spec.width_mm, 'in'):.2f} in) exceeds the "
            f"{spec.max_width_mm:g} mm ({from_mm(spec.max_width_mm, 'in'):.2f} in) "
            f"limit; set \"max_width_mm\": null to silence this"
        )
    return out
```

Then modify `figbuilder/figure.py`. Add to the imports:

```python
from figbuilder.canvas import DEFAULT_MAX_WIDTH_MM, resolve_preset, to_mm
```

Add two fields to `FigureSpec` (after `transparent`):

```python
    preset: Optional[str] = None
    max_width_mm: Optional[float] = DEFAULT_MAX_WIDTH_MM
```

Add this method to `FigureSpec`:

```python
    def set_size(self, width: float, height: Optional[float] = None,
                 unit: str = "mm") -> None:
        """Set the canvas size in mm, cm, inches, or points.

        `height=None` keeps the current height, so width can be changed alone.
        """
        self.width_mm = to_mm(width, unit)
        if height is not None:
            self.height_mm = to_mm(height, unit)
```

In `_parse`, replace the `FigureSpec(...)` construction's width/height lines so
a preset seeds the size and any explicit value overrides it:

```python
    preset = figd.get("preset")
    if preset is not None:
        pw, ph = resolve_preset(preset)
    else:
        pw, ph = 183.0, 140.0
    width_mm = float(figd["width_mm"]) if "width_mm" in figd else pw
    height_mm = float(figd["height_mm"]) if "height_mm" in figd else (
        ph if ph is not None else 140.0)

    max_width = figd["max_width_mm"] if "max_width_mm" in figd else DEFAULT_MAX_WIDTH_MM
```

and pass `width_mm=width_mm, height_mm=height_mm, preset=preset,
max_width_mm=(None if max_width is None else float(max_width))` instead of the
old `width_mm=`/`height_mm=` arguments.

In `_unparse`, extend the `"figure"` dict:

```python
        "figure": {"width_mm": spec.width_mm, "height_mm": spec.height_mm,
                   "dpi": spec.dpi, "transparent": spec.transparent,
                   "preset": spec.preset, "max_width_mm": spec.max_width_mm},
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figbuilder_canvas.py tests/test_figbuilder_figure_spec.py -v`
Expected: 12 passed in the canvas module, 6 still passing in the spec module

- [ ] **Step 5: Commit**

```bash
git add figbuilder/canvas.py figbuilder/figure.py tests/test_figbuilder_canvas.py
git commit -m "feat(figbuilder): canvas presets, unit-flexible sizing, soft max-width guard"
```

---

## Task 5: Courtship panel adapters

Wraps the ten existing `panel_*` functions so they become registry types.
`utils/courtship_figure_panels.py` is **not modified**.

**Files:**
- Modify: `figbuilder/panels/courtship.py` (created empty in Task 3)
- Test: `tests/test_figbuilder_courtship_panels.py`

**Interfaces:**
- Consumes: `PanelType`, `register` (Task 3); `array_to_segments` (Task 1)
- Produces: registered types `courtship.wing_z`, `courtship.scutellum_z`,
  `courtship.male_pitch`, `courtship.pitch_violin`, `courtship.zheight`,
  `courtship.sine_inphase`, `courtship.angle_density`, `courtship.wing_polar`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figbuilder_courtship_panels.py
"""Smoke tests: every courtship adapter draws from its declared `needs`."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import figbuilder.panels  # noqa: F401  (registers types)
from figbuilder.bundle import segments_to_array
from figbuilder.panels.base import get_panel_type

RNG = np.random.default_rng(0)
T = 400
FS = 800.0


def _t_ms():
    return np.arange(T) / FS * 1000.0


def _segs():
    return segments_to_array([{"start": 20, "end": 90, "type": "pulse"},
                              {"start": 150, "end": 260, "type": "sine"}])


CASES = {
    "courtship.wing_z": (
        {"t_ms": _t_ms(), "wingL_z": RNG.normal(size=T), "wingR_z": RNG.normal(size=T),
         "seg_L": _segs(), "seg_R": _segs()},
        {"fs": FS, "frame_range": [0, T], "time_unit": "s"},
    ),
    "courtship.scutellum_z": (
        {"t_ms": _t_ms(), "scutellum_z": RNG.normal(size=T), "segments": _segs()},
        {"fs": FS, "frame_range": [0, T], "time_unit": "s"},
    ),
    "courtship.male_pitch": (
        {"t_ms": _t_ms(), "male_pitch": RNG.normal(size=T) * 10,
         "target_pitch": RNG.normal(size=T) * 10},
        {"fs": FS, "frame_range": [0, T], "time_unit": "s"},
    ),
    "courtship.pitch_violin": (
        {"per_bout": np.abs(RNG.normal(size=37)) * 20},
        {"exemplar_idx": 3},
    ),
    "courtship.zheight": (
        {"pulse_z": RNG.normal(size=25), "sine_z": RNG.normal(size=25),
         "walking_z": RNG.normal(size=25)},
        {"kind": "violin"},
    ),
    "courtship.angle_density": (
        {"ext_pulse": np.abs(RNG.normal(size=500)) * 30,
         "ext_sine": np.abs(RNG.normal(size=500)) * 30},
        {"range_deg": [0.0, 90.0]},
    ),
    "courtship.wing_polar": (
        {"phase_diffs": RNG.uniform(-np.pi, np.pi, size=80)},
        {"center_stat": "median"},
    ),
}


@pytest.mark.parametrize("type_id", sorted(CASES))
def test_adapter_draws_without_error(type_id):
    data, spec = CASES[type_id]
    pt = get_panel_type(type_id)
    projection = "polar" if type_id.endswith("polar") else None
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projection)
    try:
        pt.draw(ax, data, spec)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("type_id", sorted(CASES))
def test_adapter_declares_the_data_it_uses(type_id):
    data, _ = CASES[type_id]
    pt = get_panel_type(type_id)
    assert set(pt.needs) <= set(data), f"{type_id} needs {pt.needs}"
    assert pt.needs, f"{type_id} must declare its needs"


def test_wing_z_adapter_converts_segment_arrays_to_dicts():
    """The panel functions expect list-of-dicts, the bundle stores structs."""
    from figbuilder.panels.courtship import _segs_arg
    out = _segs_arg(_segs())
    assert out[0] == {"start": 20, "end": 90, "type": "pulse"}


def test_polar_panel_is_flagged_so_the_renderer_uses_a_polar_axes():
    assert get_panel_type("courtship.wing_polar").projection == "polar"
    assert get_panel_type("courtship.wing_z").projection is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figbuilder_courtship_panels.py -v`
Expected: FAIL — `KeyError: unknown panel type 'courtship.wing_z'`

- [ ] **Step 3: Write minimal implementation**

Add `projection` to the `PanelType` base in `figbuilder/panels/base.py` (a
class attribute, right after `schema`):

```python
    #: matplotlib projection this panel needs, e.g. "polar". None = rectilinear.
    projection: Optional[str] = None
```

and add `Optional` to that module's `typing` import, plus `"projection":
p.projection` to the dict built by `list_panel_types`.

```python
# figbuilder/panels/courtship.py
"""Adapters exposing utils.courtship_figure_panels as registry panel types.

Each adapter is a thin shim: convert bundle-stored structured arrays back to
the list-of-dicts the panel functions expect, then forward. The panel
functions themselves are consumed unmodified.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from utils import courtship_figure_panels as cfp

from figbuilder.bundle import array_to_segments
from figbuilder.panels.base import PanelType, register


def _segs_arg(value: Any) -> List[dict]:
    """Accept a structured segment array or an already-decoded list."""
    if value is None:
        return []
    arr = np.asarray(value)
    if arr.dtype.names:
        return array_to_segments(arr)
    return list(value)


def _frame_range(spec: Dict[str, Any]) -> Optional[tuple]:
    fr = spec.get("frame_range")
    return None if fr is None else (int(fr[0]), int(fr[1]))


_TIME_SCHEMA = {
    "fs": {"type": "number", "default": 800.0, "title": "Sample rate (Hz)"},
    "frame_range": {"type": "array", "items": {"type": "integer"},
                    "title": "Frame range"},
    "time_unit": {"type": "string", "enum": ["ms", "s"], "default": "s"},
}


@register
class WingZPanel(PanelType):
    id = "courtship.wing_z"
    label = "Wing V13 z + song segments"
    needs = ["t_ms", "wingL_z", "wingR_z", "seg_L", "seg_R"]
    schema = {"type": "object", "properties": {
        **_TIME_SCHEMA,
        "min_segment_ms": {"type": "number", "default": 0.0},
    }}

    def draw(self, ax, data, spec):
        cfp.panel_wing_z_traces(
            ax, np.asarray(data["t_ms"]), np.asarray(data["wingL_z"]),
            np.asarray(data["wingR_z"]),
            _segs_arg(data.get("seg_L")), _segs_arg(data.get("seg_R")),
            fs=float(spec.get("fs", 800.0)),
            frame_range=_frame_range(spec),
            min_segment_ms=float(spec.get("min_segment_ms", 0.0)),
            time_unit=spec.get("time_unit", "s"),
        )


@register
class ScutellumZPanel(PanelType):
    id = "courtship.scutellum_z"
    label = "Scutellum (body) z"
    needs = ["t_ms", "scutellum_z"]
    schema = {"type": "object", "properties": {
        **_TIME_SCHEMA,
        "line_color": {"type": "string", "format": "color", "default": "k"},
    }}

    def draw(self, ax, data, spec):
        cfp.panel_scutellum_z_trace(
            ax, np.asarray(data["t_ms"]), np.asarray(data["scutellum_z"]),
            segments=_segs_arg(data.get("segments")),
            fs=float(spec.get("fs", 800.0)),
            frame_range=_frame_range(spec),
            line_color=spec.get("line_color", "k"),
            time_unit=spec.get("time_unit", "s"),
        )


@register
class MalePitchPanel(PanelType):
    id = "courtship.male_pitch"
    label = "Male body pitch vs target pitch"
    needs = ["t_ms", "male_pitch", "target_pitch"]
    schema = {"type": "object", "properties": {
        **_TIME_SCHEMA,
        "male_color": {"type": "string", "format": "color", "default": "#d62728"},
        "target_color": {"type": "string", "format": "color", "default": "#1f77b4"},
    }}

    def draw(self, ax, data, spec):
        cfp.panel_male_pitch(
            ax, np.asarray(data["t_ms"]), np.asarray(data["male_pitch"]),
            np.asarray(data["target_pitch"]),
            segments=None, fs=float(spec.get("fs", 800.0)),
            frame_range=_frame_range(spec),
            male_color=spec.get("male_color", "#d62728"),
            target_color=spec.get("target_color", "#1f77b4"),
            time_unit=spec.get("time_unit", "s"),
        )


@register
class PitchViolinPanel(PanelType):
    id = "courtship.pitch_violin"
    label = "Per-bout pitch alignment (violin)"
    needs = ["per_bout"]
    schema = {"type": "object", "properties": {
        "exemplar_idx": {"type": ["integer", "null"], "title": "Exemplar index"},
    }}

    def draw(self, ax, data, spec):
        cfp.panel_pitch_alignment_violin(
            ax, np.asarray(data["per_bout"]),
            exemplar_idx=spec.get("exemplar_idx"),
        )


@register
class ZHeightPanel(PanelType):
    id = "courtship.zheight"
    label = "Body height: singing vs walking"
    needs = ["pulse_z", "sine_z", "walking_z"]
    schema = {"type": "object", "properties": {
        "kind": {"type": "string", "enum": ["violin", "box"], "default": "violin"},
        "title": {"type": "string", "default": ""},
    }}

    def draw(self, ax, data, spec):
        cfp.panel_z_height_singing_vs_walking(
            ax, np.asarray(data["pulse_z"]), np.asarray(data["sine_z"]),
            np.asarray(data["walking_z"]),
            kind=spec.get("kind", "violin"),
            point_kwargs={"s": 3, "alpha": 0.25},
            title=spec.get("title", ""),
        )


@register
class AngleDensityPanel(PanelType):
    id = "courtship.angle_density"
    label = "Wing angle density: pulse vs sine"
    needs = ["ext_pulse", "ext_sine"]
    schema = {"type": "object", "properties": {
        "range_deg": {"type": "array", "items": {"type": "number"},
                      "default": [0.0, 90.0]},
        "title": {"type": "string", "default": ""},
    }}

    def draw(self, ax, data, spec):
        rd = spec.get("range_deg", [0.0, 90.0])
        cfp.panel_joint_angle_density(
            ax, np.asarray(data["ext_pulse"]), np.asarray(data["ext_sine"]),
            range_deg=(float(rd[0]), float(rd[1])),
            title=spec.get("title", ""),
        )


@register
class WingPolarPanel(PanelType):
    id = "courtship.wing_polar"
    label = "L-R wing phase difference (polar)"
    needs = ["phase_diffs"]
    projection = "polar"
    schema = {"type": "object", "properties": {
        "center_stat": {"type": "string", "enum": ["median", "mean"],
                        "default": "median"},
        "title": {"type": "string", "default": ""},
    }}

    def draw(self, ax, data, spec):
        cfp.panel_wing_phase_polar(
            ax, np.asarray(data["phase_diffs"]),
            center_stat=spec.get("center_stat", "median"),
            title=spec.get("title", ""),
        )
```

> **Implementer note:** `courtship.sine_inphase` and
> `courtship.pulse_class` are intentionally NOT in this task. Their panel
> functions (`panel_sine_wing_inphase`, `panel_pulse_classification`) take
> nested dict structures (`pulse_type_results`) that need a bundle encoding
> decision. They are added in Task 10 alongside the bundle export script that
> produces those structures.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figbuilder_courtship_panels.py -v`
Expected: 16 passed (7 parametrized × 2, plus 2 direct)

- [ ] **Step 5: Commit**

```bash
git add figbuilder/panels/courtship.py figbuilder/panels/base.py tests/test_figbuilder_courtship_panels.py
git commit -m "feat(figbuilder): courtship panel adapters over utils.courtship_figure_panels"
```

---

## Task 6: Tile renderer with ink box and disk cache

Renders ONE panel onto a transparent, **figure-sized** canvas with its axes at
the final rect. Compositing is then a pure overlay — no transforms, and labels
can never clip. Returns the ink box so the editor can align by it and warn when
a panel's labels fall off the canvas.

**Files:**
- Create: `figbuilder/render.py`
- Modify: `figbuilder/style.py` (one line — see "Font metrics" below)
- Test: `tests/test_figbuilder_render.py`

**Font metrics (Ruling 7) — do this first, it is one line.**
`render_tile` computes `ink_box` from matplotlib's text layout, but the composed
SVG is drawn by the *renderer* (browser / rsvg), which resolves the font stack
itself. When Arial is absent — as on the Hyak nodes — matplotlib lays out with
DejaVu Sans while the renderer substitutes Liberation Sans, and measurement at
6 pt showed DejaVu running **13-16% wider** ("z (mm)" +13.9%, "time (s)"
+16.1%). That inflates every `ink_box` and makes the `overflows` flag fire on
panels that actually fit.

Liberation Sans is metric-compatible with Arial and is exactly what fontconfig
substitutes, so adding it to the stack makes matplotlib's layout font match the
render font whenever Arial is missing — and changes nothing on a machine that
has Arial. In `figbuilder/style.py`, change:

```python
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
```

to:

```python
    # Liberation Sans is metric-compatible with Arial and is what fontconfig
    # substitutes when Arial is absent. Including it keeps matplotlib's LAYOUT
    # font identical to the font the SVG renderer actually DRAWS with, so
    # ink_box measurements stay truthful on machines without Arial. Ruling 7.
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
```

**Interfaces:**
- Consumes: `FigureSpec`, `PanelSpec` (Tasks 2, 4); `get_panel_type` (Task 3); `Bundle` (Task 1)
- Produces:
  - `render_tile(fig_spec, panel, data, cache_dir=None) -> TileResult`
  - `TileResult` with `.svg: bytes`, `.ink_box: tuple[float,float,float,float]` (root-figure fractions), `.cache_hit: bool`, `.overflows: bool`
  - `tile_cache_key(fig_spec, panel) -> str`
  - `panel_data(bundle, panel) -> dict[str, np.ndarray]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figbuilder_render.py
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
    a = tile_cache_key(_fig(), _panel((0.1, 0.1, 0.5, 0.5)))
    b = tile_cache_key(_fig(), _panel((0.1, 0.1, 0.5, 0.6)))
    assert a != b
    p = _panel()
    p.spec["ylabel"] = "different"
    assert tile_cache_key(_fig(), p) != tile_cache_key(_fig(), _panel())


def test_second_render_hits_the_cache(tmp_path):
    first = render_tile(_fig(), _panel(), {"y": np.linspace(0, 1, 50)},
                        cache_dir=tmp_path)
    second = render_tile(_fig(), _panel(), {"y": np.linspace(0, 1, 50)},
                         cache_dir=tmp_path)
    assert not first.cache_hit
    assert second.cache_hit
    assert first.svg == second.svg


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figbuilder_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'figbuilder.render'`

- [ ] **Step 3: Write minimal implementation**

```python
# figbuilder/render.py
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


def tile_cache_key(fig_spec: FigureSpec, panel: PanelSpec) -> str:
    payload = {
        "size_mm": [fig_spec.width_mm, fig_spec.height_mm],
        "style": fig_spec.style,
        "transparent": fig_spec.transparent,
        "panel": {"id": panel.id, "type": panel.type, "rect": list(panel.rect),
                  "data": panel.data, "spec": panel.spec},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def _ink_box(fig, ax) -> Rect:
    fig.canvas.draw()
    bb = ax.get_tightbbox().transformed(fig.transFigure.inverted())
    return (float(bb.x0), float(bb.y0), float(bb.width), float(bb.height))


def render_tile(fig_spec: FigureSpec, panel: PanelSpec,
                data: Dict[str, Any],
                cache_dir: Optional[str | Path] = None) -> TileResult:
    """Render `panel` onto a full-size transparent canvas."""
    key = tile_cache_key(fig_spec, panel)
    cache_path = Path(cache_dir) / f"{key}.svg" if cache_dir else None
    meta_path = Path(cache_dir) / f"{key}.json" if cache_dir else None
    if cache_path and cache_path.exists() and meta_path and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return TileResult(svg=cache_path.read_bytes(),
                          ink_box=tuple(meta["ink_box"]),
                          cache_hit=True, overflows=meta["overflows"])

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
        cache_path.write_bytes(svg)
        meta_path.write_text(json.dumps({"ink_box": list(ink),
                                         "overflows": bool(overflows)}))
    return TileResult(svg=svg, ink_box=ink, cache_hit=False, overflows=overflows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figbuilder_render.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add figbuilder/render.py tests/test_figbuilder_render.py
git commit -m "feat(figbuilder): full-canvas tile renderer with ink box and disk cache"
```

---

## Task 7: SVG id namespacing and the compositor

**Verified by probe:** 35 of 39 ids collide between two independently rendered
tiles. Namespacing is mandatory.

**Files:**
- Create: `figbuilder/svgutil.py`, `figbuilder/compose.py`
- Test: `tests/test_figbuilder_compose.py`

**Interfaces:**
- Consumes: `TileResult` (Task 6), `FigureSpec` (Task 2/4)
- Produces:
  - `namespace_ids(svg: bytes, prefix: str) -> lxml Element`
  - `compose(fig_spec, tiles: list[tuple[str, bytes]], annotation_elements=()) -> bytes`
  - `SVG_NS`, `qname(tag) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figbuilder_compose.py
"""SVG id namespacing and document composition."""
from __future__ import annotations

import re

import matplotlib
matplotlib.use('Agg')
import numpy as np
from lxml import etree

import figbuilder.panels  # noqa: F401
from figbuilder.compose import compose
from figbuilder.figure import FigureSpec, PanelSpec
from figbuilder.render import render_tile
from figbuilder.svgutil import namespace_ids


def _fig():
    return FigureSpec(width_mm=100.0, height_mm=60.0)


def _tile(pid, rect, color):
    p = PanelSpec(id=pid, type="line", rect=rect,
                  data={"y": {"dataset": f"/panels/{pid}/data/y"}},
                  spec={"series": [{"y": "y", "color": color}], "ylabel": "z"})
    return render_tile(_fig(), p, {"y": np.linspace(0, 1, 40)}).svg


def test_namespace_ids_rewrites_definitions_and_references():
    svg = _tile("a", (0.2, 0.2, 0.6, 0.6), "#000000")
    root = namespace_ids(svg, "a")
    ids = [e.get("id") for e in root.iter() if e.get("id")]
    assert ids and all(i.startswith("a__") for i in ids)
    blob = etree.tostring(root)
    assert b'href="#a__' in blob or b"#a__" in blob or b"url(#a__" in blob


def test_compose_produces_no_duplicate_ids_across_tiles():
    out = compose(_fig(), [("a", _tile("a", (0.1, 0.55, 0.8, 0.35), "#38bdf8")),
                           ("b", _tile("b", (0.1, 0.12, 0.8, 0.35), "#000000"))])
    ids = re.findall(rb'id="([^"]+)"', out)
    assert len(ids) == len(set(ids)), "duplicate ids in composed document"


def test_compose_wraps_each_tile_in_a_named_group():
    out = compose(_fig(), [("a", _tile("a", (0.1, 0.55, 0.8, 0.35), "#38bdf8")),
                           ("b", _tile("b", (0.1, 0.12, 0.8, 0.35), "#000000"))])
    root = etree.fromstring(out)
    top = [g.get("id") for g in root if g.get("id")]
    assert "panel_a" in top and "panel_b" in top


def test_composed_root_carries_figure_size_in_points():
    out = compose(_fig(), [("a", _tile("a", (0.2, 0.2, 0.6, 0.6), "#000000"))])
    root = etree.fromstring(out)
    assert root.get("width") == f"{100.0 / 25.4 * 72:.6f}pt"
    assert root.get("viewBox") == (
        f"0 0 {100.0 / 25.4 * 72:.6f} {60.0 / 25.4 * 72:.6f}")


def test_annotation_elements_are_appended_after_tiles():
    ann = etree.Element("{http://www.w3.org/2000/svg}text")
    ann.set("id", "letter_a")
    ann.text = "A"
    out = compose(_fig(), [("a", _tile("a", (0.2, 0.2, 0.6, 0.6), "#000000"))],
                  annotation_elements=[ann])
    root = etree.fromstring(out)
    ids = [c.get("id") for c in root]
    assert ids[-1] == "annotations"
    assert b">A<" in out


def test_compose_with_no_tiles_still_emits_a_valid_document():
    out = compose(_fig(), [])
    root = etree.fromstring(out)
    assert root.tag.endswith("svg")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figbuilder_compose.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'figbuilder.compose'`

- [ ] **Step 3: Write minimal implementation**

```python
# figbuilder/svgutil.py
"""SVG helpers, chiefly id namespacing.

matplotlib emits document-local ids (`figure_1`, `axes_1`, `line2d_1`,
`C3_0_ad4e761f83`) that are identical across independently rendered figures.
Overlaying two tiles without rewriting them silently corrupts clip paths and
marker definitions, because `url(#...)` resolves to the first match.
"""
from __future__ import annotations

import re
from typing import Set

from lxml import etree

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
NSMAP = {None: SVG_NS, "xlink": XLINK_NS}

SEP = "__"


def qname(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def _collect_ids(root) -> Set[str]:
    return {e.get("id") for e in root.iter() if e.get("id")}


def namespace_ids(svg: bytes, prefix: str):
    """Return the parsed tile with every id (and reference) prefixed."""
    root = etree.fromstring(svg)
    ids = _collect_ids(root)
    if not ids:
        return root
    blob = etree.tostring(root)
    # Longest first, so `line2d_1` cannot corrupt `line2d_11`.
    for old in sorted(ids, key=len, reverse=True):
        new = f"{prefix}{SEP}{old}"
        blob = blob.replace(f'id="{old}"'.encode(), f'id="{new}"'.encode())
        blob = blob.replace(f'#{old}"'.encode(), f'#{new}"'.encode())
        blob = blob.replace(f'#{old})'.encode(), f'#{new})'.encode())
    return etree.fromstring(blob)
```

```python
# figbuilder/compose.py
"""Compose per-panel tiles and an annotation layer into one SVG document."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from lxml import etree

from figbuilder.figure import FigureSpec
from figbuilder.svgutil import NSMAP, SVG_NS, namespace_ids, qname

PT_PER_MM = 72.0 / 25.4

#: Tile children that must not be copied into the composed document.
_SKIP = {"metadata"}


def compose(fig_spec: FigureSpec,
            tiles: Sequence[Tuple[str, bytes]],
            annotation_elements: Iterable = ()) -> bytes:
    """Overlay `tiles` (in order) and append the annotation layer."""
    w_pt = fig_spec.width_mm * PT_PER_MM
    h_pt = fig_spec.height_mm * PT_PER_MM

    root = etree.Element(qname("svg"), nsmap=NSMAP)
    # Six decimals, matching matplotlib's own pt formatting in the tiles this
    # document overlays — see Ruling 4 in the SDD ledger.
    root.set("width", f"{w_pt:.6f}pt")
    root.set("height", f"{h_pt:.6f}pt")
    root.set("viewBox", f"0 0 {w_pt:.6f} {h_pt:.6f}")
    root.set("version", "1.1")

    for panel_id, svg in tiles:
        tile_root = namespace_ids(svg, panel_id)
        g = etree.SubElement(root, qname("g"))
        g.set("id", f"panel_{panel_id}")
        for child in tile_root:
            if etree.QName(child).localname in _SKIP:
                continue
            g.append(child)

    ann: List = list(annotation_elements)
    layer = etree.SubElement(root, qname("g"))
    layer.set("id", "annotations")
    for el in ann:
        layer.append(el)

    return etree.tostring(root, xml_declaration=True, encoding="utf-8",
                          pretty_print=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figbuilder_compose.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add figbuilder/svgutil.py figbuilder/compose.py tests/test_figbuilder_compose.py
git commit -m "feat(figbuilder): id namespacing and SVG compositor"
```

---

## Task 8: Annotation SVG emitter

**Files:**
- Create: `figbuilder/annot.py`
- Test: `tests/test_figbuilder_annot.py`

**Interfaces:**
- Consumes: `FigureSpec` (Task 2/4), `qname` (Task 7)
- Produces:
  - `emit_annotations(fig_spec, annotations: list[dict]) -> list[lxml Element]`
  - `ANNOTATION_KINDS: frozenset[str]`
  - Resolves panel-parented positions via `fig_spec.panel(parent).rect`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figbuilder_annot.py
"""Annotation layer emission."""
from __future__ import annotations

import pytest
from lxml import etree

from figbuilder.annot import ANNOTATION_KINDS, emit_annotations
from figbuilder.figure import FigureSpec, PanelSpec


def _fig():
    return FigureSpec(width_mm=100.0, height_mm=50.0, panels=[
        PanelSpec(id="wing", type="line", rect=(0.1, 0.5, 0.8, 0.4)),
    ])


def test_text_annotation_becomes_real_svg_text():
    els = emit_annotations(_fig(), [
        {"id": "t1", "kind": "text", "pos_mm": [5.0, 10.0], "text": "C",
         "style": {"font_size_pt": 8, "weight": "bold"}}])
    assert len(els) == 1
    assert etree.QName(els[0]).localname == "text"
    assert els[0].text == "C"
    assert "font-weight:700" in els[0].get("style")


def test_y_is_flipped_from_figure_space_to_svg_space():
    """Figure y grows upward; SVG y grows downward."""
    els = emit_annotations(_fig(), [
        {"id": "t", "kind": "text", "pos_mm": [0.0, 0.0], "text": "x"}])
    # y=0 mm from the figure bottom -> y = height in pt
    assert float(els[0].get("y")) == pytest.approx(50.0 * 72 / 25.4)
    # ...and it is written at full precision, not 6 significant figures.
    assert els[0].get("y") == "141.732283"


def test_num_formatter_trims_trailing_zeros_and_keeps_six_decimals():
    from figbuilder.annot import _num
    assert _num(0.8) == "0.8"
    assert _num(8) == "8"
    assert _num(141.73228346456693) == "141.732283"
    assert _num(-0.0) == "0"   # must match JS toFixed, which has no -0


def test_panel_parented_annotation_offsets_from_that_panel_corner():
    els = emit_annotations(_fig(), [
        {"id": "L", "kind": "text", "parent": "wing", "pos_mm": [0.0, 0.0],
         "text": "A"}])
    # panel rect x=0.1 of 100 mm -> 10 mm -> in pt
    assert float(els[0].get("x")) == pytest.approx(10.0 * 72 / 25.4)


def test_line_and_rect_and_ellipse_emit_expected_tags():
    els = emit_annotations(_fig(), [
        {"id": "l", "kind": "line", "pos_mm": [1, 1], "to_mm": [9, 9]},
        {"id": "r", "kind": "rect", "pos_mm": [2, 2], "size_mm": [5, 4]},
        {"id": "e", "kind": "ellipse", "pos_mm": [3, 3], "size_mm": [6, 2]},
    ])
    assert [etree.QName(e).localname for e in els] == ["line", "rect", "ellipse"]


def test_arrow_emits_a_path_with_a_marker_reference():
    els = emit_annotations(_fig(), [
        {"id": "a", "kind": "arrow", "pos_mm": [1, 1], "to_mm": [9, 5]}])
    assert etree.QName(els[0]).localname == "path"
    assert "marker-end" in els[0].get("style", "")


def test_scalebar_emits_a_group_with_a_line_and_a_label():
    els = emit_annotations(_fig(), [
        {"id": "sb", "kind": "scalebar", "pos_mm": [2, 2], "length_mm": 10.0,
         "label": "1 mm"}])
    kids = [etree.QName(c).localname for c in els[0]]
    assert "line" in kids and "text" in kids


def test_every_declared_kind_is_emittable():
    for kind in ANNOTATION_KINDS:
        ann = {"id": "x", "kind": kind, "pos_mm": [1, 1], "to_mm": [5, 5],
               "size_mm": [3, 3], "text": "t", "label": "l", "length_mm": 4.0}
        assert emit_annotations(_fig(), [ann]), f"{kind} emitted nothing"


def test_unknown_kind_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown annotation kind"):
        emit_annotations(_fig(), [{"id": "z", "kind": "hologram"}])


def test_annotations_carry_their_id_for_reediting():
    els = emit_annotations(_fig(), [
        {"id": "letter_wing", "kind": "text", "pos_mm": [1, 1], "text": "A"}])
    assert els[0].get("id") == "letter_wing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figbuilder_annot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'figbuilder.annot'`

- [ ] **Step 3: Write minimal implementation**

```python
# figbuilder/annot.py
"""Annotation layer: objects -> SVG elements.

This is the Python half of a deliberately duplicated emitter; the TypeScript
half drives live editing in the browser. A conformance test (Task 14) asserts
the two agree. Duplication is preferred over making headless export depend on
a browser.

Positions are in millimetres. Figure space has y growing UPWARD from the
bottom-left (matching matplotlib); SVG has y growing downward, so every y is
flipped here exactly once.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from lxml import etree

from figbuilder.svgutil import qname

PT_PER_MM = 72.0 / 25.4


def _num(v: float) -> str:
    """Format a number identically in Python and TypeScript.

    Six decimals, trailing zeros trimmed. `:g` is NOT usable here: it gives
    6 SIGNIFICANT figures, so 141.73228346 becomes "141.732" — a 2e-6 relative
    error that breaks both pytest.approx and character-exact agreement with the
    TS emitter. See Ruling 3 in the SDD ledger.
    """
    out = f"{float(v):.6f}".rstrip("0").rstrip(".")
    # Negative zero must render as "0": JS toFixed(-0.0) has no sign, and a
    # coordinate of exactly -0.0 is reachable (a panel at the canvas edge).
    return "0" if out.lstrip("-") in ("", "0") else out


ANNOTATION_KINDS = frozenset({
    "text", "line", "arrow", "leader", "rect", "ellipse", "bracket",
    "scalebar", "image",
})

_ARROW_MARKER_ID = "fb_arrowhead"


def _style(d: Dict[str, Any]) -> str:
    parts: List[str] = []
    if "color" in d:
        parts.append(f"fill:{d['color']}")
    if "stroke" in d:
        parts.append(f"stroke:{d['stroke']}")
    if "lw_pt" in d:
        parts.append(f"stroke-width:{_num(d['lw_pt'])}")
    if "font_size_pt" in d:
        parts.append(f"font-size:{_num(d['font_size_pt'])}px")
    if d.get("weight") in ("bold", 700):
        parts.append("font-weight:700")
    if "font" in d:
        parts.append(f"font-family:{d['font']}")
    if "opacity" in d:
        parts.append(f"opacity:{_num(d['opacity'])}")
    return ";".join(parts)


class _Frame:
    """Converts annotation mm coordinates into SVG point coordinates."""

    def __init__(self, fig_spec, parent_id: Optional[str]):
        self.w_mm = fig_spec.width_mm
        self.h_mm = fig_spec.height_mm
        self.ox_mm = 0.0
        self.oy_mm = 0.0
        if parent_id:
            x, y, _, _ = fig_spec.panel(parent_id).rect
            self.ox_mm = x * self.w_mm
            self.oy_mm = y * self.h_mm

    def x(self, mm: float) -> float:
        return (self.ox_mm + float(mm)) * PT_PER_MM

    def y(self, mm: float) -> float:
        return (self.h_mm - (self.oy_mm + float(mm))) * PT_PER_MM

    def d(self, mm: float) -> float:
        return float(mm) * PT_PER_MM


def _pt(seq: Sequence[float], default=(0.0, 0.0)):
    return tuple(seq) if seq is not None else default


def emit_annotations(fig_spec, annotations: List[dict]) -> List:
    """Convert annotation objects into SVG elements, in document order."""
    out: List = []
    needs_marker = False

    for ann in annotations:
        kind = ann.get("kind")
        if kind not in ANNOTATION_KINDS:
            raise ValueError(
                f"unknown annotation kind {kind!r}; expected one of "
                f"{sorted(ANNOTATION_KINDS)}")
        f = _Frame(fig_spec, ann.get("parent"))
        st = dict(ann.get("style", {}))
        px, py = _pt(ann.get("pos_mm"))

        if kind == "text":
            el = etree.Element(qname("text"))
            el.set("x", _num(f.x(px)))
            el.set("y", _num(f.y(py)))
            st.setdefault("font", "Arial, Helvetica, sans-serif")
            el.set("style", _style(st))
            el.text = str(ann.get("text", ""))

        elif kind in ("line", "leader"):
            tx, ty = _pt(ann.get("to_mm"))
            el = etree.Element(qname("line"))
            el.set("x1", _num(f.x(px))); el.set("y1", _num(f.y(py)))
            el.set("x2", _num(f.x(tx))); el.set("y2", _num(f.y(ty)))
            st.setdefault("stroke", "#000000"); st.setdefault("lw_pt", 0.8)
            el.set("style", _style(st))

        elif kind == "arrow":
            tx, ty = _pt(ann.get("to_mm"))
            el = etree.Element(qname("path"))
            el.set("d", f"M {_num(f.x(px))},{_num(f.y(py))} "
                        f"L {_num(f.x(tx))},{_num(f.y(ty))}")
            st.setdefault("stroke", "#000000"); st.setdefault("lw_pt", 0.8)
            el.set("style", _style(st) + f";fill:none;marker-end:url(#{_ARROW_MARKER_ID})")
            needs_marker = True

        elif kind == "rect":
            w, h = _pt(ann.get("size_mm"), (1.0, 1.0))
            el = etree.Element(qname("rect"))
            el.set("x", _num(f.x(px)))
            el.set("y", _num(f.y(py + h)))
            el.set("width", _num(f.d(w))); el.set("height", _num(f.d(h)))
            st.setdefault("stroke", "#000000"); st.setdefault("lw_pt", 0.8)
            el.set("style", _style(st) + ";fill:none" if "color" not in st else _style(st))

        elif kind == "ellipse":
            w, h = _pt(ann.get("size_mm"), (1.0, 1.0))
            el = etree.Element(qname("ellipse"))
            el.set("cx", _num(f.x(px + w / 2)))
            el.set("cy", _num(f.y(py + h / 2)))
            el.set("rx", _num(f.d(w / 2))); el.set("ry", _num(f.d(h / 2)))
            st.setdefault("stroke", "#000000"); st.setdefault("lw_pt", 0.8)
            el.set("style", _style(st) + ";fill:none" if "color" not in st else _style(st))

        elif kind == "bracket":
            tx, ty = _pt(ann.get("to_mm"))
            tick = f.d(float(ann.get("tick_mm", 1.0)))
            x1, y1, x2, y2 = f.x(px), f.y(py), f.x(tx), f.y(ty)
            el = etree.Element(qname("path"))
            el.set("d", f"M {_num(x1)},{_num(y1 + tick)} "
                        f"L {_num(x1)},{_num(y1)} "
                        f"L {_num(x2)},{_num(y2)} "
                        f"L {_num(x2)},{_num(y2 + tick)}")
            st.setdefault("stroke", "#000000"); st.setdefault("lw_pt", 0.8)
            el.set("style", _style(st) + ";fill:none")

        elif kind == "scalebar":
            length = float(ann.get("length_mm", 5.0))
            el = etree.Element(qname("g"))
            ln = etree.SubElement(el, qname("line"))
            ln.set("x1", _num(f.x(px))); ln.set("y1", _num(f.y(py)))
            ln.set("x2", _num(f.x(px + length))); ln.set("y2", _num(f.y(py)))
            ln.set("style", _style({"stroke": st.get("stroke", "#000000"),
                                    "lw_pt": st.get("lw_pt", 1.2)}))
            tx_el = etree.SubElement(el, qname("text"))
            tx_el.set("x", _num(f.x(px + length / 2)))
            tx_el.set("y", _num(f.y(py) + f.d(2.0)))
            tx_el.set("style", _style({"font_size_pt": st.get("font_size_pt", 6),
                                       "font": "Arial, Helvetica, sans-serif"})
                      + ";text-anchor:middle")
            tx_el.text = str(ann.get("label", ""))

        elif kind == "image":
            w, h = _pt(ann.get("size_mm"), (10.0, 10.0))
            el = etree.Element(qname("image"))
            el.set("x", _num(f.x(px)))
            el.set("y", _num(f.y(py + h)))
            el.set("width", _num(f.d(w))); el.set("height", _num(f.d(h)))
            el.set("{http://www.w3.org/1999/xlink}href", ann.get("href", ""))

        if ann.get("id"):
            el.set("id", str(ann["id"]))
        out.append(el)

    if needs_marker:
        defs = etree.Element(qname("defs"))
        m = etree.SubElement(defs, qname("marker"))
        m.set("id", _ARROW_MARKER_ID)
        m.set("viewBox", "0 0 10 10"); m.set("refX", "9"); m.set("refY", "5")
        m.set("markerWidth", "5"); m.set("markerHeight", "5")
        m.set("orient", "auto-start-reverse")
        p = etree.SubElement(m, qname("path"))
        p.set("d", "M 0,0 L 10,5 L 0,10 z")
        out.insert(0, defs)

    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figbuilder_annot.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add figbuilder/annot.py tests/test_figbuilder_annot.py
git commit -m "feat(figbuilder): annotation layer SVG emitter"
```

---

## Task 9: Export pipeline and CLI

**Files:**
- Create: `figbuilder/export.py`, `figbuilder/cli.py`, `figbuilder/__main__.py`
- Test: `tests/test_figbuilder_export.py`

**Interfaces:**
- Consumes: everything from Tasks 1–8
- Produces:
  - `export_figure(fig_spec, bundle, out_path, formats=("svg",), cache_dir=None) -> ExportResult`
  - `ExportResult` with `.paths: dict[str, Path]`, `.warnings: list[str]`
  - `svg_to(src: Path, dst: Path, fmt: str, dpi: int) -> None` (uses `rsvg-convert`)
  - CLI: `python -m figbuilder export FIGURE.json [--bundle B.h5] [-o OUT] [--format svg,png,pdf]`
  - CLI: `python -m figbuilder bundle-info B.h5`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figbuilder_export.py
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
    fig_path, bundle_path = project
    spec, bundle = load_figure(fig_path), read_bundle(bundle_path)
    a = export_figure(spec, bundle, tmp_path / "a.svg").paths["svg"].read_bytes()
    b = export_figure(spec, bundle, tmp_path / "b.svg").paths["svg"].read_bytes()
    assert a == b


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figbuilder_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'figbuilder.export'`

- [ ] **Step 3: Write minimal implementation**

```python
# figbuilder/export.py
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
```

```python
# figbuilder/cli.py
"""figbuilder command line."""
from __future__ import annotations

import argparse
from pathlib import Path

from figbuilder.bundle import read_bundle
from figbuilder.export import export_figure
from figbuilder.figure import load_figure


def _cmd_export(args) -> int:
    fig_path = Path(args.figure)
    spec = load_figure(fig_path)
    bundle_path = Path(args.bundle) if args.bundle else fig_path.parent / spec.bundle
    out = Path(args.out) if args.out else fig_path.with_suffix(".svg")
    formats = tuple(f.strip() for f in args.format.split(",") if f.strip())
    res = export_figure(spec, read_bundle(bundle_path), out,
                        formats=formats, cache_dir=args.cache_dir)
    for w in res.warnings:
        print(f"warning: {w}")
    for fmt, p in res.paths.items():
        print(f"wrote {fmt}: {p}")
    return 0


def _cmd_bundle_info(args) -> int:
    b = read_bundle(args.bundle)
    print(f"meta: {b.meta}")
    for pid, pd in sorted(b.panels.items()):
        data = ", ".join(f"{k}{tuple(v.shape)}" for k, v in sorted(pd.data.items()))
        assets = ", ".join(f"{k}{tuple(v.shape)}" for k, v in sorted(pd.assets.items()))
        print(f"  {pid:<18} type={pd.type}")
        if data:
            print(f"      data:   {data}")
        if assets:
            print(f"      assets: {assets}")
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    from figbuilder.server import create_app

    uvicorn.run(create_app(Path(args.figure), Path(args.bundle) if args.bundle else None),
                host=args.host, port=args.port)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="figbuilder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="render figure.json to svg/png/pdf")
    e.add_argument("figure")
    e.add_argument("--bundle", default=None)
    e.add_argument("-o", "--out", default=None)
    e.add_argument("--format", default="svg", help="comma list: svg,png,pdf")
    e.add_argument("--cache-dir", default=None)
    e.set_defaults(func=_cmd_export)

    b = sub.add_parser("bundle-info", help="list a bundle's panels and datasets")
    b.add_argument("bundle")
    b.set_defaults(func=_cmd_bundle_info)

    s = sub.add_parser("serve", help="run the editor server")
    s.add_argument("figure")
    s.add_argument("--bundle", default=None)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=_cmd_serve)

    args = ap.parse_args(argv)
    return args.func(args)
```

```python
# figbuilder/__main__.py
from __future__ import annotations

import sys

from figbuilder.cli import main

sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figbuilder_export.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add figbuilder/export.py figbuilder/cli.py figbuilder/__main__.py tests/test_figbuilder_export.py
git commit -m "feat(figbuilder): export pipeline and CLI"
```

---

## Task 10: Fig 4 bundle export script

> **Runs only on the machine with `/data2` mounted.** It is NOT runnable on
> Hyak, and it has no unit test that executes the pipeline. Its correctness is
> established by Task 12's visual gate. The two extra panel adapters land here
> because their bundle encoding is only decidable once the real structures are
> in hand.

**Files:**
- Create: `scripts/figures/export_fig4_bundle.py`
- Modify: `figbuilder/panels/courtship.py` (add `courtship.sine_inphase`, `courtship.pulse_class`)
- Test: `tests/test_figbuilder_fig4_bundle.py` (structure only, synthetic input)

**Interfaces:**
- Consumes: `write_bundle`, `segments_to_array` (Task 1)
- Produces:
  - `build_fig4_panels(results, ex, extras) -> dict[str, PanelData]` — pure, testable, no I/O
  - `main()` — loads the h5s, runs the analysis, calls `build_fig4_panels`, writes the bundle
  - Registered types `courtship.sine_inphase`, `courtship.pulse_class`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figbuilder_fig4_bundle.py
"""build_fig4_panels shapes the bundle correctly from synthetic analysis output."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import figbuilder.panels  # noqa: F401
from figbuilder.panels.base import get_panel_type
from scripts.figures.export_fig4_bundle import build_fig4_panels

FS = 800.0
T = 300


def _synthetic():
    rng = np.random.default_rng(0)
    seg = [{"start": 10, "end": 80, "type": "pulse"},
           {"start": 120, "end": 220, "type": "sine"}]
    ex = {
        "T": T,
        "com_z": rng.normal(size=T),
        "song0": {
            "wing_data": {"WingL_V13": {"z": rng.normal(size=T)},
                          "WingR_V13": {"z": rng.normal(size=T)}},
            "sides": {"L": {"segments": seg}, "R": {"segments": seg}},
            "angle_L": rng.normal(size=T), "angle_R": rng.normal(size=T),
            "dominant_wing": "L",
        },
    }
    results = [{"male_valid": np.ones(T, bool),
                "male_labels": np.array(["pulse"] * 150 + ["sine"] * 150),
                "com_z": rng.normal(size=T),
                "song0": ex["song0"]} for _ in range(4)]
    extras = {
        "fs": FS, "start_frame": 0, "end_frame": T,
        "walking_z": rng.normal(size=20),
        "phase_diffs": rng.uniform(-np.pi, np.pi, 40),
        "per_bout_align": np.abs(rng.normal(size=30)) * 15,
        "male_pitch": rng.normal(size=T) * 10,
        "target_pitch": rng.normal(size=T) * 10,
        "pulse_centroids": {"Pslow": rng.normal(size=48),
                            "Pfast": rng.normal(size=48)},
        "pulse_pooled": {"Pslow": rng.normal(size=(9, 48)),
                         "Pfast": rng.normal(size=(7, 48))},
        "pulse_counts": {"Pslow": 9, "Pfast": 7},
        "ext_pulse": np.abs(rng.normal(size=200)) * 30,
        "ext_sine": np.abs(rng.normal(size=200)) * 30,
        "video_frames": [np.zeros((16, 16, 3), np.uint8) for _ in range(4)],
        "render_frames": [np.zeros((16, 16, 3), np.uint8) for _ in range(4)],
        "exemplar_bout_idx": 3,
    }
    return results, ex, extras


def test_bundle_contains_a_panel_for_every_fig4_element():
    panels = build_fig4_panels(*_synthetic())
    expected = {"wing", "scut", "sine_phase", "wing_phase_polar", "angle_2d",
                "pulse_class", "zheight", "pitch", "align_violin"}
    assert expected <= set(panels)


def test_video_and_render_frames_become_individual_image_panels():
    panels = build_fig4_panels(*_synthetic())
    assert {"video_0", "video_1", "video_2", "video_3"} <= set(panels)
    assert {"render_0", "render_1", "render_2", "render_3"} <= set(panels)
    assert panels["video_0"].assets["img"].dtype == np.uint8


def test_pulse_class_panel_stores_flat_per_type_arrays():
    """Ruling 5: the bundle is flat; the adapter re-nests into Pslow/Pfast."""
    panels = build_fig4_panels(*_synthetic())
    d = panels["pulse_class"].data
    assert {"centroid_Pslow", "centroid_Pfast",
            "pooled_Pslow", "pooled_Pfast"} <= set(d)
    assert d["centroid_Pslow"].shape == (48,)
    assert d["pooled_Pfast"].shape == (7, 48)


def test_pulse_class_adapter_renders_from_the_bundled_arrays():
    """End-to-end: flat datasets -> nested pulse_type_results -> a drawn axes."""
    import matplotlib.pyplot as plt
    panels = build_fig4_panels(*_synthetic())
    pt = get_panel_type("courtship.pulse_class")
    fig, ax = plt.subplots()
    try:
        pt.draw(ax, panels["pulse_class"].data,
                {"fs": FS, "show_std": True, "count_Pslow": 9, "count_Pfast": 7})
        labels = [ln.get_label() for ln in ax.lines]
        assert any("Pslow (n=9)" == lb for lb in labels), labels
        assert any("Pfast (n=7)" == lb for lb in labels), labels
    finally:
        plt.close(fig)


def test_segment_tables_are_stored_as_structured_arrays():
    panels = build_fig4_panels(*_synthetic())
    assert panels["wing"].data["seg_L"].dtype.names == ("start", "end", "type")


@pytest.mark.parametrize("pid", ["wing", "scut", "pitch", "align_violin",
                                 "zheight", "angle_2d", "wing_phase_polar",
                                 "sine_phase", "pulse_class"])
def test_each_panel_satisfies_its_types_declared_needs(pid):
    """Covers EVERY data-bearing panel, including sine_phase and pulse_class.

    This is the test that catches an adapter whose `needs` drifts from what
    build_fig4_panels actually emits — exactly the pulse_class defect found in
    the pre-flight audit (Ruling 5).
    """
    panels = build_fig4_panels(*_synthetic())
    pd = panels[pid]
    pt = get_panel_type(pd.type)
    assert set(pt.needs) <= set(pd.data), f"{pid} missing {set(pt.needs) - set(pd.data)}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figbuilder_fig4_bundle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.figures.export_fig4_bundle'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/__init__.py` and `scripts/figures/__init__.py` (empty) if absent, then:

```python
# scripts/figures/export_fig4_bundle.py
"""Build the Figure 4 data bundle from the courtship analysis pipeline.

Split in two deliberately:

* `build_fig4_panels` is pure — it takes already-computed analysis structures
  and shapes them into `PanelData`. It is unit-testable with synthetic input.
* `main` does the I/O: loads the combined h5s, runs `analyze_all_pairs`,
  decodes video, drives MuJoCo, and calls `build_fig4_panels`.

`main` requires the `/data2` mounts and a GPU node (MuJoCo needs
`MUJOCO_GL=egl`). It cannot run on the Hyak login node.

This file replaces the notebook's Figure 4 cell as the source of truth; the
notebook should import from here rather than duplicating the logic.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from figbuilder.bundle import PanelData, segments_to_array, write_bundle


def _mean_z_by_label(results: List[dict], label: str) -> np.ndarray:
    out: List[float] = []
    for r in results:
        v = np.asarray(r["male_valid"], dtype=bool)
        lab = np.asarray(r["male_labels"])
        z = np.asarray(r["com_z"], dtype=float)
        m = v & np.isfinite(z) & (lab == label)
        if m.any():
            out.append(float(np.mean(z[m])))
    return np.asarray(out, dtype=float)


def build_fig4_panels(results: List[dict], ex: dict,
                      extras: Dict[str, Any]) -> Dict[str, PanelData]:
    """Shape analysis structures into bundle panels. Pure; no I/O."""
    fs = float(extras["fs"])
    s0, s1 = int(extras["start_frame"]), int(extras["end_frame"])
    sel = slice(s0, s1)
    t_ms = (np.arange(s0, s1) / fs) * 1000.0

    song = ex["song0"]
    wd = song["wing_data"]
    zL = np.asarray(wd["WingL_V13"]["z"], dtype=float)
    zR = np.asarray(wd["WingR_V13"]["z"], dtype=float)
    seg_L = segments_to_array(song["sides"]["L"]["segments"])
    seg_R = segments_to_array(song["sides"]["R"]["segments"])

    angle_L = np.asarray(song["angle_L"], dtype=float)
    angle_R = np.asarray(song["angle_R"], dtype=float)
    ext_is_L = angle_L > angle_R
    ext_z = np.where(ext_is_L, zL, zR)[sel]
    fold_z = np.where(ext_is_L, zR, zL)[sel]

    panels: Dict[str, PanelData] = {
        "wing": PanelData(type="courtship.wing_z", data={
            "t_ms": t_ms, "wingL_z": zL[sel], "wingR_z": zR[sel],
            "seg_L": seg_L, "seg_R": seg_R}, attrs={"fs": fs}),
        "scut": PanelData(type="courtship.scutellum_z", data={
            "t_ms": t_ms, "scutellum_z": np.asarray(ex["com_z"], float)[sel],
            "segments": seg_L}, attrs={"fs": fs}),
        "sine_phase": PanelData(type="courtship.sine_inphase", data={
            "t_ms": t_ms - t_ms[0], "ext_z": ext_z, "fold_z": fold_z,
            "sine_segments": segments_to_array(
                [s for s in song["sides"]["L"]["segments"]
                 if s.get("type") == "sine"])}, attrs={"fs": fs}),
        "wing_phase_polar": PanelData(type="courtship.wing_polar", data={
            "phase_diffs": np.asarray(extras["phase_diffs"], float)}),
        "angle_2d": PanelData(type="courtship.angle_density", data={
            "ext_pulse": np.asarray(extras.get("ext_pulse", np.zeros(0)), float),
            "ext_sine": np.asarray(extras.get("ext_sine", np.zeros(0)), float)}),
        "pulse_class": PanelData(
            type="courtship.pulse_class",
            data={
                # Flat per-type arrays; the adapter re-nests them into the
                # {'Pslow': ..., 'Pfast': ...} dicts the panel function wants.
                **{f"centroid_{t}": np.asarray(
                    extras.get("pulse_centroids", {}).get(t, np.zeros(0)), float)
                   for t in ("Pslow", "Pfast")},
                **{f"pooled_{t}": np.asarray(
                    extras.get("pulse_pooled", {}).get(t, np.zeros((0, 0))), float)
                   for t in ("Pslow", "Pfast")},
            },
            attrs={"fs": fs}),
        "zheight": PanelData(type="courtship.zheight", data={
            "pulse_z": _mean_z_by_label(results, "pulse"),
            "sine_z": _mean_z_by_label(results, "sine"),
            "walking_z": np.asarray(extras["walking_z"], float)}),
        "pitch": PanelData(type="courtship.male_pitch", data={
            "t_ms": np.asarray(extras.get("t_ms_full", t_ms), float),
            "male_pitch": np.asarray(extras["male_pitch"], float),
            "target_pitch": np.asarray(extras["target_pitch"], float)},
            attrs={"fs": fs}),
        "align_violin": PanelData(type="courtship.pitch_violin", data={
            "per_bout": np.asarray(extras["per_bout_align"], float)},
            attrs={"exemplar_idx": int(extras.get("exemplar_bout_idx", -1))}),
    }

    for i, img in enumerate(extras.get("video_frames", [])):
        panels[f"video_{i}"] = PanelData(type="image",
                                         assets={"img": np.asarray(img, np.uint8)})
    for i, img in enumerate(extras.get("render_frames", [])):
        panels[f"render_{i}"] = PanelData(type="image",
                                          assets={"img": np.asarray(img, np.uint8)})
    return panels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/paper_figures/fig4_bundle.h5")
    ap.add_argument("--width-mm", type=float, default=183.0)
    ap.add_argument("--height-mm", type=float, default=140.0)
    args = ap.parse_args(argv)

    # Port the data-loading half of notebook cell 8 here. Each `extras` key
    # below maps to a specific region of that cell; `results` and `ex` come
    # straight from `analyze_all_pairs` + the exemplar lookup.
    #
    #   fs                 song_cfg.fs
    #   start_frame        START_FRAME after the CSV-override clamp
    #   end_frame          END_FRAME   after the CSV-override clamp
    #   t_ms_full          (arange(clip_T_full) / fs) * 1000
    #   male_pitch         cfp.body_pitch_deg_from_quat(qpos[:, 3:7])
    #   target_pitch       arcsin(target_vec_z / |target_vec|), degrees
    #   per_bout_align     compute_pitch_alignment_all_sessions(...)
    #                        ['median_abs_alignment_deg']
    #   exemplar_bout_idx  index of the exemplar in per_bout_align
    #   walking_z          load__scutellum_z(FREE_WALK_H5_PATH, per_bout=True)
    #   phase_diffs        per-sine-segment Hilbert L-R phase difference
    #   ext_pulse/ext_sine pooled |extended-wing horizontal angle| by label
    #   pulse_centroids    dict {'Pslow': (W,), 'Pfast': (W,)} — the
    #                        'centroids' entry of get_pulse_type_labels(...)
    #   pulse_pooled       dict {'Pslow': (n,W), 'Pfast': (n,W)} — its
    #                        'pooled_waveforms' entry; std shading derives from
    #                        this, so omitting it silently disables show_std
    #   pulse_counts       dict {'Pslow': int, 'Pfast': int} — goes into the
    #                        pulse_class panel's figure.json `spec`, not `data`
    #   video_frames       list of uint8 HxWx3 crops, already keypoint- and
    #                        mask-overlayed (reuse cfp.panel_video_strip_with_kp
    #                        by drawing into an offscreen axes and grabbing the
    #                        buffer, OR project + draw directly with cfp._dlt_*)
    #   render_frames      list of uint8 HxWx3 MuJoCo renders, post-crop
    #
    # Every array must be plain float64/uint8 numpy — no object dtypes, or
    # write_bundle will fail.
    raise SystemExit(
        "main() is a porting task: move notebook cell 8's data-loading half "
        "here. See REQUIRED_EXTRAS below and the docstring. "
        "build_fig4_panels() is already implemented and tested."
    )


if __name__ == "__main__":
    raise SystemExit(main())
```

Add the two remaining adapters to `figbuilder/panels/courtship.py`:

```python
@register
class SineInPhasePanel(PanelType):
    id = "courtship.sine_inphase"
    label = "Sine song: extended vs folded wing in phase"
    needs = ["t_ms", "ext_z", "fold_z"]
    schema = {"type": "object", "properties": {
        **_TIME_SCHEMA, "title": {"type": "string", "default": ""}}}

    def draw(self, ax, data, spec):
        cfp.panel_sine_wing_inphase(
            ax, np.asarray(data["t_ms"]), np.asarray(data["ext_z"]),
            np.asarray(data["fold_z"]), fs=float(spec.get("fs", 800.0)),
            frame_range=_frame_range(spec),
            sine_segments=_segs_arg(data.get("sine_segments")),
            title=spec.get("title", ""),
        )


_PULSE_TYPES = ("Pslow", "Pfast")


@register
class PulseClassPanel(PanelType):
    """Reassembles the nested `pulse_type_results` dict from flat datasets.

    `panel_pulse_classification` wants {'centroids': {'Pslow': wf, 'Pfast': wf},
    'counts': {...}, 'pooled_waveforms': {'Pslow': (n, W), ...}, 'fs': float}.
    The bundle stores flat named arrays, so per-type arrays live as
    `centroid_<T>` / `pooled_<T>` and are re-nested here. Std shading is derived
    by the panel function from `pooled_waveforms` — there is no `stds` input.
    """

    id = "courtship.pulse_class"
    label = "Pslow / Pfast typed centroids"
    needs = ["centroid_Pslow", "centroid_Pfast"]
    schema = {"type": "object", "properties": {
        "show_std": {"type": "boolean", "default": True},
        "fs": {"type": "number", "default": 800.0},
        "count_Pslow": {"type": "integer", "default": 0},
        "count_Pfast": {"type": "integer", "default": 0},
        "title": {"type": "string", "default": ""}}}

    def draw(self, ax, data, spec):
        results = {
            "centroids": {t: np.asarray(data[f"centroid_{t}"], dtype=float)
                          for t in _PULSE_TYPES if f"centroid_{t}" in data},
            "counts": {t: int(spec.get(f"count_{t}", 0)) for t in _PULSE_TYPES},
            "pooled_waveforms": {t: np.asarray(data[f"pooled_{t}"], dtype=float)
                                 for t in _PULSE_TYPES if f"pooled_{t}" in data},
            "fs": float(spec.get("fs", 800.0)),
        }
        cfp.panel_pulse_classification(
            ax, results, show_std=bool(spec.get("show_std", True)),
            title=spec.get("title", ""),
        )
```

> **Already resolved (Ruling 5).** `panel_pulse_classification`
> (`utils/courtship_figure_panels.py:2124-2127`) reads exactly four keys:
> `centroids`, `counts`, `pooled_waveforms` (each a dict keyed `'Pslow'` /
> `'Pfast'`) and `fs` (float). Std shading is derived from `pooled_waveforms`
> and needs `pooled.shape[0] > 1` to appear. There is no `stds` input. The
> adapter above already matches this — do not re-derive it.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figbuilder_fig4_bundle.py -v`
Expected: 10 passed (3 direct + 7 parametrized)

- [ ] **Step 5: Commit**

```bash
git add scripts/figures/export_fig4_bundle.py scripts/__init__.py scripts/figures/__init__.py figbuilder/panels/courtship.py tests/test_figbuilder_fig4_bundle.py
git commit -m "feat(figbuilder): fig4 bundle builder and remaining courtship adapters"
```

---

## Task 11: Seed fig4.json from the current assemble_figure

**The gotcha this task exists for:** `ax.get_position()` returns coordinates
relative to the axes' parent **SubFigure**, and `assemble_figure` nests
subfigures three deep. Probe result on a 2-level nest: `get_position()` gave
`(0.125, 0.110, 0.775, 0.770)` where the true root-figure rect was
`(0.062, 0.555, 0.388, 0.385)`. Using `get_position()` directly would place
every panel wrongly.

**Files:**
- Create: `scripts/figures/seed_fig4_layout.py`
- Test: `tests/test_figbuilder_seed_layout.py`

**Interfaces:**
- Consumes: `assemble_figure` from `utils.courtship_figure_panels`; `FigureSpec`, `PanelSpec`, `save_figure` (Tasks 2, 4)
- Produces:
  - `root_rect(fig, ax) -> tuple[float, float, float, float]`
  - `seed_layout(width_mm, height_mm, n_frames_strip, n_render_strip, n_video_frames) -> FigureSpec`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figbuilder_seed_layout.py
"""Root-figure rect extraction from nested subfigures."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from scripts.figures.seed_fig4_layout import root_rect, seed_layout


def test_root_rect_differs_from_get_position_for_nested_subfigures():
    fig = plt.figure(figsize=(6, 4))
    top, _ = fig.subfigures(2, 1)
    left, _ = top.subfigures(1, 2)
    ax = left.subplots(1, 1)
    try:
        got = root_rect(fig, ax)
        assert got != pytest.approx(ax.get_position().bounds)
        # left half, top half
        assert got[0] < 0.5 and got[1] > 0.5
    finally:
        plt.close(fig)


def test_root_rect_matches_get_position_for_a_plain_axes():
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0.2, 0.3, 0.5, 0.4])
    try:
        assert root_rect(fig, ax) == pytest.approx((0.2, 0.3, 0.5, 0.4), abs=1e-6)
    finally:
        plt.close(fig)


def test_root_rects_are_all_inside_the_unit_square():
    spec = seed_layout()
    for p in spec.panels:
        x, y, w, h = p.rect
        assert 0.0 <= x < 1.0 and 0.0 <= y < 1.0
        assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def test_seed_layout_emits_one_panel_per_fig4_axes():
    spec = seed_layout(n_video_frames=4, n_render_strip=4)
    ids = {p.id for p in spec.panels}
    assert {"wing", "scut", "sine_phase", "wing_phase_polar", "angle_2d",
            "pulse_class", "zheight", "pitch", "align_violin"} <= ids
    assert {"video_0", "video_3", "render_0", "render_3"} <= ids


def test_seed_layout_creates_groups_for_the_two_strips():
    spec = seed_layout()
    gids = {g.id for g in spec.groups}
    assert {"video_strip", "render_strip"} <= gids
    assert all(p.group == "video_strip"
               for p in spec.panels if p.id.startswith("video_"))


def test_seed_layout_creates_a_panel_letter_annotation_per_lettered_panel():
    spec = seed_layout()
    letters = {a["text"] for a in spec.annotations
               if a.get("managed") == "panel_letter"}
    assert {"A", "B", "C", "D"} <= letters


def test_seed_layout_seeds_render_params_into_panel_spec():
    """Bundle attrs are metadata only, so fs must land in figure.json spec."""
    spec = seed_layout()
    by_id = {p.id: p for p in spec.panels}
    assert by_id["wing"].spec["fs"] == pytest.approx(800.0)
    assert by_id["wing"].spec["time_unit"] == "s"
    assert by_id["wing_phase_polar"].spec["center_stat"] == "median"
    assert by_id["zheight"].spec["kind"] == "violin"


def test_seed_layout_uses_the_requested_canvas_size():
    spec = seed_layout(width_mm=183.0, height_mm=140.0)
    assert spec.width_mm == pytest.approx(183.0)
    assert spec.height_mm == pytest.approx(140.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figbuilder_seed_layout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.figures.seed_fig4_layout'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/figures/seed_fig4_layout.py
"""Convert the existing `assemble_figure` layout into a `figure.json`.

This is a one-time migration: it renders the current nested-subfigure layout,
reads every axes' true ROOT-figure rect, and writes the equivalent flat
absolute-rect design document.

Why not `ax.get_position()`: it returns coordinates relative to the axes'
parent SubFigure, and `assemble_figure` nests subfigures three deep. The
correct root-figure rect comes from the axes' window extent transformed
through the ROOT figure's `transFigure`.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figbuilder.figure import FigureSpec, GroupSpec, PanelSpec, save_figure
from utils.courtship_figure_panels import (
    DEFAULT_PANEL_LETTERS, assemble_figure,
)

#: Per-panel render parameters seeded into figure.json `spec`. The bundle's
#: `attrs` are metadata only; the render path reads `spec`. Ruling 6.
DEFAULT_PANEL_SPEC: Dict[str, Dict[str, object]] = {
    "wing":             {"fs": 800.0, "time_unit": "s", "min_segment_ms": 10.0},
    "scut":             {"fs": 800.0, "time_unit": "s"},
    "sine_phase":       {"fs": 800.0, "time_unit": "s"},
    "pitch":            {"fs": 800.0, "time_unit": "s"},
    "pulse_class":      {"fs": 800.0, "show_std": True},
    "angle_2d":         {"range_deg": [0.0, 90.0]},
    "wing_phase_polar": {"center_stat": "median"},
    "zheight":          {"kind": "violin"},
}

#: axes-dict key -> figbuilder panel type
TYPE_FOR_KEY: Dict[str, str] = {
    "wing": "courtship.wing_z",
    "scut": "courtship.scutellum_z",
    "sine_phase": "courtship.sine_inphase",
    "wing_phase_polar": "courtship.wing_polar",
    "angle_2d": "courtship.angle_density",
    "pulse_class": "courtship.pulse_class",
    "zheight": "courtship.zheight",
    "pitch": "courtship.male_pitch",
    "align_violin": "courtship.pitch_violin",
}


def root_rect(fig, ax) -> Tuple[float, float, float, float]:
    """Return an axes' rect in ROOT-figure fractions."""
    fig.canvas.draw()
    bb = ax.get_window_extent().transformed(fig.transFigure.inverted())
    return (float(bb.x0), float(bb.y0), float(bb.width), float(bb.height))


def seed_layout(width_mm: float = 183.0, height_mm: float = 140.0,
                n_frames_strip: int = 6, n_render_strip: int = 4,
                n_video_frames: int = 4) -> FigureSpec:
    fig, axd = assemble_figure(
        fig_width_mm=width_mm, fig_height_mm=height_mm,
        n_frames_strip=n_frames_strip, n_render_strip=n_render_strip,
        n_video_frames=n_video_frames,
    )
    try:
        panels: List[PanelSpec] = []
        groups: List[GroupSpec] = []

        # Render parameters live in figure.json `spec` — nothing in the render
        # path reads the bundle's `attrs`. Seed `fs` etc. here rather than
        # relying on each adapter's default. See Ruling 6 in the SDD ledger.
        for key, ptype in TYPE_FOR_KEY.items():
            ax = axd[key]
            panels.append(PanelSpec(
                id=key, type=ptype, rect=root_rect(fig, ax),
                data={}, spec=dict(DEFAULT_PANEL_SPEC.get(key, {}))))

        for strip_key, gid in (("video", "video_strip"),
                               ("render", "render_strip")):
            axes = axd[strip_key]
            rects = [root_rect(fig, a) for a in axes]
            x0 = min(r[0] for r in rects)
            y0 = min(r[1] for r in rects)
            x1 = max(r[0] + r[2] for r in rects)
            y1 = max(r[1] + r[3] for r in rects)
            gutter_frac = ((x1 - x0) - sum(r[2] for r in rects)) / max(1, len(rects) - 1)
            groups.append(GroupSpec(id=gid, axis="x",
                                    gutter_mm=max(0.0, gutter_frac * width_mm),
                                    equal=True, rect=(x0, y0, x1 - x0, y1 - y0)))
            for i, r in enumerate(rects):
                panels.append(PanelSpec(id=f"{strip_key}_{i}", type="image",
                                        rect=r, group=gid,
                                        spec={"asset": "img"}))

        by_id = {p.id: p for p in panels}
        annotations: List[dict] = []
        for key, letter in DEFAULT_PANEL_LETTERS:
            target = key if key in by_id else f"{key}_0"
            if target not in by_id:
                continue
            annotations.append({
                "id": f"letter_{target}", "kind": "text",
                "managed": "panel_letter", "parent": target,
                "pos_mm": [-4.0, by_id[target].rect[3] * height_mm + 1.0],
                "text": letter,
                "style": {"font_size_pt": 8, "weight": "bold",
                          "font": "Arial, Helvetica, sans-serif"},
            })

        spec = FigureSpec(bundle="fig4_bundle.h5", panels=panels,
                          groups=groups, annotations=annotations,
                          style={"font.size": 6.0})
        spec.set_size(width_mm, height_mm)
        return spec
    finally:
        plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/paper_figures/fig4.json")
    ap.add_argument("--width-mm", type=float, default=183.0)
    ap.add_argument("--height-mm", type=float, default=140.0)
    args = ap.parse_args(argv)
    save_figure(seed_layout(args.width_mm, args.height_mm), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figbuilder_seed_layout.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/figures/seed_fig4_layout.py tests/test_figbuilder_seed_layout.py
git commit -m "feat(figbuilder): seed fig4.json from assemble_figure with root-rect conversion"
```

---

## Task 12: M0 visual acceptance gate

Per `CLAUDE.md`, a change that moves pixels is not done until a figure shows
what it did, the expectation is stated **before** the figure is generated, and
the result is read back and reported against that expectation.

**Files:**
- Create: `scripts/viz/compare_figbuilder_export.py`
- Test: `tests/test_compare_figbuilder_export.py`

**Interfaces:**
- Consumes: `svg_to` (Task 9)
- Produces:
  - `render_svg_to_png(svg_path, png_path, dpi) -> Path`
  - `side_by_side(png_a, png_b, out_path, labels) -> Path`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compare_figbuilder_export.py
"""The M0 comparison harness."""
from __future__ import annotations

import shutil

import numpy as np
import pytest
from PIL import Image

from scripts.viz.compare_figbuilder_export import (
    render_svg_to_png, side_by_side,
)

MINIMAL_SVG = (
    b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
    b'width="72pt" height="36pt" viewBox="0 0 72 36">'
    b'<rect x="4" y="4" width="20" height="10" fill="#38bdf8"/></svg>'
)


@pytest.mark.skipif(shutil.which("rsvg-convert") is None,
                    reason="rsvg-convert not on PATH")
def test_render_svg_to_png_produces_an_opaque_image(tmp_path):
    s = tmp_path / "a.svg"
    s.write_bytes(MINIMAL_SVG)
    p = render_svg_to_png(s, tmp_path / "a.png", dpi=100)
    im = Image.open(p)
    assert im.mode == "RGB"
    assert im.size[0] > 0


def test_side_by_side_width_is_the_sum_plus_the_gutter(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.fromarray(np.zeros((40, 30, 3), np.uint8)).save(a)
    Image.fromarray(np.zeros((50, 20, 3), np.uint8)).save(b)
    out = side_by_side(a, b, tmp_path / "c.png", labels=("old", "new"), gutter=10)
    im = Image.open(out)
    assert im.size[0] == 30 + 10 + 20
    assert im.size[1] >= 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compare_figbuilder_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.viz.compare_figbuilder_export'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/viz/compare_figbuilder_export.py
"""Compare a figbuilder export against the hand-finished paper SVG.

EXPECTATION (state before looking, per CLAUDE.md):

    The figbuilder export should place every panel at the same rect as
    figures/paper_figures/fig4_050426_ETTA.svg, with visually identical
    traces, tick labels, and colors.

    Panel LETTERS and CAPTIONS are expected to DIFFER: in the reference they
    were hand-placed in Inkscape, and M0's seeded layout positions them
    programmatically. Differences confined to letter/caption placement are a
    PASS. Any difference in a panel's position, size, trace shape, axis range,
    or color is a FAIL and means the seeding or the renderer is wrong.

    If the two images differ in ways NOT confined to letters and captions,
    say so plainly and do not claim the gate passed.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw


def render_svg_to_png(svg_path: str | Path, png_path: str | Path,
                      dpi: int = 200) -> Path:
    """Rasterize an SVG onto a white background."""
    svg_path, png_path = Path(svg_path), Path(png_path)
    exe = shutil.which("rsvg-convert")
    if exe is None:
        raise RuntimeError("rsvg-convert not found on PATH")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "-f", "png", "-d", str(dpi), "-p", str(dpi),
                    str(svg_path), "-o", str(png_path)],
                   check=True, capture_output=True)
    im = Image.open(png_path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1])
        bg.save(png_path)
    return png_path


def side_by_side(png_a: str | Path, png_b: str | Path, out_path: str | Path,
                 labels: Sequence[str] = ("A", "B"), gutter: int = 24) -> Path:
    a, b = Image.open(png_a).convert("RGB"), Image.open(png_b).convert("RGB")
    w, h = a.width + gutter + b.width, max(a.height, b.height) + 24
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(a, (0, 24))
    canvas.paste(b, (a.width + gutter, 24))
    d = ImageDraw.Draw(canvas)
    d.text((4, 6), labels[0], fill="black")
    d.text((a.width + gutter + 4, 6), labels[1], fill="black")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference",
                    default="figures/paper_figures/fig4_050426_ETTA.svg")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out-dir", default="figures/2026-08-24-figbuilder-m0")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    ref_png = render_svg_to_png(args.reference, out / "reference.png", args.dpi)
    cand_png = render_svg_to_png(args.candidate, out / "candidate.png", args.dpi)
    combo = side_by_side(ref_png, cand_png, out / "side_by_side.png",
                         labels=("reference (Inkscape-finished)",
                                 "figbuilder export"))
    print(__doc__)
    print(f"wrote {combo}")
    print("Open this PNG with the Read tool and report what it shows "
          "against the stated expectation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_compare_figbuilder_export.py -v`
Expected: 2 passed (1 may skip without `rsvg-convert`)

- [ ] **Step 5: Run the gate and READ the output**

On the machine with `/data2`:

```bash
export MUJOCO_GL=egl
python -m scripts.figures.export_fig4_bundle --out figures/paper_figures/fig4_bundle.h5
python -m scripts.figures.seed_fig4_layout --out figures/paper_figures/fig4.json
python -m figbuilder export figures/paper_figures/fig4.json \
    -o figures/2026-08-24-figbuilder-m0/fig4_figbuilder.svg --format svg
python -m scripts.viz.compare_figbuilder_export \
    --candidate figures/2026-08-24-figbuilder-m0/fig4_figbuilder.svg
```

Then **open `figures/2026-08-24-figbuilder-m0/side_by_side.png` with the Read
tool** and report what it actually shows against the expectation in the module
docstring. Do not record the gate as passed without having looked. If panels
disagree in position, size, trace shape, axis range, or color, that is a
failure — say so and fix it before continuing.

- [ ] **Step 6: Commit**

```bash
git add scripts/viz/compare_figbuilder_export.py tests/test_compare_figbuilder_export.py figures/paper_figures/fig4.json
git commit -m "feat(figbuilder): M0 visual acceptance gate vs the paper SVG"
```

> `figures/` is gitignored, so `fig4.json` needs `git add -f`, OR add a
> `!figures/paper_figures/*.json` negation to `.gitignore`. Prefer the
> negation — the design artifact is meant to be tracked. Make that
> `.gitignore` edit part of this commit.

---

## Task 13: FastAPI server

**Files:**
- Create: `figbuilder/server.py`
- Modify: `requirements.txt` (add `fastapi`)
- Test: `tests/test_figbuilder_server.py`

**Interfaces:**
- Consumes: Tasks 1–9
- Produces: `create_app(figure_path, bundle_path=None) -> FastAPI` with routes
  `GET /api/bundle`, `GET /api/figure`, `POST /api/panel`, `GET /api/dataset`,
  `POST /api/export`, `GET /api/panel-types`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figbuilder_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'figbuilder.server'` (or a skip if `fastapi` is not yet installed — install it in Step 3)

- [ ] **Step 3: Write minimal implementation**

```bash
python -m pip install fastapi
echo "fastapi" >> requirements.txt
```

```python
# figbuilder/server.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figbuilder_server.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add figbuilder/server.py requirements.txt tests/test_figbuilder_server.py
git commit -m "feat(figbuilder): FastAPI server exposing bundle, tiles, and export"
```

---

## Task 14: Read-only browser canvas + annotation emitter conformance

Closes M1. The canvas displays composed tiles at their true rects with zoom and
pan; no editing. This proves tile composition works in a browser and gives M2
something to build on. The conformance test locks the two annotation emitters
together before M3 depends on both.

**Files:**
- Create: `web/package.json`, `web/vite.config.ts`, `web/index.html`, `web/src/main.tsx`, `web/src/api.ts`, `web/src/canvas/Canvas.tsx`, `web/src/annot/emit.ts`, `web/src/annot/emit.test.ts`
- Create: `tests/test_figbuilder_annot_conformance.py`
- Test: `web/src/annot/emit.test.ts` (vitest), `tests/test_figbuilder_annot_conformance.py` (pytest)

**Interfaces:**
- Consumes: `/api/figure`, `/api/panel` (Task 13); `emit_annotations` (Task 8)
- Produces:
  - TS `emitAnnotations(fig: FigureSpec, anns: Annotation[]): string` — returns an SVG fragment
  - `web/src/annot/fixtures.json` — the shared conformance fixture, read by BOTH test suites

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_figbuilder_annot_conformance.py
"""The Python and TypeScript annotation emitters must agree.

The emitter is duplicated on purpose (60 fps live editing in the browser,
headless export in Python). This test is what makes that duplication safe.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from lxml import etree

from figbuilder.annot import emit_annotations
from figbuilder.figure import FigureSpec, PanelSpec

FIXTURES = Path("web/src/annot/fixtures.json")


def _fig() -> FigureSpec:
    return FigureSpec(width_mm=100.0, height_mm=50.0, panels=[
        PanelSpec(id="wing", type="line", rect=(0.1, 0.5, 0.8, 0.4))])


def _normalize(svg: str) -> str:
    """Collapse whitespace and round numbers so the two emitters can be compared."""
    svg = re.sub(r"(\d+\.\d{3})\d+", r"\1", svg)
    return re.sub(r"\s+", " ", svg).strip()


def test_fixture_file_covers_every_annotation_kind():
    from figbuilder.annot import ANNOTATION_KINDS
    anns = json.loads(FIXTURES.read_text())
    assert {a["kind"] for a in anns} == set(ANNOTATION_KINDS)


@pytest.mark.skipif(not (Path("web/node_modules").exists()),
                    reason="web deps not installed (run `npm ci` in web/)")
def test_python_and_typescript_emitters_agree():
    anns = json.loads(FIXTURES.read_text())
    py = _normalize("".join(
        etree.tostring(e, encoding="unicode") for e in emit_annotations(_fig(), anns)))
    ts = _normalize(subprocess.run(
        ["npx", "tsx", "src/annot/emitCli.ts"], cwd="web",
        capture_output=True, text=True, check=True).stdout)
    assert py == ts
```

```typescript
// web/src/annot/emit.test.ts
import { describe, expect, it } from 'vitest';
import { emitAnnotations, num, type FigureSpec } from './emit';

const fig: FigureSpec = {
  width_mm: 100, height_mm: 50,
  panels: [{ id: 'wing', rect: [0.1, 0.5, 0.8, 0.4] }],
};

describe('emitAnnotations', () => {
  it('flips y from figure space to SVG space', () => {
    const svg = emitAnnotations(fig, [
      { id: 't', kind: 'text', pos_mm: [0, 0], text: 'x' }]);
    // Must match figbuilder/annot.py byte-for-byte, not JS full precision.
    expect(svg).toContain('y="141.732283"');
  });

  it('offsets a panel-parented annotation from that panel corner', () => {
    const svg = emitAnnotations(fig, [
      { id: 'L', kind: 'text', parent: 'wing', pos_mm: [0, 0], text: 'A' }]);
    expect(svg).toContain('x="28.346457"');
  });

  it('formats numbers the same way the Python emitter does', () => {
    expect(num(0.8)).toBe('0.8');
    expect(num(8)).toBe('8');
    expect(num(141.73228346456693)).toBe('141.732283');
    expect(num(-0)).toBe('0');
  });

  it('rejects an unknown kind by name', () => {
    expect(() => emitAnnotations(fig, [{ id: 'z', kind: 'hologram' } as never]))
      .toThrow(/unknown annotation kind/);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_figbuilder_annot_conformance.py -v`
Expected: FAIL — `FileNotFoundError: web/src/annot/fixtures.json`

- [ ] **Step 3: Write minimal implementation**

Scaffold the front-end:

```bash
mkdir -p web/src/{canvas,annot,api}
cd web && npm create vite@latest . -- --template react-ts --yes && npm install && npm install -D vitest tsx
```

`web/src/annot/fixtures.json` — one entry per kind, mirroring
`ANNOTATION_KINDS`:

```json
[
  {"id": "a_text", "kind": "text", "pos_mm": [4, 44], "text": "A",
   "style": {"font_size_pt": 8, "weight": "bold"}},
  {"id": "a_line", "kind": "line", "pos_mm": [10, 10], "to_mm": [30, 20]},
  {"id": "a_arrow", "kind": "arrow", "pos_mm": [12, 12], "to_mm": [34, 24]},
  {"id": "a_leader", "kind": "leader", "pos_mm": [14, 14], "to_mm": [36, 26]},
  {"id": "a_rect", "kind": "rect", "pos_mm": [40, 10], "size_mm": [12, 8]},
  {"id": "a_ellipse", "kind": "ellipse", "pos_mm": [56, 10], "size_mm": [10, 6]},
  {"id": "a_bracket", "kind": "bracket", "pos_mm": [70, 10], "to_mm": [88, 10],
   "tick_mm": 1.5},
  {"id": "a_scalebar", "kind": "scalebar", "pos_mm": [70, 30],
   "length_mm": 10, "label": "1 mm"},
  {"id": "a_image", "kind": "image", "pos_mm": [4, 4], "size_mm": [8, 8],
   "href": "data:image/png;base64,iVBORw0KGgo="}
]
```

`web/src/annot/emit.ts` — a direct port of `figbuilder/annot.py` (Task 8),
which is the specification. Mirror it element-for-element and
attribute-for-attribute: one y-flip, the same attribute ORDER, and the arrow
marker `defs` emitted FIRST whenever any arrow is present. The conformance
test will fail on any divergence. Structure:

```typescript
// web/src/annot/emit.ts
export const PT_PER_MM = 72 / 25.4;

/**
 * Mirrors figbuilder.annot._num EXACTLY. Six decimals, trailing zeros trimmed.
 * Do not substitute template interpolation of a raw number — that prints
 * JS full precision and diverges from Python. See Ruling 3 in the SDD ledger.
 */
export function num(v: number): string {
  const out = v.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
  // Mirrors the Python guard: "-0" and "" both collapse to "0".
  return ['', '-', '0', '-0'].includes(out) ? '0' : out;
}

export const ANNOTATION_KINDS = [
  'text', 'line', 'arrow', 'leader', 'rect', 'ellipse', 'bracket',
  'scalebar', 'image',
] as const;
export type Kind = (typeof ANNOTATION_KINDS)[number];

export interface PanelRef { id: string; rect: [number, number, number, number] }
export interface FigureSpec {
  width_mm: number; height_mm: number; panels: PanelRef[];
}
export interface Annotation {
  id?: string; kind: Kind; parent?: string;
  pos_mm?: [number, number]; to_mm?: [number, number];
  size_mm?: [number, number]; length_mm?: number; tick_mm?: number;
  text?: string; label?: string; href?: string;
  style?: Record<string, string | number>;
}

/** Mirrors figbuilder.annot._Frame exactly. */
class Frame {
  constructor(
    private wMm: number, private hMm: number,
    private oxMm = 0, private oyMm = 0,
  ) {}
  static of(fig: FigureSpec, parent?: string): Frame {
    if (!parent) return new Frame(fig.width_mm, fig.height_mm);
    const p = fig.panels.find((q) => q.id === parent);
    if (!p) throw new Error(`unknown parent panel ${parent}`);
    return new Frame(fig.width_mm, fig.height_mm,
      p.rect[0] * fig.width_mm, p.rect[1] * fig.height_mm);
  }
  x(mm: number) { return (this.oxMm + mm) * PT_PER_MM; }
  y(mm: number) { return (this.hMm - (this.oyMm + mm)) * PT_PER_MM; }
  d(mm: number) { return mm * PT_PER_MM; }
}

/** Mirrors figbuilder.annot._style — same key order, same output. */
function styleStr(d: Record<string, string | number>): string {
  const out: string[] = [];
  if ('color' in d) out.push(`fill:${d.color}`);
  if ('stroke' in d) out.push(`stroke:${d.stroke}`);
  if ('lw_pt' in d) out.push(`stroke-width:${num(Number(d.lw_pt))}`);
  if ('font_size_pt' in d) out.push(`font-size:${num(Number(d.font_size_pt))}px`);
  if (d.weight === 'bold' || d.weight === 700) out.push('font-weight:700');
  if ('font' in d) out.push(`font-family:${d.font}`);
  if ('opacity' in d) out.push(`opacity:${num(Number(d.opacity))}`);
  return out.join(';');
}

export function emitAnnotations(fig: FigureSpec, anns: Annotation[]): string {
  const parts: string[] = [];
  let needsMarker = false;

  for (const a of anns) {
    if (!(ANNOTATION_KINDS as readonly string[]).includes(a.kind)) {
      throw new Error(
        `unknown annotation kind ${a.kind}; expected one of ` +
        `${[...ANNOTATION_KINDS].sort().join(', ')}`);
    }
    const f = Frame.of(fig, a.parent);
    const st = { ...(a.style ?? {}) };
    const [px, py] = a.pos_mm ?? [0, 0];
    const idAttr = a.id ? ` id="${a.id}"` : '';

    switch (a.kind) {
      case 'text': {
        if (!('font' in st)) st.font = 'Arial, Helvetica, sans-serif';
        parts.push(
          `<text x="${num(f.x(px))}" y="${num(f.y(py))}" ` +
          `style="${styleStr(st)}"${idAttr}>${a.text ?? ''}</text>`);
        break;
      }
      case 'arrow': {
        const [tx, ty] = a.to_mm ?? [0, 0];
        if (!('stroke' in st)) st.stroke = '#000000';
        if (!('lw_pt' in st)) st.lw_pt = 0.8;
        needsMarker = true;
        parts.push(
          `<path d="M ${num(f.x(px))},${num(f.y(py))} L ${num(f.x(tx))},${num(f.y(ty))}" ` +
          `style="${styleStr(st)};fill:none;` +
          `marker-end:url(#fb_arrowhead)"${idAttr}/>`);
        break;
      }
      // ...remaining kinds: port each from figbuilder/annot.py verbatim.
      default:
        throw new Error(`kind ${a.kind} not yet ported`);
    }
  }

  if (needsMarker) {
    parts.unshift(
      '<defs><marker id="fb_arrowhead" viewBox="0 0 10 10" refX="9" refY="5" ' +
      'markerWidth="5" markerHeight="5" orient="auto-start-reverse">' +
      '<path d="M 0,0 L 10,5 L 0,10 z"/></marker></defs>');
  }
  return parts.join('');
}
```

`web/src/annot/emitCli.ts` — prints the fixture emission for the conformance
test, using the same figure as the Python side:

```typescript
// web/src/annot/emitCli.ts
import { readFileSync } from 'node:fs';
import { emitAnnotations, type Annotation, type FigureSpec } from './emit';

const fig: FigureSpec = {
  width_mm: 100, height_mm: 50,
  panels: [{ id: 'wing', rect: [0.1, 0.5, 0.8, 0.4] }],
};
const anns = JSON.parse(
  readFileSync(new URL('./fixtures.json', import.meta.url), 'utf8'),
) as Annotation[];
process.stdout.write(emitAnnotations(fig, anns));
```

`web/src/api.ts` — typed `fetch` wrappers:

```typescript
// web/src/api.ts
const BASE = import.meta.env.VITE_API ?? 'http://127.0.0.1:8765';

export interface PanelSpecDTO {
  id: string; type: string; rect: [number, number, number, number];
  data: Record<string, { dataset: string; slice?: string }>;
  spec: Record<string, unknown>; group?: string | null; z?: number;
}
export interface FigureDoc {
  figure: { width_mm: number; height_mm: number; dpi: number };
  panels: PanelSpecDTO[]; groups: unknown[]; annotations: unknown[];
}
export interface TileDTO {
  svg: string; ink_box: [number, number, number, number];
  cache_hit: boolean; overflows: boolean;
}

const get = async <T>(path: string) => {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return (await r.json()) as T;
};

export const getFigure = () => get<FigureDoc>('/api/figure');
export const getBundle = () => get<unknown>('/api/bundle');
export const getPanelTypes = () => get<unknown[]>('/api/panel-types');

export async function renderPanel(panel: PanelSpecDTO): Promise<TileDTO> {
  const r = await fetch(`${BASE}/api/panel`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ panel }),
  });
  if (!r.ok) throw new Error(`/api/panel: ${r.status}`);
  return (await r.json()) as TileDTO;
}
```

`web/src/canvas/Canvas.tsx` — read-only: fetch the figure, request one tile per
panel, overlay them in one `<svg>`. Wheel zooms, drag pans. No selection, no
dragging of panels — that is M2.

```tsx
// web/src/canvas/Canvas.tsx
import { useEffect, useRef, useState } from 'react';
import { getFigure, renderPanel, type FigureDoc, type TileDTO } from '../api';

const PT_PER_MM = 72 / 25.4;
const SEP = '__';

/** Mirrors figbuilder/svgutil.py:namespace_ids. Longest id first. */
function namespaceIds(svg: string, prefix: string): string {
  const ids = [...svg.matchAll(/id="([^"]+)"/g)].map((m) => m[1]);
  let out = svg;
  for (const id of [...new Set(ids)].sort((a, b) => b.length - a.length)) {
    const to = `${prefix}${SEP}${id}`;
    out = out.split(`id="${id}"`).join(`id="${to}"`);
    out = out.split(`#${id}"`).join(`#${to}"`);
    out = out.split(`#${id})`).join(`#${to})`);
  }
  return out;
}

/** Strip the tile's <svg> envelope; we only want its children. */
function tileInner(svg: string): string {
  const open = svg.indexOf('>', svg.indexOf('<svg'));
  const close = svg.lastIndexOf('</svg>');
  return open < 0 || close < 0 ? '' : svg.slice(open + 1, close);
}

export function Canvas() {
  const [doc, setDoc] = useState<FigureDoc | null>(null);
  const [tiles, setTiles] = useState<Record<string, TileDTO>>({});
  const [view, setView] = useState({ scale: 2, tx: 0, ty: 0 });
  const drag = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => { void getFigure().then(setDoc); }, []);

  useEffect(() => {
    if (!doc) return;
    let alive = true;
    void (async () => {
      for (const p of doc.panels) {
        const t = await renderPanel(p);
        if (!alive) return;
        setTiles((prev) => ({ ...prev, [p.id]: t }));
      }
    })();
    return () => { alive = false; };
  }, [doc]);

  if (!doc) return <p>loading…</p>;
  const wPt = doc.figure.width_mm * PT_PER_MM;
  const hPt = doc.figure.height_mm * PT_PER_MM;

  return (
    <div
      style={{ overflow: 'hidden', width: '100%', height: '100vh',
               background: '#f4f4f5', cursor: drag.current ? 'grabbing' : 'grab' }}
      onWheel={(e) => {
        e.preventDefault();
        setView((v) => ({ ...v,
          scale: Math.min(12, Math.max(0.25, v.scale * (e.deltaY < 0 ? 1.1 : 1 / 1.1))) }));
      }}
      onPointerDown={(e) => { drag.current = { x: e.clientX, y: e.clientY }; }}
      onPointerUp={() => { drag.current = null; }}
      onPointerMove={(e) => {
        if (!drag.current) return;
        const dx = e.clientX - drag.current.x;
        const dy = e.clientY - drag.current.y;
        drag.current = { x: e.clientX, y: e.clientY };
        setView((v) => ({ ...v, tx: v.tx + dx, ty: v.ty + dy }));
      }}
    >
      <svg
        width={wPt * view.scale} height={hPt * view.scale}
        viewBox={`0 0 ${wPt} ${hPt}`}
        style={{ transform: `translate(${view.tx}px, ${view.ty}px)`,
                 background: 'white', boxShadow: '0 1px 8px rgba(0,0,0,.15)' }}
      >
        {doc.panels.map((p) => {
          const t = tiles[p.id];
          if (!t) return null;
          return (
            <g key={p.id} id={`panel_${p.id}`}
               dangerouslySetInnerHTML={{
                 __html: namespaceIds(tileInner(t.svg), p.id) }} />
          );
        })}
      </svg>
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_figbuilder_annot_conformance.py -v`
Expected: 2 passed
Run: `cd web && npx vitest run`
Expected: 3 passed

- [ ] **Step 5: Verify the canvas visually**

```bash
python -m figbuilder serve figures/paper_figures/fig4.json &
cd web && npm run dev
```

Open the dev server, screenshot the canvas, and **open the screenshot with the
Read tool**. Expectation: the canvas shows the same panel arrangement as the
M0 export, at the same relative positions, with no overlapping or mispositioned
panels and no missing tiles. Report what it shows.

- [ ] **Step 6: Commit**

```bash
git add web/ tests/test_figbuilder_annot_conformance.py
git commit -m "feat(figbuilder): read-only browser canvas and annotation emitter conformance"
```
