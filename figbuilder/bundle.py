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
