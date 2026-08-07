"""Benchmark manifest: which bouts form the frozen benchmark, and where they live.

Two source layouts exist today (verified on disk):
  canonical: <source_root>/bouts/bout_XXXXX/flyN   (courtship pose/ dirs)
  flat:      <source_root>/bouts/bout_XXXXX/flyN   (legacy free-running *_bouts dirs)
Both place bouts under <source_root>/bouts; `layout` is kept explicit in the
manifest anyway because future sources may differ and the freezer/runner treat
the two assays differently (sex.json, num_animals).
"""
from __future__ import annotations

from pathlib import Path

import yaml

VALID_LAYOUTS = ("canonical", "flat")
REQUIRED_KEYS = ("run_key", "layout", "source_root", "recording_cfg",
                 "session_dir", "assay", "bout", "flies")


def load_manifest(path: Path) -> dict:
    manifest = yaml.safe_load(Path(path).read_text())
    seen = set()
    for e in manifest["bouts"]:
        missing = [k for k in REQUIRED_KEYS if k not in e]
        if missing:
            raise ValueError(f"manifest entry missing keys {missing}: {e}")
        if e["layout"] not in VALID_LAYOUTS:
            raise ValueError(f"bad layout {e['layout']!r} (want {VALID_LAYOUTS})")
        key = (e["run_key"], e["bout"])
        if key in seen:
            raise ValueError(f"duplicate manifest entry: {key}")
        seen.add(key)
        e.setdefault("tags", [])
        e.setdefault("render_frames", [])
    return manifest


def entries(manifest: dict) -> list[dict]:
    return list(manifest["bouts"])


def bout_fly_dir(entry: dict, root: Path | None = None) -> Path:
    """Bout-fly dir for entry['fly'] — source layout, or mirrored under `root`."""
    rel = Path("bouts") / f"bout_{int(entry['bout']):05d}" / f"fly{int(entry['fly'])}"
    if root is not None:
        return Path(root) / entry["run_key"] / rel
    return Path(entry["source_root"]) / rel
