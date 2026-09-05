# tests/test_coarse_pass_timeline_mvq_refusal.py
"""`scripts/viz/coarse_pass_timeline.py` must refuse an mvq-schema
`coarse_tracks.npz` (review's final-fix-wave minors).

Expectation: the SAM3-only panels this script draws (`area_ratio`,
`border_med`) are ALL-NaN on an mvq file (no masks -- see
`coarse_track.py`'s module docstring), so a silent run would emit a figure
whose top two panels are blank and say nothing -- worse than no figure at
all, since a viewer has no signal that the tool was pointed at the wrong
file. It must instead raise (via `SystemExit`, matching this script's other
CLI-usage errors) naming `scripts/viz/coarse_tracks_check.py` as the mvq
replacement.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
# Deliberately NOT `REPO / "scripts"`: `scripts/viz/` has its own `__init__.py`,
# so with `scripts/` on the path `import viz` resolves to THAT namesake
# package instead of the real `viz/` at the repo root -- the exact trap
# `run_bout.py`'s module docstring documents and `coarse_centres_check.py`'s
# own test avoids the same way.
for _p in (str(REPO / "third_party" / "jarvis_jax"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_coarse_figure_gates import _build_tracks  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, str(REPO / "scripts" / "viz" / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_refuses_an_mvq_schema_file_and_points_at_the_mvq_script(tmp_path, monkeypatch):
    # `coarse_pass_timeline.py` itself does `sys.path.insert(0, "<repo>/scripts")`
    # at import time (so its own `from coarse_pass_gates import ...` resolves).
    # That prepend OUTLIVES this test and, left in place, shadows the real
    # top-level `viz` package with `scripts/viz`'s own `__init__.py` for every
    # later test in the same session -- the exact trap `run_bout.py`'s module
    # docstring documents. Restore `sys.path` afterward so loading this one
    # script here cannot break an unrelated later test.
    _snapshot = list(sys.path)
    try:
        tracks_path, _coarse_frame = _build_tracks(tmp_path)
        mod = _load_script("coarse_pass_timeline")
        monkeypatch.setattr(sys, "argv", ["coarse_pass_timeline.py", "--tracks", tracks_path,
                                          "--out", str(tmp_path / "timeline.png")])
        with pytest.raises(SystemExit, match="coarse_tracks_check.py"):
            mod.main()
    finally:
        sys.path[:] = _snapshot
