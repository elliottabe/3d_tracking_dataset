"""The raw-export CLI must be importable: the walking->running rename
(451efb9-era) stripped 'free_walking' from identifiers, leaving
`from utils._loader import export_raw__h5` pointing at a module that does
not exist (the loader is utils/free_walking_loader.py)."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_export_cli_imports():
    spec = importlib.util.spec_from_file_location(
        "export_free_walking_raw",
        REPO / "scripts" / "export" / "export_free_walking_raw.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # raises ModuleNotFoundError if broken
    assert callable(mod.export_raw_free_running_h5)


def test_pack_cli_runs_standalone(tmp_path):
    """pack_reference_clips must bootstrap the repo root itself: running
    `python scripts/export/pack_reference_clips.py` puts scripts/export/ (not
    the repo root) on sys.path, so its `from utils...` import dies with
    ModuleNotFoundError without the bootstrap (broke the NewBouts recombine
    chain, job 38607704)."""
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "export" / "pack_reference_clips.py"),
         "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
