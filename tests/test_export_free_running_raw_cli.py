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
