import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_LAUNCHER = _REPO / "scripts" / "slurm_predict_session_dir.py"
_spec = importlib.util.spec_from_file_location("slurm_predict_session_dir", _LAUNCHER)
spsd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spsd)


def test_find_recordings_keeps_only_calibrated_dirs(tmp_path):
    root = tmp_path / "Session1"
    for rec in ("2026_04_02_11_52_43", "2026_04_02_12_11_50"):
        (root / rec / "calibration").mkdir(parents=True)
    (root / "quality_misc").mkdir(parents=True)          # no calibration/ -> excluded
    (root / "notes.txt").write_text("x")                 # not a dir -> excluded
    got = spsd.find_recordings(str(root))
    assert [os.path.basename(r) for r in got] == [
        "2026_04_02_11_52_43", "2026_04_02_12_11_50"]


def test_dry_run_emits_one_script_per_recording(tmp_path):
    """End-to-end dry-run: composes real configs, prints a processed-tree job
    per recording. Requires the 3d_tracking env (hydra)."""
    root = tmp_path / "courtship" / "Session1"
    for rec in ("recA", "recB"):
        (root / rec / "calibration").mkdir(parents=True)
    out = subprocess.run(
        [sys.executable, "scripts/slurm_predict_session_dir.py",
         "--run-name", "run4", "--session-root", str(root), "--dry-run"],
        cwd=_REPO, capture_output=True, text=True)
    assert out.returncode == 0, f"STDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}"
    s = out.stdout
    assert s.count("scripts/sam3_masks.py") == 2          # one job per recording
    assert s.count("scripts/predict_session.py") == 2
    assert "processed/courtship/Session1/recA/sam3_masks" in s
    assert "processed/courtship/Session1/recB/predictions" in s
    assert "run_id=run4" in s
