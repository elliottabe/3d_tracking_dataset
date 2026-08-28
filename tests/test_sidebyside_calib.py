"""The sidebyside right panel must use the SESSION'S OWN calibration.

`viz.config.courtship_recording()` returns the DEFAULT recording (Session0).
The view already overrides session_dir/predictions_dir, but the calibration was
still taken from that default -- so a Session1 bout rendered its right panel
from Session0's camera poses. The two panels then showed the fly from two
different viewpoints, which is precisely the failure this panel exists to
remove: measured on Session1/2026_04_02_14_54_28 bout_00018 fly0, the SAM
outline sat 28.1 px from the rendered fly, dropping to 15.4 px once the right
calibration was used (the remainder is the wing-pose disagreement).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "viz" / "views" / "sidebyside.py"
CLI = Path(__file__).resolve().parents[1] / "viz" / "cli.py"
RUN = Path(__file__).resolve().parents[1] / "scripts" / "run_bout.py"


@pytest.fixture(scope="module")
def src():
    return SRC.read_text()


def test_camera_matrices_never_read_the_default_recording(src):
    """The bug in one line: `rec["calib_dir"]` ignores the session override."""
    assert 'camera_matrices(rec["calib_dir"])' not in src, (
        "the right panel must not build cameras from the DEFAULT recording's "
        "calibration; use the session's own")
    assert src.count("camera_matrices(calib_dir)") >= 1


def test_calib_dir_defaults_to_the_overridden_session(src):
    fn = [n for n in ast.parse(src).body
          if isinstance(n, ast.FunctionDef) and n.name == "run"][0]
    body = ast.get_source_segment(src, fn)
    i = body.index("calib_dir")
    seg = body[i:i + 700]
    assert 'os.path.join(session_dir, "calibration")' in seg, (
        "when session_dir is overridden the calibration must follow it")
    assert 'getattr(args, "session_dir", None)' in seg, (
        "must fall back to the default recording ONLY when no session override")


def test_a_missing_calibration_fails_loudly(src):
    assert "calibration dir not found" in src, (
        "a wrong/missing calib dir must raise, not silently render the wrong view")


def test_cli_exposes_calib_dir():
    s = CLI.read_text()
    assert '"--calib-dir"' in s and 'dest="calib_dir"' in s


def test_run_bout_passes_this_recordings_calibration():
    s = RUN.read_text()
    assert '"--calib-dir", str(cfg.recording.calib_dir),' in s, (
        "the pipeline must hand the view its own recording's calibration")
