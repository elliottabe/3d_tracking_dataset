"""Tests for the scaling.scale_keypoints mode (scripts/scale_keypoints.py)."""
from __future__ import annotations

import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from scripts.scale_keypoints import resolve_scale_keypoints

KP = ["Scutellum", "WingL_base", "T1L_FeTi", "T1R_FeTi", "Abd_tip"]


def cfg_with(mode=None):
    scaling = {"trunk_keypoints": ["Scutellum", "WingL_base", "Abd_tip"],
               "estimator": "umeyama", "robust_stat": "median", "robust": "none"}
    if mode is not None:
        scaling["scale_keypoints"] = mode
    return OmegaConf.create({"scaling": scaling})


def test_default_is_trunk():
    assert resolve_scale_keypoints(cfg_with(), KP) == \
        ["Scutellum", "WingL_base", "Abd_tip"]


def test_trunk_explicit():
    assert resolve_scale_keypoints(cfg_with("trunk"), KP) == \
        ["Scutellum", "WingL_base", "Abd_tip"]


def test_all_uses_every_keypoint():
    assert resolve_scale_keypoints(cfg_with("all"), KP) == KP


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="scale_keypoints"):
        resolve_scale_keypoints(cfg_with("some"), KP)


def test_run_bout_direct_invocation_imports():
    """run_bout.py is launched as `python scripts/run_bout.py` (sys.path[0] = scripts/);
    a plain `from scripts...` import there crashes every pipeline launcher (regression:
    2026-08-07 benchmark run). --help exits 0 only if all module-level imports succeed."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "scripts/run_bout.py", "--help"],
                        capture_output=True, text=True, cwd=repo_root, timeout=240)
    assert "ModuleNotFoundError" not in r.stderr, r.stderr[-2000:]
    assert r.returncode == 0, r.stderr[-2000:]
