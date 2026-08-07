"""Tests for the scaling.scale_keypoints mode (scripts/scale_keypoints.py)."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from scripts.scale_keypoints import resolve_scale_keypoints

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def test_scale_keypoints_importable_both_ways():
    """run_bout.py is invoked as `python scripts/run_bout.py` (sys.path[0]=scripts/)
    AND imported as scripts.run_bout — the helper must import under both."""
    code = ("import sys; sys.path.insert(0, 'scripts'); "
            "import scale_keypoints; "
            "print(scale_keypoints.resolve_scale_keypoints.__name__)")
    r = subprocess.run([sys.executable, "-c", code],
                        capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0 and "resolve_scale_keypoints" in r.stdout, r.stderr
