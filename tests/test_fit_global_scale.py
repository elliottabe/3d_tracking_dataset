"""TDD tests for scripts/fit_global_scale.py -- pure search logic only, no GPU.

``score_scale`` (the real IK scorer) is exercised only by the GPU validation
run in the task report, never here: every test in this file either drives
the pure 1-D search primitives directly, or injects a synthetic ``score_fn``
into ``fit_global_scale`` so the search logic is covered without touching
jaxls/stac_mjx at all.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.fit_global_scale import (
    candidate_scales,
    fit_global_scale,
    golden_section,
)


# ---------------------------------------------------------------------------
# candidate_scales
# ---------------------------------------------------------------------------

def test_candidate_scales_log_spaced_and_endpoints_included():
    scales = candidate_scales(0.008, 0.016, 5)
    assert len(scales) == 5
    assert scales[0] == pytest.approx(0.008)
    assert scales[-1] == pytest.approx(0.016)
    # log spacing: consecutive ratios are constant
    ratios = [scales[i + 1] / scales[i] for i in range(len(scales) - 1)]
    assert ratios == pytest.approx([ratios[0]] * len(ratios), rel=1e-9)
    assert ratios[0] > 1.0


def test_candidate_scales_respects_n():
    assert len(candidate_scales(1.0, 100.0, 1)) == 1
    assert candidate_scales(1.0, 100.0, 1) == [1.0]
    assert len(candidate_scales(1.0, 100.0, 10)) == 10


def test_candidate_scales_deterministic():
    a = candidate_scales(0.01, 0.02, 7)
    b = candidate_scales(0.01, 0.02, 7)
    assert a == b


def test_candidate_scales_rejects_bad_bounds():
    with pytest.raises(ValueError):
        candidate_scales(0.02, 0.01, 5)
    with pytest.raises(ValueError):
        candidate_scales(-1.0, 1.0, 5)
    with pytest.raises(ValueError):
        candidate_scales(0.01, 0.02, 0)


# ---------------------------------------------------------------------------
# golden_section
# ---------------------------------------------------------------------------

def test_golden_section_finds_known_minimum():
    target = 0.0123
    f = lambda x: (math.log(x) - math.log(target)) ** 2
    best, info = golden_section(f, 0.005, 0.05, tol_rel=0.001, max_evals=60)
    assert best == pytest.approx(target, rel=0.02)
    assert info["n_evals"] <= 60
    assert info["n_evals"] >= 2


def test_golden_section_respects_max_evals_default():
    target = 0.02
    f = lambda x: (math.log(x) - math.log(target)) ** 2
    best, info = golden_section(f, 0.01, 0.04)  # default tol_rel/max_evals
    assert info["n_evals"] <= 12
    # even with only the default budget, should land in the right ballpark
    assert best == pytest.approx(target, rel=0.1)


def test_golden_section_flat_function_terminates():
    calls = {"n": 0}

    def flat(x):
        calls["n"] += 1
        return 1.0

    best, info = golden_section(flat, 0.01, 0.02, tol_rel=0.01, max_evals=12)
    assert info["n_evals"] <= 12
    assert calls["n"] <= 12
    assert 0.01 <= best <= 0.02


def test_golden_section_never_evaluates_outside_bracket():
    lo, hi = 0.01, 0.05
    seen = []

    def f(x):
        seen.append(x)
        return (math.log(x) - math.log(0.033)) ** 2

    golden_section(f, lo, hi, max_evals=20)
    assert all(lo <= x <= hi for x in seen)
    # and the evals trace reported back matches what was actually sampled
    _, info = golden_section(f, lo, hi, max_evals=20)
    assert all(lo <= x <= hi for x in info["evals"])


def test_golden_section_rejects_bad_bracket():
    with pytest.raises(ValueError):
        golden_section(lambda x: x, 0.02, 0.01)
    with pytest.raises(ValueError):
        golden_section(lambda x: x, -1.0, 1.0)


# ---------------------------------------------------------------------------
# fit_global_scale -- injected score_fn seam, synthetic bout dirs, no GPU
# ---------------------------------------------------------------------------

def _write_bout_dir(tmp_path, name, kp3d):
    d = tmp_path / name
    d.mkdir()
    np.savez(d / "kp3d.npz", kp3d=kp3d, conf3d=np.ones(kp3d.shape[:2]))
    return d


def _dummy_cfg(n_kp):
    return SimpleNamespace(model=SimpleNamespace(KP_NAMES=[f"kp{i}" for i in range(n_kp)]))


def test_fit_global_scale_picks_known_minimum_via_injected_score_fn(tmp_path):
    rng = np.random.default_rng(0)
    T, K = 40, 5
    kp3d = rng.normal(0, 1.0, (T, K, 3))
    bout_dir = _write_bout_dir(tmp_path, "bout_00001_fly0", kp3d)
    cfg = _dummy_cfg(K)

    target = 0.0117

    def fake_score_fn(cfg, kp3d, kp_names, scale, *, offsets_path, frames=None):
        # Known convex parabola in log(scale) -- GPU/jaxls never touched.
        return {"scale": scale,
                "marker_px": (math.log(scale) - math.log(target)) ** 2,
                "n_frames": kp3d.shape[0]}

    result = fit_global_scale([bout_dir], cfg, lo=0.008, hi=0.016,
                              n_frames=30, method="golden", score_fn=fake_score_fn)

    assert result["scale"] == pytest.approx(target, rel=0.02)
    assert result["method"] == "golden"
    assert result["n_frames"] == 30
    assert isinstance(result["curve"], dict) and len(result["curve"]) >= 2
    # curve keys are the scales actually evaluated, values the fake residual
    for s, v in result["curve"].items():
        assert v == pytest.approx((math.log(s) - math.log(target)) ** 2)


def test_fit_global_scale_grid_method_evaluates_full_grid(tmp_path):
    rng = np.random.default_rng(1)
    T, K = 20, 4
    kp3d = rng.normal(0, 1.0, (T, K, 3))
    bout_dir = _write_bout_dir(tmp_path, "bout_00001_fly0", kp3d)
    cfg = _dummy_cfg(K)

    target = 0.012
    calls = []

    def fake_score_fn(cfg, kp3d, kp_names, scale, *, offsets_path, frames=None):
        calls.append(scale)
        return {"scale": scale,
                "marker_px": (math.log(scale) - math.log(target)) ** 2,
                "n_frames": kp3d.shape[0]}

    result = fit_global_scale([bout_dir], cfg, lo=0.008, hi=0.016, n_frames=15,
                              method="grid", n_grid=7, score_fn=fake_score_fn)

    assert len(calls) == 7
    assert len(result["curve"]) == 7
    # grid search can only land on a sampled point -- within half a grid step
    # of the true minimum, not exactly on it
    assert result["scale"] == pytest.approx(target, rel=0.1)


def test_fit_global_scale_pools_frames_across_multiple_bout_dirs(tmp_path):
    rng = np.random.default_rng(2)
    K = 4
    kp3d_a = rng.normal(0, 1.0, (25, K, 3))
    kp3d_b = rng.normal(0, 1.0, (25, K, 3))
    d_a = _write_bout_dir(tmp_path, "bout_00001_fly0", kp3d_a)
    d_b = _write_bout_dir(tmp_path, "bout_00002_fly0", kp3d_b)
    cfg = _dummy_cfg(K)

    seen_n_frames = []

    def fake_score_fn(cfg, kp3d, kp_names, scale, *, offsets_path, frames=None):
        seen_n_frames.append(kp3d.shape[0])
        return {"scale": scale, "marker_px": 1.0, "n_frames": kp3d.shape[0]}

    result = fit_global_scale([d_a, d_b], cfg, lo=0.008, hi=0.016,
                              n_frames=30, method="grid", n_grid=3,
                              score_fn=fake_score_fn)

    # pooled across both bouts (50 available), subsampled down to n_frames=30
    assert result["n_frames"] == 30
    assert all(n == 30 for n in seen_n_frames)


def test_fit_global_scale_rejects_unknown_method(tmp_path):
    rng = np.random.default_rng(3)
    kp3d = rng.normal(0, 1.0, (10, 3, 3))
    bout_dir = _write_bout_dir(tmp_path, "bout_00001_fly0", kp3d)
    cfg = _dummy_cfg(3)
    with pytest.raises(ValueError):
        fit_global_scale([bout_dir], cfg, lo=0.008, hi=0.016, method="bogus",
                         score_fn=lambda *a, **k: {"marker_px": 0.0})
