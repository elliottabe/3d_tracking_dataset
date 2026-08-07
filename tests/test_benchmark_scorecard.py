"""Tests for scripts/benchmark/scorecard.py (aggregation, gaps, splits)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.benchmark.scorecard import (
    build_scorecard, cohort_of, scorecard_markdown, split_series, write_scorecard,
)


def test_cohort_of():
    fr = {"assay": "free_running"}
    ct = {"assay": "courtship"}
    assert cohort_of(fr, 0, None) == "free_running"
    assert cohort_of(ct, 1, 1) == "courtship_male"
    assert cohort_of(ct, 0, 1) == "courtship_female"


def test_cohort_of_courtship_requires_sex():
    with pytest.raises(ValueError, match="sex"):
        cohort_of({"assay": "courtship"}, 0, None)


def test_split_series_thresholds():
    s = np.arange(10, dtype=float)          # metric series
    prox = np.array([0.5] * 5 + [5.0] * 5)  # first half close, second apart
    out = split_series(s, prox, threshold=2.0)
    assert out["close"] == pytest.approx(np.median(s[:5]))
    assert out["apart"] == pytest.approx(np.median(s[5:]))
    assert out["all"] == pytest.approx(np.median(s))


def test_split_series_alignment_and_no_proximity():
    s = np.arange(8, dtype=float)           # e.g. jitter is (T-2,)
    prox = np.ones(10) * 5.0
    out = split_series(s, prox, threshold=2.0)
    assert out["apart"] == pytest.approx(np.median(s))
    out2 = split_series(s, None, threshold=2.0)
    assert out2["apart"] is None and out2["close"] is None


def _bout(cohort, reproj, jitter):
    return {"cohort": cohort, "run_key": "r", "bout": 1, "fly": 0, "tags": [],
            "scalars": {"reproj_px_median": reproj, "jitter_median": jitter},
            "splits": {"reproj_px": {"all": reproj, "apart": reproj, "close": None}}}


def test_build_scorecard_gap_ratios():
    per_bout = [_bout("free_running", 4.0, 0.001),
                _bout("free_running", 6.0, 0.001),
                _bout("courtship_male", 10.0, 0.002),
                _bout("courtship_female", 20.0, 0.004)]
    sc = build_scorecard(per_bout)
    assert sc["cohorts"]["free_running"]["reproj_px_median"] == pytest.approx(5.0)
    assert sc["gap_ratios"]["courtship_male"]["reproj_px_median"] == pytest.approx(2.0)
    assert sc["gap_ratios"]["courtship_female"]["jitter_median"] == pytest.approx(4.0)
    assert sc["n_bouts"]["free_running"] == 2


def test_write_and_markdown(tmp_path):
    sc = build_scorecard([_bout("free_running", 4.0, 0.001),
                          _bout("courtship_male", 8.0, 0.002)])
    p = tmp_path / "scorecard.json"
    write_scorecard(p, sc)
    assert json.loads(p.read_text())["cohorts"]
    md = scorecard_markdown(sc)
    assert "free_running" in md and "gap" in md.lower()
    assert (tmp_path / "scorecard.md").exists()
