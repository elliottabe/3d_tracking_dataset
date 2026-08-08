"""TDD tests for scripts/probe_recording.py (Task 18: scale probe).

A brand-new recording has no bout segmentation at all, so a probe needs a
SYNTHETIC bouts CSV of evenly-spaced frame windows to feed the existing SAM3 +
run_bout.py machinery -- see jarvis_jax.predict.sam3_driver.parse_bouts, which
only ever consumes a bouts CSV as a list of {bout_idx, start, end, n} windows
(it has no notion of "bout" beyond that). These tests cover the PURE window
planning + CSV round-trip + keep/drop selection logic; no GPU, no video, no
real recording.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from scripts.probe_recording import (
    plan_windows,
    write_probe_csv,
    select_windows,
)
from scripts.run_bout import should_stop_after_triangulate


# ---------------------------------------------------------------------------
# plan_windows
# ---------------------------------------------------------------------------

def test_plan_windows_yields_exactly_n_windows():
    windows = plan_windows(100_000, n_windows=16, window_len=60)
    assert len(windows) == 16


def test_plan_windows_all_same_length():
    windows = plan_windows(100_000, n_windows=16, window_len=60)
    assert all(length == 60 for _start, length in windows)


def test_plan_windows_non_overlapping_and_sorted():
    windows = plan_windows(100_000, n_windows=16, window_len=60)
    starts = [s for s, _l in windows]
    assert starts == sorted(starts)
    for (s0, l0), (s1, _l1) in zip(windows, windows[1:]):
        assert s0 + l0 <= s1, "windows must not overlap"


def test_plan_windows_deterministic():
    a = plan_windows(497_993, n_windows=16, window_len=60, margin_frac=0.02)
    b = plan_windows(497_993, n_windows=16, window_len=60, margin_frac=0.02)
    assert a == b


def test_plan_windows_evenly_spaced_gaps_are_equal():
    windows = plan_windows(100_000, n_windows=10, window_len=50)
    starts = [s for s, _l in windows]
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert len(set(gaps)) == 1, f"expected evenly-spaced starts, got gaps {gaps}"


def test_plan_windows_respects_margins():
    n = 100_000
    margin_frac = 0.02
    windows = plan_windows(n, n_windows=16, window_len=60, margin_frac=margin_frac)
    lo_bound = margin_frac * n
    hi_bound = n - margin_frac * n
    for start, length in windows:
        assert start >= lo_bound, f"window start {start} < margin {lo_bound}"
        assert start + length <= hi_bound, (
            f"window end {start + length} > {hi_bound}")


def test_plan_windows_raises_when_request_cannot_fit():
    # 100 windows of 60 frames (6000 total) cannot fit in a 1000-frame
    # recording (usable span after 2% margins is only ~960 frames).
    with pytest.raises(ValueError):
        plan_windows(1_000, n_windows=100, window_len=60)


def test_plan_windows_raises_on_nonpositive_inputs():
    with pytest.raises(ValueError):
        plan_windows(0, n_windows=1, window_len=1)
    with pytest.raises(ValueError):
        plan_windows(1000, n_windows=0, window_len=1)
    with pytest.raises(ValueError):
        plan_windows(1000, n_windows=1, window_len=0)


def test_plan_windows_exact_fit_succeeds():
    # margin=0 -> usable=1000, slot=1000//10=100 == window_len -> fits exactly.
    windows = plan_windows(1000, n_windows=10, window_len=100, margin_frac=0.0)
    assert len(windows) == 10
    assert windows[0] == (0, 100)
    assert windows[-1] == (900, 100)


# ---------------------------------------------------------------------------
# write_probe_csv <-> jarvis_jax.predict.sam3_driver.parse_bouts round-trip
# ---------------------------------------------------------------------------

SESSION_TAG = "Session0/2025_10_20_13_20_04"


def _parse_bouts_reimpl(csv_path, session_tag):
    """Minimal reimplementation of parse_bouts' CSV-reading contract (in case
    importing jarvis_jax is unavailable in some test environment) -- used only
    as a fallback so the round-trip assertion is never silently skipped."""
    out = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        has_fly_id = reader.fieldnames is not None and "fly_id" in reader.fieldnames
        for r in reader:
            if has_fly_id and session_tag and r.get("fly_id") != session_tag:
                continue
            s, e = int(r["start_frame"]), int(r["end_frame"])
            out.append({"bout_idx": int(r["bout_idx"]), "start": s, "end": e, "n": e - s + 1})
    out.sort(key=lambda b: b["bout_idx"])
    return out


def _parse_bouts(csv_path, session_tag):
    try:
        from jarvis_jax.predict.sam3_driver import parse_bouts
    except Exception:
        return _parse_bouts_reimpl(csv_path, session_tag)
    return parse_bouts(str(csv_path), session_tag)


def test_write_probe_csv_round_trips_through_parse_bouts(tmp_path):
    windows = plan_windows(100_000, n_windows=8, window_len=60)
    csv_path = write_probe_csv(tmp_path / "probe_bouts.csv", SESSION_TAG, windows)

    parsed = _parse_bouts(csv_path, SESSION_TAG)
    assert len(parsed) == len(windows)
    got = [(b["start"], b["n"]) for b in parsed]
    expected = [(start, length) for start, length in windows]
    assert got == expected


def test_write_probe_csv_bout_idx_starts_at_one_and_is_contiguous(tmp_path):
    windows = plan_windows(100_000, n_windows=5, window_len=60)
    csv_path = write_probe_csv(tmp_path / "probe_bouts.csv", SESSION_TAG, windows)
    parsed = _parse_bouts(csv_path, SESSION_TAG)
    assert [b["bout_idx"] for b in parsed] == [1, 2, 3, 4, 5]


def test_write_probe_csv_end_frame_is_inclusive(tmp_path):
    windows = [(1000, 60)]
    csv_path = write_probe_csv(tmp_path / "probe_bouts.csv", SESSION_TAG, windows)
    with open(csv_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert int(row["start_frame"]) == 1000
    assert int(row["end_frame"]) == 1059  # inclusive: end - start + 1 == 60


def test_write_probe_csv_different_session_tag_is_filtered_out(tmp_path):
    windows = plan_windows(100_000, n_windows=4, window_len=60)
    csv_path = write_probe_csv(tmp_path / "probe_bouts.csv", SESSION_TAG, windows)
    parsed = _parse_bouts(csv_path, "Session0/some_other_recording")
    assert parsed == []


# ---------------------------------------------------------------------------
# select_windows: pure keep/drop quality-gating logic
# ---------------------------------------------------------------------------

def test_select_windows_drops_insufficient_coverage():
    quality = {
        1: {"coverage_status": "ok", "within_bone_cv": 0.05},
        2: {"coverage_status": "insufficient", "within_bone_cv": 0.02},
    }
    result = select_windows(quality, keep=5)
    assert result["kept"] == [1]
    assert result["dropped"][2] == "insufficient_coverage"


def test_select_windows_drops_high_within_bone_cv():
    quality = {
        1: {"coverage_status": "ok", "within_bone_cv": 0.05},
        2: {"coverage_status": "ok", "within_bone_cv": 0.99},  # far above warn thresh
    }
    result = select_windows(quality, keep=5, cv_warn_thresh=0.15)
    assert result["kept"] == [1]
    assert result["dropped"][2] == "high_within_bone_cv"


def test_select_windows_keeps_best_by_cv_and_drops_excess():
    # 5 good windows, keep=3 -> the 3 lowest-cv windows survive.
    quality = {i: {"coverage_status": "ok", "within_bone_cv": cv}
               for i, cv in zip(range(1, 6), [0.10, 0.02, 0.30, 0.05, 0.20])}
    result = select_windows(quality, keep=3, cv_warn_thresh=0.5)
    # cv order (lowest first) is 2(0.02), 4(0.05), 1(0.10), 5(0.20), 3(0.30) --
    # top 3 by cv are {2, 4, 1}; the returned "kept" list is sorted by bout_idx
    # for a deterministic, readable report.
    assert result["kept"] == [1, 2, 4]
    assert result["dropped"][3] == "excess_oversample"
    assert result["dropped"][5] == "excess_oversample"


def test_select_windows_ties_broken_by_bout_idx_ascending():
    # Two windows tie exactly on within_bone_cv; keep=1 must deterministically
    # keep the lower bout_idx.
    quality = {
        7: {"coverage_status": "ok", "within_bone_cv": 0.05},
        3: {"coverage_status": "ok", "within_bone_cv": 0.05},
    }
    result = select_windows(quality, keep=1)
    assert result["kept"] == [3]
    assert result["dropped"][7] == "excess_oversample"


def test_select_windows_keep_greater_than_candidates_keeps_all():
    quality = {1: {"coverage_status": "ok", "within_bone_cv": 0.05},
               2: {"coverage_status": "ok", "within_bone_cv": 0.06}}
    result = select_windows(quality, keep=12)
    assert result["kept"] == [1, 2]
    assert result["dropped"] == {}


def test_select_windows_is_deterministic():
    quality = {i: {"coverage_status": "ok", "within_bone_cv": (i % 3) * 0.01}
               for i in range(1, 17)}
    a = select_windows(quality, keep=12)
    b = select_windows(quality, keep=12)
    assert a == b


# ---------------------------------------------------------------------------
# scripts.run_bout.should_stop_after_triangulate (Task 18 §3 stage-limiting)
# ---------------------------------------------------------------------------

class _FakeCfg(dict):
    """Minimal stand-in for an OmegaConf DictConfig: dict.get(...) semantics,
    with nested dicts also behaving as _FakeCfg so cfg.get("pipeline") chains."""

    def get(self, key, default=None):
        return dict.get(self, key, default)


def test_should_stop_after_triangulate_default_is_false_when_key_absent():
    # No `pipeline` block at all -- current behaviour must be unaffected.
    cfg = _FakeCfg({})
    assert should_stop_after_triangulate(cfg) is False


def test_should_stop_after_triangulate_false_when_stop_after_is_null():
    cfg = _FakeCfg({"pipeline": _FakeCfg({"stop_after": None})})
    assert should_stop_after_triangulate(cfg) is False


def test_should_stop_after_triangulate_true_when_set():
    cfg = _FakeCfg({"pipeline": _FakeCfg({"stop_after": "triangulate"})})
    assert should_stop_after_triangulate(cfg) is True


def test_should_stop_after_triangulate_false_for_other_values():
    cfg = _FakeCfg({"pipeline": _FakeCfg({"stop_after": "some_other_stage"})})
    assert should_stop_after_triangulate(cfg) is False
