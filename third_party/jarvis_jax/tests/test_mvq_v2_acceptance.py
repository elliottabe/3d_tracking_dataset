# tests/test_mvq_v2_acceptance.py
"""`scripts/benchmark/mvq_v2_acceptance.py` -- the spec §5 scorecard's
PASS/FAIL logic, markdown table, and group-merge behaviour, exercised with a
FAKE set of measured numbers (no jax, no GPU, no real checkpoint -- these
are pure-python/numpy functions by design, see the module's own "row /
scorecard plumbing" section). Also covers the calibration check group
(`check_calibration`), which needs only a JSON file on disk, and `--dry-run`
end to end via the CLI.

Each test states the expectation before asserting it (CLAUDE.md).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from scripts.benchmark import mvq_v2_acceptance as acc  # noqa: E402


# --------------------------------------------------------------------------
# make_row / build_scorecard PASS/FAIL logic
# --------------------------------------------------------------------------
def test_make_row_le_pass_and_fail():
    """EXPECTATION: an 'le' row passes when value <= threshold, fails when
    value > threshold, and reports the right status string either way."""
    ok = acc.make_row("g", "c", "m", 0.08, 0.085, "le", "ev")
    bad = acc.make_row("g", "c", "m", 0.09, 0.085, "le", "ev")
    assert ok["status"] == "PASS" and ok["pass"] is True
    assert bad["status"] == "FAIL" and bad["pass"] is False


def test_make_row_ge_eq_lt_ops():
    assert acc.make_row("g", "c", "m", 0.99, 0.99, "ge", "ev")["status"] == "PASS"
    assert acc.make_row("g", "c", "m", 0.98, 0.99, "ge", "ev")["status"] == "FAIL"
    assert acc.make_row("g", "c", "m", 0.0, 0.0, "eq", "ev")["status"] == "PASS"
    assert acc.make_row("g", "c", "m", 0.01, 0.0, "eq", "ev")["status"] == "FAIL"
    assert acc.make_row("g", "c", "m", 0.019, 0.02, "lt", "ev")["status"] == "PASS"
    assert acc.make_row("g", "c", "m", 0.02, 0.02, "lt", "ev")["status"] == "FAIL"


def test_make_row_none_or_nan_value_is_skip_not_fail():
    """EXPECTATION: a missing measurement (None, or NaN -- e.g. an empty
    cohort) is SKIPPED, never FAILED -- a step not yet run must not read as
    a regression, and must not silently read as a PASS either."""
    none_row = acc.make_row("g", "c", "m", None, 0.05, "le", "ev")
    nan_row = acc.make_row("g", "c", "m", float("nan"), 0.05, "le", "ev")
    assert none_row["status"] == "SKIP" and none_row["pass"] is None
    assert nan_row["status"] == "SKIP" and nan_row["pass"] is None


def test_skip_row_helper_carries_a_note():
    r = acc.skip_row("g", "c", "m", 0.05, "le", "ev", "GPU step skipped")
    assert r["status"] == "SKIP" and r["note"] == "GPU step skipped"


def test_build_scorecard_accepted_requires_no_fail_but_tolerates_skip():
    """EXPECTATION: `accepted` is True iff no row FAILED -- rows that PASS or
    are merely SKIPPED (not yet run) do not block acceptance; a single FAIL
    anywhere blocks it."""
    all_pass_and_skip = [
        acc.make_row("a", "c1", "m", 0.05, 0.085, "le", "ev"),
        acc.skip_row("b", "c2", "m", 0.05, "le", "ev", "not run yet"),
    ]
    sc = acc.build_scorecard(all_pass_and_skip)
    assert sc["accepted"] is True

    with_a_fail = all_pass_and_skip + [acc.make_row("c", "c3", "m", 1.0, 0.5, "le", "ev")]
    sc2 = acc.build_scorecard(with_a_fail)
    assert sc2["accepted"] is False


# --------------------------------------------------------------------------
# markdown table
# --------------------------------------------------------------------------
def test_scorecard_markdown_contains_every_row_and_the_accepted_line():
    rows = [
        acc.make_row("val", "mpjpe", "mpjpe3d_mm", 0.0812, 0.085, "le", "train_mvq.evaluate(...)"),
        acc.make_row("val", "policy_miss", "policy_miss_frac", 0.02, 0.0, "eq", "train_mvq.evaluate(...)"),
        acc.skip_row("calibration", "existence_reliability", "reliability_exist.max_gap",
                    0.05, "le", "mvq_run.json#calibration", "no calibration block yet"),
    ]
    sc = acc.build_scorecard(rows, run="/some/run/final", generated_at="2026-09-05T00:00:00")
    md = acc.scorecard_markdown(sc)
    assert "ACCEPTED: False" in md            # policy_miss (0.02 vs threshold 0.0) FAILs
    assert "mpjpe3d_mm" in md and "0.0812" in md
    assert "PASS" in md and "FAIL" in md and "SKIP" in md
    assert "no calibration block yet" in md   # note surfaced in its own section
    assert "/some/run/final" in md


def test_scorecard_markdown_handles_none_value_as_na():
    rows = [acc.skip_row("g", "c", "m", 0.05, "le", "ev", "not run")]
    md = acc.scorecard_markdown(acc.build_scorecard(rows))
    assert "n/a" in md


# --------------------------------------------------------------------------
# merge_rows: running one group later must not disturb other groups' rows
# --------------------------------------------------------------------------
def test_merge_rows_replaces_only_the_named_groups():
    existing = [
        acc.make_row("val", "mpjpe", "m", 0.08, 0.085, "le", "ev1"),
        acc.make_row("shift", "error_at_1mm", "m", 0.09, 0.10, "le", "ev2"),
    ]
    new_shift_rows = [acc.make_row("shift", "error_at_1mm", "m", 0.11, 0.10, "le", "ev3")]
    merged = acc.merge_rows(existing, new_shift_rows, groups_run={"shift"})
    by_group = {(r["group"], r["check"]): r for r in merged}
    assert len(merged) == 2
    assert by_group[("val", "mpjpe")]["evidence"] == "ev1"          # untouched
    assert by_group[("shift", "error_at_1mm")]["evidence"] == "ev3"  # replaced
    assert by_group[("shift", "error_at_1mm")]["value"] == 0.11


def test_merge_rows_group_with_zero_new_rows_drops_its_stale_rows():
    """EXPECTATION: re-running a group that this time produced no rows at
    all (e.g. every bout failed to load) still REMOVES that group's old
    rows rather than leaving a stale PASS/FAIL from a different checkpoint
    sitting in the merged scorecard."""
    existing = [acc.make_row("masked_bouts", "bout_1_pose_jump", "m", 0.0, 0.0, "eq", "ev1")]
    merged = acc.merge_rows(existing, [], groups_run={"masked_bouts"})
    assert merged == []


# --------------------------------------------------------------------------
# _jsonable / write_scorecard / load_existing_rows round-trip
# --------------------------------------------------------------------------
def test_write_and_load_scorecard_round_trips_and_nans_become_null(tmp_path):
    rows = [
        acc.make_row("val", "mpjpe", "m", float("nan"), 0.085, "le", "ev"),
        acc.make_row("shift", "miss", "m", 0.01, 0.02, "lt", "ev"),
    ]
    sc = acc.build_scorecard(rows, run="r", generated_at="t")
    jpath, mpath = acc.write_scorecard(str(tmp_path), sc)
    assert os.path.isfile(jpath) and os.path.isfile(mpath)
    with open(jpath) as f:
        raw = json.load(f)
    assert raw["rows"][0]["value"] is None    # NaN -> JSON null, not NaN (invalid JSON)
    loaded = acc.load_existing_rows(str(tmp_path))
    assert len(loaded) == 2
    assert loaded[1]["value"] == 0.01


def test_load_existing_rows_missing_file_returns_empty_list(tmp_path):
    assert acc.load_existing_rows(str(tmp_path / "nope")) == []


# --------------------------------------------------------------------------
# check_calibration: file-missing / block-missing -> SKIP, never FAIL
# --------------------------------------------------------------------------
def test_check_calibration_missing_file_is_skip(tmp_path):
    rows = acc.check_calibration(str(tmp_path / "no_such_run" / "final"))
    assert len(rows) == 1 and rows[0]["status"] == "SKIP"


def test_check_calibration_missing_block_is_skip(tmp_path):
    final = tmp_path / "run" / "final"
    final.mkdir(parents=True)
    (final / "mvq_run.json").write_text(json.dumps({"model": {}, "train": {}}))
    rows = acc.check_calibration(str(final))
    assert len(rows) == 1 and rows[0]["status"] == "SKIP"
    assert "calibration" in rows[0]["note"]


def test_check_calibration_reads_max_gap_and_grades_it(tmp_path):
    final = tmp_path / "run" / "final"
    final.mkdir(parents=True)
    (final / "mvq_run.json").write_text(json.dumps(
        {"calibration": {"reliability_exist": {"max_gap": 0.03}}}))
    rows = acc.check_calibration(str(final))
    assert len(rows) == 1
    assert rows[0]["status"] == "PASS"
    assert rows[0]["value"] == 0.03

    (final / "mvq_run.json").write_text(json.dumps(
        {"calibration": {"reliability_exist": {"max_gap": 0.09}}}))
    rows2 = acc.check_calibration(str(final))
    assert rows2[0]["status"] == "FAIL"


def test_check_calibration_accepts_a_run_dir_not_just_final(tmp_path):
    """`_final_dir` normalises a bare run dir (no trailing final/) to
    `<run>/final` when that subdir exists."""
    final = tmp_path / "run" / "final"
    final.mkdir(parents=True)
    (final / "mvq_run.json").write_text(json.dumps(
        {"calibration": {"reliability_exist": {"max_gap": 0.01}}}))
    rows = acc.check_calibration(str(tmp_path / "run"))
    assert rows[0]["status"] == "PASS"


# --------------------------------------------------------------------------
# GPU-gated check groups default to SKIP (never touch jax/GPU) unless --gpu
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fn,kwargs", [
    (acc.check_val, {}),
    (acc.check_shift, {}),
])
def test_gpu_gated_checks_skip_without_gpu_flag(fn, kwargs):
    rows = fn("/no/such/run", **kwargs)
    assert rows and all(r["status"] == "SKIP" for r in rows)


def test_check_masked_bouts_skips_without_gpu():
    rows = acc.check_masked_bouts("/no/such/run", gpu=False)
    assert len(rows) == 3    # one per bout (1, 4, 28)
    assert all(r["status"] == "SKIP" for r in rows)


def test_check_maskfree_bout117_skips_without_gpu():
    rows = acc.check_maskfree_bout117("/no/such/run", gpu=False)
    assert len(rows) == 2
    assert all(r["status"] == "SKIP" for r in rows)


def test_check_coarse_20_04_skips_without_gpu():
    rows = acc.check_coarse_20_04("/no/such/run", gpu=False)
    assert len(rows) == 3
    assert all(r["status"] == "SKIP" for r in rows)


def test_check_single_fly_skips_without_gpu():
    rows = acc.check_single_fly("/no/such/run", gpu=False)
    assert len(rows) == 3
    assert all(r["status"] == "SKIP" for r in rows)


# --------------------------------------------------------------------------
# --dry-run end to end via the CLI (subprocess, so it exercises argparse too)
# --------------------------------------------------------------------------
def test_cli_dry_run_reports_config_problems_and_commands(tmp_path):
    script = os.path.join(REPO, "scripts", "benchmark", "mvq_v2_acceptance.py")
    proc = subprocess.run(
        [sys.executable, script, "--run", str(tmp_path / "no_such_run"), "--calibration", "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "config problems found" in proc.stdout
    assert "no such directory" in proc.stdout
    assert "calibration/existence_reliability" in proc.stdout


def test_cli_requires_at_least_one_group():
    script = os.path.join(REPO, "scripts", "benchmark", "mvq_v2_acceptance.py")
    proc = subprocess.run([sys.executable, script, "--run", "/no/such/run"],
                         cwd=REPO, capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "check-group flag" in proc.stderr


def test_cli_calibration_group_writes_and_merges_scorecard(tmp_path):
    """End-to-end (no GPU): run --calibration twice against different
    calibration numbers into the SAME --out dir and confirm the second
    run's row replaces the first's rather than accumulating duplicates."""
    final = tmp_path / "run" / "final"
    final.mkdir(parents=True)
    out = tmp_path / "scorecard_out"
    script = os.path.join(REPO, "scripts", "benchmark", "mvq_v2_acceptance.py")

    (final / "mvq_run.json").write_text(json.dumps(
        {"calibration": {"reliability_exist": {"max_gap": 0.09}}}))
    proc1 = subprocess.run(
        [sys.executable, script, "--run", str(final), "--calibration", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=60)
    assert proc1.returncode == 0, proc1.stderr
    assert "accepted=False" in proc1.stdout

    (final / "mvq_run.json").write_text(json.dumps(
        {"calibration": {"reliability_exist": {"max_gap": 0.01}}}))
    proc2 = subprocess.run(
        [sys.executable, script, "--run", str(final), "--calibration", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=60)
    assert proc2.returncode == 0, proc2.stderr
    assert "accepted=True" in proc2.stdout

    with open(out / "scorecard.json") as f:
        sc = json.load(f)
    assert len(sc["rows"]) == 1
    assert sc["rows"][0]["value"] == 0.01
