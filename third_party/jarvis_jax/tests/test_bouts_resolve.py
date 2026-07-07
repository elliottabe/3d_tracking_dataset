import csv
import os
import pytest
from jarvis_jax.predict.bouts_resolve import resolve_bout_summary
from jarvis_jax.predict.sam3_driver import parse_bouts


def _write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_canonical_in_processed_used_as_is(tmp_path):
    rec = tmp_path / "rec"; proc = tmp_path / "proc"
    canonical = proc / "courtship_bout_summary.csv"
    _write(str(canonical), ["bout_idx", "start_frame", "end_frame"], [[0, 1, 2]])
    got = resolve_bout_summary(recording_dir=str(rec), processed_dir=str(proc), dataset="courtship")
    assert got == str(canonical)


def test_session0_unified_source_normalized(tmp_path):
    rec = tmp_path / "rec"; proc = tmp_path / "proc"
    _write(str(rec / "courtship_bouts_unified_summary.csv"),
           ["fly_id", "bout_idx", "start_frame", "end_frame", "source_fly"],
           [["S/rec", 1, 14045, 14557, "both"]])
    got = resolve_bout_summary(recording_dir=str(rec), processed_dir=str(proc), dataset="courtship")
    assert got == str(proc / "courtship_bout_summary.csv")
    assert os.path.isfile(got)
    rows = parse_bouts(got, "S/rec")          # round-trips through parse_bouts
    assert rows[0]["bout_idx"] == 1 and rows[0]["start"] == 14045


def test_session1_good_bouts_source(tmp_path):
    rec = tmp_path / "rec"; proc = tmp_path / "proc"
    _write(str(rec / "quality_viz" / "Predictions_3D_111_good_bouts.csv"),
           ["bout_idx", "start_frame", "end_frame", "n_frames", "mean_score"],
           [[0, 100, 600, 500, 1.37], [1, 1000, 1050, 50, 0.2]])
    got = resolve_bout_summary(recording_dir=str(rec), processed_dir=str(proc), dataset="courtship")
    assert os.path.isfile(got)
    rows = parse_bouts(got, "ignored")        # no fly_id -> all rows
    assert [r["bout_idx"] for r in rows] == [0, 1]


def test_free_running_falls_back_to_(tmp_path):
    rec = tmp_path / "rec"; proc = tmp_path / "proc"
    _write(str(rec / "_bout_summary.csv"),
           ["bout_idx", "start_frame", "end_frame"], [[0, 5, 9]])
    got = resolve_bout_summary(recording_dir=str(rec), processed_dir=str(proc), dataset="free_running")
    assert got == str(proc / "free_running_bout_summary.csv")
    assert os.path.isfile(got)


def test_no_source_raises(tmp_path):
    rec = tmp_path / "rec"; proc = tmp_path / "proc"
    os.makedirs(str(rec))
    with pytest.raises(FileNotFoundError):
        resolve_bout_summary(recording_dir=str(rec), processed_dir=str(proc), dataset="courtship")


def test_source_missing_required_column_raises(tmp_path):
    # A source file exists but lacks a required column (no start_frame) -> ValueError
    rec = tmp_path / "rec"; proc = tmp_path / "proc"
    _write(str(rec / "courtship_bout_summary.csv"),
           ["bout_idx", "end_frame", "mean_score"], [[0, 200, 0.5]])
    with pytest.raises(ValueError):
        resolve_bout_summary(recording_dir=str(rec), processed_dir=str(proc), dataset="courtship")
