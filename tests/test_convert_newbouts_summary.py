"""Tests for scripts/data/convert_newbouts_summary.py.

The NewBouts curation tree (processed/free_running/NewBouts/<ts>/) ships
`running_bouts_summary.csv` with schema
    bout,start_frame,end_frame,n_frames,duration_s,min_cycles,status,...
plus the workstation's `data3D.csv` and `info.yaml`. The batch IK route
(batch_process_predictions -> batch_run_stac -> batch_postprocess) discovers
`Predictions_3D_*` dirs, each needing `data3D.csv` and
`free_running_bouts_summary.csv` whose schema
(scripts/preprocess_keypoints_for_ik.py:load_bouts_from_csv) REQUIRES
`bout_idx,start_frame,end_frame,fly_id` -- fly_id is mandatory there, unlike
the SAM3 per-bout route.

The converter materializes `<ts>/Predictions_3D_newbouts/` with a relative
data3D.csv symlink and the converted summary: accepted rows only,
bout -> bout_idx, fly_id derived from info.yaml's recording_path
("<parent>/<ts>", stripping a trailing "videos" component).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.data.convert_newbouts_summary import (
    convert_summary, fly_id_from_info, prepare_recording)

_SRC_HEADER = ("bout,start_frame,end_frame,n_frames,duration_s,min_cycles,"
               "status,reason,total_distance_mm,mean_speed_mm_s,source,"
               "edit_kind,status_source")


def _write_src(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join([_SRC_HEADER] + rows) + "\n")
    return path


def _make_recording(tmp_path: Path, ts="2025_10_07_17_15_30",
                    recording_path="/mnt/lemebel/happyhouse_102025/session1/2025_10_07_17_15_30/",
                    rows=None) -> Path:
    rec = tmp_path / ts
    rec.mkdir()
    (rec / "data3D.csv").write_text("stub\n")
    (rec / "info.yaml").write_text(
        f"recording_path: {recording_path}\nframe_start: 0\n")
    _write_src(rec / "running_bouts_summary.csv", rows if rows is not None else [
        '1,6555,7239,685,1.37,8,accepted,"",13.8,10.1,manual,adjusted,auto',
        '2,9000,9100,101,0.2,1,rejected,"too short",nan,nan,manual,adjusted,auto',
        '3,378076,378372,297,0.594,5,accepted,"",8.4,14.2,manual,adjusted,auto',
    ])
    return rec


def test_accepted_rows_converted_with_fly_id(tmp_path):
    rec = _make_recording(tmp_path)
    out = tmp_path / "out.csv"
    n = convert_summary(rec / "running_bouts_summary.csv", out,
                        fly_id="session1/2025_10_07_17_15_30")
    assert n == 2
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["bout_idx"] for r in rows] == ["1", "3"]
    assert rows[0]["start_frame"] == "6555" and rows[0]["end_frame"] == "7239"
    assert all(r["fly_id"] == "session1/2025_10_07_17_15_30" for r in rows)
    assert "bout" not in rows[0]           # renamed, not duplicated


def test_output_satisfies_the_real_batch_consumer(tmp_path):
    """Round-trip through load_bouts_from_csv (the actual consumer), which
    REQUIRES fly_id and converts bout_idx to 0-based."""
    from scripts.preprocess_keypoints_for_ik import load_bouts_from_csv

    rec = _make_recording(tmp_path)
    out = tmp_path / "free_running_bouts_summary.csv"
    convert_summary(rec / "running_bouts_summary.csv", out, fly_id="session1/x")
    bouts = load_bouts_from_csv(out)
    assert [(b["bout_idx"], b["start_frame"], b["end_frame"]) for b in bouts] \
        == [(0, 6555, 7239), (2, 378076, 378372)]
    assert bouts[0]["fly_id"] == "session1/x"


def test_fly_id_from_info_strips_trailing_videos_component(tmp_path):
    rec = _make_recording(
        tmp_path, ts="2025_10_11_10_29_50",
        recording_path="/mnt/lemebel/happyhouse_102025/session5/2025_10_11_10_29_50/videos/")
    assert fly_id_from_info(rec / "info.yaml") == "session5/2025_10_11_10_29_50"


def test_prepare_recording_materializes_predictions_dir(tmp_path):
    rec = _make_recording(tmp_path)
    pred = prepare_recording(rec)
    assert pred == rec / "Predictions_3D_newbouts"
    assert (pred / "data3D.csv").is_symlink()
    assert (pred / "data3D.csv").read_text() == "stub\n"   # resolves
    with open(pred / "free_running_bouts_summary.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["fly_id"] == "session1/2025_10_07_17_15_30"
    # idempotent: re-running must not fail on the existing symlink
    assert prepare_recording(rec) == pred


def test_prepare_recording_zero_accepted_skips(tmp_path):
    rec = _make_recording(tmp_path, rows=[
        '1,10,20,11,0.02,0,rejected,"no",nan,nan,auto,none,auto'])
    assert prepare_recording(rec) is None
    assert not (rec / "Predictions_3D_newbouts").exists()


def test_missing_status_column_raises(tmp_path):
    src = tmp_path / "running_bouts_summary.csv"
    src.write_text("bout,start_frame,end_frame\n1,10,20\n")
    with pytest.raises(ValueError, match="status"):
        convert_summary(src, tmp_path / "out.csv", fly_id="x/y")
