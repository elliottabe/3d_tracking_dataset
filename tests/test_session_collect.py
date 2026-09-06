"""Tests for scripts/session_collect.py -- the session-level roll-up that the
mvq session chain's `collect` job runs (aggregate[*] -> collect).

Builds a fake processed tree (two recordings, one with a bout-fly that has no
stac_ik.h5 and one declared unsolvable) and asserts what lands in the summary
JSON and Markdown. Nothing here touches SLURM or the real data root.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "session_collect", REPO / "scripts" / "session_collect.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sc = _load()


# ---------------------------------------------------------------------------
# fake tree
# ---------------------------------------------------------------------------

def _session_qc(rows):
    finite = [r for r in rows if r["loo_px"] == r["loo_px"]]
    return {
        "n_bouts_flies": len(rows),
        "iou_hard_median": 0.02,
        "iou_soft_median": 0.14,
        "reproj_px_median": 2.1,
        "loo_px_median": 0.77,
        "total_frames": sum(r["n_frames"] for r in rows),
        "rows": rows,
    }


def _mvq_meta(n_frames, *, missing_f0, missing_f1, dropped_f0=0, finite_f0=1000,
              disagree_f0=0.0):
    """Only the fields the collector reads; fly0 = FEMALE, fly1 = MALE, and the
    labels are carried in the file (not assumed by the reader)."""
    return {
        "identity": "mask",
        "identity_resolved": "mask",
        "n_frames": n_frames,
        "n_missing": {"fly0": missing_f0, "fly1": missing_f1},
        "n_collapsed": {"fly0": 0, "fly1": 0},
        "sex_head_disagree_frac": {"fly0": disagree_f0, "fly1": 0.0},
        "fly_slots": {"fly0": 1, "fly1": 2},
        "fly_sex": {"fly0": "female", "fly1": "male"},
        "containment_report": {
            "enabled": True,
            "n_kp_dropped_other_mask": {"fly0": dropped_f0, "fly1": 0},
            "n_kp_dropped_spike": {"fly0": 0, "fly1": 0},
            "n_kp_finite_before": {"fly0": finite_f0, "fly1": finite_f0},
        },
    }


RUN = "pose_mvq_test"
TS_A = "2026_04_02_12_11_50"
TS_B = "2026_04_02_14_54_28"


@pytest.fixture
def processed(tmp_path):
    root = tmp_path / "processed" / "courtship"

    # --- recording A: 2 bouts, both flies fully solved -------------------
    run_a = root / "Session1" / TS_A / RUN
    (run_a / "qc").mkdir(parents=True)
    rows_a = [dict(bout=b, fly=f, iou_hard=0.02 + 0.01 * f, iou_soft=0.14,
                   reproj_px=2.0 + b, loo_px=0.5 + b, n_frames=100)
              for b in (1, 2) for f in (0, 1)]
    (run_a / "qc" / "session_qc.json").write_text(json.dumps(_session_qc(rows_a)))
    for b in (1, 2):
        bd = run_a / "bouts" / f"bout_{b:05d}"
        bd.mkdir(parents=True)
        (bd / "mvq_meta.json").write_text(json.dumps(
            _mvq_meta(100, missing_f0=10, missing_f1=0, dropped_f0=20, finite_f0=1000)))
        (bd / "coverage.json").write_text(json.dumps(
            {"n_flies": 2, "n_cams": 7, "n_frames": 100,
             "per_fly": {"0": {"status": "ok"}, "1": {"status": "ok"}}}))
        (bd / "track_qc.json").write_text(json.dumps({"status": "ok", "n_merged": 0}))
        for fly in ("fly0", "fly1"):
            fd = bd / fly
            fd.mkdir()
            (fd / "stac_ik.h5").write_bytes(b"")

    # --- recording B: 3 bouts; b1/fly0 missing stac_ik.h5, b3/fly0
    #     unsolvable, b2 has no mvq_meta.json at all --------------------
    run_b = root / "Session1" / TS_B / RUN
    (run_b / "qc").mkdir(parents=True)
    rows_b = [dict(bout=b, fly=f, iou_hard=0.03, iou_soft=0.15,
                   reproj_px=3.0, loo_px=1.0 + b, n_frames=200)
              for b in (1, 2, 3) for f in (0, 1)]
    (run_b / "qc" / "session_qc.json").write_text(json.dumps(_session_qc(rows_b)))
    for b in (1, 2, 3):
        bd = run_b / "bouts" / f"bout_{b:05d}"
        bd.mkdir(parents=True)
        if b != 2:
            (bd / "mvq_meta.json").write_text(json.dumps(
                _mvq_meta(200, missing_f0=40, missing_f1=2, dropped_f0=0,
                          finite_f0=2000, disagree_f0=0.05)))
        for fly in ("fly0", "fly1"):
            fd = bd / fly
            fd.mkdir()
            if b == 1 and fly == "fly0":
                continue                                   # no stac_ik.h5
            if b == 3 and fly == "fly0":
                (fd / "unsolvable.json").write_text(json.dumps(
                    {"status": "unsolvable", "reason": "no_finite_segment"}))
                continue
            (fd / "stac_ik.h5").write_bytes(b"")

    # a recording with no run root at all must simply not appear
    (root / "Session1" / "2026_04_02_15_12_14" / "sam3_masks").mkdir(parents=True)
    return root


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_percentile_matches_linear_interpolation():
    assert sc.percentile([1, 2, 3, 4], 50.0) == pytest.approx(2.5)
    assert sc.percentile([1, 2, 3, 4], 90.0) == pytest.approx(3.7)
    assert sc.percentile([], 50.0) is None
    assert sc.percentile([float("nan"), 5.0], 50.0) == pytest.approx(5.0)


def test_read_json_reports_instead_of_raising(tmp_path):
    warns = []
    assert sc.read_json(tmp_path / "nope.json", warns, required=True) is None
    assert any("missing" in w for w in warns)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    warns2 = []
    assert sc.read_json(bad, warns2) is None
    assert any("unreadable" in w for w in warns2)


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------

def test_collects_both_recordings(processed):
    s = sc.collect_session(processed, "Session1", RUN)
    assert s["n_recordings"] == 2
    assert [r["timestamp"] for r in s["recordings"]] == [TS_A, TS_B]


def test_per_recording_qc_numbers(processed):
    s = sc.collect_session(processed, "Session1", RUN)
    a = next(r for r in s["recordings"] if r["timestamp"] == TS_A)
    assert a["qc"]["n_bout_flies"] == 4
    # loo values 1.5,1.5,2.5,2.5 -> median 2.0, p90 2.5
    assert a["qc"]["loo_px_median"] == pytest.approx(2.0)
    assert a["qc"]["loo_px_p90"] == pytest.approx(2.5)
    assert a["bouts"]["n_bouts"] == 2
    assert a["bouts"]["n_frames_lift"] == 200          # 2 bouts x 100 frames, per fly


def test_missing_percentages_are_named_by_sex_not_by_index(processed):
    s = sc.collect_session(processed, "Session1", RUN)
    a = next(r for r in s["recordings"] if r["timestamp"] == TS_A)
    assert a["derived"]["female_fly"] == "fly0" and a["derived"]["male_fly"] == "fly1"
    # 2 bouts x 10 missing female frames of 200 lifted frames
    assert a["derived"]["female_missing_pct"] == pytest.approx(10.0)
    assert a["derived"]["male_missing_pct"] == pytest.approx(0.0)
    # 2 x 20 dropped keypoints of 2 x (1000 + 1000) tested
    assert a["derived"]["containment_dropped_pct"] == pytest.approx(1.0)


def test_missing_stac_ik_and_unsolvable_are_listed(processed):
    s = sc.collect_session(processed, "Session1", RUN)
    assert s["missing_stac_ik"] == [f"{TS_B}/bout_00001/fly0"]
    assert s["bouts_with_unsolvable"] == [f"{TS_B}/bout_00003"]
    b = next(r for r in s["recordings"] if r["timestamp"] == TS_B)
    assert b["derived"]["n_bouts_with_unsolvable"] == 1


def test_a_bout_without_mvq_meta_is_a_warning_not_a_crash(processed):
    s = sc.collect_session(processed, "Session1", RUN)
    b = next(r for r in s["recordings"] if r["timestamp"] == TS_B)
    assert b["bouts"]["n_mvq_meta"] == 2 and b["bouts"]["n_bouts"] == 3
    assert any("no mvq_meta.json for bout_00002" in w for w in b["warnings"])
    # frames come from the two bouts that DO have one
    assert b["bouts"]["n_frames_lift"] == 400


def test_totals_pool_numerators_rather_than_averaging_percentages(processed):
    s = sc.collect_session(processed, "Session1", RUN)
    t = s["total"]
    assert t["n_bouts"] == 5 and t["n_bout_flies"] == 10
    assert t["n_frames_lift"] == 600                    # 200 (A) + 400 (B)
    # female missing 2x10 (A) + 2x40 (B) = 100 of 600 frames
    assert t["female_missing_pct"] == pytest.approx(100 / 600 * 100)
    assert t["n_missing_stac_ik"] == 1
    assert t["n_bouts_with_unsolvable"] == 1


def test_empty_session_is_reported_not_raised(tmp_path):
    s = sc.collect_session(tmp_path, "SessionX", RUN)
    assert s["n_recordings"] == 0
    assert any("missing session dir" in w for w in s["warnings"])
    assert sc.render_markdown(s)          # still renders


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def test_main_writes_json_and_markdown(processed, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "session_collect.py", "--session-name", "Session1",
        "--processed", str(processed), "--run-name", RUN])
    sc.main()

    out_json = processed / "Session1" / f"{RUN}_session_summary.json"
    out_md = processed / "Session1" / f"{RUN}_session_summary.md"
    assert out_json.exists() and out_md.exists()

    s = json.loads(out_json.read_text())
    assert s["session"] == "Session1" and s["run_name"] == RUN
    assert s["n_recordings"] == 2

    md = out_md.read_text()
    assert "| recording | bouts | bout-flies solved | frames |" in md
    assert "female missing %" in md and "male missing %" in md
    assert "containment-dropped %" in md and "bouts w/ unsolvable" in md
    assert TS_A in md and TS_B in md
    assert "**TOTAL**" in md
    assert f"`{TS_B}/bout_00001/fly0`" in md          # missing stac_ik.h5 section
    assert f"`{TS_B}/bout_00003`" in md               # unsolvable section
    assert "fly0 = FEMALE, fly1 = MALE" in md
    assert "[collect] Session1/" in capsys.readouterr().out


def test_no_temp_files_left_behind(processed, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "session_collect.py", "--session-name", "Session1",
        "--processed", str(processed), "--run-name", RUN])
    sc.main()
    assert not list((processed / "Session1").glob("*.tmp"))
