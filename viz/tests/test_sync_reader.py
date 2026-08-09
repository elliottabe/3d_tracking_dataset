"""Tests for viz.core.io sync-aware frame reading.

Cameras drop frames independently, so the Nth decoded frame of one mp4 is not
necessarily the same instant as the Nth of another. `sync_plan.json` records
which canonical slot each decoded frame belongs to. Mask GENERATION already
applies this (positions_per_cam -> process_bout); the viz path did not, so an
overlay could draw a mask on the wrong frame for a camera that had dropped one
earlier in the recording. 25 courtship bouts start after a recorded drop slot.
"""
from __future__ import annotations

import json
import os

import pytest

from viz.core.io import sync_positions


def write_plan(tmp_path, cams, canonical_len=1000):
    """A sync_plan.json in the real schema; `cams` maps name -> gap slots."""
    plan = {
        "version": 1,
        "recording": "test_rec",
        "delta_ns": 1_000_000,
        "anchor_ts_ns": 0,
        "canonical_len": canonical_len,
        "predict_start": 0,
        "predict_len": canonical_len,
        "status": "reindex",
        "first_drop_slot": None,
        "cameras": {},
    }
    for name, gaps in cams.items():
        # Real schema: gaps are {"slot", "lost"} dicts, NOT bare ints. Writing
        # ints instead made SyncPlan raise, which sync_positions swallowed into
        # the positional fallback -- the test then "passed" against a plan that
        # was never parsed.
        plan["cameras"][name] = {
            "start_slot": 0,
            "decoded_len": canonical_len - len(gaps),
            "true_span": canonical_len,
            "frame_id_mode": "hole" if gaps else "reindex",
            "gaps": [{"slot": int(g), "lost": 1} for g in gaps],
        }
    (tmp_path / "sync_plan.json").write_text(json.dumps(plan))
    return tmp_path


def test_no_plan_is_positional_identity(tmp_path):
    # Session0 has no Cam*_meta.csv at all, so it always takes this path --
    # behaviour must be byte-identical to the old positional reader.
    out = sync_positions(tmp_path, ["CamA", "CamB"], start_slot=100, count=5)
    assert len(out) == 2
    for pos, pres in out:
        assert pos == [100, 101, 102, 103, 104]
        assert all(pres)


def test_camera_absent_from_plan_falls_back_to_positional(tmp_path):
    write_plan(tmp_path, {"CamA": []})
    (pos_a, _), (pos_b, pres_b) = sync_positions(
        tmp_path, ["CamA", "CamB"], start_slot=10, count=3)
    assert pos_b == [10, 11, 12] and all(pres_b)


def test_clean_camera_maps_slots_one_to_one(tmp_path):
    write_plan(tmp_path, {"CamA": []})
    (pos, pres), = sync_positions(tmp_path, ["CamA"], start_slot=50, count=4)
    assert pos == [50, 51, 52, 53]
    assert all(pres)


def test_positions_shift_after_a_dropped_slot(tmp_path):
    # The whole point: after CamA drops slot 20, canonical slot 21 lives at mp4
    # frame 20 for CamA but frame 21 everywhere else. A positional read would
    # pair slot 21's mask with CamA's slot-22 image.
    write_plan(tmp_path, {"CamA": [20], "CamB": []})
    (pos_a, pres_a), (pos_b, pres_b) = sync_positions(
        tmp_path, ["CamA", "CamB"], start_slot=18, count=6)   # slots 18..23
    assert pos_b == [18, 19, 20, 21, 22, 23]
    assert pos_a != pos_b, "a dropped slot must shift this camera's mp4 indices"
    assert pos_a[0] == 18 and pos_a[1] == 19          # before the drop: aligned
    # the dropped slot itself is either absent or maps elsewhere
    assert (not pres_a[2]) or (pos_a[2] != 20)
    # after the drop, CamA runs one mp4 frame behind the canonical slot
    assert pos_a[-1] == pos_b[-1] - 1


def test_count_and_shape_are_respected(tmp_path):
    write_plan(tmp_path, {"CamA": [5], "CamB": []})
    out = sync_positions(tmp_path, ["CamA", "CamB"], start_slot=0, count=9)
    for pos, pres in out:
        assert len(pos) == 9 and len(pres) == 9


def test_zero_count_is_empty(tmp_path):
    out = sync_positions(tmp_path, ["CamA"], start_slot=0, count=0)
    assert out == [([], [])]


def test_malformed_plan_does_not_raise(tmp_path):
    # viz must never hard-fail on sync metadata; a broken plan degrades to
    # positional rather than killing the render.
    (tmp_path / "sync_plan.json").write_text("{not json")
    (pos, pres), = sync_positions(tmp_path, ["CamA"], start_slot=7, count=3)
    assert pos == [7, 8, 9] and all(pres)


REAL = ('/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/'
        'Session1/2026_04_02_15_25_51')


@pytest.mark.skipif(not os.path.isfile(os.path.join(REAL, "sync_plan.json")),
                    reason="real recording not present")
def test_real_recording_with_a_hole_shifts_after_the_drop():
    """Cam2012853 in this recording has one gap; first_drop_slot=461848."""
    plan = json.loads(open(os.path.join(REAL, "sync_plan.json")).read())
    cams = list(plan["cameras"])
    holed = [c for c, v in plan["cameras"].items() if v["frame_id_mode"] == "hole"]
    assert holed, "expected at least one camera with a hole"
    start = int(plan["first_drop_slot"]) + 10
    out = sync_positions(REAL, cams, start_slot=start, count=3)
    by_cam = dict(zip(cams, out))
    clean = [c for c in cams if c not in holed][0]
    assert by_cam[holed[0]][0] != by_cam[clean][0], (
        "after a drop the holed camera must map to different mp4 frames")


def test_a_wrong_schema_plan_is_reported_not_silently_ignored(tmp_path, capsys):
    # Guards the failure this test file itself hit: a plan whose `gaps` are
    # bare ints raises inside SyncPlan, and a silent fallback makes it look
    # like a clean recording.
    (tmp_path / "sync_plan.json").write_text(json.dumps({
        "delta_ns": 1, "canonical_len": 10, "predict_start": 0,
        "predict_len": 10, "cameras": {"CamA": {
            "start_slot": 0, "decoded_len": 9, "gaps": [3]}}}))
    (pos, pres), = sync_positions(tmp_path, ["CamA"], start_slot=0, count=3)
    assert pos == [0, 1, 2]                      # degraded to positional
    assert "WARNING" in capsys.readouterr().out  # ...but said so
