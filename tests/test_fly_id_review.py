"""Tests for scripts/viz/fly_id_review.py (fly identity review server)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.viz.fly_id_review import (
    DEFAULT_MALE_FLY,
    MANIFEST_NAME,
    VIDEO_NAME,
    bout_dir_from_key,
    build_manifest,
    load_manifest,
    read_sex_json,
    save_manifest,
    scan_bouts,
)


def make_bout(root: Path, session: str, rec: str, bout: str,
              flies=("fly0", "fly1"), video=True, sex_json=None) -> Path:
    """Create a synthetic bout dir matching the real tree layout."""
    bout_dir = root / session / rec / "pose" / "bouts" / bout
    for fly in flies:
        d = bout_dir / fly
        d.mkdir(parents=True)
        if video:
            (d / VIDEO_NAME).write_bytes(b"\x00" * 32)
    if sex_json is not None:
        (bout_dir / "sex.json").write_text(json.dumps(sex_json))
    return bout_dir


# ---------------------------------------------------------------------------
# scan_bouts
# ---------------------------------------------------------------------------

def test_scan_finds_bouts_sorted(tmp_path):
    make_bout(tmp_path, "Session1", "recB", "bout_00002")
    make_bout(tmp_path, "Session0", "recA", "bout_00001")
    keys = list(scan_bouts(tmp_path))
    assert keys == ["Session0/recA/bout_00001", "Session1/recB/bout_00002"]


def test_scan_flags_missing_video_and_fly(tmp_path):
    make_bout(tmp_path, "Session1", "recA", "bout_00001", flies=("fly0",))
    make_bout(tmp_path, "Session1", "recA", "bout_00002", video=False)
    make_bout(tmp_path, "Session1", "recA", "bout_00003")
    bouts = scan_bouts(tmp_path)
    assert bouts["Session1/recA/bout_00001"]["warning"]  # fly1 missing entirely
    assert bouts["Session1/recA/bout_00002"]["warning"]  # videos missing
    assert bouts["Session1/recA/bout_00003"]["warning"] is None


def test_bout_dir_from_key_roundtrip(tmp_path):
    bout_dir = make_bout(tmp_path, "Session1", "recA", "bout_00007")
    assert bout_dir_from_key(tmp_path, "Session1/recA/bout_00007") == bout_dir


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------

def test_build_manifest_defaults(tmp_path):
    make_bout(tmp_path, "Session1", "recA", "bout_00001")
    m = build_manifest(tmp_path)
    e = m["bouts"]["Session1/recA/bout_00001"]
    assert e == {
        "original_male_fly": DEFAULT_MALE_FLY,
        "reviewed_male_fly": DEFAULT_MALE_FLY,
        "status": "pending",
        "source": "default",
        "reviewed_at": None,
        "applied": False,
        "warning": None,
    }
    assert m["convention"] == {"female": 0, "male": 1}
    assert m["root"] == str(tmp_path)


def test_build_manifest_reads_existing_sex_json(tmp_path):
    make_bout(tmp_path, "Session0", "recA", "bout_00001",
              sex_json={"male_fly": 0, "original_male_fly": 0,
                        "applied_swap": False, "method": "manual"})
    e = build_manifest(tmp_path)["bouts"]["Session0/recA/bout_00001"]
    assert e["original_male_fly"] == 0
    assert e["reviewed_male_fly"] == 0
    assert e["source"] == "sex.json"


def test_rescan_preserves_decisions_and_adds_new(tmp_path):
    make_bout(tmp_path, "Session1", "recA", "bout_00001")
    m1 = build_manifest(tmp_path)
    m1["bouts"]["Session1/recA/bout_00001"].update(
        {"status": "swapped", "reviewed_male_fly": 0, "reviewed_at": "2026-08-07T00:00:00+00:00"})
    make_bout(tmp_path, "Session1", "recA", "bout_00002")
    m2 = build_manifest(tmp_path, existing=m1)
    assert m2["bouts"]["Session1/recA/bout_00001"]["status"] == "swapped"
    assert m2["bouts"]["Session1/recA/bout_00001"]["reviewed_male_fly"] == 0
    assert m2["bouts"]["Session1/recA/bout_00002"]["status"] == "pending"


def test_rescan_flags_disappeared_bout(tmp_path):
    make_bout(tmp_path, "Session1", "recA", "bout_00001")
    m1 = build_manifest(tmp_path)
    import shutil
    shutil.rmtree(tmp_path / "Session1")
    m2 = build_manifest(tmp_path, existing=m1)
    assert m2["bouts"]["Session1/recA/bout_00001"]["warning"] == "bout dir missing on disk"


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_atomic(tmp_path):
    make_bout(tmp_path, "Session1", "recA", "bout_00001")
    m = build_manifest(tmp_path)
    save_manifest(tmp_path, m)
    assert load_manifest(tmp_path) == m
    assert not (tmp_path / (MANIFEST_NAME + ".tmp")).exists()
    assert not list(tmp_path.glob("*.json.tmp"))


def test_load_manifest_missing_returns_none(tmp_path):
    assert load_manifest(tmp_path) is None


def test_read_sex_json_absent_or_corrupt(tmp_path):
    bout_dir = make_bout(tmp_path, "Session1", "recA", "bout_00001")
    assert read_sex_json(bout_dir) is None
    (bout_dir / "sex.json").write_text("{not json")
    assert read_sex_json(bout_dir) is None
