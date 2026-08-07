"""Tests for scripts/viz/apply_fly_id_review.py (physical fly0/fly1 swap)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.viz.apply_fly_id_review import main, plan_swaps, swap_bout, validate
from scripts.viz.fly_id_review import (
    bout_dir_from_key, build_manifest, load_manifest, read_sex_json, save_manifest,
)
from tests.test_fly_id_review import make_bout


def entry(status="confirmed", reviewed=1, original=1, applied=False, warning=None):
    return {"original_male_fly": original, "reviewed_male_fly": reviewed,
            "status": status, "source": "default",
            "reviewed_at": "2026-08-07T00:00:00+00:00",
            "applied": applied, "warning": warning}


def manifest_of(root, **bouts):
    return {"root": str(root), "convention": {"female": 0, "male": 1}, "bouts": bouts}


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_validate_blocks_pending_and_unsure(tmp_path):
    m = manifest_of(tmp_path, a=entry("pending"), b=entry("unsure"), c=entry("confirmed"))
    errors = validate(m, allow_pending=False)
    assert len(errors) == 2
    errors = validate(m, allow_pending=True)
    assert len(errors) == 1 and "unsure" in errors[0]  # unsure ALWAYS blocks


# ---------------------------------------------------------------------------
# plan_swaps
# ---------------------------------------------------------------------------

def test_plan_swaps_selects_reviewed_male_fly0(tmp_path):
    m = manifest_of(
        tmp_path,
        swap_me=entry("swapped", reviewed=0),
        keep=entry("confirmed", reviewed=1),
        already=entry("swapped", reviewed=0, applied=True),
        broken=entry("swapped", reviewed=0, warning="missing: fly1/sidebyside.mp4"),
        pend=entry("pending", reviewed=0, original=0),
    )
    assert plan_swaps(m) == ["swap_me"]
    assert plan_swaps(m, allow_pending=True) == ["pend", "swap_me"]


# ---------------------------------------------------------------------------
# swap_bout
# ---------------------------------------------------------------------------

def test_swap_bout_exchanges_dirs_and_updates_sex_json(tmp_path):
    bout_dir = make_bout(tmp_path, "Session1", "recA", "bout_00001")
    (bout_dir / "fly0" / "MARKER_A").touch()
    (bout_dir / "fly1" / "MARKER_B").touch()
    e = entry("swapped", reviewed=0, original=1)
    swap_bout(tmp_path, "Session1/recA/bout_00001", e)
    assert (bout_dir / "fly0" / "MARKER_B").exists()  # old fly1 now fly0
    assert (bout_dir / "fly1" / "MARKER_A").exists()  # old fly0 now fly1
    sex = read_sex_json(bout_dir)
    assert sex["male_fly"] == 1
    assert sex["applied_swap"] is True
    assert sex["original_male_fly"] == 1  # entry's original preserved verbatim
    assert not (bout_dir / ".fly_swap_tmp").exists()


def test_swap_bout_stale_tmp_marker_raises(tmp_path):
    bout_dir = make_bout(tmp_path, "Session1", "recA", "bout_00001")
    (bout_dir / ".fly_swap_tmp").mkdir()
    with pytest.raises(RuntimeError, match="fly_swap_tmp"):
        swap_bout(tmp_path, "Session1/recA/bout_00001", entry("swapped", reviewed=0))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _reviewed_tree(tmp_path):
    """Two bouts: bout_00001 needs a swap, bout_00002 confirmed as-is."""
    make_bout(tmp_path, "Session1", "recA", "bout_00001")
    make_bout(tmp_path, "Session1", "recA", "bout_00002")
    m = build_manifest(tmp_path)
    m["bouts"]["Session1/recA/bout_00001"].update(
        {"status": "swapped", "reviewed_male_fly": 0,
         "reviewed_at": "2026-08-07T00:00:00+00:00"})
    m["bouts"]["Session1/recA/bout_00002"].update(
        {"status": "confirmed", "reviewed_at": "2026-08-07T00:00:00+00:00"})
    save_manifest(tmp_path, m)
    return m


def test_main_dry_run_changes_nothing(tmp_path, capsys):
    _reviewed_tree(tmp_path)
    (bout_dir_from_key(tmp_path, "Session1/recA/bout_00001") / "fly0" / "M0").touch()
    main(["--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "dry-run" in out and "bout_00001" in out
    assert (bout_dir_from_key(tmp_path, "Session1/recA/bout_00001") / "fly0" / "M0").exists()
    assert load_manifest(tmp_path)["bouts"]["Session1/recA/bout_00001"]["applied"] is False


def test_main_apply_swaps_and_is_idempotent(tmp_path, capsys):
    _reviewed_tree(tmp_path)
    bout_dir = bout_dir_from_key(tmp_path, "Session1/recA/bout_00001")
    (bout_dir / "fly0" / "M0").touch()
    main(["--root", str(tmp_path), "--apply"])
    assert (bout_dir / "fly1" / "M0").exists()  # swapped
    m = load_manifest(tmp_path)
    e = m["bouts"]["Session1/recA/bout_00001"]
    assert e["applied"] is True and e["reviewed_male_fly"] == 1
    main(["--root", str(tmp_path), "--apply"])  # second run: no-op
    assert (bout_dir / "fly1" / "M0").exists()  # NOT swapped back


def test_main_refuses_on_pending(tmp_path, capsys):
    make_bout(tmp_path, "Session1", "recA", "bout_00001")
    save_manifest(tmp_path, build_manifest(tmp_path))
    with pytest.raises(SystemExit):
        main(["--root", str(tmp_path), "--apply"])
    assert "REFUSING" in capsys.readouterr().out


def test_main_allow_pending_treats_pending_as_original(tmp_path):
    make_bout(tmp_path, "Session1", "recA", "bout_00001",
              sex_json={"male_fly": 0})  # original says male in fly0 -> needs swap
    (bout_dir_from_key(tmp_path, "Session1/recA/bout_00001") / "fly0" / "M0").touch()
    save_manifest(tmp_path, build_manifest(tmp_path))
    main(["--root", str(tmp_path), "--apply", "--allow-pending"])
    assert (bout_dir_from_key(tmp_path, "Session1/recA/bout_00001") / "fly1" / "M0").exists()


def test_main_missing_manifest_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["--root", str(tmp_path), "--apply"])
