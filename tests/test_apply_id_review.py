"""Tests for scripts/apply_id_review.py's `sex_entry` guard.

The bug these lock down: the guard used to key on `applied`, which is false on
all 160 bouts of this dataset's review while `original_male_fly !=
reviewed_male_fly` in 24 of them. `applied` records that no directory swap was
ever PERFORMED, not that none was needed -- so the guard never fired and the
script happily wrote `male_fly` labels for 24 bouts whose fly dirs hold the
other animal.
"""
from __future__ import annotations

import pytest

from scripts.apply_id_review import sex_entry


def entry(**kw):
    e = {"reviewed_male_fly": 1, "original_male_fly": 1, "status": "confirmed",
         "applied": False, "reviewed_at": "2026-08-29T22:48:41+00:00"}
    e.update(kw)
    return e


def test_agreeing_bout_is_written():
    p = sex_entry("k", entry())
    assert p["male_fly"] == 1 and p["original_male_fly"] == 1


def test_overruled_bout_is_refused_even_though_applied_is_false():
    """The exact shape of all 24 real disagreements: applied false, reviewer
    disagrees with the tracker."""
    with pytest.raises(ValueError, match="overruled the tracker"):
        sex_entry("k", entry(original_male_fly=0, reviewed_male_fly=1,
                             status="swapped", applied=False))


def test_overruled_bout_is_written_once_the_swap_is_done():
    p = sex_entry("k", entry(original_male_fly=0, reviewed_male_fly=1,
                             status="swapped"), allow_overruled=True)
    assert p["male_fly"] == 1 and p["original_male_fly"] == 0


def test_swapped_status_is_accepted():
    """24 of the 160 entries carry status 'swapped'; refusing it would refuse
    every bout the reviewer actually corrected."""
    assert sex_entry("k", entry(status="swapped"))["male_fly"] == 1


@pytest.mark.parametrize("status", ["unsure", "bad", "pending"])
def test_unreviewed_status_is_refused(status):
    with pytest.raises(ValueError, match="status"):
        sex_entry("k", entry(status=status))


@pytest.mark.parametrize("male", [None, 2, "1", True])
def test_non_binary_male_is_refused(male):
    with pytest.raises(ValueError, match="reviewed_male_fly"):
        sex_entry("k", entry(reviewed_male_fly=male, original_male_fly=male))
