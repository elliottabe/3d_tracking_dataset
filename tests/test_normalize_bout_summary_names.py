"""Rename-planning rules for the bout-summary normalizer.

This script mutates the data tree, so the selection rules are tested before it
runs: never touch OLD_-prefixed files, never re-rename an already-canonical
dir, and refuse rather than guess when a dir has two legacy names.
"""
from __future__ import annotations

import pytest

from scripts.data.normalize_bout_summary_names import CANONICAL, plan_renames


def _dir(root, name, *files):
    d = root / name
    d.mkdir()
    for f in files:
        (d / f).write_text('x')
    return d


def test_plans_rename_for_pre_rename_name(tmp_path):
    d = _dir(tmp_path, 'Predictions_3D_a', 'free_walking_bouts_summary.csv')
    assert plan_renames(tmp_path) == [
        (d / 'free_walking_bouts_summary.csv', d / CANONICAL)]


def test_plans_rename_for_singular_bout(tmp_path):
    d = _dir(tmp_path, 'Predictions_3D_a', 'free_running_bout_summary.csv')
    assert plan_renames(tmp_path) == [
        (d / 'free_running_bout_summary.csv', d / CANONICAL)]


def test_skips_already_canonical_dir(tmp_path):
    _dir(tmp_path, 'Predictions_3D_a', CANONICAL)
    assert plan_renames(tmp_path) == []


def test_canonical_wins_over_legacy_in_same_dir(tmp_path):
    """A dir holding both is already done; the legacy file is left as-is."""
    _dir(tmp_path, 'Predictions_3D_a', CANONICAL,
         'free_walking_bouts_summary.csv')
    assert plan_renames(tmp_path) == []


def test_never_touches_OLD_prefixed_file(tmp_path):
    _dir(tmp_path, 'Predictions_3D_a', 'OLD_walking_bouts_summary.csv')
    assert plan_renames(tmp_path) == []


def test_refuses_when_two_legacy_names_present(tmp_path):
    _dir(tmp_path, 'Predictions_3D_a', 'free_walking_bouts_summary.csv',
         'free_running_bout_summary.csv')
    with pytest.raises(SystemExit, match='multiple legacy names'):
        plan_renames(tmp_path)


def test_ignores_dirs_that_are_not_predictions(tmp_path):
    _dir(tmp_path, 'something_else', 'free_walking_bouts_summary.csv')
    assert plan_renames(tmp_path) == []


def test_finds_dirs_nested_under_session_folders(tmp_path):
    (tmp_path / 'session10').mkdir()
    d = _dir(tmp_path, 'session10/Predictions_3D_a',
             'free_walking_bouts_summary.csv')
    assert plan_renames(tmp_path) == [
        (d / 'free_walking_bouts_summary.csv', d / CANONICAL)]
