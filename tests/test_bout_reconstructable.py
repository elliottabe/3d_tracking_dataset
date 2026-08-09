"""Tests for scripts/qc/bout_reconstructable.py.

Bouts where a fly is outside too many cameras' field of view cannot be
triangulated at all -- that is rig geometry, not a fixable segmentation
failure -- so they are flagged and excluded rather than allowed to emit 3D
that was never observable. See the module docstring for the measurements.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.qc.bout_reconstructable import (
    MIN_VIEWS_DEFAULT,
    apply_exclusions,
    bout_verdict,
    coverage_from_conf,
    coverage_from_valid,
    scan_recording,
    write_report,
)


# ---------------------------------------------------------------------------
# coverage_from_valid
# ---------------------------------------------------------------------------

def test_all_cameras_valid_is_full_coverage():
    assert coverage_from_valid(np.ones((7, 100), bool)) == 1.0


def test_three_of_seven_cameras_is_zero_coverage_at_min_views_four():
    # Session0 bout 22's female: 3 cameras see her, 4 do not -> nothing is
    # triangulable under the shipped min_views=4 gate.
    valid = np.zeros((7, 100), bool)
    valid[:3] = True
    assert coverage_from_valid(valid, min_views=4) == 0.0
    # ...but the same data is fully usable if only 3 views are demanded.
    assert coverage_from_valid(valid, min_views=3) == 1.0


def test_partial_coverage_counts_frames_not_cameras():
    valid = np.zeros((7, 10), bool)
    valid[:4, :6] = True          # 6 frames with 4 views
    valid[:2, 6:] = True          # 4 frames with 2 views
    assert coverage_from_valid(valid, min_views=4) == pytest.approx(0.6)


def test_empty_bout_is_zero_not_nan():
    assert coverage_from_valid(np.zeros((7, 0), bool)) == 0.0


def test_valid_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\(C,T\)"):
        coverage_from_valid(np.ones((2, 7, 10), bool))


# ---------------------------------------------------------------------------
# coverage_from_conf
# ---------------------------------------------------------------------------

def test_conf_basis_counts_views_above_the_gate():
    T, C, K = 10, 7, 50
    conf = np.full((T, C, K), 0.95, np.float32)
    conf[:, 5:] = 0.2                       # 2 cameras below the view gate
    assert coverage_from_conf(conf, min_views=4, view_conf=0.6) == 1.0
    assert coverage_from_conf(conf, min_views=6, view_conf=0.6) == 0.0


def test_conf_basis_uses_median_so_a_few_occluded_keypoints_dont_disqualify():
    T, C, K = 5, 7, 50
    conf = np.full((T, C, K), 0.9, np.float32)
    conf[:, :, :10] = 0.0                   # 10 of 50 keypoints occluded
    assert coverage_from_conf(conf, min_views=7, view_conf=0.6) == 1.0


def test_conf_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\(T,C,K\)"):
        coverage_from_conf(np.ones((10, 7), np.float32))


# ---------------------------------------------------------------------------
# bout_verdict
# ---------------------------------------------------------------------------

def test_bout_kept_only_when_every_fly_clears():
    assert bout_verdict([0.95, 0.92], min_frac=0.8)[0] is True


def test_one_perfect_fly_does_not_rescue_a_bout():
    # The real Session0 bout 22 shape: fly1 perfect, fly0 unobservable. Both
    # animals are needed for any interaction measure, so the bout is dropped.
    keep, reason = bout_verdict([0.0, 1.0], min_frac=0.8)
    assert keep is False
    assert "fly0" in reason


def test_verdict_reason_names_every_failing_fly():
    keep, reason = bout_verdict([0.1, 0.2], min_frac=0.8)
    assert keep is False
    assert "fly0" in reason and "fly1" in reason


def test_threshold_boundary_is_inclusive():
    assert bout_verdict([0.8, 0.9], min_frac=0.8)[0] is True
    assert bout_verdict([0.79999, 0.9], min_frac=0.8)[0] is False


def test_no_flies_is_not_kept():
    keep, reason = bout_verdict([], min_frac=0.8)
    assert keep is False and "no flies" in reason


# ---------------------------------------------------------------------------
# scanning a synthetic recording tree
# ---------------------------------------------------------------------------

def make_recording(tmp_path, name="rec", bouts=None):
    """bouts: {idx: {"valid": (A,C,T) or None, "conf": [ (T,C,K), ... ] or None}}"""
    rec = tmp_path / name
    for idx, spec in (bouts or {}).items():
        bdir = rec / "pose" / "bouts" / f"bout_{idx:05d}"
        confs = spec.get("conf")
        n_fly = len(confs) if confs is not None else (
            spec["valid"].shape[0] if spec.get("valid") is not None else 2)
        for f in range(n_fly):
            (bdir / f"fly{f}").mkdir(parents=True, exist_ok=True)
            if confs is not None:
                np.savez(bdir / f"fly{f}" / "kp2d.npz", conf=confs[f],
                         kp2d=np.zeros(confs[f].shape + (2,), np.float32))
        if spec.get("valid") is not None:
            mdir = rec / "sam3_masks" / f"bout_{idx:05d}"
            mdir.mkdir(parents=True, exist_ok=True)
            np.savez(mdir / "sam3_masks.npz", valid=spec["valid"])
    return rec


def test_scan_prefers_mask_basis_when_both_exist(tmp_path):
    valid = np.zeros((2, 7, 50), bool)
    valid[0, :3] = True                      # fly0: only 3 cameras
    valid[1, :] = True
    conf = [np.full((50, 7, 50), 0.95, np.float32) for _ in range(2)]
    rec = make_recording(tmp_path, bouts={1: {"valid": valid, "conf": conf}})
    (row,) = scan_recording(rec)
    assert row["basis"] == "mask"            # NOT the permissive kp2d
    assert row["fractions"][0] == 0.0
    assert row["keep"] is False


def test_scan_falls_back_to_kp2d_when_masks_absent(tmp_path):
    conf = [np.full((50, 7, 50), 0.95, np.float32) for _ in range(2)]
    rec = make_recording(tmp_path, bouts={2: {"valid": None, "conf": conf}})
    (row,) = scan_recording(rec)
    assert row["basis"] == "kp2d"
    assert row["keep"] is True


def test_scan_reports_unassessable_bout_rather_than_silently_keeping(tmp_path):
    rec = tmp_path / "rec"
    (rec / "pose" / "bouts" / "bout_00003").mkdir(parents=True)
    (row,) = scan_recording(rec)
    assert row["basis"] == "none"
    assert row["keep"] is False              # must not default to keeping
    assert "cannot assess" in row["reason"]


def test_basis_can_be_forced_to_kp2d(tmp_path):
    valid = np.zeros((2, 7, 50), bool)
    valid[0, :3] = True
    valid[1, :] = True
    conf = [np.full((50, 7, 50), 0.95, np.float32) for _ in range(2)]
    rec = make_recording(tmp_path, bouts={1: {"valid": valid, "conf": conf}})
    (row,) = scan_recording(rec, basis="kp2d")
    assert row["basis"] == "kp2d" and row["keep"] is True


# ---------------------------------------------------------------------------
# report + apply
# ---------------------------------------------------------------------------

def test_write_report_counts(tmp_path):
    rows = [{"recording": "r", "bout": 1, "basis": "mask", "fractions": [1.0, 1.0],
             "keep": True, "reason": "ok"},
            {"recording": "r", "bout": 2, "basis": "mask", "fractions": [0.0, 1.0],
             "keep": False, "reason": "bad"}]
    payload = write_report(rows, tmp_path / "r.json", tmp_path / "r.csv")
    assert payload["n_keep"] == 1 and payload["n_drop"] == 1
    assert (tmp_path / "r.csv").is_file()
    assert json.loads((tmp_path / "r.json").read_text())["n_bouts"] == 2


def test_apply_writes_markers_only_for_dropped_bouts(tmp_path):
    root = tmp_path / "courtship"
    for idx in (1, 2):
        (root / "S0" / "rec" / "pose" / "bouts" / f"bout_{idx:05d}").mkdir(parents=True)
    rows = [{"session": "S0", "recording": "rec", "bout": 1, "basis": "mask",
             "fractions": [1.0, 1.0], "keep": True, "reason": "ok"},
            {"session": "S0", "recording": "rec", "bout": 2, "basis": "mask",
             "fractions": [0.0, 1.0], "keep": False, "reason": "bad"}]
    n_add, n_clear = apply_exclusions(rows, root)
    base = root / "S0" / "rec" / "pose" / "bouts"
    assert n_add == 1 and n_clear == 0
    assert not (base / "bout_00001" / "EXCLUDED.json").exists()
    assert (base / "bout_00002" / "EXCLUDED.json").exists()


def test_apply_clears_stale_markers_when_a_bout_starts_passing(tmp_path):
    # Loosening the threshold must un-exclude a bout, not leave it wrongly
    # flagged from a previous stricter run.
    root = tmp_path / "courtship"
    bdir = root / "S0" / "rec" / "pose" / "bouts" / "bout_00001"
    bdir.mkdir(parents=True)
    (bdir / "EXCLUDED.json").write_text("{}")
    rows = [{"session": "S0", "recording": "rec", "bout": 1, "basis": "mask",
             "fractions": [0.95, 0.99], "keep": True, "reason": "ok"}]
    n_add, n_clear = apply_exclusions(rows, root)
    assert n_add == 0 and n_clear == 1
    assert not (bdir / "EXCLUDED.json").exists()


def test_apply_never_deletes_pose_data(tmp_path):
    root = tmp_path / "courtship"
    bdir = root / "S0" / "rec" / "pose" / "bouts" / "bout_00002"
    (bdir / "fly0").mkdir(parents=True)
    payload = bdir / "fly0" / "outputs.h5"
    payload.write_text("precious")
    rows = [{"session": "S0", "recording": "rec", "bout": 2, "basis": "mask",
             "fractions": [0.0, 1.0], "keep": False, "reason": "bad"}]
    apply_exclusions(rows, root)
    assert payload.read_text() == "precious"
