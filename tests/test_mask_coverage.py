"""Tests for scripts/mask_coverage.py.

Per-camera SAM3 mask coverage as a first-class QC signal (Task 17). The
measured failure mode is NOT SAM3 assigning the same animal to both fly
slots -- it correctly reports "not found" per (fly, cam, frame) when it
loses track, e.g. real Session0 bout22 fly0 (female) valid in only 3/7
cameras. Triangulating from too few views then produces ill-conditioned
garbage. These tests are synthetic (no real data) and exercise:
  - per_frame_views counting,
  - coverage_report's ok/degraded/insufficient classification,
  - the zero-cameras edge case (no division by zero),
  - write_report's atomic round-trip,
  - the run_bout NaN-gating helper in isolation (no GPU/config needed).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from scripts.mask_coverage import (
    MIN_VIEWS_DEFAULT,
    coverage_report,
    load_valid,
    per_frame_views,
    write_report,
)


# ---------------------------------------------------------------------------
# per_frame_views
# ---------------------------------------------------------------------------

def test_per_frame_views_counts_mixed_validity():
    # 2 flies, 4 cams, 3 frames.
    valid = np.zeros((2, 4, 3), dtype=bool)
    valid[0, :, 0] = True                    # fly0 frame0: 4/4
    valid[0, [0, 1], 1] = True                # fly0 frame1: 2/4
    # fly0 frame2: 0/4 (left False)
    valid[1, [0, 1, 2], :] = True             # fly1: 3/4 every frame

    views = per_frame_views(valid)

    assert views.shape == (2, 3)
    np.testing.assert_array_equal(views[0], [4, 2, 0])
    np.testing.assert_array_equal(views[1], [3, 3, 3])


def test_per_frame_views_dtype_is_integer():
    valid = np.ones((1, 7, 5), dtype=bool)
    views = per_frame_views(valid)
    assert np.issubdtype(views.dtype, np.integer)


# ---------------------------------------------------------------------------
# coverage_report status classification
# ---------------------------------------------------------------------------

def _write_mask_npz(path, valid):
    np.savez(path, valid=valid)
    return path


def test_status_insufficient_when_valid_in_3_of_7_cams(tmp_path):
    # fly0 valid in cams 0,1,2 only, all frames -> median_views=3 < min_views=4.
    valid = np.zeros((1, 7, 10), dtype=bool)
    valid[0, [0, 1, 2], :] = True
    npz = _write_mask_npz(tmp_path / "sam3_masks.npz", valid)

    report = coverage_report(npz, min_views=4)

    assert report["per_fly"]["0"]["status"] == "insufficient"
    assert report["per_fly"]["0"]["median_views"] == 3.0


def test_status_ok_when_valid_in_7_of_7_cams(tmp_path):
    valid = np.ones((1, 7, 10), dtype=bool)
    npz = _write_mask_npz(tmp_path / "sam3_masks.npz", valid)

    report = coverage_report(npz, min_views=4)

    assert report["per_fly"]["0"]["status"] == "ok"
    assert report["per_fly"]["0"]["median_views"] == 7.0


def test_status_degraded_when_full_views_but_only_60pct_of_frames(tmp_path):
    # fly0 valid in all 7 cams for 60% of frames, 0 cams for the rest ->
    # median_views is still 7 (>= min_views) so NOT "insufficient", but 40%
    # of frames fall below min_views (>20% threshold) -> "degraded".
    T = 10
    valid = np.zeros((1, 7, T), dtype=bool)
    n_full = 6
    valid[0, :, :n_full] = True
    npz = _write_mask_npz(tmp_path / "sam3_masks.npz", valid)

    report = coverage_report(npz, min_views=4)

    fly0 = report["per_fly"]["0"]
    assert fly0["status"] == "degraded"
    assert fly0["median_views"] == 7.0
    assert fly0["frac_frames_below_min"] == pytest.approx(0.4)


def test_coverage_report_top_level_shape(tmp_path):
    valid = np.ones((2, 7, 5), dtype=bool)
    npz = _write_mask_npz(tmp_path / "sam3_masks.npz", valid)

    report = coverage_report(npz, min_views=4)

    assert report["n_flies"] == 2
    assert report["n_cams"] == 7
    assert report["n_frames"] == 5
    assert set(report["per_fly"].keys()) == {"0", "1"}
    fly0 = report["per_fly"]["0"]
    assert set(fly0.keys()) == {
        "cams_usable", "per_camera_frac", "median_views",
        "frac_frames_below_min", "status"}
    assert len(fly0["per_camera_frac"]) == 7


def test_coverage_report_default_min_views_is_module_default(tmp_path):
    # median_views=3 is below MIN_VIEWS_DEFAULT (4) -> insufficient without
    # having to pass min_views explicitly.
    valid = np.zeros((1, 7, 10), dtype=bool)
    valid[0, [0, 1, 2], :] = True
    npz = _write_mask_npz(tmp_path / "sam3_masks.npz", valid)

    report = coverage_report(npz)

    assert MIN_VIEWS_DEFAULT == 4
    assert report["per_fly"]["0"]["status"] == "insufficient"


# ---------------------------------------------------------------------------
# Zero-camera edge case
# ---------------------------------------------------------------------------

def test_coverage_report_fly_valid_in_zero_cameras_no_div_by_zero(tmp_path):
    valid = np.zeros((1, 7, 10), dtype=bool)   # fly0 never valid anywhere
    npz = _write_mask_npz(tmp_path / "sam3_masks.npz", valid)

    report = coverage_report(npz, min_views=4)

    fly0 = report["per_fly"]["0"]
    assert fly0["cams_usable"] == 0
    assert fly0["median_views"] == 0.0
    assert fly0["frac_frames_below_min"] == 1.0
    assert fly0["status"] == "insufficient"
    assert all(f == 0.0 for f in fly0["per_camera_frac"])
    assert not any(np.isnan(f) for f in fly0["per_camera_frac"])


def test_coverage_report_zero_frames_no_div_by_zero(tmp_path):
    valid = np.zeros((1, 7, 0), dtype=bool)
    npz = _write_mask_npz(tmp_path / "sam3_masks.npz", valid)

    report = coverage_report(npz, min_views=4)

    fly0 = report["per_fly"]["0"]
    assert report["n_frames"] == 0
    assert fly0["median_views"] == 0.0
    assert fly0["frac_frames_below_min"] == 0.0
    assert not any(np.isnan(f) for f in fly0["per_camera_frac"])


# ---------------------------------------------------------------------------
# load_valid
# ---------------------------------------------------------------------------

def test_load_valid_shape_and_dtype(tmp_path):
    valid = np.zeros((2, 7, 4), dtype=np.uint8)
    valid[1] = 1
    npz = _write_mask_npz(tmp_path / "sam3_masks.npz", valid)

    out = load_valid(npz)

    assert out.shape == (2, 7, 4)
    assert out.dtype == bool
    assert out[1].all()
    assert not out[0].any()


# ---------------------------------------------------------------------------
# write_report: atomic round-trip
# ---------------------------------------------------------------------------

def test_write_report_round_trips(tmp_path):
    report = {"n_flies": 2, "n_cams": 7, "n_frames": 10, "per_fly": {"0": {"status": "ok"}}}

    out_path = write_report(tmp_path, report)

    assert out_path == tmp_path / "coverage.json"
    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded == report


def test_write_report_leaves_no_tmp_file_behind(tmp_path):
    report = {"n_flies": 1, "n_cams": 7, "n_frames": 1, "per_fly": {}}
    write_report(tmp_path, report)
    leftovers = [p for p in os.listdir(tmp_path) if "tmp" in p]
    assert leftovers == []


def test_write_report_creates_bout_dir_if_missing(tmp_path):
    bout_dir = tmp_path / "bout_00003"
    report = {"n_flies": 1, "n_cams": 7, "n_frames": 1, "per_fly": {}}

    out_path = write_report(bout_dir, report)

    assert out_path.exists()


# ---------------------------------------------------------------------------
# run_bout NaN-gating helper (pure, no GPU/config)
# ---------------------------------------------------------------------------

from scripts.run_bout import gate_low_coverage_frames  # noqa: E402


def test_gate_frames_below_min_views_become_nan():
    T, K = 4, 3
    kp3d = np.arange(T * K * 3, dtype=np.float64).reshape(T, K, 3)
    views = np.array([7, 2, 4, 1])

    gated, n_gated = gate_low_coverage_frames(kp3d, views, min_views=4)

    assert n_gated == 2
    assert np.isnan(gated[1]).all()
    assert np.isnan(gated[3]).all()
    np.testing.assert_array_equal(gated[0], kp3d[0])
    np.testing.assert_array_equal(gated[2], kp3d[2])


def test_gate_frames_at_min_views_are_untouched():
    kp3d = np.ones((1, 2, 3), dtype=np.float64) * 5.0
    views = np.array([4])

    gated, n_gated = gate_low_coverage_frames(kp3d, views, min_views=4)

    assert n_gated == 0
    np.testing.assert_array_equal(gated, kp3d)


def test_gate_none_min_views_is_no_op():
    kp3d = np.ones((3, 2, 3), dtype=np.float64) * 9.0
    views = np.array([0, 1, 7])

    gated, n_gated = gate_low_coverage_frames(kp3d, views, min_views=None)

    assert n_gated == 0
    np.testing.assert_array_equal(gated, kp3d)


def test_gate_does_not_mutate_input_array():
    kp3d = np.ones((2, 2, 3), dtype=np.float64)
    original = kp3d.copy()
    views = np.array([0, 7])

    gate_low_coverage_frames(kp3d, views, min_views=4)

    np.testing.assert_array_equal(kp3d, original)
