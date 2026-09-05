# tests/test_coarse_pass_gates.py
"""`scripts/coarse_pass_gates.py` / `jarvis_jax.tracking.bout_gates` -- gates
read mvq tracks (P4b task 5, spec s5.1).

Expectations these tests encode (CLAUDE.md: state the expectation, check a
real invariant, index by NAME):

  * On an mvq-schema `coarse_tracks.npz` (Task 4's `write_coarse_tracks`
    schema: `exist`, `wing_angle_deg`, `sep3d`, no usable `area`), the gates
    must use EXISTENCE + WING ANGLE, not the SAM3 area-ratio gate (which can
    never pass -- `area`/`area_med` are all-NaN by construction). A
    hand-built 200-coarse-frame trace where the two flies are far apart,
    then close together with the male's wing angle at 45 deg for 60 frames,
    then apart again, must emit EXACTLY ONE bout window covering the close
    stretch, with boundaries matching the hand-built stretch (within
    `--max-gap` coarse frames -- here exact, since the stretch has no
    internal gaps to bridge).
  * On a SAM3-schema `coarse_tracks.npz` (no `exist` array, real
    `area_med`/`border_med`), the gates must be BYTE-IDENTICAL to the
    pre-task-5 script: same CSV, same windows. Pinned by literally running
    the pre-change script (`git show HEAD:scripts/coarse_pass_gates.py`,
    before this task's edit) on the same synthetic file once and hardcoding
    its output below, rather than re-deriving the expectation from the new
    code (which would prove nothing about regression).
  * `--ground-truth` scoring against a GT CSV equal to the emitted truth
    must report recall=1, precision=1, and zero start/end offsets -- the
    scoring path (`validate`) is untouched by the mvq gate work, but is
    exercised end-to-end here on the mvq schema specifically because it
    hadn't been before.

A collapsed/degenerate synthetic trace (e.g. sep3d and wing_angle constant
throughout) would pass every gate function trivially without ever exercising
`segment_runs`' contiguous-run logic; the traces below have three distinct
phases (far / close / far) so a bug that gates the WRONG phase, or drops the
segment split, is visible as a wrong window count or wrong boundaries, not
just "some windows came out."
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import coarse_pass_gates as gates  # noqa: E402

T = 200
CLOSE_S, CLOSE_E = 70, 130          # [CLOSE_S, CLOSE_E) coarse indices -- 60 frames
STRIDE = 16


def _coarse_frame():
    return (np.arange(T) * STRIDE).astype(np.int64)


def _expected_window():
    """The (start_frame, end_frame) the close stretch maps to, matching
    `coarse_to_real`'s [s,e) coarse-index -> inclusive real-frame convention."""
    cf = _coarse_frame()
    return int(cf[CLOSE_S]), int(cf[CLOSE_E - 1])


def _write_mvq_npz(path):
    """Hand-built mvq-schema trace: both flies always trackable (exist high,
    all 7 cams valid, well-separated in 2D so `separable` never trips), far
    apart (sep3d=500 units) and the male's wings folded (5 deg) EXCEPT
    during the close stretch, where they are close (sep3d=10 units, within
    the mvq proximity default of 30) AND the male's wings are extended (45
    deg, above the mvq wing-angle default of 30) -- either signal alone
    would gate the close stretch in; both agreeing is the realistic case."""
    coarse_frame = _coarse_frame()
    area_med = np.full((2, T), np.nan, np.float32)      # no masks: mvq schema
    border_med = np.full((2, T), np.nan, np.float32)
    n_valid_cams = np.full((2, T), 7, np.int16)
    sep2d_med = np.full(T, 200.0, np.float32)           # always separable (>> 15 px)
    sep3d = np.full(T, 500.0, np.float32)
    sep3d[CLOSE_S:CLOSE_E] = 10.0
    exist = np.full((2, T), 0.9, np.float32)            # always >= 0.5
    wing_angle_deg = np.full((2, T), 5.0, np.float32)
    wing_angle_deg[1, CLOSE_S:CLOSE_E] = 45.0           # male_slot = 1

    np.savez_compressed(path, coarse_frame=coarse_frame, area_med=area_med,
                        border_med=border_med, n_valid_cams=n_valid_cams,
                        sep2d_med=sep2d_med, sep3d=sep3d, exist=exist,
                        wing_angle_deg=wing_angle_deg)


def _write_sam3_npz(path):
    """Hand-built SAM3-schema trace, structurally the same three phases
    (far / close+wing-extended / far) but on the SAM3 signals: male
    area_med rises 100 -> 140 (ratio 1.4, above the 1.15 wing-extension
    default) during the close stretch; sep2d_med stays at 300 px throughout
    (never below the 120 px proximity default), so ONLY the area-ratio gate
    fires -- isolating the exact pre-task-5 code path."""
    coarse_frame = _coarse_frame()
    area_med = np.full((2, T), 100.0, np.float32)
    area_med[1, CLOSE_S:CLOSE_E] = 140.0                # male_slot = 1
    border_med = np.full((2, T), 100.0, np.float32)     # >= BORDER_MIN_PX (50)
    n_valid_cams = np.full((2, T), 7, np.int16)         # >= MIN_CAMS (3)
    sep2d_med = np.full(T, 300.0, np.float32)           # separable, never "close"
    sep3d = (sep2d_med / 2.0).astype(np.float32)        # present but unused by SAM3 gates

    np.savez_compressed(path, coarse_frame=coarse_frame, area_med=area_med,
                        border_med=border_med, n_valid_cams=n_valid_cams,
                        sep2d_med=sep2d_med, sep3d=sep3d)


# --------------------------------------------------------------------------
# (a) mvq schema -> one bout covering the close stretch
# --------------------------------------------------------------------------
def test_mvq_gates_emit_one_bout_covering_the_close_stretch(tmp_path):
    npz = tmp_path / "coarse_tracks.npz"
    _write_mvq_npz(npz)

    res = gates.run(str(npz), str(tmp_path / "bouts.csv"), session_tag="test",
                    wing_ratio_min=gates.WING_RATIO_MIN_DEFAULT,
                    proximity_max_px=gates.PROXIMITY_MAX_PX_DEFAULT,
                    min_duration=3, max_gap=2, baseline_window=gates.BASELINE_WINDOW)

    assert res["gates"]["mvq"] is True, "an npz with an `exist` array must be read as mvq schema"
    assert len(res["windows"]) == 1, res["windows"]
    got = res["windows"][0]
    assert got == _expected_window(), (got, _expected_window())

    # boundaries within --max-gap coarse frames (exact here: no internal gaps)
    exp_s, exp_e = _expected_window()
    assert abs(got[0] - exp_s) <= 2 * STRIDE
    assert abs(got[1] - exp_e) <= 2 * STRIDE

    with open(tmp_path / "bouts.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["fly_id"] == "test" and int(rows[0]["start_frame"]) == exp_s
    assert int(rows[0]["end_frame"]) == exp_e


def _write_mvq_npz_proximity_only(path):
    """Same three phases, but ONLY sep3d moves (wing_angle_deg stays at the
    5 deg baseline throughout, never reaching the 30 deg default) -- isolates
    the `close_proximity` half of `behaviour_ok`'s OR from `wing_extension`,
    per fix-round-1 MINOR 3 (the combined trace can't tell OR from AND)."""
    coarse_frame = _coarse_frame()
    area_med = np.full((2, T), np.nan, np.float32)
    border_med = np.full((2, T), np.nan, np.float32)
    n_valid_cams = np.full((2, T), 7, np.int16)
    sep2d_med = np.full(T, 200.0, np.float32)
    sep3d = np.full(T, 500.0, np.float32)
    sep3d[CLOSE_S:CLOSE_E] = 10.0
    exist = np.full((2, T), 0.9, np.float32)
    wing_angle_deg = np.full((2, T), 5.0, np.float32)     # NEVER extended

    np.savez_compressed(path, coarse_frame=coarse_frame, area_med=area_med,
                        border_med=border_med, n_valid_cams=n_valid_cams,
                        sep2d_med=sep2d_med, sep3d=sep3d, exist=exist,
                        wing_angle_deg=wing_angle_deg)


def _write_mvq_npz_wing_angle_only(path):
    """Same three phases, but ONLY wing_angle_deg moves (sep3d stays at the
    500-unit "far apart" baseline throughout, never reaching the 30-unit
    default) -- isolates `wing_extension` from `close_proximity`."""
    coarse_frame = _coarse_frame()
    area_med = np.full((2, T), np.nan, np.float32)
    border_med = np.full((2, T), np.nan, np.float32)
    n_valid_cams = np.full((2, T), 7, np.int16)
    sep2d_med = np.full(T, 200.0, np.float32)
    sep3d = np.full(T, 500.0, np.float32)                 # NEVER close
    exist = np.full((2, T), 0.9, np.float32)
    wing_angle_deg = np.full((2, T), 5.0, np.float32)
    wing_angle_deg[1, CLOSE_S:CLOSE_E] = 45.0

    np.savez_compressed(path, coarse_frame=coarse_frame, area_med=area_med,
                        border_med=border_med, n_valid_cams=n_valid_cams,
                        sep2d_med=sep2d_med, sep3d=sep3d, exist=exist,
                        wing_angle_deg=wing_angle_deg)


def test_proximity_only_trips_the_mvq_behaviour_gate(tmp_path):
    npz = tmp_path / "coarse_tracks.npz"
    _write_mvq_npz_proximity_only(npz)
    res = gates.run(str(npz), str(tmp_path / "bouts.csv"), session_tag="test",
                    wing_ratio_min=gates.WING_RATIO_MIN_DEFAULT,
                    proximity_max_px=gates.PROXIMITY_MAX_PX_DEFAULT,
                    min_duration=3, max_gap=2, baseline_window=gates.BASELINE_WINDOW)
    assert len(res["windows"]) == 1, res["windows"]
    assert res["windows"][0] == _expected_window()
    assert not res["gates"]["wing_extension"].any(), "wing angle never rose -- must not fire"
    assert res["gates"]["close_proximity"][CLOSE_S:CLOSE_E].all()


def test_wing_angle_only_trips_the_mvq_behaviour_gate(tmp_path):
    npz = tmp_path / "coarse_tracks.npz"
    _write_mvq_npz_wing_angle_only(npz)
    res = gates.run(str(npz), str(tmp_path / "bouts.csv"), session_tag="test",
                    wing_ratio_min=gates.WING_RATIO_MIN_DEFAULT,
                    proximity_max_px=gates.PROXIMITY_MAX_PX_DEFAULT,
                    min_duration=3, max_gap=2, baseline_window=gates.BASELINE_WINDOW)
    assert len(res["windows"]) == 1, res["windows"]
    assert res["windows"][0] == _expected_window()
    assert not res["gates"]["close_proximity"].any(), "flies never got close -- must not fire"
    assert res["gates"]["wing_extension"][CLOSE_S:CLOSE_E].all()


# --------------------------------------------------------------------------
# Fix round 1, IMPORTANT 1: single-fly (F=1) mvq files must not crash.
# `wing_angle_deg` has shape (1,T) (no male slot) and neither courtship
# signal exists, so `apply_gates` falls back to trackability alone (see
# `bout_gates`'s module docstring, SINGLE-FLY note) instead of indexing out
# of bounds or comparing against an empty array. This npz builds `sep3d`
# EMPTY on purpose (the shape a real single-fly SAM3 file, and an mvq file
# written before the collapse-guard fix (2026-09), both use) to prove that
# LEGACY shape still works -- `write_coarse_tracks` now emits `sep3d`
# all-NaN at shape (T,) for F<2 instead, which `compute_gate_signals`/
# `apply_gates` treat identically to this empty form (see `bout_gates`'s
# module docstring).
# --------------------------------------------------------------------------
def _write_mvq_npz_single_fly(path):
    """One fly, always well-tracked EXCEPT a dip in existence (simulating an
    occlusion) during the same [CLOSE_S,CLOSE_E) window used elsewhere --
    since there is no courtship gate for a single fly, trackability alone
    must decide the bout boundaries: TWO windows (before/after the dip), not
    one covering everything and not zero."""
    coarse_frame = _coarse_frame()
    area_med = np.full((1, T), np.nan, np.float32)
    border_med = np.full((1, T), np.nan, np.float32)
    n_valid_cams = np.full((1, T), 7, np.int16)
    sep2d_med = np.full(T, np.nan, np.float32)            # real single-fly files: no 2nd fly
    sep3d = np.array([], np.float32)                      # real single-fly files: EMPTY
    exist = np.full((1, T), 0.9, np.float32)
    exist[0, CLOSE_S:CLOSE_E] = 0.3                        # dips below EXIST_MIN (0.5)
    wing_angle_deg = np.full((1, T), 5.0, np.float32)      # no male slot (shape[0]=1)

    np.savez_compressed(path, coarse_frame=coarse_frame, area_med=area_med,
                        border_med=border_med, n_valid_cams=n_valid_cams,
                        sep2d_med=sep2d_med, sep3d=sep3d, exist=exist,
                        wing_angle_deg=wing_angle_deg)


def test_single_fly_mvq_npz_runs_without_crashing(tmp_path):
    npz = tmp_path / "coarse_tracks.npz"
    _write_mvq_npz_single_fly(npz)

    res = gates.run(str(npz), str(tmp_path / "bouts.csv"), session_tag="test",
                    wing_ratio_min=gates.WING_RATIO_MIN_DEFAULT,
                    proximity_max_px=gates.PROXIMITY_MAX_PX_DEFAULT,
                    min_duration=3, max_gap=2, baseline_window=gates.BASELINE_WINDOW)

    assert res["gates"]["mvq"] is True
    # behaviour_ok bypassed to all-True (no 2nd fly): in_bout == trackability_ok,
    # which is False exactly during the existence dip -- two windows, not one
    # spanning the whole recording and not zero.
    assert res["gates"]["behaviour_ok"].all()
    assert not res["gates"]["trackability_ok"][CLOSE_S:CLOSE_E].any()
    assert res["gates"]["trackability_ok"][:CLOSE_S].all()
    assert res["gates"]["trackability_ok"][CLOSE_E:].all()
    assert len(res["windows"]) == 2, res["windows"]
    cf = _coarse_frame()
    assert res["windows"][0] == (int(cf[0]), int(cf[CLOSE_S - 1]))
    assert res["windows"][1] == (int(cf[CLOSE_E]), int(cf[T - 1]))


def test_mvq_detection_also_works_via_meta_json_without_an_exist_array(tmp_path):
    """The npz-`exist`-array detection is the common case (Task 4 always
    writes both); `is_mvq_schema` must ALSO honour a meta json that merely
    says `source: "mvq"`, per the brief's "meta OR exist array" wording."""
    npz = tmp_path / "coarse_tracks.npz"
    coarse_frame = _coarse_frame()
    # SAM3-shaped arrays (no `exist`) but meta claims mvq -- meta must win.
    np.savez_compressed(npz, coarse_frame=coarse_frame,
                        area_med=np.full((2, T), np.nan, np.float32),
                        border_med=np.full((2, T), np.nan, np.float32),
                        n_valid_cams=np.full((2, T), 7, np.int16),
                        sep2d_med=np.full(T, 200.0, np.float32),
                        sep3d=np.full(T, 500.0, np.float32))
    import json
    with open(str(npz).rsplit(".npz", 1)[0] + ".meta.json", "w") as f:
        json.dump({"source": "mvq"}, f)

    z, meta = gates.load_tracks(str(npz))
    assert gates.is_mvq_schema(z, meta) is True
    assert gates.is_mvq_schema(z, None) is False, "without meta, no `exist` array means SAM3"


# --------------------------------------------------------------------------
# (b) SAM3 schema regression -- byte-identical to the pre-task-5 script
# --------------------------------------------------------------------------
# Captured by running `git show HEAD:scripts/coarse_pass_gates.py` (the
# script BEFORE this task's edit) on `_write_sam3_npz`'s file with
# session_tag="test" and the CLI defaults (wing_ratio_min=1.15,
# proximity_max_px=120.0, min_duration=3, max_gap=2, default baseline
# window), read back as raw bytes (`csv.writer`'s default dialect writes
# `\r\n`, which `newline=""` on write preserves literally); see the module
# docstring.
EXPECTED_SAM3_CSV = (
    b"fly_id,bout_idx,start_frame,end_frame,source_fly\r\n"
    b"test,1,1120,2064,both\r\n"
)


def test_sam3_schema_regression_byte_identical_to_pre_task5_script(tmp_path):
    npz = tmp_path / "coarse_tracks.npz"
    _write_sam3_npz(npz)
    out_csv = tmp_path / "bouts.csv"

    res = gates.run(str(npz), str(out_csv), session_tag="test",
                    wing_ratio_min=gates.WING_RATIO_MIN_DEFAULT,
                    proximity_max_px=gates.PROXIMITY_MAX_PX_DEFAULT,
                    min_duration=3, max_gap=2, baseline_window=gates.BASELINE_WINDOW)

    assert res["gates"]["mvq"] is False, "an npz with no `exist` array must be read as SAM3 schema"
    got_csv = out_csv.read_bytes()
    assert got_csv == EXPECTED_SAM3_CSV, (got_csv, EXPECTED_SAM3_CSV)


# --------------------------------------------------------------------------
# (c) --ground-truth scoring on the mvq trace: GT == truth -> perfect score
# --------------------------------------------------------------------------
def test_ground_truth_scoring_on_mvq_trace_is_perfect_when_gt_equals_truth(tmp_path):
    npz = tmp_path / "coarse_tracks.npz"
    _write_mvq_npz(npz)
    exp_s, exp_e = _expected_window()

    gt_csv = tmp_path / "gt.csv"
    with open(gt_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fly_id", "bout_idx", "start_frame", "end_frame", "source_fly"])
        w.writerow(["test", 1, exp_s, exp_e, "both"])

    res = gates.run(str(npz), str(tmp_path / "bouts.csv"), session_tag="test",
                    wing_ratio_min=gates.WING_RATIO_MIN_DEFAULT,
                    proximity_max_px=gates.PROXIMITY_MAX_PX_DEFAULT,
                    min_duration=3, max_gap=2, baseline_window=gates.BASELINE_WINDOW,
                    ground_truth_csv=str(gt_csv))

    report = res["report"]
    assert report is not None
    assert report["recall"] == 1.0
    assert report["precision"] == 1.0
    assert report["n_gt"] == 1 and report["n_windows"] == 1 and report["n_matched"] == 1
    assert report["start_offset_median"] == 0.0
    assert report["end_offset_median"] == 0.0
    assert report["start_offset_mean"] == 0.0
    assert report["end_offset_mean"] == 0.0
    assert report["unmatched_gt"] == [] and report["unmatched_windows"] == []


# --------------------------------------------------------------------------
# CLI surface: --wing-angle-min / --proximity-max-units exist with the
# documented defaults (spec s5.1 / brief).
# --------------------------------------------------------------------------
def test_cli_gains_wing_angle_min_and_proximity_max_units_with_mvq_defaults():
    args = gates.build_arg_parser().parse_args(
        ["--tracks", "t.npz", "--out-csv", "o.csv", "--session-tag", "x"])
    assert args.wing_angle_min == pytest.approx(30.0)
    assert args.proximity_max_units == pytest.approx(30.0)
    # SAM3 flags keep their pre-task-5 defaults
    assert args.wing_ratio_min == pytest.approx(1.15)
    assert args.proximity_max_px == pytest.approx(120.0)
