"""build_fig4_panels shapes the bundle correctly from synthetic analysis output."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import figbuilder.panels  # noqa: F401
from figbuilder.panels.base import get_panel_type
from scripts.figures.export_fig4_bundle import (
    build_fig4_panels, _pair_qpos, _pair_center_xyz, _resolve_session_bout,
    _load_kp3d, _processed_fly_dir, _free_running_com_z,
    _resolve_sam3_camera_index, _resolve_male_female_slots, _slot_to_raw_frame)

REPO = Path(__file__).resolve().parents[1]

FS = 800.0
T = 300


def _synthetic():
    rng = np.random.default_rng(0)
    seg = [{"start": 10, "end": 80, "type": "pulse"},
           {"start": 120, "end": 220, "type": "sine"}]
    ex = {
        "T": T,
        "com_z": rng.normal(size=T),
        "song0": {
            "wing_data": {"WingL_V13": {"z": rng.normal(size=T)},
                          "WingR_V13": {"z": rng.normal(size=T)}},
            "sides": {"L": {"segments": seg}, "R": {"segments": seg}},
            "angle_L": rng.normal(size=T), "angle_R": rng.normal(size=T),
            "dominant_wing": "L",
        },
    }
    results = [{"male_valid": np.ones(T, bool),
                "male_labels": np.array(["pulse"] * 150 + ["sine"] * 150),
                "com_z": rng.normal(size=T),
                "song0": ex["song0"]} for _ in range(4)]
    extras = {
        "fs": FS, "start_frame": 0, "end_frame": T,
        "walking_z": rng.normal(size=20),
        "phase_diffs": rng.uniform(-np.pi, np.pi, 40),
        "per_bout_align": np.abs(rng.normal(size=30)) * 15,
        "male_pitch": rng.normal(size=T) * 10,
        "target_pitch": rng.normal(size=T) * 10,
        "pulse_centroids": {"Pslow": rng.normal(size=48),
                            "Pfast": rng.normal(size=48)},
        "pulse_pooled": {"Pslow": rng.normal(size=(9, 48)),
                         "Pfast": rng.normal(size=(7, 48))},
        "pulse_counts": {"Pslow": 9, "Pfast": 7},
        "ext_pulse": np.abs(rng.normal(size=200)) * 30,
        "ext_sine": np.abs(rng.normal(size=200)) * 30,
        "video_frames": [np.zeros((16, 16, 3), np.uint8) for _ in range(4)],
        "render_frames": [np.zeros((16, 16, 3), np.uint8) for _ in range(4)],
        "exemplar_bout_idx": 3,
    }
    return results, ex, extras


def test_bundle_contains_a_panel_for_every_fig4_element():
    panels = build_fig4_panels(*_synthetic())
    expected = {"wing", "scut", "sine_phase", "wing_phase_polar", "angle_2d",
                "pulse_class", "zheight", "pitch", "align_violin"}
    assert expected <= set(panels)


def test_video_and_render_frames_become_individual_image_panels():
    panels = build_fig4_panels(*_synthetic())
    assert {"video_0", "video_1", "video_2", "video_3"} <= set(panels)
    assert {"render_0", "render_1", "render_2", "render_3"} <= set(panels)
    assert panels["video_0"].assets["img"].dtype == np.uint8


def test_pulse_class_panel_stores_flat_per_type_arrays():
    """Ruling 5: the bundle is flat; the adapter re-nests into Pslow/Pfast."""
    panels = build_fig4_panels(*_synthetic())
    d = panels["pulse_class"].data
    assert {"centroid_Pslow", "centroid_Pfast",
            "pooled_Pslow", "pooled_Pfast"} <= set(d)
    assert d["centroid_Pslow"].shape == (48,)
    assert d["pooled_Pfast"].shape == (7, 48)


def test_pulse_class_adapter_renders_from_the_bundled_arrays():
    """End-to-end: flat datasets -> nested pulse_type_results -> a drawn axes."""
    import matplotlib.pyplot as plt
    panels = build_fig4_panels(*_synthetic())
    pt = get_panel_type("courtship.pulse_class")
    fig, ax = plt.subplots()
    try:
        pt.draw(ax, panels["pulse_class"].data,
                {"fs": FS, "show_std": True, "count_Pslow": 9, "count_Pfast": 7})
        labels = [ln.get_label() for ln in ax.lines]
        assert any("Pslow (n=9)" == lb for lb in labels), labels
        assert any("Pfast (n=7)" == lb for lb in labels), labels
    finally:
        plt.close(fig)


def test_segment_tables_are_stored_as_structured_arrays():
    panels = build_fig4_panels(*_synthetic())
    assert panels["wing"].data["seg_L"].dtype.names == ("start", "end", "type")


@pytest.mark.parametrize("pid", ["wing", "scut", "pitch", "align_violin",
                                 "zheight", "angle_2d", "wing_phase_polar",
                                 "sine_phase", "pulse_class"])
def test_each_panel_satisfies_its_types_declared_needs(pid):
    """Covers EVERY data-bearing panel, including sine_phase and pulse_class.

    This is the test that catches an adapter whose `needs` drifts from what
    build_fig4_panels actually emits — exactly the pulse_class defect found in
    the pre-flight audit (Ruling 5).
    """
    panels = build_fig4_panels(*_synthetic())
    pd = panels[pid]
    pt = get_panel_type(pd.type)
    assert set(pt.needs) <= set(pd.data), f"{pid} missing {set(pt.needs) - set(pd.data)}"


def test_pair_qpos_concatenates_two_flies_side_by_side():
    """Requirement change: the render strip needs the PAIR qpos layout
    [fly0_qpos | fly1_qpos] that build_courtship_pair_visualizer's model
    expects (nq = 2 * fly_nq), not one fly's qpos alone."""
    rng = np.random.default_rng(1)
    q0 = rng.normal(size=(50, 93))
    q1 = rng.normal(size=(50, 93))
    out = _pair_qpos(q0, q1, 50)
    assert out.shape == (50, 186)
    np.testing.assert_array_equal(out[:, :93], q0)
    np.testing.assert_array_equal(out[:, 93:], q1)


def test_pair_qpos_truncates_to_the_shared_frame_count():
    rng = np.random.default_rng(2)
    q0 = rng.normal(size=(80, 93))
    q1 = rng.normal(size=(60, 93))
    out = _pair_qpos(q0, q1, 60)
    assert out.shape == (60, 186)


def test_cli_runs_standalone(tmp_path):
    """export_fig4_bundle.py must bootstrap the repo root itself: running
    `python scripts/figures/export_fig4_bundle.py` (not `-m`) puts
    scripts/figures/ on sys.path, not the repo root, so its
    `from figbuilder...`/`from utils...` imports die with ModuleNotFoundError
    without the bootstrap. Same pattern as
    scripts/export/pack_reference_clips.py (commit e030c61)."""
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "figures" / "export_fig4_bundle.py"),
         "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]


def test_cli_runs_as_module():
    """The `-m` invocation form must keep working too (it never broke, but
    both forms are asserted so a future change can't silently regress one)."""
    r = subprocess.run(
        [sys.executable, "-m", "scripts.figures.export_fig4_bundle", "--help"],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]


def test_pair_center_xyz_is_the_scutellum_midpoint():
    """Finding 2 (round 2): the video crop must be centred on the pair's
    actual position for THIS recording, not a fixed window copied from a
    different one. The centre is the elementwise midpoint of the two
    flies' Scutellum keypoint.

    Round 6: `_pair_center_xyz` now takes already-loaded kp3d arrays
    directly (the true DLT world frame from the processed pose tree), not
    `data`/`ex`/`kp_names` reaching into the combined h5's re-centred
    `kp_data`."""
    rng = np.random.default_rng(5)
    n_kp = 50
    kp0 = rng.normal(size=(20, n_kp, 3))
    kp1 = rng.normal(size=(20, n_kp, 3))
    scut_idx = 7

    out = _pair_center_xyz(kp0, kp1, scut_idx, T=20)
    expected = 0.5 * (kp0[:, 7, :] + kp1[:, 7, :])
    assert out.shape == (20, 3)
    np.testing.assert_array_equal(out, expected)


def test_pair_center_xyz_clamps_to_the_shorter_bout():
    rng = np.random.default_rng(6)
    n_kp = 50
    kp0 = rng.normal(size=(30, n_kp, 3))
    kp1 = rng.normal(size=(18, n_kp, 3))
    scut_idx = 3

    out = _pair_center_xyz(kp0, kp1, scut_idx, T=30)
    assert out.shape == (18, 3)
    np.testing.assert_array_equal(out, 0.5 * (kp0[:18, 3, :] + kp1[:18, 3, :]))


_SUMMARY_CSV_HEADER = "fly_id,bout_idx,start_frame,end_frame,source_fly\n"


def _write_summary_csv(tmp_path, rows):
    """rows: list of (fly_id, bout_idx, start_frame, end_frame).

    Round 6: `_resolve_session_bout` now points at the PROCESSED tree, whose
    CSV is named `courtship_bout_summary.csv` (singular "bout", no
    "unified") -- a different filename from the Video_recordings tree's
    `courtship_bouts_unified_summary.csv` used in rounds 3-5."""
    path = tmp_path / "courtship_bout_summary.csv"
    lines = [_SUMMARY_CSV_HEADER]
    for fly_id, bout_idx, start, end in rows:
        lines.append(f"{fly_id},{bout_idx},{start},{end},both\n")
    path.write_text("".join(lines))
    return path


def _write_fake_sam3_npz(sam3_root, bout_dir_name, n_frames, include_packed=False):
    """A tiny stand-in npz: only `valid` (2, 7, n_frames), matching the real
    shape's LAST axis (frame count) — `_resolve_session_bout` must only read
    this. `packed` is (2, 7, N, 448, 242) for real data and deliberately
    omitted by default so a test can assert the helper never touches it."""
    d = Path(sam3_root) / bout_dir_name
    d.mkdir(parents=True, exist_ok=True)
    arrays = {"valid": np.zeros((2, 7, n_frames), dtype=bool)}
    if include_packed:
        arrays["packed"] = np.zeros((2, 7, n_frames, 2, 2), dtype=np.uint8)
    np.savez(d / "sam3_masks.npz", **arrays)


def test_resolve_session_bout_exact_single_match(tmp_path):
    """Round-3 finding: the real case — recording
    Session1/2026_04_02_16_21_32, CSV bout_idx 5, start_frame 380781,
    end_frame 381558 (n=778) supplies the frame offset. The sam3 DIRECTORY
    is resolved independently, by mask frame count (round-5 fix) — here it
    is bout_00003, not bout_00005, to keep the test honest about that."""
    _write_summary_csv(tmp_path, [
        ("Session1/2026_04_02_16_21_32", 1, 248710, 249361),
        ("Session1/2026_04_02_16_21_32", 4, 344050, 346068),
        ("Session1/2026_04_02_16_21_32", 5, 380781, 381558),   # n=778
    ])
    sam3_root = tmp_path / "sam3"
    _write_fake_sam3_npz(sam3_root, "bout_00001", 652)
    _write_fake_sam3_npz(sam3_root, "bout_00002", 287)
    _write_fake_sam3_npz(sam3_root, "bout_00003", 778)
    bout_dir, start_frame = _resolve_session_bout(
        tmp_path, sam3_root, "Session1/2026_04_02_16_21_32", clip_len=778)
    assert bout_dir == "bout_00003"
    assert start_frame == 380781


def test_resolve_session_bout_zero_matches_raises_naming_clip_len(tmp_path):
    """CSV-side zero match (round-4 case) still raises before the sam3
    scan ever runs, so an empty/nonexistent sam3_root doesn't matter here."""
    _write_summary_csv(tmp_path, [
        ("Session1/2026_04_02_16_21_32", 1, 248710, 249361),
    ])
    with pytest.raises(ValueError, match="777"):
        _resolve_session_bout(
            tmp_path, tmp_path / "sam3", "Session1/2026_04_02_16_21_32",
            clip_len=777)


def test_resolve_session_bout_ambiguous_csv_match_lists_bout_idx(tmp_path):
    """Two CSV rows with the same length must raise, never silently pick
    one — this IS the failure mode the fix exists to prevent (round-4
    case, on the CSV side)."""
    _write_summary_csv(tmp_path, [
        ("Session1/2026_04_02_16_21_32", 5, 380781, 381558),   # n=778
        ("Session1/2026_04_02_16_21_32", 9, 505035, 505812),   # n=778
    ])
    with pytest.raises(ValueError, match="5") as excinfo:
        _resolve_session_bout(
            tmp_path, tmp_path / "sam3", "Session1/2026_04_02_16_21_32",
            clip_len=778)
    assert "9" in str(excinfo.value)


def test_resolve_session_bout_ignores_other_recordings(tmp_path):
    """A same-length bout from a DIFFERENT recording must not count as a
    CSV match — only the row whose fly_id equals the (suffix-stripped)
    requested recording id."""
    _write_summary_csv(tmp_path, [
        ("Session1/2025_10_20_13_20_04", 3, 1000, 1777),        # n=778, wrong recording
        ("Session1/2026_04_02_16_21_32", 5, 380781, 381558),    # n=778, right recording
    ])
    sam3_root = tmp_path / "sam3"
    _write_fake_sam3_npz(sam3_root, "bout_00007", 778)
    bout_dir, start_frame = _resolve_session_bout(
        tmp_path, sam3_root, "Session1/2026_04_02_16_21_32", clip_len=778)
    assert bout_dir == "bout_00007"
    assert start_frame == 380781


def test_resolve_session_bout_strips_fly_suffix_from_recording(tmp_path):
    """Round-4 regression: recording_of(ex) returns info/fly_ids' value,
    which carries a trailing _flyN suffix (e.g. 'Session1/..._fly1'), but
    courtship_bouts_unified_summary.csv's fly_id column has NO such suffix.
    Matching must strip the suffix before comparing, not require it to
    appear verbatim in the CSV -- this is exactly what made the round-3 fix
    resolve zero bouts on the real gate run."""
    _write_summary_csv(tmp_path, [
        ("Session1/2026_04_02_16_21_32", 5, 380781, 381558),   # n=778
    ])
    sam3_root = tmp_path / "sam3"
    _write_fake_sam3_npz(sam3_root, "bout_00004", 778)
    bout_dir, start_frame = _resolve_session_bout(
        tmp_path, sam3_root, "Session1/2026_04_02_16_21_32_fly1", clip_len=777)
    assert bout_dir == "bout_00004"
    assert start_frame == 380781


def test_resolve_session_bout_tolerates_recording_without_fly_suffix(tmp_path):
    """A caller passing an already-suffix-free recording id must still
    resolve -- rsplit("_fly", 1)[0] on a string with no "_fly" leaves it
    unchanged, so this is the same code path, not a special case."""
    _write_summary_csv(tmp_path, [
        ("Session1/2026_04_02_16_21_32", 5, 380781, 381558),
    ])
    sam3_root = tmp_path / "sam3"
    _write_fake_sam3_npz(sam3_root, "bout_00004", 778)
    bout_dir, start_frame = _resolve_session_bout(
        tmp_path, sam3_root, "Session1/2026_04_02_16_21_32", clip_len=777)
    assert bout_dir == "bout_00004"
    assert start_frame == 380781


def test_resolve_session_bout_dir_numbering_is_not_positionally_related_to_bout_idx(tmp_path):
    """Round-5 regression: this IS the real permutation. CSV bout_idx 5
    (n=778) is the exemplar, but its masks live in bout_00004 while
    bout_00005 holds an unrelated bout (CSV bout_idx 6, n=569). A
    directory-name-from-bout_idx rule would resolve the WRONG directory
    and the video strip would later IndexError against its shorter N —
    exactly what happened on the real gate run."""
    _write_summary_csv(tmp_path, [
        ("Session1/2026_04_02_16_21_32", 1, 248710, 249361),
        ("Session1/2026_04_02_16_21_32", 2, 249564, 249823),
        ("Session1/2026_04_02_16_21_32", 3, 249816, 250102),
        ("Session1/2026_04_02_16_21_32", 4, 344050, 346068),
        ("Session1/2026_04_02_16_21_32", 5, 380781, 381558),   # n=778, the exemplar
        ("Session1/2026_04_02_16_21_32", 6, 382633, 383201),   # n=569
    ])
    sam3_root = tmp_path / "sam3"
    _write_fake_sam3_npz(sam3_root, "bout_00001", 652)
    _write_fake_sam3_npz(sam3_root, "bout_00002", 287)
    _write_fake_sam3_npz(sam3_root, "bout_00003", 2019)
    _write_fake_sam3_npz(sam3_root, "bout_00004", 778)   # the exemplar's real masks
    _write_fake_sam3_npz(sam3_root, "bout_00005", 569)   # CSV bout_idx 6's masks
    bout_dir, start_frame = _resolve_session_bout(
        tmp_path, sam3_root, "Session1/2026_04_02_16_21_32_fly1", clip_len=777)
    assert bout_dir == "bout_00004"
    assert start_frame == 380781


def test_resolve_session_bout_zero_mask_matches_raises_listing_available(tmp_path):
    _write_summary_csv(tmp_path, [
        ("Session1/2026_04_02_16_21_32", 5, 380781, 381558),   # n=778
    ])
    sam3_root = tmp_path / "sam3"
    _write_fake_sam3_npz(sam3_root, "bout_00001", 652)
    _write_fake_sam3_npz(sam3_root, "bout_00002", 287)
    with pytest.raises(ValueError) as excinfo:
        _resolve_session_bout(
            tmp_path, sam3_root, "Session1/2026_04_02_16_21_32", clip_len=778)
    msg = str(excinfo.value)
    assert "778" in msg
    assert "bout_00001" in msg and "652" in msg
    assert "bout_00002" in msg and "287" in msg


def test_resolve_session_bout_ambiguous_mask_match_lists_both_dirs(tmp_path):
    """Two sam3 dirs with the same mask frame count must raise, never
    silently pick one — this is the sam3-side counterpart of the CSV
    ambiguous-match case."""
    _write_summary_csv(tmp_path, [
        ("Session1/2026_04_02_16_21_32", 5, 380781, 381558),   # n=778
    ])
    sam3_root = tmp_path / "sam3"
    _write_fake_sam3_npz(sam3_root, "bout_00004", 778)
    _write_fake_sam3_npz(sam3_root, "bout_00006", 778)
    with pytest.raises(ValueError) as excinfo:
        _resolve_session_bout(
            tmp_path, sam3_root, "Session1/2026_04_02_16_21_32", clip_len=778)
    msg = str(excinfo.value)
    assert "bout_00004" in msg
    assert "bout_00006" in msg


def test_resolve_session_bout_never_reads_packed(tmp_path):
    """The real `packed` array is (2, 7, N, 448, 242) and decompressing all
    bout dirs' worth would be very slow — the helper must resolve correctly
    from an npz whose `packed` entry is deliberately ABSENT."""
    _write_summary_csv(tmp_path, [
        ("Session1/2026_04_02_16_21_32", 5, 380781, 381558),   # n=778
    ])
    sam3_root = tmp_path / "sam3"
    _write_fake_sam3_npz(sam3_root, "bout_00004", 778, include_packed=False)
    with np.load(sam3_root / "bout_00004" / "sam3_masks.npz") as z:
        assert "packed" not in z.files   # sanity: the fixture really omits it
    bout_dir, start_frame = _resolve_session_bout(
        tmp_path, sam3_root, "Session1/2026_04_02_16_21_32", clip_len=778)
    assert bout_dir == "bout_00004"
    assert start_frame == 380781


def test_load_kp3d_returns_the_array(tmp_path):
    """Round 6: the video/pitch blocks load kp3d from the processed pose
    tree (pose/bouts/<bout>/fly{0,1}/kp3d.npz) instead of the combined h5's
    re-centred kp_data."""
    rng = np.random.default_rng(7)
    arr = rng.normal(size=(778, 50, 3))
    npz_path = tmp_path / "kp3d.npz"
    np.savez(npz_path, kp3d=arr, conf3d=rng.normal(size=(778, 50)))
    out = _load_kp3d(npz_path)
    assert out.shape == (778, 50, 3)
    np.testing.assert_array_equal(out, arr)


def test_load_kp3d_raises_a_clear_error_when_key_missing(tmp_path):
    npz_path = tmp_path / "kp3d.npz"
    np.savez(npz_path, not_kp3d=np.zeros((778, 50, 3)))
    with pytest.raises(KeyError, match="kp3d"):
        _load_kp3d(npz_path)


def test_processed_fly_dir_extracts_the_fly_suffix():
    assert _processed_fly_dir("Session1/2026_04_02_16_21_32_fly1") == "fly1"
    assert _processed_fly_dir("Session1/2026_04_02_16_21_32_fly0") == "fly0"


def test_processed_fly_dir_raises_on_missing_suffix():
    with pytest.raises(ValueError, match="fly_id"):
        _processed_fly_dir("Session1/2026_04_02_16_21_32")


def test_processed_fly_dir_resolves_the_real_inverted_case():
    """Round 7 regression: key0/key1's processed fly0/fly1 directory is NOT
    positionally fixed. Verified on the real exemplar: key0=bout_183 has
    fly_id '..._fly1', key1=bout_182 has fly_id '..._fly0' -- INVERTED
    relative to a naive key0->fly0 assumption. The directory must be
    derived from the fly_id suffix, never assumed from pair position."""
    key0_fly_id = "Session1/2026_04_02_16_21_32_fly1"
    key1_fly_id = "Session1/2026_04_02_16_21_32_fly0"
    assert _processed_fly_dir(key0_fly_id) == "fly1"
    assert _processed_fly_dir(key1_fly_id) == "fly0"


def _write_free_running_h5(tmp_path, bouts, kp_names):
    """bouts: dict bout_key -> (T, n_kp, 3) kp_data array."""
    from utils.io_dict_to_hdf5 import save as h5_save
    d = {"info": {"kp_names": kp_names}}
    for k, kp in bouts.items():
        d[k] = {"kp_data": kp}
    path = tmp_path / "free_running.h5"
    h5_save(str(path), d)
    return path


def test_free_running_com_z_subtracts_the_floor(tmp_path):
    """Round 9 finding: the estimator must subtract each bout's OWN floor
    (5th percentile of ground keypoints), not return raw z -- unambiguous
    by construction: a constant C above a constant ground level G must
    return ~C, not ~(G+C)."""
    kp_names = ["Scutellum", "Tarsus1", "Tarsus2"]
    T = 50
    G, C = 2.5, 0.3
    kp = np.zeros((T, 3, 3))
    kp[:, 0, 2] = G + C   # Scutellum
    kp[:, 1, 2] = G       # ground kp 1 (matches "Tarsus*")
    kp[:, 2, 2] = G       # ground kp 2
    h5_path = _write_free_running_h5(tmp_path, {"bout_0000": kp}, kp_names)

    out = _free_running_com_z(h5_path)
    assert out.shape == (1,)
    assert out[0] == pytest.approx(C, abs=1e-6)


def test_free_running_com_z_skips_all_nan_scutellum(tmp_path):
    """A bout whose Scutellum z is entirely NaN contributes nothing, rather
    than polluting the result with a NaN mean."""
    kp_names = ["Scutellum", "Tarsus1", "Tarsus2"]
    T = 20
    kp_good = np.zeros((T, 3, 3))
    kp_good[:, 0, 2] = 2.0 + 0.2
    kp_good[:, 1, 2] = 2.0
    kp_good[:, 2, 2] = 2.0
    kp_bad = np.zeros((T, 3, 3))
    kp_bad[:, 0, 2] = np.nan
    kp_bad[:, 1, 2] = 1.5
    kp_bad[:, 2, 2] = 1.5
    h5_path = _write_free_running_h5(
        tmp_path, {"bout_0000": kp_good, "bout_0001": kp_bad}, kp_names)

    out = _free_running_com_z(h5_path)
    assert out.shape == (1,)
    assert out[0] == pytest.approx(0.2, abs=1e-6)


def test_free_running_com_z_returns_one_entry_per_contributing_bout(tmp_path):
    kp_names = ["Scutellum", "Tarsus1", "Tarsus2"]
    T = 10

    def _mk(offset):
        kp = np.zeros((T, 3, 3))
        kp[:, 0, 2] = 3.0 + offset
        kp[:, 1, 2] = 3.0
        kp[:, 2, 2] = 3.0
        return kp

    h5_path = _write_free_running_h5(tmp_path, {
        "bout_0000": _mk(0.1), "bout_0001": _mk(0.4), "bout_0002": _mk(0.7),
    }, kp_names)

    out = _free_running_com_z(h5_path)
    assert out.ndim == 1
    assert out.shape == (3,)
    np.testing.assert_allclose(sorted(out.tolist()), [0.1, 0.4, 0.7], atol=1e-6)


def test_resolve_sam3_camera_index_from_npz_cameras():
    """Round 10 finding: the exemplar's own npz `cameras` array is NOT
    glob-sorted order -- Cam2012630 sits at index 5, not the glob-sorted
    index 0."""
    cameras = np.array(["Cam2012853", "Cam2012862", "Cam2012855", "Cam2012857",
                        "Cam2012861", "Cam2012630", "Cam2012631"])
    idx, source = _resolve_sam3_camera_index(cameras, calib_dir=None, cam="Cam2012630")
    assert idx == 5
    assert source == "npz cameras"


def test_resolve_sam3_camera_index_raises_for_absent_camera():
    cameras = np.array(["Cam2012853", "Cam2012862"])
    with pytest.raises(ValueError, match="Cam9999999"):
        _resolve_sam3_camera_index(cameras, calib_dir=None, cam="Cam9999999")


def test_resolve_sam3_camera_index_falls_back_to_glob_order(tmp_path):
    """No `cameras` array (older npz convention, e.g. the Video_recordings
    tree's Predictions_3D_* mask files) -> the legacy
    sorted(glob('Cam*_dlt.csv')) order."""
    for name in ["Cam2012630_dlt.csv", "Cam2012631_dlt.csv", "Cam2012853_dlt.csv"]:
        (tmp_path / name).write_text("")
    idx, source = _resolve_sam3_camera_index(None, tmp_path, cam="Cam2012630")
    assert idx == 0   # sorted glob order: 2012630 < 2012631 < 2012853
    assert source == "glob fallback"


def test_resolve_male_female_slots_from_real_sex_meta_json():
    """Round 10 finding: slots must be derived from sex_meta (a mask-area
    VOTE), not assumed as a hardcoded [1, 0]. (Correction, round 11 finding
    C: an earlier draft of this docstring wrongly claimed pose/bouts/*/
    sex.json said male_fly=0 for this same bout -- it does not; the real
    file says male_fly=1, AGREEING with sex_meta's male_slot=1. That was a
    false comment about a real file, corrected here -- see also
    test_resolve_male_female_slots_agrees_with_human_confirmed_sex_json
    below for the two-source cross-check this bout's REAL data exercises.)
    """
    sex_meta = ('{"male_slot": 1, "status": "kept", "method": "mask_area_vote", '
               '"male_detected_slot": 1, "agreement": 0.429, "margin": 0.038, '
               '"n_cameras": 7, "pct": 75}')
    assert _resolve_male_female_slots(sex_meta) == (1, 0)


def test_resolve_male_female_slots_when_male_slot_is_zero():
    sex_meta = '{"male_slot": 0, "status": "kept"}'
    assert _resolve_male_female_slots(sex_meta) == (0, 1)


def test_resolve_male_female_slots_falls_back_to_default_when_absent():
    assert _resolve_male_female_slots(None) == (1, 0)


def _sync_plan_with_gap(cam_name="Cam2012630", gaps=None, status="reindex"):
    """Build a `jarvis_jax.predict.frame_sync.SyncPlan` the same way the
    vendored `third_party/jarvis_jax/tests/test_synced_reader.py` does, so
    this exercises the SAME canonical mapping that module is tested
    against, not a reimplementation."""
    from jarvis_jax.predict import frame_sync as fs
    gaps = gaps or []
    total_lost = sum(g["lost"] for g in gaps)
    d = dict(delta_ns=1250000, canonical_len=1000, predict_start=0, predict_len=1000,
             status=status, first_drop_slot=(gaps[0]["slot"] if gaps else None),
             cameras={cam_name: dict(start_slot=0, decoded_len=1000 - total_lost,
                                     true_span=1000, frame_id_mode="hole", gaps=gaps)})
    return fs.SyncPlan(d)


def test_slot_to_raw_frame_before_any_gap_is_identity():
    """Round 11 finding A: a slot before any gap needs no correction."""
    plan = _sync_plan_with_gap(gaps=[{"slot": 21, "lost": 26}])
    raw, corrected = _slot_to_raw_frame(plan, "Cam2012630", 10)
    assert raw == 10
    assert corrected is True   # the PLAN applies (status=reindex); this slot just has 0 lost-before


def test_slot_to_raw_frame_past_a_gap_subtracts_frames_lost():
    """Round 11 finding A regression: this is the REAL recording's own gap
    (Cam2012630: gaps=[{"slot": 21, "lost": 26}]). Reusing the repo's own
    canonical mapping (`jarvis_jax.predict.frame_sync.SyncCam.pos`, the same
    engine `third_party/jarvis_jax/tests/test_synced_reader.py::
    test_slot_positions_maps_around_gap` already asserts `pos = slot - lost`
    against) gives slot 100 -> raw position 74, NOT 126 -- confirmed against
    the REAL sync_plan.json in this session's sanity check (canonical slot
    380781 -> raw position 380755, a SUBTRACTION of 26, matching this exact
    arithmetic direction)."""
    plan = _sync_plan_with_gap(gaps=[{"slot": 21, "lost": 26}])
    raw, corrected = _slot_to_raw_frame(plan, "Cam2012630", 100)
    assert raw == 74
    assert corrected is True


def test_slot_to_raw_frame_multiple_gaps_accumulate():
    plan = _sync_plan_with_gap(gaps=[{"slot": 21, "lost": 26}, {"slot": 500, "lost": 5}])
    raw, corrected = _slot_to_raw_frame(plan, "Cam2012630", 700)
    assert raw == 700 - 26 - 5
    assert corrected is True


def test_slot_to_raw_frame_absent_plan_is_identity():
    raw, corrected = _slot_to_raw_frame(None, "Cam2012630", 380781)
    assert raw == 380781
    assert corrected is False


def test_slot_to_raw_frame_non_reindex_status_is_identity():
    """A plan whose status is NOT 'reindex' (e.g. 'clean'/'trim') must fall
    through positionally even if it happens to carry gaps."""
    plan = _sync_plan_with_gap(gaps=[{"slot": 21, "lost": 26}], status="clean")
    raw, corrected = _slot_to_raw_frame(plan, "Cam2012630", 100)
    assert raw == 100
    assert corrected is False


def test_resolve_male_female_slots_agrees_with_human_confirmed_sex_json():
    """Round 11 finding B: this IS the real exemplar's data -- sex.json
    (human-confirmed) and sex_meta (mask-area vote) AGREE (both male=1);
    no skip should fire (no exception)."""
    sex_meta = '{"male_slot": 1, "status": "kept", "method": "mask_area_vote"}'
    sex_json = {"male_fly": 1, "original_male_fly": 1, "applied_swap": False,
               "confidence": "user", "method": "manual-gui",
               "note": "fly_id_review ... status=confirmed"}
    assert _resolve_male_female_slots(sex_meta, sex_json) == (1, 0)


def test_resolve_male_female_slots_disagreement_raises_naming_both():
    """A human-confirmed sex.json disagreeing with sex_meta must not be
    silently resolved either way -- target_pitch could be a degenerate
    male->male vector."""
    sex_meta = '{"male_slot": 0, "status": "kept", "method": "mask_area_vote"}'
    sex_json = {"male_fly": 1, "confidence": "user", "method": "manual-gui"}
    with pytest.raises(ValueError) as excinfo:
        _resolve_male_female_slots(sex_meta, sex_json)
    msg = str(excinfo.value)
    assert "1" in msg and "0" in msg


def test_resolve_male_female_slots_ignores_non_human_sex_json():
    """A sex.json present but NOT human-confirmed (confidence != 'user',
    method not manual) is not authoritative -- falls back to sex_meta
    alone, so a "disagreement" with it must NOT raise."""
    sex_meta = '{"male_slot": 0, "status": "kept", "method": "mask_area_vote"}'
    sex_json = {"male_fly": 1, "confidence": "auto", "method": "heuristic"}
    assert _resolve_male_female_slots(sex_meta, sex_json) == (0, 1)


def test_resolve_male_female_slots_uses_human_sex_json_alone():
    """sex.json alone (no sex_meta) still resolves via the human value."""
    sex_json = {"male_fly": 1, "confidence": "user", "method": "manual-gui"}
    assert _resolve_male_female_slots(None, sex_json) == (1, 0)


def test_load_kp3d_asserts_expected_keypoint_count(tmp_path):
    """Round 11 finding D: an unreordered/wrong-count kp3d array must be
    caught, not silently substitute a nearby keypoint."""
    npz_path = tmp_path / "kp3d.npz"
    np.savez(npz_path, kp3d=np.zeros((10, 40, 3)), conf3d=np.zeros((10, 40)))
    with pytest.raises(AssertionError, match="40"):
        _load_kp3d(npz_path, expected_n_kp=50)


def test_load_kp3d_passes_with_matching_keypoint_count(tmp_path):
    npz_path = tmp_path / "kp3d.npz"
    np.savez(npz_path, kp3d=np.zeros((10, 50, 3)), conf3d=np.zeros((10, 50)))
    out = _load_kp3d(npz_path, expected_n_kp=50)
    assert out.shape == (10, 50, 3)
