# tests/test_coarse_figure_gates.py
"""`scripts/viz/coarse_centres_check.py` / `scripts/viz/coarse_tracks_check.py`
-- figure gates 1-2 for the mask-free coarse pass (P4b task 6, spec §8.1/§8.2).

CPU-fast smoke test, no GPU/JAX model, no real video: builds a SYNTHETIC
`coarse_tracks.npz` (`jarvis_jax.tracking.coarse_track.write_coarse_tracks`,
the same helper `tests/test_coarse_track.py` exercises) with a hand-built,
KNOWN close/wing-extension episode, and runs each script's own `main()`
(matplotlib Agg) end to end.

Expectations these tests encode (CLAUDE.md: state the expectation, check a
real invariant):

  * `coarse_tracks_check.py` must find the reviewed bout that DOES coincide
    with the close/wing-extension episode explained (not in
    `unexplained_reviewed_bouts`) and the one that does NOT (flat signal
    throughout) reported BY BOUT ID -- a script that silently drops or
    averages away an unsupported reviewed bout would defeat the whole point
    of this gate.
  * `coarse_centres_check.py` must pick frames from all three named
    categories (inside a reviewed bout, outside, at a boundary) rather than
    silently returning fewer than requested, and (with `--video-dir`
    omitted, no GPU/real video needed here) must say plainly in its own JSON
    that a panel has no real pixels under it.
"""
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO / "third_party" / "jarvis_jax"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mvq_fixtures import CAMS, cam_P  # noqa: E402

from jarvis_jax.tracking.coarse_track import (  # noqa: E402
    FloorPlane, coarse_features, write_coarse_tracks)

KP_NAMES = ["Scutellum", "Abd_tip", "WingL_base", "WingL_V13", "WingR_base", "WingR_V13"]
STRIDE = 16
T = 200
CLOSE_S, CLOSE_E = 70, 100          # [CLOSE_S, CLOSE_E) coarse indices -- 30 frames


def _load_script(name):
    """`scripts/viz/<name>.py` loaded straight from its file (it is not a
    package member), the same idiom `tests/test_viz_compare_args.py` uses for
    a `scripts/`-only module."""
    spec = importlib.util.spec_from_file_location(
        name, str(REPO / "scripts" / "viz" / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_bouts_csv(path, rows, fly_id="test_session"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fly_id", "bout_idx", "start_frame", "end_frame"])
        for bidx, s, e in rows:
            w.writerow([fly_id, bidx, s, e])


def _build_tracks(tmp_path):
    """Two flies: far apart / wing folded EXCEPT coarse indices
    [CLOSE_S, CLOSE_E), where they are close (15 units = 1.5mm) and the
    male's (fly row 1) wing is at a KNOWN 45 deg (`WingR` held at a fixed 0
    deg the whole time, so `np.fmax` over L/R in `coarse_features` reads
    `WingL`'s angle exactly)."""
    coarse_frame = (np.arange(T) * STRIDE).astype(np.int64)
    close_mask = np.zeros(T, bool)
    close_mask[CLOSE_S:CLOSE_E] = True

    sep = np.where(close_mask, 15.0, 60.0)
    centroid = np.zeros((2, T, 3), np.float32)
    centroid[0, :, 0] = 0.0
    centroid[1, :, 0] = sep

    male_wing_deg = np.where(close_mask, 45.0, 5.0)
    K = len(KP_NAMES)
    kp3d = np.full((2, T, K, 3), np.nan, np.float32)
    idx = {n: i for i, n in enumerate(KP_NAMES)}
    for f in range(2):
        kp3d[f, :, idx["Scutellum"]] = centroid[f]
        kp3d[f, :, idx["Abd_tip"]] = centroid[f] + np.array([-10.0, 0.0, 0.0], np.float32)
        kp3d[f, :, idx["WingR_base"]] = centroid[f]
        kp3d[f, :, idx["WingR_V13"]] = centroid[f] + np.array([-8.0, 0.0, 0.0], np.float32)
        kp3d[f, :, idx["WingL_base"]] = centroid[f]
    ang0 = np.deg2rad(np.full(T, 5.0))              # female: always low
    ang1 = np.deg2rad(male_wing_deg)                 # male: KNOWN close-episode angle
    for f, ang in ((0, ang0), (1, ang1)):
        wing_vec = 8.0 * np.stack([-np.cos(ang), np.sin(ang), np.zeros(T)], axis=-1)
        kp3d[f, :, idx["WingL_V13"]] = centroid[f] + wing_vec.astype(np.float32)

    exist = np.full((2, T), 0.9, np.float32)
    slot = np.stack([np.ones(T, np.int8), np.full(T, 2, np.int8)])
    sex_prob = np.stack([np.full(T, 0.95, np.float32), np.full(T, 0.05, np.float32)])
    centre_source = np.zeros((2, T), np.int8)
    n_windows = np.ones(T, np.int8)

    tracks = {"frame": coarse_frame, "kp3d": kp3d, "centroid": centroid, "exist": exist,
             "sex_prob": sex_prob, "slot": slot, "centre_source": centre_source,
             "n_windows": n_windows, "kp_names": KP_NAMES, "W": 1936, "H": 448}
    floor = FloorPlane(np.array([0.0, 0.0, 1.0]), 0.0)
    features = coarse_features(tracks, KP_NAMES, floor=floor)

    # Check the KNOWN answer before writing (CLAUDE.md: check a real invariant,
    # not just "the file wrote without crashing").
    assert np.allclose(features["wing_angle_deg"][1][CLOSE_S:CLOSE_E], 45.0, atol=1e-3)
    assert np.allclose(features["wing_angle_deg"][1][:CLOSE_S], 5.0, atol=1e-3)
    assert np.allclose(features["dist"][CLOSE_S:CLOSE_E], 15.0, atol=1e-3)

    cam_mats = np.stack([cam_P(i).T for i in range(len(CAMS))]).astype(np.float32)
    out = str(tmp_path / "coarse_tracks.npz")
    write_coarse_tracks(out, tracks, features, CAMS, session_dir=str(tmp_path),
                       stride=STRIDE, num_animals=2, cam_mats=cam_mats)
    return out, coarse_frame


def test_coarse_tracks_check_finds_the_supported_bout_and_flags_the_unsupported_one(tmp_path):
    tracks_path, coarse_frame = _build_tracks(tmp_path)
    supported = (1, int(coarse_frame[CLOSE_S]), int(coarse_frame[CLOSE_E - 1]))
    unsupported = (2, int(coarse_frame[10]), int(coarse_frame[20]))   # flat/far the whole time
    reviewed_csv = tmp_path / "reviewed.csv"
    _write_bouts_csv(reviewed_csv, [supported, unsupported])
    gate_rows = [(1, int(coarse_frame[CLOSE_S + 1]), int(coarse_frame[CLOSE_E - 2]))]
    gate_csv = tmp_path / "gate.csv"
    _write_bouts_csv(gate_csv, gate_rows)

    mod = _load_script("coarse_tracks_check")
    out_dir = tmp_path / "out_tracks"
    png, js = mod.main(["--tracks", tracks_path, "--reviewed-bouts", str(reviewed_csv),
                       "--gate-bouts", str(gate_csv), "--out-dir", str(out_dir),
                       "--zoom-bout-idx", "1"])
    assert os.path.isfile(png)
    assert os.path.isfile(js)
    data = json.load(open(js))
    assert len(data["reviewed"]) == 2
    assert len(data["gate"]) == 1
    assert data["zoom"]["bout_idx"] == 1 and not data["zoom"]["is_fallback"]
    unexplained_idx = [u["bout_idx"] for u in data["unexplained_reviewed_bouts"]]
    assert unexplained_idx == [2]


def test_coarse_tracks_check_falls_back_when_bout_28_absent(tmp_path):
    tracks_path, coarse_frame = _build_tracks(tmp_path)
    only_bout = (7, int(coarse_frame[CLOSE_S]), int(coarse_frame[CLOSE_E - 1]))
    reviewed_csv = tmp_path / "reviewed.csv"
    _write_bouts_csv(reviewed_csv, [only_bout])

    mod = _load_script("coarse_tracks_check")
    out_dir = tmp_path / "out_tracks_fallback"
    _png, js = mod.main(["--tracks", tracks_path, "--reviewed-bouts", str(reviewed_csv),
                        "--out-dir", str(out_dir)])
    data = json.load(open(js))
    assert data["zoom"]["bout_idx"] == 7 and data["zoom"]["is_fallback"] is True


def test_coarse_centres_check_picks_all_three_categories_without_video(tmp_path):
    tracks_path, coarse_frame = _build_tracks(tmp_path)
    reviewed = [(1, int(coarse_frame[CLOSE_S]), int(coarse_frame[CLOSE_E - 1]))]
    reviewed_csv = tmp_path / "reviewed.csv"
    _write_bouts_csv(reviewed_csv, reviewed)

    mod = _load_script("coarse_centres_check")
    out_dir = tmp_path / "out_centres"
    png, js = mod.main(["--tracks", tracks_path, "--reviewed-bouts", str(reviewed_csv),
                       "--out-dir", str(out_dir)])
    assert os.path.isfile(png)
    assert os.path.isfile(js)
    data = json.load(open(js))
    cats = {r["category"] for r in data["frames"]}
    assert {"inside", "outside", "boundary"} <= cats
    for row in data["frames"]:
        for cam_rec in row["cameras"].values():
            assert cam_rec["had_real_pixels"] is False   # --video-dir omitted


def test_coarse_centres_check_frames_override(tmp_path):
    tracks_path, coarse_frame = _build_tracks(tmp_path)
    reviewed_csv = tmp_path / "reviewed.csv"
    _write_bouts_csv(reviewed_csv, [(1, int(coarse_frame[CLOSE_S]), int(coarse_frame[CLOSE_E - 1]))])

    mod = _load_script("coarse_centres_check")
    out_dir = tmp_path / "out_centres_override"
    want = [int(coarse_frame[0]), int(coarse_frame[CLOSE_S + 2])]
    _png, js = mod.main(["--tracks", tracks_path, "--reviewed-bouts", str(reviewed_csv),
                        "--out-dir", str(out_dir), "--frames", ",".join(str(f) for f in want)])
    data = json.load(open(js))
    assert [r["frame"] for r in data["frames"]] == want
