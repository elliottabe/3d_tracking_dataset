"""Raw-export bout matching must use (fly_id, start_frame) when available.

The legacy rule matched CSV rows / preproc bouts to combined bouts by
(fly_id, n_frames) — but preprocessing trims NaN edges, so the curated CSV
n_frames no longer equals the combined clip length for most bouts (real
NewBouts run: 413/2203 CSV rows matched, 1712/2203 orig_keypoints). All
three sources carry the ORIGINAL start frame (CSV `start_frame`, preproc
and combined `info/start_frames`), an exact unique key.
"""
from __future__ import annotations

import numpy as np
import pytest

from utils.free_walking_loader import export_raw_free_running_h5
from utils.io_dict_to_hdf5 import save as h5_save, load as h5_load

FID = "sessionX/2026_01_01_00_00_00"


def _fixture(tmp_path):
    # Combined h5: two bouts, SAME (trimmed) clip length -> n_frames is
    # ambiguous; distinct start_frames.
    combined = {
        "bout_000": {"kp_data": np.full((10, 6), 1.0), "qpos": np.zeros((10, 4))},
        "bout_001": {"kp_data": np.full((10, 6), 2.0), "qpos": np.zeros((10, 4))},
        "info": {
            "fly_ids": [FID, FID],
            "clip_lengths": np.array([10, 10]),
            "start_frames": np.array([100, 200]),
            "end_frames": np.array([114, 211]),
        },
    }
    ch5 = tmp_path / "combined.h5"
    h5_save(str(ch5), combined)

    # Curated CSV: n_frames reflect the UNtrimmed bouts (15, 12) -- neither
    # equals the combined clip length, so the legacy rule matches nothing.
    csv = tmp_path / "free_running_bouts_summary.csv"
    csv.write_text(
        "fly_id,bout_idx,start_frame,end_frame,n_frames\n"
        f"{FID},1,100,114,15\n"
        f"{FID},2,200,211,12\n")

    # Preproc h5: orig_keypoints with distinct values, original start_frames,
    # trimmed clip lengths.
    pre = {
        "bout_000": {"orig_keypoints": np.full((10, 2, 3), 1.0)},
        "bout_001": {"orig_keypoints": np.full((10, 2, 3), 2.0)},
        "info": {
            "fly_ids": [FID, FID],
            "clip_lengths": np.array([10, 10]),
            "start_frames": np.array([100, 200]),
        },
    }
    ph5 = tmp_path / "preprocessed_bout_v1_free_running.h5"
    h5_save(str(ph5), pre)
    return ch5, csv, ph5


def test_start_frame_matching_attaches_everything(tmp_path):
    ch5, csv, ph5 = _fixture(tmp_path)
    out = tmp_path / "raw.h5"
    summary = export_raw_free_running_h5(
        ch5, out, bout_summary_csvs=[csv], preproc_h5_paths=[ph5],
        verbose=False)
    assert summary["n_csv_matched"] == 2, summary
    assert summary["n_orig_attached"] == 2, summary
    d = h5_load(str(out))
    # attached to the RIGHT bouts (values follow start_frame identity)
    assert float(np.asarray(d["bout_000"]["orig_keypoints"]).ravel()[0]) == 1.0
    assert float(np.asarray(d["bout_001"]["orig_keypoints"]).ravel()[0]) == 2.0


def test_n_frames_fallback_still_works_without_start_frames(tmp_path):
    ch5, csv, ph5 = _fixture(tmp_path)
    # Strip start_frames from the combined info -> matcher must fall back.
    d = h5_load(str(ch5))
    del d["info"]["start_frames"]
    # make CSV n_frames match the legacy rule (clip_length + 1)
    csv.write_text(
        "fly_id,bout_idx,start_frame,end_frame,n_frames\n"
        f"{FID},1,100,114,11\n"
        f"{FID},2,200,211,11\n")
    ch5b = ch5.parent / "combined_nostart.h5"
    h5_save(str(ch5b), d)
    out = tmp_path / "raw_fallback.h5"
    summary = export_raw_free_running_h5(
        ch5b, out, bout_summary_csvs=[csv], preproc_h5_paths=[ph5],
        verbose=False)
    assert summary["n_csv_matched"] == 2, summary
    assert summary["n_orig_attached"] == 2, summary
