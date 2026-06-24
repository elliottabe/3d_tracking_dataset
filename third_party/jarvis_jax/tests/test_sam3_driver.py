import os, json
import numpy as np
import pytest
from jarvis_jax.predict.sam3_driver import (
    session_tag_for, parse_bouts, video_paths_for, bout_stats, build_manifest,
)


def test_session_tag_for():
    d = "/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04"
    assert session_tag_for(d) == "Session0/2025_10_20_13_20_04"
    assert session_tag_for(d + "/") == "Session0/2025_10_20_13_20_04"


def test_parse_bouts_filters_and_limits(tmp_path):
    csv = tmp_path / "b.csv"
    csv.write_text(
        "fly_id,bout_idx,start_frame,end_frame\n"
        "Session0/rec,1,100,200\n"
        "Session0/rec,2,300,350\n"
        "OtherSession/rec,3,0,10\n")
    rows = parse_bouts(str(csv), "Session0/rec")
    assert [r["bout_idx"] for r in rows] == [1, 2]
    assert rows[0] == {"bout_idx": 1, "start": 100, "end": 200, "n": 101}
    # limit
    assert [r["bout_idx"] for r in parse_bouts(str(csv), "Session0/rec", limit=1)] == [1]
    # explicit ids
    assert [r["bout_idx"] for r in parse_bouts(str(csv), "Session0/rec", bout_ids=[2])] == [2]


def test_video_paths_for_orders_by_camera(tmp_path):
    for cam in ("CamA", "CamB"):
        (tmp_path / f"{cam}.mp4").write_bytes(b"x")
    paths = video_paths_for(str(tmp_path), ["CamB", "CamA"])
    assert [os.path.basename(p) for p in paths] == ["CamB.mp4", "CamA.mp4"]
    with pytest.raises(FileNotFoundError):
        video_paths_for(str(tmp_path), ["CamMissing"])


class _FakeLoaded:
    # mimics LoadedBoutMasks surface used by bout_stats
    def __init__(self, valid, centroids):
        self.valid = valid                      # (A,C,F) bool
        self.centroids = centroids              # (A,C,F,2)
        self.num_animals_saved = valid.shape[0]
        self.num_cameras = valid.shape[1]
        self.num_frames = valid.shape[2]


def test_bout_stats():
    A, C, F = 2, 7, 10
    valid = np.zeros((A, C, F), bool); valid[:, :4, :] = True   # 4/7 cams valid
    cent = np.zeros((A, C, F, 2), np.float32)
    st = bout_stats(_FakeLoaded(valid, cent), num_animals=2)
    assert st["num_animals_saved"] == 2 and st["num_frames"] == 10
    assert abs(st["per_fly_valid_frac"][0] - 4 / 7) < 1e-6
    assert st["mean_cams_valid_per_frame"] == 4.0


def test_build_manifest_roundtrips(tmp_path):
    m = build_manifest("/s/dir", "S/rec", {"sam3_version": "sam3.1"},
                       [{"bout_idx": 1, "num_frames": 10}])
    assert m["session_tag"] == "S/rec" and m["n_bouts"] == 1
    assert m["sam3_settings"]["sam3_version"] == "sam3.1"
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    assert json.loads((tmp_path / "manifest.json").read_text())["n_bouts"] == 1
