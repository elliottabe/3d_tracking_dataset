import numpy as np
import pytest
from jarvis_jax.predict import frame_sync as fs
from jarvis_jax.predict import synced_reader as sr

cv2 = pytest.importorskip("cv2")


def _make_video(path, n, h=16, w=16):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), (i * 7) % 256, np.uint8))
    vw.release()


def _plan_with_gap(cam_name="Cam2012000", start_slot=0, decoded=197, gap_slot=41, lost=3):
    d = dict(delta_ns=1250000, canonical_len=200, predict_start=0, predict_len=200,
             status="reindex", first_drop_slot=gap_slot,
             cameras={cam_name: dict(start_slot=start_slot, decoded_len=decoded,
                                     true_span=decoded + lost, frame_id_mode="hole",
                                     gaps=[dict(slot=gap_slot, lost=lost)])})
    return fs.SyncPlan(d)


def test_slot_positions_none_plan_is_positional():
    pos, present = sr.slot_positions(None, "Cam2012000", start_slot=100, T=5)
    assert pos == [100, 101, 102, 103, 104]
    assert present == [True] * 5


def test_slot_positions_maps_around_gap():
    plan = _plan_with_gap()
    pos, present = sr.slot_positions(plan, "Cam2012000", start_slot=39, T=8)
    # slots 39,40 present (pos==slot); 41,42,43 dropped; 44,45,46 present (pos = slot-3)
    assert present == [True, True, False, False, False, True, True, True]
    assert pos[0] == 39 and pos[1] == 40
    assert pos[2] is None and pos[3] is None and pos[4] is None
    assert pos[5] == 41 and pos[6] == 42 and pos[7] == 43


def test_read_window_positional_shapes_and_rgb(tmp_path):
    for c in ["CamA", "CamB"]:
        _make_video(str(tmp_path / f"{c}.mp4"), 20)
    out = list(sr.read_window(str(tmp_path), ["CamA", "CamB"], None, 2, 5))
    assert len(out) == 5
    frames, present = out[0]
    assert frames.shape == (2, 16, 16, 3) and frames.dtype == np.uint8
    assert present.tolist() == [True, True]


def test_read_window_absent_slot_is_black_and_not_present(tmp_path):
    for c in ["CamA", "CamB"]:
        _make_video(str(tmp_path / f"{c}.mp4"), 20)
    # CamA drops 2 frames at slot 2 -> slots 2,3 absent for CamA
    plan = fs.SyncPlan(dict(delta_ns=1, canonical_len=20, predict_start=0, predict_len=20,
        status="reindex", first_drop_slot=2,
        cameras={"CamA": dict(start_slot=0, decoded_len=18, true_span=20, frame_id_mode="hole",
                              gaps=[dict(slot=2, lost=2)]),
                 "CamB": dict(start_slot=0, decoded_len=20, true_span=20, frame_id_mode="reindex", gaps=[])}))
    out = list(sr.read_window(str(tmp_path), ["CamA", "CamB"], plan, 0, 5))
    _f2, p2 = out[2]
    assert p2.tolist() == [False, True]           # CamA absent at slot 2
    assert int(_f2[0].max()) == 0                  # CamA frame is black
    assert int(_f2[1].max()) >= 0                  # CamB present


def test_read_window_bad_session_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(sr.read_window(str(tmp_path / "nope"), ["CamA"], None, 0, 3))
