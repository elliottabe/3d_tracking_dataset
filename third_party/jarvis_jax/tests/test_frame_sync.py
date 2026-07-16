import os
import numpy as np
from jarvis_jax.predict import frame_sync as fs


def _write_meta(path, frame_ids, ts):
    with open(path, "w") as f:
        f.write("frame_id,timestamp,timestamp_sys,ptp_offset\n")
        for fid, t in zip(frame_ids, ts):
            f.write(f"{fid},{t},{t},0\n")


def _synthetic_recording(tmp, drop_at=None, lost=0, delta=1250000, n=200, cams=3):
    """cams identical clean timelines unless drop_at set (one gap of `lost` frames)."""
    for c in range(cams):
        fids, ts = [], []
        t = 1000
        for i in range(n):
            fids.append(i); ts.append(t)
            step = delta * (1 + lost) if (drop_at is not None and i == drop_at) else delta
            t += step
        _write_meta(os.path.join(tmp, f"Cam201200{c}_meta.csv"), fids, ts)


def test_clean_recording_is_clean(tmp_path):
    _synthetic_recording(str(tmp_path))
    plan = fs.analyze_recording(str(tmp_path))
    assert plan["status"] == "clean"
    p = fs.SyncPlan(plan)
    cam = next(iter(p.cams.values()))
    # clean => pos(slot) == slot, present everywhere
    assert cam.has(50) and cam.pos(50) == 50
    assert cam.has(199)


def test_interior_drop_is_reindex_and_pos_shifts(tmp_path):
    # one camera drops 3 frames after slot 40
    _synthetic_recording(str(tmp_path), cams=3)
    # overwrite cam 0 with a 3-frame gap after index 40
    fids = list(range(200)); ts = []
    t = 1000
    for i in range(200):
        ts.append(t); t += 1250000 * (4 if i == 40 else 1)
    _write_meta(os.path.join(str(tmp_path), "Cam2012000_meta.csv"), fids, ts)
    plan = fs.analyze_recording(str(tmp_path))
    assert plan["status"] == "reindex"
    p = fs.SyncPlan(plan)
    cam0 = p.cams["Cam2012000"]
    # before the gap: pos == slot; a slot inside the gap: absent; after: pos = slot - lost
    assert cam0.has(30) and cam0.pos(30) == 30
    assert not cam0.has(42)                      # inside the 3-frame gap at slot 41..43
    assert cam0.has(60) and cam0.pos(60) == 60 - 3


def test_sync_inventory_classifies_tree(tmp_path):
    r1 = tmp_path / "recA"; r1.mkdir(); _synthetic_recording(str(r1))
    rows = fs.sync_inventory(str(tmp_path))
    assert any(row["status"] == "clean" for row in rows)
