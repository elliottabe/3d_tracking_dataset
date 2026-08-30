import pytest
from jarvis_jax.data.split_v5 import make_split, audit_split

def _merged(recs):
    """recs: {recording: n_frames}. One fly per frameset, frames spaced by 1."""
    fs = {}
    for rec, n in recs.items():
        for i in range(n):
            fs[f"{rec}/Frame_{i:06d}/fly0"] = {"recording": rec, "fly_id": 0,
                                               "frames": [], "ann_ids": []}
    return {"framesets": fs}

def _manifest(recs, female=()):
    return {"recordings": {r: {"sex": "female" if r in female else "male"}
                           for r in recs}}

def test_whole_recording_holdout_has_zero_overlap():
    m = _merged({"rec_a": 100, "rec_b": 100})
    split = make_split(m, _manifest(["rec_a", "rec_b"]), val_recordings=["rec_b"])
    assert {k.split("/")[0] for k, v in split.items() if v == "val"} == {"rec_b"}
    assert {k.split("/")[0] for k, v in split.items() if v == "train"} == {"rec_a"}
    assert audit_split(m, split)["cross_recording_leaks"] == 0

def test_female_recording_stays_in_training_with_guarded_tail_val():
    """Female data is the binding constraint (199 framesets, 7%). A female
    recording must NOT be held out whole -- only a guarded tail block."""
    m = _merged({"fem": 200})
    split = make_split(m, _manifest(["fem"], female=["fem"]),
                       val_recordings=[], female_val_frac=0.10, guard=50)
    train = [k for k, v in split.items() if v == "train"]
    val = [k for k, v in split.items() if v == "val"]
    assert len(val) == 20 and len(train) > 0
    # val must be the TAIL, contiguous
    val_frames = sorted(int(k.split("Frame_")[1].split("/")[0]) for k in val)
    assert val_frames == list(range(180, 200))

def test_guard_band_separates_female_train_from_val():
    m = _merged({"fem": 200})
    split = make_split(m, _manifest(["fem"], female=["fem"]),
                       female_val_frac=0.10, guard=50, val_recordings=[])
    a = audit_split(m, split, guard=50)
    assert a["min_guard_distance"]["fem"] >= 50, a

def test_audit_detects_a_leaky_split():
    """Regression guard for the exact defect in the old data: adjacent frames
    split across train and val."""
    m = _merged({"rec_a": 100})
    bad = {k: ("val" if i % 10 == 0 else "train")
           for i, k in enumerate(sorted(m["framesets"]))}
    a = audit_split(m, bad, guard=50)
    assert a["min_guard_distance"]["rec_a"] < 50

def test_empty_val_is_rejected():
    m = _merged({"rec_a": 10})
    with pytest.raises(ValueError, match="empty val"):
        make_split(m, _manifest(["rec_a"]), val_recordings=[], female_val_frac=0.0)
