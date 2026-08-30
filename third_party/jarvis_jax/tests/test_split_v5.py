import json

import pytest
from jarvis_jax.data.split_v5 import make_split, audit_split, write_derived

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

def test_write_derived_survives_none_ann_ids(tmp_path):
    """merge_annotations (build_v5.py, commit 89c1b08) legitimately emits
    ann_ids containing None for a camera whose fly identity could not be
    resolved. write_derived must drop those None entries rather than crash
    on sorted({int, None}) or by_id[None], and must NOT drop the frameset
    itself -- it still has >= MIN_CAMS resolvable views."""
    merged = {
        "keypoint_names": ["kp0"], "skeleton": [],
        "categories": [{"id": 1, "name": "fly"}],
        "images": [{"id": 0, "file_name": "rec_a/Cam1/Frame_000000.jpg"},
                   {"id": 1, "file_name": "rec_a/Cam2/Frame_000000.jpg"}],
        "annotations": [{"id": 0, "image_id": 0, "fly_id": 0},
                        {"id": 1, "image_id": 1, "fly_id": 0}],
        "framesets": {
            "rec_a/Frame_000000/fly0": {
                "recording": "rec_a", "fly_id": 0,
                "frames": [0, 1], "ann_ids": [0, None]},
        },
    }
    split = {"rec_a/Frame_000000/fly0": "train"}
    write_derived(merged, split, str(tmp_path))
    out = json.load(open(tmp_path / "annotations" / "instances_train.json"))
    assert "rec_a/Frame_000000/fly0" in out["framesets"], \
        "frameset with a None ann_id must survive, not be dropped"
    ann_ids_out = [a["id"] for a in out["annotations"]]
    assert ann_ids_out == [0]
    assert None not in ann_ids_out
