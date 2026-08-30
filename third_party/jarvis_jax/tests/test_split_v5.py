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

def _merged_mixed(rec, n):
    """Two flies per frameset in ONE recording, frames spaced by 1 -- the
    shape of the four recordings that label both flies."""
    fs = {}
    for i in range(n):
        for fly in (0, 1):
            fs[f"{rec}/Frame_{i:06d}/fly{fly}"] = {
                "recording": rec, "fly_id": fly, "frames": [], "ann_ids": []}
    return {"framesets": fs}

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

def test_mixed_recording_splits_each_fly_by_its_own_sex():
    """Four recordings label BOTH flies (fly_sex); a recording with one male
    + one female fly has no well-defined recording-level sex. make_split must
    resolve sex PER FRAMESET via the fly<k> suffix: fly1 (male) follows the
    normal whole-recording-holdout rule while fly0 (female) independently
    keeps the guarded-tail-in-training policy, even though both flies share
    the same recording (and the same val_recordings membership)."""
    m = _merged_mixed("mixed", 200)
    man = {"recordings": {"mixed": {"fly_sex": {"fly0": "female", "fly1": "male"}}}}
    split = make_split(m, man, val_recordings=["mixed"],
                       female_val_frac=0.10, guard=50)

    fly0 = {k: v for k, v in split.items() if k.endswith("/fly0")}
    fly1 = {k: v for k, v in split.items() if k.endswith("/fly1")}

    # fly1 (male): normal rule -- 'mixed' is in val_recordings -> held out WHOLE.
    assert set(fly1.values()) == {"val"}

    # fly0 (female): guarded-tail policy -- mostly train, small contiguous
    # tail block in val; NEVER held out whole despite 'mixed' being in
    # val_recordings (female data is the binding constraint).
    assert set(fly0.values()) == {"train", "val"}
    fly0_val = [k for k, v in fly0.items() if v == "val"]
    assert len(fly0_val) == 20
    val_frames = sorted(int(k.split("Frame_")[1].split("/")[0]) for k in fly0_val)
    assert val_frames == list(range(180, 200))

def test_two_fly_recording_without_fly_sex_behaves_as_before():
    """A recording with no fly_sex key must behave EXACTLY as before
    fly-awareness: every fly in it falls back to the recording-level
    manifest["sex"] alone, so a two-fly recording's flies get IDENTICAL
    treatment (both female-tail-guarded here), not per-fly divergence."""
    m = _merged_mixed("fem", 200)
    man = _manifest(["fem"], female=["fem"])  # recording-level sex only
    split = make_split(m, man, val_recordings=[], female_val_frac=0.10, guard=50)
    for fly in ("fly0", "fly1"):
        flyset = {k: v for k, v in split.items() if k.endswith(f"/{fly}")}
        val_frames = sorted(int(k.split("Frame_")[1].split("/")[0])
                             for k, v in flyset.items() if v == "val")
        assert val_frames == list(range(180, 200)), fly

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
