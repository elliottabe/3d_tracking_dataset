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

def test_mixed_recording_female_fly_forces_whole_frame_onto_female_policy():
    """Four recordings label BOTH flies off ONE physical capture per frame --
    fly0 and fly1 share the same 7 images, so they cannot be split
    independently by their own per-fly sex without leaking those images
    across train/val (this superseded the old per-fly-sex policy, which did
    exactly that: measured val=20/train=330 with 20/20 val frames also in
    train). Image disjointness wins: a frame goes to exactly one side and
    every fly in it follows, and a recording with ANY female fly is treated
    as female for preservation -- kept in training except the guarded tail --
    so val_recordings can no longer carve a mixed recording's male fly out
    WHOLE (that would require splitting frames the female fly also needs)."""
    m = _merged_mixed("mixed", 200)
    man = {"recordings": {"mixed": {"fly_sex": {"fly0": "female", "fly1": "male"}}}}
    split = make_split(m, man, val_recordings=["mixed"],
                       female_val_frac=0.10, guard=50)

    fly0 = {k: v for k, v in split.items() if k.endswith("/fly0")}
    fly1 = {k: v for k, v in split.items() if k.endswith("/fly1")}

    # Same frame, same physical images, for both flies -> same side, always.
    for k0, v0 in fly0.items():
        k1 = k0[:-len("fly0")] + "fly1"
        assert fly1[k1] == v0, f"{k0}={v0} but {k1}={fly1[k1]}"

    # 'mixed' has a female fly (fly0) -> the WHOLE recording (both flies)
    # gets the guarded-tail female policy; val_recordings=["mixed"] has no
    # effect (mirrors the single-fly female case) -- neither fly is held out
    # WHOLE, and fly1 (male) is NOT all-"val" despite being listed there.
    assert set(fly1.values()) == {"train", "val"}
    assert set(fly0.values()) == {"train", "val"}
    fly0_val = [k for k, v in fly0.items() if v == "val"]
    assert len(fly0_val) == 20
    val_frames = sorted(int(k.split("Frame_")[1].split("/")[0]) for k in fly0_val)
    assert val_frames == list(range(180, 200))

    # And the image-level guard agrees: this split is leak-free.
    a = audit_split(m, split, guard=50)
    assert a["cross_fly_leaked_frames"] == 0, a

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

def test_frame_level_split_has_no_cross_fly_leak():
    """Regression for the exact leak measured 2026-08-29: a two-fly recording
    where fly0/fly1 share the same 7 physical images per frame. The bug was
    resolving the split PER FLY (fly0 female -> tail into val, fly1 male ->
    normal rule -> train), so a female val frameset's underlying images were
    also present in train via fly1. val_recordings is deliberately left
    empty here -- that is the exact configuration the bug report measured
    (fly1 has no whole-recording-holdout to fall back to; it just goes to
    train, colliding with fly0's val tail)."""
    m = _merged_mixed("courtship", 200)
    man = {"recordings": {"courtship": {"fly_sex": {"fly0": "female", "fly1": "male"}}}}
    split = make_split(m, man, female_val_frac=0.10, guard=50)

    by_frame: dict[int, set[str]] = {}
    for k, v in split.items():
        fn = int(k.split("Frame_")[1].split("/")[0])
        by_frame.setdefault(fn, set()).add(v)
    leaking = {fn: sides for fn, sides in by_frame.items() if len(sides) > 1}
    assert leaking == {}, f"{len(leaking)} frames appear on both sides: {leaking}"

    a = audit_split(m, split, guard=50)
    assert a["cross_fly_leaked_frames"] == 0, a

def test_audit_detects_cross_fly_frame_leak():
    """The new image-level check in audit_split must actually be able to
    return non-zero, or it is decorative. Hand-construct the leaky shape
    directly (bypassing make_split entirely): fly0 val, fly1 train, on every
    one of 10 shared frames."""
    m = _merged_mixed("bad", 10)
    bad = {}
    for i in range(10):
        bad[f"bad/Frame_{i:06d}/fly0"] = "val"
        bad[f"bad/Frame_{i:06d}/fly1"] = "train"
    a = audit_split(m, bad, guard=50)
    assert a["cross_fly_leaked_frames"] == 10, a

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
