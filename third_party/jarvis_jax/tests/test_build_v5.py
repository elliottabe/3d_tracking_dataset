import json
import os
import pytest
from jarvis_jax.data.build_v5 import (
    SourceRec, build_manifest, discover_sources, link_media,
)

CAMS = ["Cam2012630", "Cam2012631"]

def _fake_source(tmp_path, rec, n_frames=3, first=8.1001, masks=True):
    # Shaped like the REAL trees: <root>/<subset>/{train,val}/<rec>/<Cam*>/...
    # image_root/mask_root point at the SUBSET root (siblings of many
    # recordings), never at a root already scoped to one recording -- that
    # was the fixture bug behind the now-removed `_looks_like_camera_root`
    # heuristic (Finding 2, fix round 1).
    subset = f"sub_{rec}"
    img_root = tmp_path / "src" / subset
    for c in CAMS:
        d = img_root / "train" / rec / c
        d.mkdir(parents=True)
        for i in range(n_frames):
            (d / f"Frame_{i:06d}.jpg").write_bytes(b"\xff\xd8fake")
    calib = tmp_path / "src" / subset / "calib_params" / rec
    calib.mkdir(parents=True)
    for c in ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855",
              "Cam2012857", "Cam2012861", "Cam2012862"]:
        data = ", ".join(str(v) for v in [first, 0.0074869, -0.031773, -2.828,
                                          0.0093308, -8.0788, -0.17912, 462.78,
                                          0.0, 0.0, 0.0, 1.0])
        (calib / f"{c}.yaml").write_text(
            f"%YAML:1.0\n---\nprojectionMatrix: !!opencv-matrix\n   data: [ {data} ]\n")
    mask_root = None
    if masks:
        mask_root = img_root / "sam3_masks"
        for c in CAMS:
            d = mask_root / "train" / rec / c
            d.mkdir(parents=True)
            for i in range(n_frames):
                (d / f"Frame_{i:06d}.npz").write_bytes(b"npz")
    return SourceRec(recording=rec, subset=subset, ann_paths=[],
                     calib_dir=str(calib), image_root=str(img_root),
                     mask_root=str(mask_root) if mask_root else None)

def test_manifest_records_calib_group_and_mask_availability(tmp_path):
    srcs = {"rec_a": _fake_source(tmp_path, "rec_a", first=8.1001, masks=True),
            "rec_b": _fake_source(tmp_path, "rec_b", first=8.1333, masks=False)}
    out = tmp_path / "v5"
    man = build_manifest(srcs, str(out))
    assert man["recordings"]["rec_a"]["has_masks"] is True
    assert man["recordings"]["rec_b"]["has_masks"] is False
    assert man["recordings"]["rec_a"]["calib_group"] != man["recordings"]["rec_b"]["calib_group"]
    assert man["recordings"]["rec_a"]["sex"] == "unknown"
    assert json.load(open(out / "manifest.json")) == man

def test_calibrations_are_deduplicated_not_copied_per_recording(tmp_path):
    srcs = {f"rec_{i}": _fake_source(tmp_path, f"rec_{i}", first=8.1001)
            for i in range(3)}
    out = tmp_path / "v5"
    build_manifest(srcs, str(out))
    groups = sorted(os.listdir(out / "calibrations"))
    assert groups == ["A"], f"3 identical calibrations must dedup to 1 dir, got {groups}"
    assert len(os.listdir(out / "calibrations" / "A")) == 7

def test_link_media_creates_symlinks_with_no_split_dirs(tmp_path):
    srcs = {"rec_a": _fake_source(tmp_path, "rec_a")}
    out = tmp_path / "v5"
    build_manifest(srcs, str(out))
    link_media(srcs, str(out))
    p = out / "images" / "rec_a" / "Cam2012630" / "Frame_000000.jpg"
    assert p.is_symlink(), "images must be symlinked, not copied"
    assert (out / "masks" / "rec_a" / "Cam2012630" / "Frame_000000.npz").exists()
    # The whole point: no train/ or val/ directory anywhere in the media tree.
    for root, dirs, _ in os.walk(out / "images"):
        assert "train" not in dirs and "val" not in dirs


# Finding 1 (fix round 1): discover_sources is the only function that touches
# the real filesystem and had zero test coverage -- both the implementer and
# the reviewer independently ran it and got exactly 26 recordings, but that
# verification lived only in a chat log. Pin it here.
GENERAL_MODEL_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/general_model"
V3_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"


@pytest.mark.skipif(
    not (os.path.isdir(GENERAL_MODEL_ROOT) and os.path.isdir(V3_ROOT)),
    reason="gscratch data not present",
)
def test_discover_sources_real_filesystem_smoke():
    srcs = discover_sources(GENERAL_MODEL_ROOT, V3_ROOT)
    assert len(srcs) == 26, f"expected 26 recordings, got {len(srcs)}: {sorted(srcs)}"

    # general_model lacks these 5; red_data_unified_V3 is their only source,
    # and V3 is also where their masks live.
    v3_only = [
        "2026_03_22_12_07_40",
        "2026_04_07_11_33_33",
        "2026_04_08_14_59_45",
        "2026_06_11_13_58_43",
        "2026_06_11_13_58_45",
    ]
    for rec in v3_only:
        assert rec in srcs, f"{rec} missing from discovered sources"
        assert srcs[rec].mask_root is not None, (
            f"{rec} is V3-only and must carry a non-None mask_root")

    # general_model-only recording (no V3 counterpart).
    assert "2026_07_30_13_28_99" in srcs

import json
from jarvis_jax.data.build_v5 import (
    apply_sex_labels, build_recording_sex_map, _canonical_keypoint_names,
    _keypoint_remap, _pad_keypoints, iter_resolved_slots, merge_annotations,
    SourceRec,
)

CAMS7 = ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855",
         "Cam2012857", "Cam2012861", "Cam2012862"]

def _coco(tmp_path, name, rec, n_flies):
    """One frameset over 7 cameras with n_flies annotations per image."""
    images, anns = [], []
    aid = 0
    for ci, cam in enumerate(CAMS7):
        images.append({"id": ci, "width": 1936, "height": 448,
                       "file_name": f"{rec}/{cam}/Frame_000100.jpg"})
        for k in range(n_flies):
            anns.append({"id": aid, "image_id": ci, "bbox": [k * 100.0, 0.0, 50.0, 50.0],
                         "keypoints": [1.0, 2.0, 2] * 50, "num_keypoints": 50,
                         "sex": "unknown", "behavior": "courtship"})
            aid += 1
    blob = {"keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
            "categories": [{"id": 1, "name": "Rat", "num_keypoints": 50}],
            "images": images, "annotations": anns,
            "framesets": {f"{rec}/Frame_000100": {"datasetName": rec,
                                                  "frames": list(range(7))}},
            "calibrations": {rec: {}}}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(blob))
    return str(p)

def test_two_fly_frameset_yields_two_samples_not_one(tmp_path):
    """The v3_3d 'keep first annotation' rule silently dropped the second fly.
    A 7-camera frameset with 2 annotations per image must produce TWO framesets."""
    p = _coco(tmp_path, "two", "rec_two", n_flies=2)
    srcs = {"rec_two": SourceRec("rec_two", None, [p], "", "")}
    out = tmp_path / "v5"
    out.mkdir()
    merged = merge_annotations(srcs, str(out))
    keys = sorted(merged["framesets"])
    assert keys == ["rec_two/Frame_000100/fly0", "rec_two/Frame_000100/fly1"]
    for k in keys:
        assert len(merged["framesets"][k]["frames"]) == 7
        assert len(merged["framesets"][k]["ann_ids"]) == 7
    # no annotation may appear in two framesets
    used = [a for k in keys for a in merged["framesets"][k]["ann_ids"]]
    assert len(used) == len(set(used)) == 14

def test_single_fly_frameset_unchanged(tmp_path):
    p = _coco(tmp_path, "one", "rec_one", n_flies=1)
    srcs = {"rec_one": SourceRec("rec_one", None, [p], "", "")}
    out = tmp_path / "v5"
    out.mkdir()
    merged = merge_annotations(srcs, str(out))
    assert sorted(merged["framesets"]) == ["rec_one/Frame_000100/fly0"]

def test_fly_id_is_consistent_across_cameras(tmp_path):
    p = _coco(tmp_path, "two", "rec_two", n_flies=2)
    srcs = {"rec_two": SourceRec("rec_two", None, [p], "", "")}
    out = tmp_path / "v5"
    out.mkdir()
    merged = merge_annotations(srcs, str(out))
    by_id = {a["id"]: a for a in merged["annotations"]}
    for k, fs in merged["framesets"].items():
        fly_ids = {by_id[a]["fly_id"] for a in fs["ann_ids"]}
        assert len(fly_ids) == 1, f"{k} mixes fly_ids {fly_ids} across cameras"

def test_camera_with_fewer_annotations_is_marked_absent_not_guessed(tmp_path):
    """Ruling R15. When one camera sees fewer flies than the rest, its lone
    annotation may belong to EITHER fly — verified real case:
    2026_04_08_14_59_45/Frame_149677, where Cam2012631's single annotation is
    the RIGHT fly while positional index 0 would file it as the left one.
    The merge must record that camera ABSENT for every fly, never guess."""
    p = tmp_path / "uneq.json"
    images, anns, aid = [], [], 0
    for ci, cam in enumerate(CAMS7):
        images.append({"id": ci, "width": 1936, "height": 448,
                       "file_name": f"rec_u/{cam}/Frame_000100.jpg"})
        # Cam2012631 (index 1) sees only ONE fly; every other camera sees two.
        n = 1 if ci == 1 else 2
        for k in range(n):
            anns.append({"id": aid, "image_id": ci,
                         "bbox": [500.0 if n == 1 else k * 500.0, 0.0, 50.0, 50.0],
                         "keypoints": [1.0, 2.0, 2] * 50, "num_keypoints": 50,
                         "sex": "unknown", "behavior": "courtship"})
            aid += 1
    p.write_text(json.dumps({
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "Rat", "num_keypoints": 50}],
        "images": images, "annotations": anns,
        "framesets": {"rec_u/Frame_000100": {"datasetName": "rec_u",
                                             "frames": list(range(7))}},
        "calibrations": {"rec_u": {}}}))
    srcs = {"rec_u": SourceRec("rec_u", None, [str(p)], "", "")}
    out = tmp_path / "v5"; out.mkdir()
    merged = merge_annotations(srcs, str(out))
    # BOTH flies survive (6 cameras each, above MIN_CAMS=3) ...
    assert sorted(merged["framesets"]) == ["rec_u/Frame_000100/fly0",
                                           "rec_u/Frame_000100/fly1"]
    for key in merged["framesets"]:
        ids = merged["framesets"][key]["ann_ids"]
        assert len(ids) == 7
        # ... and the disagreeing camera is ABSENT, not guessed at.
        assert ids[1] is None, f"{key} guessed an identity for Cam2012631"
        assert sum(a is not None for a in ids) == 6


def test_fly_dropped_when_too_few_cameras_resolve_it(tmp_path):
    """Below MIN_CAMS=3 resolvable cameras a fly cannot be triangulated, so it
    must be dropped rather than emitted with mostly-None slots."""
    p = tmp_path / "sparse.json"
    images, anns, aid = [], [], 0
    for ci, cam in enumerate(CAMS7):
        images.append({"id": ci, "width": 1936, "height": 448,
                       "file_name": f"rec_s/{cam}/Frame_000100.jpg"})
        n = 2 if ci < 2 else 1          # only 2 cameras resolve two flies
        for k in range(n):
            anns.append({"id": aid, "image_id": ci, "bbox": [k * 500.0, 0.0, 50.0, 50.0],
                         "keypoints": [1.0, 2.0, 2] * 50, "num_keypoints": 50,
                         "sex": "unknown", "behavior": "courtship"})
            aid += 1
    p.write_text(json.dumps({
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "Rat", "num_keypoints": 50}],
        "images": images, "annotations": anns,
        "framesets": {"rec_s/Frame_000100": {"datasetName": "rec_s",
                                             "frames": list(range(7))}},
        "calibrations": {"rec_s": {}}}))
    srcs = {"rec_s": SourceRec("rec_s", None, [str(p)], "", "")}
    out = tmp_path / "v5"; out.mkdir()
    merged = merge_annotations(srcs, str(out))
    assert merged["framesets"] == {}, "2 resolvable cameras is below MIN_CAMS"


def test_single_fly_frameset_survives_a_camera_with_no_annotation(tmp_path):
    """The bug ruling R15 also fixes: under the old `min` rule a single camera
    with zero annotations discarded the WHOLE frameset, costing 91 framesets
    from 2026_01_13_18_47_45 and every frameset of wall_frames."""
    p = tmp_path / "gap.json"
    images, anns, aid = [], [], 0
    for ci, cam in enumerate(CAMS7):
        images.append({"id": ci, "width": 1936, "height": 448,
                       "file_name": f"rec_g/{cam}/Frame_000100.jpg"})
        if ci == 3:
            continue                      # this camera saw nothing
        anns.append({"id": aid, "image_id": ci, "bbox": [10.0, 0.0, 50.0, 50.0],
                     "keypoints": [1.0, 2.0, 2] * 50, "num_keypoints": 50,
                     "sex": "female", "behavior": "general"})
        aid += 1
    p.write_text(json.dumps({
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "Rat", "num_keypoints": 50}],
        "images": images, "annotations": anns,
        "framesets": {"rec_g/Frame_000100": {"datasetName": "rec_g",
                                             "frames": list(range(7))}},
        "calibrations": {"rec_g": {}}}))
    srcs = {"rec_g": SourceRec("rec_g", None, [str(p)], "", "")}
    out = tmp_path / "v5"; out.mkdir()
    merged = merge_annotations(srcs, str(out))
    assert list(merged["framesets"]) == ["rec_g/Frame_000100/fly0"]
    ids = merged["framesets"]["rec_g/Frame_000100/fly0"]["ann_ids"]
    assert ids[3] is None and sum(a is not None for a in ids) == 6


def test_category_renamed_from_rat(tmp_path):
    p = _coco(tmp_path, "one", "rec_one", n_flies=1)
    srcs = {"rec_one": SourceRec("rec_one", None, [p], "", "")}
    out = tmp_path / "v5"
    out.mkdir()
    merged = merge_annotations(srcs, str(out))
    assert merged["categories"][0]["name"] == "fly"


def test_iter_resolved_slots_skips_none_but_keeps_the_rest():
    """Ruling R15: ann_ids runs parallel to frames but may hold None for a
    camera whose identity could not be resolved. The accessor must yield
    only the resolved (img_id, ann_id) pairs, in order, never a None ann_id
    and never dropping a resolved camera alongside it."""
    frameset = {"recording": "rec_u", "fly_id": 0,
               "frames": [10, 11, 12, 13],
               "ann_ids": [100, None, 102, 103]}
    assert list(iter_resolved_slots(frameset)) == [(10, 100), (12, 102), (13, 103)]


def test_iter_resolved_slots_on_real_merge_output(tmp_path):
    """End-to-end: feed a frameset actually produced by merge_annotations
    (the disagreeing-camera case from Ruling R15) through the accessor and
    confirm it drops exactly the one None slot merge_annotations recorded,
    keeping all 6 resolved cameras."""
    p = tmp_path / "uneq.json"
    images, anns, aid = [], [], 0
    for ci, cam in enumerate(CAMS7):
        images.append({"id": ci, "width": 1936, "height": 448,
                       "file_name": f"rec_u/{cam}/Frame_000100.jpg"})
        n = 1 if ci == 1 else 2  # Cam2012631 (index 1) sees only one fly
        for k in range(n):
            anns.append({"id": aid, "image_id": ci,
                         "bbox": [500.0 if n == 1 else k * 500.0, 0.0, 50.0, 50.0],
                         "keypoints": [1.0, 2.0, 2] * 50, "num_keypoints": 50,
                         "sex": "unknown", "behavior": "courtship"})
            aid += 1
    p.write_text(json.dumps({
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "Rat", "num_keypoints": 50}],
        "images": images, "annotations": anns,
        "framesets": {"rec_u/Frame_000100": {"datasetName": "rec_u",
                                             "frames": list(range(7))}},
        "calibrations": {"rec_u": {}}}))
    srcs = {"rec_u": SourceRec("rec_u", None, [str(p)], "", "")}
    out = tmp_path / "v5"; out.mkdir()
    merged = merge_annotations(srcs, str(out))
    for key, fs in merged["framesets"].items():
        resolved = list(iter_resolved_slots(fs))
        assert len(resolved) == 6, f"{key}: expected 6 resolved cameras, got {len(resolved)}"
        assert all(ann_id is not None for _, ann_id in resolved)
        # the accessor must not have invented an entry for the disagreeing
        # camera's img_id (frames[1]) -- it is simply absent from the result.
        assert fs["frames"][1] not in [img_id for img_id, _ in resolved]


# --- Label-loss bug fix: general_model annotations never carry `sex`, but
# V3's own annotation files (unscoped to one recording) usually do, for the
# same recording. build_recording_sex_map recovers it; merge_annotations
# uses that recovery as the fallback when a per-annotation `sex` is itself
# "unknown"/absent. -----------------------------------------------------

def test_build_recording_sex_map_recovers_unanimous_recording(tmp_path):
    p = tmp_path / "v3.json"
    p.write_text(json.dumps({
        "images": [{"id": 0, "file_name": "rec_a/Cam2012630/Frame_000100.jpg"},
                   {"id": 1, "file_name": "rec_a/Cam2012631/Frame_000100.jpg"}],
        "annotations": [{"id": 0, "image_id": 0, "sex": "male"},
                        {"id": 1, "image_id": 1, "sex": "male"}]}))
    rec_sex, mixed = build_recording_sex_map([str(p)])
    assert rec_sex == {"rec_a": "male"}
    assert mixed == []

def test_build_recording_sex_map_reports_mixed_as_unresolved(tmp_path):
    """A recording with BOTH 'male' and 'female' among its non-unknown
    values is a real data conflict, not a majority to break -- it must be
    reported, not silently resolved either way."""
    p = tmp_path / "v3.json"
    p.write_text(json.dumps({
        "images": [{"id": 0, "file_name": "rec_b/Cam2012630/Frame_000100.jpg"},
                   {"id": 1, "file_name": "rec_b/Cam2012631/Frame_000100.jpg"}],
        "annotations": [{"id": 0, "image_id": 0, "sex": "male"},
                        {"id": 1, "image_id": 1, "sex": "female"}]}))
    rec_sex, mixed = build_recording_sex_map([str(p)])
    assert "rec_b" not in rec_sex
    assert mixed == ["rec_b"]

def test_build_recording_sex_map_all_unknown_is_no_signal_not_unanimous(tmp_path):
    """4,592 'unknown' annotations (2026_06_01_15_34_04's real count) is NOT
    a unanimous 'unknown' recording -- it is no signal at all, so it must
    not appear in the map (and must not be reported as mixed either)."""
    p = tmp_path / "v3.json"
    p.write_text(json.dumps({
        "images": [{"id": 0, "file_name": "rec_c/Cam2012630/Frame_000100.jpg"}],
        "annotations": [{"id": 0, "image_id": 0, "sex": "unknown"}]}))
    rec_sex, mixed = build_recording_sex_map([str(p)])
    assert rec_sex == {}
    assert mixed == []

def _no_sex_key_coco(tmp_path, name, rec, frame, image_id_base):
    """Shaped exactly like a REAL general_model instances_*.json: annotation
    keys are bbox, category_id, id, image_id, iscrowd, keypoints,
    num_keypoints, segmentation -- no 'sex' key at all."""
    images = [{"id": image_id_base + ci, "width": 1936, "height": 448,
               "file_name": f"{rec}/{cam}/Frame_{frame}.jpg"}
              for ci, cam in enumerate(CAMS7)]
    anns = [{"id": image_id_base + ci, "image_id": image_id_base + ci,
             "bbox": [0.0, 0.0, 50.0, 50.0], "keypoints": [1.0, 2.0, 2] * 50,
             "num_keypoints": 50}
            for ci in range(7)]
    blob = {"keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
            "categories": [{"id": 1, "name": "fly", "num_keypoints": 50}],
            "images": images, "annotations": anns,
            "framesets": {f"{rec}/Frame_{frame}": {"datasetName": rec,
                                                    "frames": list(range(image_id_base, image_id_base + 7))}}}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(blob))
    return str(p)

def test_merge_annotations_recovers_sex_from_v3_when_general_model_lacks_it(tmp_path):
    """Regression for the label-loss bug: 'rec_gm' is sourced (ann_paths)
    from a general_model-style file with NO 'sex' key anywhere. Its sex must
    still be recovered from V3's OWN annotation file -- referenced here only
    via a completely different recording's SourceRec.ann_paths, exactly as
    happens for real (a V3 annotations file is not scoped to one
    recording, so it carries rec_gm's data too even though rec_gm's frames
    are read from general_model)."""
    gm = _no_sex_key_coco(tmp_path, "gm", "rec_gm", "000100", image_id_base=0)

    v3 = tmp_path / "v3.json"
    v3_images = [{"id": 100 + ci, "width": 1936, "height": 448,
                  "file_name": f"rec_v3only/{cam}/Frame_000200.jpg"}
                 for ci, cam in enumerate(CAMS7)]
    v3_anns = [{"id": 100 + ci, "image_id": 100 + ci, "bbox": [0.0, 0.0, 50.0, 50.0],
                "keypoints": [1.0, 2.0, 2] * 50, "num_keypoints": 50, "sex": "unknown"}
               for ci in range(7)]
    # rec_gm's OWN entry inside V3's (unscoped) annotation file, carrying sex.
    v3_images += [{"id": 200 + ci, "width": 1936, "height": 448,
                   "file_name": f"rec_gm/{cam}/Frame_000900.jpg"}
                  for ci, cam in enumerate(CAMS7)]
    v3_anns += [{"id": 200 + ci, "image_id": 200 + ci, "bbox": [0.0, 0.0, 50.0, 50.0],
                 "keypoints": [1.0, 2.0, 2] * 50, "num_keypoints": 50, "sex": "male"}
                for ci in range(7)]
    v3.write_text(json.dumps({
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "fly", "num_keypoints": 50}],
        "images": v3_images, "annotations": v3_anns,
        "framesets": {"rec_v3only/Frame_000200": {"datasetName": "rec_v3only",
                                                   "frames": list(range(100, 107))}}}))

    srcs = {
        "rec_gm": SourceRec("rec_gm", "some_subset", [gm], "", ""),
        "rec_v3only": SourceRec("rec_v3only", None, [str(v3)], "", ""),
    }
    out = tmp_path / "v5"; out.mkdir()
    merged = merge_annotations(srcs, str(out))

    gm_anns_out = [a for a in merged["annotations"]
                   if merged["images"][a["image_id"]]["recording"] == "rec_gm"]
    assert gm_anns_out, "expected merged annotations for rec_gm"
    assert all(a["sex"] == "male" for a in gm_anns_out), (
        "general_model-sourced annotations (no 'sex' key at all) must "
        "recover sex from V3's recording-level map instead of defaulting "
        "to 'unknown'")
    assert srcs["rec_gm"].sex == "male", (
        "the recovered sex must also be recorded on SourceRec.sex, the "
        "baseline build_manifest writes into manifest.json")

def test_merge_annotations_leaves_conflicting_recording_unknown_not_guessed(tmp_path):
    """If a recording's sex disagrees across its own annotation sources (a
    real data conflict), annotations with no sex of their own must stay
    'unknown' rather than being assigned either value."""
    gm = _no_sex_key_coco(tmp_path, "gm", "rec_mixed", "000100", image_id_base=0)

    v3 = tmp_path / "v3.json"
    v3_images = [{"id": 100 + ci, "width": 1936, "height": 448,
                  "file_name": f"rec_mixed/{cam}/Frame_000900.jpg"}
                 for ci, cam in enumerate(CAMS7)]
    v3_anns = [{"id": 100 + ci, "image_id": 100 + ci, "bbox": [0.0, 0.0, 50.0, 50.0],
                "keypoints": [1.0, 2.0, 2] * 50, "num_keypoints": 50,
                "sex": "male" if ci < 4 else "female"}
               for ci in range(7)]
    v3.write_text(json.dumps({
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "fly", "num_keypoints": 50}],
        "images": v3_images, "annotations": v3_anns, "framesets": {}}))

    srcs = {"rec_mixed": SourceRec("rec_mixed", "some_subset", [gm, str(v3)], "", "")}
    out = tmp_path / "v5"; out.mkdir()
    merged = merge_annotations(srcs, str(out))
    assert merged["annotations"], "expected merged annotations for rec_mixed"
    assert all(a["sex"] == "unknown" for a in merged["annotations"])
    assert srcs["rec_mixed"].sex == "unknown"

def test_apply_sex_labels_overrides_recovered_sex_baseline(tmp_path):
    """Ordering guarantee: apply_sex_labels (the human labels, Part 2) is
    the authority and must win over whatever merge_annotations/build_manifest
    already wrote as a recovered baseline (Part 1) -- it is called strictly
    AFTER the full build, and unconditionally overwrites the 'sex' key."""
    srcs = {"rec_a": _fake_source(tmp_path, "rec_a")}
    srcs["rec_a"].sex = "male"          # simulates Part 1's recovered baseline
    out = tmp_path / "v5"
    man = build_manifest(srcs, str(out))
    assert man["recordings"]["rec_a"]["sex"] == "male"

    man2 = apply_sex_labels(str(out), {"rec_a": "female"})
    assert man2["recordings"]["rec_a"]["sex"] == "female"
    on_disk = json.load(open(out / "manifest.json"))
    assert on_disk["recordings"]["rec_a"]["sex"] == "female"


# --- Finding 2: general_model-sourced annotations for a recording V3 ALSO
# covers put `src_ann_id` in a different id space than the V3-borrowed SAM3
# masks (~0% mask hits for exactly those 12 recordings). Fix: prefer V3's
# annotations for any recording V3 covers -- same-length framesets verified
# identical on the real tree, so this is lossless. -----------------------

def test_discover_sources_prefers_v3_annotations_for_overlap_recording(tmp_path):
    gm_root = tmp_path / "general_model"
    v3_root = tmp_path / "v3"
    rec = "rec_overlap"

    gm_calib = gm_root / "sub1" / "calib_params" / rec
    gm_calib.mkdir(parents=True)
    (gm_calib / "Cam1.yaml").write_text("dummy")
    gm_ann_dir = gm_root / "sub1" / "annotations"
    gm_ann_dir.mkdir(parents=True)
    (gm_ann_dir / "instances_train.json").write_text(
        json.dumps({"images": [], "annotations": []}))

    v3_calib = v3_root / "calib_params" / rec
    v3_calib.mkdir(parents=True)
    (v3_calib / "Cam1.yaml").write_text("dummy")
    v3_ann_dir = v3_root / "annotations"
    v3_ann_dir.mkdir(parents=True)
    v3_train = v3_ann_dir / "instances_train.json"
    v3_train.write_text(json.dumps({"images": [], "annotations": []}))

    srcs = discover_sources(str(gm_root), str(v3_root))
    assert srcs[rec].subset == "sub1", "still general_model-attributed (image_root/calib_dir)"
    assert srcs[rec].ann_paths == [str(v3_train)], (
        "a recording V3 also covers must take its ANNOTATIONS from V3, not "
        "general_model -- otherwise sex is lost and src_ann_id lands in "
        "the wrong id space for the (also V3-sourced) masks")
    assert srcs[rec].mask_root == str(v3_root / "sam3_masks")

def test_discover_sources_v3_only_recording_still_gets_v3_annotations(tmp_path):
    """A recording general_model doesn't have at all must be unaffected by
    the overlap-preference branch."""
    gm_root = tmp_path / "general_model"
    gm_root.mkdir()
    v3_root = tmp_path / "v3"
    rec = "rec_v3_only"
    v3_calib = v3_root / "calib_params" / rec
    v3_calib.mkdir(parents=True)
    (v3_calib / "Cam1.yaml").write_text("dummy")
    v3_ann_dir = v3_root / "annotations"
    v3_ann_dir.mkdir(parents=True)
    v3_train = v3_ann_dir / "instances_train.json"
    v3_train.write_text(json.dumps({"images": [], "annotations": []}))

    srcs = discover_sources(str(gm_root), str(v3_root))
    assert srcs[rec].subset is None
    assert srcs[rec].ann_paths == [str(v3_train)]


# --- Finding 3: headless/leg-amputation recordings carry 47/44 keypoints
# (a strict, in-canonical-order subset of the full 50) instead of 50. The
# loader used to drop those cameras entirely (683 of 3,717 fly-samples
# unusable). Fix: pad every annotation to the canonical 50 BY NAME, never
# positionally -- a positional pad would place leg keypoints on the head.

def test_canonical_keypoint_names_is_the_longest_schema_seen(tmp_path):
    full = tmp_path / "full.json"
    full.write_text(json.dumps({"keypoint_names": ["a", "b", "c", "d"]}))
    reduced = tmp_path / "reduced.json"
    reduced.write_text(json.dumps({"keypoint_names": ["a", "c"]}))
    assert _canonical_keypoint_names([str(full), str(reduced)]) == ["a", "b", "c", "d"]

def test_keypoint_remap_maps_by_name_not_position():
    canonical = ["head", "leg_L", "leg_R", "tail"]
    # source is missing 'leg_L' -- a REAL reduced schema is always a subset
    # in canonical order, exactly like this.
    source = ["head", "leg_R", "tail"]
    remap = _keypoint_remap(source, canonical)
    # canonical[1] ('leg_L') has no source counterpart -> None.
    assert remap == [0, None, 1, 2]

def test_keypoint_remap_raises_on_a_name_canonical_does_not_have(tmp_path):
    """A source name absent from canonical is a BLOCKED-level finding --
    the two schemas aren't simply subset/superset, so padding would be a
    guess. Must raise, never silently drop the point."""
    with pytest.raises(ValueError):
        _keypoint_remap(["head", "extra_never_seen"], ["head", "leg_L"])

def test_pad_keypoints_fills_missing_slots_invisible_not_guessed():
    # source has 2 points (x,y,v each); canonical has 3, with the source's
    # points landing at canonical slots 0 and 2.
    kps = [10.0, 20.0, 1, 30.0, 40.0, 1]
    remap = [0, None, 1]
    padded = _pad_keypoints(kps, remap)
    assert padded == [10.0, 20.0, 1, 0.0, 0.0, 0, 30.0, 40.0, 1]

def _coco_reduced_schema(tmp_path, name, rec, kp_names, n_kp):
    """A single-fly, single-camera-count-agreeing frameset whose annotations
    use a REDUCED keypoint schema (like a real headless/amputation
    recording) -- `n_kp` points instead of the canonical 50."""
    images, anns, aid = [], [], 0
    for ci, cam in enumerate(CAMS7):
        images.append({"id": ci, "width": 1936, "height": 448,
                       "file_name": f"{rec}/{cam}/Frame_000100.jpg"})
        anns.append({"id": aid, "image_id": ci, "bbox": [0.0, 0.0, 50.0, 50.0],
                     "keypoints": [1.0, 2.0, 1] * n_kp, "num_keypoints": n_kp,
                     "sex": "unknown", "behavior": "unknown"})
        aid += 1
    blob = {"keypoint_names": kp_names, "skeleton": [],
            "categories": [{"id": 1, "name": "fly", "num_keypoints": n_kp}],
            "images": images, "annotations": anns,
            "framesets": {f"{rec}/Frame_000100": {"datasetName": rec,
                                                  "frames": list(range(7))}}}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(blob))
    return str(p)

def test_merge_annotations_pads_reduced_schema_to_canonical_by_name(tmp_path):
    canonical = ["Antenna_Base", "EyeL", "EyeR", "T1L_Tro", "T1R_Tro"]
    reduced = ["T1L_Tro", "T1R_Tro"]     # a "headless" fly: missing head points
    # Canonical-schema source (a small custom schema, not the unrelated
    # 50-name one `_coco` uses, so this test controls exactly what's canonical).
    full_p = tmp_path / "canon.json"
    images, anns, aid = [], [], 0
    for ci, cam in enumerate(CAMS7):
        images.append({"id": ci, "width": 1936, "height": 448,
                       "file_name": f"rec_canon/{cam}/Frame_000200.jpg"})
        anns.append({"id": aid, "image_id": ci, "bbox": [0.0, 0.0, 50.0, 50.0],
                     "keypoints": [9.0, 9.0, 1] * len(canonical), "num_keypoints": len(canonical),
                     "sex": "unknown", "behavior": "unknown"})
        aid += 1
    full_p.write_text(json.dumps({
        "keypoint_names": canonical, "skeleton": [],
        "categories": [{"id": 1, "name": "fly", "num_keypoints": len(canonical)}],
        "images": images, "annotations": anns,
        "framesets": {"rec_canon/Frame_000200": {"datasetName": "rec_canon",
                                                  "frames": list(range(7))}}}))
    headless = _coco_reduced_schema(tmp_path, "headless", "rec_headless", reduced, len(reduced))

    srcs = {
        "rec_canon": SourceRec("rec_canon", None, [str(full_p)], "", ""),
        "rec_headless": SourceRec("rec_headless", None, [headless], "", ""),
    }
    out = tmp_path / "v5"; out.mkdir()
    merged = merge_annotations(srcs, str(out))

    assert merged["keypoint_names"] == canonical
    img_by_id = {i["id"]: i for i in merged["images"]}
    headless_anns = [a for a in merged["annotations"]
                     if img_by_id[a["image_id"]]["recording"] == "rec_headless"]
    assert headless_anns, "expected merged annotations for rec_headless"
    for a in headless_anns:
        assert len(a["keypoints"]) == len(canonical) * 3
        kps = a["keypoints"]
        invisible = {canonical[i] for i in range(len(canonical)) if kps[3 * i + 2] == 0}
        assert invisible == {"Antenna_Base", "EyeL", "EyeR"}, (
            f"wrong slots marked invisible: {invisible} (must be BY NAME, "
            f"not position, or this would place a leg keypoint on the head)")
        # the two points the reduced schema DOES have must carry real values,
        # not be zeroed out.
        for name in ("T1L_Tro", "T1R_Tro"):
            i = canonical.index(name)
            assert kps[3 * i:3 * i + 2] == [1.0, 2.0]
            assert kps[3 * i + 2] != 0
