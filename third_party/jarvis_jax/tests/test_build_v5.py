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
from jarvis_jax.data.build_v5 import iter_resolved_slots, merge_annotations, SourceRec

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
