import json
import numpy as np
import pytest
from PIL import Image

from jarvis_jax.data.v5_3d import V5FramesetDataset, frameset_batches

CAMS7 = ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855",
         "Cam2012857", "Cam2012861", "Cam2012862"]


def _v5(tmp_path, n_flies=1, n_frames=2):
    root = tmp_path / "v5"
    (root / "calibrations" / "A").mkdir(parents=True)
    for ci, c in enumerate(CAMS7):
        data = ", ".join(str(v) for v in
                         [8.1 + ci * 0.01, 0.007, -0.03, -2.8,
                          0.009, -8.07, -0.179, 462.7, 0.0, 0.0, 0.0, 1.0])
        (root / "calibrations" / "A" / f"{c}.yaml").write_text(
            "%YAML:1.0\n---\nimage_width: 1936\nimage_height: 448\n"
            "projectionMatrix: !!opencv-matrix\n   rows: 3\n   cols: 4\n   dt: d\n"
            f"   data: [ {data} ]\nscale: 10\n")
    imgs, anns, fs = [], [], {}
    iid = aid = 0
    for f in range(n_frames):
        per_fly = {k: [] for k in range(n_flies)}
        ids = []
        for cam in CAMS7:
            d = root / "images" / "rec_a" / cam
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((448, 1936, 3), np.uint8)).save(
                d / f"Frame_{f:06d}.jpg")
            imgs.append({"id": iid, "width": 1936, "height": 448, "recording": "rec_a",
                         "file_name": f"rec_a/{cam}/Frame_{f:06d}.jpg"})
            ids.append(iid)
            for k in range(n_flies):
                anns.append({"id": aid, "image_id": iid,
                             "bbox": [800.0 + k * 200, 100.0, 80.0, 80.0],
                             "keypoints": [900.0 + k * 200, 150.0, 2] * 50,
                             "num_keypoints": 50, "sex": "female", "fly_id": k,
                             "behavior": "courtship", "src_ann_id": aid})
                per_fly[k].append(aid)
                aid += 1
            iid += 1
        for k in range(n_flies):
            fs[f"rec_a/Frame_{f:06d}/fly{k}"] = {
                "recording": "rec_a", "fly_id": k,
                "frames": ids, "ann_ids": per_fly[k]}
    (root / "annotations").mkdir()
    (root / "annotations" / "instances_train.json").write_text(json.dumps({
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "fly"}],
        "images": imgs, "annotations": anns, "framesets": fs}))
    (root / "manifest.json").write_text(json.dumps({"recordings": {
        "rec_a": {"calib_group": "A", "sex": "female"}}}))
    return root


def test_two_fly_recording_yields_two_samples(tmp_path):
    root = _v5(tmp_path, n_flies=2, n_frames=2)
    ds = V5FramesetDataset(str(root), "train")
    assert len(ds) == 4          # 2 frames x 2 flies
    assert {ds[i]["fly_id"] for i in range(len(ds))} == {0, 1}


def test_sample_schema_matches_v3_loader(tmp_path):
    root = _v5(tmp_path)
    s = V5FramesetDataset(str(root), "train")[0]
    assert s["crops4"].shape == (7, 448, 448, 4) and s["crops4"].dtype == np.uint8
    assert s["centerHM"].shape == (7, 2)
    assert s["center3D"].shape == (3,)
    assert s["cameraMatrices"].shape == (7, 4, 3)
    assert s["kp3d"].shape == (50, 3)
    assert s["vis"].shape == (50,) and s["vis"].dtype == bool


def test_calib_group_filter(tmp_path):
    root = _v5(tmp_path)
    assert len(V5FramesetDataset(str(root), "train", calib_groups=["A"])) > 0
    assert len(V5FramesetDataset(str(root), "train", calib_groups=["B"])) == 0


def test_female_weight_oversamples_female_framesets(tmp_path):
    root = _v5(tmp_path, n_frames=4)
    ds = V5FramesetDataset(str(root), "train")
    n = sum(len(b["kp3d"]) for b in frameset_batches(
        ds, 2, shuffle=True, seed=0, drop_last=False, female_weight=3.0))
    assert n > len(ds), "female_weight>1 must repeat female framesets"


def test_missing_mask_is_zeros_not_a_crash(tmp_path):
    """The 9 newly-added recordings may have gaps; a missing mask must degrade
    to an empty 4th channel, never raise."""
    root = _v5(tmp_path)
    s = V5FramesetDataset(str(root), "train")[0]
    assert s["crops4"][..., 3].max() == 0


# --- sex resolution fallback (human labels live mostly on manifest.json) ---
#
# Measured on real red_data_3d_v5: per-annotation `sex` is "unknown" for 81%
# of annotations while manifest.json's `recordings[<rec>]["sex"]`/
# `["fly_sex"]` carries the human labels for all but 4 recordings. The old
# `sex=` filter here read `meta.get("sex")` (recording-level) directly, so
# a two-fly recording whose recording-level sex is "unknown" (it has no
# single answer -- one fly is male, the other female) could never match
# `sex="female"` even though its fly1 IS female. `_resolve_sex` fixes this:
# (1) the frameset's own annotation `sex` if not "unknown", else (2) the
# manifest recording's `fly_sex["fly<fly_id>"]`, else (3) the manifest
# recording's `sex`, else (4) "unknown".

def _write_calib_group(root, group="A"):
    (root / "calibrations" / group).mkdir(parents=True, exist_ok=True)
    for ci, c in enumerate(CAMS7):
        data = ", ".join(str(v) for v in
                         [8.1 + ci * 0.01, 0.007, -0.03, -2.8,
                          0.009, -8.07, -0.179, 462.7, 0.0, 0.0, 0.0, 1.0])
        (root / "calibrations" / group / f"{c}.yaml").write_text(
            "%YAML:1.0\n---\nimage_width: 1936\nimage_height: 448\n"
            "projectionMatrix: !!opencv-matrix\n   rows: 3\n   cols: 4\n   dt: d\n"
            f"   data: [ {data} ]\nscale: 10\n")


def _v5_sex_fallback(tmp_path):
    """One frameset per fallback branch. No image files are written --
    these tests only exercise __init__'s `sex=` filter and `is_female`,
    never __getitem__ (calib YAMLs ARE required: the constructor builds a
    ReprojectionTool per calib_group for every frameset it keeps).

      - recLabeled: annotation sex="male" already set -> wins outright even
        though the manifest (deliberately) disagrees ("female"); branch 1.
      - recTwoFly (fly_id 0 AND 1): both annotations are sex="unknown" (as
        every real two-fly recording's annotations are); the manifest
        recording's own sex is ALSO "unknown" but carries
        fly_sex={"fly0":"male","fly1":"female"} -- branch 2, keyed by
        fly_id, not recording.
      - recSingle: annotation sex="unknown"; manifest has no fly_sex but a
        recording-level sex="female" -- branch 3.
      - recBlank: annotation sex="unknown"; its manifest entry carries a
        calib_group (so it resolves and loads like any other recording)
        but no `sex`/`fly_sex` at all -- branch 4, stays "unknown".
    """
    root = tmp_path / "v5_sex3d"
    (root / "annotations").mkdir(parents=True)
    _write_calib_group(root, "A")

    imgs, anns, fs = [], [], {}
    iid = aid = 0

    def add_frameset(rec, fly_id, ann_sex):
        nonlocal iid, aid
        cam = CAMS7[0]
        imgs.append({"id": iid, "width": 1936, "height": 448,
                     "file_name": f"{rec}/{cam}/Frame_0.jpg"})
        anns.append({"id": aid, "image_id": iid,
                     "bbox": [800.0, 100.0, 80.0, 80.0],
                     "keypoints": [900.0, 150.0, 2] * 50, "num_keypoints": 50,
                     "sex": ann_sex, "fly_id": fly_id, "src_ann_id": aid})
        fs[f"{rec}/Frame_0/fly{fly_id}"] = {
            "recording": rec, "fly_id": fly_id,
            "frames": [iid], "ann_ids": [aid]}
        iid += 1
        aid += 1

    add_frameset("recLabeled", 0, "male")
    add_frameset("recTwoFly", 0, "unknown")
    add_frameset("recTwoFly", 1, "unknown")
    add_frameset("recSingle", 0, "unknown")
    add_frameset("recBlank", 0, "unknown")

    (root / "annotations" / "instances_train.json").write_text(json.dumps({
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "fly"}],
        "images": imgs, "annotations": anns, "framesets": fs}))
    (root / "manifest.json").write_text(json.dumps({"recordings": {
        "recLabeled": {"calib_group": "A", "sex": "female"},
        "recTwoFly": {"calib_group": "A", "sex": "unknown",
                      "fly_sex": {"fly0": "male", "fly1": "female"}},
        "recSingle": {"calib_group": "A", "sex": "female"},
        "recBlank": {"calib_group": "A"},   # no sex / fly_sex at all
    }}))
    return root


def test_sex_filter_own_annotation_wins_over_manifest(tmp_path):
    """Branch 1: an annotation that already carries a real sex is
    authoritative even when the manifest disagrees."""
    root = _v5_sex_fallback(tmp_path)
    ds_male = V5FramesetDataset(str(root), "train",
                                 recordings=["recLabeled"], sex="male")
    ds_female = V5FramesetDataset(str(root), "train",
                                   recordings=["recLabeled"], sex="female")
    assert len(ds_male) == 1
    assert len(ds_female) == 0


def test_sex_filter_uses_fly_sex_per_fly_for_two_fly_recording(tmp_path):
    """Branch 2 -- the one most likely to be got wrong: the two-fly
    recording's own annotations are BOTH sex=="unknown" and its
    recording-level manifest sex is ALSO "unknown", so resolution must key
    off fly_id via `fly_sex`, not recording."""
    root = _v5_sex_fallback(tmp_path)
    ds_male = V5FramesetDataset(str(root), "train",
                                 recordings=["recTwoFly"], sex="male")
    ds_female = V5FramesetDataset(str(root), "train",
                                   recordings=["recTwoFly"], sex="female")
    assert len(ds_male) == 1 and ds_male.keys[0].endswith("fly0")
    assert len(ds_female) == 1 and ds_female.keys[0].endswith("fly1")


def test_sex_filter_falls_back_to_recording_level_sex(tmp_path):
    """Branch 3: no fly_sex on the manifest entry -> recording-level sex."""
    root = _v5_sex_fallback(tmp_path)
    ds_female = V5FramesetDataset(str(root), "train",
                                   recordings=["recSingle"], sex="female")
    ds_male = V5FramesetDataset(str(root), "train",
                                 recordings=["recSingle"], sex="male")
    assert len(ds_female) == 1
    assert len(ds_male) == 0


def test_sex_filter_stays_unknown_when_manifest_has_nothing(tmp_path):
    """Branch 4: recording absent from the manifest entirely -> "unknown",
    never a crash on a missing manifest entry."""
    root = _v5_sex_fallback(tmp_path)
    ds_unknown = V5FramesetDataset(str(root), "train",
                                    recordings=["recBlank"], sex="unknown")
    ds_female = V5FramesetDataset(str(root), "train",
                                   recordings=["recBlank"], sex="female")
    assert len(ds_unknown) == 1
    assert len(ds_female) == 0


def test_sex_female_filter_returns_nonzero_for_two_fly_recording(tmp_path):
    """The acceptance check named in the task: V5FramesetDataset(...,
    sex="female") must return a non-zero count once the per-fly fly_sex
    fallback resolves the two-fly recording's fly1 (previously zero: every
    recording's manifest-level sex read "unknown")."""
    root = _v5_sex_fallback(tmp_path)
    ds = V5FramesetDataset(str(root), "train", sex="female")
    assert len(ds) > 0


def test_is_female_matches_resolved_per_fly_sex(tmp_path):
    """is_female shares the same blind spot as the sex= filter (it read
    meta.get("sex") directly) -- the two-fly recording's fly1 must be seen
    as female even though the RECORDING-level label is "unknown"."""
    root = _v5_sex_fallback(tmp_path)
    ds = V5FramesetDataset(str(root), "train", recordings=["recTwoFly"])
    assert len(ds) == 2
    by_key = dict(zip(ds.keys, [ds.is_female(i) for i in range(len(ds))]))
    assert by_key["recTwoFly/Frame_0/fly0"] is False
    assert by_key["recTwoFly/Frame_0/fly1"] is True
