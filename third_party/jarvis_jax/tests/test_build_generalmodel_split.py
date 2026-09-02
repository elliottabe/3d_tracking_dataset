"""Tests for the general_model-only, content-keyed split build.

The two traps this build can fall into, both of which would produce a
confident, self-consistent, completely wrong dataset:

  1. "female" CONTAINS "male". A naive substring test labels every female
     subset male, and every sex-conditioned number in the repo then reads
     fine while measuring the wrong animals.
  2. The per-fly subsets are ONE capture filed twice. If the merge treats
     them as two recordings, the same pixels land on both sides of the split
     -- the exact defect red_data_3d_v5_valfix shipped with (30.3% of val
     byte-identical to train, invisible to every name/frame/fly-level check).
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts")
_spec = importlib.util.spec_from_file_location(
    "build_generalmodel_split", os.path.join(_SCRIPTS, "build_generalmodel_split.py"))
bgs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bgs)

CAMS = [f"Cam20126{i:02d}" for i in range(7)]
N_KP = 50


def test_subset_sex_female_is_not_matched_as_male():
    """'female' contains 'male'; the female test MUST come first."""
    assert bgs.subset_sex("courtship_11_50_female") == "female"
    assert bgs.subset_sex("courtship_11_50_male") == "male"
    assert bgs.subset_sex("20_04_female_climbing") == "female"
    assert bgs.subset_sex("S6male") == "male"
    assert bgs.subset_sex("S8_male_R_amp") == "male"
    assert bgs.subset_sex("courtship_V2") == "unknown"
    assert bgs.subset_sex("grooming") == "unknown"


def _ann(ann_id, image_id, x, y=100.0):
    kp = []
    for i in range(N_KP):
        kp.extend([x + i, y + i, 1])
    return {"id": ann_id, "image_id": image_id, "category_id": 1,
            "bbox": [x, y, 60.0, 40.0], "iscrowd": 0, "num_keypoints": N_KP,
            "keypoints": kp, "segmentation": []}


def test_same_fly_separates_one_animal_labelled_twice_from_two_animals():
    a = _ann(0, 0, 100.0)
    near = _ann(1, 0, 103.0)          # same animal, re-labelled
    other = _ann(2, 0, 600.0)         # the other animal
    assert bgs.same_fly(a, near) < bgs.SAME_FLY_PX
    assert bgs.same_fly(a, other) > bgs.SAME_FLY_PX


def test_same_fly_uses_only_jointly_visible_keypoints():
    a, b = _ann(0, 0, 100.0), _ann(1, 0, 100.0)
    b["keypoints"][2] = 0                       # kp0 not visible in b
    b["keypoints"][0] = 9999                    # ... and wildly displaced
    assert bgs.same_fly(a, b) == 0.0


def _write_subset(gm, subset, rec, frames, x_of_ann, split_of_frame,
                  ambiguous_frames=()):
    """One general_model subset: calib_params, images, annotations.

    Pixel content is a deterministic function of (camera, frame) ONLY, so two
    subsets of the same capture are byte-identical -- which is exactly how the
    real corpus files the two flies of a courtship frame.
    """
    cal = os.path.join(gm, subset, "calib_params", rec)
    os.makedirs(cal, exist_ok=True)
    for c in CAMS:
        vals = ", ".join(str(float(i)) for i in range(12))
        open(os.path.join(cal, f"{c}.yaml"), "w").write(
            f"projectionMatrix:\n  rows: 3\n  cols: 4\n  data: [{vals}]\n")
    blobs = {}
    for fr in frames:
        sp = split_of_frame(fr)
        blob = blobs.setdefault(sp, {"keypoint_names": [f"kp{i}" for i in range(N_KP)],
                                     "skeleton": [], "categories": [],
                                     "images": [], "annotations": [],
                                     "framesets": {}})
        ids = []
        for c in CAMS:
            iid = len(blob["images"])
            fn = f"{rec}/{c}/Frame_{fr}.jpg"
            d = os.path.join(gm, subset, sp, rec, c)
            os.makedirs(d, exist_ok=True)
            rng = np.random.default_rng(abs(hash((c, fr))) % (2 ** 32))
            Image.fromarray(rng.integers(0, 255, (32, 64, 3), dtype=np.uint8)).save(
                os.path.join(gm, subset, sp, fn), quality=95)
            blob["images"].append({"id": iid, "file_name": fn,
                                   "width": 64, "height": 32})
            x = 100.0 if fr in ambiguous_frames else x_of_ann
            blob["annotations"].append(_ann(len(blob["annotations"]), iid, x))
            ids.append(iid)
        blob["framesets"][f"{rec}/Frame_{fr}"] = {"datasetName": rec, "frames": ids}
    ad = os.path.join(gm, subset, "annotations")
    os.makedirs(ad, exist_ok=True)
    for sp, blob in blobs.items():
        json.dump(blob, open(os.path.join(ad, f"instances_{sp}.json"), "w"))


@pytest.fixture
def built(tmp_path, monkeypatch):
    gm, out = str(tmp_path / "general_model"), str(tmp_path / "out")
    frames = list(range(1000, 1040))
    sp = lambda fr: "train" if fr < 1030 else "val"
    # ONE capture per frame, filed as two subsets under two recording names --
    # the real corpus's courtship_<id>_female / _male pattern.
    _write_subset(gm, "courtship_x_female", "2026_01_01_00_00_01", frames,
                  100.0, sp)
    # on frame 1005 the male subset's box lands on the FEMALE's animal --
    # one animal labelled twice, which no count-based rule can detect
    _write_subset(gm, "courtship_x_male", "2026_01_01_00_00_02", frames,
                  600.0, sp, ambiguous_frames={1005})
    # a genuinely unrelated recording: different frame numbers => different
    # pixels, so it must NOT join the courtship alias component
    _write_subset(gm, "solo_male", "2026_02_02_00_00_00",
                  list(range(2000, 2040)), 300.0,
                  lambda fr: "train" if fr < 2030 else "val")
    monkeypatch.setattr(bgs, "VAL_RECORDINGS", ["2026_02_02_00_00_00"])
    monkeypatch.setattr(bgs, "VAL_COMPONENTS_FORCE", [])
    monkeypatch.setattr(bgs, "BEHAVIOR", {"courtship_x_female": "courtship",
                                          "courtship_x_male": "courtship",
                                          "solo_male": "general"})
    monkeypatch.setattr(sys, "argv", ["build", "--gm", gm, "--out", out,
                                      "--guard", "2"])
    bgs.main()
    return out


def _load(out, name):
    return json.load(open(os.path.join(out, "annotations", f"instances_{name}.json")))


def test_per_fly_subsets_merge_into_one_image_with_two_annotations(built):
    """The whole point: same pixels -> ONE image carrying BOTH animals."""
    inst = json.load(open(os.path.join(built, "annotations", "instances.json")))
    per_img = {}
    for a in inst["annotations"]:
        per_img.setdefault(a["image_id"], []).append(a)
    rec_of = {i["id"]: i["recording"] for i in inst["images"]}
    two = [i for i, v in per_img.items() if len(v) == 2]
    assert two, "the two per-fly subsets did not merge onto one image"
    # every merged image is the courtship capture, and carries one of each sex
    for i in two:
        assert rec_of[i] == "2026_01_01_00_00_01"        # component id = min name
        assert {a["sex"] for a in per_img[i]} == {"female", "male"}
        assert {a["fly_id"] for a in per_img[i]} == {0, 1}
    # 40 frames x 7 cameras, minus the one frame where both subsets labelled the
    # SAME animal (recorded ABSENT for both, so it yields no merged image)
    assert len(two) == (40 - 1) * 7


def test_two_subsets_labelling_the_same_animal_are_recorded_absent(built):
    """R15, detected by displacement instead of by count: neither slot can be
    attributed, so BOTH are absent -- never silently assigned to a fly."""
    report = json.load(open(os.path.join(built, "build_report.json")))
    assert report["ambiguous_same_animal_slots"] == 7        # the 7 cameras of frame 1005
    for ex in report["ambiguous_examples"]:
        assert ex["frame"] == 1005
        assert sorted(ex["subsets"]) == ["courtship_x_female", "courtship_x_male"]
        assert ex["median_kp_px"] < bgs.SAME_FLY_PX


def test_split_has_zero_content_overlap(built):
    """Re-derive the assertion from the SHIPPED jsons, not from the builder's
    own bookkeeping: hash the actual files each side points at."""
    import hashlib
    side = {}
    for name in ("train", "val"):
        blob = _load(built, name)
        side[name] = {hashlib.md5(open(os.path.realpath(os.path.join(
            built, "images", i["file_name"])), "rb").read()).hexdigest()
            for i in blob["images"]}
    assert side["train"] and side["val"]
    assert not (side["train"] & side["val"])


def test_both_flies_of_a_capture_land_on_the_same_side(built):
    split = json.load(open(os.path.join(built, "annotations", "split.json")))
    sides = {}
    for k, v in split.items():
        cap = k.rsplit("/", 1)[0]
        sides.setdefault(cap, set()).add(v)
    assert not [c for c, s in sides.items() if len(s) > 1]


def test_sex_comes_from_the_subset_name_not_from_annotations(built):
    """The source annotations carry NO `sex` key at all -- general_model never
    records one. Every emitted annotation must still be sexed."""
    inst = json.load(open(os.path.join(built, "annotations", "instances.json")))
    assert {a["sex"] for a in inst["annotations"]} == {"female", "male"}
    man = json.load(open(os.path.join(built, "manifest.json")))["recordings"]
    assert man["2026_01_01_00_00_01"]["fly_sex"] == {"fly0": "female", "fly1": "male"}
    assert man["2026_01_01_00_00_01"]["sex_source"] == "subset_name"
    assert man["2026_02_02_00_00_00"]["sex"] == "male"


def test_build_report_states_the_two_fly_cost_of_dropping_v3(built):
    """The sourcing decision's measured cost must ship WITH the root, so a
    consumer cannot pick it up without seeing what it is missing."""
    d = json.load(open(os.path.join(built, "build_report.json")))["two_fly_deficit_vs_v3_root"]
    assert d["v3_based_root_two_fly_images"] == 1673
    assert d["lost_annotations"] == 2321
    assert d["source_recordings"] == ["2026_04_07_11_33_33", "2026_04_08_14_59_45"]
