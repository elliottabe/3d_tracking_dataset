"""TDD tests for scripts/build_detector_dataset.py.

Covers the pure logic of the unified 50-node keypoint-detector dataset build:
per-subset keypoint expansion to canonical fly50 order with COCO v=0 absence
masks, the explicit subset->rule validation, and the by-RECORDING stratified
train/val split (never by frame -- see module docstring for why).

All tests are synthetic: no dependency on the real general_model data on
disk. The end-to-end `build()` test constructs a tiny fake source_root with a
couple of fabricated subsets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.build_detector_dataset import (
    FLY50,
    FLY50_EDGES,
    SUBSET_RULES,
    SUBSET_CATEGORY,
    REQUIRED_VAL_CATEGORIES,
    SubsetRuleMismatch,
    expected_count_for_rule,
    validate_subset_keypoint_count,
    expand_to_canonical,
    CategorySplitInfo,
    SplitResult,
    stratified_recording_split,
    build,
)


NUM_FLY50 = len(FLY50)


# ---------------------------------------------------------------------------
# expand_to_canonical
# ---------------------------------------------------------------------------

def test_identity_rule_50_in_50_out_all_present():
    kps = []
    for i in range(NUM_FLY50):
        kps.extend([float(i * 10), float(i * 10 + 1), 1])
    kp50, mask50 = expand_to_canonical(kps, "identity")

    assert kp50.shape == (NUM_FLY50, 3)
    assert mask50.shape == (NUM_FLY50,)
    assert np.all(mask50)
    assert np.all(kp50[:, 2] == 2)
    for i in range(NUM_FLY50):
        assert kp50[i, 0] == i * 10
        assert kp50[i, 1] == i * 10 + 1


def test_headless_rule_47_in_50_out_head_masked():
    drop = [0, 1, 2]
    rule = ("drop", drop)
    n_in = NUM_FLY50 - len(drop)
    kps = []
    for i in range(n_in):
        kps.extend([float(1000 + i), float(2000 + i), 1])

    kp50, mask50 = expand_to_canonical(kps, rule)

    assert kp50.shape == (NUM_FLY50, 3)
    # head nodes absent
    for idx in drop:
        assert mask50[idx] == False
        assert kp50[idx, 0] == 0
        assert kp50[idx, 1] == 0
        assert kp50[idx, 2] == 0

    # remaining 47 nodes present, in order, starting right after the drop
    keep_idx = [i for i in range(NUM_FLY50) if i not in drop]
    assert len(keep_idx) == n_in
    for src, dst in enumerate(keep_idx):
        assert mask50[dst] == True
        assert kp50[dst, 2] == 2

    # Named-node spot check: Scutellum (fly50 index 3) gets the input's
    # first point (headless data starts at Scutellum).
    scutellum_idx = FLY50.index("Scutellum")
    assert scutellum_idx == 3
    assert kp50[scutellum_idx, 0] == 1000
    assert kp50[scutellum_idx, 1] == 2000


@pytest.mark.parametrize("subset_key", ["S8_male_R_amp", "S9_male_L_amp"])
def test_amputated_rules_44_in_50_out_exact_six_masked(subset_key):
    rule = SUBSET_RULES[subset_key]
    assert rule[0] == "drop"
    drop = rule[1]
    assert len(drop) == 6
    n_in = NUM_FLY50 - 6
    kps = []
    for i in range(n_in):
        kps.extend([float(i), float(i), 1])

    kp50, mask50 = expand_to_canonical(kps, rule)

    absent = [i for i in range(NUM_FLY50) if not mask50[i]]
    assert sorted(absent) == sorted(drop)
    assert mask50.sum() == 44

    # The subtle part: T1*_ThxCx on the amputated side stays PRESENT (only
    # the distal segment is amputated, not the coxa/thorax joint).
    if subset_key == "S8_male_R_amp":
        thxcx_idx = FLY50.index("T1R_ThxCx")
    else:
        thxcx_idx = FLY50.index("T1L_ThxCx")
    assert thxcx_idx not in drop
    assert mask50[thxcx_idx] == True


def test_expand_to_canonical_unknown_rule_raises():
    with pytest.raises(ValueError):
        expand_to_canonical([0.0, 0.0, 1], "not-a-real-rule")


# ---------------------------------------------------------------------------
# subset rule validation
# ---------------------------------------------------------------------------

def test_expected_count_for_rule_identity_and_drop():
    assert expected_count_for_rule("identity") == NUM_FLY50
    assert expected_count_for_rule(("drop", [0, 1, 2])) == NUM_FLY50 - 3


def test_validate_subset_keypoint_count_mismatch_raises_clear_error():
    rule = ("drop", [0, 1, 2])
    with pytest.raises(SubsetRuleMismatch, match="headless_fake"):
        validate_subset_keypoint_count("headless_fake", NUM_FLY50, rule)


def test_validate_subset_keypoint_count_match_ok():
    validate_subset_keypoint_count("full_fake", NUM_FLY50, "identity")


def test_all_declared_rules_have_consistent_expected_counts():
    # Sanity check on the explicit table itself: every rule must resolve to
    # a valid, non-negative expected count.
    for subset, rule in SUBSET_RULES.items():
        n = expected_count_for_rule(rule)
        assert 0 < n <= NUM_FLY50, (subset, rule, n)


# ---------------------------------------------------------------------------
# stratified_recording_split
# ---------------------------------------------------------------------------

def _category_recordings():
    return {
        "courtship_female": {"cf1": 100, "cf2": 150, "cf3": 270},
        "courtship_male": {"cm1": 105, "cm2": 150, "cm3": 316},
        "grooming": {"groom1": 4592},          # single recording
        "wall": {"wall1": 33},                 # single recording
        "headless": {"h1": 140, "h2": 462, "h3": 341, "h4": 224, "h5": 175},
        "amputated": {"amp1": 1334, "amp2": 2104},
        "male_general": {"male_gen1": 3351},   # not required, single recording
    }


REQUIRED = {"courtship_female", "courtship_male", "grooming", "wall", "headless", "amputated"}


def test_split_no_recording_in_both_train_and_val():
    result = stratified_recording_split(_category_recordings(), REQUIRED, seed=0)
    assert result.train_recordings.isdisjoint(result.val_recordings)


def test_split_required_multi_recording_categories_covered_in_val():
    result = stratified_recording_split(_category_recordings(), REQUIRED, seed=0)
    for cat in ("courtship_female", "courtship_male", "headless", "amputated"):
        info = result.category_info[cat]
        assert len(info.val_recordings) >= 1, cat
        assert not info.unvalidatable, cat


def test_split_single_recording_category_goes_to_train_and_flagged():
    result = stratified_recording_split(_category_recordings(), REQUIRED, seed=0)
    for cat in ("grooming", "wall", "male_general"):
        info = result.category_info[cat]
        assert info.val_recordings == []
        assert info.unvalidatable is True
        assert cat in result.unvalidatable_categories
    assert "groom1" in result.train_recordings
    assert "wall1" in result.train_recordings
    assert "male_gen1" in result.train_recordings


def test_split_achieves_target_fraction_in_reasonable_range():
    result = stratified_recording_split(_category_recordings(), REQUIRED, seed=0)
    # Not a hard requirement in the spec beyond "target ~15-20%, report it" --
    # just sanity-check it lands somewhere plausible rather than degenerate
    # (e.g. 0% or 100%).
    assert 0.05 < result.achieved_val_fraction < 0.4


def test_split_missing_required_category_raises():
    recs = _category_recordings()
    del recs["wall"]
    with pytest.raises(ValueError, match="wall"):
        stratified_recording_split(recs, REQUIRED, seed=0)


def test_split_is_deterministic_for_a_given_seed():
    r1 = stratified_recording_split(_category_recordings(), REQUIRED, seed=0)
    r2 = stratified_recording_split(_category_recordings(), REQUIRED, seed=0)
    assert r1.train_recordings == r2.train_recordings
    assert r1.val_recordings == r2.val_recordings


# ---------------------------------------------------------------------------
# build() end-to-end on a tiny synthetic source_root
# ---------------------------------------------------------------------------

def _write_subset(root: Path, subset: str, split: str, recording: str,
                   n_images: int, n_kp: int, start_frame: int = 0):
    """Write a minimal instances_{split}.json for one synthetic subset."""
    ann_dir = root / subset / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    for i in range(n_images):
        img_id = start_frame + i
        file_name = f"{recording}/CamA/Frame_{img_id}.jpg"
        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": 100,
            "height": 100,
        })
        img_path = root / subset / split / file_name
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.touch()
        kp = []
        for k in range(n_kp):
            kp.extend([float(k), float(k), 1])
        annotations.append({
            "id": img_id,
            "image_id": img_id,
            "category_id": 1,
            "iscrowd": 0,
            "bbox": [0, 0, 10, 10],
            "num_keypoints": n_kp,
            "keypoints": kp,
            "segmentation": [],
        })
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 0, "name": "Rat", "num_keypoints": n_kp}]}
    (ann_dir / f"instances_{split}.json").write_text(json.dumps(coco))


@pytest.fixture
def fake_source_root(tmp_path):
    root = tmp_path / "general_model_fake"
    # "full" category: two recordings, identity rule, 50kp
    _write_subset(root, "fake_full_a", "train", "rec_full_a", n_images=6, n_kp=NUM_FLY50)
    _write_subset(root, "fake_full_b", "train", "rec_full_b", n_images=4, n_kp=NUM_FLY50)
    # "headless" category: two recordings, drop-head rule, 47kp
    _write_subset(root, "fake_headless_a", "train", "rec_headless_a", n_images=5, n_kp=NUM_FLY50 - 3)
    _write_subset(root, "fake_headless_b", "val", "rec_headless_b", n_images=3, n_kp=NUM_FLY50 - 3)
    return root


FAKE_SUBSET_RULES = {
    "fake_full_a": "identity",
    "fake_full_b": "identity",
    "fake_headless_a": ("drop", [0, 1, 2]),
    "fake_headless_b": ("drop", [0, 1, 2]),
}
FAKE_SUBSET_CATEGORY = {
    "fake_full_a": "full",
    "fake_full_b": "full",
    "fake_headless_a": "headless",
    "fake_headless_b": "headless",
}
FAKE_REQUIRED = {"full", "headless"}


def test_build_emits_nonempty_categories_matching_fly50(fake_source_root, tmp_path):
    out_root = tmp_path / "out"
    report = build(
        fake_source_root, out_root, seed=0,
        subset_rules=FAKE_SUBSET_RULES, subset_category=FAKE_SUBSET_CATEGORY,
        required_categories=FAKE_REQUIRED,
    )
    train_coco = json.loads((out_root / "annotations" / "instances_train.json").read_text())
    val_coco = json.loads((out_root / "annotations" / "instances_val.json").read_text())

    for coco in (train_coco, val_coco):
        cats = coco["categories"]
        assert len(cats) == 1
        assert cats[0]["keypoints"] == FLY50
        assert cats[0]["skeleton"] == FLY50_EDGES

    assert isinstance(report, dict)
    assert (out_root / "build_report.json").exists()


def test_build_no_recording_split_across_train_and_val(fake_source_root, tmp_path):
    out_root = tmp_path / "out"
    build(
        fake_source_root, out_root, seed=0,
        subset_rules=FAKE_SUBSET_RULES, subset_category=FAKE_SUBSET_CATEGORY,
        required_categories=FAKE_REQUIRED,
    )
    train_coco = json.loads((out_root / "annotations" / "instances_train.json").read_text())
    val_coco = json.loads((out_root / "annotations" / "instances_val.json").read_text())

    def recordings_of(coco):
        return {im["file_name"].split("/")[0] for im in coco["images"]}

    train_recs = recordings_of(train_coco)
    val_recs = recordings_of(val_coco)
    assert train_recs.isdisjoint(val_recs)
    # every image path is reconstructible as source_root/subset/orig_split/file_name
    for im in train_coco["images"] + val_coco["images"]:
        p = fake_source_root / im["subset"] / im["orig_split"] / im["file_name"]
        assert p.exists(), p


def test_build_report_has_required_fields(fake_source_root, tmp_path):
    out_root = tmp_path / "out"
    report = build(
        fake_source_root, out_root, seed=0,
        subset_rules=FAKE_SUBSET_RULES, subset_category=FAKE_SUBSET_CATEGORY,
        required_categories=FAKE_REQUIRED,
    )
    assert "subsets" in report
    assert "categories" in report
    assert "totals" in report
    assert "val_fraction" in report["totals"]
    assert "unvalidatable_categories" in report
    for cat in ("full", "headless"):
        assert cat in report["categories"]
        assert "train_recordings" in report["categories"][cat]
        assert "val_recordings" in report["categories"][cat]
        assert "absent_node_counts" in report["categories"][cat]

    # headless subsets mask the 3 head nodes -> non-zero absent counts
    headless_absent = report["categories"]["headless"]["absent_node_counts"]
    assert headless_absent.get("Antenna_Base", 0) > 0
    # full (identity) subsets never mask anything
    full_absent = report["categories"]["full"]["absent_node_counts"]
    assert full_absent == {}


def test_build_unknown_subset_without_rule_raises(tmp_path):
    root = tmp_path / "src"
    _write_subset(root, "mystery_subset", "train", "rec_x", n_images=2, n_kp=NUM_FLY50)
    with pytest.raises(ValueError, match="mystery_subset"):
        build(root, tmp_path / "out", subset_rules={}, subset_category={},
              required_categories=set())


def test_build_val_recordings_override_is_respected(fake_source_root, tmp_path):
    out_root = tmp_path / "out"
    build(
        fake_source_root, out_root, seed=0,
        val_recordings=["rec_full_a", "rec_headless_b"],
        subset_rules=FAKE_SUBSET_RULES, subset_category=FAKE_SUBSET_CATEGORY,
        required_categories=FAKE_REQUIRED,
    )
    val_coco = json.loads((out_root / "annotations" / "instances_val.json").read_text())
    val_recs = {im["file_name"].split("/")[0] for im in val_coco["images"]}
    assert val_recs == {"rec_full_a", "rec_headless_b"}
