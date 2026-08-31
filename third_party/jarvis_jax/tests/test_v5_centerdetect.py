"""Tests for ``jarvis_jax.data.v5_centerdetect.V5CenterDetectDataset``.

Synthetic-fixture tests (fast, no real data needed) plus a guard against the
real red_data_3d_v5_valfix root when it is present -- the exact numbers the
task brief asks to be reported (train/val image and two-fly counts, and the
`num_flies` oversampling repeat factors) come from that guarded test's
printed output.
"""
import json
import os

import numpy as np
import pytest
from PIL import Image

from jarvis_jax.data.v5_centerdetect import K_MAX, V5CenterDetectDataset, batches

V5_VALFIX_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5_valfix"


def _make_fixture(tmp_path):
    """A tiny 2-recording, flat-layout v5-shaped root: 1 one-fly image + 1
    two-fly image, real (tiny) JPEGs on disk so __getitem__ is exercised."""
    root = tmp_path
    (root / "images" / "recA" / "cam1").mkdir(parents=True)
    (root / "images" / "recB" / "cam1").mkdir(parents=True)
    (root / "annotations").mkdir()

    img_one = Image.new("RGB", (200, 100), (10, 20, 30))
    img_one.save(root / "images" / "recA" / "cam1" / "Frame_0.jpg")
    img_two = Image.new("RGB", (200, 100), (40, 50, 60))
    img_two.save(root / "images" / "recB" / "cam1" / "Frame_0.jpg")

    images = [
        {"id": 0, "width": 200, "height": 100, "recording": "recA",
         "file_name": "recA/cam1/Frame_0.jpg"},
        {"id": 1, "width": 200, "height": 100, "recording": "recB",
         "file_name": "recB/cam1/Frame_0.jpg"},
    ]
    annotations = [
        {"id": 0, "image_id": 0, "bbox": [50.0, 20.0, 40.0, 30.0],
         "sex": "male", "fly_id": 0, "src_ann_id": 0},
        {"id": 1, "image_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0],
         "sex": "unknown", "fly_id": 0, "src_ann_id": 1},
        {"id": 2, "image_id": 1, "bbox": [150.0, 60.0, 20.0, 20.0],
         "sex": "unknown", "fly_id": 1, "src_ann_id": 2},
    ]
    coco = {"keypoint_names": [], "skeleton": [], "categories": [],
            "images": images, "annotations": annotations, "framesets": []}
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    (root / "annotations" / "instances_val.json").write_text(json.dumps(coco))

    manifest = {"version": 1, "calib_groups": {}, "recordings": {
        "recA": {"sex": "male", "split": "train"},
        "recB": {"sex": "unknown", "split": "train",
                 "fly_sex": {"fly0": "female", "fly1": "male"}},
    }}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return str(root)


def _make_fixture_with_zero_ann(tmp_path):
    """Same as `_make_fixture` plus a THIRD image (recC) that has media but
    ZERO annotations -- the label-defect case (unannotated, not verified
    empty) this dataset must exclude rather than treat as a background
    negative."""
    root = _make_fixture(tmp_path)
    (root_p := __import__("pathlib").Path(root))
    (root_p / "images" / "recC" / "cam1").mkdir(parents=True)
    Image.new("RGB", (200, 100), (70, 80, 90)).save(
        root_p / "images" / "recC" / "cam1" / "Frame_0.jpg")
    for split in ("train", "val"):
        ann_path = root_p / "annotations" / f"instances_{split}.json"
        coco = json.load(open(ann_path))
        coco["images"].append({"id": 2, "width": 200, "height": 100,
                               "recording": "recC",
                               "file_name": "recC/cam1/Frame_0.jpg"})
        # deliberately NO annotation added for image id 2
        json.dump(coco, open(ann_path, "w"))
    return root


def test_zero_annotation_images_are_excluded_not_kept_as_negatives(tmp_path):
    root = _make_fixture_with_zero_ann(tmp_path)
    with pytest.warns(UserWarning, match="dropped"):
        ds = V5CenterDetectDataset(root, "train")
    assert ds.n_dropped_zero_ann == 1
    assert "recC/cam1/Frame_0.jpg" not in ds.file_names
    assert len(ds) == 2                        # still just the 2 real (1-fly, 2-fly) images
    assert set(ds.num_flies) == {"1", "2"}      # never "0"
    assert ds.class_counts(key="num_flies") == {"1": 1, "2": 1}


def test_groups_by_image_not_by_annotation(tmp_path):
    root = _make_fixture(tmp_path)
    ds = V5CenterDetectDataset(root, "train")
    assert len(ds) == 2                       # 2 IMAGES, not 3 annotations
    assert sorted(ds.num_flies) == ["1", "2"]
    assert ds.two_fly_indices() == [ds.num_flies.index("2")]


def test_getitem_shapes_and_resize(tmp_path):
    root = _make_fixture(tmp_path)
    ds = V5CenterDetectDataset(root, "train", image_size=64)
    for i in range(len(ds)):
        img, centers, valid = ds[i]
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        assert centers.shape == (K_MAX, 2)
        assert valid.shape == (K_MAX,)
        assert valid.sum() == int(ds.num_flies[i])


def test_center_matches_bbox_center_rescaled(tmp_path):
    root = _make_fixture(tmp_path)
    ds = V5CenterDetectDataset(root, "train", image_size=64)
    i = ds.file_names.index("recA/cam1/Frame_0.jpg")
    img, centers, valid = ds[i]
    # bbox [50,20,40,30] on a 200x100 image -> center (70, 35); resized to
    # 64x64 -> scale (64/200, 64/100).
    assert valid[0] and not valid[1]
    expected = np.array([70.0 * 64 / 200.0, 35.0 * 64 / 100.0])
    assert np.allclose(centers[0], expected, atol=1e-3)


def test_sex_resolved_via_manifest_fly_sex_for_two_fly_recording(tmp_path):
    root = _make_fixture(tmp_path)
    ds = V5CenterDetectDataset(root, "train")
    i = ds.file_names.index("recB/cam1/Frame_0.jpg")
    assert sorted(ds.sexes[i]) == ["female", "male"]   # fly0=female, fly1=male


def test_more_than_k_max_annotations_raises(tmp_path):
    root = _make_fixture(tmp_path)
    ann_path = os.path.join(root, "annotations", "instances_train.json")
    coco = json.load(open(ann_path))
    coco["annotations"].append(
        {"id": 3, "image_id": 1, "bbox": [1.0, 1.0, 5.0, 5.0], "sex": "unknown",
         "fly_id": 2, "src_ann_id": 3})
    json.dump(coco, open(ann_path, "w"))
    with pytest.raises(ValueError, match="K_MAX"):
        V5CenterDetectDataset(root, "train")


def test_balanced_weights_reused_and_functional(tmp_path):
    root = _make_fixture(tmp_path)
    ds = V5CenterDetectDataset(root, "train")
    w = ds.balanced_weights(key="num_flies", alpha=1.0)
    assert np.isclose(w.sum(), 1.0)
    counts = ds.class_counts(key="num_flies")
    assert counts == {"1": 1, "2": 1}


def test_batches_yields_expected_shapes(tmp_path):
    root = _make_fixture(tmp_path)
    ds = V5CenterDetectDataset(root, "train", image_size=32)
    got = list(batches(ds, batch_size=2, shuffle=False, drop_last=True, num_workers=1))
    assert len(got) == 1
    imgs, centers, valid = got[0]
    assert imgs.shape == (2, 32, 32, 3)
    assert centers.shape == (2, K_MAX, 2)
    assert valid.shape == (2, K_MAX)


def test_batches_weighted_resampling_oversamples_rare_class(tmp_path):
    root = _make_fixture(tmp_path)
    ds = V5CenterDetectDataset(root, "train", image_size=16)
    weights = ds.balanced_weights(key="num_flies", alpha=1.0)   # full parity: 50/50
    idx_two = ds.num_flies.index("2")
    n = 4000
    rng = np.random.default_rng(0)
    draws = rng.choice(len(ds), size=n, replace=True, p=weights)
    frac_two = (draws == idx_two).mean()
    assert abs(frac_two - 0.5) < 0.03           # parity balancing -> ~50% draws


# ---------------------------------------------------------------------------
# Real-data guard: the actual numbers the task brief asks to be reported.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(V5_VALFIX_ROOT),
                    reason=f"real v5_valfix root not present: {V5_VALFIX_ROOT}")
def test_real_v5_valfix_counts_and_oversampling_repeat_factors():
    train = V5CenterDetectDataset(V5_VALFIX_ROOT, "train")
    val = V5CenterDetectDataset(V5_VALFIX_ROOT, "val")

    n_train_two = sum(1 for n in train.num_flies if n == "2")
    n_val_two = val.two_fly_indices()
    print(f"\n[v5_valfix] train: {len(train)} imgs, {n_train_two} two-fly")
    print(f"[v5_valfix] val:   {len(val)} imgs, {len(n_val_two)} two-fly")

    val_sexes_two = [s for i in n_val_two for s in val.sexes[i]]
    from collections import Counter
    print(f"[v5_valfix] val two-fly instance sex counts: {Counter(val_sexes_two)}")

    counts = train.class_counts(key="num_flies")
    print(f"[v5_valfix] train num_flies class_counts: {counts}")
    for alpha, max_repeat in ((0.5, 20.0),):
        weights = train.balanced_weights(key="num_flies", alpha=alpha, max_repeat=max_repeat)
        share = {c: 0.0 for c in counts}
        for w, lab in zip(weights, train.num_flies):
            share[lab] += float(w)
        print(f"[v5_valfix] balanced sampling (num_flies, alpha={alpha}, "
              f"max_repeat={max_repeat}):")
        for c in sorted(counts, key=lambda c: -counts[c]):
            n_c = counts[c]
            print(f"    {c}fly {n_c:>6} imgs ({100 * n_c / len(train):5.1f}% of data) "
                  f"-> {100 * share[c]:5.1f}% of samples "
                  f"({share[c] * len(train) / n_c:5.2f}x repeat)")

    assert len(train) > 0 and len(val) > 0
    assert n_train_two > 0
    assert len(n_val_two) > 0
