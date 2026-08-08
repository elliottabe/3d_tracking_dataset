# tests/test_v3_dataset_paths.py
"""Unit tests for V3Dataset image/mask path resolution across old-style
(V3, per-subset general_model) and new-style (V4, image-in-place with
subset/orig_split fields) COCO annotation records.

The pure-function tests (`_resolve_image_path` / `_resolve_mask_path`) need
no filesystem I/O and no real dataset. The `test_dataset_*` tests exercise
the constructor + `__getitem__` against fake tmp_path fixtures.
"""
import json
import os

import numpy as np
import pytest
from PIL import Image

from jarvis_jax.data.v3 import V3Dataset, _resolve_image_path, _resolve_mask_path


ROOT = "/some/root"
SOURCE_ROOT = "/some/source_root"


# --- _resolve_image_path -----------------------------------------------------

def test_old_style_record_resolves_under_root_split():
    img = {"id": 0, "file_name": "rec1/CamA/Frame_1.jpg"}
    path = _resolve_image_path(ROOT, "train", img, SOURCE_ROOT)
    assert path == os.path.join(ROOT, "train", "rec1/CamA/Frame_1.jpg")


def test_old_style_record_ignores_missing_source_root():
    # source_root is irrelevant for old-style records -- must not be required.
    img = {"id": 0, "file_name": "rec1/CamA/Frame_1.jpg"}
    path = _resolve_image_path(ROOT, "train", img, None)
    assert path == os.path.join(ROOT, "train", "rec1/CamA/Frame_1.jpg")


def test_new_style_record_resolves_under_source_root_subset_orig_split():
    img = {
        "id": 0,
        "file_name": "rec1/CamA/Frame_1.jpg",
        "subset": "S6male",
        "orig_split": "train",
    }
    path = _resolve_image_path(ROOT, "val", img, SOURCE_ROOT)
    # NOTE: orig_split (from the record) drives the path, NOT the dataset's
    # own train/val split -- that is the whole point of the re-split.
    assert path == os.path.join(SOURCE_ROOT, "S6male", "train", "rec1/CamA/Frame_1.jpg")


def test_new_style_record_missing_source_root_raises_clear_error():
    img = {
        "id": 0,
        "file_name": "rec1/CamA/Frame_1.jpg",
        "subset": "S6male",
        "orig_split": "train",
    }
    with pytest.raises(ValueError, match="source_root"):
        _resolve_image_path(ROOT, "val", img, None)


def test_new_style_record_requires_both_subset_and_orig_split():
    # Only `subset` present, `orig_split` missing -- should not silently
    # fall back to old-style behaviour (that would resolve to the wrong
    # file under `root`, not `source_root`).
    img = {"id": 0, "file_name": "rec1/CamA/Frame_1.jpg", "subset": "S6male"}
    with pytest.raises(ValueError, match="orig_split"):
        _resolve_image_path(ROOT, "val", img, SOURCE_ROOT)


# --- _resolve_mask_path derives consistently with _resolve_image_path ------

def test_old_style_mask_path_under_root_sam3_masks_split():
    img = {"id": 0, "file_name": "rec1/CamA/Frame_1.jpg"}
    mask_path = _resolve_mask_path(ROOT, "train", img, SOURCE_ROOT)
    assert mask_path == os.path.join(ROOT, "sam3_masks", "train", "rec1/CamA/Frame_1.npz")


def test_new_style_mask_path_under_source_root_subset_sam3_masks_orig_split():
    img = {
        "id": 0,
        "file_name": "rec1/CamA/Frame_1.jpg",
        "subset": "S6male",
        "orig_split": "train",
    }
    mask_path = _resolve_mask_path(ROOT, "val", img, SOURCE_ROOT)
    assert mask_path == os.path.join(
        SOURCE_ROOT, "S6male", "sam3_masks", "train", "rec1/CamA/Frame_1.npz")


def test_new_style_mask_path_missing_source_root_raises_clear_error():
    img = {
        "id": 0,
        "file_name": "rec1/CamA/Frame_1.jpg",
        "subset": "S6male",
        "orig_split": "train",
    }
    with pytest.raises(ValueError, match="source_root"):
        _resolve_mask_path(ROOT, "val", img, None)


# --- constructor-level behaviour (still filesystem-light via tmp_path) -----

def _make_fake_image(path, size=(500, 500)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


def _write_coco(path, images, annotations, info=None):
    path.write_text(json.dumps({
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "fly", "keypoints": [], "skeleton": []}],
        "info": info or {},
    }))


def test_dataset_getitem_old_style(tmp_path):
    root = tmp_path / "root"
    (root / "annotations").mkdir(parents=True)
    _make_fake_image(root / "val" / "rec1" / "img0.jpg")
    images = [{"id": 0, "file_name": "rec1/img0.jpg", "width": 500, "height": 500}]
    kp = [0.0, 0.0, 0] * 50
    annotations = [{
        "id": 0, "image_id": 0, "bbox": [10, 10, 20, 20], "keypoints": kp,
    }]
    _write_coco(root / "annotations" / "instances_val.json", images, annotations)

    ds = V3Dataset(str(root), "val")
    img4, kp_xy, vis = ds[0]
    assert img4.shape == (448, 448, 4)


def test_dataset_getitem_new_style_uses_source_root(tmp_path):
    root = tmp_path / "root"
    source_root = tmp_path / "source_root"
    (root / "annotations").mkdir(parents=True)
    # Image lives ONLY under source_root/<subset>/<orig_split>/..., NOT under
    # root/<split>/... -- proving resolution actually used subset/orig_split.
    _make_fake_image(source_root / "S6male" / "train" / "rec1" / "img0.jpg")

    images = [{
        "id": 0, "file_name": "rec1/img0.jpg", "width": 500, "height": 500,
        "subset": "S6male", "orig_split": "train",
    }]
    kp = [0.0, 0.0, 0] * 50
    annotations = [{
        "id": 0, "image_id": 0, "bbox": [10, 10, 20, 20], "keypoints": kp,
    }]
    _write_coco(root / "annotations" / "instances_val.json", images, annotations,
                info={"source_root": str(source_root)})

    ds = V3Dataset(str(root), "val")
    img4, kp_xy, vis = ds[0]
    assert img4.shape == (448, 448, 4)


def test_dataset_new_style_missing_source_root_raises_actionable_error(tmp_path):
    root = tmp_path / "root"
    (root / "annotations").mkdir(parents=True)
    images = [{
        "id": 0, "file_name": "rec1/img0.jpg", "width": 500, "height": 500,
        "subset": "S6male", "orig_split": "train",
    }]
    kp = [0.0, 0.0, 0] * 50
    annotations = [{
        "id": 0, "image_id": 0, "bbox": [10, 10, 20, 20], "keypoints": kp,
    }]
    _write_coco(root / "annotations" / "instances_val.json", images, annotations,
                info={})  # no source_root recorded

    ds = V3Dataset(str(root), "val")  # construction itself doesn't need files
    with pytest.raises(ValueError, match="source_root"):
        ds[0]


def test_dataset_constructor_accepts_explicit_source_root_override(tmp_path):
    # When the JSON doesn't carry source_root (e.g. an older build), the
    # constructor accepts one explicitly.
    root = tmp_path / "root"
    source_root = tmp_path / "source_root"
    (root / "annotations").mkdir(parents=True)
    _make_fake_image(source_root / "S6male" / "train" / "rec1" / "img0.jpg")

    images = [{
        "id": 0, "file_name": "rec1/img0.jpg", "width": 500, "height": 500,
        "subset": "S6male", "orig_split": "train",
    }]
    kp = [0.0, 0.0, 0] * 50
    annotations = [{
        "id": 0, "image_id": 0, "bbox": [10, 10, 20, 20], "keypoints": kp,
    }]
    _write_coco(root / "annotations" / "instances_val.json", images, annotations,
                info={})

    ds = V3Dataset(str(root), "val", source_root=str(source_root))
    img4, kp_xy, vis = ds[0]
    assert img4.shape == (448, 448, 4)
