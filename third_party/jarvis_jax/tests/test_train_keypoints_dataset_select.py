"""Trainer-side dataset-class selection (`jarvis_jax/scripts/train_keypoints.py`).

`V5Dataset` (jarvis_jax/data/v5_2d.py) was added to read red_data_3d_v5's flat
layout, but nothing in the trainer ever constructed it -- `run_training` kept
hardcoding `V3Dataset`, which builds `root/<split>/<file_name>` paths that do
not exist in a v5 root (v5 has no `train/`/`val/` directory anywhere; the
split lives only in `annotations/instances_{split}.json`). A retrain pointed
at a v5 root therefore died with `FileNotFoundError` on the very first
sample.

These tests exercise `select_dataset_cls` -- the exact function
`run_training` calls to pick a class for `train_ds`/`val_ds`/`val_ds_all` --
against real on-disk fixtures shaped like each layout, and prove the object
IT RETURNS actually loads a real sample end to end. A test that only checks
`V5Dataset` exists (as `tests/test_v5_2d.py` already does) would not catch
this: the bug was entirely in the call site, not the new class.
"""
import json

import numpy as np
import pytest
from PIL import Image

from jarvis_jax.data.v3 import V3Dataset
from jarvis_jax.data.v5_2d import V5Dataset
from jarvis_jax.scripts.train_keypoints import select_dataset_cls

# Matches the real Cam image dims used by tests/test_v5_2d.py's fixture --
# large enough that the default crop=448 needs no special-casing.
W, H = 1936, 448


def _make_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (W, H), color=color).save(path)


def _coco_blob(file_name):
    kp = [50.0, 60.0, 1] * 50
    return {
        "keypoint_names": [f"kp{i}" for i in range(50)],
        "skeleton": [],
        "categories": [{"id": 1, "name": "fly", "keypoints": [], "skeleton": []}],
        "images": [{"id": 1, "file_name": file_name, "width": W, "height": H,
                    "recording": file_name.split("/")[0]}],
        "annotations": [
            {"id": 1, "image_id": 1, "bbox": [30, 30, 60, 60], "keypoints": kp,
             "num_keypoints": 50, "sex": "male", "behavior": "general"},
        ],
        "framesets": {},
    }


def _v5_shaped_root(tmp_path):
    """images/ + manifest.json at the root, NO train/ or val/ anywhere --
    the two markers `select_dataset_cls` keys off (see its docstring)."""
    root = tmp_path / "red_data_3d_v5"
    (root / "annotations").mkdir(parents=True)
    _make_image(root / "images" / "recA" / "Cam1" / "Frame_1.jpg", (30, 60, 90))
    (root / "manifest.json").write_text(json.dumps({"recordings": {"recA": {}}}))
    coco = _coco_blob("recA/Cam1/Frame_1.jpg")
    for split in ("train", "val"):
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(coco))
    return root


def _v3_shaped_root(tmp_path):
    """train/ + val/ directories, NO images/ dir and NO manifest.json at the
    root -- the classic V3/V4 layout that must keep resolving to V3Dataset."""
    root = tmp_path / "red_data_unified_V3"
    (root / "annotations").mkdir(parents=True)
    coco = _coco_blob("recA/Cam1/Frame_1.jpg")
    for split in ("train", "val"):
        _make_image(root / split / "recA" / "Cam1" / "Frame_1.jpg", (90, 60, 30))
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(coco))
    return root


# --- selection itself --------------------------------------------------------

def test_select_dataset_cls_picks_v5_for_v5_shaped_root(tmp_path):
    root = _v5_shaped_root(tmp_path)
    assert select_dataset_cls(str(root)) is V5Dataset


def test_select_dataset_cls_picks_v3_for_v3_shaped_root(tmp_path):
    root = _v3_shaped_root(tmp_path)
    assert select_dataset_cls(str(root)) is V3Dataset


# --- the trainer's actual construction pattern (run_training) ---------------

def test_trainer_construction_pattern_loads_a_real_v5_sample(tmp_path):
    """Exercises exactly what `run_training` does: pick a class, then build
    train_ds / per-recording val_ds / full val_ds_all from it -- against a
    v5-shaped root -- and confirm a real sample loads (this is the failure
    mode reported: construction succeeded, __getitem__ died)."""
    root = _v5_shaped_root(tmp_path)
    dataset_cls = select_dataset_cls(str(root))
    assert dataset_cls is V5Dataset
    train_ds = dataset_cls(str(root), "train")
    val_ds = dataset_cls(str(root), "val", recordings=["recA"])
    val_ds_all = dataset_cls(str(root), "val")
    assert len(train_ds) == 1 and len(val_ds) == 1 and len(val_ds_all) == 1
    img4, kp_xy, vis = train_ds[0]
    assert img4.shape == (448, 448, 4) and img4.dtype == np.uint8
    assert kp_xy.shape == (50, 2)
    assert vis.shape == (50,)
    # Attributes/methods the trainer reads beyond __len__/__getitem__ (see
    # train_keypoints.py's oversample.balance_key / weighted-sampling paths
    # and eval_mpjpe's ds.heatmap_size read).
    assert train_ds.heatmap_size == 224
    assert train_ds.class_counts("category") == {"unknown": 1}
    weights = train_ds.balanced_weights(key="behavior")
    assert weights.shape == (1,) and np.isclose(weights.sum(), 1.0)


def test_trainer_construction_pattern_loads_a_real_v3_sample(tmp_path):
    """Same construction pattern against a V3/V4-shaped root: must keep
    resolving to V3Dataset and loading unchanged."""
    root = _v3_shaped_root(tmp_path)
    dataset_cls = select_dataset_cls(str(root))
    assert dataset_cls is V3Dataset
    train_ds = dataset_cls(str(root), "train")
    val_ds = dataset_cls(str(root), "val", recordings=["recA"])
    val_ds_all = dataset_cls(str(root), "val")
    assert len(train_ds) == 1 and len(val_ds) == 1 and len(val_ds_all) == 1
    img4, kp_xy, vis = train_ds[0]
    assert img4.shape == (448, 448, 4) and img4.dtype == np.uint8


def test_v5_recordings_filter_excludes_other_recordings(tmp_path):
    """The `recordings=[...]` filter used for the per-recording val split
    (line ~238 in run_training) must actually filter on the v5 path too."""
    root = _v5_shaped_root(tmp_path)
    dataset_cls = select_dataset_cls(str(root))
    filtered = dataset_cls(str(root), "val", recordings=["some_other_recording"])
    assert len(filtered) == 0
