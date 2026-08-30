import json
import os
import numpy as np
import pytest
from PIL import Image

from scripts.viz.contact_sheet import build_contact_sheet
from jarvis_jax.data.build_v5 import apply_sex_labels


def _tiny_v5(tmp_path):
    root = tmp_path / "v5"
    (root / "images" / "rec_a" / "Cam2012630").mkdir(parents=True)
    (root / "images" / "rec_a" / "Cam2012855").mkdir(parents=True)
    for cam in ("Cam2012630", "Cam2012855"):
        for i in range(3):
            Image.fromarray(np.zeros((448, 1936, 3), np.uint8)).save(
                root / "images" / "rec_a" / cam / f"Frame_{i:06d}.jpg")
    imgs, anns, fs = [], [], {}
    iid = 0
    for i in range(3):
        ids = []
        for cam in ("Cam2012630", "Cam2012855"):
            imgs.append({"id": iid, "width": 1936, "height": 448, "recording": "rec_a",
                         "file_name": f"rec_a/{cam}/Frame_{i:06d}.jpg"})
            anns.append({"id": iid, "image_id": iid, "bbox": [10.0, 10.0, 60.0, 60.0],
                         "keypoints": [20.0, 20.0, 2] * 50, "num_keypoints": 50,
                         "sex": "unknown", "behavior": "courtship", "fly_id": 0})
            ids.append(iid)
            iid += 1
        fs[f"rec_a/Frame_{i:06d}/fly0"] = {"recording": "rec_a", "fly_id": 0,
                                           "frames": ids, "ann_ids": ids}
    (root / "annotations").mkdir()
    (root / "annotations" / "instances.json").write_text(json.dumps({
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "fly"}],
        "images": imgs, "annotations": anns, "framesets": fs}))
    (root / "manifest.json").write_text(json.dumps(
        {"recordings": {"rec_a": {"sex": "unknown"}}}))
    return root


def test_contact_sheet_written_and_non_blank(tmp_path):
    root = _tiny_v5(tmp_path)
    out = build_contact_sheet(str(root), "rec_a", n_frames=2,
                              cams=("Cam2012630", "Cam2012855"),
                              out_dir=str(tmp_path / "sheets"))
    assert os.path.exists(out)
    arr = np.asarray(Image.open(out).convert("RGB"))
    assert arr.max() > 0, "sheet is entirely black — keypoints were not drawn"


def test_apply_sex_labels_updates_manifest(tmp_path):
    root = _tiny_v5(tmp_path)
    man = apply_sex_labels(str(root), {"rec_a": "female"})
    assert man["recordings"]["rec_a"]["sex"] == "female"
    assert json.load(open(root / "manifest.json"))["recordings"]["rec_a"]["sex"] == "female"


def test_apply_sex_labels_rejects_bad_value(tmp_path):
    root = _tiny_v5(tmp_path)
    with pytest.raises(ValueError, match="sex must be"):
        apply_sex_labels(str(root), {"rec_a": "F"})


def test_apply_sex_labels_rejects_unknown_recording(tmp_path):
    root = _tiny_v5(tmp_path)
    with pytest.raises(KeyError, match="not in manifest"):
        apply_sex_labels(str(root), {"nope": "male"})
