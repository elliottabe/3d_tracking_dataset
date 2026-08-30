import json
import os
import numpy as np
import pytest
from PIL import Image

from scripts.viz.contact_sheet import build_contact_sheet
from jarvis_jax.data.build_v5 import apply_sex_labels
from viz.core.colors import PALETTE

# One real keypoint name per body-region group (head/thorax/abdomen/legs),
# each placed at its own (x, y) so the four dots land at distinguishable,
# non-overlapping spots -- lets the test prove both that keypoints are drawn
# AND that each one gets its own group's PALETTE colour (not just "something
# non-black exists somewhere", which text captions alone would satisfy).
_KP_NAMES = ["EyeL", "Scutellum", "Abd_tip", "T1L_TaTip"]
_KP_XY = [(20.0, 20.0), (40.0, 20.0), (20.0, 40.0), (40.0, 40.0)]


def _tiny_v5(tmp_path):
    root = tmp_path / "v5"
    (root / "images" / "rec_a" / "Cam2012630").mkdir(parents=True)
    (root / "images" / "rec_a" / "Cam2012855").mkdir(parents=True)
    for cam in ("Cam2012630", "Cam2012855"):
        for i in range(3):
            Image.fromarray(np.zeros((448, 1936, 3), np.uint8)).save(
                root / "images" / "rec_a" / cam / f"Frame_{i:06d}.jpg")
    imgs, anns, fs = [], [], {}
    kp_flat = [v for x, y in _KP_XY for v in (x, y, 2)]
    iid = 0
    for i in range(3):
        ids = []
        for cam in ("Cam2012630", "Cam2012855"):
            imgs.append({"id": iid, "width": 1936, "height": 448, "recording": "rec_a",
                         "file_name": f"rec_a/{cam}/Frame_{i:06d}.jpg"})
            anns.append({"id": iid, "image_id": iid, "bbox": [10.0, 10.0, 60.0, 60.0],
                         "keypoints": kp_flat, "num_keypoints": len(_KP_NAMES),
                         "sex": "unknown", "behavior": "courtship", "fly_id": 0})
            ids.append(iid)
            iid += 1
        fs[f"rec_a/Frame_{i:06d}/fly0"] = {"recording": "rec_a", "fly_id": 0,
                                           "frames": ids, "ann_ids": ids}
    (root / "annotations").mkdir()
    (root / "annotations" / "instances.json").write_text(json.dumps({
        "keypoint_names": _KP_NAMES, "skeleton": [],
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

    # Genuine proof of drawing: the first cell (col=0 -> Frame_000000,
    # row=0 -> Cam2012630) must show each keypoint's dot in ITS OWN group's
    # PALETTE colour, at the location the bbox-crop math puts it. Before the
    # colour-lookup fix every keypoint fell back to the same white default
    # regardless of group, and "other"/"legs"/"abdomen" had no PALETTE entry
    # at all -- either defect would make this loop fail.
    # bbox [10,10,60,60] with pad=1.4 -> crop box (0,0,82,82); cell=256 ->
    # scale 256/82 for both axes.
    scale = 256 / 82
    expected = {name: (int(x * scale), int(y * scale), PALETTE[group])
                for name, (x, y), group in zip(
                    _KP_NAMES, _KP_XY, ("head", "thorax", "abdomen", "legs"))}
    for name, (px, py, colour) in expected.items():
        got = tuple(int(c) for c in arr[py, px])
        assert got == colour, (
            f"{name} ({'/'.join(map(str, (px, py)))}) expected colour {colour} "
            f"got {got} -- keypoint colour-coding is broken")


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
