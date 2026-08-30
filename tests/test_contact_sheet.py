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


def _tiny_v5_with_unresolved_camera(tmp_path):
    """Like _tiny_v5, but the single frameset's Cam2012630 slot has
    ann_id=None -- exactly what merge_annotations (build_v5.py, Ruling R15)
    emits when a camera's per-frame fly count disagrees with the frameset
    max. Cam2012855 stays fully resolved so the fix can be proven: that cell
    must still render even though its sibling cell is unresolved.
    """
    root = tmp_path / "v5"
    (root / "images" / "rec_u" / "Cam2012630").mkdir(parents=True)
    (root / "images" / "rec_u" / "Cam2012855").mkdir(parents=True)
    for cam in ("Cam2012630", "Cam2012855"):
        Image.fromarray(np.zeros((448, 1936, 3), np.uint8)).save(
            root / "images" / "rec_u" / cam / "Frame_000000.jpg")
    kp_flat = [v for x, y in _KP_XY for v in (x, y, 2)]
    imgs = [
        {"id": 0, "width": 1936, "height": 448, "recording": "rec_u",
         "file_name": "rec_u/Cam2012630/Frame_000000.jpg"},
        {"id": 1, "width": 1936, "height": 448, "recording": "rec_u",
         "file_name": "rec_u/Cam2012855/Frame_000000.jpg"},
    ]
    anns = [
        {"id": 0, "image_id": 1, "bbox": [10.0, 10.0, 60.0, 60.0],
         "keypoints": kp_flat, "num_keypoints": len(_KP_NAMES),
         "sex": "unknown", "behavior": "courtship", "fly_id": 0},
    ]
    fs = {"rec_u/Frame_000000/fly0": {"recording": "rec_u", "fly_id": 0,
                                      "frames": [0, 1], "ann_ids": [None, 0]}}
    (root / "annotations").mkdir()
    (root / "annotations" / "instances.json").write_text(json.dumps({
        "keypoint_names": _KP_NAMES, "skeleton": [],
        "categories": [{"id": 1, "name": "fly"}],
        "images": imgs, "annotations": anns, "framesets": fs}))
    (root / "manifest.json").write_text(json.dumps(
        {"recordings": {"rec_u": {"sex": "unknown"}}}))
    return root


def test_contact_sheet_skips_unresolved_camera_but_renders_the_rest(tmp_path):
    """Ruling R15 (build_v5.merge_annotations) legitimately emits ann_id=None
    for a camera whose fly identity could not be resolved. Before the fix,
    `ann_by_id[None]` raised KeyError and the WHOLE frameset -- including the
    camera that DID resolve -- was lost. The fix must skip only the
    unresolved cell and still render the resolved one."""
    root = _tiny_v5_with_unresolved_camera(tmp_path)
    out = build_contact_sheet(str(root), "rec_u", n_frames=1,
                              cams=("Cam2012630", "Cam2012855"),
                              out_dir=str(tmp_path / "sheets"))
    arr = np.asarray(Image.open(out).convert("RGB"))
    cell = 256
    # row 0 = Cam2012630 (unresolved: ann_id None) -> must be left as the
    # sheet's plain background fill, never crash on.
    background = (12, 12, 14)
    row0 = arr[0:cell, 0:cell]
    assert np.all(row0.reshape(-1, 3) == background), (
        "unresolved camera cell should be left as background, not crashed on")
    # row 1 = Cam2012855 (resolved) -> keypoints must still be drawn.
    scale = cell / 82  # bbox [10,10,60,60], pad=1.4 -> crop (0,0,82,82)
    for name, (x, y), group in zip(_KP_NAMES, _KP_XY,
                                    ("head", "thorax", "abdomen", "legs")):
        px, py = int(x * scale), int(y * scale) + cell
        got = tuple(int(c) for c in arr[py, px])
        assert got == PALETTE[group], (
            f"{name} not drawn on the resolved camera's row -- resolved "
            f"cameras must still render even when a sibling is unresolved")


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
