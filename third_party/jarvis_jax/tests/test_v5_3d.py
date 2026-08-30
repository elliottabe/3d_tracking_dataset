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
