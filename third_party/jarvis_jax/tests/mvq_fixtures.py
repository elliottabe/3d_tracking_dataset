# tests/mvq_fixtures.py
"""Synthetic v12-shaped root: 7 affine cameras (real calibration rotated per
camera), 1936x448 JPEGs with a bright blob per fly, framesets keyed
'<rec>/Frame_<n>/fly<k>', frames 0..n_frames-1 consecutive, and a second
fly labelled only in `two_fly_frame`."""
import json
import os
import numpy as np
from PIL import Image

CAMS = ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855",
        "Cam2012857", "Cam2012861", "Cam2012862"]
REC = "2026_01_01_00_00_00"
K = 50
P_REAL = np.array([[8.1001, 0.0074869, -0.031773, 900.0],
                   [0.0093308, -8.0788, -0.17912, 300.0],
                   [0.0, 0.0, 0.0, 1.0]], np.float64)


def _names():
    legs = [f"{s}{i}{lr}_{p}" for lr in "LR" for i in (1, 2, 3) for s in "T"
            for p in ("Tro", "FeTi", "TiTa", "TaT1", "TaT3", "TaTip")]
    head = ["Antenna_Base", "EyeL", "EyeR", "Scutellum", "Abd_A4", "Abd_tip",
            "WingL_base", "WingL_V12", "WingL_V13", "T1L_ThxCx"]
    right = ["WingR_base", "WingR_V12", "WingR_V13", "T1R_ThxCx"]
    names = head + legs[:18] + right + legs[18:]
    assert len(names) == K, len(names)
    return names


def cam_P(i):
    th = 2 * np.pi * i / 7
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    P = P_REAL.copy(); P[:2, :3] = P[:2, :3] @ Rz
    return P


def fly_points(fly, frame, rng):
    """(K,3) world points: a 20-unit body around a per-fly centre that drifts 1 unit/frame."""
    centre = np.array([0.0, 0.0, 0.0]) if fly == 0 else np.array([40.0, 5.0, 0.0])
    centre = centre + frame * np.array([1.0, 0.5, 0.0])
    return centre + rng.normal(size=(K, 3)) * np.array([10.0, 4.0, 2.0])


def make_v12_root(tmp_path, *, n_frames=3, two_fly_frame=1, img_w=1936, img_h=448):
    root = tmp_path / "v12"; root.mkdir()
    names = _names()
    (root / "calibrations" / "A").mkdir(parents=True)
    for i, c in enumerate(CAMS):
        vals = ", ".join(f"{v:.16g}" for v in cam_P(i).ravel())
        (root / "calibrations" / "A" / f"{c}.yaml").write_text(
            "%%YAML:1.0\n---\nimage_width: %d\nimage_height: %d\n"
            "projectionMatrix: !!opencv-matrix\n   rows: 3\n   cols: 4\n   dt: d\n"
            "   data: [ %s ]\nscale: 10\n" % (img_w, img_h, vals))
    rng = np.random.default_rng(0)
    images, anns, framesets = [], [], {}
    img_id = ann_id = 0
    for f in range(n_frames):
        flies = [0, 1] if f == two_fly_frame else [0]
        pts = {fly: fly_points(fly, f, rng) for fly in flies}
        per_fly = {fly: {"frames": [], "ann_ids": []} for fly in flies}
        for i, c in enumerate(CAMS):
            P = cam_P(i)
            img = np.zeros((img_h, img_w, 3), np.uint8)
            fn = f"{REC}/{c}/Frame_{f}.jpg"
            img_id += 1
            images.append({"id": img_id, "width": img_w, "height": img_h,
                           "recording": REC, "file_name": fn})
            mask_ids, mask_arrs = [], []
            for fly in flies:
                X = np.concatenate([pts[fly], np.ones((K, 1))], 1)
                uv = (X @ P.T)[:, :2]
                kps = np.stack([uv[:, 0], uv[:, 1], np.ones(K)], 1)
                cx, cy = int(uv[:, 0].mean()), int(uv[:, 1].mean())
                y0, y1 = max(cy - 12, 0), cy + 12
                x0, x1 = max(cx - 12, 0), cx + 12
                img[y0:y1, x0:x1] = 255
                m = np.zeros((img_h, img_w), np.uint8)
                m[y0:y1, x0:x1] = 1
                mask_ids.append(ann_id); mask_arrs.append(m)
                anns.append({"id": ann_id, "image_id": img_id,
                             "bbox": [float(uv[:, 0].min()), float(uv[:, 1].min()),
                                      float(np.ptp(uv[:, 0])), float(np.ptp(uv[:, 1]))],
                             "keypoints": [float(v) for v in kps.ravel()], "num_keypoints": K,
                             "sex": "female" if fly == 0 else "male", "fly_id": fly,
                             "subset": "synthetic", "src_ann_id": ann_id})
                per_fly[fly]["frames"].append(img_id); per_fly[fly]["ann_ids"].append(ann_id)
                ann_id += 1
            os.makedirs(root / "images" / REC / c, exist_ok=True)
            Image.fromarray(img).save(root / "images" / REC / c / f"Frame_{f}.jpg", quality=90)
            # mask npz, same layout _load_mask reads: masks/<rec>/<cam>/Frame_<f>.npz
            # with ann_ids/matched/masks keyed by each annotation's own `id` (== src_ann_id here)
            os.makedirs(root / "masks" / REC / c, exist_ok=True)
            np.savez(root / "masks" / REC / c / f"Frame_{f}.npz",
                     ann_ids=np.array(mask_ids, np.int64),
                     matched=np.ones(len(mask_ids), bool),
                     masks=np.stack(mask_arrs).astype(np.uint8))
        for fly in flies:
            framesets[f"{REC}/Frame_{f}/fly{fly}"] = {"recording": REC, "fly_id": fly,
                                                     "subset": "synthetic", **per_fly[fly]}
    coco = {"keypoint_names": names, "skeleton": [], "categories": [{"id": 1, "name": "fly"}],
            "images": images, "annotations": anns, "framesets": framesets}
    (root / "annotations").mkdir()
    for split in ("train", "val"):
        json.dump(coco, open(root / "annotations" / f"instances_{split}.json", "w"))
    json.dump(names, open(root / "annotations" / "keypoint_names.json", "w"))
    json.dump({"version": "synthetic", "recordings": {REC: {
        "calib_group": "A", "sex": "mixed", "behavior": "courtship", "n_flies": 2,
        "fly_sex": {"fly0": "female", "fly1": "male"}, "split": "train"}}},
              open(root / "manifest.json", "w"))
    return str(root)
