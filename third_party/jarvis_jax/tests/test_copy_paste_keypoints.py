"""CopyPasteKeypointDataset: host keypoints/mask untouched, donor keypoints
land where the donor was pasted, real two-fly rows pass through."""
import json
import numpy as np
from PIL import Image

from jarvis_jax.data.copy_paste import CopyPasteKeypointDataset
from jarvis_jax.data.distractor import DistractorKeypointDataset
from jarvis_jax.data.v5_2d import V5Dataset


def _root(tmp_path, img_w=200, img_h=120, crop=64):
    names = ["EyeL", "EyeR", "T1L_TaTip", "T1R_TaTip"]
    root = tmp_path
    (root / "annotations").mkdir()
    images, anns = [], []

    def add(rec, cam, frame, bbox, kps, color, img_id, ann_id, extra=None):
        d = root / "images" / rec / cam; d.mkdir(parents=True, exist_ok=True)
        arr = np.full((img_h, img_w, 3), 30, np.uint8)
        bx, by, bw, bh = bbox; arr[by:by + bh, bx:bx + bw] = color
        Image.fromarray(arr).save(d / f"{frame}.jpg")
        images.append({"id": img_id, "width": img_w, "height": img_h, "file_name": f"{rec}/{cam}/{frame}.jpg"})
        anns.append({"id": ann_id, "image_id": img_id, "bbox": list(bbox), "sex": "unknown", "fly_id": 0,
                     "src_ann_id": ann_id, "keypoints": [float(v) for xyv in kps for v in xyv], "num_keypoints": 4})
        m = np.zeros((img_h, img_w), bool); m[by:by + bh, bx:bx + bw] = True
        md = root / "masks" / rec / cam; md.mkdir(parents=True, exist_ok=True)
        np.savez(md / f"{frame}.npz", masks=m[None], ann_ids=np.array([ann_id]), matched=np.array([True]))

    # host: recA/cam1/Frame_0 single fly at (20,20)-(50,40); donor: recA/cam1/Frame_1 at (120,30)
    add("recA", "cam1", "Frame_0", (20, 20, 30, 20), [(25, 25, 2), (45, 25, 2), (22, 38, 2), (48, 38, 2)], 200, 0, 0)
    add("recA", "cam1", "Frame_1", (120, 30, 30, 20), [(125, 35, 2), (145, 35, 2), (122, 48, 2), (148, 48, 0)], 120, 1, 1)
    add("recB", "cam2", "Frame_0", (60, 40, 30, 20), [(65, 45, 2), (85, 45, 2), (62, 58, 2), (88, 58, 2)], 90, 2, 2)
    coco = {"keypoint_names": names, "images": images, "annotations": anns, "categories": []}
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    return root, crop


def test_paste_fills_distractor_rows_and_leaves_host_alone(tmp_path):
    # crop 160 so a donor 70-80 px from the host lands INSIDE the crop (the
    # real crop is 448 with separations 40-300); a 64 crop would exclude it
    root, crop = _root(tmp_path, crop=160)
    base = V5Dataset(str(root), "train", crop=crop, heatmap_size=crop // 2)
    inner = DistractorKeypointDataset(base)
    # separation 70-80 px: host (30 px wide) and donor (30 px wide) never overlap
    ds = CopyPasteKeypointDataset(inner, p=1.0, seed=0, sep_low=70, sep_high=80, sep_near_boundary=75)
    img_ref, kp_ref, vis_ref = inner[0]
    img4, kp, vis, meta = ds.sample_with_meta(0)
    K = 4
    np.testing.assert_array_equal(kp[:K], kp_ref[:K]); np.testing.assert_array_equal(vis[:K], vis_ref[:K])
    np.testing.assert_array_equal(img4[..., 3], img_ref[..., 3])          # mask channel untouched
    assert meta["donor_file_name"] == "recA/cam1/Frame_1.jpg"              # same camera, never cam2
    assert vis[K:].sum() == 3 and not vis[K + 3]                           # donor's v=0 point stays invisible
    # donor keypoints sit on pasted pixels (donor colour 120 blended by the
    # feathered alpha near the bbox edge; background is 30, host 200 is elsewhere)
    for k in range(K):
        if vis[K + k]:
            x, y = (kp[K + k] * 2).astype(int)                            # heatmap px -> crop px
            assert 60 < int(img4[y, x, 0]) < 160, (k, img4[y, x, 0])
    # RGB did change somewhere (a paste happened) but, with no overlap, not at the host's own keypoints
    assert meta["sep_achieved_px"] >= 60
    assert (img4[..., :3] != img_ref[..., :3]).any()
    for k in range(K):
        x, y = (kp[k] * 2).astype(int)
        assert img4[y, x, 0] == img_ref[y, x, 0]


def test_p_zero_is_identity_and_len_forwarded(tmp_path):
    root, crop = _root(tmp_path)
    inner = DistractorKeypointDataset(V5Dataset(str(root), "train", crop=crop, heatmap_size=crop // 2))
    ds = CopyPasteKeypointDataset(inner, p=0.0)
    assert len(ds) == 3
    a, b = inner[0], ds[0]
    np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
