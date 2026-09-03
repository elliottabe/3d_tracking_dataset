"""Distractor-aware supervision: same-part index, repulsion footprint, and the
dataset wrapper that appends the OTHER fly's keypoints to every sample.

Why: on the v12 mask-off detector the tarsal-tip misses land on the other
fly's leg tips / floor reflections while the true tip barely responds
(docs/benchmark/2026-09-03-maskoff-attention). The repulsion target says
"the other fly's TaTip is NOT your TaTip" at the pixels where that confusion
happens; the dataset wrapper is what carries those pixels to the loss.
"""
import json
import os

import numpy as np
import jax.numpy as jnp
import pytest
from PIL import Image

from jarvis_jax.data.distractor import (
    build_part_index, part_name, repulsion_footprint,
    DistractorKeypointDataset, DistractorGrayFillDataset)
from jarvis_jax.data.transforms import crop_origin, transform_keypoints

NAMES = ["Antenna_Base", "EyeL", "EyeR", "Scutellum", "WingL_V12", "WingR_V12",
         "WingL_V13", "T1L_TaTip", "T2R_TaTip", "T3L_TaT3"]


def test_part_name_strips_side_and_leg_only():
    assert part_name("T1L_TaTip") == "TaTip" == part_name("T2R_TaTip")
    assert part_name("T3L_TaT3") == "TaT3"
    assert part_name("EyeL") == "Eye" == part_name("EyeR")
    assert part_name("WingL_V12") == "Wing_V12" == part_name("WingR_V12")
    assert part_name("WingL_V13") == "Wing_V13"
    assert part_name("Scutellum") == "Scutellum"
    assert part_name("Antenna_Base") == "Antenna_Base"


def test_build_part_index_groups_same_part_across_legs_and_sides():
    part_of_k, parts = build_part_index(NAMES)
    assert part_of_k.shape == (len(NAMES),) and part_of_k.dtype == np.int32
    k = {n: i for i, n in enumerate(NAMES)}
    assert part_of_k[k["T1L_TaTip"]] == part_of_k[k["T2R_TaTip"]]
    assert part_of_k[k["EyeL"]] == part_of_k[k["EyeR"]]
    assert part_of_k[k["WingL_V12"]] == part_of_k[k["WingR_V12"]]
    assert part_of_k[k["WingL_V12"]] != part_of_k[k["WingL_V13"]]
    assert part_of_k[k["T1L_TaTip"]] != part_of_k[k["T3L_TaT3"]]
    assert len(parts) == part_of_k.max() + 1 == 7


def test_repulsion_footprint_lands_on_same_part_channels_only():
    part_of_k, _ = build_part_index(NAMES)
    K = len(NAMES); H = 32
    kp = np.zeros((1, K, 2), np.float32); vis = np.zeros((1, K), bool)
    k = {n: i for i, n in enumerate(NAMES)}
    kp[0, k["T1L_TaTip"]] = (10, 12); vis[0, k["T1L_TaTip"]] = True     # distractor's T1L tip
    fp = np.asarray(repulsion_footprint(jnp.asarray(kp), jnp.asarray(vis), part_of_k,
                                        heatmap_size=H, sigma=2.0))
    assert fp.shape == (1, H, H, K)
    # every TaTip channel of the TARGET is repelled at the distractor's tip ...
    assert fp[0, 12, 10, k["T2R_TaTip"]] > 0.99
    assert fp[0, 12, 10, k["T1L_TaTip"]] > 0.99
    # ... and no unrelated channel is
    assert fp[0, :, :, k["EyeL"]].max() == 0.0
    assert fp[0, :, :, k["T3L_TaT3"]].max() == 0.0
    # invisible distractor keypoints contribute nothing
    vis[:] = False
    fp0 = np.asarray(repulsion_footprint(jnp.asarray(kp), jnp.asarray(vis), part_of_k,
                                         heatmap_size=H, sigma=2.0))
    assert fp0.max() == 0.0


# ---------------------------------------------------------------- fixture
def _make_root(tmp_path, *, crop=64, img_w=160, img_h=120):
    """One two-fly image (anns 0,1) and one single-fly image (ann 2), 4 kps."""
    names = ["EyeL", "EyeR", "T1L_TaTip", "T1R_TaTip"]
    root = tmp_path
    (root / "images" / "recA" / "cam1").mkdir(parents=True)
    (root / "annotations").mkdir()
    for f in ("Frame_0", "Frame_1"):
        Image.fromarray(np.full((img_h, img_w, 3), 40, np.uint8)).save(
            root / "images" / "recA" / "cam1" / f"{f}.jpg")
    images = [{"id": 0, "width": img_w, "height": img_h, "file_name": "recA/cam1/Frame_0.jpg"},
              {"id": 1, "width": img_w, "height": img_h, "file_name": "recA/cam1/Frame_1.jpg"}]

    def ann(aid, img, bbox, kps):
        return {"id": aid, "image_id": img, "bbox": list(bbox), "sex": "unknown", "fly_id": 0,
                "keypoints": [float(v) for xyv in kps for v in xyv], "num_keypoints": 4}
    kp_a = [(30, 30, 2), (34, 30, 2), (20, 50, 2), (44, 50, 2)]
    kp_b = [(60, 40, 2), (64, 40, 2), (50, 58, 2), (74, 58, 0)]       # one invisible
    kp_c = [(80, 60, 2), (84, 60, 2), (70, 80, 2), (94, 80, 2)]
    anns = [ann(0, 0, (16, 24, 32, 32), kp_a), ann(1, 0, (46, 34, 32, 30), kp_b),
            ann(2, 1, (66, 54, 32, 32), kp_c)]
    coco = {"keypoint_names": names, "images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "fly", "keypoints": names}]}
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    (root / "annotations" / "keypoint_names.json").write_text(json.dumps(names))
    return root, kp_a, kp_b, kp_c, crop


def test_distractor_dataset_appends_other_fly_keypoints_in_crop_coords(tmp_path):
    from jarvis_jax.data.v5_2d import V5Dataset
    root, kp_a, kp_b, kp_c, crop = _make_root(tmp_path)
    base = V5Dataset(str(root), "train", crop=crop, heatmap_size=crop // 2)
    ds = DistractorKeypointDataset(base)
    assert len(ds) == 3 and ds.has_distractor(0) and ds.has_distractor(1) and not ds.has_distractor(2)

    img4, kp, vis = ds[0]
    K = 4
    assert img4.shape == (crop, crop, 4) and kp.shape == (2 * K, 2) and vis.shape == (2 * K,)
    # primary rows are untouched
    img_b, kp_b0, vis_b0 = base[0]
    np.testing.assert_array_equal(kp[:K], kp_b0); np.testing.assert_array_equal(vis[:K], vis_b0)
    # distractor rows are ann 1's keypoints through the HOST's crop
    x0, y0 = crop_origin(base.bboxes[0], 160, 120, crop)
    exp_xy, exp_v = transform_keypoints(np.asarray(kp_b, np.float32), x0, y0, crop, crop // 2)
    np.testing.assert_allclose(kp[K:][exp_v], exp_xy[exp_v])
    np.testing.assert_array_equal(vis[K:], exp_v)
    assert not vis[K + 3]                       # ann 1's T1R_TaTip is v=0

    # single-fly image: distractor rows all invisible
    _, kp2, vis2 = ds[2]
    assert not vis2[K:].any()


def test_grayfill_dataset_passes_keypoints_through_and_fills_only_with_masks(tmp_path):
    from jarvis_jax.data.v5_2d import V5Dataset
    root, *_ , crop = _make_root(tmp_path)
    base = V5Dataset(str(root), "train", crop=crop, heatmap_size=crop // 2)
    ds = DistractorGrayFillDataset(DistractorKeypointDataset(base), str(root), p=1.0, seed=0)
    img4, kp, vis = ds[0]
    ref, kp_ref, vis_ref = DistractorKeypointDataset(base)[0]
    assert kp.shape == (8, 2)
    np.testing.assert_array_equal(kp, kp_ref); np.testing.assert_array_equal(vis, vis_ref)
    # no mask npz on disk -> nothing to fill, image unchanged, counted
    np.testing.assert_array_equal(img4, ref)
    assert ds.n_no_distractor == 1 and ds.n_filled == 0


def test_grayfill_resolves_own_mask_by_src_ann_id(tmp_path):
    """Regression: a mask npz keyed by src_ann_id (not the merged id) must be
    recognised as the target's OWN mask -- the first version treated it as a
    distractor and erased the fly on 1187 S8_male_R_amp train rows and the
    whole headless_22_50_female val recording (72 px under the fill)."""
    from jarvis_jax.data.v5_2d import V5Dataset
    root, kp_a, kp_b, kp_c, crop = _make_root(tmp_path)
    # rewrite annotations with src_ann_id != id, and write npz keyed by src ids
    ann_path = root / "annotations" / "instances_train.json"
    coco = json.loads(ann_path.read_text())
    for a in coco["annotations"]:
        a["id"] = a["id"] + 1000; a["src_ann_id"] = a["id"] - 1000
    ann_path.write_text(json.dumps(coco))
    (root / "masks" / "recA" / "cam1").mkdir(parents=True)
    m = np.zeros((120, 160), bool)
    m0 = m.copy(); m0[24:56, 16:48] = True           # ann 1000 (src 0) fly A
    m1 = m.copy(); m1[34:64, 46:78] = True           # ann 1001 (src 1) fly B
    np.savez(root / "masks" / "recA" / "cam1" / "Frame_0.npz", masks=np.stack([m0, m1]),
             ann_ids=np.array([0, 1]), matched=np.array([True, True]))
    m2 = m.copy(); m2[54:86, 66:98] = True           # ann 1002 (src 2), single fly
    np.savez(root / "masks" / "recA" / "cam1" / "Frame_1.npz", masks=m2[None],
             ann_ids=np.array([2]), matched=np.array([True]))
    # paint fly B bright so a fill (crop-mean grey) is visible; the fixture jpg is flat
    arr = np.full((120, 160, 3), 40, np.uint8); arr[m1] = 200
    Image.fromarray(arr).save(root / "images" / "recA" / "cam1" / "Frame_0.jpg")
    base = V5Dataset(str(root), "train", crop=crop, heatmap_size=crop // 2)
    ds = DistractorGrayFillDataset(base, str(root), p=1.0, seed=0, dilate=2, protect=4)
    # single-fly image: its own (src-keyed) mask is NOT a distractor -> unchanged
    assert ds.fill(base[2][0], 2) is None
    # two-fly image: only fly B's pixels change for target A
    img4, _, _ = base[0]
    out = ds.fill(img4, 0)
    assert out is not None
    changed = (out[..., :3] != img4[..., :3]).any(-1)
    x0, y0 = crop_origin(base.bboxes[0], 160, 120, crop)
    own = m0[y0:y0 + crop, x0:x0 + crop]
    assert not changed[own[:changed.shape[0], :changed.shape[1]]].any(), "the target's own mask was filled"
    assert changed.any()
