"""Tests for ``jarvis_jax.data.copy_paste.CopyPasteCenterDetectDataset``.

All synthetic fixtures (small, real tiny JPEGs + real mask .npz files on
disk so __getitem__/_synthesize are exercised end to end) -- no real dataset
root or GPU needed.
"""
import json
import os

import numpy as np
import pytest
from PIL import Image

from jarvis_jax.data.copy_paste import (
    CopyPasteCenterDetectDataset,
    DEFAULT_SEP_HIGH_PX,
    DEFAULT_SEP_LOW_PX,
    DEFAULT_SEP_NEAR_BOUNDARY_PX,
    DEFAULT_SEP_NEAR_FRAC,
    _composite,
    _feathered_sprite,
    sample_separation_px,
)
from jarvis_jax.data.v5_centerdetect import V5CenterDetectDataset


def _write_mask_npz(path, mask_bool, ann_id):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path,
             masks=mask_bool[None].astype(bool),
             bboxes=np.zeros((1, 4), dtype=np.float32),
             scores=np.ones((1,), dtype=np.float32),
             ann_ids=np.array([ann_id], dtype=np.int64),
             kp_coverage=np.ones((1,), dtype=np.float32),
             matched=np.array([True]),
             extra_masks=np.zeros((0,) + mask_bool.shape, dtype=bool),
             extra_boxes=np.zeros((0, 4), dtype=np.float32),
             extra_scores=np.zeros((0,), dtype=np.float32))


def _make_fixture(tmp_path, *, img_w=200, img_h=100, offset=(0, 0)):
    """recA/cam1: 2 single-fly images (Frame_0, Frame_1) -- same-recording
    donor pool available for each other.
    recC/cam1: 1 single-fly image (Frame_0) -- no same-recording donor, must
    fall back to the same-camera (recA) pool.
    recB/cam2: 1 single-fly image, a DIFFERENT camera -- must never be
    selected as a donor for any cam1 host.
    recD/cam1: 1 real TWO-fly image -- must pass through untouched.
    """
    root = tmp_path
    images, annotations = [], []
    img_id = 0
    ann_id = 0
    ox, oy = offset

    def add_single(rec, cam, bbox, color):
        nonlocal img_id, ann_id
        (root / "images" / rec / cam).mkdir(parents=True, exist_ok=True)
        arr = np.full((img_h, img_w, 3), 30, dtype=np.uint8)
        Image.fromarray(arr).save(root / "images" / rec / cam / "Frame_0.jpg")
        images.append({"id": img_id, "width": img_w, "height": img_h,
                       "recording": rec, "file_name": f"{rec}/{cam}/Frame_0.jpg"})
        annotations.append({"id": ann_id, "image_id": img_id, "bbox": list(bbox),
                            "sex": "unknown", "fly_id": 0, "src_ann_id": ann_id})
        mask = np.zeros((img_h, img_w), dtype=bool)
        bx, by, bw, bh = [int(round(v)) for v in bbox]
        mask[by:by + bh, bx:bx + bw] = True
        _write_mask_npz(str(root / "masks" / rec / cam / "Frame_0.npz"), mask, ann_id)
        this_img_id, this_ann_id = img_id, ann_id
        img_id += 1
        ann_id += 1
        return this_img_id, this_ann_id

    add_single("recA", "cam1", (20 + ox, 20 + oy, 30, 20), 80)
    # second recA/cam1 frame needs its own file, so give it Frame_1 by
    # writing directly (add_single always writes Frame_0 -- inline here).
    (root / "images" / "recA" / "cam1").mkdir(parents=True, exist_ok=True)
    arr = np.full((img_h, img_w, 3), 30, dtype=np.uint8)
    Image.fromarray(arr).save(root / "images" / "recA" / "cam1" / "Frame_1.jpg")
    images.append({"id": img_id, "width": img_w, "height": img_h,
                   "recording": "recA", "file_name": "recA/cam1/Frame_1.jpg"})
    annotations.append({"id": ann_id, "image_id": img_id,
                        "bbox": [120 + ox, 30 + oy, 30, 20],
                        "sex": "unknown", "fly_id": 0, "src_ann_id": ann_id})
    mask = np.zeros((img_h, img_w), dtype=bool)
    mask[30 + oy:50 + oy, 120 + ox:150 + ox] = True
    _write_mask_npz(str(root / "masks" / "recA" / "cam1" / "Frame_1.npz"), mask, ann_id)
    img_id += 1
    ann_id += 1

    add_single("recC", "cam1", (60, 40, 30, 20), 90)   # no same-rec donor
    add_single("recB", "cam2", (60, 40, 30, 20), 100)  # different camera -- must never be chosen

    # recD/cam1: real two-fly image.
    (root / "images" / "recD" / "cam1").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((img_h, img_w, 3), 30, dtype=np.uint8)).save(
        root / "images" / "recD" / "cam1" / "Frame_0.jpg")
    images.append({"id": img_id, "width": img_w, "height": img_h,
                   "recording": "recD", "file_name": "recD/cam1/Frame_0.jpg"})
    annotations.append({"id": ann_id, "image_id": img_id, "bbox": [10, 10, 20, 20],
                        "sex": "unknown", "fly_id": 0, "src_ann_id": ann_id})
    ann_id += 1
    annotations.append({"id": ann_id, "image_id": img_id, "bbox": [150, 60, 20, 20],
                        "sex": "unknown", "fly_id": 1, "src_ann_id": ann_id})
    two_fly_img_id = img_id
    ann_id += 1
    img_id += 1

    (root / "annotations").mkdir()
    coco = {"keypoint_names": [], "skeleton": [], "categories": [],
            "images": images, "annotations": annotations, "framesets": []}
    for split in ("train", "val"):
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(coco))
    manifest = {"version": 1, "calib_groups": {}, "recordings": {
        r: {"sex": "unknown", "split": "train"} for r in ("recA", "recB", "recC", "recD")
    }}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return str(root), two_fly_img_id


def _train_ds(root, image_size=64):
    return V5CenterDetectDataset(root, "train", image_size=image_size)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_sample_separation_range_and_low_end_weight():
    rng = np.random.default_rng(0)
    draws = np.array([sample_separation_px(rng) for _ in range(20000)])
    assert draws.min() >= DEFAULT_SEP_LOW_PX
    assert draws.max() <= DEFAULT_SEP_HIGH_PX
    frac_below_boundary = (draws < DEFAULT_SEP_NEAR_BOUNDARY_PX).mean()
    assert abs(frac_below_boundary - DEFAULT_SEP_NEAR_FRAC) < 0.02
    # Real weight at the low end: median well below the midpoint of the
    # full [30, 400] range.
    assert np.median(draws) < (DEFAULT_SEP_LOW_PX + DEFAULT_SEP_HIGH_PX) / 2.0


def test_feathered_sprite_is_not_a_hard_edge():
    mask = np.zeros((60, 60), dtype=bool)
    mask[20:40, 20:40] = True
    rgb = np.zeros((60, 60, 3), dtype=np.uint8)
    rgb[mask] = 200
    crop_rgb, alpha, center, origin = _feathered_sprite(
        rgb, mask, (20, 20, 20, 20), pad=15, sigma=3.0)
    # Interior of the mask should be ~1, well outside ~0, but there must be
    # SOME strictly-intermediate values near the boundary (feathered, not a
    # step function).
    assert alpha.max() > 0.9
    assert alpha.min() < 0.05
    n_intermediate = np.sum((alpha > 0.05) & (alpha < 0.95))
    assert n_intermediate > 0


def test_composite_clips_at_frame_edge_without_raising():
    base = np.full((50, 50, 3), 10, dtype=np.uint8)
    sprite_rgb = np.full((20, 20, 3), 250, dtype=np.uint8)
    sprite_alpha = np.ones((20, 20), dtype=np.float32)
    out = _composite(base, sprite_rgb, sprite_alpha, (-10, -10))
    assert out.shape == base.shape
    # Only the in-bounds quadrant of the sprite should have been applied.
    assert out[0, 0, 0] == 250
    assert out[15, 15, 0] == 10  # outside the sprite's on-frame remainder


# ---------------------------------------------------------------------------
# CopyPasteCenterDetectDataset
# ---------------------------------------------------------------------------

def test_val_split_rejected(tmp_path):
    root, _ = _make_fixture(tmp_path)
    val_ds = V5CenterDetectDataset(root, "val", image_size=64)
    with pytest.raises(ValueError, match="TRAIN"):
        CopyPasteCenterDetectDataset(val_ds, p=1.0)


def test_two_fly_row_passes_through_untouched(tmp_path):
    root, two_fly_img_id = _make_fixture(tmp_path)
    ds = _train_ds(root)
    cp = CopyPasteCenterDetectDataset(ds, p=1.0, seed=0)
    i = ds.file_names.index("recD/cam1/Frame_0.jpg")
    img_a, centers_a, valid_a = ds[i]
    img_b, centers_b, valid_b = cp[i]
    assert np.array_equal(img_a, img_b)
    assert np.allclose(centers_a, centers_b)
    assert np.array_equal(valid_a, valid_b)


def test_never_pastes_across_cameras(tmp_path):
    root, _ = _make_fixture(tmp_path)
    ds = _train_ds(root)
    cp = CopyPasteCenterDetectDataset(ds, p=1.0, seed=0)
    i = ds.file_names.index("recA/cam1/Frame_0.jpg")
    seen_donor_cams = set()
    for seed in range(30):
        _, _, _, meta = cp.sample_with_meta(i, seed=seed)
        donor_cam = meta["donor_file_name"].split("/")[1]
        seen_donor_cams.add(donor_cam)
    assert seen_donor_cams == {"cam1"}   # never cam2 (recB)


def test_prefers_same_recording_when_available(tmp_path):
    root, _ = _make_fixture(tmp_path)
    ds = _train_ds(root)
    cp = CopyPasteCenterDetectDataset(ds, p=1.0, seed=0)
    i = ds.file_names.index("recA/cam1/Frame_0.jpg")
    for seed in range(10):
        _, _, _, meta = cp.sample_with_meta(i, seed=seed)
        assert meta["same_recording"] is True
        assert meta["donor_file_name"] == "recA/cam1/Frame_1.jpg"


def test_falls_back_to_same_camera_when_no_same_recording_donor(tmp_path):
    root, _ = _make_fixture(tmp_path)
    ds = _train_ds(root)
    cp = CopyPasteCenterDetectDataset(ds, p=1.0, seed=0)
    i = ds.file_names.index("recC/cam1/Frame_0.jpg")
    for seed in range(10):
        _, _, _, meta = cp.sample_with_meta(i, seed=seed)
        assert meta["same_recording"] is False
        assert meta["donor_file_name"].split("/")[1] == "cam1"
        assert meta["donor_file_name"].split("/")[0] != "recC"


def test_zorder_is_randomized_across_draws(tmp_path):
    root, _ = _make_fixture(tmp_path)
    ds = _train_ds(root)
    cp = CopyPasteCenterDetectDataset(ds, p=1.0, seed=0)
    i = ds.file_names.index("recA/cam1/Frame_0.jpg")
    tops = {cp.sample_with_meta(i, seed=s)[3]["top_is_donor"] for s in range(30)}
    assert tops == {True, False}


def test_getitem_shapes_when_synthesized(tmp_path):
    root, _ = _make_fixture(tmp_path)
    ds = _train_ds(root, image_size=48)
    cp = CopyPasteCenterDetectDataset(ds, p=1.0, seed=0)
    i = ds.file_names.index("recA/cam1/Frame_0.jpg")
    img, centers, valid = cp[i]
    assert img.shape == (48, 48, 3)
    assert img.dtype == np.uint8
    assert centers.shape == (2, 2)
    assert valid.tolist() == [True, True]


def test_p_zero_never_synthesizes(tmp_path):
    root, _ = _make_fixture(tmp_path)
    ds = _train_ds(root)
    cp = CopyPasteCenterDetectDataset(ds, p=0.0, seed=0)
    i = ds.file_names.index("recA/cam1/Frame_0.jpg")
    img_a, centers_a, valid_a = ds[i]
    img_b, centers_b, valid_b = cp[i]
    assert np.array_equal(img_a, img_b)
    assert valid_b.tolist() == [True, False]


def test_center_labels_self_consistent_and_unclamped_matches_sampled(tmp_path):
    # A large canvas so the clamp essentially never binds -- checks that the
    # achieved separation/direction reproduces the SAMPLED one when there is
    # room, i.e. the placement geometry (not just the clamp fallback) is
    # correct.
    # Offset so the host bbox sits well away from every edge -- otherwise
    # (as in the other tests, where the fixture's bboxes hug the origin
    # deliberately) the clamp binds on roughly 3/4 of directions just from
    # the host being near a corner, which is a fixture-geometry artifact,
    # not a placement-logic one.
    root, _ = _make_fixture(tmp_path, img_w=2000, img_h=1000, offset=(900, 450))
    ds = _train_ds(root)
    cp = CopyPasteCenterDetectDataset(ds, p=1.0, seed=0)
    i = ds.file_names.index("recA/cam1/Frame_0.jpg")
    n_unclamped = 0
    for seed in range(40):
        _, _, _, meta = cp.sample_with_meta(i, seed=seed)
        host = np.array(meta["host_center_xy"])
        target = np.array(meta["target_center_xy"])
        achieved = float(np.hypot(*(target - host)))
        assert abs(achieved - meta["sep_achieved_px"]) < 1e-6
        if abs(meta["sep_achieved_px"] - meta["sep_sampled_px"]) < 1.0:
            n_unclamped += 1
    assert n_unclamped > 30   # most draws unclamped on this big a canvas


def test_getattr_forwarding(tmp_path):
    root, _ = _make_fixture(tmp_path)
    ds = _train_ds(root)
    cp = CopyPasteCenterDetectDataset(ds, p=0.3, seed=0)
    assert cp.num_flies == ds.num_flies
    assert cp.class_counts(key="num_flies") == ds.class_counts(key="num_flies")
    assert len(cp) == len(ds)


def test_donor_never_the_host_itself(tmp_path):
    root, _ = _make_fixture(tmp_path)
    ds = _train_ds(root)
    cp = CopyPasteCenterDetectDataset(ds, p=1.0, seed=0)
    i = ds.file_names.index("recA/cam1/Frame_0.jpg")
    for seed in range(20):
        _, _, _, meta = cp.sample_with_meta(i, seed=seed)
        assert meta["donor_file_name"] != meta["host_file_name"]
