# tests/test_v5_2d.py
"""Unit tests for the v5 2D keypoint loader (`jarvis_jax.data.v5_2d.V5Dataset`).

red_data_3d_v5 has NO train/ or val/ directory anywhere -- the split lives
only in `annotations/instances_{train,val}.json`, and images/masks are
resolved flat: `images/<rec>/<cam>/Frame_N.jpg`, `masks/<rec>/<cam>/Frame_N.npz`.
`jarvis_jax.data.v3.V3Dataset` builds `root/<split>/<file_name>` paths, which
do not exist in this layout by design (see module docstring in v5_2d.py) --
the first test below reproduces that reported failure directly, before any
of the V5Dataset tests establish the fix.
"""
import json
import os

import numpy as np
import pytest
from PIL import Image

from jarvis_jax.data.v3 import V3Dataset


# --- v5-shaped fixture ------------------------------------------------------

def _make_image(path, w=200, h=150, color=(30, 60, 90)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), color=color).save(path)


def _make_mask_npz(path, ann_ids, masks, matched):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, masks=np.asarray(masks, dtype=np.uint8),
             ann_ids=np.asarray(ann_ids, dtype=np.int64),
             matched=np.asarray(matched, dtype=bool))


def _v5_root(tmp_path):
    """Build a v5-shaped tree: annotations/, images/, masks/, with NO train/
    or val/ directory anywhere (that is the whole point of v5's layout).

    Three annotations exercise the three cases the real dataset actually
    has, each verified against real red_data_3d_v5 data before writing this
    fixture:
      - ann 101 (image recA): its mask npz is keyed by `src_ann_id` (777) --
        the scheme used by masks borrowed from red_data_unified_V3.
      - ann 202 (image recB): its `src_ann_id` (999) does NOT appear in the
        npz; only the merged `id` (202) does -- the scheme used by masks
        generated fresh for the 9 newly-added recordings.
      - ann 303 (image recC): no mask npz exists at all for its image --
        must degrade to an all-zero mask channel, never raise. Its keypoints
        are also padded-with-an-absent-joint, mimicking the real
        headless/amputee recordings (kp array is always 50 long; joint 0
        here carries v=0).
    """
    root = tmp_path / "red_data_3d_v5"
    (root / "annotations").mkdir(parents=True)

    W, H = 1936, 448  # matches real Cam image dims (H == crop, exercises x0-clamping)
    _make_image(root / "images" / "recA" / "Cam1" / "Frame_1.jpg", W, H)
    mask_a = np.zeros((H, W), dtype=np.uint8)
    mask_a[40:80, 40:80] = 1
    _make_mask_npz(root / "masks" / "recA" / "Cam1" / "Frame_1.npz",
                   ann_ids=[777], masks=[mask_a], matched=[True])

    _make_image(root / "images" / "recB" / "Cam1" / "Frame_2.jpg", W, H)
    mask_b = np.zeros((H, W), dtype=np.uint8)
    mask_b[20:60, 20:60] = 1
    _make_mask_npz(root / "masks" / "recB" / "Cam1" / "Frame_2.npz",
                   ann_ids=[202], masks=[mask_b], matched=[True])

    _make_image(root / "images" / "recC" / "Cam1" / "Frame_3.jpg", W, H)
    # no masks/recC/... at all

    images = [
        {"id": 1, "file_name": "recA/Cam1/Frame_1.jpg", "width": W, "height": H,
         "recording": "recA"},
        {"id": 2, "file_name": "recB/Cam1/Frame_2.jpg", "width": W, "height": H,
         "recording": "recB"},
        {"id": 3, "file_name": "recC/Cam1/Frame_3.jpg", "width": W, "height": H,
         "recording": "recC"},
    ]

    kp_full = [50.0, 60.0, 1] * 50
    kp_padded = list(kp_full)
    kp_padded[2] = 0  # joint 0 absent (headless/amputee-style padding)

    annotations = [
        {"id": 101, "image_id": 1, "bbox": [30, 30, 60, 60], "keypoints": kp_full,
         "num_keypoints": 50, "sex": "male", "behavior": "general", "src_ann_id": 777},
        {"id": 202, "image_id": 2, "bbox": [10, 10, 60, 60], "keypoints": kp_full,
         "num_keypoints": 50, "sex": "female", "behavior": "courtship", "src_ann_id": 999},
        {"id": 303, "image_id": 3, "bbox": [5, 5, 60, 60], "keypoints": kp_padded,
         "num_keypoints": 49, "sex": "male", "behavior": "unknown", "src_ann_id": 303},
    ]

    coco = {
        "keypoint_names": [f"kp{i}" for i in range(50)],
        "skeleton": [],
        "categories": [{"id": 1, "name": "fly", "keypoints": [], "skeleton": []}],
        "images": images,
        "annotations": annotations,
        "framesets": {},
    }
    for split in ("train", "val"):
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(coco))
    return root


# --- reproduce the reported bug ---------------------------------------------

def test_v3dataset_cannot_read_v5_layout_by_design(tmp_path):
    """V3Dataset builds `root/<split>/<file_name>`; v5 has no `<split>/`
    directory at all, so this must fail exactly as reported: construction
    succeeds (it only reads the COCO json), __getitem__ dies with
    FileNotFoundError on the image path."""
    root = _v5_root(tmp_path)
    ds = V3Dataset(str(root), "train")
    assert len(ds) == 3
    with pytest.raises(FileNotFoundError):
        ds[0]


# --- V5Dataset (the fix) -----------------------------------------------------

def test_v5dataset_getitem_shapes_and_dtypes(tmp_path):
    from jarvis_jax.data.v5_2d import V5Dataset
    root = _v5_root(tmp_path)
    ds = V5Dataset(str(root), "train")
    assert len(ds) == 3
    img4, kp, vis = ds[0]
    assert img4.shape == (448, 448, 4) and img4.dtype == np.uint8
    assert kp.shape == (50, 2) and kp.dtype == np.float32
    assert vis.shape == (50,) and vis.dtype == np.bool_


def test_v5dataset_mask_found_via_src_ann_id_first(tmp_path):
    from jarvis_jax.data.v5_2d import V5Dataset
    root = _v5_root(tmp_path)
    ds = V5Dataset(str(root), "train")
    img4, _, _ = ds[0]  # ann 101: src_ann_id=777 matches the npz directly
    assert img4[..., 3].max() > 0


def test_v5dataset_mask_found_via_id_fallback(tmp_path):
    from jarvis_jax.data.v5_2d import V5Dataset
    root = _v5_root(tmp_path)
    ds = V5Dataset(str(root), "train")
    img4, _, _ = ds[1]  # ann 202: src_ann_id=999 misses; falls back to id=202
    assert img4[..., 3].max() > 0


def test_v5dataset_missing_mask_file_is_zeros_not_a_crash(tmp_path):
    from jarvis_jax.data.v5_2d import V5Dataset
    root = _v5_root(tmp_path)
    ds = V5Dataset(str(root), "train")
    img4, _, _ = ds[2]  # ann 303: no npz at all for its image
    assert img4[..., 3].max() == 0


def test_v5dataset_rgb_channel_always_nonzero(tmp_path):
    from jarvis_jax.data.v5_2d import V5Dataset
    root = _v5_root(tmp_path)
    ds = V5Dataset(str(root), "train")
    for i in range(len(ds)):
        img4, _, _ = ds[i]
        assert img4[..., :3].max() > 0


def test_v5dataset_padded_absent_joint_marked_not_visible(tmp_path):
    """Mirrors the real headless/amputee recordings: keypoints arrive as a
    full 50-slot array with the absent joint's v=0 -- transform_keypoints
    must mark it (and only it) not-visible."""
    from jarvis_jax.data.v5_2d import V5Dataset
    root = _v5_root(tmp_path)
    ds = V5Dataset(str(root), "train")
    _, kp, vis = ds[2]
    assert kp.shape == (50, 2)
    assert not vis[0]
    assert vis[1:].all()


def test_v5dataset_recordings_filter_keeps_only_requested(tmp_path):
    from jarvis_jax.data.v5_2d import V5Dataset
    root = _v5_root(tmp_path)
    ds = V5Dataset(str(root), "train", recordings=["recA"])
    assert len(ds) == 1
    assert ds.file_names[0].startswith("recA/")


def test_v5dataset_val_split_reads_its_own_json(tmp_path):
    from jarvis_jax.data.v5_2d import V5Dataset
    root = _v5_root(tmp_path)
    ds = V5Dataset(str(root), "val")
    assert len(ds) == 3


def test_v5dataset_sex_behavior_tags_and_sampling_weights_reused(tmp_path):
    """sampling_weights/balanced_weights are shared with V3Dataset (pure
    functions of self.sex/self.behavior/self.category) -- confirm V5Dataset
    actually populates those attributes so reuse isn't a no-op."""
    from jarvis_jax.data.v5_2d import V5Dataset
    root = _v5_root(tmp_path)
    ds = V5Dataset(str(root), "train")
    assert ds.sex == ["male", "female", "male"]
    assert ds.behavior == ["general", "courtship", "unknown"]
    w = ds.sampling_weights("female", "courtship", 11)
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w[1], 11 / 13) and np.allclose([w[0], w[2]], 1 / 13)


def test_v5dataset_batches_stacks(tmp_path):
    from jarvis_jax.data.v5_2d import V5Dataset, batches
    root = _v5_root(tmp_path)
    ds = V5Dataset(str(root), "train")
    it = batches(ds, batch_size=2, shuffle=False, drop_last=False)
    img4, kp, vis = next(it)
    assert img4.shape == (2, 448, 448, 4)
    assert kp.shape == (2, 50, 2)
    assert vis.shape == (2, 50)


# --- sex resolution fallback (human labels live mostly on manifest.json) ---
#
# Measured on real red_data_3d_v5: per-annotation `sex` is "unknown" for 81%
# of annotations (only 8 source recordings carried it) while manifest.json's
# `recordings[<rec>]["sex"]`/`["fly_sex"]` carries the human labels for all
# but 4 recordings. `balanced_weights(key="sex")` reading `self.sex` alone
# therefore balanced across a bogus 19,330-strong "unknown" bucket instead of
# the true male/female split. `_resolve_sex` below is the fix: each
# annotation's sex is (1) its own `sex` if not "unknown", else (2) the
# manifest recording's `fly_sex["fly<fly_id>"]` (the 4 two-fly recordings),
# else (3) the manifest recording's `sex`, else (4) "unknown".

def _v5_root_sex_fallback(tmp_path):
    """One annotation per fallback branch, verified against the real shape
    of the defect: most annotations carry sex=="unknown" and the human
    label lives on manifest.json instead, either per-recording (`sex`) or,
    for the 4 two-fly recordings, per-fly (`fly_sex`, keyed by "fly<fly_id>"
    -- NOT by recording, since a mixed-sex two-fly recording has no single
    recording-level answer).

      - recLabeled: annotation sex="male" already set -> wins outright even
        though the manifest (deliberately) disagrees ("female"); branch 1.
      - recTwoFly (fly_id 0 AND 1): both annotations are sex="unknown" (as
        every real two-fly recording's annotations are); the manifest
        recording's own sex is ALSO "unknown" but carries
        fly_sex={"fly0":"male","fly1":"female"} -- branch 2, keyed by
        fly_id, not recording.
      - recSingle: annotation sex="unknown"; manifest has no fly_sex but a
        recording-level sex="female" -- branch 3.
      - recBlank: annotation sex="unknown"; recBlank is absent from the
        manifest entirely -- branch 4, stays "unknown".

    No image files are written -- these tests only exercise `self.sex`
    (populated in `__init__`), never `__getitem__`.
    """
    root = tmp_path / "v5_sex"
    (root / "annotations").mkdir(parents=True)

    images, annotations = [], []
    iid = aid = 0

    def add(rec, cam, fly_id, ann_sex):
        nonlocal iid, aid
        images.append({"id": iid, "file_name": f"{rec}/{cam}/Frame_0.jpg",
                        "width": 100, "height": 100})
        annotations.append({
            "id": aid, "image_id": iid, "bbox": [0, 0, 10, 10],
            "keypoints": [1.0, 1.0, 1] * 50, "num_keypoints": 50,
            "sex": ann_sex, "fly_id": fly_id, "src_ann_id": aid,
        })
        iid += 1
        aid += 1

    add("recLabeled", "Cam1", 0, "male")
    add("recTwoFly", "Cam1", 0, "unknown")
    add("recTwoFly", "Cam1", 1, "unknown")
    add("recSingle", "Cam1", 0, "unknown")
    add("recBlank", "Cam1", 0, "unknown")

    coco = {
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "fly", "keypoints": [], "skeleton": []}],
        "images": images, "annotations": annotations, "framesets": {},
    }
    for split in ("train", "val"):
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(coco))

    manifest = {"recordings": {
        "recLabeled": {"sex": "female"},          # disagrees; must lose to branch 1
        "recTwoFly": {"sex": "unknown",
                      "fly_sex": {"fly0": "male", "fly1": "female"}},
        "recSingle": {"sex": "female"},
        # recBlank intentionally absent from the manifest
    }}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_v5dataset_sex_own_annotation_wins_over_manifest(tmp_path):
    """Branch 1: an annotation that already carries a real sex is
    authoritative even when the manifest disagrees."""
    from jarvis_jax.data.v5_2d import V5Dataset
    root = _v5_root_sex_fallback(tmp_path)
    ds = V5Dataset(str(root), "train", recordings=["recLabeled"])
    assert ds.sex == ["male"]


def test_v5dataset_sex_falls_back_to_per_fly_fly_sex(tmp_path):
    """Branch 2: the two-fly recording's own annotations are BOTH
    sex=="unknown" (as every real one is) -- resolution must key off
    fly_id, not recording, since fly0 and fly1 disagree."""
    from jarvis_jax.data.v5_2d import V5Dataset
    root = _v5_root_sex_fallback(tmp_path)
    ds = V5Dataset(str(root), "train", recordings=["recTwoFly"])
    assert ds.sex == ["male", "female"]


def test_v5dataset_sex_falls_back_to_recording_level_sex(tmp_path):
    """Branch 3: no fly_sex on the manifest entry -> recording-level sex."""
    from jarvis_jax.data.v5_2d import V5Dataset
    root = _v5_root_sex_fallback(tmp_path)
    ds = V5Dataset(str(root), "train", recordings=["recSingle"])
    assert ds.sex == ["female"]


def test_v5dataset_sex_stays_unknown_when_manifest_has_nothing(tmp_path):
    """Branch 4: recording absent from the manifest entirely -> "unknown",
    never a crash on a missing manifest entry."""
    from jarvis_jax.data.v5_2d import V5Dataset
    root = _v5_root_sex_fallback(tmp_path)
    ds = V5Dataset(str(root), "train", recordings=["recBlank"])
    assert ds.sex == ["unknown"]


def test_v5dataset_missing_manifest_json_degrades_to_unknown_not_a_crash(tmp_path):
    """A root with no manifest.json at all (the older `_v5_root` fixture
    above) must not crash V5Dataset -- resolution just can't reach past
    branch 1, which is fine since that fixture's annotations already carry
    real sex values."""
    from jarvis_jax.data.v5_2d import V5Dataset
    root = _v5_root(tmp_path)  # no manifest.json in this older fixture
    ds = V5Dataset(str(root), "train")
    assert ds.sex == ["male", "female", "male"]


# --- real-data regression (skipped when the dataset isn't present) ---------

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5"
have_data = os.path.isdir(ROOT)
skip = pytest.mark.skipif(not have_data, reason="v5 data root not present")


@skip
def test_real_v5_train_and_val_construct_and_getitem_succeed():
    from jarvis_jax.data.v5_2d import V5Dataset
    for split in ("train", "val"):
        ds = V5Dataset(ROOT, split)
        assert len(ds) > 0
        img4, kp, vis = ds[0]
        assert img4.shape == (448, 448, 4) and img4.dtype == np.uint8
        assert kp.shape == (50, 2)
        assert vis.shape == (50,)


@skip
def test_real_v5_mask_coverage_near_100_percent():
    from jarvis_jax.data.v5_2d import V5Dataset
    ds = V5Dataset(ROOT, "train")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(200, len(ds)), replace=False)
    nonzero = sum(1 for i in idx if ds[int(i)][0][..., 3].max() > 0)
    frac = nonzero / len(idx)
    assert frac > 0.9, f"mask coverage too low: {frac:.3f}"


@skip
def test_real_v5_headless_recording_padding_survives():
    from jarvis_jax.data.v5_2d import V5Dataset
    ds = V5Dataset(ROOT, "train", recordings=["2026_06_09_15_46_55"])
    assert len(ds) > 0
    _, kp, vis = ds[0]
    assert kp.shape == (50, 2) and vis.shape == (50,)
    assert not vis.all()  # headless: at least one joint is padded-absent


@skip
def test_real_v5_amputee_recording_padding_survives():
    from jarvis_jax.data.v5_2d import V5Dataset
    ds = V5Dataset(ROOT, "train", recordings=["2026_02_13_13_44_49"])
    assert len(ds) > 0
    _, kp, vis = ds[0]
    assert kp.shape == (50, 2) and vis.shape == (50,)
    assert not vis.all()
