"""NumPy 2D keypoint dataset for red_data_3d_v5.

`jarvis_jax.data.v3.V3Dataset` builds image/mask paths as
`root/<split>/<file_name>` and `root/sam3_masks/<split>/<stem>` -- both
components baked into the directory layout. red_data_3d_v5 deliberately has
no `train/`/`val/` directory anywhere: the split lives ONLY in
`annotations/instances_{train,val}.json`, and images/masks resolve flat
(`images/<rec>/<cam>/Frame_N.jpg`, `masks/<rec>/<cam>/Frame_N.npz`) regardless
of split. Feeding a v5 root into V3Dataset dies with FileNotFoundError on the
very first sample (see tests/test_v5_2d.py::test_v3dataset_cannot_read_v5_layout_by_design
for the reproduction) -- V3/V4 datasets are untouched by this module and keep
working exactly as before.

This is a SEPARATE module rather than a branch inside V3Dataset, mirroring
the precedent already set for the 3D loaders in this repo: `data/v5_3d.py`
is its own module next to `data/v3_3d.py` rather than a layout-detection
branch bolted onto V3FramesetDataset, because the two trees differ in more
than one field (flat vs split-rooted paths, and a genuinely different mask
key -- see `_load_mask` below) and conflating them risks exactly the kind of
"detect layout, then quietly do the wrong thing for one of them" bug this
repo has been bitten by before. `V5Dataset` mirrors `data/v5_3d.py`'s path
and mask-key conventions (flat `images/`/`masks/` trees, try `src_ann_id`
then the merged `id`) rather than inventing a second dialect, and reuses
`V3Dataset`'s pure per-annotation weighting methods (`sampling_weights`,
`balanced_weights`, `class_counts`) verbatim since those only touch
`self.sex`/`self.behavior`/`self.category`/`len(self)` and are identical
either way. `batches()` is likewise reused unchanged from `data.v3` (it only
duck-types `len(ds)`/`ds[i]`).

Mask lookup tries BOTH id keys, in this order (verified against real
red_data_3d_v5 data, see `_load_mask`):
  1. `src_ann_id` -- masks borrowed from red_data_unified_V3's own
     sam3_masks/ store `ann_ids` in V3's annotation id space.
  2. the merged annotation `id` -- masks generated fresh for the 9 newly-
     added recordings key `ann_ids` by the merged v5 id instead (there is no
     V3 copy to borrow an id space from).
Trying only one key measured ~0.1% mask coverage; trying both measured 100%.
A missing/unmatched mask degrades to an all-zero channel -- it must NEVER
raise, since real recordings (e.g. the headless/amputee ones) legitimately
lack masks for some or all frames -- but it must also never be the SILENT
default for samples that DO have a real mask (that was an hour lost to a
loader returning zeros for 99.9% of samples without erroring). The
acceptance check for this module is therefore a measured non-zero-mask
fraction on real sampled data, not merely "it didn't crash".

Keypoints arrive already padded to the full 50-slot schema at the annotation
level (verified: headless/amputee annotations carry `num_keypoints` < 50 but
`len(keypoints) == 150`, with the absent joints' `v` set to 0) -- so no
special-casing is needed here; `transforms.transform_keypoints` already
marks any v<=0 joint not-visible, which is exactly the desired behaviour for
a padded-absent joint too.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from jarvis_jax.data.transforms import crop_origin, transform_keypoints
from jarvis_jax.data.v3 import V3Dataset, batches  # noqa: F401  (batches: generic, reused as-is)


def _load_mask(root, file_name, ann_id, src_ann_id, img_w, img_h):
    """Mask for ONE annotation, mirroring `data/v5_3d.py::_load_mask`.

    `file_name` is "<rec>/<cam>/Frame_N.jpg" (v5's flat image-record shape);
    the matching mask lives at "masks/<rec>/<cam>/Frame_N.npz" with no split
    component. Tries `src_ann_id` first, then `ann_id` (the merged id) --
    see module docstring for why both keys are required.
    """
    rec, cam, fn = file_name.split("/")
    path = os.path.join(root, "masks", rec, cam,
                         os.path.splitext(fn)[0] + ".npz")
    if not os.path.exists(path):
        return np.zeros((img_h, img_w), dtype=np.float32)
    try:
        with np.load(path, allow_pickle=True) as z:
            masks, ids, matched = z["masks"], z["ann_ids"], z["matched"]
            if masks.shape[0] == 0:
                return np.zeros((img_h, img_w), dtype=np.float32)
            sel = np.where((ids == src_ann_id) & matched)[0]
            if sel.size == 0:
                sel = np.where((ids == ann_id) & matched)[0]
            if sel.size == 0:
                return np.zeros((img_h, img_w), dtype=np.float32)
            return masks[sel[0]].astype(np.float32)
    except Exception:
        return np.zeros((img_h, img_w), dtype=np.float32)


class V5Dataset:
    """Per-annotation 2D keypoint dataset over red_data_3d_v5.

    Same `__getitem__` contract as `V3Dataset` (img4 (448,448,4) uint8,
    kp_xy (50,2) float32, vis (50,) bool) so it's a drop-in replacement for
    callers currently constructing `V3Dataset` against a v5 root -- only the
    constructor and internal path/mask resolution differ.

    Parameters
    ----------
    root : str
        Root of the v5 dataset (contains annotations/, images/, masks/,
        calibrations/, manifest.json).
    split : str
        'train' or 'val' -- selects `annotations/instances_{split}.json`
        only; it does NOT select a directory (v5 has none).
    recordings : list[str] | None
        If given, only include annotations whose recording is in this list.
    """

    def __init__(self, root, split, *, crop=448, heatmap_size=224, sigma=2.0,
                 recordings=None):
        self.root = root
        self.split = split
        self.crop = crop
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        ann_path = os.path.join(root, "annotations", f"instances_{split}.json")
        with open(ann_path) as f:
            coco = json.load(f)
        id2file = {im["id"]: im["file_name"] for im in coco["images"]}
        id2wh = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}

        self.file_names = []
        self.bboxes = []
        self.keypoints = []
        self.ann_ids = []
        self.src_ann_ids = []   # mask-lookup key tried FIRST (see _load_mask)
        self.img_wh = []
        self.sex = []          # per-annotation "male"/"female"/"unknown"
        self.behavior = []     # per-annotation behavior tag ("unknown" if absent)
        self.category = []     # per-annotation category tag ("unknown" if absent)
        for a in coco["annotations"]:
            fn = id2file[a["image_id"]]
            if recordings is not None and not any(
                    fn.startswith(r + "/") for r in recordings):
                continue
            self.file_names.append(fn)
            self.bboxes.append(np.asarray(a["bbox"], dtype=np.float32))
            self.keypoints.append(
                np.asarray(a["keypoints"], dtype=np.float32).reshape(-1, 3))
            ann_id = int(a["id"])
            self.ann_ids.append(ann_id)
            self.src_ann_ids.append(a.get("src_ann_id", ann_id))
            self.img_wh.append(id2wh[a["image_id"]])
            self.sex.append(a.get("sex", "unknown"))
            self.behavior.append(a.get("behavior", "unknown"))
            self.category.append(a.get("category", "unknown"))

    def __len__(self):
        return len(self.file_names)

    # Pure functions of self.sex/self.behavior/self.category/len(self) --
    # identical to V3Dataset's, reused verbatim rather than duplicated.
    sampling_weights = V3Dataset.sampling_weights
    balanced_weights = V3Dataset.balanced_weights
    class_counts = V3Dataset.class_counts

    def __getitem__(self, i):
        fn = self.file_names[i]
        bbox = self.bboxes[i]
        img_w, img_h = self.img_wh[i]

        img_path = os.path.join(self.root, "images", fn)
        with Image.open(img_path) as pil:
            img = np.asarray(pil.convert("RGB"), dtype=np.uint8)   # (H,W,3) 0-255
        mask = _load_mask(self.root, fn, self.ann_ids[i], self.src_ann_ids[i],
                           img_w, img_h)                            # float32 0/1

        x0, y0 = crop_origin(bbox, img_w, img_h, self.crop)
        rgb_crop = img[y0:y0 + self.crop, x0:x0 + self.crop]                 # uint8
        mask_crop = mask[y0:y0 + self.crop, x0:x0 + self.crop][..., None]
        img4 = np.concatenate(
            [rgb_crop, mask_crop.astype(np.uint8)], axis=-1)                 # (448,448,4) uint8

        kp_xy, vis = transform_keypoints(
            self.keypoints[i], x0, y0, self.crop, self.heatmap_size)
        return img4, kp_xy.astype(np.float32), vis
