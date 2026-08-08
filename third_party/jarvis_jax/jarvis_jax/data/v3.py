"""NumPy V3 keypoint dataset: COCO annotations + matched SAM masks -> 4-channel
448 uint8 crops and (50,2) heatmap-coord keypoints."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from jarvis_jax.data.transforms import (
    crop_origin, transform_keypoints,
)


def _resolve_image_path(root, split, img_record, source_root):
    """Resolve the on-disk path for one COCO image record.

    New-style records (from `scripts/build_detector_dataset.py`, e.g. the
    V4 unified detector dataset) reference images IN PLACE rather than
    copying them: they carry explicit `subset`/`orig_split` fields and the
    real file lives at `<source_root>/<subset>/<orig_split>/<file_name>`.
    Old-style records (V3 and the per-subset general_model dirs) have
    neither field and resolve as `<root>/<split>/<file_name>`, unchanged
    from prior behaviour.

    `orig_split` (not the dataset's own `split`) drives the new-style path
    -- that is the whole point of the V4 re-split: a recording can land in
    this dataset's "val" while its images still physically live under the
    source subset's "train" directory.
    """
    subset = img_record.get("subset")
    orig_split = img_record.get("orig_split")
    if subset is not None or orig_split is not None:
        if subset is None:
            raise ValueError(
                "image record has 'orig_split' but no 'subset' -- both "
                "fields are required together for in-place (V4-style) "
                f"image resolution: {img_record!r}")
        if orig_split is None:
            raise ValueError(
                "image record has 'subset' but no 'orig_split' -- both "
                "fields are required together for in-place (V4-style) "
                f"image resolution: {img_record!r}")
        if not source_root:
            raise ValueError(
                "image record has subset/orig_split (in-place V4-style "
                "reference) but no 'source_root' is available -- pass "
                "source_root=... to V3Dataset, or ensure the dataset JSON's "
                f"info.source_root is set. record: {img_record!r}")
        return os.path.join(source_root, subset, orig_split, img_record["file_name"])
    return os.path.join(root, split, img_record["file_name"])


def _resolve_mask_path(root, split, img_record, source_root):
    """Resolve the sam3 mask .npz path matching `_resolve_image_path`'s
    choice of image location: `<...>/sam3_masks/<split_dir>/<stem>.npz`
    rooted at whichever tree (`root` or `<source_root>/<subset>`) the
    image itself was resolved under."""
    subset = img_record.get("subset")
    orig_split = img_record.get("orig_split")
    stem = os.path.splitext(img_record["file_name"])[0] + ".npz"
    if subset is not None or orig_split is not None:
        if subset is None or orig_split is None:
            raise ValueError(
                "image record has only one of 'subset'/'orig_split' -- "
                f"both are required together: {img_record!r}")
        if not source_root:
            raise ValueError(
                "image record has subset/orig_split (in-place V4-style "
                "reference) but no 'source_root' is available -- pass "
                "source_root=... to V3Dataset, or ensure the dataset JSON's "
                f"info.source_root is set. record: {img_record!r}")
        return os.path.join(source_root, subset, "sam3_masks", orig_split, stem)
    return os.path.join(root, "sam3_masks", split, stem)


class V3Dataset:
    def __init__(self, root, split, *, crop=448, heatmap_size=224, sigma=7.0,
                 recordings=None, source_root=None):
        self.root = root
        self.split = split
        self.crop = crop
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        ann_path = os.path.join(root, "annotations", f"instances_{split}.json")
        with open(ann_path) as f:
            coco = json.load(f)
        self.source_root = source_root or coco.get("info", {}).get("source_root")
        id2file = {im["id"]: im["file_name"] for im in coco["images"]}
        id2img = {im["id"]: im for im in coco["images"]}
        id2wh = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}

        self.file_names = []
        self.img_records = []  # full COCO image dict (carries subset/orig_split, if any)
        self.bboxes = []
        self.keypoints = []
        self.ann_ids = []
        self.img_wh = []
        self.sex = []          # per-annotation "male"/"female"/"unknown" (for weighted sampling)
        self.behavior = []     # per-annotation "general"/"courtship"/"grooming"/"unknown"
        for a in coco["annotations"]:
            fn = id2file[a["image_id"]]
            if recordings is not None and not any(
                fn.startswith(r + "/") for r in recordings):
                continue
            self.file_names.append(fn)
            self.img_records.append(id2img[a["image_id"]])
            self.bboxes.append(np.asarray(a["bbox"], dtype=np.float32))
            self.keypoints.append(
                np.asarray(a["keypoints"], dtype=np.float32).reshape(-1, 3))
            self.ann_ids.append(int(a["id"]))
            self.img_wh.append(id2wh[a["image_id"]])
            self.sex.append(a.get("sex", "unknown"))
            self.behavior.append(a.get("behavior", "unknown"))

    def __len__(self):
        return len(self.file_names)

    def sampling_weights(self, target_sex, target_behavior, factor):
        """Per-annotation sampling weights (sum-normalised): annotations whose
        (sex, behavior) match the target get `factor`x the base weight of 1,
        everything else 1. `target_sex`/`target_behavior` may be None to match
        any value on that axis. Used to oversample the under-represented
        female-courtship class (2.2% of train) during weighted sampling."""
        w = np.ones(len(self), dtype=np.float64)
        for i in range(len(self)):
            if ((target_sex is None or self.sex[i] == target_sex) and
                    (target_behavior is None or self.behavior[i] == target_behavior)):
                w[i] = float(factor)
        return w / w.sum()

    def _load_mask(self, img_record, ann_id, img_w, img_h):
        npz_path = _resolve_mask_path(self.root, self.split, img_record, self.source_root)
        if not os.path.exists(npz_path):
            return np.zeros((img_h, img_w), dtype=np.float32)
        try:
            z = np.load(npz_path, allow_pickle=True)
            masks, ids, matched = z["masks"], z["ann_ids"], z["matched"]
            if masks.shape[0] == 0:
                return np.zeros((img_h, img_w), dtype=np.float32)
            sel = np.where((ids == ann_id) & matched)[0]
            if sel.size == 0:
                return np.zeros((img_h, img_w), dtype=np.float32)
            return masks[sel[0]].astype(np.float32)
        except Exception:
            return np.zeros((img_h, img_w), dtype=np.float32)

    def __getitem__(self, i):
        img_record = self.img_records[i]
        bbox = self.bboxes[i]
        img_w, img_h = self.img_wh[i]

        img_path = _resolve_image_path(self.root, self.split, img_record, self.source_root)
        with Image.open(img_path) as pil:
            img = np.asarray(pil.convert("RGB"), dtype=np.uint8)   # (H,W,3) 0-255
        mask = self._load_mask(img_record, self.ann_ids[i], img_w, img_h)  # float32 0/1

        x0, y0 = crop_origin(bbox, img_w, img_h, self.crop)
        rgb_crop = img[y0:y0 + self.crop, x0:x0 + self.crop]                 # uint8
        mask_crop = mask[y0:y0 + self.crop, x0:x0 + self.crop][..., None]
        img4 = np.concatenate(
            [rgb_crop, mask_crop.astype(np.uint8)], axis=-1)                 # (448,448,4) uint8

        kp_xy, vis = transform_keypoints(
            self.keypoints[i], x0, y0, self.crop, self.heatmap_size)
        return img4, kp_xy.astype(np.float32), vis


def batches(ds, batch_size, *, shuffle=True, seed=0, drop_last=True, weights=None,
            num_workers: int = 8):
    """Yield (imgs, kps, viss) batches over the dataset for one epoch.

    weights: optional (n,) sum-normalised per-annotation sampling probabilities
    (e.g. from ds.sampling_weights). When given, one epoch draws n indices WITH
    REPLACEMENT from `weights` (oversampling the rare class); when None, a plain
    uniform shuffle (each annotation once).

    num_workers: when > 1, per-batch samples (ds[j] -- JPEG decode + mask npz
    load, both GIL-releasing) are fetched concurrently via a thread pool that is
    created once for the whole epoch and reused across batches. Order of results
    within a batch always matches `sel` (index selection is untouched), so
    batches are byte-identical to the serial (num_workers<=1) path for the same
    seed. When num_workers <= 1, falls back to the original serial fetch."""
    n = len(ds)
    rng = np.random.default_rng(seed)
    if weights is not None:
        idx = rng.choice(n, size=n, replace=True, p=weights)
    elif shuffle:
        idx = np.arange(n); rng.shuffle(idx)
    else:
        idx = np.arange(n)
    stop = (n // batch_size) * batch_size if drop_last else n

    if num_workers is not None and num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for s in range(0, stop, batch_size):
                sel = idx[s:s + batch_size]
                imgs, kps, viss = zip(*executor.map(lambda j: ds[int(j)], sel))
                yield (np.stack(imgs), np.stack(kps), np.stack(viss))
    else:
        for s in range(0, stop, batch_size):
            sel = idx[s:s + batch_size]
            imgs, kps, viss = zip(*(ds[int(j)] for j in sel))
            yield (np.stack(imgs), np.stack(kps), np.stack(viss))
