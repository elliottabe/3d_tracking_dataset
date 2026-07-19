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


class V3Dataset:
    def __init__(self, root, split, *, crop=448, heatmap_size=224, sigma=7.0,
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
        self.img_wh = []
        self.sex = []          # per-annotation "male"/"female"/"unknown" (for weighted sampling)
        self.behavior = []     # per-annotation "general"/"courtship"/"grooming"/"unknown"
        for a in coco["annotations"]:
            fn = id2file[a["image_id"]]
            if recordings is not None and not any(
                fn.startswith(r + "/") for r in recordings):
                continue
            self.file_names.append(fn)
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

    def _load_mask(self, file_name, ann_id, img_w, img_h):
        npz_path = os.path.join(
            self.root, "sam3_masks", self.split,
            os.path.splitext(file_name)[0] + ".npz")
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
        fn = self.file_names[i]
        bbox = self.bboxes[i]
        img_w, img_h = self.img_wh[i]

        with Image.open(os.path.join(self.root, self.split, fn)) as pil:
            img = np.asarray(pil.convert("RGB"), dtype=np.uint8)   # (H,W,3) 0-255
        mask = self._load_mask(fn, self.ann_ids[i], img_w, img_h)  # float32 0/1

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
