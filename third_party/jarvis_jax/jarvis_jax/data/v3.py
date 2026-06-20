"""NumPy V3 keypoint dataset: COCO annotations + matched SAM masks -> 4-channel
448 crops and 224x224x50 Gaussian heatmaps."""
import json
import os

import numpy as np
from PIL import Image

from jarvis_jax.data.transforms import (
    crop_origin, normalize_rgb, transform_keypoints, gaussian_heatmaps,
)


class V3Dataset:
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
        self.img_wh = []
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

    def __len__(self):
        return len(self.file_names)

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
            img = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
        mask = self._load_mask(fn, self.ann_ids[i], img_w, img_h)

        x0, y0 = crop_origin(bbox, img_w, img_h, self.crop)
        rgb_crop = normalize_rgb(img[y0:y0 + self.crop, x0:x0 + self.crop])
        mask_crop = mask[y0:y0 + self.crop, x0:x0 + self.crop][..., None]
        img4 = np.concatenate([rgb_crop, mask_crop], axis=-1)

        hm_xy, vis = transform_keypoints(
            self.keypoints[i], x0, y0, self.crop, self.heatmap_size)
        hm = gaussian_heatmaps(hm_xy, vis, self.heatmap_size, self.sigma)
        return img4, hm, vis


def batches(ds, batch_size, *, shuffle=True, seed=0, drop_last=True):
    n = len(ds)
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    stop = (n // batch_size) * batch_size if drop_last else n
    for s in range(0, stop, batch_size):
        sel = idx[s:s + batch_size]
        imgs, hms, viss = zip(*(ds[int(j)] for j in sel))
        yield (np.stack(imgs), np.stack(hms), np.stack(viss))
