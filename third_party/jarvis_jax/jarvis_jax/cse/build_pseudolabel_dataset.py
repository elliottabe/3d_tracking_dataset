"""Assemble gated silhouette pseudo-labels into the V3 COCO dataset format.

Writes the exact on-disk layout `jarvis_jax.data.v3.V3Dataset` reads:
  - `<out_root>/annotations/instances_<split>.json`      COCO images+annotations
  - `<out_root>/<split>/<file_name>`                      frame jpg
  - `<out_root>/sam3_masks/<split>/<file_noext>.npz`      per-annotation SAM mask

Note: `V3Dataset.__getitem__` opens the frame at `<root>/<split>/<file_name>`
(not `<root>/<file_name>`) -- see `jarvis_jax/data/v3.py`.
"""
import os, json
import numpy as np
import cv2


def bbox_from_mask(mask):
    ys, xs = np.where(np.asarray(mask, bool))
    if xs.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def _pad_to_at_least(arr, min_h, min_w):
    """Zero-pad an (H,W) or (H,W,C) array on the bottom/right so it is at
    least `min_h` x `min_w`. Content stays anchored at the origin, so pixel
    coordinates (keypoints, bbox) computed against the un-padded array remain
    valid. `V3Dataset.__getitem__` crops a fixed `crop`x`crop` window with no
    padding of its own, so the on-disk frame must already be >= crop-sized in
    both dimensions (true for real camera frames; only matters for tiny
    synthetic frames such as in tests)."""
    h, w = arr.shape[:2]
    pad_h, pad_w = max(0, min_h - h), max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    pad_spec = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, pad_spec, mode="constant")


def write_pseudolabel_coco(out_root, records, *, split="train", crop=448):
    os.makedirs(os.path.join(out_root, "annotations"), exist_ok=True)
    images, annotations = [], []
    for i, r in enumerate(records):
        kp = np.asarray(r["keypoints"], float)
        if not (kp[:, 2] > 0).any():
            continue                                    # no visible label -> skip
        fn = r["file_name"]

        rgb = _pad_to_at_least(np.asarray(r["rgb"], np.uint8), crop, crop)
        mask = _pad_to_at_least(np.asarray(r["mask"], bool), crop, crop)
        img_h, img_w = rgb.shape[:2]

        img_path = os.path.join(out_root, split, fn)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        npz_path = os.path.join(out_root, "sam3_masks", split,
                                os.path.splitext(fn)[0] + ".npz")
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(npz_path, masks=mask[None],
                 ann_ids=np.array([i], int), matched=np.array([True], bool))

        images.append({"id": i, "file_name": fn,
                       "width": int(img_w), "height": int(img_h)})
        annotations.append({"id": i, "image_id": i, "category_id": 1,
                            "bbox": [float(x) for x in r["bbox"]],
                            "keypoints": kp.reshape(-1).tolist()})
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "fly"}]}
    ann_path = os.path.join(out_root, "annotations", f"instances_{split}.json")
    with open(ann_path, "w") as f:
        json.dump(coco, f)
    return ann_path
