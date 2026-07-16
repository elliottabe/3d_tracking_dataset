"""Assemble gated silhouette pseudo-labels into the V3 COCO dataset format.

Writes the exact on-disk layout `jarvis_jax.data.v3.V3Dataset` reads:
  - `<out_root>/annotations/instances_<split>.json`      COCO images+annotations
  - `<out_root>/<split>/<file_name>`                      frame jpg
  - `<out_root>/sam3_masks/<split>/<file_noext>.npz`      per-annotation SAM mask

Note: `V3Dataset.__getitem__` opens the frame at `<root>/<split>/<file_name>`
(not `<root>/<file_name>`) -- see `jarvis_jax/data/v3.py`.

`PseudoLabelWriter` is the incremental (per-bout-streaming) version of this
writer -- see its docstring. `write_pseudolabel_coco` is a thin one-shot
wrapper over it, kept for callers that pass every record at once.
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


class PseudoLabelWriter:
    """Incremental version of `write_pseudolabel_coco`'s on-disk writer.

    A whole-run pseudo-label build can span hundreds of bouts x 2 flies x many
    recordings; holding every bout's full-res `rgb`/`mask` arrays in one
    Python list until a single final write call (the original
    `write_pseudolabel_coco` contract) uses hundreds of GB of RAM. This class
    lets a caller flush one bout's records to disk immediately via
    `.add_records(records)` -- jpgs + the per-file stacked-mask npz are
    written right away and the heavy `rgb`/`mask` arrays are NOT retained;
    only lightweight COCO image/annotation dicts accumulate in `self.images`/
    `self.annotations`. Call `.finalize()` once at the end to write
    `annotations/instances_{split}.json`.

    CRUCIAL: two flies sharing the same courtship frame (and therefore the
    same `file_name`) MUST be passed to the SAME `.add_records(...)` call (as
    the original single-call `write_pseudolabel_coco` naturally did by
    grouping the whole records list by `file_name`) -- see the module
    docstring / the historical np.savez-clobber bug this grouping guards
    against. Records from different `.add_records()` calls are grouped
    independently, so passing the two flies of one bout across two SEPARATE
    calls would silently reproduce that bug.

    Image ids and annotation ids are drawn from monotonically increasing
    counters that are NEVER reset across `.add_records()` calls, so every id
    stays unique dataset-wide regardless of how many calls are made.
    """

    def __init__(self, out_root, *, split="train", crop=448):
        self.out_root = out_root
        self.split = split
        self.crop = crop
        os.makedirs(os.path.join(out_root, "annotations"), exist_ok=True)
        self.images = []
        self.annotations = []
        self._next_img_id = 0
        self._next_ann_id = 0

    def add_records(self, records):
        """Write this batch's jpgs + per-file stacked-mask npz immediately,
        and append lightweight COCO metadata to `self.images`/
        `self.annotations`. Retains no `rgb`/`mask` arrays after returning."""
        # Group by file_name first: two flies in the same courtship frame
        # share a file_name and MUST land in one npz with one image id --
        # writing per-record would let the second record's np.savez silently
        # clobber the first's npz (see class docstring / historical bug).
        by_file = {}
        for r in records:
            kp = np.asarray(r["keypoints"], float)
            if not (kp[:, 2] > 0).any():
                continue                                # no visible label -> skip
            by_file.setdefault(r["file_name"], []).append(r)

        for fn, recs in by_file.items():
            r0 = recs[0]
            rgb = _pad_to_at_least(np.asarray(r0["rgb"], np.uint8), self.crop, self.crop)
            img_h, img_w = rgb.shape[:2]

            img_path = os.path.join(self.out_root, self.split, fn)
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            img_id = self._next_img_id
            self._next_img_id += 1

            masks, ann_ids = [], []
            for r in recs:
                kp = np.asarray(r["keypoints"], float)
                mask = _pad_to_at_least(np.asarray(r["mask"], bool), self.crop, self.crop)
                ann_id = self._next_ann_id
                self._next_ann_id += 1
                masks.append(mask)
                ann_ids.append(ann_id)
                self.annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": 1,
                    "bbox": [float(x) for x in r["bbox"]],
                    "keypoints": kp.reshape(-1).tolist()})

            npz_path = os.path.join(self.out_root, "sam3_masks", self.split,
                                    os.path.splitext(fn)[0] + ".npz")
            os.makedirs(os.path.dirname(npz_path), exist_ok=True)
            np.savez(npz_path, masks=np.stack(masks, axis=0),
                     ann_ids=np.array(ann_ids, int),
                     matched=np.ones(len(ann_ids), bool))

            self.images.append({"id": img_id, "file_name": fn,
                                "width": int(img_w), "height": int(img_h)})
            # r, recs, rgb, masks (and their backing rgb/mask arrays) go out
            # of scope at the end of this loop iteration / this method --
            # nothing heavy is retained on self.

    def finalize(self):
        """Write `annotations/instances_{split}.json` from every
        `.add_records()` call so far and return its path."""
        coco = {"images": self.images, "annotations": self.annotations,
                "categories": [{"id": 1, "name": "fly"}]}
        ann_path = os.path.join(self.out_root, "annotations",
                                f"instances_{self.split}.json")
        with open(ann_path, "w") as f:
            json.dump(coco, f)
        return ann_path


def write_pseudolabel_coco(out_root, records, *, split="train", crop=448):
    """Thin one-shot wrapper over `PseudoLabelWriter` (kept for callers/tests
    that don't need incremental streaming): writes the whole `records` list
    in one `add_records` call and finalizes immediately."""
    writer = PseudoLabelWriter(out_root, split=split, crop=crop)
    writer.add_records(records)
    return writer.finalize()
