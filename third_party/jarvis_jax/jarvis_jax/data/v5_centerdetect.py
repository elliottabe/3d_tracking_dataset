"""Per-IMAGE (not per-annotation) dataset for training CenterDetect (JAX
EfficientTrack ``model_size="medium"``) on red_data_3d_v5-shaped roots.

CenterDetect answers a different question than the KeypointDetect loaders in
this package (``data/v3.py``/``data/v5_2d.py``): "where is EVERY animal in
this frame", not "here is one animal's crop, where are its joints". So the
dataset row here is one FULL FRAME (resized to a square, matching JARVIS's
own ``Dataset2D(mode='CenterDetect')`` -- see
``third_party/JARVIS-HybridNet/jarvis/dataset/dataset2D.py::_build_augpipe``,
which does ``iaa.Resize(IMAGE_SIZE)``, i.e. an aspect-distorting resize to a
SQUARE target, not a crop), carrying the (1 or 2) animal centers present
in that frame -- zero-annotation images are excluded entirely (see below).

Center = the annotation's own bbox center (``bbox[0]+bbox[2]/2``,
``bbox[1]+bbox[3]/2``), the same convention JARVIS's
``Dataset2D._get_item_center`` uses, so a JAX-trained checkpoint's peak
locations are directly comparable to the PyTorch baseline's.

K_MAX=2 is an assumption about this data (courtship = 2 flies, free-running
= 1), not a general multi-animal cap -- checked (not assumed) at construction
time; a 3rd annotation on one image raises rather than silently truncating.

**Zero-annotation images are EXCLUDED, not treated as background negatives.**
``red_data_3d_v5``'s own image count (22,449 train / 1,729 val) is larger
than its annotated-image count (22,040 / 1,690) by 409 / 39 images that
carry media but no annotation. An earlier version of this module read that
gap as "pure background frames" and kept them as `num_flies=="0"` training
rows -- WRONG: they are unannotated, not empty. Three were rendered and
inspected (`figures/2026-08-31-zerofly/zero_annotation_frames.png`); all
three show a plainly visible, unambiguous fly. The likely upstream cause is
the v5 builder's ``MIN_CAMS=3`` rule (a frameset with fewer than 3
resolvable per-camera annotations is dropped) leaving `link_media` having
already linked that frameset's images. Keeping them as negatives would
train the model to predict NOTHING on frames that contain a fly -- exactly
the collapse this whole task exists to fix, and worse than not balancing at
all (measured: at alpha=0.5 they got a 5.49x sampling boost). This is a
DATASET defect, not a CenterDetect-specific one: every PER-ANNOTATION
consumer (``V5Dataset``/keypoint training) is silently unaffected by it
(zero-annotation images are simply invisible to a per-annotation iterator),
but any OTHER per-image consumer built against this root will hit the same
trap. ``__init__`` asserts/warns loudly (naming the count) rather than
silently excluding, so a future change to the source data that legitimately
introduces verified-empty frames does not silently reintroduce this exact
mistake unnoticed.

Sex resolution reuses ``jarvis_jax.data.v5_2d._resolve_sex`` UNCHANGED (see
that module's docstring for the fallback chain and why the raw per-annotation
``sex`` field must never be read alone) -- carried per-instance (not padded)
for val-set reporting split by sex; balanced OVERSAMPLING here is keyed on
``num_flies`` (this task's actual ask), not sex.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from jarvis_jax.data.v3 import V3Dataset  # for balanced_weights/class_counts reuse
from jarvis_jax.data.v5_2d import _resolve_sex

K_MAX = 2


class V5CenterDetectDataset:
    """One row per IMAGE. ``__getitem__(i)`` returns:
        img       (image_size, image_size, 3) uint8 RGB, aspect-distorting
                  resize of the full frame (matches JARVIS's own CenterDetect
                  preprocessing -- see module docstring).
        centers   (K_MAX, 2) float32 (x, y) in image_size-space, zero-padded
                  past the real instance count.
        valid     (K_MAX,) bool -- which `centers` rows are real instances.

    ``self.num_flies`` (one "1"/"2" string per image) feeds
    ``balanced_weights(key="num_flies")`` (reused verbatim from
    ``V3Dataset`` -- a pure function of ``getattr(self, key)``/``len(self)``,
    identical either way, per the precedent already set for the keypoint
    v5 loader).
    """

    def __init__(self, root, split, *, image_size=320, recordings=None):
        self.root = root
        self.split = split
        self.image_size = int(image_size)

        ann_path = os.path.join(root, "annotations", f"instances_{split}.json")
        with open(ann_path) as f:
            coco = json.load(f)
        id2img = {im["id"]: im for im in coco["images"]}

        manifest_path = os.path.join(root, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                self.manifest = json.load(f).get("recordings", {})
        else:
            self.manifest = {}

        by_image = defaultdict(list)
        for a in coco["annotations"]:
            by_image[a["image_id"]].append(a)

        self.file_names = []      # "<rec>/<cam>/Frame_N.jpg"
        self.img_wh = []          # (w, h) original
        self.centers = []         # list[np.ndarray (n_i, 2)] full-image px
        self.sexes = []           # list[list[str]] aligned with centers, len n_i
        self.num_flies = []       # str per image ("1"/"2"), for balanced_weights
        self.n_dropped_zero_ann = 0   # images with media but NO annotation -- see below
        # Iterate every IMAGE, but EXCLUDE zero-annotation ones (module
        # docstring: they are unannotated, not empty -- keeping them as
        # `num_flies=="0"` negatives was measured to train the model to
        # predict nothing on frames that DO contain a fly). Loud, not
        # silent: warn naming the count, so a future change to the source
        # data cannot silently reintroduce this as an unnoticed regression.
        for image_id, im in id2img.items():
            anns = by_image.get(image_id, [])
            fn = im["file_name"]
            if recordings is not None and not any(
                    fn.startswith(r + "/") for r in recordings):
                continue
            if len(anns) == 0:
                self.n_dropped_zero_ann += 1
                continue
            if len(anns) > K_MAX:
                raise ValueError(
                    f"V5CenterDetectDataset: image {fn!r} has {len(anns)} "
                    f"annotations, more than K_MAX={K_MAX} -- this dataset "
                    "assumes at most 2 flies per frame; a 3rd instance would "
                    "silently be dropped by every fixed-K_MAX consumer "
                    "downstream (target rendering, decode-side top-2), so "
                    "this raises instead of guessing which one to keep.")
            rec_meta = self.manifest.get(fn.split("/")[0], {})
            ctrs = []
            sexes = []
            for a in anns:
                bx, by, bw, bh = a["bbox"]
                ctrs.append([bx + bw / 2.0, by + bh / 2.0])
                sexes.append(_resolve_sex(a.get("sex", "unknown"),
                                          a.get("fly_id", 0), rec_meta))
            self.file_names.append(fn)
            self.img_wh.append((im["width"], im["height"]))
            self.centers.append(np.asarray(ctrs, dtype=np.float64).reshape(-1, 2))
            self.sexes.append(sexes)
            self.num_flies.append(str(len(anns)))

        if self.n_dropped_zero_ann:
            import warnings
            warnings.warn(
                f"V5CenterDetectDataset({split!r}): dropped "
                f"{self.n_dropped_zero_ann} image(s) with media but ZERO "
                "annotations (unannotated, not verified-empty -- see module "
                "docstring). Excluded from the dataset entirely, not kept "
                "as background negatives.", stacklevel=2)
            print(f"[V5CenterDetectDataset:{split}] WARNING: excluded "
                  f"{self.n_dropped_zero_ann} zero-annotation image(s) "
                  "(unannotated, not empty -- not used as negatives)")

    def __len__(self):
        return len(self.file_names)

    # Pure functions of self.num_flies/len(self) -- identical to V3Dataset's
    # (see module docstring); reused verbatim rather than duplicated.
    balanced_weights = V3Dataset.balanced_weights
    class_counts = V3Dataset.class_counts

    def two_fly_indices(self):
        """Row indices with exactly 2 valid instances -- the held-out-val
        two-peak-rate evaluation set."""
        return [i for i, n in enumerate(self.num_flies) if n == "2"]

    def single_fly_indices(self):
        """Row indices with exactly 1 valid instance -- the held-out-val
        FALSE-POSITIVE (spurious second peak) evaluation set: a model that
        has learned to always emit two confident peaks would show up here,
        not in ``two_fly_indices()`` (see the copy-paste-synthesis task
        brief's acceptance criterion 2 -- a two-peak-rate gain bought with
        single-fly false positives is not a gain)."""
        return [i for i, n in enumerate(self.num_flies) if n == "1"]

    def __getitem__(self, i):
        fn = self.file_names[i]
        img_w, img_h = self.img_wh[i]
        img_path = os.path.join(self.root, "images", fn)
        with Image.open(img_path) as pil:
            pil = pil.convert("RGB").resize(
                (self.image_size, self.image_size), Image.BILINEAR)
            img = np.asarray(pil, dtype=np.uint8)

        sx = self.image_size / float(img_w)
        sy = self.image_size / float(img_h)
        ctrs = self.centers[i]
        n = ctrs.shape[0]

        centers_out = np.zeros((K_MAX, 2), dtype=np.float32)
        valid = np.zeros((K_MAX,), dtype=bool)
        if n > 0:
            centers_out[:n, 0] = ctrs[:, 0] * sx
            centers_out[:n, 1] = ctrs[:, 1] * sy
            valid[:n] = True
        return img, centers_out, valid


def batches(ds, batch_size, *, shuffle=True, seed=0, drop_last=True, weights=None,
            num_workers: int = 8):
    """Yield (imgs, centers, valid) batches over the dataset for one epoch.
    Mirrors ``jarvis_jax.data.v3.batches`` exactly (weighted-with-replacement
    resampling when ``weights`` is given, threaded per-batch fetch when
    ``num_workers>1``) -- kept as its own copy rather than a shared/generic
    function because it unpacks a 3-tuple that is NOT drop-in interchangeable
    with v3.batches's (img4, kp_xy, vis) -- v3.batches's callers assume a
    448x448x4 image and (K,2)/( K,) keypoint/vis arrays; conflating the two
    risks exactly the kind of "looks generic, silently wrong shape" bug the
    v5_2d.py module docstring warns about for dataset layouts."""
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
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for s in range(0, stop, batch_size):
                sel = idx[s:s + batch_size]
                imgs, centers, valid = zip(*executor.map(lambda j: ds[int(j)], sel))
                yield (np.stack(imgs), np.stack(centers), np.stack(valid))
    else:
        for s in range(0, stop, batch_size):
            sel = idx[s:s + batch_size]
            imgs, centers, valid = zip(*(ds[int(j)] for j in sel))
            yield (np.stack(imgs), np.stack(centers), np.stack(valid))
