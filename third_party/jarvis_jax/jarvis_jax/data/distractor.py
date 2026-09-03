"""Distractor-aware supervision for the 2D keypoint detector.

WHY THIS EXISTS (docs/benchmark/2026-09-03-maskoff-attention/notes.md). On the
v12 root the mask-off ViTPose misses tarsal tips by 30-340 px on held-out
val, and the misses are not an attention failure: the backbone attends to the
same places as a checkpoint that gets those crops right. The heatmap simply
fires on tarsus-LIKE structure elsewhere -- the OTHER fly's leg tips, the
floor-reflection line -- while the true tip responds at a median 0.31 of the
max. Nothing in the training signal says those pixels are wrong: the loss's
background term averages over ~50,000 pixels at weight 0.1, so a confident
blob of ~300 px on the other fly costs almost nothing, and two-fly crops are
1.7% of train (32% of val) with no instance cue for the mask-off arm.

Three pieces, each independently switchable from configs/train/vit2d.yaml:

1. ``DistractorKeypointDataset`` appends the OTHER annotation(s) on the same
   image, mapped through the HOST's crop, as K extra keypoint rows:
   ``kp (2K,2), vis (2K,)``. Rows ``[K:]`` are never rendered as targets;
   the train step splits them off (``n_keypoints``) and turns them into the
   repulsion footprint below. Single-fly samples carry ``vis[K:] == False``.
2. ``repulsion_footprint`` renders those rows as Gaussians and takes, per
   TARGET channel k, the max over the distractor's keypoints of the SAME
   PART (``part_name``: every ``T*_TaTip`` is "TaTip", ``EyeL/EyeR`` is
   "Eye", ``Wing[LR]_V12`` is "Wing_V12"). The loss then penalises the
   target's channel k at those pixels -- "any tarsal tip of the other fly is
   not my T1L tip" -- excluding pixels inside the target's own foreground.
3. ``DistractorGrayFillDataset`` applies, with probability ``p`` per draw,
   the SAME distractor gray-fill inference already performs
   (``predict_2d``/``session_frameset``: dilate the other fly's SAM mask,
   protect the target's, fill with the crop mean). Training never did this,
   so a two-fly crop was ambiguous at train time and deployed under a
   different input distribution. Note the SAM mask is the BODY silhouette:
   the other fly's legs stay visible after the fill, which is exactly the
   structure the detector confuses, so 1-2 remain meaningful with the fill on.
   ``p < 1`` keeps unfilled two-fly crops in the stream so the model also
   learns to discriminate when a mask is missing or wrong.

Wrapper shape mirrors ``data/mask_zero.py::ZeroMaskDataset``: forward every
attribute the loop needs via ``__getattr__``. Order in ``train_keypoints``:
V5Dataset -> DistractorKeypointDataset -> [CopyPasteKeypointDataset] ->
[DistractorGrayFillDataset] -> [ZeroMaskDataset].
"""
from __future__ import annotations

import itertools
import os
import re
from collections import defaultdict

import numpy as np

from jarvis_jax.data.transforms import crop_origin, transform_keypoints

_LEG = re.compile(r"^T[1-3][LR]_(.+)$")
_WING = re.compile(r"^Wing[LR]_(.+)$")
_EYE = re.compile(r"^Eye[LR]$")


def part_name(name: str) -> str:
    """Anatomical part shared across legs and sides: 'T2R_TaTip' -> 'TaTip',
    'EyeL' -> 'Eye', 'WingR_V12' -> 'Wing_V12'; midline names unchanged."""
    m = _LEG.match(name)
    if m:
        return m.group(1)
    m = _WING.match(name)
    if m:
        return "Wing_" + m.group(1)
    if _EYE.match(name):
        return "Eye"
    return name


def build_part_index(names):
    """(part_of_k int32 (K,), parts list[str]) -- keypoint -> part id."""
    parts = sorted({part_name(n) for n in names})
    idx = {p: i for i, p in enumerate(parts)}
    return np.asarray([idx[part_name(n)] for n in names], dtype=np.int32), parts


def repulsion_footprint(kp_d, vis_d, part_of_k, *, heatmap_size=224, sigma=7.0):
    """(B,K,2) distractor keypoints (heatmap px) + (B,K) vis -> (B,H,W,K)
    footprint for the TARGET's channels: channel k = max Gaussian over the
    distractor's keypoints whose part equals part_of_k[k]. JAX, jit-safe
    (``part_of_k`` is a static numpy array)."""
    import jax.numpy as jnp
    from jarvis_jax.data.device import render_heatmaps
    fp = render_heatmaps(kp_d, vis_d, heatmap_size=heatmap_size, sigma=sigma)   # (B,H,W,K)
    part_of_k = np.asarray(part_of_k)
    n_parts = int(part_of_k.max()) + 1
    per_part = [fp[..., np.where(part_of_k == p)[0]].max(axis=-1) for p in range(n_parts)]
    fp_part = jnp.stack(per_part, axis=-1)                                      # (B,H,W,P)
    return fp_part[..., part_of_k]                                               # (B,H,W,K)


class DistractorKeypointDataset:
    """Wrap a V5Dataset so each sample carries the other fly's keypoints as
    rows [K:] (see module docstring). Must wrap the V5Dataset DIRECTLY (needs
    its raw ``keypoints``/``bboxes``/``img_wh``/``crop``/``heatmap_size``)."""

    def __init__(self, ds):
        self._ds = ds
        by_file = defaultdict(list)
        for i, fn in enumerate(ds.file_names):
            by_file[fn].append(i)
        self._others = [[j for j in by_file[fn] if j != i] for i, fn in enumerate(ds.file_names)]
        self.n_keypoints = int(ds.keypoints[0].shape[0])

    def __len__(self):
        return len(self._ds)

    def __getattr__(self, k):
        return getattr(self._ds, k)

    def has_distractor(self, i) -> bool:
        return bool(self._others[i])

    def __getitem__(self, i):
        img4, kp, vis = self._ds[i]
        K = kp.shape[0]
        d_kp = np.zeros((K, 2), np.float32)
        d_vis = np.zeros((K,), bool)
        if self._others[i]:
            w, h = self._ds.img_wh[i]
            x0, y0 = crop_origin(self._ds.bboxes[i], int(w), int(h), self._ds.crop)
            for j in self._others[i]:
                kxy, kv = transform_keypoints(self._ds.keypoints[j], x0, y0,
                                              self._ds.crop, self._ds.heatmap_size)
                take = kv & ~d_vis          # >2 flies: first visible wins per keypoint
                d_kp[take] = kxy[take]
                d_vis |= kv
        return (img4, np.concatenate([kp, d_kp], axis=0).astype(np.float32),
                np.concatenate([vis, d_vis], axis=0))


class DistractorGrayFillDataset:
    """Gray-fill the OTHER flies out of a crop with probability ``p`` per draw,
    exactly as inference does (see module docstring). Reads the root's own
    masks/<rec>/<cam>/<frame>.npz, so it composes with ZeroMaskDataset and
    DistractorKeypointDataset in either order and passes kp/vis through.

    Mirrors ``session_frameset``: dilate the distractor by ``dilate`` px,
    subtract the target dilated by the LARGER ``protect`` radius (so the
    target's own extended wing/legs are not erased), fill with the crop mean
    taken BEFORE filling. The decision to fill is drawn from an RNG keyed by
    (seed, index, draw counter), so the same sample is filled on some epochs
    and raw on others."""

    def __init__(self, ds, root, *, p=0.5, dilate=15, protect=60, seed=0, crop=None):
        self._ds, self.root, self.p = ds, root, float(p)
        self.dilate, self.protect, self.seed = int(dilate), int(protect), int(seed)
        self.crop = int(crop if crop is not None else getattr(ds, "crop", 448))
        self._draws = itertools.count()
        self.n_filled = 0
        self.n_no_distractor = 0
        self.n_skipped = 0
        # A mask npz keys its rows by whatever id the mask job used: the MERGED
        # annotation id for most recordings, the SOURCE id (src_ann_id) for
        # some (S8_male_R_amp: 1187/1330 train rows; headless_22_50_female:
        # 126/140 val rows). `_load_mask` tries both; the first version of this
        # fill compared against the merged id only and therefore treated the
        # target's OWN mask as a distractor on those recordings -- erasing the
        # fly it was supposed to protect (72 px on that val recording). Resolve
        # "mine" by EITHER key, and "distractor" as a row keyed by another
        # annotation on the same image; rows matching neither are ignored.
        self._keys_by_file = defaultdict(list)     # file -> [(ann_id, src_ann_id)]
        src = getattr(ds, "src_ann_ids", None)
        for i, fn in enumerate(ds.file_names):
            aid = int(ds.ann_ids[i])
            self._keys_by_file[fn].append((aid, int(src[i]) if src is not None else aid))

    def __len__(self):
        return len(self._ds)

    def __getattr__(self, k):
        return getattr(self._ds, k)

    @staticmethod
    def _dil(m, r):
        """Euclidean dilation by ``r`` px via a distance transform: identical
        to binary_dilation with a disc structure (verified pixel-for-pixel at
        r=15 and r=60) and ~500x faster at r=60 on a 448 crop (4 ms vs 2.1 s),
        which is the difference between a data-bound and a GPU-bound loader."""
        if r <= 0 or not m.any():
            return m
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~m) <= r

    def fill(self, img4, i):
        """Return the gray-filled copy of ``img4`` for sample ``i`` (None if
        there is no distractor mask to fill)."""
        fn = self._ds.file_names[i]
        rec, cam, base = fn.split("/")
        npz = os.path.join(self.root, "masks", rec, cam, os.path.splitext(base)[0] + ".npz")
        if not os.path.exists(npz):
            return None
        with np.load(npz) as z:
            masks, ids, matched = z["masks"], z["ann_ids"], z["matched"]
        me = int(self._ds.ann_ids[i])
        mine_keys = {me, int(getattr(self._ds, "src_ann_ids", self._ds.ann_ids)[i])}
        other_keys = set()
        for aid, sid in self._keys_by_file[fn]:
            if aid != me:
                other_keys |= {aid, sid}
        other_keys -= mine_keys
        keep = [j for j in range(len(ids)) if matched[j] and int(ids[j]) in other_keys]
        mine = [j for j in range(len(ids)) if matched[j] and int(ids[j]) in mine_keys]
        if not keep:
            return None
        w, h = self._ds.img_wh[i]
        x0, y0 = crop_origin(self._ds.bboxes[i], int(w), int(h), crop=self.crop)
        sl = (slice(y0, y0 + self.crop), slice(x0, x0 + self.crop))
        d = np.zeros(masks.shape[1:], bool)
        for j in keep:
            d |= masks[j].astype(bool)
        t = np.zeros(masks.shape[1:], bool)
        for j in mine:
            t |= masks[j].astype(bool)
        d, t = d[sl], t[sl]
        hh, ww = d.shape
        d = self._dil(d, self.dilate) & ~self._dil(t, self.protect)
        if not d.any():
            return None
        out = img4.copy()
        region = out[:hh, :ww, :3]
        mean_color = region.reshape(-1, 3).mean(axis=0)       # BEFORE filling
        region[d] = mean_color.astype(out.dtype)
        return out

    def __getitem__(self, i):
        img4, kp, vis = self._ds[i]
        rng = np.random.default_rng([self.seed, int(i), next(self._draws)])
        if rng.random() >= self.p:
            self.n_skipped += 1
            return img4, kp, vis
        out = self.fill(img4, i)
        if out is None:
            self.n_no_distractor += 1
            return img4, kp, vis
        self.n_filled += 1
        return out, kp, vis
