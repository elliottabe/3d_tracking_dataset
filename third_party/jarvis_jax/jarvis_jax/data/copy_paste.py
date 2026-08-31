"""Copy-paste synthesis of two-fly CenterDetect training frames.

Only 1,487 of 22,040 ``V5CenterDetectDataset`` train images contain two
flies (6.7%) -- oversampling those 1,487 images (``balanced_weights(key=
"num_flies")``) reuses the SAME images every epoch and cannot add pose,
position, or separation diversity. The other 20,553 train images have
exactly ONE fly, each with its own SAM mask already on disk
(``masks/<rec>/<cam>/Frame_N.npz``, the same store ``data/v5_2d.py::
_load_mask`` reads). Cutting a fly out along its mask and pasting it into a
DIFFERENT single-fly frame turns that into unlimited two-fly training data
with exact known centers and controllable separation -- including the close
separations (real courtship pairs sit at ~267-336px; the model's failures
are CLOSER than that) that are scarcest in the real data and hardest for the
model.

SAM3 is used ONCE, OFFLINE, at DATASET-BUILD time (the masks were already
generated for the mask-channel/QC pipeline) -- this module never calls SAM3
itself, it only reads the masks already on disk to train the model that
replaces SAM3 at inference.

Design constraints (see the coordinating task brief):

  1. **Same camera only.** Each of the 7 cameras (Cam2012630/631/853/855/857/
     861/862) has its own viewpoint, lighting, and platform geometry (one is
     top-down, the rest oblique ~29-31 degrees) -- pasting across cameras
     would be a physically impossible image, and even though every v5 frame
     happens to share the same (1936, 448) pixel canvas, the apparent SCALE
     of a fly at a given real-world size still differs by camera (lens/
     distance), so cross-camera pasting could not be scale-matched even with
     equal-sized canvases. Same-recording is preferred within that (closer
     lighting/platform-state match) but same-camera is the hard floor --
     never relaxed, even when the same-recording pool is empty (donor pool
     falls back to same-camera-any-recording, never to a different camera).
  2. **Randomised z-order.** Real overlaps occlude in both directions;
     always pasting the donor on top would teach a z-order prior that isn't
     real. ``top_is_donor`` is an independent coin flip per sample.
  3. **Separation sampled with real weight at the close end.** See
     ``sample_separation_px``.
  4. **Feathered edges.** The donor's binary SAM mask is Gaussian-blurred
     before compositing (``feather_sigma_px``) so the paste has a soft alpha
     ramp, not a hard silhouette edge -- a hard edge is a free shortcut
     feature ("is there a paste seam here") the model would happily learn
     instead of "is there a second fly here".
  5. **Train only.** ``__init__`` raises if the wrapped dataset's split is
     not "train" -- contaminating val with synthetic frames would make the
     whole experiment unfalsifiable (the acceptance criterion is real-val
     two-peak rate; a model that learns to detect paste seams would ace a
     synthetic val for the wrong reason and this guard is what stops that
     from ever being possible to do by accident).

Wrapper shape mirrors ``data/mask_zero.py::ZeroMaskDataset`` and
``data/center_channel.py::CenterChannelDataset``: wraps a
``V5CenterDetectDataset`` instance, forwards everything the training loop
needs (``num_flies``, ``balanced_weights``, ``class_counts``, ``sexes``,
``file_names``, ...) via ``__getattr__``, and reproduces the wrapped
dataset's own image-loading/resize convention here rather than depending on
new fields on ``V5CenterDetectDataset`` -- so this module is the only thing
that changes if the composite recipe changes, and importing it changes
nothing for any caller that does not.

Only TRUE single-fly rows (``num_flies == "1"``) are ever candidates to
receive a paste, and only a `p` fraction of THOSE draws are actually
synthesized (the rest pass through as real, unmodified single-fly frames)
-- real two-fly rows are never touched (they are already real, and pasting
a 3rd fly onto them would violate this dataset's own K_MAX=2 assumption).
`p` is the single knob that trades off the two acceptance metrics against
each other: too high and the model rarely sees a genuine single-fly frame
(the false-positive/hallucination risk the task brief calls out by name);
too low and the two-peak-rate gain shrinks back toward the oversampling-only
ceiling. Combined with `num_flies` oversampling upstream (real 2-fly frames
already ~21.2% of samples at alpha=0.5), the TOTAL two-fly-labelled sample
fraction is approximately::

    0.212 + 0.788 * p

``p=0.3`` (this module's CLI default, not hardcoded here) targets ~45%,
deliberately short of "the majority of what the model sees is two flies" --
stated here, in one place, so it is easy to revisit against the measured
false-positive rate.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from jarvis_jax.data.v5_2d import _load_mask

# Real courtship two-fly centroid separation sits ~267-336px (task brief).
# The failures this synthesis exists to fix are CLOSER than that, so the
# default range starts well below it (30px -- near-total overlap) and the
# low/high split point (180px) puts the majority of mass (`near_frac`)
# strictly below the real range, with the remainder spanning up through and
# a bit past it (180-400px) for continuity/coverage. Not swept -- a
# deliberate, documented choice; see `sample_separation_px`.
DEFAULT_SEP_LOW_PX = 30.0
DEFAULT_SEP_HIGH_PX = 400.0
DEFAULT_SEP_NEAR_BOUNDARY_PX = 180.0
DEFAULT_SEP_NEAR_FRAC = 0.6

# Gaussian std (px, full-image resolution) the donor's binary mask is
# blurred by before alpha-compositing -- a soft ramp a few px wide, not a
# silhouette. Not swept; a few px is enough that the paste boundary is not a
# perfect step function without starting to look like a fog around the fly.
DEFAULT_FEATHER_SIGMA_PX = 3.0


def sample_separation_px(rng, *, low=DEFAULT_SEP_LOW_PX, high=DEFAULT_SEP_HIGH_PX,
                          near_boundary=DEFAULT_SEP_NEAR_BOUNDARY_PX,
                          near_frac=DEFAULT_SEP_NEAR_FRAC):
    """Sample one target centroid separation (px), a 2-component mixture:
    with probability `near_frac`, Uniform(low, near_boundary) (the close,
    hard-for-the-model end this synthesis exists to cover); otherwise
    Uniform(near_boundary, high) (spanning up through and past the real
    ~267-336px courtship range). `rng`: numpy Generator (``np.random.
    default_rng``), for reproducibility per-sample."""
    if rng.random() < near_frac:
        return float(rng.uniform(low, near_boundary))
    return float(rng.uniform(near_boundary, high))


def _load_rgb_full(root, file_name):
    path = os.path.join(root, "images", file_name)
    with Image.open(path) as pil:
        return np.asarray(pil.convert("RGB"), dtype=np.uint8)


def _feathered_sprite(rgb_full, mask_full, bbox, *, pad, sigma):
    """Crop a padded window around `bbox` from `rgb_full`/`mask_full` and
    return (crop_rgb (h,w,3) uint8, alpha (h,w) float32 in [0,1] feathered,
    (cx, cy) bbox center IN CROP coordinates, (x0, y0) crop origin in the
    source image). Blur is applied to the CROP (not the full image) for
    speed; `pad` must be large enough that the blur's falloff is fully
    captured within the crop (>= ~4 sigma) so cropping-then-blurring is
    equivalent to blurring the full mask then cropping (blur commutes with
    translation, not with a tight crop that clips its own tails)."""
    h_img, w_img = mask_full.shape
    bx, by, bw, bh = bbox
    x0 = max(0, int(np.floor(bx)) - pad)
    y0 = max(0, int(np.floor(by)) - pad)
    x1 = min(w_img, int(np.ceil(bx + bw)) + pad)
    y1 = min(h_img, int(np.ceil(by + bh)) + pad)
    crop_rgb = rgb_full[y0:y1, x0:x1]
    crop_mask = mask_full[y0:y1, x0:x1].astype(np.float32)
    alpha = np.clip(gaussian_filter(crop_mask, sigma=sigma), 0.0, 1.0)
    cx = bx + bw / 2.0 - x0
    cy = by + bh / 2.0 - y0
    return crop_rgb, alpha, (cx, cy), (x0, y0)


def _composite(base_rgb, sprite_rgb, sprite_alpha, top_left_xy):
    """Alpha-composite `sprite_rgb`/`sprite_alpha` onto a COPY of `base_rgb`
    at integer top-left `top_left_xy`, clipped to `base_rgb`'s bounds (a
    sprite that lands fully or partly off-frame is simply clipped, not an
    error -- callers that need the WHOLE sprite on-frame should clamp the
    placement center beforehand; see `CopyPasteCenterDetectDataset`)."""
    out = base_rgb.copy()
    h_img, w_img = out.shape[:2]
    x0, y0 = int(round(top_left_xy[0])), int(round(top_left_xy[1]))
    sh, sw = sprite_alpha.shape
    dx0, dy0 = max(x0, 0), max(y0, 0)
    dx1, dy1 = min(x0 + sw, w_img), min(y0 + sh, h_img)
    if dx0 >= dx1 or dy0 >= dy1:
        return out
    sx0, sy0 = dx0 - x0, dy0 - y0
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
    a = sprite_alpha[sy0:sy1, sx0:sx1][..., None]
    region = out[dy0:dy1, dx0:dx1].astype(np.float32)
    src = sprite_rgb[sy0:sy1, sx0:sx1].astype(np.float32)
    out[dy0:dy1, dx0:dx1] = np.clip(region * (1.0 - a) + src * a, 0, 255).astype(np.uint8)
    return out


class CopyPasteCenterDetectDataset:
    """Wrap a ``V5CenterDetectDataset`` (TRAIN split only) so a `p` fraction
    of single-fly draws become a synthetic two-fly composite: a donor fly,
    cut out along its own SAM mask from a DIFFERENT single-fly image on the
    SAME camera (same recording preferred), feather-blended in at a sampled
    separation/direction from the host fly, with independently randomised
    top/bottom z-order.

    Every other row (real two-fly images; single-fly draws that lose the `p`
    coin flip) passes through completely unmodified.

    Args:
        ds: a ``V5CenterDetectDataset`` instance, split == "train".
        root: dataset root (defaults to ``ds.root``) -- only needed because
            ``V5CenterDetectDataset.__getitem__`` returns the RESIZED image,
            not the full-resolution one this module composites in.
        p: probability that a single-fly draw is turned into a synthetic
           composite (see module docstring for the resulting overall
           two-fly sample fraction).
        sep_low/sep_high/sep_near_boundary/sep_near_frac: passed to
           `sample_separation_px`.
        feather_sigma: passed to `_feathered_sprite`.
        seed: base seed for this wrapper's own RNG (separate from any
           shuffling/oversampling RNG upstream -- deterministic given the
           SAME (index, draw) sequence, but not tied to it).
    """

    def __init__(self, ds, *, root=None, p=0.3,
                 sep_low=DEFAULT_SEP_LOW_PX, sep_high=DEFAULT_SEP_HIGH_PX,
                 sep_near_boundary=DEFAULT_SEP_NEAR_BOUNDARY_PX,
                 sep_near_frac=DEFAULT_SEP_NEAR_FRAC,
                 feather_sigma=DEFAULT_FEATHER_SIGMA_PX, seed=0):
        if ds.split != "train":
            raise ValueError(
                f"CopyPasteCenterDetectDataset must wrap the TRAIN split "
                f"only -- got split={ds.split!r}. Synthesising two-fly val "
                f"frames would make the whole experiment unfalsifiable "
                f"(the acceptance criterion IS real-val two-peak rate).")
        self._ds = ds
        self.root = root or ds.root
        self.p = float(p)
        self.sep_low = float(sep_low)
        self.sep_high = float(sep_high)
        self.sep_near_boundary = float(sep_near_boundary)
        self.sep_near_frac = float(sep_near_frac)
        self.feather_sigma = float(feather_sigma)
        self._seed = int(seed)

        # Per-image annotation metadata (ann_id, src_ann_id, bbox) for every
        # SINGLE-fly row -- V5CenterDetectDataset itself does not expose
        # these (only the already-scaled bbox CENTER, and no ann id at all,
        # neither of which is enough to look a mask up or cut a sprite), so
        # this re-reads the same COCO json V5CenterDetectDataset already
        # parsed. Re-parsing (rather than adding these fields to
        # V5CenterDetectDataset) keeps this module's dependency on the
        # shared dataset class to its PUBLIC attributes only.
        ann_path = os.path.join(self.root, "annotations",
                                 f"instances_{ds.split}.json")
        with open(ann_path) as f:
            coco = json.load(f)
        file_to_imgid = {im["file_name"]: im["id"] for im in coco["images"]}
        anns_by_image = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_image[a["image_id"]].append(a)

        self._single_idx = []          # ds row indices with num_flies == "1"
        self._ann_meta = {}            # file_name -> (ann_id, src_ann_id, bbox np.float64[4])
        by_camera = defaultdict(list)
        by_rec_camera = defaultdict(list)
        for i, fn in enumerate(ds.file_names):
            if ds.num_flies[i] != "1":
                continue
            img_id = file_to_imgid[fn]
            anns = anns_by_image[img_id]
            if len(anns) != 1:
                # Should be impossible (num_flies=="1" means exactly one
                # non-dropped annotation) -- guard rather than silently
                # picking anns[0] if this invariant is ever violated.
                raise ValueError(
                    f"CopyPasteCenterDetectDataset: {fn!r} has num_flies=='1' "
                    f"but {len(anns)} annotations in the COCO file -- "
                    f"dataset/annotation mismatch.")
            a = anns[0]
            ann_id = int(a["id"])
            src_ann_id = int(a.get("src_ann_id", ann_id))
            bbox = np.asarray(a["bbox"], dtype=np.float64)
            self._single_idx.append(i)
            self._ann_meta[fn] = (ann_id, src_ann_id, bbox)
            rec, cam, _ = fn.split("/")
            by_camera[cam].append(i)
            by_rec_camera[(rec, cam)].append(i)
        self._by_camera = by_camera
        self._by_rec_camera = by_rec_camera

    def __len__(self):
        return len(self._ds)

    def __getattr__(self, name):
        return getattr(self._ds, name)

    # -- donor selection --------------------------------------------------

    def _donor_pool(self, host_idx):
        """Row indices eligible to donate a fly onto `host_idx`: same
        recording+camera preferred, else same camera (any recording) --
        NEVER a different camera (see module docstring, constraint 1).
        Excludes `host_idx` itself."""
        fn = self._ds.file_names[host_idx]
        rec, cam, _ = fn.split("/")
        pool = [j for j in self._by_rec_camera.get((rec, cam), ()) if j != host_idx]
        same_recording = bool(pool)
        if not pool:
            pool = [j for j in self._by_camera.get(cam, ()) if j != host_idx]
        return pool, same_recording

    # -- one synthesis --------------------------------------------------

    def _synthesize(self, host_idx, rng):
        fn = self._ds.file_names[host_idx]
        img_w, img_h = self._ds.img_wh[host_idx]
        host_ann_id, host_src_id, host_bbox = self._ann_meta[fn]

        pool, same_recording = self._donor_pool(host_idx)
        donor_idx = int(rng.choice(pool))
        donor_fn = self._ds.file_names[donor_idx]
        donor_ann_id, donor_src_id, donor_bbox = self._ann_meta[donor_fn]

        host_rgb = _load_rgb_full(self.root, fn)
        donor_rgb = _load_rgb_full(self.root, donor_fn)
        host_mask = _load_mask(self.root, fn, host_ann_id, host_src_id, img_w, img_h)
        d_img_w, d_img_h = self._ds.img_wh[donor_idx]
        donor_mask = _load_mask(self.root, donor_fn, donor_ann_id, donor_src_id,
                                 d_img_w, d_img_h)

        pad = max(20, int(round(4 * self.feather_sigma)))
        donor_crop_rgb, donor_alpha, donor_center_in_crop, _ = _feathered_sprite(
            donor_rgb, donor_mask, donor_bbox, pad=pad, sigma=self.feather_sigma)
        host_crop_rgb, host_alpha, host_center_in_crop, host_crop_origin = _feathered_sprite(
            host_rgb, host_mask, host_bbox, pad=pad, sigma=self.feather_sigma)

        host_center = np.array([host_bbox[0] + host_bbox[2] / 2.0,
                                 host_bbox[1] + host_bbox[3] / 2.0])

        sep_sampled = sample_separation_px(
            rng, low=self.sep_low, high=self.sep_high,
            near_boundary=self.sep_near_boundary, near_frac=self.sep_near_frac)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        raw_target = host_center + sep_sampled * np.array([np.cos(theta), np.sin(theta)])

        half_w = donor_crop_rgb.shape[1] / 2.0
        half_h = donor_crop_rgb.shape[0] / 2.0
        lo_x, hi_x = sorted((half_w, img_w - half_w))
        lo_y, hi_y = sorted((half_h, img_h - half_h))
        target_x = float(np.clip(raw_target[0], lo_x, hi_x))
        target_y = float(np.clip(raw_target[1], lo_y, hi_y))
        sep_achieved = float(np.hypot(target_x - host_center[0], target_y - host_center[1]))

        dst_top_left = (target_x - donor_center_in_crop[0],
                         target_y - donor_center_in_crop[1])

        top_is_donor = bool(rng.random() < 0.5)
        composite = _composite(host_rgb, donor_crop_rgb, donor_alpha, dst_top_left)
        if not top_is_donor:
            # Re-lay the host's OWN fly on top at its OWN (untranslated)
            # location, using its own feathered alpha -- this is what makes
            # the host occlude the donor in their overlap region instead of
            # the donor always winning ties (constraint 2).
            composite = _composite(composite, host_crop_rgb, host_alpha, host_crop_origin)

        meta = {
            "host_file_name": fn, "donor_file_name": donor_fn,
            "same_recording": same_recording, "top_is_donor": top_is_donor,
            "sep_sampled_px": sep_sampled, "sep_achieved_px": sep_achieved,
            "host_center_xy": (float(host_center[0]), float(host_center[1])),
            "target_center_xy": (target_x, target_y),
        }
        return composite, host_center, np.array([target_x, target_y]), meta

    def __getitem__(self, i):
        img, centers, valid = self._ds[i]
        if self._ds.num_flies[i] != "1":
            return img, centers, valid   # real two-fly row -- untouched

        # One RNG stream per (seed, row index): first draw decides whether
        # to synthesize at all, the rest (donor pick, separation, angle,
        # z-order) come from the SAME stream inside `_synthesize` -- both
        # deterministic given (seed, i), but callers that want fresh
        # variety across epochs pass a different seed per epoch (mirrors
        # `batches()`'s own per-epoch `seed=seed+epoch` convention).
        rng = np.random.default_rng([self._seed, i])
        if rng.random() >= self.p:
            return img, centers, valid   # coin flip: stays a real single-fly frame
        pool, _ = self._donor_pool(i)
        if not pool:
            # No eligible same-camera donor at all (should not happen on the
            # real dataset -- ~2900 single-fly images per camera on
            # average -- but degrade to "stays real" rather than crashing
            # on rng.choice(empty) for a pathological/tiny root).
            return img, centers, valid

        composite_full, host_c, target_c, _meta = self._synthesize(i, rng)
        img_w, img_h = self._ds.img_wh[i]
        pil = Image.fromarray(composite_full).resize(
            (self._ds.image_size, self._ds.image_size), Image.BILINEAR)
        img_out = np.asarray(pil, dtype=np.uint8)

        sx = self._ds.image_size / float(img_w)
        sy = self._ds.image_size / float(img_h)
        centers_out = np.zeros_like(centers)
        centers_out[0] = [host_c[0] * sx, host_c[1] * sy]
        centers_out[1] = [target_c[0] * sx, target_c[1] * sy]
        valid_out = np.array([True, True])
        return img_out, centers_out.astype(np.float32), valid_out

    def sample_with_meta(self, i, *, seed=None):
        """Like ``__getitem__`` but ALWAYS synthesizes (ignores `p`) and
        also returns the composite's metadata dict -- for
        `--dump-samples`/visual verification, not used by training. Raises
        if row `i` is not single-fly (nothing to paste onto in this row)."""
        if self._ds.num_flies[i] != "1":
            raise ValueError(f"sample_with_meta: row {i} is not single-fly "
                             f"(num_flies={self._ds.num_flies[i]!r})")
        rng = np.random.default_rng([self._seed if seed is None else seed, i, 1])
        composite_full, host_c, target_c, meta = self._synthesize(i, rng)
        img_w, img_h = self._ds.img_wh[i]
        pil = Image.fromarray(composite_full).resize(
            (self._ds.image_size, self._ds.image_size), Image.BILINEAR)
        img_out = np.asarray(pil, dtype=np.uint8)
        sx = self._ds.image_size / float(img_w)
        sy = self._ds.image_size / float(img_h)
        centers_out = np.array([[host_c[0] * sx, host_c[1] * sy],
                                 [target_c[0] * sx, target_c[1] * sy]], dtype=np.float32)
        return composite_full, img_out, centers_out, meta
