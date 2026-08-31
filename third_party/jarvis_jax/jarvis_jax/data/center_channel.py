"""Center-channel (instance-cue) ablation wrapper for 2D keypoint datasets.

``CenterChannelDataset`` wraps ANY dataset with the ``V3Dataset``/``V5Dataset``
``__getitem__`` contract (img4 (H,W,4) uint8, kp_xy (K,2) float32, vis (K,)
bool) and REPLACES the 4th (normally SAM-mask) channel with a Gaussian blob
centered on the TARGET annotation's own bbox center, in crop-space -- an
explicit "this is the fly you are keypointing" instance cue, standard
practice in multi-person pose estimation, in place of the SAM mask
silhouette.

Only the TARGET fly's peak is ever rendered, never a second animal's, even
though ``bboxes``/``img_wh`` in principle let a caller look up other
annotations on the same image. Feeding both flies' peaks would put the
ambiguity this channel exists to remove right back into the input: the model
would again have to guess which of two hot blobs it is supposed to key point.

Why this needs a NEW dataset attribute lookup (``ds.bboxes[i]``/
``ds.img_wh[i]``/``ds.crop``) rather than a value already returned by
``__getitem__``: the wrapped dataset's own ``__getitem__`` only returns the
final crop + heatmap-space keypoints, not the crop origin or the bbox center
in crop pixel coordinates. Both ``V3Dataset`` and ``V5Dataset`` already carry
per-row ``bboxes``/``img_wh``/``crop`` (used internally to build the exact
same crop), so recomputing ``crop_origin`` here reproduces the wrapped
dataset's own crop deterministically -- no new data dependency, no drift risk
between "the crop that was cut" and "the center this wrapper draws".

Mirrors ``jarvis_jax/data/mask_zero.py::ZeroMaskDataset``'s wrapper shape
(``__getattr__`` forwarding for everything the training loop needs --
``sampling_weights``, ``balanced_weights``, ``class_counts``, ``sex``,
``file_names``, ``heatmap_size``, ...) for the same reason: model capacity
stays fixed (still in_ch=4), and a wrapper that never existed before this
ablation cannot change behaviour for any caller that does not import it.

Evaluation-time note (see the coordinating task brief): at TRAIN time the
channel is always rendered from the GT bbox center (this file). Reporting
accuracy against that alone repeats the exact train/inference mismatch that
made the mask-off arm's zeroed-channel number ambiguous -- what actually
deploys is a channel driven by CenterDetect's PREDICTED peak, which will
disagree with the GT center by however good/bad CenterDetect's localization
is. ``predicted_centers_xy`` (optional, full-image pixel coordinates, one
entry per dataset row, aligned to the wrapped dataset's own row order) lets a
caller substitute predicted centers post-hoc for that "what actually
deploys" evaluation without a second wrapper class or touching training.
"""
from __future__ import annotations

import numpy as np

from jarvis_jax.data.transforms import crop_origin, gaussian_blob

# The fly spans roughly 150 px in a 448 crop (task sizing note). 20 px is
# ~13% of that span: wide enough that the channel carries a real "this
# region" signal after the 0..1 -> 0..255 -> back-to-0..1 uint8 round trip
# (see device.py::normalize_image_center_channel's docstring for why that
# round trip is necessary at all) -- a much smaller sigma would leave only a
# few nonzero uint8 pixels, most of the blob's mass quantized away; a much
# larger sigma starts to resemble a silhouette rather than a center cue,
# undoing the point of testing "identity without extent". Not swept --
# stated explicitly here, in one place, so it is easy to revisit.
DEFAULT_CENTER_SIGMA_PX = 20.0


class CenterChannelDataset:
    """Wrap ``ds``, replacing channel index 3 with a Gaussian at the target
    annotation's own bbox center (crop-space).

    Args:
        ds: wrapped dataset (``V3Dataset``/``V5Dataset``-shaped: needs
            ``bboxes``, ``img_wh``, and optionally ``crop`` per row).
        sigma: Gaussian std, px, in crop-space. Defaults to
            ``DEFAULT_CENTER_SIGMA_PX``.
        predicted_centers_xy: optional ``(len(ds), 2)`` array of (x, y) in
            FULL-IMAGE pixel coordinates, one row per dataset index, to draw
            the blob from INSTEAD of the GT bbox center -- the "what actually
            deploys" eval path (see module docstring). ``None`` (default) =
            GT bbox center (the training / ceiling-eval path).
    """

    def __init__(self, ds, *, sigma=None, predicted_centers_xy=None):
        self._ds = ds
        self._sigma = float(sigma) if sigma is not None else DEFAULT_CENTER_SIGMA_PX
        if predicted_centers_xy is not None:
            predicted_centers_xy = np.asarray(predicted_centers_xy, dtype=np.float64)
            if predicted_centers_xy.shape != (len(ds), 2):
                raise ValueError(
                    "predicted_centers_xy must have shape (len(ds), 2), got "
                    f"{predicted_centers_xy.shape} for len(ds)={len(ds)}")
        self._predicted = predicted_centers_xy

    def __len__(self):
        return len(self._ds)

    def _target_center_full_image(self, i):
        if self._predicted is not None:
            return float(self._predicted[i, 0]), float(self._predicted[i, 1])
        bbox = self._ds.bboxes[i]
        return float(bbox[0] + bbox[2] / 2.0), float(bbox[1] + bbox[3] / 2.0)

    def __getitem__(self, i):
        img4, kp_xy, vis = self._ds[i]
        crop = getattr(self._ds, "crop", img4.shape[0])
        img_w, img_h = self._ds.img_wh[i]
        # Reproduce the wrapped dataset's own crop origin exactly (see module
        # docstring) -- this is the SAME bbox used to cut the crop, so the
        # target-center coordinate below lands in the SAME crop-pixel frame
        # regardless of whether the predicted or GT center is used.
        bbox = self._ds.bboxes[i]
        x0, y0 = crop_origin(bbox, img_w, img_h, crop)
        fx, fy = self._target_center_full_image(i)
        cx, cy = fx - x0, fy - y0
        blob01 = gaussian_blob(crop, cx, cy, self._sigma)         # (crop,crop) float32, peak 1.0
        blob_u8 = np.clip(np.round(blob01 * 255.0), 0, 255).astype(np.uint8)
        img4 = img4.copy()
        img4[..., 3] = blob_u8
        return img4, kp_xy, vis

    def __getattr__(self, name):
        # Only reached for attributes not found on CenterChannelDataset
        # itself -- forwards everything else (sex, behavior, file_names,
        # heatmap_size, bboxes, img_wh, sampling_weights, balanced_weights,
        # class_counts, ...) to the wrapped dataset.
        return getattr(self._ds, name)
