"""Decode-side top-2 peak extraction for CenterDetect (multi-animal, single
output channel).

Port of the PyTorch production decode
(``third_party/JARVIS-HybridNet/jarvis/prediction/multi_peak.py::
extract_top_k_peaks``): iterative-NMS peak extraction over a single-channel
heatmap, ALWAYS returning exactly `k` peaks with NO confidence threshold --
so downstream code always has two candidates to work with. This cannot
rescue a genuinely absent second peak (if the model never learned to raise
one, the 2nd "peak" returned is just the strongest surviving background
pixel after excluding the 1st peak's suppression disk); it converts a
weak-but-real second peak into a usable detection for free once training
(oversampling + per-instance loss) has taught the model to raise one.

`suppression_radius=15` (heatmap px, matching production) is kept as the
default deliberately, per the task brief: no measured cause was found here
to change it (this port does not sweep it, and the JAX resolution and
receptive field are the same as PyTorch's -- k, output size, and sigma all
transcribed from the SAME `_MODEL_SIZE_TABLE`/HeatmapGenerator convention),
so changing the constant with no evidence would just be a second unguarded
free parameter.
"""
from __future__ import annotations

import numpy as np


def extract_top_k_peaks(heatmap, *, k=2, suppression_radius=15):
    """(B,H,W) or (B,H,W,1) heatmap -> (peaks_xy (B,k,2), conf (B,k)).

    Iterative NMS: take the global argmax, record it, zero out a disk of
    radius `suppression_radius` around it, repeat `k` times. No confidence
    threshold anywhere in this function -- callers that want a
    "is this a real second animal" decision apply their own threshold to the
    returned `conf`, matching production's `assign_peaks_across_cameras`
    (this function only ever hands it two unconditioned candidates).

    Pure numpy (this runs at inference/eval time on already-materialized
    heatmaps, not inside a training gradient -- no need for it to be
    differentiable or traced).
    """
    hm = np.asarray(heatmap)
    if hm.ndim == 4:
        if hm.shape[-1] != 1:
            raise ValueError(
                f"extract_top_k_peaks expects a single-channel heatmap, got "
                f"shape {hm.shape}")
        hm = hm[..., 0]
    b, h, w = hm.shape
    working = hm.copy()

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    peaks = np.zeros((b, k, 2), dtype=np.float32)
    conf = np.zeros((b, k), dtype=np.float32)
    for j in range(k):
        flat = working.reshape(b, h * w)
        m = flat.argmax(axis=1)                      # (B,)
        py = (m // w).astype(np.float32)
        px = (m % w).astype(np.float32)
        peaks[:, j, 0] = px
        peaks[:, j, 1] = py
        conf[:, j] = flat[np.arange(b), m]
        for bi in range(b):
            mask = ((xx - px[bi]) ** 2 + (yy - py[bi]) ** 2) <= suppression_radius ** 2
            working[bi][mask] = -np.inf
    return peaks, conf


def peaks_to_full_image(peaks_xy, heatmap_size, img_w, img_h):
    """Rescale (..., 2) heatmap-space (x, y) peaks to ORIGINAL full-image
    pixel coordinates, undoing the SAME non-uniform (aspect-distorting)
    resize ``V5CenterDetectDataset``/JARVIS's own CenterDetect preprocessing
    applies (separate x/y scale factors -- matches
    ``Dataset2D._build_augpipe``'s ``scale_width``/``scale_height``, NOT a
    single uniform scale)."""
    peaks_xy = np.asarray(peaks_xy, dtype=np.float64)
    out = peaks_xy.copy()
    out[..., 0] *= img_w / float(heatmap_size)
    out[..., 1] *= img_h / float(heatmap_size)
    return out


def two_peak_hit(peaks_full_xy, gt_centers_xy, *, capture_radius_px=40.0):
    """One frame's decision: did the top-2 peaks (full-image px) correctly
    separate two GT centers (full-image px)?

    Matches the multianimal-collapse measurement's own definition (see
    ``.superpowers/sdd/2026-08-29-coarse-to-fine-3d/centerdetect-multianimal.md``
    Step 1): each GT fly's NEAREST peak must be a DIFFERENT peak index, and
    both nearest distances must be within `capture_radius_px` -- otherwise
    ("collapsed to one" OR "one/both GT flies have no nearby peak") this is
    NOT a hit. Returns (is_two_peak_hit: bool, dists: (2,) nearest distance
    per GT fly in px, matched_peak_idx: (2,) which peak matched which GT).
    """
    peaks_full_xy = np.asarray(peaks_full_xy, dtype=np.float64)   # (k,2), k>=2
    gt = np.asarray(gt_centers_xy, dtype=np.float64)              # (2,2)
    d = np.linalg.norm(peaks_full_xy[None, :, :] - gt[:, None, :], axis=-1)  # (2,k)
    nearest_idx = d.argmin(axis=1)                                 # (2,)
    nearest_dist = d[np.arange(2), nearest_idx]                    # (2,)
    is_hit = bool(nearest_idx[0] != nearest_idx[1]
                 and nearest_dist[0] <= capture_radius_px
                 and nearest_dist[1] <= capture_radius_px)
    return is_hit, nearest_dist, nearest_idx


def grayfill_peaks(forward, imgs_u8, *, n_peaks=2, gray_radius_xy=(34, 46),
                   fill="mean", return_heatmaps=False):
    """Find `n_peaks` centres by ERASING each one from the IMAGE and re-running.

    ``extract_top_k_peaks`` suppresses a disk in the HEATMAP, which cannot
    raise a peak the network declined to commit to: the first fly is still in
    the input, still competing, so a second fly it half-saw stays half-seen.
    Measured on real courtship bouts, that is the actual failure -- the second
    peak's LOCATION is often right (11-44 px) while its confidence collapses
    (conf2/conf1 0.02-0.05 when the pair sits near a frame edge).

    This instead gray-fills the found animal out of the RGB input and asks
    again, so each subsequent animal is detected on a frame where it is the
    ONLY animal and can take the full confidence of a first peak. It is the
    same trick JARVIS already uses for the KEYPOINT detector, where the
    distractor fly's mask pixels are replaced by the crop mean
    (``predict.session_frameset``) -- applied here to centre detection, and
    without needing masks: the erased region is an ellipse around the peak.

    Cost is `n_peaks` forward passes instead of one, which is still far
    cheaper than a SAM pass over the same frames.

    Args:
        forward: callable (B,H,W,3) float32 image batch -> (B,h,w[,1]) heatmap.
            Must apply whatever normalisation the model expects.
        imgs_u8: (B,H,W,3) uint8 model-input-sized images.
        n_peaks: how many animals to find -- JARVIS's ``num_animals``. 1
            reduces to a plain argmax; there is no upper limit beyond cost.
        gray_radius_xy: (rx, ry) ellipse semi-axes IN MODEL-INPUT PIXELS for
            the erased region. Default (34, 46) covers one fly in a 1936x448
            frame squashed to 320x320 (the squash is anisotropic -- ~6.0x in x
            but ~1.4x in y -- so a circle in model space is badly wrong; this
            is a circle in REAL space).
        fill: "mean" (JARVIS's convention) or "median", computed per image over
            the pixels not yet erased.

    Returns:
        (peaks_xy (B,n_peaks,2) in HEATMAP coords, conf (B,n_peaks)); each
        conf is that animal's own first-peak confidence, not a residual.
        With `return_heatmaps`, also a list of the per-pass heatmaps.
    """
    imgs = np.asarray(imgs_u8)
    if imgs.ndim != 4 or imgs.shape[-1] != 3:
        raise ValueError(f"imgs_u8 must be (B,H,W,3), got {imgs.shape}")
    if int(n_peaks) < 1:
        raise ValueError(f"n_peaks must be >= 1, got {n_peaks}")
    rx, ry = (float(gray_radius_xy[0]), float(gray_radius_xy[1]))
    if rx <= 0 or ry <= 0:
        raise ValueError(f"gray_radius_xy must be positive, got {gray_radius_xy}")

    b, H, W = imgs.shape[:3]
    work = imgs.astype(np.float32).copy()
    erased = np.zeros((b, H, W), dtype=bool)
    iyy, ixx = np.mgrid[0:H, 0:W].astype(np.float32)

    peaks = np.zeros((b, int(n_peaks), 2), dtype=np.float32)
    conf = np.zeros((b, int(n_peaks)), dtype=np.float32)
    hms = []
    for j in range(int(n_peaks)):
        hm = np.asarray(forward(work))
        if hm.ndim == 4:
            hm = hm[..., 0]
        hms.append(hm)
        hh, hw = hm.shape[1:3]
        flat = hm.reshape(b, hh * hw)
        m = flat.argmax(axis=1)
        py, px = (m // hw).astype(np.float32), (m % hw).astype(np.float32)
        peaks[:, j, 0], peaks[:, j, 1] = px, py
        conf[:, j] = flat[np.arange(b), m]
        if j == int(n_peaks) - 1:
            break
        # Erase this animal from the INPUT, in image coords.
        cx = px * (W / float(hw))
        cy = py * (H / float(hh))
        for i in range(b):
            disk = (((ixx - cx[i]) / rx) ** 2 + ((iyy - cy[i]) / ry) ** 2) <= 1.0
            keep = ~erased[i]
            src = work[i][keep] if keep.any() else work[i].reshape(-1, 3)
            val = (src.mean(axis=0) if fill == "mean"
                   else np.median(src.reshape(-1, 3), axis=0))
            work[i][disk] = val
            erased[i] |= disk
    return (peaks, conf, hms) if return_heatmaps else (peaks, conf)
