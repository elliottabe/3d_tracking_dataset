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
