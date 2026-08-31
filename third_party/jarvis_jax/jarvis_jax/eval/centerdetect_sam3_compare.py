"""Pure aggregation for the CenterDetect-vs-SAM3-centroid localisation
comparison (the acceptance bar is SAM3, not just a peak count -- see the
task brief). Kept separate from ``scripts/eval_centerdetect_sam3.py``'s video
I/O + model inference orchestration so the statistics themselves are
unit-testable without a real bout video, checkpoint, or GPU.
"""
from __future__ import annotations

import numpy as np


def nearest_peak_distances(peaks_full_xy, sam3_centroids_xy, sam3_valid):
    """One (frame, camera) sample: distance from each valid SAM3 centroid to
    its NEAREST decoded peak (full-image px, both top-2 peaks considered).

    Args:
        peaks_full_xy: (k, 2) decoded peaks, full-image px.
        sam3_centroids_xy: (2, 2) SAM3 centroids for this sample (slot, xy).
        sam3_valid: (2,) bool -- which SAM3 slots are valid this sample.

    Returns:
        list of (slot, dist_px) for each valid SAM3 slot.
    """
    peaks_full_xy = np.asarray(peaks_full_xy, dtype=np.float64)
    out = []
    for slot in range(sam3_centroids_xy.shape[0]):
        if not sam3_valid[slot]:
            continue
        c = np.asarray(sam3_centroids_xy[slot], dtype=np.float64)
        d = np.linalg.norm(peaks_full_xy - c[None, :], axis=-1)
        out.append((slot, float(d.min())))
    return out


def summarize(samples):
    """``samples``: list of per-(frame,camera) dicts, each with at minimum
    ``peaks_full_xy`` (k,2), ``sam3_centroids_xy`` (2,2), ``sam3_valid`` (2,).
    Returns overall distance distribution + the two-peak-hit rate (using
    ``jarvis_jax.eval.centerdetect_decode.two_peak_hit`` when both SAM3 slots
    are valid for that sample).

    This mirrors the multianimal-collapse measurement's own definitions (see
    ``.superpowers/sdd/2026-08-29-coarse-to-fine-3d/centerdetect-multianimal.md``)
    so the JAX numbers are directly comparable to the PyTorch baseline/retrain
    numbers already on record.
    """
    from jarvis_jax.eval.centerdetect_decode import two_peak_hit

    all_dists = []
    hits = []
    n_both_valid = 0
    for s in samples:
        peaks = np.asarray(s["peaks_full_xy"], dtype=np.float64)
        cents = np.asarray(s["sam3_centroids_xy"], dtype=np.float64)
        valid = np.asarray(s["sam3_valid"], dtype=bool)
        for _, d in nearest_peak_distances(peaks, cents, valid):
            all_dists.append(d)
        if valid.all():
            n_both_valid += 1
            is_hit, _, _ = two_peak_hit(peaks, cents, capture_radius_px=s.get("capture_radius_px", 40.0))
            hits.append(is_hit)

    all_dists = np.asarray(all_dists, dtype=np.float64)
    hits = np.asarray(hits, dtype=bool)
    out = {
        "n_samples": len(samples),
        "n_both_valid": n_both_valid,
        "n_dist_obs": int(all_dists.size),
        "two_peak_rate": float(hits.mean()) if hits.size else float("nan"),
    }
    if all_dists.size:
        out.update({
            "dist_mean_px": float(all_dists.mean()),
            "dist_median_px": float(np.median(all_dists)),
            "dist_p10_px": float(np.percentile(all_dists, 10)),
            "dist_p90_px": float(np.percentile(all_dists, 90)),
            "dist_max_px": float(all_dists.max()),
        })
    return out
