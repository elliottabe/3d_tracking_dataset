"""Mesh-vs-mask IoU helpers used by the per-bout QC report.

Extracted from ``run_silhouette_polish.py`` when the silhouette polish was
deleted. The polish DRIVER was dead, but these three functions are not:
``qc.silhouette_iou_report`` calls ``iou_of_projected_verts`` and
``soft_iou_of_verts`` on every bout (``qc.py``, ``qc_perframe.py``), so they
ship with the live QC path and keep their unit tests.

``filled_tri_iou`` is the honest one -- it rasterises the projected mesh FACES
and IoUs that against the SAM mask. The other two splat projected VERTICES,
which under-reports coverage for a sparse vertex subset; they are kept because
the QC report's numbers are a continuous time series that would break if the
definition changed under it.
"""
from __future__ import annotations

import numpy as np

def iou_of_projected_verts(verts2d, mask_shape, ref_mask) -> float:
    """Coarse silhouette HARD-IoU: splat projected 2-D verts to a binary mask
    (round to pixel), IoU vs ref_mask. Out-of-bounds verts dropped. Pure NumPy.
    The before/after DELTA is what matters (a point-splat is sparse vs a filled
    silhouette); the baseline-comparable absolute is soft_iou_of_verts below."""
    H, W = mask_shape
    pred = np.zeros((H, W), dtype=bool)
    v = np.asarray(verts2d)
    xs = np.round(v[:, 0]).astype(int)
    ys = np.round(v[:, 1]).astype(int)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    pred[ys[ok], xs[ok]] = True
    ref = np.asarray(ref_mask, dtype=bool)
    inter = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    return float(inter / union) if union > 0 else 0.0


def filled_tri_iou(verts2d, faces, mask_shape, ref_mask) -> float:
    """Filled-triangle silhouette hard-IoU: rasterize the projected mesh faces
    (cv2.fillPoly) and IoU vs ref_mask. The overlay methodology (a true filled
    silhouette, not a point splat)."""
    import cv2
    H, W = mask_shape
    sil = np.zeros((H, W), np.uint8)
    v = np.asarray(verts2d)
    tris = v[np.asarray(faces)].astype(np.int32)     # (F,3,2)
    cv2.fillPoly(sil, tris, 1)
    sil = sil > 0
    ref = np.asarray(ref_mask, bool)
    inter = np.logical_and(sil, ref).sum()
    union = np.logical_or(sil, ref).sum()
    return float(inter / union) if union > 0 else 0.0


def soft_iou_of_verts(verts2d, mask, *, sigma: float = 1.3, splat_k: int = 2) -> float:
    """EVAL-ONLY soft-IoU (baseline-comparable). NOT an objective term.

    Windowed-Gaussian splat of the projected 2-D verts into a soft occupancy
    grid on the mask's own pixel frame, then soft-IoU vs the (binary) SAM mask,
    exactly mirroring silhouette_fit.py::soft_sil + its IoU loss (extracted here
    as a reusable, report-time-only function). Runs once per frame at report
    time -- never inside the LM loop -- so rasterizing ALL verts is fine and the
    locked "no soft-IoU objective" scope is preserved (this is a METRIC only).

    Args:
        verts2d: (V, 2) projected mesh verts in pixel (x, y) for this camera.
        mask: (H, W) SAM mask (bool / {0,1}).
        sigma: Gaussian splat sigma (matches silhouette_fit default 1.3).
        splat_k: half-window in pixels for the splat (K=2 in silhouette_fit).

    Returns:
        soft-IoU in [0, 1]; 0.0 if the mask is empty.
    """
    mask = np.asarray(mask).astype(np.float32)
    H, W = mask.shape
    if mask.sum() == 0:
        return 0.0
    v = np.asarray(verts2d, dtype=np.float64)
    grid = np.zeros((H, W), dtype=np.float64)
    bx = np.floor(v[:, 0]).astype(int)
    by = np.floor(v[:, 1]).astype(int)
    for di in range(-splat_k, splat_k + 1):
        for dj in range(-splat_k, splat_k + 1):
            cx = bx + dj
            cy = by + di
            w = np.exp(-(((cx + 0.5 - v[:, 0]) ** 2 + (cy + 0.5 - v[:, 1]) ** 2))
                       / (2 * sigma ** 2))
            valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
            np.add.at(grid, (np.clip(cy, 0, H - 1), np.clip(cx, 0, W - 1)),
                      np.where(valid, w, 0.0))
    soft = 1.0 - np.exp(-grid)                       # soft occupancy in [0,1)
    inter = float((soft * mask).sum())
    union = float((soft + mask - soft * mask).sum())
    return inter / union if union > 0 else 0.0


