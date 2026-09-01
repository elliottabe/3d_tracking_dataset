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

def _in_bounds_pixels(verts2d, H, W):
    """Unique flat pixel indices the rounded verts land on, in-bounds only.

    This is exactly the set of True pixels of the old dense ``pred`` mask:
    rounding to pixel then dropping out-of-bounds. Non-finite verts round to
    INT64_MIN and are dropped by the bounds test, same as before.
    """
    v = np.asarray(verts2d)
    with np.errstate(invalid="ignore"):
        xs = np.round(v[:, 0]).astype(np.int64)
        ys = np.round(v[:, 1]).astype(np.int64)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if not ok.any():
        return np.zeros(0, np.int64)
    return np.unique(ys[ok] * W + xs[ok])


def iou_of_projected_verts(verts2d, mask_shape, ref_mask) -> float:
    """Coarse silhouette HARD-IoU: splat projected 2-D verts to a binary mask
    (round to pixel), IoU vs ref_mask. Out-of-bounds verts dropped. Pure NumPy.
    The before/after DELTA is what matters (a point-splat is sparse vs a filled
    silhouette); the baseline-comparable absolute is soft_iou_of_verts below.

    Sparse since 2026-09-01: instead of materialising an (H, W) ``pred`` mask
    and running full-frame ``logical_and``/``logical_or`` (867k px per camera
    per frame in the live QC path), it counts on the |verts| pixels the splat
    actually touches. |pred & ref| and |pred | ref| are integer counts, so this
    is BIT-IDENTICAL, not merely close -- see
    tests/test_mesh_iou_fast_equivalence.py.
    """
    H, W = mask_shape
    ref = np.asarray(ref_mask, dtype=bool)
    if ref.shape != (H, W):
        raise ValueError(f"ref_mask shape {ref.shape} != mask_shape {(H, W)}")
    return _hard_iou_from_pixels(_in_bounds_pixels(verts2d, H, W),
                                 ref.reshape(-1), int(ref.sum()))


def _hard_iou_from_pixels(flat, ref_flat, ref_n) -> float:
    """|pred & ref| / |pred | ref| from pred's unique flat pixel indices."""
    if flat.size == 0:
        return float(0 / ref_n) if ref_n > 0 else 0.0
    inter = int(np.count_nonzero(ref_flat[flat]))
    union = ref_n + int(flat.size - inter)
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


def _soft_splat_bbox(verts2d, H, W, sigma, splat_k):
    """(soft, y0, x0) -- the soft-occupancy grid over ONLY the bounding box the
    splat can reach, plus its offset in the full frame. ``None`` when nothing
    lands in frame (soft occupancy identically zero).

    Bit-identical to building the full (H, W) grid and slicing it:
      * a vert whose whole (2k+1)^2 window is out of frame contributed
        ``np.where(valid, w, 0.0) == 0.0`` at every offset, and adding 0.0 to a
        float64 accumulator is exact -- so dropping it changes nothing;
      * the surviving (offset, vert) pairs are accumulated in the SAME order
        (offset-major, vert-inner), which is what fixes the float64 rounding of
        overlapping splats;
      * ``1 - exp(-0) == 0`` exactly, so every pixel outside the box is
        genuinely zero rather than negligible.
    """
    v = np.asarray(verts2d, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        bx = np.floor(v[:, 0])
        by = np.floor(v[:, 1])
    keep = (np.isfinite(bx) & np.isfinite(by)
            & (bx + splat_k >= 0) & (bx - splat_k < W)
            & (by + splat_k >= 0) & (by - splat_k < H))
    if not keep.any():
        return None
    vx = v[keep, 0]
    vy = v[keep, 1]
    bxi = bx[keep].astype(np.int64)
    byi = by[keep].astype(np.int64)
    x0 = max(0, int(bxi.min()) - splat_k)
    x1 = min(W, int(bxi.max()) + splat_k + 1)
    y0 = max(0, int(byi.min()) - splat_k)
    y1 = min(H, int(byi.max()) + splat_k + 1)
    # offsets in the original loop order (di outer, dj inner) so the C-order
    # ravel below reproduces the old np.add.at accumulation order exactly.
    off = np.arange(-splat_k, splat_k + 1)
    di, dj = np.meshgrid(off, off, indexing="ij")
    cx = bxi[None, :] + dj.reshape(-1, 1)                 # (K*K, V)
    cy = byi[None, :] + di.reshape(-1, 1)
    valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    cxv = cx[valid]
    cyv = cy[valid]
    dx = cxv + 0.5 - np.broadcast_to(vx, cx.shape)[valid]
    dy = cyv + 0.5 - np.broadcast_to(vy, cy.shape)[valid]
    w = np.exp(-((dx ** 2 + dy ** 2)) / (2 * sigma ** 2))
    grid = np.zeros((y1 - y0, x1 - x0), dtype=np.float64)
    np.add.at(grid, (cyv - y0, cxv - x0), w)
    return 1.0 - np.exp(-grid), y0, x0


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

    Bounding-boxed since 2026-09-01 (the dense version rasterised, exp'd and
    summed 867k px per camera per frame -- ~7 ms x 14049 (frame, camera) pairs
    per bout in live QC). The grid itself is bit-identical (see
    ``_soft_splat_bbox``); the two REDUCTIONS are algebraically rearranged, so
    they can differ in the last ulp:
        inter = sum(soft * mask)          -- zero outside the box
        union = sum(soft + mask - soft*mask) = |mask| + sum(soft * (1 - mask))
    Measured on Session0 bout 28 fly0 (840 real (frame, camera) pairs): max
    absolute deviation 1.1e-16, max relative 5.9e-16, i.e. 1-2 ulp.
    ``|mask|`` is counted on the mask's own dtype rather than a float32 copy;
    for a bool/{0,1} mask below 2**24 pixels both are exactly the pixel count.
    """
    mask = np.asarray(mask)
    H, W = mask.shape
    mask_sum = float(mask.sum())
    if mask_sum == 0:
        return 0.0
    return _soft_iou_from_bbox(_soft_splat_bbox(verts2d, H, W, sigma, splat_k),
                               mask, mask_sum)


def _soft_iou_from_bbox(splat, mask, mask_sum) -> float:
    if splat is None:
        return 0.0
    soft, y0, x0 = splat
    h, w = soft.shape
    m = mask[y0:y0 + h, x0:x0 + w].astype(np.float32)
    inter = float((soft * m).sum())
    union = mask_sum + float((soft * (1.0 - m)).sum())
    return inter / union if union > 0 else 0.0


def hard_soft_iou_of_verts(verts2d, mask, *, sigma: float = 1.3,
                           splat_k: int = 2):
    """``(hard, soft)`` IoU for one (camera, frame) in one pass.

    Identical to calling ``iou_of_projected_verts(verts2d, mask.shape, mask)``
    and ``soft_iou_of_verts(verts2d, mask)``, but counts ``|mask|`` once: that
    full-frame reduction is 0.25 ms per call on this rig's (448, 1936) masks and
    the QC path needs it for both metrics.
    """
    mask = np.asarray(mask)
    H, W = mask.shape
    ref = mask.astype(bool, copy=False)
    ref_n = int(ref.sum())
    hard = _hard_iou_from_pixels(_in_bounds_pixels(verts2d, H, W),
                                 ref.reshape(-1), ref_n)
    # |mask| for the soft-IoU denominator. For a bool mask (what the live QC
    # path passes -- bout_masks unpacks SAM3 to bool) it is provably the same
    # count as `ref_n`, so the shared reduction is exact; anything else gets its
    # own sum rather than silently assuming {0,1}.
    mask_sum = float(ref_n) if mask.dtype == bool else float(mask.sum())
    soft = 0.0 if mask_sum == 0 else _soft_iou_from_bbox(
        _soft_splat_bbox(verts2d, H, W, sigma, splat_k), mask, mask_sum)
    return hard, soft
