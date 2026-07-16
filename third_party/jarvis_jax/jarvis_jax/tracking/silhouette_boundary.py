# jarvis_jax/cse/silhouette_boundary.py
"""Offline SAM-mask boundary extraction + uniform boundary-point sampling.

Pure NumPy/scipy (NO JAX, NO qpos): produces the Phase-6 silhouette *target*.
The boundary is the set of foreground pixels adjacent to background (a binary
erosion XOR); sampled points are evenly spaced along the boundary's arc order.
Coordinates are (x, y) = (column, row) to match affine_camera.project_affine's
uv convention. Fixed per (frame, camera); not a function of qpos.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage


def mask_boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Return (K,2) float (x,y) boundary-pixel coordinates of a 2-D mask.

    A boundary pixel is a foreground pixel with >=1 4-connected background
    neighbor (foreground minus its 4-connected erosion).
    """
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return np.zeros((0, 2), dtype=np.float64)
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    eroded = ndimage.binary_erosion(m, structure=struct, border_value=0)
    boundary = m & ~eroded
    ys, xs = np.where(boundary)  # row, col
    return np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)


def _order_boundary(bp: np.ndarray) -> np.ndarray:
    """Order boundary pixels by a nearest-neighbor walk (approx arc order).

    Greedy chain from an arbitrary start; good enough for even *arc-order*
    subsampling of a single closed contour (the fly silhouette).
    """
    n = bp.shape[0]
    if n <= 2:
        return bp
    remaining = list(range(n))
    order = [remaining.pop(0)]
    while remaining:
        last = bp[order[-1]]
        d = np.linalg.norm(bp[remaining] - last, axis=1)
        k = int(np.argmin(d))
        order.append(remaining.pop(k))
    return bp[np.asarray(order)]


def sample_boundary_points(mask: np.ndarray, n_points: int, *, seed: int = 0) -> np.ndarray:
    """Return exactly (n_points, 2) float (x,y) points along the mask boundary.

    Empty mask -> all-NaN (n_points,2). Fewer boundary pixels than n_points ->
    deterministic sampling WITH replacement so the shape is always fixed.
    """
    bp = mask_boundary_pixels(mask)
    if bp.shape[0] == 0:
        return np.full((n_points, 2), np.nan, dtype=np.float64)
    ordered = _order_boundary(bp)
    n = ordered.shape[0]
    if n >= n_points:
        # even arc-order subsample
        idx = np.linspace(0, n - 1, n_points).round().astype(int)
        idx = np.clip(idx, 0, n - 1)
        return ordered[idx]
    # too few pixels: deterministic sample with replacement
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=n_points)
    return ordered[idx]
