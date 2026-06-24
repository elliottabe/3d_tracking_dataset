"""GT-free 3D ROI center estimation for inference (SAM3 mask centroids)."""
import numpy as np


def quantize_center3d(pts, grid_spacing: int = 1):
    """Quantize a set of 3D points to V3's center3D lattice.

    Per axis: midrange of the non-zero coords, snapped to the grid:
        mid = (max + min) / grid_spacing / 2 ; center[d] = int(mid) * grid_spacing
    A single point reduces to int(p/grid_spacing)*grid_spacing (lattice snap).

    Args:
        pts: (K, 3) candidate 3D points (e.g. visible keypoints, or one centroid).
        grid_spacing: world units per grid step (V3 uses 1).
    Returns:
        (3,) float32 quantized center.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)
    out = []
    for d in range(3):
        coords = pts[:, d]
        nz = coords[coords != 0]
        if nz.size == 0:
            nz = coords
        mid = (float(np.max(nz)) + float(np.min(nz))) / float(grid_spacing) / 2.0
        out.append(int(mid) * grid_spacing)
    return np.array(out, dtype=np.float32)
