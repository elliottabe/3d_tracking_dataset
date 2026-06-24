"""GT-free 3D ROI center estimation for inference (SAM3 mask centroids)."""
import numpy as np
import jax.numpy as jnp


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


def mask_centroids(crops4):
    """Per-camera SAM3 mask centroid in crop pixel space (x, y).

    Args:
        crops4: (B, nc, H, W, 4) uint8 array, mask in channel 3 (last).

    Returns:
        centroids: (B, nc, 2) float32 [x, y] in crop pixel space.
        valid: (B, nc) bool indicating whether mask is non-empty.
    """
    m = crops4[..., 3].astype(jnp.float32)            # (B,nc,H,W)
    H, W = m.shape[-2], m.shape[-1]
    xs = jnp.arange(W, dtype=jnp.float32)
    ys = jnp.arange(H, dtype=jnp.float32)
    msum = m.sum(axis=(-1, -2))                        # (B,nc)
    denom = jnp.clip(msum, 1.0, None)
    cx = (m.sum(axis=-2) * xs).sum(-1) / denom         # sum over y -> (B,nc,W) * xs
    cy = (m.sum(axis=-1) * ys).sum(-1) / denom         # sum over x -> (B,nc,H) * ys
    centroids = jnp.stack([cx, cy], axis=-1)           # (B,nc,2) [x,y]
    valid = msum > 0
    return centroids, valid


def centroids_to_fullpx(centroids, centerHM, crop: int = 448):
    """Map crop-space centroids to full-image px: centroid + (centerHM - crop/2).

    Args:
        centroids: (B, nc, 2) float32 in crop-space [x, y].
        centerHM: (B, nc, 2) float32 crop center in full-image px.
        crop: crop size in pixels (default 448).

    Returns:
        (B, nc, 2) float32 full-image pixel coordinates.
    """
    return centroids + centerHM - crop / 2.0


def triangulate_dlt_batched(points2d, cameraMatrices, valid):
    """Batched DLT triangulation. Invalid cameras contribute zero rows.

    Args:
        points2d:       (B, nc, 2) full-image px [u, v].
        cameraMatrices: (B, nc, 4, 3) = P.T (P is the 3x4 DLT matrix).
        valid:          (B, nc) bool.
    Returns:
        (B, 3) triangulated points (perspective-divided).
    """
    points2d = jnp.asarray(points2d, jnp.float32)
    P = jnp.swapaxes(jnp.asarray(cameraMatrices, jnp.float32), -1, -2)  # (B,nc,3,4)
    u = points2d[..., 0:1]; v = points2d[..., 1:2]                      # (B,nc,1)
    row_u = u * P[:, :, 2, :] - P[:, :, 0, :]                           # (B,nc,4)
    row_v = v * P[:, :, 2, :] - P[:, :, 1, :]                           # (B,nc,4)
    w = valid[..., None].astype(jnp.float32)                            # (B,nc,1)
    A = jnp.concatenate([row_u * w, row_v * w], axis=1)                 # (B,2nc,4)
    _, _, Vh = jnp.linalg.svd(A, full_matrices=False)                   # Vh (B,4,4)
    X = Vh[:, -1, :]                                                    # (B,4) null vec
    return X[:, :3] / X[:, 3:4]


def estimate_center3d_from_masks(crops4, centerHM, cameraMatrices,
                                 *, grid_spacing: int = 1, crop: int = 448):
    """GT-free center3D from triangulated SAM3 mask centroids, lattice-snapped.

    Composes mask_centroids -> centroids_to_fullpx -> triangulate_dlt_batched
    -> per-item quantize_center3d.

    Args:
        crops4:         (B, nc, H, W, 4) uint8 array with mask in channel 3.
        centerHM:       (B, nc, 2) float32 crop center in full-image pixels.
        cameraMatrices: (B, nc, 4, 3) float32 DLT projection matrices.
        grid_spacing:   world units per grid step (default 1, matching V3).
        crop:           crop size in pixels (default 448).

    Returns:
        center3D: (B, 3) float32 lattice-snapped 3D centers.
        n_valid:  (B,) int32 number of valid cameras per batch item.
                  Items with n_valid < 2 have center3D = zeros.
    """
    centroids, valid = mask_centroids(crops4)                 # (B,nc,2),(B,nc)
    full = centroids_to_fullpx(centroids, centerHM, crop)     # (B,nc,2)
    pts3d = triangulate_dlt_batched(full, cameraMatrices, valid)  # (B,3)
    pts3d = np.asarray(pts3d)
    n_valid = np.asarray(valid).sum(axis=1).astype(np.int32)  # (B,)
    out = np.zeros((pts3d.shape[0], 3), np.float32)
    for i in range(pts3d.shape[0]):
        if n_valid[i] >= 2 and np.all(np.isfinite(pts3d[i])):
            out[i] = quantize_center3d(pts3d[i][None, :], grid_spacing)
    return out, n_valid


def project_center_to_cameras(center3D, cameraMatrices):
    """Project one 3D point to each camera's full-image px (perspective divide).

    Inverse convention of triangulate_dlt_batched: cameraMatrices (nc,4,3) = P.T,
    so proj = [X,Y,Z,1] @ M -> (3,), px = proj[:2]/proj[2].
    """
    M = np.asarray(cameraMatrices, np.float64)            # (nc,4,3)
    ph = np.concatenate([np.asarray(center3D, np.float64), [1.0]])  # (4,)
    proj = ph @ M                                          # (nc,3)
    return (proj[:, :2] / proj[:, 2:3]).astype(np.float32)  # (nc,2)
