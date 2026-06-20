"""Pure NumPy transforms for the V3 keypoint loader: cropping, RGB
normalization, keypoint coordinate transforms, and Gaussian heatmap rendering.
"""
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_origin(bbox, img_w, img_h, crop=448):
    """Top-left (x0, y0) of a `crop`x`crop` window centered on the bbox center,
    clamped so the window stays inside the image. bbox is [x, y, w, h]."""
    cx = bbox[0] + bbox[2] / 2.0
    cy = bbox[1] + bbox[3] / 2.0
    x0 = int(round(cx - crop / 2.0))
    y0 = int(round(cy - crop / 2.0))
    x0 = max(0, min(x0, img_w - crop))
    y0 = max(0, min(y0, img_h - crop))
    return x0, y0


def normalize_rgb(rgb01):
    """(H,W,3) RGB in [0,1] -> ImageNet-normalized float32."""
    return ((rgb01 - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


def transform_keypoints(kps, x0, y0, crop=448, heatmap_size=224):
    """Map full-image keypoints into heatmap coords.

    kps: (50,3) = [x, y, v]. Returns (hm_xy (50,2) float32, vis (50,) bool).
    A keypoint is visible if v>0 and its heatmap coord is inside the grid.
    """
    scale = heatmap_size / float(crop)
    xy = kps[:, :2].astype(np.float32)
    v = kps[:, 2]
    hm_xy = np.empty((kps.shape[0], 2), dtype=np.float32)
    hm_xy[:, 0] = (xy[:, 0] - x0) * scale
    hm_xy[:, 1] = (xy[:, 1] - y0) * scale
    inb = (
        (hm_xy[:, 0] >= 0) & (hm_xy[:, 0] <= heatmap_size - 1)
        & (hm_xy[:, 1] >= 0) & (hm_xy[:, 1] <= heatmap_size - 1)
    )
    vis = (v > 0) & inb
    return hm_xy, vis


def gaussian_heatmaps(hm_xy, vis, heatmap_size=224, sigma=2.0):
    """Render (heatmap_size, heatmap_size, K) Gaussian heatmaps (peak 1.0).

    hm_xy: (K, 2) array with columns [x (col), y (row)]."""
    k = hm_xy.shape[0]
    hm = np.zeros((heatmap_size, heatmap_size, k), dtype=np.float32)
    grid = np.arange(heatmap_size, dtype=np.float32)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")  # (H,W)
    two_s2 = 2.0 * sigma * sigma
    for j in range(k):
        if not vis[j]:
            continue
        cx, cy = hm_xy[j]  # hm_xy columns are [x (col), y (row)]
        g = np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / two_s2)
        hm[:, :, j] = g
    return hm
