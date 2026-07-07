"""Tile arrangement + cropping for multi-camera viz montages."""
import numpy as np
from viz.core import overlays

def crop_to_points(img, uv, pad=55):
    p = np.asarray(uv, float).reshape(-1, 2)
    p = p[np.isfinite(p).all(1)]
    if len(p) == 0:
        return img, (0, 0)
    x0, y0 = np.maximum(p.min(0) - pad, 0).astype(int)
    x1, y1 = (p.max(0) + pad).astype(int)
    x1 = min(img.shape[1], x1); y1 = min(img.shape[0], y1)
    return img[y0:y1, x0:x1].copy(), (int(x0), int(y0))

def montage(tiles, cols=2):
    if not tiles:
        return np.zeros((1, 1, 3), np.uint8)
    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * h:r * h + t.shape[0], c * w:c * w + t.shape[1]] = t
    return grid

def banner(width, items, height=None):
    height = height or (len(items) * 18 + 10)
    b = np.zeros((height, width, 3), np.uint8)
    return overlays.legend(b, items)
