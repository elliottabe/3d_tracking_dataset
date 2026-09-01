"""Cropped signed-distance fields from SAM masks, in ORIGINAL image pixels.

For each (frame, camera) mask: crop to the fly bbox + margin, resize to a fixed
grid, and compute a SIGNED distance field -- negative inside, positive outside --
so a differentiable containment residual can bilinearly sample it at projected
mesh vertices. Pure NumPy/SciPy/cv2; no JAX, no qpos.

Recovered from `silhouette_sdf.py` (deleted in 0bc36fe with the silhouette
polish) and re-verified rather than trusted: the sign convention checks out
(-25.3 inside / +27.8 outside on a test box), but its ORIGINAL-pixel scaling was
WRONG and is fixed here.

THE BUG, and why it mattered. The old code resized the crop ANISOTROPICALLY to a
square grid, ran an isotropic distance transform on that grid, then rescaled by a
single AVERAGED factor:

    px_scale = ((x1 - x0) / W + (y1 - y0) / H) / 2.0     # one factor, two axes
    sdf = sdf_resized * px_scale

No single factor can be correct in both axes. Measured on a 40x60 box (crop
108x72 -> grid 64x64, a 1.50x anisotropy) it read -25.3 px at the box centre
against a true inradius of 20.0 px: a 27% direction-dependent error, which for a
containment penalty mis-weights the pull depending on which way a vertex lies
outside the mask.

THE FIX. `scipy.ndimage.distance_transform_edt` takes per-axis pixel spacing via
`sampling=`, so the transform itself can work in original pixels and no rescale
is needed. With `sampling=(dy, dx)` the same test box reads exactly -20.0.
"""
from __future__ import annotations

import numpy as np

BIG = 1.0e6          # SDF fill for a (frame, camera) with no usable mask


def mask_bbox(mask, margin=0.4):
    """(x0, y0, x1, y1) float bbox of the mask, grown by `margin` of its size.
    None if the mask is empty."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return None
    ys, xs = np.where(m)
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (x0 - mx, y0 - my, x1 + mx, y1 + my)


def mask_to_sdf_crop(mask, bbox, out_hw):
    """Crop -> resize -> SIGNED distance field in ORIGINAL pixels.

    Returns (sdf (H,W) float32, grid_scale (2,) float32, grid_offset (2,)
    float32), or None if the crop is empty. The sampling convention is
    ``grid_xy = (orig_xy - grid_offset) * grid_scale``, unchanged from the
    original so existing containment code keeps working.

    Distances are in ORIGINAL pixels EXACTLY, not approximately: the per-axis
    spacing goes into the distance transform instead of being averaged into one
    post-hoc factor.
    """
    import cv2
    from scipy import ndimage

    m = np.asarray(mask).astype(np.uint8)
    H, W = int(out_hw[0]), int(out_hw[1])
    x0, y0, x1, y1 = bbox
    x0i, y0i = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1i, y1i = min(m.shape[1], int(np.ceil(x1))), min(m.shape[0], int(np.ceil(y1)))
    if x1i <= x0i or y1i <= y0i:
        return None
    crop = m[y0i:y1i, x0i:x1i]
    if crop.sum() == 0:
        return None
    rc = cv2.resize(crop, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    # per-axis ORIGINAL px per resized px -- this is the fix
    dy = (y1i - y0i) / float(H)
    dx = (x1i - x0i) / float(W)
    din = ndimage.distance_transform_edt(rc, sampling=(dy, dx))    # >0 inside
    dout = ndimage.distance_transform_edt(~rc, sampling=(dy, dx))  # >0 outside
    sdf = (dout - din).astype(np.float32)                          # ORIGINAL px

    grid_scale = np.array([W / (x1i - x0i), H / (y1i - y0i)], np.float32)
    grid_offset = np.array([x0i, y0i], np.float32)
    return sdf, grid_scale, grid_offset


def sdf_stack_from_masks(masks, valid, *, out_hw=(128, 128), bbox_margin=0.4):
    """(T,C,H,W) masks + (T,C) valid -> per-(frame,camera) SDF stack.

    Replaces the recovered `build_sdf_stack`, which indexed a COCO annotation
    file. The pipeline already carries masks as an array (sam3_masks.npz), so
    taking them directly removes a dataset dependency the pipeline does not have.

    Returns (sdf (T,C,H,W) float32, grid_scale (T,C,2), grid_offset (T,C,2),
    present (T,C) bool). A (frame, camera) with no usable mask is left
    `present=False` and its SDF filled with BIG, so a containment residual gated
    on `present` contributes nothing there rather than pulling toward garbage.
    """
    masks = np.asarray(masks)
    valid = np.asarray(valid, bool)
    T, C = masks.shape[:2]
    H, W = int(out_hw[0]), int(out_hw[1])
    sdf = np.full((T, C, H, W), BIG, np.float32)
    gs = np.ones((T, C, 2), np.float32)
    go = np.zeros((T, C, 2), np.float32)
    present = np.zeros((T, C), bool)
    for t in range(T):
        for c in range(C):
            if not valid[t, c]:
                continue
            bb = mask_bbox(masks[t, c], bbox_margin)
            if bb is None:
                continue
            out = mask_to_sdf_crop(masks[t, c], bb, (H, W))
            if out is None:
                continue
            sdf[t, c], gs[t, c], go[t, c] = out
            present[t, c] = True
    return sdf, gs, go, present
