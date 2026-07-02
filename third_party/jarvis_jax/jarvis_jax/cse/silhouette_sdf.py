"""Offline cropped signed-distance-field precompute for the silhouette
containment residual (pure NumPy/SciPy/cv2, no JAX, no qpos).

For each (frame, camera) SAM mask we crop to the fly bbox + margin, resize to a
fixed grid, and compute a SIGNED distance field (negative inside, positive
outside) rescaled to ORIGINAL-image pixels. The containment residual bilinearly
samples this field at projected mesh verts; grid coords are
grid_xy = (orig_xy - grid_offset) * grid_scale. Cropping keeps the field small
(so the whole (T, n_cam, H, W) stack fits on device as a closed-over constant).
"""
from __future__ import annotations
import json
import os
import numpy as np
import cv2
from scipy import ndimage

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.cse.silhouette_ik_solve import (
    _cam2img_for_frame, _ann_for_image, _load_sam_mask,
)

_BIG = 1e4  # sdf fill for absent cameras (present=False gates it anyway)


def _load_coco_index(root, split):
    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann_multi = {}
    for an in coco["annotations"]:
        id2ann_multi.setdefault(an["image_id"], []).append(an)
    return id2file, id2ann_multi


def _mask_bbox(mask, margin):
    """(x0,y0,x1,y1) fly bbox expanded by `margin` fraction of side; None if empty."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return None
    ys, xs = np.where(m)
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (x0 - mx, y0 - my, x1 + mx, y1 + my)


def _mask_to_sdf_crop(mask, bbox, out_hw):
    """Crop mask to bbox, resize to out_hw, signed distance in ORIGINAL px.

    Returns (sdf (H,W) f32, grid_scale (2,) f32, grid_offset (2,) f32) or None
    if the crop is empty. grid_xy = (orig_xy - grid_offset) * grid_scale.
    """
    m = np.asarray(mask).astype(np.uint8)
    H, W = out_hw
    x0, y0, x1, y1 = bbox
    x0i, y0i = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1i, y1i = min(m.shape[1], int(np.ceil(x1))), min(m.shape[0], int(np.ceil(y1)))
    if x1i <= x0i or y1i <= y0i:
        return None
    crop = m[y0i:y1i, x0i:x1i]
    if crop.sum() == 0:
        return None
    rc = cv2.resize(crop, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    din = ndimage.distance_transform_edt(rc)     # >0 inside (dist to background)
    dout = ndimage.distance_transform_edt(~rc)   # >0 outside (dist to foreground)
    sdf_resized = dout - din                     # + outside, - inside (resized px)
    px_scale = ((x1i - x0i) / W + (y1i - y0i) / H) / 2.0   # resized px -> original px
    sdf = (sdf_resized * px_scale).astype(np.float32)
    grid_scale = np.array([W / (x1i - x0i), H / (y1i - y0i)], np.float32)
    grid_offset = np.array([x0i, y0i], np.float32)
    return sdf, grid_scale, grid_offset


def build_sdf_stack(root, split, fs_imgids, calib_dir, *, out_hw=(128, 128),
                    bbox_margin=0.4, ann_id_by_image=None):
    """Per-(frame,camera) cropped signed-distance stack + crop transforms."""
    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    n_cam = rt.num_cameras
    cam_Ms = np.stack([rt._camera_list[c].cameraMatrix[:2, :3] for c in range(n_cam)]).astype(np.float32)
    cam_ts = np.stack([rt._camera_list[c].cameraMatrix[:2, 3] for c in range(n_cam)]).astype(np.float32)

    id2file, id2ann_multi = _load_coco_index(root, split)
    fs_list = list(fs_imgids)
    T = len(fs_list)
    H, W = out_hw
    sdf = np.full((T, n_cam, H, W), _BIG, np.float32)
    grid_scale = np.ones((T, n_cam, 2), np.float32)
    grid_offset = np.zeros((T, n_cam, 2), np.float32)
    present = np.zeros((T, n_cam), bool)

    for t in range(T):
        cam2img = _cam2img_for_frame(fs_list[t], id2file, cam_names)
        for c in range(n_cam):
            iid = cam2img.get(c)
            if iid is None:
                continue
            ann = _ann_for_image(id2ann_multi, iid, ann_id_by_image)
            if ann is None:
                continue
            mask = _load_sam_mask(root, split, id2file.get(int(iid), ""), ann["id"])
            if mask is None:
                continue
            mask = np.asarray(mask)
            bbox = _mask_bbox(mask, bbox_margin)
            if bbox is None:
                continue
            out = _mask_to_sdf_crop(mask, bbox, out_hw)
            if out is None:
                continue
            sdf[t, c], grid_scale[t, c], grid_offset[t, c] = out
            present[t, c] = True

    return dict(sdf=sdf, grid_scale=grid_scale, grid_offset=grid_offset,
                present=present, cam_Ms=cam_Ms, cam_ts=cam_ts, cam_names=cam_names)
