"""Assemble per-(frame,camera) silhouette targets for the silhouette IK.

For each bout frame + camera, loads the SAM mask for the selected fly, erodes it
by `erode_px` (strips the reflection/shadow halo), samples n_points uniform
boundary points, and returns them as an UNPACKED dict (boundary/conf_p/present +
per-camera affine matrices) that the solver selects per frame via a FrameVar.
Cameras with no mask this frame get NaN boundary blocks + present=False (the
Chamfer residual is NaN-safe).
"""
from __future__ import annotations
import json
import os
import numpy as np
from scipy import ndimage

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.cse.silhouette_boundary import sample_boundary_points
from jarvis_jax.cse.silhouette_ik_solve import (
    _cam2img_for_frame, _ann_for_image, _load_sam_mask,
)


def _load_coco_index(root, split):
    """Return (id2file, id2ann_multi, cam_names_placeholder). cam_names comes
    from the ReprojectionTool at call time; this returns the coco maps."""
    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann_multi = {}
    for an in coco["annotations"]:
        id2ann_multi.setdefault(an["image_id"], []).append(an)
    return id2file, id2ann_multi, None


def build_silhouette_targets(
    root, split, fs_imgids, calib_dir, *,
    n_points: int = 128, erode_px: int = 8, ann_id_by_image=None, seed: int = 0,
):
    """Per-(frame,camera) eroded-mask boundary points + confidence, unpacked.

    Eroding by `erode_px` strips the SAM shadow/reflection halo so the coverage
    Chamfer pulls the mesh to the TRUE fly boundary, not the inflated ring.
    Absent camera -> boundary NaN, conf_p 0, present False.

    Returns dict(boundary (T,C,n_points,2) f32, conf_p (T,C,n_points) f32,
    present (T,C) bool, cam_Ms (C,2,3) f32, cam_ts (C,2) f32, cam_names, n_pts).
    """
    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    n_cam = rt.num_cameras
    cam_Ms = np.stack([rt._camera_list[c].cameraMatrix[:2, :3] for c in range(n_cam)]).astype(np.float32)
    cam_ts = np.stack([rt._camera_list[c].cameraMatrix[:2, 3] for c in range(n_cam)]).astype(np.float32)

    id2file, id2ann_multi, _ = _load_coco_index(root, split)
    fs_list = list(fs_imgids)
    T = len(fs_list)
    boundary = np.full((T, n_cam, n_points, 2), np.nan, np.float32)
    conf_p = np.zeros((T, n_cam, n_points), np.float32)
    present = np.zeros((T, n_cam), bool)

    struct = ndimage.generate_binary_structure(2, 1)
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
            mask = np.asarray(mask).astype(bool)
            if erode_px > 0:
                eroded = ndimage.binary_erosion(mask, structure=struct, iterations=erode_px,
                                                border_value=0)
                if eroded.any():
                    mask = eroded
            bpts = sample_boundary_points(mask, n_points, seed=seed)
            boundary[t, c] = bpts
            conf_p[t, c] = np.where(np.isfinite(bpts).all(axis=-1), 1.0, 0.0)
            present[t, c] = bool(np.isfinite(bpts).any())

    return dict(boundary=boundary, conf_p=conf_p, present=present,
                cam_Ms=cam_Ms, cam_ts=cam_ts, cam_names=cam_names, n_pts=n_points)


def silhouette_fk_indices(mesh_npz, subset: str = "fps_300", exclude_seg_ids=None):
    """FULL-vertex-array indices for the silhouette factor's projected subset.

    INDEX-SPACE HAZARD (Phase-4 carry-forward, mirrors
    silhouette_ik_solve._wing_fk_indices): the mesh npz's ``fps_*`` arrays hold
    indices INTO the full ``vertices``/``vertex_geom`` arrays (values 0..61665),
    so they are ALREADY full-array space and can be passed straight to
    ``silhouette_ik.make_fk_repose(indices=...)``. In contrast,
    ``silhouette_landmarks.wing_side_vertices`` /
    ``active_parts.excluded_fps_indices`` return indices INTO the fps subset
    (0..299) -- those MUST be bridged via ``fps[idx]`` before FK, or they
    silently select the wrong vertices (verified thorax-vertex bug in
    _wing_fk_indices' docstring). This helper only ever returns full-array
    indices, and applies ``exclude_seg_ids`` in full-array space.

    Args:
        mesh_npz: canonical mesh npz path.
        subset: which fps subset to use ("fps_300" default).
        exclude_seg_ids: optional list of segment ids to drop (full-array
            filtering via ``vertex_segment``), for Phase-4 active-parts.

    Returns:
        np.ndarray (M,) int32 full-vertex-array indices.
    """
    z = np.load(mesh_npz, allow_pickle=True)
    fps = np.asarray(z[subset], dtype=np.int64)   # full-array indices already
    if exclude_seg_ids:
        seg = np.asarray(z["vertex_segment"])     # (61666,) full-array
        excl = set(int(s) for s in exclude_seg_ids)
        keep = np.array([int(seg[i]) not in excl for i in fps])
        fps = fps[keep]
    return fps.astype(np.int32)
