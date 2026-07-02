"""Assemble per-(frame,camera) silhouette targets for the Phase-6 IK.

For each bout frame + camera, loads the SAM mask for the selected fly, samples
n_points uniform boundary points, and packs them with the camera's affine DLT
matrix into the flat SilVar layout the sibling solver threads. Cameras with no
mask this frame get NaN boundary blocks (the Chamfer residual is NaN-safe).
"""
from __future__ import annotations
import json
import os
import numpy as np

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.cse.silhouette_boundary import sample_boundary_points
from jarvis_jax.cse.silhouette_ik_solve import (
    _cam2img_for_frame, _ann_for_image, _load_sam_mask,
)
from jarvis_jax.cse.silhouette_joint_ik import SIL_PER_CAM, pack_sil_value


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
    n_points: int = 128, ann_id_by_image=None, seed: int = 0,
):
    """Build (sil_data, meta) for the silhouette IK.

    Args:
        root, split: dataset root + split (coco annotations + sam3_masks).
        fs_imgids: (T, n_cam) array (or list of per-frame {cam_idx:image_id}
            dicts) of coco image_ids per frame/camera.
        calib_dir: (refined) calibration dir for ReprojectionTool.
        n_points: boundary points sampled per camera per frame.
        ann_id_by_image: optional image_id->chosen ann id (multi-fly identity).
        seed: boundary-sampling seed (deterministic).

    Returns:
        sil_data: (T, num_cameras*SIL_PER_CAM(n_points)) float32.
        meta: dict(n_cam, n_pts, cam_names).
    """
    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    n_cam = rt.num_cameras
    cam_mats = [rt._camera_list[c].cameraMatrix for c in range(n_cam)]

    id2file, id2ann_multi, _ = _load_coco_index(root, split)

    fs_list = list(fs_imgids)
    T = len(fs_list)
    rows = []
    for t in range(T):
        cam2img = _cam2img_for_frame(fs_list[t], id2file, cam_names)
        boundary = np.full((n_cam, n_points, 2), np.nan, dtype=np.float64)
        for c in range(n_cam):
            iid = cam2img.get(c)
            if iid is None:
                continue
            ann = _ann_for_image(id2ann_multi, iid, ann_id_by_image)
            if ann is None:
                continue
            fn = id2file.get(int(iid), "")
            mask = _load_sam_mask(root, split, fn, ann["id"])
            if mask is None:
                continue
            boundary[c] = sample_boundary_points(np.asarray(mask), n_points, seed=seed)
        rows.append(pack_sil_value(cam_mats, boundary))

    sil_data = np.stack(rows, axis=0).astype(np.float32)
    meta = dict(n_cam=n_cam, n_pts=n_points, cam_names=cam_names)
    return sil_data, meta
