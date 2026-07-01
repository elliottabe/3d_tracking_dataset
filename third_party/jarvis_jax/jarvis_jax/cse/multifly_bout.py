"""Per-fly bout builder: de-collapse a 2-fly recording into one bout h5 per fly.

The stock ``cse_labels.build_bout`` indexes COCO annotations via
``id2ann.setdefault(image_id, a)`` (first-ann-wins per image), which silently
mixes the two flies together on a courtship (multi-animal) recording. Given
the Phase-3 cross-camera identity map (``identity_link.link_recording``),
this builds one bout h5 PER FLY by selecting that fly's chosen annotation per
camera (via the identity map), triangulating its 50 keypoints in model order,
and scaling to model cm with the SAME convention ``cse_labels.build_bout``
uses -- so the downstream STAC + IK pipeline (``run_stac_bout.run``,
``silhouette_ik_solve.build_solver_inputs``), built for single-fly bouts,
consumes each per-fly bout h5 unchanged.

Keypoint convention (matches ``cse_labels.build_bout`` exactly)
-----------------------------------------------------------------
The ``keypoints`` dataset stores ``kp_mm * s`` (triangulated calibration-mm
keypoints scaled by the recording's isotropic Umeyama scale into model-cm
units), NOT raw mm. ``scale`` is written as a separate attr. This mirrors
``cse_labels.build_bout``'s docstring (``kp_stac = kp_mm * s``) and variable
name (``kp_scaled``) precisely -- storing raw mm here would silently
double-scale (or under-scale) any downstream STAC solve that assumes the
``build_bout`` convention.
"""
from __future__ import annotations

import json
import os

import h5py
import numpy as np
import mujoco

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.cse.cse_labels import (
    model_kp_order, model_rest_keypoints, umeyama_scale, _reorder_index,
)


def build_fly_bout(coco_path, calib_dir, recording, identity_map, fly_id,
                   anatomy_yaml, model_xml, out_h5, *, split="val"):
    """Build one fly's bout h5 from the cross-camera identity map.

    Parameters
    ----------
    coco_path : str
        Path to the COCO json (e.g. ``instances_val.json``).
    calib_dir : str
        Calibration directory for THIS recording (already includes the
        recording name, e.g. ``.../calib_params/<recording>``) -- matches
        ``identity_link.link_recording``'s ``calib_dir`` argument, NOT
        ``cse_labels.build_bout``'s ``calib_root`` (which joins ``rec``
        itself).
    recording : str
        Recording ("datasetName") this identity map was built for.
    identity_map : dict[str, dict[int, dict[int, int]]]
        Output of ``identity_link.link_recording``:
        ``{fs_key: {fly_id: {cam_idx: ann_id}}}``.
    fly_id : int
        Which fly slot to extract from ``identity_map``.
    anatomy_yaml : str
        Path to the stac-mjx anatomy yaml giving ``KEYPOINT_MODEL_PAIRS``
        (model keypoint order); same as ``cse_labels.model_kp_order`` input.
    model_xml : str
        MJCF model xml (for ``model_rest_keypoints``).
    out_h5 : str
        Output bout h5 path.
    split : str
        Retained for API symmetry with ``identity_link.link_recording``;
        does not itself filter ``coco_path``.

    Returns
    -------
    (out_h5, scale) : tuple[str, float]
    """
    coco = json.load(open(coco_path))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    src_names = coco["keypoint_names"]

    kp_order = model_kp_order(anatomy_yaml)
    reorder = _reorder_index(src_names, kp_order)          # dst(model) -> src(coco)
    K = len(kp_order)

    rt = ReprojectionTool(calib_dir)
    nc = rt.num_cameras

    kp3d_rows, vis_rows, fs_keys, fs_imgids = [], [], [], []
    for fs_key, per_fly in identity_map.items():
        cam2ann = per_fly.get(fly_id, {})
        if len(cam2ann) < 2:
            continue
        kp2d = np.zeros((nc, K, 3))
        row_imgids = [-1] * nc
        for cam, aid in cam2ann.items():
            ann = ann_by_id[aid]
            row_imgids[cam] = int(ann["image_id"])
            raw = np.asarray(ann["keypoints"], float).reshape(-1, 3)  # coco order
            kp2d[cam] = raw[reorder]
        p3d = np.zeros((K, 3)); vmask = np.zeros(K, bool)
        for j in range(K):
            cams = [c for c in cam2ann if kp2d[c, j, 2] > 0]
            if len(cams) >= 2:
                p3d[j] = rt.reconstruct_point(kp2d[:, j, :2], cams_to_use=cams)
                vmask[j] = True
        if vmask.sum() < 2:
            continue
        kp3d_rows.append(p3d); vis_rows.append(vmask)
        fs_keys.append(fs_key); fs_imgids.append(row_imgids)

    if not kp3d_rows:
        raise RuntimeError(f"no framesets with >=2 cameras for fly {fly_id} in {recording}")
    kp3d = np.asarray(kp3d_rows)                            # (T,K,3) mm
    vis = np.asarray(vis_rows)                              # (T,K)

    model = mujoco.MjModel.from_xml_path(model_xml)
    rest = model_rest_keypoints(model, kp_order)
    # one scale for this fly's bout from the per-frameset median keypoint cloud
    # (mirrors cse_labels.build_bout exactly: kp_stac = kp_mm * s).
    med = np.nanmedian(np.where(vis[..., None], kp3d, np.nan), axis=0)  # (K,3)
    s = umeyama_scale(med, rest, np.all(np.isfinite(med), 1))
    kp_scaled = (kp3d * s).astype(np.float32)                # (T,K,3) model cm
    kp_scaled[~vis] = 0.0

    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("keypoints", data=kp_scaled)
        f.create_dataset("kp_names", data=np.array(kp_order, dtype="S20"))
        f.create_dataset("vis", data=vis)
        f.attrs["scale"] = s
        f.attrs["recording"] = recording
        f.create_dataset("fs_keys", data=np.array(fs_keys, dtype="S64"))
        f.create_dataset("fs_imgids", data=np.asarray(fs_imgids, np.int64))
    print(f"[build_fly_bout] {recording} fly{fly_id}: {len(kp_scaled)} framesets, "
          f"scale={s:.4f} -> {out_h5}")
    return out_h5, s
