"""Per-frame model->mm similarity bridge (the `s, R, t` Stage E FKs through).

Extracted verbatim from ``tracking.polish.polish_bout`` when the silhouette
polish was deleted. The polish itself was dead -- with
``silhouette_weight == containment_weight == 0`` (the shipped default) it
early-returned ``q = q_init``, and ``qpos_refined.npz`` was measured equal to
the STAC ``qpos`` at maxabsdiff 0.0 on both flies of Session0 bout 28. But it
was NOT inert: the same function computed the bridges, and a bridge is not
optional -- ``build_fly_outputs`` maps model units to mm through it, and a
``None`` bridge is the "this frame is NaN" contract. So the bridge came out
first and the silhouette went away around it.

`bridge_mode` (``cfg.ik.bridge_mode``):
  'keypoint' -- full per-frame Umeyama(FK sites, kp3d) similarity. The default,
      and the tightest marker fit when the keypoints are multiview-consistent.
      Needs >= 3 valid keypoints on the frame. Uses NO masks.
  'mask'     -- s = 1/kp_scale fixed, translation from the triangulated SAM
      mask centroid, rotation still from keypoint Umeyama. For when the 2D
      keypoints are unreliable and global placement must be anchored to the
      masks. Needs `masks_dict` and >= 2 valid-mask cameras on the frame.

Only 'mask' touches masks at all, so under the default the IK no longer has a
mask dependency -- `masks_dict` is optional and unread.
"""
from __future__ import annotations

import numpy as np


def _mask_centroids_3d(masks, valid, cam_mats):
    """(T,C,H,W) masks + (T,C) valid + (C,4,3) DLT cams -> (T,3) triangulated 3D
    mask centroids and (T,) bool triangulable (>=2 valid-mask cams)."""
    from scipy import ndimage
    from jarvis_jax.tracking.triangulate import triangulate_keypoints
    T, C = masks.shape[:2]
    cen2d = np.full((T, C, 1, 2), np.nan, np.float32)
    cconf = np.zeros((T, C, 1), np.float32)
    for t in range(T):
        for c in range(C):
            if not valid[t, c] or not masks[t, c].any():
                continue
            cy, cx = ndimage.center_of_mass(masks[t, c])   # (row, col) = (y, x)
            cen2d[t, c, 0] = (cx, cy)                       # (x, y) pixel
            cconf[t, c, 0] = 1.0
    cen3d, _ = triangulate_keypoints(cen2d, cconf, cam_mats, conf_thresh=0.5)
    cen3d = cen3d[:, 0, :]                                  # (T,3), NaN where <2 views
    return cen3d.astype(np.float32), np.isfinite(cen3d).all(-1)


def compute_bridges(stac_h5, model_xml, kp3d_mm, kp3d_conf, calib_dir, *,
                    kp_scale=1.0, bridge_mode="keypoint", masks_dict=None):
    """Return ``(qpos, bridge_s, bridge_R, bridge_t, bridge_ok)``.

    ``qpos`` is the STAC pose as ``build_solver_inputs`` resolves it (``q_init``),
    returned rather than re-read from the h5 so callers keep byte-identical
    behaviour with the polish this replaced. ``bridge_ok[t]`` false means frame
    t has no usable bridge; Stage E turns those into NaN rows.
    """
    import jax.numpy as jnp
    import stac_mjx.utils as stac_utils
    from jarvis_jax.tracking.ik_solve import build_solver_inputs, _umeyama

    if bridge_mode not in ("keypoint", "mask"):
        raise ValueError(f"bridge_mode must be 'keypoint' or 'mask', got {bridge_mode!r}")
    if bridge_mode == "mask" and masks_dict is None:
        raise ValueError("bridge_mode='mask' needs masks_dict; 'keypoint' does not")

    inp = build_solver_inputs(stac_h5, model_xml)
    T = inp["q_init"].shape[0]
    q_init = np.asarray(inp["q_init"])[:T]

    s_const = 1.0 / float(kp_scale)
    if bridge_mode == "mask":
        from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
        cam_mats = np.asarray(ReprojectionTool(calib_dir).camera_matrices, np.float32)
        cen3d, cen_ok = _mask_centroids_3d(masks_dict["masks"], masks_dict["valid"],
                                           cam_mats)
    else:
        cen3d = cen_ok = None

    bridge_s = np.full((T,), s_const, np.float32)
    bridge_R = np.broadcast_to(np.eye(3, dtype=np.float32), (T, 3, 3)).copy()
    bridge_t = np.zeros((T, 3), np.float32)
    bridge_ok = np.zeros(T, bool)

    for t in range(T):
        d0 = inp["mjx_data"].replace(qpos=jnp.asarray(q_init[t]))
        d0 = stac_utils.kinematics(inp["mjx_model"], d0)
        d0 = stac_utils.com_pos(inp["mjx_model"], d0)
        sites0 = np.asarray(stac_utils.get_site_xpos(d0, inp["site_idxs"]))
        # Require finite FK sites too: STAC can emit NaN qpos on sparse frames
        # (-> NaN sites0), which would make _umeyama's SVD non-convergent. Those
        # frames are left un-bridged (bridge_ok stays False -> NaN output).
        kok = (np.isfinite(kp3d_mm[t]).all(-1) & (kp3d_conf[t] > 0)
               & np.isfinite(sites0).all(-1))
        if bridge_mode == "keypoint":
            if kok.sum() < 3:
                continue
            try:
                s, R, tr = _umeyama(sites0[kok], np.asarray(kp3d_mm[t])[kok])
            except np.linalg.LinAlgError:
                continue
            if not (np.isfinite(s) and np.isfinite(R).all() and np.isfinite(tr).all()):
                continue
            bridge_s[t] = s; bridge_R[t] = R; bridge_t[t] = tr; bridge_ok[t] = True
        else:  # 'mask'
            if not cen_ok[t]:
                continue
            if kok.sum() >= 3:
                try:
                    _, R, _ = _umeyama(sites0[kok], np.asarray(kp3d_mm[t])[kok])
                except np.linalg.LinAlgError:
                    R = np.eye(3, dtype=np.float32)
            else:
                R = np.eye(3, dtype=np.float32)
            model_cen = sites0.mean(axis=0)     # place FK body centroid at 3D mask centroid
            bridge_t[t] = cen3d[t] - s_const * (R @ model_cen)
            bridge_R[t] = R; bridge_ok[t] = True

    return q_init, bridge_s, bridge_R, bridge_t, bridge_ok
