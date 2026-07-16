"""Courtship silhouette-containment polish: STAC h5 + unpacked SAM masks +
triangulated mm keypoints -> refined qpos via SilhouetteJaxlsBatchSolver.

Reuses the solver, bridge, and DOF/vertex logic; replaces the COCO mask/keypoint
providers of run_silhouette_polish with courtship sources.
"""
from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from scipy import ndimage

import stac_mjx.utils as stac_utils
from jarvis_jax.tracking.silhouette_ik import load_anatomy, make_fk_repose
from jarvis_jax.tracking.silhouette_ik_solve import build_solver_inputs, _umeyama
from jarvis_jax.tracking.silhouette_dof import appendage_vertex_indices, build_appendage_dof_mask
from jarvis_jax.tracking.silhouette_targets import silhouette_fk_indices
from jarvis_jax.tracking.silhouette_sdf import _mask_bbox, _mask_to_sdf_crop, _BIG
from jarvis_jax.tracking.silhouette_boundary import sample_boundary_points
from jarvis_jax.tracking.silhouette_joint_ik import SilhouetteJaxlsBatchSolver


def courtship_targets_from_masks(masks, present, cam_affine, *, erode_px, n_points,
                                 sdf_hw, bbox_margin):
    """(T,C) unpacked masks -> boundary/conf_p/present + SDF stack + affine cams."""
    cam_Ms, cam_ts = cam_affine
    T, C, H, W = masks.shape
    Hh, Ww = sdf_hw
    boundary = np.full((T, C, n_points, 2), np.nan, np.float32)
    conf_p = np.zeros((T, C, n_points), np.float32)
    pres = np.zeros((T, C), bool)
    sdf = np.full((T, C, Hh, Ww), _BIG, np.float32)
    gscale = np.ones((T, C, 2), np.float32); goff = np.zeros((T, C, 2), np.float32)
    struct = ndimage.generate_binary_structure(2, 1)
    for t in range(T):
        for c in range(C):
            if not present[t, c] or not masks[t, c].any():
                continue
            m = masks[t, c]
            er = ndimage.binary_erosion(m, structure=struct, iterations=erode_px, border_value=0) \
                if erode_px > 0 else m
            if not er.any():
                er = m
            bpts = sample_boundary_points(er, n_points, seed=0)
            boundary[t, c] = bpts
            conf_p[t, c] = np.isfinite(bpts).all(-1).astype(np.float32)
            bbox = _mask_bbox(m, bbox_margin)
            out = _mask_to_sdf_crop(m, bbox, sdf_hw) if bbox is not None else None
            if out is not None:
                sdf[t, c], gscale[t, c], goff[t, c] = out
                pres[t, c] = True
    return dict(boundary=boundary, conf_p=conf_p, present=pres, sdf=sdf,
                grid_scale=gscale, grid_offset=goff,
                cam_Ms=np.asarray(cam_Ms, np.float32), cam_ts=np.asarray(cam_ts, np.float32))


def _mask_centroids_3d(masks, valid, cam_mats):
    """(T,C,H,W) masks + (T,C) valid + (C,4,3) DLT cams -> (T,3) triangulated 3D
    silhouette centroids and (T,) bool triangulable (>=2 valid-mask cams).

    The SAM masks are the reliable observation for courtship (the 2D keypoints
    are OOD/inconsistent), so the global placement of the mesh is anchored to the
    per-frame mask centroid rather than the noisy keypoint cloud."""
    from jarvis_jax.tracking.triangulate import triangulate_keypoints
    T, C, H, W = masks.shape
    cen2d = np.full((T, C, 1, 2), np.nan, np.float32)
    cconf = np.zeros((T, C, 1), np.float32)
    for t in range(T):
        for c in range(C):
            if not valid[t, c] or not masks[t, c].any():
                continue
            cy, cx = ndimage.center_of_mass(masks[t, c])   # (row, col) = (y, x)
            cen2d[t, c, 0] = (cx, cy)                       # (x, y) pixel
            cconf[t, c, 0] = 1.0
    cen3d, _ = triangulate_keypoints(cen2d, cconf, cam_mats, conf_thresh=0.5)  # (T,1,3)
    cen3d = cen3d[:, 0, :]                                  # (T,3), NaN where <2 views
    ok = np.isfinite(cen3d).all(-1)
    return cen3d.astype(np.float32), ok


def polish_bout(stac_h5, cfg, kp3d_mm, kp3d_conf, masks_dict, calib_dir, *, kp_scale=1.0,
                bridge_mode="mask"):
    """Refine qpos with silhouette+containment. kp3d_mm (T,K,3), masks_dict from
    load_bout_masks (fly), calib_dir has Cam*.yaml.

    kp_scale is the trunk-Procrustes body-size scale applied to the keypoints
    before STAC (kp3d * kp_scale ~= model units), so the STAC-fitted model is at
    model scale; the bridge maps model -> mm with a fixed s = 1/kp_scale.

    Bridge (model->mm) is MASK-DRIVEN: the courtship 2D keypoints are unreliable,
    so global placement is anchored to the reliable SAM masks --
      s = 1/kp_scale                 (robust per-fly body size; constant)
      t : places the FK body centroid at the triangulated 3D mask centroid
      R : orientation from the keypoint Umeyama fit when >=3 keypoints are valid,
          else identity (the solver's qpos root quaternion, composed with R and
          refined by the silhouette cost, carries the rest of the orientation).
    A frame is usable (bridge != None) when its mask centroid triangulates from
    >=2 cameras -- no longer gated on keypoint count.

    Returns:
        (qpos_refined, bridges): qpos_refined is (T,nq) float array. bridges is
        a length-T list of (s, R (3,3), t (3,)) per-frame model->mm similarity
        bridges (the same bridge_s[t]/bridge_R[t]/bridge_t[t] used inside the
        solve), or None for frames where frame_ok[t] is False -- matching
        build_fly_outputs' "None bridge -> NaN frame" contract.
    """
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    sil = cfg.silhouette
    inp = build_solver_inputs(stac_h5, sil.xml)
    T = inp["q_init"].shape[0]
    q_init = np.asarray(inp["q_init"])[:T]; kp_data = inp["kp_data"][:T]
    anat = load_anatomy(sil.xml, sil.mesh_npz); fk = make_fk_repose(anat)
    cov_idx = silhouette_fk_indices(sil.mesh_npz, subset=sil.mesh_subset)
    cont_idx = appendage_vertex_indices(sil.mesh_npz, subset=sil.mesh_subset,
                                        include=tuple(sil.appendage_include))
    sil_qs = build_appendage_dof_mask(anat["m"], include=tuple(sil.appendage_include))
    conf_v = jnp.ones((len(cont_idx),))

    rt = ReprojectionTool(calib_dir)
    cam_Ms = np.stack([rt._camera_list[c].cameraMatrix[:2, :3] for c in range(rt.num_cameras)])
    cam_ts = np.stack([rt._camera_list[c].cameraMatrix[:2, 3] for c in range(rt.num_cameras)])
    cam_mats = np.asarray(rt.camera_matrices, np.float32)   # (C,4,3) for centroid triangulation

    tg = courtship_targets_from_masks(masks_dict["masks"], masks_dict["valid"],
                                      (cam_Ms, cam_ts), erode_px=sil.erode_px,
                                      n_points=sil.n_points, sdf_hw=tuple(sil.sdf_hw),
                                      bbox_margin=sil.bbox_margin)

    # Per-frame model->mm bridges. Two modes:
    #   'mask'     : s=1/kp_scale (const), t=triangulated SAM-mask centroid, R=keypoint-Umeyama
    #                -- robust when the 2D keypoints are unreliable (global placement anchored to
    #                the masks). Frame usable iff its mask centroid triangulates from >=2 cams.
    #   'keypoint' : full per-frame Umeyama(sites0, kp3d) similarity (s,R,t) -- tightest marker
    #                fit when the keypoints are multiview-consistent (e.g. after the camera-order
    #                fix). Frame usable iff >=3 valid keypoints.
    s_const = 1.0 / float(kp_scale)
    cen3d, cen_ok = (_mask_centroids_3d(masks_dict["masks"], masks_dict["valid"], cam_mats)
                     if bridge_mode == "mask" else (None, None))
    bridge_s = np.full((T,), s_const, np.float32)
    bridge_R = np.broadcast_to(np.eye(3, dtype=np.float32), (T, 3, 3)).copy()
    bridge_t = np.zeros((T, 3), np.float32)
    frame_ok = np.zeros(T, bool)
    for t in range(T):
        d0 = inp["mjx_data"].replace(qpos=jnp.asarray(q_init[t]))
        d0 = stac_utils.kinematics(inp["mjx_model"], d0); d0 = stac_utils.com_pos(inp["mjx_model"], d0)
        sites0 = np.asarray(stac_utils.get_site_xpos(d0, inp["site_idxs"]))
        # Require finite FK sites too: STAC can emit NaN qpos on sparse frames
        # (-> NaN sites0), which would make _umeyama's SVD non-convergent. Those
        # frames are simply left un-bridged (frame_ok stays False -> NaN output),
        # matching the "None bridge -> NaN frame" contract downstream.
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
            bridge_s[t] = s; bridge_R[t] = R; bridge_t[t] = tr; frame_ok[t] = True
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
            model_cen = sites0.mean(axis=0)           # place FK body centroid at 3D mask centroid
            tr = cen3d[t] - s_const * (R @ model_cen)
            bridge_R[t] = R; bridge_t[t] = tr; frame_ok[t] = True
    present_g = tg["present"].copy(); present_g[~frame_ok] = False
    conf_p_g = tg["conf_p"].copy(); conf_p_g[~frame_ok] = 0.0

    optimizer = str(getattr(sil, "optimizer", "adam"))
    if float(sil.silhouette_weight) == 0.0 and float(sil.containment_weight) == 0.0:
        # Polish disabled (both weights 0): Stage-D is a pass-through of the STAC +
        # keypoint-bridge pose. This is the current default -- the silhouette polish
        # cannot un-curl the STAC legs (one-sided containment) and drifts off the
        # reliable keypoints; see configs/silhouette/default.yaml. Machinery below
        # stays ready for when a weight is set.
        q = jnp.asarray(q_init)
    elif optimizer == "adam":
        # Gradient-based (Adam) refinement of the appendage DOFs only; root+body
        # stay frozen at the keypoint fit q_init. Replaces the jaxls GN/LM solve,
        # which cannot navigate the pixel-scale nonlinear silhouette cost (jaxls
        # uses non-scale-invariant lambda*I damping -> every GN step rejected).
        # See silhouette_refine for the full rationale. Coverage (chamfer) and
        # containment (SDF-relu) weights + a keypoint anchor are configurable;
        # the diagnosis found containment the effective term and coverage able to
        # fight it, so the config defaults to containment-dominant.
        from jarvis_jax.tracking.silhouette_refine import refine_appendages_adam
        q = refine_appendages_adam(
            jnp.asarray(q_init), fk_repose=fk,
            cov_vert_indices=cov_idx, cont_vert_indices=cont_idx,
            cam_Ms=jnp.asarray(cam_Ms), cam_ts=jnp.asarray(cam_ts),
            boundary_all=jnp.asarray(tg["boundary"]), conf_p_all=jnp.asarray(conf_p_g),
            sdf_all=jnp.asarray(tg["sdf"]), grid_scale_all=jnp.asarray(tg["grid_scale"]),
            grid_offset_all=jnp.asarray(tg["grid_offset"]), present_all=jnp.asarray(present_g),
            conf_v=conf_v, bridge_s_all=jnp.asarray(bridge_s),
            bridge_R_all=jnp.asarray(bridge_R), bridge_t_all=jnp.asarray(bridge_t),
            opt_mask=sil_qs, lb=inp["lb"], ub=inp["ub"],
            silhouette_weight=float(sil.silhouette_weight),
            containment_weight=float(sil.containment_weight),
            smooth_weight=float(sil.smooth_weight),
            anchor_weight=float(getattr(sil, "anchor_weight", 0.0)),
            limit_weight=float(getattr(sil, "limit_weight", 10.0)),
            beta=float(sil.beta), huber_delta=float(sil.huber_delta), margin=float(sil.margin),
            n_steps=int(getattr(sil, "refine_steps", 200)),
            lr=float(getattr(sil, "refine_lr", 1e-2)))
    else:
        solver = SilhouetteJaxlsBatchSolver(
            n_iter=int(sil.n_iter), smooth_weight=float(sil.smooth_weight),
            beta=float(sil.beta), huber_delta=float(sil.huber_delta),
            cg_tolerance_max=float(sil.cg_tolerance_max),
            cg_tolerance_min=float(sil.cg_tolerance_min))
        q = solver.solve_trajectory(
            q_init=jnp.asarray(q_init), mjx_model=inp["mjx_model"], mjx_data_template=inp["mjx_data"],
            kp_data=kp_data, qs_to_opt=inp["qs_to_opt"], kps_to_opt=inp["kps_to_opt"],
            lb=inp["lb"], ub=inp["ub"], site_idxs=inp["site_idxs"], q_reg_weights=inp["q_reg_weights"],
            fk_repose=fk, cov_vert_indices=cov_idx, cont_vert_indices=cont_idx,
            cam_Ms=jnp.asarray(cam_Ms), cam_ts=jnp.asarray(cam_ts),
            boundary_all=jnp.asarray(tg["boundary"]), conf_p_all=jnp.asarray(conf_p_g),
            sil_qs_mask=sil_qs, silhouette_weight=float(sil.silhouette_weight),
            sdf_all=jnp.asarray(tg["sdf"]), grid_scale_all=jnp.asarray(tg["grid_scale"]),
            grid_offset_all=jnp.asarray(tg["grid_offset"]), present_all=jnp.asarray(present_g),
            conf_v=conf_v, containment_weight=float(sil.containment_weight), margin=float(sil.margin),
            bridge_s_all=jnp.asarray(bridge_s), bridge_R_all=jnp.asarray(bridge_R),
            bridge_t_all=jnp.asarray(bridge_t))

    bridges = [
        (float(bridge_s[t]), bridge_R[t].copy(), bridge_t[t].copy()) if frame_ok[t] else None
        for t in range(T)
    ]
    return np.asarray(q), bridges
