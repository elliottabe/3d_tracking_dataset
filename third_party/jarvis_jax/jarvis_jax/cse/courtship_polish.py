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
from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
from jarvis_jax.cse.silhouette_ik_solve import build_solver_inputs, _umeyama
from jarvis_jax.cse.silhouette_dof import appendage_vertex_indices, build_appendage_dof_mask
from jarvis_jax.cse.silhouette_targets import silhouette_fk_indices
from jarvis_jax.cse.silhouette_sdf import _mask_bbox, _mask_to_sdf_crop, _BIG
from jarvis_jax.cse.silhouette_boundary import sample_boundary_points
from jarvis_jax.cse.silhouette_joint_ik import SilhouetteJaxlsBatchSolver


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


def polish_bout(stac_h5, cfg, kp3d_mm, kp3d_conf, masks_dict, calib_dir):
    """Refine qpos with silhouette+containment. kp3d_mm (T,K,3), masks_dict from
    load_bout_masks (fly), calib_dir has Cam*.yaml.

    Returns:
        (qpos_refined, bridges): qpos_refined is (T,nq) float array. bridges is
        a length-T list of (s, R (3,3), t (3,)) per-frame model->mm similarity
        bridges (the same bridge_s[t]/bridge_R[t]/bridge_t[t] used inside the
        solve), or None for frames where frame_ok[t] is False (fewer than 3
        valid triangulated keypoints) -- matching build_fly_outputs' "None
        bridge -> NaN frame" contract.
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

    tg = courtship_targets_from_masks(masks_dict["masks"], masks_dict["valid"],
                                      (cam_Ms, cam_ts), erode_px=sil.erode_px,
                                      n_points=sil.n_points, sdf_hw=tuple(sil.sdf_hw),
                                      bbox_margin=sil.bbox_margin)

    # per-frame model->mm bridges from q_init sites vs triangulated kp3d_mm
    bridge_s = np.ones((T,), np.float32)
    bridge_R = np.broadcast_to(np.eye(3, dtype=np.float32), (T, 3, 3)).copy()
    bridge_t = np.zeros((T, 3), np.float32)
    frame_ok = np.zeros(T, bool)
    for t in range(T):
        kok = np.isfinite(kp3d_mm[t]).all(-1) & (kp3d_conf[t] > 0)
        if kok.sum() < 3:
            continue
        d0 = inp["mjx_data"].replace(qpos=jnp.asarray(q_init[t]))
        d0 = stac_utils.kinematics(inp["mjx_model"], d0); d0 = stac_utils.com_pos(inp["mjx_model"], d0)
        sites0 = np.asarray(stac_utils.get_site_xpos(d0, inp["site_idxs"]))
        s, R, tr = _umeyama(sites0[kok], np.asarray(kp3d_mm[t])[kok])
        bridge_s[t] = s; bridge_R[t] = R; bridge_t[t] = tr; frame_ok[t] = True
    present_g = tg["present"].copy(); present_g[~frame_ok] = False
    conf_p_g = tg["conf_p"].copy(); conf_p_g[~frame_ok] = 0.0

    solver = SilhouetteJaxlsBatchSolver(n_iter=int(sil.n_iter), smooth_weight=float(sil.smooth_weight),
                                        beta=float(sil.beta), huber_delta=float(sil.huber_delta))
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
