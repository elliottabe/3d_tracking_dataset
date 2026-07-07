"""Validation driver: coverage (boundary-Chamfer) + containment (SDF-relu)
silhouette refinement on the MALE recording 2026_03_18_15_31_22, warm-started
from the STAC fit.

Reports the filled-triangle silhouette IoU before/after (primary success
measure; the "overlay methodology" -- rasterize the projected mesh faces and
IoU vs the SAM mask, not a point splat), plus two do-no-harm guards: multi-view
keypoint reprojection RMSE and MPJPE vs the STAC-init pose. HONEST success
criterion: IoU improves and neither guard materially worsens. `iou_of_projected_verts`
and `soft_iou_of_verts` are retained (point-splat / soft-IoU report helpers from
the prior Chamfer-only driver) for their existing unit-test coverage; `run_polish`
itself now reports via `filled_tri_iou` only.

Coordinator-run on GPU (the heavy solve is NOT a pytest). A tiny CPU smoke lives
in tests/test_run_silhouette_polish.py.
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np


def iou_of_projected_verts(verts2d, mask_shape, ref_mask) -> float:
    """Coarse silhouette HARD-IoU: splat projected 2-D verts to a binary mask
    (round to pixel), IoU vs ref_mask. Out-of-bounds verts dropped. Pure NumPy.
    The before/after DELTA is what matters (a point-splat is sparse vs a filled
    silhouette); the baseline-comparable absolute is soft_iou_of_verts below."""
    H, W = mask_shape
    pred = np.zeros((H, W), dtype=bool)
    v = np.asarray(verts2d)
    xs = np.round(v[:, 0]).astype(int)
    ys = np.round(v[:, 1]).astype(int)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    pred[ys[ok], xs[ok]] = True
    ref = np.asarray(ref_mask, dtype=bool)
    inter = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    return float(inter / union) if union > 0 else 0.0


def filled_tri_iou(verts2d, faces, mask_shape, ref_mask) -> float:
    """Filled-triangle silhouette hard-IoU: rasterize the projected mesh faces
    (cv2.fillPoly) and IoU vs ref_mask. The overlay methodology (a true filled
    silhouette, not a point splat)."""
    import cv2
    H, W = mask_shape
    sil = np.zeros((H, W), np.uint8)
    v = np.asarray(verts2d)
    tris = v[np.asarray(faces)].astype(np.int32)     # (F,3,2)
    cv2.fillPoly(sil, tris, 1)
    sil = sil > 0
    ref = np.asarray(ref_mask, bool)
    inter = np.logical_and(sil, ref).sum()
    union = np.logical_or(sil, ref).sum()
    return float(inter / union) if union > 0 else 0.0


def soft_iou_of_verts(verts2d, mask, *, sigma: float = 1.3, splat_k: int = 2) -> float:
    """EVAL-ONLY soft-IoU (baseline-comparable). NOT an objective term.

    Windowed-Gaussian splat of the projected 2-D verts into a soft occupancy
    grid on the mask's own pixel frame, then soft-IoU vs the (binary) SAM mask,
    exactly mirroring silhouette_fit.py::soft_sil + its IoU loss (extracted here
    as a reusable, report-time-only function). Runs once per frame at report
    time -- never inside the LM loop -- so rasterizing ALL verts is fine and the
    locked "no soft-IoU objective" scope is preserved (this is a METRIC only).

    Args:
        verts2d: (V, 2) projected mesh verts in pixel (x, y) for this camera.
        mask: (H, W) SAM mask (bool / {0,1}).
        sigma: Gaussian splat sigma (matches silhouette_fit default 1.3).
        splat_k: half-window in pixels for the splat (K=2 in silhouette_fit).

    Returns:
        soft-IoU in [0, 1]; 0.0 if the mask is empty.
    """
    mask = np.asarray(mask).astype(np.float32)
    H, W = mask.shape
    if mask.sum() == 0:
        return 0.0
    v = np.asarray(verts2d, dtype=np.float64)
    grid = np.zeros((H, W), dtype=np.float64)
    bx = np.floor(v[:, 0]).astype(int)
    by = np.floor(v[:, 1]).astype(int)
    for di in range(-splat_k, splat_k + 1):
        for dj in range(-splat_k, splat_k + 1):
            cx = bx + dj
            cy = by + di
            w = np.exp(-(((cx + 0.5 - v[:, 0]) ** 2 + (cy + 0.5 - v[:, 1]) ** 2))
                       / (2 * sigma ** 2))
            valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
            np.add.at(grid, (np.clip(cy, 0, H - 1), np.clip(cx, 0, W - 1)),
                      np.where(valid, w, 0.0))
    soft = 1.0 - np.exp(-grid)                       # soft occupancy in [0,1)
    inter = float((soft * mask).sum())
    union = float((soft + mask - soft * mask).sum())
    return inter / union if union > 0 else 0.0


def run_polish(
    recording, *, ik_h5, model_xml, mesh_npz, root, split="val", calib_dir=None,
    n_points=128, silhouette_weight=0.3, beta=8.0, huber_delta=0.0,
    max_frames=0, smooth_weight=0.0, n_iter=50, out_dir,   # dataset is NOT temporally coherent -> no temporal smoothing (see configs/silhouette)
    erode_px=8, sdf_hw=(128, 128), bbox_margin=0.4, margin=0.0,
    containment_weight=0.3, mesh_subset="fps_300",
    appendage_include=("wing", "leg", "abdomen"),
):
    import h5py
    import jax.numpy as jnp
    import stac_mjx.utils as stac_utils
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.cse.silhouette_ik_solve import (
        build_solver_inputs, _umeyama, _model_to_mm, _triangulate_kp_mm,
        _cam2img_for_frame, _ann_for_image, _load_sam_mask,
        _DEFAULT_REFINED_CALIB_DIR, _DEFAULT_FACTORY_CALIB_DIR,
    )
    from jarvis_jax.cse.silhouette_targets import (
        build_silhouette_targets, silhouette_fk_indices,
    )
    from jarvis_jax.cse.silhouette_sdf import build_sdf_stack
    from jarvis_jax.cse.silhouette_dof import build_appendage_dof_mask, appendage_vertex_indices
    from jarvis_jax.cse.silhouette_joint_ik import SilhouetteJaxlsBatchSolver

    if calib_dir is None:
        calib_dir = (_DEFAULT_REFINED_CALIB_DIR
                     if os.path.exists(_DEFAULT_REFINED_CALIB_DIR)
                     else _DEFAULT_FACTORY_CALIB_DIR)

    inputs = build_solver_inputs(ik_h5, model_xml)
    T_full = inputs["q_init"].shape[0]
    T = T_full if max_frames <= 0 else min(max_frames, T_full)
    q_init = inputs["q_init"][:T]
    kp_data = inputs["kp_data"][:T]

    bout_h5 = os.path.join(os.path.dirname(os.path.dirname(ik_h5)), f"{recording}_bout.h5")
    with h5py.File(bout_h5, "r") as f:
        fs_imgids = f["fs_imgids"][()][:T]

    tg = build_silhouette_targets(root, split, fs_imgids, calib_dir,
                                  n_points=n_points, erode_px=erode_px)
    sdf = build_sdf_stack(root, split, fs_imgids, calib_dir,
                          out_hw=sdf_hw, bbox_margin=bbox_margin)

    anat = load_anatomy(model_xml, mesh_npz)
    fk = make_fk_repose(anat)
    faces = np.asarray(anat["faces"])
    full_idx = np.arange(len(anat["vlocal"]), dtype=np.int32)
    cov_idx = silhouette_fk_indices(mesh_npz, subset=mesh_subset)
    cont_idx = appendage_vertex_indices(mesh_npz, subset=mesh_subset, include=appendage_include)
    sil_qs = build_appendage_dof_mask(anat["m"], include=appendage_include)
    conf_v = jnp.ones((len(cont_idx),))

    cam_Ms = jnp.asarray(tg["cam_Ms"]); cam_ts = jnp.asarray(tg["cam_ts"])

    # calibration + coco maps (reused for the bridge precompute AND the metrics)
    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann_multi = {}
    for an in coco["annotations"]:
        id2ann_multi.setdefault(an["image_id"], []).append(an)
    coco_kpnames = coco["keypoint_names"]
    kp_names = list(inputs["kp_names"])
    name2coco = {n: i for i, n in enumerate(coco_kpnames)}

    # FIXED per-frame model->mm bridges from the STAC init pose (same Umeyama the
    # metrics use). The in-solver silhouette/containment costs FK verts in MODEL
    # frame; these bridges map them to mm BEFORE the affine (mm->px) projection --
    # without them the mesh projects in model units through mm cameras and the fit
    # is destroyed. Frames with <3 triangulated kp keep an identity bridge.
    bridge_s = np.ones((T,), np.float32)
    bridge_R = np.broadcast_to(np.eye(3, dtype=np.float32), (T, 3, 3)).copy()
    bridge_t = np.zeros((T, 3), np.float32)
    frame_ok = np.zeros(T, bool)          # frames with a valid (>=3 kp) bridge
    for t in range(T):
        cam2img = _cam2img_for_frame(fs_imgids[t], id2file, cam_names)
        kp_mm, kok = _triangulate_kp_mm(rt, kp_names, coco_kpnames, cam2img, id2ann_multi)
        if kok.sum() < 3:
            continue
        d0 = inputs["mjx_data"].replace(qpos=jnp.asarray(q_init[t]))
        d0 = stac_utils.kinematics(inputs["mjx_model"], d0)
        d0 = stac_utils.com_pos(inputs["mjx_model"], d0)
        sites0 = np.asarray(stac_utils.get_site_xpos(d0, inputs["site_idxs"]))
        s, R, tr = _umeyama(sites0[kok], kp_mm[kok])
        bridge_s[t] = s; bridge_R[t] = R; bridge_t[t] = tr
        frame_ok[t] = True
    bridge_s = jnp.asarray(bridge_s); bridge_R = jnp.asarray(bridge_R)
    bridge_t = jnp.asarray(bridge_t)

    # Gate the silhouette OFF on frames without a valid bridge (<3 triangulated
    # kp -> identity bridge): projecting model-frame verts through mm cameras
    # there would inject spurious residuals (esp. on OOD females). Zero their
    # containment present + coverage conf so only keypoint/reg/smoothness act.
    present_g = np.asarray(sdf["present"]).copy(); present_g[~frame_ok] = False
    conf_p_g = np.asarray(tg["conf_p"]).copy(); conf_p_g[~frame_ok] = 0.0
    present_g = jnp.asarray(present_g); conf_p_g = jnp.asarray(conf_p_g)

    solver = SilhouetteJaxlsBatchSolver(
        n_iter=n_iter, smooth_weight=smooth_weight, beta=beta, huber_delta=huber_delta)

    def _solve(sw, cw):
        return np.asarray(solver.solve_trajectory(
            q_init=q_init, mjx_model=inputs["mjx_model"], mjx_data_template=inputs["mjx_data"],
            kp_data=kp_data, qs_to_opt=inputs["qs_to_opt"], kps_to_opt=inputs["kps_to_opt"],
            lb=inputs["lb"], ub=inputs["ub"], site_idxs=inputs["site_idxs"],
            q_reg_weights=inputs["q_reg_weights"], fk_repose=fk,
            cov_vert_indices=cov_idx, cont_vert_indices=cont_idx,
            cam_Ms=cam_Ms, cam_ts=cam_ts, boundary_all=jnp.asarray(tg["boundary"]),
            conf_p_all=conf_p_g, sil_qs_mask=sil_qs, silhouette_weight=sw,
            sdf_all=jnp.asarray(sdf["sdf"]), grid_scale_all=jnp.asarray(sdf["grid_scale"]),
            grid_offset_all=jnp.asarray(sdf["grid_offset"]), present_all=present_g,
            conf_v=conf_v, containment_weight=cw, margin=margin,
            bridge_s_all=bridge_s, bridge_R_all=bridge_R, bridge_t_all=bridge_t))

    q_before = _solve(0.0, 0.0)                          # keypoint-only baseline
    q_after = _solve(silhouette_weight, containment_weight)

    def _metrics(qtraj):
        mjx_model = inputs["mjx_model"]; mjx_data_template = inputs["mjx_data"]
        site_idxs = inputs["site_idxs"]
        ious, reprojs, mpjpes = [], [], []
        for t in range(T):
            cam2img = _cam2img_for_frame(fs_imgids[t], id2file, cam_names)
            kp_mm, kok = _triangulate_kp_mm(rt, kp_names, coco_kpnames, cam2img, id2ann_multi)
            if kok.sum() < 3:
                continue
            data_t = mjx_data_template.replace(qpos=jnp.asarray(qtraj[t]))
            data_t = stac_utils.kinematics(mjx_model, data_t)
            data_t = stac_utils.com_pos(mjx_model, data_t)
            sites_model = np.asarray(stac_utils.get_site_xpos(data_t, site_idxs))
            s, R, tr = _umeyama(sites_model[kok], kp_mm[kok])
            # MPJPE vs STAC init pose (both FK'd sites -> mm via same bridge)
            data_s = mjx_data_template.replace(qpos=jnp.asarray(q_init[t]))
            data_s = stac_utils.kinematics(mjx_model, data_s)
            data_s = stac_utils.com_pos(mjx_model, data_s)
            sites_stac = np.asarray(stac_utils.get_site_xpos(data_s, site_idxs))
            mpjpes.append(float(np.mean(np.linalg.norm(
                _model_to_mm(sites_model, s, R, tr) - _model_to_mm(sites_stac, s, R, tr), axis=1))))
            # filled-tri IoU on the FULL mesh (overlay methodology), affine projection
            verts_model = np.asarray(fk(jnp.asarray(qtraj[t].astype(np.float32)), 1.0, full_idx))
            verts_mm = _model_to_mm(verts_model, s, R, tr)
            for c, iid in cam2img.items():
                ann = _ann_for_image(id2ann_multi, iid, None)
                if ann is None:
                    continue
                mask = _load_sam_mask(root, split, id2file[iid], ann["id"])
                if mask is None:
                    continue
                mask = np.asarray(mask)
                uv = verts_mm @ np.asarray(tg["cam_Ms"])[c].T + np.asarray(tg["cam_ts"])[c]
                ious.append(filled_tri_iou(uv, faces, mask.shape, mask))
            for j, nm in enumerate(kp_names):
                ci = name2coco.get(nm)
                if ci is None:
                    continue
                for c, iid in cam2img.items():
                    ann = _ann_for_image(id2ann_multi, iid, None)
                    if ann is None:
                        continue
                    kp2d = np.asarray(ann["keypoints"], float).reshape(-1, 3)
                    if kp2d[ci, 2] > 0:
                        uvp = _model_to_mm(sites_model[j][None], s, R, tr)[0]
                        uvp = uvp @ np.asarray(tg["cam_Ms"])[c].T + np.asarray(tg["cam_ts"])[c]
                        reprojs.append(float(np.linalg.norm(uvp - kp2d[ci, :2])))
        return (float(np.mean(ious)) if ious else float("nan"),
                float(np.sqrt(np.mean(np.square(reprojs)))) if reprojs else float("nan"),
                float(np.mean(mpjpes)) if mpjpes else float("nan"))

    iou_b, reproj_b, mpjpe_b = _metrics(q_before)
    iou_a, reproj_a, mpjpe_a = _metrics(q_after)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{recording}_polish_qpos.npz"),
             q_before=q_before, q_after=q_after)
    return dict(iou_before=iou_b, iou_after=iou_a,
                reproj_px_before=reproj_b, reproj_px_after=reproj_a,
                mpjpe_stac_before=mpjpe_b, mpjpe_stac_after=mpjpe_a, n_frames=T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="2026_03_18_15_31_22")
    ap.add_argument("--ik-h5", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--n-points", type=int, default=128)
    ap.add_argument("--silhouette-weight", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--huber-delta", type=float, default=0.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--smooth-weight", type=float, default=0.0)  # dataset not temporally coherent
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--containment-weight", type=float, default=0.3)
    ap.add_argument("--erode-px", type=int, default=8)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--sdf-hw", type=int, default=128)
    a = ap.parse_args()
    rep = run_polish(
        a.recording, ik_h5=a.ik_h5, model_xml=a.xml, mesh_npz=a.mesh, root=a.root,
        split=a.split, calib_dir=a.calib_dir, n_points=a.n_points,
        silhouette_weight=a.silhouette_weight, beta=a.beta, huber_delta=a.huber_delta,
        max_frames=a.max_frames, smooth_weight=a.smooth_weight, n_iter=a.n_iter,
        out_dir=a.out_dir, containment_weight=a.containment_weight,
        erode_px=a.erode_px, margin=a.margin, sdf_hw=(a.sdf_hw, a.sdf_hw),
    )
    print("PHASE6 POLISH REPORT")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print(f"  IoU delta (filled-tri, primary): {rep['iou_after'] - rep['iou_before']:+.4f} "
          f"(higher is better)")
    print(f"  reproj RMSE delta (px, do-no-harm): {rep['reproj_px_after'] - rep['reproj_px_before']:+.4f} "
          f"(must NOT materially worsen)")
    print(f"  MPJPE-vs-STAC delta (mm, do-no-harm): "
          f"{rep['mpjpe_stac_after'] - rep['mpjpe_stac_before']:+.4f} "
          f"(must NOT materially worsen)")


if __name__ == "__main__":
    main()
