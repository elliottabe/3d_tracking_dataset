"""Phase 6 validation driver: joint 2D+3D silhouette-Chamfer refinement on the
MALE recording 2026_03_18_15_31_22, warm-started from the STAC fit.

Reports silhouette IoU before/after, multi-view reprojection before/after, and
the 3-D marker residual delta. HONEST success criterion: IoU improves (up to the
~0.76 collision-mesh cap), reprojection does NOT regress, 3-D markers not
materially worse. A neutral/small result is acceptable and must be reported
truthfully (the collision-mesh silhouette IoU cap is a known ceiling).

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
    max_frames=0, smooth_weight=0.1, n_iter=50, out_dir,
):
    import h5py
    import jax.numpy as jnp
    import stac_mjx.io_dict_to_hdf5 as ioh5
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

    ik_raw = ioh5.load(ik_h5)
    marker_sites = np.asarray(ik_raw["marker_sites"])[:T]

    bout_h5 = os.path.join(os.path.dirname(os.path.dirname(ik_h5)), f"{recording}_bout.h5")
    with h5py.File(bout_h5, "r") as f:
        fs_imgids = f["fs_imgids"][()][:T]

    # --- assemble silhouette targets ---
    sil_data, meta = build_silhouette_targets(
        root, split, fs_imgids, calib_dir, n_points=n_points,
    )
    anat = load_anatomy(model_xml, mesh_npz)
    fk = make_fk_repose(anat)
    vert_indices = silhouette_fk_indices(mesh_npz, subset="fps_300")

    small = dict(inputs)
    small["q_init"] = q_init
    small["kp_data"] = kp_data

    solver = SilhouetteJaxlsBatchSolver(
        n_iter=n_iter, smooth_weight=smooth_weight, beta=beta, huber_delta=huber_delta)

    def _solve(weight, sd):
        return np.asarray(solver.solve_trajectory(
            q_init=small["q_init"], mjx_model=small["mjx_model"],
            mjx_data_template=small["mjx_data"], kp_data=small["kp_data"],
            qs_to_opt=small["qs_to_opt"], kps_to_opt=small["kps_to_opt"],
            lb=small["lb"], ub=small["ub"], site_idxs=small["site_idxs"],
            q_reg_weights=small["q_reg_weights"],
            fk_repose=fk, vert_indices=vert_indices, sil_data=sd,
            n_pts=n_points, silhouette_weight=weight,
        ))

    q_before = _solve(0.0, None)                       # 3-D only (baseline)
    q_after = _solve(silhouette_weight, sil_data)      # joint 2D+3D

    # --- metrics (IoU + reproj + marker residual), before vs after ---
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

    def _metrics(qtraj):
        ious, soft_ious, reprojs, mresids = [], [], [], []
        for t in range(T):
            cam2img = _cam2img_for_frame(fs_imgids[t], id2file, cam_names)
            kp_mm, kok = _triangulate_kp_mm(rt, kp_names, coco_kpnames, cam2img, id2ann_multi)
            if kok.sum() < 3:
                continue
            s, R, tr = _umeyama(marker_sites[t][kok], kp_mm[kok])
            verts_model = np.asarray(fk(jnp.asarray(qtraj[t].astype(np.float32)), 1.0, vert_indices))
            verts_mm = _model_to_mm(verts_model, s, R, tr)
            # 3-D marker residual (FK sites vs triangulated kp) -- proxy: distance
            # of triangulated markers to their FK positions (kok subset).
            mresids.append(float(np.mean(np.linalg.norm(
                _model_to_mm(marker_sites[t][kok], s, R, tr) - kp_mm[kok], axis=1))))
            for c, iid in cam2img.items():
                ann = _ann_for_image(id2ann_multi, iid, None)
                if ann is None:
                    continue
                fn = id2file[iid]
                mask = _load_sam_mask(root, split, fn, ann["id"])
                if mask is None:
                    continue
                mask = np.asarray(mask)
                uv = np.stack([rt.reproject_point(verts_mm[k])[c] for k in range(0, len(verts_mm))], 0)
                ious.append(iou_of_projected_verts(uv, mask.shape, mask))
                # baseline-comparable soft-IoU (eval-only; not an objective term)
                soft_ious.append(soft_iou_of_verts(uv, mask))
            # reprojection of the 50 kp sites
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
                        uv_pred = rt.reproject_point(_model_to_mm(marker_sites[t][j][None], s, R, tr)[0])[c]
                        reprojs.append(float(np.linalg.norm(uv_pred - kp2d[ci, :2])))
        return (float(np.mean(ious)) if ious else float("nan"),
                float(np.mean(soft_ious)) if soft_ious else float("nan"),
                float(np.mean(reprojs)) if reprojs else float("nan"),
                float(np.mean(mresids)) if mresids else float("nan"))

    iou_b, soft_b, reproj_b, mres_b = _metrics(q_before)
    iou_a, soft_a, reproj_a, mres_a = _metrics(q_after)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{recording}_polish_qpos.npz"),
             q_before=q_before, q_after=q_after)

    return dict(
        iou_before=iou_b, iou_after=iou_a,
        soft_iou_before=soft_b, soft_iou_after=soft_a,
        reproj_px_before=reproj_b, reproj_px_after=reproj_a,
        marker_resid_before=mres_b, marker_resid_after=mres_a,
        n_frames=T,
    )


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
    ap.add_argument("--smooth-weight", type=float, default=0.1)
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    rep = run_polish(
        a.recording, ik_h5=a.ik_h5, model_xml=a.xml, mesh_npz=a.mesh, root=a.root,
        split=a.split, calib_dir=a.calib_dir, n_points=a.n_points,
        silhouette_weight=a.silhouette_weight, beta=a.beta, huber_delta=a.huber_delta,
        max_frames=a.max_frames, smooth_weight=a.smooth_weight, n_iter=a.n_iter,
        out_dir=a.out_dir,
    )
    print("PHASE6 POLISH REPORT")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print(f"  soft-IoU delta: {rep['soft_iou_after'] - rep['soft_iou_before']:+.4f} "
          f"(baseline-comparable; cap ~0.76; neutral/small is acceptable and reported truthfully)")
    print(f"  hard-IoU delta: {rep['iou_after'] - rep['iou_before']:+.4f} (point-splat proxy)")
    print(f"  reproj delta (px): {rep['reproj_px_after'] - rep['reproj_px_before']:+.4f} "
          f"(must NOT regress)")
    print(f"  marker-resid delta (mm): {rep['marker_resid_after'] - rep['marker_resid_before']:+.4f} "
          f"(3-D markers must not be materially worse)")


if __name__ == "__main__":
    main()
