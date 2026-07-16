# jarvis_jax/cse/qc.py
"""Phase 7 / component 8: bundled QC for a fitted fly trajectory.

Pure-ish functions (no FK -- callers pass already-FK'd 3-D points/mesh):
  * per_camera_reproj_error -- all 50 kp; generalizes reproj_validate.py's
    per-cam reprojection-error to every keypoint (was wing-tip-only there).
  * loo_reproj -- generalizes reproj_validate.py's leave-one-out wing-tip test
    to all 50 keypoints: per kp, triangulate from all visible cams; per
    held-out cam re-triangulate from the OTHER visible cams (>=2), reproject
    into the held-out cam, pixel error vs its own 2-D.
  * silhouette_iou_report -- reuses Phase-6 iou_of_projected_verts /
    soft_iou_of_verts over the posed mesh subset vs SAM masks.
  * qc_report -- bundles all three across frames into one dict, saves json.
"""
from __future__ import annotations
import json
import numpy as np


def per_camera_reproj_error(rt, kp3d_mm, kp2d_by_cam, vis_by_cam):
    """Median per-camera pixel reprojection error over visible keypoints."""
    kp3d_mm = np.asarray(kp3d_mm, float)
    out = {}
    for c, kp2d in kp2d_by_cam.items():
        vis = np.asarray(vis_by_cam.get(c, np.ones(len(kp3d_mm), bool)), bool)
        errs = []
        for j in range(len(kp3d_mm)):
            if not vis[j]:
                continue
            uv = rt.reproject_point(kp3d_mm[j])[c]
            errs.append(float(np.linalg.norm(uv - np.asarray(kp2d[j], float))))
        if errs:
            out[c] = float(np.median(errs))
    return out


def loo_reproj(rt, kp2d_by_cam, vis_by_cam):
    """Leave-one-out reprojection error over all keypoints.

    For each kp: gather cams where it is visible; for each held-out visible
    cam with >=2 OTHER visible cams, triangulate from the others, reproject
    into the held-out cam, pixel error vs that cam's own observation.
    """
    cams = sorted(kp2d_by_cam)
    if not cams:
        return {"per_kp": np.zeros(0), "median": float("nan"), "n": 0}
    n_kp = len(next(iter(kp2d_by_cam.values())))
    per_kp = np.full(n_kp, np.nan)
    all_errs = []
    for j in range(n_kp):
        vis_cams = [c for c in cams
                    if np.asarray(vis_by_cam.get(c, np.ones(n_kp, bool)))[j]]
        if len(vis_cams) < 3:
            continue
        obs = np.zeros((rt.num_cameras, 2))
        for c in vis_cams:
            obs[c] = np.asarray(kp2d_by_cam[c][j], float)
        kp_errs = []
        for held in vis_cams:
            others = [c for c in vis_cams if c != held]
            if len(others) < 2:
                continue
            X = rt.reconstruct_point(obs, cams_to_use=others)
            proj = rt.reproject_point(X)[held]
            kp_errs.append(float(np.linalg.norm(proj - obs[held])))
        if kp_errs:
            per_kp[j] = float(np.mean(kp_errs))
            all_errs.extend(kp_errs)
    return {"per_kp": per_kp,
            "median": float(np.median(all_errs)) if all_errs else float("nan"),
            "n": len(all_errs)}


def silhouette_iou_report(rt, mesh_mm, masks_by_cam):
    """Hard+soft IoU of the projected posed mesh subset vs SAM masks, per cam."""
    from jarvis_jax.tracking.run_silhouette_polish import (
        iou_of_projected_verts, soft_iou_of_verts)
    mesh_mm = np.asarray(mesh_mm, float)
    hard, soft = {}, {}
    for c, mask in masks_by_cam.items():
        if mask is None or len(mesh_mm) == 0:
            continue
        mask = np.asarray(mask)
        uv = np.stack([rt.reproject_point(mesh_mm[k])[c] for k in range(len(mesh_mm))], 0)
        hard[c] = iou_of_projected_verts(uv, mask.shape, mask)
        soft[c] = soft_iou_of_verts(uv, mask)
    hm = float(np.mean(list(hard.values()))) if hard else float("nan")
    sm = float(np.mean(list(soft.values()))) if soft else float("nan")
    return {"hard": hard, "soft": soft, "hard_mean": hm, "soft_mean": sm}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def qc_report(rt, *, kp3d_by_frame, mesh_by_frame, kp2d_by_frame,
              vis_by_frame, masks_by_frame, out_json=None):
    """Bundle per-camera reproj, LOO reproj, and silhouette IoU over frames."""
    T = len(kp3d_by_frame)
    per_cam_all, loo_all, iou_hard_all, iou_soft_all = [], [], [], []
    for t in range(T):
        pc = per_camera_reproj_error(rt, kp3d_by_frame[t], kp2d_by_frame[t], vis_by_frame[t])
        per_cam_all.extend(pc.values())
        lo = loo_reproj(rt, kp2d_by_frame[t], vis_by_frame[t])
        if lo["n"]:
            loo_all.append(lo["median"])
        sr = silhouette_iou_report(rt, mesh_by_frame[t], masks_by_frame[t])
        if np.isfinite(sr["hard_mean"]):
            iou_hard_all.append(sr["hard_mean"])
        if np.isfinite(sr["soft_mean"]):
            iou_soft_all.append(sr["soft_mean"])

    def _agg(xs):
        return float(np.median(xs)) if xs else float("nan")

    report = {
        "per_camera_reproj_px": {"median": _agg(per_cam_all), "n": len(per_cam_all)},
        "loo_reproj_px": {"median": _agg(loo_all), "n_frames": len(loo_all)},
        "silhouette_iou": {"hard_median": _agg(iou_hard_all),
                           "soft_median": _agg(iou_soft_all),
                           "n_frames": len(iou_hard_all)},
        "n_frames": T,
    }
    if out_json is not None:
        import os
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(_jsonable(report), f, indent=2)
    return report
