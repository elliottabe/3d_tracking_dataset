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
  * ik_reproj_report -- the metric NONE of the above three actually is: FK'd
    FITTED model sites vs. the MEASURED triangulated kp3d, both reprojected
    with the SAME calibration and compared to the SAME detected kp2d/vis
    gate. per_camera_reproj_error/loo_reproj never touch the fitted pose
    (loo_reproj is a triangulation-only, pre-IK check); silhouette_iou is
    structurally tiny for a sparse mesh-vertex subset regardless of fit
    quality. This is the one metric that can tell "IK is bad" apart from
    "2D/triangulation is bad".
  * qc_report -- bundles all four across frames into one dict, saves json.
"""
from __future__ import annotations
import json
import numpy as np


def per_camera_reproj_error(rt, kp3d_mm, kp2d_by_cam, vis_by_cam):
    """Median per-camera pixel reprojection error over visible keypoints.

    Skips a (cam, kp) pair if the 3-D point, its reprojection, or the
    observed 2-D point is non-finite -- NOT just "not visible". Regression:
    a frame whose bridge failed (e.g. the fly briefly out of frame) leaves
    `kp3d_mm` all-NaN for that frame while `vis` (detector confidence) can
    still be True; before this guard, ONE such NaN reprojection error landed
    in `errs` and poisoned that (frame, cam)'s median to NaN via plain
    `np.median`, which then poisoned this whole run's `per_camera_reproj_px`
    in qc_report (measured: fly0/Session0 bout 28, 483/2007 frames all-NaN,
    164949 (cam,kp) pairs still `vis=True` inside them -- median NaN, n=12232
    -- while fly1, with no such frames, reported a real number)."""
    kp3d_mm = np.asarray(kp3d_mm, float)
    out = {}
    for c, kp2d in kp2d_by_cam.items():
        vis = np.asarray(vis_by_cam.get(c, np.ones(len(kp3d_mm), bool)), bool)
        errs = []
        for j in range(len(kp3d_mm)):
            if not vis[j]:
                continue
            xyz = kp3d_mm[j]
            obs = np.asarray(kp2d[j], float)
            if not (np.all(np.isfinite(xyz)) and np.all(np.isfinite(obs))):
                continue
            uv = rt.reproject_point(xyz)[c]
            if not np.all(np.isfinite(uv)):
                continue
            errs.append(float(np.linalg.norm(uv - obs)))
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
    from jarvis_jax.tracking.mesh_iou import (
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


def _ik_reproj_samples(rt, kp3d_mm, kp2d_by_cam, vis_by_cam):
    """Per-(cam, kp) reprojection errors for ONE 3-D source, ONE frame,
    pooled over every visible camera. Returns parallel ``(kp_idx, err_px)``
    arrays; skips any (cam, kp) pair where the source point, its
    reprojection, or the observed kp2d is non-finite (same NaN-poisoning
    guard as ``per_camera_reproj_error`` above)."""
    kp3d_mm = np.asarray(kp3d_mm, float)
    n_kp = len(kp3d_mm)
    idx_out, err_out = [], []
    for c, kp2d in kp2d_by_cam.items():
        vis = np.asarray(vis_by_cam.get(c, np.ones(n_kp, bool)), bool)
        for j in range(n_kp):
            if not vis[j]:
                continue
            xyz = kp3d_mm[j]
            obs = np.asarray(kp2d[j], float)
            if not (np.all(np.isfinite(xyz)) and np.all(np.isfinite(obs))):
                continue
            uv = rt.reproject_point(xyz)[c]
            if not np.all(np.isfinite(uv)):
                continue
            idx_out.append(j)
            err_out.append(float(np.linalg.norm(uv - obs)))
    return np.asarray(idx_out, int), np.asarray(err_out, float)


def _median_or_nan(err):
    return float(np.median(err)) if len(err) else float("nan")


def _ratio_or_nan(fitted, measured):
    if np.isfinite(fitted) and np.isfinite(measured) and measured != 0:
        return float(fitted / measured)
    return float("nan")


def ik_reproj_report(rt, *, kp3d_fitted_by_frame, kp2d_by_frame, vis_by_frame,
                     kp3d_measured_by_frame=None, kp_names=None, group_defs=None):
    """IK reprojection accuracy: FITTED (FK'd model sites, e.g. outputs.h5's
    ``kp3d_mm``) vs. MEASURED (raw triangulated, e.g. kp3d.npz's ``kp3d``)
    3-D, both reprojected with the recording's OWN calibration (`rt`) and
    compared against the SAME detected ``kp2d_by_frame`` gated by
    ``vis_by_frame`` (conf > cfg.detector.conf_thresh upstream). Reports
    fitted, measured, and their ratio -- the ratio is the actual signal,
    since a fitted number alone can't distinguish "IK is bad" from "2D is
    bad".

    ``kp3d_measured_by_frame=None`` computes fitted-only (measured/ratio come
    back as NaN) -- callers without a measured 3-D source (e.g. the COCO-val
    Phase-7 driver) still get a report, just not the ratio.

    ``kp_names``: adds a ``"per_keypoint"`` breakdown keyed by real keypoint
    name. ``group_defs`` (a ``{group_name: [kp_idx, ...]}`` dict, e.g. built
    by the caller from ``viz.core.colors.keypoint_groups`` -- this module
    deliberately has no dependency on the top-level ``viz`` package) adds a
    ``"per_group"`` breakdown. Both are omitted from the report if not given.
    """
    T = len(kp3d_fitted_by_frame)
    have_measured = kp3d_measured_by_frame is not None
    idx_fit_parts, err_fit_parts = [], []
    idx_meas_parts, err_meas_parts = [], []
    for t in range(T):
        i_f, e_f = _ik_reproj_samples(rt, kp3d_fitted_by_frame[t], kp2d_by_frame[t], vis_by_frame[t])
        idx_fit_parts.append(i_f); err_fit_parts.append(e_f)
        if have_measured:
            i_m, e_m = _ik_reproj_samples(rt, kp3d_measured_by_frame[t], kp2d_by_frame[t], vis_by_frame[t])
            idx_meas_parts.append(i_m); err_meas_parts.append(e_m)

    idx_fit = np.concatenate(idx_fit_parts) if idx_fit_parts else np.zeros(0, int)
    err_fit = np.concatenate(err_fit_parts) if err_fit_parts else np.zeros(0, float)
    if have_measured:
        idx_meas = np.concatenate(idx_meas_parts) if idx_meas_parts else np.zeros(0, int)
        err_meas = np.concatenate(err_meas_parts) if err_meas_parts else np.zeros(0, float)
    else:
        idx_meas = np.zeros(0, int)
        err_meas = np.zeros(0, float)

    def _entry(fmask, mmask):
        f = err_fit[fmask]
        m = err_meas[mmask] if have_measured else np.zeros(0, float)
        mf, mm = _median_or_nan(f), (_median_or_nan(m) if have_measured else float("nan"))
        return {"fitted_median_px": mf, "measured_median_px": mm,
                "ratio_fitted_over_measured": _ratio_or_nan(mf, mm),
                "n_fitted": int(f.size), "n_measured": int(m.size)}

    report = _entry(np.ones(err_fit.shape, bool), np.ones(err_meas.shape, bool))

    if kp_names is not None:
        kp_names = list(kp_names)
        report["per_keypoint"] = {
            name: _entry(idx_fit == j, idx_meas == j) for j, name in enumerate(kp_names)
        }

    if group_defs is not None:
        report["per_group"] = {
            gname: _entry(np.isin(idx_fit, list(idxs)), np.isin(idx_meas, list(idxs)))
            for gname, idxs in group_defs.items()
        }

    return report


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
              vis_by_frame, masks_by_frame, out_json=None,
              kp3d_measured_by_frame=None, kp_names=None, group_defs=None):
    """Bundle per-camera reproj, LOO reproj, silhouette IoU, and (additive)
    the IK reprojection metric over frames.

    ``kp3d_by_frame`` is the FITTED (FK'd) 3-D used by every metric here
    exactly as before. ``kp3d_measured_by_frame``/``kp_names``/``group_defs``
    are new, optional, additive kwargs (default None): passing them adds an
    ``"ik_reproj"`` key with fitted-vs-measured reprojection medians/ratio
    (overall, plus per-keypoint/per-group if ``kp_names``/``group_defs`` are
    given) WITHOUT touching any existing key -- see
    ``jarvis_jax.tracking.qc.ik_reproj_report``. Omitting them (the default)
    is byte-for-byte the previous report; existing qc.json files and callers
    (e.g. the COCO-val Phase-7 driver) are unaffected.
    """
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
        # nanmedian, not median: per_cam_all/loo_all/iou_*_all should no
        # longer CONTAIN a NaN after the per_camera_reproj_error guard above,
        # but this is the second, defensive layer -- one stray non-finite
        # value must never poison every OTHER good value's median the way it
        # did before (see per_camera_reproj_error's docstring for the
        # measured real-run instance of this).
        xs = [x for x in xs if np.isfinite(x)]
        return float(np.median(xs)) if xs else float("nan")

    report = {
        "per_camera_reproj_px": {"median": _agg(per_cam_all), "n": len(per_cam_all)},
        "loo_reproj_px": {"median": _agg(loo_all), "n_frames": len(loo_all)},
        "silhouette_iou": {"hard_median": _agg(iou_hard_all),
                           "soft_median": _agg(iou_soft_all),
                           "n_frames": len(iou_hard_all)},
        "n_frames": T,
    }

    if kp3d_measured_by_frame is not None or kp_names is not None:
        report["ik_reproj"] = ik_reproj_report(
            rt, kp3d_fitted_by_frame=kp3d_by_frame, kp2d_by_frame=kp2d_by_frame,
            vis_by_frame=vis_by_frame, kp3d_measured_by_frame=kp3d_measured_by_frame,
            kp_names=kp_names, group_defs=group_defs)

    if out_json is not None:
        import os
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(_jsonable(report), f, indent=2)
    return report
