# jarvis_jax/cse/qc.py
"""Phase 7 / component 8: bundled QC for a fitted fly trajectory.

Pure-ish functions (no FK -- callers pass already-FK'd 3-D points/mesh):
  * per_camera_reproj_error -- all 50 kp; generalizes reproj_validate.py's
    per-cam reprojection-error to every keypoint (was wing-tip-only there).
  * loo_reproj -- generalizes reproj_validate.py's leave-one-out wing-tip test
    to all 50 keypoints: per kp, triangulate from all visible cams; per
    held-out cam re-triangulate from the OTHER visible cams (>=2), reproject
    into the held-out cam, pixel error vs its own 2-D.
  * mesh_mask_iou_report -- reuses Phase-6 iou_of_projected_verts /
    soft_iou_of_verts over the posed mesh subset vs SAM masks.
  * ik_reproj_report -- the metric NONE of the above three actually is: FK'd
    FITTED model sites vs. the MEASURED triangulated kp3d, both reprojected
    with the SAME calibration and compared to the SAME detected kp2d/vis
    gate. per_camera_reproj_error/loo_reproj never touch the fitted pose
    (loo_reproj is a triangulation-only, pre-IK check); mesh_mask_iou is
    structurally tiny for a sparse mesh-vertex subset regardless of fit
    quality. This is the one metric that can tell "IK is bad" apart from
    "2D/triangulation is bad".
  * frame_metrics -- the per-(frame, camera) arrays qc_report AND
    qc_perframe.per_frame_qc both need; compute once, pass to both.
  * qc_report -- bundles all four across frames into one dict, saves json.

VECTORISED 2026-09-01 (behaviour-preserving). Every function here used to call
``rt.reproject_point`` / ``rt.reconstruct_point`` once per (point, camera,
frame) from Python -- ~7M scalar calls per bout, which made qc.json (3m39s) +
qc_perframe.npz (2m52s) half of a 35-minute bout. They now use the batched
``ReprojectionTool.reproject_points`` / ``.reconstruct_points``, which are
bit-identical to the scalar versions (einsum keeps the dot order; numpy's
stacked SVD calls the same LAPACK routine per matrix), and the mesh-vs-mask
IoU is bounding-boxed in ``tracking.mesh_iou``. An ``rt`` that only implements
the scalar API (test doubles, the JARVIS torch adapter in
``predict.sam3_driver``) still works: the SAME code path just falls back to
looping the scalar primitive, so there is one algorithm, not two.
"""
from __future__ import annotations
import json
import numpy as np


# ---------------------------------------------------------------------------
# Batched reprojection / triangulation, with a scalar fallback.
# ---------------------------------------------------------------------------

def _reproject_all(rt, pts):
    """(..., 3) world points -> (..., num_cam, 2) pixels.

    Uses ``rt.reproject_points`` (batched, bit-identical -- see
    ``jarvis_jax.geometry.reprojection_tool``) when the tool provides it, else
    loops ``rt.reproject_point`` so any object with only the scalar API keeps
    working.
    """
    pts = np.asarray(pts, dtype=float)
    batch = getattr(rt, "reproject_points", None)
    if batch is not None:
        return np.asarray(batch(pts))
    flat = pts.reshape(-1, 3)
    out = np.stack([np.asarray(rt.reproject_point(p), float) for p in flat])
    return out.reshape(*pts.shape[:-1], out.shape[-2], out.shape[-1])


def _reconstruct_all(rt, obs, cams):
    """Triangulate N systems. ``obs`` (N, n, 2) are the observations of the
    cameras named by ``cams`` (N, n), in DLT-row order.

    Uses ``rt.reconstruct_points`` (batched, bit-identical) when available,
    else loops ``rt.reconstruct_point`` with a full-length ``points2d`` array
    exactly as the scalar callers did (it only reads ``cams_to_use`` rows).
    """
    obs = np.asarray(obs, dtype=float)
    cams = np.asarray(cams, dtype=int)
    batch = getattr(rt, "reconstruct_points", None)
    if batch is not None:
        return np.asarray(batch(obs, cams))
    n_cam = int(getattr(rt, "num_cameras", int(cams.max()) + 1 if cams.size else 0))
    out = np.empty((cams.shape[0], 3), dtype=float)
    for i in range(cams.shape[0]):
        full = np.zeros((n_cam, 2))
        full[cams[i]] = obs[i]
        out[i] = rt.reconstruct_point(full, cams_to_use=list(cams[i]))
    return out


def _px_err(uv, obs):
    """Row-wise pixel distance, BIT-IDENTICAL to the per-point
    ``np.linalg.norm(uv - obs)`` it replaced.

    ``np.linalg.norm`` of a 1-D vector is ``sqrt(np.dot(x, x))``, i.e. a BLAS
    ddot, which contracts ``x0*x0 + x1*x1`` into an FMA. So the obvious
    ``np.linalg.norm(d, axis=-1)`` / ``sqrt((d*d).sum(-1))`` disagree in the
    last ulp on ~8% of pairs (measured). The batched matmul below goes through
    the SAME BLAS kernel: 0 mismatches in 200k random pairs, and faster than
    ``np.linalg.norm`` besides."""
    d = np.ascontiguousarray(np.asarray(uv, float) - np.asarray(obs, float))
    return np.sqrt(np.matmul(d[..., None, :], d[..., :, None])[..., 0, 0])


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
    n_kp = len(kp3d_mm)
    uv_all = _reproject_all(rt, kp3d_mm)                  # (K, C, 2)
    xyz_ok = np.isfinite(kp3d_mm).all(-1)                 # (K,)
    out = {}
    for c, kp2d in kp2d_by_cam.items():
        vis = np.asarray(vis_by_cam.get(c, np.ones(n_kp, bool)), bool)
        obs = np.asarray(kp2d, float)
        uv = uv_all[..., c, :]
        good = (vis & xyz_ok & np.isfinite(obs).all(-1)
                & np.isfinite(uv).all(-1))
        if good.any():
            out[c] = float(np.median(_px_err(uv[good], obs[good])))
    return out


def loo_reproj(rt, kp2d_by_cam, vis_by_cam):
    """Leave-one-out reprojection error over all keypoints.

    For each kp: gather cams where it is visible; for each held-out visible
    cam with >=2 OTHER visible cams, triangulate from the others, reproject
    into the held-out cam, pixel error vs that cam's own observation.

    Batched over keypoints: every held-out solve for a kp with V visible cams
    uses V-1 others, so all kps sharing the same V share one (G*V, 2(V-1), 4)
    DLT stack and one stacked SVD -- at most 5 batched solves per frame
    instead of up to 350 scalar ones.
    """
    cams = sorted(kp2d_by_cam)
    if not cams:
        return {"per_kp": np.zeros(0), "median": float("nan"), "n": 0}
    n_kp = len(next(iter(kp2d_by_cam.values())))
    cam_ids = np.asarray(cams, int)
    per_kp = np.full(n_kp, np.nan)
    vis = np.stack([np.asarray(vis_by_cam.get(c, np.ones(n_kp, bool)), bool)
                    for c in cams], axis=1)               # (K, nc)
    obs = np.stack([np.asarray(kp2d_by_cam[c], float) for c in cams], axis=1)
    n_vis = vis.sum(axis=1)                               # (K,)
    all_errs = []
    for V in np.unique(n_vis[n_vis >= 3]):
        V = int(V)
        js = np.nonzero(n_vis == V)[0]                    # (G,)
        # visible cameras per kp, ascending -- `cams` is sorted, so this is the
        # same `vis_cams` list the scalar version built.
        sel = np.nonzero(vis[js])[1].reshape(len(js), V)  # (G, V) column idx
        cam_sel = cam_ids[sel]                            # (G, V) camera ids
        obs_sel = np.take_along_axis(obs[js], sel[..., None], axis=1)  # (G,V,2)
        errs = np.empty((len(js), V), float)
        for h in range(V):
            keep = [k for k in range(V) if k != h]        # `others`, in order
            X = _reconstruct_all(rt, obs_sel[:, keep], cam_sel[:, keep])
            proj = _reproject_all(rt, X)                  # (G, C, 2)
            held = cam_sel[:, h]
            errs[:, h] = _px_err(proj[np.arange(len(js)), held], obs_sel[:, h])
        per_kp[js] = errs.mean(axis=1)
        all_errs.append(errs.reshape(-1))
    all_errs = np.concatenate(all_errs) if all_errs else np.zeros(0)
    return {"per_kp": per_kp,
            "median": float(np.median(all_errs)) if all_errs.size else float("nan"),
            "n": int(all_errs.size)}


def mesh_mask_iou_report(rt, mesh_mm, masks_by_cam):
    """Hard+soft IoU of the projected posed mesh subset vs SAM masks, per cam.

    Was `silhouette_iou_report`, renamed 2026-09-01: it has nothing to do with
    the deleted silhouette POLISH (0bc36fe) -- it is a mesh-vs-mask overlap
    check that live QC runs on every bout, which is why mesh_iou.py survived
    the removal. The old name kept implying the polish was still in the
    pipeline. The qc.json key changed with it: `silhouette_iou` ->
    `mesh_mask_iou`, and scripts/benchmark/{metrics,select_bouts}.py accept
    EITHER, because the frozen 13-bout baseline's qc.json predates the rename.

    NOT A COVERAGE FRACTION. It splats projected VERTICES of a SPARSE mesh
    subset (fps_500), which mesh_iou.py's own docstring notes under-reports;
    `filled_tri_iou` -- which rasterises the projected FACES -- is the honest
    measure but needs subset faces this array does not carry. So the numbers
    run ~0.025 hard / ~0.15 soft and always have (phase-0 baseline recorded
    0.0287/0.0251). Treat it as a relative proxy between runs, never as
    "2.5% of the fly is covered", and do not gate on it.

    The mesh is reprojected ONCE for all cameras (it used to be reprojected
    per camera, per vertex, from Python: 500 x 7 scalar calls per frame).
    """
    from jarvis_jax.tracking.mesh_iou import hard_soft_iou_of_verts
    mesh_mm = np.asarray(mesh_mm, float)
    hard, soft = {}, {}
    if len(mesh_mm):
        uv_all = _reproject_all(rt, mesh_mm)              # (V, C, 2)
        for c, mask in masks_by_cam.items():
            if mask is None:
                continue
            hard[c], soft[c] = hard_soft_iou_of_verts(uv_all[..., c, :],
                                                      np.asarray(mask))
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
    uv_all = _reproject_all(rt, kp3d_mm)                  # (K, C, 2)
    xyz_ok = np.isfinite(kp3d_mm).all(-1)
    idx_out, err_out = [], []
    for c, kp2d in kp2d_by_cam.items():
        vis = np.asarray(vis_by_cam.get(c, np.ones(n_kp, bool)), bool)
        obs = np.asarray(kp2d, float)
        uv = uv_all[..., c, :]
        good = (vis & xyz_ok & np.isfinite(obs).all(-1)
                & np.isfinite(uv).all(-1))
        if good.any():
            idx_out.append(np.nonzero(good)[0])
            err_out.append(_px_err(uv[good], obs[good]))
    if not idx_out:
        return np.zeros(0, int), np.zeros(0, float)
    return (np.concatenate(idx_out).astype(int),
            np.concatenate(err_out).astype(float))


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


# ---------------------------------------------------------------------------
# The (frame, camera) metrics qc_report and per_frame_qc SHARE.
# ---------------------------------------------------------------------------

def frame_metrics(rt, *, kp3d_by_frame, mesh_by_frame, kp2d_by_frame,
                  vis_by_frame, masks_by_frame):
    """Per-(frame, camera) mesh-vs-mask IoU and keypoint reprojection error.

    ``qc_report`` (median across the bout, for qc.json) and
    ``qc_perframe.per_frame_qc`` (one row per frame, for the pseudo-label
    Gate A) computed the SAME two things independently, so a bout paid for
    ``mesh_mask_iou_report`` + ``per_camera_reproj_error`` twice -- the single
    largest cost in each stage. Compute this once and pass it to both as
    ``frame_metrics=``; both still compute it themselves when it is omitted,
    so every existing caller is unaffected.

    Returns ``{"cameras": [...], "hard_iou": (T,C), "soft_iou": (T,C),
    "reproj_px": (T,C), "mask_present": (T,C) bool}``; NaN marks a (frame,
    camera) the corresponding metric skipped (no mask / no usable keypoint).
    """
    T = len(kp3d_by_frame)
    cams = sorted({c for t in range(T) for c in kp2d_by_frame[t]}
                  | {c for t in range(T) for c in masks_by_frame[t]})
    col = {c: i for i, c in enumerate(cams)}
    hard = np.full((T, len(cams)), np.nan)
    soft = np.full((T, len(cams)), np.nan)
    reproj = np.full((T, len(cams)), np.nan)
    present = np.zeros((T, len(cams)), bool)
    for t in range(T):
        for c, m in masks_by_frame[t].items():
            present[t, col[c]] = m is not None
        sr = mesh_mask_iou_report(rt, mesh_by_frame[t], masks_by_frame[t])
        for c, v in sr["hard"].items():
            hard[t, col[c]] = v
        for c, v in sr["soft"].items():
            soft[t, col[c]] = v
        for c, v in per_camera_reproj_error(rt, kp3d_by_frame[t],
                                           kp2d_by_frame[t], vis_by_frame[t]).items():
            reproj[t, col[c]] = v
    return {"cameras": cams, "hard_iou": hard, "soft_iou": soft,
            "reproj_px": reproj, "mask_present": present}


# Private alias: `qc_report`/`per_frame_qc` take a `frame_metrics=` KWARG that
# shadows the function name inside their bodies.
_frame_metrics = frame_metrics


def _row_mean(a):
    """Per-row mean over the non-NaN entries; NaN for an all-NaN row.

    Equals ``np.mean(list(d.values()))`` over the present cameras: NaN entries
    contribute an exact 0.0 to ``nansum`` and are excluded from the count."""
    cnt = np.count_nonzero(~np.isnan(a), axis=1)
    with np.errstate(invalid="ignore"):
        out = np.where(cnt > 0, np.nansum(a, axis=1) / np.maximum(cnt, 1), np.nan)
    return out


def _row_median(a):
    """Per-row median over the non-NaN entries; NaN for an all-NaN row
    (``np.nanmedian`` without its all-NaN RuntimeWarning)."""
    out = np.full(a.shape[0], np.nan)
    rows = np.nonzero(np.count_nonzero(~np.isnan(a), axis=1) > 0)[0]
    for i in rows:
        r = a[i]
        out[i] = np.median(r[~np.isnan(r)])
    return out


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
              kp3d_measured_by_frame=None, kp_names=None, group_defs=None,
              frame_metrics=None):
    """Bundle per-camera reproj, LOO reproj, mesh-vs-mask IoU, and (additive)
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

    ``frame_metrics``: an already-computed ``frame_metrics(...)`` result for
    these same inputs, so a caller that also writes qc_perframe.npz does not
    pay for the shared (frame, camera) metrics twice. None (default) computes
    them here, unchanged.
    """
    T = len(kp3d_by_frame)
    fm = frame_metrics
    if fm is None:
        fm = _frame_metrics(
            rt, kp3d_by_frame=kp3d_by_frame, mesh_by_frame=mesh_by_frame,
            kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
            masks_by_frame=masks_by_frame)

    per_cam_all = fm["reproj_px"][~np.isnan(fm["reproj_px"])]
    iou_hard_all = _row_mean(fm["hard_iou"])
    iou_soft_all = _row_mean(fm["soft_iou"])
    iou_hard_all = iou_hard_all[np.isfinite(iou_hard_all)]
    iou_soft_all = iou_soft_all[np.isfinite(iou_soft_all)]

    loo_all = []
    for t in range(T):
        lo = loo_reproj(rt, kp2d_by_frame[t], vis_by_frame[t])
        if lo["n"]:
            loo_all.append(lo["median"])

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
        "mesh_mask_iou": {"hard_median": _agg(iou_hard_all),
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
