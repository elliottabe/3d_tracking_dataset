"""DLT triangulation of per-camera 2-D keypoints into 3-D mm keypoints + conf.

Geometric (no learned lift): per keypoint, use the cameras where the 2-D peak
confidence >= conf_thresh (>=2 required) and solve the DLT via
center3d.triangulate_dlt_batched. Keypoints with <2 confident views -> NaN
(down-weighted downstream by conf3d=0).
"""
from __future__ import annotations
import numpy as np
from jarvis_jax.geometry.center3d import triangulate_dlt_batched


def view_median_conf(conf):
    """conf (T,C,K) -> (T,C) median peak confidence across keypoints per view.

    The per-view summary the gate in `triangulate_keypoints` thresholds on.
    Median rather than mean so a handful of genuinely occluded keypoints cannot
    drag an otherwise-good view under the threshold.
    """
    conf = np.asarray(conf, np.float32)
    if conf.ndim != 3:
        raise ValueError(f"view_median_conf expects (T,C,K), got {conf.shape}")
    return np.median(conf, axis=2)


def _reproject_px(cam_mats, X):
    """X (M,3) world -> (M,C,2) pixels under the (C,4,3) matrices (p_h @ M)."""
    ph = np.concatenate([np.asarray(X, np.float64),
                         np.ones((len(X), 1), np.float64)], axis=1)   # (M,4)
    pr = np.einsum("mj,cjk->mck", ph, np.asarray(cam_mats, np.float64))
    return (pr[..., :2] / pr[..., 2:3]).astype(np.float32)


def _point_residuals(pts, val, cam_mats, idx):
    """Residuals of the all-valid-view DLT solve for the points in idx.

    Returns (len(idx), C) float32; invalid views hold -1. Non-finite solves
    leave the whole row at -1 (nothing droppable there).
    """
    from jarvis_jax.geometry.center3d import triangulate_dlt_batched
    C = val.shape[1]
    cams = np.broadcast_to(cam_mats[None], (len(idx), C, 4, 3))
    X = np.asarray(triangulate_dlt_batched(pts[idx], cams, val[idx]))
    finite = np.isfinite(X).all(1)
    resid = np.full((len(idx), C), -1.0, np.float32)
    if finite.any():
        uv = _reproject_px(cam_mats, X[finite])                    # (m,C,2)
        r = np.linalg.norm(uv - pts[idx][finite], axis=-1)         # (m,C)
        r[~val[idx][finite]] = -1.0
        resid[finite] = r
    return resid


# A point enters consensus only when some view is TRIGGER_FACTOR*thr_px
# inconsistent, and a consensus must keep >= MIN_CONSENSUS_VIEWS views to be
# applied. Measured on the frozen 13-bout benchmark (2026-08-14 notes.md):
# without the trigger margin, bouts whose 2D is diffusely noisy just above
# thr_px (hard wall/occlusion poses) get "fixed" on most frames and the
# winning view-subset churns frame to frame -- spike rate INCREASED up to 80%
# on such bouts. Real single-view swaps are 100-400 px, so a 3x margin
# separates them cleanly from diffuse noise. Likewise a 2-view consensus can
# lock onto one of two equally-sized clusters arbitrarily (female b8: +25%
# spikes); requiring 3 views keeps the solve overdetermined or leaves the
# point alone.
TRIGGER_FACTOR = 3.0
MIN_CONSENSUS_VIEWS = 3


def _reject_outlier_views(pts, val, cam_mats, thr_px):
    """Restrict each grossly-inconsistent point to its largest self-consistent
    view set.

    pts (N,C,2), val (N,C) bool -> new val. A point qualifies when its
    all-valid-view DLT solve leaves some view reprojecting more than
    TRIGGER_FACTOR * thr_px from its own 2D (an unambiguous swap, not diffuse
    noise -- see the constants above). For those points (and only those), run
    a deterministic consensus search: triangulate from EVERY valid view pair,
    count how many valid views reproject within thr_px of that two-view
    solution, and keep the largest consensus set (ties broken by lower mean
    inlier residual) provided it keeps >= MIN_CONSENSUS_VIEWS views. NOT
    greedy worst-view dropping -- with 2+ swapped views the dragged
    least-squares solution can hang its worst residual on a GOOD view, so
    greedy removal walks downhill discarding good views (measured: err 248 ->
    534 mm on a 2-of-6-outlier point). Points with < 3 valid views are never
    touched (two disagreeing rays cannot vote; dropping one trades a fixable
    point for a NaN).
    """
    from itertools import combinations

    from jarvis_jax.geometry.center3d import triangulate_dlt_batched
    val = np.asarray(val, bool)
    N, C = val.shape
    eligible = np.flatnonzero(val.sum(1) >= 3)
    if len(eligible) == 0:
        return val
    resid = _point_residuals(pts, val, cam_mats, eligible)
    idx = eligible[resid.max(1) > TRIGGER_FACTOR * thr_px]         # (n,) active
    if len(idx) == 0:
        return val
    p, v = pts[idx], val[idx]                                      # (n,C,2),(n,C)
    best_count = np.full(len(idx), -1, np.int64)
    best_meanr = np.full(len(idx), np.inf, np.float32)
    best_inl = v.copy()                          # fall back to "keep all"
    cams2 = np.broadcast_to(cam_mats[None], (len(idx), C, 4, 3))
    for i, j in combinations(range(C), 2):
        ok = v[:, i] & v[:, j]                                     # (n,)
        if not ok.any():
            continue
        pair_val = np.zeros_like(v)
        pair_val[:, i] = pair_val[:, j] = True
        X = np.asarray(triangulate_dlt_batched(p, cams2, pair_val))
        finite = np.isfinite(X).all(1) & ok
        if not finite.any():
            continue
        uv = _reproject_px(cam_mats, X[finite])
        r = np.linalg.norm(uv - p[finite], axis=-1)                # (m,C)
        inl = (r <= thr_px) & v[finite]
        cnt = inl.sum(1)
        meanr = np.where(cnt > 0, np.where(inl, r, 0).sum(1) / np.maximum(cnt, 1),
                         np.inf).astype(np.float32)
        rows = np.flatnonzero(finite)
        better = (cnt > best_count[rows]) | (
            (cnt == best_count[rows]) & (meanr < best_meanr[rows]))
        upd = rows[better]
        best_count[upd] = cnt[better]
        best_meanr[upd] = meanr[better]
        best_inl[upd] = inl[better]
    # A smaller consensus (2-view lock-on, or a degenerate pair) is refused:
    # keep the original views for those points (identical to the plain solve).
    keep = best_count >= MIN_CONSENSUS_VIEWS
    out = val.copy()
    out[idx[keep]] = best_inl[keep]
    return out


def triangulate_keypoints(kp2d, conf, cam_mats, *, conf_thresh: float = 0.3,
                          view_conf_thresh: float | None = None,
                          reproj_resid_px: float | None = None):
    """kp2d (T,C,K,2), conf (T,C,K), cam_mats (C,4,3) -> (kp3d (T,K,3), conf3d (T,K)).

    NOTE: kp2d must be FINITE even where conf < conf_thresh (invalid views are zeroed by
    validity masking inside triangulate_dlt_batched; a NaN/Inf pixel would poison that
    point's SVD via 0*NaN).

    `conf_thresh` gates individual keypoints. `view_conf_thresh` (None = off)
    additionally gates whole (frame, camera) views on their median keypoint
    confidence, dropping every keypoint of a view that falls below it.

    The per-view gate exists because the detector emits all K channels for
    whatever crop it is given, including one containing no fly, and reports
    high PER-KEYPOINT confidence while doing so -- so `conf_thresh` alone
    cannot stop a misplaced crop from entering the DLT and producing a
    confident, wrong 3D point. That is worse than dropping the view: a NaN is
    visibly missing downstream, a plausible wrong point is not. Measured on 273
    real views per class (scripts/calibrate_view_gate.py), per-view medians are
    0.963 with a fly present against 0.379 without, so a threshold of 0.6 keeps
    100% of genuine views while rejecting ~92% of empty ones. It is a strong
    mitigation, not a cure -- ~5% of empty crops still score above 0.9.

    `reproj_resid_px` (None = off) rejects OUTLIER views by consensus: any
    point whose all-view solve leaves a view reprojecting more than
    TRIGGER_FACTOR times this many px from its own 2D is re-solved from its
    largest self-consistent view set, with this value as the inlier band
    (see `_reject_outlier_views`). This is the gate confidence cannot
    provide: the dominant 3D jitter spikes are one camera whose 2D swapped to
    the wrong leg at conf 0.4-0.8 -- confidently wrong, invisible to both
    thresholds above, but ~100s of px inconsistent with the other views
    (measured: docs/benchmark/2026-08-14-jax-vs-jarvis-stability/notes.md in
    the parent repo -- 12 px touches 1.8% of view-points and removes most
    accel spikes on the Session6 clip). conf3d averages only surviving views.
    """
    kp2d = np.asarray(kp2d, np.float32); conf = np.asarray(conf, np.float32)
    cam_mats = np.asarray(cam_mats, np.float32)
    T, C, K, _ = kp2d.shape
    valid = conf >= conf_thresh                                   # (T,C,K)
    if view_conf_thresh is not None:
        view_ok = view_median_conf(conf) >= float(view_conf_thresh)   # (T,C)
        valid = valid & view_ok[:, :, None]
    kp3d = np.full((T, K, 3), np.nan, np.float32)
    conf3d = np.zeros((T, K), np.float32)
    # batch over (T*K) points; each has C views
    pts = kp2d.transpose(0, 2, 1, 3).reshape(T * K, C, 2)         # (TK,C,2)
    val = valid.transpose(0, 2, 1).reshape(T * K, C)             # (TK,C)
    if reproj_resid_px is not None:
        val = _reject_outlier_views(pts, val, cam_mats, float(reproj_resid_px))
    cams = np.broadcast_to(cam_mats[None], (T * K, C, 4, 3))
    nvalid = val.sum(1)                                          # (TK,)
    ok = nvalid >= 2
    if ok.any():
        X = np.asarray(triangulate_dlt_batched(pts[ok], cams[ok], val[ok]))   # (n,3)
        finite = np.isfinite(X).all(1)
        idx = np.where(ok)[0][finite]
        flat3d = kp3d.reshape(T * K, 3); flatc = conf3d.reshape(T * K)
        flat3d[idx] = X[finite]
        cmean = (conf.transpose(0, 2, 1).reshape(T * K, C) * val).sum(1) / np.maximum(nvalid, 1)
        flatc[idx] = cmean[idx]
        kp3d = flat3d.reshape(T, K, 3); conf3d = flatc.reshape(T, K)
    return kp3d, conf3d


def refine_from_seed(seed_kp3d, kp2d, conf, cam_mats, *, gate_px: float = 15.0,
                     conf_thresh: float = 0.3, min_views: int = 2):
    """Stage 3: continuous sub-voxel 3D, gated by a volumetric seed.

    The two lifters fail orthogonally. Plain DLT is continuous and sub-pixel
    but has no outlier rejection, so a single swapped view drags the solve
    (measured: 834 jitter spikes on a 921-frame clip). The volumetric net fuses
    every view and is robust (156 spikes) but quantizes at ~0.17 mm, which is
    larger than the tarsal segments that ARE the leg kinematics.

    This takes the robustness from the volume and the precision from the DLT:
    the seed says where the joint is and therefore which views are lying; the
    surviving views are triangulated continuously.

    Args:
        seed_kp3d: (T,K,3) stage-1/2 estimate.
        kp2d:      (T,C,K,2) detector 2D, must be FINITE everywhere.
        conf:      (T,C,K).
        cam_mats:  (C,4,3).
        gate_px:   a view is dropped if its 2D is further than this from the
                   seed's reprojection.
        min_views: below this, the seed is returned unchanged rather than
                   producing a confident wrong point.

    Returns:
        (kp3d (T,K,3), conf3d (T,K), n_views (T,K))
    """
    seed_kp3d = np.asarray(seed_kp3d, np.float64)
    kp2d = np.asarray(kp2d, np.float64)
    conf = np.asarray(conf, np.float64)
    T, C, K = conf.shape

    out = seed_kp3d.copy()
    out_conf = np.zeros((T, K), np.float32)
    n_used = np.zeros((T, K), np.int32)

    for t in range(T):
        for k in range(K):
            X0 = seed_kp3d[t, k]
            if not np.all(np.isfinite(X0)):
                continue
            proj = _reproject_px(cam_mats, X0[None, :])[0]  # (C,2)
            d = np.linalg.norm(kp2d[t, :, k, :] - proj, axis=-1)
            keep = (conf[t, :, k] >= conf_thresh) & (d <= gate_px)
            idx = np.nonzero(keep)[0]
            n_used[t, k] = idx.size
            if idx.size < min_views:
                continue
            A = []
            for c in idx:
                M = cam_mats[c]                              # (4,3)
                x, y = kp2d[t, c, k]
                A.append(x * M[:, 2] - M[:, 0])
                A.append(y * M[:, 2] - M[:, 1])
            _, _, Vt = np.linalg.svd(np.stack(A))
            h = Vt[-1]
            if abs(h[3]) < 1e-12:
                continue
            out[t, k] = h[:3] / h[3]
            out_conf[t, k] = float(np.mean(conf[t, idx, k]))
    return out.astype(np.float32), out_conf, n_used
