"""JAX/NumPy affine cross-camera identity linker.

Each synchronized frameset has up to ``n_flies`` COCO annotations PER camera.
Within a camera the anns have distinct ``id``s, but which ann in cam-i is the
same physical fly as which ann in cam-j is not given. This module recovers that
cross-camera identity by triangulation consistency: for a candidate assignment
of per-camera anns to fly slots, triangulate each fly's keypoints (affine DLT)
and reproject; the true assignment minimizes the multi-view reprojection
residual. Mirrors the ALGORITHM of JARVIS-HybridNet BoutMasks.assign_identities
(anchor enumeration + min-residual + nearest-reproj for short cameras), but is
written from scratch against the affine primitives -- the torch tracker is NOT
imported. Cameras are telecentric (affine): projection has no perspective divide.
"""
from __future__ import annotations

import itertools

import numpy as np


def _dlt_affine(cam_mats, cam_ids, pts2d) -> np.ndarray:
    """Affine DLT-SVD triangulation (same math as ReprojectionTool.reconstruct_point).

    Args:
        cam_mats: (n_cam, 3, 4) per-camera 3x4 DLT matrices.
        cam_ids: absolute camera indices to use (>=2).
        pts2d: (n_cam, 2) observations, row indexed by absolute camera id.
    Returns:
        (3,) triangulated point; zeros if <2 cameras.
    """
    cam_ids = list(cam_ids)
    if len(cam_ids) < 2:
        return np.zeros(3)
    A = np.zeros((2 * len(cam_ids), 4))
    for i, c in enumerate(cam_ids):
        P = np.asarray(cam_mats[c], float)      # (3,4)
        uv = np.asarray(pts2d[c], float)        # (2,)
        A[2 * i:2 * i + 2] = uv.reshape(2, 1) * P[2].reshape(1, 4) - P[0:2]
    _, _, Vh = np.linalg.svd(A)
    Xh = Vh[-1]
    return (Xh / Xh[3])[:3]


def _project_affine_point(P, X) -> np.ndarray:
    """uv = P[:2,:3] @ X + P[:2,3] (affine; no perspective divide)."""
    P = np.asarray(P, float); X = np.asarray(X, float)
    return X @ P[:2, :3].T + P[:2, 3]


def score_assignment(kp2d_per_cam, cams_present, cam_mats) -> float:
    """Mean multi-view reprojection residual (px) for ONE fly's per-camera kps.

    Args:
        kp2d_per_cam: (n_cam, K, 3) coco [x, y, v]; v>0 == present.
        cams_present: absolute camera ids that carry this fly's detection.
        cam_mats: (n_cam, 3, 4) affine DLT matrices.
    Returns:
        Mean per-(keypoint, present-camera) pixel residual over triangulable
        keypoints; float('inf') if none triangulate.
    """
    kp2d_per_cam = np.asarray(kp2d_per_cam, float)
    cams_present = list(cams_present)
    K = kp2d_per_cam.shape[1]
    resid = []
    for j in range(K):
        cams_j = [c for c in cams_present if kp2d_per_cam[c, j, 2] > 0]
        if len(cams_j) < 2:
            continue
        pts = np.zeros((kp2d_per_cam.shape[0], 2))
        for c in cams_j:
            pts[c] = kp2d_per_cam[c, j, :2]
        X = _dlt_affine(cam_mats, cams_j, pts)
        for c in cams_j:
            uv = _project_affine_point(cam_mats[c], X)
            resid.append(float(np.linalg.norm(uv - kp2d_per_cam[c, j, :2])))
    return float(np.mean(resid)) if resid else float("inf")


def _ann_kp(ann) -> np.ndarray:
    """(K, 3) coco [x, y, v] from a COCO annotation dict."""
    return np.asarray(ann["keypoints"], float).reshape(-1, 3)


def _fly_center2d(kp) -> np.ndarray:
    """Mean of visible (v>0) 2-D keypoints; NaN if none visible."""
    vis = kp[:, 2] > 0
    return kp[vis, :2].mean(0) if vis.any() else np.full(2, np.nan)


def link_frameset(anns_by_cam, cam_mats, *, n_flies=2, ref_cam=None,
                  residual_gate_px=40.0):
    """Cross-camera identity for one frameset. See module docstring / Interfaces."""
    n_cam = np.asarray(cam_mats).shape[0]
    K = _ann_kp(next(iter(anns_by_cam.values()))[0]).shape[0]

    full_cams = [c for c in anns_by_cam if len(anns_by_cam[c]) >= n_flies]
    if not full_cams:
        return {fid: {} for fid in range(n_flies)}
    if ref_cam is None or ref_cam not in full_cams:
        ref_cam = full_cams[0]

    # Reference camera fixes fly slots: ref ann index i -> fly i.
    ref_anns = anns_by_cam[ref_cam][:n_flies]

    # For each other full camera, choose the permutation of its first n_flies
    # anns onto fly slots that minimizes total residual across (ref, cam).
    perms = list(itertools.permutations(range(n_flies)))
    # assign[fly][cam] = local ann index chosen for that fly at that camera.
    assign = {fid: {ref_cam: fid} for fid in range(n_flies)}
    for cam in full_cams:
        if cam == ref_cam:
            continue
        cam_anns = anns_by_cam[cam][:n_flies]
        best_perm, best_total = None, float("inf")
        for perm in perms:
            total = 0.0
            for fly in range(n_flies):
                obs = np.zeros((n_cam, K, 3))
                obs[ref_cam] = _ann_kp(ref_anns[fly])
                obs[cam] = _ann_kp(cam_anns[perm[fly]])
                total += score_assignment(obs, [ref_cam, cam], cam_mats)
            if total < best_total:
                best_total, best_perm = total, perm
        for fly in range(n_flies):
            assign[fly][cam] = best_perm[fly]

    # Gate: mean residual of the resolved full-camera assignment.
    gate_resid = []
    for fly in range(n_flies):
        obs = np.zeros((n_cam, K, 3))
        cams_fly = list(assign[fly])
        for cam in cams_fly:
            obs[cam] = _ann_kp(anns_by_cam[cam][assign[fly][cam]])
        r = score_assignment(obs, cams_fly, cam_mats)
        if np.isfinite(r):
            gate_resid.append(r)
    if not gate_resid or float(np.mean(gate_resid)) > residual_gate_px:
        return {fid: {} for fid in range(n_flies)}

    # Triangulate each fly's center from its resolved full cameras, to place
    # the short cameras (fewer than n_flies anns) by nearest reprojection.
    centers = {}
    for fly in range(n_flies):
        cams_fly = list(assign[fly])
        pts = np.zeros((n_cam, 2))
        used = []
        for cam in cams_fly:
            ctr = _fly_center2d(_ann_kp(anns_by_cam[cam][assign[fly][cam]]))
            if np.all(np.isfinite(ctr)):
                pts[cam] = ctr; used.append(cam)
        centers[fly] = _dlt_affine(cam_mats, used, pts) if len(used) >= 2 else None

    for cam in anns_by_cam:
        if cam in full_cams:
            continue
        for local_i, ann in enumerate(anns_by_cam[cam]):
            ctr = _fly_center2d(_ann_kp(ann))
            if not np.all(np.isfinite(ctr)):
                continue
            dists = []
            for fly in range(n_flies):
                if centers[fly] is None:
                    dists.append(np.inf); continue
                uv = _project_affine_point(cam_mats[cam], centers[fly])
                dists.append(float(np.linalg.norm(uv - ctr)))
            assign[int(np.argmin(dists))][cam] = local_i

    # Convert local ann indices -> ann ids.
    out = {}
    for fly in range(n_flies):
        out[fly] = {cam: int(anns_by_cam[cam][li]["id"]) for cam, li in assign[fly].items()}
    return out
