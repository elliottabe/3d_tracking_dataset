"""Courtship SAM3 mask adapter: bit-packed per-bout masks -> full-frame bool.

`sam3_masks.npz` stores `packed (A, C, T, H, ceil(W/8)) uint8` (np.packbits along
width), `valid (A,C,T) bool`, `shape [H,W]`, `centroids (A,C,T,2)`. Unpacking
gives the FULL-FRAME (H,W) mask in the calibration pixel frame (verified: fly0/fly1
land at the correct full-frame columns), so no centroid offset is needed.

CAMERA IDENTITY (bug fix): the C axis originally carried NO identifying
metadata, and a prior run wrote cameras in a SCRAMBLED order vs the
calibration/videos -- silently corrupting the whole courtship 3D pipeline
(triangulating stored per-camera mask centroids under the identity mapping
gave 46px reprojection residual; the correct permutation gave 6.8px). Two
independent defenses now guard against this:

  1. Name-based reordering (`load_bout_masks(..., expected_cameras=...)`):
     safe/exact -- only applies when the npz carries a `cameras` (C,) name
     array (written by `sam3_driver.run_sam3_masks` going forward).
  2. Calibration-based detection (`detect_camera_order`,
     `verify_mask_camera_order`): a brute-force geometric permutation search
     against the calibration, for legacy npz files with no `cameras` array.
     This is a GUARD, not an auto-fix -- see `verify_mask_camera_order` and
     `scripts/fix_mask_camera_order.py`.
"""
from __future__ import annotations

import itertools

import numpy as np


def unpack_one(packed, fly: int, cam: int, frame: int, W: int) -> np.ndarray:
    """(H,W) bool full-frame mask for one (fly,cam,frame)."""
    return np.unpackbits(packed[fly, cam, frame], axis=-1)[:, :W].astype(bool)


def _camera_permutation(npz_cameras, expected_cameras) -> np.ndarray:
    """Index array `perm` (len(expected_cameras),) such that indexing an
    array's camera axis with `arr[..., perm, ...]` (axis=1 for our (A,C,...)
    / (T,C,...) layouts) reorders it from `npz_cameras` order into
    `expected_cameras` order, i.e. `arr[:, perm][:, i]` is the data for
    `expected_cameras[i]`.

    Raises ValueError naming any `expected_cameras` entry absent from
    `npz_cameras`.
    """
    npz_cameras = [str(c) for c in list(npz_cameras)]
    index = {name: i for i, name in enumerate(npz_cameras)}
    missing = [c for c in expected_cameras if c not in index]
    if missing:
        raise ValueError(
            f"expected camera(s) {missing} not present in npz `cameras` "
            f"{npz_cameras}")
    return np.array([index[c] for c in expected_cameras], dtype=int)


def load_bout_masks(npz_path: str, fly: int, *, expected_cameras=None) -> dict:
    """Full-frame masks for one fly across the bout: masks (T,C,H,W) bool,
    valid (T,C) bool, centroids (T,C,2) float32.

    Camera-axis identity:
      - If the npz has a `cameras` (C,) name array AND `expected_cameras` is
        given, the C axis of masks/valid/centroids is reordered by NAME
        lookup (not position) into `expected_cameras` order, and the
        returned `cameras` is `list(expected_cameras)`. Raises ValueError if
        any `expected_cameras` name is absent from the npz's `cameras`.
      - If the npz has `cameras` but `expected_cameras` is None, no reorder
        happens; `cameras` in the output is the npz's stored order.
      - If the npz has NO `cameras` array (legacy files predating this
        fix), `expected_cameras` is ignored -- there is no name to reorder
        by -- and the stored C-axis order is returned UNCHANGED (no
        `cameras` key in the output). Use `detect_camera_order` /
        `verify_mask_camera_order` (calibration-based) to check such files.

    Backward compatible: existing callers passing no `expected_cameras` see
    the same masks/valid/T/C/H/W as before this fix (centroids and, when
    available, cameras are additive fields).
    """
    z = np.load(npz_path)
    packed = z["packed"]; H, W = int(z["shape"][0]), int(z["shape"][1])
    A, C, T = packed.shape[0], packed.shape[1], packed.shape[2]
    valid = np.asarray(z["valid"])[fly].transpose(1, 0)              # (T,C)
    centroids = np.asarray(z["centroids"], np.float32)[fly].transpose(1, 0, 2)  # (T,C,2)
    masks = np.zeros((T, C, H, W), bool)
    for c in range(C):
        for t in range(T):
            masks[t, c] = np.unpackbits(packed[fly, c, t], axis=-1)[:, :W].astype(bool)

    out_cameras = None
    if "cameras" in z.files:
        npz_cameras = [str(c) for c in np.asarray(z["cameras"]).tolist()]
        if expected_cameras is not None:
            perm = _camera_permutation(npz_cameras, expected_cameras)
            masks = masks[:, perm]
            valid = valid[:, perm]
            centroids = centroids[:, perm]
            out_cameras = list(expected_cameras)
        else:
            out_cameras = npz_cameras

    out = dict(masks=masks, valid=valid, centroids=centroids, T=T, C=C, H=H, W=W)
    if out_cameras is not None:
        out["cameras"] = out_cameras
    return out


# ---------------------------------------------------------------------------
# Calibration-based camera-order detection / guard.
# ---------------------------------------------------------------------------

def _mean_reproj_resid_for_perm(centroids, t, perm, rt) -> float:
    """Mean per-camera reprojection residual (px) for one frame `t` under
    the hypothesis that mask-index `i`'s centroid belongs to calibration
    camera `perm[i]`, for all i."""
    C = len(perm)
    pts2d = np.zeros((C, 2), dtype=np.float64)
    for i, cam in enumerate(perm):
        pts2d[cam] = centroids[t, i]
    X = rt.reconstruct_point(pts2d)
    repro = rt.reproject_point(X)
    return float(np.linalg.norm(repro - pts2d, axis=-1).mean())


def detect_camera_order(centroids, valid, rt, *, n_frames: int = 20) -> dict:
    """Brute-force geometric detection of a mask file's true camera order.

    Over up to `n_frames` frames where EVERY camera is valid, and for each
    permutation of the C cameras (mask-index i -> calibration-camera
    `perm[i]`), triangulates the assigned centroids (`rt.reconstruct_point`),
    reprojects (`rt.reproject_point`), and takes the median (over sampled
    frames) of the per-frame mean reprojection residual. Mirrors the
    approach that first uncovered the camera-scramble bug (identity mapping:
    46px; correct permutation: 6.8px, on real data).

    Parameters
    ----------
    centroids : (T, C, 2) array
        Per-camera centroid track for ONE fly (e.g. `load_bout_masks`'s
        `centroids`, or the npz's raw `centroids[fly]` transposed to
        (T,C,2)).
    valid : (T, C) bool
        Matching validity mask, same (T,C) axis order as `centroids`.
    rt : ReprojectionTool
        Calibration-order reprojection tool; `rt.num_cameras` must equal C.
    n_frames : int
        Max number of all-camera-valid frames sampled (median over these).

    Returns
    -------
    dict with:
      identity_resid : float -- median residual (px) under the identity
          mapping (mask-index i -> calibration camera i).
      best_perm : tuple[int, ...] -- length-C permutation; best_perm[i] is
          the calibration-camera index mask-index i's centroid should be
          assigned to for the LOWEST residual (may equal identity).
      best_resid : float -- the residual achieved by best_perm.

    Raises
    ------
    ValueError
      - if C (from `centroids`) != rt.num_cameras.
      - if C > 8 (brute-forcing C! permutations would hang).
      - if no frame has every camera valid.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    T, C = centroids.shape[0], centroids.shape[1]
    if C != rt.num_cameras:
        raise ValueError(
            f"detect_camera_order: centroids has C={C} cameras but "
            f"rt.num_cameras={rt.num_cameras}")
    if C > 8:
        raise ValueError(
            f"detect_camera_order: C={C} cameras is too many for a "
            f"brute-force permutation search ({C}! permutations) -- "
            f"refusing to hang.")

    all_valid_frames = np.where(valid.all(axis=1))[0]
    if all_valid_frames.size == 0:
        raise ValueError(
            "detect_camera_order: no frame has every camera valid -- "
            "cannot geometrically verify camera order for this bout.")
    frame_idx = all_valid_frames[:n_frames]

    identity = tuple(range(C))
    identity_resid = None
    best_perm, best_resid = None, np.inf
    for perm in itertools.permutations(range(C)):
        resids = [_mean_reproj_resid_for_perm(centroids, t, perm, rt) for t in frame_idx]
        med = float(np.median(resids))
        if perm == identity:
            identity_resid = med
        if med < best_resid:
            best_perm, best_resid = perm, med

    return dict(identity_resid=float(identity_resid), best_perm=best_perm,
               best_resid=float(best_resid))


def verify_mask_camera_order(centroids, valid, rt, *, max_resid: float = 20.0,
                             tol: float = 2.0, n_frames: int = 20) -> dict:
    """Calibration-based guard for a mask file's camera axis order.

    Calls `detect_camera_order`. If the identity mapping's residual is
    already acceptable (`identity_resid <= max_resid`) this returns cleanly
    (aligned). Otherwise, if some OTHER permutation is both clearly better
    (`identity_resid - best_resid > tol`) and itself acceptable
    (`best_resid <= max_resid`), this is strong geometric evidence of a
    camera-order scramble -- raise RuntimeError naming the detected
    permutation and both residuals. This function NEVER auto-remaps (a
    geometric false-positive could mangle correct masks); it only fails
    loud so a human/tool (`scripts/fix_mask_camera_order.py`) can act.

    Returns the `detect_camera_order` dict when it does not raise (either
    aligned, or the evidence is ambiguous -- identity is bad but no
    permutation is confidently better either, which likely indicates a
    different problem, e.g. bad triangulation geometry, not a camera-order
    scramble).
    """
    det = detect_camera_order(centroids, valid, rt, n_frames=n_frames)
    if det["identity_resid"] <= max_resid:
        return det
    if det["identity_resid"] - det["best_resid"] > tol and det["best_resid"] <= max_resid:
        raise RuntimeError(
            f"Mask camera order looks WRONG: identity-mapping reprojection "
            f"residual is {det['identity_resid']:.1f}px, but permutation "
            f"{det['best_perm']} (mask-index -> calibration-camera-index) "
            f"gives {det['best_resid']:.1f}px. This looks like a "
            f"camera-order scramble (the courtship SAM3 mask bug this guard "
            f"exists to catch). Regenerate this bout's sam3_masks.npz, or "
            f"run scripts/fix_mask_camera_order.py to correct it in place -- "
            f"refusing to proceed on a possibly mis-ordered mask set.")
    print(f"[verify_mask_camera_order] WARNING: identity residual "
         f"{det['identity_resid']:.1f}px exceeds max_resid={max_resid} but no "
         f"permutation is confidently better (best={det['best_resid']:.1f}px "
         f"via {det['best_perm']}) -- not a clear camera-order scramble; "
         f"proceeding, but this bout's masks/geometry may have other issues.")
    return det
