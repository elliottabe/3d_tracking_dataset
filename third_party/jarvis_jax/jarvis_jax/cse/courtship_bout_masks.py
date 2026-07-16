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


# ---------------------------------------------------------------------------
# Partial-visibility-tolerant detection + in-place AUTOFIX.
#
# `detect_camera_order` / `verify_mask_camera_order` above REQUIRE a frame
# where EVERY camera is valid and RAISE otherwise -- too strict for courtship
# bouts where a fly is never simultaneously visible in all cameras (a bout
# with no all-camera-valid frame hard-fails the whole task) -- and they only
# GUARD (never correct). The functions below (a) verify using frames where
# only a SUBSET of cameras is valid (triangulating from that subset via
# `rt.reconstruct_point(cams_to_use=...)`), and (b) auto-correct a
# CONFIDENTLY-scrambled legacy mask file IN PLACE (with a .bak), warning and
# proceeding on the sparse/ambiguous cases rather than failing the bout.
# ---------------------------------------------------------------------------

def _masked_resid_for_perm(centroids, valid_row, t, perm, rt):
    """Per-frame mean reprojection residual (px) under `perm`, using ONLY the
    cameras valid at frame `t`. Assigns mask-index i's centroid to calibration
    camera `perm[i]` for every i where `valid_row[i]`, triangulates from just
    those cameras, and averages the reprojection error over them. Returns
    (resid_px, n_cams_used); resid is np.nan when fewer than 2 cameras valid."""
    C = len(perm)
    pts2d = np.zeros((C, 2), dtype=np.float64)
    cams_used = []
    for i, cam in enumerate(perm):
        if valid_row[i]:
            pts2d[cam] = centroids[t, i]
            cams_used.append(cam)
    if len(cams_used) < 2:
        return float("nan"), len(cams_used)
    X = rt.reconstruct_point(pts2d, cams_to_use=cams_used)
    repro = rt.reproject_point(X)
    err = float(np.linalg.norm(repro[cams_used] - pts2d[cams_used], axis=-1).mean())
    return err, len(cams_used)


def detect_camera_order_robust(centroids, valid, rt, *, n_frames: int = 30,
                               min_valid_cams: int = 4,
                               early_accept_resid: float | None = None) -> dict:
    """Partial-visibility-tolerant sibling of `detect_camera_order`.

    Samples up to `n_frames` frames that have at least `min_valid_cams` valid
    cameras (falling back to the frames with the MOST valid cameras, provided
    that maximum is >= 3, when none reach the threshold). For every camera
    permutation, takes the median (over sampled frames) of the masked
    per-frame reprojection residual (`_masked_resid_for_perm`, which uses only
    each frame's valid cameras). Unlike `detect_camera_order`, this NEVER
    raises on sparse validity -- it reports `n_frames_used == 0` when the
    masks are too sparse to verify geometrically (fewer than 3 cameras valid
    in every frame).

    When `early_accept_resid` is given and the IDENTITY mapping's residual is
    already <= it, returns immediately with `best_perm == identity` WITHOUT
    the full C! permutation search -- the common aligned case (avoids ~C!
    triangulations per bout).

    Returns dict(identity_resid, best_perm, best_resid, n_frames_used,
    min_cams_used); identity_resid/best_resid are np.nan when
    n_frames_used == 0. Raises ValueError only for C != rt.num_cameras or
    C > 8 (mirroring `detect_camera_order`)."""
    centroids = np.asarray(centroids, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    T, C = centroids.shape[0], centroids.shape[1]
    if C != rt.num_cameras:
        raise ValueError(
            f"detect_camera_order_robust: centroids has C={C} cameras but "
            f"rt.num_cameras={rt.num_cameras}")
    if C > 8:
        raise ValueError(
            f"detect_camera_order_robust: C={C} cameras is too many for a "
            f"brute-force permutation search ({C}! permutations) -- refusing "
            f"to hang.")

    n_valid_per_frame = valid.sum(axis=1)
    order = np.argsort(-n_valid_per_frame)                 # most-valid frames first
    eligible = order[n_valid_per_frame[order] >= min_valid_cams]
    if eligible.size == 0:
        max_valid = int(n_valid_per_frame.max()) if T else 0
        if max_valid >= 3:                                 # still triangulable
            eligible = order[n_valid_per_frame[order] >= max_valid]
        else:
            return dict(identity_resid=float("nan"), best_perm=tuple(range(C)),
                        best_resid=float("nan"), n_frames_used=0,
                        min_cams_used=int(max_valid))
    frame_idx = eligible[:n_frames]
    identity = tuple(range(C))
    min_cams_used = int(n_valid_per_frame[frame_idx].min())

    # Fast path: if identity is already good enough, skip the C! search.
    if early_accept_resid is not None:
        id_resids = [r for r in (_masked_resid_for_perm(centroids, valid[t], t, identity, rt)[0]
                                 for t in frame_idx) if np.isfinite(r)]
        id_med = float(np.median(id_resids)) if id_resids else float("nan")
        if np.isfinite(id_med) and id_med <= early_accept_resid:
            return dict(identity_resid=id_med, best_perm=identity, best_resid=id_med,
                        n_frames_used=int(frame_idx.size), min_cams_used=min_cams_used)

    identity_resid, best_perm, best_resid = None, None, np.inf
    for perm in itertools.permutations(range(C)):
        resids = []
        for t in frame_idx:
            r, _ = _masked_resid_for_perm(centroids, valid[t], t, perm, rt)
            if np.isfinite(r):
                resids.append(r)
        if not resids:
            continue
        med = float(np.median(resids))
        if perm == identity:
            identity_resid = med
        if med < best_resid:
            best_perm, best_resid = perm, med

    return dict(
        identity_resid=float(identity_resid) if identity_resid is not None else float("nan"),
        best_perm=best_perm if best_perm is not None else identity,
        best_resid=float(best_resid) if np.isfinite(best_resid) else float("nan"),
        n_frames_used=int(frame_idx.size),
        min_cams_used=min_cams_used)


def _per_camera_identity_resid(centroids, valid, rt, *, n_frames: int = 40):
    """Median per-camera reprojection residual (px, length C) under the IDENTITY
    mapping, over up to `n_frames` all-camera-valid frames. Returns None when
    no all-valid frame exists. Used to NAME the camera(s) whose mask centroids
    are bad (a SAM3 per-camera tracking error), which is what a high overall
    residual almost always means now (see check_bout_camera_order)."""
    centroids = np.asarray(centroids, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    T, C = centroids.shape[0], centroids.shape[1]
    frames = np.where(valid.all(axis=1))[0][:n_frames]
    if frames.size == 0:
        return None
    per_cam = np.zeros((frames.size, C))
    for k, t in enumerate(frames):
        X = rt.reconstruct_point(centroids[t])
        repro = rt.reproject_point(X)
        per_cam[k] = np.linalg.norm(repro - centroids[t], axis=-1)
    return np.median(per_cam, axis=0)


def check_bout_camera_order(npz_path, fly, expected_cameras, rt, *,
                            max_resid: float = 20.0, n_frames: int = 30) -> dict:
    """NON-MUTATING QC check of a bout's mask camera-axis geometry. NEVER writes
    the file and NEVER raises on data problems (so one odd bout can't block a
    whole session) -- returns a status dict and logs a WARNING that NAMES the
    worst camera when mask-centroid reprojection is poor.

    Status is one of:
      'aligned'    -- centroids (in the order the pipeline uses them, after any
                      `load_bout_masks` name-based reorder) triangulate cleanly.
      'suspect'    -- identity reprojection is poor. In the CURRENT pipeline the
                      SAM3 write path is order-preserving and self-labelling
                      (masks packed by cam_idx + a `cameras` array in the same
                      order), so this is almost always a per-camera SAM3
                      TRACKING error -- a bad mask centroid in one/few cameras,
                      NOT a camera-axis scramble. The warning names the worst
                      camera. (A genuine legacy camera-order scramble would also
                      land here; correct it deliberately with
                      scripts/fix_mask_camera_order.py only after confirming.)
      'unverified' -- no all-camera-valid frame to verify against; proceeds.

    Why non-mutating: mask CENTROIDS are coarse, and a single mis-tracked camera
    makes a brute-force permutation search find a spuriously-lower-residual
    ordering -- so AUTO-REMAPPING on this signal corrupts correctly-ordered
    masks (observed on real data: a bout with one bad camera at 129px was
    "recovered" to a wrong permutation). Camera identity is trusted from the
    self-labelling `cameras` array `sam3_driver` writes; this check only
    surfaces bad geometry for QC and downstream robust handling."""
    with np.load(npz_path) as z:
        npz_cameras = ([str(c) for c in np.asarray(z["cameras"]).tolist()]
                       if "cameras" in z.files else None)
        valid = np.asarray(z["valid"])
        centroids = np.asarray(z["centroids"])

    A, C = valid.shape[0], valid.shape[1]
    if not (0 <= fly < A):
        raise ValueError(f"{npz_path}: fly={fly} out of range (A={A})")

    # `perm_name[i]` = stored index `load_bout_masks(expected_cameras=)` places
    # at output index i (identity for legacy files with no `cameras` array), so
    # we check the centroids AS THE PIPELINE WILL USE THEM.
    perm_name = (_camera_permutation(npz_cameras, expected_cameras)
                 if npz_cameras is not None else np.arange(C, dtype=int))
    centroids_pv = centroids[fly].transpose(1, 0, 2)[:, perm_name]   # (T,C,2)
    valid_pv = valid[fly].transpose(1, 0)[:, perm_name]              # (T,C)

    per_cam = _per_camera_identity_resid(centroids_pv, valid_pv, rt, n_frames=max(n_frames, 40))
    if per_cam is None:
        return dict(status="unverified", identity_resid=float("nan"),
                    worst_cam=None, npz=npz_path)

    overall = float(np.mean(per_cam))
    worst = int(np.argmax(per_cam))
    if overall <= max_resid:
        return dict(status="aligned", identity_resid=overall,
                    worst_cam=None, npz=npz_path)

    worst_name = (list(rt.cameras)[worst] if worst < len(list(rt.cameras))
                  else f"cam{worst}")
    print(f"[camera-order] WARNING: {npz_path} fly{fly}: mask-centroid "
          f"reprojection is poor (mean {overall:.1f}px). Worst camera: "
          f"cam{worst} ({worst_name}) at {per_cam[worst]:.1f}px vs "
          f"{np.median(per_cam):.1f}px median -- almost certainly a per-camera "
          f"SAM3 tracking error in this bout, NOT a camera scramble (the write "
          f"path is order-preserving + self-labelling). Proceeding on the "
          f"`cameras` label untouched; the bad camera degrades this bout's "
          f"silhouette/bridge only. If a TRUE camera-order scramble is ever "
          f"confirmed, correct it with scripts/fix_mask_camera_order.py.")
    return dict(status="suspect", identity_resid=overall, worst_cam=worst,
                worst_cam_resid=float(per_cam[worst]), npz=npz_path)
