"""Coarse localiser: CenterDetect peaks -> 3D fly centres -> window plans.

Spec: `docs/specs/2026-09-04-mvq-maskfree-frontend-design.md` §4.1. This
module is pure geometry (numpy) plus a thin `CenterDetect` inference
wrapper; it has no notion of bouts, frames-over-time, or the mvq lifter --
those live in `coarse_track.py`, which calls `CenterDetector.peaks`,
`lift_peaks_to_centres`, `cluster_centres` and `plan_windows` per sampled
frame.

Pipeline, per frame:

  1. `CenterDetector.peaks(frames)` -- one CenterDetect forward pass per
     camera, top-2 peaks each (`extract_top_k_peaks`), rescaled to
     full-image pixels (`peaks_to_full_image`). CenterDetect has NO
     cross-camera identity: camera A's "peak 0" and camera B's "peak 0" are
     not guaranteed to be the same animal, and a camera can be missing a
     peak (dim second blob below `min_score`) or have one moved by mask/BG
     noise. `lift_peaks_to_centres` must not assume peak *index* is animal
     identity (CLAUDE.md's keypoint/camera-order warning, restated one level
     up: never assume two independently-decoded per-camera indices agree).

  2. `lift_peaks_to_centres` resolves that with a greedy RANSAC-flavoured
     search: seed a 3D candidate from every (camera pair, peak pair)
     combination that is geometrically possible (triangulate_dlt_batched),
     reproject each candidate to every camera, and count INLIERS -- cameras
     whose reprojection lands within `max_resid_px` of *either* of that
     camera's two peaks (so a swapped 0/1 order in one camera is invisible
     to this check: it just picks whichever slot is closer). The candidate
     with the most inliers wins (ties broken by summed peak score), gets
     refined by a full DLT over its own inlier set, and those specific
     peaks are consumed (marked NaN) so the next iteration finds a
     DIFFERENT animal instead of re-discovering the same one. Stops when
     the best remaining candidate has fewer than `min_views` inliers, or
     `max_animals` centres have been found.

  3. `cluster_centres` / `plan_windows` merge centres that are the same
     physical animal seen twice (e.g. a spurious extra candidate) or that
     are close enough to share one crop window, respectively -- both in
     WORLD units (0.1 mm; `min_sep_units=15.0` == 1.5 mm, `merge_dist_units
     =30.0` == 3 mm, per the design doc). Padding is always NaN for centres
     (so a fixed-shape (max_animals, 3) / (A, 3) array survives however many
     animals were actually found), never a variable-length list.

Every distance/centre in this module is 0.1 mm world units and every pixel
is FULL-IMAGE px (not heatmap px, not crop px) unless named otherwise --
see CLAUDE.md on labelling with real names and units.

CAMERA-ORDER CONTRACT (CLAUDE.md's camera-order trap, restated here because
this is the one place a caller's per-camera array first meets a per-camera
calibration array): every per-camera axis in this module -- `peaks[c]`,
`scores[c]` and `cam_mats[c]` -- MUST be the same camera `c`, in the
CANONICAL order (`cfg.recording.cameras`, == the calibration glob order,
== `ReprojectionTool`'s own key order). Nothing here re-sorts or matches by
name; a caller holding an array in a DIFFERENT camera order (e.g. a mask
npz's own stored `cameras` array) must permute it into canonical order
FIRST, exactly as `tracking.lift_mvq.MVQRunner` requires for its `cameras`
argument. `lift_peaks_to_centres` checks the one thing it can verify
cheaply -- that `peaks` and `cam_mats` agree on camera COUNT -- and raises
`ValueError` on a mismatch; it cannot detect a same-length but
wrongly-ordered camera axis, which is why the contract is stated here too.
"""
from __future__ import annotations

import numpy as np

from jarvis_jax.eval.centerdetect_decode import extract_top_k_peaks, peaks_to_full_image
from jarvis_jax.geometry.center3d import triangulate_dlt_batched

CENTERDETECT_INPUT_SIZE = 320
CENTERDETECT_HEATMAP_SIZE = 160


def _restore_centerdetect(ckpt_dir):
    """Restore a CenterDetect `EfficientTrack(num_joints=1, in_channels=3,
    model_size="medium")` checkpoint. Exact pattern of
    `jarvis_jax/scripts/viz_centerdetect_fp.py::_restore`: eval_shape ->
    replicated-sharding ShapeDtypeStruct targets -> Orbax StandardCheckpointer
    restore -> nnx.merge -> `.eval()`. Kept import-lazy (jax/orbax/flax) so
    this module imports on a CPU-only test box with no checkpoint present.
    """
    import jax
    import orbax.checkpoint as ocp
    from flax import nnx
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jarvis_jax.models.efficienttrack import EfficientTrack

    ctor = lambda: EfficientTrack(num_joints=1, in_channels=3,
                                  model_size="medium", rngs=nnx.Rngs(0))
    gdef, abstract = nnx.split(nnx.eval_shape(ctor))
    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())
    target = jax.tree_util.tree_map(
        lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=repl), abstract)
    model = nnx.merge(gdef, ocp.StandardCheckpointer().restore(ckpt_dir, target=target))
    model.eval()
    return model


def _peaks_from_heatmap(hm, img_w, img_h, min_score=0.2):
    """Pure post-processing: `(C,160,160)` or `(C,160,160,1)` CenterDetect
    heatmap -> full-image-px peaks. Isolated from `CenterDetector.peaks` so
    it can be unit-tested on a hand-built heatmap with no checkpoint.

    Args:
        hm: (C, H_hm, W_hm[, 1]) heatmap, single channel (num_joints=1).
        img_w, img_h: ORIGINAL full-image size (before the 320x320 squash);
            `peaks_to_full_image` applies the anisotropic x/y scale this
            implies, matching the training-side resize.
        min_score: a peak with score below this is reported as NaN (both
            coordinates) rather than a false-confident location -- so a
            camera with only one real animal doesn't hand a phantom second
            centre to `lift_peaks_to_centres`.

    Returns:
        peaks: (C, 2, 2) float32 full-image [x, y] px, NaN where score < min_score.
        scores: (C, 2) float32 RAW heatmap confidence, always returned
            unthresholded/un-NaN'd -- a peak being unusable is signalled by
            its COORDINATES being NaN, not by its score. Callers (e.g.
            `lift_peaks_to_centres`) gate on the peak, not the score.
    """
    hm = np.asarray(hm)
    heatmap_size = hm.shape[1] if hm.ndim >= 3 else hm.shape[0]
    peaks_hm, conf = extract_top_k_peaks(hm, k=2, suppression_radius=15)
    full = peaks_to_full_image(peaks_hm, heatmap_size, img_w, img_h).astype(np.float32)
    conf = conf.astype(np.float32)
    below = conf < min_score
    full[below] = np.nan
    return full, conf


def _centerdetect_preprocess(frame):
    """One full RGB frame -> the 320x320 CenterDetect model input.

    Resizes with PIL `Image.BILINEAR` -- NOT `cv2.resize(..., INTER_LINEAR)`
    -- because that is what `jarvis_jax/data/v5_centerdetect.py` uses at
    TRAINING time. PIL's resize is antialiased (a proper low-pass box/tent
    filter before subsampling); cv2's `INTER_LINEAR` point-samples with no
    antialiasing. At this dataset's ~6x horizontal squash (1936 -> 320) that
    difference is not cosmetic: measured mean |delta| of 11/255 between the
    two resized images -- exactly the confident, self-consistent,
    wrong-for-a-reason-no-metric-catches failure mode CLAUDE.md warns about,
    here as a train/inference preprocessing mismatch rather than an index
    bug.

    Args:
        frame: (H, W, 3) uint8 RGB.
    Returns:
        (320, 320, 3) float32, ImageNet-normalised.
    """
    from PIL import Image
    from jarvis_jax.data.device import IMAGENET_MEAN_J, IMAGENET_STD_J

    resized = np.asarray(Image.fromarray(frame).resize(
        (CENTERDETECT_INPUT_SIZE, CENTERDETECT_INPUT_SIZE), Image.BILINEAR))
    mean = np.asarray(IMAGENET_MEAN_J, np.float32)
    std = np.asarray(IMAGENET_STD_J, np.float32)
    return ((resized.astype(np.float32) / 255.0 - mean) / std).astype(np.float32)


class CenterDetector:
    """Thin CenterDetect inference wrapper: full frames -> top-2 peaks per
    camera in full-image px. GPU/checkpoint-heavy -- never exercised in unit
    tests (those hit `_peaks_from_heatmap` and `_centerdetect_preprocess`
    directly, on synthetic heatmaps/frames).
    """

    def __init__(self, ckpt_dir, *, min_score: float = 0.2):
        self.min_score = float(min_score)
        self._model = _restore_centerdetect(ckpt_dir)

    def peaks(self, frames):
        """frames: (C, H, W, 3) uint8 RGB full frames -- camera `c` here
        must be the SAME camera as `cam_mats[c]` later passed to
        `lift_peaks_to_centres` (see module docstring's camera-order
        contract); this method itself has no camera identity of its own to
        check that against, so the contract is the caller's to keep.

        Returns:
            peaks: (C, 2, 2) float32 full-image px, NaN where score < min_score.
            scores: (C, 2) float32 raw confidence (unthresholded; see
                `_peaks_from_heatmap`).
        """
        import jax.numpy as jnp

        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames must be (C,H,W,3), got {frames.shape}")
        c, img_h, img_w = frames.shape[0], frames.shape[1], frames.shape[2]

        x = jnp.asarray(np.stack([_centerdetect_preprocess(frames[i]) for i in range(c)]))
        hm = np.asarray(self._model(x, use_running_average=True))  # (C,160,160,1)
        return _peaks_from_heatmap(hm, img_w, img_h, self.min_score)


# ---------------------------------------------------------------------------
# Greedy multi-view lift
# ---------------------------------------------------------------------------

def _seed_candidates(peaks, cam_mats):
    """All (camera pair x peak pair) triangulation candidates with both
    peaks non-NaN. Returns (M, 3) candidate 3D points, or an (0, 3) array if
    fewer than 2 cameras have any usable peak."""
    n_cams = peaks.shape[0]
    valid_peak = ~np.isnan(peaks).any(axis=-1)  # (C,2)

    idx_i, idx_j, slot_i, slot_j = [], [], [], []
    for i in range(n_cams):
        for j in range(i + 1, n_cams):
            for pi in range(2):
                if not valid_peak[i, pi]:
                    continue
                for pj in range(2):
                    if not valid_peak[j, pj]:
                        continue
                    idx_i.append(i); idx_j.append(j)
                    slot_i.append(pi); slot_j.append(pj)

    m = len(idx_i)
    if m == 0:
        return np.zeros((0, 3), np.float32)

    points2d = np.zeros((m, n_cams, 2), np.float32)
    valid = np.zeros((m, n_cams), bool)
    for k in range(m):
        points2d[k, idx_i[k]] = peaks[idx_i[k], slot_i[k]]
        points2d[k, idx_j[k]] = peaks[idx_j[k], slot_j[k]]
        valid[k, idx_i[k]] = True
        valid[k, idx_j[k]] = True
    cam_mats_b = np.broadcast_to(cam_mats, (m,) + cam_mats.shape)
    return np.asarray(triangulate_dlt_batched(points2d, cam_mats_b, valid))


def _project_batch(centres3d, cam_mats):
    """(N,3) world points, (C,4,3) cam_mats -> (N,C,2) full-image px.

    The batched/vectorised equivalent of
    `jarvis_jax.geometry.center3d.project_center_to_cameras` (same
    `p_h @ M`, perspective-divide convention) over N candidate points at
    once, so `_score_candidates` needs one einsum instead of a Python loop
    over candidates.
    """
    M = np.asarray(cam_mats, np.float64)
    ph = np.concatenate(
        [np.asarray(centres3d, np.float64), np.ones((centres3d.shape[0], 1))], axis=1)
    proj = np.einsum("nd,cdk->nck", ph, M)
    return proj[..., :2] / proj[..., 2:3]


def _score_candidates(candidates, peaks, scores, cam_mats, max_resid_px):
    """For every candidate, count inlier cameras (nearest of that camera's
    two peaks within `max_resid_px`) and the summed score of the matched
    peaks. Returns (n_inliers (M,), summed_score (M,), best_slot (M,C),
    inlier (M,C) bool)."""
    proj = _project_batch(candidates, cam_mats)                      # (M,C,2)
    diff = peaks[None, :, :, :] - proj[:, :, None, :]                 # (M,C,2,2)
    dist = np.linalg.norm(diff, axis=-1)                              # (M,C,2)
    dist_filled = np.where(np.isnan(dist), np.inf, dist)
    best_slot = np.argmin(dist_filled, axis=-1)                       # (M,C)
    best_dist = np.take_along_axis(dist_filled, best_slot[..., None], axis=-1)[..., 0]
    inlier = best_dist <= max_resid_px                                # (M,C)
    n_inliers = inlier.sum(axis=-1)                                   # (M,)

    m, n_cams = best_slot.shape
    scores_b = np.broadcast_to(scores, (m, n_cams, 2))
    matched_score = np.take_along_axis(scores_b, best_slot[..., None], axis=-1)[..., 0]
    matched_score = np.where(np.isnan(matched_score), 0.0, matched_score)
    summed_score = np.where(inlier, matched_score, 0.0).sum(axis=-1)  # (M,)
    return n_inliers, summed_score, best_slot, inlier


def lift_peaks_to_centres(peaks, scores, cam_mats, *, min_views=3,
                          max_resid_px=25.0, max_animals=2):
    """CenterDetect peaks (no cross-camera identity) -> up to `max_animals`
    triangulated 3D centres, greedily, most-consistent animal first.

    Args:
        peaks: (C, 2, 2) float32 full-image px, NaN for a missing peak.
            `peaks[c]` MUST be the same camera as `cam_mats[c]` -- see the
            module docstring's camera-order contract.
        scores: (C, 2) float32 RAW per-peak confidence (as returned by
            `_peaks_from_heatmap`/`CenterDetector.peaks`, unthresholded). A
            peak being unusable is driven entirely by ITS COORDINATES being
            NaN in `peaks` (that is what gates it out of `_seed_candidates`
            and drives its reprojection distance to NaN -> inf -> not an
            inlier in `_score_candidates`) -- `scores` need not be NaN'd in
            lockstep; a NaN score is tolerated (treated as 0 when summing)
            but never required.
        cam_mats: (C, 4, 3) DLT projection matrices (`ReprojectionTool.camera_matrices`).
        min_views: stop once the best remaining candidate has fewer inlier
            cameras than this (default 3 -- triangulation needs >=2, this
            requires one spare to reject a single bad view).
        max_resid_px: reprojection distance below which a camera counts as
            an inlier for a candidate centre.
        max_animals: stop after finding this many centres.

    Returns:
        centres: (max_animals, 3) float32, NaN-padded.
        n_views: (max_animals,) int32, 0-padded.
        score: (max_animals,) float32, 0-padded (summed inlier peak score).

    Raises:
        ValueError: if `peaks` and `cam_mats` disagree on camera count --
            the one camera-order mismatch this function can detect (it
            cannot tell a same-length but wrongly-ORDERED camera axis; that
            is on the caller, per the module docstring).
    """
    peaks = np.array(peaks, dtype=np.float32, copy=True)
    scores = np.array(scores, dtype=np.float32, copy=True)
    cam_mats = np.asarray(cam_mats, np.float32)
    if peaks.shape[0] != cam_mats.shape[0]:
        raise ValueError(
            f"peaks camera axis ({peaks.shape[0]}) must match cam_mats "
            f"camera axis ({cam_mats.shape[0]}) -- peaks[c]/scores[c] and "
            f"cam_mats[c] must be the SAME camera, in canonical order "
            f"(see module docstring's camera-order contract).")
    n_cams = peaks.shape[0]

    out_centres = np.full((max_animals, 3), np.nan, np.float32)
    out_views = np.zeros((max_animals,), np.int32)
    out_score = np.zeros((max_animals,), np.float32)

    for a in range(max_animals):
        candidates = _seed_candidates(peaks, cam_mats)
        if candidates.shape[0] == 0:
            break
        n_inliers, summed_score, best_slot, inlier = _score_candidates(
            candidates, peaks, scores, cam_mats, max_resid_px)
        best = np.lexsort((-summed_score, -n_inliers))[0]
        if n_inliers[best] < min_views:
            break

        inlier_cams = np.flatnonzero(inlier[best])
        refine_points = np.zeros((n_cams, 2), np.float32)
        refine_valid = np.zeros((n_cams,), bool)
        for c_idx in inlier_cams:
            slot = best_slot[best, c_idx]
            refine_points[c_idx] = peaks[c_idx, slot]
            refine_valid[c_idx] = True
        refined = np.asarray(triangulate_dlt_batched(
            refine_points[None], cam_mats[None], refine_valid[None]))[0]

        out_centres[a] = refined
        out_views[a] = int(n_inliers[best])
        out_score[a] = float(summed_score[best])

        for c_idx in inlier_cams:
            peaks[c_idx, best_slot[best, c_idx]] = np.nan
            scores[c_idx, best_slot[best, c_idx]] = np.nan

    return out_centres, out_views, out_score


# ---------------------------------------------------------------------------
# Clustering / window planning
# ---------------------------------------------------------------------------

def cluster_centres(centres, *, min_sep_units=15.0):
    """Merge centres closer than `min_sep_units` (default 15.0 == 1.5 mm in
    0.1 mm world units) by repeatedly averaging the closest pair below
    threshold. NaN-padded back to the input length so the shape never
    depends on how many merges happened.

    Args:
        centres: (N, 3) float32, may already contain NaN rows (ignored).
    Returns:
        (N, 3) float32, NaN-padded.
    """
    centres = np.asarray(centres, np.float32)
    n = centres.shape[0]
    pts = [centres[i].copy() for i in range(n) if not np.isnan(centres[i]).any()]

    merged = True
    while merged and len(pts) > 1:
        merged = False
        best = None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if d < min_sep_units and (best is None or d < best[0]):
                    best = (d, i, j)
        if best is not None:
            _, i, j = best
            new_pt = (pts[i] + pts[j]) / 2.0
            pts = [p for k, p in enumerate(pts) if k not in (i, j)] + [new_pt]
            merged = True

    out = np.full((n, 3), np.nan, np.float32)
    for k, p in enumerate(pts):
        out[k] = p
    return out


def plan_windows(centres, *, merge_dist_units=30.0):
    """One crop window per centre, merging two centres within
    `merge_dist_units` (default 30.0 == 3 mm) into a single window at their
    midpoint (mean, for >2-way merges via transitive closure).

    Args:
        centres: (A, 3) float32, NaN rows are ignored (assigned window -1).
    Returns:
        window_centres: (W, 3) float32, one row per resulting window.
        assignment: (A,) int, window_centres row index for each input centre
            (-1 for a NaN input row).
    """
    centres = np.asarray(centres, np.float32)
    a = centres.shape[0]
    valid = ~np.isnan(centres).any(axis=1)
    idxs = np.flatnonzero(valid)

    parent = {int(i): int(i) for i in idxs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for ii in range(len(idxs)):
        for jj in range(ii + 1, len(idxs)):
            i, j = int(idxs[ii]), int(idxs[jj])
            if np.linalg.norm(centres[i] - centres[j]) <= merge_dist_units:
                union(i, j)

    assignment = np.full(a, -1, dtype=int)
    order = []
    root_to_widx = {}
    for i in idxs:
        i = int(i)
        r = find(i)
        if r not in root_to_widx:
            root_to_widx[r] = len(order)
            order.append(r)
        assignment[i] = root_to_widx[r]

    window_centres = np.zeros((len(order), 3), np.float32)
    for widx, r in enumerate(order):
        members = centres[[int(i) for i in idxs if find(int(i)) == r]]
        window_centres[widx] = members.mean(axis=0)

    return window_centres, assignment
