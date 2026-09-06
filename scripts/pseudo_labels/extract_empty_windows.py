"""Empty-window negatives for the existence head (spec 2026-09-05 §3.5, Task 5).

Run (real recordings, controller ruling 2026-09-05). The edge stratum runs
CenterDetect, so this needs a GPU node and the usual jax preamble
(`module load cuda/12.9.1; export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6;
unset LD_LIBRARY_PATH JAX_PLATFORMS`); `--edge-frac 0` falls back to a
CPU-only, interior-only run:

    PYTHONPATH=third_party/jarvis_jax:. python scripts/pseudo_labels/extract_empty_windows.py \\
        --tracks 2025_10_20_13_20_04=/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/coarse_mvq_p3b/coarse_tracks.npz \\
        --lift-root 2026_04_02_12_11_50=/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session1/2026_04_02_12_11_50/pose_mvq_p3b \\
        ... (one --lift-root per remaining recording) \\
        --out /gscratch/portia/eabe/data/Johnson_lab/red_data_3d_v12_pseudo_negatives_20260905 \\
        --n 2000 --edge-frac 0.25

Two centroid sources, one per recording (controller ruling 2026-09-05 --
`coarse_tracks.npz` exists only for Session0/2025_10_20_13_20_04):

  (a) `--tracks REC=coarse_tracks.npz` -- the mask-free coarse pass
      (`jarvis_jax.tracking.coarse_track.coarse_pass`/`write_coarse_tracks`):
      `X3d` (F,T,3), `exist` (F,T), `coarse_frame` (T,), `trackable` (F,T),
      plus the sibling `.meta.json` for `session_dir`/`calib_dir`/`cameras`/
      `W`/`H`. Tracked-centroid coverage spans the WHOLE recording at the
      coarse stride (16).
  (b) `--lift-root REC=pose_mvq_p3b` -- the masked p3b campaign's own bout
      lifts: per-frame fly centroids = nanmean over keypoints of
      `bouts/bout_*/fly{0,1}/kp3d.npz`'s `kp3d` (T,K,3), placed at the bout's
      `frame_start` (`mvq_meta.json`), ONLY for frames inside a bout.

Either way a source provides exactly one thing: `frame -> the tracked fly
centroids OF THAT FRAME`. Every clearance gate below is measured against
those, at the candidate's own frame, never at a neighbouring one.

STRATA (both drawn per recording; `--edge-frac`, default 0.25, is the edge
share of each recording's anchor quota):

  * `interior` -- reject-sample a centre uniformly inside the
    tracked-centroid bounding box (inflated `--bbox-inflate`, 20 %) at a
    randomly drawn available frame.
  * `edge` -- CenterDetect FALSE PEAKS (spec §3.5, the gate-bout-1 failure
    mode: a window on empty floor or an arena fixture where the existence
    head fired). For `--edge-scan-frames` sampled frames per recording the
    CenterDetect checkpoint (`--centerdetect`) is run over all 7 cameras and
    its peaks are lifted with `coarse_centres.lift_peaks_to_centres`; the
    lifted centres the two real flies do NOT explain are the candidates.
    See `false_peak_centres` for why this needs `--cd-peaks` > 2 rather than
    just `max_animals >= 3`. The scan visits only frames where EVERY fly is
    tracked (`--edge-allow-untracked` to relax): with a fly's centroid
    missing, "no fly is in this window" cannot be checked at all.

BOTH strata go through the SAME three clearance gates, at the candidate's
own frame -- `>= --min-dist-units` (60 units, 6 mm) from EVERY tracked
centroid of that exact frame, `>= --min-height-units` (6 units) above the
recording's fitted floor (`coarse_track.fit_floor`), and projecting inside
`>= --min-cams` (5) cameras (`coarse_centres._project_batch`). The distance
gate is what makes an edge candidate a NEGATIVE: a lifted CenterDetect peak
that is within 6 mm of a real fly is that fly (measured 2026-09-06: 49/52
residual peaks in a courtship bout are secondary responses on the flies'
own bodies, 2.7-32 units away), and only a peak that clears it is a window
with no animal in it. `--edge-units` additionally caps how far OUTSIDE the
recording's own tracked-centroid convex hull a candidate may sit, so a
two-view triangulation ghost in free space is not sold as an arena fixture;
a candidate INSIDE the hull (signed distance <= 0, "empty floor") is never
rejected by it.

Fix round 1 (2026-09-06) replaced the original edge source, which mined
`coarse_tracks.npz` for reads with `exist < 0.2`: `MVQRunner.read_typed`
returns None below `exist_thresh`, so a sub-threshold read is never written
and the real file's `exist` never leaves 0.5-1.0. That source could only
ever yield 0 rows, and did.

Every recording's negatives write ONE unique video frame each (a v12
frameset is keyed `<rec>/Frame_<n>/...`, and `pseudo_export.write_pseudo_export`
resolves exactly one negative row per (rec, frame) group -- see the module
docstring's "one negative record per (rec,frame)" note below for why two
negatives can never legally share a frame here).

T=2 PARTNERS (coordinator ruling 2026-09-06, T=2 loader review): every drawn
negative ("anchor") carries partner framesets at `--partner-deltas` (default
1, 4, 16 video frames FORWARD -- `f0 + delta`, not the positive anchors'
bidirectional endpoint search, since an empty window has no motion to prefer
a direction from), at the SAME world centre, so a T=2 window is not a frozen
identical pair. A partner is written ONLY where the same three clearance
gates (dist / height / cams) hold AT ITS OWN FRAME -- never assumed from the
anchor -- and is OMITTED otherwise (the loader then simply does not build
that Delta pair, exactly as for real pseudo-labels); the anchor's own
frameset carries `partners: {"1": frame, "4": frame, "16": frame}` for
whichever deltas cleared (`pseudo_export`'s existing convention, see
`extract_p3b_pseudolabels.scan_bout`), and the partner rows are written as
independent, unique-framed negatives with `role: "partner"` so
`write_pseudo_export`'s own `per_role`/`partner_availability` summaries count
anchors and partners separately with no code changes needed there.

Output is its own v12-format root (`jarvis_jax.data.pseudo_export.
write_pseudo_export`, unchanged): every annotation (anchor or partner) is
all-zero, `fly_id: -1`, `negative: true`, and the frameset's `center3D` is
the sampled window centre -- `V12WindowDataset`'s negative branch (Plan B
Task 1) reads `fly_valid` all-False and an existence target of 0 for every
slot from exactly this; T=2 pairing of two negative framesets is the OTHER
half of the T=2 loader review, not this script's concern -- this script only
needs both framesets to already exist, correctly gated, at the right frames.

WEIGHT (coordinator ruling 2026-09-06): negatives train at FULL weight
(`--weight`, default 1.0), not the 0.3 the positive pseudo-labels carry.
A negative's supervision is the existence target alone, and that target is
not a guess from a checkpoint -- the window provably has no fly in it (three
independent geometric gates say so), so there is nothing to down-weight.
`source` stays `"pseudo"` and `role` stays `"negative"`/`"partner"`, so the
loader still separates these from the real labelled set.

EXPECTATION for `figures/2026-09-mvq/v2_pseudo/negatives_check.png` (state it
before looking, CLAUDE.md): a 16-panel contact sheet of drawn empty crops
across several recordings, with AT LEAST 6 panels from the `edge` stratum --
every panel shows bare substrate or an arena fixture, and NO FLY. A panel
with a fly's body in it is a mis-sampled negative that would teach the
existence head to suppress a real animal; in an EDGE panel specifically it
would mean the CenterDetect false-peak candidates are not being routed
through the same distance/height/camera gates as the interior ones (the fix
round 1 review finding), i.e. the gate gap is still open. The run must not
be committed until this has been read back and confirmed clean (or the
offending stratum/gate is reported as a concern).
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "third_party", "jarvis_jax"), _REPO, os.path.join(_REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_EXPORT_NAMES = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902"
TRACKABLE_EXIST = 0.5   # jarvis_jax.tracking.coarse_track.TRACKABLE_EXIST, restated (no import
                        # of coarse_track needed just for the constant)


# --------------------------------------------------------------------- geometry
def _hull_signed_distance(points, hull):
    """(N,3) points -> (N,) SIGNED distance to `hull`'s nearest bounding facet
    plane: NEGATIVE inside the hull, POSITIVE outside, magnitude = distance to
    the nearest supporting hyperplane.

    `hull.equations` is `[A | b]` per facet with `A` unit-normalized by qhull
    so that `A . x + b <= 0` for every point of the hull; the max of that over
    facets is <= 0 strictly inside and > 0 outside. That is the standard
    convex-polytope proxy for "distance to the hull surface" -- exact when the
    foot of the perpendicular lands within that facet, a slight underestimate
    near a facet's own edge/vertex.

    The SIGN carries the meaning `--edge-units` needs (fix round 1): an
    arena-edge false peak sits OUTSIDE the cloud of places a fly's centroid
    was ever tracked, and how far outside is exactly how suspicious it is (a
    two-view triangulation ghost can be metres away). A candidate INSIDE the
    hull is an empty-floor window -- equally a §3.5 negative, and never
    rejected for its hull distance.
    """
    eq = np.asarray(hull.equations, np.float64)
    A, b = eq[:, :-1], eq[:, -1]
    return np.max(np.asarray(points, np.float64) @ A.T + b[None, :], axis=1)


def _project_inside_count(point, cam_mats, W, H):
    """One (3,) world point -> how many of `cam_mats`'s cameras it projects
    inside a (W,H) image, via `coarse_centres._project_batch` (the same
    perspective-divide convention `write_coarse_tracks`/`coarse_pass` use)."""
    from jarvis_jax.tracking.coarse_centres import _project_batch
    uv = _project_batch(np.asarray(point, np.float64)[None], cam_mats)[0]     # (C,2)
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(uv).all(-1)
        inside = finite & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) \
            & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
    return int(inside.sum())


# ----------------------------------------------------------------- recording data
@dataclasses.dataclass
class RecData:
    """Everything `sample_negatives` needs about one recording, source-agnostic.

    `frame_to_cent[frame]` is the list of that frame's OWN tracked-and-finite
    fly centroids (0, 1 or 2 of them); `frame_has_untracked[frame]` is True
    if at least one fly is untracked there (NaN/exist<0.5), for
    `--require-tracked`. `hull` is the `scipy.spatial.ConvexHull` of every
    tracked centroid of the recording, the reference `--edge-units` measures
    against. `edge_candidates` starts EMPTY for both sources and is filled by
    `scan_false_peaks` (a CenterDetect pass over sampled frames) -- it is not
    something either source file already contains.
    """
    recording: str
    source: str                 # "tracks" | "lift_root"
    source_path: str
    session_dir: str
    calib_dir: str
    cameras: list
    cam_mats: np.ndarray        # (C,4,3) float64
    W: int
    H: int
    checkpoint: str
    frames_avail: np.ndarray    # (N,) int64, candidate video frames
    frame_to_cent: dict
    frame_has_untracked: dict
    bbox_lo: np.ndarray
    bbox_hi: np.ndarray
    floor: object                # jarvis_jax.tracking.coarse_track.FloorPlane
    hull: object                 # scipy.spatial.ConvexHull of every tracked centroid
    edge_candidates: list = dataclasses.field(default_factory=list)


def build_recdata_from_tracks(recording, tracks_path, *, bbox_inflate=0.20):
    """Source (a): a mask-free `coarse_tracks.npz` + its sibling `.meta.json`."""
    from jarvis_jax.tracking.coarse_track import fit_floor
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from scipy.spatial import ConvexHull

    with np.load(tracks_path, allow_pickle=True) as z:
        X3d = np.asarray(z["X3d"], np.float64)                      # (F,T,3)
        exist = np.asarray(z["exist"], np.float64)                  # (F,T)
        coarse_frame = np.asarray(z["coarse_frame"], np.int64)      # (T,)
        trackable = (np.asarray(z["trackable"], bool) if "trackable" in z.files
                    else (np.isfinite(X3d).all(-1) & (exist >= TRACKABLE_EXIST)))
    meta_path = tracks_path.rsplit(".npz", 1)[0] + ".meta.json"
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"{tracks_path}: no sibling {meta_path} (need session_dir/"
                                f"calib_dir/cameras/W/H)")
    meta = json.load(open(meta_path))
    cameras = [str(c) for c in meta["cameras"]]
    calib_dir = meta.get("calib_dir")
    if not calib_dir:
        raise ValueError(f"{meta_path}: no calib_dir")
    rt = ReprojectionTool(calib_dir)
    if list(rt.cameras.keys()) != cameras:
        raise ValueError(f"{tracks_path}: calibration glob order {list(rt.cameras.keys())} != "
                         f"meta cameras {cameras}")
    cam_mats = np.asarray(rt.camera_matrices, np.float64)
    W, H = int(meta["W"]), int(meta["H"])
    F, T = X3d.shape[0], X3d.shape[1]

    frame_to_cent, frame_has_untracked = {}, {}
    for t in range(T):
        fr = int(coarse_frame[t])
        frame_to_cent[fr] = [X3d[f, t].copy() for f in range(F) if trackable[f, t]]
        frame_has_untracked[fr] = any(not np.isfinite(X3d[f, t]).all() for f in range(F))

    pts_all = np.concatenate([X3d[f][trackable[f]] for f in range(F)], axis=0)
    if pts_all.shape[0] < 4:
        raise ValueError(f"{tracks_path}: only {pts_all.shape[0]} trackable centroids -- need "
                         f">= 4 (a floor fit needs >= 3, a convex hull in 3D needs >= 4)")
    lo, hi = pts_all.min(0), pts_all.max(0)
    center, half = (lo + hi) / 2.0, (hi - lo) / 2.0 * (1.0 + float(bbox_inflate))
    bbox_lo, bbox_hi = center - half, center + half

    floor = fit_floor(X3d, exist=exist, n_fit=F * T)   # n_fit is an upper bound -- fit_floor
                                                        # already restricts to finite&trackable
                                                        # points before slicing by n_fit, so this
                                                        # uses every trackable centroid, not just
                                                        # the first 2000 (extract_p3b_pseudolabels'
                                                        # own `n_fit=len(pts)` convention).
    hull = ConvexHull(pts_all)

    return RecData(recording=recording, source="tracks", source_path=tracks_path,
                   session_dir=str(meta.get("session_dir", "")), calib_dir=str(calib_dir),
                   cameras=cameras, cam_mats=cam_mats, W=W, H=H,
                   checkpoint=str(meta.get("checkpoint", "unknown")),
                   frames_avail=coarse_frame, frame_to_cent=frame_to_cent,
                   frame_has_untracked=frame_has_untracked, bbox_lo=bbox_lo, bbox_hi=bbox_hi,
                   floor=floor, hull=hull, edge_candidates=[])


def build_recdata_from_liftroot(recording, lift_root, *, bbox_inflate=0.20, num_animals=2):
    """Source (b): the masked p3b campaign's own bout lifts.

    Returns `None` (never raises) when `lift_root` has no finished bouts for
    `recording` -- the caller skips it with a message (controller ruling:
    Session1 lifts are being re-run and some recordings are not done yet)."""
    from jarvis_jax.tracking.coarse_track import fit_floor
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from scipy.spatial import ConvexHull
    from pseudo_labels.extract_p3b_pseudolabels import discover_bouts
    from coarse_pass_mvq import video_size

    refs = [r for r in discover_bouts([lift_root]) if r.recording == recording]
    if not refs:
        return None
    refs = sorted(refs, key=lambda r: r.bout)
    cameras = list(refs[0].cameras)
    calib_dir = refs[0].calib_dir
    session_dir = refs[0].session_dir
    rt = ReprojectionTool(calib_dir)
    if list(rt.cameras.keys()) != cameras:
        raise ValueError(f"{lift_root}/{recording}: calibration glob order "
                         f"{list(rt.cameras.keys())} != bout cameras {cameras}")
    cam_mats = np.asarray(rt.camera_matrices, np.float64)
    _n_frames, W, H = video_size(session_dir, cameras[0])
    checkpoint = json.load(open(os.path.join(refs[0].bout_dir, "mvq_meta.json"))).get(
        "checkpoint", "unknown")

    frame_to_cent, frame_has_untracked, all_pts = {}, {}, []
    n_overlap = 0
    for ref in refs:
        cent = []
        for f in range(num_animals):
            with np.load(os.path.join(ref.bout_dir, f"fly{f}", "kp3d.npz"),
                        allow_pickle=True) as z:
                kp3d = np.asarray(z["kp3d"], np.float64)          # (T,K,3)
            with np.errstate(invalid="ignore"):
                cent.append(np.nanmean(kp3d, axis=1))              # (T,3), all-NaN if no kp finite
        for t in range(ref.n_frames):
            fr = ref.frame_start + t
            if fr in frame_to_cent:
                # Adjacent campaign bouts can overlap by a handful of frames
                # at their boundary (measured: bout detection windows, not a
                # corrupt export -- e.g. 2026_04_02_15_25_51 bout_00013/14
                # share 6 frames). Keep whichever bout's read came FIRST
                # (sorted by bout number) and count the rest, rather than
                # aborting the whole recording over a boundary overlap.
                n_overlap += 1
                continue
            ents = [cent[f][t] for f in range(num_animals) if np.isfinite(cent[f][t]).all()]
            frame_to_cent[fr] = ents
            frame_has_untracked[fr] = len(ents) < num_animals
            all_pts.extend(ents)
    if n_overlap:
        print(f"[extract_empty] {recording}: {n_overlap} frame(s) shared by adjacent bouts "
              f"under {lift_root}; kept the first bout's read", flush=True)

    pts_all = np.asarray(all_pts, np.float64)
    if pts_all.shape[0] < 4:
        raise ValueError(f"{lift_root}/{recording}: only {pts_all.shape[0]} finite fly centroids "
                         f"across {len(refs)} bouts -- need >= 4")
    lo, hi = pts_all.min(0), pts_all.max(0)
    center, half = (lo + hi) / 2.0, (hi - lo) / 2.0 * (1.0 + float(bbox_inflate))
    bbox_lo, bbox_hi = center - half, center + half

    # No per-frame existence here (controller ruling): every point already
    # passed into `all_pts` is finite by construction, so `fit_floor` gets no
    # `exist` gate and `n_fit` is every point there is.
    floor = fit_floor(pts_all, n_fit=pts_all.shape[0])

    frames_avail = np.array(sorted(frame_to_cent), np.int64)
    return RecData(recording=recording, source="lift_root", source_path=lift_root,
                   session_dir=session_dir, calib_dir=calib_dir, cameras=cameras,
                   cam_mats=cam_mats, W=W, H=H, checkpoint=str(checkpoint),
                   frames_avail=frames_avail, frame_to_cent=frame_to_cent,
                   frame_has_untracked=frame_has_untracked, bbox_lo=bbox_lo, bbox_hi=bbox_hi,
                   floor=floor, hull=ConvexHull(pts_all), edge_candidates=[])


# ------------------------------------------------- edge stratum: CenterDetect
DEFAULT_CENTERDETECT = ("/gscratch/portia/eabe/data/Johnson_lab/jax_centerdetect_runs/"
                        "cd_focal_bg30/ckpt/epoch_004")


class CenterDetectPeaks:
    """`(C,H,W,3) uint8 frames -> (peaks (C,k,2) px, scores (C,k))`, k >= 3.

    `coarse_centres.CenterDetector` is the production wrapper and this class
    runs EXACTLY its model, preprocessing and decode -- `_restore_centerdetect`
    (same Orbax/nnx restore), `_centerdetect_preprocess` (PIL BILINEAR, the
    train-time resize, NOT cv2), `extract_top_k_peaks(suppression_radius=15)`
    + `peaks_to_full_image`, `min_score` NaN-ing the coordinates. The one
    difference is `k`.

    Why k must exceed 2 (measured 2026-09-06, and the reason the review's
    "just pass max_animals >= 3" is not sufficient on its own):
    `CenterDetector.peaks` hard-codes `k=2`, and with two real flies in the
    frame those two peaks per camera ARE the two flies -- a lift with
    `max_animals=4` over that array returned 32/32 centres within 12 units of
    a tracked fly on `2026_04_02_16_21_32`, i.e. no false peak survives to be
    found, because there is no unconsumed peak left to seed one. `--cd-peaks`
    (default 8) keeps the weaker responses -- arena fixtures, the chamber
    edge, wall reflections -- on the table for `false_peak_centres`.

    GPU/checkpoint-heavy; never constructed in a unit test (the tests inject
    a fake `peak_reader` instead, exactly as `coarse_track`'s own tests inject
    a fake detector).
    """

    def __init__(self, ckpt_dir, *, k=8, min_score=0.2, workers=7):
        from concurrent.futures import ThreadPoolExecutor
        from jarvis_jax.tracking.coarse_centres import _restore_centerdetect
        self.k = int(k)
        if self.k < 3:
            raise ValueError(f"--cd-peaks must be >= 3 to leave a non-fly peak to lift "
                             f"(got {self.k}); k=2 is the production top-2 and is fully "
                             f"consumed by the two real flies")
        self.min_score = float(min_score)
        self._model = _restore_centerdetect(ckpt_dir)
        # PIL's resize releases the GIL, and the 7 cameras' 1936x448 -> 320x320
        # squash is the single most expensive step of the scan when run
        # serially (measured 48.6 ms serial vs 8.1 ms over 7 threads).
        self._pool = ThreadPoolExecutor(int(workers))

    def __call__(self, frames):
        import jax.numpy as jnp
        from jarvis_jax.tracking.coarse_centres import _centerdetect_preprocess
        from jarvis_jax.eval.centerdetect_decode import extract_top_k_peaks, peaks_to_full_image

        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames must be (C,H,W,3), got {frames.shape}")
        img_h, img_w = int(frames.shape[1]), int(frames.shape[2])
        x = jnp.asarray(np.stack(list(self._pool.map(
            _centerdetect_preprocess, [frames[i] for i in range(frames.shape[0])]))))
        hm = np.asarray(self._model(x, use_running_average=True))     # (C,160,160,1)
        pk, conf = extract_top_k_peaks(hm, k=self.k, suppression_radius=15)
        full = peaks_to_full_image(pk, hm.shape[1], img_w, img_h).astype(np.float32)
        conf = conf.astype(np.float32)
        full[conf < self.min_score] = np.nan
        return full, conf

    def close(self):
        self._pool.shutdown(wait=False)


def _compact_peaks(peaks, scores):
    """(C,k,2)/(C,k) with NaN holes -> the first two SURVIVING peaks of each
    camera moved into slots 0/1, plus how many peaks are still unconsumed.

    `lift_peaks_to_centres` seeds triangulation candidates from peak slots 0
    and 1 only (`coarse_centres._seed_candidates`'s `for pi in range(2)`), so
    a peak sitting in slot 5 can be matched as an inlier but can never START a
    candidate. Compacting between rounds is what lets every peak get its turn
    without touching that module.
    """
    C = peaks.shape[0]
    out = np.full((C, 2, 2), np.nan, np.float32)
    out_s = np.zeros((C, 2), np.float32)
    n_left = 0
    for c in range(C):
        keep = np.flatnonzero(np.isfinite(peaks[c]).all(-1))
        n_left += int(keep.size)
        for j, s in enumerate(keep[:2]):
            out[c, j] = peaks[c, s]
            out_s[c, j] = scores[c, s]
    return out, out_s, n_left


def _consume_peaks(peaks, scores, centres, cam_mats, tol_px):
    """NaN out every peak within `tol_px` of any finite centre's reprojection
    (in place). That is the same "this peak is explained, do not re-discover
    it" bookkeeping `lift_peaks_to_centres` does internally per animal, lifted
    out here so it also works ACROSS calls."""
    from jarvis_jax.tracking.coarse_centres import _project_batch
    good = np.asarray([c for c in np.asarray(centres) if np.isfinite(c).all()], np.float64)
    if good.shape[0] == 0:
        return 0
    proj = _project_batch(good, cam_mats)                              # (A,C,2)
    n = 0
    with np.errstate(invalid="ignore"):
        for a in range(proj.shape[0]):
            d = np.linalg.norm(peaks - proj[a][:, None, :], axis=-1)   # (C,k)
            hit = np.isfinite(d) & (d <= tol_px)
            n += int(hit.sum())
            peaks[hit] = np.nan
            scores[hit] = np.nan
    return n


def _lift_residual_px(centre, peaks, cam_mats, max_resid_px):
    """Median over INLIER cameras of the distance from `centre`'s reprojection
    to that camera's nearest ORIGINAL peak, in full-image px -- "how well the
    peaks that voted for this centre actually agree with it". NaN if no camera
    is within `max_resid_px` (cannot happen for a centre `lift_peaks_to_centres`
    returned, which needed >= min_views inliers, but reported honestly rather
    than defaulted to 0)."""
    from jarvis_jax.tracking.coarse_centres import _project_batch
    proj = _project_batch(np.asarray(centre, np.float64)[None], cam_mats)[0]   # (C,2)
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(np.asarray(peaks, np.float64) - proj[:, None, :], axis=-1)  # (C,k)
        d = np.where(np.isfinite(d), d, np.inf)
        best = d.min(axis=1)
    inl = best[best <= max_resid_px]
    return float(np.median(inl)) if inl.size else float("nan")


def false_peak_centres(peaks, scores, cam_mats, *, num_animals=2, max_animals=3,
                       min_views=3, max_resid_px=25.0, max_rounds=8):
    """CenterDetect peaks -> the lifted 3D centres the REAL FLIES do not explain.

    Args:
        peaks: (C, k, 2) full-image px, NaN for a peak below `min_score`.
            `peaks[c]` is the SAME camera as `cam_mats[c]`, canonical order
            (`coarse_centres`'s camera-order contract -- the caller keeps it;
            here that is `RecData.cameras`/`RecData.cam_mats`, both read from
            the recording's own calibration glob).
        scores: (C, k) raw peak confidence.
        num_animals: how many real flies the production read takes off the
            table first (2 for courtship).
        max_animals: `lift_peaks_to_centres`'s cap PER ROUND. >= 3 is what the
            review asks for; on its own it is not enough, see
            `CenterDetectPeaks` -- the peaks must outnumber the flies too.

    Returns:
        `[{"xyz": (3,) float64, "cd_score": float, "cd_n_views": int,
           "cd_resid_px": float, "round": int}, ...]`, strongest first within
        each round.

    Method. Round 0 is the PRODUCTION read: `lift_peaks_to_centres` over the
    top-2 peaks per camera with `max_animals=num_animals` -- exactly what
    `coarse_track.coarse_pass` does -- and every peak within `max_resid_px` of
    one of those centres' reprojections is consumed. What is left is, by
    construction, the responses that are not the flies. Each further round
    compacts the survivors (`_compact_peaks`) and lifts them again, consuming
    as it goes, until nothing lifts or `max_rounds` is spent.

    Nothing here decides whether a returned centre is a usable negative --
    that is the three clearance gates' job, applied by `scan_false_peaks` and
    again, authoritatively, by `sample_negatives`. Most of these centres are
    NOT negatives: measured 2026-09-06 on a courtship bout, 49/52 sat 2.7-32
    units from a real fly (secondary responses on the animals' own bodies)
    and were rejected on distance.
    """
    from jarvis_jax.tracking.coarse_centres import lift_peaks_to_centres

    peaks0 = np.array(peaks, np.float32, copy=True)
    work = np.array(peaks, np.float32, copy=True)
    work_s = np.array(scores, np.float32, copy=True)
    cam_mats = np.asarray(cam_mats)

    real, _nv, _sc = lift_peaks_to_centres(work[:, :2], work_s[:, :2], cam_mats,
                                           min_views=min_views, max_resid_px=max_resid_px,
                                           max_animals=int(num_animals))
    _consume_peaks(work, work_s, real, cam_mats, max_resid_px)

    out = []
    for rnd in range(1, int(max_rounds) + 1):
        comp, comp_s, n_left = _compact_peaks(work, work_s)
        if n_left == 0:
            break
        centres, n_views, score = lift_peaks_to_centres(
            comp, comp_s, cam_mats, min_views=min_views, max_resid_px=max_resid_px,
            max_animals=int(max_animals))
        if not np.isfinite(centres).all(-1).any():
            break
        _consume_peaks(work, work_s, centres, cam_mats, max_resid_px)
        for a in range(centres.shape[0]):
            xyz = np.asarray(centres[a], np.float64)
            if not np.isfinite(xyz).all():
                continue
            out.append({"xyz": xyz, "cd_score": float(score[a]),
                        "cd_n_views": int(n_views[a]),
                        "cd_resid_px": _lift_residual_px(xyz, peaks0, cam_mats, max_resid_px),
                        "round": int(rnd)})
    return out


def plan_scan_frames(rd, n_frames, *, rng, block=64, min_gap=16, max_jump=300,
                     require_tracked=False):
    """Which frames of `rd` the CenterDetect scan visits, in blocks.

    Only frames the source actually covers are eligible (`frame_to_cent`), so
    every candidate found can be gated against that frame's own tracked
    centroids; `--require-tracked` drops frames where a fly is untracked,
    exactly as it does for the interior stratum.

    The frames come in CONTIGUOUS BLOCKS rather than uniformly at random for a
    pure IO reason: `SessionFrames` decodes forward through a gap of up to 300
    frames and seeks otherwise, and on these H.264 files that is the
    difference between 243 ms and 3,024 ms per 7-camera frame (measured
    2026-09-06 on 2025_10_20_13_20_04). Within a block, consecutive picks are
    >= `min_gap` (16) video frames apart so a block still samples 16x its own
    length of video.

    Returned FLAT but in RANDOM BLOCK ORDER (frames ascending inside each
    block, blocks shuffled), which matters because `scan_false_peaks` stops
    as soon as it has enough candidates: a globally sorted plan would make
    that early stop sample only the beginning of the recording. False peaks
    are also strongly clustered in time -- one phantom episode yields a run of
    consecutive scanned frames (measured 2026-09-06: 25 of 25 candidates in a
    192-frame scan came from a handful of episodes, their hull distance
    drifting smoothly) -- so which blocks are visited dominates the yield, and
    they must be drawn without positional bias.
    """
    avail = np.asarray(sorted(
        f for f in rd.frame_to_cent
        if not (require_tracked and rd.frame_has_untracked.get(f, False))), np.int64)
    n_frames = int(n_frames)
    if n_frames <= 0 or avail.size == 0:
        return []
    n_blocks = max(1, -(-n_frames // int(block)))
    starts = rng.permutation(avail.size)[:min(n_blocks, avail.size)]
    picked, out = set(), []
    for s in starts:
        last, got, i = None, 0, int(s)
        while i < avail.size and got < int(block):
            f = int(avail[i])
            if last is None or min_gap <= f - last <= max_jump:
                if f not in picked:
                    picked.add(f)
                    out.append(f)
                last, got = f, got + 1
            elif f - last > max_jump:
                break
            i += 1
        if len(out) >= n_frames:
            break
    return out[:n_frames]


def scan_false_peaks(rd, peak_reader, frames, *, n_target, min_dist_units=60.0,
                     min_height_units=6.0, min_cams=5, edge_units=100.0, num_animals=2,
                     max_animals=3, min_views=3, max_resid_px=25.0, progress_every=200):
    """Run `peak_reader` over `frames` and collect the arena-edge candidates.

    Args:
        peak_reader: `(recording, frame) -> (peaks (C,k,2), scores (C,k))` in
            `rd.cameras` order. `CenterDetectPeaks` behind a frame reader in
            production; a fake in the unit tests.
        n_target: stop early once this many candidates have been collected
            (an oversampled multiple of the recording's edge quota, so the
            sampler still has a choice).

    Returns `(candidates, stats)`. A candidate is
    `{"frame", "xyz", "cd_score", "cd_n_views", "cd_resid_px", "hull_dist_units"}`.

    The three clearance gates are applied HERE only to decide when the scan
    has found enough and to keep the pool honest -- `sample_negatives` applies
    them again, authoritatively, to whatever pool it is handed, so a bad
    candidate from any source (including a test fake) can never reach the
    export. Belt and braces on purpose: the review found the previous edge
    loop calling `_accept` with no gate check at all.
    """
    cands = []
    stats = {"frames_scanned": 0, "frames_planned": len(frames), "lifted": 0,
             "rejections": {"dist": 0, "height": 0, "cams": 0, "no_coverage": 0,
                            "outside_hull": 0}}
    t0 = time.time()
    for i, frame in enumerate(frames):
        if len(cands) >= int(n_target):
            break
        peaks, scores = peak_reader(rd.recording, int(frame))
        stats["frames_scanned"] += 1
        for c in false_peak_centres(peaks, scores, rd.cam_mats, num_animals=num_animals,
                                    max_animals=max_animals, min_views=min_views,
                                    max_resid_px=max_resid_px):
            stats["lifted"] += 1
            xyz = c["xyz"]
            ok, info = _clears_gates(rd, int(frame), xyz, min_dist_units=min_dist_units,
                                     min_height_units=min_height_units, min_cams=min_cams)
            if not ok:
                stats["rejections"][info["reason"]] += 1
                continue
            hd = float(_hull_signed_distance(xyz[None], rd.hull)[0])
            if hd > float(edge_units):
                stats["rejections"]["outside_hull"] += 1
                continue
            cands.append({"frame": int(frame), "xyz": xyz, "cd_score": c["cd_score"],
                          "cd_n_views": c["cd_n_views"], "cd_resid_px": c["cd_resid_px"],
                          "hull_dist_units": hd})
        if progress_every and (i + 1) % progress_every == 0:
            el = time.time() - t0
            print(f"[extract_empty] {rd.recording}: CD scan {i + 1}/{len(frames)} frames, "
                  f"{len(cands)}/{n_target} candidates, {el / (i + 1):.2f}s/frame", flush=True)
    stats["n_candidates"] = len(cands)
    stats["seconds"] = round(time.time() - t0, 1)
    return cands, stats


# --------------------------------------------------------------------- sampling
DEFAULT_PARTNER_DELTAS = (1, 4, 16)


def _clears_gates(rd, frame, cand, *, min_dist_units, min_height_units, min_cams):
    """Whether `cand` (3,) at `frame` clears the same three gates every
    negative is drawn under. Returns `(ok, info)`.

    `frame` must be a KNOWN key of `rd.frame_to_cent` -- an ABSENT key means
    no tracked-centroid coverage at that exact frame (source (a)'s stride-16
    coarse grid between samples, or a lift-root frame outside every bout),
    which is treated as "cannot verify this frame at all", never as
    "vacuously clear" (CLAUDE.md: never assume a gate passed for lack of
    data to check it against). `info` names the failing gate and its value on
    rejection, or every gate's value on success.
    """
    if frame not in rd.frame_to_cent:
        return False, {"reason": "no_coverage"}
    cents = rd.frame_to_cent[frame]
    best_d = float("inf")
    for c in cents:
        d = float(np.linalg.norm(cand - c))
        best_d = min(best_d, d)
        if d < min_dist_units:
            return False, {"reason": "dist", "min_dist_to_tracked_units": round(d, 3)}
    h = float(rd.floor.height(cand[None])[0])
    if h < min_height_units:
        return False, {"reason": "height", "height_above_floor_units": round(h, 3)}
    n_in = _project_inside_count(cand, rd.cam_mats, rd.W, rd.H)
    if n_in < min_cams:
        return False, {"reason": "cams", "n_cams_inside": int(n_in)}
    return True, {"min_dist_to_tracked_units": None if not cents else round(best_d, 3),
                  "height_above_floor_units": round(h, 3), "n_cams_inside": int(n_in)}


def _link_partners(rd, f0, cand, stratum, used_frames, *, deltas, min_dist_units,
                   min_height_units, min_cams, pstats):
    """T=2 partners of one accepted negative at (`f0`, `cand`).

    FORWARD-ONLY (`f0 + delta`), not the positive anchors' bidirectional
    endpoint search: an empty window has no motion to prefer a direction
    from (coordinator ruling 2026-09-06, T=2 loader review). A delta is
    OMITTED -- never guessed -- when the partner frame has no
    tracked-centroid coverage to verify against, fails a clearance gate AT
    ITS OWN FRAME (the SAME `>= min_dist_units` / `>= min_height_units` /
    `>= min_cams` gates, re-checked there, not assumed from the anchor), or
    collides with a frame some other negative or partner already claimed.

    Returns `(partners, partner_rows)`. `partners` is `{delta_str: frame}`
    for the ANCHOR's own record -- `pseudo_export`'s exact convention, see
    `extract_p3b_pseudolabels.scan_bout`'s `partners[d] = ...`. `partner_rows`
    are shaped like an anchor row (`frame`, `center3D`, `stratum`, `gates`)
    plus `role: "partner"`, `anchor_frame` and `delta`, so `main` writes each
    as its own `PseudoRecord(role="partner")` -- a SEPARATE frameset, same
    centre, same negative contract (fly_id -1, all-zero annotations, full
    frames), never the anchor's own frameset repeated.
    """
    partners, partner_rows = {}, []
    for delta in deltas:
        pf = int(f0) + int(delta)
        if pf in used_frames:
            pstats[str(delta)]["collision"] += 1
            continue
        ok, info = _clears_gates(rd, pf, cand, min_dist_units=min_dist_units,
                                 min_height_units=min_height_units, min_cams=min_cams)
        if not ok:
            pstats[str(delta)][info["reason"]] += 1
            continue
        used_frames.add(pf)
        partners[str(delta)] = pf
        partner_rows.append({"frame": pf, "center3D": np.array(cand, np.float64),
                             "stratum": dict(stratum), "gates": info,
                             "role": "partner", "anchor_frame": int(f0), "delta": int(delta)})
        pstats[str(delta)]["taken"] += 1
    return partners, partner_rows


def sample_negatives(rd: RecData, n_want, *, edge_frac=0.25, edge_want=None,
                     min_dist_units=60.0, min_height_units=6.0, min_cams=5,
                     require_tracked=False, rng=None, max_tries=300,
                     partner_deltas=DEFAULT_PARTNER_DELTAS):
    """Draw up to `n_want` empty-window ANCHOR negatives for one recording,
    each carrying T=2 partner framesets at `partner_deltas` where they clear
    the gates (spec §3.5 + coordinator ruling 2026-09-06: partners are
    additional framesets layered on the anchor quota, not counted against
    it -- exactly how the positive pseudo-labels' T=2 partners work).

    Returns `(anchor_rows, partner_rows, stats)`. An anchor row is a dict
    with `frame`, `center3D` ((3,) float64 world units), `stratum`
    (`{"kind": "edge"|"interior"}`, plus `cd_score`/`hull_dist_units`/
    `cd_resid_px`/`cd_n_views` for an edge row), `gates` (the quantities that
    admitted it, named) and `partners` (`{delta_str: partner_frame}`, only
    the deltas that cleared -- see `_link_partners`). A partner row has the
    same shape plus `role: "partner"`, `anchor_frame` and `delta`. Every
    row's `frame` -- anchor or partner, from either list -- is unique WITHIN
    this recording (one v12 frameset per (recording, frame); a collision
    would silently merge in `write_pseudo_export`'s per-frame grouping).

    `edge_want` overrides `round(edge_frac * n_want)` with an absolute count,
    which is how `main` moves one recording's unmet edge quota to a recording
    whose CenterDetect scan found more qualifying false peaks.

    BOTH STRATA GO THROUGH `_clears_gates` (fix round 1). The edge loop used
    to call `_accept` directly on whatever the candidate pool held, trusting
    the pool -- so the distance/height/camera gates that define "no fly in
    this window" were enforced for interior draws only. They are the whole
    reason an edge candidate counts as a negative, and a candidate pool is
    just a proposal, whatever built it.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n_want = int(n_want)
    if edge_want is None:
        n_edge_want = int(round(float(edge_frac) * n_want)) if rd.edge_candidates else 0
    else:
        n_edge_want = int(edge_want)
    deltas = tuple(int(d) for d in partner_deltas)

    picked, partner_rows, used_frames = [], [], set()
    pstats = {str(d): collections.Counter() for d in deltas}
    stats = {"recording": rd.recording, "source": rd.source, "n_want": n_want,
             "edge_pool": len(rd.edge_candidates), "edge_want": n_edge_want,
             "edge_taken": 0, "interior_taken": 0,
             "rejections": {"duplicate_frame": 0, "dist": 0, "height": 0, "cams": 0},
             "edge_rejections": {"duplicate_frame": 0, "dist": 0, "height": 0, "cams": 0,
                                 "no_coverage": 0},
             "partner_deltas": list(deltas)}

    def _accept(frame, cand, stratum, gates):
        used_frames.add(frame)
        p, prows = _link_partners(rd, frame, cand, stratum, used_frames, deltas=deltas,
                                  min_dist_units=min_dist_units,
                                  min_height_units=min_height_units, min_cams=min_cams,
                                  pstats=pstats)
        picked.append({"frame": frame, "center3D": np.asarray(cand, np.float64),
                       "stratum": dict(stratum), "gates": gates, "role": "negative",
                       "partners": p})
        partner_rows.extend(prows)

    # ---- edge stratum: CenterDetect false peaks (`scan_false_peaks`), gated
    # here exactly like an interior draw -- see the docstring.
    pool = list(rd.edge_candidates)
    rng.shuffle(pool)
    for cand in pool:
        if stats["edge_taken"] >= n_edge_want:
            break
        frame = int(cand["frame"])
        xyz = np.asarray(cand["xyz"], np.float64)
        if frame in used_frames:
            stats["edge_rejections"]["duplicate_frame"] += 1
            continue
        ok, info = _clears_gates(rd, frame, xyz, min_dist_units=min_dist_units,
                                 min_height_units=min_height_units, min_cams=min_cams)
        if not ok:
            stats["edge_rejections"][info["reason"]] += 1
            continue
        _accept(frame, xyz,
                {"kind": "edge", "cd_score": round(float(cand["cd_score"]), 3),
                 "cd_n_views": int(cand["cd_n_views"]),
                 "cd_resid_px": round(float(cand["cd_resid_px"]), 3),
                 "hull_dist_units": round(float(cand["hull_dist_units"]), 3)},
                info)
        stats["edge_taken"] += 1

    # ---- interior stratum (fills the rest of the quota)
    frames_pool = [int(f) for f in rd.frames_avail]
    if require_tracked:
        frames_pool = [f for f in frames_pool if not rd.frame_has_untracked.get(f, False)]
    if frames_pool:
        rng.shuffle(frames_pool)
        budget = len(frames_pool) * int(max_tries)
        fi = tries = 0
        while len(picked) < n_want and tries < budget:
            frame = frames_pool[fi % len(frames_pool)]
            fi += 1
            tries += 1
            if frame in used_frames:
                continue
            cand = rng.uniform(rd.bbox_lo, rd.bbox_hi)
            ok, info = _clears_gates(rd, frame, cand, min_dist_units=min_dist_units,
                                     min_height_units=min_height_units, min_cams=min_cams)
            if not ok:
                # `no_coverage` cannot happen here: `frame` is drawn from
                # `rd.frames_avail`, exactly the key set `frame_to_cent` was
                # built over -- only dist/height/cams reject an interior draw.
                stats["rejections"][info["reason"]] += 1
                continue
            _accept(frame, cand, {"kind": "interior"}, info)
    stats["interior_taken"] = len(picked) - stats["edge_taken"]
    stats["edge_shortfall"] = max(0, n_edge_want - stats["edge_taken"])
    stats["shortfall"] = max(0, n_want - len(picked))
    stats["n_anchors"] = len(picked)
    stats["n_partners"] = len(partner_rows)
    stats["partners"] = {d: dict(c) for d, c in pstats.items()}
    return picked, partner_rows, stats


# ------------------------------------------------------------------------ figure
def _pick_round_robin(by_rec, n, rng):
    """Up to `n` rows drawn round-robin over recordings (a random one from
    each in turn), so no single recording can fill the sheet."""
    pools = {rec: list(rows) for rec, rows in by_rec.items() if rows}
    for rows in pools.values():
        rng.shuffle(rows)
    out, keys = [], sorted(pools)
    rng.shuffle(keys)
    i = 0
    while len(out) < n and pools:
        k = keys[i % len(keys)]
        if pools[k]:
            out.append(pools[k].pop())
        if not pools[k]:
            del pools[k]
            keys.remove(k)
            if not keys:
                break
            i = i % len(keys)
            continue
        i += 1
    return out


def write_contact_sheet(out_path, rows_by_rec, recdatas, frame_reader, *, n_panels=16,
                        min_edge_panels=6, seed=0):
    """`figures/2026-09-mvq/v2_pseudo/negatives_check.png`: `n_panels` crops of
    drawn empty windows, spanning recordings and both strata.

    `min_edge_panels` of them come from the `edge` (CenterDetect false-peak)
    stratum whenever that many edge negatives exist -- the sheet's job in fix
    round 1 is specifically to show whether the edge candidates are as
    fly-free as the interior ones, and a sheet that happens to draw all
    interior panels cannot answer that (the previous run's 12 panels were
    all interior, because the edge pool was empty).

    EXPECTATION (module docstring, restated at the point of use): every panel
    shows bare substrate or an arena fixture and NO fly. Read back with the
    Read tool before committing -- a fly in an INTERIOR panel means a gate
    threshold is too loose; a fly in an EDGE panel means the false-peak
    candidates are not going through those gates at all.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jarvis_jax.data.transforms import crop_origin
    from jarvis_jax.tracking.coarse_centres import _project_batch

    rng = np.random.default_rng(seed)
    flat = [(rec, row) for rec, rows in rows_by_rec.items() for row in rows]
    if not flat:
        raise ValueError("no negatives to render")
    # up to 2 crops (best-margin cameras) per negative, stratified across
    # recordings within each stratum.
    per_neg = 2
    n_negs = max(1, -(-n_panels // per_neg))
    n_edge_negs = min(n_negs, -(-int(min_edge_panels) // per_neg))
    by_kind = {"edge": collections.defaultdict(list), "interior": collections.defaultdict(list)}
    for rec, row in flat:
        by_kind.setdefault(row["stratum"]["kind"], collections.defaultdict(list))[rec].append(
            (rec, row))
    chosen = _pick_round_robin(by_kind["edge"], n_edge_negs, rng)
    rest = {k: v for kind, recs in by_kind.items() if kind != "edge"
            for k, v in recs.items()}
    chosen += _pick_round_robin(rest, n_negs - len(chosen), rng)
    if len(chosen) < n_negs:                      # not enough interior either: top up on edge
        taken = {id(row) for _rec, row in chosen}
        left = {rec: [(r, row) for r, row in rows if id(row) not in taken]
                for rec, rows in by_kind["edge"].items()}
        chosen += _pick_round_robin(left, n_negs - len(chosen), rng)

    panels = []
    for rec, row in chosen:
        rd = recdatas[rec]
        uv = _project_batch(np.asarray(row["center3D"], np.float64)[None], rd.cam_mats)[0]
        with np.errstate(invalid="ignore"):
            margin = np.minimum(np.minimum(uv[:, 0], rd.W - 1 - uv[:, 0]),
                                np.minimum(uv[:, 1], rd.H - 1 - uv[:, 1]))
            margin = np.where(np.isfinite(margin), margin, -np.inf)
        order = np.argsort(-margin)
        cams = [c for c in order if margin[c] > 0][:per_neg]
        for c in cams:
            panels.append((rec, rd.cameras[int(c)], row, uv[int(c)]))
    # edge panels first so the stratum under review reads across the top rows
    panels.sort(key=lambda p: (p[2]["stratum"]["kind"] != "edge", p[0], int(p[2]["frame"])))
    panels = panels[:n_panels]

    ncols = 4
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    n_edge = 0
    for ax, (rec, cam, row, (u, v)) in zip(axes, panels):
        frame_no = int(row["frame"])
        frames, present = frame_reader(rec, frame_no)
        ci = recdatas[rec].cameras.index(cam)
        img = frames[ci]
        x0, y0 = crop_origin([float(u), float(v), 0, 0], img.shape[1], img.shape[0], 448)
        crop = img[y0:y0 + 448, x0:x0 + 448]
        ax.imshow(crop)
        st = row["stratum"]
        kind = st["kind"]
        n_edge += kind == "edge"
        extra = (f"\ncd_score {st.get('cd_score')} resid {st.get('cd_resid_px')} px, "
                 f"hull {st.get('hull_dist_units')} u" if kind == "edge" else
                 f"\n{row['gates'].get('min_dist_to_tracked_units')} u from the nearest fly")
        ax.set_title(f"{rec}\n{cam} Frame_{frame_no} [{kind}]{extra}", fontsize=7,
                     color=("#c05000" if kind == "edge" else "#202020"))
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Empty-window negatives -- every panel should show NO fly "
                 f"({n_edge} edge / {len(panels) - n_edge} interior)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return {"n_panels": len(panels), "n_edge_panels": int(n_edge),
           "panels": [{"recording": rec, "camera": cam, "frame": int(row["frame"]),
                       "kind": row["stratum"]["kind"],
                       "min_dist_to_tracked_units": row["gates"].get(
                           "min_dist_to_tracked_units")} for rec, cam, row, _ in panels]}


# --------------------------------------------------------------------- verify
def verify_negative_export(root, n=20, split="train", seed=0):
    """Open the REAL export with `V12WindowDataset` and read up to `n` random
    negative windows.

    `extract_p3b_pseudolabels.verify_export` is the loader round trip for a
    REAL (labelled) export -- it asserts `has3d[0,0].any()`, i.e. a host with
    triangulated 3D, which is FALSE by construction for every window here
    (a negative has no host and `has3d` all-zero, per `V12WindowDataset`'s
    NEGATIVES docstring). Calling it against an all-negative root fails on
    its first window, not because the export is wrong. This checks the
    contract that DOES apply to a negative window instead: `is_negative` is
    True, `fly_valid` has no True slot, `center3D` is finite, >= 2 cameras
    are valid, and at least one valid camera's crop is not all-black (an
    unwritten or misnamed JPEG).
    """
    from jarvis_jax.data.v12_windows import V12WindowDataset
    ds = V12WindowDataset(root, split, T=1, train=False)
    if len(ds) == 0:
        raise AssertionError(f"{root}: V12WindowDataset found 0 windows for split {split!r} -- "
                             f"the export or the loader's negative branch is broken")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(int(n), len(ds)), replace=False)
    n_cams = 0
    for i in idx:
        s = ds[int(i)]
        rec, fly, frame = ds.windows[int(i)]
        if fly >= 0:
            raise AssertionError(f"window {rec}/Frame_{frame}: fly id {fly} >= 0 (not a negative)")
        if not bool(s["is_negative"]):
            raise AssertionError(f"window {rec}/Frame_{frame}: is_negative is False")
        if int(s["fly_valid"].sum()) != 0:
            raise AssertionError(f"window {rec}/Frame_{frame}: fly_valid has a True slot")
        if not np.isfinite(s["center3D"]).all():
            raise AssertionError(f"window {rec}/Frame_{frame}: non-finite center3D")
        if int(s["cam_valid"].sum()) < 2:
            raise AssertionError(f"window {rec}/Frame_{frame}: < 2 valid cameras")
        if not s["crops"][0][s["cam_valid"][0]].any():
            raise AssertionError(f"window {rec}/Frame_{frame}: all-black crops")
        n_cams += int(s["cam_valid"].sum())
    out = {"windows_checked": len(idx), "n_windows": len(ds),
           "mean_valid_cameras": round(n_cams / max(len(idx), 1), 2), "all_negative": True}
    print(f"[verify] {out}", flush=True)
    return out


# --------------------------------------------------------------------------- CLI
def _parse_kv(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected REC=PATH, got {item!r}")
        rec, path = item.split("=", 1)
        out[rec] = path
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", action="append", default=[], metavar="REC=coarse_tracks.npz",
                    help="source (a), repeatable")
    ap.add_argument("--lift-root", action="append", default=[], metavar="REC=pose_mvq_p3b",
                    help="source (b), repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--export-names-from", default=DEFAULT_EXPORT_NAMES)
    ap.add_argument("--n", type=int, default=2000, help="total negatives, spread over recordings")
    ap.add_argument("--n-per-recording", type=int, default=None,
                    help="override the even split of --n across recordings with a source")
    ap.add_argument("--edge-frac", type=float, default=0.25,
                    help="share of each recording's anchor quota drawn from CenterDetect "
                         "false peaks (spec §3.5); 0 disables the scan entirely")
    ap.add_argument("--edge-units", type=float, default=100.0,
                    help="how far OUTSIDE the recording's tracked-centroid convex hull an "
                         "edge candidate may sit (units, 0.1 mm; 100 = 1 cm). A candidate "
                         "INSIDE the hull (empty floor) is never rejected by this")
    ap.add_argument("--centerdetect", default=DEFAULT_CENTERDETECT,
                    help="CenterDetect checkpoint dir for the edge stratum (needs a GPU)")
    ap.add_argument("--cd-peaks", type=int, default=8,
                    help="peaks per camera to decode; MUST exceed the 2 the production "
                         "wrapper takes or the two real flies consume them all")
    ap.add_argument("--cd-min-score", type=float, default=0.2,
                    help="CenterDetect peak threshold (production default)")
    ap.add_argument("--cd-max-animals", type=int, default=3,
                    help="lift_peaks_to_centres max_animals per residual round (>= 3)")
    ap.add_argument("--cd-min-views", type=int, default=3)
    ap.add_argument("--cd-max-resid-px", type=float, default=25.0)
    ap.add_argument("--edge-scan-frames", type=int, default=1200,
                    help="frames per recording the CenterDetect scan may visit")
    ap.add_argument("--edge-scan-block", type=int, default=64,
                    help="scan frames per contiguous block (video seeks are 12x a "
                         "decode-through; see plan_scan_frames)")
    ap.add_argument("--edge-oversample", type=float, default=3.0,
                    help="stop a recording's scan at edge_quota * this many candidates")
    ap.add_argument("--edge-allow-untracked", action="store_true",
                    help="let the CenterDetect scan visit frames where a fly's centroid "
                         "is NaN. OFF by default, unlike the interior stratum: at such a "
                         "frame the missing fly could be anywhere, INCLUDING inside the "
                         "candidate window, so 'this window has no fly' is unverifiable "
                         "there (CLAUDE.md: never assume a gate passed for lack of data "
                         "to check it against). Costs little -- 97-100 %% of the lift-root "
                         "recordings' in-bout frames have both flies, and 44 %% of "
                         "2025_10_20_13_20_04's coarse frames (13,652 of 31,125)")
    ap.add_argument("--min-dist-units", type=float, default=60.0, help="6 mm")
    ap.add_argument("--min-height-units", type=float, default=6.0)
    ap.add_argument("--min-cams", type=int, default=5)
    ap.add_argument("--bbox-inflate", type=float, default=0.20)
    ap.add_argument("--require-tracked", action="store_true",
                    help="skip a frame entirely if either fly's centroid is NaN there. "
                         "Applies to the EDGE stratum too: the CenterDetect scan only "
                         "visits frames that survive this filter, so a false peak found "
                         "while one fly was untracked is never admitted -- that fly could "
                         "be anywhere, including inside the window")
    ap.add_argument("--max-tries", type=int, default=300,
                    help="rejection-sampling attempts per candidate frame")
    ap.add_argument("--partner-deltas", default=",".join(str(d) for d in DEFAULT_PARTNER_DELTAS),
                    help="comma list of T=2 partner spacings, forward-only from each anchor "
                         "(coordinator ruling 2026-09-06)")
    ap.add_argument("--num-animals", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight", type=float, default=1.0,
                    help="loss weight written into the manifest and every frameset. "
                         "1.0 (NOT the positives' 0.3): a negative's only supervision is "
                         "an existence target of 0, and that is a geometric fact about "
                         "the window, not a checkpoint's guess (coordinator ruling "
                         "2026-09-06)")
    ap.add_argument("--no-images", action="store_true", help="write the JSON only (schema smoke test)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--figures-dir", default="figures/2026-09-mvq/v2_pseudo")
    ap.add_argument("--n-figure-panels", type=int, default=16)
    ap.add_argument("--n-figure-edge-panels", type=int, default=8,
                    help="how many of --n-figure-panels must come from the edge stratum")
    return ap


def main(argv=None):
    from jarvis_jax.data.pseudo_export import PseudoRecord, write_pseudo_export
    from pseudo_labels.extract_p3b_pseudolabels import SessionFrames

    a = build_parser().parse_args(argv)
    t0 = time.time()

    tracks_specs = _parse_kv(a.tracks)
    lift_specs = _parse_kv(a.lift_root)
    dup = set(tracks_specs) & set(lift_specs)
    if dup:
        raise SystemExit(f"given both --tracks and --lift-root for {sorted(dup)}")
    if not tracks_specs and not lift_specs:
        raise SystemExit("no --tracks/--lift-root given")

    recdatas, skipped = {}, {}
    for rec, path in tracks_specs.items():
        ts = time.time()
        print(f"[extract_empty] {rec} (tracks): loading {path} ...", flush=True)
        try:
            recdatas[rec] = build_recdata_from_tracks(rec, path, bbox_inflate=a.bbox_inflate)
        except Exception as e:
            print(f"[extract_empty] {rec} (tracks): SKIPPED ({type(e).__name__}: {e})", flush=True)
            skipped[rec] = f"{type(e).__name__}: {e}"
            continue
        print(f"[extract_empty] {rec} (tracks): loaded in {time.time() - ts:.1f}s, "
              f"{len(recdatas[rec].frames_avail)} candidate frames", flush=True)
    for rec, path in lift_specs.items():
        ts = time.time()
        print(f"[extract_empty] {rec} (lift_root): loading {path} ...", flush=True)
        try:
            rd = build_recdata_from_liftroot(rec, path, bbox_inflate=a.bbox_inflate,
                                             num_animals=a.num_animals)
        except Exception as e:
            print(f"[extract_empty] {rec} (lift_root): SKIPPED ({type(e).__name__}: {e})", flush=True)
            skipped[rec] = f"{type(e).__name__}: {e}"
            continue
        if rd is None:
            print(f"[extract_empty] {rec} (lift_root): no finished bouts under {path}; skipped",
                  flush=True)
            skipped[rec] = "no finished bouts"
            continue
        recdatas[rec] = rd
        print(f"[extract_empty] {rec} (lift_root): loaded in {time.time() - ts:.1f}s, "
              f"{len(rd.frames_avail)} candidate frames", flush=True)

    if not recdatas:
        raise SystemExit(f"no recording had a usable source; skipped: {skipped}")

    recs_sorted = sorted(recdatas)
    if a.n_per_recording is not None:
        quotas = {rec: int(a.n_per_recording) for rec in recs_sorted}
    else:
        base, rem = divmod(int(a.n), len(recs_sorted))
        quotas = {rec: base + (1 if i < rem else 0) for i, rec in enumerate(recs_sorted)}

    partner_deltas = tuple(int(d) for d in a.partner_deltas.split(",") if d.strip())

    cameras = list(recdatas[recs_sorted[0]].cameras)
    for rec in recs_sorted:
        if list(recdatas[rec].cameras) != cameras:
            raise ValueError(f"{rec}: cameras {recdatas[rec].cameras} != the canonical "
                             f"{cameras} (write_pseudo_export needs ONE camera axis)")
    frames = SessionFrames({rec: recdatas[rec].session_dir for rec in recs_sorted}, cameras)

    # ---- edge stratum: one CenterDetect pass per recording (§3.5 false peaks)
    scan_stats = {}
    if a.edge_frac > 0:
        t_scan = time.time()
        detector = CenterDetectPeaks(a.centerdetect, k=a.cd_peaks, min_score=a.cd_min_score)
        peak_reader = lambda rec, frame: detector(frames(rec, int(frame))[0])   # noqa: E731
        scan_rng = np.random.default_rng(a.seed + 101)
        for rec in recs_sorted:
            rd = recdatas[rec]
            want = int(round(a.edge_frac * quotas[rec]))
            plan = plan_scan_frames(rd, a.edge_scan_frames, rng=scan_rng,
                                    block=a.edge_scan_block,
                                    require_tracked=(a.require_tracked
                                                     or not a.edge_allow_untracked))
            cands, st = scan_false_peaks(
                rd, peak_reader, plan, n_target=int(np.ceil(want * a.edge_oversample)),
                min_dist_units=a.min_dist_units, min_height_units=a.min_height_units,
                min_cams=a.min_cams, edge_units=a.edge_units, num_animals=a.num_animals,
                max_animals=a.cd_max_animals, min_views=a.cd_min_views,
                max_resid_px=a.cd_max_resid_px)
            rd.edge_candidates = cands
            st["edge_quota"] = want
            scan_stats[rec] = st
            print(f"[extract_empty] {rec}: CD scan {st['frames_scanned']}/{len(plan)} frames "
                  f"in {st['seconds']}s -> {len(cands)} false-peak candidates "
                  f"(quota {want}, lifted {st['lifted']}, rejections {st['rejections']})",
                  flush=True)
        detector.close()
        print(f"[extract_empty] CenterDetect scan total "
              f"{sum(s['n_candidates'] for s in scan_stats.values())} candidates in "
              f"{time.time() - t_scan:.0f}s", flush=True)

    rng = np.random.default_rng(a.seed)
    rows_by_rec, partner_rows_by_rec, stats_by_rec = {}, {}, {}
    for rec in recs_sorted:
        rd = recdatas[rec]
        rows, prows, stats = sample_negatives(
            rd, quotas[rec], edge_frac=a.edge_frac, min_dist_units=a.min_dist_units,
            min_height_units=a.min_height_units, min_cams=a.min_cams,
            require_tracked=a.require_tracked, rng=rng, max_tries=a.max_tries,
            partner_deltas=partner_deltas)
        rows_by_rec[rec] = rows
        partner_rows_by_rec[rec] = prows
        stats_by_rec[rec] = stats
        print(f"[extract_empty] {rec} ({rd.source}): quota {quotas[rec]} -> {len(rows)} anchors "
              f"+ {len(prows)} partners (edge {stats['edge_taken']}/{stats['edge_want']}, "
              f"interior {stats['interior_taken']}, shortfall {stats['shortfall']}, rejections "
              f"{stats['rejections']}, partners {stats['partners']})", flush=True)

    # A recording whose OWN tracked footprint is too small clears the gates
    # nowhere (measured 2026-09-05: 2026_04_02_17_52_50's bounding box is only
    # 5.6 units tall, below --min-height-units alone) and legitimately
    # shortfalls at any quota; recordings with spare capacity (their own
    # shortfall was 0, meaning the gates barely bit) take up the slack so the
    # total stays close to --n instead of silently losing one recording's
    # share, one round of bumped quotas -- not a retry loop, so an unlucky
    # donor can leave a residual shortfall, reported honestly either way.
    total_shortfall = sum(s["shortfall"] for s in stats_by_rec.values())
    if total_shortfall > 0:
        donors = [rec for rec in recs_sorted if stats_by_rec[rec]["shortfall"] == 0]
        if donors:
            bonus, rem = divmod(total_shortfall, len(donors))
            print(f"[extract_empty] redistributing {total_shortfall} shortfall across "
                  f"{len(donors)} recording(s) with spare capacity: {donors}", flush=True)
            for i, rec in enumerate(donors):
                extra = bonus + (1 if i < rem else 0)
                if extra <= 0:
                    continue
                rd = recdatas[rec]
                new_quota = quotas[rec] + extra
                rows, prows, stats = sample_negatives(
                    rd, new_quota, edge_frac=a.edge_frac, min_dist_units=a.min_dist_units,
                    min_height_units=a.min_height_units, min_cams=a.min_cams,
                    require_tracked=a.require_tracked, rng=rng, max_tries=a.max_tries,
                    partner_deltas=partner_deltas)
                rows_by_rec[rec] = rows
                partner_rows_by_rec[rec] = prows
                stats_by_rec[rec] = stats
                quotas[rec] = new_quota
                print(f"[extract_empty] {rec}: quota bumped {quotas[rec] - extra} -> "
                      f"{new_quota} -> got {len(rows)} anchors + {len(prows)} partners "
                      f"(shortfall {stats['shortfall']})", flush=True)

    # A recording whose CenterDetect scan found fewer qualifying false peaks
    # than its own edge quota hands the difference to recordings whose scan
    # found MORE than they can use, so the export's overall edge share stays
    # near --edge-frac instead of tracking the weakest recording (Task 5 fix
    # round 1 brief). The recording's own anchor quota is unchanged -- it just
    # fills more of it from the interior stratum.
    realised = sum(len(v) for v in rows_by_rec.values())
    edge_taken = sum(s["edge_taken"] for s in stats_by_rec.values())
    edge_deficit = int(round(a.edge_frac * realised)) - edge_taken
    if edge_deficit > 0:
        spare = {rec: len(recdatas[rec].edge_candidates) - stats_by_rec[rec]["edge_taken"]
                 for rec in recs_sorted}
        donors = sorted((rec for rec in recs_sorted if spare[rec] > 0),
                        key=lambda r: -spare[r])
        if donors:
            print(f"[extract_empty] edge deficit {edge_deficit} "
                  f"({edge_taken}/{int(round(a.edge_frac * realised))}); donors with spare "
                  f"candidates: {[(r, spare[r]) for r in donors]}", flush=True)
            left = edge_deficit
            for rec in donors:
                if left <= 0:
                    break
                extra = min(left, spare[rec])
                rd = recdatas[rec]
                new_edge = stats_by_rec[rec]["edge_taken"] + extra
                rows, prows, stats = sample_negatives(
                    rd, quotas[rec], edge_want=new_edge, min_dist_units=a.min_dist_units,
                    min_height_units=a.min_height_units, min_cams=a.min_cams,
                    require_tracked=a.require_tracked, rng=rng, max_tries=a.max_tries,
                    partner_deltas=partner_deltas)
                gained = stats["edge_taken"] - stats_by_rec[rec]["edge_taken"]
                rows_by_rec[rec] = rows
                partner_rows_by_rec[rec] = prows
                stats_by_rec[rec] = stats
                left -= max(0, gained)
                print(f"[extract_empty] {rec}: edge quota raised to {new_edge} -> "
                      f"{stats['edge_taken']} edge + {stats['interior_taken']} interior "
                      f"({len(rows)} anchors, {len(prows)} partners)", flush=True)
            if left > 0:
                print(f"[extract_empty] edge deficit {left} UNMET -- no recording has "
                      f"spare qualifying false peaks; the shortfall is filled with "
                      f"interior negatives and reported", flush=True)
        else:
            # Every recording's whole candidate pool is already used, so there
            # is nothing to move. This is the normal outcome, not an error:
            # measured 2026-09-06, CenterDetect produces a qualifying false
            # peak on 0.36 % of frames where both flies are tracked (28 in
            # 7,808), so a 25 % edge share is not reachable at any quota. The
            # remainder is interior, and the realised share is reported.
            print(f"[extract_empty] edge deficit {edge_deficit}: every recording's "
                  f"false-peak pool is fully used ({edge_taken} of "
                  f"{int(round(a.edge_frac * realised))} wanted); the rest of the quota "
                  f"is interior negatives", flush=True)

    total = sum(len(v) for v in rows_by_rec.values())
    total_partners = sum(len(v) for v in partner_rows_by_rec.values())
    print(f"[extract_empty] {total} anchor negatives + {total_partners} T=2 partners across "
          f"{len(recs_sorted)} recordings (requested {a.n} anchors)", flush=True)

    # ---- records
    export_names = json.load(open(os.path.join(a.export_names_from, "annotations",
                                               "keypoint_names.json")))
    K = len(export_names)
    records = []
    for rec in recs_sorted:
        rd = recdatas[rec]
        C = len(rd.cameras)
        # unused for a negative record (pseudo_export skips straight to the
        # zero-annotation branch) -- placeholders only, minimally shaped.
        kp3d0, kp2d0 = np.zeros((1, K, 3), np.float32), np.zeros((1, C, K, 2), np.float32)
        vis0, sex0 = np.zeros((1, C, K), bool), np.array([-1], np.int8)
        for row in rows_by_rec[rec]:
            records.append(PseudoRecord(
                recording=rec, frame=int(row["frame"]), host_fly=None,
                kp3d=kp3d0, kp2d=kp2d0, vis=vis0, sex=sex0,
                stratum=row["stratum"], gates=row["gates"],
                partners=row.get("partners", {}), role="negative",
                center3D=np.asarray(row["center3D"], np.float64)))
        for row in partner_rows_by_rec[rec]:
            records.append(PseudoRecord(
                recording=rec, frame=int(row["frame"]), host_fly=None,
                kp3d=kp3d0, kp2d=kp2d0, vis=vis0, sex=sex0,
                stratum=row["stratum"], gates=row["gates"], partners={}, role="partner",
                center3D=np.asarray(row["center3D"], np.float64)))

    if not records:
        raise SystemExit("no negatives were drawn at all -- nothing to write")

    recordings = {rec: {"calib_dir": recdatas[rec].calib_dir, "calib_group": rec,
                        "fly_sex": {}, "kp_names": export_names, "n_flies": a.num_animals,
                        "behavior": "courtship", "sex": "mixed",
                        "sex_source": recdatas[rec].source}
                  for rec in recs_sorted}

    checkpoints = sorted({recdatas[rec].checkpoint for rec in recs_sorted})
    summary = write_pseudo_export(
        a.out, records, export_names=export_names, cameras=cameras, recordings=recordings,
        checkpoint=checkpoints[0] if checkpoints else "unknown",
        gates={"min_dist_units": a.min_dist_units, "min_height_units": a.min_height_units,
              "min_cams": a.min_cams, "edge_frac": a.edge_frac, "edge_units": a.edge_units,
              "cd_peaks": a.cd_peaks, "cd_min_score": a.cd_min_score,
              "cd_max_animals": a.cd_max_animals, "cd_min_views": a.cd_min_views,
              "cd_max_resid_px": a.cd_max_resid_px, "bbox_inflate": a.bbox_inflate,
              "require_tracked": bool(a.require_tracked),
              "edge_allow_untracked": bool(a.edge_allow_untracked)},
        weight=a.weight, frame_reader=frames, mask_reader=None,
        write_images=not a.no_images,
        extra_manifest={"role": "negative", "partner_deltas": list(partner_deltas),
                        "sources": {rec: {"kind": recdatas[rec].source,
                                         "path": recdatas[rec].source_path}
                                   for rec in recs_sorted},
                        "quotas": quotas, "skipped": skipped,
                        "checkpoints": checkpoints,
                        "centerdetect": a.centerdetect if a.edge_frac > 0 else None,
                        "edge_scan": scan_stats,
                        "per_recording_stats": stats_by_rec})

    # ---- figure (Step 4): read this back before committing (CLAUDE.md)
    fig_path = os.path.join(a.figures_dir, "negatives_check.png")
    fig_summary = None
    try:
        fig_summary = write_contact_sheet(fig_path, rows_by_rec, recdatas, frames,
                                          n_panels=a.n_figure_panels,
                                          min_edge_panels=a.n_figure_edge_panels, seed=a.seed)
        print(f"[extract_empty] figure: {fig_path} ({fig_summary['n_panels']} panels, "
              f"{fig_summary['n_edge_panels']} edge)", flush=True)
    except Exception as e:
        print(f"[extract_empty] figure FAILED ({type(e).__name__}: {e}); export still written",
              flush=True)
    frames.close()

    report = {"total_anchors": total, "total_partners": total_partners,
             "total_edge_anchors": sum(s["edge_taken"] for s in stats_by_rec.values()),
             "total_interior_anchors": sum(s["interior_taken"] for s in stats_by_rec.values()),
             "partner_deltas": list(partner_deltas), "requested": a.n, "quotas": quotas,
             "weight": a.weight, "edge_frac": a.edge_frac,
             "per_recording": stats_by_rec, "edge_scan": scan_stats, "skipped": skipped,
             "summary": summary, "figure": fig_path, "figure_panels": fig_summary,
             "seconds": round(time.time() - t0, 1)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "negatives_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)

    if not a.no_verify:
        try:
            report["verify"] = verify_negative_export(a.out, n=min(20, total))
        except Exception as e:
            print(f"[extract_empty] verify FAILED ({type(e).__name__}: {e}) -- export still "
                  f"written; see the concern in the report", flush=True)
            report["verify"] = {"error": f"{type(e).__name__}: {e}"}
        with open(os.path.join(a.out, "negatives_report.json"), "w") as f:
            json.dump(report, f, indent=1, default=float)

    # The figures/ copy is written LAST, after the verify result is in the
    # report (review fix): it is the copy that survives beside the PNG when
    # the export root is regenerated, and a copy missing the loader round trip
    # is the one someone would read as "verify was never run".
    os.makedirs(a.figures_dir, exist_ok=True)
    with open(os.path.join(a.figures_dir, "negatives_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)

    print(f"[extract_empty] wrote {summary['n_framesets']} negative framesets / "
          f"{summary['n_images']} images to {a.out} in {time.time() - t0:.0f}s", flush=True)
    return report


if __name__ == "__main__":
    main()
