"""Empty-window negatives for the existence head (spec 2026-09-05 §3.5, Task 5).

Run (real recordings, controller ruling 2026-09-05):

    JAX_PLATFORMS=cpu python scripts/pseudo_labels/extract_empty_windows.py \\
        --tracks 2025_10_20_13_20_04=/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/coarse_mvq_p3b/coarse_tracks.npz \\
        --lift-root 2026_04_02_12_11_50=/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session1/2026_04_02_12_11_50/pose_mvq_p3b \\
        ... (one --lift-root per remaining recording) \\
        --out /gscratch/portia/eabe/data/Johnson_lab/red_data_3d_v12_pseudo_negatives_20260905 \\
        --n 2000

Two centroid sources, one per recording (controller ruling 2026-09-05 --
`coarse_tracks.npz` exists only for Session0/2025_10_20_13_20_04):

  (a) `--tracks REC=coarse_tracks.npz` -- the mask-free coarse pass
      (`jarvis_jax.tracking.coarse_track.coarse_pass`/`write_coarse_tracks`):
      `X3d` (F,T,3), `exist` (F,T), `coarse_frame` (T,), `trackable` (F,T),
      plus the sibling `.meta.json` for `session_dir`/`calib_dir`/`cameras`/
      `W`/`H`. CenterDetect peaks that lifted to a centre but read a typed
      slot with `exist < --exist-edge-thresh` (0.2) ARE available here (the
      false-peak windows §3.5 wants) -- `--edge-frac` applies only to this
      source.
  (b) `--lift-root REC=pose_mvq_p3b` -- the masked p3b campaign's own bout
      lifts: per-frame fly centroids = nanmean over keypoints of
      `bouts/bout_*/fly{0,1}/kp3d.npz`'s `kp3d` (T,K,3), placed at the bout's
      `frame_start` (`mvq_meta.json`), ONLY for frames inside a bout. There
      is no per-frame existence read here and no CenterDetect peak to be a
      false one, so `--edge-frac` is silently 0 for this source and the
      manifest's per-recording `stratum` counts say so.

Sampling, per recording (spec §3.5 / Task 5 brief Step 2-3): draw
`--n-per-recording` (default: `--n` split evenly, remainder to the first
recordings alphabetically) frames uniformly over the recording's available
frames, and per frame reject-sample a centre uniformly inside the
tracked-centroid bounding box (inflated `--bbox-inflate`, 20 %), keeping it
only if it is `>= --min-dist-units` (60 units, 6 mm) from EVERY tracked
centroid of that exact frame, `>= --min-height-units` (6 units) above the
recording's fitted floor (`coarse_track.fit_floor`), and projects inside
`>= --min-cams` (5) cameras (`coarse_centres._project_batch`). Arena-edge
windows (`--edge-frac`, source (a) only) are drawn instead from CenterDetect
reads with `exist < --exist-edge-thresh` whose centre lands within
`--edge-units` of the convex hull of the recording's OWN trackable
centroids -- the false peaks the existence head must learn to refuse
(gate-bout-1 failure mode), not a fresh random draw.

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

EXPECTATION for `figures/2026-09-mvq/v2_pseudo/negatives_check.png` (state it
before looking, CLAUDE.md): a 12-panel contact sheet of drawn empty crops
across >= 2 recordings and both strata (interior + edge, where available) --
every panel shows bare substrate or an arena fixture, and NO FLY. A panel
with a fly's body in it is a mis-sampled negative that would teach the
existence head to suppress a real animal, and the run must not be committed
until this has been read back and confirmed clean (or the offending stratum/
gate is reported as a concern).
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
def _hull_distance(points, hull):
    """(N,3) points -> (N,) DISTANCE to `hull`'s nearest bounding facet plane.

    `hull.equations` is `[A | b]` per facet with `A` unit-normalized by qhull
    so that `A . x + b <= 0` for every point of the hull; the max of that over
    facets is <= 0 strictly inside and > 0 outside, and its magnitude is the
    distance to the nearest supporting hyperplane -- exact when the foot of
    the perpendicular lands within that facet, a slight underestimate near a
    facet's own edge/vertex. That is the standard convex-polytope proxy for
    "distance to the hull surface" and is what "within --edge-units of the
    convex hull" (Task 5 brief step 1c) means here: exact at the recording's
    own hull vertices (which is where the false-peak candidates in practice
    land -- an arena-edge fixture is a physical extreme of the tracked
    cloud), and a lower bound elsewhere, so it never UNDER-restricts the
    "near the edge" test.
    """
    eq = np.asarray(hull.equations, np.float64)
    A, b = eq[:, :-1], eq[:, -1]
    signed = np.max(np.asarray(points, np.float64) @ A.T + b[None, :], axis=1)
    return np.abs(signed)


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
    `--require-tracked`. `edge_candidates` is `[(frame, xyz, hull_dist), ...]`
    and is only ever non-empty for source (a) (controller ruling: source (b)
    has no CenterDetect peaks to be false ones).
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
    edge_candidates: list


def build_recdata_from_tracks(recording, tracks_path, *, bbox_inflate=0.20,
                              edge_units=20.0, exist_edge_thresh=0.2):
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

    edge_candidates = []
    with np.errstate(invalid="ignore"):
        for f in range(F):
            for t in range(T):
                e = exist[f, t]
                if not (np.isfinite(e) and e < exist_edge_thresh):
                    continue
                xyz = X3d[f, t]
                if not np.isfinite(xyz).all():
                    continue
                d = float(_hull_distance(xyz[None], hull)[0])
                if d <= edge_units:
                    edge_candidates.append((int(coarse_frame[t]), xyz.copy(), d))

    return RecData(recording=recording, source="tracks", source_path=tracks_path,
                   session_dir=str(meta.get("session_dir", "")), calib_dir=str(calib_dir),
                   cameras=cameras, cam_mats=cam_mats, W=W, H=H,
                   checkpoint=str(meta.get("checkpoint", "unknown")),
                   frames_avail=coarse_frame, frame_to_cent=frame_to_cent,
                   frame_has_untracked=frame_has_untracked, bbox_lo=bbox_lo, bbox_hi=bbox_hi,
                   floor=floor, edge_candidates=edge_candidates)


def build_recdata_from_liftroot(recording, lift_root, *, bbox_inflate=0.20, num_animals=2):
    """Source (b): the masked p3b campaign's own bout lifts.

    Returns `None` (never raises) when `lift_root` has no finished bouts for
    `recording` -- the caller skips it with a message (controller ruling:
    Session1 lifts are being re-run and some recordings are not done yet)."""
    from jarvis_jax.tracking.coarse_track import fit_floor
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
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
                   floor=floor, edge_candidates=[])


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


def _link_partners(rd, f0, cand, kind, used_frames, *, deltas, min_dist_units,
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
                             "stratum": {"kind": kind}, "gates": info,
                             "role": "partner", "anchor_frame": int(f0), "delta": int(delta)})
        pstats[str(delta)]["taken"] += 1
    return partners, partner_rows


def sample_negatives(rd: RecData, n_want, *, edge_frac=0.25, min_dist_units=60.0,
                     min_height_units=6.0, min_cams=5, require_tracked=False,
                     rng=None, max_tries=300, partner_deltas=DEFAULT_PARTNER_DELTAS):
    """Draw up to `n_want` empty-window ANCHOR negatives for one recording,
    each carrying T=2 partner framesets at `partner_deltas` where they clear
    the gates (spec §3.5 + coordinator ruling 2026-09-06: partners are
    additional framesets layered on the anchor quota, not counted against
    it -- exactly how the positive pseudo-labels' T=2 partners work).

    Returns `(anchor_rows, partner_rows, stats)`. An anchor row is a dict
    with `frame`, `center3D` ((3,) float64 world units), `stratum`
    (`{"kind": "edge"|"interior"}`), `gates` (the quantities that admitted
    it, named) and `partners` (`{delta_str: partner_frame}`, only the deltas
    that cleared -- see `_link_partners`). A partner row has the same shape
    plus `role: "partner"`, `anchor_frame` and `delta`. Every row's `frame`
    -- anchor or partner, from either list -- is unique WITHIN this
    recording (one v12 frameset per (recording, frame); a collision would
    silently merge in `write_pseudo_export`'s per-frame grouping).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n_want = int(n_want)
    n_edge_want = int(round(float(edge_frac) * n_want)) if rd.edge_candidates else 0
    deltas = tuple(int(d) for d in partner_deltas)

    picked, partner_rows, used_frames = [], [], set()
    pstats = {str(d): collections.Counter() for d in deltas}
    stats = {"recording": rd.recording, "source": rd.source, "n_want": n_want,
             "edge_pool": len(rd.edge_candidates), "edge_want": n_edge_want,
             "edge_taken": 0, "interior_taken": 0,
             "rejections": {"duplicate_frame": 0, "dist": 0, "height": 0, "cams": 0},
             "partner_deltas": list(deltas)}

    def _accept(frame, cand, kind, gates):
        used_frames.add(frame)
        p, prows = _link_partners(rd, frame, cand, kind, used_frames, deltas=deltas,
                                  min_dist_units=min_dist_units,
                                  min_height_units=min_height_units, min_cams=min_cams,
                                  pstats=pstats)
        picked.append({"frame": frame, "center3D": np.asarray(cand, np.float64),
                       "stratum": {"kind": kind}, "gates": gates, "role": "negative",
                       "partners": p})
        partner_rows.extend(prows)

    # ---- edge stratum (source (a) only; empty pool otherwise)
    pool = list(rd.edge_candidates)
    rng.shuffle(pool)
    for frame, xyz, hdist in pool:
        if stats["edge_taken"] >= n_edge_want:
            break
        frame = int(frame)
        if frame in used_frames:
            stats["rejections"]["duplicate_frame"] += 1
            continue
        _accept(frame, xyz, "edge", {"hull_dist_units": round(float(hdist), 3)})
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
            _accept(frame, cand, "interior", info)
    stats["interior_taken"] = len(picked) - stats["edge_taken"]
    stats["shortfall"] = max(0, n_want - len(picked))
    stats["n_anchors"] = len(picked)
    stats["n_partners"] = len(partner_rows)
    stats["partners"] = {d: dict(c) for d, c in pstats.items()}
    return picked, partner_rows, stats


# ------------------------------------------------------------------------ figure
def write_contact_sheet(out_path, rows_by_rec, recdatas, frame_reader, *, n_panels=12, seed=0):
    """`figures/2026-09-mvq/v2_pseudo/negatives_check.png`: `n_panels` crops of
    drawn empty windows, spanning recordings and strata where available.

    EXPECTATION (module docstring, restated at the point of use): every panel
    shows bare substrate or an arena fixture and NO fly. Read back with the
    Read tool before committing -- a panel with a fly body in it means a
    stratum or gate threshold let a real animal through.
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
    # recordings/strata so one recording or one kind cannot fill the sheet.
    per_neg = 2
    n_negs = max(1, -(-n_panels // per_neg))
    by_key = collections.defaultdict(list)
    for rec, row in flat:
        by_key[(rec, row["stratum"]["kind"])].append((rec, row))
    keys = sorted(by_key)
    rng.shuffle(keys)
    chosen = []
    ki = 0
    while len(chosen) < n_negs and keys:
        k = keys[ki % len(keys)]
        pool = by_key[k]
        if pool:
            chosen.append(pool.pop(rng.integers(len(pool))))
        else:
            keys.remove(k)
            continue
        ki += 1
        if all(not v for v in by_key.values()):
            break

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
    panels = panels[:n_panels]

    ncols = 4
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (rec, cam, row, (u, v)) in zip(axes, panels):
        frame_no = int(row["frame"])
        frames, present = frame_reader(rec, frame_no)
        ci = recdatas[rec].cameras.index(cam)
        img = frames[ci]
        x0, y0 = crop_origin([float(u), float(v), 0, 0], img.shape[1], img.shape[0], 448)
        crop = img[y0:y0 + 448, x0:x0 + 448]
        ax.imshow(crop)
        kind = row["stratum"]["kind"]
        ax.set_title(f"{rec}\n{cam} Frame_{frame_no} ({kind})", fontsize=8)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle("Empty-window negatives -- every panel should show NO fly", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return {"n_panels": len(panels),
           "panels": [{"recording": rec, "camera": cam, "frame": int(row["frame"]),
                       "kind": row["stratum"]["kind"]} for rec, cam, row, _ in panels]}


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
    ap.add_argument("--edge-frac", type=float, default=0.25)
    ap.add_argument("--edge-units", type=float, default=20.0,
                    help="units from the tracked-centroid convex hull (2 mm); source (a) only")
    ap.add_argument("--exist-edge-thresh", type=float, default=0.2)
    ap.add_argument("--min-dist-units", type=float, default=60.0, help="6 mm")
    ap.add_argument("--min-height-units", type=float, default=6.0)
    ap.add_argument("--min-cams", type=int, default=5)
    ap.add_argument("--bbox-inflate", type=float, default=0.20)
    ap.add_argument("--require-tracked", action="store_true",
                    help="skip a frame entirely if either fly's centroid is NaN there")
    ap.add_argument("--max-tries", type=int, default=300,
                    help="rejection-sampling attempts per candidate frame")
    ap.add_argument("--partner-deltas", default=",".join(str(d) for d in DEFAULT_PARTNER_DELTAS),
                    help="comma list of T=2 partner spacings, forward-only from each anchor "
                         "(coordinator ruling 2026-09-06)")
    ap.add_argument("--num-animals", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight", type=float, default=0.3)
    ap.add_argument("--no-images", action="store_true", help="write the JSON only (schema smoke test)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--figures-dir", default="figures/2026-09-mvq/v2_pseudo")
    ap.add_argument("--n-figure-panels", type=int, default=12)
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
            recdatas[rec] = build_recdata_from_tracks(
                rec, path, bbox_inflate=a.bbox_inflate, edge_units=a.edge_units,
                exist_edge_thresh=a.exist_edge_thresh)
        except Exception as e:
            print(f"[extract_empty] {rec} (tracks): SKIPPED ({type(e).__name__}: {e})", flush=True)
            skipped[rec] = f"{type(e).__name__}: {e}"
            continue
        print(f"[extract_empty] {rec} (tracks): loaded in {time.time() - ts:.1f}s, "
              f"{len(recdatas[rec].edge_candidates)} false-peak candidates", flush=True)
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

    rng = np.random.default_rng(a.seed)
    rows_by_rec, partner_rows_by_rec, stats_by_rec = {}, {}, {}
    for rec in recs_sorted:
        rd = recdatas[rec]
        edge_frac = a.edge_frac if rd.source == "tracks" else 0.0
        rows, prows, stats = sample_negatives(
            rd, quotas[rec], edge_frac=edge_frac, min_dist_units=a.min_dist_units,
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
                edge_frac = a.edge_frac if rd.source == "tracks" else 0.0
                rows, prows, stats = sample_negatives(
                    rd, new_quota, edge_frac=edge_frac, min_dist_units=a.min_dist_units,
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

    cameras = list(recdatas[recs_sorted[0]].cameras)
    for rec in recs_sorted:
        if list(recdatas[rec].cameras) != cameras:
            raise ValueError(f"{rec}: cameras {recdatas[rec].cameras} != the canonical "
                             f"{cameras} (write_pseudo_export needs ONE camera axis)")
    recordings = {rec: {"calib_dir": recdatas[rec].calib_dir, "calib_group": rec,
                        "fly_sex": {}, "kp_names": export_names, "n_flies": a.num_animals,
                        "behavior": "courtship", "sex": "mixed",
                        "sex_source": recdatas[rec].source}
                  for rec in recs_sorted}

    frames = SessionFrames({rec: recdatas[rec].session_dir for rec in recs_sorted}, cameras)
    checkpoints = sorted({recdatas[rec].checkpoint for rec in recs_sorted})
    summary = write_pseudo_export(
        a.out, records, export_names=export_names, cameras=cameras, recordings=recordings,
        checkpoint=checkpoints[0] if checkpoints else "unknown",
        gates={"min_dist_units": a.min_dist_units, "min_height_units": a.min_height_units,
              "min_cams": a.min_cams, "edge_frac": a.edge_frac, "edge_units": a.edge_units,
              "exist_edge_thresh": a.exist_edge_thresh, "bbox_inflate": a.bbox_inflate,
              "require_tracked": bool(a.require_tracked)},
        weight=a.weight, frame_reader=frames, mask_reader=None,
        write_images=not a.no_images,
        extra_manifest={"role": "negative", "partner_deltas": list(partner_deltas),
                        "sources": {rec: {"kind": recdatas[rec].source,
                                         "path": recdatas[rec].source_path}
                                   for rec in recs_sorted},
                        "quotas": quotas, "skipped": skipped,
                        "checkpoints": checkpoints,
                        "per_recording_stats": stats_by_rec})

    # ---- figure (Step 4): read this back before committing (CLAUDE.md)
    fig_path = os.path.join(a.figures_dir, "negatives_check.png")
    fig_summary = None
    try:
        fig_summary = write_contact_sheet(fig_path, rows_by_rec, recdatas, frames,
                                          n_panels=a.n_figure_panels, seed=a.seed)
        print(f"[extract_empty] figure: {fig_path} ({fig_summary['n_panels']} panels)", flush=True)
    except Exception as e:
        print(f"[extract_empty] figure FAILED ({type(e).__name__}: {e}); export still written",
              flush=True)
    frames.close()

    report = {"total_anchors": total, "total_partners": total_partners,
             "partner_deltas": list(partner_deltas), "requested": a.n, "quotas": quotas,
             "per_recording": stats_by_rec, "skipped": skipped,
             "summary": summary, "figure": fig_path, "figure_panels": fig_summary,
             "seconds": round(time.time() - t0, 1)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "negatives_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    os.makedirs(a.figures_dir, exist_ok=True)
    with open(os.path.join(a.figures_dir, "negatives_report.json"), "w") as f:
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

    print(f"[extract_empty] wrote {summary['n_framesets']} negative framesets / "
          f"{summary['n_images']} images to {a.out} in {time.time() - t0:.0f}s", flush=True)
    return report


if __name__ == "__main__":
    main()
