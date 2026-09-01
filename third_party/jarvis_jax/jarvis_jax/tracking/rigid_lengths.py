"""Project triangulated 3-D keypoints onto rigid segment lengths.

WHAT THIS IS NOT. `utils.keypoint_filter._detect_bone_length_outliers` already
exists and is a REJECTER: it NaNs both endpoints of an edge whose length
deviates. And wings are deliberately EXCLUDED from it (and from savgol /
spike smoothing, via `preserve_raw_patterns: ["Wing"]`) because temporal
smoothing halved the wing speed of a singing male. So wing keypoints currently
reach STAC with no length constraint at all.

WHY A PROJECTION IS DIFFERENT. This is per-frame geometry, not a temporal
filter. It preserves arbitrarily fast motion exactly -- it only corrects the
RADIAL component along each segment, leaving the direction (which multi-view
triangulation gets right) untouched. That is precisely where the wing error
lives: measured on bout_00001 fly0, the 2-D detections sit ON the wing outline
(WingL_V12 0.020 of fly size from the silhouette edge, conf 0.96, 7/7 views)
yet the triangulated 3-D cannot reproject onto them (10.93 px median vs 4.64 px
for WingL_V13) because one camera drags V12 ~30% of the blade chord inboard
ALONG the vein. A length constraint pins the radius back without touching the
direction or the timing.

THE RISK, named because it is the same shape as the bug that motivated this
module: if a target length is estimated from bad frames, the projection moves
every point confidently to the WRONG radius, and the resulting vein-length CV
will look perfect (it is perfect by construction). So `estimate_bone_lengths`
gates its input, reports n/CV/spread per edge, and REFUSES an edge it cannot
measure -- and the acceptance test is 2-D reprojection error against the
detector, never the length CV.
"""
from __future__ import annotations

import numpy as np


def estimate_bone_lengths(kp3d, conf3d, kp_names, edges, *, min_conf=0.5,
                          min_frames=50, max_cv=0.35):
    """Measure each edge's rigid length. Returns (targets, report).

    `targets[e]` is the median length over frames where BOTH endpoints are
    finite and above `min_conf`; None for an edge with fewer than `min_frames`
    such frames, or whose own CV exceeds `max_cv` (a segment that variable is
    not behaving rigidly, so a target measured from it would be fiction).
    """
    kp3d = np.asarray(kp3d, float)
    conf3d = np.asarray(conf3d, float)
    targets, report = {}, {}
    for ei, (a, b) in enumerate(np.asarray(edges)):
        ok = (np.isfinite(kp3d[:, a]).all(-1) & np.isfinite(kp3d[:, b]).all(-1)
              & (conf3d[:, a] > min_conf) & (conf3d[:, b] > min_conf))
        L = np.linalg.norm(kp3d[ok, a] - kp3d[ok, b], axis=-1)
        L = L[np.isfinite(L) & (L > 0)]
        name = f"{kp_names[a]}->{kp_names[b]}"
        if L.size < min_frames:
            targets[ei] = None
            report[name] = dict(n=int(L.size), target=None,
                                reason=f"only {L.size} usable frames < {min_frames}")
            continue
        med = float(np.median(L))
        cv = float(np.std(L) / med) if med > 0 else np.inf
        if cv > max_cv:
            targets[ei] = None
            report[name] = dict(n=int(L.size), target=None, cv=cv,
                                reason=f"CV {cv:.2f} > max_cv {max_cv} -- not rigid "
                                       f"enough to define a target")
            continue
        targets[ei] = med
        report[name] = dict(n=int(L.size), target=med, cv=cv,
                            p5=float(np.percentile(L, 5)),
                            p95=float(np.percentile(L, 95)))
    return targets, report


def enforce_bone_lengths(kp3d, conf3d, kp_names, edges, targets, *, iters=1,
                         anchor_patterns=("Scutellum",), min_conf=0.0,
                         max_shift=None, split="distal"):
    """Return a copy of `kp3d` with each edge's length set to its target.

    Per frame, `iters` sweeps over `edges` IN THE GIVEN ORDER, which must be
    proximal->distal along each chain (`courtship_skeleton_edges` already is).

    `split` decides which endpoint absorbs the correction:
      'distal'     -- move only edge[1]. Correct for a CHAIN and exact in ONE
                      sweep: each edge's proximal end has already been placed
                      by the more-proximal edge, so nothing downstream can
                      disturb it. This is the default. For the wing it is also
                      the physically right choice: the hinge is fixed to the
                      thorax and the vein landmarks carry the radial error.
      'confidence' -- both ends move, shares INVERSELY proportional to
                      confidence. Better for an isolated edge, but on a chain a
                      distal edge's correction perturbs the shared middle point
                      and breaks the edge proximal to it, so it needs several
                      `iters` and still only converges approximately.

    Either way the direction is never changed, and no information crosses
    frames -- fast motion (wing song) survives exactly.

    Keypoints matching `anchor_patterns` never move.

    Direction is never changed, and no information crosses frames -- fast
    motion (wing song) survives exactly.

    `max_shift`: if set, an edge whose required correction exceeds this (world
    units) is SKIPPED rather than applied, so a wild triangulation cannot be
    dragged onto the target and made to look plausible.
    """
    kp3d = np.asarray(kp3d, float)
    conf3d = np.asarray(conf3d, float)
    out = kp3d.copy()
    anchored = np.array([any(p in n for p in (anchor_patterns or ()))
                         for n in kp_names], bool)
    edges = np.asarray(edges)
    n_applied = n_skipped = 0

    for t in range(out.shape[0]):
        for _ in range(int(iters)):
            for ei, (a, b) in enumerate(edges):
                L = targets.get(ei)
                if L is None:
                    continue
                pa, pb = out[t, a], out[t, b]
                if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                    continue
                if conf3d[t, a] <= min_conf or conf3d[t, b] <= min_conf:
                    continue
                d = pb - pa
                n = float(np.linalg.norm(d))
                if n <= 1e-9:
                    continue                      # coincident: no axis to move along
                err = n - L
                if max_shift is not None and abs(err) > max_shift:
                    n_skipped += 1
                    continue
                u = d / n
                ca, cb = float(conf3d[t, a]), float(conf3d[t, b])
                if anchored[a] and anchored[b]:
                    continue
                if anchored[a]:
                    sa, sb = 0.0, 1.0
                elif anchored[b]:
                    sa, sb = 1.0, 0.0
                elif split == "distal":
                    sa, sb = 0.0, 1.0
                elif split == "confidence":
                    s = ca + cb
                    sa, sb = ((cb / s, ca / s) if s > 0 else (0.5, 0.5))
                else:
                    raise ValueError(
                        f"split must be 'distal' or 'confidence', got {split!r}")
                out[t, a] = pa + u * (err * sa)
                out[t, b] = pb - u * (err * sb)
                n_applied += 1
    return out, dict(n_applied=int(n_applied), n_skipped_max_shift=int(n_skipped))


def segment_length_report(kp3d, kp_names, edges):
    """Per-edge (mean, cv) -- for BEFORE/AFTER tables. Remember that a small CV
    after enforcement is tautological; judge by reprojection error instead."""
    kp3d = np.asarray(kp3d, float)
    out = {}
    for (a, b) in np.asarray(edges):
        L = np.linalg.norm(kp3d[:, a] - kp3d[:, b], axis=-1)
        L = L[np.isfinite(L)]
        if L.size:
            out[f"{kp_names[a]}->{kp_names[b]}"] = (float(L.mean()),
                                                    float(L.std() / L.mean()))
    return out


def repair_flipped_segments(kp3d, conf3d, kp_names, edges, targets, *,
                            rel_tol=0.5, max_gap=5, also_nan=()):
    """NaN frames whose rigid segment length is impossible, then interpolate.

    This is the narrow companion to `enforce_bone_lengths`, and it exists
    because that one CANNOT fix the failure it was built for. The residual wing
    defect on Session0 bout 28 fly1 is an ANGULAR error: on frames 3/10/11 the
    3-D WingL_V12-V13 length reads ~21.6u against a measured 4.8u, because a
    landmark sits a whole wing away. Projecting onto the right LENGTH just
    slides it along a wrong direction (measured: WingL_V12's 13 jumps stayed
    13). And the fit cannot reveal it either -- the model's wing is rigid, so
    outputs.h5 reports a perfect vein length while qpos[7] swings 0.7 rad
    (~40 deg) and the fitted site jumps 13-17 mm against a 0.33 mm baseline.

    So: do not try to correct those frames, DELETE them. A frame whose rigid
    length is off by `rel_tol` is not a measurement, and 3 bad frames bracketed
    by good neighbours is exactly what interpolation is for.

    Deliberately narrow, because wings are excluded from the Stage-B2 smoothing
    (`preserve_raw_patterns`) to protect the song and this must not smuggle
    smoothing back in:
      * only frames FLAGGED by the invariant are touched; every other frame is
        returned byte-identical;
      * only the endpoints of the violating edge (plus `also_nan`) move;
      * a run longer than `max_gap` is REFUSED, not interpolated -- a long
        violation is a regime, not a spike, and filling it would fabricate
        data. Those frames are left NaN so downstream sees a gap, not a guess;
      * a run touching the start or end of the bout has no bracket and is also
        left NaN.

    Returns (kp3d_repaired, report).
    """
    kp3d = np.asarray(kp3d, float).copy()
    T = kp3d.shape[0]
    report = {}
    for ei, (a, b) in enumerate(np.asarray(edges)):
        L = targets.get(ei)
        if L is None or L <= 0:
            continue
        cur = np.linalg.norm(kp3d[:, a] - kp3d[:, b], axis=-1)
        bad = np.isfinite(cur) & (np.abs(cur - L) / L > rel_tol)
        if not bad.any():
            continue
        name = f"{kp_names[a]}->{kp_names[b]}"
        idxs = [kp_names.index(n) for n in also_nan] + [int(a), int(b)]
        # contiguous runs of flagged frames
        runs, s = [], None
        for t in range(T + 1):
            f = bool(bad[t]) if t < T else False
            if f and s is None:
                s = t
            elif not f and s is not None:
                runs.append((s, t)); s = None
        filled = refused = 0
        for (s0, s1) in runs:
            n = s1 - s0
            if n > max_gap or s0 == 0 or s1 >= T:
                # no bracket, or too long to be a spike -> leave NaN
                for i in idxs:
                    kp3d[s0:s1, i] = np.nan
                refused += n
                continue
            for i in idxs:
                lo, hi = kp3d[s0 - 1, i], kp3d[s1, i]
                if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
                    kp3d[s0:s1, i] = np.nan
                    continue
                w = (np.arange(1, n + 1) / (n + 1.0))[:, None]
                kp3d[s0:s1, i] = lo[None, :] * (1 - w) + hi[None, :] * w
            filled += n
        report[name] = dict(n_flagged=int(bad.sum()), n_runs=len(runs),
                            n_interpolated=int(filled), n_left_nan=int(refused),
                            target=float(L), rel_tol=float(rel_tol),
                            frames=np.flatnonzero(bad).tolist()[:40])
    return kp3d, report
