"""Decide wing L/R correspondence ACROSS views before triangulating.

THE FAILURE THIS FIXES. Each camera labels its own two wings independently, and
on some frames a view puts both wing labels on ONE wing. Measured on Session0
bout 28 fly1 (male), frames 0-299: on the 11 frames whose 3-D WingL_V12-V13
length flips from ~4.8u to ~21.6u, FOUR of seven cameras (631, 855, 857, 861)
collapse |WingL_V12 - WingR_V12| to 0.30-0.54 body lengths while 630/853/862
keep them 2.0-2.6 apart. Camera 855 collapses them on 94% of ALL frames.

Consensus triangulation cannot help: with 4 of 7 views wrong it follows the
majority and its reproj_resid_px gate throws out the three CORRECT views. And a
rigid-length constraint cannot help either -- this is an angular/identity error,
not a radial one (verified: enforcing the vein length left WingL_V12's 13 jumps
at 13).

THE FIX. Stop trusting per-view labels. A whole-wing swap in one view is a
binary unknown, so for each frame search the flip pattern over cameras that
makes the two 3-D wings most consistent with ALL views, then relabel the 2-D
accordingly. Geometry decides the correspondence; the count does not.

Two deliberate choices:
  * scoring triangulates with NO consensus gating, because gating is exactly
    what discards the correct minority -- the whole point is to let three good
    views outweigh four bad ones on geometry.
  * the search covers all 2**C patterns (no camera is privileged as a
    reference), and ties -- a pattern and its global complement always score
    identically, since swapping every view just renames left and right -- are
    broken by FEWEST FLIPS, which keeps the detector's global L/R convention.
    That is a weak use of the majority (global naming only, not per-frame
    correspondence) and is what `anchor_cameras` can override.
"""
from __future__ import annotations

import itertools

import numpy as np

WING_GROUPS = (("WingL_base", "WingL_V12", "WingL_V13"),
               ("WingR_base", "WingR_V12", "WingR_V13"))


def _project(kp3d, cam_mats):
    """(T,K,3) + (C,4,3) DLT matrices -> (T,C,K,2) pixels."""
    P = np.asarray(cam_mats, np.float64)                     # (C,4,3)
    h = np.concatenate([np.asarray(kp3d, np.float64),
                        np.ones(kp3d.shape[:-1] + (1,))], -1)   # (T,K,4)
    uvw = np.einsum("tkj,cjm->tckm", h, P)                    # (T,C,K,3)
    return uvw[..., :2] / np.maximum(uvw[..., 2:3], 1e-9)


def resolve_wing_lr(kp2d, conf2d, cam_mats, kp_names, *, groups=WING_GROUPS,
                    conf_thresh=0.3, anchor_cameras=(), max_cameras=10):
    """Return (kp2d_fixed, conf2d_fixed, report).

    For every frame, choose which cameras have their wing labels swapped so the
    reconstructed pair best explains all views, then swap those views' 2-D wing
    labels. Downstream triangulation is unchanged -- it just receives 2-D whose
    L/R correspondence is consistent across cameras.

    `anchor_cameras`: names or indices held un-flipped (use for a camera you
    trust; leave empty to let geometry decide with the fewest-flips tie-break).
    """
    from jarvis_jax.tracking.triangulate import triangulate_keypoints

    kp2d = np.asarray(kp2d, np.float32).copy()
    conf2d = np.asarray(conf2d, np.float32).copy()
    T, C = kp2d.shape[:2]
    if C > max_cameras:
        raise ValueError(f"{C} cameras -> 2**{C} patterns; raise max_cameras "
                         f"deliberately if you really want that")
    li = [kp_names.index(n) for n in groups[0]]
    ri = [kp_names.index(n) for n in groups[1]]
    idx = np.array(li + ri)                                   # (2G,)
    G = len(li)
    swapped = np.array(ri + li)                               # same points, L<->R

    anchor_ix = {int(a) for a in (anchor_cameras or ())
                 if isinstance(a, (int, np.integer))}

    patterns = [p for p in itertools.product((0, 1), repeat=C)
                if not any(p[a] for a in anchor_ix)]
    best_cost = np.full(T, np.inf, np.float64)
    best_pat = np.zeros((T, C), np.int8)
    best_nflip = np.full(T, C + 1, np.int16)

    for pat in patterns:
        pat = np.asarray(pat, bool)
        # per-camera index list: swapped where this pattern flips that view
        take = np.where(pat[:, None], swapped[None, :], idx[None, :])   # (C,2G)
        sel = np.broadcast_to(take[None], (T, C, 2 * G))
        obs = np.take_along_axis(kp2d, sel[..., None].repeat(2, axis=-1), axis=2)
        cf = np.take_along_axis(conf2d, sel, axis=2)
        # NO consensus gating here -- gating is what discards the correct minority.
        k3, c3 = triangulate_keypoints(obs, cf, cam_mats, conf_thresh=conf_thresh,
                                       view_conf_thresh=None, reproj_resid_px=None)
        pred = _project(k3, cam_mats)                          # (T,C,2G,2)
        d = np.linalg.norm(pred - obs, axis=-1)                # (T,C,2G)
        m = (cf > conf_thresh) & np.isfinite(d)
        cost = np.where(m.any(axis=(1, 2)),
                        np.nansum(np.where(m, d, 0.0), axis=(1, 2))
                        / np.maximum(m.sum(axis=(1, 2)), 1), np.inf)
        nflip = int(pat.sum())
        better = (cost < best_cost - 1e-9) | (
            (np.abs(cost - best_cost) <= 1e-9) & (nflip < best_nflip))
        best_cost = np.where(better, cost, best_cost)
        best_nflip = np.where(better, nflip, best_nflip)
        best_pat[better] = pat

    # ---- global handedness, resolved TEMPORALLY ----------------------------
    # A pattern and its complement ALWAYS score identically: flipping every view
    # just renames left and right, so geometry cannot choose between them. And
    # "fewest flips" is not a tie-break, it is a bias -- on a frame where 4 of 7
    # views are wrong it prefers the 3-flip complement and renames the wings for
    # that frame alone. (Hamming distance to an all-zero reference IS the flip
    # count, so a majority-vote reference reduces to the same bias.)
    #
    # Temporal continuity does break it, and is the only local information that
    # can: wings move smoothly, so a global L<->R swap makes the left wing jump
    # to where the right one was -- a large, unmistakable discontinuity. Note
    # the complement's 3-D needs no re-triangulation: it is exactly this
    # solution with L and R exchanged.
    #
    # Anchor on the frame needing the fewest flips (an unambiguous frame), then
    # sweep outward choosing, at each frame, the option closer to its already
    # resolved neighbour.
    take = np.where(best_pat[:, :, None].astype(bool),
                    swapped[None, None, :], idx[None, None, :])
    sel = np.broadcast_to(take, (T, C, 2 * G))
    obs = np.take_along_axis(kp2d, sel[..., None].repeat(2, axis=-1), axis=2)
    cf = np.take_along_axis(conf2d, sel, axis=2)
    k3, _ = triangulate_keypoints(obs, cf, cam_mats, conf_thresh=conf_thresh,
                                  view_conf_thresh=None, reproj_resid_px=None)
    L3, R3 = k3[:, :G, :], k3[:, G:, :]                       # (T,G,3) each
    comp = np.zeros(T, bool)                                  # take the complement?
    anchor = int(np.argmin(best_pat.sum(1)))

    def _d(a, b):
        v = np.linalg.norm(a - b, axis=-1)
        return float(np.nansum(v)) if np.isfinite(v).any() else np.inf

    for step in (1, -1):
        t = anchor
        while 0 <= t + step < T:
            prv, t = t, t + step
            pL = R3[prv] if comp[prv] else L3[prv]
            pR = L3[prv] if comp[prv] else R3[prv]
            keep = _d(L3[t], pL) + _d(R3[t], pR)
            flip = _d(R3[t], pL) + _d(L3[t], pR)
            comp[t] = flip < keep
    if anchor_ix:
        a = np.array(sorted(anchor_ix))
        comp &= ~((1 - best_pat[:, a]).any(axis=1))
    best_pat[comp] = 1 - best_pat[comp]

    # apply
    n_swaps = 0
    for c in range(C):
        f = best_pat[:, c].astype(bool)
        if not f.any():
            continue
        n_swaps += int(f.sum())
        t_ix = np.flatnonzero(f)
        for a, b in zip(li, ri):
            # explicit 3-way swap: kp2d[f, c, [a,b]] = kp2d[f, c, [b,a]] does
            # NOT broadcast (bool mask + index list) and raises IndexError.
            tmp = kp2d[t_ix, c, a].copy()
            kp2d[t_ix, c, a] = kp2d[t_ix, c, b]
            kp2d[t_ix, c, b] = tmp
            ctmp = conf2d[t_ix, c, a].copy()
            conf2d[t_ix, c, a] = conf2d[t_ix, c, b]
            conf2d[t_ix, c, b] = ctmp
    report = dict(
        n_frames=int(T), n_cameras=int(C), n_view_swaps=int(n_swaps),
        anchor_frame=int(anchor),
        frames_complemented_for_temporal_consistency=int(comp.sum()),
        frames_with_any_swap=int((best_pat.sum(1) > 0).sum()),
        per_camera_swap_rate={i: float(best_pat[:, i].mean()) for i in range(C)},
        median_cost_px=float(np.median(best_cost[np.isfinite(best_cost)]))
        if np.isfinite(best_cost).any() else None)
    return kp2d, conf2d, report


def detect_wing_lr_collapse(kp2d, conf2d, kp_names, *, groups=WING_GROUPS,
                            body_pair=("Scutellum", "Abd_tip"),
                            abs_floor=0.6, rel_frac=0.35, conf_thresh=0.3):
    """Which (frame, camera) put BOTH wing labels on the SAME wing?

    A collapse is not a swap. If a view's WingL_V12 and WingR_V12 sit on top of
    each other, permuting the labels changes nothing -- that view carries no L/R
    information at all, so no cross-view assignment can repair it (verified:
    ``resolve_wing_lr`` fires on 2 of 1500 male frames and leaves 11 of 13
    jumps). The repair is to stop letting that view vote.

    Measured on Session0 bout 28 fly1, |WingL_V12 - WingR_V12| / body length on
    the 11 flipped-vein frames: 630/853/862 = 2.04/2.50/2.64 (wings resolved),
    631/855/857/861 = 0.54/0.30/0.30/0.37 (collapsed). Camera 855 is collapsed
    on 94% of ALL frames. Clean separation, so a threshold works.

    Flagged when EITHER
      * separation < `abs_floor` body lengths (an absolute collapse), OR
      * separation < `rel_frac` x that camera's OWN median separation (a view
        that normally resolves the wings but has stopped on this frame).
    The second catches 631/857/861 (1.5-2.1 normally, 0.3 on flip frames); the
    first catches 855, which never resolves them.

    Note this deliberately does NOT try to tell a detector error from a genuine
    projective overlap (a near-lateral view where the wings really do superpose).
    Either way the view cannot inform L/R, so excluding it is right in both
    cases -- which is why the criterion needs no appeal to intent.

    Returns (collapsed (T,C) bool, info).
    """
    kp2d = np.asarray(kp2d, np.float64)
    conf2d = np.asarray(conf2d, np.float64)
    T, C = kp2d.shape[:2]
    li = [kp_names.index(n) for n in groups[0]]
    ri = [kp_names.index(n) for n in groups[1]]
    ib, ia = kp_names.index(body_pair[0]), kp_names.index(body_pair[1])

    body = np.linalg.norm(kp2d[:, :, ia, :] - kp2d[:, :, ib, :], axis=-1)  # (T,C)
    body = np.where(body > 1.0, body, np.nan)
    # separation of CORRESPONDING landmarks, averaged over the wing's points
    seps = []
    for a, b in zip(li, ri):
        ok = (conf2d[:, :, a] > conf_thresh) & (conf2d[:, :, b] > conf_thresh)
        d = np.linalg.norm(kp2d[:, :, a, :] - kp2d[:, :, b, :], axis=-1) / body
        seps.append(np.where(ok, d, np.nan))
    sep = np.nanmean(np.stack(seps), axis=0)                    # (T,C)

    with np.errstate(invalid="ignore"):
        med = np.nanmedian(sep, axis=0)                         # (C,) per-camera norm
        collapsed = (sep < abs_floor) | (sep < rel_frac * med[None, :])
    collapsed &= np.isfinite(sep)
    info = dict(
        per_camera_median_sep={i: (None if not np.isfinite(med[i]) else float(med[i]))
                               for i in range(C)},
        per_camera_collapse_rate={i: float(collapsed[:, i].mean()) for i in range(C)},
        frames_with_any={"n": int((collapsed.any(1)).sum()), "of": int(T)},
        views_collapsed_hist={k: int((collapsed.sum(1) == k).sum())
                              for k in range(C + 1)})
    return collapsed, info


def mask_collapsed_wing_views(conf2d, collapsed, kp_names, *, groups=WING_GROUPS,
                              min_views_kept=4):
    """Zero the confidence of the wing keypoints in every collapsed view.

    Triangulation then ignores those views for the wings only -- the rest of the
    skeleton keeps all cameras.

    `min_views_kept` is not optional book-keeping, it is the guard that makes
    this safe. A fly with FOLDED wings has both wings genuinely superposed in
    most views -- that is anatomy, not a detector error -- so the collapse test
    fires nearly everywhere. Measured on Session0 bout 28 fly0 (female, wings
    folded): median separations 0.25-0.57 body lengths, cameras 855/857/861
    flagged on 99-100% of frames, 4-6 of 7 views masked, and masking anyway
    turned 117 bad frames into 246 with 88 new all-NaN frames. On a frame where
    fewer than `min_views_kept` views would survive, NOTHING is masked: an
    under-determined triangulation is worse than a biased one, and a fly whose
    wings cannot be told apart in 5 of 7 views has no L/R to recover.

    Returns (conf2d_masked, info).
    """
    conf2d = np.asarray(conf2d).copy()
    collapsed = np.asarray(collapsed).copy()
    C = collapsed.shape[1]
    kept = C - collapsed.sum(1)
    too_few = kept < int(min_views_kept)
    collapsed[too_few] = False                       # leave those frames alone
    wing = [kp_names.index(n) for g in groups for n in g]
    tt, cc = np.where(collapsed)
    for i in wing:
        conf2d[tt, cc, i] = 0.0
    info = dict(frames_masked=int((collapsed.any(1)).sum()),
                frames_left_alone_too_few_views=int(too_few.sum()),
                views_masked=int(collapsed.sum()),
                min_views_kept=int(min_views_kept))
    return conf2d, info
