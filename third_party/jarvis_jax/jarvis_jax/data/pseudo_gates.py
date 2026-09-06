"""Admission gates for P3b pseudo-labels (spec 2026-09-05 §3.2). Pure numpy
over ONE bout's on-disk arrays: no model, no GPU, so the whole gate stack is
unit-testable. Thresholds live in `GateThresholds` and nowhere else -- the
extractor, the gallery and the notes all read them from the same object, and
the resolved values are written into the export manifest's `gates` block.

Anchors and partners are separate concepts. `admit_bout.frame` is the set of
DECORRELATED ANCHOR frames (>= `decorrelation` frames apart); a T=2 partner
frame at `Delta` from an anchor is never itself an anchor and never enters
the spacing count -- Task 2 derives it directly from `partners[delta]` at
each anchor `a` (forward: `partners[delta][a]`; backward:
`partners[delta][a - delta]`). `partner_masks` only checks the two named
frames of the pair (endpoint semantics, per spec §3.2's "when those frames
also pass the gates") -- the model at training time is shown just those two
frames, never anything in between.

The containment gate is EXPECTED to reject flies with a poor mask fit (e.g.
the female fly on some bouts, see docs/benchmark/2026-09-mvq/
v2-pseudolabel-notes.md) -- there is no adaptive per-fly threshold here.
Task 2 must run a per-stratum yield census across all 160 bouts before
sampling, rather than assume any single bout yields a balanced female/male
draw.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import NamedTuple

import numpy as np

MM_PER_UNIT = 0.1


@dataclasses.dataclass(frozen=True)
class GateThresholds:
    exist_min: float = 0.8            # spec §3.2, per present fly
    max_step_units: float = 3.0       # 0.3 mm per keypoint per frame
    reproj_px: float = 3.0            # per-view median over keypoints
    reproj_min_views: int = 5
    contain_frac: float = 0.9
    contain_min_cams: int = 5
    contain_dilate_px: float = 6.0
    decorrelation: int = 16
    deltas: tuple = (1, 4, 16)
    conf_min: float = 0.5             # per-view visibility below this is not scored


class BoutArrays(NamedTuple):
    kp3d: np.ndarray        # (F,T,K,3) MODEL order (what lift_mvq wrote)
    kp2d: np.ndarray        # (F,T,C,K,2) full-frame px, canonical camera order
    conf: np.ndarray        # (F,T,C,K) per-view visibility
    kp_names: list
    cameras: list
    meta: dict


class GateResult(NamedTuple):
    frame: np.ndarray                  # (T,) anchor frames after decorrelation
    fly: np.ndarray                    # (F,T) per-fly admission (False = absent/rejected)
    reasons: dict                      # gate name -> bool array, True = THIS gate rejected
    quant: dict                        # gate name -> the raw quantity, for histograms
    partners: dict                     # delta -> (T,) bool, both frames admitted


def load_bout_arrays(bout_dir, *, cameras, n_flies=2):
    """Read `fly{0..n}/kp{2,3}d.npz` + `mvq_meta.json`, checking the keypoint
    axis by NAME across flies and the camera axis by NAME against `cameras`."""
    kp3d, kp2d, conf, names = [], [], [], None
    for f in range(n_flies):
        z3 = np.load(os.path.join(bout_dir, f"fly{f}", "kp3d.npz"), allow_pickle=True)
        z2 = np.load(os.path.join(bout_dir, f"fly{f}", "kp2d.npz"), allow_pickle=True)
        n3 = [str(s) for s in z3["kp_names"]]
        if names is None:
            names = n3
        elif n3 != names:
            raise ValueError(f"{bout_dir} fly{f}: kp_names differ between flies")
        cam = [str(c) for c in z2["cameras"]]
        if cam != [str(c) for c in cameras]:
            raise ValueError(f"{bout_dir} fly{f}: kp2d camera order {cam} != canonical {list(cameras)}")
        kp3d.append(z3["kp3d"]); kp2d.append(z2["kp2d"]); conf.append(z2["conf"])
    meta = json.load(open(os.path.join(bout_dir, "mvq_meta.json")))
    return BoutArrays(np.stack(kp3d), np.stack(kp2d), np.stack(conf),
                      names, [str(c) for c in cameras], meta)


def existence_gate(meta, thr):
    e = np.asarray(meta["per_frame"]["exist"], np.float32)          # (F,T), -1 = no instance
    return e >= thr.exist_min, e


def identity_gate(meta, thr):
    """(T,) True where the frame is REJECTED: not mask identity, collapsed, or
    any containment drop. `identity_source` is per frame in `mvq_meta.json`."""
    pf = meta["per_frame"]; T = int(meta["n_frames"])
    src = np.array([s == "mask" for s in pf["identity_source"]], bool)
    collapsed = np.asarray(pf["collapsed"], bool)
    rep = (meta.get("containment_report") or {}).get("per_frame") or {}
    drops = np.asarray(rep.get("n_dropped", np.zeros((2, T)))).sum(0) > 0
    return ~(src & ~collapsed & ~drops)


def step_gate(kp3d, thr):
    """(F,T) admitted, and (F,T) the max per-keypoint step in units. A frame is
    rejected if EITHER neighbour step exceeds the threshold (a spike is bad in
    both frames of the pair it appears in)."""
    d = np.linalg.norm(np.diff(kp3d, axis=1), axis=-1)             # (F,T-1,K)
    with np.errstate(invalid="ignore"):
        s = np.nanmax(d, axis=-1)
    prev = np.concatenate([np.zeros((kp3d.shape[0], 1)), s], 1)
    nxt = np.concatenate([s, np.zeros((kp3d.shape[0], 1))], 1)
    worst = np.fmax(prev, nxt)
    return worst <= thr.max_step_units, worst


def reprojection_gate(kp2d, kp3d, cam_mats, conf, thr):
    """(F,T) admitted and (F,T,C) per-view median |2D head - reprojected 3D| px."""
    from jarvis_jax.tracking.lift_mvq import project_points
    F, T, C, K, _ = kp2d.shape
    med = np.full((F, T, C), np.nan, np.float64)
    for f in range(F):
        for t in range(T):
            uv = project_points(cam_mats, kp3d[f, t])               # (C,K,2)
            d = np.linalg.norm(kp2d[f, t] - uv, axis=-1)            # (C,K)
            d = np.where(conf[f, t] >= thr.conf_min, d, np.nan)
            with np.errstate(invalid="ignore"):
                med[f, t] = np.nanmedian(d, axis=-1)
    ok = (np.nan_to_num(med, nan=np.inf) <= thr.reproj_px).sum(-1) >= thr.reproj_min_views
    return ok, med


def containment_gate(kp3d, store, cam_mats, thr):
    """(F,T) admitted and (F,T) the best `contain_min_cams`-th containment
    fraction: per camera, the share of the fly's reprojected keypoints inside
    its OWN mask dilated by `contain_dilate_px`. Thresholds are ABSOLUTE (spec
    §3.2 requires >= 5 cameras) -- a bout with fewer valid cameras than
    `contain_min_cams` simply fails this gate; it is never relaxed."""
    from jarvis_jax.tracking.lift_mvq import _disk_offsets, _inside_mask_dilated, project_points
    dy, dx = _disk_offsets(thr.contain_dilate_px)
    F, T = kp3d.shape[:2]; C = len(store.cameras)
    cam_mats = np.asarray(cam_mats)
    if cam_mats.shape[0] != C:
        raise ValueError(
            f"containment_gate: cam_mats has {cam_mats.shape[0]} cameras but "
            f"the mask store has {C} ({list(store.cameras)}) -- they cannot "
            f"be aligned by name at this call site; pass matching arrays.")
    frac = np.zeros((F, T), np.float64)
    for f in range(F):
        for t in range(T):
            uv = project_points(cam_mats, kp3d[f, t])
            ok = np.isfinite(uv).all(-1)
            per_cam = []
            for c in range(C):
                if not store.valid_at(f, t)[c] or not ok[c].any():
                    continue
                inside = _inside_mask_dilated(store.mask_at(f, c, t), uv[c], ok[c], dy, dx)
                per_cam.append(inside[ok[c]].mean())
            per_cam = sorted(per_cam, reverse=True)
            frac[f, t] = per_cam[thr.contain_min_cams - 1] if len(per_cam) >= thr.contain_min_cams else 0.0
    return frac >= thr.contain_frac, frac


def decorrelate(admit, spacing, protect=None):
    """Greedy first-fit thinning: keep an admitted frame only if it is at
    least `spacing` frames after the last kept one. `protect` (T,), if given,
    marks frames exempt from the spacing check. NOT how T=2 partners work:
    `admit_bout` does NOT pass `protect` here -- partner frames are derived
    separately, at each decorrelated anchor, from `partner_masks`'s output;
    they are never anchors themselves and never affect this spacing count
    (wiring partners in as `protect` here would let them chain into new
    anchors and collapse the >= `spacing` invariant)."""
    keep = np.zeros_like(admit, bool)
    last = -10 ** 9
    for t in np.flatnonzero(admit):
        if t - last >= spacing or (protect is not None and protect[t]):
            keep[t] = True; last = t
    return keep


def partner_masks(admit, deltas):
    """{delta: (T,) bool} -- True where frame t AND frame t+delta BOTH pass.
    Endpoint semantics only (spec §3.2: a partner counts "when those frames
    also pass the gates") -- a T=2 pair is exactly the two named frames; the
    model is shown only those two, so nothing strictly between them (which
    is never sampled) affects whether the pair is usable."""
    admit = np.asarray(admit, bool)
    T = admit.shape[0]
    out = {}
    for d in deltas:
        d = int(d)
        m = np.zeros(T, bool)
        n = T - d
        if n > 0:
            m[:n] = admit[:n] & admit[d:d + n]
        out[d] = m
    return out


def admit_bout(arrays, store, cam_mats, thr, *, use_identity=True):
    if store is not None and list(arrays.cameras) != list(store.cameras):
        raise ValueError(
            f"admit_bout: kp2d camera order {list(arrays.cameras)} != mask "
            f"store camera order {list(store.cameras)} -- cam_mats must "
            f"align with BOTH by name, never assumed positional.")
    cam_mats = np.asarray(cam_mats)
    if cam_mats.shape[0] != len(arrays.cameras):
        raise ValueError(
            f"admit_bout: cam_mats has {cam_mats.shape[0]} cameras but "
            f"arrays.cameras has {len(arrays.cameras)} ({list(arrays.cameras)}) "
            f"-- they cannot be aligned by name at this call site.")
    ex_ok, ex_q = existence_gate(arrays.meta, thr)
    st_ok, st_q = step_gate(arrays.kp3d, thr)
    rp_ok, rp_q = reprojection_gate(arrays.kp2d, arrays.kp3d, cam_mats, arrays.conf, thr)
    finite = np.isfinite(arrays.kp3d).all((-1, -2))
    if store is not None:
        ct_ok, ct_q = containment_gate(arrays.kp3d, store, cam_mats, thr)
    else:                                    # single-fly pass: no masks (spec §3.4)
        ct_ok = np.ones_like(ex_ok); ct_q = np.ones_like(ex_q)
    id_bad = identity_gate(arrays.meta, thr) if use_identity else np.zeros(ex_ok.shape[1], bool)
    fly = ex_ok & st_ok & rp_ok & ct_ok & finite & ~id_bad[None, :]
    frame_any = fly.any(0) & ~id_bad
    # Anchors are decorrelated ALONE (no `protect`): T=2 partners are derived
    # separately by the caller, at each anchor, from `partners[delta]` --
    # see module docstring. Feeding partners into `decorrelate` as `protect`
    # would let them chain into new anchors and collapse the >= `spacing`
    # invariant.
    frame = decorrelate(frame_any, thr.decorrelation)
    partners = partner_masks(frame_any, thr.deltas)
    reasons = {"exist": ~ex_ok, "step": ~st_ok, "reproj": ~rp_ok, "contain": ~ct_ok,
               "nonfinite": ~finite, "identity": id_bad}
    quant = {"exist": ex_q, "step_units": st_q, "reproj_px": rp_q, "contain_frac": ct_q}
    return GateResult(frame, fly, reasons, quant, partners)
