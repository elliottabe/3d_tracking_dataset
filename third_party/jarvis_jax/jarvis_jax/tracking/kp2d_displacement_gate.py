"""Per-frame, per-camera 2D displacement gate for a correlated landmark
majority-flip artifact -- applied pre-triangulation, alongside the other
Stage-A/B 2D filters (see kp2d_oneeuro.py for the sibling temporal filter).

## What this catches, and why the other filters/gates cannot

Measured (Session0 ``2025_10_20_13_20_04`` bout 28, male/fly1): the 2D
``WingL_V13`` landmark jumps between two different, BOTH-VISIBLE wing
structures ~170-330 px apart, at high confidence (0.91-0.94) on both sides
of the jump, with the camera majority flipping frame to frame. Every other
guard in this pipeline was tried and ruled out on this exact artifact before
this gate was written (see
``.superpowers/sdd/2026-08-29-coarse-to-fine-3d/wing-gating.md`` for the full
measurement):

- confidence gating (``conf_thresh``, ``view_conf_thresh``): both candidate
  positions score 0.91-0.94, i.e. the bad detection is not less confident
  than the good one.
- multi-view consensus (``triangulate.reproj_resid_px`` /
  ``_reject_outlier_views``): built to reject a LONE dissenting camera; here
  the camera MAJORITY is what flips, so there is no minority to outvote.
- ``triangulate.refine_from_seed``: same shape of failure -- a seed-gated
  re-solve cannot help when most views agree on the wrong structure.
- ``kp2d_oneeuro`` (1-Euro temporal smoothing): its speed-adaptive cutoff
  (``beta``) treats the artifact's frame-to-frame jump exactly like fast
  real motion (0.5% suppression at the production default; any
  ``beta >= 0.1`` collapses back to <=3%); only killing the adaptivity
  entirely (``beta=0``, i.e. a plain fixed low-pass) suppresses it, and that
  measurably flattens real wing flicks (16.5-53% amplitude loss).

## What this does instead, and its ACTUAL, narrow basis

A per-(camera, keypoint) check on raw frame-to-frame pixel displacement: if
a keypoint hops further than its own threshold between two otherwise-
confident frames, the ARRIVAL frame's view is dropped (see "why drop, not
interpolate" below). Measured on the same bout: **82.5% artifact suppression
(p95 |delta wing angle| in the artifact window, 27.2 deg -> 4.8 deg) with
100.0% real-motion preservation across all five confirmed real-motion
windows** (peak-to-trough amplitude unchanged to the frame).

**This works for a narrower reason than "big 2D jumps are suspicious" might
suggest, and that reason is the whole basis for trusting it at all.** A
motion-energy check (does the underlying image actually change where the
landmark says it moved?) was run alongside this measurement specifically to
ask whether "the pixels changed" separates real jumps from artifact jumps.
It does not: restricted to the one displacement band where both classes have
real support (20-60 px), motion energy alone scores AUC 0.473 -- chance.
The artifact is only separable here because ITS hop (measured 120-330 px)
happens to EXCEED this bout's largest confirmed real wing displacement
(~130 px) -- a property of this one landmark-confusion pair on this one
bout, not a general property of "big" 2D motion.

**LIMITATION, stated plainly: this gate catches displacement-SCALE outliers
only.** It was validated on ONE bout, ONE fly's ground truth (male/fly1,
bout 28). It provides no protection whatsoever against a landmark confusion
whose hop is the SAME pixel scale as real motion for that keypoint --
exactly the case a same-scale confusion would be. If a future artifact
(different keypoint, different bout, different camera rig) hops within the
range of that keypoint's real motion, this gate will not see it, and no
figure or metric downstream will distinguish "gate had nothing to catch"
from "gate is blind here" -- that ambiguity is inherent to a magnitude-only
rule and is not fixed by tuning the threshold.

## Why drop, not interpolate

The task that motivated this gate specified NaN-ing the dropped view. Per
``triangulate.triangulate_keypoints``'s own contract, that is NOT how a view
is removed from the DLT solve in this codebase: "kp2d must be FINITE even
where conf < conf_thresh (invalid views are zeroed by validity masking
inside triangulate_dlt_batched; a NaN/Inf pixel would poison that point's
SVD via 0*NaN)". Every existing view-drop gate in the Stage-B path
(``view_conf_thresh``, the kp-mask-agreement gate in ``scripts/run_bout.py``)
removes a view by zeroing its CONFIDENCE, not by writing NaN into ``kp2d``.
This module follows that same contract: ``displacement_gate_kp2d`` returns
an updated ``conf`` with flagged (frame, camera, keypoint) views set to 0
and leaves ``kp2d`` byte-identical. ``triangulate_keypoints`` then does
exactly what the task asked for anyway -- a keypoint with <2 confident views
comes out NaN in ``kp3d`` -- it is just implemented via the confidence
channel, which is the pipeline's one existing mechanism for "this view does
not count," not a second, parallel one.

## Default-off / per-keypoint contract

Disabled (the default) is a strict no-op: ``conf`` is returned unchanged
(a ``.copy()``, not the same object, but every value identical), so a run
with this gate off is byte-identical to a run predating this module.
Thresholds are per-keypoint (a fallback ``default_px`` applies to any
keypoint not explicitly listed) because a single global number is wrong: a
wing tip legitimately moves 40x faster, frame to frame, than a thorax
landmark on the same fly (see ``configs/detector/vitpose_v3.yaml``'s
``kp2d_displacement_gate`` block and
``.superpowers/sdd/2026-08-29-coarse-to-fine-3d/displacement-gate.md`` for
the derivation and the full per-keypoint table).
"""
from __future__ import annotations
import numpy as np


def resolve_thresholds_px(kp_names, *, default_px: float, per_keypoint_px=None):
    """kp_names (K,) -> (K,) float64 array of per-keypoint px thresholds.

    ``per_keypoint_px`` (dict[name -> px] | None) overrides ``default_px`` by
    exact name match; every OTHER keypoint still gets its own entry (equal to
    ``default_px``) rather than sharing one array-wide scalar downstream --
    this is what keeps the gate "per-keypoint, not global" even for names the
    config does not call out explicitly.
    """
    per_keypoint_px = dict(per_keypoint_px or {})
    thr = np.full(len(kp_names), float(default_px), np.float64)
    unknown = set(per_keypoint_px) - set(kp_names)
    if unknown:
        raise ValueError(
            f"kp2d_displacement_gate.per_keypoint_px names not in kp_names: "
            f"{sorted(unknown)}")
    for i, name in enumerate(kp_names):
        if name in per_keypoint_px:
            thr[i] = float(per_keypoint_px[name])
    return thr


def displacement_gate_kp2d(kp2d, conf, kp_names, *, default_px: float,
                           per_keypoint_px=None, conf_thresh: float = 0.3):
    """Drop (zero the confidence of) any (frame, camera, keypoint) ARRIVAL
    view whose raw 2D position hopped further than that keypoint's own
    threshold from the immediately preceding frame -- see the module
    docstring for what this is (a displacement-magnitude outlier gate) and
    is NOT (a general "is this real motion" classifier).

    Args:
        kp2d: (T,C,K,2) float, full-frame pixel coordinates. NEVER modified
            or read as invalid where low-confidence -- displacement is
            computed on the raw signal exactly like every other Stage-A/B
            gate that only ever touches ``conf`` (view_conf_thresh, the
            kp-mask-agreement gate).
        conf: (T,C,K) float, per-keypoint peak confidence.
        kp_names: (K,) keypoint names, in the SAME order as the ``kp2d``/
            ``conf`` last axis (cfg.model.KP_NAMES order, i.e. called AFTER
            ``reorder_detector_to_model`` in ``scripts/run_bout.py``).
        default_px: fallback per-keypoint threshold (px) for any keypoint
            not named in ``per_keypoint_px``.
        per_keypoint_px: optional dict[name -> px] of per-keypoint overrides.
        conf_thresh: a transition is only evaluated (and can only be
            flagged) when BOTH its start and arrival frame are already
            confident (``conf >= conf_thresh``) at that (camera, keypoint).
            This is deliberate, not incidental: ``predict_bout_2d`` zero-
            fills ``kp2d``/``conf`` for a (frame, camera) with an empty crop
            (see its docstring), and comparing a real detection against that
            zero-fill would read as a spurious huge "hop" and wrongly flag a
            perfectly good detection. Requiring both endpoints confident also
            matches what was actually measured: the artifact's own two
            candidate positions are BOTH high-confidence (0.91-0.94), so
            restricting to confident-to-confident transitions costs nothing
            on the case this gate was built for.

    Returns:
        conf' (T,C,K), same dtype as the input ``conf`` (float32 in
        production): a copy with flagged views zeroed.
        ``kp2d`` is returned unchanged (not returned) -- callers keep using
        their own ``kp2d`` array; only ``conf`` changes. Disabled / no
        transition ever exceeds threshold -> ``conf' == conf`` elementwise
        (a copy, not the same object).
    """
    kp2d = np.asarray(kp2d, np.float64)
    conf = np.array(conf, copy=True)          # preserve caller's dtype (float32 in production)
    T, C, K, two = kp2d.shape
    if two != 2:
        raise ValueError(f"kp2d last axis must be 2 (x,y), got {two}")
    if conf.shape != (T, C, K):
        raise ValueError(f"conf shape {conf.shape} != kp2d's (T,C,K)={(T, C, K)}")
    if K != len(kp_names):
        raise ValueError(f"kp_names length {len(kp_names)} != K={K}")
    if T < 2:
        return conf

    thr = resolve_thresholds_px(kp_names, default_px=default_px,
                                per_keypoint_px=per_keypoint_px)   # (K,)
    valid = conf >= float(conf_thresh)                            # (T,C,K)
    disp = np.linalg.norm(kp2d[1:] - kp2d[:-1], axis=-1)          # (T-1,C,K)
    both_confident = valid[1:] & valid[:-1]                       # (T-1,C,K)
    flagged = both_confident & (disp > thr[None, None, :])        # (T-1,C,K)

    conf[1:][flagged] = 0.0
    return conf
