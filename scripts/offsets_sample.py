"""Per-fly STAC marker-offset sample, pooled over a recording.

Marker offsets (`offsets_fly<f>.h5`) are a per-INDIVIDUAL constant, like body
scale. They used to be fit once per run root on whichever bout-fly reached the
stage first and shared with both flies; on Session0/2025_10_20_13_20_04 bout 28
that sample was fly0 (the female) frames 0..499, so the male ran IK with her
offsets -- his abdomen fitted 1.10x too long, wings 1.01-1.06x.

This module builds the sample that `fit_offsets_once` should see for ONE fly:

1. pool every triangulated bout of that fly under the run root
   (`estimate_recording_scale.bout_kp3d_paths`: kp3d_filt.npz, else kp3d.npz);
2. keep a frame only if every keypoint is finite, its per-frame MINIMUM
   conf3d is >= `min_conf`, and its implied body scale (per-frame Umeyama on
   the trunk markers) lies within `mad_k` MADs of the fly's pooled median --
   a bad triangulation must not enter the fit even when confidence looks fine;
3. take `n_frames` stratified round-robin across bouts, best confidence first
   within each bout, so no single bout defines the offsets.

The frames are NOT a time series, so the fit must run with the jaxls temporal
smoothness term off: see `offsets_fit_cfg`.

Per-fly is only meaningful when fly0/fly1 is a stable individual across bouts
(sex-canonicalized: every bout carries sex.json). `resolve_offsets_path`
refuses to fall back to a shared file unless `stac.allow_shared_offsets`.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scripts.estimate_recording_scale import bout_kp3d_paths, _bout_idx_from_path
except ModuleNotFoundError:  # direct invocation: sys.path[0] is scripts/, not the repo root
    from estimate_recording_scale import bout_kp3d_paths, _bout_idx_from_path

SHARED_OFFSETS_NAME = "offsets.h5"


def per_fly_offsets_name(fly: int) -> str:
    return f"offsets_fly{int(fly)}.h5"


@dataclass
class OffsetsSample:
    kp3d: np.ndarray                       # (N, K, 3), rows in `frames` order
    frames: List[Tuple[int, int]]          # (bout_idx, frame_idx) per row
    provenance: dict                       # JSON-serialisable; written beside the h5


def load_fly_bouts(run_root, fly: int) -> Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """{bout_idx: (kp3d (F,K,3) mm, conf3d (F,K) or None)} for ONE fly.

    kp3d_filt.npz is preferred per bout (what ik_only consumes), kp3d.npz is
    the fallback. conf3d is read from whichever file was chosen; None when the
    file has no such key, in which case the confidence gate is skipped for
    that bout and recorded as such.
    """
    out = {}
    for p in bout_kp3d_paths(Path(run_root), int(fly)):
        with np.load(p, allow_pickle=False) as z:
            kp = np.asarray(z["kp3d"], dtype=np.float64)
            conf = np.asarray(z["conf3d"], dtype=np.float64) if "conf3d" in z.files else None
        out[_bout_idx_from_path(p)] = (kp, conf)
    return out


def _frame_gates(kp3d: np.ndarray, conf: Optional[np.ndarray], min_conf: float):
    """(candidate mask, per-frame min conf, counts) for one bout."""
    finite = np.isfinite(kp3d).all(axis=(1, 2))
    if conf is not None:
        mconf = np.where(np.isfinite(conf), conf, -np.inf).min(axis=1)
        conf_ok = mconf >= float(min_conf)
    else:
        mconf = np.zeros(len(kp3d))
        conf_ok = np.ones(len(kp3d), dtype=bool)
    counts = {"n_frames": int(len(kp3d)),
              "nonfinite": int((~finite).sum()),
              "low_conf": int((finite & ~conf_ok).sum())}
    return finite & conf_ok, mconf, counts


def select_offsets_sample(kp3d_by_bout: Dict[int, np.ndarray],
                          conf_by_bout: Dict[int, Optional[np.ndarray]],
                          *, n_frames: int, min_conf: float,
                          scale_by_bout: Optional[Dict[int, np.ndarray]] = None,
                          mad_k: float = 3.0) -> OffsetsSample:
    """Gate every frame of every bout, then draw a stratified sample.

    `scale_by_bout` (optional) maps bout -> per-frame implied body scale,
    aligned with that bout's frames (NaN where unavailable). The gate pools
    the candidates' scales, takes median and MAD, and rejects frames further
    than `mad_k` * 1.4826 * MAD from the median. Zero spread rejects nothing.
    Deterministic: no randomness anywhere.
    """
    bouts = sorted(kp3d_by_bout)
    cand, mconf, counts = {}, {}, {}
    any_conf = False
    for b in bouts:
        c = conf_by_bout.get(b)
        any_conf |= c is not None
        cand[b], mconf[b], counts[b] = _frame_gates(np.asarray(kp3d_by_bout[b]), c, min_conf)
        counts[b]["scale_outlier"] = 0

    scale_gate = {"applied": False, "median": None, "mad": None, "mad_k": float(mad_k)}
    if scale_by_bout is not None:
        pooled = np.concatenate([np.asarray(scale_by_bout[b], float)[cand[b]] for b in bouts]) \
            if bouts else np.zeros(0)
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size:
            med = float(np.median(pooled))
            mad = float(np.median(np.abs(pooled - med)))
            scale_gate.update(applied=True, median=med, mad=mad)
            if mad > 0:
                tol = float(mad_k) * 1.4826 * mad
                for b in bouts:
                    s = np.asarray(scale_by_bout[b], float)
                    bad = cand[b] & np.isfinite(s) & (np.abs(s - med) > tol)
                    counts[b]["scale_outlier"] = int(bad.sum())
                    cand[b] &= ~bad

    # per-bout candidate lists, best confidence first (stable on frame index)
    queues = {}
    for b in bouts:
        idx = np.flatnonzero(cand[b])
        order = np.lexsort((idx, -mconf[b][idx]))
        queues[b] = list(idx[order])
        counts[b]["candidates"] = int(len(idx))

    chosen: List[Tuple[int, int]] = []
    while len(chosen) < int(n_frames) and any(queues[b] for b in bouts):
        for b in bouts:
            if queues[b] and len(chosen) < int(n_frames):
                chosen.append((int(b), int(queues[b].pop(0))))
    for b in bouts:
        counts[b]["selected"] = sum(1 for bb, _ in chosen if bb == b)

    if not chosen:
        raise ValueError(
            "offsets sample: no frame passed the gates "
            f"(min_conf={min_conf}, mad_k={mad_k}); per bout: "
            + "; ".join(f"bout {b}: {counts[b]}" for b in bouts))

    kp = np.stack([np.asarray(kp3d_by_bout[b])[f] for b, f in chosen]).astype(np.float64)
    prov = {
        "n_bouts": len(bouts),
        "n_requested": int(n_frames),
        "n_selected": len(chosen),
        "min_conf": float(min_conf),
        "conf_gate_applied": bool(any_conf),
        "scale_gate": scale_gate,
        "bouts": {str(b): counts[b] for b in bouts},
        "frames": [[b, f] for b, f in chosen],
    }
    return OffsetsSample(kp3d=kp, frames=chosen, provenance=prov)


def aligned_per_frame_scales(kp3d: np.ndarray, per_frame_scales_fn) -> np.ndarray:
    """Per-frame implied scale aligned to `kp3d`'s frames (NaN where any
    keypoint is non-finite). `per_frame_scales_fn` is
    `estimate_recording_scale.per_frame_scales` partially applied with
    kp_names/model_xml; it DROPS non-finite frames, so it is only called on
    the all-finite subset, where its output is one value per input frame."""
    out = np.full(len(kp3d), np.nan)
    finite = np.isfinite(kp3d).all(axis=(1, 2))
    if finite.any():
        s = np.asarray(per_frame_scales_fn(kp3d[finite]), float)
        if s.shape[0] != int(finite.sum()):
            raise RuntimeError("per_frame_scales dropped frames from an all-finite input; "
                               "cannot align scales to frames")
        out[finite] = s
    return out


def resolve_offsets_path(run_root, fly: int, identity: str, *, allow_shared: bool,
                         reason: str = "") -> Tuple[str, str]:
    """(path, mode) with mode 'per_fly' or 'shared'.

    Per-fly needs identity == 'canonical' (every bout has a usable sex.json,
    so fly0/fly1 names one individual across the recording). Otherwise a
    shared file would blend two animals' anatomy, so this REFUSES unless
    `stac.allow_shared_offsets` is set, and names the cause.
    """
    if identity == "canonical":
        return os.path.join(str(run_root), per_fly_offsets_name(fly)), "per_fly"
    if not allow_shared:
        raise RuntimeError(
            f"fly{fly}: refusing SHARED marker offsets: fly identity is not canonical "
            f"({reason or 'unknown'}), so fly0/fly1 is not a stable individual and one "
            f"offsets.h5 would blend both animals. Fix the cause "
            f"(scripts/apply_id_review.py / canonicalize sex), or set "
            f"stac.allow_shared_offsets=true to accept a shared fit deliberately.")
    return os.path.join(str(run_root), SHARED_OFFSETS_NAME), "shared"


def offsets_fit_cfg(cfg):
    """A deep copy of `cfg` for the offsets fit only: the sample is a set of
    non-consecutive frames, so the jaxls temporal smoothness term (and any
    per-DOF smoothness multiplier) must be OFF -- otherwise the solver couples
    frames that are seconds apart. The caller's cfg is left untouched."""
    from omegaconf import OmegaConf
    fit = copy.deepcopy(cfg)
    was_struct = OmegaConf.is_struct(fit)
    OmegaConf.set_struct(fit, False)     # some of these keys may be absent
    fit.anatomy.model.JAXLS_SMOOTH_WEIGHT = 0.0
    fit.anatomy.model.JAXLS_SMOOTH_Q_MULT = None
    # With the smoothness chain gone the frames are independent, and
    # JaxlsBatchSolver then takes its vmapped per-frame path automatically
    # (stac_core_jaxls._solve_independent: T=1 problem, dense per-frame
    # factorisation, per-frame termination). Chunking would only add Python
    # loop overhead on top of that, so run the whole sample in one call.
    fit.anatomy.model.JAXLS_CHUNK_SIZE = 0
    OmegaConf.set_struct(fit, bool(was_struct))
    return fit
