#!/usr/bin/env python3
"""Recording-level robust body-scale estimator.

Replaces the scale-from-first-bout defect in ``scripts/run_bout.py``: that
driver used to fit ONE trunk Procrustes scale from whichever bout/fly got
there first and apply it to every bout and both flies. Measured on real
frozen data (Session0 2025_10_20_13_20_04), one bad bout-fly (bout22/fly0)
gave a scale 38x too small (0.000338 vs production's 0.010994) because only
68% of that bout-fly's frames had all 5 trunk markers finite in the raw
triangulated kp3d -- the temporal filter's gap-fill inflated the trunk point
cloud. Other bout-flies of the same recording spread 0.0098-0.0130 (+-20%).

Body size is constant across a recording's bouts for a given individual, so
this spread is estimator noise, not signal. This module:

  1. computes a per-bout-fly scale (default ``scale_keypoints='rigid_segment'``:
     ``per_bout_segment_scale``, a pose-invariant direct rigid-leg-segment
     length measurement -- see that function's docstring; the older
     ``'trunk'``/``'all'`` modes use ``per_frame_scales``, a Procrustes/
     norm-ratio FIT over a trunk-marker cloud, which cloud-spread estimators
     confound with pose),
  2. pools (frames, or per-pair values for rigid_segment) across a fly's
     bouts and rejects outlier bouts by a MAD threshold on their per-bout
     medians (``robust_scale``),
  3. drives that per (run_root, fly) -- UNLESS the recording's fly-slot
     identity is not stable across bouts, in which case all bout-flies are
     pooled into a single recording-level scale instead (see
     ``estimate_run_root`` docstring: fly0/fly1 is only a stable individual
     label once the recording has been sex-canonicalized).

``rigid_segment`` additionally derives physics-based keypoint-quality
diagnostics (``within_bone_cv``/``across_bone_cv``, see
``segment_scale_diagnostics``) since a rigid-segment measurement doubles as a
check that keypoints obey basic skeletal physics.

CLI:
    python scripts/estimate_recording_scale.py --run-root <path> \\
        [--estimator umeyama|norm_ratio] \\
        [--scale-keypoints trunk|all|rigid_segment] [--include-thorax] \\
        [--anatomy configs/anatomy/v1.yaml] [--out <path>] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from jarvis_jax.tracking.scale import _umeyama_scale_per_frame, DEFAULT_TRUNK_KEYPOINTS
try:
    from scripts.benchmark.metrics import proximity_bl
except ModuleNotFoundError:  # direct invocation: sys.path[0] is scripts/, not repo root
    from benchmark.metrics import proximity_bl

_BOUT_DIR_RE = re.compile(r"^bout_(\d+)$")
_FLY_DIR_RE = re.compile(r"^fly(\d+)$")

# Slot-coincidence guard: when SAM3 loses one animal in a camera it can lock
# that camera's "other" slot onto the SAME animal, so the "other" fly's 3D is
# triangulated from a mix of both animals. Measured across 160 processed
# two-fly bouts: 4 had >20% of frames with slots within 0.35 body-lengths of
# each other (the median bout: 0.00%) -- S0 bout22/fly0, exactly the bout
# that produced the 38x scale defect, is one of them. A duplicated slot
# contains a REAL fly (just the wrong one), so its scale can look perfectly
# normal -- MAD outlier rejection alone will not reliably catch it.
COINCIDENT_BL_THRESH = 0.35
DEFAULT_COINCIDENT_FRAC_THRESH = 0.20

# Cache MjModel-derived tracking-site data per xml path so estimate_run_root
# (which calls per_frame_scales once per bout-fly, i.e. dozens of times per
# recording) doesn't re-parse the MJCF on every call.
_SITE_CACHE: Dict[str, Dict[str, int]] = {}


def _tracking_site_idx(model_xml: str) -> Dict[str, int]:
    idx = _SITE_CACHE.get(model_xml)
    if idx is not None:
        return idx
    import mujoco
    mj = mujoco.MjModel.from_xml_path(str(model_xml))
    idx = {}
    for i in range(mj.nsite):
        site_name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SITE, i)
        if site_name and site_name.startswith("tracking[") and site_name.endswith("]"):
            idx[site_name[len("tracking["):-1]] = i
    idx["__mj__"] = mj  # stash the model too (avoids a second parse for xpos)
    _SITE_CACHE[model_xml] = idx
    return idx


def _ref_positions(model_xml: str, present: List[str]) -> np.ndarray:
    import mujoco
    idx = _tracking_site_idx(model_xml)
    mj = idx["__mj__"]
    d = mujoco.MjData(mj)
    mujoco.mj_forward(mj, d)
    return np.array([d.site_xpos[idx[n]] for n in present])


# ---------------------------------------------------------------------------
# Rigid-segment body-scale estimator.
#
# Keypoints at the two ends of one rigid bone are a fixed distance apart no
# matter how the fly moves (rotation/translation of the whole body cancels
# out exactly, and -- unlike a trunk-marker Procrustes/norm-ratio fit -- so
# does internal pose: a leg bending elsewhere doesn't change THIS bone's
# length). Measured by hand: T1L_FeTi->T1L_TiTa (front-left tibia) has model
# length 0.05103 model-units, observed median 4.495 data-units over 1174
# frames (CV 2.9%), implied scale 0.011353 = model/observed. Pooling the
# implied scale over every such pair and taking a robust median is a
# pose-invariant DIRECT measurement, not a fit -- there is no `estimator`
# choice (umeyama vs norm_ratio) the way there is for per_frame_scales.
# ---------------------------------------------------------------------------

LEG_NAMES = ("T1L", "T1R", "T2L", "T2R", "T3L", "T3R")
# Consecutive joints along one leg's kinematic chain, thorax to tarsus tip.
LEG_JOINT_CHAIN = ("ThxCx", "Tro", "FeTi", "TiTa", "TaT1")

# Thorax-plate pairs, NOT included by default (include_thorax=False). Measured
# on real data these imply a scale ~10.2% LARGER than the leg-chain scale --
# i.e. the model's trunk proportions don't quite match the animal's. Since IK
# joint angles are determined by the LEG chain (not the trunk plates), the
# leg-only default is deliberate: it matches the kinematic chain STAC solves.
THORAX_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("WingL_base", "WingR_base"),
    ("Scutellum", "WingL_base"),
    ("Scutellum", "WingR_base"),
)

# within_bone_cv threshold above which a fly's rigid-segment scale estimate
# is flagged untrustworthy. Measured on real data (Session0, 4 bouts): male
# (good keypoints) within-bone CV 3.6-6.2%, across-bone CV 6.0-6.7%, scale
# stable 0.01142-0.01178 across bouts; female (known-bad keypoints)
# within-bone CV 20-50%, across-bone CV 29-57%, scale swings 0.0081-0.0130
# (60%). Total separation between the two populations -- 0.15 sits well
# above the male ceiling (~0.062) and well below the female floor (~0.20).
WITHIN_BONE_CV_WARN_THRESH = 0.15


def rigid_segment_pairs(kp_names: List[str], *, include_thorax: bool = False
                        ) -> List[Tuple[str, str]]:
    """Consecutive-joint keypoint pairs spanning one rigid skeletal segment
    each, restricted to pairs where BOTH names are present in ``kp_names``
    (missing names -- e.g. the real v1 model has no ThxCx tracking site for
    T2/T3 legs -- silently drop that one pair, not an error).

    ``include_thorax=True`` adds the 3 thorax-plate pairs (see
    ``THORAX_PAIRS``); default False since they imply a different (larger)
    scale than the leg chain the IK solves against.
    """
    names = set(kp_names)
    pairs: List[Tuple[str, str]] = []
    for leg in LEG_NAMES:
        chain = [f"{leg}_{joint}" for joint in LEG_JOINT_CHAIN]
        for a, b in zip(chain[:-1], chain[1:]):
            if a in names and b in names:
                pairs.append((a, b))
    if include_thorax:
        for a, b in THORAX_PAIRS:
            if a in names and b in names:
                pairs.append((a, b))
    return pairs


def _segment_pair_measurements(kp3d: np.ndarray, kp_names: List[str], model_xml: str, *,
                               include_thorax: bool = False
                               ) -> List[Tuple[Tuple[str, str], np.ndarray, float]]:
    """Shared internals for ``per_bout_segment_scale``/``segment_scale_diagnostics``.

    For each usable rigid pair (both names present in ``kp_names`` AND the
    model's tracking sites, >=1 finite observed-distance frame): the
    per-frame observed distances (NaN frames dropped) and the model's
    rest-pose distance between the same two tracking sites.

    Returns a list of ``((name_a, name_b), finite_dists, model_dist)``.
    Raises ValueError if no pair is usable.
    """
    pairs = rigid_segment_pairs(kp_names, include_thorax=include_thorax)
    if not pairs:
        raise ValueError(
            f"_segment_pair_measurements: no rigid-segment pairs available "
            f"for the given kp_names (include_thorax={include_thorax})")

    name_to_idx = {n: i for i, n in enumerate(kp_names)}
    tracking_site_idx = _tracking_site_idx(model_xml)
    needed = sorted({n for pair in pairs for n in pair if n in tracking_site_idx})
    ref_arr = _ref_positions(model_xml, needed)
    ref_pos = dict(zip(needed, ref_arr))

    P = np.asarray(kp3d, dtype=np.float64)
    measurements: List[Tuple[Tuple[str, str], np.ndarray, float]] = []
    for a, b in pairs:
        if a not in ref_pos or b not in ref_pos:
            continue  # not a real model tracking site -- can't get a model length
        ia, ib = name_to_idx[a], name_to_idx[b]
        d_obs = np.linalg.norm(P[:, ia, :] - P[:, ib, :], axis=-1)
        finite = d_obs[np.isfinite(d_obs)]
        if finite.size == 0:
            continue
        model_dist = float(np.linalg.norm(ref_pos[a] - ref_pos[b]))
        if model_dist <= 0:
            continue
        measurements.append(((a, b), finite, model_dist))

    if not measurements:
        raise ValueError(
            "_segment_pair_measurements: no rigid-segment pair had any finite "
            "observed-distance frame")
    return measurements


def segment_scale_diagnostics(kp3d: np.ndarray, kp_names: List[str], model_xml: str, *,
                              include_thorax: bool = False) -> dict:
    """Physics-based keypoint-quality diagnostics from rigid-segment
    measurements, for ONE bout-fly.

    Two independent properties a rigid skeleton must have:

      1. RIGIDITY -- one bone's length cannot change across frames.
         ``within_bone_cv`` = mean over pairs of (std/mean of that pair's
         observed distance across frames). High = keypoints are jittery/bad.
      2. AGREEMENT -- different bones must all imply the same overall body
         scale. ``across_bone_cv`` = std/mean of the per-pair implied scales.
         High = keypoints are internally inconsistent (e.g. systematically
         wrong on one limb) even if each individual bone looks rigid.

    These are independent signals: per-frame jitter on one bone raises
    ``within_bone_cv`` without moving ``across_bone_cv`` (the jitter's
    across-frame median is unbiased); a systematically-wrong bone length
    raises ``across_bone_cv`` without moving ``within_bone_cv`` (that bone is
    still perfectly rigid, just the wrong rigid length). See
    ``WITHIN_BONE_CV_WARN_THRESH`` for measured male-vs-female values.

    Returns ``{"scale": float, "per_pair_scale": np.ndarray,
    "within_bone_cv": float, "across_bone_cv": float, "n_pairs_used": int}``.
    Raises ValueError if no pair is usable (propagated from
    ``_segment_pair_measurements``).
    """
    measurements = _segment_pair_measurements(kp3d, kp_names, model_xml,
                                              include_thorax=include_thorax)
    per_pair_scale = []
    within_cvs = []
    for _pair, d_obs, model_dist in measurements:
        mean = float(np.mean(d_obs))
        if mean > 0:
            within_cvs.append(float(np.std(d_obs)) / mean)
        med = float(np.median(d_obs))
        if med > 0:
            per_pair_scale.append(model_dist / med)

    if not per_pair_scale:
        raise ValueError(
            "segment_scale_diagnostics: every usable pair had a degenerate "
            "(non-positive) median observed distance")

    per_pair_scale = np.asarray(per_pair_scale, dtype=np.float64)
    within_bone_cv = float(np.mean(within_cvs)) if within_cvs else float("nan")
    mean_scale = float(np.mean(per_pair_scale))
    across_bone_cv = float(np.std(per_pair_scale) / mean_scale) if mean_scale > 0 else float("nan")

    return {
        "scale": float(np.median(per_pair_scale)),
        "per_pair_scale": per_pair_scale,
        "within_bone_cv": within_bone_cv,
        "across_bone_cv": across_bone_cv,
        "n_pairs_used": int(per_pair_scale.size),
    }


def per_bout_segment_scale(kp3d: np.ndarray, kp_names: List[str], model_xml: str, *,
                           include_thorax: bool = False) -> np.ndarray:
    """Per-pair implied body-scale (data -> model) for ONE bout-fly, from
    rigid leg-segment lengths (see module docstring above).

    For each usable rigid pair (``rigid_segment_pairs``): the median-over-
    frames observed distance between the two keypoints, and the model's
    rest-pose distance between the same two tracking sites; implied scale =
    model_distance / observed_median. NaN-safe (frames with a non-finite
    marker are dropped per pair, not propagated); pairs with no finite
    frames are dropped. Pose (per-frame rotation/translation, and any OTHER
    joint bending) does not affect a pair's own implied scale.

    Returns the 1-D array of per-pair implied scales. Raises ValueError if no
    pair is usable. NOTE: there is no ``estimator`` argument -- a rigid
    segment's length is measured directly, not fit by Procrustes/norm-ratio.
    """
    return segment_scale_diagnostics(kp3d, kp_names, model_xml,
                                     include_thorax=include_thorax)["per_pair_scale"]


def bout_kp3d_paths(run_root: Path, fly: int) -> List[Path]:
    """Every ``<run_root>/bouts/bout_*/fly<fly>/kp3d_filt.npz``, falling back
    to ``kp3d.npz`` per bout when the filtered file is absent, sorted by bout
    index."""
    run_root = Path(run_root)
    bouts_dir = run_root / "bouts"
    if not bouts_dir.exists():
        return []
    entries = []
    for d in sorted(bouts_dir.iterdir()):
        if not d.is_dir():
            continue
        m = _BOUT_DIR_RE.match(d.name)
        if not m:
            continue
        fly_dir = d / f"fly{fly}"
        filt = fly_dir / "kp3d_filt.npz"
        raw = fly_dir / "kp3d.npz"
        if filt.exists():
            entries.append((int(m.group(1)), filt))
        elif raw.exists():
            entries.append((int(m.group(1)), raw))
    entries.sort(key=lambda t: t[0])
    return [p for _, p in entries]


def _bout_idx_from_path(p: Path) -> int:
    m = _BOUT_DIR_RE.match(p.parent.parent.name)
    if not m:
        raise ValueError(f"cannot parse bout index from path: {p}")
    return int(m.group(1))


def per_frame_scales(kp3d: np.ndarray, kp_names: List[str], model_xml: str, *,
                     trunk_names: Optional[List[str]] = None,
                     estimator: str = "umeyama") -> np.ndarray:
    """(F,) per-frame scale estimates (data -> model) for ONE bout-fly.

    Frames with any non-finite selected marker are dropped, not propagated.
    ``estimator='umeyama'`` uses the per-frame Umeyama similarity solve
    (``jarvis_jax.tracking.scale._umeyama_scale_per_frame``); ``'norm_ratio'``
    uses ref_spread / data_spread per frame.
    """
    names = list(trunk_names) if trunk_names else list(DEFAULT_TRUNK_KEYPOINTS)
    name_to_idx = {n: i for i, n in enumerate(kp_names)}
    tracking_site_idx = _tracking_site_idx(model_xml)
    present = [n for n in names if n in name_to_idx and n in tracking_site_idx]
    if len(present) < 3:
        raise ValueError(
            f"per_frame_scales needs >=3 trunk markers present in both kp_names "
            f"and the model's tracking[...] sites; requested {names}, found {present}")

    ref = _ref_positions(model_xml, present)
    ref_centered = ref - ref.mean(axis=0)
    ref_spread = float(np.sqrt((ref_centered ** 2).sum()))

    tidx = [name_to_idx[n] for n in present]
    P = np.asarray(kp3d, dtype=np.float64)[:, tidx, :]
    valid = np.all(np.isfinite(P), axis=(1, 2))
    P = P[valid]
    if P.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    if estimator == "norm_ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            spreads = np.sqrt(((P - P.mean(axis=1, keepdims=True)) ** 2).sum(axis=(1, 2)))
            return ref_spread / spreads
    elif estimator == "umeyama":
        return _umeyama_scale_per_frame(P, ref_centered, robust="none")
    else:
        raise ValueError(f"per_frame_scales: unknown estimator {estimator!r}")


def robust_scale(per_bout: Dict[int, np.ndarray], *, mad_k: float = 3.0) -> dict:
    """Pool per-frame scales across a fly's (or a pooled set of) bouts and
    reject outlier bouts.

    ``per_bout`` maps an integer key (bout index, or any hashable int-like
    disambiguator) to a per-frame scale array. All frames are pooled and
    their median taken as the initial estimate; a bout is flagged an outlier
    when its own median deviates from that pooled median by more than
    ``mad_k * MAD`` (MAD computed over the per-bout medians); the final scale
    re-pools frames EXCLUDING flagged bouts. Deterministic; no randomness.
    """
    filtered: Dict[int, np.ndarray] = {}
    for bi, arr in per_bout.items():
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size:
            filtered[bi] = a

    if not filtered:
        raise ValueError("robust_scale: no usable (finite, positive) frames in any bout")

    bout_idxs = sorted(filtered.keys())
    per_bout_median = {bi: float(np.median(filtered[bi])) for bi in bout_idxs}
    medians = np.array([per_bout_median[bi] for bi in bout_idxs])

    pooled_all = np.concatenate([filtered[bi] for bi in bout_idxs])
    pooled_median = float(np.median(pooled_all))

    med_of_medians = float(np.median(medians))
    mad = float(np.median(np.abs(medians - med_of_medians)))

    if mad <= 0:
        outlier_bouts = []
    else:
        outlier_bouts = [bi for bi in bout_idxs
                         if abs(per_bout_median[bi] - pooled_median) > mad_k * mad]

    keep = [bi for bi in bout_idxs if bi not in outlier_bouts] or bout_idxs
    final_scale = float(np.median(np.concatenate([filtered[bi] for bi in keep])))

    if len(medians) > 1 and med_of_medians:
        p10, p90 = np.percentile(medians, [10, 90])
        spread_pct = float((p90 - p10) / med_of_medians * 100)
    else:
        spread_pct = 0.0

    # Third stability signal (alongside spread_pct): plain CV (std/mean) of
    # the per-bout medians, over the SAME population as spread_pct (all
    # bouts, before outlier exclusion) -- "does this fly's scale hold steady
    # across its bouts?" Generic to every scale_keypoints mode.
    mean_of_medians = float(np.mean(medians))
    scale_cv_across_bouts = (float(np.std(medians) / mean_of_medians)
                             if len(medians) > 1 and mean_of_medians else 0.0)

    return {
        "scale": final_scale,
        "n_bouts": len(bout_idxs),
        "n_frames": int(sum(a.size for a in filtered.values())),
        "per_bout_median": per_bout_median,
        "outlier_bouts": outlier_bouts,
        "spread_pct": spread_pct,
        "scale_cv_across_bouts": scale_cv_across_bouts,
    }


def coincident_fraction(kp3d_a: np.ndarray, kp3d_b: np.ndarray) -> float:
    """Fraction of frames where the two slots' 3D centroids are closer than
    ``COINCIDENT_BL_THRESH`` (0.35) "body lengths".

    Body length is the median per-frame RMS keypoint spread of slot A --
    matches ``scripts.benchmark.metrics.proximity_bl``'s convention exactly
    (reused here) so the two agree. NaN-safe; returns 0.0 when either input
    has no usable (finite) frames.
    """
    import warnings

    a = np.asarray(kp3d_a, dtype=np.float64)
    b = np.asarray(kp3d_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        d = proximity_bl(a, b)
    d = np.asarray(d)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    return float(np.mean(d < COINCIDENT_BL_THRESH))


def _find_duplicate_slot_bouts(kp3d_by_fly_bout: Dict[int, Dict[int, np.ndarray]], *,
                               coincident_thresh: float = DEFAULT_COINCIDENT_FRAC_THRESH
                               ) -> List[int]:
    """Bout indices where >= 2 fly slots are coincident often enough to be
    the same physical animal (see ``coincident_fraction``). A bout with only
    one fly slot present is never flagged."""
    by_bout: Dict[int, Dict[int, np.ndarray]] = {}
    for fly, per_bout in kp3d_by_fly_bout.items():
        for idx, kp3d in per_bout.items():
            by_bout.setdefault(idx, {})[fly] = kp3d

    flagged = []
    for idx in sorted(by_bout):
        flies = sorted(by_bout[idx])
        if len(flies) < 2:
            continue
        for i in range(len(flies)):
            for j in range(i + 1, len(flies)):
                frac = coincident_fraction(by_bout[idx][flies[i]], by_bout[idx][flies[j]])
                if frac > coincident_thresh:
                    flagged.append(idx)
                    break
            else:
                continue
            break
    return flagged


def _bout_dirs(run_root: Path) -> List[Path]:
    bouts_dir = Path(run_root) / "bouts"
    if not bouts_dir.exists():
        return []
    return [d for d in sorted(bouts_dir.iterdir())
            if d.is_dir() and _BOUT_DIR_RE.match(d.name)]


def _usable_male_fly(value) -> Optional[int]:
    """``male_fly`` is only usable if it is a real ``int`` (not a bool, not a
    numeric string, not a float) equal to 0 or 1. A well-formed sex.json that
    simply omits the key, or sets it to ``null``, yields ``None`` from
    ``dict.get`` -- treated as unusable here, same as any other malformed
    value, rather than silently forming a length-1 {None} agreement set that
    would defeat this whole gate."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value not in (0, 1):
        return None
    return value


def _determine_identity(run_root: Path) -> "tuple[str, str]":
    """Whether fly0/fly1 is a stable individual label across this
    recording's bouts.

    Identity is assigned PER BOUT by the tracker; the fly0/fly1 slot is
    otherwise arbitrary. Only once a recording has been sex-canonicalized
    (every bout dir carries a ``sex.json`` sibling of the fly dirs, written
    by the canonicalization step, all agreeing on a USABLE ``male_fly``, see
    ``_usable_male_fly``) is fly0/fly1 a stable per-individual label.
    Measured counter-example (uncanonicalized 2026_04_02_14_54_28): fly0 >
    fly1 in bout 12 but fly0 < fly1 in bout 17 -- pooling per-slot across
    bouts there would blend the two individuals.

    Returns (identity, reason) with identity in {"canonical", "unknown"}.
    """
    bout_dirs = _bout_dirs(run_root)
    if not bout_dirs:
        return "unknown", "no bout directories found"
    n = len(bout_dirs)
    male_flies = []
    missing_file = 0
    bad_value = 0
    for d in bout_dirs:
        sex_path = d / "sex.json"
        if not sex_path.exists():
            missing_file += 1
            continue
        try:
            with open(sex_path) as f:
                raw_value = json.load(f).get("male_fly")
        except (OSError, json.JSONDecodeError):
            missing_file += 1
            continue
        usable = _usable_male_fly(raw_value)
        if usable is None:
            bad_value += 1
            continue
        male_flies.append(usable)
    if missing_file or bad_value:
        parts = []
        if missing_file:
            parts.append(f"{missing_file}/{n} bouts lack (or have unreadable) sex.json")
        if bad_value:
            parts.append(f"{bad_value}/{n} bouts have sex.json without a usable male_fly")
        return "unknown", "; ".join(parts)
    uniq = sorted(set(male_flies))
    if len(uniq) != 1:
        return "unknown", f"bouts disagree on male_fly: {uniq}"
    return "canonical", f"all {n} bouts have sex.json, male_fly={uniq[0]}"


def _fly_ids(run_root: Path) -> List[int]:
    bouts_dir = Path(run_root) / "bouts"
    if not bouts_dir.exists():
        return []
    ids = set()
    for bout_dir in bouts_dir.iterdir():
        if not bout_dir.is_dir() or not _BOUT_DIR_RE.match(bout_dir.name):
            continue
        for fly_dir in bout_dir.iterdir():
            m = _FLY_DIR_RE.match(fly_dir.name) if fly_dir.is_dir() else None
            if m:
                ids.add(int(m.group(1)))
    return sorted(ids)


def _load_kp3d_by_fly_bout(run_root: Path, fly_ids: List[int]
                           ) -> Dict[int, Dict[int, np.ndarray]]:
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for fly in fly_ids:
        per_bout = {}
        for p in bout_kp3d_paths(run_root, fly):
            idx = _bout_idx_from_path(p)
            with np.load(p) as z:
                per_bout[idx] = z["kp3d"]
        if per_bout:
            out[fly] = per_bout
    return out


def _per_fly_scale_arrays(kp3d_by_fly_bout: Dict[int, Dict[int, np.ndarray]],
                          exclude_bouts: set, kp_names: List[str], model_xml: str,
                          trunk_names: List[str], estimator: str, *,
                          scale_keypoints: str = "trunk", include_thorax: bool = False
                          ) -> Dict[int, Dict[int, np.ndarray]]:
    """Per bout-fly scale arrays, feeding ``robust_scale``.

    For ``scale_keypoints='rigid_segment'`` each bout-fly's array is its
    per-pair implied scales (``per_bout_segment_scale``) -- so
    ``robust_scale``'s own per-bout median of that array IS "the median of
    that bout's per-pair implied scales", per spec. Every other mode keeps
    the existing per-FRAME scale array (``per_frame_scales``).
    """
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for fly, per_bout in kp3d_by_fly_bout.items():
        scale_bouts = {}
        for idx, kp3d in per_bout.items():
            if idx in exclude_bouts:
                continue
            if scale_keypoints == "rigid_segment":
                try:
                    scales = per_bout_segment_scale(kp3d, kp_names, model_xml,
                                                    include_thorax=include_thorax)
                except ValueError:
                    continue
            else:
                scales = per_frame_scales(kp3d, kp_names, model_xml,
                                          trunk_names=trunk_names, estimator=estimator)
            if scales.size:
                scale_bouts[idx] = scales
        if scale_bouts:
            out[fly] = scale_bouts
    return out


def _per_fly_segment_diagnostics(kp3d_by_fly_bout: Dict[int, Dict[int, np.ndarray]],
                                 exclude_bouts: set, kp_names: List[str], model_xml: str, *,
                                 include_thorax: bool = False
                                 ) -> Dict[int, Dict[int, dict]]:
    """Per bout-fly ``segment_scale_diagnostics`` (within/across-bone CV),
    only computed for ``scale_keypoints='rigid_segment'``."""
    out: Dict[int, Dict[int, dict]] = {}
    for fly, per_bout in kp3d_by_fly_bout.items():
        bout_diags = {}
        for idx, kp3d in per_bout.items():
            if idx in exclude_bouts:
                continue
            try:
                bout_diags[idx] = segment_scale_diagnostics(
                    kp3d, kp_names, model_xml, include_thorax=include_thorax)
            except ValueError:
                continue
        if bout_diags:
            out[fly] = bout_diags
    return out


def _attach_segment_quality(diag: dict, bout_diags: Dict[int, dict]) -> None:
    """Mutate a ``robust_scale`` result in place with rigid-segment
    keypoint-quality aggregates (mean over the KEPT, non-outlier bouts).

    Mirrors ``robust_scale``'s OWN fallback for its ``final_scale`` (``keep =
    [...] or bout_idxs``): if every bout happened to be flagged an outlier
    (e.g. near-zero real spread makes the MAD threshold degenerate), fall
    back to using all bouts rather than aggregating over an empty set.
    """
    all_bouts = list(diag["per_bout_median"])
    kept = [bi for bi in all_bouts if bi not in diag["outlier_bouts"]] or all_bouts
    within = [bout_diags[bi]["within_bone_cv"] for bi in kept if bi in bout_diags]
    across = [bout_diags[bi]["across_bone_cv"] for bi in kept if bi in bout_diags]
    npairs = [bout_diags[bi]["n_pairs_used"] for bi in kept if bi in bout_diags]
    diag["within_bone_cv"] = float(np.mean(within)) if within else float("nan")
    diag["across_bone_cv"] = float(np.mean(across)) if across else float("nan")
    diag["n_pairs_used"] = int(round(float(np.mean(npairs)))) if npairs else 0


def estimate_run_root(run_root: Path, cfg, *, estimator: str = "umeyama",
                      scale_keypoints: str = "trunk", include_thorax: bool = False,
                      coincident_thresh: float = DEFAULT_COINCIDENT_FRAC_THRESH) -> dict:
    """Recording-level robust body-scale estimate.

    ``cfg`` needs ``cfg.model.KP_NAMES`` (list of keypoint names) and
    ``cfg.mjcf_path`` (MuJoCo model XML path).

    Two independent defects are guarded against:

    1. Fly identity is NOT guaranteed stable across a recording's bouts: it
       is assigned per bout by the tracker, so pooling per fly-slot across
       bouts silently blends two different individuals unless the recording
       has been sex-canonicalized (see ``_determine_identity``). When
       canonical, this returns a per-fly scale (``scale_by_fly``); otherwise
       it pools every bout-fly into one recording-level scale and sets
       ``scale_by_fly`` to ``None`` -- still strictly better than either a
       first-bout estimate or a slot-blended per-fly split (measured
       per-individual spread ~8% vs. per-bout estimator noise ~20%).
    2. Some bouts have BOTH fly slots tracking the SAME animal (SAM3 locks a
       camera's "other" slot onto the visible animal when the other is
       occluded). Any bout where two slots are coincident more than
       ``coincident_thresh`` of frames (``coincident_fraction``) has BOTH
       slots excluded from all pooling, before MAD outlier rejection runs --
       a duplicated slot contains a real (just wrong) fly, so its per-frame
       scale looks normal and MAD rejection alone would not catch it.

    ``scale_keypoints``:
      - ``'trunk'``: rigid-trunk-marker Procrustes/norm-ratio fit (``estimator``
        selects which), the historical default.
      - ``'all'``: same fit, every keypoint.
      - ``'rigid_segment'``: pose-invariant DIRECT rigid-segment-length
        measurement (see module docstring / ``rigid_segment_pairs``) -- the
        pipeline default as of Task 16, since it matches the LEG kinematic
        chain STAC's IK actually solves and is not confounded by pose.
        ``estimator`` is IGNORED for this mode (there is nothing to fit); a
        ``UserWarning`` is raised if it was set to anything but the default
        ``'umeyama'``, so a caller relying on it silently doesn't get fooled.
        Additionally attaches physics-based keypoint-quality diagnostics
        (``within_bone_cv``, ``across_bone_cv``, ``n_pairs_used`` -- see
        ``segment_scale_diagnostics``) to each fly's (or the pooled) entry
        under ``diagnostics``.
      ``include_thorax`` (rigid_segment only): also use the 3 thorax-plate
      pairs (default False -- see ``THORAX_PAIRS``).

    Returns::

        {"scale": float,                       # back-compat scalar
         "scale_by_fly": {"0": f, "1": f} | None,
         "estimator": estimator, "scale_keypoints": scale_keypoints,
         "method": "recording_robust",
         "identity": "canonical" | "unknown", "identity_reason": str,
         "duplicate_slot_bouts": [bout_idx, ...],
         "diagnostics": {...}}                 # per-fly or {"pooled": ...};
                                                # each entry additionally has
                                                # "scale_cv_across_bouts" (all
                                                # modes) and, for
                                                # scale_keypoints=
                                                # 'rigid_segment',
                                                # "within_bone_cv" /
                                                # "across_bone_cv" /
                                                # "n_pairs_used".
    """
    run_root = Path(run_root)
    kp_names = list(cfg.model.KP_NAMES)
    model_xml = str(cfg.mjcf_path)

    if scale_keypoints == "trunk":
        trunk_names = list(DEFAULT_TRUNK_KEYPOINTS)
    elif scale_keypoints == "all":
        trunk_names = list(kp_names)
    elif scale_keypoints == "rigid_segment":
        trunk_names = None  # unused -- see the estimator-ignored warning below
        if estimator != "umeyama":
            warnings.warn(
                "estimate_run_root: `estimator` is ignored for "
                f"scale_keypoints='rigid_segment' (got estimator={estimator!r}) "
                "-- a rigid segment's length is measured directly, not fit.",
                UserWarning, stacklevel=2)
    else:
        raise ValueError(
            f"estimate_run_root: scale_keypoints must be 'trunk', 'all', or "
            f"'rigid_segment', got {scale_keypoints!r}")

    identity, identity_reason = _determine_identity(run_root)
    fly_ids = _fly_ids(run_root)
    kp3d_by_fly_bout = _load_kp3d_by_fly_bout(run_root, fly_ids)
    if not kp3d_by_fly_bout:
        raise ValueError(f"estimate_run_root: no usable bout-fly kp3d under {run_root}")

    duplicate_slot_bouts = _find_duplicate_slot_bouts(
        kp3d_by_fly_bout, coincident_thresh=coincident_thresh)
    exclude_bouts = set(duplicate_slot_bouts)

    per_fly_per_bout = _per_fly_scale_arrays(
        kp3d_by_fly_bout, exclude_bouts, kp_names, model_xml,
        trunk_names, estimator, scale_keypoints=scale_keypoints,
        include_thorax=include_thorax)

    if not per_fly_per_bout:
        raise ValueError(
            f"estimate_run_root: no usable bout-fly kp3d under {run_root} after "
            f"excluding duplicate-slot bouts {duplicate_slot_bouts}")

    segment_diag_by_fly: Dict[int, Dict[int, dict]] = {}
    if scale_keypoints == "rigid_segment":
        segment_diag_by_fly = _per_fly_segment_diagnostics(
            kp3d_by_fly_bout, exclude_bouts, kp_names, model_xml,
            include_thorax=include_thorax)

    result = {
        "estimator": estimator,
        "scale_keypoints": scale_keypoints,
        "method": "recording_robust",
        "identity": identity,
        "identity_reason": identity_reason,
        "duplicate_slot_bouts": duplicate_slot_bouts,
    }

    if identity == "canonical":
        scale_by_fly = {}
        diagnostics = {}
        for fly, per_bout in per_fly_per_bout.items():
            diag = robust_scale(per_bout)
            if scale_keypoints == "rigid_segment":
                _attach_segment_quality(diag, segment_diag_by_fly.get(fly, {}))
            scale_by_fly[str(fly)] = diag["scale"]
            diagnostics[str(fly)] = diag
        result["scale_by_fly"] = scale_by_fly
        result["scale"] = float(np.median(list(scale_by_fly.values())))
        result["diagnostics"] = diagnostics
    else:
        # Fly slot is not a stable individual label here -- pool every
        # bout-fly together instead of blending two individuals per slot.
        # robust_scale only needs hashable, sortable keys (its "bout index"
        # is really just a per-entry disambiguator); human-readable
        # "fly<F>:bout<idx>" strings keep any outlier_bouts diagnostic
        # readable instead of an opaque encoded int.
        pooled = {}
        pooled_segment_diag = {}
        for fly, per_bout in per_fly_per_bout.items():
            for idx, arr in per_bout.items():
                pooled[f"fly{fly}:bout{idx}"] = arr
        for fly, bout_diags in segment_diag_by_fly.items():
            for idx, d in bout_diags.items():
                pooled_segment_diag[f"fly{fly}:bout{idx}"] = d
        diag = robust_scale(pooled)
        if scale_keypoints == "rigid_segment":
            _attach_segment_quality(diag, pooled_segment_diag)
        result["scale_by_fly"] = None
        result["scale"] = diag["scale"]
        result["diagnostics"] = {"pooled": diag}

    return result


def _load_anatomy_cfg(anatomy_path: str):
    from omegaconf import OmegaConf
    repo_root = Path(__file__).resolve().parents[1]
    anatomy_full = Path(anatomy_path)
    if not anatomy_full.is_absolute():
        anatomy_full = repo_root / anatomy_full
    raw = OmegaConf.load(str(anatomy_full))
    # `mjcf_path` in the anatomy yaml interpolates ${paths.body_model_dir};
    # supply it directly (this repo's own models/ dir) rather than pulling in
    # hydra's full paths config-group machinery, which this standalone CLI
    # has no need for otherwise. Accessed lazily (no eager OmegaConf.resolve)
    # so the yaml's OTHER self-referential interpolations (e.g.
    # model.MJCF_PATH: ${anatomy.mjcf_path}, unused here) don't need to
    # resolve too.
    container = OmegaConf.create({"paths": {"body_model_dir": str(repo_root / "models")}})
    return OmegaConf.merge(container, raw)


def _print_scale_diag(label: str, s: Optional[float], diag: dict) -> None:
    """Print one fly's (or "pooled") robust_scale diagnostics, including the
    scale_cv_across_bouts signal (all modes) and, for scale_keypoints=
    'rigid_segment', the within/across-bone-CV keypoint-quality signals plus
    a WARNING when within_bone_cv exceeds WITHIN_BONE_CV_WARN_THRESH."""
    scale_val = diag["scale"] if s is None else s
    outliers = diag["outlier_bouts"]
    print(f"  {label}: scale={scale_val:.6f} n_bouts={diag['n_bouts']} "
          f"n_frames={diag['n_frames']} spread_pct={diag['spread_pct']:.2f}% "
          f"scale_cv_across_bouts={diag.get('scale_cv_across_bouts', 0.0) * 100:.2f}%"
          + (f" outlier_bouts={outliers}" if outliers else ""))
    if "within_bone_cv" in diag:
        print(f"    within_bone_cv={diag['within_bone_cv'] * 100:.2f}% "
              f"across_bone_cv={diag['across_bone_cv'] * 100:.2f}% "
              f"n_pairs_used={diag['n_pairs_used']}")
        if diag["within_bone_cv"] > WITHIN_BONE_CV_WARN_THRESH:
            print(f"    WARNING: {label} within_bone_cv "
                  f"{diag['within_bone_cv'] * 100:.1f}% exceeds "
                  f"{WITHIN_BONE_CV_WARN_THRESH * 100:.0f}% -- keypoints are "
                  f"physically inconsistent (bone lengths not constant across "
                  f"frames); this {label}'s scale estimate should not be trusted.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Recording-level robust body-scale estimator "
                    "(replaces the scale-from-first-bout defect).")
    ap.add_argument("--run-root", required=True, help="<recording>/pose dir")
    ap.add_argument("--estimator", default="umeyama", choices=["umeyama", "norm_ratio"],
                    help="ignored when --scale-keypoints=rigid_segment")
    ap.add_argument("--scale-keypoints", default="rigid_segment",
                    choices=["trunk", "all", "rigid_segment"])
    ap.add_argument("--include-thorax", action="store_true",
                    help="rigid_segment only: also use the 3 thorax-plate pairs "
                         "(default: leg chain only, see THORAX_PAIRS)")
    ap.add_argument("--anatomy", default="configs/anatomy/v1.yaml",
                    help="anatomy config for KP_NAMES + mjcf path")
    ap.add_argument("--out", default=None, help="default <run-root>/scale.json")
    ap.add_argument("--dry-run", action="store_true", help="print, don't write")
    args = ap.parse_args(argv)

    run_root = Path(args.run_root)
    cfg = _load_anatomy_cfg(args.anatomy)
    result = estimate_run_root(run_root, cfg, estimator=args.estimator,
                               scale_keypoints=args.scale_keypoints,
                               include_thorax=args.include_thorax)

    out_path = Path(args.out) if args.out else run_root / "scale.json"

    print(f"[estimate_recording_scale] run_root={run_root}")
    print(f"  identity={result['identity']} ({result['identity_reason']})")
    if result["duplicate_slot_bouts"]:
        print(f"  duplicate_slot_bouts={result['duplicate_slot_bouts']} "
              f"(both fly slots coincident >{DEFAULT_COINCIDENT_FRAC_THRESH:.0%} "
              f"of frames -- likely SAM3 locked both slots onto the same "
              f"animal; both slots excluded from scale pooling for these bouts)")
    if result["scale_by_fly"] is None:
        print("  WARNING: fly identity is not stable across this recording's bouts "
              "-- per-fly scales were NOT computed; using one pooled "
              "recording-level scale for all bout-flies instead.")
        _print_scale_diag("pooled", None, result["diagnostics"]["pooled"])
    else:
        for fly in sorted(result["scale_by_fly"]):
            s = result["scale_by_fly"][fly]
            diag = result["diagnostics"][fly]
            _print_scale_diag(f"fly{fly}", s, diag)
    print(f"  overall scale (back-compat): {result['scale']:.6f}")

    if args.dry_run:
        print(f"[dry-run] would write {out_path}")
        return

    tmp = str(out_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, str(out_path))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
