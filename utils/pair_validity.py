"""Per-frame validity mask for paired (two-fly) keypoint bouts.

Used by the courtship pipeline to tag, for each frame of each bout, whether
each fly is "tracked well" (filter passed, identity confident, on the ground)
and whether both flies are simultaneously tracked well. Downstream code can
then separate paired analyses (both flies valid) from solo analyses (only
one fly valid) without re-running preprocessing.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# pair_state encoding
PAIR_NONE = np.uint8(0)
PAIR_FLY0_ONLY = np.uint8(1)
PAIR_FLY1_ONLY = np.uint8(2)
PAIR_BOTH = np.uint8(3)


@dataclass
class PairValidityConfig:
    enabled: bool = True
    critical_kp_patterns: Sequence[str] = field(
        default_factory=lambda: [
            "*Tarsus*",
            "*Claw*",
            "*Scutellum*",
            "*Scutum*",
        ]
    )
    ground_kp_patterns: Sequence[str] = field(
        default_factory=lambda: ["*Tarsus*", "*Claw*"]
    )
    ground_epsilon_mm: float = 0.05
    floor_percentile: float = 5.0
    swap_guard_frames: int = 5
    min_paired_frames: int = 30
    min_solo_frames: int = 30
    # Identity-collapse detector. When > 0, frames where the two flies'
    # ``colocation_centroid_kp`` are closer than this threshold are marked
    # invalid for BOTH flies (see ``compute_colocation_mask``). 0 disables
    # the check. For courtship, ~1.0 mm (half a Drosophila body length) is
    # a reasonable floor.
    min_pair_separation_mm: float = 0.0
    colocation_centroid_kp: str = "Scutellum"


def _match_indices(kp_names: Sequence[str], patterns: Sequence[str]) -> List[int]:
    idx = []
    for i, name in enumerate(kp_names):
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            idx.append(i)
    return idx


def _filter_valid(kp: np.ndarray, critical_idx: Sequence[int]) -> np.ndarray:
    """A frame is filter-valid iff every critical keypoint is finite."""
    if len(critical_idx) == 0:
        return np.ones(kp.shape[0], dtype=bool)
    sub = kp[:, list(critical_idx), :]
    return np.all(np.isfinite(sub), axis=(1, 2))


def _ground_valid(
    kp: np.ndarray, ground_idx: Sequence[int], percentile: float, epsilon: float
) -> Tuple[np.ndarray, float]:
    """Per-frame: at least one ground keypoint is within epsilon of the floor.

    Floor z is estimated as the `percentile`-th percentile of the ground
    keypoints' z values across the bout (ignoring NaNs).
    """
    if len(ground_idx) == 0:
        return np.ones(kp.shape[0], dtype=bool), float("nan")
    zs = kp[:, list(ground_idx), 2]  # (T, K)
    finite_zs = zs[np.isfinite(zs)]
    if finite_zs.size == 0:
        return np.zeros(kp.shape[0], dtype=bool), float("nan")
    floor_z = float(np.percentile(finite_zs, percentile))
    # min over ground kps per frame; nan propagates → invalid
    with np.errstate(invalid="ignore"):
        min_z = np.nanmin(zs, axis=1)
    grounded = np.isfinite(min_z) & (min_z <= floor_z + epsilon)
    return grounded, floor_z


def _guard_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a True mask by `radius` frames on each side."""
    if radius <= 0 or not mask.any():
        return mask
    T = mask.shape[0]
    out = mask.copy()
    for shift in range(1, radius + 1):
        out[shift:] |= mask[:-shift]
        out[:-shift] |= mask[shift:]
    return out


def compute_colocation_mask(
    fly0_kp: np.ndarray,
    fly1_kp: np.ndarray,
    kp_names: Sequence[str],
    min_separation_mm: float,
    centroid_kp: str = "Scutellum",
) -> np.ndarray:
    """Per-frame mask flagging identity-collapse frames in a paired bout.

    A frame is True iff the two flies' ``centroid_kp`` positions are
    measurable and closer than ``min_separation_mm``. The JARVIS multi-peak
    tracker occasionally produces two output tracks that both lock onto the
    same physical animal (e.g. when one fly is occluded); that failure mode
    shows up as an inter-fly centroid distance that collapses to ~0.

    NaN handling: frames with any NaN in either centroid are returned as
    False — we only invalidate frames we can actually measure. Downstream
    validity logic will still drop NaN frames via ``_filter_valid``.

    Args:
        fly0_kp, fly1_kp: (T, N, 3) keypoints in a *shared* world frame.
            Passing egocentric arrays is useless because each fly is at the
            origin in its own frame.
        kp_names: ordered keypoint names (length N, shared by both flies).
        min_separation_mm: inter-centroid distance threshold. 0 disables
            the check and returns an all-False mask.
        centroid_kp: keypoint name to use as the body centroid. Must be in
            ``kp_names``. If missing, the function returns an all-False
            mask (fail-open).

    Returns:
        (T,) bool ndarray. True on collapse frames.
    """
    T = fly0_kp.shape[0]
    if min_separation_mm <= 0 or T == 0:
        return np.zeros(T, dtype=bool)
    if fly0_kp.shape != fly1_kp.shape:
        return np.zeros(T, dtype=bool)
    try:
        idx = list(kp_names).index(centroid_kp)
    except ValueError:
        return np.zeros(T, dtype=bool)
    c0 = fly0_kp[:, idx, :]
    c1 = fly1_kp[:, idx, :]
    diff = c0 - c1
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(diff, axis=1)
    mask = np.isfinite(dist) & (dist < float(min_separation_mm))
    return mask


def compute_single_fly_validity(
    fly_kp: np.ndarray,
    kp_names: Sequence[str],
    cfg: Optional[PairValidityConfig] = None,
    edge_nan_mask: Optional[np.ndarray] = None,
) -> dict:
    """Per-frame validity mask for a *single* fly (no cross-fly check).

    This is the honest version of what the per-fly preprocessing stage can
    actually compute, since at that point only one fly's keypoints are in
    memory. The previous code wedged ``compute_pair_validity`` into this
    spot by passing the same array twice — that produced
    ``valid_both ≡ valid_fly0`` and a fake ``identity_valid`` mask. Use this
    helper instead and let the real cross-fly mask be built later in
    ``batch_split_valid_bouts.py`` after both flies are loaded.

    Args:
        fly_kp: (T, N, 3) keypoints. Already filtered + interpolated.
        kp_names: ordered keypoint names (length N).
        cfg: pair validity config.
        edge_nan_mask: optional (T, N) bool — True on frame-keypoints that were
            originally in a leading/trailing NaN run. Any frame with an edge
            NaN on ANY keypoint is marked not-filter-OK (hence not-valid).
            This handles the case where short edge gaps were filled by bounded
            linear extrapolation: the values are finite but still phantom, so
            the plain ``isfinite`` check in ``_filter_valid`` would otherwise
            trust them.

    Returns a dict with keys ``valid_fly``, ``filter_ok``, ``ground_ok``,
    ``floor_z``, ``n_frames``.
    """
    if cfg is None:
        cfg = PairValidityConfig()
    T = fly_kp.shape[0]
    critical_idx = _match_indices(kp_names, cfg.critical_kp_patterns)
    ground_idx = _match_indices(kp_names, cfg.ground_kp_patterns)
    filt = _filter_valid(fly_kp, critical_idx)
    grd, floor = _ground_valid(
        fly_kp, ground_idx, cfg.floor_percentile, cfg.ground_epsilon_mm
    )
    if edge_nan_mask is not None:
        edge_bad = np.asarray(edge_nan_mask, dtype=bool)
        if edge_bad.shape == (T, fly_kp.shape[1]):
            # A frame is phantom if ANY keypoint sat in an edge NaN run.
            filt = filt & ~edge_bad.any(axis=1)
    return dict(
        valid_fly=filt & grd,
        filter_ok=filt,
        ground_ok=grd,
        floor_z=floor,
        n_frames=T,
    )


def compute_pair_validity(
    fly0_kp: np.ndarray,
    fly1_kp: np.ndarray,
    kp_names: Sequence[str],
    cfg: Optional[PairValidityConfig] = None,
    swap_state: Optional[np.ndarray] = None,
    edge_nan_mask_fly0: Optional[np.ndarray] = None,
    edge_nan_mask_fly1: Optional[np.ndarray] = None,
) -> dict:
    """Compute per-frame validity masks for a paired bout.

    Parameters
    ----------
    fly0_kp, fly1_kp : ndarray (T, N, 3)
        Filtered keypoints for each fly over the same frame range.
    kp_names : list[str]
        Keypoint names (length N, shared by both flies).
    cfg : PairValidityConfig, optional
    swap_state : ndarray (T,) bool, optional
        Cumulative relink state from utils.identity_relink.relink_pair —
        True where the assignment was flipped relative to the input. The
        keypoints passed in are assumed to already be relink-corrected, so
        only *toggle events* in this state mark uncertain frames; toggles
        are dilated by `cfg.swap_guard_frames` on each side.
    edge_nan_mask_fly0, edge_nan_mask_fly1 : ndarray (T, N) bool, optional
        Per-fly edge-NaN masks carried over from preprocessing. Frames where
        ANY keypoint was in a leading/trailing NaN run (whether filled by
        bounded extrapolation or left NaN) are marked not-valid for that fly.
        Callers in the joint-relink pathway should pass these *after* applying
        any swap_state-driven row exchange so the masks remain aligned with
        the post-swap keypoints.

    Returns
    -------
    dict with keys:
        valid_fly0, valid_fly1, valid_both : (T,) bool
        pair_state : (T,) uint8    (0 none, 1 fly0, 2 fly1, 3 both)
        bout_pair_class : str      'paired'|'fly0_only'|'fly1_only'|'mixed'|'empty'
        floor_z_fly0, floor_z_fly1 : float
        identity_valid : (T,) bool
        pair_colocated : (T,) bool  — True on frames where the two flies'
            centroid keypoints are within ``cfg.min_pair_separation_mm``
            (identity-collapse detector; False everywhere when disabled).
        n_colocated : int
    """
    if cfg is None:
        cfg = PairValidityConfig()
    assert fly0_kp.shape == fly1_kp.shape, (
        f"shape mismatch {fly0_kp.shape} vs {fly1_kp.shape}"
    )
    T = fly0_kp.shape[0]

    critical_idx = _match_indices(kp_names, cfg.critical_kp_patterns)
    ground_idx = _match_indices(kp_names, cfg.ground_kp_patterns)

    filt0 = _filter_valid(fly0_kp, critical_idx)
    filt1 = _filter_valid(fly1_kp, critical_idx)

    def _edge_bad(mask, fly_kp):
        if mask is None:
            return None
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != (T, fly_kp.shape[1]):
            return None
        return arr.any(axis=1)

    edge_bad0 = _edge_bad(edge_nan_mask_fly0, fly0_kp)
    edge_bad1 = _edge_bad(edge_nan_mask_fly1, fly1_kp)
    if edge_bad0 is not None:
        filt0 = filt0 & ~edge_bad0
    if edge_bad1 is not None:
        filt1 = filt1 & ~edge_bad1

    grd0, floor0 = _ground_valid(
        fly0_kp, ground_idx, cfg.floor_percentile, cfg.ground_epsilon_mm
    )
    grd1, floor1 = _ground_valid(
        fly1_kp, ground_idx, cfg.floor_percentile, cfg.ground_epsilon_mm
    )

    if swap_state is not None:
        swap = np.asarray(swap_state, dtype=bool)
        if swap.shape[0] != T:
            raise ValueError(
                f"swap_state length {swap.shape[0]} != bout length {T}"
            )
        # Only the toggle events are uncertain — frames deep inside a stable
        # (un)swapped segment are reliable because the relink correction has
        # already been applied to the keypoints.
        toggles = np.zeros(T, dtype=bool)
        if T >= 2:
            toggles[1:] = swap[1:] != swap[:-1]
        guarded = _guard_dilate(toggles, cfg.swap_guard_frames)
        identity_valid = ~guarded
    else:
        identity_valid = np.ones(T, dtype=bool)

    # Identity-collapse detector. When enabled (min_pair_separation_mm > 0),
    # frames where the two flies' centroid keypoints are closer than the
    # threshold are treated as invalid for *both* flies — we have no way to
    # tell which track was real, so the safe policy is to drop both.
    pair_colocated = compute_colocation_mask(
        fly0_kp,
        fly1_kp,
        kp_names,
        min_separation_mm=cfg.min_pair_separation_mm,
        centroid_kp=cfg.colocation_centroid_kp,
    )
    not_colocated = ~pair_colocated

    valid_fly0 = filt0 & grd0 & identity_valid & not_colocated
    valid_fly1 = filt1 & grd1 & identity_valid & not_colocated
    valid_both = valid_fly0 & valid_fly1

    pair_state = (
        valid_fly0.astype(np.uint8) | (valid_fly1.astype(np.uint8) << 1)
    )

    n_both = int(valid_both.sum())
    n_fly0 = int(valid_fly0.sum())
    n_fly1 = int(valid_fly1.sum())
    bout_pair_class = classify_bout(
        n_both, n_fly0, n_fly1, cfg.min_paired_frames, cfg.min_solo_frames
    )

    return dict(
        valid_fly0=valid_fly0,
        valid_fly1=valid_fly1,
        valid_both=valid_both,
        pair_state=pair_state,
        identity_valid=identity_valid,
        pair_colocated=pair_colocated,
        bout_pair_class=bout_pair_class,
        floor_z_fly0=floor0,
        floor_z_fly1=floor1,
        n_frames=T,
        n_valid_both=n_both,
        n_valid_fly0=n_fly0,
        n_valid_fly1=n_fly1,
        n_colocated=int(pair_colocated.sum()),
    )


def classify_bout(
    n_both: int,
    n_fly0: int,
    n_fly1: int,
    min_paired: int,
    min_solo: int,
) -> str:
    has_pair = n_both >= min_paired
    has_f0 = n_fly0 >= min_solo
    has_f1 = n_fly1 >= min_solo
    if has_pair and (n_fly0 - n_both >= min_solo or n_fly1 - n_both >= min_solo):
        return "mixed"
    if has_pair:
        return "paired"
    if has_f0 and not has_f1:
        return "fly0_only"
    if has_f1 and not has_f0:
        return "fly1_only"
    if has_f0 and has_f1:
        # both have solo frames but never paired enough → mixed solo
        return "mixed"
    return "empty"


def pair_validity_config_from_dict(d) -> PairValidityConfig:
    """Build a PairValidityConfig from a plain dict / OmegaConf node."""
    if d is None:
        return PairValidityConfig()
    get = (lambda k, default: d.get(k, default)) if hasattr(d, "get") else (
        lambda k, default: getattr(d, k, default)
    )
    return PairValidityConfig(
        enabled=bool(get("enabled", True)),
        critical_kp_patterns=list(get("critical_kp_patterns",
                                      PairValidityConfig().critical_kp_patterns)),
        ground_kp_patterns=list(get("ground_kp_patterns",
                                    PairValidityConfig().ground_kp_patterns)),
        ground_epsilon_mm=float(get("ground_epsilon_mm", 0.05)),
        floor_percentile=float(get("floor_percentile", 5.0)),
        swap_guard_frames=int(get("swap_guard_frames", 5)),
        min_paired_frames=int(get("min_paired_frames", 30)),
        min_solo_frames=int(get("min_solo_frames", 30)),
        min_pair_separation_mm=float(get("min_pair_separation_mm", 0.0)),
        colocation_centroid_kp=str(get("colocation_centroid_kp", "Scutellum")),
    )


if __name__ == "__main__":
    # Quick synthetic smoke test: fly1 airborne in [50,60), identity swap at 80.
    rng = np.random.default_rng(0)
    T, N = 120, 6
    names = ["Scutum", "Scutellum", "ForeTarsus", "MidTarsus", "HindTarsus", "ForeClaw"]
    z_base = np.full((T, N), -0.1)
    z_base[:, :2] = -0.08  # trunk slightly higher
    fly0 = np.stack([np.zeros((T, N)), np.zeros((T, N)), z_base], axis=-1)
    fly1 = np.stack([np.ones((T, N)), np.zeros((T, N)), z_base.copy()], axis=-1)
    fly1[50:60, :, 2] = 0.5  # airborne
    swap = np.zeros(T, dtype=bool)
    swap[80:] = True
    out = compute_pair_validity(fly0, fly1, names, swap_state=swap)
    print("n_both:", out["n_valid_both"], "expected ~50")
    print("class:", out["bout_pair_class"])
    assert not out["valid_fly1"][55]
    assert not out["valid_both"][85]   # within ±5 of toggle at frame 80
    assert out["valid_both"][110]      # deep inside swapped segment, far from toggle
    print("OK")
