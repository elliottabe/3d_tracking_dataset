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


def compute_pair_validity(
    fly0_kp: np.ndarray,
    fly1_kp: np.ndarray,
    kp_names: Sequence[str],
    cfg: Optional[PairValidityConfig] = None,
    swap_state: Optional[np.ndarray] = None,
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

    Returns
    -------
    dict with keys:
        valid_fly0, valid_fly1, valid_both : (T,) bool
        pair_state : (T,) uint8    (0 none, 1 fly0, 2 fly1, 3 both)
        bout_pair_class : str      'paired'|'fly0_only'|'fly1_only'|'mixed'|'empty'
        floor_z_fly0, floor_z_fly1 : float
        identity_valid : (T,) bool
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

    valid_fly0 = filt0 & grd0 & identity_valid
    valid_fly1 = filt1 & grd1 & identity_valid
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
        bout_pair_class=bout_pair_class,
        floor_z_fly0=floor0,
        floor_z_fly1=floor1,
        n_frames=T,
        n_valid_both=n_both,
        n_valid_fly0=n_fly0,
        n_valid_fly1=n_fly1,
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
