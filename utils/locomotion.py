"""Running, turning and COM-height features for one fly in one bout.

Designed to feed the paper-figure notebook: computes per-frame forward /
lateral speed and turn rate from a centroid keypoint, per-frame COM
height above the estimated floor plane, and a stopped / running state
gate. All numeric features are returned in data-native units unless the
caller supplies ``body_length`` for normalisation.

Reuses :func:`utils.pair_validity._ground_valid` for the floor estimate
so this module and ``compute_pair_validity`` agree on what "ground"
means in a given bout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter

from utils.pair_validity import _ground_valid, _match_indices


@dataclass
class LocomotionConfig:
    fs: float = 800.0
    centroid_kp: str = "Scutellum"
    forward_from_kp: str = "Scutellum"
    forward_to_kp: str = "Antenna_Base"

    # Savitzky-Golay smoothing of the centroid trajectory before
    # differentiating (frames). 1 disables smoothing.
    smooth_window: int = 7
    smooth_polyorder: int = 2

    # Floor / ground detection (reused from PairValidityConfig defaults).
    ground_kp_patterns: Sequence[str] = field(
        default_factory=lambda: ("*claw*", "*Claw*", "Tarsus*", "*_Ti*")
    )
    ground_epsilon: float = 0.05
    floor_percentile: float = 5.0

    # Stopped / running classifier thresholds, in body-length / s.
    running_speed_bl: float = 0.5
    stopped_speed_bl: float = 0.3  # hysteresis floor
    min_run_ms: float = 40.0        # minimum state duration before flipping


# -----------------------------------------------------------------------------
# Centroid kinematics
# -----------------------------------------------------------------------------


def _smooth_xy(xy: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Row-wise Savitzky-Golay smoothing of a (T, 2 or 3) trajectory.

    Falls back to the raw signal if the bout is shorter than ``window``.
    """
    if window <= 1 or xy.shape[0] < window:
        return xy
    out = np.empty_like(xy, dtype=float)
    for c in range(xy.shape[1]):
        out[:, c] = savgol_filter(xy[:, c], window, polyorder, mode="interp")
    return out


def compute_centroid_velocity(
    kp_world: np.ndarray,
    kp_names: Sequence[str],
    cfg: Optional[LocomotionConfig] = None,
    body_length: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Forward / lateral speed + turn rate derived from the body axis.

    Forward axis is a per-frame unit vector from ``forward_from_kp``
    (Scutellum) to ``forward_to_kp`` (Antenna_Base). Speed is projected
    onto this axis and its perpendicular in the horizontal plane.

    Returns a dict with (T,) arrays:
        ``speed``         — |centroid velocity| in data-units / s
        ``forward_speed`` — signed projection onto the body heading
        ``lateral_speed`` — signed perpendicular component (left +)
        ``turn_rate``     — d(heading)/dt in deg/s (horizontal plane)
        ``heading``       — (T,) rad, horizontal-plane heading angle
        ``speed_bl``      — ``speed / body_length`` if body_length given
        ``forward_speed_bl``, ``lateral_speed_bl`` — normalized variants
    """
    if cfg is None:
        cfg = LocomotionConfig()
    kp_world = np.asarray(kp_world)
    if kp_world.ndim == 2:
        kp_world = kp_world.reshape(kp_world.shape[0], -1, 3)
    T = kp_world.shape[0]

    names = list(kp_names)
    i_cent = names.index(cfg.centroid_kp)
    i_from = names.index(cfg.forward_from_kp)
    i_to = names.index(cfg.forward_to_kp)

    centroid = kp_world[:, i_cent, :].astype(float)
    centroid_s = _smooth_xy(centroid, cfg.smooth_window, cfg.smooth_polyorder)

    # Horizontal-plane velocity (use full 3D for magnitude, xy for heading).
    vel = np.gradient(centroid_s, 1.0 / cfg.fs, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    # Body heading vector (horizontal projection).
    body_vec = kp_world[:, i_to, :] - kp_world[:, i_from, :]
    body_xy = body_vec[:, :2]
    body_norm = np.linalg.norm(body_xy, axis=1, keepdims=True) + 1e-12
    body_hat = body_xy / body_norm
    perp_hat = np.stack([-body_hat[:, 1], body_hat[:, 0]], axis=1)  # 90° CCW

    forward_speed = np.sum(vel[:, :2] * body_hat, axis=1)
    lateral_speed = np.sum(vel[:, :2] * perp_hat, axis=1)

    heading = np.arctan2(body_xy[:, 1], body_xy[:, 0])  # radians
    # Unwrap then differentiate so turn rate doesn't spike at ±π wraps.
    heading_u = np.unwrap(heading)
    turn_rate = np.degrees(np.gradient(heading_u, 1.0 / cfg.fs))

    out: Dict[str, np.ndarray] = {
        "speed": speed,
        "forward_speed": forward_speed,
        "lateral_speed": lateral_speed,
        "turn_rate": turn_rate,
        "heading": heading,
    }
    if body_length is not None and np.isfinite(body_length) and body_length > 0:
        out["speed_bl"] = speed / body_length
        out["forward_speed_bl"] = forward_speed / body_length
        out["lateral_speed_bl"] = lateral_speed / body_length
    return out


# -----------------------------------------------------------------------------
# COM height above the floor
# -----------------------------------------------------------------------------


def compute_com_height(
    kp_world: np.ndarray,
    kp_names: Sequence[str],
    cfg: Optional[LocomotionConfig] = None,
    centroid_kp: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """Per-frame COM Z above the estimated floor plane.

    Floor Z is the ``floor_percentile``-th percentile of Z over the
    matched ground keypoints for this single fly, using the same helper
    that ``pair_validity.compute_pair_validity`` uses, so the two
    modules agree on what "ground" means.

    Returns ``(com_z, floor_z)`` where ``com_z`` is (T,) and equals
    centroid Z minus ``floor_z``.
    """
    if cfg is None:
        cfg = LocomotionConfig()
    kp_world = np.asarray(kp_world)
    if kp_world.ndim == 2:
        kp_world = kp_world.reshape(kp_world.shape[0], -1, 3)

    names = list(kp_names)
    ground_idx = _match_indices(names, cfg.ground_kp_patterns)
    _, floor_z = _ground_valid(
        kp_world, ground_idx, cfg.floor_percentile, cfg.ground_epsilon
    )
    i_cent = names.index(centroid_kp or cfg.centroid_kp)
    com_z = kp_world[:, i_cent, 2].astype(float) - floor_z
    return com_z, floor_z


# -----------------------------------------------------------------------------
# Running state (stopped / running) with hysteresis
# -----------------------------------------------------------------------------


def classify_running_state(
    speed_bl: np.ndarray, cfg: Optional[LocomotionConfig] = None
) -> np.ndarray:
    """Per-frame state ∈ {'running', 'stopped'} from a body-length speed
    signal. Uses a hysteresis gate plus a minimum-run-length filter so
    the labels aren't dominated by single-frame jitter.
    """
    if cfg is None:
        cfg = LocomotionConfig()
    T = len(speed_bl)
    state = np.full(T, "stopped", dtype=object)
    if T == 0:
        return state

    running = False
    for t in range(T):
        s = speed_bl[t]
        if running:
            if s < cfg.stopped_speed_bl:
                running = False
        else:
            if s > cfg.running_speed_bl:
                running = True
        state[t] = "running" if running else "stopped"

    # Minimum-run-length filter: collapse runs shorter than min_run_ms.
    min_run = max(1, int(round(cfg.min_run_ms * 1e-3 * cfg.fs)))
    if min_run > 1:
        state = _apply_min_run(state, min_run)
    return state


def _apply_min_run(state: np.ndarray, min_run: int) -> np.ndarray:
    out = state.copy()
    T = len(out)
    i = 0
    while i < T:
        j = i
        while j < T and out[j] == out[i]:
            j += 1
        run_len = j - i
        if run_len < min_run and i > 0:
            # Extend the previous state over this short run.
            out[i:j] = out[i - 1]
        i = j
    return out


# -----------------------------------------------------------------------------
# Song-conditioned aggregation
# -----------------------------------------------------------------------------


def summarize_by_song(
    song_labels: np.ndarray,
    metrics: Dict[str, np.ndarray],
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Group per-frame metrics by song label and compute mean / std.

    Parameters
    ----------
    song_labels : (T,) object array of label strings (from the song
        classifier). Typical labels: 'pulse', 'sine', 'waggle', 'quiet'.
    metrics : dict of (T,) arrays keyed by metric name.
    valid_mask : (T,) bool or None. Only True frames are included.

    Returns
    -------
    dict ``{label: {metric: {'mean': float, 'std': float, 'n': int}}}``.
    """
    song_labels = np.asarray(song_labels)
    T = len(song_labels)
    if valid_mask is None:
        valid_mask = np.ones(T, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    unique_labels = sorted(set(song_labels.tolist()))
    for lbl in unique_labels:
        sel = valid_mask & (song_labels == lbl)
        n = int(sel.sum())
        out[lbl] = {}
        for mname, arr in metrics.items():
            a = np.asarray(arr)
            if n == 0:
                out[lbl][mname] = {"mean": float("nan"), "std": float("nan"), "n": 0}
                continue
            vals = a[sel]
            finite = np.isfinite(vals)
            if finite.sum() == 0:
                out[lbl][mname] = {"mean": float("nan"), "std": float("nan"), "n": n}
                continue
            out[lbl][mname] = {
                "mean": float(np.mean(vals[finite])),
                "std": float(np.std(vals[finite])),
                "n": n,
            }
    return out


def running_fraction(state: np.ndarray) -> float:
    """Fraction of frames labelled 'running' (NaN-safe)."""
    T = len(state)
    if T == 0:
        return 0.0
    return float(np.sum(np.asarray(state) == "running") / T)
