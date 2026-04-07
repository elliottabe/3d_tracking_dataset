"""
Identity relink for two-animal multi-fly tracking.

Detects and corrects per-frame identity swaps between two animals tracked
independently across frames. The typical failure pattern in JARVIS multi-animal
output is that fly1's keypoints momentarily snap onto fly0's body (or vice
versa) for one or more frames, producing a sustained "teleport" in the per-fly
3D trajectory.

This module operates on raw 3D keypoints (T, N, 3) for both flies in the same
global frame coordinates (i.e. before any per-fly Procrustes alignment) and
returns swap-corrected per-fly arrays plus a swap log.

Algorithm
---------
For each frame t starting at t=1:
  1. Predict each fly's expected centroid as ``prev_centroid + ema_velocity``.
  2. Build a 2x2 cost matrix between the two predicted positions and the two
     observed centroids using ``||predicted - observed||`` plus a body-length
     consistency term.
  3. Compare cost(identity) vs cost(swapped). If swapped is significantly
     better (ratio < ``swap_ratio``), swap the two flies' entire skeletons at
     this frame and propagate the swap forward.
  4. Update EMA velocity and EMA body length on the post-swap arrays.

The swap state is *cumulative*: once frame t is swapped, every subsequent
frame is also swapped until the next swap-detection flips it back.

The function never extrapolates across NaNs — if either fly's centroid is
NaN at frame t, the previous swap state is held without updating velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class RelinkConfig:
    trunk_keypoints: Tuple[str, ...] = ("Scutellum", "Postnotum", "Scutum")
    body_length_pair: Tuple[str, str] = ("Antenna_Base", "Abd_tip")
    velocity_alpha: float = 0.5      # EMA weight for velocity update (1 = no smoothing)
    body_length_alpha: float = 0.05  # EMA weight for body length tracking
    body_length_weight: float = 0.5  # cost weight relative to centroid distance
    swap_ratio: float = 0.7          # swap if cost_swapped < cost_current * swap_ratio
    min_swap_advantage: float = 0.0  # additional absolute margin (in same units as kp)
    require_min_displacement: float = 0.0  # only run swap test if max obs displacement >= this


def _resolve_indices(kp_names: List[str], names: Tuple[str, ...]) -> List[int]:
    out = []
    for i, n in enumerate(kp_names):
        if any(t in n for t in names):
            out.append(i)
    return out


def _centroid(kp: np.ndarray, idx: List[int]) -> np.ndarray:
    """(T, N, 3) -> (T, 3) using nanmean over the chosen indices."""
    if len(idx) == 0:
        return np.nanmean(kp, axis=1)
    return np.nanmean(kp[:, idx, :], axis=1)


def _body_length(kp: np.ndarray, i_a: int, i_b: int) -> np.ndarray:
    """(T, N, 3) -> (T,) Euclidean distance between two keypoints per frame."""
    return np.linalg.norm(kp[:, i_a, :] - kp[:, i_b, :], axis=-1)


def relink_pair(
    fly0: np.ndarray,
    fly1: np.ndarray,
    kp_names: List[str],
    cfg: Optional[RelinkConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Relink two flies' keypoint arrays by detecting per-frame identity swaps.

    Args:
        fly0, fly1: (T, N, 3) per-fly 3D keypoints in the same world frame.
        kp_names:   list of N keypoint names (used to find centroid + body length).
        cfg:        RelinkConfig (defaults are conservative for fly courtship).

    Returns:
        relinked_fly0: (T, N, 3) corrected fly0 keypoints
        relinked_fly1: (T, N, 3) corrected fly1 keypoints
        log: dict with keys
            'swap_state'   : (T,) bool — True where the assignment is flipped
                             relative to the input
            'swap_frames'  : list[int] — frame indices where the state toggled
            'n_swap_segments': int — number of toggle events
            'fraction_swapped': float — fraction of frames in swapped state
            'cost_current' : (T,) float — frame-by-frame current-assignment cost
            'cost_swapped' : (T,) float — frame-by-frame swapped-assignment cost
    """
    if cfg is None:
        cfg = RelinkConfig()
    assert fly0.shape == fly1.shape, f"shape mismatch {fly0.shape} vs {fly1.shape}"
    T, N, _ = fly0.shape

    trunk_idx = _resolve_indices(kp_names, cfg.trunk_keypoints)
    if len(trunk_idx) == 0:
        # Fall back to mean over all keypoints — better than failing
        trunk_idx = list(range(N))

    try:
        i_a = kp_names.index(cfg.body_length_pair[0])
        i_b = kp_names.index(cfg.body_length_pair[1])
        have_body_len = True
    except ValueError:
        have_body_len = False
        i_a = i_b = 0

    cent0 = _centroid(fly0, trunk_idx)  # (T, 3)
    cent1 = _centroid(fly1, trunk_idx)
    if have_body_len:
        bl0 = _body_length(fly0, i_a, i_b)
        bl1 = _body_length(fly1, i_a, i_b)
    else:
        bl0 = bl1 = np.zeros(T)

    swap_state = np.zeros(T, dtype=bool)  # cumulative state
    cost_current_log = np.zeros(T)
    cost_swapped_log = np.zeros(T)

    # Running EMA state on the *current-identity* trajectories.
    pred0 = cent0[0].copy()
    pred1 = cent1[0].copy()
    vel0 = np.zeros(3)
    vel1 = np.zeros(3)
    ema_bl0 = bl0[0] if have_body_len and np.isfinite(bl0[0]) else 0.0
    ema_bl1 = bl1[0] if have_body_len and np.isfinite(bl1[0]) else 0.0

    cur_state = False  # whether current frame's assignment is swapped vs input

    for t in range(1, T):
        # Observed centroids under the *input* assignment, but if we're already
        # in swapped state we should reason about the corrected centroids.
        if cur_state:
            obs0 = cent1[t]; obs1 = cent0[t]
            obs_bl0 = bl1[t]; obs_bl1 = bl0[t]
        else:
            obs0 = cent0[t]; obs1 = cent1[t]
            obs_bl0 = bl0[t]; obs_bl1 = bl1[t]

        valid = np.all(np.isfinite(obs0)) and np.all(np.isfinite(obs1))
        if not valid:
            swap_state[t] = cur_state
            continue

        # Predicted positions from EMA velocity
        p0 = pred0 + vel0
        p1 = pred1 + vel1

        d_current = np.linalg.norm(p0 - obs0) + np.linalg.norm(p1 - obs1)
        d_swapped = np.linalg.norm(p0 - obs1) + np.linalg.norm(p1 - obs0)

        if cfg.body_length_weight > 0 and have_body_len and ema_bl0 > 0 and ema_bl1 > 0:
            d_current += cfg.body_length_weight * (
                abs(obs_bl0 - ema_bl0) + abs(obs_bl1 - ema_bl1)
            )
            d_swapped += cfg.body_length_weight * (
                abs(obs_bl1 - ema_bl0) + abs(obs_bl0 - ema_bl1)
            )

        cost_current_log[t] = d_current
        cost_swapped_log[t] = d_swapped

        # Decide whether to flip
        max_obs_step = max(np.linalg.norm(obs0 - pred0), np.linalg.norm(obs1 - pred1))
        eligible = max_obs_step >= cfg.require_min_displacement
        do_flip = (
            eligible
            and d_swapped < d_current * cfg.swap_ratio - cfg.min_swap_advantage
        )

        if do_flip:
            cur_state = not cur_state
            # Swap the obs we just used so EMA tracks the corrected fly
            obs0, obs1 = obs1, obs0
            obs_bl0, obs_bl1 = obs_bl1, obs_bl0

        swap_state[t] = cur_state

        # Update EMA on the (post-flip) observations
        new_v0 = obs0 - pred0
        new_v1 = obs1 - pred1
        a = cfg.velocity_alpha
        vel0 = (1 - a) * vel0 + a * new_v0
        vel1 = (1 - a) * vel1 + a * new_v1
        pred0 = obs0
        pred1 = obs1

        if have_body_len:
            b = cfg.body_length_alpha
            if np.isfinite(obs_bl0) and obs_bl0 > 0:
                ema_bl0 = (1 - b) * ema_bl0 + b * obs_bl0 if ema_bl0 > 0 else obs_bl0
            if np.isfinite(obs_bl1) and obs_bl1 > 0:
                ema_bl1 = (1 - b) * ema_bl1 + b * obs_bl1 if ema_bl1 > 0 else obs_bl1

    relinked_fly0 = np.where(swap_state[:, None, None], fly1, fly0)
    relinked_fly1 = np.where(swap_state[:, None, None], fly0, fly1)

    toggles = np.flatnonzero(np.diff(swap_state.astype(np.int8)) != 0) + 1
    log = dict(
        swap_state=swap_state,
        swap_frames=toggles.tolist(),
        n_swap_segments=int(len(toggles)),
        fraction_swapped=float(swap_state.mean()),
        cost_current=cost_current_log,
        cost_swapped=cost_swapped_log,
    )
    return relinked_fly0, relinked_fly1, log
