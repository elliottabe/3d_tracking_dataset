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
from typing import Dict, List, Optional, Tuple

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
    # Contact-aware criteria (used by _should_flip in addition to swap_ratio).
    bl_tube_factor: float = 0.25     # body-length-tube relative half-width
    # Predicted-step ceiling. The effective threshold per frame is
    #     max(max_step_abs, max_step_bl * ema_body_length)
    # so users can express it either as an absolute distance (in the same
    # units as the input keypoints) or as a fraction of body length per
    # frame. The body-length-relative form is unit-agnostic and the
    # recommended default — fly courtship rarely exceeds ~half a body
    # length per frame at 60 fps.
    max_step_bl: float = 0.5         # body-length multiples per frame
    max_step_abs: float = 0.0        # absolute floor in input kp units
    nan_resume_frames: int = 3       # NaN-gap length that triggers a forced re-test
    bl_seed_window: int = 20         # window size for median body-length seeding
    # Body-length Viterbi (preferred when the two flies have distinguishable
    # body lengths). Replaces the greedy EMA loop with a global 2-state
    # Viterbi over body length, which is robust to close contact because the
    # per-fly body length is a stable identity signal that doesn't drift.
    # Falls back to the greedy loop automatically when body length is not
    # available or the two flies look identical.
    use_bl_viterbi: bool = True
    bl_viterbi_transition_weight: float = 5.0   # multiple of (base1-base0)^2
    bl_viterbi_min_separation: float = 0.02     # min |base1 - base0| / mean to use viterbi
    # Position-continuity weight in the Viterbi transitions. The bl-only
    # Viterbi gets confused when the upstream tracker briefly mislabels
    # internal keypoints (causing a body length flip without a real position
    # swap). With ``bl_viterbi_position_weight > 0`` the transition cost
    # includes ``||c0[t] - c1[t-1]||² + ||c1[t] - c0[t-1]||²`` (the cost of
    # the labels actually swapping) — small when the flies are close, large
    # when they are far apart. This makes wrong swaps far more expensive
    # than the bl emission gain, while leaving real contact-time swaps cheap.
    # Set to 0 to disable and fall back to a pure bl-only Viterbi.
    bl_viterbi_position_weight: float = 1.0
    # Body-length outlier rejection. Frames where either fly's body length
    # falls outside [bl_outlier_low * base_s, bl_outlier_high * base_l] are
    # treated as tracking errors: emission cost and position-aware transition
    # cost are zeroed at those frames, so a collapsed-body glitch cannot seed
    # a fake swap segment. The baselines are the robust per-frame
    # ``median(min(bl0, bl1))`` and ``median(max(bl0, bl1))`` used inside
    # the Viterbi. 0.7 handles "soft" collapses — tracker mislabelings that
    # drop bl to 70-95% of baseline, still below the Antenna_Base→Abd_tip
    # range fruit flies can reach by natural bending. 0.5 was too permissive
    # and let soft collapses seed false swap segments; see
    # ``docs/superpowers/specs/2026-04-11-identity-relink-bl-outlier-fix-design.md``.
    bl_viterbi_outlier_low: float = 0.7   # reject bl < low * base_smaller
    bl_viterbi_outlier_high: float = 2.0  # reject bl > high * base_larger
    # Minimum length (in frames) for a swap segment to survive the Viterbi
    # backtrace. Any True-run shorter than this is flipped back to False.
    # One-sided: we suppress spurious short *swaps*, never spurious short
    # un-swaps. Real courtship identity swaps require sustained contact
    # (typically many frames), while single-frame tracking glitches that
    # sneak through outlier rejection often manifest as 1-3 frame blips.
    bl_viterbi_min_swap_frames: int = 5


def _resolve_indices(kp_names: List[str], names: Tuple[str, ...]) -> List[int]:
    out = []
    for i, n in enumerate(kp_names):
        if any(t in n for t in names):
            out.append(i)
    return out


def _resolve_one(kp_names: List[str], target: str) -> Optional[int]:
    """Find a single keypoint index by exact match first, then substring.

    Handles both bare names ('Antenna_Base') and wrapped names
    ('tracking[Antenna_Base]_fly') used by some upstream pipelines.
    """
    try:
        return kp_names.index(target)
    except ValueError:
        pass
    for i, n in enumerate(kp_names):
        if target in n:
            return i
    return None


def _centroid(kp: np.ndarray, idx: List[int]) -> np.ndarray:
    """(T, N, 3) -> (T, 3) using nanmean over the chosen indices."""
    if len(idx) == 0:
        return np.nanmean(kp, axis=1)
    return np.nanmean(kp[:, idx, :], axis=1)


def _body_length(kp: np.ndarray, i_a: int, i_b: int) -> np.ndarray:
    """(T, N, 3) -> (T,) Euclidean distance between two keypoints per frame."""
    return np.linalg.norm(kp[:, i_a, :] - kp[:, i_b, :], axis=-1)


def _bl_viterbi_swap_state(
    bl0: np.ndarray,
    bl1: np.ndarray,
    transition_weight: float,
    min_separation: float,
    cent0: Optional[np.ndarray] = None,
    cent1: Optional[np.ndarray] = None,
    position_weight: float = 1.0,
    outlier_low: float = 0.5,
    outlier_high: float = 2.0,
    min_swap_frames: int = 5,
) -> Tuple[Optional[np.ndarray], dict]:
    """Globally optimal binary swap-state from per-frame body lengths.

    Models a 2-state HMM:
      state 0 = input labels are correct (fly0 is the smaller fly iff base0 < base1)
      state 1 = labels are swapped

    Emission cost at frame t under state s is the squared deviation of each
    label's body length from the corresponding *physical* fly's baseline.
    Baselines are estimated robustly as the per-frame
    ``median(min(bl0, bl1))`` and ``median(max(bl0, bl1))`` over valid frames,
    so they're independent of the input labeling. Whichever role the input
    fly0 plays in the *majority* of frames (smaller or larger) defines
    state 0.

    Returns ``(swap_state, info)`` or ``(None, info)`` when the two flies
    cannot be distinguished by body length (separation < min_separation
    relative to the mean baseline).
    """
    info = dict(used=False, base_smaller=None, base_larger=None,
                relative_separation=None, transition_cost=None,
                ref_fly0_smaller=None, position_aware=False,
                d_switch_median=None, d_stay_median=None,
                n_outlier_frames=0, n_short_runs_suppressed=0)
    bl0 = np.asarray(bl0, dtype=float)
    bl1 = np.asarray(bl1, dtype=float)
    T = bl0.shape[0]
    if T == 0:
        return None, info

    finite = np.isfinite(bl0) & np.isfinite(bl1) & (bl0 > 0) & (bl1 > 0)
    if finite.sum() < max(20, T // 20):
        return None, info

    bv0 = bl0[finite]
    bv1 = bl1[finite]
    smaller = np.minimum(bv0, bv1)
    larger = np.maximum(bv0, bv1)
    base_s = float(np.median(smaller))
    base_l = float(np.median(larger))
    base_mean = 0.5 * (base_s + base_l)
    rel_sep = (base_l - base_s) / base_mean if base_mean > 0 else 0.0
    info.update(base_smaller=base_s, base_larger=base_l,
                relative_separation=rel_sep)
    if rel_sep < min_separation:
        return None, info

    # Reject frames where either fly's body length is physically implausible
    # relative to the robust baselines. The typical failure mode is a tracker
    # briefly mis-attributing an internal keypoint, collapsing bl to well
    # below base_smaller (e.g. 9-15 when baselines are 24/27). Left in place,
    # those frames would flip the emission cost to prefer the swapped state
    # and, when combined with a phantom-collision d_switch, seed a spurious
    # multi-frame swap segment.
    bl_low = outlier_low * base_s
    bl_high = outlier_high * base_l
    plausible = (
        (bl0 >= bl_low) & (bl0 <= bl_high)
        & (bl1 >= bl_low) & (bl1 <= bl_high)
    )
    valid = finite & plausible
    info["n_outlier_frames"] = int(finite.sum() - valid.sum())
    if valid.sum() < max(20, T // 20):
        return None, info

    # Reference: in the input, is fly0 *usually* the smaller fly?
    n_fly0_smaller = int(np.sum(bv0 < bv1))
    n_fly1_smaller = int(np.sum(bv1 < bv0))
    ref_fly0_smaller = n_fly0_smaller >= n_fly1_smaller
    info["ref_fly0_smaller"] = bool(ref_fly0_smaller)

    # Per-frame emission cost under each state. Invalid frames pay zero
    # emission (transitions only), so the optimizer naturally interpolates.
    if ref_fly0_smaller:
        # state 0: fly0 ↔ smaller, fly1 ↔ larger
        c0 = (bl0 - base_s) ** 2 + (bl1 - base_l) ** 2
        c1 = (bl0 - base_l) ** 2 + (bl1 - base_s) ** 2
    else:
        c0 = (bl0 - base_l) ** 2 + (bl1 - base_s) ** 2
        c1 = (bl0 - base_s) ** 2 + (bl1 - base_l) ** 2
    # Mask invalid frames to zero emission so they don't bias toward either state.
    c0 = np.where(valid, c0, 0.0)
    c1 = np.where(valid, c1, 0.0)

    trans_base = transition_weight * (base_l - base_s) ** 2
    info["transition_cost"] = float(trans_base)

    # Position-aware transition costs. The latent state s_t encodes whether
    # the input labels at frame t are flipped vs the physical flies. The
    # *physical* trajectories must be continuous, so:
    #
    #   stay  (s_{t-1}==s_t):  ||c0[t]-c0[t-1]||² + ||c1[t]-c1[t-1]||²
    #   switch (s_{t-1}!=s_t): ||c0[t]-c1[t-1]||² + ||c1[t]-c0[t-1]||²
    #
    # ``d_switch[t]`` is small only when the flies are physically close at
    # the transition frame — this is the right structural prior for identity
    # swaps (they require contact). Body length plays the role of the
    # *emission* and disambiguates the absolute labeling, while position
    # continuity governs *when* the labels are allowed to flip.
    d_stay = np.zeros(T, dtype=np.float64)
    d_switch = np.zeros(T, dtype=np.float64)
    have_pos = False
    if cent0 is not None and cent1 is not None and position_weight > 0:
        c0_arr = np.asarray(cent0, dtype=float)
        c1_arr = np.asarray(cent1, dtype=float)
        if c0_arr.shape == c1_arr.shape and c0_arr.shape[0] == T and c0_arr.ndim >= 2:
            d_stay[1:] = (
                np.sum((c0_arr[1:] - c0_arr[:-1]) ** 2, axis=-1)
                + np.sum((c1_arr[1:] - c1_arr[:-1]) ** 2, axis=-1)
            )
            d_switch[1:] = (
                np.sum((c0_arr[1:] - c1_arr[:-1]) ** 2, axis=-1)
                + np.sum((c1_arr[1:] - c0_arr[:-1]) ** 2, axis=-1)
            )
            # NaN-safe: if either centroid is missing, treat both costs as 0
            # so the optimizer falls back to the bl emission alone for that
            # transition. Same treatment for transitions that touch a
            # body-length-outlier frame — a phantom collision at an outlier
            # frame would otherwise produce an artificially-small d_switch
            # and seed a spurious swap.
            bad = ~(np.isfinite(d_stay) & np.isfinite(d_switch))
            trans_valid = np.zeros(T, dtype=bool)
            trans_valid[1:] = valid[1:] & valid[:-1]
            bad |= ~trans_valid
            d_stay = np.where(bad, 0.0, d_stay)
            d_switch = np.where(bad, 0.0, d_switch)
            d_stay *= position_weight
            d_switch *= position_weight
            have_pos = True
            info["position_aware"] = True
            valid_idx = np.flatnonzero(~bad)
            if valid_idx.size > 1:
                info["d_switch_median"] = float(np.median(d_switch[valid_idx[1:]]))
                info["d_stay_median"] = float(np.median(d_stay[valid_idx[1:]]))

    # Forward Viterbi
    dp = np.empty((T, 2), dtype=np.float64)
    bt = np.empty((T, 2), dtype=np.int8)
    dp[0, 0] = c0[0]
    dp[0, 1] = c1[0]
    bt[0, 0] = 0
    bt[0, 1] = 1
    for t in range(1, T):
        emit0 = c0[t]
        emit1 = c1[t]
        ds = d_stay[t]
        dw = d_switch[t]
        # State 0 at t (input labels are correct here)
        cost00 = dp[t - 1, 0] + ds + emit0                 # stay in state 0
        cost10 = dp[t - 1, 1] + dw + trans_base + emit0    # switch from state 1
        if cost00 <= cost10:
            dp[t, 0] = cost00
            bt[t, 0] = 0
        else:
            dp[t, 0] = cost10
            bt[t, 0] = 1
        # State 1 at t (input labels are flipped here)
        cost11 = dp[t - 1, 1] + ds + emit1                 # stay in state 1
        cost01 = dp[t - 1, 0] + dw + trans_base + emit1    # switch from state 0
        if cost11 <= cost01:
            dp[t, 1] = cost11
            bt[t, 1] = 1
        else:
            dp[t, 1] = cost01
            bt[t, 1] = 0

    # Backtrace
    state = np.zeros(T, dtype=np.int8)
    state[T - 1] = 0 if dp[T - 1, 0] <= dp[T - 1, 1] else 1
    for t in range(T - 1, 0, -1):
        state[t - 1] = bt[t, state[t]]

    state_bool = state.astype(bool)

    # Minimum-run suppression (one-sided): flip any True-run shorter than
    # ``min_swap_frames`` back to False. Real courtship identity swaps
    # persist across many frames of sustained contact; short blips that
    # survive the outlier mask are almost always detector noise or
    # boundary effects around a glitch we already zero-cost'd.
    if min_swap_frames > 1 and state_bool.any():
        i = 0
        n_suppressed = 0
        while i < T:
            if state_bool[i]:
                j = i
                while j < T and state_bool[j]:
                    j += 1
                if (j - i) < min_swap_frames:
                    state_bool[i:j] = False
                    n_suppressed += 1
                i = j
            else:
                i += 1
        info["n_short_runs_suppressed"] = n_suppressed

    info["used"] = True
    return state_bool, info


def _seed_body_length(bl: np.ndarray, window: int) -> float:
    """Median body length over the first ``window`` valid frames, or 0.0."""
    if bl.size == 0:
        return 0.0
    take = min(window, max(1, bl.shape[0] // 2)) if bl.shape[0] >= 2 else bl.shape[0]
    head = bl[:take]
    finite = head[np.isfinite(head) & (head > 0)]
    if finite.size == 0:
        return 0.0
    return float(np.median(finite))


def _should_flip(
    d_current: float,
    d_swapped: float,
    obs0: np.ndarray,
    obs1: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    obs_bl0: float,
    obs_bl1: float,
    ema_bl0: float,
    ema_bl1: float,
    have_body_len: bool,
    cfg: RelinkConfig,
) -> bool:
    """Contact-aware swap decision used inside the relink loop.

    Returns True if any of three independent criteria fire:
      1. existing multiplicative ratio (``swap_ratio``);
      2. body-length-tube violation that the swap fixes;
      3. predicted-step violation that the swap fixes (threshold is the
         max of ``max_step_abs`` and ``max_step_bl * mean(ema_bl)``).

    Designed to recover swaps in close courtship contact where (1) alone
    almost never trips because both costs are tiny and similar.
    """
    # 1. Original ratio rule
    ratio_flip = d_swapped < d_current * cfg.swap_ratio - cfg.min_swap_advantage

    # 2. Body-length-tube rule. The "tube" is ema_bl_i ± bl_tube_factor*ema_bl_i.
    bl_flip = False
    if have_body_len and ema_bl0 > 0 and ema_bl1 > 0:
        tube0 = cfg.bl_tube_factor * ema_bl0
        tube1 = cfg.bl_tube_factor * ema_bl1
        cur_bad = (
            abs(obs_bl0 - ema_bl0) > tube0 or abs(obs_bl1 - ema_bl1) > tube1
        )
        swap_ok = (
            abs(obs_bl1 - ema_bl0) <= tube0 and abs(obs_bl0 - ema_bl1) <= tube1
        )
        if cur_bad and swap_ok:
            bl_flip = True

    # 3. Predicted-step ceiling. Threshold is unit-agnostic via body length:
    #    `max_step_bl * mean_body_length`, with an optional absolute floor.
    step_flip = False
    bl_mean = 0.5 * (ema_bl0 + ema_bl1) if (have_body_len and ema_bl0 > 0 and ema_bl1 > 0) else 0.0
    step_thresh = max(cfg.max_step_abs, cfg.max_step_bl * bl_mean)
    if step_thresh > 0:
        cur_step = max(
            float(np.linalg.norm(obs0 - p0)),
            float(np.linalg.norm(obs1 - p1)),
        )
        if cur_step > step_thresh:
            swap_step = max(
                float(np.linalg.norm(obs1 - p0)),
                float(np.linalg.norm(obs0 - p1)),
            )
            if swap_step <= step_thresh:
                step_flip = True

    return ratio_flip or bl_flip or step_flip


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

    i_a = _resolve_one(kp_names, cfg.body_length_pair[0])
    i_b = _resolve_one(kp_names, cfg.body_length_pair[1])
    if i_a is not None and i_b is not None:
        have_body_len = True
    else:
        have_body_len = False
        i_a = i_b = 0

    cent0 = _centroid(fly0, trunk_idx)  # (T, 3)
    cent1 = _centroid(fly1, trunk_idx)
    if have_body_len:
        bl0 = _body_length(fly0, i_a, i_b)
        bl1 = _body_length(fly1, i_a, i_b)
    else:
        bl0 = bl1 = np.zeros(T)

    # Try the body-length Viterbi first. When the two flies have a clear
    # body-length separation (males ~smaller than females in fly courtship)
    # this gives a globally optimal swap-state without the EMA-drift
    # failure mode of the greedy loop.
    bl_vit_state = None
    bl_vit_info: dict = {"used": False}
    if cfg.use_bl_viterbi and have_body_len:
        bl_vit_state, bl_vit_info = _bl_viterbi_swap_state(
            bl0, bl1,
            transition_weight=cfg.bl_viterbi_transition_weight,
            min_separation=cfg.bl_viterbi_min_separation,
            cent0=cent0,
            cent1=cent1,
            position_weight=cfg.bl_viterbi_position_weight,
            outlier_low=cfg.bl_viterbi_outlier_low,
            outlier_high=cfg.bl_viterbi_outlier_high,
            min_swap_frames=cfg.bl_viterbi_min_swap_frames,
        )

    if bl_vit_state is not None:
        # Body-length Viterbi succeeded — use it as the authoritative swap state
        # and short-circuit the greedy loop. We still build the cost logs from
        # body length so callers can introspect.
        swap_state = bl_vit_state
        # Cost diagnostics: emission cost under the chosen vs the alternative state.
        cost_current_log = np.zeros(T)
        cost_swapped_log = np.zeros(T)

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
            method="bl_viterbi",
            bl_viterbi=bl_vit_info,
        )
        return relinked_fly0, relinked_fly1, log

    swap_state = np.zeros(T, dtype=bool)  # cumulative state
    cost_current_log = np.zeros(T)
    cost_swapped_log = np.zeros(T)

    # Running EMA state on the *current-identity* trajectories.
    pred0 = cent0[0].copy()
    pred1 = cent1[0].copy()
    vel0 = np.zeros(3)
    vel1 = np.zeros(3)
    # Seed body-length EMAs from the median of the first window so short
    # bouts don't sit at zero for the whole loop (was: bl[0] only).
    if have_body_len:
        ema_bl0 = _seed_body_length(bl0, cfg.bl_seed_window)
        ema_bl1 = _seed_body_length(bl1, cfg.bl_seed_window)
    else:
        ema_bl0 = 0.0
        ema_bl1 = 0.0

    cur_state = False  # whether current frame's assignment is swapped vs input
    nan_gap = 0  # consecutive invalid-frame counter for NaN-resume guard

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
            nan_gap += 1
            continue

        # Predicted positions from EMA velocity
        p0 = pred0 + vel0
        p1 = pred1 + vel1

        # NaN-resume guard: if we just emerged from a long invalid span the EMA
        # velocity is stale, so the standard predicted-distance test is
        # unreliable. Fall back to comparing observations against the
        # *last-seen* positions (pred0/pred1 — i.e. velocity ignored), which
        # is much more robust to swaps that happened during the gap.
        long_gap = nan_gap >= cfg.nan_resume_frames
        if long_gap:
            p0 = pred0
            p1 = pred1
        nan_gap = 0

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

        # Decide whether to flip. Long-gap resumes always run the test (the
        # require_min_displacement gate would otherwise suppress it).
        max_obs_step = max(np.linalg.norm(obs0 - pred0), np.linalg.norm(obs1 - pred1))
        eligible = long_gap or (max_obs_step >= cfg.require_min_displacement)
        do_flip = eligible and _should_flip(
            d_current=d_current,
            d_swapped=d_swapped,
            obs0=obs0,
            obs1=obs1,
            p0=p0,
            p1=p1,
            obs_bl0=obs_bl0,
            obs_bl1=obs_bl1,
            ema_bl0=ema_bl0,
            ema_bl1=ema_bl1,
            have_body_len=have_body_len,
            cfg=cfg,
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
        method="greedy",
        bl_viterbi=bl_vit_info,
    )
    return relinked_fly0, relinked_fly1, log


def relink_pair_bouted(
    fly0_bouts: Dict[str, np.ndarray],
    fly1_bouts: Dict[str, np.ndarray],
    kp_names: List[str],
    cfg: Optional[RelinkConfig] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, dict]]:
    """Bout-aware joint relink for two flies tracked over the same windows.

    Calls ``relink_pair`` once per shared bout key. Each call gets a fresh
    EMA / velocity / body-length state — fixing the cross-bout EMA bleed
    bug in the previous concatenated-relink usage. Returns *both* flies'
    corrected per-bout arrays so callers (e.g. ``batch_split_valid_bouts``)
    can apply a single consistent correction to fly0 and fly1 simultaneously
    instead of running two independent per-fly relinks that can disagree.

    Args:
        fly0_bouts: dict mapping bout key -> (T_b, N, 3) keypoint array.
        fly1_bouts: dict mapping the same keys -> (T_b, N, 3) keypoint array.
        kp_names:   list of N keypoint names shared by both flies.
        cfg:        RelinkConfig (defaults are conservative for fly courtship).

    Returns:
        relinked_fly0: dict {bout_key: (T_b, N, 3)} corrected fly0 arrays.
        relinked_fly1: dict {bout_key: (T_b, N, 3)} corrected fly1 arrays.
        logs:          dict {bout_key: log_dict} from each per-bout call,
                       plus a "_summary" entry with aggregate counts.
    """
    if cfg is None:
        cfg = RelinkConfig()

    shared_keys = sorted(set(fly0_bouts.keys()) & set(fly1_bouts.keys()))
    relinked_fly0: Dict[str, np.ndarray] = {}
    relinked_fly1: Dict[str, np.ndarray] = {}
    logs: Dict[str, dict] = {}

    total_frames = 0
    swapped_frames = 0
    swap_segments = 0
    bouts_with_swap = 0
    skipped: List[str] = []

    for bk in shared_keys:
        a = np.asarray(fly0_bouts[bk])
        b = np.asarray(fly1_bouts[bk])
        # Trim to the shorter bout if the two per-fly h5s disagreed slightly.
        if a.shape != b.shape:
            T = min(a.shape[0], b.shape[0])
            if T == 0 or a.ndim != 3 or b.ndim != 3 or a.shape[1:] != b.shape[1:]:
                skipped.append(bk)
                relinked_fly0[bk] = a
                relinked_fly1[bk] = b
                continue
            a = a[:T]
            b = b[:T]

        rl0, rl1, log = relink_pair(a, b, kp_names, cfg)
        relinked_fly0[bk] = rl0
        relinked_fly1[bk] = rl1
        logs[bk] = log

        total_frames += int(rl0.shape[0])
        swapped_frames += int(np.asarray(log["swap_state"], dtype=bool).sum())
        swap_segments += int(log["n_swap_segments"])
        if log["n_swap_segments"] > 0:
            bouts_with_swap += 1

    logs["_summary"] = dict(
        n_bouts=len(shared_keys),
        n_bouts_with_swap=bouts_with_swap,
        n_swap_segments=swap_segments,
        total_frames=total_frames,
        swapped_frames=swapped_frames,
        fraction_swapped=(swapped_frames / total_frames) if total_frames else 0.0,
        skipped_bouts=skipped,
    )

    return relinked_fly0, relinked_fly1, logs
