"""
Keypoint filtering and smoothing utilities for 3D tracking data.

Stacked cleaning pipeline applied before Procrustes alignment:
  1. Confidence masking  — sets low-confidence keypoints to NaN
  2. Bone-length outlier detection — flags frames where a segment length
     deviates more than `threshold_std` sigma (MAD-based) from the temporal median
  2b. Centroid-jump detection — flags frames where the body centroid teleports
      (identity switch in multi-fly tracking)
  3. Median-filter spike detection + spline interpolation — identifies remaining
     trajectory spikes and fills all NaN gaps via cubic spline
  4. Savitzky-Golay smoothing — optional final trajectory smoothing

All operations work on numpy arrays of shape (T, N, 3).

Usage::

    from utils.keypoint_filter import filter_keypoints, load_confidence_from_csv
    filtered, report, edge_nan_mask = filter_keypoints(kp_array, confidence, skeleton_edges, cfg.preprocessing.filtering)
"""

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator
from typing import Optional, Sequence, Tuple, Dict, List
from omegaconf import DictConfig
import traceback as _traceback
from utils.centroid_jump_check import mask_centroid_jumps

# Set non-interactive backend before any pyplot import so figure saving works
# headlessly (on servers without a display).  This must happen at import time
# of this module, before matplotlib.pyplot is ever loaded anywhere.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _nan_interp_1d(
    vals: np.ndarray,
    use_spline: bool = True,
    max_edge_extrap: int = 0,
    edge_fit_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate NaN gaps in a 1-D signal. Skips if >50% of values are NaN.

    Interior gaps are filled with a monotone PCHIP cubic interpolant (C¹-smooth,
    no overshoot between anchors). Leading/trailing NaN runs are handled with
    bounded *linear extrapolation*: a degree-1 fit on the `edge_fit_window`
    nearest valid points is evaluated at up to `max_edge_extrap` frames into the
    edge run. Anything beyond that cap stays NaN and is dropped downstream by
    `pair_validity`. This replaces the old constant-hold edge fill, which
    produced flat tails whenever a bout had poorly-tracked trailing frames.

    Args:
        vals: (T,) 1-D signal with possible NaNs.
        use_spline: use PCHIP cubic interpolation for interior gaps (True) or
            plain linear interpolation (False).
        max_edge_extrap: maximum number of leading/trailing frames to fill by
            linear extrapolation. 0 = never extrapolate edges (leave NaN).
        edge_fit_window: number of valid frames (nearest the edge) used to fit
            the linear extrapolation model.

    Returns:
        out: (T,) filled signal. May still contain NaN where edge runs exceeded
            the cap, or where the input was too sparse to interpolate safely.
        phantom_mask: (T,) bool — True for any frame that was originally in a
            leading/trailing NaN run, regardless of whether it was filled by
            extrapolation. Interior NaNs that were filled by PCHIP are NOT
            marked phantom. Used by `pair_validity.compute_single_fly_validity`
            to mark these frames as not-trusted even when they were filled.
    """
    T = len(vals)
    phantom_mask = np.zeros(T, dtype=bool)
    nans = np.isnan(vals)
    if not np.any(nans):
        return vals, phantom_mask

    finite_idx = np.where(~nans)[0]
    if finite_idx.size == 0:
        # Fully NaN column: every frame is "phantom" by construction.
        phantom_mask[:] = True
        return vals, phantom_mask

    # Always record original edge-NaN positions as phantom, even if we later
    # fill them by extrapolation.
    x_min, x_max = finite_idx[0], finite_idx[-1]
    if x_min > 0:
        phantom_mask[:x_min] = True
    if x_max < T - 1:
        phantom_mask[x_max + 1:] = True

    if np.mean(~nans) < 0.5 or np.sum(~nans) < 4:
        # Too sparse — leave as NaN rather than interpolate wildly.
        return vals, phantom_mask

    x_good = finite_idx
    v_good = vals[~nans]
    x_bad = np.where(nans)[0]
    out = vals.copy()

    # Split NaN positions into interior (between first/last valid) and edges
    interior_mask = (x_bad >= x_min) & (x_bad <= x_max)
    x_interior = x_bad[interior_mask]

    # Interior gaps: PCHIP (or linear fallback). Always filled — these are
    # interpolation, not extrapolation.
    if len(x_interior) > 0:
        if use_spline:
            try:
                pchip = PchipInterpolator(x_good, v_good, extrapolate=False)
                out[x_interior] = pchip(x_interior)
            except Exception:
                out[x_interior] = np.interp(x_interior, x_good, v_good)
        else:
            out[x_interior] = np.interp(x_interior, x_good, v_good)

    # Edge gaps: bounded linear extrapolation. No more constant-hold.
    if max_edge_extrap > 0:
        n_good = len(x_good)
        fit_n = min(edge_fit_window, n_good)

        # Leading edge: indices [0, x_min) — fill the `max_edge_extrap` frames
        # closest to x_min (i.e. [x_min - n_fill, x_min)).
        if x_min > 0:
            n_fill = int(min(x_min, max_edge_extrap))
            fill_idx = np.arange(x_min - n_fill, x_min)
            if fit_n >= 2:
                slope, intercept = np.polyfit(
                    x_good[:fit_n].astype(float), v_good[:fit_n], 1
                )
                out[fill_idx] = slope * fill_idx.astype(float) + intercept
            elif fit_n == 1:
                out[fill_idx] = v_good[0]

        # Trailing edge: indices (x_max, T) — fill the `max_edge_extrap` frames
        # closest to x_max (i.e. [x_max + 1, x_max + 1 + n_fill)).
        if x_max < T - 1:
            n_trail = T - 1 - x_max
            n_fill = int(min(n_trail, max_edge_extrap))
            fill_idx = np.arange(x_max + 1, x_max + 1 + n_fill)
            if fit_n >= 2:
                slope, intercept = np.polyfit(
                    x_good[-fit_n:].astype(float), v_good[-fit_n:], 1
                )
                out[fill_idx] = slope * fill_idx.astype(float) + intercept
            elif fit_n == 1:
                out[fill_idx] = v_good[-1]

    return out, phantom_mask


def _nan_medfilt1d(vals: np.ndarray, kernel_size: int) -> np.ndarray:
    """NaN-aware 1-D median filter. Each output is the median of valid values in the window."""
    half = kernel_size // 2
    T = len(vals)
    out = np.full(T, np.nan)
    for t in range(T):
        window = vals[max(0, t - half):min(T, t + half + 1)]
        valid = window[np.isfinite(window)]
        if len(valid) > 0:
            out[t] = np.median(valid)
    return out


def _mask_low_confidence(
    kp_array: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    kp_names: Optional[List[str]] = None,
    exclude_keypoint_patterns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Set keypoints with confidence < threshold to NaN.

    Args:
        kp_array:   (T, N, 3)  3-D keypoint positions
        confidence: (T, N)     confidence scores in [0, 1]
        threshold:  scalar     detections below this are masked
        kp_names:   list of N keypoint names (used for exclusion matching)
        exclude_keypoint_patterns: list of substrings; keypoints whose name
                    contains a pattern are not masked

    Returns:
        kp_masked: (T, N, 3) copy with NaN where confidence was low
        bad_mask:  (T, N) bool — True where keypoints were masked
    """
    bad_mask = confidence < threshold                      # (T, N)

    # Zero out bad_mask for excluded keypoints
    n_excluded = 0
    if kp_names and exclude_keypoint_patterns:
        for idx, name in enumerate(kp_names):
            for pat in exclude_keypoint_patterns:
                if pat in name:
                    n_before = int(np.sum(bad_mask[:, idx]))
                    bad_mask[:, idx] = False
                    n_excluded += n_before
                    break

    kp_masked = kp_array.copy().astype(float)
    kp_masked[bad_mask] = np.nan                           # broadcast over xyz dim
    n_bad = int(np.sum(bad_mask))
    frac = n_bad / bad_mask.size * 100
    exclude_msg = f" (excluded {n_excluded} wing keypoint-frames)" if n_excluded else ""
    print(f"  [confidence] masked {n_bad}/{bad_mask.size} keypoint-frames "
          f"({frac:.1f}%) below threshold={threshold}{exclude_msg}")
    return kp_masked, bad_mask


def _detect_bone_length_outliers(
    kp_array: np.ndarray,
    edges: np.ndarray,
    threshold_std: float = 3.0,
    kp_names: Optional[List[str]] = None,
    exclude_keypoint_patterns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Flag frames where a bone's length deviates more than threshold_std sigma
    (MAD-based) from its temporal median. Both endpoint keypoints are set to NaN.

    Args:
        kp_array:      (T, N, 3) — may already contain NaNs
        edges:         (E, 2)    skeleton edge index pairs
        threshold_std: deviation cutoff in robust-sigma units
        kp_names:      list of N keypoint names (used for exclusion matching)
        exclude_keypoint_patterns: list of substrings; edges where *either*
                       endpoint name contains a pattern are skipped

    Returns:
        kp_flagged: copy with additional NaN outliers
        report:     {edge_index: n_frames_flagged}
    """
    kp_flagged = kp_array.copy()
    report: Dict[int, int] = {}
    total_flagged = 0
    n_excluded = 0

    # Build set of keypoint indices to exclude from bone-length checking
    excluded_kp_indices = set()
    if kp_names and exclude_keypoint_patterns:
        for idx, name in enumerate(kp_names):
            for pat in exclude_keypoint_patterns:
                if pat in name:
                    excluded_kp_indices.add(idx)
                    break

    # Vectorized bone length computation for all edges at once
    # edges shape: (E, 2) -> compute all E bone lengths across T frames
    idx_i = edges[:, 0]  # (E,)
    idx_j = edges[:, 1]  # (E,)
    # all_lengths shape: (T, E)
    all_lengths = np.linalg.norm(
        kp_array[:, idx_i, :] - kp_array[:, idx_j, :], axis=2
    )

    for ei in range(len(edges)):
        i, j = edges[ei]

        # Skip edges involving excluded keypoints
        if i in excluded_kp_indices or j in excluded_kp_indices:
            n_excluded += 1
            continue

        lengths = all_lengths[:, ei]
        valid = np.isfinite(lengths)
        if np.sum(valid) < 5:
            continue

        median_len = np.median(lengths[valid])
        mad = np.median(np.abs(lengths[valid] - median_len))
        sigma = mad / 0.6745  # MAD → σ for Gaussian
        if sigma < 1e-8:
            continue

        outlier = (np.abs(lengths - median_len) > threshold_std * sigma) & valid
        n = int(np.sum(outlier))
        if n > 0:
            kp_flagged[outlier, i, :] = np.nan
            kp_flagged[outlier, j, :] = np.nan
            total_flagged += n
            report[ei] = n

    exclude_msg = f" (skipped {n_excluded} excluded edges)" if n_excluded else ""
    if total_flagged:
        print(f"  [bone-length] flagged {total_flagged} keypoint-frames across "
              f"{len(report)} bones (>{threshold_std}σ from median length){exclude_msg}")
    else:
        print(f"  [bone-length] no outliers found (threshold={threshold_std}σ){exclude_msg}")
    return kp_flagged, report


def _medfilt_interpolate(
    kp_array: np.ndarray,
    medfilt_kernel: int = 5,
    spike_threshold_std: float = 5.0,
    use_spline: bool = True,
) -> np.ndarray:
    """
    NaN-aware median-filter spike detection + spline gap interpolation.

    Per coordinate per keypoint:
      1. Compute NaN-aware median filter to get a reference trajectory.
      2. Flag values deviating > spike_threshold_std * MAD-sigma as spikes → NaN.
      3. Spline-interpolate all NaN gaps (original + newly flagged).
      4. Forward/backward fill any remaining edge NaNs (too-sparse gaps).

    Args:
        kp_array:           (T, N, 3) — may contain NaN
        medfilt_kernel:     window size for median filter (must be odd)
        spike_threshold_std: sigma cutoff for spike detection
        use_spline:         use cubic spline for interpolation (False = linear)
    """
    if medfilt_kernel % 2 == 0:
        medfilt_kernel += 1

    T, N, _ = kp_array.shape
    kp_out = kp_array.copy()
    total_spikes = 0

    for n in range(N):
        for d in range(3):
            vals = kp_out[:, n, d].copy()
            finite_mask = np.isfinite(vals)

            if np.sum(finite_mask) < medfilt_kernel:
                # Not enough data for median filter — just interpolate interior
                # gaps. Edges (if any) are left NaN here and handled later by
                # `_interpolate_nan_gaps` with the real config knobs.
                filled, _ = _nan_interp_1d(vals, use_spline=use_spline)
                kp_out[:, n, d] = filled
                continue

            # NaN-aware median filter reference
            vals_med = _nan_medfilt1d(vals, medfilt_kernel)

            # Spike detection: MAD of deviations from median filter
            deviations = np.abs(vals - vals_med)
            valid_devs = deviations[finite_mask]
            mad = np.median(np.abs(valid_devs - np.median(valid_devs)))
            sigma = mad / 0.6745

            if sigma > 1e-8:
                spike_mask = (deviations > spike_threshold_std * sigma) & finite_mask
                n_spikes = int(np.sum(spike_mask))
                if n_spikes > 0:
                    vals[spike_mask] = np.nan
                    total_spikes += n_spikes

            # Interior-only fill; the subsequent `_interpolate_nan_gaps` pass in
            # `filter_keypoints` applies bounded edge extrapolation using the
            # real config-driven limits.
            filled, _ = _nan_interp_1d(vals, use_spline=use_spline)
            kp_out[:, n, d] = filled

    if total_spikes:
        print(f"  [medfilt+interp] detected {total_spikes} coordinate-spikes "
              f"(>{spike_threshold_std}σ from medfilt); interpolated all NaN gaps "
              f"(kernel={medfilt_kernel})")
    else:
        print(f"  [medfilt+interp] no spikes detected; interpolated NaN gaps "
              f"(kernel={medfilt_kernel})")
    return kp_out


def despike_isolated_spikes(
    arr: np.ndarray,
    threshold_factor: float = 10.0,
    max_iterations: int = 1,
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """Remove tracking glitches via velocity-reversal detection.

    Each pass finds frames where the jump in is large, the jump out is
    large, and the two jumps have opposite signs (immediate reversal).
    Flagged frames are replaced with the average of their neighbours.

    With ``max_iterations=1`` (default), only true single-frame spikes are
    fixed — safe for signals with fast oscillations like male wing song.

    With ``max_iterations>1``, multi-frame glitches are peeled from the
    outside in: a 3-frame spike becomes a 2-frame spike after pass 1,
    then a 1-frame spike after pass 2, fully fixed by pass 3.  Use higher
    values only for signals where multi-frame tracking errors are expected
    and real fast oscillations are absent (e.g. non-singing flies).

    Works on arrays of any shape whose first axis is time:
    ``(T,)``, ``(T, D)``, ``(T, N, 3)``, etc.

    Parameters
    ----------
    arr : ndarray
        Input array.  First axis is time.
    threshold_factor : float
        A frame is flagged when its inward *and* outward velocity both
        exceed ``threshold_factor × median(|diff|)`` for that signal.
    max_iterations : int
        Number of passes.  1 = single-frame only (conservative, safe for
        song).  Higher values peel multi-frame glitches layer by layer.
    verbose : bool
        Print a summary line to stdout.

    Returns
    -------
    (cleaned, n_fixed) : tuple[ndarray, int]
        *cleaned* has the same shape/dtype as *arr*.  *n_fixed* is the total
        number of spike frames replaced across all signals.
    """
    if arr.ndim == 0 or arr.shape[0] < 3:
        return arr.copy(), 0

    T = arr.shape[0]
    orig_shape = arr.shape
    # Flatten to (T, C) so we can iterate columns
    flat = arr.reshape(T, -1).astype(np.float64, copy=True)
    C = flat.shape[1]
    total_fixed = 0

    for c in range(C):
        x = flat[:, c]

        # Compute threshold once from the original signal's velocity scale
        v0 = np.diff(x)
        abs_v0 = np.abs(v0)
        finite = np.isfinite(abs_v0)
        if finite.sum() < 3:
            continue
        med_v = float(np.median(abs_v0[finite]))
        if med_v < 1e-12:
            continue
        thresh = threshold_factor * med_v

        for _iteration in range(max_iterations):
            v = np.diff(x)
            abs_v = np.abs(v)

            big = abs_v > thresh
            big_in  = big[:-1]
            big_out = big[1:]
            reversal = (v[:-1] * v[1:]) < 0
            spike = big_in & big_out & reversal

            idx = np.nonzero(spike)[0] + 1
            if idx.size == 0:
                break

            n_this = 0
            for t in idx:
                left, right = x[t - 1], x[t + 1]
                if np.isfinite(left) and np.isfinite(right):
                    x[t] = (left + right) / 2.0
                    n_this += 1
            total_fixed += n_this
            if n_this == 0:
                break

    out = flat.reshape(orig_shape)
    if verbose and total_fixed:
        tag = "single-frame" if max_iterations == 1 else f"up to {max_iterations} passes"
        print(f"  [isolated-spike] fixed {total_fixed} spike frames ({tag}, "
              f">{threshold_factor}\u00d7 median velocity, with reversal)")
    return out, total_fixed


def repair_wing_tip_identity_swaps(
    kp0: np.ndarray,
    kp1: np.ndarray,
    kp_names: Sequence[str],
    wing_kps: Tuple[str, str] = ('WingL_V13', 'WingR_V13'),
    threshold_mm: float = 0.10,
    max_flicker_frames: int = 30,
    max_iterations: int = 3,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Fix short-run wing-tip identity swaps between two tracked flies.

    During close-contact courtship the JARVIS multi-animal tracker occasionally
    assigns fly0's wing-tip to fly1 (or vice versa) for a short contiguous run
    of frames ("flicker").  Per-fly despike can't remove this because each
    fly's trajectory contains a real wing-tip position — just the wrong one.

    Each wing is handled independently.  For every transition ``i`` (between
    frames ``i`` and ``i+1``) we compare two hypotheses for this single wing:

    * ``keep``  — the two tracks continue as-is
    * ``swap``  — fly0's and fly1's wing identities switch at frame ``i+1``

    Transitions where ``cost(keep) − cost(swap) > threshold_mm`` are marked
    as flip events.  The per-transition cost is label-symmetric under a
    global swap, so absolute parity can't be recovered from cost alone.
    To avoid any global-parity ambiguity, we only act on **short** runs
    between consecutive flip events: when two flip events at ``t_a < t_b``
    are separated by ``t_b − t_a <= max_flicker_frames`` frames, the region
    ``[t_a+1, t_b]`` is treated as a flicker and its wing identities are
    exchanged.  Long regions are left untouched — if the tracker held a
    state for hundreds of frames, we trust that state.

    Only the two wing-tip keypoints are modified; all other keypoints pass
    through unchanged.  ``max_iterations`` is retained for symmetry with
    :func:`despike_isolated_spikes`; iterations exit early once no further
    flips qualify.

    Parameters
    ----------
    kp0, kp1 : ndarray, shape (T, N, 3)
        Per-fly keypoints, aligned in time.
    kp_names : sequence of str
        Keypoint names; must contain both entries of ``wing_kps``.
    wing_kps : (str, str)
        Names of the left and right wing-tip keypoints to repair.
    threshold_mm : float
        Minimum cost reduction (in position units, typically mm) required
        for a transition to count as a flip event.
    max_flicker_frames : int
        Maximum length (in frames) of a swapped run to repair.  Runs longer
        than this are assumed to be persistent states and left alone.
    max_iterations : int
        Maximum repair passes.  Exits early once a pass produces no flips.
    verbose : bool
        Print a summary line.

    Returns
    -------
    (kp0_out, kp1_out, n_frames_swapped) : tuple[ndarray, ndarray, int]
        Repaired keypoint arrays and total number of frame-level wing
        identity swaps applied across both wings and all iterations.
    """
    if kp0.shape != kp1.shape:
        raise ValueError(
            f"kp0/kp1 shape mismatch: {kp0.shape} vs {kp1.shape}")
    if kp0.ndim != 3 or kp0.shape[-1] != 3:
        raise ValueError(
            f"expected (T, N, 3) arrays, got kp0.shape={kp0.shape}")
    T = kp0.shape[0]
    if T < 3:
        return kp0.copy(), kp1.copy(), 0

    try:
        i_tips = [kp_names.index(n) for n in wing_kps]
    except ValueError as e:
        raise ValueError(
            f"wing keypoints {wing_kps} not in kp_names") from e

    out0 = kp0.astype(np.float64, copy=True)
    out1 = kp1.astype(np.float64, copy=True)
    total_swaps = 0

    def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(a - b, axis=-1)
        return np.where(np.isfinite(d), d, np.inf)

    def _short_run_assignment(parity: np.ndarray) -> np.ndarray:
        """True in every contiguous ``parity==True`` run of length
        ``<= max_flicker_frames``; False elsewhere."""
        assign = np.zeros(T, dtype=bool)
        in_run = False
        run_start = 0
        for t in range(T):
            if parity[t] and not in_run:
                in_run = True
                run_start = t
            elif not parity[t] and in_run:
                if t - run_start <= max_flicker_frames:
                    assign[run_start:t] = True
                in_run = False
        if in_run and (T - run_start) <= max_flicker_frames:
            assign[run_start:T] = True
        return assign

    def _assignment_for_wing(tip_idx: int) -> Tuple[np.ndarray, int]:
        """Return (per-frame assignment bool, n_frames_swapped) for one wing."""
        tip0 = out0[:, tip_idx]
        tip1 = out1[:, tip_idx]
        keep = _dist(tip0[1:], tip0[:-1]) + _dist(tip1[1:], tip1[:-1])
        swap = _dist(tip1[1:], tip0[:-1]) + _dist(tip0[1:], tip1[:-1])
        flip = (keep - swap) > threshold_mm
        if not flip.any():
            return np.zeros(T, dtype=bool), 0
        # Absolute parity is unrecoverable from cost alone, so we evaluate
        # both interpretations and take whichever places more short runs —
        # i.e., covers more plausible flicker frames.  The short-run length
        # cap ensures neither parity ever proposes flipping a long region.
        parity0 = np.concatenate(([False], np.cumsum(flip) % 2 == 1))
        assign0 = _short_run_assignment(parity0)
        assign1 = _short_run_assignment(~parity0)
        chosen = assign0 if assign0.sum() >= assign1.sum() else assign1
        return chosen, int(chosen.sum())

    for _it in range(max_iterations):
        n_this = 0
        for tip_idx in i_tips:
            assign, n = _assignment_for_wing(tip_idx)
            if n == 0:
                continue
            tmp = out0[assign, tip_idx].copy()
            out0[assign, tip_idx] = out1[assign, tip_idx]
            out1[assign, tip_idx] = tmp
            n_this += n
        if n_this == 0:
            break
        total_swaps += n_this

    if verbose and total_swaps:
        print(f"  [wing-tip swap repair] swapped {total_swaps} frame-wing "
              f"identities (>{threshold_mm:.3f} mm per-transition gain, "
              f"runs \u2264 {max_flicker_frames} frames, "
              f"{'/'.join(wing_kps)})")
    return out0.astype(kp0.dtype, copy=False), out1.astype(kp1.dtype, copy=False), total_swaps


def medfilt_despike(
    arr: np.ndarray,
    kernel: int = 7,
    threshold_factor: float = 10.0,
    max_replace_frac: float = 0.10,
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """Replace frames that deviate from a local median by more than a
    velocity-based threshold.

    Designed for non-singing flies where multi-frame tracking excursions
    are common (keypoint drifts to wrong feature for several frames).
    NOT safe for fast oscillatory signals like male wing song — the median
    filter will flatten real wing beats.

    Parameters
    ----------
    arr : ndarray
        Input array, shape ``(T,)``, ``(T, D)``, or ``(T, N, 3)``.
    kernel : int
        Median filter kernel size (must be odd).
    threshold_factor : float
        A frame is replaced when ``|signal - medfilt| > threshold_factor
        × median(|diff(signal)|)``.  Uses the same velocity scale as
        :func:`despike_isolated_spikes`.
    max_replace_frac : float
        Safety cap: skip a signal column if more than this fraction of
        frames would be replaced (avoids destroying good data).
    verbose : bool
        Print summary.

    Returns
    -------
    (cleaned, n_fixed) : tuple[ndarray, int]
    """
    from scipy.signal import medfilt as _medfilt

    if arr.ndim == 0 or arr.shape[0] < kernel:
        return arr.copy(), 0

    T = arr.shape[0]
    orig_shape = arr.shape
    flat = arr.reshape(T, -1).astype(np.float64, copy=True)
    C = flat.shape[1]
    total_fixed = 0

    for c in range(C):
        sig = flat[:, c]
        med = _medfilt(sig, kernel_size=kernel)
        dev = np.abs(sig - med)

        v = np.abs(np.diff(sig))
        vf = v[np.isfinite(v)]
        if len(vf) < 5:
            continue
        med_v = float(np.median(vf))
        if med_v < 1e-12:
            continue
        thresh = threshold_factor * med_v

        bad = dev > thresh
        n_bad = int(bad.sum())
        if n_bad > 0 and n_bad < T * max_replace_frac:
            sig[bad] = med[bad]
            total_fixed += n_bad

    out = flat.reshape(orig_shape)
    if verbose and total_fixed:
        print(f"  [medfilt-despike] replaced {total_fixed} frames "
              f"(kernel={kernel}, >{threshold_factor}\u00d7 median velocity)")
    return out, total_fixed


def _interpolate_nan_gaps(
    kp_array: np.ndarray,
    use_spline: bool = True,
    max_edge_extrap_frames: int = 0,
    edge_fit_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate all NaN gaps in a (T, N, 3) keypoint array.

    Per coordinate: PCHIP interpolate interior NaN gaps. Leading/trailing NaN
    runs are filled by *bounded linear extrapolation* (up to
    ``max_edge_extrap_frames`` frames of linear fit on the nearest
    ``edge_fit_window`` valid points); any edge frames beyond that cap remain
    NaN and are dropped downstream by pair_validity.

    Args:
        kp_array: (T, N, 3) — may contain NaN.
        use_spline: cubic PCHIP (True) or linear (False) for interior gaps.
        max_edge_extrap_frames: max frames of leading/trailing linear
            extrapolation. 0 disables edge extrapolation entirely.
        edge_fit_window: number of valid frames used to fit the edge linear
            model.

    Returns:
        kp_out: (T, N, 3) filled array. May still contain NaN on frame-keypoints
            whose edge NaN run exceeded ``max_edge_extrap_frames``.
        edge_nan_mask: (T, N) bool — True for any (frame, keypoint) that was
            originally in a leading/trailing NaN run on *any* of its three
            coordinates, regardless of whether the frame was filled by
            extrapolation. Used by
            ``pair_validity.compute_single_fly_validity`` to mark phantom
            frames as untrusted.
    """
    T, N, _ = kp_array.shape
    # Reshape to (T, N*3) so we can iterate over columns without nested loops.
    kp_flat = kp_array.reshape(T, -1).copy()
    phantom_flat = np.zeros(kp_flat.shape, dtype=bool)
    nan_counts = np.isnan(kp_flat).sum(axis=0)  # per-column NaN count
    total_filled = 0

    cols_with_nans = np.where(nan_counts > 0)[0]
    for c in cols_with_nans:
        vals = kp_flat[:, c]
        n_nan_before = int(nan_counts[c])

        filled, phantom = _nan_interp_1d(
            vals,
            use_spline=use_spline,
            max_edge_extrap=max_edge_extrap_frames,
            edge_fit_window=edge_fit_window,
        )
        kp_flat[:, c] = filled
        phantom_flat[:, c] = phantom

        total_filled += n_nan_before - int(np.sum(np.isnan(filled)))

    kp_out = kp_flat.reshape(T, N, 3)
    # OR across x/y/z coords for each (T, N) — a frame-keypoint is "edge
    # phantom" if any of its coordinates sat in an edge NaN run.
    edge_nan_mask = np.any(phantom_flat.reshape(T, N, 3), axis=2)

    n_edge_frames = int(np.any(edge_nan_mask, axis=1).sum())
    if total_filled > 0:
        msg = (f"  [interpolation] filled {total_filled} NaN coordinate-values "
               f"({'spline' if use_spline else 'linear'}, "
               f"edge_extrap≤{max_edge_extrap_frames})")
        if n_edge_frames:
            msg += f"; {n_edge_frames} frames flagged edge-phantom"
        print(msg)
    else:
        print(f"  [interpolation] no NaN gaps to fill")
    return kp_out, edge_nan_mask


def _confidence_weighted_smooth(
    kp_array: np.ndarray,
    confidence: np.ndarray,
    smooth_window: int = 11,
    smooth_polyorder: int = 3,
    conf_low: float = 0.3,
    conf_high: float = 0.8,
) -> np.ndarray:
    """
    Blend Savitzky-Golay smoothed trajectory with raw using confidence.

    For each keypoint at each frame:
      - confidence >= conf_high: keep raw value (trust tracker)
      - confidence <= conf_low:  use fully smoothed value
      - in between: linear blend

    Args:
        kp_array:        (T, N, 3) — should be NaN-free (run after interpolation)
        confidence:      (T, N)    continuous confidence scores in [0, 1]
        smooth_window:   Savitzky-Golay window length (must be odd)
        smooth_polyorder: polynomial order for Savitzky-Golay
        conf_low:        confidence at/below which smoothing is fully applied
        conf_high:       confidence at/above which raw data is fully trusted
    """
    T, N, _ = kp_array.shape
    if smooth_window % 2 == 0:
        smooth_window += 1
    smooth_window = min(smooth_window, T if T % 2 != 0 else T - 1)
    if smooth_window <= smooth_polyorder:
        print(f"  [conf-smooth] skipped (window={smooth_window} <= polyorder={smooth_polyorder})")
        return kp_array

    kp_out = kp_array.copy()

    # Compute blend weight from confidence: 0 = fully smoothed, 1 = fully raw
    denom = max(conf_high - conf_low, 1e-8)
    weight = np.clip((confidence - conf_low) / denom, 0.0, 1.0)  # (T, N)

    # Expand weight to (T, N, 3) for broadcasting
    weight_3d = weight[:, :, np.newaxis]  # (T, N, 3)

    # Find keypoints that need blending (any frame with weight < 1)
    needs_blend = np.any(weight < 1.0, axis=0)  # (N,)

    # Apply savgol per coordinate column for keypoints that need it
    # Reshape to (T, N*3) for efficient column-wise savgol
    kp_flat = kp_out.reshape(T, -1)
    smoothed_flat = kp_flat.copy()
    has_nan = np.any(np.isnan(kp_flat), axis=0)  # (N*3,)

    for c in range(kp_flat.shape[1]):
        kp_idx = c // 3
        if not needs_blend[kp_idx] or has_nan[c]:
            continue
        smoothed_flat[:, c] = signal.savgol_filter(kp_flat[:, c], smooth_window, smooth_polyorder)

    smoothed = smoothed_flat.reshape(T, N, 3)
    kp_out = weight_3d * kp_out + (1.0 - weight_3d) * smoothed

    total_blended = int(np.sum(needs_blend * np.sum(weight < 1.0, axis=0)))

    frac = total_blended / (T * N) * 100 if (T * N) > 0 else 0
    print(f"  [conf-smooth] blended {total_blended}/{T * N} keypoint-frames "
          f"({frac:.1f}%) (conf_low={conf_low}, conf_high={conf_high}, "
          f"window={smooth_window})")
    return kp_out


def _savgol_smooth(
    kp_array: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Savitzky-Golay smoothing along the time axis.

    Args:
        kp_array:      (T, N, 3) — assumed NaN-free after interpolation step
        window_length: filter window (must be odd, > polyorder)
        polyorder:     polynomial order
    """
    T, N, _ = kp_array.shape
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, T if T % 2 != 0 else T - 1)
    if window_length <= polyorder:
        print(f"  [savgol] skipped (window_length={window_length} ≤ polyorder={polyorder})")
        return kp_array

    kp_out = kp_array.copy()
    for n in range(N):
        for d in range(3):
            vals = kp_out[:, n, d]
            if np.any(np.isnan(vals)):
                continue  # skip columns still containing NaN
            kp_out[:, n, d] = signal.savgol_filter(vals, window_length, polyorder)

    print(f"  [savgol] smoothed (window={window_length}, polyorder={polyorder})")
    return kp_out


# ─── Visualization ─────────────────────────────────────────────────────────────

def plot_filtering_report(
    kp_raw: np.ndarray,
    kp_filtered: np.ndarray,
    report: Dict,
    kp_names: Optional[List[str]],
    fig_dir,
    bout_name: str = "bout",
    n_traj_keypoints: int = 8,
) -> None:
    """
    Save two diagnostic figures comparing raw and filtered keypoints.

    Figure 1 — ``{bout_name}_filter_overview.png``
        A three-panel overview:
        - Top-left:  T × N heatmap of ||raw - filtered|| (mm); shows *where* and
          *which* keypoints were modified.
        - Top-right: Horizontal bar chart — fraction of frames modified per
          keypoint (sorted descending), great for judging per-keypoint noise level.
        - Bottom:    Total change magnitude summed across all keypoints per frame;
          exposes global trouble spots in the recording.

    Figure 2 — ``{bout_name}_filter_trajectories.png``
        For the ``n_traj_keypoints`` most-modified keypoints, plots x / y / z
        timeseries.  Raw trajectory = light-gray line; filtered = steel-blue.
        Frames where the value changed are shaded red.  Only produced when at
        least one keypoint was actually modified.

    Args:
        kp_raw:           (T, N, 3) raw keypoint positions before filtering.
        kp_filtered:      (T, N, 3) final cleaned keypoint positions.
        report:           Dict returned by ``filter_keypoints`` (used for title
                          annotations; keys: ``confidence_masked``,
                          ``bone_outliers``).
        kp_names:         List of N keypoint name strings (None → use indices).
        fig_dir:          Directory where figures are saved (created if needed).
        bout_name:        Short label used in filenames and figure titles.
        n_traj_keypoints: Maximum number of keypoints shown in Figure 2.
    """
    from pathlib import Path

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Ensure plain numpy arrays (safe against JAX / other array types)
    kp_raw = np.asarray(kp_raw, dtype=float)
    kp_filtered = np.asarray(kp_filtered, dtype=float)

    T, N, _ = kp_raw.shape
    labels = kp_names if (kp_names and len(kp_names) == N) else [str(i) for i in range(N)]

    # -- Change magnitude (T, N) ------------------------------------------------
    # Where raw was NaN (pre-existing missing data), treat as "not a change"
    raw_for_diff = np.where(np.isnan(kp_raw), kp_filtered, kp_raw)
    change_mag = np.sqrt(np.nansum((kp_filtered - raw_for_diff) ** 2, axis=-1))  # (T, N)
    frac_changed = np.mean(change_mag > 1e-8, axis=0)           # (N,)
    total_change_per_frame = change_mag.sum(axis=-1)             # (T,)

    n_modified_kp = int(np.sum(frac_changed > 0))
    total_modified_frames = int(np.sum(total_change_per_frame > 1e-8))

    # Build a short summary string from the report
    summary_parts = [f"{T} frames, {N} keypoints"]
    if 'confidence_masked' in report and report['confidence_masked'] > 0:
        summary_parts.append(f"conf-masked: {report['confidence_masked']}")
    if 'bone_outliers' in report:
        n_bone = sum(report['bone_outliers'].values())
        if n_bone > 0:
            summary_parts.append(f"bone-outliers: {n_bone} frame-edges")
    summary_parts.append(f"frames modified: {total_modified_frames} ({100*total_modified_frames/max(T,1):.1f}%)")
    summary = " | ".join(summary_parts)

    # ── Figure 1: Overview ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[3, 1], width_ratios=[5, 2],
                           hspace=0.45, wspace=0.35)

    # Heatmap
    ax_heat = fig.add_subplot(gs[0, 0])
    vmax = float(np.percentile(change_mag[change_mag > 1e-10], 95)) if change_mag.max() > 1e-10 else 1.0
    im = ax_heat.imshow(change_mag.T, aspect='auto', origin='upper',
                        cmap='hot_r', vmin=0, vmax=vmax, interpolation='none')
    ax_heat.set_xlabel('Frame', fontsize=9)
    ax_heat.set_ylabel('Keypoint', fontsize=9)
    ax_heat.set_yticks(range(N))
    ax_heat.set_yticklabels(labels, fontsize=max(4, min(7, 80 // N)))
    ax_heat.set_title(f'Change magnitude per keypoint/frame — {bout_name}', fontsize=10)
    plt.colorbar(im, ax=ax_heat, label='||raw − filtered|| (mm)', shrink=0.8)

    # Bar chart: % frames changed per keypoint
    ax_bar = fig.add_subplot(gs[0, 1])
    sorted_idx = np.argsort(frac_changed)
    bar_colors = ['#c0392b' if f > 0.1 else '#2980b9' for f in frac_changed[sorted_idx]]
    ax_bar.barh(range(N), frac_changed[sorted_idx] * 100, color=bar_colors)
    ax_bar.set_yticks(range(N))
    ax_bar.set_yticklabels([labels[i] for i in sorted_idx],
                            fontsize=max(4, min(7, 80 // N)))
    ax_bar.set_xlabel('% frames modified', fontsize=9)
    ax_bar.set_title('Modification rate by keypoint', fontsize=10)
    ax_bar.axvline(10, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Total change per frame
    ax_tot = fig.add_subplot(gs[1, :])
    t = np.arange(T)
    ax_tot.fill_between(t, total_change_per_frame, alpha=0.55, color='#2980b9')
    ax_tot.plot(t, total_change_per_frame, linewidth=0.6, color='#1a5276')
    ax_tot.set_xlabel('Frame', fontsize=9)
    ax_tot.set_ylabel('Σ change (mm)', fontsize=9)
    ax_tot.set_title('Total modification magnitude per frame', fontsize=10)
    ax_tot.set_xlim(0, T - 1)

    fig.suptitle(f'Filter overview — {bout_name}\n{summary}', fontsize=11, y=1.01)

    save_path = fig_dir / f"{bout_name}_filter_overview.png"
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [filter viz] Overview saved to: {save_path}")

    # ── Figure 2: Trajectory comparison ────────────────────────────────────────
    if n_modified_kp == 0:
        return  # Nothing was modified — skip trajectory figure

    top_k_idx = np.argsort(frac_changed)[::-1][:n_traj_keypoints]
    n_rows = len(top_k_idx)

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 2.8 * n_rows), squeeze=False)
    t = np.arange(T)

    for row, kp_i in enumerate(top_k_idx):
        name = labels[kp_i]
        for col, dim_label in enumerate(['x', 'y', 'z']):
            ax = axes[row, col]
            raw_vals = kp_raw[:, kp_i, col]
            filt_vals = kp_filtered[:, kp_i, col]
            raw_for_cmp = np.where(np.isnan(raw_vals), filt_vals, raw_vals)
            changed_mask = np.abs(filt_vals - raw_for_cmp) > 1e-8

            # Compute y-axis bounds from data
            all_v = np.concatenate([raw_vals[~np.isnan(raw_vals)], filt_vals[~np.isnan(filt_vals)]])
            if len(all_v) == 0:
                continue
            y_min, y_max = float(all_v.min()), float(all_v.max())
            margin = max((y_max - y_min) * 0.05, 0.5)
            y_lo, y_hi = y_min - margin, y_max + margin

            if np.any(changed_mask):
                ax.fill_between(t, y_lo, y_hi, where=changed_mask,
                                color='salmon', alpha=0.35, zorder=0, label='_nolegend_')

            ax.plot(t, raw_vals, color='#bdc3c7', linewidth=1.4, label='raw', zorder=1)
            ax.plot(t, filt_vals, color='#2980b9', linewidth=1.0, label='filtered', zorder=2)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlim(0, T - 1)
            ax.tick_params(labelsize=6)

            if row == 0:
                ax.set_title(dim_label, fontsize=10)
                if col == 2:
                    ax.legend(fontsize=7, loc='upper right',
                              handles=[
                                  plt.Line2D([0], [0], color='#bdc3c7', lw=1.4, label='raw'),
                                  plt.Line2D([0], [0], color='#2980b9', lw=1.0, label='filtered'),
                                  plt.Rectangle((0, 0), 1, 1, fc='salmon', alpha=0.35, label='changed'),
                              ])
            if col == 0:
                ax.set_ylabel(name, fontsize=8)
        if row == n_rows - 1:
            axes[row, 1].set_xlabel('Frame', fontsize=9)

    fig.suptitle(
        f'Trajectory comparison — {bout_name}\n'
        f'(top {n_rows} most-modified keypoints; red = changed frames)',
        fontsize=11, y=1.01
    )
    plt.tight_layout()

    save_path = fig_dir / f"{bout_name}_filter_trajectories.png"
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [filter viz] Trajectories saved to: {save_path}")


# ─── Public API ────────────────────────────────────────────────────────────────

def filter_keypoints(
    kp_array: np.ndarray,
    confidence: Optional[np.ndarray],
    skeleton_edges: np.ndarray,
    filter_cfg: DictConfig,
    fig_dir=None,
    bout_name: str = "bout",
    kp_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Apply the full filtering pipeline to a (T, N, 3) keypoint array.

    Steps (each enabled/disabled via filter_cfg sub-keys):
      1. confidence masking           (filter_cfg.confidence.enabled)
      2. bone-length outliers         (filter_cfg.bone_length.enabled, default OFF)
     2b. centroid-jump masking        (filter_cfg.centroid_jump.enabled, default OFF)
     2c. isolated-spike removal       (filter_cfg.isolated_spike.enabled, default ON)
      3. medfilt spike detection      (filter_cfg.medfilt.enabled, default OFF)
      4. interpolate NaN gaps         (filter_cfg.interpolation.enabled)
      5. confidence-weighted smooth   (filter_cfg.confidence_smooth.enabled)
      6. savgol smoothing             (filter_cfg.savgol.enabled, default OFF)

    The interpolation stage replaces the legacy constant-hold edge fill with
    bounded linear extrapolation (controlled by
    ``filter_cfg.interpolation.max_edge_extrap_frames`` and ``.edge_fit_window``)
    and returns a ``(T, N)`` ``edge_nan_mask`` identifying frame-keypoints that
    were originally in a leading/trailing NaN run. Downstream,
    ``pair_validity.compute_single_fly_validity`` uses this mask to mark those
    phantom frames as untrusted (``valid_fly = False``) even when they were
    filled by extrapolation. Subsequent stages (``confidence_smooth``,
    ``savgol``) already skip columns containing NaN, so any edge frames left
    un-extrapolated propagate as NaN safely.

    Args:
        kp_array:       (T, N, 3)  raw keypoint positions (mm)
        confidence:     (T, N)     confidence scores in [0, 1], or None
        skeleton_edges: (E, 2)     skeleton edge index pairs
        filter_cfg:     OmegaConf DictConfig with filtering sub-keys
        fig_dir:        If provided, save diagnostic figures here.
        bout_name:      Label used in figure filenames/titles.
        kp_names:       List of N keypoint names for plot axis labels.

    Returns:
        kp_filtered:   (T, N, 3) cleaned array
        report:        per-step summary dict
        edge_nan_mask: (T, N) bool — True where a frame-keypoint was in a
            leading/trailing NaN run in the input (phantom after any bounded
            extrapolation). Passed to pair_validity for honest validity masks.
    """
    report: Dict = {}
    kp_raw = kp_array.astype(float)
    kp = kp_raw.copy()
    T_raw, N_raw = kp_raw.shape[:2]
    edge_nan_mask = np.zeros((T_raw, N_raw), dtype=bool)
    print("\n--- Keypoint Filtering ---")

    # Step 1: confidence masking — NaN out low-confidence frames
    if filter_cfg.get('confidence', {}).get('enabled', False) and confidence is not None:
        threshold = filter_cfg.confidence.get('threshold', 0.5)
        conf_exclude = list(filter_cfg.confidence.get('exclude_keypoint_patterns', []))
        kp, conf_mask = _mask_low_confidence(
            kp, confidence, threshold,
            kp_names=kp_names, exclude_keypoint_patterns=conf_exclude,
        )
        report['confidence_masked'] = int(np.sum(conf_mask))

    # Step 2: bone-length outlier detection (optional, default OFF)
    if filter_cfg.get('bone_length', {}).get('enabled', False):
        std_thresh = filter_cfg.bone_length.get('threshold_std', 3.0)
        exclude_patterns = list(filter_cfg.bone_length.get('exclude_keypoint_patterns', []))
        kp, bone_rpt = _detect_bone_length_outliers(
            kp, skeleton_edges, std_thresh,
            kp_names=kp_names, exclude_keypoint_patterns=exclude_patterns,
        )
        report['bone_outliers'] = bone_rpt

    # Step 2b: centroid-jump detection for identity switches (optional, default OFF)
    # Detect on the *raw* input (kp_raw) so prior bone-length masking doesn't
    # NaN out the centroid and hide the jump. Apply the mask to the in-progress kp.
    if filter_cfg.get('centroid_jump', {}).get('enabled', False) and kp_names is not None:
        from utils.centroid_jump_check import detect_centroid_jumps
        cj_cfg = filter_cfg.centroid_jump
        trunk_kps = list(cj_cfg.get('trunk_keypoints', ['Scutellum', 'Postnotum', 'Scutum']))
        thresh = cj_cfg.get('threshold_mm', 2.0)
        win = cj_cfg.get('window', 1)
        bad_mask, cj_report = detect_centroid_jumps(
            kp_raw, kp_names, trunk_kps, threshold_mm=thresh,
        )
        if win > 0 and bad_mask.any():
            expanded = bad_mask.copy()
            for off in range(1, win + 1):
                expanded[off:] |= bad_mask[:-off]
                expanded[:-off] |= bad_mask[off:]
            bad_mask = expanded
        n_masked = int(bad_mask.sum())
        cj_report['n_frames_masked'] = n_masked
        cj_report['window'] = win
        kp[bad_mask] = np.nan
        report['centroid_jumps'] = cj_report
        print(f"  [centroid jump] Flagged {cj_report['n_frames_flagged']} frames "
              f"(masked {n_masked} with window={win}) [computed on raw input]")

    # Step 2c: isolated spike removal (velocity-reversal detector).
    # With max_iterations=1 (default): single-frame only, safe for all signals.
    # With max_iterations>1: peels multi-frame glitches from outside in.
    if filter_cfg.get('isolated_spike', {}).get('enabled', True):
        _iso_cfg = filter_cfg.get('isolated_spike', {})
        _iso_thresh = _iso_cfg.get('threshold_factor', 10.0)
        _iso_iters = _iso_cfg.get('max_iterations', 1)
        kp, _n_iso = despike_isolated_spikes(
            kp, threshold_factor=_iso_thresh, max_iterations=_iso_iters, verbose=True)
        report['isolated_spikes'] = _n_iso

    # Step 2d: median-filter despike (optional, default OFF).
    # Catches multi-frame tracking excursions that lack a velocity reversal.
    # NOT safe for fast oscillations (male wing song) — enable only for
    # non-singing flies or when song analysis is not needed.
    if filter_cfg.get('medfilt_despike', {}).get('enabled', False):
        _mfd_cfg = filter_cfg.get('medfilt_despike', {})
        _mfd_kernel = _mfd_cfg.get('kernel', 7)
        _mfd_thresh = _mfd_cfg.get('threshold_factor', 10.0)
        kp, _n_mfd = medfilt_despike(
            kp, kernel=_mfd_kernel, threshold_factor=_mfd_thresh, verbose=True)
        report['medfilt_despike'] = _n_mfd

    # Step 3: medfilt spike detection + spline interpolation (optional, default OFF)
    if filter_cfg.get('medfilt', {}).get('enabled', False):
        kernel = filter_cfg.medfilt.get('kernel', 5)
        spike_std = filter_cfg.medfilt.get('spike_threshold_std', 5.0)
        use_spline = filter_cfg.medfilt.get('use_spline', True)
        kp = _medfilt_interpolate(kp, medfilt_kernel=kernel,
                                  spike_threshold_std=spike_std, use_spline=use_spline)

    # Step 4: interpolate NaN gaps (from confidence masking and/or outlier steps)
    if filter_cfg.get('interpolation', {}).get('enabled', True):
        interp_cfg = filter_cfg.interpolation
        use_spline = interp_cfg.get('use_spline', True)
        max_edge_extrap = int(interp_cfg.get('max_edge_extrap_frames', 0))
        edge_fit_window = int(interp_cfg.get('edge_fit_window', 5))
        kp, edge_nan_mask = _interpolate_nan_gaps(
            kp,
            use_spline=use_spline,
            max_edge_extrap_frames=max_edge_extrap,
            edge_fit_window=edge_fit_window,
        )
        report['edge_nan_frames'] = int(np.any(edge_nan_mask, axis=1).sum())

    # Step 5: confidence-weighted smoothing — blend raw + smoothed by confidence
    if filter_cfg.get('confidence_smooth', {}).get('enabled', False) and confidence is not None:
        window = filter_cfg.confidence_smooth.get('smooth_window', 11)
        order = filter_cfg.confidence_smooth.get('smooth_polyorder', 3)
        c_low = filter_cfg.confidence_smooth.get('conf_low', 0.3)
        c_high = filter_cfg.confidence_smooth.get('conf_high', 0.8)
        kp = _confidence_weighted_smooth(kp, confidence,
                                         smooth_window=window, smooth_polyorder=order,
                                         conf_low=c_low, conf_high=c_high)

    # Step 6: Savitzky-Golay final smoothing (optional, default OFF)
    if filter_cfg.get('savgol', {}).get('enabled', False):
        window = filter_cfg.savgol.get('window_length', 11)
        order = filter_cfg.savgol.get('polyorder', 3)
        kp = _savgol_smooth(kp, window_length=window, polyorder=order)

    print("--- Filtering complete ---\n")

    if fig_dir is not None:
        try:
            plot_filtering_report(kp_raw, kp, report, kp_names, fig_dir, bout_name)
        except Exception as exc:
            print(f"  [filter viz] Warning: figure generation failed — {exc}")
            _traceback.print_exc()

    return kp, report, edge_nan_mask


def load_confidence_from_csv(
    csv_path,
    frame_indices: Optional[np.ndarray],
    csv_kp_names: List[str],
    csv_to_filtered_idx: Dict,
    filtered_node_names: List[str],
) -> Optional[np.ndarray]:
    """
    Load per-keypoint confidence for a single bout, reordered to filtered skeleton order.

    The CSV has columns: <KeypointName>_confidence for each keypoint.

    Args:
        csv_path:            path to data3D.csv
        frame_indices:       integer frame indices to load, or None (load all)
        csv_kp_names:        keypoint names as they appear in the CSV
        csv_to_filtered_idx: maps CSV keypoint name → filtered skeleton index
        filtered_node_names: ordered list of matched node names

    Returns:
        (T, N) confidence array in filtered skeleton order, or None if no
        confidence columns found in the CSV.
    """
    import pandas as pd

    df = pd.read_csv(csv_path, header=[0, 1])
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                  for col in df.columns.values]

    conf_columns = [c for c in df.columns if c.endswith('_confidence')]
    if not conf_columns:
        print("  [filter] No confidence columns found — skipping confidence masking")
        return None

    conf_kp_names = [c[:-len('_confidence')] for c in conf_columns]
    conf_raw = df[conf_columns]
    if frame_indices is not None:
        conf_raw = conf_raw.iloc[frame_indices]

    conf_array = conf_raw.values.astype(float)
    T = conf_array.shape[0]
    N = len(filtered_node_names)
    confidence = np.ones((T, N), dtype=float)  # default = high confidence

    for csv_name, new_idx in csv_to_filtered_idx.items():
        if csv_name in conf_kp_names:
            col_idx = conf_kp_names.index(csv_name)
            confidence[:, new_idx] = conf_array[:, col_idx]

    return confidence


def load_confidence_concatenated(
    csv_path,
    bouts: List[Dict],
    csv_kp_names: List[str],
    csv_to_filtered_idx: Dict,
    filtered_node_names: List[str],
) -> Optional[np.ndarray]:
    """
    Load per-keypoint confidence for multiple bouts, concatenated in bout order.
    Mirrors load_concatenated_bouts but for confidence columns.

    Args:
        csv_path:            path to data3D.csv
        bouts:               list of bout dicts with 'start_frame', 'end_frame'
        csv_kp_names:        keypoint names as they appear in the CSV
        csv_to_filtered_idx: maps CSV keypoint name → filtered skeleton index
        filtered_node_names: ordered list of matched node names

    Returns:
        (T_total, N) array in filtered skeleton order, or None if no confidence
        columns found.
    """
    import pandas as pd
    from pathlib import Path
    from utils.fly_detection import build_compact_frame_map

    df = pd.read_csv(csv_path, header=[0, 1])
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                  for col in df.columns.values]

    conf_columns = [c for c in df.columns if c.endswith('_confidence')]
    if not conf_columns:
        return None

    conf_kp_names = [c[:-len('_confidence')] for c in conf_columns]
    conf_data = df[conf_columns]
    n_frames_available = len(conf_data)
    N = len(filtered_node_names)
    conf_bouts = []

    tracking_info_path = Path(csv_path).parent / "tracking_info.json"
    compact_map = build_compact_frame_map(tracking_info_path, n_frames_available)

    for bout in bouts:
        if compact_map is not None:
            rows = [compact_map[f]
                    for f in range(bout['start_frame'], bout['end_frame'])
                    if f in compact_map]
            if not rows:
                continue
            frame_idx = np.array(rows)
        else:
            if bout['start_frame'] >= n_frames_available or bout['end_frame'] > n_frames_available:
                continue
            frame_idx = np.arange(bout['start_frame'], bout['end_frame'])
        bout_conf_raw = conf_data.iloc[frame_idx].values.astype(float)
        T_bout = len(frame_idx)
        bout_conf = np.ones((T_bout, N), dtype=float)
        for csv_name, new_idx in csv_to_filtered_idx.items():
            if csv_name in conf_kp_names:
                col_idx = conf_kp_names.index(csv_name)
                bout_conf[:, new_idx] = bout_conf_raw[:, col_idx]
        conf_bouts.append(bout_conf)

    if not conf_bouts:
        return None
    return np.concatenate(conf_bouts, axis=0)
