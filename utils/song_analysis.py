"""Courtship song detection and classification.

Ported from the exploratory ``notebooks/Courtship_Song_Analysis.ipynb``
sandbox (cells 3, 18, 20, 21, 23, 25) into a reusable module so the
paper-figure notebook can stay thin. The two are kept behaviourally
equivalent; thresholds match the sandbox defaults except where noted.

The pipeline, per fly per bout, is:

1. Build a wing-tip signal dict from the RAW world-frame keypoints
   ``kp_data`` (NOT the IK-reconstructed egocentric positions — see
   comment in ``compute_wing_tip_signal``).
2. Detect frames where any wing pair is oscillating above a
   |dZ/dt| activity threshold → ``is_singing`` mask + dominant wing.
3. For each wing independently, run an FFT-based pulse/sine/waggle/
   quiet classifier on the tip Z trace, then override the pulse label
   with a peak-based detector on the bilateral max ``|dZ/dt|`` so
   sparse pulses that FFT smooths out still get caught.
4. Merge per-side results into a bout summary with ``song_fraction``
   (pulse + sine + waggle frames / total frames) for downstream male
   / female identification.

No dependencies beyond numpy / scipy. This module does not touch h5 or
configuration files; the notebook does that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, spectrogram as sp_spectrogram


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

#: Canonical keypoint names the song detector needs to find on a
#: ``kp_data`` / ``xpos_egocentric`` array. The sandbox notebook hard-
#: codes indices 0-9; we look them up from ``kp_names`` instead so the
#: module survives changes to keypoint ordering.
_REQUIRED_KP = (
    "Scutellum",
    "WingL_base",
    "WingR_base",
    "Antenna_Base",
    "WingL_V12",
    "WingL_V13",
    "WingR_V12",
    "WingR_V13",
)


@dataclass
class SongAnalysisConfig:
    """All tunable parameters for the song detector.

    Defaults reproduce
    ``notebooks/Courtship_Song_Analysis.ipynb`` cell 20 exactly so
    results from the paper notebook match the sandbox. The comments
    note which frame-rate / unit assumptions each value makes.
    """

    # Frame rate -------------------------------------------------------------
    fs: float = 800.0  # Hz — Johnson-lab courtship camera rate

    # Spectrogram windowing --------------------------------------------------
    fft_nperseg: int = 48  # ~60 ms, freq res ~16.7 Hz
    fft_hop: int = 8       # ~10 ms
    fft_song_band: Tuple[float, float] = (80.0, 400.0)
    fft_sine_band: Tuple[float, float] = (100.0, 180.0)
    fft_pulse_band: Tuple[float, float] = (140.0, 400.0)
    fft_waggle_band: Tuple[float, float] = (5.0, 25.0)

    # FFT classifier thresholds ---------------------------------------------
    song_power_threshold: float = 2e-8
    pulse_peak_freq_min: float = 140.0
    pulse_freq_ratio_min: float = 0.35
    waggle_power_ratio_min: float = 1.0
    min_segment_frames: int = 16

    # Peak-based pulse detection (operates on bilateral max |dZ/dt|) --------
    use_peak_pulse: bool = True
    pulse_detect_height: float = 25.0       # min |dZ/dt| (data-units / s)
    pulse_detect_prominence: float = 15.0
    pulse_detect_min_dist_ms: float = 15.0
    pulse_detect_half_width_ms: float = 12.0
    # Max gap between consecutive detected peaks that still counts as the
    # same pulse train. Normal D. melanogaster inter-pulse intervals are
    # ~35 ms; a single missed peak doubles that to ~70 ms, two missed peaks
    # gives ~105 ms. 150 ms tolerates up to ~3 dropped peaks in a row
    # while still being well below the typical >300 ms inter-train pause,
    # so it never accidentally merges two distinct pulse trains.
    pulse_train_max_gap_ms: float = 150.0
    # Minimum ratio of peak height to local baseline |dZ/dt|. True pulses
    # stand ≥3-4× above the inter-pulse baseline; individual sine-carrier
    # cycles only rise ~1.5-2.5× above the sustained carrier energy, so
    # this cleanly rejects false pulses inside strong sine song.
    # Set to 0 to disable the filter.
    pulse_baseline_ratio_min: float = 3.0
    # ± window (ms) used to compute the local |dZ/dt| baseline at each
    # candidate peak. Must be wider than one inter-pulse interval (~35 ms)
    # so a pulse train's baseline is dominated by inter-pulse silence
    # rather than the pulses themselves.
    pulse_baseline_window_ms: float = 40.0
    # ± window (ms) around the peak itself to EXCLUDE from the baseline
    # computation (removes the peak's own width from the median).
    pulse_baseline_exclude_ms: float = 5.0

    # Per-pulse feature extraction (Clemens 2018 adapted to 800 Hz) ---------
    # 25 ms total window → 21 samples at 800 Hz (±10 frames + center).
    pulse_window_ms: float = 25.0
    # Min pulses in a bout before per-bout Pslow/Pfast fractions are trusted
    # as summary statistics (per-pulse features are still extracted either
    # way; this just gates downstream aggregation).
    pulse_feature_min_n: int = 40
    # Spectral center-of-mass threshold from Clemens (e^-1 of peak
    # magnitude). Frequencies whose |X(f)| exceeds this fraction of the
    # spectrum's maximum contribute to the carrier-frequency estimate.
    pulse_spectral_thresh: float = 0.3679

    # Wing activity detection (gating + dominant-wing choice) ---------------
    wing_activity_window: int = 100     # frames for smoothed |dZ/dt|
    wing_activity_threshold: float = 1.0
    singing_min_bout_frames: int = 80
    singing_max_gap_frames: int = 200

    # Tips used for the spectrogram (paired: one left + one right) ----------
    left_tip: str = "WingL_V13"
    right_tip: str = "WingR_V13"

    # Default wing joint indices in qpos (may be overridden per-model) -----
    qpos_wing_yaw_L: int = 7
    qpos_wing_roll_L: int = 8
    qpos_wing_pitch_L: int = 9
    qpos_wing_yaw_R: int = 10
    qpos_wing_roll_R: int = 11
    qpos_wing_pitch_R: int = 12

    def ms_to_frames(self, ms: float) -> int:
        return max(1, int(round(ms * 1e-3 * self.fs)))


# -----------------------------------------------------------------------------
# Keypoint index helpers
# -----------------------------------------------------------------------------


def resolve_kp_indices(kp_names: Sequence[str]) -> Dict[str, int]:
    """Return a ``{name: index}`` dict for the keypoints the detector needs.

    Raises ``KeyError`` if any required keypoint is missing, so callers fail
    loudly instead of silently computing garbage.
    """
    names = list(kp_names)
    out: Dict[str, int] = {}
    missing: List[str] = []
    for kp in _REQUIRED_KP:
        try:
            out[kp] = names.index(kp)
        except ValueError:
            missing.append(kp)
    if missing:
        raise KeyError(
            f"song_analysis: missing required keypoint(s) {missing}. "
            f"Available: {names}"
        )
    return out


# -----------------------------------------------------------------------------
# Wing geometry (kinematic features, not used by the classifier itself)
# -----------------------------------------------------------------------------


def compute_wing_extension_angles(
    xpos_ego: np.ndarray,
    kp_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Wing extension angle (deg) from egocentric keypoints.

    For each wing, returns the angle between the body axis
    (Scutellum → Antenna_Base) and the wing vector
    (wing base → midpoint of V12, V13).
    """
    scut = kp_idx["Scutellum"]
    ant = kp_idx["Antenna_Base"]
    wl_b = kp_idx["WingL_base"]
    wr_b = kp_idx["WingR_base"]
    wl12 = kp_idx["WingL_V12"]
    wl13 = kp_idx["WingL_V13"]
    wr12 = kp_idx["WingR_V12"]
    wr13 = kp_idx["WingR_V13"]

    body_axis = xpos_ego[:, ant] - xpos_ego[:, scut]
    body_axis /= np.linalg.norm(body_axis, axis=1, keepdims=True) + 1e-12

    tip_L = 0.5 * (xpos_ego[:, wl12] + xpos_ego[:, wl13])
    vec_L = tip_L - xpos_ego[:, wl_b]
    vec_L /= np.linalg.norm(vec_L, axis=1, keepdims=True) + 1e-12

    tip_R = 0.5 * (xpos_ego[:, wr12] + xpos_ego[:, wr13])
    vec_R = tip_R - xpos_ego[:, wr_b]
    vec_R /= np.linalg.norm(vec_R, axis=1, keepdims=True) + 1e-12

    dot_L = np.clip(np.sum(body_axis * vec_L, axis=1), -1, 1)
    dot_R = np.clip(np.sum(body_axis * vec_R, axis=1), -1, 1)
    return np.degrees(np.arccos(dot_L)), np.degrees(np.arccos(dot_R))


def extract_wing_joint_angles(
    qpos: np.ndarray, cfg: SongAnalysisConfig
) -> Dict[str, np.ndarray]:
    """Return the six wing DOFs from qpos as a dict (radians)."""
    return {
        "yaw_L":   qpos[:, cfg.qpos_wing_yaw_L],
        "roll_L":  qpos[:, cfg.qpos_wing_roll_L],
        "pitch_L": qpos[:, cfg.qpos_wing_pitch_L],
        "yaw_R":   qpos[:, cfg.qpos_wing_yaw_R],
        "roll_R":  qpos[:, cfg.qpos_wing_roll_R],
        "pitch_R": qpos[:, cfg.qpos_wing_pitch_R],
    }


# -----------------------------------------------------------------------------
# Wing activity / singing gate
# -----------------------------------------------------------------------------


def compute_wing_tip_signal(
    kp_world: np.ndarray, kp_idx: Dict[str, int]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract wing tip (x, y, z) traces from a (T, N, 3) keypoint array.

    Uses RAW world-frame keypoints (``kp_data``) rather than the IK-
    reconstructed ``xpos_egocentric`` because the IK output misses real
    strokes in some bouts (noted in cell 21 of the sandbox). Absolute
    position doesn't matter for FFT power or peak detection — only the
    oscillation shape.
    """
    tip_map = {
        "WingL_V12": kp_idx["WingL_V12"],
        "WingL_V13": kp_idx["WingL_V13"],
        "WingR_V12": kp_idx["WingR_V12"],
        "WingR_V13": kp_idx["WingR_V13"],
    }
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, i in tip_map.items():
        out[name] = {
            "x": np.asarray(kp_world[:, i, 0]),
            "y": np.asarray(kp_world[:, i, 1]),
            "z": np.asarray(kp_world[:, i, 2]),
        }
    return out


def _wing_activity(
    wing_data: Dict[str, Dict[str, np.ndarray]],
    fs: float,
    window: int,
) -> Dict[str, np.ndarray]:
    """Windowed mean ``|dZ/dt|`` per wing tip."""
    out: Dict[str, np.ndarray] = {}
    for tip in ("WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13"):
        z = wing_data[tip]["z"]
        dz = np.abs(np.diff(z, prepend=z[0]) * fs)
        out[tip] = uniform_filter1d(dz, size=window)
    return out


def detect_singing_frames(
    wing_data: Dict[str, Dict[str, np.ndarray]],
    cfg: SongAnalysisConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
    """Mark frames with active wing oscillation and pick the dominant wing.

    Returns ``(is_singing, activities, dominant_wing)`` where
    ``dominant_wing`` is ``'L'`` or ``'R'``.
    """
    activities = _wing_activity(
        wing_data, cfg.fs, cfg.wing_activity_window
    )
    thr = cfg.wing_activity_threshold

    left_active = (
        (activities["WingL_V12"] > thr) & (activities["WingL_V13"] > thr)
    )
    right_active = (
        (activities["WingR_V12"] > thr) & (activities["WingR_V13"] > thr)
    )
    is_singing = left_active | right_active
    T = len(is_singing)

    # Bridge short gaps.
    changes = np.diff(is_singing.astype(int))
    ons = np.where(changes == 1)[0] + 1
    offs = np.where(changes == -1)[0] + 1
    if is_singing[0]:
        ons = np.concatenate([[0], ons])
    if is_singing[-1]:
        offs = np.concatenate([offs, [T]])
    n = min(len(ons), len(offs))
    ons, offs = ons[:n], offs[:n]
    for i in range(len(ons) - 1):
        if ons[i + 1] - offs[i] <= cfg.singing_max_gap_frames:
            is_singing[offs[i]:ons[i + 1]] = True

    # Drop short segments.
    changes = np.diff(is_singing.astype(int))
    ons = np.where(changes == 1)[0] + 1
    offs = np.where(changes == -1)[0] + 1
    if is_singing[0]:
        ons = np.concatenate([[0], ons])
    if is_singing[-1]:
        offs = np.concatenate([offs, [T]])
    n = min(len(ons), len(offs))
    ons, offs = ons[:n], offs[:n]
    for on, off in zip(ons, offs):
        if off - on < cfg.singing_min_bout_frames:
            is_singing[on:off] = False

    left_power = (
        float(np.mean(activities["WingL_V12"][is_singing]))
        if is_singing.any() else 0.0
    )
    right_power = (
        float(np.mean(activities["WingR_V12"][is_singing]))
        if is_singing.any() else 0.0
    )
    dominant_wing = "L" if left_power > right_power else "R"
    return is_singing, activities, dominant_wing


# -----------------------------------------------------------------------------
# Per-side FFT classifier
# -----------------------------------------------------------------------------


def _interp_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isnan(x).any():
        return x
    idx = np.arange(len(x))
    good = ~np.isnan(x)
    if good.sum() == 0:
        return np.zeros_like(x)
    return np.interp(idx, idx[good], x[good])


def _band_power(Sxx: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (f >= lo) & (f <= hi)
    return Sxx[mask, :].sum(axis=0)


def _peak_freq(Sxx: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (f >= lo) & (f <= hi)
    if mask.sum() == 0:
        return np.zeros(Sxx.shape[1])
    sub = Sxx[mask, :]
    f_sub = f[mask]
    return f_sub[np.argmax(sub, axis=0)]


def _merge_short_segments(labels: np.ndarray, min_len: int) -> np.ndarray:
    if len(labels) == 0:
        return labels
    out = labels.copy()
    changes = np.where(out[1:] != out[:-1])[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(out)]])
    for s_i, e_i in zip(starts, ends):
        if e_i - s_i >= min_len:
            continue
        left_label = out[s_i - 1] if s_i > 0 else None
        right_label = out[e_i] if e_i < len(out) else None
        if left_label is not None:
            out[s_i:e_i] = left_label
        elif right_label is not None:
            out[s_i:e_i] = right_label
    return out


def _classify_one_side_fft(
    z: np.ndarray, cfg: SongAnalysisConfig
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """FFT-only classifier for one wing's tip Z trace. Returns
    ``(frame_labels, window_features)``. Pulse vs sine is determined by
    the peak frequency inside the song band AND the pulse/song power
    ratio. Waggle is a low-frequency (5-25 Hz) carrier."""
    z = _interp_nan(np.asarray(z))
    n_frames = len(z)
    nperseg_eff = min(cfg.fft_nperseg, n_frames)
    noverlap = max(0, nperseg_eff - cfg.fft_hop)
    if n_frames < 2:
        return (
            np.full(n_frames, "quiet", dtype=object),
            {
                "window_centers": np.array([], dtype=int),
                "song_power": np.array([]),
                "pulse_power": np.array([]),
                "waggle_power": np.array([]),
                "peak_freq": np.array([]),
                "pulse_freq_ratio": np.array([]),
                "pulse_event_frames": np.array([], dtype=int),
            },
        )

    f_axis, t_axis, Sxx = sp_spectrogram(
        z, fs=cfg.fs, nperseg=nperseg_eff, noverlap=noverlap, detrend="linear"
    )
    n_windows = len(t_axis)

    song_p = _band_power(Sxx, f_axis, *cfg.fft_song_band)
    pulse_p = _band_power(Sxx, f_axis, *cfg.fft_pulse_band)
    waggle_p = _band_power(Sxx, f_axis, *cfg.fft_waggle_band)
    peak_f = _peak_freq(Sxx, f_axis, *cfg.fft_song_band)

    with np.errstate(divide="ignore", invalid="ignore"):
        pulse_freq_ratio = np.where(song_p > 0, pulse_p / song_p, 0.0)

    window_labels = np.full(n_windows, "quiet", dtype=object)
    for i in range(n_windows):
        is_singing = song_p[i] >= cfg.song_power_threshold
        is_waggle = (
            waggle_p[i] >= cfg.waggle_power_ratio_min * song_p[i]
            and waggle_p[i] > 0
        )
        if is_waggle and not is_singing:
            window_labels[i] = "waggle"
        elif is_singing:
            if (
                peak_f[i] >= cfg.pulse_peak_freq_min
                and pulse_freq_ratio[i] >= cfg.pulse_freq_ratio_min
            ):
                window_labels[i] = "pulse"
            else:
                window_labels[i] = "sine"

    frame_labels = np.full(n_frames, "quiet", dtype=object)
    window_centers = (t_axis * cfg.fs).astype(int)
    for i in range(n_windows):
        w_start = max(0, window_centers[i] - cfg.fft_hop // 2)
        w_end = min(n_frames, window_centers[i] + cfg.fft_hop // 2)
        frame_labels[w_start:w_end] = window_labels[i]
    if n_windows > 0:
        frame_labels[: max(0, window_centers[0])] = window_labels[0]
        frame_labels[min(n_frames, window_centers[-1]):] = window_labels[-1]

    frame_labels = _merge_short_segments(frame_labels, cfg.min_segment_frames)

    return frame_labels, {
        "window_centers": window_centers,
        "song_power": song_p,
        "pulse_power": pulse_p,
        "waggle_power": waggle_p,
        "peak_freq": peak_f,
        "pulse_freq_ratio": pulse_freq_ratio,
        "pulse_event_frames": np.array([], dtype=int),
    }


def _bilateral_pulse_mask(
    z_L: np.ndarray, z_R: np.ndarray, cfg: SongAnalysisConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Run peak detection on ``max(|dZ/dt|_L, |dZ/dt|_R)``. Returns a
    per-frame mask (with intra-train gaps filled) and the peak frame
    indices themselves."""
    sig_L = np.abs(np.diff(z_L, prepend=z_L[0]) * cfg.fs)
    sig_R = np.abs(np.diff(z_R, prepend=z_R[0]) * cfg.fs)
    dz_max = np.maximum(sig_L, sig_R)

    min_dist = cfg.ms_to_frames(cfg.pulse_detect_min_dist_ms)
    half_w = cfg.ms_to_frames(cfg.pulse_detect_half_width_ms)
    peaks, _ = find_peaks(
        dz_max,
        height=cfg.pulse_detect_height,
        prominence=cfg.pulse_detect_prominence,
        distance=min_dist,
    )

    # Relative peak-to-baseline gate. For a true pulse the inter-pulse
    # |dZ/dt| sits near the noise floor, so peak/baseline is large (≥4).
    # For a strong sine carrier the baseline *is* the signal energy, so
    # peak/baseline collapses toward ~1.5-2.5 and these candidates get
    # dropped here before we paint the pulse mask.
    if len(peaks) and cfg.pulse_baseline_ratio_min > 0:
        w_base = cfg.ms_to_frames(cfg.pulse_baseline_window_ms)
        ex = cfg.ms_to_frames(cfg.pulse_baseline_exclude_ms)
        kept = []
        for p in peaks:
            lo = max(0, p - w_base)
            hi = min(len(dz_max), p + w_base + 1)
            local = dz_max[lo:hi]
            center_lo = max(0, (p - ex) - lo)
            center_hi = min(len(local), (p + ex + 1) - lo)
            local = np.concatenate([local[:center_lo], local[center_hi:]])
            if len(local) == 0:
                kept.append(p)
                continue
            baseline = float(np.median(local))
            ratio = dz_max[p] / max(baseline, 1e-9)
            if ratio >= cfg.pulse_baseline_ratio_min:
                kept.append(p)
        peaks = np.asarray(kept, dtype=int)

    mask = np.zeros(len(dz_max), dtype=bool)
    for pk in peaks:
        mask[max(0, pk - half_w):min(len(dz_max), pk + half_w + 1)] = True

    # Bridge gaps between consecutive pulses in a single train.
    max_gap = cfg.ms_to_frames(cfg.pulse_train_max_gap_ms)
    for a, b in zip(peaks[:-1], peaks[1:]):
        if (b - a) <= max_gap:
            mask[a:b + 1] = True

    return mask, peaks


# -----------------------------------------------------------------------------
# Per-pulse feature extraction (Clemens et al. 2018, adapted to 800 Hz)
# -----------------------------------------------------------------------------


def extract_pulse_waveforms(
    wing_z: np.ndarray,
    peak_frames: np.ndarray,
    cfg: SongAnalysisConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice z-scored, sign-aligned pulse windows around each peak.

    Parameters
    ----------
    wing_z : (T,) float
        Tip Z trace (the same one the detector ran on). NaNs are
        interpolated with :func:`_interp_nan` first so a single missing
        frame next to a pulse does not drop the window.
    peak_frames : (N,) int
        Peak indices from ``window_features['pulse_event_frames']``.
    cfg : SongAnalysisConfig
        Supplies ``fs`` and ``pulse_window_ms``.

    Returns
    -------
    waveforms : (M, W) float32
        One row per surviving peak. Each row is linearly detrended,
        z-scored to unit std, and sign-flipped so its center sample is
        non-negative (Clemens alignment convention). ``W =
        2 * ms_to_frames(pulse_window_ms / 2) + 1``.
    kept_frames : (M,) int
        Subset of ``peak_frames`` whose full ±half_w window fits inside
        the bout. Callers that need to re-index other per-peak arrays
        should use this.
    """
    peak_frames = np.asarray(peak_frames, dtype=int)
    half_w = cfg.ms_to_frames(cfg.pulse_window_ms / 2.0)
    W = 2 * half_w + 1
    if peak_frames.size == 0:
        return np.zeros((0, W), dtype=np.float32), peak_frames

    z = _interp_nan(np.asarray(wing_z, dtype=float))
    T = len(z)

    inside = (peak_frames - half_w >= 0) & (peak_frames + half_w + 1 <= T)
    kept_frames = peak_frames[inside]
    if kept_frames.size == 0:
        return np.zeros((0, W), dtype=np.float32), kept_frames

    # Gather windows in one shot.
    offsets = np.arange(-half_w, half_w + 1)
    idx = kept_frames[:, None] + offsets[None, :]
    waves = z[idx].astype(np.float32)

    # Per-window detrend (remove linear baseline over the window).
    x = np.arange(W, dtype=np.float32)
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = float((x_centered * x_centered).sum()) + 1e-12
    slope = (waves * x_centered).sum(axis=1, keepdims=True) / denom
    intercept = waves.mean(axis=1, keepdims=True) - slope * x_mean
    waves = waves - (slope * x + intercept)

    # z-score (per window).
    std = waves.std(axis=1, keepdims=True)
    std = np.where(std < 1e-9, 1.0, std)
    waves = waves / std

    # Sign-align: make the center sample non-negative.
    center = waves[:, half_w]
    flip = np.where(center < 0, -1.0, 1.0).astype(np.float32)
    waves = waves * flip[:, None]

    return waves, kept_frames


def compute_pulse_symmetry(waveforms: np.ndarray) -> np.ndarray:
    """Per-pulse symmetry index s (Clemens 2018 methods).

        s = (a · flip(b)) / (||a|| ||b||)

    where ``a`` is the first half and ``b`` the second half of the
    aligned window. Pslow ≈ +0.3, Pfast ≈ −0.2. Returns (N,) float.
    """
    waves = np.asarray(waveforms, dtype=float)
    if waves.ndim != 2 or waves.shape[0] == 0:
        return np.zeros(0, dtype=float)
    W = waves.shape[1]
    half = W // 2
    a = waves[:, :half]
    # skip the center sample when W is odd so |a| == |b|
    b_start = W - half
    b = waves[:, b_start:]
    b_flip = b[:, ::-1]
    num = (a * b_flip).sum(axis=1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    out = np.zeros_like(num)
    good = denom > 1e-12
    out[good] = num[good] / denom[good]
    return out


def compute_pulse_carrier_freq(
    waveforms: np.ndarray,
    fs: float,
    thresh: float = 0.3679,
) -> np.ndarray:
    """Per-pulse spectral center-of-mass frequency (Hz).

    For each aligned pulse window, compute the rFFT magnitude, threshold
    at ``thresh * max(|X(f)|)``, and return the energy-weighted mean
    frequency across the surviving bins. Falls back to argmax on any
    degenerate spectrum. Returns (N,) float.
    """
    waves = np.asarray(waveforms, dtype=float)
    if waves.ndim != 2 or waves.shape[0] == 0:
        return np.zeros(0, dtype=float)
    N, W = waves.shape
    freqs = np.fft.rfftfreq(W, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(waves, axis=1))
    peak_mag = mag.max(axis=1, keepdims=True)
    mask = mag >= (thresh * peak_mag)
    weighted = mag * mask
    wsum = weighted.sum(axis=1)
    out = np.zeros(N, dtype=float)
    good = wsum > 1e-12
    out[good] = (weighted[good] * freqs[None, :]).sum(axis=1) / wsum[good]
    # Fallback to argmax for any degenerate rows.
    bad = ~good
    if bad.any():
        out[bad] = freqs[mag[bad].argmax(axis=1)]
    return out


def compute_pulse_wing_angle(
    angle_dominant: Optional[np.ndarray],
    peak_frames: np.ndarray,
    cfg: SongAnalysisConfig,
) -> Optional[np.ndarray]:
    """Mean extended-wing angle (deg) over ± ``pulse_window_ms``/2 around
    each peak. Returns (N,) float or ``None`` if ``angle_dominant`` is
    ``None`` (e.g. bout without egocentric positions)."""
    if angle_dominant is None:
        return None
    peak_frames = np.asarray(peak_frames, dtype=int)
    if peak_frames.size == 0:
        return np.zeros(0, dtype=float)
    ang = np.asarray(angle_dominant, dtype=float)
    T = len(ang)
    half_w = cfg.ms_to_frames(cfg.pulse_window_ms / 2.0)
    out = np.full(peak_frames.shape, np.nan, dtype=float)
    for i, p in enumerate(peak_frames):
        lo = max(0, p - half_w)
        hi = min(T, p + half_w + 1)
        seg = ang[lo:hi]
        if seg.size:
            out[i] = float(np.nanmean(np.abs(seg)))
    return out


def classify_song_both_sides(
    wing_data: Dict[str, Dict[str, np.ndarray]],
    cfg: SongAnalysisConfig,
) -> Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Run the FFT classifier on L and R wings independently, then override
    the pulse label with the bilateral peak-based pulse mask (same mask
    applied to both sides, matching the sandbox behaviour).
    """
    z_L = _interp_nan(np.asarray(wing_data[cfg.left_tip]["z"]))
    z_R = _interp_nan(np.asarray(wing_data[cfg.right_tip]["z"]))

    labels_L, feats_L = _classify_one_side_fft(z_L, cfg)
    labels_R, feats_R = _classify_one_side_fft(z_R, cfg)

    if cfg.use_peak_pulse and len(z_L) > 1:
        mask, peaks = _bilateral_pulse_mask(z_L, z_R, cfg)
        # FFT-only pulse calls are noisy for sparse pulses; let the peak
        # detector be authoritative (demotes previous pulse → sine, then
        # overwrites with the bilateral mask).
        labels_L = labels_L.copy()
        labels_R = labels_R.copy()
        labels_L[labels_L == "pulse"] = "sine"
        labels_R[labels_R == "pulse"] = "sine"
        labels_L[mask] = "pulse"
        labels_R[mask] = "pulse"
        feats_L["pulse_event_frames"] = peaks
        feats_R["pulse_event_frames"] = peaks

    return {"L": (labels_L, feats_L), "R": (labels_R, feats_R)}


# -----------------------------------------------------------------------------
# Segmentation + summaries
# -----------------------------------------------------------------------------


def segments_from_labels(
    frame_labels: np.ndarray, fs: float
) -> List[Dict]:
    """Run-length encode per-frame labels into a list of segment dicts."""
    T = len(frame_labels)
    out: List[Dict] = []
    if T == 0:
        return out
    current = frame_labels[0]
    seg_start = 0
    for i in range(1, T):
        if frame_labels[i] != current:
            out.append({
                "start": seg_start,
                "end": i - 1,
                "type": current,
                "n_frames": i - seg_start,
                "duration_s": (i - seg_start) / fs,
            })
            current = frame_labels[i]
            seg_start = i
    out.append({
        "start": seg_start,
        "end": T - 1,
        "type": current,
        "n_frames": T - seg_start,
        "duration_s": (T - seg_start) / fs,
    })
    return out


def _side_summary(
    frame_labels: np.ndarray, fs: float, valid_mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    T = len(frame_labels)
    if valid_mask is None:
        valid_mask = np.ones(T, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
    eff = int(valid_mask.sum())
    if eff == 0:
        return {
            "n_frames": 0, "duration_s": 0.0,
            "frac_pulse": 0.0, "frac_sine": 0.0, "frac_waggle": 0.0,
            "frac_quiet": 0.0, "song_fraction": 0.0,
        }
    lbl = frame_labels[valid_mask]
    n_p = int((lbl == "pulse").sum())
    n_s = int((lbl == "sine").sum())
    n_w = int((lbl == "waggle").sum())
    n_q = int((lbl == "quiet").sum())
    return {
        "n_frames": eff,
        "duration_s": eff / fs,
        "frac_pulse": n_p / eff,
        "frac_sine": n_s / eff,
        "frac_waggle": n_w / eff,
        "frac_quiet": n_q / eff,
        "song_fraction": (n_p + n_s + n_w) / eff,
    }


# -----------------------------------------------------------------------------
# Public entry point: one fly, one bout
# -----------------------------------------------------------------------------


def analyze_fly_song(
    kp_world: np.ndarray,
    xpos_ego: Optional[np.ndarray],
    qpos: Optional[np.ndarray],
    kp_names: Sequence[str],
    cfg: Optional[SongAnalysisConfig] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict:
    """Full per-fly per-bout song analysis.

    Parameters
    ----------
    kp_world : (T, N, 3) or (T, N*3) float array
        RAW world-frame keypoints — used for the oscillation signal.
    xpos_ego : (T, N, 3) or None
        Egocentric positions — used only for wing extension angles
        (kinematic feature, not for classification). Pass ``None`` to
        skip.
    qpos : (T, D) or None
        Joint angles — used only for extracting wing DOFs for plotting.
        Pass ``None`` to skip.
    kp_names : sequence of keypoint name strings matching the N axis.
    cfg : SongAnalysisConfig or None
        Detector configuration. Defaults reproduce the sandbox
        behaviour.
    valid_mask : (T,) bool or None
        If provided, per-frame validity mask (e.g. ``valid_fly0 &
        ~pair_colocated``). Summaries are computed over valid frames
        only, but frame_labels cover all T frames so the caller can
        still plot them.

    Returns a dict with:
        ``wing_data``          — per-tip xyz traces
        ``sides``              — {'L': {frame_labels, segments, summary,
                                          window_features,
                                          pulse_features},
                                  'R': {...}}
        ``dominant_wing``      — 'L' or 'R'
        ``is_singing``         — (T,) bool gate
        ``activities``         — per-tip windowed |dZ/dt|
        ``angle_L, angle_R``   — (T,) wing extension angles (deg) if
                                  xpos_ego was provided
        ``joints``             — dict of wing DOFs if qpos was provided
        ``summary``            — dominant-wing song_fraction + bout
                                  metadata (bout length, fs)
    """
    if cfg is None:
        cfg = SongAnalysisConfig()
    kp_idx = resolve_kp_indices(kp_names)

    kp_world = np.asarray(kp_world)
    if kp_world.ndim == 2:
        kp_world = kp_world.reshape(kp_world.shape[0], -1, 3)
    T = kp_world.shape[0]

    wing_data = compute_wing_tip_signal(kp_world, kp_idx)
    is_singing, activities, dominant_wing = detect_singing_frames(wing_data, cfg)

    # FFT + peak classification -------------------------------------------
    side_results = classify_song_both_sides(wing_data, cfg)
    sides: Dict[str, Dict] = {}
    for sname, (frame_labels, features) in side_results.items():
        sides[sname] = {
            "frame_labels": frame_labels,
            "window_features": features,
            "segments": segments_from_labels(frame_labels, cfg.fs),
            "summary": _side_summary(frame_labels, cfg.fs, valid_mask),
        }

    # Wing extension angles (geometric feature, for plotting) -------------
    angle_L = angle_R = None
    if xpos_ego is not None:
        xpe = np.asarray(xpos_ego)
        if xpe.ndim == 2:
            xpe = xpe.reshape(xpe.shape[0], -1, 3)
        angle_L, angle_R = compute_wing_extension_angles(xpe, kp_idx)

    # Per-pulse features (Clemens 2018, adapted) --------------------------
    # The bilateral peak detector stores the same peak frames under both
    # 'L' and 'R' sides; we still compute waveform-based features per side
    # because the tip-Z oscillation shape differs between wings.
    side_tip = {"L": cfg.left_tip, "R": cfg.right_tip}
    side_angle = {"L": angle_L, "R": angle_R}
    for sname in ("L", "R"):
        feats = sides[sname]["window_features"]
        peak_frames = np.asarray(
            feats.get("pulse_event_frames", np.array([], dtype=int)),
            dtype=int,
        )
        wing_z = wing_data[side_tip[sname]]["z"]
        waveforms, kept = extract_pulse_waveforms(wing_z, peak_frames, cfg)
        symmetry = compute_pulse_symmetry(waveforms)
        carrier_hz = compute_pulse_carrier_freq(
            waveforms, cfg.fs, thresh=cfg.pulse_spectral_thresh
        )
        wing_angle = compute_pulse_wing_angle(side_angle[sname], kept, cfg)
        ipi_ms = (
            np.diff(kept).astype(float) * 1000.0 / cfg.fs
            if kept.size > 1 else np.array([], dtype=float)
        )
        sides[sname]["pulse_features"] = {
            "peak_frames": kept,
            "waveforms": waveforms,
            "symmetry": symmetry,
            "carrier_hz": carrier_hz,
            "wing_angle": wing_angle,
            "ipi_ms": ipi_ms,
        }

    joints = None
    if qpos is not None:
        qp = np.asarray(qpos)
        if qp.shape[1] > max(cfg.qpos_wing_pitch_R, 0):
            joints = extract_wing_joint_angles(qp, cfg)

    dom_summary = sides[dominant_wing]["summary"]

    return {
        "wing_data": wing_data,
        "sides": sides,
        "dominant_wing": dominant_wing,
        "is_singing": is_singing,
        "activities": activities,
        "angle_L": angle_L,
        "angle_R": angle_R,
        "joints": joints,
        "summary": {
            "n_frames": T,
            "duration_s": T / cfg.fs,
            "dominant_wing": dominant_wing,
            "song_fraction": dom_summary["song_fraction"],
            "frac_pulse": dom_summary["frac_pulse"],
            "frac_sine": dom_summary["frac_sine"],
            "frac_waggle": dom_summary["frac_waggle"],
            "frac_quiet": dom_summary["frac_quiet"],
            "valid_n_frames": dom_summary["n_frames"],
        },
    }
