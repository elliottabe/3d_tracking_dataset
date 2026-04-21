"""Courtship song detection and classification.

Ported from the exploratory ``notebooks/Courtship_Song_Analysis.ipynb``
sandbox (cells 3, 18, 20, 21, 23, 25) into a reusable module so the
paper-figure notebook can stay thin. The two are kept behaviourally
equivalent; thresholds match the sandbox defaults except where noted.

The primary pipeline (default) is the FFT-spectrogram classifier paired
with a derivative-peak pulse detector. Per fly per bout:

1. Build a wing-tip signal dict from the RAW world-frame keypoints
   ``kp_data`` (NOT the IK-reconstructed egocentric positions — see
   comment in ``compute_wing_tip_signal``).
2. Detect frames where any wing pair is oscillating above a
   |dZ/dt| activity threshold → ``is_singing`` mask + dominant wing.
3. For each wing independently, run an FFT classifier on the tip Z
   trace: compute a spectrogram, threshold the song-band power, and
   label windows as ``sine`` or ``pulse`` based on the peak frequency
   in the song band and the pulse/song power ratio.
4. Override the pulse label from step 3 with a bilateral peak detector
   on ``max(|dZ/dt|_L, |dZ/dt|_R)`` — this catches sparse pulses the
   FFT window smooths out and gives a clean per-pulse event list for
   waveform feature extraction. Any peak whose local |dZ/dt| baseline
   is too high (sine-carrier territory) is rejected before painting
   the pulse mask.
5. Merge per-side results into a bout summary with ``song_fraction``
   (pulse + sine frames / total frames) for downstream male / female
   identification.

A secondary pipeline (Butterworth bandpass + Hilbert envelope for
pulse, Thomson multitaper F-test for sine — closer to the paper text)
runs in parallel and is exposed under ``*_new`` / ``frame_labels_new``
keys for comparison. It is not the default output because on hand-
validated courtship bouts it under-detects sustained sine and leaks
on harmonic-rich bouts; the FFT + peak detector is more robust in
practice.

The detector does not emit a separate ``waggle`` label — frames that
would otherwise have been labelled waggle are treated as ``quiet``.

No dependencies beyond numpy / scipy. This module does not touch h5 or
configuration files; the notebook does that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import (
    butter,
    filtfilt,
    find_peaks,
    hilbert,
    spectrogram as sp_spectrogram,
)
from scipy.signal.windows import dpss
from scipy.stats import f as _f_dist


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

    # ------------------------------------------------------------------ #
    # New pulse detector (Butterworth bandpass + Hilbert envelope, per paper)
    # ------------------------------------------------------------------ #
    pulse_band_hz: Tuple[float, float] = (200.0, 380.0)
    pulse_filter_order: int = 4
    pulse_envelope_lp_hz: float = 25.0
    pulse_noise_floor_percentile: float = 50.0  # median of bandpassed |x|
    pulse_threshold_mult: float = 3.0           # T = mult * noise_floor.
                                                #  Lowered from 4.0 after
                                                #  hand-validation on bout 37:
                                                #  a 12-pulse train at 32 ms
                                                #  IPI had envelope peaks
                                                #  right at 4×median of the
                                                #  bandpassed signal, so
                                                #  only 4/12 survived. 3× is
                                                #  a stable noise floor the
                                                #  envelope rarely touches
                                                #  outside real pulses.
    pulse_min_dist_ms: float = 15.0
    pulse_dedupe_window_ms: float = 10.0        # keep-largest window
    pulse_isolation_window_ms: float = 120.0    # drop if no neighbor within
    pulse_train_max_median_ipi_ms: float = 80.0  # drop trains above this
    pulse_train_min_n: int = 2
    pulse_mask_half_width_ms: float = 12.0
    # Spectral-ratio gate: reject candidate peaks where the local
    # pulse-band (200-380 Hz) envelope is weaker than this fraction of the
    # local sine-band (80-200 Hz) envelope. Kills false pulses produced by
    # sine-harmonic leakage on all-sine bouts where the noise-floor
    # estimate collapses. 0.5 preserves real pulses (bout 37: 12/12) while
    # eliminating sine-leakage false pulses (bout 61: 7→0).
    # Set to 0 to disable.
    pulse_spectral_ratio_min: float = 0.5

    # ------------------------------------------------------------------ #
    # New sine detector (multitaper F-test, per paper)
    # ------------------------------------------------------------------ #
    sine_band_hz: Tuple[float, float] = (80.0, 200.0)
    sine_filter_order: int = 4
    sine_window_ms: float = 100.0               # → 80 samples @ 800 Hz
    sine_hop_ms: float = 10.0                   # → 8 samples
    sine_test_band_hz: Tuple[float, float] = (90.0, 175.0)
    sine_dpss_NW: float = 3.0
    sine_dpss_K: int = 5                        # ≈ 2*NW - 1
    sine_f_test_p: float = 0.05                 # Raw per-window F-test p
                                                #  threshold (no Bonferroni
                                                #  correction). Chaining + the
                                                #  80 ms min-segment gate
                                                #  already require 8+
                                                #  consecutive freq-consistent
                                                #  windows, which provides
                                                #  strong segment-level
                                                #  control. Bonferroni on top
                                                #  of that was over-conservative
                                                #  because courtship sine is
                                                #  modulated (F~6-10, not
                                                #  pure-line-huge) — applying
                                                #  the correction dropped ~2/3
                                                #  of real sine windows on
                                                #  bouts 23/27.
    sine_noise_floor_percentile: float = 25.0   # per-window power gate uses
                                                #  this percentile of window
                                                #  band-power as floor
    sine_noise_floor_mult: float = 0.5          # gate = mult * pct(power).
                                                #  0.5 * pct25 only drops the
                                                #  lowest-energy windows; 1.0
                                                #  * median drops half.
    sine_freq_tolerance: float = 0.20           # ±20% for chaining
    sine_min_segment_ms: float = 80.0
    sine_pulse_overlap_max: float = 0.5         # reject windows with more
                                                #  than this fraction inside
                                                #  the pulse mask

    # ------------------------------------------------------------------ #
    # Legacy FFT/peak detector (kept for parallel comparison output)
    # ------------------------------------------------------------------ #
    # Spectrogram windowing (legacy only) -----------------------------------
    fft_nperseg: int = 48  # ~60 ms, freq res ~16.7 Hz
    fft_hop: int = 8       # ~10 ms
    fft_song_band: Tuple[float, float] = (80.0, 400.0)
    fft_sine_band: Tuple[float, float] = (100.0, 180.0)
    fft_pulse_band: Tuple[float, float] = (140.0, 400.0)

    # FFT classifier thresholds ---------------------------------------------
    song_power_threshold: float = 2e-8
    pulse_peak_freq_min: float = 140.0
    pulse_freq_ratio_min: float = 0.35
    min_segment_frames: int = 16

    # Legacy peak-based pulse detection (bilateral max |dZ/dt|) -------------
    use_peak_pulse: bool = True
    pulse_detect_height: float = 25.0       # min |dZ/dt| (data-units / s)
    pulse_detect_prominence: float = 15.0
    pulse_detect_min_dist_ms: float = 15.0
    pulse_detect_half_width_ms: float = 12.0
    # Max gap between consecutive detected peaks that still counts as the
    # same pulse train. Normal D. melanogaster inter-pulse intervals are
    # ~35 ms; a single missed peak doubles that to ~70 ms, two missed peaks
    # gives ~105 ms, three gives ~140 ms. 135 ms tolerates up to 3 dropped
    # peaks while staying below the shortest observed sine-straddling gap
    # (148.8 ms in pair bout 23 — last pulse before and first pulse after a
    # sine passage). True inter-train silences are typically >250 ms, so
    # 135 ms stays well below that floor.
    pulse_train_max_gap_ms: float = 135.0
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


def compute_wing_horizontal_angles(
    xpos_ego: np.ndarray,
    kp_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Wing horizontal-plane angle (deg) from egocentric keypoints.

    Builds a body frame from Scutellum→Antenna_Base (fore-aft) and the
    left/right wing bases (lateral), projects each wing vector into the
    horizontal plane (perpendicular to the derived dorsal axis), and
    returns the unsigned angle between the projection and the body
    fore-aft line. 0° ≈ wing parallel to body axis (rest); 90° ≈ wing
    perpendicular to the body axis (fully extended).
    """
    scut = kp_idx["Scutellum"]
    ant = kp_idx["Antenna_Base"]
    wl_b = kp_idx["WingL_base"]
    wr_b = kp_idx["WingR_base"]
    wl12 = kp_idx["WingL_V12"]
    wl13 = kp_idx["WingL_V13"]
    wr12 = kp_idx["WingR_V12"]
    wr13 = kp_idx["WingR_V13"]

    fwd = xpos_ego[:, ant] - xpos_ego[:, scut]
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-12

    lat_raw = xpos_ego[:, wr_b] - xpos_ego[:, wl_b]
    dorsal = np.cross(fwd, lat_raw)
    dorsal /= np.linalg.norm(dorsal, axis=1, keepdims=True) + 1e-12
    lat = np.cross(dorsal, fwd)
    lat /= np.linalg.norm(lat, axis=1, keepdims=True) + 1e-12

    def _angle(t12: int, t13: int, base: int) -> np.ndarray:
        tip = 0.5 * (xpos_ego[:, t12] + xpos_ego[:, t13])
        vec = tip - xpos_ego[:, base]
        vec_h = vec - np.sum(vec * dorsal, axis=1, keepdims=True) * dorsal
        x = np.sum(vec_h * fwd, axis=1)
        y = np.sum(vec_h * lat, axis=1)
        return np.degrees(np.arctan2(np.abs(y), np.abs(x)))

    return _angle(wl12, wl13, wl_b), _angle(wr12, wr13, wr_b)


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


# -----------------------------------------------------------------------------
# Paper-accurate detectors (Butterworth + Hilbert pulse, multitaper sine)
# -----------------------------------------------------------------------------
#
# These replace the original FlySongSegmenter (Arthur et al. 2013) pipeline,
# adapted to kinematic wing-tip Z traces at 800 Hz. The two detectors run in
# parallel to the legacy ones and their outputs are surfaced under the primary
# summary/label keys; the legacy outputs go to ``*_legacy`` keys.


def _butter_bandpass(
    low_hz: float, high_hz: float, fs: float, order: int
) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    return butter(order, [low, high], btype="bandpass")


def _butter_lowpass(
    cutoff_hz: float, fs: float, order: int
) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    wn = min(cutoff_hz / nyq, 0.999)
    return butter(order, wn, btype="lowpass")


def _bilateral_pulse_mask_butterworth(
    z_L: np.ndarray, z_R: np.ndarray, cfg: SongAnalysisConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Paper-accurate pulse detector.

    Pipeline (adapted from FlySongSegmenter):

    1. 4th-order zero-phase Butterworth bandpass at ``pulse_band_hz``
       applied to each tip Z trace.
    2. Hilbert amplitude envelope per side, then bilateral ``max``.
    3. Zero-phase 4th-order Butterworth low-pass at ``pulse_envelope_lp_hz``
       to smooth the envelope.
    4. Noise floor from the bandpassed bilateral signal's
       ``pulse_noise_floor_percentile`` value; threshold
       ``T = pulse_threshold_mult * noise_floor``.
    5. ``find_peaks`` on the smoothed envelope with ``height=T`` and
       minimum inter-peak distance ``pulse_min_dist_ms``.
    6. Dedupe: within ``pulse_dedupe_window_ms`` keep only the
       largest-amplitude peak.
    7. Isolation gate: drop peaks with no neighbor within
       ``pulse_isolation_window_ms``.
    8. Per-train IPI gate: group peaks into trains; drop trains whose
       median IPI exceeds ``pulse_train_max_median_ipi_ms`` or that
       contain fewer than ``pulse_train_min_n`` peaks.
    9. Paint a per-frame mask ``±pulse_mask_half_width_ms`` around each
       surviving peak and fill intra-train gaps.

    Returns ``(mask, kept_peak_frames)``.
    """
    z_L = _interp_nan(np.asarray(z_L, dtype=float))
    z_R = _interp_nan(np.asarray(z_R, dtype=float))
    n = len(z_L)
    if n < 16:
        return np.zeros(n, dtype=bool), np.array([], dtype=int)

    b_bp, a_bp = _butter_bandpass(
        cfg.pulse_band_hz[0],
        cfg.pulse_band_hz[1],
        cfg.fs,
        cfg.pulse_filter_order,
    )
    b_lp, a_lp = _butter_lowpass(
        cfg.pulse_envelope_lp_hz, cfg.fs, cfg.pulse_filter_order
    )

    bp_L = filtfilt(b_bp, a_bp, z_L)
    bp_R = filtfilt(b_bp, a_bp, z_R)
    env_L = np.abs(hilbert(bp_L))
    env_R = np.abs(hilbert(bp_R))
    env = np.maximum(env_L, env_R)
    env_lp = filtfilt(b_lp, a_lp, env)

    # Sine-band envelope (80-200 Hz) for the spectral-ratio gate below.
    # Computed alongside the pulse-band envelope so we can compare the two
    # at each candidate peak and reject sine-harmonic leakage.
    if cfg.pulse_spectral_ratio_min > 0:
        b_bp_s, a_bp_s = _butter_bandpass(
            cfg.sine_band_hz[0],
            cfg.sine_band_hz[1],
            cfg.fs,
            cfg.pulse_filter_order,
        )
        bp_L_s = filtfilt(b_bp_s, a_bp_s, z_L)
        bp_R_s = filtfilt(b_bp_s, a_bp_s, z_R)
        env_sine = np.maximum(np.abs(hilbert(bp_L_s)), np.abs(hilbert(bp_R_s)))
    else:
        env_sine = None

    bp_max = np.maximum(np.abs(bp_L), np.abs(bp_R))
    noise_floor = float(
        np.percentile(bp_max, cfg.pulse_noise_floor_percentile)
    )
    if not np.isfinite(noise_floor) or noise_floor <= 0:
        noise_floor = float(np.median(np.abs(bp_max))) + 1e-12
    threshold = cfg.pulse_threshold_mult * noise_floor

    min_dist = cfg.ms_to_frames(cfg.pulse_min_dist_ms)
    peaks, _ = find_peaks(env_lp, height=threshold, distance=min_dist)

    # Dedupe within a 10 ms window (keep the largest-amplitude peak).
    if len(peaks):
        dedupe_w = max(1, cfg.ms_to_frames(cfg.pulse_dedupe_window_ms))
        amps = env_lp[peaks]
        keep_mask = np.ones(len(peaks), dtype=bool)
        order = np.argsort(-amps)  # descending amplitude
        occupied = np.zeros(n, dtype=bool)
        for rank in order:
            p = peaks[rank]
            lo = max(0, p - dedupe_w // 2)
            hi = min(n, p + dedupe_w // 2 + 1)
            if occupied[lo:hi].any():
                keep_mask[rank] = False
            else:
                occupied[lo:hi] = True
        peaks = peaks[keep_mask]
        peaks.sort()

    # Spectral-ratio gate: reject peaks whose local pulse-band envelope is
    # weaker than ``pulse_spectral_ratio_min`` × local sine-band envelope.
    # Real pulses have a sharp broadband transient that dominates the
    # 200-380 Hz band; sine-harmonic leakage into the pulse band is always
    # accompanied by much larger 80-200 Hz energy, so the ratio < 1.
    if len(peaks) and env_sine is not None:
        half_w = max(1, cfg.ms_to_frames(cfg.pulse_mask_half_width_ms))
        kept = []
        for p in peaks:
            lo = max(0, p - half_w)
            hi = min(n, p + half_w + 1)
            env_p_local = float(env[lo:hi].mean())
            env_s_local = float(env_sine[lo:hi].mean())
            if env_s_local <= 0:
                kept.append(p)
                continue
            if env_p_local >= cfg.pulse_spectral_ratio_min * env_s_local:
                kept.append(p)
        peaks = np.asarray(kept, dtype=int)

    # Isolation gate: drop peaks with no neighbor within 120 ms.
    if len(peaks) > 1:
        iso_w = cfg.ms_to_frames(cfg.pulse_isolation_window_ms)
        diffs = np.diff(peaks)
        near_prev = np.concatenate([[np.inf], diffs]) <= iso_w
        near_next = np.concatenate([diffs, [np.inf]]) <= iso_w
        peaks = peaks[near_prev | near_next]
    elif len(peaks) == 1:
        peaks = np.array([], dtype=int)  # a lone peak is never a train

    # Group into trains (break on gap > isolation window) and apply the
    # median-IPI gate.
    kept: List[int] = []
    if len(peaks):
        iso_w = cfg.ms_to_frames(cfg.pulse_isolation_window_ms)
        max_median_ipi = cfg.ms_to_frames(cfg.pulse_train_max_median_ipi_ms)
        train_start = 0
        for i in range(1, len(peaks) + 1):
            is_last = i == len(peaks)
            if is_last or (peaks[i] - peaks[i - 1]) > iso_w:
                train = peaks[train_start:i]
                if len(train) >= cfg.pulse_train_min_n:
                    median_ipi = float(np.median(np.diff(train)))
                    if median_ipi <= max_median_ipi:
                        kept.extend(int(p) for p in train)
                train_start = i
    peaks = np.asarray(kept, dtype=int)

    mask = np.zeros(n, dtype=bool)
    if len(peaks):
        half_w = cfg.ms_to_frames(cfg.pulse_mask_half_width_ms)
        for p in peaks:
            mask[max(0, p - half_w):min(n, p + half_w + 1)] = True
        # Fill intra-train gaps so the mask is contiguous within a train.
        iso_w = cfg.ms_to_frames(cfg.pulse_isolation_window_ms)
        for a, b in zip(peaks[:-1], peaks[1:]):
            if (b - a) <= iso_w:
                mask[a:b + 1] = True

    return mask, peaks


def _thomson_f_test(
    x: np.ndarray, tapers: np.ndarray, K: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thomson (1982) line-component F-test.

    Parameters
    ----------
    x : (N,) signal.
    tapers : (K, N) DPSS tapers (unit-energy).
    K : number of tapers.

    Returns
    -------
    freqs     : (N//2 + 1,) rFFT frequency bins, unit fraction of ``fs``.
    F_stat    : (len(freqs),) F-statistic per frequency bin.
    amp       : (len(freqs),) complex line-component amplitude estimate.
    """
    N = len(x)
    # X_k(f): rFFT of tapered signal for each taper.
    tapered = tapers * x[None, :]                   # (K, N)
    Xk = np.fft.rfft(tapered, axis=1)               # (K, Nf)
    # DC sum of each taper: only even-symmetric tapers (k=0,2,4,...) have
    # non-zero DC, odd tapers cancel out — but we include all to stay
    # general and they contribute ~0.
    U = tapers.sum(axis=1)                          # (K,)
    U_sq_sum = float((U * U).sum())
    if U_sq_sum <= 0:
        nf = Xk.shape[1]
        return (
            np.fft.rfftfreq(N, d=1.0),
            np.zeros(nf),
            np.zeros(nf, dtype=complex),
        )
    # μ(f) = Σ_k X_k(f) · U_k / Σ_k U_k²
    mu = (Xk * U[:, None]).sum(axis=0) / U_sq_sum   # (Nf,)
    # Residuals: X_k(f) - μ(f) · U_k
    resid = Xk - mu[None, :] * U[:, None]           # (K, Nf)
    resid_sq = (np.abs(resid) ** 2).sum(axis=0)     # (Nf,)
    num = (K - 1) * (np.abs(mu) ** 2) * U_sq_sum
    denom = np.where(resid_sq > 0, resid_sq, np.inf)
    F_stat = num / denom
    freqs = np.fft.rfftfreq(N, d=1.0)
    return freqs, F_stat, mu


def _classify_sine_multitaper(
    z: np.ndarray,
    pulse_mask: np.ndarray,
    cfg: SongAnalysisConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Paper-accurate sine detector (Thomson multitaper F-test).

    Returns ``(frame_labels, window_features)`` where frame_labels contain
    only ``{"pulse", "sine", "quiet"}`` (pulse frames come from the
    provided ``pulse_mask``).
    """
    z = _interp_nan(np.asarray(z, dtype=float))
    n = len(z)
    pulse_mask = np.asarray(pulse_mask, dtype=bool)
    if pulse_mask.shape[0] != n:
        pulse_mask = np.zeros(n, dtype=bool)

    empty_features = {
        "window_centers": np.array([], dtype=int),
        "window_freq": np.array([], dtype=float),
        "window_F": np.array([], dtype=float),
        "window_p": np.array([], dtype=float),
        "window_is_sine": np.array([], dtype=bool),
        "sine_segments": np.zeros((0, 2), dtype=int),
        "pulse_event_frames": np.array([], dtype=int),
    }

    if n < 32:
        frame_labels = np.full(n, "quiet", dtype=object)
        frame_labels[pulse_mask] = "pulse"
        return frame_labels, empty_features

    # Pulse pre-mask: replace pulse samples with linear interp so the
    # bandpass filter isn't driven by pulse energy.
    z_masked = z.copy()
    if pulse_mask.any() and not pulse_mask.all():
        idx_all = np.arange(n)
        good = ~pulse_mask
        z_masked[pulse_mask] = np.interp(
            idx_all[pulse_mask], idx_all[good], z[good]
        )

    b_bp, a_bp = _butter_bandpass(
        cfg.sine_band_hz[0],
        cfg.sine_band_hz[1],
        cfg.fs,
        cfg.sine_filter_order,
    )
    z_bp = filtfilt(b_bp, a_bp, z_masked)

    W = cfg.ms_to_frames(cfg.sine_window_ms)
    hop = max(1, cfg.ms_to_frames(cfg.sine_hop_ms))
    if W < 8 or W > n:
        frame_labels = np.full(n, "quiet", dtype=object)
        frame_labels[pulse_mask] = "pulse"
        return frame_labels, empty_features

    NW = cfg.sine_dpss_NW
    K = max(2, int(cfg.sine_dpss_K))
    tapers = dpss(W, NW, Kmax=K)  # (K, W), unit-energy
    p_threshold = cfg.sine_f_test_p
    dof2 = 2 * K - 2

    starts = np.arange(0, n - W + 1, hop, dtype=int)
    n_win = len(starts)
    if n_win == 0:
        frame_labels = np.full(n, "quiet", dtype=object)
        frame_labels[pulse_mask] = "pulse"
        return frame_labels, empty_features

    freq_fraction = np.fft.rfftfreq(W, d=1.0)
    freq_hz = freq_fraction * cfg.fs
    test_lo, test_hi = cfg.sine_test_band_hz
    test_band = (freq_hz >= test_lo) & (freq_hz <= test_hi)
    if not test_band.any():
        frame_labels = np.full(n, "quiet", dtype=object)
        frame_labels[pulse_mask] = "pulse"
        return frame_labels, empty_features

    window_centers = starts + W // 2
    window_freq = np.zeros(n_win, dtype=float)
    window_F = np.zeros(n_win, dtype=float)
    window_p = np.ones(n_win, dtype=float)
    window_pulse_frac = np.zeros(n_win, dtype=float)
    window_power = np.zeros(n_win, dtype=float)

    for i, s in enumerate(starts):
        w_mask = pulse_mask[s:s + W]
        window_pulse_frac[i] = float(w_mask.mean())
        seg = z_bp[s:s + W]
        # Simple band-power proxy (mean-squared bandpassed amplitude).
        window_power[i] = float(np.mean(seg * seg))
        _, F_stat, _ = _thomson_f_test(seg, tapers, K)
        if F_stat.shape[0] != freq_hz.shape[0]:
            continue
        F_band = F_stat[test_band]
        f_band = freq_hz[test_band]
        if F_band.size == 0:
            continue
        idx_max = int(np.argmax(F_band))
        window_F[i] = float(F_band[idx_max])
        window_freq[i] = float(f_band[idx_max])
        window_p[i] = float(_f_dist.sf(window_F[i], 2, dof2))

    # Raw per-window F-test p threshold. We deliberately skip the Bonferroni
    # correction across test-band bins because the downstream chaining gate
    # (need ≥ ``sine_min_segment_ms / hop`` freq-consistent windows in a row
    # within ±``sine_freq_tolerance``) already provides segment-level
    # multiple-comparison control. Courtship sine is modulated, producing
    # moderate F stats (~5-15) with raw p ~0.02-0.08 per window; Bonferroni
    # at n_bins ≈ 9 dropped the effective threshold to ~0.0056, which
    # rejected ~2/3 of real sine windows on hand-validated bouts 23/27.
    p_threshold_bonf = p_threshold

    # Per-window power gate: mult × percentile of band-power across windows.
    # Median (50th pct) is too aggressive when a bout has loud pulse regions
    # whose interp-filtered residue inflates the median — genuine sine can
    # end up below it. A lower percentile (e.g. 25th) tracks the quiet-window
    # floor more faithfully.
    nz_power = window_power[window_power > 0]
    floor_power = (
        float(np.percentile(nz_power, cfg.sine_noise_floor_percentile))
        if nz_power.size > 0
        else 0.0
    )
    power_gate = cfg.sine_noise_floor_mult * floor_power
    is_candidate = (
        (window_p < p_threshold_bonf)
        & (window_pulse_frac <= cfg.sine_pulse_overlap_max)
        & (window_power >= power_gate)
    )

    # Chain consecutive candidate windows with ±freq_tol frequency
    # consistency. When a candidate's frequency doesn't match the running
    # reference, we drop THIS window (don't close + reopen) so that noise
    # regions with scattered F-test hits at random frequencies don't
    # produce spurious short segments.
    segments: List[Tuple[int, int, float, int]] = []
    seg_open = False
    seg_start = 0
    seg_end = 0
    seg_freqs: List[float] = []
    freq_tol = cfg.sine_freq_tolerance
    max_gap_windows = 2

    def _close_segment() -> None:
        nonlocal seg_open, seg_freqs
        if seg_open and seg_freqs:
            segments.append(
                (seg_start, seg_end, float(np.median(seg_freqs)), len(seg_freqs))
            )
        seg_open = False
        seg_freqs = []

    prev_cand_idx = -10 ** 9
    for i in range(n_win):
        if not is_candidate[i]:
            if seg_open and (i - prev_cand_idx) > max_gap_windows:
                _close_segment()
            continue
        f_i = window_freq[i]
        s_frame = starts[i]
        e_frame = starts[i] + W
        if not seg_open:
            seg_open = True
            seg_start = s_frame
            seg_end = e_frame
            seg_freqs = [f_i]
            prev_cand_idx = i
            continue
        # Gap timeout: close out the old segment and start a new one.
        if (i - prev_cand_idx) > max_gap_windows:
            _close_segment()
            seg_open = True
            seg_start = s_frame
            seg_end = e_frame
            seg_freqs = [f_i]
            prev_cand_idx = i
            continue
        f_ref = float(np.median(seg_freqs))
        if abs(f_i - f_ref) <= freq_tol * f_ref:
            seg_end = e_frame
            seg_freqs.append(f_i)
            prev_cand_idx = i
        # else: frequency inconsistent; skip this window without closing.
    _close_segment()

    # Length gate + minimum-candidate gate: reject segments shorter than
    # ``sine_min_segment_ms`` in frames OR segments whose span is padded
    # out mostly by the window width itself (need enough matching windows
    # to demonstrate a sustained signal).
    min_len = cfg.ms_to_frames(cfg.sine_min_segment_ms)
    min_cand_windows = max(2, min_len // max(1, hop))
    segments = [
        (s, e, f)
        for (s, e, f, n_cand) in segments
        if (e - s) >= min_len and n_cand >= min_cand_windows
    ]

    frame_labels = np.full(n, "quiet", dtype=object)
    for s, e, _ in segments:
        frame_labels[s:e] = "sine"
    # Pulse takes precedence over any overlapping sine.
    frame_labels[pulse_mask] = "pulse"

    seg_array = (
        np.asarray([[s, e] for (s, e, _) in segments], dtype=int)
        if segments
        else np.zeros((0, 2), dtype=int)
    )

    features = {
        "window_centers": window_centers,
        "window_freq": window_freq,
        "window_F": window_F,
        "window_p": window_p,
        "window_is_sine": is_candidate,
        "sine_segments": seg_array,
        "pulse_event_frames": np.array([], dtype=int),
    }
    return frame_labels, features


# -----------------------------------------------------------------------------
# Legacy FFT / peak-derivative detectors (kept for parallel comparison)
# -----------------------------------------------------------------------------


def _classify_one_side_fft_legacy(
    z: np.ndarray, cfg: SongAnalysisConfig
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """FFT classifier (primary) for one wing's tip Z trace. Returns
    ``(frame_labels, window_features)``. Pulse vs sine is determined by
    the peak frequency inside the song band AND the pulse/song power
    ratio. Non-singing windows fall through to ``quiet``."""
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
    peak_f = _peak_freq(Sxx, f_axis, *cfg.fft_song_band)

    with np.errstate(divide="ignore", invalid="ignore"):
        pulse_freq_ratio = np.where(song_p > 0, pulse_p / song_p, 0.0)

    window_labels = np.full(n_windows, "quiet", dtype=object)
    for i in range(n_windows):
        is_singing = song_p[i] >= cfg.song_power_threshold
        if is_singing:
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
        "peak_freq": peak_f,
        "pulse_freq_ratio": pulse_freq_ratio,
        "pulse_event_frames": np.array([], dtype=int),
    }


def _bilateral_pulse_mask_legacy(
    z_L: np.ndarray, z_R: np.ndarray, cfg: SongAnalysisConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Derivative-peak pulse detector on ``max(|dZ/dt|_L, |dZ/dt|_R)``.
    Returns a per-frame mask (with intra-train gaps filled) and the peak
    frame indices themselves. This is the primary pulse detector."""
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
) -> Dict[str, Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]]]:
    """Run both detector pipelines on L/R wings and return both label
    streams. ``legacy`` is the primary (FFT + derivative-peak) detector;
    ``new`` is the experimental (Butterworth + multitaper) detector kept
    for side-by-side comparison.

    Returned shape:
        {"L": {"legacy": (labels, features),
               "new":    (labels, features)},
         "R": {"legacy": (labels, features),
               "new":    (labels, features)}}
    """
    z_L = _interp_nan(np.asarray(wing_data[cfg.left_tip]["z"]))
    z_R = _interp_nan(np.asarray(wing_data[cfg.right_tip]["z"]))

    # ---- NEW pipeline: Butterworth + Hilbert pulse, multitaper sine ----
    mask_new, peaks_new = _bilateral_pulse_mask_butterworth(z_L, z_R, cfg)
    labels_L_new, feats_L_new = _classify_sine_multitaper(z_L, mask_new, cfg)
    labels_R_new, feats_R_new = _classify_sine_multitaper(z_R, mask_new, cfg)
    feats_L_new["pulse_event_frames"] = peaks_new
    feats_R_new["pulse_event_frames"] = peaks_new

    # ---- LEGACY pipeline: FFT classifier + bilateral peak pulse mask ----
    labels_L_leg, feats_L_leg = _classify_one_side_fft_legacy(z_L, cfg)
    labels_R_leg, feats_R_leg = _classify_one_side_fft_legacy(z_R, cfg)
    if cfg.use_peak_pulse and len(z_L) > 1:
        mask_leg, peaks_leg = _bilateral_pulse_mask_legacy(z_L, z_R, cfg)
        labels_L_leg = labels_L_leg.copy()
        labels_R_leg = labels_R_leg.copy()
        labels_L_leg[labels_L_leg == "pulse"] = "sine"
        labels_R_leg[labels_R_leg == "pulse"] = "sine"
        labels_L_leg[mask_leg] = "pulse"
        labels_R_leg[mask_leg] = "pulse"
        feats_L_leg["pulse_event_frames"] = peaks_leg
        feats_R_leg["pulse_event_frames"] = peaks_leg

    return {
        "L": {
            "new": (labels_L_new, feats_L_new),
            "legacy": (labels_L_leg, feats_L_leg),
        },
        "R": {
            "new": (labels_R_new, feats_R_new),
            "legacy": (labels_R_leg, feats_R_leg),
        },
    }


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
            "frac_pulse": 0.0, "frac_sine": 0.0,
            "frac_quiet": 0.0, "song_fraction": 0.0,
        }
    lbl = frame_labels[valid_mask]
    n_p = int((lbl == "pulse").sum())
    n_s = int((lbl == "sine").sum())
    n_q = int((lbl == "quiet").sum())
    return {
        "n_frames": eff,
        "duration_s": eff / fs,
        "frac_pulse": n_p / eff,
        "frac_sine": n_s / eff,
        "frac_quiet": n_q / eff,
        "song_fraction": (n_p + n_s) / eff,
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
                                          pulse_features,
                                          frame_labels_new, segments_new,
                                          summary_new, window_features_new},
                                  'R': {...}}. Primary keys come from the
                                  FFT + derivative-peak detector; ``*_new``
                                  keys come from the experimental
                                  Butterworth + multitaper detector.
        ``dominant_wing``      — 'L' or 'R'
        ``is_singing``         — (T,) bool gate
        ``activities``         — per-tip windowed |dZ/dt|
        ``angle_L, angle_R``   — (T,) wing extension angles (deg) if
                                  xpos_ego was provided
        ``joints``             — dict of wing DOFs if qpos was provided
        ``summary``            — dominant-wing song_fraction + bout
                                  metadata (bout length, fs) from the
                                  primary detector
        ``summary_new``        — same shape, from the experimental detector
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
    # classify_song_both_sides returns both the primary (FFT + derivative-peak)
    # and the experimental (Butterworth+Hilbert + multitaper) outputs. The
    # FFT + peak detector feeds the primary keys; the experimental detector
    # populates the ``*_new`` keys for side-by-side comparison. We also mirror
    # the primary keys under ``*_legacy`` for backward compatibility with
    # notebook code that expects the old naming.
    side_results = classify_song_both_sides(wing_data, cfg)
    sides: Dict[str, Dict] = {}
    for sname, streams in side_results.items():
        leg_labels, leg_features = streams["legacy"]
        new_labels, new_features = streams["new"]
        leg_segments = segments_from_labels(leg_labels, cfg.fs)
        leg_summary = _side_summary(leg_labels, cfg.fs, valid_mask)
        sides[sname] = {
            "frame_labels": leg_labels,
            "window_features": leg_features,
            "segments": leg_segments,
            "summary": leg_summary,
            # Alias of the primary keys (legacy naming retained).
            "frame_labels_legacy": leg_labels,
            "window_features_legacy": leg_features,
            "segments_legacy": leg_segments,
            "summary_legacy": leg_summary,
            # Experimental Butterworth + multitaper detector.
            "frame_labels_new": new_labels,
            "window_features_new": new_features,
            "segments_new": segments_from_labels(new_labels, cfg.fs),
            "summary_new": _side_summary(new_labels, cfg.fs, valid_mask),
        }

    # Wing extension angles (geometric feature, for plotting) -------------
    angle_L = angle_R = None
    horiz_angle_L = horiz_angle_R = None
    if xpos_ego is not None:
        xpe = np.asarray(xpos_ego)
        if xpe.ndim == 2:
            xpe = xpe.reshape(xpe.shape[0], -1, 3)
        angle_L, angle_R = compute_wing_extension_angles(xpe, kp_idx)
        horiz_angle_L, horiz_angle_R = compute_wing_horizontal_angles(xpe, kp_idx)

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
    dom_summary_new = sides[dominant_wing]["summary_new"]
    primary_summary = {
        "n_frames": T,
        "duration_s": T / cfg.fs,
        "dominant_wing": dominant_wing,
        "song_fraction": dom_summary["song_fraction"],
        "frac_pulse": dom_summary["frac_pulse"],
        "frac_sine": dom_summary["frac_sine"],
        "frac_quiet": dom_summary["frac_quiet"],
        "valid_n_frames": dom_summary["n_frames"],
    }

    return {
        "wing_data": wing_data,
        "sides": sides,
        "dominant_wing": dominant_wing,
        "is_singing": is_singing,
        "activities": activities,
        "angle_L": angle_L,
        "angle_R": angle_R,
        "horiz_angle_L": horiz_angle_L,
        "horiz_angle_R": horiz_angle_R,
        "joints": joints,
        "summary": primary_summary,
        # Alias of ``summary`` for backward compatibility with notebooks
        # that expected the old naming (when the new detector was primary).
        "summary_legacy": primary_summary,
        "summary_new": {
            "n_frames": T,
            "duration_s": T / cfg.fs,
            "dominant_wing": dominant_wing,
            "song_fraction": dom_summary_new["song_fraction"],
            "frac_pulse": dom_summary_new["frac_pulse"],
            "frac_sine": dom_summary_new["frac_sine"],
            "frac_quiet": dom_summary_new["frac_quiet"],
            "valid_n_frames": dom_summary_new["n_frames"],
        },
    }
