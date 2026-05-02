"""Courtship bout detection extracted from Sandbox_Strict.ipynb.

Provides detect_courtship_bouts() and reclassify_bouts_with_fft() so that
batch_bout_detection.py can run without the notebook.
"""

from collections import Counter

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import spectrogram as sp_spectrogram, csd

# ── Constants ─────────────────────────────────────────────────────────────────

JARVIS_SCALE = 10.0
FPS = 800

SCUTELLUM = "Scutellum"
COURT_ABD_TIP = "Abd_tip"

# Confidence filter
COURT_CONF_THRESHOLD = 0.70
COURT_CONF_GAP_BRIDGE = 5
COURT_CONF_WINGS_ONLY = True       # True = confidence filter only checks wing keypoints

# Upright filter
COURT_UPRIGHT_ENABLED = True

# Wing oscillation detection
COURT_WING_TIPS = ["WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13"]
COURT_WING_ACTIVITY_WINDOW = 100   # frames
COURT_WING_THRESHOLD = 15.0        # mm/s mean |dZ/dt|

# Pulse/sine classification (speed-based, refined by FFT)
COURT_PULSE_CLASSIFY_SPEED = 8.0   # mm/s — above = pulse, below = sine
COURT_MIN_SEGMENT_FRAMES = 40

# Bout parameters
COURT_MIN_BOUT_FRAMES = 80
COURT_MAX_GAP = 200

# Arena
COURT_ARENA_X_MM = 23.5
COURT_ARENA_Y_MM = 5.5

# Leg tips (for upright check)
COURT_ALL_LEG_TIPS = ["T1L_TaTip", "T1R_TaTip", "T2L_TaTip", "T2R_TaTip",
                       "T3L_TaTip", "T3R_TaTip"]

# Diagnostics
COURT_SHOW_BOUNDARY_DIAG = True
COURT_BOUNDARY_DIAG_FRAMES = 50

COURT_FILTER_NAMES = ['confidence', 'upright', 'wing_oscillation']

# ── FFT constants ─────────────────────────────────────────────────────────────

FFT_NPERSEG = 128
FFT_HOP = 32
FFT_SONG_BAND = (80, 350)
FFT_SINE_BAND = (80, 200)
FFT_PULSE_BAND = (150, 350)
FFT_WAGGLE_BAND = (5, 25)
FFT_SONG_POWER_THRESHOLD = 5e-7
FFT_PULSE_PEAK_FREQ_MIN = 137.5
FFT_PULSE_FREQ_RATIO_MIN = 0.30
FFT_WAGGLE_PHASE_MIN = 2.0
FFT_WAGGLE_BILATERAL_MIN = 0.4
FFT_MIN_SEGMENT_FRAMES = 40


# ── Shared utilities ──────────────────────────────────────────────────────────

def get_all_keypoint_names(df):
    cols = df.columns.tolist()
    kp_names = []
    seen = set()
    for col in cols:
        base_name = col.split('.')[0] if '.' in col else col
        if base_name not in seen:
            seen.add(base_name)
            kp_names.append(base_name)
    return kp_names


def extract_xyzc(df, kp_name, scale=JARVIS_SCALE):
    cols = df.columns.tolist()
    start_idx = cols.index(kp_name)
    x = df.iloc[:, start_idx].values.astype(float) / scale
    y = df.iloc[:, start_idx + 1].values.astype(float) / scale
    z = df.iloc[:, start_idx + 2].values.astype(float) / scale
    conf = df.iloc[:, start_idx + 3].values.astype(float)
    return x, y, z, conf


def bridge_short_gaps(mask, max_gap):
    if max_gap <= 0:
        return mask
    bridged = mask.copy()
    in_gap = False
    gap_start = 0
    for i in range(len(bridged)):
        if not bridged[i]:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                gap_len = i - gap_start
                if gap_len <= max_gap and gap_start > 0:
                    bridged[gap_start:i] = True
                in_gap = False
    return bridged


def compute_instant_speed(scutellum_data, start, end, fps=FPS):
    x = scutellum_data['x'][start:end+1]
    y = scutellum_data['y'][start:end+1]
    dx = np.diff(x)
    dy = np.diff(y)
    displacement = np.sqrt(dx**2 + dy**2)
    speed = displacement * fps
    speed = np.concatenate([[speed[0] if len(speed) > 0 else 0], speed])
    mean_speed = np.nanmean(speed)
    max_speed = np.nanmax(speed)
    return speed, mean_speed, max_speed


def compute_total_distance(scutellum_data, start, end):
    x = scutellum_data['x'][start:end+1]
    y = scutellum_data['y'][start:end+1]
    valid = ~np.isnan(x) & ~np.isnan(y)
    if valid.sum() < 2:
        return 0.0, 0.0
    x_valid, y_valid = x[valid], y[valid]
    dx, dy = np.diff(x_valid), np.diff(y_valid)
    total_distance = float(np.sum(np.sqrt(dx**2 + dy**2)))
    net_displacement = float(np.sqrt((x_valid[-1] - x_valid[0])**2 +
                                      (y_valid[-1] - y_valid[0])**2))
    return total_distance, net_displacement


def find_contiguous_bouts_with_gap_bridging(valid_mask, min_frames, max_gap):
    bridged = valid_mask.copy()
    in_gap = False
    gap_start = 0
    for i in range(len(bridged)):
        if not bridged[i]:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                gap_len = i - gap_start
                if gap_len <= max_gap and gap_start > 0 and bridged[gap_start - 1]:
                    bridged[gap_start:i] = True
                in_gap = False
    bouts = []
    in_bout = False
    start = 0
    for i, valid in enumerate(bridged):
        if valid and not in_bout:
            start = i
            in_bout = True
        elif not valid and in_bout:
            if i - start >= min_frames:
                bouts.append((start, i - 1))
            in_bout = False
    if in_bout and len(bridged) - start >= min_frames:
        bouts.append((start, len(bridged) - 1))
    return bouts


# ── Wing activity ─────────────────────────────────────────────────────────────

def compute_wing_activity(df, scale=JARVIS_SCALE, fps=FPS,
                          wing_tips=None, window=None):
    if wing_tips is None:
        wing_tips = COURT_WING_TIPS
    if window is None:
        window = COURT_WING_ACTIVITY_WINDOW

    activities = {}
    wing_data = {}
    for tip in wing_tips:
        x, y, z, conf = extract_xyzc(df, tip, scale)
        wing_data[tip] = {'x': x, 'y': y, 'z': z, 'conf': conf}
        dz = np.abs(np.diff(z, prepend=z[0]) * fps)
        nan_mask = np.isnan(dz)
        filled = np.where(nan_mask, 0.0, dz)
        smoothed = uniform_filter1d(filled, size=window)
        valid_frac = uniform_filter1d((~nan_mask).astype(float), size=window)
        activities[tip] = np.where(valid_frac > 0.01, smoothed / valid_frac, np.nan)

    return activities, wing_data


# ── Frame-level filter ────────────────────────────────────────────────────────

def apply_courtship_filter(df, scale=JARVIS_SCALE, fps=FPS):
    """Apply confidence + upright + wing-oscillation filters.

    Returns:
        valid_mask, leg_data, scut_data, wing_data, activities,
        filter_masks, diagnostics, abd_data
    """
    n_frames = len(df)

    leg_data = {}
    for tip in COURT_ALL_LEG_TIPS:
        x, y, z, conf = extract_xyzc(df, tip, scale)
        leg_data[tip] = {'x': x, 'y': y, 'z': z, 'conf': conf}

    scut_x, scut_y, scut_z, scut_conf = extract_xyzc(df, SCUTELLUM, scale)
    scut_data = {'x': scut_x, 'y': scut_y, 'z': scut_z, 'conf': scut_conf}

    abd_data = {}
    try:
        abd_x, abd_y, abd_z, abd_conf = extract_xyzc(df, COURT_ABD_TIP, scale)
        abd_data = {'x': abd_x, 'y': abd_y, 'z': abd_z, 'conf': abd_conf}
    except (ValueError, IndexError):
        pass

    activities, wing_data = compute_wing_activity(df, scale, fps)

    # Filter 1: Confidence
    conf_mask = np.ones(n_frames, dtype=bool)
    n_checked = 0
    if COURT_CONF_WINGS_ONLY:
        for tip in COURT_WING_TIPS:
            conf_mask &= (wing_data[tip]['conf'] >= COURT_CONF_THRESHOLD)
            n_checked += 1
    else:
        for kp in get_all_keypoint_names(df):
            try:
                _, _, _, conf = extract_xyzc(df, kp, scale)
                conf_mask &= (conf >= COURT_CONF_THRESHOLD)
                n_checked += 1
            except (ValueError, IndexError):
                pass

    if COURT_CONF_GAP_BRIDGE > 0:
        conf_mask = bridge_short_gaps(conf_mask, COURT_CONF_GAP_BRIDGE)

    # Filter 2: Upright
    upright_mask = np.ones(n_frames, dtype=bool)
    if COURT_UPRIGHT_ENABLED:
        for tip in COURT_ALL_LEG_TIPS:
            upright_mask &= (scut_z > leg_data[tip]['z'])

    # Filter 3: Wing oscillation
    left_active = ((activities['WingL_V12'] > COURT_WING_THRESHOLD) &
                   (activities['WingL_V13'] > COURT_WING_THRESHOLD))
    right_active = ((activities['WingR_V12'] > COURT_WING_THRESHOLD) &
                    (activities['WingR_V13'] > COURT_WING_THRESHOLD))
    wing_mask = left_active | right_active

    valid_mask = conf_mask & upright_mask & wing_mask

    filter_masks = {
        'confidence': conf_mask,
        'upright': upright_mask,
        'wing_oscillation': wing_mask,
        '_conf_mask_for_detection': conf_mask,
    }

    conf_mode = "wings only" if COURT_CONF_WINGS_ONLY else "all keypoints"
    diagnostics = {
        'total_frames': n_frames,
        'confidence_pass': int(conf_mask.sum()),
        'confidence_pass_pct': 100 * conf_mask.sum() / n_frames,
        'confidence_kps_checked': n_checked,
        'confidence_mode': conf_mode,
        'upright_pass': int(upright_mask.sum()),
        'upright_pass_pct': 100 * upright_mask.sum() / n_frames,
        'wing_oscillation_pass': int(wing_mask.sum()),
        'wing_oscillation_pass_pct': 100 * wing_mask.sum() / n_frames,
        'combined_pass': int(valid_mask.sum()),
        'combined_pass_pct': 100 * valid_mask.sum() / n_frames,
    }

    return valid_mask, leg_data, scut_data, wing_data, activities, filter_masks, diagnostics, abd_data


# ── Pulse/sine classification ─────────────────────────────────────────────────

def classify_courtship_segments(bout_info, scut_data, fps=FPS):
    """Classify frames within a courtship bout as pulse or sine by speed."""
    start, end = bout_info['start'], bout_info['end']
    speed, _, _ = compute_instant_speed(scut_data, start, end, fps)

    smooth_window = min(COURT_MIN_SEGMENT_FRAMES, len(speed))
    smoothed = uniform_filter1d(speed, size=smooth_window) if smooth_window > 1 else speed

    is_pulse = smoothed > COURT_PULSE_CLASSIFY_SPEED
    segments = []
    current_type = 'pulse' if is_pulse[0] else 'sine'
    seg_start_local = 0

    for i in range(1, len(is_pulse)):
        frame_type = 'pulse' if is_pulse[i] else 'sine'
        if frame_type != current_type:
            segments.append({
                'start': start + seg_start_local,
                'end': start + i - 1,
                'type': current_type,
                'n_frames': i - seg_start_local,
                'duration_s': (i - seg_start_local) / fps,
            })
            current_type = frame_type
            seg_start_local = i

    segments.append({
        'start': start + seg_start_local,
        'end': end,
        'type': current_type,
        'n_frames': end - start - seg_start_local + 1,
        'duration_s': (end - start - seg_start_local + 1) / fps,
    })

    n_pulse = int(is_pulse.sum())
    n_sine = len(is_pulse) - n_pulse
    bout_info['segments'] = segments
    bout_info['n_pulse_frames'] = n_pulse
    bout_info['n_sine_frames'] = n_sine
    bout_info['pct_pulse'] = 100 * n_pulse / len(is_pulse)
    bout_info['pct_sine'] = 100 * n_sine / len(is_pulse)


def diagnose_courtship_boundaries(bout_start, bout_end, filter_masks,
                                   n_frames_context=COURT_BOUNDARY_DIAG_FRAMES):
    total_frames = len(filter_masks['confidence'])
    pre_start = max(0, bout_start - n_frames_context)
    pre_range = slice(pre_start, bout_start)
    post_end = min(total_frames, bout_end + n_frames_context + 1)
    post_range = slice(bout_end + 1, post_end)

    pre_failures = {'n_frames_analyzed': bout_start - pre_start}
    post_failures = {'n_frames_analyzed': post_end - bout_end - 1}

    for name in COURT_FILTER_NAMES:
        mask = filter_masks[name]
        pre_failures[name] = int((~mask[pre_range]).sum())
        post_failures[name] = int((~mask[post_range]).sum())

    pre_items = [(k, v) for k, v in pre_failures.items() if k != 'n_frames_analyzed']
    post_items = [(k, v) for k, v in post_failures.items() if k != 'n_frames_analyzed']
    pre_max = max(pre_items, key=lambda x: x[1]) if pre_items else ('none', 0)
    post_max = max(post_items, key=lambda x: x[1]) if post_items else ('none', 0)

    return {
        'pre_bout_failures': pre_failures,
        'post_bout_failures': post_failures,
        'primary_cause_before': pre_max[0] if pre_max[1] > 0 else None,
        'primary_cause_after': post_max[0] if post_max[1] > 0 else None,
    }


# ── Main detection pipeline ───────────────────────────────────────────────────

def detect_courtship_bouts(df, scale=JARVIS_SCALE, fps=FPS, verbose=True):
    """Unified courtship bout detection with pulse/sine classification.

    Step 1: frame-level filters (confidence + upright + wing oscillation)
    Step 2: find contiguous bouts within confidence-valid regions
    Step 3: classify frames as pulse/sine by scutellum speed

    Returns:
        bouts, leg_data, scut_data, wing_data, activities,
        filter_masks, diagnostics, abd_data
    """
    if verbose:
        print("=" * 70)
        print("COURTSHIP BOUT DETECTION (unified pulse + sine)")
        print("=" * 70)

    valid_mask, leg_data, scut_data, wing_data, activities, filter_masks, diagnostics, abd_data = \
        apply_courtship_filter(df, scale, fps)

    if verbose:
        print(f"\n  Total frames: {diagnostics['total_frames']}")
        print(f"  Confidence pass: {diagnostics['confidence_pass_pct']:.1f}%")
        print(f"  Upright pass: {diagnostics['upright_pass_pct']:.1f}%")
        print(f"  Wing oscillation pass: {diagnostics['wing_oscillation_pass_pct']:.1f}%")
        print(f"  Combined valid: {diagnostics['combined_pass_pct']:.1f}%")

    # Find bouts only within contiguous confidence-valid regions
    conf_mask_det = filter_masks['_conf_mask_for_detection']
    conf_regions = []
    in_conf = False
    cr_start = 0
    for i, v in enumerate(conf_mask_det):
        if v and not in_conf:
            cr_start = i
            in_conf = True
        elif not v and in_conf:
            conf_regions.append((cr_start, i - 1))
            in_conf = False
    if in_conf:
        conf_regions.append((cr_start, len(conf_mask_det) - 1))

    bouts_raw = []
    for cr_s, cr_e in conf_regions:
        region_valid = valid_mask[cr_s:cr_e+1]
        for bs, be in find_contiguous_bouts_with_gap_bridging(
                region_valid, COURT_MIN_BOUT_FRAMES, COURT_MAX_GAP):
            bouts_raw.append((cr_s + bs, cr_s + be))

    if verbose:
        print(f"\n  Courtship bouts found: {len(bouts_raw)}")

    court_bouts = []
    for i, (start, end) in enumerate(bouts_raw):
        n = end - start + 1
        _, mean_speed, max_speed = compute_instant_speed(scut_data, start, end, fps)
        total_dist, net_disp = compute_total_distance(scut_data, start, end)

        bout_info = {
            'start': start,
            'end': end,
            'n_frames': n,
            'duration_s': n / fps,
            'mean_speed_mm_s': float(mean_speed),
            'max_speed_mm_s': float(max_speed),
            'total_distance_mm': float(total_dist),
            'bout_idx': i + 1,
        }

        classify_courtship_segments(bout_info, scut_data, fps)

        if COURT_SHOW_BOUNDARY_DIAG:
            bout_info['boundary_diagnostics'] = diagnose_courtship_boundaries(
                start, end, filter_masks, COURT_BOUNDARY_DIAG_FRAMES
            )

        court_bouts.append(bout_info)

    diagnostics['n_courtship_bouts'] = len(court_bouts)

    return court_bouts, leg_data, scut_data, wing_data, activities, filter_masks, diagnostics, abd_data


# ── FFT reclassification ──────────────────────────────────────────────────────

def _interp_nan(z):
    nans = np.isnan(z)
    if not nans.any():
        return z.copy()
    out = z.copy()
    idx = np.arange(len(z))
    valid = ~nans
    if valid.sum() < 2:
        return np.zeros_like(z)
    out[nans] = np.interp(idx[nans], idx[valid], z[valid])
    return out


def _band_power(Sxx, f, fmin, fmax):
    mask = (f >= fmin) & (f <= fmax)
    if mask.sum() == 0:
        return np.zeros(Sxx.shape[1])
    return np.mean(Sxx[mask, :], axis=0)


def _peak_freq(Sxx, f, fmin, fmax):
    mask = (f >= fmin) & (f <= fmax)
    if mask.sum() == 0:
        return np.zeros(Sxx.shape[1])
    f_band = f[mask]
    peak_idx = np.argmax(Sxx[mask, :], axis=0)
    return f_band[peak_idx]


def _merge_short_segments(labels, min_len):
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


def classify_song_modes_fft(wing_data, start, end, fps=FPS):
    """FFT-based per-frame classification: pulse / sine / waggle / quiet."""
    n_frames = end - start + 1
    nperseg = min(FFT_NPERSEG, n_frames)
    noverlap = max(0, nperseg - FFT_HOP)

    tips_L = ['WingL_V12', 'WingL_V13']
    tips_R = ['WingR_V12', 'WingR_V13']

    spectrograms = {}
    f_axis = t_axis = None
    for tip in tips_L + tips_R:
        z = _interp_nan(wing_data[tip]['z'][start:end+1])
        f, t, Sxx = sp_spectrogram(z, fs=fps, nperseg=nperseg,
                                    noverlap=noverlap, detrend='linear')
        spectrograms[tip] = Sxx
        if f_axis is None:
            f_axis, t_axis = f, t

    n_windows = len(t_axis)
    Sxx_L = (spectrograms['WingL_V12'] + spectrograms['WingL_V13']) / 2
    Sxx_R = (spectrograms['WingR_V12'] + spectrograms['WingR_V13']) / 2

    song_L = _band_power(Sxx_L, f_axis, *FFT_SONG_BAND)
    song_R = _band_power(Sxx_R, f_axis, *FFT_SONG_BAND)
    pulse_L = _band_power(Sxx_L, f_axis, *FFT_PULSE_BAND)
    pulse_R = _band_power(Sxx_R, f_axis, *FFT_PULSE_BAND)
    waggle_L = _band_power(Sxx_L, f_axis, *FFT_WAGGLE_BAND)
    waggle_R = _band_power(Sxx_R, f_axis, *FFT_WAGGLE_BAND)
    peak_f_L = _peak_freq(Sxx_L, f_axis, *FFT_SONG_BAND)
    peak_f_R = _peak_freq(Sxx_R, f_axis, *FFT_SONG_BAND)

    dominant_song = np.maximum(song_L, song_R)
    dominant_side = np.where(song_R >= song_L, 'R', 'L')
    dominant_pulse = np.where(song_R >= song_L, pulse_R, pulse_L)
    dominant_peak_f = np.where(song_R >= song_L, peak_f_R, peak_f_L)

    with np.errstate(divide='ignore', invalid='ignore'):
        pulse_freq_ratio = np.where(dominant_song > 0,
                                     dominant_pulse / dominant_song, 0.0)

    max_waggle = np.maximum(waggle_L, waggle_R)
    min_waggle = np.minimum(waggle_L, waggle_R)
    with np.errstate(divide='ignore', invalid='ignore'):
        bilateral_ratio = np.where(max_waggle > 0, min_waggle / max_waggle, 0.0)

    z_L_mean = (_interp_nan(wing_data['WingL_V12']['z'][start:end+1]) +
                _interp_nan(wing_data['WingL_V13']['z'][start:end+1])) / 2
    z_R_mean = (_interp_nan(wing_data['WingR_V12']['z'][start:end+1]) +
                _interp_nan(wing_data['WingR_V13']['z'][start:end+1])) / 2

    f_csd, Pxy = csd(z_L_mean, z_R_mean, fs=fps, nperseg=nperseg, noverlap=noverlap)
    csd_phase = np.abs(np.angle(Pxy))
    waggle_f_mask = (f_csd >= FFT_WAGGLE_BAND[0]) & (f_csd <= FFT_WAGGLE_BAND[1])
    mean_waggle_phase = float(np.mean(csd_phase[waggle_f_mask])) if waggle_f_mask.sum() > 0 else 0.0

    window_labels = np.full(n_windows, 'quiet', dtype=object)
    for i in range(n_windows):
        is_singing = dominant_song[i] >= FFT_SONG_POWER_THRESHOLD
        is_waggle = (max_waggle[i] > dominant_song[i] and
                     bilateral_ratio[i] >= FFT_WAGGLE_BILATERAL_MIN and
                     mean_waggle_phase >= FFT_WAGGLE_PHASE_MIN)
        if is_waggle and not is_singing:
            window_labels[i] = 'waggle'
        elif is_singing:
            if (dominant_peak_f[i] >= FFT_PULSE_PEAK_FREQ_MIN and
                    pulse_freq_ratio[i] >= FFT_PULSE_FREQ_RATIO_MIN):
                window_labels[i] = 'pulse'
            else:
                window_labels[i] = 'sine'

    frame_labels = np.full(n_frames, 'quiet', dtype=object)
    window_centers = (t_axis * fps).astype(int)

    for i in range(n_windows):
        w_start = max(0, window_centers[i] - FFT_HOP // 2)
        w_end = min(n_frames, window_centers[i] + FFT_HOP // 2)
        frame_labels[w_start:w_end] = window_labels[i]

    if n_windows > 0:
        frame_labels[:max(0, window_centers[0])] = window_labels[0]
        frame_labels[min(n_frames, window_centers[-1]):] = window_labels[-1]

    frame_labels = _merge_short_segments(frame_labels, FFT_MIN_SEGMENT_FRAMES)

    window_features = {
        'window_centers': window_centers + start,
        'dominant_song_power': dominant_song,
        'dominant_side': dominant_side,
        'peak_freq': dominant_peak_f,
        'pulse_freq_ratio': pulse_freq_ratio,
    }
    return frame_labels, window_features


def reclassify_bouts_with_fft(bouts, wing_data, fps=FPS, verbose=True):
    """Replace speed-based pulse/sine classification with FFT spectral analysis.

    Updates bouts in-place with pct_pulse, pct_sine, pct_waggle, dominant_wing.
    Returns the same bouts list.
    """
    if verbose:
        print("=" * 70)
        print("FFT-BASED SONG MODE RECLASSIFICATION")
        print("=" * 70)

    for bout in bouts:
        start, end = bout['start'], bout['end']
        n_frames = end - start + 1

        if n_frames < FFT_NPERSEG // 2:
            bout.setdefault('pct_waggle', 0.0)
            bout.setdefault('dominant_wing', None)
            bout.setdefault('fft_classified', False)
            continue

        frame_labels, wf = classify_song_modes_fft(wing_data, start, end, fps)

        # Build segments from contiguous label runs
        segments = []
        current_type = frame_labels[0]
        seg_start_local = 0
        for i in range(1, len(frame_labels)):
            if frame_labels[i] != current_type:
                seg_n = i - seg_start_local
                segments.append({'start': start + seg_start_local,
                                  'end': start + i - 1,
                                  'type': current_type,
                                  'n_frames': seg_n,
                                  'duration_s': seg_n / fps})
                current_type = frame_labels[i]
                seg_start_local = i
        seg_n = n_frames - seg_start_local
        segments.append({'start': start + seg_start_local, 'end': end,
                         'type': current_type, 'n_frames': seg_n,
                         'duration_s': seg_n / fps})

        song_segments = [s for s in segments if s['type'] != 'quiet']
        if not song_segments:
            dom_power = np.max(wf['dominant_song_power']) if len(wf['dominant_song_power']) > 0 else 0
            fallback_type = 'sine'
            if dom_power >= FFT_SONG_POWER_THRESHOLD:
                if (np.median(wf['peak_freq']) >= FFT_PULSE_PEAK_FREQ_MIN and
                        np.median(wf['pulse_freq_ratio']) >= FFT_PULSE_FREQ_RATIO_MIN):
                    fallback_type = 'pulse'
            song_segments = [{'start': start, 'end': end, 'type': fallback_type,
                               'n_frames': n_frames, 'duration_s': n_frames / fps}]

        bout['segments'] = song_segments
        bout['fft_classified'] = True

        total = sum(s['n_frames'] for s in song_segments)
        n_pulse = sum(s['n_frames'] for s in song_segments if s['type'] == 'pulse')
        n_sine = sum(s['n_frames'] for s in song_segments if s['type'] == 'sine')
        n_waggle = sum(s['n_frames'] for s in song_segments if s['type'] == 'waggle')
        bout['n_pulse_frames'] = n_pulse
        bout['n_sine_frames'] = n_sine
        bout['n_waggle_frames'] = n_waggle
        bout['pct_pulse'] = 100 * n_pulse / total if total > 0 else 0.0
        bout['pct_sine'] = 100 * n_sine / total if total > 0 else 0.0
        bout['pct_waggle'] = 100 * n_waggle / total if total > 0 else 0.0

        if len(wf['dominant_side']) > 0:
            side_counts = Counter(wf['dominant_side'])
            bout['dominant_wing'] = side_counts.most_common(1)[0][0]
        else:
            bout['dominant_wing'] = None

    if verbose:
        print(f"\n  Reclassified {sum(1 for b in bouts if b.get('fft_classified'))} bouts")

    return bouts
