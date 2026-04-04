"""Walking bout detection extracted from Bout_prediction.ipynb.

Provides detect_walking_bouts() and all helpers so that
batch_bout_detection.py can run without the notebook.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.stats import pearsonr, spearmanr

# ── Constants ─────────────────────────────────────────────────────────────────

JARVIS_SCALE = 10.0
FPS = 800

SCUTELLUM = "Scutellum"
LEG_TIPS = ["T1L_TaTip", "T1R_TaTip", "T2L_TaTip", "T2R_TaTip", "T3L_TaTip", "T3R_TaTip"]

# Hard filters
CONFIDENCE_FILTER_ENABLED = True
CONFIDENCE_THRESHOLD = 0.80
CONFIDENCE_GAP_BRIDGE = 15
UPRIGHT_FILTER_ENABLED = True

# Soft filters
FLOOR_Z_FILTER_ENABLED = True
FLOOR_Z_FILTER_MODE = "split_bout"
FLOOR_Z_THRESHOLD = 0.40

Y_WALL_UPPER_ENABLED = True
Y_WALL_UPPER_MODE = "split_bout"
Y_WALL_MAX = 5.4
Y_WALL_UPPER_MARGIN = 0.0

Y_WALL_LOWER_ENABLED = True
Y_WALL_LOWER_MODE = "split_bout"
Y_WALL_MIN = 0
Y_WALL_LOWER_MARGIN = 0.0

X_WALL_ENABLED = True
X_WALL_MODE = "split_bout"
X_WALL_MARGIN = 0.0
X_WALL_TIPS = ["T1L_TaTip", "T1R_TaTip", "T3L_TaTip", "T3R_TaTip"]

IMMOBILITY_FILTER_ENABLED = True
IMMOBILITY_FILTER_MODE = "split_bout"
MAX_STATIONARY_FRAMES = 25
STATIONARY_SPEED_THRESHOLD = 5

# Walking validation
MIN_WALKING_CYCLES = 2
MIN_DISTANCE_MM = 5
MAX_IMMOBILITY_FRAMES = 100
MIN_BOUT_FRAMES = 100
MAX_SWING_DURATION = 35
SWING_PROMINENCE = 0.05
ENFORCE_STRAIGHTNESS = False
STRAIGHTNESS_THRESHOLD = 0.2

# Arena
ARENA_X_MM = 23.5
ARENA_Y_MM = 5.5

SHOW_BOUNDARY_DIAGNOSTICS = True
BOUNDARY_DIAGNOSTIC_FRAMES = 50


# ── Data extraction ───────────────────────────────────────────────────────────

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


def extract_all_data(df, scale=JARVIS_SCALE):
    leg_tip_data = {}
    for tip in LEG_TIPS:
        x, y, z, conf = extract_xyzc(df, tip, scale)
        leg_tip_data[tip] = {'x': x, 'y': y, 'z': z, 'conf': conf}
    scut_x, scut_y, scut_z, scut_conf = extract_xyzc(df, SCUTELLUM, scale)
    scutellum_data = {'x': scut_x, 'y': scut_y, 'z': scut_z, 'conf': scut_conf}
    all_kp_names = get_all_keypoint_names(df)
    all_keypoint_conf = {}
    for kp in all_kp_names:
        try:
            _, _, _, conf = extract_xyzc(df, kp, scale)
            all_keypoint_conf[kp] = conf
        except (ValueError, IndexError):
            pass
    return leg_tip_data, scutellum_data, all_keypoint_conf


# ── Frame-level filters ───────────────────────────────────────────────────────

def bridge_short_gaps(mask, max_gap):
    """Bridge runs of False <= max_gap that are bounded by True on both sides."""
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


def apply_comprehensive_walking_filter(
    df,
    scale=JARVIS_SCALE,
    confidence_filter_enabled=CONFIDENCE_FILTER_ENABLED,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    confidence_gap_bridge=CONFIDENCE_GAP_BRIDGE,
    upright_filter_enabled=UPRIGHT_FILTER_ENABLED,
    floor_z_filter_enabled=FLOOR_Z_FILTER_ENABLED,
    floor_z_filter_mode=FLOOR_Z_FILTER_MODE,
    floor_z_threshold=FLOOR_Z_THRESHOLD,
    y_wall_upper_enabled=Y_WALL_UPPER_ENABLED,
    y_wall_upper_mode=Y_WALL_UPPER_MODE,
    y_wall_max=Y_WALL_MAX,
    y_wall_upper_margin=Y_WALL_UPPER_MARGIN,
    y_wall_lower_enabled=Y_WALL_LOWER_ENABLED,
    y_wall_lower_mode=Y_WALL_LOWER_MODE,
    y_wall_min=Y_WALL_MIN,
    y_wall_lower_margin=Y_WALL_LOWER_MARGIN,
    x_wall_enabled=X_WALL_ENABLED,
    x_wall_mode=X_WALL_MODE,
    x_wall_margin=X_WALL_MARGIN,
    x_wall_tips=X_WALL_TIPS,
    arena_x_mm=ARENA_X_MM,
):
    n_frames = len(df)
    leg_tip_data, scutellum_data, all_keypoint_conf = extract_all_data(df, scale)

    # Confidence
    confidence_mask = np.ones(n_frames, dtype=bool)
    n_recovered = 0
    if confidence_filter_enabled:
        for kp_name, conf_arr in all_keypoint_conf.items():
            confidence_mask &= (conf_arr >= confidence_threshold)
        if confidence_gap_bridge > 0:
            n_before = int((~confidence_mask).sum())
            confidence_mask = bridge_short_gaps(confidence_mask, confidence_gap_bridge)
            n_recovered = n_before - int((~confidence_mask).sum())

    # Upright
    upright_mask = np.ones(n_frames, dtype=bool)
    if upright_filter_enabled:
        for tip in LEG_TIPS:
            upright_mask &= (scutellum_data['z'] > leg_tip_data[tip]['z'])

    # Floor Z violations
    floor_violation_mask = np.zeros(n_frames, dtype=bool)
    if floor_z_filter_enabled:
        for tip in LEG_TIPS:
            floor_violation_mask |= (leg_tip_data[tip]['z'] < floor_z_threshold)

    # Y-wall upper violations
    y_wall_upper_violation_mask = np.zeros(n_frames, dtype=bool)
    if y_wall_upper_enabled:
        for tip in LEG_TIPS:
            y_wall_upper_violation_mask |= (leg_tip_data[tip]['y'] >= y_wall_max - y_wall_upper_margin)

    # Y-wall lower violations
    y_wall_lower_violation_mask = np.zeros(n_frames, dtype=bool)
    if y_wall_lower_enabled:
        for tip in LEG_TIPS:
            y_wall_lower_violation_mask |= (leg_tip_data[tip]['y'] <= y_wall_min + y_wall_lower_margin)

    # X-wall violations
    x_wall_violation_mask = np.zeros(n_frames, dtype=bool)
    if x_wall_enabled:
        for tip in x_wall_tips:
            x = leg_tip_data[tip]['x']
            x_wall_violation_mask |= (x >= arena_x_mm - x_wall_margin)
            x_wall_violation_mask |= (x <= x_wall_margin)

    # Valid mask (hard filters + soft filters if not "ignore")
    valid_mask = confidence_mask & upright_mask
    if floor_z_filter_enabled and floor_z_filter_mode != "ignore":
        valid_mask &= ~floor_violation_mask
    if y_wall_upper_enabled and y_wall_upper_mode != "ignore":
        valid_mask &= ~y_wall_upper_violation_mask
    if y_wall_lower_enabled and y_wall_lower_mode != "ignore":
        valid_mask &= ~y_wall_lower_violation_mask
    if x_wall_enabled and x_wall_mode != "ignore":
        valid_mask &= ~x_wall_violation_mask

    filter_masks = {
        'confidence': confidence_mask,
        'upright': upright_mask,
        'floor_violation': floor_violation_mask,
        'y_wall_upper_violation': y_wall_upper_violation_mask,
        'y_wall_lower_violation': y_wall_lower_violation_mask,
        'x_wall_violation': x_wall_violation_mask,
    }

    diagnostics = {
        'total_frames': n_frames,
        'confidence_pass': int(confidence_mask.sum()),
        'upright_pass': int(upright_mask.sum()),
        'floor_violations': int(floor_violation_mask.sum()),
        'y_wall_upper_violations': int(y_wall_upper_violation_mask.sum()),
        'y_wall_lower_violations': int(y_wall_lower_violation_mask.sum()),
        'x_wall_violations': int(x_wall_violation_mask.sum()),
        'combined_pass': int(valid_mask.sum()),
        'confidence_pass_pct': 100 * confidence_mask.sum() / n_frames,
        'upright_pass_pct': 100 * upright_mask.sum() / n_frames,
        'floor_violation_pct': 100 * floor_violation_mask.sum() / n_frames,
        'y_wall_upper_violation_pct': 100 * y_wall_upper_violation_mask.sum() / n_frames,
        'y_wall_lower_violation_pct': 100 * y_wall_lower_violation_mask.sum() / n_frames,
        'x_wall_violation_pct': 100 * x_wall_violation_mask.sum() / n_frames,
        'combined_pass_pct': 100 * valid_mask.sum() / n_frames,
        'n_keypoints_checked': len(all_keypoint_conf),
        'confidence_frames_bridged': n_recovered,
    }

    return valid_mask, leg_tip_data, scutellum_data, filter_masks, diagnostics


# ── Bout detection ────────────────────────────────────────────────────────────

def find_contiguous_bouts_with_gap_bridging(valid_mask, min_frames=MIN_BOUT_FRAMES,
                                             max_gap=MAX_IMMOBILITY_FRAMES):
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


def compute_instant_speed(scutellum_data, start, end, fps=FPS):
    x = scutellum_data['x'][start:end+1]
    y = scutellum_data['y'][start:end+1]
    dx = np.diff(x)
    dy = np.diff(y)
    displacement = np.sqrt(dx**2 + dy**2)
    speed = displacement * fps
    speed = np.concatenate([[speed[0] if len(speed) > 0 else 0], speed])
    mean_speed = float(np.nanmean(speed))
    max_speed = float(np.nanmax(speed))
    return speed, mean_speed, max_speed


def split_bouts_at_violations(
    bouts,
    leg_tip_data,
    scutellum_data,
    filter_masks,
    fps=FPS,
    floor_z_filter_enabled=FLOOR_Z_FILTER_ENABLED,
    floor_z_filter_mode=FLOOR_Z_FILTER_MODE,
    y_wall_upper_enabled=Y_WALL_UPPER_ENABLED,
    y_wall_upper_mode=Y_WALL_UPPER_MODE,
    y_wall_lower_enabled=Y_WALL_LOWER_ENABLED,
    y_wall_lower_mode=Y_WALL_LOWER_MODE,
    x_wall_enabled=X_WALL_ENABLED,
    x_wall_mode=X_WALL_MODE,
    immobility_filter_enabled=IMMOBILITY_FILTER_ENABLED,
    immobility_filter_mode=IMMOBILITY_FILTER_MODE,
    max_stationary=MAX_STATIONARY_FRAMES,
    stationary_thresh=STATIONARY_SPEED_THRESHOLD,
    min_frames=MIN_BOUT_FRAMES,
):
    split_bouts_list = []
    bout_excluded_masks = []

    for start, end in bouts:
        n = end - start + 1
        split_violation = np.zeros(n, dtype=bool)
        excluded_mask = np.zeros(n, dtype=bool)

        if floor_z_filter_enabled:
            viol = filter_masks['floor_violation'][start:end+1]
            if floor_z_filter_mode == "split_bout":
                split_violation |= viol
            elif floor_z_filter_mode == "exclude_frames":
                excluded_mask |= viol

        if y_wall_upper_enabled:
            viol = filter_masks['y_wall_upper_violation'][start:end+1]
            if y_wall_upper_mode == "split_bout":
                split_violation |= viol
            elif y_wall_upper_mode == "exclude_frames":
                excluded_mask |= viol

        if y_wall_lower_enabled:
            viol = filter_masks['y_wall_lower_violation'][start:end+1]
            if y_wall_lower_mode == "split_bout":
                split_violation |= viol
            elif y_wall_lower_mode == "exclude_frames":
                excluded_mask |= viol

        if x_wall_enabled:
            viol = filter_masks['x_wall_violation'][start:end+1]
            if x_wall_mode == "split_bout":
                split_violation |= viol
            elif x_wall_mode == "exclude_frames":
                excluded_mask |= viol

        if immobility_filter_enabled:
            speed, _, _ = compute_instant_speed(scutellum_data, start, end, fps)
            is_stationary = speed < stationary_thresh
            immobility_viol = np.zeros(n, dtype=bool)
            run_len = 0
            run_start = 0
            for i, stat in enumerate(is_stationary):
                if stat:
                    if run_len == 0:
                        run_start = i
                    run_len += 1
                else:
                    if run_len > max_stationary:
                        immobility_viol[run_start:run_start + run_len] = True
                    run_len = 0
            if run_len > max_stationary:
                immobility_viol[run_start:run_start + run_len] = True
            if immobility_filter_mode == "split_bout":
                split_violation |= immobility_viol
            elif immobility_filter_mode == "exclude_frames":
                excluded_mask |= immobility_viol

        valid = ~split_violation
        in_valid = False
        sub_start = 0
        for i, v in enumerate(valid):
            if v and not in_valid:
                sub_start = i
                in_valid = True
            elif not v and in_valid:
                if i - sub_start >= min_frames:
                    split_bouts_list.append((start + sub_start, start + i - 1))
                    bout_excluded_masks.append(excluded_mask[sub_start:i].copy())
                in_valid = False
        if in_valid and (n - sub_start) >= min_frames:
            split_bouts_list.append((start + sub_start, end))
            bout_excluded_masks.append(excluded_mask[sub_start:].copy())

    return split_bouts_list, bout_excluded_masks


def get_excluded_frames_for_bout(start, end, filter_masks,
                                  floor_z_filter_mode=FLOOR_Z_FILTER_MODE,
                                  y_wall_upper_mode=Y_WALL_UPPER_MODE,
                                  y_wall_lower_mode=Y_WALL_LOWER_MODE,
                                  x_wall_mode=X_WALL_MODE,
                                  immobility_filter_mode=IMMOBILITY_FILTER_MODE):
    n = end - start + 1
    excluded = np.zeros(n, dtype=bool)
    if floor_z_filter_mode == "exclude_frames":
        excluded |= filter_masks['floor_violation'][start:end+1]
    if y_wall_upper_mode == "exclude_frames":
        excluded |= filter_masks['y_wall_upper_violation'][start:end+1]
    if y_wall_lower_mode == "exclude_frames":
        excluded |= filter_masks['y_wall_lower_violation'][start:end+1]
    if x_wall_mode == "exclude_frames":
        excluded |= filter_masks['x_wall_violation'][start:end+1]
    return excluded


# ── Swing phase detection ─────────────────────────────────────────────────────

def detect_swing_phases(z_signal, max_swing_duration=MAX_SWING_DURATION,
                        min_prominence=SWING_PROMINENCE):
    z_clean = z_signal.copy()
    nan_mask = np.isnan(z_clean)
    if nan_mask.all():
        return np.array([]), 0
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        z_clean[nan_mask] = np.interp(np.where(nan_mask)[0], valid_idx, z_clean[valid_idx])
    z_smooth = uniform_filter1d(z_clean, size=5)
    peaks, _ = find_peaks(z_smooth, prominence=min_prominence, distance=8,
                          width=(1, max_swing_duration))
    return peaks, len(peaks)


def count_swing_phases_all_legs(leg_tip_data, start, end,
                                 max_swing_duration=MAX_SWING_DURATION,
                                 min_prominence=SWING_PROMINENCE):
    per_leg_cycles = {}
    for tip in LEG_TIPS:
        z_bout = leg_tip_data[tip]['z'][start:end+1]
        _, swing_count = detect_swing_phases(z_bout, max_swing_duration, min_prominence)
        per_leg_cycles[tip] = swing_count
    min_cycles = min(per_leg_cycles.values()) if per_leg_cycles else 0
    return per_leg_cycles, min_cycles


# ── Distance computation ──────────────────────────────────────────────────────

def compute_total_distance(scutellum_data, start, end):
    x = scutellum_data['x'][start:end+1]
    y = scutellum_data['y'][start:end+1]
    valid = ~np.isnan(x) & ~np.isnan(y)
    if valid.sum() < 2:
        return 0.0, 0.0
    x_valid, y_valid = x[valid], y[valid]
    dx, dy = np.diff(x_valid), np.diff(y_valid)
    total_distance = float(np.sum(np.sqrt(dx**2 + dy**2)))
    net_displacement = float(np.sqrt((x_valid[-1] - x_valid[0])**2 + (y_valid[-1] - y_valid[0])**2))
    return total_distance, net_displacement


# ── Bout validation ───────────────────────────────────────────────────────────

def validate_walking_bouts(
    bouts,
    leg_tip_data,
    scutellum_data,
    min_cycles=MIN_WALKING_CYCLES,
    min_distance_mm=MIN_DISTANCE_MM,
    max_swing_duration=MAX_SWING_DURATION,
    min_prominence=SWING_PROMINENCE,
    enforce_straightness=ENFORCE_STRAIGHTNESS,
    straightness_threshold=STRAIGHTNESS_THRESHOLD,
):
    valid_bouts = []
    rejected_bouts = []

    for start, end in bouts:
        per_leg_cycles, min_leg_cycles = count_swing_phases_all_legs(
            leg_tip_data, start, end, max_swing_duration, min_prominence
        )
        total_distance, net_displacement = compute_total_distance(scutellum_data, start, end)
        straightness = net_displacement / total_distance if total_distance > 0 else 0

        bout_info = {
            'start': start,
            'end': end,
            'n_frames': end - start + 1,
            'min_cycles': min_leg_cycles,
            'per_leg_cycles': per_leg_cycles,
            'total_distance_mm': total_distance,
            'net_displacement_mm': net_displacement,
            'straightness': straightness,
        }

        all_legs_pass = all(c >= min_cycles for c in per_leg_cycles.values())
        passes_distance = total_distance >= min_distance_mm
        passes_straightness = (not enforce_straightness) or (straightness >= straightness_threshold)

        if all_legs_pass and passes_distance and passes_straightness:
            bout_info['valid'] = True
            valid_bouts.append(bout_info)
        else:
            bout_info['valid'] = False
            reasons = []
            if not all_legs_pass:
                failing = [f"{leg}:{c}" for leg, c in per_leg_cycles.items() if c < min_cycles]
                reasons.append(f"legs with <{min_cycles} cycles: {failing}")
            if not passes_distance:
                reasons.append(f"distance: {total_distance:.2f}mm < {min_distance_mm}mm")
            if not passes_straightness:
                reasons.append(f"straightness: {straightness:.2f} < {straightness_threshold}")
            bout_info['rejection_reason'] = reasons
            rejected_bouts.append(bout_info)

    return valid_bouts, rejected_bouts


# ── Main pipeline ─────────────────────────────────────────────────────────────

def detect_walking_bouts(
    df,
    scale=JARVIS_SCALE,
    confidence_filter_enabled=CONFIDENCE_FILTER_ENABLED,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    confidence_gap_bridge=CONFIDENCE_GAP_BRIDGE,
    upright_filter_enabled=UPRIGHT_FILTER_ENABLED,
    floor_z_filter_enabled=FLOOR_Z_FILTER_ENABLED,
    floor_z_filter_mode=FLOOR_Z_FILTER_MODE,
    floor_z_threshold=FLOOR_Z_THRESHOLD,
    y_wall_upper_enabled=Y_WALL_UPPER_ENABLED,
    y_wall_upper_mode=Y_WALL_UPPER_MODE,
    y_wall_max=Y_WALL_MAX,
    y_wall_upper_margin=Y_WALL_UPPER_MARGIN,
    y_wall_lower_enabled=Y_WALL_LOWER_ENABLED,
    y_wall_lower_mode=Y_WALL_LOWER_MODE,
    y_wall_min=Y_WALL_MIN,
    y_wall_lower_margin=Y_WALL_LOWER_MARGIN,
    x_wall_enabled=X_WALL_ENABLED,
    x_wall_mode=X_WALL_MODE,
    x_wall_margin=X_WALL_MARGIN,
    x_wall_tips=X_WALL_TIPS,
    arena_x_mm=ARENA_X_MM,
    immobility_filter_enabled=IMMOBILITY_FILTER_ENABLED,
    immobility_filter_mode=IMMOBILITY_FILTER_MODE,
    max_stationary_frames=MAX_STATIONARY_FRAMES,
    stationary_speed_threshold=STATIONARY_SPEED_THRESHOLD,
    min_bout_frames=MIN_BOUT_FRAMES,
    max_immobility_gap=MAX_IMMOBILITY_FRAMES,
    min_walking_cycles=MIN_WALKING_CYCLES,
    min_distance_mm=MIN_DISTANCE_MM,
    max_swing_duration=MAX_SWING_DURATION,
    swing_prominence=SWING_PROMINENCE,
    enforce_straightness=ENFORCE_STRAIGHTNESS,
    straightness_threshold=STRAIGHTNESS_THRESHOLD,
    show_boundary_diagnostics=SHOW_BOUNDARY_DIAGNOSTICS,
    boundary_diagnostic_frames=BOUNDARY_DIAGNOSTIC_FRAMES,
    verbose=True,
):
    """Complete walking bout detection pipeline.

    Returns:
        valid_bouts, rejected_bouts, leg_tip_data, scutellum_data, filter_masks, diagnostics
    """
    if verbose:
        print("=" * 70)
        print("WALKING BOUT DETECTION PIPELINE")
        print("=" * 70)

    # Step 1: frame-level filters
    valid_mask, leg_tip_data, scutellum_data, filter_masks, diagnostics = (
        apply_comprehensive_walking_filter(
            df, scale,
            confidence_filter_enabled, confidence_threshold, confidence_gap_bridge,
            upright_filter_enabled,
            floor_z_filter_enabled, floor_z_filter_mode, floor_z_threshold,
            y_wall_upper_enabled, y_wall_upper_mode, y_wall_max, y_wall_upper_margin,
            y_wall_lower_enabled, y_wall_lower_mode, y_wall_min, y_wall_lower_margin,
            x_wall_enabled, x_wall_mode, x_wall_margin, x_wall_tips, arena_x_mm,
        )
    )

    if verbose:
        print(f"\n  Total frames: {diagnostics['total_frames']}")
        print(f"  Confidence pass: {diagnostics['confidence_pass_pct']:.1f}% "
              f"(bridged: {diagnostics['confidence_frames_bridged']})")
        print(f"  Combined valid: {diagnostics['combined_pass_pct']:.1f}%")

    # Step 2: find candidate bouts with gap bridging
    candidate_bouts = find_contiguous_bouts_with_gap_bridging(
        valid_mask, min_bout_frames, max_immobility_gap
    )

    if verbose:
        print(f"\n  Candidate bouts: {len(candidate_bouts)}")

    # Step 3: split at violations
    split_candidate_bouts, _ = split_bouts_at_violations(
        candidate_bouts, leg_tip_data, scutellum_data, filter_masks, FPS,
        floor_z_filter_enabled, floor_z_filter_mode,
        y_wall_upper_enabled, y_wall_upper_mode,
        y_wall_lower_enabled, y_wall_lower_mode,
        x_wall_enabled, x_wall_mode,
        immobility_filter_enabled, immobility_filter_mode,
        max_stationary_frames, stationary_speed_threshold,
        min_bout_frames,
    )

    if verbose:
        print(f"  After splitting: {len(split_candidate_bouts)}")

    # Step 4: validate
    valid_bouts, rejected_bouts = validate_walking_bouts(
        split_candidate_bouts, leg_tip_data, scutellum_data,
        min_walking_cycles, min_distance_mm,
        max_swing_duration, swing_prominence,
        enforce_straightness, straightness_threshold,
    )

    # Step 5: add excluded frame masks and bout_idx
    for i, bout in enumerate(valid_bouts):
        excluded_mask = get_excluded_frames_for_bout(
            bout['start'], bout['end'], filter_masks,
            floor_z_filter_mode, y_wall_upper_mode, y_wall_lower_mode,
            x_wall_mode, immobility_filter_mode,
        )
        bout['excluded_frame_mask'] = excluded_mask
        bout['n_excluded_frames'] = int(excluded_mask.sum())
        bout['n_valid_frames'] = bout['n_frames'] - bout['n_excluded_frames']
        bout['bout_idx'] = i + 1

    if verbose:
        print(f"\n  Valid bouts: {len(valid_bouts)}")
        print(f"  Rejected bouts: {len(rejected_bouts)}")
        print("=" * 70)

    diagnostics['n_valid_bouts'] = len(valid_bouts)
    diagnostics['n_rejected_bouts'] = len(rejected_bouts)

    return valid_bouts, rejected_bouts, leg_tip_data, scutellum_data, filter_masks, diagnostics
