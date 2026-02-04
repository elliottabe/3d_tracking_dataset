"""
Enhanced Walking Cycle Detection with Body Displacement Validation

This module provides functions to validate walking cycles by checking that
leg swing phases are accompanied by actual body displacement.

A valid walking cycle requires BOTH:
1. Leg Z peak (swing phase)
2. Body displacement during that swing window

This filters out cases where the fly is just moving its legs without walking.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


def detect_swing_phases_with_displacement(
    z_signal,
    scutellum_xy,
    max_swing_duration=40,
    min_prominence=0.05,
    min_displacement_per_cycle=0.1,
    window_frames=20
):
    """
    Detect swing phases that are accompanied by body displacement.

    A valid walking swing is:
    1. A peak in leg Z (swing phase)
    2. Body displacement during that swing cycle window

    Args:
        z_signal: Z coordinate array for one leg tip (N,)
        scutellum_xy: (N, 2) array of scutellum [x, y] positions
        max_swing_duration: Maximum frames for a valid swing
        min_prominence: Minimum peak prominence (mm)
        min_displacement_per_cycle: Minimum body displacement (mm) during swing window
        window_frames: Frames around each peak to measure displacement

    Returns:
        valid_peaks: Array of peak indices that have body displacement
        all_peaks: All detected peaks (before displacement filter)
        displacements: Body displacement for each peak
        valid_count: Number of valid walking cycles
    """
    # Handle NaN values in Z signal
    z_clean = z_signal.copy()
    nan_mask = np.isnan(z_clean)

    if nan_mask.all():
        return np.array([]), np.array([]), np.array([]), 0

    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        z_clean[nan_mask] = np.interp(
            np.where(nan_mask)[0],
            valid_idx,
            z_clean[valid_idx]
        )

    # Light smoothing
    z_smooth = uniform_filter1d(z_clean, size=5)

    # Find all peaks
    all_peaks, properties = find_peaks(
        z_smooth,
        prominence=min_prominence,
        distance=8,
        width=(1, max_swing_duration)
    )

    if len(all_peaks) == 0:
        return np.array([]), np.array([]), np.array([]), 0

    # For each peak, compute body displacement in a window around it
    valid_peaks = []
    displacements = []
    n_frames = len(z_signal)

    for peak_idx in all_peaks:
        # Define window around peak
        win_start = max(0, peak_idx - window_frames)
        win_end = min(n_frames - 1, peak_idx + window_frames)

        # Get scutellum positions in window
        scut_x = scutellum_xy[win_start:win_end+1, 0]
        scut_y = scutellum_xy[win_start:win_end+1, 1]

        # Handle NaN
        valid_scut = ~np.isnan(scut_x) & ~np.isnan(scut_y)
        if valid_scut.sum() < 2:
            displacements.append(0.0)
            continue

        scut_x_valid = scut_x[valid_scut]
        scut_y_valid = scut_y[valid_scut]

        # Compute total path distance in window (cumulative displacement)
        dx = np.diff(scut_x_valid)
        dy = np.diff(scut_y_valid)
        path_distance = np.sum(np.sqrt(dx**2 + dy**2))

        displacements.append(path_distance)

        # Check if displacement is sufficient
        if path_distance >= min_displacement_per_cycle:
            valid_peaks.append(peak_idx)

    return np.array(valid_peaks), all_peaks, np.array(displacements), len(valid_peaks)


def count_valid_walking_cycles_all_legs(
    leg_tip_data,
    scutellum_data,
    start,
    end,
    max_swing_duration=40,
    min_prominence=0.05,
    min_displacement_per_cycle=0.1,
    window_frames=20,
    leg_tips=None
):
    """
    Count valid walking cycles for all 6 legs.

    A valid walking cycle requires BOTH:
    1. Leg Z peak (swing phase)
    2. Body displacement during that swing

    Args:
        leg_tip_data: Dict mapping leg tip names to dicts with 'x', 'y', 'z' arrays
        scutellum_data: Dict with 'x', 'y', 'z' arrays for scutellum
        start, end: Bout frame boundaries
        max_swing_duration: Maximum frames for swing detection
        min_prominence: Peak prominence threshold (mm)
        min_displacement_per_cycle: Minimum body displacement per cycle (mm)
        window_frames: Frames around peak to measure displacement
        leg_tips: List of leg tip keypoint names (default: all 6)

    Returns:
        per_leg_valid_cycles: Dict of valid cycle counts per leg
        per_leg_total_peaks: Dict of total peaks per leg (without displacement filter)
        min_valid_cycles: Minimum valid cycles across all legs
        cycle_details: Dict with peak and displacement info per leg
    """
    if leg_tips is None:
        leg_tips = ['T1L_TaTip', 'T2L_TaTip', 'T3L_TaTip',
                    'T1R_TaTip', 'T2R_TaTip', 'T3R_TaTip']

    per_leg_valid_cycles = {}
    per_leg_total_peaks = {}
    cycle_details = {}

    # Get scutellum XY for the bout
    scut_x = scutellum_data['x'][start:end+1]
    scut_y = scutellum_data['y'][start:end+1]
    scutellum_xy = np.column_stack([scut_x, scut_y])

    for tip in leg_tips:
        z_bout = leg_tip_data[tip]['z'][start:end+1]

        valid_peaks, all_peaks, displacements, valid_count = detect_swing_phases_with_displacement(
            z_bout,
            scutellum_xy,
            max_swing_duration,
            min_prominence,
            min_displacement_per_cycle,
            window_frames
        )

        per_leg_valid_cycles[tip] = valid_count
        per_leg_total_peaks[tip] = len(all_peaks)
        cycle_details[tip] = {
            'valid_peaks': valid_peaks + start,  # Convert to global frame indices
            'all_peaks': all_peaks + start,
            'displacements': displacements,
            'valid_count': valid_count,
            'total_peaks': len(all_peaks)
        }

    min_valid_cycles = min(per_leg_valid_cycles.values()) if per_leg_valid_cycles else 0

    return per_leg_valid_cycles, per_leg_total_peaks, min_valid_cycles, cycle_details


def validate_walking_bouts_with_displacement(
    bouts,
    leg_tip_data,
    scutellum_data,
    min_cycles=2,
    min_distance_mm=3.0,
    max_swing_duration=40,
    min_prominence=0.05,
    min_displacement_per_cycle=0.1,
    window_frames=20,
    enforce_straightness=False,
    straightness_threshold=0.2,
    verbose=True
):
    """
    Validate candidate bouts with the enhanced displacement-aware cycle counting.

    Criteria:
    - Each leg must have >= min_cycles VALID swing phases (peaks WITH body displacement)
    - Total path distance >= min_distance_mm
    - Straightness >= threshold (if enforce_straightness is True)

    Args:
        bouts: List of (start, end) tuples for candidate bouts
        leg_tip_data: Dict of leg data
        scutellum_data: Dict of scutellum data
        min_cycles: Minimum valid walking cycles per leg
        min_distance_mm: Minimum total path distance
        max_swing_duration: Max frames for swing detection
        min_prominence: Peak prominence threshold
        min_displacement_per_cycle: Min body displacement per swing (mm)
        window_frames: Frames around peak to measure displacement
        enforce_straightness: Whether to filter by straightness
        straightness_threshold: Min straightness ratio
        verbose: Print rejection reasons

    Returns:
        valid_bouts: List of validated bout dicts with metadata
        rejected_bouts: List of rejected bout dicts with rejection reason
    """
    valid_bouts = []
    rejected_bouts = []

    for start, end in bouts:
        # Count valid walking cycles (with displacement check)
        per_leg_valid, per_leg_total, min_leg_valid, cycle_details = count_valid_walking_cycles_all_legs(
            leg_tip_data, scutellum_data, start, end,
            max_swing_duration, min_prominence,
            min_displacement_per_cycle, window_frames
        )

        # Also count original peaks (without displacement filter) for comparison
        # This is already in per_leg_total

        # Compute distance
        scut_x = scutellum_data['x'][start:end+1]
        scut_y = scutellum_data['y'][start:end+1]
        valid_mask = ~np.isnan(scut_x) & ~np.isnan(scut_y)

        if valid_mask.sum() < 2:
            total_distance = 0.0
            net_displacement = 0.0
        else:
            x_valid = scut_x[valid_mask]
            y_valid = scut_y[valid_mask]
            dx = np.diff(x_valid)
            dy = np.diff(y_valid)
            total_distance = np.sum(np.sqrt(dx**2 + dy**2))
            net_displacement = np.sqrt(
                (x_valid[-1] - x_valid[0])**2 +
                (y_valid[-1] - y_valid[0])**2
            )

        straightness = net_displacement / total_distance if total_distance > 0 else 0

        bout_info = {
            'start': start,
            'end': end,
            'n_frames': end - start + 1,
            'min_valid_cycles': min_leg_valid,
            'per_leg_valid_cycles': per_leg_valid,
            'per_leg_total_peaks': per_leg_total,
            'cycle_details': cycle_details,
            'total_distance_mm': total_distance,
            'net_displacement_mm': net_displacement,
            'straightness': straightness
        }

        # Check if ALL legs have enough VALID cycles
        all_legs_pass = all(
            cycles >= min_cycles for cycles in per_leg_valid.values()
        )
        passes_distance = total_distance >= min_distance_mm
        passes_straightness = (not enforce_straightness) or (straightness >= straightness_threshold)

        if all_legs_pass and passes_distance and passes_straightness:
            bout_info['valid'] = True
            valid_bouts.append(bout_info)
        else:
            # Determine rejection reason
            reasons = []
            if not all_legs_pass:
                failing_legs = [leg for leg, cyc in per_leg_valid.items() if cyc < min_cycles]
                # Show how many peaks were rejected due to no displacement
                rejected_info = []
                for leg in failing_legs:
                    total = per_leg_total[leg]
                    valid = per_leg_valid[leg]
                    rejected_info.append(f"{leg}: {valid}/{total} valid")
                reasons.append(f"Insufficient valid cycles: {', '.join(rejected_info)}")
            if not passes_distance:
                reasons.append(f"Distance {total_distance:.2f}mm < {min_distance_mm}mm")
            if not passes_straightness:
                reasons.append(f"Straightness {straightness:.2f} < {straightness_threshold}")

            bout_info['valid'] = False
            bout_info['rejection_reason'] = '; '.join(reasons)
            rejected_bouts.append(bout_info)

            if verbose:
                print(f"Rejected bout [{start}-{end}]: {bout_info['rejection_reason']}")

    return valid_bouts, rejected_bouts


def detect_grooming_frames(
    leg_tip_data,
    start,
    end,
    z_lift_percentile=75,
    min_overlap_frames=3,
    smooth_window=5
):
    """
    Detect grooming behavior based on simultaneous leg lifting.

    Grooming is characterized by:
    - Front grooming: T1L and T1R lifted at the same time
    - Rear grooming: T3L and T3R lifted at the same time

    During normal walking, contralateral legs alternate - they should NOT
    be lifted simultaneously.

    Args:
        leg_tip_data: Dict mapping leg tip names to dicts with 'x', 'y', 'z' arrays
        start, end: Bout frame boundaries
        z_lift_percentile: Percentile threshold for "lifted" detection.
            A leg is considered lifted when Z > this percentile of its own Z values.
        min_overlap_frames: Minimum consecutive frames of overlap to count as grooming
        smooth_window: Smoothing window for Z signal

    Returns:
        grooming_mask: Boolean array of shape (end-start+1,) where True = grooming frame
        front_grooming_mask: Boolean mask for front leg grooming (T1L+T1R)
        rear_grooming_mask: Boolean mask for rear leg grooming (T3L+T3R)
        grooming_info: Dict with diagnostic information
    """
    n_frames = end - start + 1

    # Leg pairs to check for grooming
    leg_pairs = {
        'front': ('T1L_TaTip', 'T1R_TaTip'),
        'rear': ('T3L_TaTip', 'T3R_TaTip')
    }

    # Detect "lifted" state for each relevant leg
    lifted_masks = {}
    lift_thresholds = {}

    for pair_name, (leg_L, leg_R) in leg_pairs.items():
        for leg in [leg_L, leg_R]:
            z = leg_tip_data[leg]['z'][start:end+1].copy()

            # Handle NaN
            nan_mask = np.isnan(z)
            if nan_mask.all():
                lifted_masks[leg] = np.zeros(n_frames, dtype=bool)
                lift_thresholds[leg] = np.nan
                continue

            if nan_mask.any():
                valid_idx = np.where(~nan_mask)[0]
                z[nan_mask] = np.interp(
                    np.where(nan_mask)[0],
                    valid_idx,
                    z[valid_idx]
                )

            # Smooth the signal
            z_smooth = uniform_filter1d(z, size=smooth_window)

            # Compute threshold: leg is "lifted" when above this percentile
            threshold = np.nanpercentile(z_smooth, z_lift_percentile)
            lift_thresholds[leg] = threshold

            # Detect lifted frames
            lifted_masks[leg] = z_smooth > threshold

    # Detect grooming: both legs in a pair lifted simultaneously
    front_grooming_mask = np.zeros(n_frames, dtype=bool)
    rear_grooming_mask = np.zeros(n_frames, dtype=bool)

    leg_L, leg_R = leg_pairs['front']
    front_overlap = lifted_masks[leg_L] & lifted_masks[leg_R]

    leg_L, leg_R = leg_pairs['rear']
    rear_overlap = lifted_masks[leg_L] & lifted_masks[leg_R]

    # Filter by minimum consecutive frames
    if min_overlap_frames > 1:
        front_grooming_mask = _filter_short_segments(front_overlap, min_overlap_frames)
        rear_grooming_mask = _filter_short_segments(rear_overlap, min_overlap_frames)
    else:
        front_grooming_mask = front_overlap
        rear_grooming_mask = rear_overlap

    # Combined grooming mask
    grooming_mask = front_grooming_mask | rear_grooming_mask

    # Diagnostic info
    grooming_info = {
        'front_grooming_frames': np.sum(front_grooming_mask),
        'rear_grooming_frames': np.sum(rear_grooming_mask),
        'total_grooming_frames': np.sum(grooming_mask),
        'grooming_fraction': np.sum(grooming_mask) / n_frames,
        'lift_thresholds': lift_thresholds,
        'lifted_masks': lifted_masks
    }

    return grooming_mask, front_grooming_mask, rear_grooming_mask, grooming_info


def _filter_short_segments(mask, min_length):
    """
    Filter out segments shorter than min_length.

    Args:
        mask: Boolean array
        min_length: Minimum consecutive True values to keep

    Returns:
        Filtered boolean array
    """
    if min_length <= 1:
        return mask

    result = np.zeros_like(mask)
    in_segment = False
    segment_start = 0

    for i in range(len(mask)):
        if mask[i]:
            if not in_segment:
                in_segment = True
                segment_start = i
        else:
            if in_segment:
                segment_length = i - segment_start
                if segment_length >= min_length:
                    result[segment_start:i] = True
                in_segment = False

    # Handle segment at end
    if in_segment:
        segment_length = len(mask) - segment_start
        if segment_length >= min_length:
            result[segment_start:] = True

    return result


def compute_grooming_fraction(
    leg_tip_data,
    start,
    end,
    z_lift_percentile=75,
    min_overlap_frames=3
):
    """
    Compute the fraction of frames in a bout that show grooming behavior.

    Args:
        leg_tip_data: Dict mapping leg tip names to dicts with 'z' arrays
        start, end: Bout frame boundaries
        z_lift_percentile: Percentile threshold for "lifted" detection
        min_overlap_frames: Minimum consecutive frames to count as grooming

    Returns:
        grooming_fraction: Fraction of frames classified as grooming (0.0 to 1.0)
        grooming_info: Dict with detailed breakdown
    """
    grooming_mask, front_mask, rear_mask, info = detect_grooming_frames(
        leg_tip_data, start, end, z_lift_percentile, min_overlap_frames
    )
    return info['grooming_fraction'], info


def validate_bouts_no_grooming(
    bouts,
    leg_tip_data,
    max_grooming_fraction=0.1,
    z_lift_percentile=75,
    min_overlap_frames=3,
    verbose=True
):
    """
    Filter out bouts that contain significant grooming behavior.

    A bout is rejected if the grooming fraction exceeds max_grooming_fraction.

    Args:
        bouts: List of (start, end) tuples or dicts with 'start'/'end' keys
        leg_tip_data: Dict of leg data
        max_grooming_fraction: Maximum allowed fraction of grooming frames
        z_lift_percentile: Percentile threshold for lift detection
        min_overlap_frames: Minimum overlap frames to count as grooming
        verbose: Print rejection info

    Returns:
        valid_bouts: Bouts that pass the grooming filter
        rejected_bouts: Bouts rejected due to grooming
        grooming_info_per_bout: Dict mapping bout index to grooming info
    """
    valid_bouts = []
    rejected_bouts = []
    grooming_info_per_bout = {}

    for i, bout in enumerate(bouts):
        # Handle both tuple and dict formats
        if isinstance(bout, tuple):
            start, end = bout
        else:
            start, end = bout['start'], bout['end']

        grooming_fraction, info = compute_grooming_fraction(
            leg_tip_data, start, end, z_lift_percentile, min_overlap_frames
        )
        grooming_info_per_bout[i] = info

        if grooming_fraction <= max_grooming_fraction:
            valid_bouts.append(bout)
        else:
            rejected_bouts.append(bout)
            if verbose:
                print(f"Rejected bout [{start}-{end}]: grooming fraction "
                      f"{grooming_fraction:.1%} > {max_grooming_fraction:.1%} "
                      f"(front: {info['front_grooming_frames']} frames, "
                      f"rear: {info['rear_grooming_frames']} frames)")

    return valid_bouts, rejected_bouts, grooming_info_per_bout


def print_cycle_comparison(bout_info):
    """
    Print comparison of total peaks vs valid cycles for a bout.

    Helps diagnose how many leg movements were filtered out due to
    lack of body displacement.
    """
    print(f"\nBout [{bout_info['start']}-{bout_info['end']}] ({bout_info['n_frames']} frames)")
    print(f"  Total distance: {bout_info['total_distance_mm']:.2f} mm")
    print(f"  Leg cycle analysis (valid/total peaks):")

    for leg in sorted(bout_info['per_leg_valid_cycles'].keys()):
        valid = bout_info['per_leg_valid_cycles'][leg]
        total = bout_info['per_leg_total_peaks'][leg]
        pct = 100 * valid / total if total > 0 else 0
        status = "✓" if valid >= 2 else "✗"
        print(f"    {status} {leg}: {valid}/{total} ({pct:.0f}% with displacement)")


def plot_grooming_diagnosis(
    leg_tip_data,
    start,
    end,
    z_lift_percentile=75,
    min_overlap_frames=3,
    figsize=(14, 8),
    title=None
):
    """
    Plot diagnostic visualization for grooming detection in a bout.

    Shows Z traces for front and rear leg pairs with lifted regions
    and detected grooming (overlap) highlighted.

    Args:
        leg_tip_data: Dict mapping leg tip names to dicts with 'z' arrays
        start, end: Bout frame boundaries
        z_lift_percentile: Percentile threshold for lift detection
        min_overlap_frames: Minimum overlap frames for grooming
        figsize: Figure size
        title: Optional title

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    grooming_mask, front_mask, rear_mask, info = detect_grooming_frames(
        leg_tip_data, start, end, z_lift_percentile, min_overlap_frames
    )

    n_frames = end - start + 1
    frames = np.arange(start, end + 1)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Front legs (T1)
    ax = axes[0]
    z_T1L = leg_tip_data['T1L_TaTip']['z'][start:end+1]
    z_T1R = leg_tip_data['T1R_TaTip']['z'][start:end+1]

    ax.plot(frames, z_T1L, 'b-', label='T1L (left front)', alpha=0.8)
    ax.plot(frames, z_T1R, 'r-', label='T1R (right front)', alpha=0.8)

    # Shade lifted regions
    lifted_T1L = info['lifted_masks']['T1L_TaTip']
    lifted_T1R = info['lifted_masks']['T1R_TaTip']

    ax.fill_between(frames, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=lifted_T1L, alpha=0.15, color='blue', label='T1L lifted')
    ax.fill_between(frames, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=lifted_T1R, alpha=0.15, color='red', label='T1R lifted')

    # Highlight grooming (overlap)
    ax.fill_between(frames, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=front_mask, alpha=0.4, color='purple',
                    label=f'GROOMING ({info["front_grooming_frames"]} frames)')

    # Threshold lines
    ax.axhline(info['lift_thresholds']['T1L_TaTip'], color='blue', linestyle='--', alpha=0.5)
    ax.axhline(info['lift_thresholds']['T1R_TaTip'], color='red', linestyle='--', alpha=0.5)

    ax.set_ylabel('Z (mm)')
    ax.set_title(f'Front Legs (T1L + T1R) - Grooming = both lifted')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Rear legs (T3)
    ax = axes[1]
    z_T3L = leg_tip_data['T3L_TaTip']['z'][start:end+1]
    z_T3R = leg_tip_data['T3R_TaTip']['z'][start:end+1]

    ax.plot(frames, z_T3L, 'b-', label='T3L (left rear)', alpha=0.8)
    ax.plot(frames, z_T3R, 'r-', label='T3R (right rear)', alpha=0.8)

    # Shade lifted regions
    lifted_T3L = info['lifted_masks']['T3L_TaTip']
    lifted_T3R = info['lifted_masks']['T3R_TaTip']

    ax.fill_between(frames, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=lifted_T3L, alpha=0.15, color='blue', label='T3L lifted')
    ax.fill_between(frames, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=lifted_T3R, alpha=0.15, color='red', label='T3R lifted')

    # Highlight grooming (overlap)
    ax.fill_between(frames, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=rear_mask, alpha=0.4, color='purple',
                    label=f'GROOMING ({info["rear_grooming_frames"]} frames)')

    # Threshold lines
    ax.axhline(info['lift_thresholds']['T3L_TaTip'], color='blue', linestyle='--', alpha=0.5)
    ax.axhline(info['lift_thresholds']['T3R_TaTip'], color='red', linestyle='--', alpha=0.5)

    ax.set_ylabel('Z (mm)')
    ax.set_xlabel('Frame')
    ax.set_title(f'Rear Legs (T3L + T3R) - Grooming = both lifted')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    if title is None:
        title = f'Grooming Detection: frames [{start}-{end}]'
    fig.suptitle(f"{title}\nTotal grooming: {info['grooming_fraction']:.1%} "
                 f"({info['total_grooming_frames']}/{n_frames} frames)",
                 fontsize=12)
    plt.tight_layout()

    return fig, axes
