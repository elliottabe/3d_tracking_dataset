"""Visualization functions for walking and courtship bout figures.

Extracted from notebooks/Sandbox_Strict.ipynb for headless/batch use.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

from utils.bout_detection import (
    FPS, LEG_TIPS, ARENA_X_MM, ARENA_Y_MM,
    FLOOR_Z_THRESHOLD, STATIONARY_SPEED_THRESHOLD,
    Y_WALL_MAX, Y_WALL_UPPER_MARGIN, Y_WALL_MIN, Y_WALL_LOWER_MARGIN,
    compute_instant_speed,
)
from utils.courtship_detection import (
    COURT_WING_TIPS, COURT_ALL_LEG_TIPS, COURT_CONF_THRESHOLD,
    COURT_PULSE_CLASSIFY_SPEED, COURT_ARENA_X_MM, COURT_ARENA_Y_MM,
)

# Constants not present in existing utils
STANCE_Z_THRESHOLD = 0.55   # mm — leg in stance when Z <= this
SHOW_GAIT_DIAGRAM = False    # include gait phase panel in walking figures
DEFAULT_DPI = 150            # output DPI for batch PNGs

# Courtship segment shading
_SEG_COLORS = {
    'pulse':  '#E76F5133',
    'sine':   '#2A9D8F33',
    'waggle': '#E9C46A33',
}
_SEG_EDGE_COLORS = {
    'pulse':  '#E76F51',
    'sine':   '#2A9D8F',
    'waggle': '#E9C46A',
}

Z_PLOT_MIN = 0.0
Z_PLOT_MAX = 5.0


# ── shared helpers ────────────────────────────────────────────────────────────

def _add_bout_boundaries(ax, actual_start, actual_end, add_legend=False):
    """Add bout boundary shading and vertical lines to an axis."""
    ax.axvspan(actual_start, actual_end, color='#2ca02c', alpha=0.08, zorder=0,
               label='Detected bout' if add_legend else None)
    ax.axvline(x=actual_start, color='#d62728', ls='--', lw=1.5, alpha=0.8, zorder=3,
               label='Bout start/end' if add_legend else None)
    ax.axvline(x=actual_end,   color='#d62728', ls='--', lw=1.5, alpha=0.8, zorder=3)


# ── walking helpers ───────────────────────────────────────────────────────────

def _format_boundary_cause(cause, failures_dict, check_start, check_end, leg_tip_data):
    if cause is None:
        return None
    n_analyzed = failures_dict.get('n_frames_analyzed', 0)
    if cause == 'confidence':
        n_fail = failures_dict.get('confidence', 0)
        return f'Low confidence ({n_fail}/{n_analyzed}f)'
    elif cause == 'y_wall_upper' and leg_tip_data is not None:
        max_y, worst_leg = -np.inf, ''
        for tip in LEG_TIPS:
            y_vals = leg_tip_data[tip]['y'][check_start:check_end]
            if len(y_vals) > 0:
                tip_max = np.nanmax(y_vals)
                if tip_max > max_y:
                    max_y, worst_leg = tip_max, tip.replace('_TaTip', '')
        limit = Y_WALL_MAX - Y_WALL_UPPER_MARGIN
        if np.isfinite(max_y):
            return f'Near upper wall ({worst_leg} Y={max_y:.2f}, lim={limit:.2f}mm)'
        return f'Near upper wall ({failures_dict.get("y_wall_upper", 0)}/{n_analyzed}f)'
    elif cause == 'y_wall_lower' and leg_tip_data is not None:
        min_y, worst_leg = np.inf, ''
        for tip in LEG_TIPS:
            y_vals = leg_tip_data[tip]['y'][check_start:check_end]
            if len(y_vals) > 0:
                tip_min = np.nanmin(y_vals)
                if tip_min < min_y:
                    min_y, worst_leg = tip_min, tip.replace('_TaTip', '')
        limit = Y_WALL_MIN + Y_WALL_LOWER_MARGIN
        if np.isfinite(min_y):
            return f'Near lower wall ({worst_leg} Y={min_y:.2f}, lim={limit:.2f}mm)'
        return f'Near lower wall ({failures_dict.get("y_wall_lower", 0)}/{n_analyzed}f)'
    elif cause == 'floor' and leg_tip_data is not None:
        min_z, worst_leg = np.inf, ''
        for tip in LEG_TIPS:
            z_vals = leg_tip_data[tip]['z'][check_start:check_end]
            if len(z_vals) > 0:
                tip_min = np.nanmin(z_vals)
                if tip_min < min_z:
                    min_z, worst_leg = tip_min, tip.replace('_TaTip', '')
        if np.isfinite(min_z):
            return f'Floor Z ({worst_leg} Z={min_z:.2f}, lim={FLOOR_Z_THRESHOLD:.2f}mm)'
        return f'Floor Z violation ({failures_dict.get("floor", 0)}/{n_analyzed}f)'
    elif cause == 'upright':
        return f'Not upright ({failures_dict.get("upright", 0)}/{n_analyzed}f)'
    return cause


def _get_boundary_reasons(bout_info, speed, ctx_start, start, end, ctx_end, fps,
                           leg_tip_data=None):
    bd = bout_info.get('boundary_diagnostics', None)
    start_reason = end_reason = None
    n_check = 10

    if bd:
        pre  = bd.get('pre_bout_failures', {})
        post = bd.get('post_bout_failures', {})
        before_start = max(0, start - n_check)
        start_reason = _format_boundary_cause(
            bd.get('primary_cause_before'), pre, before_start, start, leg_tip_data)
        data_len = len(leg_tip_data[LEG_TIPS[0]]['y']) if leg_tip_data else end + n_check + 1
        after_end = min(data_len, end + 1 + n_check)
        end_reason = _format_boundary_cause(
            bd.get('primary_cause_after'), post, end + 1, after_end, leg_tip_data)

    if speed is not None and len(speed) > 0:
        pre_idx  = start - ctx_start
        post_idx = end   - ctx_start
        check_n = min(n_check, pre_idx)
        if check_n > 0 and start_reason is None:
            pre_speed = np.nanmean(speed[pre_idx - check_n:pre_idx])
            if pre_speed < STATIONARY_SPEED_THRESHOLD:
                start_reason = f'Prolonged immobility (speed={pre_speed:.1f}mm/s)'
        check_n = min(n_check, len(speed) - post_idx - 1)
        if check_n > 0 and end_reason is None:
            post_speed = np.nanmean(speed[post_idx + 1:post_idx + 1 + check_n])
            if post_speed < STATIONARY_SPEED_THRESHOLD:
                end_reason = f'Prolonged immobility (speed={post_speed:.1f}mm/s)'

    return start_reason, end_reason


def compute_stance_swing_phases(leg_tip_data, start, end,
                                 stance_threshold=STANCE_Z_THRESHOLD):
    phases = {}
    for tip in LEG_TIPS:
        z = leg_tip_data[tip]['z'][start:end + 1]
        phases[tip] = z <= stance_threshold
    return phases


def plot_gait_phase_diagram(leg_tip_data, start, end, frames,
                             stance_threshold=STANCE_Z_THRESHOLD,
                             ax=None, excluded_mask=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 3))

    phases = compute_stance_swing_phases(leg_tip_data, start, end, stance_threshold)
    leg_order = ['T3R_TaTip', 'T2R_TaTip', 'T1R_TaTip', 'T3L_TaTip', 'T2L_TaTip']
    leg_labels = ['R3', 'R2', 'R1', 'L3', 'L2']
    bar_height = 0.6

    if excluded_mask is not None and excluded_mask.any():
        in_ex = False
        ex_start = 0
        for i in range(len(frames)):
            if excluded_mask[i] and not in_ex:
                ex_start, in_ex = i, True
            elif not excluded_mask[i] and in_ex:
                ax.axvspan(frames[ex_start], frames[i - 1], color='lightgray', alpha=0.5, zorder=0)
                in_ex = False
        if in_ex:
            ax.axvspan(frames[ex_start], frames[-1], color='lightgray', alpha=0.5, zorder=0)

    for y_pos, (leg, label) in enumerate(zip(leg_order, leg_labels)):
        stance_mask = phases[leg]
        in_stance = False
        stance_start_idx = 0
        for i, is_stance in enumerate(stance_mask):
            if is_stance and not in_stance:
                stance_start_idx, in_stance = i, True
            elif not is_stance and in_stance:
                ax.barh(y_pos, frames[i - 1] - frames[stance_start_idx] + 1,
                        left=frames[stance_start_idx], height=bar_height,
                        color='black', edgecolor='black', linewidth=0.5)
                in_stance = False
        if in_stance:
            ax.barh(y_pos, frames[-1] - frames[stance_start_idx] + 1,
                    left=frames[stance_start_idx], height=bar_height,
                    color='black', edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(len(leg_order)))
    ax.set_yticklabels(leg_labels, fontsize=10)
    ax.set_ylabel('Leg', fontsize=11)
    ax.set_title(f'Gait Phase Diagram (black = stance, Z ≤ {stance_threshold}mm)', fontsize=12)
    ax.set_xlim(frames[0] - 5, frames[-1] + 5)
    ax.set_ylim(-0.5, len(leg_order) - 0.5)
    ax.grid(True, axis='x', alpha=0.3, linestyle=':')
    legend_elements = [
        Patch(facecolor='black', edgecolor='black', label='Stance'),
        Patch(facecolor='white', edgecolor='black', label='Swing'),
    ]
    if excluded_mask is not None and excluded_mask.any():
        legend_elements.append(
            Patch(facecolor='lightgray', alpha=0.5, edgecolor='gray', label='Excluded'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=len(legend_elements))
    return ax


# ── walking figure ────────────────────────────────────────────────────────────

def plot_walking_bout_figure(
    bout_info,
    leg_tip_data,
    scutellum_data,
    frame_offset=0,
    fps=FPS,
    arena_x_mm=ARENA_X_MM,
    arena_y_mm=ARENA_Y_MM,
    save_path=None,
    dpi=DEFAULT_DPI,
    show_gait_diagram=SHOW_GAIT_DIAGRAM,
    stance_threshold=STANCE_Z_THRESHOLD,
    context_frames=200,
):
    """Create 5-panel walking bout figure (leg Z, gait, body height, speed, XY)."""
    start    = bout_info['start']
    end      = bout_info['end']
    bout_idx = bout_info.get('bout_idx', '?')

    data_len  = len(scutellum_data['x'])
    ctx_start = max(0, start - context_frames)
    ctx_end   = min(data_len - 1, end + context_frames)

    actual_start     = start     + frame_offset
    actual_end       = end       + frame_offset
    actual_ctx_start = ctx_start + frame_offset
    actual_ctx_end   = ctx_end   + frame_offset
    frames = np.arange(actual_ctx_start, actual_ctx_end + 1)
    n_frames_bout = end - start + 1

    excluded_mask = bout_info.get('excluded_frame_mask', None)
    if excluded_mask is not None and context_frames > 0:
        pad_before = start - ctx_start
        pad_after  = ctx_end - end
        excluded_mask = np.concatenate([
            np.zeros(pad_before, dtype=bool),
            excluded_mask,
            np.zeros(pad_after,  dtype=bool),
        ])

    scut_x = scutellum_data['x'][ctx_start:ctx_end + 1]
    scut_y = scutellum_data['y'][ctx_start:ctx_end + 1]
    scut_z = scutellum_data['z'][ctx_start:ctx_end + 1]

    speed, _, _           = compute_instant_speed(scutellum_data, ctx_start, ctx_end, fps)
    _, mean_speed, max_speed = compute_instant_speed(scutellum_data, start, end, fps)

    start_reason, end_reason = _get_boundary_reasons(
        bout_info, speed, ctx_start, start, end, ctx_end, fps,
        leg_tip_data=leg_tip_data,
    )

    if show_gait_diagram:
        fig = plt.figure(figsize=(14, 17))
        gs  = fig.add_gridspec(5, 1, height_ratios=[1, 0.5, 0.7, 0.7, 1.2], hspace=0.35)
        ax1     = fig.add_subplot(gs[0])
        ax_gait = fig.add_subplot(gs[1], sharex=ax1)
        ax2     = fig.add_subplot(gs[2], sharex=ax1)
        ax3     = fig.add_subplot(gs[3], sharex=ax1)
        ax4     = fig.add_subplot(gs[4])
    else:
        fig = plt.figure(figsize=(14, 14))
        gs  = fig.add_gridspec(4, 1, height_ratios=[1, 0.7, 0.7, 1.2], hspace=0.35)
        ax1     = fig.add_subplot(gs[0])
        ax_gait = None
        ax2     = fig.add_subplot(gs[1], sharex=ax1)
        ax3     = fig.add_subplot(gs[2], sharex=ax1)
        ax4     = fig.add_subplot(gs[3])

    leg_colors = {
        'T1R_TaTip': '#457B9D',
        'T2L_TaTip': '#F4A261',
        'T2R_TaTip': '#2A9D8F',
        'T3L_TaTip': '#9B2226',
        'T3R_TaTip': '#1D3557',
    }

    # Panel 1: Leg Tip Z
    _add_bout_boundaries(ax1, actual_start, actual_end, add_legend=True)
    for tip, color in leg_colors.items():
        z_leg = leg_tip_data[tip]['z'][ctx_start:ctx_end + 1]
        ax1.plot(frames, z_leg, label=tip.replace('_TaTip', ''), color=color, lw=1.2, alpha=0.8)
    ax1.axhline(y=FLOOR_Z_THRESHOLD, color='gray', ls=':', lw=1.5, alpha=0.7,
                label=f'Floor ({FLOOR_Z_THRESHOLD}mm)')
    ax1.set_ylabel('Leg Tip Z (mm)', fontsize=12)
    ax1.set_title(f'Leg Tip Z Trajectories | Bout {bout_idx}', fontsize=13)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, ncol=1)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    if start_reason:
        ax1.annotate(
            start_reason, xy=(actual_start, 1), xycoords=('data', 'axes fraction'),
            xytext=(6, -8), textcoords='offset points',
            fontsize=8, color='#d62728', fontweight='bold', ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d62728', alpha=0.85))
    if end_reason:
        ax1.annotate(
            end_reason, xy=(actual_end, 1), xycoords=('data', 'axes fraction'),
            xytext=(-6, -8), textcoords='offset points',
            fontsize=8, color='#d62728', fontweight='bold', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d62728', alpha=0.85))

    # Panel 2 (optional): Gait phase
    if show_gait_diagram and ax_gait is not None:
        _add_bout_boundaries(ax_gait, actual_start, actual_end)
        plot_gait_phase_diagram(leg_tip_data, ctx_start, ctx_end, frames,
                                stance_threshold=stance_threshold, ax=ax_gait,
                                excluded_mask=excluded_mask)
        plt.setp(ax_gait.get_xticklabels(), visible=False)

    # Panel 3: Body height
    _add_bout_boundaries(ax2, actual_start, actual_end)
    ax2.plot(frames, scut_z, color='#2D3142', lw=1.8, label='Scutellum Z')
    ax2.fill_between(frames, np.nanmin(scut_z), scut_z, alpha=0.3, color='#2D3142')
    scut_z_mean = np.nanmean(scutellum_data['z'][start:end + 1])
    ax2.axhline(y=scut_z_mean, color='#E76F51', ls='--', lw=1.5,
                label=f'Bout mean: {scut_z_mean:.2f} mm')
    ax2.set_ylabel('Body Height Z (mm)', fontsize=12)
    ax2.set_title('Scutellum Z (Body Height)', fontsize=13)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Panel 4: Speed
    _add_bout_boundaries(ax3, actual_start, actual_end)
    ax3.plot(frames, speed, color='#264653', lw=1.2)
    ax3.axhline(y=mean_speed, color='#E76F51', ls='--', lw=1.5,
                label=f'Bout mean: {mean_speed:.1f} mm/s')
    ax3.fill_between(frames, 0, speed, alpha=0.3, color='#264653')
    ax3.set_ylabel('Speed (mm/s)', fontsize=12)
    ax3.set_xlabel('Frame', fontsize=11)
    ax3.set_title(f'Instantaneous Speed | Bout max: {max_speed:.1f} mm/s', fontsize=13)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    # Panel 5: XY trajectory
    rect_x = [0, arena_x_mm, arena_x_mm, 0, 0]
    rect_y = [0, 0, arena_y_mm, arena_y_mm, 0]
    ax4.plot(rect_x, rect_y, 'k-', lw=2, label=f'Arena ({arena_x_mm}×{arena_y_mm}mm)')

    pre_len = start - ctx_start
    if pre_len > 1:
        pre_x = scutellum_data['x'][ctx_start:start + 1]
        pre_y = scutellum_data['y'][ctx_start:start + 1]
        v = ~np.isnan(pre_x) & ~np.isnan(pre_y)
        if v.sum() > 1:
            ax4.plot(pre_x[v], pre_y[v], color='gray', ls='--', lw=1.2, alpha=0.5,
                     label='Context (before)')
    if ctx_end - end > 1:
        post_x = scutellum_data['x'][end:ctx_end + 1]
        post_y = scutellum_data['y'][end:ctx_end + 1]
        v = ~np.isnan(post_x) & ~np.isnan(post_y)
        if v.sum() > 1:
            ax4.plot(post_x[v], post_y[v], color='gray', ls=':', lw=1.2, alpha=0.5,
                     label='Context (after)')

    bout_x = scutellum_data['x'][start:end + 1]
    bout_y = scutellum_data['y'][start:end + 1]
    v = ~np.isnan(bout_x) & ~np.isnan(bout_y)
    x_v, y_v = bout_x[v], bout_y[v]
    if len(x_v) > 1:
        pts  = np.array([x_v, y_v]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        t    = np.linspace(0, 1, len(segs))
        lc   = LineCollection(segs, cmap='viridis', norm=plt.Normalize(0, 1))
        lc.set_array(t)
        lc.set_linewidth(2.5)
        ax4.add_collection(lc)
        ax4.plot(x_v[0],  y_v[0],  'go', ms=12, label='Bout start', zorder=5)
        ax4.plot(x_v[-1], y_v[-1], 'r^', ms=12, label='Bout end',   zorder=5)

    pad_x = arena_x_mm * 0.05
    pad_y = arena_y_mm * 0.15
    ax4.set_xlim(-pad_x, arena_x_mm + pad_x)
    ax4.set_ylim(-pad_y, arena_y_mm + pad_y)
    ax4.set_aspect('equal')
    ax4.set_xlabel('X (mm)', fontsize=12)
    ax4.set_ylabel('Y (mm)', fontsize=12)
    ax4.set_title(f'XY Trajectory | Distance: {bout_info["total_distance_mm"]:.2f} mm',
                  fontsize=13)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4, shrink=0.6, pad=0.02)
    cbar.set_label('Time (normalized)', fontsize=10)

    duration_s = n_frames_bout / fps
    excl_info  = ''
    if excluded_mask is not None:
        n_excl = bout_info.get('n_excluded_frames', 0)
        if n_excl > 0:
            excl_info = f' | {n_excl} frames excluded'
    ctx_info = f' | {context_frames}f context' if context_frames > 0 else ''
    boundary_str = ''
    parts = []
    if start_reason:
        parts.append(f'start: {start_reason}')
    if end_reason:
        parts.append(f'end: {end_reason}')
    if parts:
        boundary_str = f'\nBoundary causes: {" | ".join(parts)}'

    fig.suptitle(
        f'Walking Bout Analysis | Frames {actual_start}-{actual_end} | '
        f'{duration_s:.2f}s | {bout_info["min_cycles"]} min cycles{excl_info}{ctx_info}'
        f'{boundary_str}',
        fontsize=13, fontweight='bold', y=0.998,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fmt = save_path.suffix.lstrip('.') or 'png'
        fig.savefig(save_path, format=fmt, dpi=dpi, bbox_inches='tight')

    return fig


# ── courtship helpers ─────────────────────────────────────────────────────────

def _add_segment_shading(ax, segments, frame_offset=0):
    added = set()
    for seg in segments:
        stype = seg['type']
        if stype == 'quiet':
            continue
        s = seg['start'] + frame_offset
        e = seg['end']   + frame_offset
        color = _SEG_COLORS.get(stype, '#cccccc33')
        label = stype.capitalize() if stype not in added else None
        added.add(stype)
        ax.axvspan(s, e, color=color, zorder=1, label=label)


def _add_segment_freq_annotations(ax, segments, window_features, frame_offset=0):
    if window_features is None:
        return
    wc = window_features.get('window_centers', None)
    pf = window_features.get('peak_freq', None)
    if wc is None or pf is None:
        return
    for seg in segments:
        stype = seg['type']
        if stype == 'quiet':
            continue
        s, e = seg['start'], seg['end']
        in_seg = (wc >= s) & (wc <= e)
        if in_seg.sum() == 0:
            continue
        seg_freqs = pf[in_seg]
        median_f = np.nanmedian(seg_freqs)
        min_f    = np.nanmin(seg_freqs)
        max_f    = np.nanmax(seg_freqs)
        mid_frame   = (s + e) / 2 + frame_offset
        edge_color  = _SEG_EDGE_COLORS.get(stype, '#999999')
        freq_str = f'{median_f:.0f}Hz' if min_f == max_f else f'{min_f:.0f}-{max_f:.0f}Hz'
        ax.annotate(
            freq_str,
            xy=(mid_frame, 0.02), xycoords=('data', 'axes fraction'),
            fontsize=7, color=edge_color, fontweight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=edge_color,
                      alpha=0.85, lw=0.7),
        )


# ── courtship figure ──────────────────────────────────────────────────────────

def plot_courtship_bout_figure(
    bout_info, leg_data, scut_data, wing_data, wing_activities, filter_masks,
    frame_offset=0, fps=FPS, save_path=None, dpi=DEFAULT_DPI,
    context_frames=200, abd_data=None, window_features=None,
):
    """Create 5-panel courtship bout figure (wing Z, leg Z, speed, confidence, XY)."""
    start    = bout_info['start']
    end      = bout_info['end']
    bout_idx = bout_info.get('bout_idx', '?')
    pct_pulse  = bout_info.get('pct_pulse',  0)
    pct_sine   = bout_info.get('pct_sine',   0)
    pct_waggle = bout_info.get('pct_waggle', 0)
    segments   = bout_info.get('segments', [])
    dom_wing   = bout_info.get('dominant_wing', '?')

    data_len  = len(scut_data['x'])
    ctx_start = max(0, start - context_frames)
    ctx_end   = min(data_len - 1, end + context_frames)

    actual_start = start + frame_offset
    actual_end   = end   + frame_offset
    frames = np.arange(ctx_start + frame_offset, ctx_end + frame_offset + 1)

    speed, _, _           = compute_instant_speed(scut_data, ctx_start, ctx_end, fps)
    _, mean_speed, _      = compute_instant_speed(scut_data, start, end, fps)

    bd           = bout_info.get('boundary_diagnostics', None)
    start_reason = bd.get('primary_cause_before') if bd else None
    end_reason   = bd.get('primary_cause_after')  if bd else None

    mode_parts = []
    if pct_pulse  > 0: mode_parts.append(f'pulse={pct_pulse:.0f}%')
    if pct_sine   > 0: mode_parts.append(f'sine={pct_sine:.0f}%')
    if pct_waggle > 0: mode_parts.append(f'waggle={pct_waggle:.0f}%')
    mode_str   = ' '.join(mode_parts) if mode_parts else 'no song'
    duration_s = bout_info['n_frames'] / fps

    fig, axes = plt.subplots(
        5, 1, figsize=(14, 18),
        gridspec_kw={'height_ratios': [1.2, 1, 0.7, 0.7, 1], 'hspace': 0.28},
    )
    ax_wing, ax_leg, ax_speed, ax_conf, ax_xy = axes

    boundary_str = ''
    parts = []
    if start_reason: parts.append(f'start: {start_reason}')
    if end_reason:   parts.append(f'end: {end_reason}')
    if parts:
        boundary_str = f'  |  {" | ".join(parts)}'

    fig.suptitle(
        f'Courtship Bout {bout_idx}  |  Frames {actual_start}-{actual_end}  |  '
        f'{duration_s:.3f}s  |  Wing: {dom_wing}  |  {mode_str}{boundary_str}',
        fontsize=12, fontweight='bold', y=0.995,
    )

    wing_colors = {
        'WingL_V12': '#7B2D8E', 'WingL_V13': '#A855F7',
        'WingR_V12': '#0369A1', 'WingR_V13': '#38BDF8',
    }

    # Panel 1: Wing + Abd Z
    _add_bout_boundaries(ax_wing, actual_start, actual_end, add_legend=True)
    _add_segment_shading(ax_wing, segments, frame_offset)
    for tip in COURT_WING_TIPS:
        z = wing_data[tip]['z'][ctx_start:ctx_end + 1].copy()
        z = np.where((z >= Z_PLOT_MIN) & (z <= Z_PLOT_MAX), z, np.nan)
        ax_wing.plot(frames, z, label=tip.replace('Wing', 'W'),
                     color=wing_colors.get(tip, 'gray'), lw=1.5, alpha=0.9)
    if abd_data and 'z' in abd_data:
        abd_z = abd_data['z'][ctx_start:ctx_end + 1].copy()
        abd_z = np.where((abd_z >= Z_PLOT_MIN) & (abd_z <= Z_PLOT_MAX), abd_z, np.nan)
        ax_wing.plot(frames, abd_z, label='Abd_tip', color='#D4A017', lw=1.8, alpha=0.85)
    _add_segment_freq_annotations(ax_wing, segments, window_features, frame_offset)
    ax_wing.set_ylabel('Z (mm)', fontsize=12)
    ax_wing.set_title('Wing + Abdomen Z', fontsize=11)
    ax_wing.set_ylim(Z_PLOT_MIN, Z_PLOT_MAX)
    ax_wing.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, ncol=1)
    ax_wing.grid(True, alpha=0.3)
    if start_reason:
        ax_wing.annotate(
            start_reason, xy=(actual_start, 0.95), xycoords=('data', 'axes fraction'),
            xytext=(6, 0), textcoords='offset points', fontsize=7, color='#d62728',
            fontweight='bold', ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#d62728', alpha=0.85))
    if end_reason:
        ax_wing.annotate(
            end_reason, xy=(actual_end, 0.95), xycoords=('data', 'axes fraction'),
            xytext=(-6, 0), textcoords='offset points', fontsize=7, color='#d62728',
            fontweight='bold', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#d62728', alpha=0.85))

    # Panel 2: Leg Z
    leg_colors = {
        'T1L_TaTip': '#E63946', 'T1R_TaTip': '#457B9D',
        'T2L_TaTip': '#F4A261', 'T2R_TaTip': '#2A9D8F',
        'T3L_TaTip': '#9B2226', 'T3R_TaTip': '#1D3557',
    }
    _add_bout_boundaries(ax_leg, actual_start, actual_end)
    _add_segment_shading(ax_leg, segments, frame_offset)
    for tip in COURT_ALL_LEG_TIPS:
        z = leg_data[tip]['z'][ctx_start:ctx_end + 1].copy()
        z = np.where((z >= Z_PLOT_MIN) & (z <= Z_PLOT_MAX), z, np.nan)
        ax_leg.plot(frames, z, label=tip.replace('_TaTip', ''),
                    color=leg_colors.get(tip, 'gray'), lw=1.0, alpha=0.8)
    ax_leg.set_ylabel('Leg Tip Z (mm)', fontsize=12)
    ax_leg.set_title('Leg Z Trajectories', fontsize=11)
    ax_leg.set_ylim(Z_PLOT_MIN, Z_PLOT_MAX)
    ax_leg.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, ncol=1)
    ax_leg.grid(True, alpha=0.3)
    ax_leg.sharex(ax_wing)

    # Panel 3: Speed
    _add_bout_boundaries(ax_speed, actual_start, actual_end)
    _add_segment_shading(ax_speed, segments, frame_offset)
    ax_speed.plot(frames, speed, color='#264653', lw=1.2)
    ax_speed.fill_between(frames, 0, speed, alpha=0.3, color='#264653')
    ax_speed.axhline(y=COURT_PULSE_CLASSIFY_SPEED, color='#E76F51', ls='--', lw=1.5,
                     label=f'Pulse/sine threshold ({COURT_PULSE_CLASSIFY_SPEED} mm/s)')
    ax_speed.axhline(y=mean_speed, color='#2A9D8F', ls=':', lw=1.5,
                     label=f'Bout mean: {mean_speed:.1f} mm/s')
    ax_speed.set_ylabel('Speed (mm/s)', fontsize=12)
    ax_speed.set_title('Scutellum Speed', fontsize=11)
    ax_speed.legend(loc='upper right', fontsize=9)
    ax_speed.grid(True, alpha=0.3)
    ax_speed.set_ylim(bottom=0)
    ax_speed.sharex(ax_wing)

    # Panel 4: Wing confidence
    _add_bout_boundaries(ax_conf, actual_start, actual_end)
    for tip in COURT_WING_TIPS:
        c = wing_data[tip]['conf'][ctx_start:ctx_end + 1].astype(float)
        ax_conf.plot(frames, c, label=tip.replace('Wing', 'W'),
                     color=wing_colors.get(tip, 'gray'), lw=1.0, alpha=0.85)
    ax_conf.axhline(y=COURT_CONF_THRESHOLD, color='red', ls='--', lw=1.2,
                    label=f'Threshold ({COURT_CONF_THRESHOLD})')
    conf_mask_ctx = filter_masks['confidence'][ctx_start:ctx_end + 1]
    fail_trans = np.diff(np.concatenate([[False], ~conf_mask_ctx, [False]]).astype(int))
    for fs, fe in zip(np.where(fail_trans == 1)[0], np.where(fail_trans == -1)[0]):
        ax_conf.axvspan(frames[fs], frames[min(fe, len(frames) - 1)],
                        color='red', alpha=0.12, zorder=0)
    ax_conf.set_ylabel('Confidence', fontsize=11)
    ax_conf.set_title('Wing Keypoint Confidence  (red = filter fails)', fontsize=11)
    ax_conf.set_ylim(-0.05, 1.05)
    ax_conf.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, ncol=1)
    ax_conf.grid(True, alpha=0.3)
    ax_conf.sharex(ax_wing)

    # Panel 5: Wing XY trajectory
    arena_x, arena_y = COURT_ARENA_X_MM, COURT_ARENA_Y_MM
    ax_xy.plot([0, arena_x, arena_x, 0, 0], [0, 0, arena_y, arena_y, 0],
               'k-', lw=2, label=f'Arena ({arena_x}×{arena_y}mm)')

    if dom_wing == 'L':
        xy_tips = [('WingL_V12', '#7B2D8E'), ('WingL_V13', '#A855F7')]
    elif dom_wing == 'R':
        xy_tips = [('WingR_V12', '#0369A1'), ('WingR_V13', '#38BDF8')]
    else:
        xy_tips = [('WingR_V12', '#0369A1'), ('WingL_V12', '#7B2D8E')]

    for tip, color in xy_tips:
        if start - ctx_start > 1:
            pre_x = wing_data[tip]['x'][ctx_start:start + 1]
            pre_y = wing_data[tip]['y'][ctx_start:start + 1]
            v = ~np.isnan(pre_x) & ~np.isnan(pre_y)
            if v.sum() > 1:
                ax_xy.plot(pre_x[v], pre_y[v], color='gray', ls='--', lw=0.8, alpha=0.4)
        if ctx_end - end > 1:
            post_x = wing_data[tip]['x'][end:ctx_end + 1]
            post_y = wing_data[tip]['y'][end:ctx_end + 1]
            v = ~np.isnan(post_x) & ~np.isnan(post_y)
            if v.sum() > 1:
                ax_xy.plot(post_x[v], post_y[v], color='gray', ls=':', lw=0.8, alpha=0.4)
        bx, by = wing_data[tip]['x'][start:end + 1], wing_data[tip]['y'][start:end + 1]
        v = ~np.isnan(bx) & ~np.isnan(by)
        xv, yv = bx[v], by[v]
        if len(xv) > 1:
            pts  = np.array([xv, yv]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            t    = np.linspace(0, 1, len(segs))
            lc   = LineCollection(segs, cmap='viridis', norm=plt.Normalize(0, 1))
            lc.set_array(t)
            lc.set_linewidth(2.0)
            ax_xy.add_collection(lc)
        tip_label = tip.replace('Wing', 'W')
        if len(xv) > 0:
            ax_xy.plot(xv[0],  yv[0],  'o', color=color, ms=8, zorder=5,
                       label=f'{tip_label} start')
            ax_xy.plot(xv[-1], yv[-1], '^', color=color, ms=8, zorder=5,
                       label=f'{tip_label} end')

    pad_x = arena_x * 0.05
    pad_y = arena_y * 0.15
    ax_xy.set_xlim(-pad_x, arena_x + pad_x)
    ax_xy.set_ylim(-pad_y, arena_y + pad_y)
    ax_xy.set_aspect('equal')
    ax_xy.set_xlabel('X (mm)', fontsize=12)
    ax_xy.set_ylabel('Y (mm)', fontsize=12)
    ax_xy.set_title(
        f'Wing XY Trajectory (dominant: {dom_wing}) | '
        f'Scut dist: {bout_info.get("total_distance_mm", 0):.2f} mm',
        fontsize=11,
    )
    ax_xy.legend(loc='upper right', fontsize=8, ncol=2)
    ax_xy.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_xy, shrink=0.6, pad=0.02)
    cbar.set_label('Time (normalized)', fontsize=10)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fmt = save_path.suffix.lstrip('.') or 'png'
        fig.savefig(save_path, format=fmt, dpi=dpi, bbox_inches='tight')

    return fig
