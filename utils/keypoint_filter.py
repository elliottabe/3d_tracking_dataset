"""
Keypoint filtering and smoothing utilities for 3D tracking data.

Stacked cleaning pipeline applied before Procrustes alignment:
  1. Confidence masking  — sets low-confidence keypoints to NaN
  2. Bone-length outlier detection — flags frames where a segment length
     deviates more than `threshold_std` sigma (MAD-based) from the temporal median
  3. Median-filter spike detection + spline interpolation — identifies remaining
     trajectory spikes and fills all NaN gaps via cubic spline
  4. Savitzky-Golay smoothing — optional final trajectory smoothing

All operations work on numpy arrays of shape (T, N, 3).

Usage::

    from utils.keypoint_filter import filter_keypoints, load_confidence_from_csv
    filtered, report = filter_keypoints(kp_array, confidence, skeleton_edges, cfg.preprocessing.filtering)
"""

import numpy as np
from scipy import signal
from scipy.interpolate import splrep, splev
from typing import Optional, Tuple, Dict, List
from omegaconf import DictConfig
import traceback as _traceback

# Set non-interactive backend before any pyplot import so figure saving works
# headlessly (on servers without a display).  This must happen at import time
# of this module, before matplotlib.pyplot is ever loaded anywhere.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _nan_interp_1d(vals: np.ndarray, use_spline: bool = True) -> np.ndarray:
    """Interpolate NaN gaps in a 1-D signal. Skips if >50% of values are NaN."""
    nans = np.isnan(vals)
    if not np.any(nans):
        return vals
    if np.mean(~nans) < 0.5 or np.sum(~nans) < 4:
        return vals  # too sparse — leave as NaN rather than extrapolate wildly

    x_good = np.where(~nans)[0]
    v_good = vals[~nans]
    x_bad = np.where(nans)[0]
    out = vals.copy()

    if use_spline:
        try:
            spl = splrep(x_good, v_good, k=3, s=0)
            out[nans] = splev(x_bad, spl)
        except Exception:
            out[nans] = np.interp(x_bad, x_good, v_good)
    else:
        out[nans] = np.interp(x_bad, x_good, v_good)
    return out


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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Set keypoints with confidence < threshold to NaN.

    Args:
        kp_array:   (T, N, 3)  3-D keypoint positions
        confidence: (T, N)     confidence scores in [0, 1]
        threshold:  scalar     detections below this are masked

    Returns:
        kp_masked: (T, N, 3) copy with NaN where confidence was low
        bad_mask:  (T, N) bool — True where keypoints were masked
    """
    bad_mask = confidence < threshold                      # (T, N)
    kp_masked = kp_array.copy().astype(float)
    kp_masked[bad_mask] = np.nan                           # broadcast over xyz dim
    n_bad = int(np.sum(bad_mask))
    frac = n_bad / bad_mask.size * 100
    print(f"  [confidence] masked {n_bad}/{bad_mask.size} keypoint-frames "
          f"({frac:.1f}%) below threshold={threshold}")
    return kp_masked, bad_mask


def _detect_bone_length_outliers(
    kp_array: np.ndarray,
    edges: np.ndarray,
    threshold_std: float = 3.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Flag frames where a bone's length deviates more than threshold_std sigma
    (MAD-based) from its temporal median. Both endpoint keypoints are set to NaN.

    Args:
        kp_array:      (T, N, 3) — may already contain NaNs
        edges:         (E, 2)    skeleton edge index pairs
        threshold_std: deviation cutoff in robust-sigma units

    Returns:
        kp_flagged: copy with additional NaN outliers
        report:     {edge_index: n_frames_flagged}
    """
    kp_flagged = kp_array.copy()
    report: Dict[int, int] = {}
    total_flagged = 0

    for ei, (i, j) in enumerate(edges):
        lengths = np.linalg.norm(kp_array[:, i, :] - kp_array[:, j, :], axis=1)
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

    if total_flagged:
        print(f"  [bone-length] flagged {total_flagged} keypoint-frames across "
              f"{len(report)} bones (>{threshold_std}σ from median length)")
    else:
        print(f"  [bone-length] no outliers found (threshold={threshold_std}σ)")
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
                # Not enough data for median filter — just interpolate gaps
                kp_out[:, n, d] = _nan_interp_1d(vals, use_spline=use_spline)
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

            kp_out[:, n, d] = _nan_interp_1d(vals, use_spline=use_spline)

    # Fill any remaining NaN gaps: linear interp for internal gaps,
    # nearest-neighbour for leading/trailing edges.
    for n in range(N):
        for d in range(3):
            vals = kp_out[:, n, d]
            nans = np.isnan(vals)
            if np.any(nans) and np.any(~nans):
                valid_idx = np.where(~nans)[0]
                vals[nans] = np.interp(np.where(nans)[0], valid_idx, vals[valid_idx])
                kp_out[:, n, d] = vals

    if total_spikes:
        print(f"  [medfilt+interp] detected {total_spikes} coordinate-spikes "
              f"(>{spike_threshold_std}σ from medfilt); interpolated all NaN gaps "
              f"(kernel={medfilt_kernel})")
    else:
        print(f"  [medfilt+interp] no spikes detected; interpolated NaN gaps "
              f"(kernel={medfilt_kernel})")
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
) -> Tuple[np.ndarray, Dict]:
    """
    Apply the full filtering pipeline to a (T, N, 3) keypoint array.

    Steps (each enabled/disabled via filter_cfg sub-keys):
      1. confidence masking   (filter_cfg.confidence.enabled)
      2. bone-length outliers (filter_cfg.bone_length.enabled)
      3. medfilt + interp     (filter_cfg.medfilt.enabled)
      4. savgol smoothing     (filter_cfg.savgol.enabled)

    Args:
        kp_array:       (T, N, 3)  raw keypoint positions (mm)
        confidence:     (T, N)     confidence scores in [0, 1], or None
        skeleton_edges: (E, 2)     skeleton edge index pairs
        filter_cfg:     OmegaConf DictConfig with filtering sub-keys
        fig_dir:        If provided, save diagnostic figures here.
        bout_name:      Label used in figure filenames/titles.
        kp_names:       List of N keypoint names for plot axis labels.

    Returns:
        kp_filtered: (T, N, 3) cleaned array
        report:      per-step summary dict
    """
    report: Dict = {}
    kp_raw = kp_array.astype(float)
    kp = kp_raw.copy()
    print("\n--- Keypoint Filtering ---")

    # Step 1: confidence masking
    if filter_cfg.get('confidence', {}).get('enabled', False) and confidence is not None:
        threshold = filter_cfg.confidence.get('threshold', 0.5)
        kp, conf_mask = _mask_low_confidence(kp, confidence, threshold)
        report['confidence_masked'] = int(np.sum(conf_mask))

    # Step 2: bone-length outlier detection
    if filter_cfg.get('bone_length', {}).get('enabled', True):
        std_thresh = filter_cfg.bone_length.get('threshold_std', 3.0)
        kp, bone_rpt = _detect_bone_length_outliers(kp, skeleton_edges, std_thresh)
        report['bone_outliers'] = bone_rpt

    # Step 3: medfilt spike detection + spline interpolation
    if filter_cfg.get('medfilt', {}).get('enabled', True):
        kernel = filter_cfg.medfilt.get('kernel', 5)
        spike_std = filter_cfg.medfilt.get('spike_threshold_std', 5.0)
        use_spline = filter_cfg.medfilt.get('use_spline', True)
        kp = _medfilt_interpolate(kp, medfilt_kernel=kernel,
                                  spike_threshold_std=spike_std, use_spline=use_spline)

    # Step 4: Savitzky-Golay final smoothing
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

    return kp, report


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

    for bout in bouts:
        # Mirror the same bounds check as load_concatenated_bouts
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
