#!/usr/bin/env python3
"""Batch walking + courtship bout detection from predictions_index.csv.

Usage:
    python batch_bout_detection.py --index /path/to/predictions_index.csv
    python batch_bout_detection.py --index /path/to/predictions_index.csv --fps 800 --skip-existing
    python batch_bout_detection.py --index /path/to/predictions_index.csv --no-courtship
    python batch_bout_detection.py --index /path/to/predictions_index.csv --no-walking --save-figs --max-figs N

For each row in predictions_index.csv (columns: prediction_folder, fly_id):
  - Auto-detects single-fly (data3D.csv) or two-fly (data3D_fly0.csv + data3D_fly1.csv) format
  - Runs walking bout detection on each fly → walking_bouts[_flyN]_summary.csv
  - Runs courtship bout detection on each fly → courtship_bouts[_flyN]_summary.csv

Also saves combined CSVs alongside the predictions_index.csv:
  {predictions_index_dir}/walking_bouts_combined.csv
  {predictions_index_dir}/courtship_bouts_combined.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use('Agg')  # headless rendering — must be set before any plt import

sys.path.insert(0, "/home/user/src/3d_tracking_dataset")
from utils.bout_detection import (
    detect_walking_bouts, compute_instant_speed, FPS,
)
from utils.courtship_detection import (
    detect_courtship_bouts, reclassify_bouts_with_fft,
)
from utils.bout_viz import plot_walking_bout_figure, plot_courtship_bout_figure


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data3d(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def detect_fly_files(prediction_folder: Path) -> list[tuple[Path, str]]:
    """Return list of (csv_path, fly_suffix) for this folder.

    Two-fly format:  [(data3D_fly0.csv, '_fly0'), (data3D_fly1.csv, '_fly1')]
    Single-fly format: [(data3D.csv, '')]
    """
    fly0 = prediction_folder / "data3D_fly0.csv"
    fly1 = prediction_folder / "data3D_fly1.csv"
    single = prediction_folder / "data3D.csv"

    if fly0.exists() and fly1.exists():
        return [(fly0, "_fly0"), (fly1, "_fly1")]
    elif single.exists():
        return [(single, "")]
    else:
        raise FileNotFoundError(
            f"No data3D.csv or data3D_fly0/1.csv found in {prediction_folder}"
        )


# ── Walking bout output ───────────────────────────────────────────────────────

def walking_bouts_to_dataframe(valid_bouts: list, scutellum_data: dict,
                                fly_id: str, fps: float) -> pd.DataFrame:
    rows = []
    for bout in valid_bouts:
        start, end = bout['start'], bout['end']
        n_frames = bout['n_frames']
        dur = n_frames / fps

        speed, mean_speed, max_speed = compute_instant_speed(scutellum_data, start, end, fps)

        scut_z = scutellum_data['z'][start:end+1]
        scut_z_mean = float(np.nanmean(scut_z))
        scut_z_std = float(np.nanstd(scut_z))
        scut_z_min = float(np.nanmin(scut_z))
        scut_z_max = float(np.nanmax(scut_z))

        valid_mask = ~np.isnan(scut_z) & ~np.isnan(speed)
        if valid_mask.sum() > 10:
            r_p, p_p = pearsonr(scut_z[valid_mask], speed[valid_mask])
            r_s, p_s = spearmanr(scut_z[valid_mask], speed[valid_mask])
        else:
            r_p, p_p, r_s, p_s = np.nan, np.nan, np.nan, np.nan

        rows.append({
            'fly_id': fly_id,
            'bout_idx': bout['bout_idx'],
            'start_frame': start,
            'end_frame': end,
            'n_frames': n_frames,
            'duration_s': dur,
            'min_cycles': bout['min_cycles'],
            'total_distance_mm': bout['total_distance_mm'],
            'net_displacement_mm': bout['net_displacement_mm'],
            'mean_speed_mm_s': mean_speed,
            'max_speed_mm_s': max_speed,
            'scut_z_mean': scut_z_mean,
            'scut_z_std': scut_z_std,
            'scut_z_min': scut_z_min,
            'scut_z_max': scut_z_max,
            'height_speed_pearson_r': r_p,
            'height_speed_pearson_p': p_p,
            'height_speed_spearman_r': r_s,
            'height_speed_spearman_p': p_s,
        })
    return pd.DataFrame(rows)


# ── Courtship bout output ─────────────────────────────────────────────────────

def courtship_bouts_to_dataframe(bouts: list, fly_id: str) -> pd.DataFrame:
    rows = []
    for bout in bouts:
        bd = bout.get('boundary_diagnostics', {})
        pre = bd.get('pre_bout_failures', {})
        post = bd.get('post_bout_failures', {})
        rows.append({
            'fly_id': fly_id,
            'bout_idx': bout['bout_idx'],
            'start_frame': bout['start'],
            'end_frame': bout['end'],
            'n_frames': bout['n_frames'],
            'duration_s': bout['duration_s'],
            'mean_speed_mm_s': bout['mean_speed_mm_s'],
            'max_speed_mm_s': bout['max_speed_mm_s'],
            'total_distance_mm': bout['total_distance_mm'],
            'pct_pulse': bout.get('pct_pulse', np.nan),
            'pct_sine': bout.get('pct_sine', np.nan),
            'pct_waggle': bout.get('pct_waggle', np.nan),
            'dominant_wing': bout.get('dominant_wing', None),
            'fft_classified': bout.get('fft_classified', False),
            'start_reason': bd.get('primary_cause_before'),
            'end_reason': bd.get('primary_cause_after'),
        })
    return pd.DataFrame(rows)


# ── Per-folder processing ─────────────────────────────────────────────────────

def _save_walking_figs(valid_bouts, leg_tip_data, scutellum_data, fly_label,
                        fps, fig_dir, max_figs):
    """Save per-bout walking figures as PNG, limited to the N longest bouts."""
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    bouts_to_plot = sorted(valid_bouts, key=lambda b: b['n_frames'], reverse=True)
    if max_figs:
        bouts_to_plot = bouts_to_plot[:max_figs]
    print(f"    Saving {len(bouts_to_plot)} walking figures → {fig_dir.name}/")
    for bout in bouts_to_plot:
        idx = bout['bout_idx']
        save_path = fig_dir / f"walking_bout_{idx:03d}.png"
        fig = plot_walking_bout_figure(
            bout, leg_tip_data, scutellum_data,
            fps=fps, save_path=save_path,
        )
        plt.close(fig)


def _save_courtship_figs(bouts, leg_tip_data, scutellum_data, wing_data,
                          wing_activities, filter_masks, abd_data,
                          fps, fig_dir, max_figs):
    """Save per-bout courtship figures as PNG, limited to the N longest bouts."""
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    bouts_to_plot = sorted(bouts, key=lambda b: b['n_frames'], reverse=True)
    if max_figs:
        bouts_to_plot = bouts_to_plot[:max_figs]
    print(f"    Saving {len(bouts_to_plot)} courtship figures → {fig_dir.name}/")
    for bout in bouts_to_plot:
        idx = bout['bout_idx']
        save_path = fig_dir / f"courtship_bout_{idx:03d}.png"
        fig = plot_courtship_bout_figure(
            bout, leg_tip_data, scutellum_data, wing_data, wing_activities,
            filter_masks, fps=fps, save_path=save_path, abd_data=abd_data,
            window_features=bout.get('window_features'),
        )
        plt.close(fig)


def process_one(prediction_folder: Path, fly_id: str, fps: float,
                skip_existing: bool, run_walking: bool, run_courtship: bool,
                save_figs: bool = False,
                max_figs: int | None = 50) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Process all flies in one prediction folder.

    Returns:
        (walking_dfs, courtship_dfs) — list of DataFrames, one per fly.
    """
    fly_files = detect_fly_files(prediction_folder)
    walking_dfs = []
    courtship_dfs = []

    for csv_path, suffix in fly_files:
        fly_label = f"{fly_id}{suffix}"

        # ── Walking ──────────────────────────────────────────────────────────
        if run_walking:
            walk_out = prediction_folder / f"walking_bouts{suffix}_summary.csv"
            if skip_existing and walk_out.exists():
                print(f"  [skip walking] {fly_label}")
                walk_df = pd.read_csv(walk_out)
            else:
                print(f"  Walking: {fly_label} ...")
                df = load_data3d(csv_path)
                valid_bouts, rejected_bouts, leg_tip_data, scutellum_data, _, _ = \
                    detect_walking_bouts(df, verbose=False)
                print(f"    → {len(valid_bouts)} valid, {len(rejected_bouts)} rejected")
                walk_df = walking_bouts_to_dataframe(valid_bouts, scutellum_data,
                                                      fly_label, fps)
                walk_df.to_csv(walk_out, index=False)
                print(f"    Saved: {walk_out}")
                if save_figs and valid_bouts:
                    _save_walking_figs(
                        valid_bouts, leg_tip_data, scutellum_data, fly_label, fps,
                        fig_dir=prediction_folder / f"walking_figs{suffix}",
                        max_figs=max_figs,
                    )
            if len(walk_df) > 0:
                walk_df = walk_df.copy()
                walk_df["prediction_folder"] = str(prediction_folder)
                walking_dfs.append(walk_df)

        # ── Courtship ─────────────────────────────────────────────────────
        if run_courtship:
            court_out = prediction_folder / f"courtship_bouts{suffix}_summary.csv"
            if skip_existing and court_out.exists():
                print(f"  [skip courtship] {fly_label}")
                court_df = pd.read_csv(court_out)
            else:
                print(f"  Courtship: {fly_label} ...")
                df = load_data3d(csv_path)
                bouts, leg_tip_data, scut_data, wing_data, wing_activities, \
                    filter_masks, diagnostics, abd_data = \
                    detect_courtship_bouts(df, verbose=False)
                print(f"    → {len(bouts)} courtship bouts (raw)")
                if bouts:
                    bouts = reclassify_bouts_with_fft(bouts, wing_data, verbose=False)
                print(f"    → {len(bouts)} bouts after FFT reclassification")
                court_df = courtship_bouts_to_dataframe(bouts, fly_label)
                court_df.to_csv(court_out, index=False)
                print(f"    Saved: {court_out}")
                if save_figs and bouts:
                    _save_courtship_figs(
                        bouts, leg_tip_data, scut_data, wing_data,
                        wing_activities, filter_masks, abd_data, fps,
                        fig_dir=prediction_folder / f"courtship_figs{suffix}",
                        max_figs=max_figs,
                    )
            if len(court_df) > 0:
                court_df = court_df.copy()
                court_df["prediction_folder"] = str(prediction_folder)
                courtship_dfs.append(court_df)

    return walking_dfs, courtship_dfs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch walking + courtship bout detection"
    )
    parser.add_argument("--index", required=True, help="Path to predictions_index.csv")
    parser.add_argument("--fps", type=float, default=FPS,
                        help=f"Frame rate in Hz (default: {FPS})")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip folders that already have output summary CSVs")
    parser.add_argument("--no-walking", action="store_true",
                        help="Skip walking bout detection")
    parser.add_argument("--no-courtship", action="store_true",
                        help="Skip courtship bout detection")
    parser.add_argument("--save-figs", action="store_true",
                        help="Save per-bout diagnostic figures as PNG")
    parser.add_argument("--max-figs", type=int, default=50,
                        help="Max figures to save per fly, sorted by duration "
                             "(default: 50; 0 = save all)")
    args = parser.parse_args()

    run_walking  = not args.no_walking
    run_courtship = not args.no_courtship
    max_figs = args.max_figs if args.max_figs > 0 else None

    if not run_walking and not run_courtship:
        print("Error: --no-walking and --no-courtship are both set; nothing to do.",
              file=sys.stderr)
        sys.exit(1)

    index_path = Path(args.index)
    if not index_path.exists():
        print(f"Error: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    index_df = pd.read_csv(index_path)
    required = {"prediction_folder", "fly_id"}
    if not required.issubset(index_df.columns):
        print(f"Error: missing columns: {required - set(index_df.columns)}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(index_df)} prediction folders from {index_path.name} ...")
    all_walking = []
    all_courtship = []
    errors = []

    for _, row in index_df.iterrows():
        prediction_folder = Path(str(row["prediction_folder"]))
        fly_id = str(row["fly_id"])
        print(f"\n[{fly_id}] {prediction_folder.name}")
        try:
            walk_dfs, court_dfs = process_one(
                prediction_folder, fly_id, args.fps,
                args.skip_existing, run_walking, run_courtship,
                save_figs=args.save_figs, max_figs=max_figs,
            )
            all_walking.extend(walk_dfs)
            all_courtship.extend(court_dfs)
        except Exception as exc:
            msg = f"  [error] {fly_id}: {exc}"
            print(msg, file=sys.stderr)
            errors.append(msg)

    print()

    if run_walking:
        if all_walking:
            combined = pd.concat(all_walking, ignore_index=True)
            out_path = index_path.parent / "walking_bouts_combined.csv"
            combined.to_csv(out_path, index=False)
            print(f"Walking combined: {out_path}  ({len(combined)} bouts)")
        else:
            print("No walking bouts detected across all flies.")

    if run_courtship:
        if all_courtship:
            combined = pd.concat(all_courtship, ignore_index=True)
            out_path = index_path.parent / "courtship_bouts_combined.csv"
            combined.to_csv(out_path, index=False)
            print(f"Courtship combined: {out_path}  ({len(combined)} bouts)")
        else:
            print("No courtship bouts detected across all flies.")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(e)


if __name__ == "__main__":
    main()
