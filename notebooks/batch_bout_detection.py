#!/usr/bin/env python3
"""Batch walking bout detection from predictions_index.csv.

Usage:
    python batch_bout_detection.py --index /path/to/predictions_index.csv
    python batch_bout_detection.py --index /path/to/predictions_index.csv --fps 800 --skip-existing

For each row in predictions_index.csv (columns: prediction_folder, fly_id):
  - Loads {prediction_folder}/data3D.csv
  - Runs detect_walking_bouts
  - Saves {prediction_folder}/walking_bouts_summary.csv

Also saves a combined CSV alongside the predictions_index.csv:
  {predictions_index_dir}/walking_bouts_combined.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.bout_detection import (
    detect_walking_bouts, compute_instant_speed, FPS,
)


def load_data3d(prediction_folder: Path) -> pd.DataFrame:
    csv_path = prediction_folder / "data3D.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"data3D.csv not found in {prediction_folder}")
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def bouts_to_dataframe(valid_bouts: list, scutellum_data: dict,
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


def process_one(prediction_folder: Path, fly_id: str, fps: float,
                skip_existing: bool) -> pd.DataFrame | None:
    out_csv = prediction_folder / "walking_bouts_summary.csv"

    if skip_existing and out_csv.exists():
        print(f"  [skip] {fly_id} — walking_bouts_summary.csv already exists")
        return pd.read_csv(out_csv)

    print(f"  Processing {fly_id} ...")
    df = load_data3d(prediction_folder)
    valid_bouts, rejected_bouts, _, scutellum_data, _, _ = detect_walking_bouts(
        df, verbose=False
    )
    print(f"    → {len(valid_bouts)} valid bouts, {len(rejected_bouts)} rejected")

    bout_df = bouts_to_dataframe(valid_bouts, scutellum_data, fly_id, fps)
    bout_df.to_csv(out_csv, index=False)
    print(f"    Saved: {out_csv}")
    return bout_df


def main():
    parser = argparse.ArgumentParser(
        description="Batch walking bout detection from predictions_index.csv"
    )
    parser.add_argument("--index", required=True, help="Path to predictions_index.csv")
    parser.add_argument("--fps", type=float, default=FPS,
                        help=f"Frame rate in Hz (default: {FPS})")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip folders that already have walking_bouts_summary.csv")
    args = parser.parse_args()

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
    all_dfs = []
    errors = []

    for _, row in index_df.iterrows():
        prediction_folder = Path(str(row["prediction_folder"]))
        fly_id = str(row["fly_id"])
        try:
            bout_df = process_one(prediction_folder, fly_id, args.fps, args.skip_existing)
            if bout_df is not None and len(bout_df) > 0:
                bout_df = bout_df.copy()
                bout_df["prediction_folder"] = str(prediction_folder)
                all_dfs.append(bout_df)
        except Exception as exc:
            msg = f"  [error] {fly_id}: {exc}"
            print(msg, file=sys.stderr)
            errors.append(msg)

    print()
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = index_path.parent / "walking_bouts_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Combined CSV: {combined_path}")
        print(f"Total bouts:  {len(combined)} across {len(all_dfs)} flies")
    else:
        print("No bouts detected across all flies.")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(e)


if __name__ == "__main__":
    main()
