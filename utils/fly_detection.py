"""
Fly detection utility for multi-fly datasets.

Detects whether a Predictions_3D folder contains single-fly or dual-fly data
based on the presence of fly-suffixed CSV files.
"""

from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd


def build_unified_bouts_csv(folder: Path, dataset: str,
                            force: bool = False) -> Optional[Path]:
    """
    Merge per-fly bouts summary CSVs into a single unified bouts list so that
    both flies are preprocessed over the same set of frame ranges. This is a
    prerequisite for the identity-relink stage which needs both flies' raw
    keypoints over the same time windows.

    Reads ``<dataset>_bouts_fly0_summary.csv`` and ``<dataset>_bouts_fly1_summary.csv``
    and writes ``<dataset>_bouts_unified_summary.csv`` to the same folder.

    The merge:
      - Strips ``_fly0`` / ``_fly1`` suffixes from the ``fly_id`` column so the
        unified csv stores session-level identifiers only. The processing
        script appends the actual fly suffix at load time.
      - Drops exact duplicate windows (same start_frame, end_frame, fly_id).
      - Sorts by (fly_id, start_frame).
      - Renumbers ``bout_idx`` sequentially starting at 1.

    Returns the path to the unified csv, or None if the per-fly summaries do
    not exist.
    """
    fly0_csv = folder / f"{dataset}_bouts_fly0_summary.csv"
    fly1_csv = folder / f"{dataset}_bouts_fly1_summary.csv"
    if not (fly0_csv.exists() and fly1_csv.exists()):
        return None

    out_csv = folder / f"{dataset}_bouts_unified_summary.csv"
    if out_csv.exists() and not force:
        return out_csv

    df0 = pd.read_csv(fly0_csv)
    df1 = pd.read_csv(fly1_csv)
    df = pd.concat([df0, df1], ignore_index=True)

    # Strip _fly0 / _fly1 suffix from fly_id (session-level only)
    if "fly_id" in df.columns:
        df["fly_id"] = df["fly_id"].astype(str).str.replace(r"_fly\d+$", "",
                                                              regex=True)

    df = df.drop_duplicates(subset=["fly_id", "start_frame", "end_frame"])
    df = df.sort_values(["fly_id", "start_frame"]).reset_index(drop=True)
    df["bout_idx"] = range(1, len(df) + 1)

    df.to_csv(out_csv, index=False)
    return out_csv


def detect_flies(folder: Path, dataset: str) -> List[Dict]:
    """
    Detect single-fly vs dual-fly data layout in a Predictions_3D folder.

    Checks for fly-suffixed CSV files (e.g., data3D_fly0.csv). If found,
    returns one entry per fly. Otherwise returns a single entry with no suffix,
    matching the original single-fly pipeline behavior.

    Args:
        folder: Path to a Predictions_3D_* folder.
        dataset: Dataset name (e.g., 'courtship', 'free_walking').

    Returns:
        List of dicts, each with keys:
            - fly_id:    str or None ('fly0', 'fly1', or None for single-fly)
            - suffix:    str to append to output filenames ('_fly0', '_fly1', or '')
            - csv:       CSV filename for this fly's keypoint data
            - bouts_csv: Bouts summary CSV filename for this fly
    """
    flies = []

    # Check for dual-fly layout: data3D_fly0.csv, data3D_fly1.csv
    fly_csvs = sorted(folder.glob("data3D_fly*.csv"))
    if fly_csvs:
        # For dual-fly we always build a unified bouts list so identity-relink
        # has both flies' keypoints over the same frame windows.
        unified_csv = build_unified_bouts_csv(folder, dataset)
        for csv_path in fly_csvs:
            stem = csv_path.stem  # data3D_fly0
            fly_id = stem.replace("data3D_", "")  # fly0

            if unified_csv is not None:
                bouts_csv_name = unified_csv.name
            else:
                # Fallback: per-fly bouts (older single-fly behaviour)
                bouts_csv_name = f"{dataset}_bouts_{fly_id}_summary.csv"
                bouts_csv_path = folder / bouts_csv_name
                if not bouts_csv_path.exists():
                    for f in folder.iterdir():
                        if f.name.lower() == bouts_csv_name.lower():
                            bouts_csv_name = f.name
                            break

            flies.append({
                'fly_id': fly_id,
                'suffix': f"_{fly_id}",
                'csv': csv_path.name,
                'bouts_csv': bouts_csv_name,
            })
    else:
        # Single-fly layout: data3D.csv
        flies.append({
            'fly_id': None,
            'suffix': '',
            'csv': 'data3D.csv',
            'bouts_csv': f'{dataset}_bouts_summary.csv',
        })

    return flies
