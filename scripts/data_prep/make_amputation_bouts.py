#!/usr/bin/env python3
"""
Generate per-folder ``amputation_bouts_summary.csv`` for the amputation dataset.

The amputation recordings ship a ``running_bouts_summary_full.csv`` whose
``fly_id`` column may list several recordings (one per session timestamp), while
each ``Predictions_3D_*`` folder holds a single recording's ``data3D.csv``. The
preprocessing pipeline (``preprocess_keypoints_for_ik.py``) expects a per-folder
``<dataset>_bouts_summary.csv`` with columns
``[bout_idx, start_frame, end_frame, fly_id]`` containing only this recording's
bouts.

This script, per folder:
  1. Reads ``info.yaml`` to get the recording id (basename of ``recording_path``).
  2. Filters ``running_bouts_summary_full.csv`` to rows whose ``fly_id`` matches
     that recording (falls back to all rows if nothing matches / single fly_id).
  3. Writes ``amputation_bouts_summary.csv`` with the required columns and
     ``bout_idx`` renumbered from 1.

Usage:
    # One folder
    python scripts/make_amputation_bouts.py --folder /path/to/Predictions_3D_XXXX

    # All Predictions_3D_* folders under a base dir
    python scripts/make_amputation_bouts.py --base-dir /gscratch/.../Johnson_lab/Amputation
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REQUIRED_COLS = ["bout_idx", "start_frame", "end_frame", "fly_id"]
SOURCE_CSV = "running_bouts_summary_full.csv"
OUTPUT_CSV = "amputation_bouts_summary.csv"


def recording_id_from_info(folder: Path) -> str | None:
    """Return the recording id (basename of info.yaml:recording_path), or None."""
    info_path = folder / "info.yaml"
    if not info_path.exists():
        return None
    with open(info_path) as f:
        info = yaml.safe_load(f) or {}
    rec = info.get("recording_path")
    if not rec:
        return None
    return Path(str(rec).rstrip("/")).name


def make_bouts_for_folder(folder: Path, force: bool = False) -> bool:
    """Write amputation_bouts_summary.csv for one folder. Returns True if written."""
    src = folder / SOURCE_CSV
    out = folder / OUTPUT_CSV
    if not src.exists():
        print(f"  [skip] {folder.name}: no {SOURCE_CSV}")
        return False
    if out.exists() and not force:
        print(f"  [skip] {folder.name}: {OUTPUT_CSV} exists (use --force)")
        return False

    df = pd.read_csv(src)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"  [error] {folder.name}: {SOURCE_CSV} missing columns {missing}")
        return False

    rec_id = recording_id_from_info(folder)
    n_fly_ids = df["fly_id"].nunique()
    if rec_id is not None and n_fly_ids > 1:
        mask = df["fly_id"].astype(str).str.contains(rec_id, regex=False)
        if mask.any():
            df = df[mask].copy()
            print(f"  [info] {folder.name}: filtered to recording '{rec_id}' "
                  f"({len(df)} bouts)")
        else:
            print(f"  [warn] {folder.name}: recording '{rec_id}' not found in "
                  f"fly_id column; keeping all {len(df)} bouts")
    elif rec_id is None:
        print(f"  [warn] {folder.name}: no recording_path in info.yaml; "
              f"keeping all {len(df)} bouts")

    df = df.sort_values(["start_frame"]).reset_index(drop=True)
    df["bout_idx"] = range(1, len(df) + 1)
    df[REQUIRED_COLS].to_csv(out, index=False)
    print(f"  [ok] {folder.name}: wrote {OUTPUT_CSV} ({len(df)} bouts)")
    return True


def find_folders(base_dir: Path) -> list:
    pattern = "Predictions_3D_*"
    if base_dir.is_dir() and base_dir.match(pattern):
        return [base_dir]
    return sorted(f for f in base_dir.rglob(pattern) if f.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--folder", type=str, help="A single Predictions_3D_* folder")
    g.add_argument("--base-dir", type=str,
                   help="Base dir to search for Predictions_3D_* folders")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing amputation_bouts_summary.csv")
    args = parser.parse_args()

    if args.folder:
        folders = [Path(args.folder)]
    else:
        base = Path(args.base_dir)
        if not base.exists():
            print(f"Error: base dir not found: {base}")
            sys.exit(1)
        folders = find_folders(base)
        if not folders:
            print(f"No Predictions_3D_* folders under {base}")
            sys.exit(1)

    print(f"Generating {OUTPUT_CSV} for {len(folders)} folder(s):")
    n_written = 0
    for folder in folders:
        if make_bouts_for_folder(folder, force=args.force):
            n_written += 1
    print(f"\nDone: wrote {n_written}/{len(folders)} files.")


if __name__ == "__main__":
    main()
