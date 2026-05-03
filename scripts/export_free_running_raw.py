#!/usr/bin/env python3
"""
Export a "raw-data only" free-running h5 from an existing combined free-running
h5 (e.g. ik_output_combined_v1_free_running.h5).

Free-running data has no fly0/fly1 pairing and no song-analysis filter, so
this exporter is structurally simpler than the courtship one: it keeps every
bout in the source combined h5, drops geometric_angles, and best-effort
attaches per-Predictions_3D bout-summary metadata + un-rescaled orig_keypoints
when matching paths are supplied.

Usage:
    python scripts/export_free_running_raw.py \\
        /data2/.../free_running/Data_analysis/analysis/v1/ik_output_combined_v1_free_running.h5 \\
        --out /data2/.../free_running/free_running_raw_combined_v1.h5 \\
        --bout-summary-csvs /data2/.../free_running/Predictions_3D_*/free_running_bouts_summary.csv \\
        --preproc-h5-paths /data2/.../free_running/Predictions_3D_*/preprocessing/preprocessed_bout_v1_free_running.h5
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils.free_running_loader import export_raw_free_running_h5


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('combined_h5', type=Path,
                   help='Existing combined free-running h5 to export from.')
    p.add_argument('--out', type=Path, required=True,
                   help='Destination path for the exported h5.')
    p.add_argument('--bout-summary-csvs', nargs='*', type=Path, default=None,
                   help='free_running_bouts_summary.csv files (one per '
                        'Predictions_3D dir). Shell-glob friendly.')
    p.add_argument('--preproc-h5-paths', nargs='*', type=Path, default=None,
                   help='preprocessed_bout_*_free_running.h5 files (one per '
                        'Predictions_3D dir). Used to attach orig_keypoints.')
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite --out if it already exists.')
    p.add_argument('--quiet', action='store_true',
                   help='Suppress progress output.')
    args = p.parse_args()

    if not args.combined_h5.exists():
        print(f'Error: combined_h5 not found: {args.combined_h5}', file=sys.stderr)
        sys.exit(1)

    summary = export_raw_free_running_h5(
        args.combined_h5,
        args.out,
        bout_summary_csvs=args.bout_summary_csvs,
        preproc_h5_paths=args.preproc_h5_paths,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print(f'\nsummary: {summary}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
