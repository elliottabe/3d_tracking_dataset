#!/usr/bin/env python3
"""
Export a "raw-data only" combined courtship h5 from one or more existing
combined courtship h5 files.

The output contains only fly0/fly1 paired bouts that survive the same filters
as utils.courtship_loader.analyze_all_pairs (default
min_song_bout_frames=100, min_bilateral_dz_p95=12.0). For the production
courtship inputs this yields 110 bouts (55 pairs x 2 flies).

Per-bout payload preserved when present: kp_data, xpos_egocentric, qpos, qvel,
xpos, xquat, site_xpos, geometric_angles. When --preproc-search-paths is
supplied, orig_keypoints from the matching preprocessing h5 is attached
best-effort. No song-analysis results are saved.

Usage:
    python scripts/export_courtship_raw.py SESSION0.h5 SESSION1.h5 \\
        --out raw_courtship.h5 \\
        --preproc-search-paths /data2/.../Session0 /data2/.../Session1 \\
        --analysis-cache /tmp/analyze_all_pairs.pkl
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils.courtship_loader import export_raw_courtship_h5


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('h5_paths', nargs='+', type=Path,
                   help='Combined courtship h5 files to merge.')
    p.add_argument('--out', type=Path, required=True,
                   help='Destination path for the exported h5.')
    p.add_argument('--preproc-search-paths', nargs='*', type=Path, default=None,
                   help='Roots to search for Predictions_3D_*/preprocessing/...'
                        ' h5s (used to attach orig_keypoints). Omit to skip.')
    p.add_argument('--glob-pattern', type=str,
                   default='Predictions_3D_*/preprocessing/preprocessed_bout_v1_*_merged.h5',
                   help='Glob pattern relative to each --preproc-search-paths entry.')
    p.add_argument('--analysis-cache', type=Path, default=None,
                   help='Pickle cache for analyze_all_pairs (recommended).')
    p.add_argument('--min-song-bout-frames', type=int, default=100,
                   help='Drop pairs shorter than this many frames.')
    p.add_argument('--min-bilateral-dz-p95', type=float, default=12.0,
                   help='Drop pairs whose male wing-tip dz p95 (mm/s) is below '
                        'this. Filters non-singing pairs.')
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite --out if it already exists.')
    p.add_argument('--quiet', action='store_true',
                   help='Suppress progress output.')
    args = p.parse_args()

    for h in args.h5_paths:
        if not h.exists():
            print(f'Error: input not found: {h}', file=sys.stderr)
            sys.exit(1)

    summary = export_raw_courtship_h5(
        args.h5_paths,
        args.out,
        preproc_search_paths=args.preproc_search_paths,
        glob_pattern=args.glob_pattern,
        analysis_cache=args.analysis_cache,
        min_song_bout_frames=args.min_song_bout_frames,
        min_bilateral_dz_p95=args.min_bilateral_dz_p95,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print(f'\nsummary: {summary}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
