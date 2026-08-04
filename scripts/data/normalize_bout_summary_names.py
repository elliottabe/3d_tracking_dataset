"""Normalize free-running bout-summary CSV filenames to the canonical name.

Commit 451efb9 renamed the dataset walking -> running in code and configs, but
the data files on disk kept their old names. This brings them in line so
utils/fly_detection.py's f'{dataset}_bouts_summary.csv' resolves everywhere.

Renames (per Predictions_3D_* dir):
    free_walking_bouts_summary.csv -> free_running_bouts_summary.csv
    free_running_bout_summary.csv  -> free_running_bouts_summary.csv

Leaves OLD_walking_bouts_summary.csv alone -- the OLD_ prefix marks it as
deliberately superseded.

Idempotent and reversible. --dry-run prints the plan without touching anything;
--undo reverses a previous run using the sidecar manifest.

Usage:
    python scripts/data/normalize_bout_summary_names.py --root <dir> --dry-run
    python scripts/data/normalize_bout_summary_names.py --root <dir>
    python scripts/data/normalize_bout_summary_names.py --root <dir> --undo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CANONICAL = 'free_running_bouts_summary.csv'
LEGACY = ['free_walking_bouts_summary.csv', 'free_running_bout_summary.csv']
MANIFEST = '.bout_summary_rename_manifest.json'


def plan_renames(root: Path) -> list[tuple[Path, Path]]:
    """One (src, dst) per dir that has a legacy name and no canonical file."""
    renames = []
    for d in sorted(root.rglob('Predictions_3D_*')):
        if not d.is_dir():
            continue
        if (d / CANONICAL).is_file():
            continue  # already normalized
        found = [d / name for name in LEGACY if (d / name).is_file()]
        if len(found) > 1:
            raise SystemExit(
                f'ERROR: {d} has multiple legacy names {[f.name for f in found]}; '
                f'resolve by hand')
        if found:
            renames.append((found[0], d / CANONICAL))
    return renames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--undo', action='store_true')
    args = ap.parse_args()

    manifest_path = args.root / MANIFEST

    if args.undo:
        if not manifest_path.is_file():
            raise SystemExit(f'ERROR: no manifest at {manifest_path}')
        entries = json.loads(manifest_path.read_text())
        for dst, src in entries:  # reverse direction
            Path(dst).rename(src)
            print(f'undo: {Path(dst).name} -> {Path(src).name}  in {Path(src).parent}')
        manifest_path.unlink()
        print(f'reverted {len(entries)} renames')
        return 0

    renames = plan_renames(args.root)
    if not renames:
        print('nothing to do -- all Predictions_3D_* dirs already canonical')
        return 0

    for src, dst in renames:
        print(f'{"[dry-run] " if args.dry_run else ""}{src.name} -> {dst.name}'
              f'  in {src.parent}')
    if args.dry_run:
        print(f'\n{len(renames)} would be renamed')
        return 0

    for src, dst in renames:
        src.rename(dst)
    manifest_path.write_text(
        json.dumps([[str(s), str(d)] for s, d in renames], indent=2))
    print(f'\nrenamed {len(renames)}; manifest at {manifest_path} (--undo to revert)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
