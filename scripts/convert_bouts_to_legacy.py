#!/usr/bin/env python3
"""
Materialise legacy-format files inside Predictions_3D_* folders that only
contain the newer bout_*/ layout.

For each Predictions_3D_* folder with bout_<N>/fly{0,1}.csv subdirectories,
this script calls utils.fly_detection.aggregate_per_bout_predictions which
writes:
    data3D_fly{0,1}.csv
    tracking_info.json
    <dataset>_bouts_fly{0,1}_summary.csv
    <dataset>_bouts_unified_summary.csv

Idempotent: folders already carrying data3D_fly0.csv are skipped unless
--force is passed. Folders that do not contain a bout_*/ layout are also
skipped (nothing to convert).

Usage:
    python scripts/convert_bouts_to_legacy.py --dataset courtship \\
        --base-dir /path/to/Session/<timestamp>
    python scripts/convert_bouts_to_legacy.py --dataset courtship \\
        --base-dir .../Predictions_3D_34662304 --force
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils.fly_detection import aggregate_per_bout_predictions


def find_prediction_folders(base_dir: Path) -> list[Path]:
    """Return Predictions_3D_* folders under base_dir (or base_dir itself)."""
    pattern = "Predictions_3D_*"
    if base_dir.is_dir() and base_dir.match(pattern):
        return [base_dir]
    return sorted({p for p in base_dir.rglob(pattern) if p.is_dir()})


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default="courtship",
                   help="Dataset name (used for the bouts summary filenames)")
    p.add_argument("--base-dir", type=Path, required=True,
                   help="Folder to scan (may itself be a Predictions_3D_*)")
    p.add_argument("--force", action="store_true",
                   help="Rewrite legacy files even if outputs are up-to-date")
    p.add_argument("--dry-run", action="store_true",
                   help="List folders that would be converted and exit")
    args = p.parse_args()

    if not args.base_dir.exists():
        print(f"Error: base-dir not found: {args.base_dir}", file=sys.stderr)
        sys.exit(1)

    folders = find_prediction_folders(args.base_dir)
    if not folders:
        print(f"No Predictions_3D_* folders under {args.base_dir} — nothing to do.")
        return 0

    print(f"Found {len(folders)} Predictions_3D_* folder(s) under {args.base_dir}")

    n_converted = 0
    n_skipped = 0
    n_noop = 0
    for folder in folders:
        has_bouts = any(folder.glob("bout_*/fly0.csv"))
        has_legacy = (folder / "data3D_fly0.csv").exists()

        if not has_bouts:
            print(f"  - {folder.name}: no bout_*/ layout, skipping")
            n_skipped += 1
            continue
        if has_legacy and not args.force:
            print(f"  - {folder.name}: legacy files present, skipping "
                  f"(use --force to rewrite)")
            n_skipped += 1
            continue
        if args.dry_run:
            print(f"  * {folder.name}: would convert (dry-run)")
            continue

        print(f"  * {folder.name}: converting...")
        wrote = aggregate_per_bout_predictions(folder, args.dataset,
                                               force=args.force)
        if wrote:
            n_converted += 1
            print(f"    wrote data3D_fly[0,1].csv, tracking_info.json, "
                  f"{args.dataset}_bouts_fly[0,1]_summary.csv, "
                  f"{args.dataset}_bouts_unified_summary.csv")
        else:
            n_noop += 1
            print(f"    no-op (outputs up-to-date)")

    print(f"\nDone: {n_converted} converted, {n_noop} up-to-date, "
          f"{n_skipped} skipped (of {len(folders)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
