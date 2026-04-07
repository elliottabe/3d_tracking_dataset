"""
Batch process all Predictions_3D_* folders in a dataset.

This script finds all Predictions_3D folders and runs the preprocessing pipeline
on each one. It handles:
- Finding all prediction folders automatically
- Detecting single-fly vs dual-fly layouts (fly0/fly1)
- Running preprocess_keypoints_for_ik.py for each folder (and each fly)
- Logging results (success/failure/skipped)
- Resuming processing (skips folders with existing outputs)

Usage:
    # Process all folders
    python scripts/batch_process_predictions.py

    # Force reprocess even if outputs exist
    python scripts/batch_process_predictions.py --force

    # Dry run (show what would be processed)
    python scripts/batch_process_predictions.py --dry-run

    # Process specific anatomy version
    python scripts/batch_process_predictions.py --anatomy v1

    # Process courtship dataset (auto-detects fly0/fly1)
    python scripts/batch_process_predictions.py --dataset courtship
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path for utility imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.fly_detection import detect_flies


def find_prediction_folders(base_dir: Path) -> list:
    """
    Find all Predictions_3D_* folders in the base directory.

    Args:
        base_dir: Base directory to search

    Returns:
        List of Path objects for prediction folders, sorted by name
    """
    pattern = "Predictions_3D_*"
    folders = sorted(base_dir.glob(pattern))

    # Filter to only directories
    folders = [f for f in folders if f.is_dir()]

    return folders


def check_prerequisites(folder: Path, fly_info: dict) -> tuple:
    """
    Check if folder has required input files for a specific fly.

    Args:
        folder: Path to prediction folder
        fly_info: Dict from detect_flies() with 'csv' and 'bouts_csv' keys

    Returns:
        (has_inputs, missing_files) tuple
    """
    required_files = [
        fly_info['csv'],
        fly_info['bouts_csv'],
    ]

    missing = []
    for filename in required_files:
        if not (folder / filename).exists():
            missing.append(filename)

    return (len(missing) == 0, missing)


def check_outputs_exist(folder: Path, anatomy: str, dataset: str, fly_suffix: str) -> tuple:
    """
    Check if preprocessing outputs already exist.

    Args:
        folder: Path to prediction folder
        anatomy: Anatomy version (e.g., 'v1', 'v2_muscles')
        dataset: Dataset name
        fly_suffix: Fly suffix (e.g., '_fly0' or '' for single-fly)

    Returns:
        (exists, output_path) tuple
    """
    output_file = f"preprocessed_bout_{anatomy}_{dataset}{fly_suffix}.h5"
    output_path = folder / "preprocessing" / output_file

    return (output_path.exists(), output_path)


def run_preprocessing(folder: Path, anatomy: str, dataset: str, paths: str,
                      fly_info: dict, dry_run: bool = False) -> dict:
    """
    Run preprocessing for a single prediction folder and fly.

    Args:
        folder: Path to prediction folder
        anatomy: Anatomy version (e.g., 'v1')
        dataset: Dataset name (e.g., 'free_walking', 'courtship')
        paths: Paths config to use (e.g., 'workstation', 'hyak')
        fly_info: Dict from detect_flies() with fly-specific file info
        dry_run: If True, don't actually run, just show command

    Returns:
        Dictionary with status information
    """
    fly_label = fly_info['fly_id'] or 'single'
    result = {
        'folder': folder.name,
        'fly': fly_label,
        'status': 'unknown',
        'message': '',
        'command': ''
    }

    # Build command
    script_path = Path(__file__).parent / "preprocess_keypoints_for_ik.py"

    suffix = fly_info['suffix']
    bout_name = f"preprocessed_bout_{anatomy}_{dataset}{suffix}"

    cmd = [
        sys.executable,
        str(script_path),
        f"paths={paths}",
        f"dataset={dataset}",
        f"anatomy={anatomy}",
        f"paths.data_dir={folder}",
        f"preprocessing.csv_path={fly_info['csv']}",
        f"preprocessing.bouts_csv={fly_info['bouts_csv']}",
        f"preprocessing.bout_name={bout_name}",
    ]

    result['command'] = ' '.join(cmd)

    if dry_run:
        result['status'] = 'dry-run'
        result['message'] = f'Would run preprocessing for {fly_label} (dry-run mode)'
        return result

    # Run command
    try:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {folder.name} [{fly_label}]")
        print(f"{'='*80}")
        print(f"Command: {' '.join(cmd)}\n")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            print(line, end='', flush=True)

        proc.wait(timeout=600)

        if proc.returncode == 0:
            result['status'] = 'success'
            result['message'] = f'Preprocessing completed successfully for {fly_label}'
        else:
            result['status'] = 'failed'
            result['message'] = f'Exit code {proc.returncode}'

    except subprocess.TimeoutExpired:
        result['status'] = 'timeout'
        result['message'] = 'Processing timed out after 10 minutes'
    except Exception as e:
        result['status'] = 'error'
        result['message'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Batch process all Predictions_3D folders'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='free_walking',
        choices=['free_walking', 'courtship'],
        help='Dataset type (default: free_walking)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Base directory containing Predictions_3D folders (default: /data2/users/eabe/datasets/Johnson_lab/<dataset>)'
    )
    parser.add_argument(
        '--anatomy',
        type=str,
        default='v1',
        help='Anatomy version to use (v1, v2_muscles, etc.)'
    )
    parser.add_argument(
        '--paths',
        type=str,
        default='workstation',
        help='Paths config to use (workstation, hyak, desktop, mbook)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if outputs exist'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually running'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (default: batch_process_TIMESTAMP.log)'
    )

    args = parser.parse_args()

    if args.base_dir is None:
        args.base_dir = f'/data2/users/eabe/datasets/Johnson_lab/{args.dataset}'

    # Setup logging
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        # Create logs directory if it doesn't exist
        logs_dir = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"batch_process_{timestamp}.log"

    # Find all prediction folders
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"BATCH PREPROCESSING - {args.dataset.upper()} PREDICTIONS")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Anatomy: {args.anatomy}")
    print(f"Paths config: {args.paths}")
    print(f"Force reprocess: {args.force}")
    print(f"Dry run: {args.dry_run}")
    print(f"Log file: {log_path}")
    print()

    folders = find_prediction_folders(base_dir)

    if not folders:
        print(f"No Predictions_3D_* folders found in {base_dir}")
        sys.exit(1)

    print(f"Found {len(folders)} prediction folders:")
    for i, folder in enumerate(folders, 1):
        print(f"  [{i}] {folder.name}")
    print()

    # Process each folder
    results = []

    for folder in folders:
        print(f"\n{'-'*80}")
        print(f"Checking: {folder.name}")
        print(f"{'-'*80}")

        # Detect fly layout
        flies = detect_flies(folder, args.dataset)
        fly_labels = [f['fly_id'] or 'single' for f in flies]
        print(f"  Detected flies: {fly_labels}")

        for fly_info in flies:
            fly_label = fly_info['fly_id'] or 'single'
            fly_suffix = fly_info['suffix']

            # Check prerequisites for this fly
            has_inputs, missing_files = check_prerequisites(folder, fly_info)
            if not has_inputs:
                result = {
                    'folder': folder.name, 'fly': fly_label,
                    'status': 'missing_inputs',
                    'message': f"Missing files: {', '.join(missing_files)}",
                    'command': ''
                }
                print(f"  [{fly_label}] Skipping - missing input files: {', '.join(missing_files)}")
                results.append(result)
                continue

            # Check if outputs exist
            outputs_exist, output_path = check_outputs_exist(
                folder, args.anatomy, args.dataset, fly_suffix
            )
            if outputs_exist and not args.force:
                result = {
                    'folder': folder.name, 'fly': fly_label,
                    'status': 'skipped',
                    'message': f"Output exists: {output_path.name}",
                    'command': ''
                }
                print(f"  [{fly_label}] Skipping - output already exists: {output_path.name}")
                print("    Use --force to reprocess")
                results.append(result)
                continue

            # Run preprocessing
            print(f"  [{fly_label}] Prerequisites OK - running preprocessing...")
            folder_result = run_preprocessing(
                folder, args.anatomy, args.dataset, args.paths, fly_info, args.dry_run
            )
            results.append(folder_result)

    # Print summary
    print(f"\n\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}\n")

    status_counts = {}
    for result in results:
        status = result['status']
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"Total items: {len(results)}")
    for status, count in sorted(status_counts.items()):
        icon = {
            'success': '+',
            'failed': 'X',
            'error': 'X',
            'timeout': 'T',
            'skipped': '-',
            'missing_inputs': '!',
            'dry-run': '>'
        }.get(status, '?')
        print(f"  [{icon}] {status}: {count}")

    print(f"\nDetailed results:")
    for result in results:
        icon = {
            'success': '+',
            'failed': 'X',
            'error': 'X',
            'timeout': 'T',
            'skipped': '-',
            'missing_inputs': '!',
            'dry-run': '>'
        }.get(result['status'], '?')
        fly_str = f" [{result.get('fly', 'single')}]"
        print(f"  [{icon}] {result['folder']}{fly_str}: {result['status']} - {result['message']}")

    # Write log file
    with open(log_path, 'w') as f:
        f.write(f"Batch Processing Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Base directory: {base_dir}\n")
        f.write(f"  Dataset: {args.dataset}\n")
        f.write(f"  Anatomy: {args.anatomy}\n")
        f.write(f"  Paths: {args.paths}\n")
        f.write(f"  Force: {args.force}\n")
        f.write(f"  Dry-run: {args.dry_run}\n\n")

        f.write(f"Summary:\n")
        f.write(f"  Total items: {len(results)}\n")
        for status, count in sorted(status_counts.items()):
            f.write(f"  {status}: {count}\n")
        f.write(f"\n")

        f.write(f"Detailed Results:\n")
        f.write(f"{'-'*80}\n")
        for result in results:
            f.write(f"\nFolder: {result['folder']}\n")
            f.write(f"Fly: {result.get('fly', 'single')}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Message: {result['message']}\n")
            if result['command']:
                f.write(f"Command: {result['command']}\n")

    print(f"\nLog saved to: {log_path}")

    # Exit with error code if any failures
    if status_counts.get('failed', 0) > 0 or status_counts.get('error', 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
