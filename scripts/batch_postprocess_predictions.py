"""
Batch postprocess all Predictions_3D_* folders after STAC IK has run.

This script finds all Predictions_3D folders and runs the postprocessing pipeline
on each one (after STAC IK solver has been run). It handles:
- Finding all prediction folders with STAC outputs
- Running postprocess_stac_data.py for each folder
- Logging results (success/failure/skipped)
- Resuming processing (skips folders with existing outputs)

Usage:
    # Postprocess all folders with STAC outputs
    python scripts/batch_postprocess_predictions.py
    
    # Force reprocess even if outputs exist
    python scripts/batch_postprocess_predictions.py --force
    
    # Dry run (show what would be processed)
    python scripts/batch_postprocess_predictions.py --dry-run
    
    # Process specific anatomy version
    python scripts/batch_postprocess_predictions.py --anatomy v1
    
    # Process with specific paths config
    python scripts/batch_postprocess_predictions.py --paths workstation
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse



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


def check_stac_outputs_exist(folder: Path, anatomy: str, dataset_suffix: str) -> tuple:
    """
    Check if STAC IK outputs exist (required for postprocessing).

    Args:
        folder: Path to prediction folder
        anatomy: Anatomy version (e.g., 'v1', 'v2_muscles')
        dataset_suffix: Short dataset suffix used in filenames (e.g., 'free', 'courtship')

    Returns:
        (exists, stac_path) tuple
    """
    stac_file = f"Fruitfly_ik_{anatomy}_{dataset_suffix}.h5"
    stac_path = folder / stac_file

    return (stac_path.exists(), stac_path)


def check_postprocess_outputs_exist(folder: Path, anatomy: str) -> tuple:
    """
    Check if postprocessing outputs already exist.
    
    Args:
        folder: Path to prediction folder
        anatomy: Anatomy version (e.g., 'v1', 'v2_muscles')
        
    Returns:
        (exists, output_path) tuple
    """
    output_file = f"ik_output_{anatomy}.h5"
    output_path = folder / output_file
    
    return (output_path.exists(), output_path)


def run_postprocessing(folder: Path, anatomy: str, dataset: str, paths: str, dry_run: bool = False) -> dict:
    """
    Run postprocessing for a single prediction folder.
    
    Args:
        folder: Path to prediction folder
        anatomy: Anatomy version (e.g., 'v1')
        dataset: Dataset name (e.g., 'free_walking', 'courtship')
        paths: Paths config to use (e.g., 'workstation', 'hyak')
        dry_run: If True, don't actually run, just show command

    Returns:
        Dictionary with status information
    """
    result = {
        'folder': folder.name,
        'status': 'unknown',
        'message': '',
        'command': ''
    }

    # Build command
    script_path = Path(__file__).parent / "postprocess_stac_data.py"

    cmd = [
        sys.executable,  # Use same Python interpreter
        str(script_path),
        f"paths={paths}",
        f"dataset={dataset}",
        f"anatomy={anatomy}",
        f"paths.data_dir={folder}",
    ]
    
    result['command'] = ' '.join(cmd)
    
    if dry_run:
        result['status'] = 'dry-run'
        result['message'] = 'Would run postprocessing (dry-run mode)'
        return result
    
    # Run command
    try:
        print(f"\n{'='*80}")
        print(f"POSTPROCESSING: {folder.name}")
        print(f"{'='*80}")
        print(f"Command: {' '.join(cmd)}\n")
        
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if proc.returncode == 0:
            result['status'] = 'success'
            result['message'] = 'Postprocessing completed successfully'
        else:
            result['status'] = 'failed'
            result['message'] = f'Exit code {proc.returncode}'
            # Print stderr for debugging
            if proc.stderr:
                print(f"\nSTDERR:\n{proc.stderr}")
        
        # Print stdout
        if proc.stdout:
            print(proc.stdout)
            
    except subprocess.TimeoutExpired:
        result['status'] = 'timeout'
        result['message'] = 'Processing timed out after 10 minutes'
    except Exception as e:
        result['status'] = 'error'
        result['message'] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Batch postprocess all Predictions_3D folders (after STAC IK)'
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
        help='Path to log file (default: batch_postprocess_TIMESTAMP.log)'
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
        log_path = logs_dir / f"batch_postprocess_{timestamp}.log"
    
    # Find all prediction folders
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"❌ Error: Base directory not found: {base_dir}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"BATCH POSTPROCESSING - {args.dataset.upper()} PREDICTIONS")
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
        print(f"❌ No Predictions_3D_* folders found in {base_dir}")
        sys.exit(1)
    
    print(f"Found {len(folders)} prediction folders:")
    for i, folder in enumerate(folders, 1):
        print(f"  [{i}] {folder.name}")
    print()
    
    # Process each folder
    results = []
    
    for folder in folders:
        folder_result = {
            'folder': folder.name,
            'status': 'unknown',
            'message': '',
            'command': ''
        }
        
        print(f"\n{'-'*80}")
        print(f"Checking: {folder.name}")
        print(f"{'-'*80}")
        
        # Check if STAC outputs exist
        stac_exists, stac_path = check_stac_outputs_exist(folder, args.anatomy, args.dataset)
        if not stac_exists:
            folder_result['status'] = 'missing_stac'
            folder_result['message'] = f"STAC output not found: {stac_path.name}"
            print(f"⚠ Skipping - STAC IK output not found: {stac_path.name}")
            print("  Run STAC IK solver first")
            results.append(folder_result)
            continue
        
        # Check if postprocessing outputs exist
        outputs_exist, output_path = check_postprocess_outputs_exist(folder, args.anatomy)
        if outputs_exist and not args.force:
            folder_result['status'] = 'skipped'
            folder_result['message'] = f"Output exists: {output_path.name}"
            print(f"✓ Skipping - output already exists: {output_path.name}")
            print("  Use --force to reprocess")
            results.append(folder_result)
            continue
        
        # Run postprocessing
        print(f"✓ STAC output found - running postprocessing...")
        folder_result = run_postprocessing(folder, args.anatomy, args.dataset, args.paths, args.dry_run)
        results.append(folder_result)
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("BATCH POSTPROCESSING SUMMARY")
    print(f"{'='*80}\n")
    
    status_counts = {}
    for result in results:
        status = result['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"Total folders: {len(results)}")
    for status, count in sorted(status_counts.items()):
        icon = {
            'success': '✓',
            'failed': '✗',
            'error': '✗',
            'timeout': '⏱',
            'skipped': '⊘',
            'missing_stac': '⚠',
            'dry-run': '▷'
        }.get(status, '?')
        print(f"  {icon} {status}: {count}")
    
    print(f"\nDetailed results:")
    for result in results:
        icon = {
            'success': '✓',
            'failed': '✗',
            'error': '✗',
            'timeout': '⏱',
            'skipped': '⊘',
            'missing_stac': '⚠',
            'dry-run': '▷'
        }.get(result['status'], '?')
        print(f"  {icon} {result['folder']}: {result['status']} - {result['message']}")
    
    # Write log file
    with open(log_path, 'w') as f:
        f.write(f"Batch Postprocessing Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Base directory: {base_dir}\n")
        f.write(f"  Dataset: {args.dataset}\n")
        f.write(f"  Anatomy: {args.anatomy}\n")
        f.write(f"  Paths: {args.paths}\n")
        f.write(f"  Force: {args.force}\n")
        f.write(f"  Dry-run: {args.dry_run}\n\n")
        
        f.write(f"Summary:\n")
        f.write(f"  Total folders: {len(results)}\n")
        for status, count in sorted(status_counts.items()):
            f.write(f"  {status}: {count}\n")
        f.write(f"\n")
        
        f.write(f"Detailed Results:\n")
        f.write(f"{'-'*80}\n")
        for result in results:
            f.write(f"\nFolder: {result['folder']}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Message: {result['message']}\n")
            if result['command']:
                f.write(f"Command: {result['command']}\n")
    
    print(f"\n✓ Log saved to: {log_path}")
    
    # Exit with error code if any failures
    if status_counts.get('failed', 0) > 0 or status_counts.get('error', 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
