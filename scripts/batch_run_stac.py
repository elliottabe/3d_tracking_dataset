#!/usr/bin/env python3
"""
Batch STAC IK Processing Script

This script runs STAC inverse kinematics solver on all preprocessed prediction folders.
It automatically finds folders with preprocessed_bout_*.h5 files and runs the STAC IK pipeline
on each one using run_stac_fly_model.py which handles:
- Multi-bout format concatenation
- Automatic padding to max_length (if enabled)
- MOCAP_SCALE_FACTOR application

Supports dual-fly datasets (courtship): automatically detects fly-suffixed preprocessed
files and processes each fly independently with appropriate Hydra overrides.

Usage:
    # Dry run to see what would be processed
    python scripts/batch_run_stac.py --anatomy v1 --dry-run

    # Process all folders
    python scripts/batch_run_stac.py --anatomy v1

    # Force reprocessing even if outputs exist
    python scripts/batch_run_stac.py --anatomy v1 --force

    # Custom GPU memory fraction (default 0.9)
    python scripts/batch_run_stac.py --anatomy v1 --gpu-mem-fraction 0.8

    # Pass additional STAC config overrides
    python scripts/batch_run_stac.py --anatomy v1 --stac-overrides "dataset.stac.n_fit_frames=401"

    # Process courtship dataset (auto-detects fly0/fly1)
    python scripts/batch_run_stac.py --dataset courtship --anatomy v1

Directory structure:
    /data2/users/eabe/datasets/Johnson_lab/free_walking/
        Predictions_3D_20260202-171900/
            preprocessed_bout_v1.h5          <- Input
            Fruitfly_fit_v1_free.h5         <- Output (fit_offsets stage)
            Fruitfly_ik_v1_free.h5          <- Output (ik_only stage)
    /data2/users/eabe/datasets/Johnson_lab/courtship/
        Predictions_3D_34327248/
            preprocessed_bout_v1_courtship_fly0.h5  <- Input (fly0)
            preprocessed_bout_v1_courtship_fly1.h5  <- Input (fly1)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import os

# Add project root to path for utility imports
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from utils import io_dict_to_hdf5 as ioh5  # noqa: E402


def merge_fly_preprocessed(fly_files: list[Path], out_path: Path,
                           force: bool = False) -> Path:
    """
    Merge per-fly preprocessed bout h5 files into a single combined file so
    STAC can run once over all bouts (avoids JAX recompile per fly).

    Concatenates bout_NNN entries (renumbering sequentially) and merges
    info/clip_lengths, info/fly_ids, info/source_flies as parallel lists.
    """
    if out_path.exists() and not force:
        return out_path

    print(f"  Merging {len(fly_files)} per-fly preprocessed files into {out_path.name}")

    combined: dict = {}
    info: dict = {
        'clip_lengths': [],
        'fly_ids': [],
        'source_flies': [],
    }
    bout_counter = 0

    for fp in fly_files:
        d = ioh5.load(fp, enable_jax=False)
        bout_keys = sorted(k for k in d.keys() if k != 'info')
        sub_info = d.get('info', {})
        sub_fly_ids = list(sub_info.get('fly_ids', []))
        sub_source = list(sub_info.get('source_flies', []))
        sub_clip = list(sub_info.get('clip_lengths', []))

        for i, bk in enumerate(bout_keys):
            new_key = f'bout_{bout_counter:03d}'
            combined[new_key] = d[bk]
            if i < len(sub_clip):
                info['clip_lengths'].append(int(sub_clip[i]))
            else:
                info['clip_lengths'].append(int(d[bk]['keypoints'].shape[0]))
            info['fly_ids'].append(sub_fly_ids[i] if i < len(sub_fly_ids) else '')
            info['source_flies'].append(sub_source[i] if i < len(sub_source) else '')
            bout_counter += 1

    combined['info'] = info
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ioh5.save(out_path, combined)
    print(f"  ✓ Combined {bout_counter} bouts → {out_path}")
    return out_path


def get_stac_environment(gpu_mem_fraction: float = 0.9) -> dict:
    """
    Create environment dict with settings for headless GPU rendering and JAX optimization.

    Args:
        gpu_mem_fraction: Fraction of GPU memory to allocate (0.0-1.0)

    Returns:
        dict: Environment variables for subprocess
    """
    env = os.environ.copy()

    # Headless rendering
    env['MUJOCO_GL'] = 'egl'
    env['PYOPENGL_PLATFORM'] = 'egl'

    # GPU memory management
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(gpu_mem_fraction)

    # JAX compilation cache
    env['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
    env['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '-1'
    env['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'

    # GPU optimizations
    env['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

    cache_dir = Path(env['JAX_COMPILATION_CACHE_DIR'])
    cache_dir.mkdir(exist_ok=True)
    env['XLA_FLAGS'] += f' --xla_gpu_per_fusion_autotune_cache_dir={cache_dir}'

    return env


def _default_stac_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "stac-mjx"


def _default_stac_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "stac-mjx"


MERGED_SUFFIX = '_merged'


def find_preprocessed_files(base_dir: Path, anatomy_name: str, dataset: str,
                            force_merge: bool = False) -> list[tuple[Path, str, str]]:
    """
    Find all Predictions_3D_* folders with preprocessed bout files.

    Detects both single-fly (preprocessed_bout_v1_courtship.h5) and dual-fly
    (preprocessed_bout_v1_courtship_fly0.h5) layouts.

    Args:
        base_dir: Base directory containing prediction folders
        anatomy_name: Anatomy version (e.g., 'v1', 'v2')
        dataset: Dataset name (e.g., 'free_walking', 'courtship')

    Returns:
        List of (folder_path, version_name, fly_suffix) tuples.
        fly_suffix is '' for single-fly or '_fly0'/'_fly1' for dual-fly.
    """
    items = []

    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return []

    # Allow base_dir to itself be a single Predictions_3D_* folder (per-folder slurm jobs)
    if base_dir.is_dir() and base_dir.match("Predictions_3D_*"):
        candidate_folders = [base_dir]
    else:
        candidate_folders = sorted(base_dir.glob("Predictions_3D_*"))
    for folder in candidate_folders:
        if not folder.is_dir():
            continue

        version_name = folder.name
        preproc_dir = folder / "preprocessing"

        # Check for fly-suffixed files first; if present, merge them into a
        # single _merged h5 so STAC can run once and avoid recompiling per fly.
        # Prefer *_paired.h5 variants (validity-filtered by the pair step) when
        # they exist — they replace the raw per-fly files for the STAC run.
        paired_stem = f"preprocessed_bout_{anatomy_name}_{dataset}"
        paired_fly_files = sorted(preproc_dir.glob(f"{paired_stem}_fly*_paired.h5"))
        if paired_fly_files:
            fly_files = paired_fly_files
        else:
            fly_files = sorted(preproc_dir.glob(
                f"{paired_stem}_fly*.h5"))
            # Exclude any stray _paired.h5 that isn't one of the per-fly files
            fly_files = [f for f in fly_files if not f.name.endswith("_paired.h5")]
        if fly_files:
            merged_path = preproc_dir / f"preprocessed_bout_{anatomy_name}_{dataset}{MERGED_SUFFIX}.h5"
            try:
                merge_fly_preprocessed(fly_files, merged_path, force=force_merge)
                items.append((folder, version_name, MERGED_SUFFIX))
            except Exception as e:
                print(f"  ⚠ Failed to merge per-fly files in {folder.name}: {e}")
        else:
            # Check for standard single-fly file
            standard_file = preproc_dir / f"preprocessed_bout_{anatomy_name}_{dataset}.h5"
            if standard_file.exists():
                items.append((folder, version_name, ''))
            else:
                print(f"  Skipping {folder.name}: No preprocessed file found")

    return items


def check_stac_outputs_exist(folder: Path, anatomy_name: str, dataset: str, fly_suffix: str = '') -> tuple[bool, bool]:
    """
    Check if STAC output files already exist.

    Args:
        folder: Prediction folder path
        anatomy_name: Anatomy version
        dataset: Dataset name
        fly_suffix: Fly suffix (e.g., '_fly0' or '')

    Returns:
        (fit_offsets_exists, ik_only_exists) tuple
    """
    fit_file = folder / "stac" / f"Fruitfly_fit_{anatomy_name}_{dataset}{fly_suffix}.h5"
    ik_file = folder / "stac" / f"Fruitfly_ik_{anatomy_name}_{dataset}{fly_suffix}.h5"

    return fit_file.exists(), ik_file.exists()


def run_stac(
    folder: Path,
    version_name: str,
    anatomy_name: str,
    dataset: str,
    stac_dir: Path,
    fly_suffix: str = '',
    gpu_mem_fraction: float = 0.9,
    stac_overrides: str = "",
    paths_config: str = "workstation",
    dry_run: bool = False
) -> tuple[bool, str]:
    """
    Run STAC IK solver on a single folder/fly.

    Args:
        folder: Path to prediction folder
        version_name: Version name for config override (folder name)
        anatomy_name: Anatomy version
        dataset: Dataset name
        stac_dir: Path to stac-mjx directory
        fly_suffix: Fly suffix for input/output filenames (e.g., '_fly0' or '')
        gpu_mem_fraction: Fraction of GPU memory to allocate
        stac_overrides: Additional Hydra config overrides
        dry_run: If True, only print command without running

    Returns:
        (success, message) tuple
    """
    cmd = [
        sys.executable,
        "run_stac_fly_model.py",
        f"paths={paths_config}",
        f"dataset={dataset}",
        f"anatomy={anatomy_name}",
        f"version={version_name}",
        # Override base_dir/data_dir to the actual folder so nested layouts
        # (e.g. <dataset>/sessionX/Predictions_3D_*) are handled correctly
        # instead of the default <dataset>/<version> assumption.
        f"paths.base_dir={folder}",
        f"paths.data_dir={folder}",
        "run_id=stac",
    ]

    # Add fly-specific overrides for input/output filenames
    if fly_suffix:
        input_filename = f"preprocessed_bout_{anatomy_name}_{dataset}{fly_suffix}.h5"
        fit_path = f"Fruitfly_fit_{anatomy_name}_{dataset}{fly_suffix}.h5"
        ik_path = f"Fruitfly_ik_{anatomy_name}_{dataset}{fly_suffix}.h5"
        cmd.extend([
            f"dataset.preprocessing.input_filename={input_filename}",
            f"dataset.stac.fit_offsets_path={fit_path}",
            f"dataset.stac.ik_only_path={ik_path}",
        ])

    # Add any additional STAC overrides
    if stac_overrides:
        overrides = [o.strip() for o in stac_overrides.split() if o.strip()]
        cmd.extend(overrides)

    fly_label = fly_suffix.lstrip('_') if fly_suffix else 'single'

    if dry_run:
        print(f"  Would run: cd {stac_dir} && {' '.join(cmd)}")
        return True, "Dry run"

    # Set up environment
    env = get_stac_environment(gpu_mem_fraction)

    # Log file
    log_file = folder / "stac" / f"stac_batch_{anatomy_name}{fly_suffix}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  Running STAC IK solver [{fly_label}]...")
        print(f"  Log file: {log_file}")

        with open(log_file, 'w') as lf:
            lf.write(f"STAC run started: {datetime.now().isoformat()}\n")
            lf.write(f"Command: {' '.join(cmd)}\n")
            lf.write(f"{'='*80}\n\n")
            lf.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=stac_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in proc.stdout:
                sys.stdout.write(f"  | {line}")
                sys.stdout.flush()
                lf.write(line)
                lf.flush()

            proc.wait(timeout=6*3600)

            lf.write(f"\n{'='*80}\n")
            lf.write(f"STAC run finished: {datetime.now().isoformat()}\n")
            lf.write(f"Return code: {proc.returncode}\n")

        if proc.returncode == 0:
            return True, "Success"
        else:
            return False, f"Failed with return code {proc.returncode} (see {log_file})"

    except subprocess.TimeoutExpired:
        proc.kill()
        return False, "Timeout after 6 hours"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch run STAC IK solver on preprocessed prediction folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be processed
  python scripts/batch_run_stac.py --anatomy v1 --dry-run

  # Process all folders
  python scripts/batch_run_stac.py --anatomy v1

  # Force reprocessing
  python scripts/batch_run_stac.py --anatomy v1 --force

  # Process courtship (auto-detects fly0/fly1)
  python scripts/batch_run_stac.py --dataset courtship --anatomy v1
        """
    )

    parser.add_argument(
        '--anatomy',
        type=str,
        default='v1',
        help='Anatomy version (v1, v2, etc.)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='free_walking',
        choices=['free_walking', 'courtship', 'stationary'],
        help='Dataset type (default: free_walking)'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=None,
        help='Base directory containing Predictions_3D_* folders'
    )
    parser.add_argument(
        '--paths',
        type=str,
        default='workstation',
        help='Paths config (workstation, hyak, etc.)'
    )
    parser.add_argument(
        '--stac-dir',
        type=Path,
        default=_default_stac_dir(),
        help='Path to stac-mjx directory'
    )
    parser.add_argument(
        '--gpu-mem-fraction',
        type=float,
        default=0.9,
        help='Fraction of GPU memory for JAX (0.0-1.0, default 0.9)'
    )
    parser.add_argument(
        '--stac-overrides',
        type=str,
        default='',
        help='Additional Hydra config overrides for STAC'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if output files exist'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be done without actually running'
    )

    args = parser.parse_args()

    if args.base_dir is None:
        args.base_dir = Path(f'/data2/users/eabe/datasets/Johnson_lab/{args.dataset}')

    # Validate stac-mjx directory
    if not args.stac_dir.exists():
        print(f"Error: stac-mjx directory not found: {args.stac_dir}")
        print("   Make sure the stac-mjx submodule is initialized:")
        print("   git submodule update --init --recursive")
        return 1

    run_stac_script = args.stac_dir / "run_stac_fly_model.py"
    if not run_stac_script.exists():
        print(f"Error: run_stac_fly_model.py not found in {args.stac_dir}")
        return 1

    # Find all preprocessed files
    print(f"\n{'='*80}")
    print(f"STAC Batch Processing")
    print(f"{'='*80}")
    print(f"Base directory: {args.base_dir}")
    print(f"Anatomy: {args.anatomy}")
    print(f"Dataset: {args.dataset}")
    print(f"STAC directory: {args.stac_dir}")
    print(f"GPU memory fraction: {args.gpu_mem_fraction}")
    if args.stac_overrides:
        print(f"STAC overrides: {args.stac_overrides}")
    print(f"Force reprocessing: {args.force}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*80}\n")

    preprocessed_items = find_preprocessed_files(
        args.base_dir, args.anatomy, args.dataset, force_merge=args.force
    )

    if not preprocessed_items:
        print("No prediction folders with preprocessed data found!")
        print(f"   Looking for: {args.base_dir}/Predictions_3D_*/preprocessing/preprocessed_bout_{args.anatomy}_{args.dataset}*.h5")
        print("   Run batch_process_predictions.py first to create preprocessed files.")
        return 1

    print(f"Found {len(preprocessed_items)} preprocessed item(s):\n")

    # Process each item
    results = []
    skipped = []

    for folder, version_name, fly_suffix in preprocessed_items:
        fly_label = fly_suffix.lstrip('_') if fly_suffix else 'single'
        print(f"  {folder.name} [{fly_label}]")

        # Check if outputs already exist
        fit_exists, ik_exists = check_stac_outputs_exist(folder, args.anatomy, args.dataset, fly_suffix)

        if (fit_exists and ik_exists) and not args.force:
            print(f"    Skipping: Output files already exist (use --force to reprocess)")
            skipped.append(f"{folder.name} [{fly_label}]")
            print()
            continue

        if fit_exists or ik_exists:
            print(f"    Partial outputs exist:")
            if fit_exists:
                print(f"      + Fruitfly_fit_{args.anatomy}_{args.dataset}{fly_suffix}.h5")
            else:
                print(f"      - Fruitfly_fit_{args.anatomy}_{args.dataset}{fly_suffix}.h5")
            if ik_exists:
                print(f"      + Fruitfly_ik_{args.anatomy}_{args.dataset}{fly_suffix}.h5")
            else:
                print(f"      - Fruitfly_ik_{args.anatomy}_{args.dataset}{fly_suffix}.h5")
            if args.force:
                print(f"      Reprocessing due to --force flag")

        # Run STAC
        success, message = run_stac(
            folder,
            version_name,
            args.anatomy,
            args.dataset,
            args.stac_dir,
            fly_suffix,
            args.gpu_mem_fraction,
            args.stac_overrides,
            args.paths,
            args.dry_run
        )

        results.append((f"{folder.name} [{fly_label}]", success, message))

        if success:
            print(f"  [+] {message}")
        else:
            print(f"  [X] {message}")

        print()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if skipped:
        print(f"Skipped {len(skipped)} item(s) with existing outputs:")
        for name in skipped:
            print(f"   - {name}")
        print()

    if results:
        successful = sum(1 for _, success, _ in results if success)
        failed = len(results) - successful

        print(f"Successful: {successful}/{len(results)}")
        if failed > 0:
            print(f"Failed: {failed}/{len(results)}")
            print("\nFailed items:")
            for name, success, message in results:
                if not success:
                    print(f"   - {name}: {message}")

    # Create log file
    if not args.dry_run and results:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"batch_stac_{timestamp}.log"

        with open(log_file, 'w') as f:
            f.write(f"STAC Batch Processing Log - {timestamp}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Anatomy: {args.anatomy}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"GPU memory fraction: {args.gpu_mem_fraction}\n")
            if args.stac_overrides:
                f.write(f"STAC overrides: {args.stac_overrides}\n")
            f.write(f"Force: {args.force}\n")
            f.write(f"\n")

            if skipped:
                f.write(f"Skipped items ({len(skipped)}):\n")
                for name in skipped:
                    f.write(f"  - {name}\n")
                f.write(f"\n")

            if results:
                f.write(f"Processed items ({len(results)}):\n")
                for name, success, message in results:
                    status = "[+]" if success else "[X]"
                    f.write(f"  {status} {name}: {message}\n")

        print(f"\nLog saved to: {log_file}")

    print(f"{'='*80}\n")

    # Return appropriate exit code
    if results:
        failed = sum(1 for _, success, _ in results if not success)
        return 0 if failed == 0 else 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
