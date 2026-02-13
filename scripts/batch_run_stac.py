#!/usr/bin/env python3
"""
Batch STAC IK Processing Script

This script runs STAC inverse kinematics solver on all preprocessed prediction folders.
It automatically finds folders with preprocessed_bout_*.h5 files and runs the STAC IK pipeline
on each one using run_stac_fly_model.py which handles:
- Multi-bout format concatenation
- Automatic padding to max_length (if enabled)
- MOCAP_SCALE_FACTOR application

The script sets up proper environment variables for headless GPU rendering (MUJOCO_GL),
GPU memory management (XLA_PYTHON_CLIENT_MEM_FRACTION), and JAX compilation caching
following the pattern in stac-mjx/run_stac_fly_model.py.

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
    
    # Skip fit_offsets stage (only run ik_only)
    python scripts/batch_run_stac.py --anatomy v1 --stac-overrides "dataset.stac.skip_fit_offsets=True"

Directory structure:
    /data2/users/eabe/datasets/Johnson_lab/free_walking/
        Predictions_3D_20260202-171900/
            preprocessed_bout_v1.h5          <- Input
            Fruitfly_fit_v1_free.h5         <- Output (fit_offsets stage)
            Fruitfly_ik_v1_free.h5          <- Output (ik_only stage)
        Predictions_3D_20260203-103416/
            ...
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import os


def get_stac_environment(gpu_mem_fraction: float = 0.9) -> dict:
    """
    Create environment dict with settings for headless GPU rendering and JAX optimization.
    
    Based on stac-mjx/demos/run_stac_fly_model.py environment setup:
    - MUJOCO_GL='egl': Use EGL for headless rendering (no display needed)
    - PYOPENGL_PLATFORM='egl': PyOpenGL backend for headless
    - XLA_PYTHON_CLIENT_MEM_FRACTION: Fraction of GPU memory for JAX (default 0.9)
    - JAX compilation cache: Speed up repeated runs
    - XLA_FLAGS: Enable GPU optimizations
    
    Args:
        gpu_mem_fraction: Fraction of GPU memory to allocate (0.0-1.0)
    
    Returns:
        dict: Environment variables for subprocess
    """
    env = os.environ.copy()
    
    # Headless rendering (critical for cluster/server execution)
    env['MUJOCO_GL'] = 'egl'
    env['PYOPENGL_PLATFORM'] = 'egl'
    
    # GPU memory management
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(gpu_mem_fraction)
    
    # JAX compilation cache (speeds up repeated runs)
    env['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
    env['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '-1'
    env['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'
    
    # GPU optimizations
    env['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
    
    # Per-fusion autotune cache
    cache_dir = Path(env['JAX_COMPILATION_CACHE_DIR'])
    cache_dir.mkdir(exist_ok=True)
    env['XLA_FLAGS'] += f' --xla_gpu_per_fusion_autotune_cache_dir={cache_dir}'
    
    return env


def find_prediction_folders(base_dir: Path, anatomy_name: str) -> list[tuple[Path, str]]:
    """
    Find all Predictions_3D_* folders that have preprocessed bout files.
    
    Args:
        base_dir: Base directory containing prediction folders
        anatomy_name: Anatomy version (e.g., 'v1', 'v2')
    
    Returns:
        List of (folder_path, version_name) tuples
    """
    prediction_folders = []
    
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return []
    
    # Find all Predictions_3D_* directories
    for folder in sorted(base_dir.glob("Predictions_3D_*")):
        if not folder.is_dir():
            continue
        
        # Check if preprocessed file exists
        preprocessed_file = folder / f"preprocessed_bout_{anatomy_name}.h5"
        if not preprocessed_file.exists():
            print(f"⚠️  Skipping {folder.name}: No preprocessed file found")
            continue
        
        # Extract version name (folder name)
        version_name = folder.name
        prediction_folders.append((folder, version_name))
    
    return prediction_folders


def check_stac_outputs_exist(folder: Path, anatomy_name: str) -> tuple[bool, bool]:
    """
    Check if STAC output files already exist.
    
    Args:
        folder: Prediction folder path
        anatomy_name: Anatomy version (e.g., 'v1', 'v2')
    
    Returns:
        (fit_offsets_exists, ik_only_exists) tuple
    """
    fit_file = folder / f"Fruitfly_fit_{anatomy_name}_free.h5"
    ik_file = folder / f"Fruitfly_ik_{anatomy_name}_free.h5"
    
    return fit_file.exists(), ik_file.exists()


def run_stac(
    folder: Path,
    version_name: str,
    anatomy_name: str,
    stac_dir: Path,
    gpu_mem_fraction: float = 0.9,
    stac_overrides: str = "",
    dry_run: bool = False
) -> tuple[bool, str]:
    """
    Run STAC IK solver on a single folder.
    
    Args:
        folder: Path to prediction folder
        version_name: Version name for config override (folder name)
        anatomy_name: Anatomy version (e.g., 'v1', 'v2')
        stac_dir: Path to stac-mjx directory
        gpu_mem_fraction: Fraction of GPU memory to allocate
        stac_overrides: Additional Hydra config overrides
        dry_run: If True, only print command without running
    
    Returns:
        (success, message) tuple
    """
    # Build command with Hydra overrides
    # Use run_stac_fly_model.py which handles multi-bout concatenation and padding
    cmd = [
        sys.executable,  # Current Python interpreter
        "run_stac_fly_model.py",
        f"paths=workstation",
        f"dataset=free_walking",
        f"anatomy={anatomy_name}",
        f"version={version_name}",
        "run_id=batch_stac",
    ]
    
    # Add any additional STAC overrides
    if stac_overrides:
        # Split on spaces, preserving quoted values
        overrides = [o.strip() for o in stac_overrides.split() if o.strip()]
        cmd.extend(overrides)
    
    if dry_run:
        print(f"  Would run: cd {stac_dir} && {' '.join(cmd)}")
        return True, "Dry run"

    # Set up environment with GPU/rendering settings
    env = get_stac_environment(gpu_mem_fraction)

    # Log file co-located with the data
    log_file = folder / f"stac_batch_{anatomy_name}.log"

    try:
        print(f"  Running STAC IK solver...")
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

            # Stream output to both console and log file
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
  
  # Adjust GPU memory
  python scripts/batch_run_stac.py --anatomy v1 --gpu-mem-fraction 0.8
  
  # Pass STAC config overrides
  python scripts/batch_run_stac.py --anatomy v1 --stac-overrides "dataset.stac.n_fit_frames=401"
        """
    )
    
    parser.add_argument(
        '--anatomy',
        type=str,
        default='v1',
        help='Anatomy version (v1, v2, etc.)'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path('/data2/users/eabe/datasets/Johnson_lab/free_walking'),
        help='Base directory containing Predictions_3D_* folders'
    )
    parser.add_argument(
        '--stac-dir',
        type=Path,
        default=Path('/home/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx'),
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
        help='Additional Hydra config overrides for STAC (e.g., "dataset.stac.n_fit_frames=401")'
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
    
    # Validate stac-mjx directory
    if not args.stac_dir.exists():
        print(f"❌ Error: stac-mjx directory not found: {args.stac_dir}")
        print("   Make sure the stac-mjx submodule is initialized:")
        print("   git submodule update --init --recursive")
        return 1
    
    run_stac_script = args.stac_dir / "run_stac_fly_model.py"
    if not run_stac_script.exists():
        print(f"❌ Error: run_stac_fly_model.py not found in {args.stac_dir}")
        return 1
    
    # Find all prediction folders with preprocessed data
    print(f"\n{'='*80}")
    print(f"STAC Batch Processing")
    print(f"{'='*80}")
    print(f"Base directory: {args.base_dir}")
    print(f"Anatomy: {args.anatomy}")
    print(f"STAC directory: {args.stac_dir}")
    print(f"GPU memory fraction: {args.gpu_mem_fraction}")
    if args.stac_overrides:
        print(f"STAC overrides: {args.stac_overrides}")
    print(f"Force reprocessing: {args.force}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*80}\n")
    
    prediction_folders = find_prediction_folders(args.base_dir, args.anatomy)
    
    if not prediction_folders:
        print("❌ No prediction folders with preprocessed data found!")
        print(f"   Looking for: {args.base_dir}/Predictions_3D_*/preprocessed_bout_{args.anatomy}.h5")
        print("   Run batch_process_predictions.py first to create preprocessed files.")
        return 1
    
    print(f"Found {len(prediction_folders)} prediction folder(s) with preprocessed data:\n")
    
    # Process each folder
    results = []
    skipped = []
    
    for folder, version_name in prediction_folders:
        print(f"📁 {folder.name}")
        
        # Check if outputs already exist
        fit_exists, ik_exists = check_stac_outputs_exist(folder, args.anatomy)
        
        if (fit_exists and ik_exists) and not args.force:
            print(f"  ⏭️  Skipping: Output files already exist")
            print(f"     (Use --force to reprocess)")
            skipped.append(folder.name)
            print()
            continue
        
        if fit_exists or ik_exists:
            print(f"  ℹ️  Partial outputs exist:")
            if fit_exists:
                print(f"     ✓ Fruitfly_fit_{args.anatomy}_free.h5")
            else:
                print(f"     ✗ Fruitfly_fit_{args.anatomy}_free.h5")
            if ik_exists:
                print(f"     ✓ Fruitfly_ik_{args.anatomy}_free.h5")
            else:
                print(f"     ✗ Fruitfly_ik_{args.anatomy}_free.h5")
            if args.force:
                print(f"     Reprocessing due to --force flag")
        
        # Run STAC
        success, message = run_stac(
            folder,
            version_name,
            args.anatomy,
            args.stac_dir,
            args.gpu_mem_fraction,
            args.stac_overrides,
            args.dry_run
        )
        
        results.append((folder.name, success, message))
        
        if success:
            print(f"  ✅ {message}")
        else:
            print(f"  ❌ {message}")
        
        print()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if skipped:
        print(f"⏭️  Skipped {len(skipped)} folder(s) with existing outputs:")
        for name in skipped:
            print(f"   - {name}")
        print()
    
    if results:
        successful = sum(1 for _, success, _ in results if success)
        failed = len(results) - successful
        
        print(f"✅ Successful: {successful}/{len(results)}")
        if failed > 0:
            print(f"❌ Failed: {failed}/{len(results)}")
            print("\nFailed folders:")
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
            f.write(f"GPU memory fraction: {args.gpu_mem_fraction}\n")
            if args.stac_overrides:
                f.write(f"STAC overrides: {args.stac_overrides}\n")
            f.write(f"Force: {args.force}\n")
            f.write(f"\n")
            
            if skipped:
                f.write(f"Skipped folders ({len(skipped)}):\n")
                for name in skipped:
                    f.write(f"  - {name}\n")
                f.write(f"\n")
            
            if results:
                f.write(f"Processed folders ({len(results)}):\n")
                for name, success, message in results:
                    status = "✅" if success else "❌"
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
