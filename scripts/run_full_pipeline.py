#!/usr/bin/env python3
"""
Full Pipeline Orchestration Script

This script runs the complete 3D tracking data processing pipeline:
1. Preprocessing - Extract and align keypoints
2. STAC IK - Run inverse kinematics solver
3. Postprocessing - Compute velocities and egocentric positions
4. Combine - Merge all results into single file

Usage:
    # Run complete pipeline
    python scripts/run_full_pipeline.py --anatomy v1
    
    # Dry run to see what would happen
    python scripts/run_full_pipeline.py --anatomy v1 --dry-run
    
    # Run specific steps only
    python scripts/run_full_pipeline.py --anatomy v1 --steps preprocess,stac
    
    # Force reprocessing all steps
    python scripts/run_full_pipeline.py --anatomy v1 --force
    
    # Skip steps that are already complete
    python scripts/run_full_pipeline.py --anatomy v1 --skip-completed
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time

# Default data root — all datasets live under this directory
DATA_ROOT = Path('/data2/users/eabe/datasets/Johnson_lab')


class PipelineRunner:
    """Orchestrates the full pipeline execution."""

    def __init__(
        self,
        anatomy: str,
        dataset: str = "free_walking",
        base_dir: Path | None = None,
        paths_config: str = "workstation",
        force: bool = False,
        skip_completed: bool = False,
        dry_run: bool = False,
        stac_overrides: str = "",
        gpu_mem_fraction: float = 0.9
    ):
        self.anatomy = anatomy
        self.dataset = dataset
        self.base_dir = base_dir if base_dir is not None else DATA_ROOT / dataset
        self.paths_config = paths_config
        self.force = force
        self.skip_completed = skip_completed
        self.dry_run = dry_run
        self.stac_overrides = stac_overrides
        self.gpu_mem_fraction = gpu_mem_fraction

        # Script paths
        self.scripts_dir = Path(__file__).parent
        self.project_dir = self.scripts_dir.parent

        # Track timing and results
        self.step_results = {}
        self.step_times = {}
    
    def print_banner(self, text: str, char: str = "="):
        """Print a formatted banner."""
        width = 80
        print(f"\n{char * width}")
        print(text.center(width))
        print(f"{char * width}\n")
    
    def print_step_header(self, step_num: int, total_steps: int, step_name: str):
        """Print step header."""
        self.print_banner(f"STEP {step_num}/{total_steps}: {step_name}", char="-")
    
    def run_command(
        self,
        cmd: list[str],
        step_name: str,
        timeout: int = 3600
    ) -> tuple[bool, str]:
        """
        Run a command and return success status.
        
        Args:
            cmd: Command to run
            step_name: Name of the step for logging
            timeout: Timeout in seconds
        
        Returns:
            (success, message) tuple
        """
        if self.dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return True, "Dry run"
        
        print(f"Executing: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            self.step_times[step_name] = elapsed
            
            # Print output
            if result.stdout:
                print(result.stdout)
            
            if result.returncode == 0:
                return True, f"Success (took {elapsed:.1f}s)"
            else:
                error_msg = result.stderr[-1000:] if result.stderr else "Unknown error"
                return False, f"Failed with return code {result.returncode}\n{error_msg}"
        
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            self.step_times[step_name] = elapsed
            return False, f"Timeout after {elapsed:.1f}s"
        except Exception as e:
            elapsed = time.time() - start_time
            self.step_times[step_name] = elapsed
            return False, f"Exception: {str(e)}"
    
    def step_1_preprocess(self) -> bool:
        """Step 1: Preprocessing."""
        self.print_step_header(1, 4, "PREPROCESSING")
        
        cmd = [
            sys.executable,
            str(self.scripts_dir / "batch_process_predictions.py"),
            f"--dataset={self.dataset}",
            f"--anatomy={self.anatomy}",
            f"--paths={self.paths_config}",
            f"--base-dir={self.base_dir}",
        ]
        
        if self.force:
            cmd.append("--force")
        if self.dry_run:
            cmd.append("--dry-run")
        
        success, message = self.run_command(cmd, "preprocess", timeout=1800)
        self.step_results["preprocess"] = (success, message)
        
        if success:
            print(f"✅ Preprocessing complete: {message}")
        else:
            print(f"❌ Preprocessing failed: {message}")
        
        return success
    
    def step_2_stac(self) -> bool:
        """Step 2: STAC IK solver."""
        self.print_step_header(2, 4, "STAC IK SOLVER")
        
        cmd = [
            sys.executable,
            str(self.scripts_dir / "batch_run_stac.py"),
            f"--dataset={self.dataset}",
            f"--anatomy={self.anatomy}",
            f"--base-dir={self.base_dir}",
            f"--gpu-mem-fraction={self.gpu_mem_fraction}",
        ]
        
        if self.stac_overrides:
            cmd.append(f"--stac-overrides={self.stac_overrides}")
        if self.force:
            cmd.append("--force")
        if self.dry_run:
            cmd.append("--dry-run")
        
        success, message = self.run_command(cmd, "stac", timeout=7200)
        self.step_results["stac"] = (success, message)
        
        if success:
            print(f"✅ STAC IK complete: {message}")
        else:
            print(f"❌ STAC IK failed: {message}")
        
        return success
    
    def step_3_postprocess(self) -> bool:
        """Step 3: Postprocessing."""
        self.print_step_header(3, 4, "POSTPROCESSING")
        
        cmd = [
            sys.executable,
            str(self.scripts_dir / "batch_postprocess_predictions.py"),
            f"--dataset={self.dataset}",
            f"--anatomy={self.anatomy}",
            f"--paths={self.paths_config}",
            f"--base-dir={self.base_dir}",
        ]
        
        if self.force:
            cmd.append("--force")
        if self.dry_run:
            cmd.append("--dry-run")
        
        success, message = self.run_command(cmd, "postprocess", timeout=1800)
        self.step_results["postprocess"] = (success, message)
        
        if success:
            print(f"✅ Postprocessing complete: {message}")
        else:
            print(f"❌ Postprocessing failed: {message}")
        
        return success
    
    def step_4_combine(self) -> bool:
        """Step 4: Combine all results."""
        self.print_step_header(4, 4, "COMBINE DATA")
        
        cmd = [
            sys.executable,
            str(self.scripts_dir / "combine_data.py"),
            f"paths={self.paths_config}",
            f"dataset={self.dataset}",
            f"anatomy={self.anatomy}",
        ]
        
        if self.dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            print("Note: combine_data.py doesn't have dry-run mode")
            self.step_results["combine"] = (True, "Dry run")
            return True
        
        success, message = self.run_command(cmd, "combine", timeout=600)
        self.step_results["combine"] = (success, message)
        
        if success:
            print(f"✅ Combine complete: {message}")
        else:
            print(f"❌ Combine failed: {message}")
        
        return success
    
    def run(self, steps: list[str] = None) -> bool:
        """
        Run the pipeline.
        
        Args:
            steps: List of steps to run (default: all)
                   Options: 'preprocess', 'stac', 'postprocess', 'combine'
        
        Returns:
            True if all steps succeeded
        """
        if steps is None:
            steps = ['preprocess', 'stac', 'postprocess', 'combine']
        
        # Validate steps
        valid_steps = {'preprocess', 'stac', 'postprocess', 'combine'}
        invalid = set(steps) - valid_steps
        if invalid:
            print(f"❌ Invalid steps: {invalid}")
            print(f"   Valid steps: {valid_steps}")
            return False
        
        # Print configuration
        self.print_banner("3D TRACKING DATA PIPELINE")
        print(f"Configuration:")
        print(f"  Anatomy: {self.anatomy}")
        print(f"  Dataset: {self.dataset}")
        print(f"  Base directory: {self.base_dir}")
        print(f"  Paths config: {self.paths_config}")
        print(f"  Steps to run: {', '.join(steps)}")
        print(f"  Force reprocessing: {self.force}")
        print(f"  Skip completed: {self.skip_completed}")
        print(f"  Dry run: {self.dry_run}")
        if self.stac_overrides:
            print(f"  STAC overrides: {self.stac_overrides}")
        print()
        
        # Confirm if not dry run
        if not self.dry_run:
            response = input("Continue with pipeline execution? [y/N]: ")
            if response.lower() != 'y':
                print("Pipeline cancelled by user")
                return False
        
        start_time = time.time()
        
        # Run steps in order
        step_map = {
            'preprocess': self.step_1_preprocess,
            'stac': self.step_2_stac,
            'postprocess': self.step_3_postprocess,
            'combine': self.step_4_combine,
        }
        
        for step_name in steps:
            step_func = step_map[step_name]
            
            # Run step
            success = step_func()
            
            # Check if we should continue
            if not success and not self.dry_run:
                print(f"\n❌ Step '{step_name}' failed. Stopping pipeline.")
                self.print_summary()
                return False
        
        # All steps completed
        elapsed = time.time() - start_time
        self.print_banner("PIPELINE COMPLETE")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        self.print_summary()
        
        return all(success for success, _ in self.step_results.values())
    
    def print_summary(self):
        """Print summary of pipeline execution."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        if not self.step_results:
            print("No steps executed")
            return
        
        for step_name, (success, message) in self.step_results.items():
            status = "✅" if success else "❌"
            time_str = f" ({self.step_times.get(step_name, 0):.1f}s)" if step_name in self.step_times else ""
            print(f"{status} {step_name.upper()}{time_str}: {message}")
        
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete 3D tracking data processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline for free_walking with anatomy v1
  python scripts/run_full_pipeline.py --anatomy v1

  # Run for courtship dataset with v2_muscles anatomy
  python scripts/run_full_pipeline.py --dataset courtship --anatomy v2_muscles

  # Dry run
  python scripts/run_full_pipeline.py --anatomy v1 --dry-run

  # Run specific steps
  python scripts/run_full_pipeline.py --anatomy v1 --steps preprocess,stac

  # Force reprocess everything
  python scripts/run_full_pipeline.py --anatomy v1 --force

  # Custom base directory
  python scripts/run_full_pipeline.py --dataset courtship --anatomy v1 \\
      --base-dir /path/to/custom/data

  # Custom STAC settings
  python scripts/run_full_pipeline.py --anatomy v1 \\
      --stac-overrides "dataset.stac.n_fit_frames=401"

Pipeline Steps:
  1. preprocess  - Extract and align keypoints from raw data
  2. stac        - Run inverse kinematics solver
  3. postprocess - Compute velocities and egocentric positions
  4. combine     - Merge all results into single file
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
        help='Dataset name (free_walking, courtship, etc.)'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=None,
        help='Base directory containing Predictions_3D_* folders (default: DATA_ROOT/<dataset>)'
    )
    parser.add_argument(
        '--paths',
        type=str,
        default='workstation',
        help='Paths config (workstation, hyak, etc.)'
    )
    parser.add_argument(
        '--steps',
        type=str,
        default='preprocess,stac,postprocess,combine',
        help='Comma-separated list of steps to run (preprocess,stac,postprocess,combine)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing all steps even if outputs exist'
    )
    parser.add_argument(
        '--skip-completed',
        action='store_true',
        help='Skip steps that are already complete (not implemented yet)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be done without actually running'
    )
    parser.add_argument(
        '--gpu-mem-fraction',
        type=float,
        default=0.9,
        help='GPU memory fraction for STAC (0.0-1.0, default 0.9)'
    )
    parser.add_argument(
        '--stac-overrides',
        type=str,
        default='',
        help='Additional Hydra config overrides for STAC'
    )
    
    args = parser.parse_args()
    
    # Parse steps
    steps = [s.strip() for s in args.steps.split(',') if s.strip()]
    
    # Create pipeline runner
    runner = PipelineRunner(
        anatomy=args.anatomy,
        dataset=args.dataset,
        base_dir=args.base_dir,
        paths_config=args.paths,
        force=args.force,
        skip_completed=args.skip_completed,
        dry_run=args.dry_run,
        stac_overrides=args.stac_overrides,
        gpu_mem_fraction=args.gpu_mem_fraction
    )
    
    # Run pipeline
    success = runner.run(steps)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
