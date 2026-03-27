"""Fix xpos/xquat ordering in raw STAC output files (Fruitfly_ik_*.h5).

The STAC solver's _package_data() used order="F" (Fortran/column-major) when
reshaping xpos and xquat from (n_bouts, max_T, ...) to flat arrays, producing
timestep-major ordering. All other arrays (qpos, marker_sites, kp_data) use
C-order (bout-major). This script reverses the F-order reshape so all arrays
are consistently bout-major.

Usage:
    python fix_stac_xpos_ordering.py /path/to/base_dir [--dry-run] [--no-backup]

Example:
    python fix_stac_xpos_ordering.py /data2/users/eabe/datasets/Johnson_lab/free_walking
"""

import sys
import argparse
import re
import shutil
from pathlib import Path

import h5py
import numpy as np


def find_preprocessed_file(stac_path: Path) -> Path | None:
    """Find the preprocessed_bout file corresponding to a Fruitfly_ik file."""
    parent = stac_path.parent
    # Extract suffix: Fruitfly_ik_{suffix}.h5
    match = re.match(r"Fruitfly_ik_(.+)\.h5", stac_path.name)
    if not match:
        return None
    suffix = match.group(1)

    # Try common naming patterns
    candidates = [
        parent / f"preprocessed_bout_{suffix}.h5",
        parent / f"stationary_preprocessed_bout_{suffix}.h5",
    ]
    # Also try without trailing qualifiers (e.g., _stationary_free -> just the base)
    parts = suffix.rsplit("_", 1)
    if len(parts) > 1:
        candidates.append(parent / f"preprocessed_bout_{parts[0]}.h5")
        candidates.append(parent / f"stationary_preprocessed_bout_{parts[0]}.h5")

    for c in candidates:
        if c.exists():
            return c
    return None


def count_bouts_in_preprocessed(preproc_path: Path) -> int:
    """Count the number of bouts in a preprocessed file."""
    with h5py.File(preproc_path, "r") as f:
        bout_keys = [k for k in f.keys() if k.startswith("bout_")]
        return len(bout_keys)


def fix_ordering(arr: np.ndarray, n_bouts: int, max_t: int) -> np.ndarray:
    """Convert an array from F-order (timestep-major) to C-order (bout-major).

    The F-order reshape of (n_bouts, max_T, ...) produces indices like:
        [bout0_t0, bout1_t0, ..., boutN_t0, bout0_t1, bout1_t1, ...]
    We need to undo this to get:
        [bout0_t0, bout0_t1, ..., bout0_tM, bout1_t0, bout1_t1, ...]
    """
    extra_dims = arr.shape[1:]
    # Undo F-order flatten: reconstruct (max_T, n_bouts, ...)
    arr_2d = arr.reshape(max_t, n_bouts, *extra_dims)
    # Transpose to (n_bouts, max_T, ...)
    perm = (1, 0) + tuple(range(2, arr_2d.ndim))
    arr_2d = arr_2d.transpose(perm)
    # Flatten back to (n_bouts * max_T, ...) in C-order
    return arr_2d.reshape(-1, *extra_dims)


def verify_fix(stac_path: Path, qpos: np.ndarray, xpos: np.ndarray, model_path: str) -> bool:
    """Verify fixed xpos matches FK from qpos for several frames."""
    try:
        import mujoco
    except ImportError:
        print("    [WARN] mujoco not available, skipping FK verification")
        return True

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Skip verification if model doesn't match the data dimensions
    if model.nq != qpos.shape[1] or model.nbody != xpos.shape[1]:
        print(f"    [SKIP] Model mismatch (nq={model.nq} vs {qpos.shape[1]}, "
              f"nbody={model.nbody} vs {xpos.shape[1]}), skipping FK verification")
        return True

    n_checks = min(5, qpos.shape[0])
    all_ok = True
    for i in range(n_checks):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        if not np.allclose(xpos[i, 1, :], data.xpos[1, :], atol=1e-3):
            print(f"    [FAIL] Frame {i}: xpos thorax {xpos[i, 1, :]} != FK {data.xpos[1, :]}")
            all_ok = False
    if all_ok:
        print(f"    [OK] FK verification passed for {n_checks} frames")
    return all_ok


def process_file(
    stac_path: Path,
    preproc_path: Path,
    model_path: str | None = None,
    dry_run: bool = False,
    backup: bool = True,
) -> bool:
    """Fix xpos/xquat ordering in a single STAC file."""
    print(f"\n{'='*60}")
    print(f"Processing: {stac_path.name}")
    print(f"  Preprocessed: {preproc_path.name}")

    n_bouts = count_bouts_in_preprocessed(preproc_path)
    print(f"  n_bouts: {n_bouts}")

    if n_bouts <= 1:
        print("  Skipping: single bout (F-order = C-order when n_bouts=1)")
        return True

    with h5py.File(stac_path, "r") as f:
        total_frames = f["xpos"].shape[0]
        xpos_shape = f["xpos"].shape
        xquat_shape = f["xquat"].shape

    if total_frames % n_bouts != 0:
        print(f"  [ERROR] total_frames={total_frames} not divisible by n_bouts={n_bouts}")
        return False

    max_t = total_frames // n_bouts
    print(f"  max_T: {max_t}, total_frames: {total_frames}")
    print(f"  xpos: {xpos_shape}, xquat: {xquat_shape}")

    if dry_run:
        print("  [DRY RUN] Would fix xpos and xquat ordering")
        return True

    # Read, fix, write
    if backup:
        backup_path = stac_path.with_suffix(".h5.bak")
        if not backup_path.exists():
            shutil.copy2(stac_path, backup_path)
            print(f"  Backed up to: {backup_path.name}")

    with h5py.File(stac_path, "r") as f:
        xpos = np.array(f["xpos"])
        xquat = np.array(f["xquat"])
        qpos = np.array(f["qpos"])

    xpos_fixed = fix_ordering(xpos, n_bouts, max_t)
    xquat_fixed = fix_ordering(xquat, n_bouts, max_t)

    # Quick sanity: frame 0 should be unchanged (it's bout 0, time 0 in both orderings)
    assert np.allclose(xpos[0], xpos_fixed[0]), "Frame 0 should be unchanged!"

    # Verify with FK if model path provided
    if model_path:
        if not verify_fix(stac_path, qpos, xpos_fixed, model_path):
            print("  [ERROR] FK verification failed, not saving")
            return False

    # Write fixed arrays back
    with h5py.File(stac_path, "r+") as f:
        f["xpos"][...] = xpos_fixed
        f["xquat"][...] = xquat_fixed

    print("  [DONE] Fixed xpos and xquat in-place")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fix xpos/xquat ordering in raw STAC output files")
    parser.add_argument("base_dir", type=Path, help="Base directory containing Predictions_3D_* folders")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying files")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating .bak backup files")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to MuJoCo XML for FK verification (optional)")
    parser.add_argument("--pattern", type=str, default="Fruitfly_ik_*.h5",
                        help="Glob pattern for STAC files (default: Fruitfly_ik_*.h5)")
    args = parser.parse_args()

    if not args.base_dir.exists():
        print(f"Error: {args.base_dir} does not exist")
        sys.exit(1)

    stac_files = sorted(args.base_dir.rglob(args.pattern))
    print(f"Found {len(stac_files)} STAC files under {args.base_dir}")

    success = 0
    skipped = 0
    failed = 0

    for sf in stac_files:
        preproc = find_preprocessed_file(sf)
        if preproc is None:
            print(f"\n[SKIP] No preprocessed file found for {sf.name} in {sf.parent}")
            skipped += 1
            continue

        ok = process_file(
            sf,
            preproc,
            model_path=args.model,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Summary: {success} fixed, {skipped} skipped, {failed} failed")
    if args.dry_run:
        print("(dry run - no files were modified)")


if __name__ == "__main__":
    main()
