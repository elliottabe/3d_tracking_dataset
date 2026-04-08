#!/usr/bin/env python3
"""Batch pair per-fly preprocessed bouts into paired h5 files.

Walks every Predictions_3D_* folder under a dataset root, finds matching
``preprocessed_bout_<anatomy>_<dataset>_fly0.h5`` / ``_fly1.h5`` pairs, and
produces the paired outputs used by STAC:

  <preproc_dir>/preprocessed_bout_<a>_<d>_paired.h5        (full paired data)
  <preproc_dir>/preprocessed_bout_<a>_<d>_fly0_paired.h5   (STAC input fly0)
  <preproc_dir>/preprocessed_bout_<a>_<d>_fly1_paired.h5   (STAC input fly1)
  <preproc_dir>/preprocessed_bout_<a>_<d>_paired_index.json

Folders that only contain single-fly data (e.g. free_walking) are skipped
silently — this step is a no-op for single-fly datasets, so the pipeline can
call it unconditionally.

Usage:
    python scripts/batch_pair_bouts.py --dataset courtship --anatomy v1 \\
        --base-dir /path/to/Johnson_lab/courtship

    # or on a single Predictions_3D_* folder (slurm-per-folder mode)
    python scripts/batch_pair_bouts.py --dataset courtship --anatomy v1 \\
        --base-dir /path/.../Predictions_3D_34327248
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.merge_paired_bouts import merge_paired  # noqa: E402
from scripts.run_stac_paired import split_paired_to_per_fly  # noqa: E402
from utils.pair_validity import (  # noqa: E402
    PairValidityConfig,
    pair_validity_config_from_dict,
)
from utils import io_dict_to_hdf5 as ioh5  # noqa: E402


def _find_folders(base_dir: Path) -> list[Path]:
    if base_dir.is_dir() and base_dir.match("Predictions_3D_*"):
        return [base_dir]
    return sorted({p for p in base_dir.rglob("Predictions_3D_*") if p.is_dir()})


def _load_pv_cfg(path: Path) -> PairValidityConfig:
    try:
        d = ioh5.load(path, enable_jax=False)
        pv = d.get("info", {}).get("pair_validity", None)
        if pv is not None:
            return pair_validity_config_from_dict(pv)
    except Exception:
        pass
    return PairValidityConfig()


def process_folder(folder: Path, anatomy: str, dataset: str,
                   min_valid_frames: int, force: bool,
                   dry_run: bool) -> dict:
    preproc = folder / "preprocessing"
    result = {"folder": str(folder), "status": "skipped", "message": ""}
    if not preproc.exists():
        result["message"] = "no preprocessing/ dir"
        return result

    stem = f"preprocessed_bout_{anatomy}_{dataset}"
    fly0 = preproc / f"{stem}_fly0.h5"
    fly1 = preproc / f"{stem}_fly1.h5"
    if not fly0.exists() or not fly1.exists():
        result["message"] = "no fly0/fly1 pair (single-fly dataset or not preprocessed yet)"
        return result

    paired = preproc / f"{stem}_paired.h5"
    out_fly0 = preproc / f"{stem}_fly0_paired.h5"
    out_fly1 = preproc / f"{stem}_fly1_paired.h5"
    index_path = preproc / f"{stem}_paired_index.json"

    print(f"\n[pair] {folder.name}")
    print(f"  fly0:   {fly0.name}")
    print(f"  fly1:   {fly1.name}")
    print(f"  paired: {paired.name}")

    if dry_run:
        result["status"] = "dry-run"
        result["message"] = f"would pair {fly0.name} + {fly1.name}"
        return result

    try:
        pv_cfg = _load_pv_cfg(fly0)
        merge_paired(fly0, fly1, paired, pv_cfg, force=force)
        split_paired_to_per_fly(
            paired, out_fly0, out_fly1, index_path,
            min_valid_frames=min_valid_frames, force=force,
        )
        result["status"] = "success"
        result["message"] = str(paired.name)
    except Exception as e:
        import traceback
        traceback.print_exc()
        result["status"] = "error"
        result["message"] = str(e)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--anatomy", default="v1")
    ap.add_argument("--base-dir", type=Path, required=True)
    ap.add_argument("--min-valid-frames", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.base_dir.exists():
        print(f"Error: base dir not found: {args.base_dir}", file=sys.stderr)
        sys.exit(1)

    folders = _find_folders(args.base_dir)
    if not folders:
        print(f"No Predictions_3D_* folders under {args.base_dir}")
        # Not an error — single-fly datasets or empty trees are fine.
        sys.exit(0)

    print(f"Found {len(folders)} prediction folder(s)")
    results = [process_folder(f, args.anatomy, args.dataset,
                              args.min_valid_frames, args.force, args.dry_run)
               for f in folders]

    print("\n" + "=" * 60)
    print("PAIRING SUMMARY")
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # Only treat real errors as failure; "skipped" (single-fly) is fine.
    if counts.get("error", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
