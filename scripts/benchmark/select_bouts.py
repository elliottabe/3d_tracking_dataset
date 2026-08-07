"""Rank candidate benchmark bouts from existing pipeline outputs (human picks).

Usage:
  python -m scripts.benchmark.select_bouts \
      --courtship-root /gscratch/portia/eabe/data/Johnson_lab/processed/courtship \
      --freerun-root   /gscratch/portia/eabe/data/Johnson_lab/free_running \
      > candidates.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def scan_bout_fly(fly_dir: Path) -> dict | None:
    qc_path = fly_dir / "qc.json"
    if not qc_path.is_file():
        return None
    qc = json.loads(qc_path.read_text())
    return {"n_frames": qc.get("n_frames"),
            "reproj_px": qc.get("per_camera_reproj_px", {}).get("median"),
            "soft_iou": qc.get("silhouette_iou", {}).get("soft_median")}


def proximity_median(bout_dir: Path) -> float | None:
    from scripts.benchmark.metrics import proximity_bl
    srcs = []
    for f in (0, 1):
        d = bout_dir / f"fly{f}"
        p = d / ("kp3d_filt.npz" if (d / "kp3d_filt.npz").exists() else "kp3d.npz")
        if not p.exists():
            return None
        srcs.append(np.load(p)["kp3d"])
    return float(np.nanmedian(proximity_bl(srcs[0], srcs[1])))


def _suggestion(reproj_px: float | None, proximity: float | None) -> str:
    """Assign curation hint based on metrics. Thresholds: proximity < 2.0, reproj > 12.0."""
    if proximity is not None and proximity < 2.0:
        return "close_interaction"
    if reproj_px is not None and reproj_px > 12.0:
        return "hard"
    return "clean"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--courtship-root", type=Path, default=None)
    ap.add_argument("--freerun-root", type=Path, default=None)
    ap.add_argument("--top", type=int, default=None,
                    help="Keep only top N bouts (sorted by reproj_px descending)")
    args = ap.parse_args(argv)

    rows = []

    if args.courtship_root:
        for bout_dir in sorted(args.courtship_root.glob(
                "Session*/*/pose/bouts/bout_*")):
            prox = proximity_median(bout_dir)
            for f in (0, 1):
                row = scan_bout_fly(bout_dir / f"fly{f}")
                if row:
                    run_key = f"{bout_dir.parents[3].name}/{bout_dir.parents[2].name}"
                    suggestion = _suggestion(row["reproj_px"], prox)
                    rows.append([run_key, bout_dir.name, f, "courtship",
                                row["n_frames"], row["reproj_px"],
                                row["soft_iou"], prox, suggestion])

    if args.freerun_root:
        for bout_dir in sorted(args.freerun_root.glob("*_bouts/bouts/bout_*")):
            row = scan_bout_fly(bout_dir / "fly0")
            if row:
                run_key = bout_dir.parents[1].name
                suggestion = _suggestion(row["reproj_px"], None)
                rows.append([run_key, bout_dir.name, 0, "free_running",
                            row["n_frames"], row["reproj_px"],
                            row["soft_iou"], "", suggestion])

    # Sort by reproj_px descending (None sorts last)
    rows.sort(key=lambda r: (r[5] is None, -r[5] if r[5] is not None else 0))

    # Truncate if --top specified
    if args.top is not None:
        rows = rows[:args.top]

    # Write CSV
    w = csv.writer(sys.stdout)
    w.writerow(["run_key", "bout", "fly", "assay", "n_frames", "reproj_px",
                "soft_iou", "proximity_median", "suggestion"])
    for row in rows:
        w.writerow(row)


if __name__ == "__main__":
    main()
