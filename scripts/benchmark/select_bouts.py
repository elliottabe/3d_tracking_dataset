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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--courtship-root", type=Path, default=None)
    ap.add_argument("--freerun-root", type=Path, default=None)
    args = ap.parse_args(argv)
    w = csv.writer(sys.stdout)
    w.writerow(["assay", "run", "bout", "fly", "n_frames", "reproj_px",
                "soft_iou", "proximity_median"])
    if args.courtship_root:
        for bout_dir in sorted(args.courtship_root.glob(
                "Session*/*/pose/bouts/bout_*")):
            prox = proximity_median(bout_dir)
            for f in (0, 1):
                row = scan_bout_fly(bout_dir / f"fly{f}")
                if row:
                    w.writerow(["courtship",
                                f"{bout_dir.parents[3].name}/{bout_dir.parents[2].name}",
                                bout_dir.name, f, row["n_frames"],
                                row["reproj_px"], row["soft_iou"], prox])
    if args.freerun_root:
        for bout_dir in sorted(args.freerun_root.glob("*_bouts/bouts/bout_*")):
            row = scan_bout_fly(bout_dir / "fly0")
            if row:
                w.writerow(["free_running", bout_dir.parents[1].name,
                            bout_dir.name, 0, row["n_frames"],
                            row["reproj_px"], row["soft_iou"], ""])


if __name__ == "__main__":
    main()
