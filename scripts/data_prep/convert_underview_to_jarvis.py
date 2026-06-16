#!/usr/bin/env python3
"""Convert a Johnson-lab "underview" muscle-imaging CSV into JARVIS data3D.csv.

The underview rig exports a single-header CSV with 6 columns per keypoint
(``<name>_x,_y,_z,_error,_ncams,_score``) plus a trailing per-frame body frame
(``M_00..M_22``, ``center_0..center_2``) and ``fnum``. Its marker names
(``R1A``, ``thorax-tether``, ...) do not match the fruit-fly body model.

The rest of the pipeline (``scripts/preprocess_keypoints_for_ik.py``) reads the
JARVIS ``data3D.csv`` format: a 2-row header where level 0 is the node name
(repeated for each of its columns) and level 1 is ``x,y,z,confidence``, with
node names matching ``data/fly50.json`` / the model ``KP_NAMES``.

This script renames the underview markers onto model node names (see
``UNDERVIEW_TO_MODEL`` below), maps ``score -> confidence``, drops the unused
markers/columns, and writes:

  * ``data3D.csv``                      — JARVIS-format keypoints
  * ``muscle_imaging_bouts_summary.csv`` — one bout spanning every frame

Leg points A-E run proximal->distal = coxa joint / femur joint / tibia / tarsus
/ claw, which map onto the model's ``ThxCx, Tro, FeTi, TiTa, TaTip``. The model
has no coxa keypoint for the middle/hind legs, so ``R2A``/``R3A`` are dropped.

Usage:
    python scripts/data_prep/convert_underview_to_jarvis.py \
        /path/to/2026-05-18--16-06-57_fly1_underview.csv
    python scripts/data_prep/convert_underview_to_jarvis.py INPUT.csv \
        --output-dir /some/dir --fly-id fly1
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Underview marker name -> model/skeleton node name (data/fly50.json order).
# Markers not listed (head-tip, R2A, R3A) are intentionally dropped.
UNDERVIEW_TO_MODEL = {
    # Body markers
    "head-top": "Antenna_Base",     # orientation front anchor
    "thorax-tether": "Scutellum",   # root + trunk-scale marker
    "abdomen-tip": "Abd_tip",       # trunk + orientation rear
    # Left front leg (T1L): A-E = coxa/femur/tibia/tarsus/claw
    "L1A": "T1L_ThxCx", "L1B": "T1L_Tro", "L1C": "T1L_FeTi",
    "L1D": "T1L_TiTa", "L1E": "T1L_TaTip",
    # Right front leg (T1R)
    "R1A": "T1R_ThxCx", "R1B": "T1R_Tro", "R1C": "T1R_FeTi",
    "R1D": "T1R_TiTa", "R1E": "T1R_TaTip",
    # Right middle leg (T2R) — model has no coxa kp, so R2A (coxa) is dropped
    "R2B": "T2R_Tro", "R2C": "T2R_FeTi", "R2D": "T2R_TiTa", "R2E": "T2R_TaTip",
    # Right hind leg (T3R) — R3A (coxa) dropped
    "R3B": "T3R_Tro", "R3C": "T3R_FeTi", "R3D": "T3R_TiTa", "R3E": "T3R_TaTip",
}

# Order to emit nodes in the output (data/fly50.json node order, present subset).
FLY50_ORDER = [
    "Antenna_Base", "EyeL", "EyeR", "Scutellum", "Abd_A4", "Abd_tip",
    "WingL_base", "WingL_V12", "WingL_V13",
    "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip",
    "T2L_Tro", "T2L_FeTi", "T2L_TiTa", "T2L_TaT1", "T2L_TaT3", "T2L_TaTip",
    "T3L_Tro", "T3L_FeTi", "T3L_TiTa", "T3L_TaT1", "T3L_TaT3", "T3L_TaTip",
    "WingR_base", "WingR_V12", "WingR_V13",
    "T1R_ThxCx", "T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip",
    "T2R_Tro", "T2R_FeTi", "T2R_TiTa", "T2R_TaT1", "T2R_TaT3", "T2R_TaTip",
    "T3R_Tro", "T3R_FeTi", "T3R_TiTa", "T3R_TaT1", "T3R_TaT3", "T3R_TaTip",
]


def parse_fly_id(filename: str) -> str:
    """Pull a ``flyN`` token from the recording filename (default ``fly0``)."""
    m = re.search(r"(fly\d+)", filename)
    return m.group(1) if m else "fly0"


def check_world_frame(df: pd.DataFrame) -> None:
    """Assert the per-frame body frame (M, center) is identity/zero.

    When M/center are non-trivial the exported x,y,z live in a per-frame
    body frame and would need the inverse transform applied before they are in
    a consistent world frame. This converter only handles the world-frame case.
    """
    m_cols = [f"M_{i}{j}" for i in range(3) for j in range(3)]
    have_m = all(c in df.columns for c in m_cols)
    have_c = all(f"center_{i}" in df.columns for i in range(3))
    if not (have_m and have_c):
        print("  [frame] no M/center columns found — assuming world-frame x,y,z")
        return
    M = df[m_cols].to_numpy().reshape(-1, 3, 3)
    center = df[[f"center_{i}" for i in range(3)]].to_numpy()
    eye = np.broadcast_to(np.eye(3), M.shape)
    m_ok = np.allclose(M, eye, atol=1e-6)
    c_ok = np.allclose(center, 0.0, atol=1e-6)
    if not (m_ok and c_ok):
        sys.exit(
            "ERROR: M is not identity or center is not zero in this file.\n"
            "  The x,y,z are in a per-frame body frame; this converter only\n"
            "  supports world-frame exports. Apply the inverse (M, center)\n"
            "  transform upstream, or extend this script, before converting."
        )
    print("  [frame] M == I and center == 0 for all frames — world-frame x,y,z")


def convert(input_csv: Path, output_dir: Path, fly_id: str) -> None:
    print(f"Loading underview CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    n_frames = len(df)
    print(f"  {n_frames} frames, {df.shape[1]} columns")

    check_world_frame(df)

    # Underview marker names present in the file (have an _x column).
    csv_markers = [c[:-2] for c in df.columns if c.endswith("_x")]
    mapped = {u: UNDERVIEW_TO_MODEL[u] for u in csv_markers if u in UNDERVIEW_TO_MODEL}
    dropped = [u for u in csv_markers if u not in UNDERVIEW_TO_MODEL]
    print(f"  Mapping {len(mapped)} markers -> model nodes; dropping {dropped}")

    # Emit nodes in fly50 order, restricted to the mapped subset.
    model_to_underview = {v: k for k, v in mapped.items()}
    nodes_out = [n for n in FLY50_ORDER if n in model_to_underview]
    missing = [n for n in model_to_underview if n not in nodes_out]
    if missing:
        # Mapped to a name not in FLY50_ORDER — would silently vanish; fail loud.
        sys.exit(f"ERROR: mapped node(s) not in fly50 order: {missing}")

    # Build the JARVIS frame: MultiIndex columns (node, x|y|z|confidence).
    data = {}
    for node in nodes_out:
        u = model_to_underview[node]
        data[(node, "x")] = df[f"{u}_x"].to_numpy()
        data[(node, "y")] = df[f"{u}_y"].to_numpy()
        data[(node, "z")] = df[f"{u}_z"].to_numpy()
        data[(node, "confidence")] = df[f"{u}_score"].to_numpy()
    out = pd.DataFrame(data)
    out.columns = pd.MultiIndex.from_tuples(out.columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    data3d_path = output_dir / "data3D.csv"
    out.to_csv(data3d_path, index=False)
    print(f"✓ Wrote {data3d_path}  ({len(nodes_out)} nodes x 4 cols, {n_frames} rows)")

    # One-bout summary so batch preprocessing runs (load_bouts_from_csv maps
    # bout_idx -> bout_idx-1, so bout_idx=1 yields bout_000).
    bouts = pd.DataFrame(
        [{"bout_idx": 1, "start_frame": 0, "end_frame": n_frames - 1, "fly_id": fly_id}]
    )
    bouts_path = output_dir / "muscle_imaging_bouts_summary.csv"
    bouts.to_csv(bouts_path, index=False)
    print(f"✓ Wrote {bouts_path}  (1 bout, frames 0-{n_frames - 1}, fly_id={fly_id})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_csv", type=Path, help="underview CSV to convert")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="output dir (default: same dir as input)")
    ap.add_argument("--fly-id", type=str, default=None,
                    help="fly id for the bouts summary (default: parsed from filename)")
    args = ap.parse_args()

    if not args.input_csv.exists():
        sys.exit(f"ERROR: input not found: {args.input_csv}")
    output_dir = args.output_dir or args.input_csv.parent
    fly_id = args.fly_id or parse_fly_id(args.input_csv.name)
    convert(args.input_csv, output_dir, fly_id)


if __name__ == "__main__":
    main()
