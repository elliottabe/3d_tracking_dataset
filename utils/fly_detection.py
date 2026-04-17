"""
Fly detection utility for multi-fly datasets.

Detects whether a Predictions_3D folder contains single-fly or dual-fly data
based on the presence of fly-suffixed CSV files.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd


_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint32)


def build_compact_frame_map(tracking_info_path: Path,
                            n_csv_rows: int) -> Optional[Dict[int, int]]:
    """If the CSV is compact (bouts-mode JARVIS output), build a mapping from
    original video frame number -> compact CSV row index.

    Returns None if the CSV is sparse (legacy full-video output).
    Detection: if tracking_info.json exists, has a 'bouts' array, and the sum
    of bout lengths matches n_csv_rows, the CSV is compact.
    """
    if not tracking_info_path.exists():
        return None
    import json
    with open(tracking_info_path) as f:
        info = json.load(f)
    bouts = info.get('bouts')
    if not bouts:
        return None
    compact_total = sum(b['end'] - b['start'] + 1 for b in bouts)
    if compact_total != n_csv_rows:
        return None
    frame_map: Dict[int, int] = {}
    row = 0
    for b in bouts:
        for frame in range(b['start'], b['end'] + 1):
            frame_map[frame] = row
            row += 1
    return frame_map


def build_unified_bouts_csv(folder: Path, dataset: str,
                            force: bool = False) -> Optional[Path]:
    """
    Merge per-fly bouts summary CSVs into a single unified bouts list so that
    both flies are preprocessed over the same set of frame ranges. This is a
    prerequisite for the identity-relink stage which needs both flies' raw
    keypoints over the same time windows.

    Reads ``<dataset>_bouts_fly0_summary.csv`` and ``<dataset>_bouts_fly1_summary.csv``
    and writes ``<dataset>_bouts_unified_summary.csv`` to the same folder.

    The merge:
      - Strips ``_fly0`` / ``_fly1`` suffixes from the ``fly_id`` column so the
        unified csv stores session-level identifiers only. The processing
        script appends the actual fly suffix at load time.
      - Drops exact duplicate windows (same start_frame, end_frame, fly_id).
      - Sorts by (fly_id, start_frame).
      - Renumbers ``bout_idx`` sequentially starting at 1.

    Returns the path to the unified csv, or None if the per-fly summaries do
    not exist.
    """
    fly0_csv = folder / f"{dataset}_bouts_fly0_summary.csv"
    fly1_csv = folder / f"{dataset}_bouts_fly1_summary.csv"
    if not (fly0_csv.exists() and fly1_csv.exists()):
        return None

    out_csv = folder / f"{dataset}_bouts_unified_summary.csv"
    if out_csv.exists() and not force:
        return out_csv

    df0 = pd.read_csv(fly0_csv)
    df1 = pd.read_csv(fly1_csv)
    df0["source_fly"] = "fly0"
    df1["source_fly"] = "fly1"
    df = pd.concat([df0, df1], ignore_index=True)

    # Strip _fly0 / _fly1 suffix from fly_id (session-level only)
    if "fly_id" in df.columns:
        df["fly_id"] = df["fly_id"].astype(str).str.replace(r"_fly\d+$", "",
                                                              regex=True)

    df = df.drop_duplicates(
        subset=["fly_id", "source_fly", "start_frame", "end_frame"]
    )
    df = df.sort_values(["fly_id", "source_fly", "start_frame"]).reset_index(drop=True)
    df["bout_idx"] = range(1, len(df) + 1)

    df.to_csv(out_csv, index=False)
    return out_csv


def _decide_sex_swap(npz_path: Path,
                     min_ratio: float = 1.05) -> Tuple[bool, float, float]:
    """
    Decide whether to swap fly0<->fly1 so fly0 becomes the male (smaller).

    Reads ``sam3_masks.npz`` (keys ``packed`` shape [A, C, F, H, W_pack]
    uint8, bit-packed masks) and computes per-fly mask-pixel area on an
    every-10th-frame subsample via a popcount lookup. Female D. melanogaster
    are ~15-20% larger; swap if fly0 is larger by at least ``min_ratio``.

    Returns (swap, area0, area1). If the NPZ is missing or both areas are
    within ``min_ratio`` of each other, returns (False, …).
    """
    if not npz_path.exists():
        return False, 0.0, 0.0
    with np.load(npz_path) as d:
        if 'packed' not in d:
            return False, 0.0, 0.0
        packed = d['packed']  # (A, C, F, H, W_pack)
    if packed.shape[0] < 2:
        return False, 0.0, 0.0
    F = packed.shape[2]
    step = max(1, F // 50)
    subset = packed[:, :, ::step]  # (A, C, F_sub, H, W_pack)
    # popcount per byte then sum
    pop = _POPCOUNT[subset]
    area0 = float(pop[0].sum())
    area1 = float(pop[1].sum())
    denom = min(area0, area1)
    if denom <= 0:
        return False, area0, area1
    ratio = max(area0, area1) / denom
    swap = (area0 > area1) and (ratio >= min_ratio)
    return swap, area0, area1


def _find_unified_bouts_csv(folder: Path, dataset: str) -> Optional[Path]:
    """Walk up parent dirs to find <dataset>_bouts_unified_summary.csv."""
    target = f"{dataset}_bouts_unified_summary.csv"
    for candidate in [folder, *folder.parents]:
        p = candidate / target
        if p.exists():
            return p
    return None


def _synthesize_unified_from_bouts(folder: Path,
                                   bout_dirs: List[Path]) -> pd.DataFrame:
    """Build a unified bouts DataFrame from the per-bout CSV frame columns.

    Used for new-format folders that have bout_*/fly{0,1}.csv but no
    pre-existing <dataset>_bouts_unified_summary.csv. The fly_id is derived
    from folder path components (``<SessionX>/<timestamp>``), matching the
    convention used by build_unified_bouts_csv. source_fly is always 'both'
    since every bout dir carries both fly0 and fly1 CSVs.
    """
    fly_id = f"{folder.parent.parent.name}/{folder.parent.name}"
    rows = []
    for i, bout_dir in enumerate(bout_dirs, start=1):
        probe = pd.read_csv(bout_dir / "fly0.csv", header=[0, 1])
        frame_col = [c for c in probe.columns if c[0] == 'frame']
        if not frame_col:
            raise ValueError(
                f"No 'frame' column found in {bout_dir}/fly0.csv — cannot "
                f"synthesize bouts summary for new-format folder {folder}."
            )
        frames = probe[frame_col[0]].values
        rows.append({
            'fly_id': fly_id,
            'bout_idx': i,
            'start_frame': int(frames.min()),
            'end_frame': int(frames.max()),
            'source_fly': 'both',
        })
    return pd.DataFrame(rows,
                        columns=['fly_id', 'bout_idx', 'start_frame',
                                 'end_frame', 'source_fly'])


def _concat_per_bout_csvs(bout_csvs: List[Path]) -> pd.DataFrame:
    """
    Read bout_*/flyX.csv (per-bout compact CSVs with 2-row MultiIndex header)
    and concatenate in bout order.

    - Drops the `frame` column (pipeline uses tracking_info.json for bout
      windows, keyed on start/end in the per-fly summary).
    - Renames the row-1 label `conf` -> `confidence` so
      `load_confidence_from_csv` picks up `<kp>_confidence` after MultiIndex
      flatten.
    """
    frames = []
    for p in bout_csvs:
        df = pd.read_csv(p, header=[0, 1])
        # Drop frame column (level-0 label is 'frame', level-1 also 'frame')
        drop_cols = [c for c in df.columns if c[0] == 'frame']
        df = df.drop(columns=drop_cols)
        # Rename level-1 'conf' -> 'confidence'
        df.columns = pd.MultiIndex.from_tuples(
            [(lvl0, 'confidence' if lvl1 == 'conf' else lvl1)
             for lvl0, lvl1 in df.columns]
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate_per_bout_predictions(folder: Path,
                                    dataset: str,
                                    force: bool = False) -> bool:
    """
    Materialize old-style aggregate predictions from a per-bout layout.

    Turns::

        folder/bout_<N>/fly{0,1}.csv  +  sam3_masks.npz

    into the pipeline-expected flat layout::

        folder/data3D_fly{0,1}.csv
        folder/tracking_info.json
        folder/<dataset>_bouts_fly{0,1}_summary.csv
        folder/<dataset>_bouts_unified_summary.csv

    Also applies a mask-area sex-swap per bout so fly0 is the male (smaller
    mask). Per-bout decisions are recorded under ``sex_swaps`` in
    tracking_info.json for traceability.

    Idempotent: returns False without writing if all outputs exist and are
    newer than every per-bout CSV (unless ``force=True``).
    """
    bout_dirs = sorted(
        d for d in folder.iterdir()
        if d.is_dir() and d.name.startswith('bout_')
        and (d / 'fly0.csv').exists() and (d / 'fly1.csv').exists()
    )
    if not bout_dirs:
        return False

    data3d_paths = [folder / f"data3D_fly{n}.csv" for n in (0, 1)]
    summary_paths = [folder / f"{dataset}_bouts_fly{n}_summary.csv"
                     for n in (0, 1)]
    unified_out = folder / f"{dataset}_bouts_unified_summary.csv"
    tracking_info = folder / "tracking_info.json"
    outputs = [*data3d_paths, *summary_paths, unified_out, tracking_info]

    if not force and all(p.exists() for p in outputs):
        newest_input = max(
            (d / f"fly{n}.csv").stat().st_mtime
            for d in bout_dirs for n in (0, 1)
        )
        oldest_output = min(p.stat().st_mtime for p in outputs)
        if oldest_output >= newest_input:
            return False

    unified_src = _find_unified_bouts_csv(folder, dataset)
    if unified_src is None:
        unified_df = _synthesize_unified_from_bouts(folder, bout_dirs)
    else:
        unified_df = pd.read_csv(unified_src)

    bouts_info: List[Dict[str, int]] = []
    sex_swaps: List[bool] = []
    # Concatenated DataFrames per OUTPUT fly index after sex-swap applied.
    fly_frames: List[List[pd.DataFrame]] = [[], []]

    for bout_dir in bout_dirs:
        swap, a0, a1 = _decide_sex_swap(bout_dir / "sam3_masks.npz")
        sex_swaps.append(swap)
        # Peek at the fly0 CSV's `frame` column for bout range
        probe = pd.read_csv(bout_dir / "fly0.csv", header=[0, 1])
        frame_col = [c for c in probe.columns if c[0] == 'frame']
        frames = probe[frame_col[0]].values if frame_col else None
        start = int(frames.min()) if frames is not None else -1
        end = int(frames.max()) if frames is not None else -1
        bouts_info.append({'start': start, 'end': end,
                           'area0': a0, 'area1': a1, 'swap': swap})

        out_fly0_src = bout_dir / ("fly1.csv" if swap else "fly0.csv")
        out_fly1_src = bout_dir / ("fly0.csv" if swap else "fly1.csv")
        fly_frames[0].append(_concat_per_bout_csvs([out_fly0_src]))
        fly_frames[1].append(_concat_per_bout_csvs([out_fly1_src]))
        print(f"  [aggregate] {bout_dir.name}: frames=[{start},{end}] "
              f"area0={a0:.0f} area1={a1:.0f} swap={swap}")

    for n in (0, 1):
        combined = pd.concat(fly_frames[n], ignore_index=True)
        combined.to_csv(data3d_paths[n], index=False)

    # tracking_info.json — compact-frame-map schema
    tracking_info.write_text(json.dumps({
        'bouts': [{'start': b['start'], 'end': b['end']} for b in bouts_info],
        'sex_swaps': [b['swap'] for b in bouts_info],
        'areas': [{'a0': b['area0'], 'a1': b['area1']} for b in bouts_info],
    }, indent=2))

    # Filter unified summary to materialized bouts; renumber bout_idx.
    materialized_ranges = {(b['start'], b['end']) for b in bouts_info}
    unified_filtered = unified_df[
        unified_df.apply(
            lambda r: (int(r['start_frame']), int(r['end_frame']))
            in materialized_ranges,
            axis=1,
        )
    ].copy()
    unified_filtered = unified_filtered.sort_values(
        ['fly_id', 'source_fly', 'start_frame']).reset_index(drop=True)
    unified_filtered['bout_idx'] = range(1, len(unified_filtered) + 1)
    unified_filtered.to_csv(unified_out, index=False)

    # Per-fly bouts summaries — filter by source_fly, renumber bout_idx.
    for n in (0, 1):
        src_tags = {'both', f'fly{n}'}
        per_fly = unified_filtered[
            unified_filtered['source_fly'].isin(src_tags)].copy()
        per_fly = per_fly.drop(columns=['source_fly'])
        per_fly = per_fly.sort_values(['fly_id', 'start_frame']).reset_index(drop=True)
        per_fly['bout_idx'] = range(1, len(per_fly) + 1)
        per_fly.to_csv(summary_paths[n], index=False)

    return True


def detect_flies(folder: Path, dataset: str) -> List[Dict]:
    """
    Detect single-fly vs dual-fly data layout in a Predictions_3D folder.

    Checks for fly-suffixed CSV files (e.g., data3D_fly0.csv). If found,
    returns one entry per fly. Otherwise returns a single entry with no suffix,
    matching the original single-fly pipeline behavior.

    Args:
        folder: Path to a Predictions_3D_* folder.
        dataset: Dataset name (e.g., 'courtship', 'free_walking').

    Returns:
        List of dicts, each with keys:
            - fly_id:    str or None ('fly0', 'fly1', or None for single-fly)
            - suffix:    str to append to output filenames ('_fly0', '_fly1', or '')
            - csv:       CSV filename for this fly's keypoint data
            - bouts_csv: Bouts summary CSV filename for this fly
    """
    flies = []

    # Per-bout layout (bout_*/fly{0,1}.csv)? Materialize aggregates in place
    # and fall through to the standard dual-fly branch below.
    if (not sorted(folder.glob("data3D_fly*.csv"))
            and sorted(folder.glob("bout_*/fly*.csv"))):
        aggregate_per_bout_predictions(folder, dataset)

    # Check for dual-fly layout: data3D_fly0.csv, data3D_fly1.csv
    fly_csvs = sorted(folder.glob("data3D_fly*.csv"))
    if fly_csvs:
        # For dual-fly we always build a unified bouts list so identity-relink
        # has both flies' keypoints over the same frame windows.
        unified_csv = build_unified_bouts_csv(folder, dataset)
        for csv_path in fly_csvs:
            stem = csv_path.stem  # data3D_fly0
            fly_id = stem.replace("data3D_", "")  # fly0

            if unified_csv is not None:
                bouts_csv_name = unified_csv.name
            else:
                # Fallback: per-fly bouts (older single-fly behaviour)
                bouts_csv_name = f"{dataset}_bouts_{fly_id}_summary.csv"
                bouts_csv_path = folder / bouts_csv_name
                if not bouts_csv_path.exists():
                    for f in folder.iterdir():
                        if f.name.lower() == bouts_csv_name.lower():
                            bouts_csv_name = f.name
                            break

            flies.append({
                'fly_id': fly_id,
                'suffix': f"_{fly_id}",
                'csv': csv_path.name,
                'bouts_csv': bouts_csv_name,
            })
    else:
        # Single-fly layout: data3D.csv
        flies.append({
            'fly_id': None,
            'suffix': '',
            'csv': 'data3D.csv',
            'bouts_csv': f'{dataset}_bouts_summary.csv',
        })

    return flies
