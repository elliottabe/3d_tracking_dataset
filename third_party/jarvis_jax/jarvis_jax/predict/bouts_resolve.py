"""Resolve a recording's canonical bout-summary CSV.

The pipeline reads one per-recording file `{dataset}_bout_summary.csv` from the
processed dir. This module locates that file, creating it by normalizing the
first available source (Session0 unified CSV or Session1 quality_viz good_bouts
CSV). Legacy `free_walking*` files are accepted when dataset == 'free_running'.
"""
import csv
import glob
import os

_REQUIRED = ("bout_idx", "start_frame", "end_frame")


def _read_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _datasets_to_try(dataset):
    out = [dataset]
    if dataset == "free_running":
        out.append("free_walking")   # legacy files not yet renamed
    return out


def _find_source(recording_dir, dataset):
    """First available (rows, fieldnames) from a bout source, or None."""
    for ds in _datasets_to_try(dataset):
        for name in (f"{ds}_bout_summary.csv", f"{ds}_bouts_unified_summary.csv"):
            p = os.path.join(recording_dir, name)
            if os.path.isfile(p):
                return _read_rows(p)
        gv = sorted(
            glob.glob(os.path.join(recording_dir, "quality_viz", "*_good_bouts.csv")),
            key=os.path.getmtime)
        if gv:
            return _read_rows(gv[-1])   # newest by mtime
    return None


def resolve_bout_summary(*, recording_dir, processed_dir, dataset):
    """Path to <processed_dir>/<dataset>_bout_summary.csv, created (normalized)
    from the first available source if absent. Raises FileNotFoundError when no
    source exists."""
    canonical = os.path.join(processed_dir, f"{dataset}_bout_summary.csv")
    if os.path.isfile(canonical):
        return canonical
    found = _find_source(recording_dir, dataset)
    if found is None:
        raise FileNotFoundError(
            f"No bout source for dataset '{dataset}' under {recording_dir} "
            f"(looked for {dataset}_bout_summary.csv, "
            f"{dataset}_bouts_unified_summary.csv, quality_viz/*_good_bouts.csv).")
    rows, fieldnames = found
    for r in rows:
        missing = [c for c in _REQUIRED if c not in r]
        if missing:
            raise ValueError(
                f"bout source missing column(s) {missing}; has {list(r.keys())}")
    os.makedirs(processed_dir, exist_ok=True)
    with open(canonical, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return canonical
