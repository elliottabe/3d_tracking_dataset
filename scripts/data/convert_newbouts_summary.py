"""Materialize batch-IK input dirs from the NewBouts curated bout tree.

The NewBouts curation tree (processed/free_running/NewBouts/<ts>/) ships, per
recording, the workstation's 3D keypoint predictions (`data3D.csv`), an
`info.yaml` (with the workstation `recording_path`), and a curated
`running_bouts_summary.csv` with schema
    bout,start_frame,end_frame,n_frames,duration_s,min_cycles,status,...
where `status` marks manual accept/reject decisions.

The batch IK route (scripts/batch_process_predictions.py ->
batch_run_stac.py -> batch_postprocess_predictions.py, discovered via
rglob("Predictions_3D_*")) needs, per dir, `data3D.csv` plus
`free_running_bouts_summary.csv` whose schema
(preprocess_keypoints_for_ik.load_bouts_from_csv) REQUIRES
`bout_idx,start_frame,end_frame,fly_id`.

So per recording this script creates `<ts>/Predictions_3D_newbouts/` with:
  - `data3D.csv`   relative symlink -> ../data3D.csv
  - `info.yaml`    relative symlink -> ../info.yaml (provenance)
  - `free_running_bouts_summary.csv`  accepted rows only, bout -> bout_idx
    (numbering preserved for traceability), fly_id = "<parent>/<ts>" derived
    from info.yaml's recording_path (a trailing "videos" component is
    stripped, e.g. .../session5/<ts>/videos/ -> "session5/<ts>").

All-rejected recordings get NO Predictions_3D_newbouts dir, so the batch
drivers never see them.

Usage:
    # one recording
    python scripts/data/convert_newbouts_summary.py \
        --recording /gscratch/.../processed/free_running/NewBouts/2025_10_07_17_15_30
    # every recording under the root
    python scripts/data/convert_newbouts_summary.py \
        --root /gscratch/.../processed/free_running/NewBouts
Idempotent: outputs are deterministically derived and re-running overwrites.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

SOURCE_NAME = "running_bouts_summary.csv"
OUT_NAME = "free_running_bouts_summary.csv"
PRED_DIR_NAME = "Predictions_3D_newbouts"
_REQUIRED = ("bout", "start_frame", "end_frame", "status")


def fly_id_from_info(info_yaml: Path) -> str:
    """"<parent>/<ts>" from info.yaml's recording_path, e.g.
    /mnt/lemebel/happyhouse_102025/session5/<ts>/videos/ -> "session5/<ts>".
    Falls back to "NewBouts/<recording-dir-name>" when unparseable."""
    try:
        for line in info_yaml.read_text().splitlines():
            if line.startswith("recording_path:"):
                parts = [p for p in line.split(":", 1)[1].strip().split("/") if p]
                if parts and parts[-1] == "videos":
                    parts = parts[:-1]
                if len(parts) >= 2:
                    return f"{parts[-2]}/{parts[-1]}"
    except OSError:
        pass
    return f"NewBouts/{info_yaml.parent.name}"


def convert_summary(src: Path, out: Path, *, fly_id: str) -> int:
    """Convert one running_bouts_summary.csv -> batch-route schema at `out`.

    Returns the number of accepted rows written. Writes nothing (and returns
    0) when no row is accepted."""
    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in _REQUIRED if c not in fields]
        if missing:
            raise ValueError(
                f"{src}: missing column(s) {missing}; has {fields} -- not a "
                f"NewBouts running_bouts_summary.csv?")
        rows = [r for r in reader if r["status"] == "accepted"]
    if not rows:
        return 0
    out_fields = ["fly_id"] + ["bout_idx" if c == "bout" else c for c in fields]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["bout_idx"] = r.pop("bout")
            r["fly_id"] = fly_id
            w.writerow(r)
    return len(rows)


def _relink(link: Path, target_name: str) -> None:
    """(Re-)create a relative symlink `link` -> ../<target_name>."""
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(os.path.join("..", target_name), link)


def prepare_recording(rec_dir: Path) -> Path | None:
    """Materialize <rec_dir>/Predictions_3D_newbouts/ for the batch route.

    Returns the predictions dir, or None when the recording has no accepted
    bouts (in which case nothing is created)."""
    src = rec_dir / SOURCE_NAME
    fly_id = fly_id_from_info(rec_dir / "info.yaml")
    pred = rec_dir / PRED_DIR_NAME
    pred.mkdir(exist_ok=True)
    n = convert_summary(src, pred / OUT_NAME, fly_id=fly_id)
    if n == 0:
        (pred / OUT_NAME).unlink(missing_ok=True)
        try:
            pred.rmdir()
        except OSError:
            pass   # keep non-empty dirs (existing outputs) intact
        return None
    _relink(pred / "data3D.csv", "data3D.csv")
    if (rec_dir / "info.yaml").is_file():
        _relink(pred / "info.yaml", "info.yaml")
    return pred


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--recording", type=Path,
                   help=f"one NewBouts recording dir (contains {SOURCE_NAME})")
    g.add_argument("--root", type=Path,
                   help="NewBouts root; prepares every <ts>/ under it")
    a = ap.parse_args(argv)

    recs = [a.recording] if a.recording else sorted(
        d for d in a.root.iterdir() if (d / SOURCE_NAME).is_file())
    if not recs:
        print(f"no {SOURCE_NAME} found", file=sys.stderr)
        return 1
    total = prepared = 0
    for d in recs:
        if not (d / SOURCE_NAME).is_file():
            print(f"skip {d.name}: no {SOURCE_NAME}", file=sys.stderr)
            continue
        pred = prepare_recording(d)
        if pred is None:
            print(f"{d.name}: 0 accepted bout(s)  (skipped)")
            continue
        with open(pred / OUT_NAME, newline="") as f:
            n = sum(1 for _ in f) - 1
        fid = fly_id_from_info(d / "info.yaml")
        total += n
        prepared += 1
        print(f"{d.name}: {n} accepted bout(s)  fly_id={fid}")
    print(f"total: {total} accepted bouts across {prepared} prepared "
          f"recording(s) (of {len(recs)} scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
