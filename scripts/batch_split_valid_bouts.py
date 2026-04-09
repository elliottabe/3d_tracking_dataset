#!/usr/bin/env python3
"""Classify per-fly preprocessed bouts into valid-bucket files for STAC.

Replaces the old paired-bout merge/split flow (``batch_pair_bouts.py`` →
``merge_paired_bouts.py`` + ``run_stac_paired.py``).

For each ``Predictions_3D_*`` folder we read the per-fly preprocessed h5s
created by ``preprocess_keypoints_for_ik.py`` (which already carry a per-frame
``valid_fly`` mask in each bout) and emit up to three bucket files used as
STAC inputs:

  preprocessing/preprocessed_bout_<a>_<d>_fly0_only.h5  (fly0 valid, fly1 not)
  preprocessing/preprocessed_bout_<a>_<d>_fly1_only.h5  (fly1 valid, fly0 not)
  preprocessing/preprocessed_bout_<a>_<d>_both.h5       (both valid; one entry
                                                          per fly, fly0 first)

A bout is the **union** of the three conditions: if both flies are valid in
the same bout it lands in ``_both`` (not in either ``_only`` file). If only
one fly clears its threshold, the bout lands in that fly's ``_only`` file.
Empty buckets are not written. Folders that produce no bouts at all are
reported and skipped without crashing the pipeline.

A small ``preprocessed_bout_<a>_<d>_split_index.json`` manifest is written
alongside the bucket files for debugging / downstream alignment.

Usage:
    python scripts/batch_split_valid_bouts.py --dataset courtship --anatomy v2 \\
        --base-dir /path/to/Johnson_lab/courtship
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils import io_dict_to_hdf5 as ioh5  # noqa: E402


BUCKETS = ("fly0_only", "fly1_only", "both")


def _find_folders(base_dir: Path) -> list[Path]:
    if base_dir.is_dir() and base_dir.match("Predictions_3D_*"):
        return [base_dir]
    return sorted({p for p in base_dir.rglob("Predictions_3D_*") if p.is_dir()})


def _get_thresholds(d0: dict) -> tuple[int, int]:
    """Pull min_solo_frames / min_paired_frames from preprocessed info if present."""
    pv = (d0.get("info", {}) or {}).get("pair_validity", {}) or {}
    min_solo = int(pv.get("min_solo_frames", 30))
    min_paired = int(pv.get("min_paired_frames", 30))
    return min_solo, min_paired


def _bout_count(mask) -> int:
    return int(np.asarray(mask, dtype=bool).sum())


def _copy_bout(src_bout: dict) -> dict:
    """Shallow copy a bout dict — values are arrays/lists we don't mutate."""
    return {k: v for k, v in src_bout.items()}


def classify_and_split(
    fly0_path: Path,
    fly1_path: Path,
    out_paths: dict[str, Path],
    index_path: Path,
    force: bool,
) -> dict:
    """Read per-fly preprocessed h5s and emit bucket files. Returns manifest."""
    if not force and all(p.exists() for p in out_paths.values()) and index_path.exists():
        with open(index_path) as fh:
            return json.load(fh)

    d0 = ioh5.load(fly0_path, enable_jax=False)
    d1 = ioh5.load(fly1_path, enable_jax=False)

    info0 = d0.get("info", {}) or {}
    info1 = d1.get("info", {}) or {}
    min_solo, min_paired = _get_thresholds(d0)

    bout_keys0 = sorted(k for k in d0.keys() if k != "info")
    bout_keys1 = sorted(k for k in d1.keys() if k != "info")
    shared = sorted(set(bout_keys0) & set(bout_keys1))

    fly_ids0 = list(info0.get("fly_ids", []))
    fly_ids1 = list(info1.get("fly_ids", []))
    clip_lengths0 = list(info0.get("clip_lengths", []))
    start_frames0 = list(info0.get("start_frames", []))
    end_frames0 = list(info0.get("end_frames", []))
    start_frames1 = list(info1.get("start_frames", []))
    end_frames1 = list(info1.get("end_frames", []))

    def _idx(keys, k):
        try:
            return keys.index(k)
        except ValueError:
            return None

    bucket_data: dict[str, dict] = {b: {} for b in BUCKETS}
    bucket_info: dict[str, dict] = {
        b: {"fly_ids": [], "source_flies": [], "clip_lengths": [],
            "start_frames": [], "end_frames": [], "bucket": []}
        for b in BUCKETS
    }
    for b in BUCKETS:
        for k in ("kp_names", "skeleton_edges"):
            if k in info0:
                bucket_info[b][k] = info0[k]
        # Echo pair_validity config so downstream stages can introspect
        if "pair_validity" in info0:
            bucket_info[b]["pair_validity"] = info0["pair_validity"]

    counters = {b: 0 for b in BUCKETS}
    index_entries: list[dict] = []

    for bk in shared:
        b0 = d0[bk]
        b1 = d1[bk]
        v0 = np.asarray(b0.get("valid_fly", []), dtype=bool)
        v1 = np.asarray(b1.get("valid_fly", []), dtype=bool)
        if v0.shape != v1.shape or v0.size == 0:
            # Defensive: if masks are missing or shape-mismatched, skip the bout.
            index_entries.append({
                "src_bout_key": bk,
                "skipped": True,
                "reason": f"valid_fly shape mismatch or empty (v0={v0.shape}, v1={v1.shape})",
            })
            continue

        n0 = _bout_count(v0)
        n1 = _bout_count(v1)
        n_both = _bout_count(v0 & v1)

        i0 = _idx(bout_keys0, bk)
        T = int(clip_lengths0[i0]) if i0 is not None and i0 < len(clip_lengths0) else int(v0.size)
        base_id0 = fly_ids0[i0] if i0 is not None and i0 < len(fly_ids0) else f"{bk}_fly0"
        sf0 = int(start_frames0[i0]) if i0 is not None and i0 < len(start_frames0) else -1
        ef0 = int(end_frames0[i0]) if i0 is not None and i0 < len(end_frames0) else -1
        i1 = _idx(bout_keys1, bk)
        base_id1 = fly_ids1[i1] if i1 is not None and i1 < len(fly_ids1) else f"{bk}_fly1"
        sf1 = int(start_frames1[i1]) if i1 is not None and i1 < len(start_frames1) else -1
        ef1 = int(end_frames1[i1]) if i1 is not None and i1 < len(end_frames1) else -1

        entry = {
            "src_bout_key": bk,
            "n_frames": T,
            "n_valid_fly0": n0,
            "n_valid_fly1": n1,
            "n_valid_both": n_both,
            "bucket": None,
            "out_keys": {},
        }

        if n_both >= min_paired:
            # 'both' wins — emit two entries (fly0 then fly1) so STAC sees both.
            new_key0 = f"bout_{counters['both']:03d}"
            counters["both"] += 1
            new_key1 = f"bout_{counters['both']:03d}"
            counters["both"] += 1

            bucket_data["both"][new_key0] = _copy_bout(b0)
            bucket_data["both"][new_key1] = _copy_bout(b1)
            for new_key, base, src_fly, sf, ef in (
                (new_key0, base_id0, "fly0", sf0, ef0),
                (new_key1, base_id1, "fly1", sf1, ef1),
            ):
                bucket_info["both"]["fly_ids"].append(str(base))
                bucket_info["both"]["source_flies"].append(src_fly)
                bucket_info["both"]["clip_lengths"].append(T)
                bucket_info["both"]["start_frames"].append(int(sf))
                bucket_info["both"]["end_frames"].append(int(ef))
                bucket_info["both"]["bucket"].append("both")
            entry["bucket"] = "both"
            entry["out_keys"] = {"fly0": new_key0, "fly1": new_key1}
        else:
            if n0 >= min_solo:
                new_key = f"bout_{counters['fly0_only']:03d}"
                counters["fly0_only"] += 1
                bucket_data["fly0_only"][new_key] = _copy_bout(b0)
                bucket_info["fly0_only"]["fly_ids"].append(str(base_id0))
                bucket_info["fly0_only"]["source_flies"].append("fly0")
                bucket_info["fly0_only"]["clip_lengths"].append(T)
                bucket_info["fly0_only"]["start_frames"].append(int(sf0))
                bucket_info["fly0_only"]["end_frames"].append(int(ef0))
                bucket_info["fly0_only"]["bucket"].append("fly0_only")
                entry["bucket"] = "fly0_only"
                entry["out_keys"] = {"fly0": new_key}
            if n1 >= min_solo:
                new_key = f"bout_{counters['fly1_only']:03d}"
                counters["fly1_only"] += 1
                bucket_data["fly1_only"][new_key] = _copy_bout(b1)
                bucket_info["fly1_only"]["fly_ids"].append(str(base_id1))
                bucket_info["fly1_only"]["source_flies"].append("fly1")
                bucket_info["fly1_only"]["clip_lengths"].append(T)
                bucket_info["fly1_only"]["start_frames"].append(int(sf1))
                bucket_info["fly1_only"]["end_frames"].append(int(ef1))
                bucket_info["fly1_only"]["bucket"].append("fly1_only")
                # If neither solo passed, bucket stays None.
                if entry["bucket"] is None:
                    entry["bucket"] = "fly1_only"
                    entry["out_keys"] = {"fly1": new_key}
                elif entry["bucket"] == "fly0_only":
                    entry["bucket"] = "fly0_only+fly1_only"
                    entry["out_keys"]["fly1"] = new_key

        index_entries.append(entry)

    # Write bucket files (skip empties)
    written = {}
    for b in BUCKETS:
        if counters[b] == 0:
            continue
        bucket_data[b]["info"] = bucket_info[b]
        out = out_paths[b]
        out.parent.mkdir(parents=True, exist_ok=True)
        ioh5.save(out, bucket_data[b])
        written[b] = str(out)
        print(f"  ✓ {b}: {counters[b]} entries → {out.name}")

    manifest = {
        "fly0_h5": str(fly0_path),
        "fly1_h5": str(fly1_path),
        "min_solo_frames": min_solo,
        "min_paired_frames": min_paired,
        "n_shared_bouts": len(shared),
        "counts": counters,
        "outputs": written,
        "bouts": index_entries,
    }
    with open(index_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def process_folder(folder: Path, anatomy: str, dataset: str,
                   force: bool, dry_run: bool) -> dict:
    preproc = folder / "preprocessing"
    result = {"folder": str(folder), "status": "skipped", "message": "", "counts": {}}
    if not preproc.exists():
        result["message"] = "no preprocessing/ dir"
        return result

    stem = f"preprocessed_bout_{anatomy}_{dataset}"
    fly0 = preproc / f"{stem}_fly0.h5"
    fly1 = preproc / f"{stem}_fly1.h5"
    if not fly0.exists() or not fly1.exists():
        result["message"] = "no fly0/fly1 pair (single-fly dataset)"
        return result

    out_paths = {
        "fly0_only": preproc / f"{stem}_fly0_only.h5",
        "fly1_only": preproc / f"{stem}_fly1_only.h5",
        "both": preproc / f"{stem}_both.h5",
    }
    index_path = preproc / f"{stem}_split_index.json"

    print(f"\n[split] {folder.name}")
    if dry_run:
        result["status"] = "dry-run"
        result["message"] = f"would split {fly0.name} + {fly1.name}"
        return result

    try:
        manifest = classify_and_split(fly0, fly1, out_paths, index_path, force=force)
        counts = manifest.get("counts", {})
        result["counts"] = counts
        total = sum(counts.get(b, 0) for b in BUCKETS)
        if total == 0:
            print(f"  [skip] no valid bouts captured in {folder.name}")
            result["status"] = "no_bouts"
            result["message"] = "no bouts above thresholds"
        else:
            result["status"] = "success"
            result["message"] = ", ".join(f"{b}={counts.get(b,0)}" for b in BUCKETS)
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
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.base_dir.exists():
        print(f"Error: base dir not found: {args.base_dir}", file=sys.stderr)
        sys.exit(1)

    folders = _find_folders(args.base_dir)
    if not folders:
        print(f"No Predictions_3D_* folders under {args.base_dir}")
        sys.exit(0)

    print(f"Found {len(folders)} prediction folder(s)")
    results = [process_folder(f, args.anatomy, args.dataset, args.force, args.dry_run)
               for f in folders]

    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # 'no_bouts' and 'skipped' are not failures — pipeline should continue.
    if counts.get("error", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
