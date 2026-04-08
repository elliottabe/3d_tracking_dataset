#!/usr/bin/env python3
"""Split a paired dual-fly h5 back into per-fly STAC-ready h5 files.

Reads a paired h5 produced by ``merge_paired_bouts.py`` and materialises two
per-fly h5 files in the legacy single-fly layout that the existing STAC
pipeline (``run_stac_fly_model.py`` / ``batch_run_stac.py``) already knows how
to consume. A bout is only emitted for a given fly if that fly has at least
``--min-valid-frames`` valid frames, so bouts where only one fly is tracked
well still feed the STAC run for that fly.

A ``paired_index.json`` sidecar is written alongside the per-fly files that
maps each output bout key to the fly0/fly1 bout keys in the paired file and
stores the per-frame ``pair_state``, so downstream rendering/analysis code can
align the two STAC IK outputs frame-by-frame.

Typical workflow (courtship):
    1. python scripts/preprocess_keypoints_for_ik.py dataset=courtship \\
           preprocessing.csv_path=data3D_fly0.csv preprocessing.bout_name=...fly0
    2. (same for fly1)
    3. python scripts/merge_paired_bouts.py --fly0 ...fly0.h5 --fly1 ...fly1.h5 \\
           --out ...paired.h5
    4. python scripts/run_stac_paired.py --paired ...paired.h5
    5. python scripts/batch_run_stac.py --dataset courtship --anatomy v1
       (picks up the two new per-fly files automatically)
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


def _uint8_to_intlist(arr):
    return [int(x) for x in np.asarray(arr).tolist()]


def split_paired_to_per_fly(
    paired_path: Path,
    out_fly0: Path,
    out_fly1: Path,
    index_path: Path,
    min_valid_frames: int = 30,
    force: bool = False,
) -> dict:
    if (out_fly0.exists() and out_fly1.exists() and index_path.exists()
            and not force):
        print(f"✓ Per-fly split already exists (use --force to overwrite):")
        print(f"    {out_fly0}")
        print(f"    {out_fly1}")
        print(f"    {index_path}")
        return {}

    print(f"Loading paired h5: {paired_path}")
    d = ioh5.load(paired_path, enable_jax=False)
    info = d.get("info", {})
    base_ids = list(info.get("fly_ids", []))
    clip_lengths = list(info.get("clip_lengths", []))
    bout_keys = sorted(k for k in d.keys() if k != "info")
    assert len(bout_keys) == len(base_ids) == len(clip_lengths), \
        f"length mismatch bouts={len(bout_keys)} ids={len(base_ids)} clips={len(clip_lengths)}"

    fly0_out: dict = {}
    fly1_out: dict = {}
    fly0_info = {"fly_ids": [], "source_flies": [], "clip_lengths": []}
    fly1_info = {"fly_ids": [], "source_flies": [], "clip_lengths": []}
    for k in ("kp_names", "skeleton_edges"):
        if k in info:
            fly0_info[k] = info[k]
            fly1_info[k] = info[k]

    index: list[dict] = []
    n_f0 = 0
    n_f1 = 0

    for bk, base, T in zip(bout_keys, base_ids, clip_lengths):
        b = d[bk]
        valid0 = np.asarray(b["valid_fly0"], dtype=bool)
        valid1 = np.asarray(b["valid_fly1"], dtype=bool)
        pair_state = np.asarray(b["pair_state"], dtype=np.uint8)

        entry = {
            "paired_bout_key": bk,
            "base_id": base,
            "n_frames": int(T),
            "pair_state": _uint8_to_intlist(pair_state),
            "fly0_bout_key": None,
            "fly1_bout_key": None,
            "fly0_valid_frames": int(valid0.sum()),
            "fly1_valid_frames": int(valid1.sum()),
            "paired_frames": int((pair_state == 3).sum()),
        }

        if int(valid0.sum()) >= min_valid_frames:
            new_key = f"bout_{n_f0:03d}"
            fly0_out[new_key] = {
                "keypoints": np.asarray(b["keypoints_fly0"]),
                "kp_names": info.get("kp_names", []),
            }
            if "orig_keypoints_fly0" in b:
                fly0_out[new_key]["orig_keypoints"] = np.asarray(b["orig_keypoints_fly0"])
            if "alignment_info_fly0" in b:
                fly0_out[new_key]["alignment_info"] = b["alignment_info_fly0"]
            fly0_out[new_key]["valid_fly"] = valid0
            fly0_out[new_key]["pair_state"] = pair_state
            fly0_info["fly_ids"].append(f"{base}_fly0")
            fly0_info["source_flies"].append("fly0")
            fly0_info["clip_lengths"].append(int(T))
            entry["fly0_bout_key"] = new_key
            n_f0 += 1

        if int(valid1.sum()) >= min_valid_frames:
            new_key = f"bout_{n_f1:03d}"
            fly1_out[new_key] = {
                "keypoints": np.asarray(b["keypoints_fly1"]),
                "kp_names": info.get("kp_names", []),
            }
            if "orig_keypoints_fly1" in b:
                fly1_out[new_key]["orig_keypoints"] = np.asarray(b["orig_keypoints_fly1"])
            if "alignment_info_fly1" in b:
                fly1_out[new_key]["alignment_info"] = b["alignment_info_fly1"]
            fly1_out[new_key]["valid_fly"] = valid1
            fly1_out[new_key]["pair_state"] = pair_state
            fly1_info["fly_ids"].append(f"{base}_fly1")
            fly1_info["source_flies"].append("fly1")
            fly1_info["clip_lengths"].append(int(T))
            entry["fly1_bout_key"] = new_key
            n_f1 += 1

        index.append(entry)

    fly0_out["info"] = fly0_info
    fly1_out["info"] = fly1_info

    out_fly0.parent.mkdir(parents=True, exist_ok=True)
    ioh5.save(out_fly0, fly0_out)
    ioh5.save(out_fly1, fly1_out)

    manifest = {
        "paired_h5": str(paired_path),
        "fly0_h5": str(out_fly0),
        "fly1_h5": str(out_fly1),
        "min_valid_frames": int(min_valid_frames),
        "n_fly0_bouts": n_f0,
        "n_fly1_bouts": n_f1,
        "bouts": index,
    }
    with open(index_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n✓ Wrote {n_f0} fly0 bouts → {out_fly0}")
    print(f"✓ Wrote {n_f1} fly1 bouts → {out_fly1}")
    print(f"✓ Wrote pairing manifest → {index_path}")
    print("\nNext: run STAC on both per-fly files, e.g.")
    print("  python scripts/batch_run_stac.py --dataset courtship --anatomy v1")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paired", type=Path, required=True,
                    help="Paired h5 from merge_paired_bouts.py")
    ap.add_argument("--out-fly0", type=Path, default=None,
                    help="Output per-fly0 h5 (default: alongside paired, "
                         "with _fly0_paired suffix)")
    ap.add_argument("--out-fly1", type=Path, default=None)
    ap.add_argument("--index", type=Path, default=None,
                    help="Output pairing manifest json "
                         "(default: alongside paired h5)")
    ap.add_argument("--min-valid-frames", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    stem = args.paired.stem
    base_stem = stem[: -len("_paired")] if stem.endswith("_paired") else stem
    out_fly0 = args.out_fly0 or (
        args.paired.with_name(f"{base_stem}_fly0_paired.h5"))
    out_fly1 = args.out_fly1 or (
        args.paired.with_name(f"{base_stem}_fly1_paired.h5"))
    index_path = args.index or (
        args.paired.with_name(f"{base_stem}_paired_index.json"))

    split_paired_to_per_fly(
        args.paired, out_fly0, out_fly1, index_path,
        min_valid_frames=args.min_valid_frames, force=args.force,
    )


if __name__ == "__main__":
    main()
