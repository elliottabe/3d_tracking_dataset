"""Prove a dataset root's train/val split is disjoint BY BYTES, from disk.

    python scripts/analysis/verify_split_content.py --root <dataset_root>

This is an ACCEPTANCE GATE, not a report. It re-derives everything from the
files the trainer will actually open -- it reads `annotations/instances_train.
json` and `instances_val.json`, resolves every image path they reference to a
real file, and md5s that file. It deliberately shares NO code and NO cached
state with the builder: the builder's own guard (split_v5.write_derived) can
only prove the split it intended to write is clean, whereas this proves the
split that is on disk is clean.

WHY THIS EXISTS. `red_data_3d_v5` and `red_data_3d_v5_valfix` both shipped
without a content check and both leaked ~30% of val -- the same footage was
ingested twice under two recording names, so every name-, frame- and
fly-level check passed while the pixels were on both sides. Every val number
measured on those roots is optimistic. Names are a proxy; bytes are the
ground truth, and they are cheap to check.

Checks, all fatal:
  1. every referenced image path exists and resolves
  2. zero md5 shared between train and val          <- the one that matters
  3. no md5 appears twice WITHIN a side (self-duplication wastes training
     capacity and silently reweights the sampler)
  4. every annotation's image_id is present in its own side's image list
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=128)
    args = ap.parse_args()

    sides, missing = {}, []
    for name in ("train", "val"):
        p = os.path.join(args.root, "annotations", f"instances_{name}.json")
        blob = json.load(open(p))
        paths = {}
        for im in blob["images"]:
            fp = os.path.join(args.root, "images", im["file_name"])
            if not os.path.exists(fp):
                missing.append(fp)
            paths[im["id"]] = fp
        ids = set(paths)
        orphan = [a["id"] for a in blob["annotations"] if a["image_id"] not in ids]
        sides[name] = {"paths": paths, "blob": blob, "orphan": orphan}
        print(f"{name:5s}: {len(blob['images']):6d} images  "
              f"{len(blob['annotations']):6d} annotations  "
              f"{len(blob['framesets']):5d} framesets")

    if missing:
        raise SystemExit(f"FAIL: {len(missing)} referenced images do not exist, "
                         f"e.g. {missing[:3]}")
    for name, d in sides.items():
        if d["orphan"]:
            raise SystemExit(f"FAIL: {name} has {len(d['orphan'])} annotations "
                             f"whose image_id is not in its own image list")

    todo = sorted({p for d in sides.values() for p in d["paths"].values()})
    print(f"hashing {len(todo)} files from disk ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        h = dict(zip(todo, ex.map(_md5, todo)))

    md5s, dupes = {}, {}
    for name, d in sides.items():
        c = collections.Counter(h[p] for p in d["paths"].values())
        md5s[name] = set(c)
        dupes[name] = {k: v for k, v in c.items() if v > 1}
        print(f"{name:5s}: {len(d['paths']):6d} paths -> "
              f"{len(c):6d} unique contents  ({len(dupes[name])} duplicated)")

    overlap = md5s["train"] & md5s["val"]
    print("\n=== ASSERTION: train content INTERSECT val content == 0 ===")
    print(f"    |train| = {len(md5s['train'])}   |val| = {len(md5s['val'])}")
    print(f"    |intersection| = {len(overlap)}")
    ok = True
    if overlap:
        ok = False
        print(f"    FAIL -- examples: {sorted(overlap)[:5]}")
    else:
        print("    PASS -- the two sides share no image content")
    for name in ("train", "val"):
        if dupes[name]:
            ok = False
            print(f"    FAIL -- {name} duplicates {len(dupes[name])} contents "
                  f"internally, e.g. {sorted(dupes[name])[:3]}")
    if not ok:
        raise SystemExit("VERIFICATION FAILED")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
