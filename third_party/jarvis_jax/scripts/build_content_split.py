"""Rebuild the red_data_3d_v5 split keyed on IMAGE CONTENT, into a new root.

    python third_party/jarvis_jax/scripts/build_content_split.py \
        --src  .../red_data/red_data_3d_v5_valfix \
        --out  .../red_data/red_data_3d_v6_contentsplit

READ-ONLY on `--src` and on both upstream corpora. Emits a new root whose
`images/` is a single symlink back at the source tree (the pixels are already
there; copying 4.4 GB to change a JSON would be silly) and whose
`annotations/` carry a split that is disjoint BY BYTES, not by name.

WHAT WAS WRONG. `red_data_3d_v5_valfix` splits by recording name. Six groups
of recording names are the same footage re-ingested (see
jarvis_jax.data.content_index), and three of its four VAL_RECORDINGS have
their twin in train. Measured: 511/1729 val images (29.6%) and 620/1871 val
annotations (33.1%) byte-identical to something in train.

WHAT THIS DOES. The split atom becomes the capture `(alias component, frame)`,
so every path holding the same bytes lands on the same side. The holdout unit
becomes the alias COMPONENT.

POLICY, and the judgement in it.
  * Held out WHOLE (unseen fly + unseen session):
      2026_03_18_15_31_22                      group B, courtship male
      2026_06_18_19_23_03 + 2026_06_19_11_09_36  group A, courtship M+F
      2026_05_27_11_56_05 + 2026_05_27_11_57_05  group C, courtship M+F
    The last two are female-inclusive, which the female-preservation rule
    normally forbids holding out whole. Spent deliberately via
    `val_components_force`: they are the two SMALLEST female-inclusive
    components (38 of 569 female framesets, 6.7%), and without them there is
    no unseen-session female number at all and no group-C val at all -- the
    old split got both only by leaking. Cost is stated, not hidden.
  * Every other female-inclusive component keeps the existing policy: stays
    in training, contributes a guarded 10% tail. That is where two-fly/overlap
    val coverage now comes from (components B and C), as "unseen pose, same
    session" rather than the leaked "unseen session" it used to claim.
  * Male-only components not listed: all train, unchanged.

ANNOTATIONS ARE NOT DEDUPLICATED HERE. On an aliased capture the two
recordings can carry (a) the SAME fly labelled twice -- median 0.00 px apart,
mean 1.3-1.5, tail to ~10 px on Antenna_Base / Scutellum / distal tarsi -- and
(b) a second fly that exists on only one side. (b) is real data and must be
kept; (a) is redundant but its two labels DISAGREE, and which one is right is
a human call, so dropping either here would prejudge the adjudication that
scripts/analysis/label_disagreement.py exists to support. Both are kept, and
every annotation is tagged with its capture so the pairs are recoverable.
Redundancy oversamples those captures; it does not leak, because both copies
are on the same side by construction.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from jarvis_jax.data.content_index import (            # noqa: E402
    alias_components, capture_of, content_groups, hash_paths)
from jarvis_jax.data.split_v5 import (                 # noqa: E402
    audit_split, make_split, write_derived)

VAL_RECORDINGS = ["2026_03_18_15_31_22"]
VAL_COMPONENTS_FORCE = ["2026_06_18_19_23_03", "2026_05_27_11_56_05"]


def resolve_image_paths(src_root: str, images: list[dict]) -> dict[int, str]:
    """coco image id -> the REAL file, through the symlink into whichever
    upstream corpus stores it. `os.path.realpath` normalises /gscratch ->
    /mmfs1/gscratch, so paths from here are comparable to each other but not
    to a hand-written /gscratch string; never join the two spaces."""
    return {im["id"]: os.path.realpath(
        os.path.join(src_root, "images", im["file_name"])) for im in images}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hash-cache", default=None,
                    help="json {path: md5}; written if absent, reused if present")
    ap.add_argument("--workers", type=int, default=192)
    ap.add_argument("--female-val-frac", type=float, default=0.10)
    # 200, not split_v5's 50. Measured on this data (RMS grey difference of a
    # 64x16 thumbnail, 0-255): frames 1 apart score 0.39, random pairs 96.6.
    # At guard=50 the nearest val image sat at 1.39 and 14 were under 2.0 --
    # i.e. still a few frames from a train image. guard=200 lifts the floor to
    # 1.58 with 3 under 2.0, costs 42 more TRAIN framesets to the band, and
    # costs val nothing (the band only ever removes train). 500 buys nothing
    # further.
    ap.add_argument("--guard", type=int, default=200)
    args = ap.parse_args()

    inst = json.load(open(os.path.join(args.src, "annotations", "instances.json")))
    man = json.load(open(os.path.join(args.src, "manifest.json")))
    real = resolve_image_paths(args.src, inst["images"])

    cache = {}
    if args.hash_cache and os.path.exists(args.hash_cache):
        cache = json.load(open(args.hash_cache))
    todo = sorted({p for p in real.values() if p not in cache})
    if todo:
        print(f"hashing {len(todo)} files ...", flush=True)
        cache.update(hash_paths(todo, workers=args.workers))
        if args.hash_cache:
            json.dump(cache, open(args.hash_cache, "w"))
    path_hash = {p: cache[p] for p in real.values()}
    image_hashes = {i: cache[p] for i, p in real.items()}

    rec_of_img = {im["id"]: im["recording"] for im in inst["images"]}
    path_rec = {}
    for i, p in real.items():
        path_rec[p] = rec_of_img[i]
    aliases = alias_components(path_hash, lambda p: path_rec[p])
    comps = collections.defaultdict(list)
    for r, c in aliases.items():
        comps[c].append(r)
    multi = {c: sorted(v) for c, v in comps.items() if len(v) > 1}
    print(f"{len(path_hash)} images -> {len(set(path_hash.values()))} unique "
          f"contents; {len(multi)} alias components:")
    for c, v in sorted(multi.items()):
        print("   ", " == ".join(v))

    split = make_split(inst, man, val_recordings=VAL_RECORDINGS,
                       val_components_force=VAL_COMPONENTS_FORCE,
                       female_val_frac=args.female_val_frac, guard=args.guard,
                       aliases=aliases)
    audit = audit_split(inst, split, guard=args.guard, aliases=aliases,
                        image_hashes=image_hashes)
    print("audit:", json.dumps({k: v for k, v in audit.items()
                                if not k.startswith("leaky")
                                and k != "min_guard_distance"}, indent=2))
    if (audit["cross_fly_leaked_frames"] or audit["cross_capture_leaks"]
            or audit["content_overlap"]):
        raise SystemExit(f"REFUSING to write a leaky split: {audit}")

    os.makedirs(args.out, exist_ok=True)
    for name in ("images", "masks", "calibrations"):
        link = os.path.join(args.out, name)
        srcdir = os.path.join(args.src, name)
        if not os.path.lexists(link) and os.path.isdir(srcdir):
            os.symlink(srcdir, link)

    # capture tag on every image, so the pairs stay recoverable downstream
    caps = {im["id"]: capture_of(im["recording"],
                                 int(im["file_name"].rsplit("Frame_", 1)[1]
                                     .split(".")[0]), aliases)
            for im in inst["images"]}
    for im in inst["images"]:
        im["capture"] = caps[im["id"]]
        im["content_md5"] = image_hashes[im["id"]]

    write_derived(inst, split, args.out, image_hashes=image_hashes)
    with open(os.path.join(args.out, "annotations", "instances.json"), "w") as f:
        json.dump(inst, f)

    groups = content_groups(path_hash)
    with open(os.path.join(args.out, "content_index.json"), "w") as f:
        json.dump({"alias_components": {c: sorted(v) for c, v in comps.items()},
                   "n_images": len(path_hash),
                   "n_unique_content": len(groups),
                   "duplicate_groups": {h: v for h, v in groups.items()
                                        if len(v) > 1}}, f, indent=1)

    man["version"] = "red_data_3d_v6_contentsplit"
    man["source_root"] = args.src
    man["alias_components"] = {c: sorted(v) for c, v in sorted(multi.items())}
    for rec in man["recordings"]:
        man["recordings"][rec]["alias_component"] = aliases.get(rec, rec)
        man["recordings"][rec]["split"] = None
    for k, v in split.items():
        r = inst["framesets"][k]["recording"]
        cur = man["recordings"][r].get("split")
        man["recordings"][r]["split"] = v if cur in (None, v) else "mixed"
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    with open(os.path.join(args.out, "build_report.json"), "w") as f:
        json.dump({"split_audit": audit, "val_recordings": VAL_RECORDINGS,
                   "val_components_force": VAL_COMPONENTS_FORCE,
                   "alias_components": {c: sorted(v) for c, v in sorted(multi.items())},
                   "n_images": len(path_hash),
                   "n_unique_content": len(groups)}, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
