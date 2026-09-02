"""Build a content-keyed train/val root sourcing frames from general_model ONLY.

    python third_party/jarvis_jax/scripts/build_generalmodel_split.py \
        --gm  .../red_data/general_model \
        --out .../red_data/red_data_3d_v7_generalmodel

READ-ONLY on `general_model`, on `red_data_unified_V3` and on every existing
root. Emits a new tree whose `images/` and `masks/` are symlinks back at
general_model and whose `annotations/` carry a split that is disjoint BY BYTES.

WHY THIS EXISTS, and how it differs from build_content_split.py.
`red_data_3d_v5*` / `v6_contentsplit` merge general_model AND
red_data_unified_V3, which duplicates most of the corpus: 14,182 of V3's
14,973 image contents are byte-identical to a general_model image, and 18,430
of its 20,751 annotations match a general_model annotation to under 5 px. The
user's ruling is to source frames from general_model only.

THE ONE THING general_model FILES DIFFERENTLY. A two-fly courtship capture is
filed as TWO subsets under TWO recording timestamps, one per animal:

    courtship_25_51_female/  ->  2026_06_19_11_09_36
    courtship_25_51_male/    ->  2026_06_18_19_23_03    (same footage, 1 s apart)

so a name-keyed merge sees two single-fly recordings where there is one two-fly
capture. This module merges by CONTENT: one image per md5, both animals'
annotations attached, both on the same side of the split. Measured over
general_model's 19,929 image references -> 19,334 unique contents, that
recovers 499 genuinely two-fly images (bbox centroids 241-392 px apart --
checked, not assumed; two labels on one animal sit under 10 px).

WHAT THIS ROOT DOES NOT HAVE. V3 is the only source of the second-fly labels on
the courtship_V2 capture (V3 recordings 2026_04_07_11_33_33 and
2026_04_08_14_59_45): 2,321 annotations, 2,014 of them the second animal of a
two-fly frame, and 1,225 of them on 791 image contents general_model does not
hold at all. So this root carries 499 two-fly images where the V3-based root
carries 1,673. That is a real, measured cost of the sourcing decision and it is
recorded in build_report.json under `two_fly_deficit_vs_v3_root`, not hidden.

IDENTITY COMES FROM THE SUBSET, NOT FROM POSITION. `build_v5.merge_annotations`
assigns fly_id by position within a camera's annotation list, guarded by
CONTROLLER RULING R15 (a camera whose fly count disagrees with the frameset max
is recorded ABSENT rather than guessed). Here the subset directory names the
animal, so identity is read off the filing and is immune to the chimera failure
R15 exists to prevent. R15's spirit still applies in one place: if the two
subsets' annotations on one camera land on the SAME animal (median visible
keypoint displacement < 50 px -- 21 contents, a labelling defect), neither can
be attributed, so BOTH slots are recorded ABSENT for that camera.

SEX IS FULLY RECOVERABLE WITHOUT V3, and this was verified rather than assumed.
V3's annotations carry a non-"unknown" `sex` for exactly 8 recordings; the
general_model subset names encode the same 8 (`S6male`, `female`,
`courtship_{11_50,25_51,28_34}_{female,male}`) and agree with V3 on 8/8. Every
other recording's sex in the v5 manifest is a HUMAN label applied via
`build_v5.apply_sex_labels`, never V3-derived, and is carried forward here from
`--sex-fallback-manifest` with its provenance recorded per recording in
`manifest.json["recordings"][rec]["sex_source"]`. Note "female" contains
"male": the subset parser must test female first.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from jarvis_jax.data.build_v5 import (                 # noqa: E402
    MIN_CAMS, _canonical_keypoint_names, _keypoint_remap, _pad_keypoints)
from jarvis_jax.data.calib_groups import CAM_GLOB, group_calibrations   # noqa: E402
from jarvis_jax.data.content_index import (            # noqa: E402
    alias_components, content_groups, hash_paths)
from jarvis_jax.data.split_v5 import (                 # noqa: E402
    audit_split, make_split, write_derived)

# Same policy as build_content_split.py so the two roots are comparable.
VAL_RECORDINGS = ["2026_03_18_15_31_22"]                     # courtship_V3, male, group B
VAL_COMPONENTS_FORCE = ["2026_06_18_19_23_03",               # courtship_25_51 pair
                        "2026_05_27_11_56_05"]               # courtship_11_50 pair

# Median visible-keypoint displacement below which two annotations are the SAME
# animal. Task 15 measured this distribution to be bimodal with an empty gap:
# 95% of best matches sit under 5 px, 0.08% land in 5-186 px, the other animal
# sits at >= 186 px. 50 px is inside the gap.
SAME_FLY_PX = 50.0

# Behaviour is not recorded on any general_model annotation (nor on any V3 one
# -- every root so far has `behavior: "unknown"` on every annotation). The
# subset directory is the only behaviour signal in the corpus, so it is written
# to the MANIFEST, which no loader reads for behaviour, and the annotation
# field is left "unknown" exactly as the v5/v6 roots have it -- changing it
# would silently re-weight V5Dataset2D.balanced_weights.
BEHAVIOR = {
    "20_04_female_climbing": "climbing", "S6male": "general",
    "S8_male_R_amp": "amputation", "S9_male_L_amp": "amputation",
    "courtship_11_50_female": "courtship", "courtship_11_50_male": "courtship",
    "courtship_25_51_female": "courtship", "courtship_25_51_male": "courtship",
    "courtship_28_34_female": "courtship", "courtship_28_34_male": "courtship",
    "courtship_V2": "courtship", "courtship_V3": "courtship",
    "courtship_V4": "courtship", "female": "general", "grooming": "grooming",
    "headless_22_50": "headless", "headless_24_04": "headless",
    "headless_24_04_1": "headless", "headless_56_42": "headless",
    "headless_56_42_1": "headless", "wall_frames": "wall",
}


def subset_sex(subset: str) -> str:
    """Sex from the subset directory name. "female" CONTAINS "male", so the
    female test must come first -- getting that backwards labels every female
    subset male, and every sex-conditioned metric in this repo would then read
    fine while measuring the wrong animals."""
    s = subset.lower()
    if "female" in s:
        return "female"
    if "male" in s:
        return "male"
    return "unknown"


def _kp_xy(ann):
    k = ann["keypoints"]
    return [(k[3 * i], k[3 * i + 1], k[3 * i + 2]) for i in range(len(k) // 3)]


def same_fly(a, b) -> float:
    """Median displacement over keypoints visible in BOTH annotations."""
    ka, kb = _kp_xy(a), _kp_xy(b)
    d = [((ka[i][0] - kb[i][0]) ** 2 + (ka[i][1] - kb[i][1]) ** 2) ** 0.5
         for i in range(min(len(ka), len(kb)))
         if ka[i][2] > 0 and kb[i][2] > 0]
    return statistics.median(d) if d else float("inf")


class _DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def load_general_model(gm_root: str):
    """Every (subset, split) annotation file, flattened.

    Returns `records`: one entry per source image, and `subsets`: subset ->
    recording. Image paths are `<subset>/<split>/<file_name>` -- verified
    against the real tree (19,929/19,929 resolve)."""
    records, subsets, ann_paths = [], {}, []
    for sub in sorted(os.listdir(gm_root)):
        cp = os.path.join(gm_root, sub, "calib_params")
        if not os.path.isdir(cp):
            continue
        recs = sorted(os.listdir(cp))
        if len(recs) != 1:
            raise ValueError(f"{sub}: expected one recording in calib_params, got {recs}")
        subsets[sub] = recs[0]
        for split in ("train", "val"):
            p = os.path.join(gm_root, sub, "annotations", f"instances_{split}.json")
            if not os.path.exists(p):
                continue
            ann_paths.append(p)
            blob = json.load(open(p))
            by_img = collections.defaultdict(list)
            for a in blob["annotations"]:
                by_img[a["image_id"]].append(a)
            fs_of_img = {}
            for key, fsv in blob.get("framesets", {}).items():
                for iid in fsv["frames"]:
                    fs_of_img[iid] = (sub, split, key)
            for im in blob["images"]:
                rec, cam, fname = im["file_name"].split("/")
                records.append({
                    "subset": sub, "split": split, "recording": rec, "camera": cam,
                    "frame": int(fname.split("_")[1].split(".")[0]),
                    "path": os.path.join(gm_root, sub, split, im["file_name"]),
                    "width": im["width"], "height": im["height"],
                    "kp_names": blob.get("keypoint_names", []),
                    "skeleton": blob.get("skeleton", []),
                    "anns": sorted(by_img.get(im["id"], []), key=lambda a: a["id"]),
                    "frameset": fs_of_img.get(im["id"]),
                })
    return records, subsets, sorted(set(ann_paths))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hash-cache", default=None)
    ap.add_argument("--sex-fallback-manifest", default=None,
                    help="manifest.json to carry HUMAN sex labels forward from "
                         "for subsets whose name does not encode sex")
    ap.add_argument("--workers", type=int, default=192)
    ap.add_argument("--female-val-frac", type=float, default=0.10)
    ap.add_argument("--guard", type=int, default=200)
    args = ap.parse_args()

    records, subsets, ann_paths = load_general_model(args.gm)
    print(f"{len(subsets)} subsets, {len(records)} source image references")

    # ---- content identity -------------------------------------------------
    cache = {}
    if args.hash_cache and os.path.exists(args.hash_cache):
        cache = json.load(open(args.hash_cache))
    real = {r["path"]: os.path.realpath(r["path"]) for r in records}
    todo = sorted({p for p in real.values() if p not in cache})
    if todo:
        print(f"hashing {len(todo)} files ...", flush=True)
        cache.update(hash_paths(todo, workers=args.workers))
        if args.hash_cache:
            json.dump(cache, open(args.hash_cache, "w"))
    for r in records:
        r["md5"] = cache[real[r["path"]]]
    contents = collections.defaultdict(list)
    for r in records:
        contents[r["md5"]].append(r)
    print(f"{len(records)} image refs -> {len(contents)} unique contents")

    # A content must be one camera of one frame, or the capture atom below is
    # not well defined. Verified 0/0 on the 2026-09-02 tree; asserted so a
    # future ingest cannot break it silently.
    for h, lst in contents.items():
        if len({r["camera"] for r in lst}) > 1 or len({r["frame"] for r in lst}) > 1:
            raise ValueError(f"content {h[:8]} spans several cameras/frames: "
                             f"{[(r['recording'], r['camera'], r['frame']) for r in lst]}")

    # ---- captures: source framesets joined by shared content ---------------
    # NOT joined on frame number. Counters free-run per session, so two
    # recordings can collide on a frame number while sharing zero bytes
    # (measured: min |dframe| = 4 between unrelated recordings). Byte identity
    # is the only evidence used.
    dsu = _DSU()
    for h, lst in contents.items():
        fs = [r["frameset"] for r in lst if r["frameset"]]
        for f in fs:
            dsu.find(f)
        for f in fs[1:]:
            dsu.union(fs[0], f)
    cap_members = collections.defaultdict(set)
    for r in records:
        if r["frameset"]:
            cap_members[dsu.find(r["frameset"])].add(r["md5"])
    print(f"{len(cap_members)} captures")

    # ---- canonical image per content ---------------------------------------
    kp_names = _canonical_keypoint_names(ann_paths)
    skeleton = next((r["skeleton"] for r in records if r["skeleton"]), [])
    canon = {}
    for h, lst in contents.items():
        pick = min(lst, key=lambda r: (r["recording"], r["subset"], r["split"]))
        canon[h] = pick
    order = sorted(contents, key=lambda h: (canon[h]["recording"], canon[h]["camera"],
                                            canon[h]["frame"]))
    img_id = {h: i for i, h in enumerate(order)}
    out_images = [{
        "id": img_id[h], "width": canon[h]["width"], "height": canon[h]["height"],
        "recording": canon[h]["recording"],
        "file_name": f"{canon[h]['recording']}/{canon[h]['camera']}/Frame_{canon[h]['frame']}.jpg",
        "content_md5": h,
    } for h in order]

    # ---- annotations + framesets -------------------------------------------
    out_anns, framesets = [], {}
    next_ann = 0
    n_ambiguous = 0
    ambiguous_log = []
    rec_slots = collections.defaultdict(set)
    for cap in sorted(cap_members, key=lambda c: (canon[min(cap_members[c])]["recording"],
                                                  canon[min(cap_members[c])]["frame"])):
        hs = cap_members[cap]
        cams = sorted({canon[h]["camera"] for h in hs})
        by_cam = {canon[h]["camera"]: h for h in hs}
        cap_rec = min(canon[h]["recording"] for h in hs)
        cap_frame = canon[next(iter(hs))]["frame"]
        slots = sorted({r["subset"] for h in hs for r in contents[h] if r["anns"]})
        if not slots:
            continue
        rec_slots[cap_rec].update(slots)
        frames = [img_id[by_cam[c]] for c in cams]

        # per camera, per slot: the one annotation that slot contributes
        picked = {}
        for c in cams:
            h = by_cam[c]
            for r in contents[h]:
                for a in r["anns"]:
                    if (c, r["subset"]) in picked:
                        raise ValueError(f"{r['subset']} has >1 annotation on "
                                         f"{r['recording']}/{c}/Frame_{r['frame']}")
                    picked[(c, r["subset"])] = (r, a)

        # R15, detected properly: two slots on one camera that land on the SAME
        # animal cannot be attributed to either. Absent for both.
        drop = set()
        for c in cams:
            here = [(s, picked[(c, s)]) for s in slots if (c, s) in picked]
            for i in range(len(here)):
                for j in range(i + 1, len(here)):
                    d = same_fly(here[i][1][1], here[j][1][1])
                    if d < SAME_FLY_PX:
                        drop.add((c, here[i][0]))
                        drop.add((c, here[j][0]))
                        n_ambiguous += 1
                        ambiguous_log.append({
                            "content_md5": by_cam[c], "camera": c,
                            "recording": cap_rec, "frame": cap_frame,
                            "subsets": [here[i][0], here[j][0]],
                            "median_kp_px": round(d, 2)})

        for k, sub in enumerate(slots):
            ann_ids = []
            for c in cams:
                if (c, sub) not in picked or (c, sub) in drop:
                    ann_ids.append(None)
                    continue
                r, a = picked[(c, sub)]
                remap = (None if r["kp_names"] == kp_names
                         else _keypoint_remap(r["kp_names"], kp_names))
                out_anns.append({
                    "id": next_ann, "image_id": img_id[by_cam[c]],
                    "bbox": a["bbox"],
                    "keypoints": (a["keypoints"] if remap is None
                                  else _pad_keypoints(a["keypoints"], remap)),
                    "num_keypoints": a.get("num_keypoints", len(kp_names)),
                    "sex": subset_sex(sub),
                    "behavior": "unknown",     # see BEHAVIOR comment at module top
                    "fly_id": k, "subset": sub,
                    "src_ann_id": a["id"],     # general_model subset id space (masks)
                })
                ann_ids.append(next_ann)
                next_ann += 1
            if sum(a is not None for a in ann_ids) < MIN_CAMS:
                continue
            framesets[f"{cap_rec}/Frame_{cap_frame}/fly{k}"] = {
                "recording": cap_rec, "fly_id": k, "subset": sub,
                "frames": frames, "ann_ids": ann_ids}

    print(f"{len(out_anns)} annotations, {len(framesets)} fly-samples; "
          f"{n_ambiguous} camera/slot pairs recorded ABSENT because two subsets "
          f"labelled the same animal")

    merged = {"keypoint_names": kp_names, "skeleton": skeleton,
              "categories": [{"id": 1, "name": "fly", "num_keypoints": len(kp_names)}],
              "images": out_images, "annotations": out_anns, "framesets": framesets}

    # ---- manifest ----------------------------------------------------------
    fallback = {}
    if args.sex_fallback_manifest:
        fallback = json.load(open(args.sex_fallback_manifest))["recordings"]
    rec_subsets = collections.defaultdict(set)
    for sub, rec in subsets.items():
        rec_subsets[rec].add(sub)
    calib_of = {rec: os.path.join(args.gm, sorted(rec_subsets[rec])[0],
                                  "calib_params", rec)
                for rec in rec_subsets}
    used_recs = sorted({im["recording"] for im in out_images})
    groups = group_calibrations({r: calib_of[r] for r in used_recs})
    calib_out = os.path.join(args.out, "calibrations")
    for rec, grp in groups.items():
        dst = os.path.join(calib_out, grp)
        if os.path.isdir(dst):
            continue
        os.makedirs(dst, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(calib_of[rec], CAM_GLOB))):
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))

    man = {"version": os.path.basename(args.out.rstrip("/")),
           "source_root": args.gm,
           "calib_groups": sorted(set(groups.values())), "recordings": {}}
    for rec in used_recs:
        slots = sorted(rec_slots[rec])
        sexes = {s: subset_sex(s) for s in slots}
        fly_sex = {f"fly{k}": sexes[s] for k, s in enumerate(slots)}
        known = {v for v in sexes.values() if v != "unknown"}
        if known:
            sex, src = (next(iter(known)) if len(known) == 1 else "unknown"), "subset_name"
        else:
            sex = fallback.get(rec, {}).get("sex", "unknown")
            src = "carried_from_manifest" if sex != "unknown" else "unknown"
            fly_sex = {f"fly{k}": sex for k in range(len(slots))}
        entry = {"subset": "+".join(slots), "calib_group": groups[rec],
                 "n_framesets": len({k for k in framesets if k.startswith(f"{rec}/")}),
                 "n_flies": len(slots),
                 "behavior": "+".join(sorted({BEHAVIOR.get(s, "unknown") for s in slots})),
                 "sex": sex, "sex_source": src, "has_masks": False, "split": None}
        if len(set(fly_sex.values())) > 1:
            entry["fly_sex"] = fly_sex
        man["recordings"][rec] = entry

    # ---- media -------------------------------------------------------------
    # 19k symlinks on gpfs are metadata-latency bound, not CPU bound -- the
    # same reason hash_paths threads. Serial with a per-file makedirs+lexists
    # measured ~8 links/s (40 min for this tree); mkdir once per
    # (recording, camera) plus 64 threads brings it to seconds.
    # FileExistsError IS the idempotency guard, so there is no extra stat.
    def _link(job):
        src, dst = job
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass

    img_jobs, mask_jobs = [], []
    for h in order:
        r = canon[h]
        img_jobs.append((os.path.realpath(r["path"]),
                         os.path.join(args.out, "images", r["recording"],
                                      r["camera"], f"Frame_{r['frame']}.jpg")))
        src = os.path.join(args.gm, r["subset"], "sam3_masks", r["split"],
                           r["recording"], r["camera"], f"Frame_{r['frame']}.npz")
        if os.path.exists(src):
            mask_jobs.append((os.path.realpath(src),
                              os.path.join(args.out, "masks", r["recording"],
                                           r["camera"], f"Frame_{r['frame']}.npz")))
            man["recordings"][r["recording"]]["has_masks"] = True
    for jobs in (img_jobs, mask_jobs):
        for d in {os.path.dirname(dst) for _, dst in jobs}:
            os.makedirs(d, exist_ok=True)
        with ThreadPoolExecutor(max_workers=64) as ex:
            list(ex.map(_link, jobs))
    n_masks = len(mask_jobs)
    print(f"linked {len(img_jobs)} images, {n_masks} masks", flush=True)

    # ---- split -------------------------------------------------------------
    path_hash = {os.path.realpath(r["path"]): r["md5"] for r in records}
    path_rec = {os.path.realpath(r["path"]): r["recording"] for r in records}
    aliases = alias_components(path_hash, lambda p: path_rec[p])
    # after the content merge every image already carries its canonical
    # recording, so aliases collapse -- keep them anyway: holding a component
    # together is exactly what makes a partial holdout impossible.
    aliases = {canon_rec: aliases.get(canon_rec, canon_rec) for canon_rec in used_recs}
    image_hashes = {im["id"]: im["content_md5"] for im in out_images}

    split = make_split(merged, man, val_recordings=VAL_RECORDINGS,
                       val_components_force=VAL_COMPONENTS_FORCE,
                       female_val_frac=args.female_val_frac, guard=args.guard,
                       aliases=aliases)
    audit = audit_split(merged, split, guard=args.guard, aliases=aliases,
                        image_hashes=image_hashes)
    print("audit:", json.dumps({k: v for k, v in audit.items()
                                if not k.startswith("leaky")
                                and k != "min_guard_distance"}, indent=2))
    if (audit["cross_fly_leaked_frames"] or audit["cross_capture_leaks"]
            or audit["content_overlap"]):
        raise SystemExit(f"REFUSING to write a leaky split: {audit}")

    write_derived(merged, split, args.out, image_hashes=image_hashes)
    with open(os.path.join(args.out, "annotations", "instances.json"), "w") as f:
        json.dump(merged, f)

    for rec in man["recordings"]:
        man["recordings"][rec]["split"] = None
    for k, v in split.items():
        rec = framesets[k]["recording"]
        cur = man["recordings"][rec].get("split")
        man["recordings"][rec]["split"] = v if cur in (None, v) else "mixed"
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)

    groups_c = content_groups(path_hash)
    flies_on_image = collections.defaultdict(set)
    for a in out_anns:
        flies_on_image[a["image_id"]].add(a["fly_id"])
    n_two = sum(1 for v in flies_on_image.values() if len(v) >= 2)
    with open(os.path.join(args.out, "content_index.json"), "w") as f:
        json.dump({"n_image_refs": len(records), "n_unique_content": len(contents),
                   "duplicate_groups": {h: sorted(v) for h, v in groups_c.items()
                                        if len(v) > 1}}, f, indent=1)
    with open(os.path.join(args.out, "build_report.json"), "w") as f:
        json.dump({
            "source": "general_model ONLY (red_data_unified_V3 deliberately excluded)",
            "n_subsets": len(subsets), "n_recordings": len(used_recs),
            "n_image_refs": len(records), "n_unique_content": len(contents),
            "n_captures": len(cap_members), "n_annotations": len(out_anns),
            "n_fly_samples": len(framesets),
            "two_fly_images": n_two,
            "two_fly_deficit_vs_v3_root": {
                "v3_based_root_two_fly_images": 1673,
                "this_root_two_fly_images": n_two,
                "lost_annotations": 2321,
                "lost_second_fly_annotations": 2014,
                "v3_only_contents": 791,
                "source_recordings": ["2026_04_07_11_33_33", "2026_04_08_14_59_45"],
                "note": "V3 is the only source of the second animal's labels on "
                        "the courtship_V2 capture. Dropping it costs those labels."},
            "ambiguous_same_animal_slots": n_ambiguous,
            "ambiguous_examples": ambiguous_log[:20],
            "split_audit": audit,
            "val_recordings": VAL_RECORDINGS,
            "val_components_force": VAL_COMPONENTS_FORCE,
            "guard": args.guard, "female_val_frac": args.female_val_frac,
        }, f, indent=2)
    print(f"two-fly images in this root: {n_two}  (V3-based root: 1673)")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
