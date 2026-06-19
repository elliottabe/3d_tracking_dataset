"""Compile a properly-handled unified JARVIS dataset from general_model subsets
(+ unified_v2-only extras), with a RECORDING-LEVEL train/val split and per-
annotation sex/behavior tags.

Why: the existing red_data_unified_V2 used a random frameset-level val split, so
val shares recordings with train (optimistic), and most courtship-FEMALE labels
were never included. This tool fixes both: it ingests every sex-labeled subset,
tags each annotation with sex+behavior, and splits by RECORDING so no recording
is in both train and val.

Sources:
  - every <general_model>/<subset>/ (sex/behavior parsed from subset name)
  - the recordings present ONLY in <unified_v2> (tagged sex/behavior=unknown)

Output (COCO/JARVIS):
  <out>/{train,val}/<rec>/<cam>/Frame_<N>.jpg            (file copies)
  <out>/annotations/instances_{train,val}.json          (+ sex/behavior per ann)
  <out>/calib_params/<rec>/*.yaml
  <out>/build_report.json

Run dry first:
  python build_unified_dataset.py --dry_run
"""
import argparse
import json
import os
import shutil
from collections import defaultdict, Counter
from pathlib import Path

GM_DEFAULT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/general_model"
UNI_DEFAULT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V2"
OUT_DEFAULT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"


def parse_sex_behavior(subset_name):
    """Map a general_model subset directory name -> (sex, behavior)."""
    n = subset_name.lower()
    if n.endswith("_female") or n == "female":
        sex = "female"
    elif n.endswith("_male") or n == "s6male":
        sex = "male"
    else:
        sex = "unknown"
    if "courtship" in n:
        behavior = "courtship"
    elif "groom" in n:
        behavior = "grooming"
    elif n in ("female", "s6male"):
        behavior = "general"
    else:
        behavior = "unknown"
    return sex, behavior


def rec_of(file_name):
    return file_name.split("/")[0]


def cam_of(file_name):
    return file_name.split("/")[1]


def frame_of(file_name):
    return int(file_name.split("/")[-1].replace("Frame_", "").replace(".jpg", ""))


def load_source(root, splits, sex, behavior, recording_filter=None):
    """Yield per-image entries from a source's split JSONs.

    recording_filter: if set, keep only images whose recording is in this set.
    Returns (entries, meta) where entries is a list of dicts and meta carries
    keypoint_names/skeleton/categories/calibrations seen.
    """
    entries = []
    meta = {}
    for split in splits:
        jpath = Path(root) / "annotations" / f"instances_{split}.json"
        if not jpath.exists():
            continue
        blob = json.load(open(jpath))
        meta.setdefault("keypoint_names", blob.get("keypoint_names"))
        meta.setdefault("skeleton", blob.get("skeleton"))
        meta.setdefault("categories", blob.get("categories"))
        meta.setdefault("calibrations", {})
        meta["calibrations"].update(blob.get("calibrations", {}))
        anns_by_img = defaultdict(list)
        for a in blob["annotations"]:
            anns_by_img[a["image_id"]].append(a)
        for im in blob["images"]:
            fn = im["file_name"]
            rec = rec_of(fn)
            if recording_filter is not None and rec not in recording_filter:
                continue
            entries.append({
                "file_name": fn,
                "recording": rec,
                "cam": cam_of(fn),
                "frame": frame_of(fn),
                "abs_path": Path(root) / split / fn,
                "width": im["width"],
                "height": im["height"],
                "annotations": anns_by_img.get(im["id"], []),
                "sex": sex,
                "behavior": behavior,
                "src_root": str(root),
            })
    return entries, meta


def choose_val_recordings(rec_info, val_frac, min_group_for_holdout=2):
    """Pick val recordings: per (sex,behavior) group with >=2 recordings, hold
    out EXACTLY ONE recording -- the one whose frame count is closest to
    val_frac of the group total (so val is proportional, not the largest rec,
    and the group keeps the bulk for training). Single-recording groups stay in
    train (can't split one recording without leakage). Returns val rec set."""
    groups = defaultdict(list)
    for rec, info in rec_info.items():
        groups[(info["sex"], info["behavior"])].append(rec)
    val = set()
    for key, recs in groups.items():
        if len(recs) < min_group_for_holdout:
            continue
        total = sum(rec_info[r]["n_imgs"] for r in recs)
        target = val_frac * total
        pick = min(recs, key=lambda r: abs(rec_info[r]["n_imgs"] - target))
        val.add(pick)
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--general_model", default=GM_DEFAULT)
    ap.add_argument("--unified_v2", default=UNI_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--val_frac", type=float, default=0.2,
                    help="target fraction of frames per category held out for val")
    ap.add_argument("--val_recordings", nargs="*", default=None,
                    help="explicit recordings to force into val (overrides auto)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    GM, UNI = args.general_model, args.unified_v2

    # ---- discover general_model recordings to know what unified_v2 adds ----
    gm_subsets = sorted(d for d in os.listdir(GM)
                        if os.path.isdir(os.path.join(GM, d, "annotations")))
    all_entries = []
    metas = []
    gm_recordings = set()
    for sub in gm_subsets:
        sex, beh = parse_sex_behavior(sub)
        ents, meta = load_source(os.path.join(GM, sub), ("train", "val"), sex, beh)
        for e in ents:
            gm_recordings.add(e["recording"])
        all_entries += ents
        metas.append(meta)

    # ---- unified_v2 extras: recordings present only in unified_v2 ----
    uni_ents_all, uni_meta = load_source(UNI, ("train", "val"), "unknown", "unknown")
    uni_recordings = set(e["recording"] for e in uni_ents_all)
    extra_recs = uni_recordings - gm_recordings
    uni_extra = [e for e in uni_ents_all if e["recording"] in extra_recs]
    all_entries += uni_extra
    metas.append(uni_meta)

    # ---- consistency of keypoint_names/skeleton ----
    kp = metas[0]["keypoint_names"]
    sk = metas[0]["skeleton"]
    for m in metas:
        if m.get("keypoint_names") and m["keypoint_names"] != kp:
            raise SystemExit("keypoint_names differ across sources; abort.")
    categories = metas[0]["categories"]
    calibrations_all = {}
    for m in metas:
        calibrations_all.update(m.get("calibrations", {}))

    # ---- dedup by file_name (recordings are disjoint across subsets; this
    #      collapses any train/val duplication within a subset). Keep the
    #      entry with the most annotations / a known sex. ----
    by_fn = {}
    for e in all_entries:
        k = e["file_name"]
        if k not in by_fn:
            by_fn[k] = e
        else:
            old = by_fn[k]
            better = (len(e["annotations"]), e["sex"] != "unknown") > \
                     (len(old["annotations"]), old["sex"] != "unknown")
            if better:
                by_fn[k] = e
    entries = list(by_fn.values())

    # ---- group into framesets per (recording, frame); keep complete only ----
    fs = defaultdict(dict)  # (rec, frame) -> {cam: entry}
    rec_cams = defaultdict(set)
    for e in entries:
        fs[(e["recording"], e["frame"])][e["cam"]] = e
        rec_cams[e["recording"]].add(e["cam"])
    complete_fs = {}
    dropped_incomplete = 0
    for (rec, fr), cams in fs.items():
        if set(cams) == rec_cams[rec]:
            complete_fs[(rec, fr)] = cams
        else:
            dropped_incomplete += 1

    # ---- per-recording stats + sex/behavior (from its entries) ----
    rec_info = {}
    for (rec, fr), cams in complete_fs.items():
        any_e = next(iter(cams.values()))
        info = rec_info.setdefault(rec, {"sex": any_e["sex"],
                                         "behavior": any_e["behavior"],
                                         "n_fs": 0, "n_imgs": 0, "n_anns": 0})
        info["n_fs"] += 1
        info["n_imgs"] += len(cams)
        info["n_anns"] += sum(len(e["annotations"]) for e in cams.values())

    # ---- choose val recordings (recording-level holdout) ----
    if args.val_recordings:
        val_recs = set(args.val_recordings)
    else:
        val_recs = choose_val_recordings(rec_info, args.val_frac)
    split_of = {r: ("val" if r in val_recs else "train") for r in rec_info}

    # ---- report ----
    print(f"\n{'='*78}\nRecording-level split plan  (out={args.out})\n{'='*78}")
    print(f"{'recording':<24}{'sex':<9}{'behavior':<11}{'split':<7}{'framesets':>10}{'anns':>8}")
    for rec in sorted(rec_info, key=lambda r: (rec_info[r]['behavior'], rec_info[r]['sex'], r)):
        i = rec_info[rec]
        print(f"{rec:<24}{i['sex']:<9}{i['behavior']:<11}{split_of[rec]:<7}"
              f"{i['n_fs']:>10}{i['n_anns']:>8}")

    print(f"\n{'='*78}\nPer-split x (sex,behavior) annotation counts\n{'='*78}")
    agg = defaultdict(lambda: Counter())
    for rec, i in rec_info.items():
        agg[split_of[rec]][(i["sex"], i["behavior"])] += i["n_anns"]
    for sp in ("train", "val"):
        print(f"-- {sp} --")
        for k in sorted(agg[sp]):
            print(f"   {k[0]:<8} {k[1]:<10} : {agg[sp][k]:>6} anns")
        print(f"   TOTAL anns: {sum(agg[sp].values())}")

    # leakage assertion
    leak = val_recs & (set(rec_info) - val_recs - val_recs)  # always empty by construction
    assert not (val_recs & set(r for r in rec_info if split_of[r] == "train")), \
        "recording in both splits!"
    print(f"\nframesets: {len(complete_fs)} complete  (dropped {dropped_incomplete} incomplete)")
    print(f"female-courtship in val: "
          f"{[r for r in val_recs if rec_info[r]['sex']=='female' and rec_info[r]['behavior']=='courtship']}")

    if args.dry_run:
        print("\n[dry-run] no files written. Re-run without --dry_run to build.")
        return

    # ---- write dataset ----
    out = Path(args.out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    (out / "calib_params").mkdir(exist_ok=True)

    split_entries = {"train": [], "val": []}
    for (rec, fr), cams in complete_fs.items():
        sp = split_of[rec]
        for cam, e in cams.items():
            split_entries[sp].append(((rec, fr), cam, e))

    # copy calibrations for used recordings
    used_recs = set(rec_info)
    for rec in used_recs:
        # find a source root that has this recording's calib
        src = None
        for base in [GM] + [os.path.join(GM, s) for s in gm_subsets] + [UNI]:
            cand = Path(base) / "calib_params" / rec
            if cand.is_dir():
                src = cand
                break
        if src is None:
            print(f"  WARN: no calib_params for {rec}")
            continue
        dst = out / "calib_params" / rec
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    for sp in ("train", "val"):
        images_out, anns_out, framesets_out = [], [], {}
        next_img, next_ann = 0, 0
        # group by frameset for ordering + framesets emission
        per_fs = defaultdict(list)
        for fskey, cam, e in split_entries[sp]:
            per_fs[fskey].append((cam, e))
        for fskey in sorted(per_fs):
            rec, fr = fskey
            cam_ids = []
            for cam, e in sorted(per_fs[fskey]):
                fn = f"{rec}/{cam}/Frame_{fr}.jpg"
                dst = out / sp / fn
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(e["abs_path"].resolve(), dst)
                images_out.append({"file_name": fn, "id": next_img,
                                   "width": e["width"], "height": e["height"],
                                   "coco_url": "", "flickr_url": "",
                                   "date_captured": ""})
                for a in e["annotations"]:
                    na = dict(a)
                    na["id"] = next_ann
                    na["image_id"] = next_img
                    na["sex"] = e["sex"]
                    na["behavior"] = e["behavior"]
                    anns_out.append(na)
                    next_ann += 1
                cam_ids.append(next_img)
                next_img += 1
            framesets_out[f"{rec}/Frame_{fr}"] = {"datasetName": rec,
                                                  "frames": cam_ids}
        calibs = {r: {c: f"calib_params/{r}/{c}.yaml"
                      for c in rec_cams[r]
                      if (out / "calib_params" / r / f"{c}.yaml").exists()}
                  for r in used_recs if split_of[r] == sp}
        blob = {"keypoint_names": kp, "skeleton": sk, "categories": categories,
                "calibrations": calibs, "images": images_out,
                "annotations": anns_out, "framesets": framesets_out}
        with open(out / "annotations" / f"instances_{sp}.json", "w") as f:
            json.dump(blob, f)
        print(f"  wrote {sp}: {len(images_out)} imgs, {len(anns_out)} anns")

    report = {"sources": {"general_model": GM, "unified_v2_extras": sorted(extra_recs)},
              "val_recordings": sorted(val_recs),
              "split_of": split_of,
              "rec_info": rec_info,
              "n_framesets": len(complete_fs),
              "dropped_incomplete": dropped_incomplete}
    json.dump(report, open(out / "build_report.json", "w"), indent=2)
    print(f"  wrote {out}/build_report.json\ndone.")


if __name__ == "__main__":
    main()
