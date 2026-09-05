#!/usr/bin/env python3
"""Stage the 2026-09-02 raw label export over general_model, one subset per recording.

WHY. `courtship_label_2026_09_02` holds NINE recordings. The v10 root sourced
only two of them (the wall recording and `2025_10_20_13_20_04_male`); the other
seven were still read from `general_model`, whose annotations are incomplete --
some framesets are labelled on fewer than 7 cameras. Measured, general_model vs
export, complete-7-camera framesets:

    courtship_11_50_female  13/15 -> 15/15      courtship_25_51_male  18/22 -> 22/22
    courtship_11_50_male    15/15 -> 15/15      courtship_28_34_female 19/46 -> 46/46
    courtship_25_51_female  19/23 -> 22/23      courtship_28_34_male   43/46 -> 46/46
    20_04_female_climbing   15/15 -> 15/15

+62 annotations overall (1,205 -> 1,267), every one of them a camera view that
was labelled in the export and missing in general_model. `courtship_28_34_female`
is the worst and matters most: it is the courtship FEMALE, the fly this pipeline
fails on and the class balanced sampling lifts 3.2x.

UNION, NOT REPLACEMENT. general_model has one frameset the export does not:
frame 431686 of the 25_51 pair, complete on 7 cameras for BOTH flies (14
annotations). Replacing outright would silently delete it, so anything
general_model has and the export lacks is backfilled.

IMAGES. Eight of the nine export recordings ship annotations only -- no jpgs --
so images are symlinked back into general_model via the converter's
`--link-images-from`. Verified beforehand: every camera view the export labels
has an image (0 missing), and where two general_model subsets hold the same
frame (the two flies of one capture are filed under two FABRICATED recording
ids a minute apart, e.g. 2026_05_27_11_56_05 and _11_57_05) the two copies are
byte-identical.

CALIBRATION COMES FROM general_model, NOT FROM THE EXPORT (2026-09-03). All six
`2026_04_02_*` export dirs ship ONE byte-identical `calibration/` -- the
12_11_50 one. For 15_25_51 and 17_28_34 the exported 3D reprojects 4-15 px off
the exported 2D through it, and 0.002 px through the calibration the
same-named general_model subset carries (checked for all seven recordings
here; 11_50 and 20_04 agree either way). The 3D trainer triangulates the 2D
labels with the shipped calibration, so the v12 root built from the export's
calibration trained on wrong 3D for those two recordings. Each conversion is
therefore given `--calib-from <gm>/<subset>/calib_params/<rec>` and the
converter PROVES the shipped calibration against the recording's own labels
before writing it (see red3d2jarvis.py).

RECORDING IDS ARE THE TRUE CAPTURE TIMESTAMPS, taken from the export directory
names, not general_model's fabricated ones. That is not cosmetic: filing both
flies of a capture under ONE id makes them one capture to the split builder by
construction, which is the leak CAPTURE_GROUPS had to be declared to prevent.

Run:
    python scripts/data_prep/stage_export_corrections.py \
        --raw   /gscratch/.../red_data/courtship_label_2026_09_02 \
        --gm    /gscratch/.../red_data/general_model \
        --stage /gscratch/.../red_data/courtship_labels_2026_09_02 \
        --farm  /gscratch/.../red_data/general_model_2026_09_02_v11
"""
import argparse
import collections
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONVERTER = HERE / "red3d2jarvis.py"

# export dir -> (staged subset name, TRUE capture id, sex, general_model subsets
#                whose image trees hold this recording's frames)
RECORDINGS = [
    ("2026_04_02_12_11_50_female", "courtship_11_50_female", "2026_04_02_12_11_50",
     "female", ["courtship_11_50_female", "courtship_11_50_male"]),
    ("2026_04_02_12_11_50_male", "courtship_11_50_male", "2026_04_02_12_11_50",
     "male", ["courtship_11_50_female", "courtship_11_50_male"]),
    ("2026_04_02_15_25_51_female", "courtship_25_51_female", "2026_04_02_15_25_51",
     "female", ["courtship_25_51_female", "courtship_25_51_male"]),
    ("2026_04_02_15_25_51_male", "courtship_25_51_male", "2026_04_02_15_25_51",
     "male", ["courtship_25_51_female", "courtship_25_51_male"]),
    ("2026_04_02_17_28_34_female", "courtship_28_34_female", "2026_04_02_17_28_34",
     "female", ["courtship_28_34_female", "courtship_28_34_male"]),
    ("2026_04_02_17_28_34_male", "courtship_28_34_male", "2026_04_02_17_28_34",
     "male", ["courtship_28_34_female", "courtship_28_34_male"]),
    ("2025_10_20_13_20_04_female_climbing", "20_04_female_climbing",
     "2025_10_20_13_20_04", "female", ["20_04_female_climbing"]),
]

# Already staged for v10, reused as-is (their export dirs are the other two).
ALREADY_STAGED = ["courtship_20_04_male", "wall_frames_15_06_46_male"]

CITATION = ("raw export courtship_label_2026_09_02/{exp}; supersedes "
            "general_model/{subs} whose annotations were incomplete "
            "(missing camera views on labelled framesets)")


def gm_index(gm_root, subset):
    """{(cam, frame): (file_name, [annotation, ...])} for one general_model subset."""
    out = collections.defaultdict(lambda: [None, []])
    for split in ("train", "val"):
        p = Path(gm_root) / subset / "annotations" / f"instances_{split}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        id2im = {im["id"]: im for im in d["images"]}
        for a in d["annotations"]:
            im = id2im[a["image_id"]]
            parts = im["file_name"].replace("\\", "/").split("/")
            cam = parts[1]
            frame = int(os.path.splitext(parts[-1])[0].split("_")[-1])
            rec = ent = out[(cam, frame)]
            ent[0] = (str(Path(gm_root) / subset / split / im["file_name"]), im)
            ent[1].append(a)
    return out


def export_frames(raw_root, exp):
    d = Path(raw_root) / exp
    frames = collections.defaultdict(set)
    for c in sorted(f for f in os.listdir(d) if f.startswith("Cam") and f.endswith(".csv")):
        with open(d / c) as f:
            for r in csv.reader(f):
                if r and r[0].strip().lstrip("-").isdigit():
                    frames[int(r[0])].add(c[:-4])
    return frames


def assert_same_keypoint_order(staged_json, gm_idx, rec_id, tol_px=2.0):
    """PROVE the converted subset and general_model use the same 50-slot order.

    The backfill copies general_model keypoint arrays verbatim into a file the
    converter wrote. If the two orders differed, the copied frames would carry
    scrambled anatomy that every downstream metric would rate as fine -- the
    exact failure this repo has already paid for twice. So: for framesets the
    two sources SHARE, the coordinates must agree. Raises rather than warns.
    """
    d = json.loads(Path(staged_json).read_text())
    id2f = {im["id"]: im["file_name"] for im in d["images"]}
    checked = worst = 0
    for a in d["annotations"]:
        parts = id2f[a["image_id"]].split("/")
        cam = parts[1]
        frame = int(os.path.splitext(parts[-1])[0].split("_")[-1])
        ent = gm_idx.get((cam, frame))
        if not ent or not ent[1]:
            continue
        conv = a["keypoints"]
        for cand in ent[1]:
            ref = cand["keypoints"]
            dmax = 0.0
            for i in range(0, min(len(conv), len(ref)), 3):
                if conv[i + 2] > 0 and ref[i + 2] > 0:
                    dmax = max(dmax, abs(conv[i] - ref[i]), abs(conv[i + 1] - ref[i + 1]))
            if dmax <= tol_px:
                checked += 1
                worst = max(worst, dmax)
                break
    if checked == 0:
        raise SystemExit(
            f"FATAL: {staged_json}: no frameset matched general_model within "
            f"{tol_px} px on ANY keypoint order -- cannot prove the two sources "
            f"share a keypoint order, so backfilling would scramble anatomy.")
    return checked, worst


def backfill(staged_dir, rec_id, gm_idxs, exp_frames):
    """Add general_model framesets the export lacks; symlink their images.

    MUST also write the top-level `framesets` entry. That dict -- not the
    images list -- is what `jarvis_jax/data/build_v5.py` iterates
    (`for key, fsv in blob.get("framesets", {}).items()`), so images and
    annotations added without it are INVISIBLE to the builder: it materialises
    the jpgs into the root and then ships neither annotation nor split entry.
    The first version of this function did exactly that and silently lost
    frame 431686 of the 25_51 pair -- 14 annotations that looked present in
    the staged subset and were absent from the built root. Asserted below.
    """
    ann_p = Path(staged_dir) / "annotations" / "instances_train.json"
    d = json.loads(ann_p.read_text())
    have = {(im["file_name"].split("/")[1],
             int(os.path.splitext(im["file_name"].split("/")[-1])[0].split("_")[-1]))
            for im in d["images"]}
    next_img = max((im["id"] for im in d["images"]), default=-1) + 1
    next_ann = max((a["id"] for a in d["annotations"]), default=-1) + 1
    added_imgs = added_anns = 0
    seen = set()
    per_frame_ids = collections.defaultdict(list)
    for gm_idx in gm_idxs:
        for (cam, frame), (src, anns) in sorted(gm_idx.items()):
            if (cam, frame) in have or (cam, frame) in seen or frame in exp_frames:
                continue
            seen.add((cam, frame))
            src_path, src_im = src
            rel = f"{rec_id}/{cam}/Frame_{frame}.jpg"
            dst = Path(staged_dir) / "train" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.symlink(os.path.realpath(src_path), dst)
            d["images"].append({"coco_url": "", "date_captured": "", "file_name": rel,
                                "flickr_url": "", "height": src_im["height"],
                                "id": next_img, "width": src_im["width"]})
            for a in anns:
                na = dict(a)
                na["id"] = next_ann; na["image_id"] = next_img
                d["annotations"].append(na)
                next_ann += 1; added_anns += 1
            per_frame_ids[frame].append(next_img)
            next_img += 1; added_imgs += 1

    for frame, ids in per_frame_ids.items():
        d.setdefault("framesets", {})[f"{rec_id}/Frame_{frame}"] = {
            "datasetName": rec_id, "frames": sorted(ids)}

    # Every image must belong to a frameset, or the builder will not see it.
    in_fs = {i for fsv in d.get("framesets", {}).values() for i in fsv["frames"]}
    orphans = [im["file_name"] for im in d["images"] if im["id"] not in in_fs]
    if orphans:
        raise SystemExit(
            f"FATAL: {staged_dir}: {len(orphans)} images belong to no frameset "
            f"entry, so build_v5 would silently drop them: {orphans[:5]}")

    if added_imgs:
        ann_p.write_text(json.dumps(d))
    return added_imgs, added_anns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--gm", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--farm", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    Path(a.stage).mkdir(parents=True, exist_ok=True)
    report = []
    for exp, subset, rec_id, sex, gm_subs in RECORDINGS:
        out = Path(a.stage) / subset
        # the SAME-NAMED general_model subset's calibration (see module doc)
        cal_root = Path(a.gm) / subset / "calib_params"
        cal_dirs = sorted(p for p in cal_root.iterdir() if p.is_dir()) \
            if cal_root.is_dir() else []
        if len(cal_dirs) != 1:
            raise SystemExit(f"FATAL: {cal_root}: expected exactly one "
                             f"recording dir of Cam*.yaml, found {cal_dirs}")
        cmd = [sys.executable, str(CONVERTER),
               "-i", str(Path(a.raw) / exp), "-o", str(out),
               "--subset-name", subset, "--recording-id", rec_id,
               "--sex", sex, "--sex-source", "dirname",
               "--citation", CITATION.format(exp=exp, subs="+".join(gm_subs)),
               "--calib-from", str(cal_dirs[0]),
               "--link-images-from"] + [str(Path(a.gm) / s) for s in gm_subs]
        print("\n$", " ".join(cmd), flush=True)
        if a.dry_run:
            continue
        if out.exists():
            subprocess.run(["rm", "-rf", str(out)], check=True)
        subprocess.run(cmd, check=True)

        idxs = [gm_index(a.gm, s) for s in gm_subs]
        # Reference the SAME-NAMED general_model subset, not gm_subs[0]: for a
        # male conversion gm_subs[0] is the female subset, i.e. a different
        # animal, and the proof would (correctly) fail on every frameset.
        ref = gm_index(a.gm, subset) if subset in gm_subs else idxs[0]
        ann_p = out / "annotations" / "instances_train.json"
        checked, worst = assert_same_keypoint_order(ann_p, ref, rec_id)
        ef = set(export_frames(a.raw, exp))
        ai, aa = backfill(out, rec_id, idxs, ef)
        n = json.loads(ann_p.read_text())
        report.append((subset, len(n["images"]), len(n["annotations"]), checked, worst, ai, aa))
        print(f"  [{subset}] kp-order proven on {checked} shared annotations "
              f"(worst {worst:.2f} px); backfilled +{ai} images / +{aa} annotations")

    if a.dry_run:
        return
    farm = Path(a.farm)
    farm.mkdir(parents=True, exist_ok=True)
    staged_names = {s for _, s, _, _, _ in RECORDINGS} | set(ALREADY_STAGED)
    replaced = {s for _, _, _, _, subs in RECORDINGS for s in subs}
    replaced |= {"courtship_V2", "courtship_V3", "courtship_V4"}   # -> courtship_20_04_male
    n_links = 0
    for name in sorted(staged_names):
        link = farm / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(str(Path(a.stage) / name), link); n_links += 1
    for name in sorted(os.listdir(a.gm)):
        if name in replaced or name in staged_names:
            continue
        if not (Path(a.gm) / name).is_dir():
            continue
        link = farm / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(str(Path(a.gm) / name), link); n_links += 1

    print(f"\n=== staged ===")
    print(f"{'subset':<26}{'images':>8}{'anns':>7}{'kp-proof':>10}{'worstpx':>9}{'+img':>6}{'+ann':>6}")
    for r in report:
        print(f"{r[0]:<26}{r[1]:>8}{r[2]:>7}{r[3]:>10}{r[4]:>9.2f}{r[5]:>6}{r[6]:>6}")
    print(f"\nfarm {a.farm}: {n_links} subsets "
          f"({len(staged_names)} from the export, {n_links - len(staged_names)} from general_model)")


if __name__ == "__main__":
    main()
