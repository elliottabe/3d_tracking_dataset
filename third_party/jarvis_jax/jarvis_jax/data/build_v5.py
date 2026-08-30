"""Build red_data_3d_v5: one correctly-organized root for 3D training.

Two organizational defects in the old layout drive this module:

1. The SPLIT WAS THE DIRECTORY LAYOUT. general_model/<subset>/ held train/ and
   val/ dirs over the SAME recording, so the split was frame-level and leaked
   (~50% of val framesets had a train frameset within +/-3 frames), and
   regenerating it meant moving image files. Here the split is METADATA
   (split.json, Task 5) and the media tree has no train/ or val/ dir at all.
2. Calibrations were duplicated per subset (21 copies of 3 distinct
   calibrations), hiding the fact that calibration identity is a real
   experimental variable.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field

from jarvis_jax.data.calib_groups import CAM_GLOB, group_calibrations

import glob


@dataclass
class SourceRec:
    recording: str
    subset: str | None
    ann_paths: list[str]
    calib_dir: str
    image_root: str
    mask_root: str | None = None
    n_framesets: int = 0
    n_flies: int = 0
    behavior: str = "unknown"
    sex: str = "unknown"
    extra: dict = field(default_factory=dict)


def discover_sources(general_model_root: str, v3_root: str) -> dict[str, SourceRec]:
    """Enumerate every labeled recording across BOTH source trees.

    general_model contributes 21 recordings; red_data_unified_V3 contributes 5
    more that general_model lacks (2026_03_22, 2026_04_07, 2026_04_08, and the
    two 2026_06_11 recordings). The union is 26 -- V3's 3D training only ever
    used 17.
    """
    srcs: dict[str, SourceRec] = {}
    for subset in sorted(os.listdir(general_model_root)):
        cp = os.path.join(general_model_root, subset, "calib_params")
        if not os.path.isdir(cp):
            continue
        for rec in sorted(os.listdir(cp)):
            anns = sorted(glob.glob(os.path.join(
                general_model_root, subset, "annotations", "instances_*.json")))
            srcs[rec] = SourceRec(
                recording=rec, subset=subset, ann_paths=anns,
                calib_dir=os.path.join(cp, rec),
                image_root=os.path.join(general_model_root, subset),
                mask_root=None)
    v3_calib = os.path.join(v3_root, "calib_params")
    v3_anns = sorted(glob.glob(os.path.join(v3_root, "annotations", "instances_*.json")))
    for rec in sorted(os.listdir(v3_calib)):
        if rec in srcs:
            # Already covered by general_model, but V3 is where its masks live.
            srcs[rec].mask_root = os.path.join(v3_root, "sam3_masks")
            continue
        srcs[rec] = SourceRec(
            recording=rec, subset=None, ann_paths=v3_anns,
            calib_dir=os.path.join(v3_calib, rec),
            image_root=v3_root,
            mask_root=os.path.join(v3_root, "sam3_masks"))
    return srcs


def build_manifest(sources: dict[str, SourceRec], out_root: str) -> dict:
    """Write manifest.json and the deduplicated calibrations/ tree."""
    os.makedirs(out_root, exist_ok=True)
    groups = group_calibrations({r: s.calib_dir for r, s in sources.items()})

    calib_out = os.path.join(out_root, "calibrations")
    for rec, grp in groups.items():
        dst = os.path.join(calib_out, grp)
        if os.path.isdir(dst):
            continue
        os.makedirs(dst, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(sources[rec].calib_dir, CAM_GLOB))):
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))

    man = {
        "version": "red_data_3d_v5",
        "calib_groups": sorted(set(groups.values())),
        "recordings": {
            rec: {
                "subset": s.subset,
                "calib_group": groups[rec],
                "n_framesets": s.n_framesets,
                "n_flies": s.n_flies,
                "behavior": s.behavior,
                "sex": s.sex,                      # filled by the Task 6 sexing pass
                "has_masks": s.mask_root is not None,
                "split": None,                     # filled by Task 5
            }
            for rec, s in sorted(sources.items())
        },
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    return man


def link_media(sources: dict[str, SourceRec], out_root: str, *,
               copy: bool = False) -> None:
    """Materialize images/ and masks/ as ONE flat per-recording tree.

    Symlinks by default: the sources are stable on gscratch and copying costs
    4.4 GB for no benefit. `copy=True` if the tree must be self-contained.

    Known limitations of the idempotency guard (`os.path.lexists`) below --
    documented, not fixed, since current usage is a single build over stable,
    symlink-only sources:

    - A broken symlink (its source later removed) is never healed or even
      reported: `lexists` is True for a dangling link, so a rerun silently
      skips it and the destination stays broken.
    - A stale regular file left behind by an earlier `copy=True` run
      permanently blocks a symlink on a later `copy=False` run over the same
      recording: `lexists` sees the file and skips it, so the destination
      never becomes a symlink even though `copy` changed between runs.
    """
    for rec, s in sources.items():
        for kind, src_root in (("images", s.image_root), ("masks", s.mask_root)):
            if src_root is None:
                continue
            for split in ("train", "val", ""):
                base = os.path.join(src_root, split, rec) if split else os.path.join(src_root, rec)
                if not os.path.isdir(base):
                    continue
                for cam in sorted(os.listdir(base)):
                    src_cam = os.path.join(base, cam)
                    if not os.path.isdir(src_cam):
                        continue
                    dst_cam = os.path.join(out_root, kind, rec, cam)
                    os.makedirs(dst_cam, exist_ok=True)
                    for fn in sorted(os.listdir(src_cam)):
                        dst = os.path.join(dst_cam, fn)
                        if os.path.lexists(dst):
                            continue
                        if copy:
                            shutil.copy2(os.path.join(src_cam, fn), dst)
                        else:
                            os.symlink(os.path.join(src_cam, fn), dst)


import re
from collections import defaultdict

_FRAME_RE = re.compile(r"Frame_(\d+)")
# Minimum cameras a fly must appear in to be usable. 3 matches the robust
# triangulation path, which needs >=3 views for its consensus check.
MIN_CAMS = 3


def merge_annotations(sources: dict[str, SourceRec], out_root: str) -> dict:
    """Merge every source COCO file into one instances.json with per-fly identity.

    CONTROLLER RULING R15 -- how a fly's identity is resolved ACROSS cameras.
    Real courtship data frequently has one camera drop an annotation (occlusion),
    so per-camera annotation counts DISAGREE within a frameset. Three rules were
    considered and two are wrong:

      * `min(counts)` (the original draft) requires all 7 cameras to agree before
        counting a fly at all. It silently discarded 316 framesets with unequal
        counts, and dropped 8 recordings' worth of SINGLE-fly data too wherever
        some camera had zero annotations -- 91 framesets from 2026_01_13_18_47_45
        and 73 from 2026_03_22_12_07_40 among them. `2026_07_30_13_28_99`
        (wall_frames) collapsed to ZERO framesets.
      * `max(counts)` with positional indexing reaches the right TOTAL but creates
        CHIMERAS. Verified counterexample: in
        `2026_04_08_14_59_45/Frame_149677`, Cam2012631 carries a single annotation
        at bbox x=507 while the other six cameras carry two, at x~30-60 (left fly)
        and x~546-575 (right fly). That lone annotation is the RIGHT fly, so
        `cam_anns[0]` would file it as fly0 -- a fly whose views come from two
        different animals, which every downstream metric would report as fine.

    The rule used here: a camera whose annotation count EQUALS the frameset max
    contributes `cam_anns[k]` -- position-based identity is verified safe in that
    regime (two-fly framesets triangulate at 0.39 px, identical to single-fly
    controls). A camera that disagrees on the count cannot be assigned by
    position, so it is recorded as ABSENT (`None`) for every fly in that
    frameset. A fly is emitted only if at least MIN_CAMS cameras remain.

    Consumers must therefore treat `ann_ids` as possibly containing `None`.

    THE FIX: v3_3d.py mapped image_id -> FIRST annotation, commented "shouldn't
    happen for single-fly". It does happen -- on 2026_04_08 (214 framesets),
    2026_04_07 (30), and the two 2026_06_11 recordings (20) -- silently dropping
    264 complete framesets / 1,673 annotations. A chimera hypothesis (fly A in
    one camera mixed with fly B in another) was TESTED AND REFUTED: two-fly
    framesets triangulate at 0.39 px, identical to single-fly controls, so
    annotation order IS consistent across cameras. Ordering by annotation id
    within an image therefore assigns fly_id consistently.
    """
    out_images: list[dict] = []
    out_anns: list[dict] = []
    framesets: dict[str, dict] = {}
    kp_names: list[str] = []
    skeleton: list = []
    next_img, next_ann = 0, 0

    for rec, s in sorted(sources.items()):
        seen_paths: set[str] = set()
        img_key_to_id: dict[tuple[str, str], int] = {}
        anns_by_img: dict[int, list[dict]] = defaultdict(list)

        for ann_path in s.ann_paths:
            with open(ann_path) as f:
                blob = json.load(f)
            kp_names = kp_names or blob.get("keypoint_names", [])
            skeleton = skeleton or blob.get("skeleton", [])
            src_img = {i["id"]: i for i in blob["images"]}
            src_anns = defaultdict(list)
            for a in blob["annotations"]:
                src_anns[a["image_id"]].append(a)

            for key, fsv in blob.get("framesets", {}).items():
                if fsv.get("datasetName") != rec:
                    continue
                frame_no = _FRAME_RE.search(key)
                if frame_no is None:
                    continue
                frame = frame_no.group(1)
                # Deduplicate across the source's train/ and val/ files.
                if (rec, frame) in seen_paths:
                    continue
                seen_paths.add((rec, frame))

                per_cam: list[list[dict]] = []
                cam_img_ids: list[int] = []
                for iid in fsv["frames"]:
                    info = src_img.get(iid)
                    if info is None:
                        break
                    parts = info["file_name"].split("/")
                    cam = parts[-2]
                    ikey = (cam, frame)
                    if ikey not in img_key_to_id:
                        img_key_to_id[ikey] = next_img
                        out_images.append({
                            "id": next_img, "width": info["width"],
                            "height": info["height"], "recording": rec,
                            "file_name": f"{rec}/{cam}/Frame_{frame}.jpg"})
                        next_img += 1
                    cam_img_ids.append(img_key_to_id[ikey])
                    # Deterministic order => consistent fly_id across cameras.
                    per_cam.append(sorted(src_anns.get(iid, []), key=lambda a: a["id"]))

                if len(cam_img_ids) != len(fsv["frames"]) or not per_cam:
                    continue
                # CONTROLLER RULING R15 -- see the block comment above for why
                # this is NOT `min` (silently drops data) and NOT `max`
                # (creates chimeras).
                n_flies = max(len(v) for v in per_cam)
                if n_flies == 0:
                    continue

                for k in range(n_flies):
                    ann_ids = []
                    for img_id, cam_anns in zip(cam_img_ids, per_cam):
                        if len(cam_anns) != n_flies:
                            # This camera disagrees on how many flies it sees,
                            # so positional index k is NOT safe here. Record the
                            # camera as ABSENT for this fly rather than guessing.
                            ann_ids.append(None)
                            continue
                        a = cam_anns[k]
                        out_anns.append({
                            "id": next_ann, "image_id": img_id,
                            "bbox": a["bbox"], "keypoints": a["keypoints"],
                            "num_keypoints": a.get("num_keypoints", 50),
                            "sex": a.get("sex", "unknown"),
                            "behavior": a.get("behavior", "unknown"),
                            "fly_id": k,
                            "src_ann_id": a["id"],   # masks are keyed by this
                        })
                        ann_ids.append(next_ann)
                        next_ann += 1
                    if sum(a is not None for a in ann_ids) < MIN_CAMS:
                        continue   # too few views to triangulate; drop the fly
                    framesets[f"{rec}/Frame_{frame}/fly{k}"] = {
                        "recording": rec, "fly_id": k,
                        "frames": cam_img_ids, "ann_ids": ann_ids}

        s.n_framesets = len({k for k in framesets if k.startswith(f"{rec}/")})
        s.n_flies = sum(1 for k in framesets if k.startswith(f"{rec}/"))

    merged = {
        "keypoint_names": kp_names, "skeleton": skeleton,
        "categories": [{"id": 1, "name": "fly", "num_keypoints": 50}],
        "images": out_images, "annotations": out_anns, "framesets": framesets,
    }
    ann_dir = os.path.join(out_root, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    with open(os.path.join(ann_dir, "instances.json"), "w") as f:
        json.dump(merged, f)
    return merged


_VALID_SEX = {"male", "female", "unknown"}


def apply_sex_labels(v5_root: str, labels: dict[str, str]) -> dict:
    """Write manually-determined sex into manifest.json.

    Keys are recording names, or '<recording>/fly<k>' for the four two-fly
    recordings where the two slots differ.
    """
    path = os.path.join(v5_root, "manifest.json")
    with open(path) as f:
        man = json.load(f)
    for key, sex in labels.items():
        if sex not in _VALID_SEX:
            raise ValueError(f"sex must be one of {sorted(_VALID_SEX)}, got {sex!r}")
        rec = key.split("/")[0]
        if rec not in man["recordings"]:
            raise KeyError(f"{rec!r} not in manifest")
        if "/" in key:
            man["recordings"][rec].setdefault("fly_sex", {})[key.split("/")[1]] = sex
        else:
            man["recordings"][rec]["sex"] = sex
    with open(path, "w") as f:
        json.dump(man, f, indent=2)
    return man
