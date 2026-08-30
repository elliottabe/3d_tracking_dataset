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
from collections.abc import Iterator
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

    For the 12 recordings general_model AND V3 both cover, V3's annotations
    -- not general_model's -- are used, for two reasons verified against the
    real tree (identical frameset counts for all 12; same frames/keypoints,
    just re-numbered ids):
      1. general_model annotations never carry `sex` at all; V3's do.
      2. The SAM3 masks for these 12 (borrowed from V3, `mask_root` below)
         are keyed by V3's own annotation ids. Sourcing frame/annotation
         data from general_model instead put `src_ann_id` in a DIFFERENT id
         space than the mask files' `ann_ids` -- 0% mask hits for exactly
         these 12 recordings, verified as the dominant cause of an
         overall ~30% hit rate across all 17 V3-mask recordings (the other
         5, V3-only, already matched at ~100%: (5*100 + 12*0)/17 ~= 29%).
    Only image_root/calib_dir stay general_model's -- that's where the
    actual jpgs and this rig's calibration files live; nothing there
    disagrees between the two trees.
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
            # Already covered by general_model, but V3 is where its masks
            # live AND (see docstring) the annotation source of record.
            srcs[rec].ann_paths = v3_anns
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


def build_recording_sex_map(ann_paths: list[str]) -> tuple[dict[str, str], list[str]]:
    """Recover a per-recording sex label from raw COCO annotation files.

    BUG THIS FIXES: `general_model/<subset>/annotations/instances_*.json` has
    no `sex` field on its annotations at all (keys are just `bbox,
    category_id, id, image_id, iscrowd, keypoints, num_keypoints,
    segmentation`) -- only `red_data_unified_V3/annotations/` records it.
    `discover_sources` prefers general_model for the 21 recordings it
    covers, so `merge_annotations` used to read `a.get("sex", "unknown")`
    straight off THOSE sex-less annotations and silently default everyone to
    "unknown" (3,922 male / 1,229 female annotations lost). V3's own
    annotation files, however, cover many of the SAME recordings (by name,
    via `file_name`'s leading path component) and DO carry sex there, so it
    can be recovered without touching a single pixel of image data.

    Callers pass the union of every source's `ann_paths` -- for the current
    tree this always includes the V3-only recordings' `ann_paths`, which
    point at the very same `red_data_unified_V3/annotations/instances_*.json`
    files that also hold annotations for the general_model-sourced
    recordings, since a V3 annotation file is not scoped to one recording.

    A recording's sex is recovered ONLY when every annotation that carries a
    non-"unknown" `sex` for it AGREES -- "unknown" is silently ignored (a
    recording with 4,592 'unknown' and 0 anything-else has no signal, not a
    unanimous 'unknown'), but a recording that shows BOTH 'male' and
    'female' among its real (non-"unknown") values is never resolved: that
    is reported back as `mixed` so the caller can leave it 'unknown' rather
    than guess. The four two-fly recordings (2026_04_07_11_33_33,
    2026_04_08_14_59_45, and the two 2026_06_11 recordings) fall out of this
    naturally: V3 never recorded per-fly sex for them, so every annotation
    is "unknown" and they get no map entry -- correctly, since a single
    recording-level label would be wrong for a recording with one male and
    one female fly. Their sex comes only from the human labels
    (`apply_sex_labels`), keyed per fly slot.

    Returns `(recording -> sex, sorted list of mixed recordings)`.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    for path in sorted(set(ann_paths)):
        with open(path) as f:
            blob = json.load(f)
        img_rec = {img["id"]: img["file_name"].split("/")[0]
                   for img in blob.get("images", [])}
        for a in blob.get("annotations", []):
            sex = a.get("sex", "unknown")
            if sex == "unknown":
                continue
            rec = img_rec.get(a["image_id"])
            if rec is not None:
                seen[rec].add(sex)

    rec_sex: dict[str, str] = {}
    mixed: list[str] = []
    for rec, sexes in seen.items():
        if len(sexes) == 1:
            rec_sex[rec] = next(iter(sexes))
        else:
            mixed.append(rec)
    return rec_sex, sorted(mixed)


def _canonical_keypoint_names(ann_paths: list[str]) -> list[str]:
    """The FULL keypoint schema, used to pad every reduced-schema source
    (headless recordings: 47 keypoints, missing Antenna_Base/EyeL/EyeR;
    single-leg-amputation recordings: 44, missing one T1 leg's 6 points) up
    to a common slot count. Every reduced schema is verified (BY NAME, not
    just by length) to be a strict, in-canonical-order subset of the
    longest `keypoint_names` list seen anywhere in the tree, so that
    longest list IS the canonical schema. This is recomputed from the data
    -- never hardcoded -- so a future reduced schema this wasn't checked
    against fails loudly in `_keypoint_remap` instead of silently
    corrupting anatomy.
    """
    best: list[str] = []
    for path in sorted(set(ann_paths)):
        with open(path) as f:
            blob = json.load(f)
        names = blob.get("keypoint_names", [])
        if len(names) > len(best):
            best = names
    return best


def _keypoint_remap(source_names: list[str], canonical: list[str]) -> list[int | None]:
    """Index map, BY NAME, from a canonical slot to its index in a source
    keypoints array shaped like `source_names` -- never positional. This
    repo has a documented history of positional keypoint-order bugs that
    scrambled anatomy while every downstream metric looked fine (LOO/IoU,
    residual/NaN checks were all blind to it); mapping by name is how this
    fix avoids repeating it.

    Every name in `source_names` must exist in `canonical` -- if the two
    schemas are not simply subset/superset, silently padding would be a
    guess, so this raises instead (a BLOCKED-level finding to report, not
    paper over).
    """
    missing = [n for n in source_names if n not in canonical]
    if missing:
        raise ValueError(
            f"keypoint name(s) {missing} exist in a source schema but not "
            f"in the canonical {len(canonical)}-keypoint schema -- cannot "
            f"pad by name; this needs a human decision, not a guess")
    name_to_src_idx = {n: i for i, n in enumerate(source_names)}
    return [name_to_src_idx.get(n) for n in canonical]


def _pad_keypoints(kps: list[float], remap: list[int | None]) -> list[float]:
    """Expand a flat (x, y, v) x N_source keypoints array to the canonical
    schema via `remap` (canonical slot -> source index, or None).

    A canonical slot with no source counterpart is padded (0.0, 0.0, 0):
    v=0 is the existing "not visible" sentinel, and that IS the correct
    meaning here -- an amputated leg or a removed head is an anatomically
    ABSENT structure, not a missing observation, so the model must learn
    nothing about it from these frames and IK must not attempt to fit it.
    Never interpolate or infer a value for these slots.
    """
    out: list[float] = []
    for src_idx in remap:
        if src_idx is None:
            out.extend((0.0, 0.0, 0))
        else:
            out.extend(kps[3 * src_idx:3 * src_idx + 3])
    return out


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
    skeleton: list = []
    next_img, next_ann = 0, 0

    # Recording-level sex recovery (see build_recording_sex_map docstring):
    # general_model annotations never carry `sex`, but V3's own annotation
    # files -- referenced somewhere in `sources` for any recording V3 has
    # data on, whether or not that recording's SourceRec.ann_paths point at
    # them -- do. Scanning the union up front lets a general_model-sourced
    # recording recover its sex from V3's copy of the same recording.
    all_ann_paths = sorted({p for s in sources.values() for p in s.ann_paths})
    rec_sex_map, mixed_sex_recs = build_recording_sex_map(all_ann_paths)
    if mixed_sex_recs:
        print(f"merge_annotations: {len(mixed_sex_recs)} recording(s) show "
              f"CONFLICTING non-unknown sex across V3 annotations -- left "
              f"'unknown' rather than guessed: {mixed_sex_recs}")

    # Canonical keypoint schema (see _canonical_keypoint_names docstring):
    # headless/leg-amputation recordings carry 47/44 keypoints, a strict
    # in-order subset of the full 50. Every emitted annotation is padded to
    # this schema BY NAME so the loader always sees a uniform-length array.
    kp_names = _canonical_keypoint_names(all_ann_paths)

    for rec, s in sorted(sources.items()):
        recovered_sex = rec_sex_map.get(rec, "unknown")
        s.sex = recovered_sex
        seen_paths: set[str] = set()
        img_key_to_id: dict[tuple[str, str], int] = {}
        anns_by_img: dict[int, list[dict]] = defaultdict(list)

        for ann_path in s.ann_paths:
            with open(ann_path) as f:
                blob = json.load(f)
            skeleton = skeleton or blob.get("skeleton", [])
            blob_kp_names = blob.get("keypoint_names", [])
            # Fast path: this source already uses the canonical schema (the
            # overwhelming majority of annotations) -- no remap needed.
            kp_remap = (None if blob_kp_names == kp_names else
                        _keypoint_remap(blob_kp_names, kp_names))
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
                        # THE FIX (label-loss bug): a general_model annotation
                        # has no "sex" key at all, so `a.get(..., "unknown")`
                        # always fell back to "unknown" for the 21 recordings
                        # general_model is preferred for -- 3,922 male and
                        # 1,229 female annotations silently dropped. Recover
                        # it from V3's recording-level map when the
                        # per-annotation value is itself "unknown"/absent.
                        ann_sex = a.get("sex", "unknown")
                        if ann_sex == "unknown":
                            ann_sex = recovered_sex
                        # Pad a reduced-schema source (headless/amputation
                        # recordings) to the canonical 50 keypoints BY NAME
                        # -- see _pad_keypoints: missing slots are (0,0,0),
                        # i.e. genuinely not-visible, never guessed at.
                        kps = (a["keypoints"] if kp_remap is None
                               else _pad_keypoints(a["keypoints"], kp_remap))
                        out_anns.append({
                            "id": next_ann, "image_id": img_id,
                            "bbox": a["bbox"], "keypoints": kps,
                            "num_keypoints": a.get("num_keypoints", 50),
                            "sex": ann_sex,
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


def iter_resolved_slots(frameset: dict) -> Iterator[tuple[int, int]]:
    """Yield (img_id, ann_id) for a frameset's cameras that resolved to a fly.

    `frameset["ann_ids"]` runs parallel to `frameset["frames"]` (image ids)
    but may legitimately contain `None`: per CONTROLLER RULING R15 in
    `merge_annotations` above, a camera whose per-frame fly count disagreed
    with the frameset max cannot be assigned an identity by position, so it
    is recorded ABSENT (`None`) for every fly in that frameset rather than
    guessed at. `frameset["frames"]` itself is never `None` -- merge_annotations
    aborts the whole frameset if any camera's image info is missing.

    A frameset is only ever emitted once at least MIN_CAMS cameras resolved,
    so this always yields >= MIN_CAMS pairs -- but callers must not assume
    every camera in `frames`/`ann_ids` is present, and must not index the two
    lists positionally without this filter. This is the third call site to
    trip on the raw `None` (after `write_derived` and the frameset loader),
    so the skip lives here once instead of being reimplemented per consumer.
    """
    for img_id, ann_id in zip(frameset["frames"], frameset["ann_ids"]):
        if ann_id is not None:
            yield img_id, ann_id


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
