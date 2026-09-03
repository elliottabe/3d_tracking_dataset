#!/usr/bin/env python3
"""Build a unified 50-node keypoint-detector dataset from ALL of general_model.

Every general_model subset is a single COCO-style annotation pair
(``annotations/instances_{train,val}.json``) whose ``keypoints`` arrays are
SHORT (50, 47, or 44 floats-per-node) with NO ``categories[0]["keypoints"]``
names -- the original defect that made the per-subset column mapping
unrecoverable from the file alone. This module:

  1. Expands every subset's keypoints to the canonical 50-node fly50 order
     (``data/fly50.json`` -- HEAD FIRST: Antenna_Base, EyeL, EyeR, Scutellum,
     ...), using an EXPLICIT, hand-verified per-subset rule table
     (``SUBSET_RULES``) rather than inferring the mapping from array length.
     Missing nodes get COCO's "not labelled" convention: (0, 0, v=0).
  2. Writes non-empty ``categories[0]["keypoints"]``/``["skeleton"]`` so the
     mapping is recoverable from the file going forward.
  3. Re-splits the WHOLE merged pool into train/val BY RECORDING (never by
     frame -- frames within a recording are near-duplicates at 800fps, so a
     frame-level split leaks), stratified so validation covers every
     category in ``REQUIRED_VAL_CATEGORIES``.
  4. References images rather than copying them: each output image keeps
     its original short ``file_name`` plus explicit ``subset``/``orig_split``
     fields, so the real file lives at
     ``<source_root>/<subset>/<orig_split>/<file_name>``.

Note: fly50 order (this module) is NOT the same as `cfg.model.KP_NAMES`
(configs/anatomy/v1.yaml, Scutellum-first model order) -- that is a
separate, model-side convention bridged at inference by
`reorder_detector_to_model`. Do not cross the two.

Do NOT confuse ``orig_split`` (the train/val directory the source subset's
JSON came from -- an accident of how general_model was originally, and
badly, split) with the train/val split this module *produces* -- the whole
point of the rebuild is that the two need not agree.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FLY50_PATH = REPO_ROOT / "data" / "fly50.json"


def load_fly50(path: Path = FLY50_PATH) -> tuple[list[str], list[list[int]]]:
    d = json.loads(Path(path).read_text())
    return list(d["node_names"]), [list(e) for e in d["edges"]]


FLY50, FLY50_EDGES = load_fly50()
NUM_FLY50 = len(FLY50)

# ---------------------------------------------------------------------------
# Explicit per-subset mapping rules (verified by distance-matrix matching
# against an intact reference across all 20 general_model subsets -- see
# .superpowers/sdd/2026-08-07-benchmark-and-scale-ab/progress.md). A rule is
# either the literal string "identity" (subset already emits all 50 fly50
# nodes in fly50 order) or ("drop", [fly50_indices]) meaning the subset
# omits exactly those fly50 nodes (and only those), so its N-length array
# maps onto fly50 order with those indices removed.
# ---------------------------------------------------------------------------
SUBSET_RULES: dict[str, tuple] = {
    # 50 keypoints -> identity in fly50 order.
    "S6male": "identity",
    "courtship_11_50_female": "identity",
    "courtship_11_50_male": "identity",
    "courtship_25_51_female": "identity",
    "courtship_25_51_male": "identity",
    "courtship_28_34_female": "identity",
    "courtship_28_34_male": "identity",
    "courtship_V2": "identity",
    "courtship_V3": "identity",
    "courtship_V4": "identity",
    # The 2026-09-02 re-export of the ONE recording (2025_10_20_13_20_04) that
    # courtship_V2/V3/V4 were three arbitrary slices of. Verified 50 names in
    # fly50 order -- it reproduces all 4,721 of their annotations exactly.
    "courtship_20_04_male": "identity",
    "female": "identity",
    # Female climbing the arena wall during courtship (Session0
    # 2025_10_20_13_20_04, frames inside bout_00028). Verified 50 names in
    # IDENTICAL order to the other courtship subsets, so identity.
    "20_04_female_climbing": "identity",
    "grooming": "identity",
    "wall_frames": "identity",
    # 47 keypoints -> fly50 minus head nodes {Antenna_Base, EyeL, EyeR}.
    "headless_22_50": ("drop", [0, 1, 2]),
    "headless_24_04": ("drop", [0, 1, 2]),
    "headless_24_04_1": ("drop", [0, 1, 2]),
    "headless_56_42": ("drop", [0, 1, 2]),
    "headless_56_42_1": ("drop", [0, 1, 2]),
    # 44 keypoints -> fly50 minus the T1 DISTAL segment on the amputated
    # side, retaining T1*_ThxCx (the coxa/thorax joint is not amputated).
    "S8_male_R_amp": ("drop", [32, 33, 34, 35, 36, 37]),  # T1R_{Tro..TaTip}
    "S9_male_L_amp": ("drop", [10, 11, 12, 13, 14, 15]),  # T1L_{Tro..TaTip}
    # "p4_active_parts_viz" intentionally absent -- it has no annotations.
}

# Subsets known to carry no annotations at all -- skipped outright rather
# than treated as an unmapped-subset error.
SKIP_SUBSETS: frozenset[str] = frozenset({"p4_active_parts_viz"})

# ---------------------------------------------------------------------------
# Category grouping, for split stratification and reporting. A "category"
# is the coarse behavioural/anatomical bucket the benchmark cares about;
# several subsets can share one (e.g. 3 separate courtship-female
# recordings). Every recording maps to exactly one category via its
# subset's entry here.
# ---------------------------------------------------------------------------
SUBSET_CATEGORY: dict[str, str] = {
    "courtship_11_50_female": "courtship_female",
    "courtship_25_51_female": "courtship_female",
    "courtship_28_34_female": "courtship_female",
    # Categorised as courtship_female, not "wall": it is a courtship female
    # (the fly this data exists to improve), and "wall" is the separate
    # single-fly wall_frames set (which is MALE). Climbing is the novel CONDITION within
    # courtship_female, not a different behavioural bucket.
    "20_04_female_climbing": "courtship_female",
    "courtship_11_50_male": "courtship_male",
    "courtship_25_51_male": "courtship_male",
    "courtship_28_34_male": "courtship_male",
    "grooming": "grooming",
    "wall_frames": "wall",
    "headless_22_50": "headless",
    "headless_24_04": "headless",
    "headless_24_04_1": "headless",
    "headless_56_42": "headless",
    "headless_56_42_1": "headless",
    "S8_male_R_amp": "amputated",
    "S9_male_L_amp": "amputated",
    # Included in the build but not part of the required-coverage list below
    # (each is a single recording, so is inherently unvalidatable; see
    # stratified_recording_split).
    "S6male": "male_general",
    "female": "female_general",
    "courtship_V2": "courtship_other",
    "courtship_V3": "courtship_other",
    "courtship_V4": "courtship_other",
    "courtship_20_04_male": "courtship_male",
}

# Categories the val set MUST cover with >=1 recording (when they have >=2
# recordings to choose from -- see stratified_recording_split docstring for
# the single-recording exception).
REQUIRED_VAL_CATEGORIES: frozenset[str] = frozenset({
    "courtship_female", "courtship_male", "grooming", "wall", "headless",
    "amputated",
})

# Per-annotation (sex, behavior) tags. These are NOT present in the source
# subset jsons -- V3 derived them per-subset and the training sampler reads
# them off each annotation (V3Dataset.sampling_weights), so dropping them
# silently degrades weighted sampling to uniform. The values below for the
# 14 subsets that V3 also contained were recovered empirically by joining V4
# annotations to V3 on file_name: every subset resolved to exactly one
# (sex, behavior) pair, so this table reproduces V3 exactly.
#
# The 6 subsets new in V4 (headless_*, S8/S9 amputated, wall_frames) had no
# V3 counterpart. Their sex is taken from the subset name where it states one
# ("S8_male_R_amp" -> male; wall_frames is the male-on-wall set -- corrected
# 2026-09-02, it was recorded female here and that was wrong) and is
# "unknown" for the headless prep, where the name does not say. Behavior is
# "general" for all six: none is a courtship or grooming recording.
SUBSET_SEX_BEHAVIOR: dict[str, tuple[str, str]] = {
    # --- recovered from V3 (exact) ---
    "S6male": ("male", "general"),
    "female": ("female", "general"),
    "grooming": ("unknown", "grooming"),
    "courtship_11_50_female": ("female", "courtship"),
    "courtship_25_51_female": ("female", "courtship"),
    "courtship_28_34_female": ("female", "courtship"),
    # Session0 2025_10_20 female climbing the arena wall mid-courtship. Tagged
    # ("female", "courtship") so it joins the SAME under-represented class the
    # oversampler already targets, rather than forming a class of its own that
    # weighted sampling would have to be re-tuned for.
    "20_04_female_climbing": ("female", "courtship"),
    "courtship_11_50_male": ("male", "courtship"),
    "courtship_25_51_male": ("male", "courtship"),
    "courtship_28_34_male": ("male", "courtship"),
    "courtship_V2": ("unknown", "courtship"),
    "courtship_V3": ("unknown", "courtship"),
    "courtship_V4": ("unknown", "courtship"),
    "courtship_20_04_male": ("male", "courtship"),
    # --- new in V4 (no V3 counterpart) ---
    "S8_male_R_amp": ("male", "general"),
    "S9_male_L_amp": ("male", "general"),
    # CORRECTED 2026-09-02 by the user: wall_frames is MALE. Its source is
    # 2025_10_12_10_56_07_male_climbing; the subset's internal recording id
    # (2026_07_30_13_28_99) does not carry the capture timestamp, so this
    # mapping is not derivable from the data.
    "wall_frames": ("male", "general"),
    "headless_22_50": ("unknown", "general"),
    "headless_24_04": ("unknown", "general"),
    "headless_24_04_1": ("unknown", "general"),
    "headless_56_42": ("unknown", "general"),
    "headless_56_42_1": ("unknown", "general"),
}


class SubsetRuleMismatch(ValueError):
    """A subset's observed keypoint count contradicts its declared rule."""


def expected_count_for_rule(rule) -> int:
    if rule == "identity":
        return NUM_FLY50
    if isinstance(rule, (tuple, list)) and len(rule) == 2 and rule[0] == "drop":
        return NUM_FLY50 - len(rule[1])
    raise ValueError(f"unrecognized rule: {rule!r}")


def validate_subset_keypoint_count(subset: str, observed: int, rule) -> None:
    """Fail loudly if `observed` (a subset's keypoints-per-annotation count)
    does not match what `rule` says it should be."""
    expected = expected_count_for_rule(rule)
    if observed != expected:
        raise SubsetRuleMismatch(
            f"subset {subset!r}: rule {rule!r} expects {expected} keypoints "
            f"per annotation but observed {observed} -- the SUBSET_RULES "
            f"table is out of date for this subset (re-verify the mapping, "
            f"do not silently reinterpret it)."
        )


def expand_to_canonical(kps_flat, rule) -> tuple[np.ndarray, np.ndarray]:
    """Expand one annotation's flat [x,y,v]*N keypoints array to canonical
    fly50 order.

    Returns (kp50, mask50): kp50 is (50, 3) float64 with columns [x, y, v] in
    fly50 node order; mask50 is (50,) bool, True where the node is present in
    the source data. Present nodes keep their (x, y) with v=2 (COCO
    "labelled and visible"); absent nodes get (0, 0, v=0) -- v=0 is the COCO
    convention for "not labelled", which is what lets a keypoint loss skip
    them.
    """
    kp = np.asarray(kps_flat, dtype=np.float64).reshape(-1, 3)
    out = np.zeros((NUM_FLY50, 3), dtype=np.float64)
    mask = np.zeros((NUM_FLY50,), dtype=bool)

    if rule == "identity":
        if kp.shape[0] != NUM_FLY50:
            raise ValueError(
                f"identity rule expects {NUM_FLY50} keypoints, got {kp.shape[0]}"
            )
        out[:, :2] = kp[:, :2]
        out[:, 2] = 2.0
        mask[:] = True
        return out, mask

    if isinstance(rule, (tuple, list)) and len(rule) == 2 and rule[0] == "drop":
        drop = set(rule[1])
        keep_idx = [i for i in range(NUM_FLY50) if i not in drop]
        if kp.shape[0] != len(keep_idx):
            raise ValueError(
                f"drop rule (dropping {sorted(drop)}) expects {len(keep_idx)} "
                f"keypoints, got {kp.shape[0]}"
            )
        for src, dst in enumerate(keep_idx):
            out[dst, :2] = kp[src, :2]
            out[dst, 2] = 2.0
            mask[dst] = True
        return out, mask

    raise ValueError(f"unrecognized rule: {rule!r}")


# ---------------------------------------------------------------------------
# Split-by-recording stratification
# ---------------------------------------------------------------------------

@dataclass
class CategorySplitInfo:
    category: str
    train_recordings: list
    val_recordings: list
    unvalidatable: bool
    train_annotations: int = 0
    val_annotations: int = 0


@dataclass
class SplitResult:
    train_recordings: set
    val_recordings: set
    category_info: dict
    achieved_val_fraction: float
    unvalidatable_categories: list = field(default_factory=list)


def _summarize_split(category_recordings: dict[str, dict[str, int]],
                      train_set: set, val_set: set) -> SplitResult:
    category_info: dict[str, CategorySplitInfo] = {}
    unvalidatable: list[str] = []
    total_ann = 0
    val_ann = 0
    for cat in sorted(category_recordings):
        recs = category_recordings[cat]
        train_r = sorted(r for r in recs if r in train_set)
        val_r = sorted(r for r in recs if r in val_set)
        is_unvalidatable = len(recs) <= 1 or len(val_r) == 0
        if is_unvalidatable:
            unvalidatable.append(cat)
        tcount = sum(recs[r] for r in train_r)
        vcount = sum(recs[r] for r in val_r)
        total_ann += tcount + vcount
        val_ann += vcount
        category_info[cat] = CategorySplitInfo(
            category=cat, train_recordings=train_r, val_recordings=val_r,
            unvalidatable=is_unvalidatable, train_annotations=tcount,
            val_annotations=vcount,
        )
    frac = (val_ann / total_ann) if total_ann else 0.0
    return SplitResult(set(train_set), set(val_set), category_info, frac,
                        unvalidatable)


def stratified_recording_split(
    category_recordings: dict[str, dict[str, int]],
    required_categories: Iterable[str],
    target_frac: tuple[float, float] = (0.15, 0.20),
    seed: int = 0,
) -> SplitResult:
    """Split recordings into train/val, never splitting a recording across
    both.

    Every category with >=2 recordings gets exactly one recording guaranteed
    into val -- the SMALLEST by annotation count, so coverage costs as
    little train data as possible -- so every `required_categories` entry
    with enough data is represented; a category with only 1 recording
    cannot be split at all and goes entirely to train, flagged
    `unvalidatable` in the result (this applies to ANY category with a
    single recording, required or not -- see module docstring). More
    recordings are then greedily added to val (smallest annotation count
    first, never draining a category to zero train recordings) until the
    overall annotation fraction reaches `target_frac[0]`; overshooting
    `target_frac[1]` is accepted rather than leaving a required category
    uncovered. `seed` only breaks ties (equal annotation counts) -- the
    selection is otherwise deterministic by count, not random, so that
    "add coverage as cheaply as possible" is the actual policy rather than
    an artifact of shuffle order.
    """
    required_categories = set(required_categories)
    missing = required_categories - set(category_recordings)
    if missing:
        raise ValueError(
            f"required categories missing from category_recordings: "
            f"{sorted(missing)}"
        )

    rng = random.Random(seed)
    train_set: set[str] = set()
    val_set: set[str] = set()
    candidates: list[tuple[str, str]] = []  # (recording, category)

    for cat in sorted(category_recordings):
        recs = category_recordings[cat]
        rec_ids = sorted(recs)
        if len(rec_ids) <= 1:
            train_set.update(rec_ids)
            continue
        rng.shuffle(rec_ids)  # tie-break equal counts deterministically per seed
        rec_ids.sort(key=lambda r: recs[r])
        val_pick, remaining = rec_ids[0], rec_ids[1:]
        val_set.add(val_pick)
        train_set.update(remaining)
        candidates.extend((r, cat) for r in remaining)

    total_ann = sum(v for recs in category_recordings.values() for v in recs.values())
    val_ann = sum(category_recordings[cat][r] for cat in category_recordings
                  for r in category_recordings[cat] if r in val_set)
    lo, _hi = target_frac

    rng.shuffle(candidates)
    candidates.sort(key=lambda rc: category_recordings[rc[1]][rc[0]])
    train_count_by_cat = {
        cat: sum(1 for r in recs if r in train_set)
        for cat, recs in category_recordings.items()
    }
    for r, cat in candidates:
        if total_ann and val_ann / total_ann >= lo:
            break
        if train_count_by_cat[cat] <= 1:
            continue  # never drain a category's train pool to zero
        train_set.discard(r)
        val_set.add(r)
        train_count_by_cat[cat] -= 1
        val_ann += category_recordings[cat][r]

    return _summarize_split(category_recordings, train_set, val_set)


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------

def _load_subset_coco(source_root: Path, subset: str) -> dict[str, dict]:
    out = {}
    ann_dir = source_root / subset / "annotations"
    for split in ("train", "val"):
        p = ann_dir / f"instances_{split}.json"
        if p.exists():
            out[split] = json.loads(p.read_text())
    return out


def discover_subsets(source_root: Path, subset_rules: dict,
                      skip_subsets: frozenset,
                      subset_sex_behavior: dict | None = None) -> list[str]:
    subsets = []
    for child in sorted(source_root.iterdir()):
        if not child.is_dir() or child.name in skip_subsets:
            continue
        if not (child / "annotations").is_dir():
            continue
        if child.name not in subset_rules:
            raise ValueError(
                f"subset {child.name!r} has an annotations/ dir but no rule "
                f"in SUBSET_RULES -- add an explicit mapping (or to "
                f"SKIP_SUBSETS if it truly has no usable annotations) "
                f"before building."
            )
        tags = (SUBSET_SEX_BEHAVIOR if subset_sex_behavior is None
                else subset_sex_behavior)
        if child.name not in tags:
            # Fail loudly rather than defaulting to ("unknown", "unknown"): a
            # silently-untagged subset is invisible to the weighted sampler,
            # which is precisely the regression that made V4's first build
            # train uniformly while appearing to oversample.
            raise ValueError(
                f"subset {child.name!r} has no SUBSET_SEX_BEHAVIOR entry -- "
                f"add its (sex, behavior) tags before building, or weighted "
                f"sampling will silently ignore it."
            )
        subsets.append(child.name)
    return subsets


def build(source_root: Path, out_root: Path, *, val_recordings=None, seed: int = 0,
          subset_rules: dict | None = None, subset_category: dict | None = None,
          required_categories=None, skip_subsets: frozenset | None = None,
          subset_sex_behavior: dict | None = None,
          target_val_frac: tuple[float, float] = (0.15, 0.20)) -> dict:
    """Merge every general_model subset into one unified 50-node COCO
    dataset, split by recording, and write it (+ a build report) to
    `out_root`.

    Images are referenced, not copied: each output image's `file_name` is
    unchanged from the source, with explicit `subset`/`orig_split` fields
    added so `<source_root>/<subset>/<orig_split>/<file_name>` resolves to
    the real file.
    """
    source_root = Path(source_root)
    out_root = Path(out_root)
    subset_rules = SUBSET_RULES if subset_rules is None else subset_rules
    subset_category = SUBSET_CATEGORY if subset_category is None else subset_category
    required_categories = (REQUIRED_VAL_CATEGORIES if required_categories is None
                            else set(required_categories))
    skip_subsets = SKIP_SUBSETS if skip_subsets is None else skip_subsets
    subset_sex_behavior = (SUBSET_SEX_BEHAVIOR if subset_sex_behavior is None
                            else subset_sex_behavior)

    subsets = discover_subsets(source_root, subset_rules, skip_subsets,
                                subset_sex_behavior)

    image_records = []  # every image, annotated or not -- these become merged_images
    ann_records = []    # every annotation (expanded to canonical) -- become merged_annotations
    per_subset_report: dict[str, dict] = {}
    recording_category: dict[str, str] = {}

    def _register_recording(recording: str, category: str) -> None:
        prev_cat = recording_category.get(recording)
        if prev_cat is not None and prev_cat != category:
            raise ValueError(
                f"recording {recording!r} appears under two categories "
                f"({prev_cat!r} and {category!r}) -- a recording must be a "
                f"single split unit"
            )
        recording_category[recording] = category

    for subset in subsets:
        rule = subset_rules[subset]
        category = subset_category.get(subset)
        if category is None:
            raise ValueError(f"subset {subset!r} has no entry in subset_category")

        coco_by_split = _load_subset_coco(source_root, subset)
        subset_images = 0
        subset_anns = 0
        checked = False
        for orig_split, coco in coco_by_split.items():
            id2img = {im["id"]: im for im in coco["images"]}
            subset_images += len(coco["images"])

            # Register every image (annotated or not) -- images are the
            # unit that gets referenced downstream, and every one needs a
            # recording->category assignment for the split.
            for img in coco["images"]:
                recording = img["file_name"].split("/")[0]
                _register_recording(recording, category)
                image_records.append({
                    "subset": subset, "orig_split": orig_split,
                    "recording": recording, "category": category, "image": img,
                })

            for ann in coco["annotations"]:
                subset_anns += 1
                observed = len(ann["keypoints"]) // 3
                if not checked:
                    validate_subset_keypoint_count(subset, observed, rule)
                    checked = True
                kp50, mask50 = expand_to_canonical(ann["keypoints"], rule)
                img = id2img[ann["image_id"]]
                recording = img["file_name"].split("/")[0]
                ann_records.append({
                    "subset": subset, "orig_split": orig_split,
                    "recording": recording, "category": category,
                    "image": img, "ann": ann, "kp50": kp50, "mask50": mask50,
                })
        per_subset_report[subset] = {
            "rule": list(rule) if isinstance(rule, tuple) else rule,
            "category": category,
            "images": subset_images,
            "annotations": subset_anns,
        }

    recording_ann_count: dict[str, int] = defaultdict(int)
    for r in ann_records:
        recording_ann_count[r["recording"]] += 1

    category_recordings: dict[str, dict[str, int]] = defaultdict(dict)
    for recording, category in recording_category.items():
        category_recordings[category][recording] = recording_ann_count[recording]

    if val_recordings is not None:
        val_set = set(val_recordings)
        train_set = set(recording_category) - val_set
        split_result = _summarize_split(category_recordings, train_set, val_set)
    else:
        split_result = stratified_recording_split(
            category_recordings, required_categories, target_val_frac, seed)

    new_split_of = {r: "train" for r in split_result.train_recordings}
    new_split_of.update({r: "val" for r in split_result.val_recordings})

    merged_images: dict[str, list] = {"train": [], "val": []}
    merged_annotations: dict[str, list] = {"train": [], "val": []}
    image_id_counters = {"train": 0, "val": 0}
    ann_id_counters = {"train": 0, "val": 0}
    image_key_to_new_id: dict[tuple, int] = {}
    absent_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for rec in image_records:
        new_split = new_split_of[rec["recording"]]
        key = (rec["subset"], rec["orig_split"], rec["image"]["id"])
        if key not in image_key_to_new_id:
            new_img_id = image_id_counters[new_split]
            image_id_counters[new_split] += 1
            image_key_to_new_id[key] = new_img_id
            merged_images[new_split].append({
                "id": new_img_id,
                "file_name": rec["image"]["file_name"],
                "width": rec["image"].get("width"),
                "height": rec["image"].get("height"),
                "subset": rec["subset"],
                "orig_split": rec["orig_split"],
                "recording": rec["recording"],
            })

    for rec in ann_records:
        new_split = new_split_of[rec["recording"]]
        key = (rec["subset"], rec["orig_split"], rec["image"]["id"])
        new_img_id = image_key_to_new_id[key]

        mask50 = rec["mask50"]
        kp50 = rec["kp50"]
        flat_kp = []
        for (x, y, v) in kp50:
            flat_kp.extend([float(x), float(y), int(v)])
        bbox = rec["ann"].get("bbox", [])
        area = float(bbox[2]) * float(bbox[3]) if len(bbox) == 4 else 0.0

        new_ann_id = ann_id_counters[new_split]
        ann_id_counters[new_split] += 1
        sex, behavior = subset_sex_behavior[rec["subset"]]
        merged_annotations[new_split].append({
            "id": new_ann_id,
            "image_id": new_img_id,
            "category_id": 1,
            "iscrowd": rec["ann"].get("iscrowd", 0),
            "bbox": bbox,
            "area": area,
            "num_keypoints": int(mask50.sum()),
            "keypoints": flat_kp,
            "segmentation": rec["ann"].get("segmentation", []),
            # Sampling metadata. `sex`/`behavior` reproduce V3's schema (read by
            # V3Dataset.sampling_weights); `category` is the finer axis added in
            # V4 -- it distinguishes wall/headless/amputated, which all collapse
            # to the same (sex, behavior) pair and so cannot be balanced apart
            # on the V3 axes alone. Used by V3Dataset.balanced_weights.
            "sex": sex,
            "behavior": behavior,
            "category": rec["category"],
        })

        for i, present in enumerate(mask50):
            if not present:
                absent_counts[rec["category"]][FLY50[i]] += 1

    categories = [{
        "id": 1,
        "name": "fly",
        "supercategory": "animal",
        "keypoints": FLY50,
        "skeleton": FLY50_EDGES,
    }]

    # Top-level `keypoint_names`/`skeleton` mirror the V3 COCO json exactly, so
    # a V4 root is a drop-in replacement for a V3 one. Several consumers read
    # them from the top level rather than from categories[0] -- notably
    # jarvis_jax.scripts.train_keypoints (build_lr_swap, the left/right flip
    # augmentation table) which does a bare `["keypoint_names"]` and dies with a
    # KeyError otherwise. `skeleton` uses V3's dict schema (keypointA/keypointB,
    # not index pairs); build_skeleton_edges accepts both, but matching V3 keeps
    # the two roots byte-comparable for anything that copies the field through.
    skeleton_named = [
        {"keypointA": FLY50[a], "keypointB": FLY50[b],
         "length": 0.0, "name": f"Joint {i + 1}"}
        for i, (a, b) in enumerate(FLY50_EDGES)
    ]

    ann_dir = out_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        payload = {
            "keypoint_names": FLY50,
            "skeleton": skeleton_named,
            "images": merged_images[split],
            "annotations": merged_annotations[split],
            "categories": categories,
            "info": {
                "source_root": str(source_root),
                "seed": seed,
                "keypoint_order": "fly50 (data/fly50.json), head-first",
            },
        }
        (ann_dir / f"instances_{split}.json").write_text(json.dumps(payload))

    totals = {
        "images": len(merged_images["train"]) + len(merged_images["val"]),
        "annotations": len(merged_annotations["train"]) + len(merged_annotations["val"]),
        "train_images": len(merged_images["train"]),
        "train_annotations": len(merged_annotations["train"]),
        "val_images": len(merged_images["val"]),
        "val_annotations": len(merged_annotations["val"]),
        "val_fraction": split_result.achieved_val_fraction,
    }

    categories_report = {}
    for cat, info in split_result.category_info.items():
        categories_report[cat] = {
            "required": cat in required_categories,
            "unvalidatable": info.unvalidatable,
            "train_recordings": info.train_recordings,
            "val_recordings": info.val_recordings,
            "train_annotations": info.train_annotations,
            "val_annotations": info.val_annotations,
            "absent_node_counts": dict(absent_counts.get(cat, {})),
        }

    report = {
        "source_root": str(source_root),
        "out_root": str(out_root),
        "seed": seed,
        "fly50_nodes": FLY50,
        "subsets": per_subset_report,
        "categories": categories_report,
        "totals": totals,
        "unvalidatable_categories": split_result.unvalidatable_categories,
    }
    (out_root / "build_report.json").write_text(json.dumps(report, indent=2))
    return report


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-recordings", type=str, default=None,
                    help="comma-separated recording ids to force into val "
                         "(overrides automatic stratified selection)")
    return p.parse_args()


def main():
    args = _parse_args()
    val_recordings = (args.val_recordings.split(",") if args.val_recordings
                       else None)
    report = build(args.source_root, args.out_root,
                    val_recordings=val_recordings, seed=args.seed)
    print(json.dumps(report["totals"], indent=2))
    print(f"Unvalidatable categories: {report['unvalidatable_categories']}")
    print(f"Report written to {Path(args.out_root) / 'build_report.json'}")


if __name__ == "__main__":
    main()
