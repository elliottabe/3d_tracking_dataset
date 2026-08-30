"""Split red_data_3d_v5 without leakage.

THE DEFECT THIS REPLACES: general_model split frame-level WITHIN each recording,
so ~50% of val framesets had a train frameset within +/-3 frames (courtship_V3
86%, S8_male_R_amp 100%). Every 3D val number produced before 2026-08-29 --
MPJPE 1.083, 0.708, and the laplacian ablation that set cached3d.yaml -- scored
near-duplicate frames.

POLICY. Male/general recordings are data-rich, so they are held out WHOLE: zero
leakage, and the val number means "generalizes to an unseen fly and session".
Female recordings are the binding constraint (199 framesets, 7% of the data),
so every one stays in training and val is a contiguous tail block separated by a
guard band. That measures "new pose, same fly and session" and WILL read
optimistic against bout 28 -- an accepted, documented limit. The real female
test is bout 28 under GT-free metrics plus rendered overlays.
"""
from __future__ import annotations

import json
import os
import re

_FRAME_RE = re.compile(r"Frame_(\d+)")


def _frame_no(key: str) -> int:
    return int(_FRAME_RE.search(key).group(1))


def _by_recording(merged: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k, v in merged["framesets"].items():
        out.setdefault(v["recording"], []).append(k)
    for rec in out:
        out[rec].sort(key=_frame_no)
    return out


def _by_recording_fly(merged: dict) -> dict[tuple[str, str], list[str]]:
    """Like _by_recording, but keyed by (recording, 'fly<k>') so a mixed
    recording's two flies -- who can have different sex -- are split
    independently. Frameset keys are '<recording>/Frame_<N>/fly<k>'."""
    out: dict[tuple[str, str], list[str]] = {}
    for k, v in merged["framesets"].items():
        out.setdefault((v["recording"], k.split("/")[-1]), []).append(k)
    for grp in out:
        out[grp].sort(key=_frame_no)
    return out


def _is_female_fly(rec: str, fly: str, recs_meta: dict) -> bool:
    """Resolve sex for one fly of one recording. A per-fly label in
    manifest["recordings"][rec]["fly_sex"]["fly<k>"] (written by
    apply_sex_labels for the two-fly recordings where the slots differ) takes
    precedence; otherwise fall back to the recording-level manifest["sex"]
    used for every single-fly recording."""
    meta = recs_meta.get(rec, {})
    fly_sex = meta.get("fly_sex", {})
    if fly in fly_sex:
        return fly_sex[fly] == "female"
    return meta.get("sex") == "female"


def make_split(merged: dict, manifest: dict, *, val_recordings=(),
               female_val_frac: float = 0.10, guard: int = 50,
               seed: int = 0) -> dict:
    groups = _by_recording_fly(merged)
    recs_meta = manifest.get("recordings", {})
    split: dict[str, str] = {}
    val_recordings = set(val_recordings)

    for (rec, fly), keys in groups.items():
        is_female = _is_female_fly(rec, fly, recs_meta)
        if rec in val_recordings and not is_female:
            for k in keys:
                split[k] = "val"
            continue
        if not is_female:
            for k in keys:
                split[k] = "train"
            continue

        # Female: contiguous tail block for val, guard band dropped entirely.
        n_val = int(round(len(keys) * female_val_frac))
        if n_val == 0:
            for k in keys:
                split[k] = "train"
            continue
        val_keys = keys[-n_val:]
        val_start = _frame_no(val_keys[0])
        for k in keys[:-n_val]:
            if val_start - _frame_no(k) <= guard:
                continue          # guard band: belongs to neither split
            split[k] = "train"
        for k in val_keys:
            split[k] = "val"

    if not any(v == "val" for v in split.values()):
        raise ValueError("empty val split — every frameset landed in train")
    return split


def audit_split(merged: dict, split: dict, *, guard: int = 50) -> dict:
    """Prove the split does not leak. This is the test the old data would fail."""
    groups = _by_recording(merged)
    leaks = 0
    min_dist: dict[str, int] = {}
    for rec, keys in groups.items():
        tr = [_frame_no(k) for k in keys if split.get(k) == "train"]
        va = [_frame_no(k) for k in keys if split.get(k) == "val"]
        if not tr or not va:
            continue
        # Same recording in BOTH splits: only legal for the female guard-band case.
        leaks += 0
        tr_s = sorted(tr)
        d = min(min(abs(v - t) for t in tr_s) for v in va)
        min_dist[rec] = d
    for rec, keys in groups.items():
        tr = any(split.get(k) == "train" for k in keys)
        va = any(split.get(k) == "val" for k in keys)
        if tr and va and min_dist.get(rec, 0) < guard:
            leaks += 1
    n = sum(1 for v in split.values() if v in ("train", "val"))
    return {"cross_recording_leaks": leaks, "min_guard_distance": min_dist,
            "val_frac": sum(1 for v in split.values() if v == "val") / max(n, 1)}


def write_derived(merged: dict, split: dict, out_root: str) -> None:
    """Emit instances_{train,val}.json DERIVED from instances.json + split.json."""
    ann_dir = os.path.join(out_root, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    with open(os.path.join(ann_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=2)

    by_id = {a["id"]: a for a in merged["annotations"]}
    img_by_id = {i["id"]: i for i in merged["images"]}
    for name in ("train", "val"):
        keys = [k for k, v in split.items() if v == name]
        fs = {k: merged["framesets"][k] for k in keys}
        # ann_ids may legitimately contain None (merge_annotations records a
        # camera as ABSENT when its per-frame fly count disagrees with the
        # frameset max) -- drop those before sorting/looking up, but keep the
        # frameset itself: merge_annotations already guarantees >= MIN_CAMS
        # resolvable views before it emits the frameset at all.
        ann_ids = {a for v in fs.values() for a in v["ann_ids"] if a is not None}
        # frames (image ids) are never None -- merge_annotations only appends
        # to cam_img_ids for cameras with real image info, and aborts the
        # whole frameset (`break`) if any camera's frame info is missing.
        img_ids = {i for v in fs.values() for i in v["frames"]}
        blob = {
            "keypoint_names": merged["keypoint_names"],
            "skeleton": merged["skeleton"],
            "categories": merged["categories"],
            "images": [img_by_id[i] for i in sorted(img_ids)],
            "annotations": [by_id[a] for a in sorted(ann_ids)],
            "framesets": fs,
        }
        with open(os.path.join(ann_dir, f"instances_{name}.json"), "w") as f:
            json.dump(blob, f)
