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

FRAME-LEVEL, NOT FLY-LEVEL (2026-08-29 fix). Four recordings label BOTH flies
in the same frames, and both flies' framesets point at the SAME 7 physical
images -- there is one capture per frame, not one per fly. Splitting those
recordings per (recording, fly) -- as a prior fix round did, to let a mixed
recording's two flies (who can have different sex) get different treatment --
let fly0's female-tail val frames collide with fly1's male-normal-rule train
frames on the exact same underlying images: a val sample whose pixels are also
in train. The fix keeps the split decision at (recording, frame) granularity:
every fly sharing a frame lands on the same side. The female-preservation
policy is expressed as "a frame joins the guarded-tail-into-val policy (and
ignores val_recordings, exactly as a single-fly female recording already did)
if ANY fly annotated in it is female; otherwise the frame follows the normal
whole-recording-holdout rule." This means a recording with even one female fly
can no longer be held out WHOLE via val_recordings -- the female-preservation
intent (scarce data stays in training) outranks the male's normal rule for any
frame they share, because sharing pixels means they must share a split side,
and it is train (mostly) that is more valuable for the female-heavy case.

CONTENT-KEYED, NOT NAME-KEYED (2026-09-02 fix). Everything above assumed a
recording NAME identifies footage. It does not: the same capture was ingested
twice under different session names, so `2026_06_15_12_12_33` (held out WHOLE
as val) is byte-identical to `2026_06_15_12_12_34` (in train) -- the same
frames, labelled once per fly and filed one second apart. Measured over all
39,536 upstream jpgs: 6 alias groups, and 29.6% of val images (33.1% of val
annotations) byte-identical to a train image. The frame-level rule above was
enforced WITHIN a recording and was blind to it.

The holdout unit is therefore the ALIAS COMPONENT (jarvis_jax.data.
content_index.alias_components -- recordings joined transitively by byte
identity), and the split atom is the CAPTURE `(component, frame)`. Pass
`aliases=` to get this; `aliases=None` reproduces the old name-keyed
behaviour exactly (each recording is its own component) and is kept only so
the pre-existing unit tests still describe the rule they were written for.
Real builds MUST pass aliases and MUST pass image_hashes to audit_split /
write_derived, which then assert zero train/val content overlap.
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


def _frame_groups(merged: dict,
                  aliases: dict[str, str] | None = None,
                  ) -> dict[str, dict[int, list[str]]]:
    """component -> {frame_no: [frameset keys sharing that physical capture]}.

    Two flies annotated in the same recording+frame share the SAME physical
    images (this dataset labels both flies off one multi-camera capture), so
    every key in one inner list MUST end up on the same side of the split --
    that invariant is the entire reason to group this way instead of by
    (recording, fly).

    `aliases` (recording -> component id) extends the same invariant across
    RECORDING NAMES, which is where the 2026-09-02 leak lived: two names for
    one capture put the same pixels in both splits. With aliases=None each
    recording is its own component and this is the pre-2026-09-02 grouping."""
    aliases = aliases or {}
    out: dict[str, dict[int, list[str]]] = {}
    for k, v in merged["framesets"].items():
        comp = aliases.get(v["recording"], v["recording"])
        out.setdefault(comp, {}).setdefault(_frame_no(k), []).append(k)
    return out


def _group_is_female(frames: dict[int, list[str]], merged: dict,
                     recs_meta: dict) -> bool:
    """True if ANY fly annotated anywhere in this component is female. Sex is
    a per-(recording, fly) label that is constant across frames, so in
    practice this is true for either none or all of a component's frames --
    but it is cheap and correct to check every key rather than assume. The
    recording is read off the frameset (not the group id) because a component
    can span several recording names."""
    for keys in frames.values():
        for k in keys:
            rec = merged["framesets"][k]["recording"]
            if _is_female_fly(rec, k.split("/")[-1], recs_meta):
                return True
    return False


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
               seed: int = 0, aliases: dict[str, str] | None = None,
               val_components_force=(), train_components_force=()) -> dict:
    """Assign every frameset to train/val/dropped-guard-band.

    `val_recordings` names recordings to hold out WHOLE. With `aliases` the
    holdout expands to the naming recording's whole alias component, because
    holding out half a component is exactly the leak this replaces.

    `val_components_force` is the deliberate escape hatch: recordings whose
    whole component goes to val EVEN IF it contains a female fly. Female
    preservation normally vetoes a whole holdout, but the only footage that
    can produce an unseen-session FEMALE val number is a female-inclusive
    component, so refusing categorically means never measuring the thing the
    pipeline is worst at. Spend it explicitly, on small components, and say
    how much female training data it costs.

    `train_components_force` is the MIRROR escape hatch, and it exists because
    the female tail-block rule is automatic and therefore fires on components
    where a 10% tail is not a sample of anything. `wall_frames` yields SEVEN
    viable framesets (its camera coverage tops out at 6/7 and four of its
    eleven frames fall below MIN_CAMS); 10% of seven is one frameset, ~4
    annotations, which cannot support a metric while removing it from training
    removes 100% of the corpus's wall-adjacent female supervision -- the exact
    regime the pipeline fails at. Naming a component here keeps it WHOLLY in
    train. It is a deliberate statement that this regime is unmeasurable by
    held-out MPJPE and must be judged GT-free, not an accident of a rounding
    rule. Forcing a component both ways is a contradiction and raises."""
    recs_meta = manifest.get("recordings", {})
    aliases = aliases or {}
    val_groups = {aliases.get(r, r) for r in val_recordings}
    forced_groups = {aliases.get(r, r) for r in val_components_force}
    train_forced = {aliases.get(r, r) for r in train_components_force}
    both = (val_groups | forced_groups) & train_forced
    if both:
        raise ValueError(f"components forced to BOTH train and val: {sorted(both)}")
    by_rec = _frame_groups(merged, aliases)
    split: dict[str, str] = {}

    for rec, frames in by_rec.items():
        frame_nos = sorted(frames)
        is_female = _group_is_female(frames, merged, recs_meta)

        if not is_female or rec in forced_groups or rec in train_forced:
            side = ("val" if (rec in val_groups or rec in forced_groups)
                    and rec not in train_forced else "train")
            for fn in frame_nos:
                for k in frames[fn]:
                    split[k] = side
            continue

        # Female-inclusive recording (>=1 fly in it is female): the decision
        # is made per FRAME (both flies together), not per fly, so fly0's
        # tail-into-val and fly1's same-frame samples can never disagree.
        # val_recordings is ignored here, same as the single-fly-female case
        # always did -- scarce female data stays in training regardless.
        n_val = int(round(len(frame_nos) * female_val_frac))
        if n_val == 0:
            for fn in frame_nos:
                for k in frames[fn]:
                    split[k] = "train"
            continue
        val_frame_nos = set(frame_nos[-n_val:])
        val_start = min(val_frame_nos)
        for fn in frame_nos:
            if fn in val_frame_nos:
                side = "val"
            elif val_start - fn <= guard:
                continue          # guard band: belongs to neither split
            else:
                side = "train"
            for k in frames[fn]:
                split[k] = side

    if not any(v == "val" for v in split.values()):
        raise ValueError("empty val split — every frameset landed in train")
    return split


def _cross_fly_leaks(merged: dict, split: dict) -> list[tuple[str, int]]:
    """(recording, frame_no) pairs where framesets of the SAME frame --
    i.e. different flies pointed at the same physical images -- landed on
    opposite sides of the split. Any non-empty result is a real pixel leak:
    the exact frame-level defect a prior fly-level split introduced (fly0's
    val tail colliding with fly1's train, same 7 images, same frame)."""
    sides: dict[tuple[str, int], set[str]] = {}
    for k, v in merged["framesets"].items():
        side = split.get(k)
        if side not in ("train", "val"):
            continue
        key = (v["recording"], _frame_no(k))
        sides.setdefault(key, set()).add(side)
    return sorted(key for key, s in sides.items() if len(s) > 1)


def audit_split(merged: dict, split: dict, *, guard: int = 50,
                aliases: dict[str, str] | None = None,
                image_hashes: dict[int, str] | None = None) -> dict:
    """Prove the split does not leak. This is the test the old data would fail.

    `image_hashes` (coco image id -> content md5) turns on the only check that
    would have caught the 2026-09-02 defect: whether any pixels are on both
    sides. Names, frame numbers and fly ids were all consistent in the leaky
    split -- the bytes were not."""
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
    leaky_frames = _cross_fly_leaks(merged, split)
    out = {"cross_recording_leaks": leaks, "min_guard_distance": min_dist,
           "val_frac": sum(1 for v in split.values() if v == "val") / max(n, 1),
           # Image-level check (2026-08-29): a frame (same physical images
           # across every fly annotated in it) must never appear on both
           # sides. Zero is the only leak-free value.
           "cross_fly_leaked_frames": len(leaky_frames),
           "leaky_frame_ids": leaky_frames}

    # Capture-level check (2026-09-02): two RECORDING NAMES for one capture.
    caps: dict[str, set[str]] = {}
    for grp, frames in _frame_groups(merged, aliases).items():
        for fn, keys in frames.items():
            sides = {split.get(k) for k in keys} & {"train", "val"}
            if sides:
                caps.setdefault(f"{grp}/Frame_{fn:06d}", set()).update(sides)
    split_caps = sorted(c for c, s in caps.items() if len(s) > 1)
    out["cross_capture_leaks"] = len(split_caps)
    out["leaky_capture_ids"] = split_caps[:20]

    # Content-level check: the ground truth. Everything above is a proxy.
    if image_hashes is not None:
        side_h: dict[str, set[str]] = {"train": set(), "val": set()}
        for k, v in merged["framesets"].items():
            s = split.get(k)
            if s in side_h:
                side_h[s].update(image_hashes[i] for i in v["frames"]
                                 if i in image_hashes)
        ov = side_h["train"] & side_h["val"]
        out["content_overlap"] = len(ov)
        out["content_overlap_examples"] = sorted(ov)[:5]
        out["val_content"] = len(side_h["val"])
        out["train_content"] = len(side_h["train"])
    return out


def write_derived(merged: dict, split: dict, out_root: str, *,
                  image_hashes: dict[int, str] | None = None) -> None:
    """Emit instances_{train,val}.json DERIVED from instances.json + split.json.

    BUILD-TIME GUARD. When `image_hashes` is supplied this REFUSES to write a
    split whose train and val share image content. Two builds shipped without
    that assertion (red_data_3d_v5, red_data_3d_v5_valfix) and both leaked
    29.6% of val; every val number measured on them -- including the
    mask-on/mask-off detector A/B -- is optimistic. A split is not leak-free
    because its recording names are disjoint; it is leak-free when its bytes
    are, and that is cheap to check here."""
    if image_hashes is not None:
        from jarvis_jax.data.content_index import assert_disjoint
        side_h: dict[str, set[str]] = {"train": set(), "val": set()}
        for k, v in merged["framesets"].items():
            s = split.get(k)
            if s in side_h:
                side_h[s].update(image_hashes[i] for i in v["frames"]
                                 if i in image_hashes)
        assert_disjoint(side_h["train"], side_h["val"], what=out_root)

    ann_dir = os.path.join(out_root, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    with open(os.path.join(ann_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=2)

    # keypoint_names.json is the ONLY artifact that lets
    # jarvis_jax.tracking.predict_2d.verify_detector_kp_order actually verify
    # a checkpoint's keypoint order instead of merely warning: it resolves
    # ckpt -> .hydra/overrides.yaml -> paths.data_root -> this file. A root
    # built without it silently downgrades that guard to warn-only, which is
    # how a keypoint-order bug stayed invisible before. The names are already
    # in `merged`, so there is no reason to make it a manual copy step.
    with open(os.path.join(ann_dir, "keypoint_names.json"), "w") as f:
        json.dump(merged["keypoint_names"], f, indent=2)

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
