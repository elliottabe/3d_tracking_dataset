"""Deterministic fly sexing + identity canonicalization for the pose pipeline.

The male is the fly whose wing-splay FLUCTUATES most over a courtship bout
(unilateral wing-extension song), measured as a coefficient of variation. This
module is pure numpy + stdlib (no jax/torch/mujoco) so it runs on the login
node. See docs/superpowers/specs/2026-07-17-fly-sexing-canonicalization-design.md.
"""
from __future__ import annotations

import io
import json
import os
import zipfile

import numpy as np

WING_SIDES = ("L", "R")


def _kp_index(kp_names):
    """Map keypoint name -> column index (model KP order)."""
    return {n: i for i, n in enumerate(kp_names)}


def wing_song_cv(kp3d, conf3d, kp_index, *, conf_min=0.2, min_frames=20):
    """Per-fly wing-song signal: CV (std/mean) of the per-frame max wing-splay angle.

    Body axis = Abd_tip - Scutellum. Wing vectors = WingX_V13 - WingX_base for
    X in {L,R}; per-frame splay = angle(wing, body axis) in degrees, taken as the
    max over the two wings. A frame's splay is valid only when Abd_tip, Scutellum,
    and that side's base+tip all have conf > conf_min. Returns nan if fewer than
    min_frames valid frames or a nonpositive mean."""
    kp3d = np.asarray(kp3d, float)
    conf3d = np.asarray(conf3d, float)

    def col(n):
        return kp_index[n]

    axis = kp3d[:, col("Abd_tip"), :] - kp3d[:, col("Scutellum"), :]
    axis = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9)
    body_ok = (conf3d[:, col("Abd_tip")] > conf_min) & (conf3d[:, col("Scutellum")] > conf_min)

    splays = []
    for s in WING_SIDES:
        w = kp3d[:, col(f"Wing{s}_V13"), :] - kp3d[:, col(f"Wing{s}_base"), :]
        w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-9)
        cosang = np.clip((w * axis).sum(axis=1), -1.0, 1.0)
        a = np.degrees(np.arccos(cosang))
        ok = (body_ok
              & (conf3d[:, col(f"Wing{s}_V13")] > conf_min)
              & (conf3d[:, col(f"Wing{s}_base")] > conf_min))
        splays.append(np.where(ok, a, np.nan))

    with np.errstate(invalid="ignore"):
        splay = np.nanmax(np.stack(splays, axis=1), axis=1)   # (T,) nan where both wings invalid
    valid = splay[np.isfinite(splay)]
    if valid.size < min_frames:
        return float("nan")
    mean = float(valid.mean())
    if mean <= 0:
        return float("nan")
    return float(valid.std() / mean)


def sex_bout_from_pose(cv0, cv1, *, mask_sex_meta=None, ratio_thr=1.5, high_ratio=2.5):
    """Decide the male slot from the two flies' wing-song CVs (4-branch rule)."""
    cv = [cv0, cv1]
    male_pose = None
    ratio = None
    finite = [c for c in cv if c is not None and np.isfinite(c)]
    if len(finite) == 2 and min(cv) > 0:
        male_pose = int(np.argmax(cv))
        ratio = float(max(cv) / min(cv))

    if ratio is not None and ratio >= ratio_thr:
        conf = "high" if ratio >= high_ratio else "medium"
        return dict(male_fly=male_pose, confidence=conf, method="pose_wing_cv",
                    cv=[cv0, cv1], cv_ratio=ratio, note="")
    if mask_sex_meta is not None:
        return dict(male_fly=1, confidence="low", method="mask_fallback",
                    cv=[cv0, cv1], cv_ratio=ratio,
                    note="weak pose signal; mask sexed bout to slot 1")
    return dict(male_fly=None, confidence="unknown", method="unresolved",
                cv=[cv0, cv1], cv_ratio=ratio,
                note="weak pose signal, no mask sex_meta")


# ---------------------------------------------------------------------------
# Human ID review: the authority whenever one exists.
#
# The wing-song CV below is a heuristic and is wrong often enough to matter
# (measured on this dataset's 160 reviewed courtship bouts, the human overruled
# the tracker's original assignment in 24 of them). When a human has reviewed a
# bout, that decision wins.
#
# WHICH CARRIER, AND WHY THE ORDER MATTERS. A review manifest
# (`id_review_reviewed_*.json`) names a POSE FLY-DIR index in the tree the
# reviewer watched. A re-run does not reproduce that tree's fly ordering, so
# replaying `reviewed_male_fly` onto a fresh run's fly0/fly1 can invert the
# label -- the same index-space trap that made the two courtship mask sets
# disagree on 24 bouts. The durable carrier is the MASK: once
# scripts/canonicalize_sam_masks.py has resolved the review into the mask's own
# slot space and written `sex_meta.method = "human_id_review"`, the pose fly
# dirs inherit that slot ordering directly (run_bout writes fly{slot}). So the
# mask's copy of the decision outranks a manifest entry, and a manifest entry
# is used only when the mask carries no human decision -- and then only by a
# caller that knows it is looking at the reviewed tree.
# ---------------------------------------------------------------------------

HUMAN_REVIEW_METHOD = "human_id_review"      # sex_meta.method written by the
                                             # review canonicalizer
_REVIEWED_STATUS = ("confirmed", "swapped")


def _binary(x):
    """x if it is exactly 0 or 1 (and not a bool), else None."""
    if isinstance(x, bool) or not isinstance(x, int) or x not in (0, 1):
        return None
    return x


def load_review(path):
    """The `{bout_key: entry}` mapping from an id_review JSON, either shape
    (`{root, convention, bouts}` or a bare mapping)."""
    with open(path) as f:
        doc = json.load(f)
    return doc.get("bouts", doc) if isinstance(doc, dict) else {}


def review_key_for(bout_dir, pose_dir="pose"):
    """'<...>/Session1/<rec>/pose/bouts/bout_00004' ->
    'Session1/<rec>/bout_00004', the key an id_review manifest uses."""
    parts = os.path.normpath(str(bout_dir)).split(os.sep)
    try:
        i = len(parts) - 1 - parts[::-1].index(pose_dir)
    except ValueError:
        return None
    if i < 2 or len(parts) < i + 3:
        return None
    return "/".join([parts[i - 2], parts[i - 1], parts[-1]])


def review_from_mask_meta(mask_sex_meta):
    """The human decision baked into a canonicalized mask npz, or None.

    Only `sex_meta.method == "human_id_review"` counts. sam3_driver writes a
    `sex_meta` of its own from the mask-area vote; that is a heuristic and must
    not be promoted to 'user' authority just because it lives in the same
    field.
    """
    if not mask_sex_meta:
        return None
    if str(mask_sex_meta.get("method")) != HUMAN_REVIEW_METHOD:
        return None
    slot = _binary(mask_sex_meta.get("male_slot"))
    if slot is None:
        return None
    return dict(male_fly=slot, confidence="user",
                method="human_id_review_masks", cv_ratio=None,
                review_file=mask_sex_meta.get("review_file"),
                review_reviewed_at=mask_sex_meta.get("review_reviewed_at"),
                note=f"male = mask slot {slot} per "
                     f"{mask_sex_meta.get('review_file', 'the human ID review')}")


def review_from_manifest_entry(entry):
    """The human decision from an id_review manifest entry, or None.

    Deliberately ignores `applied`: in this dataset every entry carries
    `applied: false` while 24 of them disagree with `original_male_fly`, so it
    records "no directory swap was performed", not "the tracker was right".
    """
    if not entry:
        return None
    if str(entry.get("status", "unknown")) not in _REVIEWED_STATUS:
        return None
    male = _binary(entry.get("reviewed_male_fly"))
    if male is None:
        return None
    return dict(male_fly=male, confidence="user",
                method="human_id_review_manifest", cv_ratio=None,
                review_reviewed_at=entry.get("reviewed_at"),
                note=f"male = fly{male} per the human ID review "
                     f"(status={entry.get('status')}); the manifest indexes the "
                     f"REVIEWED tree's fly dirs")


_SWAP_TMP = ".fly_swap_tmp"


def read_sex_meta(npz_path):
    """Parse the SAM3 mask `sex_meta` (0-d JSON-string 'sex_meta.npy') from a
    sam3_masks.npz. Returns the dict, or None if the file/entry is absent or
    unreadable."""
    if not npz_path or not os.path.exists(npz_path):
        return None
    try:
        with zipfile.ZipFile(npz_path) as zf:
            if "sex_meta.npy" not in zf.namelist():
                return None
            arr = np.load(io.BytesIO(zf.read("sex_meta.npy")), allow_pickle=True)
        return json.loads(str(arr))
    except (zipfile.BadZipFile, OSError, ValueError):
        return None


def _swap_fly_dirs(bout_dir):
    """Swap fly0 <-> fly1 crash-safely via a temp name. On entry, complete a
    partial swap left by a prior crash (temp dir present)."""
    f0 = os.path.join(bout_dir, "fly0")
    f1 = os.path.join(bout_dir, "fly1")
    tmp = os.path.join(bout_dir, _SWAP_TMP)

    if os.path.exists(tmp):                       # recover partial swap; tmp holds original fly0
        if not os.path.exists(f0) and os.path.exists(f1):     # crashed after step 1
            os.rename(f1, f0)
            os.rename(tmp, f1)
            return
        if os.path.exists(f0) and not os.path.exists(f1):     # crashed after step 2
            os.rename(tmp, f1)
            return
        raise RuntimeError(
            f"ambiguous partial-swap state in {bout_dir}: fly0/fly1 both present "
            f"with {_SWAP_TMP}; resolve manually")

    os.rename(f0, tmp)      # step 1
    os.rename(f1, f0)       # step 2
    os.rename(tmp, f1)      # step 3


def _jsonnum(x):
    """float for JSON, or None for None/non-finite."""
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else x


def canonicalize_bout(bout_dir, kp_names, *, mask_sex_meta=None,
                      review_entry=None, male_slot=1, ratio_thr=1.5,
                      high_ratio=2.5, conf_min=0.2, min_frames=20,
                      dry_run=False, verbose=True):
    """Canonicalize one bout so male == fly{male_slot}, human review first.

    Authority, in order:
      1. `mask_sex_meta` written by scripts/canonicalize_sam_masks.py
         (`method == "human_id_review"`) -- the human decision expressed in the
         mask slot space the pose fly dirs are built from;
      2. `review_entry`, an id_review manifest entry -- only meaningful when
         `bout_dir` is the tree the reviewer watched (see the note above
         review_from_mask_meta);
      3. the wing-song CV heuristic (sex_bout_from_pose).

    The wing-song CV is computed either way and reported as `heuristic_male_fly`
    / `heuristic_agrees`, so a human/heuristic disagreement is visible in
    sex.json rather than silently discarded. Physically swaps the fly dirs when
    the male is not already at male_slot (unless dry_run) and writes sex.json
    (unless dry_run). Idempotent. Returns the decision dict.
    """
    kp_index = _kp_index(kp_names)

    def load_cv(fly):
        p = os.path.join(bout_dir, f"fly{fly}", "kp3d.npz")
        if not os.path.exists(p):
            return None
        with np.load(p) as d:
            return wing_song_cv(d["kp3d"], d["conf3d"], kp_index,
                                conf_min=conf_min, min_frames=min_frames)

    cv0, cv1 = load_cv(0), load_cv(1)
    if cv0 is None or cv1 is None:
        heuristic = dict(male_fly=None, confidence="unknown",
                         method="missing_kp3d", cv=[cv0, cv1], cv_ratio=None,
                         note="a fly kp3d.npz is missing")
    else:
        heuristic = sex_bout_from_pose(cv0, cv1, mask_sex_meta=mask_sex_meta,
                                       ratio_thr=ratio_thr, high_ratio=high_ratio)

    decision = review_from_mask_meta(mask_sex_meta)
    if decision is None:
        decision = review_from_manifest_entry(review_entry)
    authority = "human_review" if decision is not None else "heuristic"
    if decision is None:
        decision = heuristic

    male = decision["male_fly"]
    applied_swap = male is not None and male != male_slot
    if applied_swap and not dry_run:
        _swap_fly_dirs(bout_dir)

    hm = heuristic["male_fly"]
    out = dict(
        male_fly=(male_slot if male is not None else None),
        original_male_fly=male,
        applied_swap=bool(applied_swap),
        confidence=decision["confidence"],
        method=decision["method"],
        authority=authority,
        heuristic_male_fly=hm,
        heuristic_method=heuristic["method"],
        heuristic_agrees=(None if hm is None or male is None else bool(hm == male)),
        review_file=decision.get("review_file"),
        review_reviewed_at=decision.get("review_reviewed_at"),
        wing_cv_original={"fly0": _jsonnum(cv0), "fly1": _jsonnum(cv1)},
        cv_ratio=_jsonnum(heuristic.get("cv_ratio")),
        mask_song_cv=(mask_sex_meta.get("song_cv") if mask_sex_meta else None),
        note=decision["note"],
    )
    if verbose:
        print(f"[sexing] {os.path.basename(os.path.normpath(str(bout_dir)))}: "
              f"authority={authority} method={out['method']} "
              f"male=fly{male} swap={applied_swap} "
              f"heuristic=fly{hm} agrees={out['heuristic_agrees']}")
    if not dry_run:
        with open(os.path.join(bout_dir, "sex.json"), "w") as f:
            json.dump(out, f, indent=2)
    return out
