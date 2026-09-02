#!/usr/bin/env python3
"""Build a canonicalized COPY of a SAM3 mask set: fly slot 1 is the male, in
every bout, with the human ID review as the authority.

WHY A COPY, AND WHY AT THE MASK LEVEL. The mask npz is upstream of everything --
2D keypoints, triangulation, STAC/IK, every per-fly figure -- so fixing identity
here means a reprocess inherits it for free, with no per-stage swap logic. And
the source stays the master: this script only ever reads it (`--verify-source`
re-hashes it afterwards to prove that).

WHY THE MALE SLOT IS MEASURED, NEVER REPLAYED AS AN INDEX. `reviewed_male_fly`
is the pose fly-DIR index the human judged (the GUI serves
`pose/bouts/bout_*/fly<k>/sidebyside.mp4`). A mask npz's fly axis is a
different index space, and the two courtship mask sets DISAGREE. Measured
2026-09-02 with the vote below over all 160 reviewed bouts: pose fly0 lands on
mask slot 0 in 160/160 bouts of `processed/.../sam3_masks`, but in only 135/160
of `Video_recordings/.../Predictions_3D_sam3*` -- the other 25 (21 of
Session0's 30, plus 4 in Session1) have the two slots REVERSED, every one of
them unanimous across all 7 cameras. Copying `reviewed_male_fly` straight onto
a slot index would therefore have mislabelled the sex of 25 bouts while every
stage reported success. So for each bout this script VOTES the reviewed fly's
2D keypoints against the centroids of the file it is about to write, per camera
matched BY NAME, and refuses when the vote is unusable. Same argument, same
method as scripts/qc/remap_review_to_new_masks.py.

Relation to the neighbours:
  * scripts/recanonicalize_masks.py -- swaps masks IN PLACE using the mask-area
    vote as the authority. Automatic, no human input, mutates the source.
  * scripts/canonicalize_session_sex.py -- swaps POSE fly dirs after a run.
    Operates on pose output, not masks.
  * this script -- human review as the authority, writes a separate tree,
    source untouched. Use it when a human has reviewed the bouts.

Typical use:

    python scripts/canonicalize_sam_masks.py \\
        --review  /gscratch/portia/eabe/data/Johnson_lab/processed/courtship/id_review_reviewed_20260829.json \\
        --src-root /gscratch/portia/eabe/data/Johnson_lab/processed/_courtship_backup \\
        --pose-root /gscratch/portia/eabe/data/Johnson_lab/processed/_courtship_backup \\
        --video-root /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship \\
        --dest /gscratch/portia/eabe/data/Johnson_lab/processed/courtship_canonical \\
        --dry-run

Then drop `--dry-run`. Point the pipeline at the result with

    recording.predictions_dir=<dest>/<Session>/<recording>/sam3_masks

(the same tail as configs/recording/session*.yaml, so only the dataset segment
of the interpolation changes).

Only `sam3_masks.npz` is carried across. The `maskvid_*.mp4` / `_still.png`
renders beside some sources are NOT copied: they are pictures of the
PRE-canonicalization slot colouring, so in the canonical tree they would show
fly1-orange painted on the female for exactly the bouts this script fixed --
worse than absent. `*.bak` files are snapshots of an older state of the source
and have no meaning in a freshly derived tree.

Heavy: reads and rewrites ~2 GB of compressed masks. Run on a compute node.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# Arrays whose axis 0 is the FLY SLOT and which therefore must move together.
# `packed`/`centroids`/`valid` are written by sam3_driver; `in_frame` is the
# per-(fly, camera, frame) visibility code appended after the repairs.
FLY_AXIS_MEMBERS = ("packed", "centroids", "valid", "in_frame")

# Arrays that must NOT be reversed. `shape` is the trap: it is [H, W], so its
# leading dimension is also 2 and a blind "reverse everything with a 2 in
# front" transposes every mask unpack downstream.
NON_FLY_MEMBERS = ("cameras", "shape", "version", "sex_meta", "sync",
                   "suspect_cameras", "gap_repair")

_ACCEPTED_STATUS = ("confirmed", "swapped")


class Refusal(Exception):
    """A condition the script must not guess its way past."""


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

def review_male_pose_fly(key, entry):
    """The reviewed male's POSE fly-dir index, or raise.

    `applied` is deliberately ignored. Every entry in this dataset's review
    carries `applied: false` while 24 of them have
    `original_male_fly != reviewed_male_fly`, so `applied` does not mean "the
    tracker already agreed" -- it means no directory swap was ever performed.
    The authority is `reviewed_male_fly` alone.
    """
    if entry is None:
        raise Refusal(f"{key}: no review entry -- an unreviewed bout has no "
                      f"authority to canonicalize against")
    status = str(entry.get("status", "unknown"))
    if status not in _ACCEPTED_STATUS:
        raise Refusal(f"{key}: review status is {status!r}, not one of "
                      f"{_ACCEPTED_STATUS}")
    male = entry.get("reviewed_male_fly")
    if isinstance(male, bool) or not isinstance(male, int) or male not in (0, 1):
        raise Refusal(f"{key}: reviewed_male_fly is {male!r}, not 0 or 1")
    return male


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------

def canonical_cameras(calib_dir):
    """The canonical camera order: sorted `Cam*.yaml` in the calibration dir.

    This is exactly how ReprojectionTool builds its camera list, and therefore
    the order of kp2d/kp3d's camera axis. Reimplemented with glob so this
    script stays pure numpy + stdlib (no cv2/jax import for a directory
    listing).
    """
    files = sorted(glob.glob(os.path.join(str(calib_dir), "Cam*.yaml")))
    if not files:
        raise Refusal(f"no Cam*.yaml calibration files in {calib_dir}")
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


# ---------------------------------------------------------------------------
# npz members
# ---------------------------------------------------------------------------

def read_members(path):
    """Every array in an npz, as a plain dict."""
    with np.load(str(path), allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def swap_fly_axis(members):
    """Reverse axis 0 of every fly-indexed array; leave the rest alone.

    Refuses on a member classified as neither -- a member the mask writer
    added after this script was written is exactly the case where guessing
    leaves one fly axis behind.
    """
    unknown = [k for k in members
               if k not in FLY_AXIS_MEMBERS and k not in NON_FLY_MEMBERS]
    if unknown:
        raise Refusal(
            f"unclassified npz member(s) {sorted(unknown)}: add each to "
            f"FLY_AXIS_MEMBERS (axis 0 is the fly slot) or NON_FLY_MEMBERS "
            f"in scripts/canonicalize_sam_masks.py before swapping")
    out = {}
    for k, v in members.items():
        out[k] = v[::-1] if k in FLY_AXIS_MEMBERS else v
    return out


def read_sex_meta_member(members):
    """The source's own `sex_meta`, parsed, or None."""
    if "sex_meta" not in members:
        return None
    try:
        return json.loads(str(members["sex_meta"]))
    except ValueError:
        return {"unparseable": str(members["sex_meta"])}


# ---------------------------------------------------------------------------
# resolving the reviewed fly onto a slot of THIS file
# ---------------------------------------------------------------------------

def _kp_centroids(kz, ci, thresh):
    """(T, 2) mean of the confident keypoints in camera index `ci`; NaN where
    none. `np.linalg.norm(v, -1)` would pass -1 as `ord`, not `axis` -- always
    axis=-1 (see scripts/qc/audit_fly_assignment.py)."""
    g = kz["conf"][:, ci] >= thresh
    return np.nanmean(np.where(g[..., None], kz["kp2d"][:, ci], np.nan), axis=1)


def resolve_pose_fly_to_slot(mask_npz, pose_bout_dir, cams, *, sep_px=200.0,
                             thresh=0.5, decisive=0.5, min_frames=20,
                             min_cams=2, min_agree=0.5, min_frame_agree=0.75):
    """{pose fly k: (slot, camera_agreement, n_decisive_frames)} for `mask_npz`.

    There are only two hypotheses -- (fly0, fly1) -> (slot0, slot1) ["identity"]
    or -> (slot1, slot0) ["swapped"] -- so they are scored JOINTLY rather than
    per fly:

        cost_identity = |kp0 - c0| + |kp1 - c1|
        cost_swapped  = |kp0 - c1| + |kp1 - c0|

    per camera (matched BY NAME between the npz's `cameras` array and the
    canonical order `cams` that kp2d's camera axis uses) and per frame where
    both mask centroids are valid and more than `sep_px` apart. A frame votes
    only when the two costs differ by more than `decisive` x that separation.
    That gate is what makes the joint form robust: when both pose flies track
    the SAME animal the two costs are equal, so a collapsed frame abstains
    instead of casting a coin-flip vote. Each camera then votes by its own
    frame majority, and the cameras are majority-voted -- so one camera whose
    SAM3 masks sit on a reflection cannot outvote six good ones just by having
    more frames (the same per-camera robustness as
    jarvis_jax.predict.sam3_driver.sex_male_by_size).

    `camera_agreement` is |sum(votes)| / n_cameras (1.0 unanimous, 0.0 even
    split), matching sex_male_by_size's definition -- NOT a fraction of
    cameras. Measured over this dataset's 160 reviewed bouts it is 1.000 for
    every bout, with 100% frame agreement, so the thresholds below are guards
    against future data rather than tuned knobs.

    Raises Refusal on a missing kp2d, no separated frames, too few decisive
    frames or cameras (the collapse signature), or agreement below threshold.
    """
    with np.load(str(mask_npz), allow_pickle=True) as z:
        names = [str(x) for x in z["cameras"]]
        missing = [c for c in cams if c not in names]
        if missing:
            raise Refusal(f"{mask_npz}: camera(s) {missing} absent from the npz "
                          f"(it stores {names})")
        order = [names.index(c) for c in cams]
        cent = np.asarray(z["centroids"])[:, order]
        val = np.asarray(z["valid"], bool)[:, order]
    if cent.shape[0] < 2:
        raise Refusal(f"{mask_npz}: only {cent.shape[0]} fly slot(s); "
                      f"canonicalization requires 2 flies")

    kp = {}
    for k in (0, 1):
        p = Path(pose_bout_dir) / f"fly{k}" / "kp2d.npz"
        if not p.is_file():
            raise Refusal(f"{pose_bout_dir}: missing fly{k}/kp2d.npz -- cannot "
                          f"measure which mask slot the reviewed fly sits on")
        kp[k] = np.load(p)
    try:
        votes, strengths, n_dec, n_qual = [], [], 0, 0
        for ci in range(len(cams)):
            k0 = _kp_centroids(kp[0], ci, thresh)
            k1 = _kp_centroids(kp[1], ci, thresh)
            T = min(len(k0), len(k1), cent.shape[2])
            sep = np.linalg.norm(cent[0, ci, :T] - cent[1, ci, :T], axis=-1)
            c_id = (np.linalg.norm(k0[:T] - cent[0, ci, :T], axis=-1)
                    + np.linalg.norm(k1[:T] - cent[1, ci, :T], axis=-1))
            c_sw = (np.linalg.norm(k0[:T] - cent[1, ci, :T], axis=-1)
                    + np.linalg.norm(k1[:T] - cent[0, ci, :T], axis=-1))
            qual = (val[0, ci, :T] & val[1, ci, :T] & (sep > sep_px)
                    & np.isfinite(c_id) & np.isfinite(c_sw))
            dec = qual & (np.abs(c_id - c_sw) > decisive * sep)
            n_qual += int(qual.sum())
            nd = int(dec.sum())
            if nd < min_frames:
                continue
            frac_id = int((dec & (c_id < c_sw)).sum()) / nd
            votes.append(1 if frac_id > 0.5 else -1)
            strengths.append(max(frac_id, 1.0 - frac_id))
            n_dec += nd
    finally:
        for f in kp.values():
            f.close()

    if n_qual == 0:
        raise Refusal(
            f"{pose_bout_dir}: no separated frames -- the two mask centroids "
            f"are never more than {sep_px} px apart, so the flies cannot be "
            f"told apart in any view")
    if len(votes) < min_cams:
        raise Refusal(
            f"{pose_bout_dir}: only {len(votes)} camera(s) with >= {min_frames} "
            f"decisive frames (need {min_cams}); {n_qual} frames had the masks "
            f"separated but the two slot assignments cost the same, which is "
            f"the signature of a collapse -- both pose flies tracking one "
            f"animal. Re-review this bout rather than guessing.")

    s = int(np.sum(votes))
    if s == 0:
        raise Refusal(f"{pose_bout_dir}: cameras split evenly ({len(votes)} "
                      f"votes) on which slot each fly is")
    cam_agree = abs(s) / len(votes)
    if cam_agree < min_agree:
        raise Refusal(f"{pose_bout_dir}: camera agreement {cam_agree:.2f} "
                      f"< {min_agree} on the fly -> slot assignment")
    identity = s > 0
    # How firmly did the cameras that voted WITH the majority hold that view?
    # Averaged over those cameras only, and never pooled over frames: pooling
    # would re-weight by frame count and hand the decision back to a single
    # long, broken view -- the thing the per-camera vote exists to prevent.
    win = 1 if identity else -1
    frame_agree = float(np.mean([g for v, g in zip(votes, strengths) if v == win]))
    if frame_agree < min_frame_agree:
        raise Refusal(f"{pose_bout_dir}: the {sum(1 for v in votes if v == win)} "
                      f"agreeing camera(s) are only {frame_agree:.2f} sure "
                      f"frame-by-frame (< {min_frame_agree})")
    slots = (0, 1) if identity else (1, 0)
    return {0: (slots[0], cam_agree, n_dec), 1: (slots[1], cam_agree, n_dec)}


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _atomic_savez(dst, members):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **members)
    os.replace(tmp, dst)


def canonicalize_bout_npz(src, dst, *, male_slot_src, provenance, male_slot=1):
    """Write `dst` = `src` with the male in slot `male_slot`. Never touches `src`.

    `male_slot_src` is the slot the male occupies in `src`, as MEASURED by
    resolve_pose_fly_to_slot -- not an index copied from anywhere else.
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise Refusal(f"{src}: refusing to canonicalize in place; the source "
                      f"mask set is the master and must stay byte-identical")
    if male_slot_src not in (0, 1):
        raise Refusal(f"{src}: male_slot_src={male_slot_src!r}, not 0 or 1")

    members = read_members(src)
    prior = read_sex_meta_member(members)     # capture BEFORE the swap; re-reading
                                              # the npz here would re-load ~1.5 GB
    if "packed" not in members or members["packed"].shape[0] != 2:
        got = None if "packed" not in members else members["packed"].shape[0]
        raise Refusal(f"{src}: needs 2 flies on the mask axis, found {got}")

    applied_swap = male_slot_src != male_slot
    if applied_swap:
        members = swap_fly_axis(members)
    else:
        swap_fly_axis(members)      # classification check even on pass-through

    sex_meta = {
        "male_slot": int(male_slot),
        "status": "swapped" if applied_swap else "kept",
        "method": "human_id_review",
        "applied_swap": bool(applied_swap),
        "original_male_slot": int(male_slot_src),
        "canonicalized_at": _dt.datetime.now(_dt.timezone.utc)
                               .replace(microsecond=0).isoformat(),
        "tool": "scripts/canonicalize_sam_masks.py",
        "source_npz": str(src),
        "prior_sex_meta": prior,
    }
    sex_meta.update(provenance or {})
    members["sex_meta"] = np.array(json.dumps(sex_meta))
    _atomic_savez(dst, members)
    return {"status": sex_meta["status"], "applied_swap": applied_swap,
            "original_male_slot": int(male_slot_src), "dst": str(dst)}


# ---------------------------------------------------------------------------
# source-integrity manifest
# ---------------------------------------------------------------------------

def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def source_manifest(paths):
    """{path: {sha256, size, mtime_ns}} for every source npz."""
    out = {}
    for p in sorted(str(x) for x in paths):
        st = os.stat(p)
        out[p] = {"sha256": sha256_file(p), "size": st.st_size,
                  "mtime_ns": st.st_mtime_ns}
    return out


def diff_manifest(before, after):
    """Human-readable list of differences; empty means byte-identical."""
    diffs = []
    for p in sorted(set(before) | set(after)):
        a, b = before.get(p), after.get(p)
        if a is None:
            diffs.append(f"{p}: APPEARED")
        elif b is None:
            diffs.append(f"{p}: DISAPPEARED")
        elif a["sha256"] != b["sha256"]:
            diffs.append(f"{p}: CONTENT CHANGED {a['sha256'][:12]} -> "
                         f"{b['sha256'][:12]}")
        elif a["size"] != b["size"]:
            diffs.append(f"{p}: SIZE CHANGED {a['size']} -> {b['size']}")
        elif a["mtime_ns"] != b["mtime_ns"]:
            diffs.append(f"{p}: mtime changed (content identical)")
    return diffs


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

def find_mask_dir(src_root, session, recording):
    """The bout-holding mask dir for one recording, in either known layout:
    `<rec>/sam3_masks` (processed tree) or `<rec>/Predictions_3D_sam3*` (video
    tree). Refuses on none or several -- a hand-named Predictions_3D_* dir
    picked by luck is how the pipeline read a stale mask set for weeks."""
    base = Path(src_root) / session / recording
    direct = base / "sam3_masks"
    if direct.is_dir():
        return direct
    cands = sorted(p for p in base.glob("Predictions_3D_sam3*") if p.is_dir())
    if not cands:
        raise Refusal(f"{base}: no sam3_masks/ and no Predictions_3D_sam3* dir")
    if len(cands) > 1:
        raise Refusal(f"{base}: {len(cands)} candidate mask dirs "
                      f"({[c.name for c in cands]}); name one explicitly")
    return cands[0]


# ---------------------------------------------------------------------------
# per-bout worker
# ---------------------------------------------------------------------------

def plan_bout(key, entry, *, src_npz, pose_bout, cams, vote_kw):
    """Decide what to do with one bout. Returns a plan dict; raises Refusal."""
    male_pose_fly = review_male_pose_fly(key, entry)
    if not Path(src_npz).is_file():
        raise Refusal(f"{key}: no mask npz at {src_npz}")
    votes = resolve_pose_fly_to_slot(src_npz, pose_bout, cams, **vote_kw)
    male_slot_src, agree, nframes = votes[male_pose_fly]
    other_slot = votes[1 - male_pose_fly][0]
    return {
        "key": key,
        "src": str(src_npz),
        "male_pose_fly": male_pose_fly,
        "male_slot_src": int(male_slot_src),
        "female_slot_src": int(other_slot),
        "vote_agreement": round(float(agree), 4),
        "vote_frames": int(nframes),
        "action": "swap" if male_slot_src != 1 else "pass",
        "review_status": str(entry.get("status")),
        "review_original_male_fly": entry.get("original_male_fly"),
        "review_reviewed_at": entry.get("reviewed_at"),
    }


def _run_one(job):
    plan, dst, provenance = job
    try:
        res = canonicalize_bout_npz(plan["src"], dst,
                                    male_slot_src=plan["male_slot_src"],
                                    provenance=provenance)
        return {**plan, **res, "ok": True}
    except Refusal as e:
        return {**plan, "ok": False, "refusal": str(e)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", required=True, help="id_review_reviewed_*.json")
    ap.add_argument("--src-root", required=True,
                    help="root holding <Session>/<recording>/{sam3_masks|"
                         "Predictions_3D_sam3*}; READ ONLY")
    ap.add_argument("--pose-root", required=True,
                    help="root holding <Session>/<recording>/pose/bouts/bout_*/"
                         "fly{0,1}/kp2d.npz -- the tree the human reviewed")
    ap.add_argument("--video-root", required=True,
                    help="root holding <Session>/<recording>/calibration "
                         "(defines the canonical camera order)")
    ap.add_argument("--dest", required=True,
                    help="output root; writes <dest>/<Session>/<recording>/"
                         "sam3_masks/bout_*/sam3_masks.npz")
    ap.add_argument("--pose-dir", default="pose")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite outputs that already exist")
    ap.add_argument("--verify-source", action="store_true",
                    help="sha256 every source npz before and after the run and "
                         "report any difference")
    ap.add_argument("--sep-px", type=float, default=200.0,
                    help="min mask-centroid separation for a frame to count")
    ap.add_argument("--decisive", type=float, default=0.5,
                    help="a frame votes only if the two slot assignments differ "
                         "by more than this x the mask separation")
    ap.add_argument("--min-frames", type=int, default=20,
                    help="decisive frames a camera needs to cast a vote")
    ap.add_argument("--min-cams", type=int, default=2)
    ap.add_argument("--min-agree", type=float, default=0.5,
                    help="min |sum(camera votes)| / n_cameras")
    ap.add_argument("--min-frame-agree", type=float, default=0.75,
                    help="min mean frame-level certainty of the agreeing cameras")
    a = ap.parse_args(argv)

    review_path = Path(a.review)
    review_doc = json.loads(review_path.read_text())
    review = review_doc.get("bouts", review_doc)
    convention = review_doc.get("convention")
    if convention and convention != {"female": 0, "male": 1}:
        raise SystemExit(f"unexpected review convention {convention}; this "
                         f"script canonicalizes to male = fly1")
    provenance_base = {
        "review_file": review_path.name,
        "review_path": str(review_path),
        "review_sha256": sha256_file(review_path),
        "review_convention": convention,
    }

    dest = Path(a.dest)
    vote_kw = dict(sep_px=a.sep_px, decisive=a.decisive, min_frames=a.min_frames,
                   min_cams=a.min_cams, min_agree=a.min_agree,
                   min_frame_agree=a.min_frame_agree)

    plans, refusals, cams_cache = [], [], {}
    for key in sorted(review):
        session, rec, bname = key.split("/")
        entry = review[key]
        try:
            cal = Path(a.video_root) / session / rec / "calibration"
            if cal not in cams_cache:
                cams_cache[cal] = canonical_cameras(cal)
            cams = cams_cache[cal]
            src_npz = find_mask_dir(a.src_root, session, rec) / bname / "sam3_masks.npz"
            pose_bout = (Path(a.pose_root) / session / rec / a.pose_dir
                         / "bouts" / bname)
            plan = plan_bout(key, entry, src_npz=src_npz, pose_bout=pose_bout,
                             cams=cams, vote_kw=vote_kw)
        except Refusal as e:
            refusals.append({"key": key, "refusal": str(e)})
            print(f"REFUSE  {key}: {e}", flush=True)
            continue
        plan["dst"] = str(dest / session / rec / "sam3_masks" / bname
                          / "sam3_masks.npz")
        plans.append(plan)
        print(f"{plan['action'].upper():5s}  {key}  male=pose fly"
              f"{plan['male_pose_fly']} -> src slot {plan['male_slot_src']} "
              f"(agree {plan['vote_agreement']:.3f}, n={plan['vote_frames']})"
              f"  review={plan['review_status']}", flush=True)

    n_swap = sum(1 for p in plans if p["action"] == "swap")
    print(f"\n{len(plans)} bout(s) to write: {n_swap} swap, "
          f"{len(plans) - n_swap} pass-through; {len(refusals)} refused")

    if a.dry_run:
        print("\n(dry run -- nothing written)")
        return {"plans": plans, "refusals": refusals}

    before = source_manifest([p["src"] for p in plans]) if a.verify_source else None

    jobs = [(p, p["dst"], provenance_base) for p in plans
            if a.overwrite or not Path(p["dst"]).exists()]
    print(f"writing {len(jobs)} bout(s) with {a.jobs} worker(s) "
          f"({len(plans) - len(jobs)} already present)", flush=True)
    results = []
    if jobs:
        if a.jobs > 1:
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                for r in ex.map(_run_one, jobs):
                    results.append(r)
                    print(f"  {r['status'] if r['ok'] else 'REFUSE':8s} "
                          f"{r['key']}", flush=True)
        else:
            for j in jobs:
                r = _run_one(j)
                results.append(r)
                print(f"  {r['status'] if r['ok'] else 'REFUSE':8s} {r['key']}",
                      flush=True)
    for r in results:
        if not r["ok"]:
            refusals.append({"key": r["key"], "refusal": r["refusal"]})

    manifest = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc)
                           .replace(microsecond=0).isoformat(),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "review": provenance_base,
        "src_root": str(a.src_root),
        "pose_root": str(a.pose_root),
        "dest": str(dest),
        "vote": vote_kw,
        "n_swapped": sum(1 for r in results if r.get("status") == "swapped"),
        "n_kept": sum(1 for r in results if r.get("status") == "kept"),
        "n_refused": len(refusals),
        "bouts": {p["key"]: p for p in plans},
        "refusals": refusals,
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "canonicalization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nwrote {manifest['n_swapped']} swapped + {manifest['n_kept']} kept "
          f"-> {dest}\nmanifest: {dest / 'canonicalization_manifest.json'}")

    if before is not None:
        after = source_manifest([p["src"] for p in plans])
        diffs = diff_manifest(before, after)
        if diffs:
            print("\nSOURCE CHANGED -- this must never happen:")
            for d in diffs:
                print("   " + d)
            raise SystemExit(2)
        print(f"\nsource verified byte-identical: {len(before)} npz, sha256 "
              f"unchanged")
    return manifest


if __name__ == "__main__":
    main()
