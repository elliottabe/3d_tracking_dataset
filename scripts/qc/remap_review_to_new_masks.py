#!/usr/bin/env python3
"""Carry human fly-ID review decisions from the old pose tree onto a new one.

WHY A REMAP IS NEEDED AT ALL. The review GUI shows `pose/.../fly<k>/sidebyside.mp4`,
whose crop window is computed from that fly's `kp2d` (viz/views/sidebyside.py).
So the reviewer judged the animal the KEYPOINTS were on -- not the animal in mask
slot k. In the old tree those disagreed for many bouts (`predictions_dir` pointed
at a stale mask set: measured 38.9% wrong-fly, eight bouts fully swapped), and
the mask regeneration independently reshuffled slots. Replaying a decision by
slot index would therefore silently mislabel sex on exactly the bouts that were
already broken.

HOW. Rather than chain two error-prone mappings (old-video -> stale slot ->
new slot), measure the composite directly: for each old fly k, find which NEW
mask slot its keypoint centroid actually sits on, per camera per frame, and vote.
`reviewed_male_fly = k` then becomes `male_fly = vote[k]` in the new tree.

Bouts where the vote is unusable are NOT guessed. They are written to the new
manifest with status `unsure` so the GUI re-serves them:
  * both old flies voted onto the same new slot (the collapse case -- the
    reviewer saw one animal twice, so the decision cannot be recovered)
  * fewer than --min-frames separated frames to vote on
  * agreement below --min-agree

Usage:
    python scripts/qc/remap_review_to_new_masks.py --dry-run
    python scripts/qc/remap_review_to_new_masks.py --apply
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "third_party" / "jarvis_jax"))
sys.path.insert(0, str(PROJECT_DIR))

PROC = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship"
VIDEO_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship"
BACKUP = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship_review_2026-08-09"


def vote_old_fly_to_new_slot(mask_npz, old_pose, bout, cams,
                             sep_px=200.0, thresh=0.5, margin=0.35):
    """{old_fly: (new_slot, agreement, n_frames)} from kp-centroid proximity."""
    from scripts.qc.audit_fly_assignment import kp_centroids
    with np.load(mask_npz, allow_pickle=True) as z:
        order = [[str(x) for x in z["cameras"]].index(c) for c in cams]
        cent = np.asarray(z["centroids"])[:, order]
        val = np.asarray(z["valid"], bool)[:, order]
    if cent.shape[0] < 2:
        return {}
    out = {}
    for k in (0, 1):
        p = Path(old_pose) / "bouts" / f"bout_{bout:05d}" / f"fly{k}" / "kp2d.npz"
        if not p.is_file():
            return {}
        with np.load(p) as kz:
            votes = []
            for ci in range(len(cams)):
                kc = kp_centroids(kz, ci, thresh)
                T = min(len(kc), cent.shape[2])
                sep = np.linalg.norm(cent[0, ci, :T] - cent[1, ci, :T], axis=-1)
                d0 = np.linalg.norm(kc[:T] - cent[0, ci, :T], axis=-1)
                d1 = np.linalg.norm(kc[:T] - cent[1, ci, :T], axis=-1)
                s = (val[0, ci, :T] & val[1, ci, :T] & (sep > sep_px)
                     & np.isfinite(d0) & np.isfinite(d1)
                     & (np.minimum(d0, d1) < margin * sep))
                if s.any():
                    votes.append((int((s & (d1 < d0)).sum()), int(s.sum())))
        n1 = sum(v[0] for v in votes); n = sum(v[1] for v in votes)
        if not n:
            return {}
        slot = 1 if n1 * 2 > n else 0
        agree = (n1 if slot == 1 else n - n1) / n
        out[k] = (slot, agree, n)
    return out


def main(argv=None):
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=PROC)
    ap.add_argument("--old-pose", default="pose")
    ap.add_argument("--new-pose", default="pose_v3")
    ap.add_argument("--review", default=os.path.join(BACKUP, "id_review.json"))
    ap.add_argument("--min-frames", type=int, default=20)
    ap.add_argument("--min-agree", type=float, default=0.90)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not (a.apply or a.dry_run):
        raise SystemExit("pass --dry-run or --apply")

    review = json.loads(Path(a.review).read_text())
    review = review.get("bouts", review)
    stamp = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()

    cams_cache, new_manifest, tally = {}, {}, Counter()
    detail = []
    for key, ent in sorted(review.items()):
        sess, rec, bname = key.split("/")
        bout = int(bname.split("_")[1])
        cal = os.path.join(VIDEO_ROOT, sess, rec, "calibration")
        if cal not in cams_cache:
            cams_cache[cal] = (list(ReprojectionTool(cal).cameras.keys())
                               if os.path.isdir(cal) else None)
        cams = cams_cache[cal]
        recdir = os.path.join(a.root, sess, rec)
        mask = os.path.join(recdir, "sam3_masks", f"bout_{bout:05d}", "sam3_masks.npz")
        status = ent.get("status")
        rmf = ent.get("reviewed_male_fly")

        v = (vote_old_fly_to_new_slot(mask, os.path.join(recdir, a.old_pose),
                                      bout, cams)
             if cams and os.path.isfile(mask) else {})
        reason = None
        if status in ("unsure", "bad"):
            reason = f"kept status={status}"
        elif not v:
            reason = "no vote (missing kp2d/mask or no separated frames)"
        elif v[0][0] == v[1][0]:
            reason = f"both old flies -> new slot {v[0][0]} (collapse)"
        elif min(v[0][2], v[1][2]) < a.min_frames:
            reason = f"only {min(v[0][2], v[1][2])} separated frames"
        elif min(v[0][1], v[1][1]) < a.min_agree:
            reason = f"agreement {min(v[0][1], v[1][1]):.2f} < {a.min_agree}"

        new_ent = dict(ent)
        if reason and status not in ("unsure", "bad"):
            new_ent["status"] = "unsure"
            new_ent["remap_note"] = f"needs re-review: {reason}"
            tally["needs_review"] += 1
        elif status in ("unsure", "bad"):
            tally[f"kept_{status}"] += 1
        else:
            new_slot = v[rmf][0]
            new_ent["reviewed_male_fly"] = new_slot
            new_ent["original_male_fly"] = new_slot if status == "confirmed" else 1 - new_slot
            new_ent["remap_note"] = (
                f"remapped from {a.old_pose} fly{rmf} -> slot {new_slot} "
                f"(agree {v[rmf][1]:.2f}, n={v[rmf][2]})")
            tally["remapped_same" if new_slot == rmf else "remapped_flipped"] += 1
            detail.append((key, rmf, new_slot, v[rmf][1], v[rmf][2]))
        new_manifest[key] = new_ent

    print(f"{len(review)} reviewed bouts -> {a.new_pose}")
    for k, n in sorted(tally.items()):
        print(f"   {k:20s} {n}")
    flips = [d for d in detail if d[1] != d[2]]
    print(f"\n   slot unchanged: {len(detail)-len(flips)}   flipped: {len(flips)}")
    for key, o, n, ag, nf in flips[:10]:
        print(f"      {key:44s} fly{o} -> slot{n}  (agree {ag:.2f}, n={nf})")

    if not a.apply:
        print("\n(dry run -- nothing written)")
        return new_manifest

    written = 0
    for key, ent in new_manifest.items():
        sess, rec, bname = key.split("/")
        bdir = Path(a.root) / sess / rec / a.new_pose / "bouts" / bname
        if not bdir.is_dir():
            continue
        if ent.get("status") in ("confirmed", "swapped"):
            (bdir / "sex.json").write_text(json.dumps({
                "male_fly": int(ent["reviewed_male_fly"]),
                "original_male_fly": int(ent["original_male_fly"]),
                "applied_swap": False,
                "confidence": "user",
                "method": "manual-gui-remapped",
                "note": f"remapped from {a.old_pose} {stamp}; "
                        f"{ent.get('remap_note', '')}",
            }, indent=2))
            written += 1
    # MUST be the GUI's schema: {root, convention, bouts}. A bare
    # {bout_key: entry} mapping made fly_id_review's load_manifest see no prior
    # decisions and overwrite the file with 160 pending entries.
    out = Path(a.root) / f"id_review_{a.new_pose}.json"
    out.write_text(json.dumps({
        "root": str(a.root),
        "convention": {"female": 0, "male": 1},
        "bouts": new_manifest,
    }, indent=2, sort_keys=True))
    print(f"\nwrote {written} sex.json into {a.new_pose}/ and {out}")
    return new_manifest


if __name__ == "__main__":
    main()
