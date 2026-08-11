#!/usr/bin/env python3
"""Flag unusable tracking PER FLY, not per bout.

Human review of all 160 courtship bouts produced one verdict per bout, but the
finding it produced was per fly: "several with bad female tracking, all the male
tracking looks good." The GUI's `bad` status cannot express that -- marking a
bout bad to exclude a broken female would also discard a male fit that is fine.
This computes a per-(bout, fly) verdict so the male survives.

THE GATE: LOO reprojection > 30 px, or a NaN per-camera reprojection median.

Chosen against rendered frames, not from the distribution alone. Ordering female
bout-flies by per-camera reprojection error and looking at
figures/2026-08-10-per-fly-quality/female_quality_examples.png:

  * 112 px and 44 px -- keypoints split between the fly and a detached wall or
    reflection mask, skeleton stretched across the frame. Broken.
  * 28 px -- keypoints sit ON the animal and look usable.

So a 25 px per-camera-reproj gate (which the male distribution alone would
justify: male max is 22.6 px) rejects usable data. LOO separates the same cases
correctly -- 13.7 px for the usable 28 px case versus 59 and 143 px for the two
broken ones -- because leave-one-out error measures cross-camera DISAGREEMENT,
which is what a keypoint stranded on a reflection produces. NaN is included
because a missing median is a failure to compute, not a pass.

Validation: the one bout a human independently marked `bad`
(Session1/2026_04_02_15_25_51 bout 8) has female LOO 59.3 px / conf 0.73 against
male 5.1 px / conf 0.96, and the gate flags the female while clearing the male.
Across the set it flags 36 of 160 females and 2 of 160 males, matching the
reported asymmetry.

Usage:
    python scripts/qc/per_fly_quality.py --pose-dir pose
    python scripts/qc/per_fly_quality.py --pose-dir pose --json-out docs/qc/per_fly_quality.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

PROC = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship"


def fly_metrics(bout_fly_dir: Path) -> dict | None:
    """Per-fly QC numbers, or None if the fly was not solved."""
    qc = bout_fly_dir / "qc.json"
    kp = bout_fly_dir / "kp2d.npz"
    if not qc.is_file():
        return None
    q = json.loads(qc.read_text())
    conf = float("nan")
    if kp.is_file():
        with np.load(kp) as z:
            conf = float(np.median(z["conf"]))
    return {
        "reproj_px": q.get("per_camera_reproj_px", {}).get("median", float("nan")),
        "loo_px": q.get("loo_reproj_px", {}).get("median", float("nan")),
        "median_conf": conf,
        "n_frames": q.get("n_frames", 0),
    }


def verdict(m: dict, loo_max: float = 30.0) -> tuple[bool, list[str]]:
    """(usable, reasons_it_is_not). See the gate rationale in the docstring."""
    reasons = []
    if not np.isfinite(m["reproj_px"]):
        reasons.append("per-camera reproj median is NaN")
    if np.isfinite(m["loo_px"]) and m["loo_px"] > loo_max:
        reasons.append(f"LOO reproj {m['loo_px']:.0f}px > {loo_max:.0f}px")
    elif not np.isfinite(m["loo_px"]):
        reasons.append("LOO reproj is NaN")
    return (not reasons), reasons


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=PROC)
    ap.add_argument("--pose-dir", default="pose")
    ap.add_argument("--loo-max", type=float, default=30.0)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)

    root = Path(a.root)
    mpath = root / (f"id_review_{a.pose_dir}.json" if a.pose_dir != "pose"
                    else "id_review.json")
    review = {}
    if mpath.is_file():
        raw = json.loads(mpath.read_text())
        review = raw.get("bouts", raw)
    else:
        print(f"note: no review manifest at {mpath}; sex labels unavailable, "
              f"every fly reported as sex='unknown'")

    out, tally = {}, Counter()
    for sess in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for rec in sorted(p.name for p in (root / sess).iterdir() if p.is_dir()):
            bdir = root / sess / rec / a.pose_dir / "bouts"
            if not bdir.is_dir():
                continue
            for b in sorted(bdir.glob("bout_*")):
                key = f"{sess}/{rec}/{b.name}"
                ent = review.get(key, {})
                # An undecided bout has no trustworthy sex; do not guess one.
                male = (int(ent["reviewed_male_fly"])
                        if ent.get("status") in ("confirmed", "swapped") else None)
                for fly in (0, 1):
                    m = fly_metrics(b / f"fly{fly}")
                    if m is None:
                        continue
                    ok, reasons = verdict(m, a.loo_max)
                    sex = ("unknown" if male is None
                           else ("male" if fly == male else "female"))
                    out[f"{key}/fly{fly}"] = {
                        **m, "sex": sex, "usable": ok, "reasons": reasons,
                        "review_status": ent.get("status", "unreviewed"),
                    }
                    tally[f"{sex}:{'usable' if ok else 'BAD'}"] += 1

    print(f"{len(out)} bout-flies under {a.pose_dir}/  "
          f"(gate: LOO > {a.loo_max:.0f}px or NaN)")
    for k in sorted(tally):
        print(f"   {k:18s} {tally[k]}")
    badf = {k.rsplit('/', 1)[0] for k, v in out.items()
            if not v["usable"] and v["sex"] == "female"}
    badm = {k.rsplit('/', 1)[0] for k, v in out.items()
            if not v["usable"] and v["sex"] == "male"}
    print(f"   bouts where only the female is unusable "
          f"(male still good): {len(badf - badm)}")

    if a.json_out:
        p = Path(a.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"gate": {"loo_max_px": a.loo_max, "reproj_nan_is_bad": True},
             "pose_dir": a.pose_dir, "flies": out}, indent=2, sort_keys=True))
        print(f"wrote {p}")
    return out


if __name__ == "__main__":
    main()
