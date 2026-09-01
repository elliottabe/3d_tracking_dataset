#!/usr/bin/env python3
"""Emit per-bout ``sex.json`` from the human fly-identity review.

Why this exists: ``scale.json``'s per-fly body scale is gated on
``_determine_identity`` finding a ``sex.json`` in EVERY bout dir, all agreeing
on ``male_fly``. Without it the fly0/fly1 slot is assigned per bout by the
tracker and is not a stable individual label -- measured counter-example, an
uncanonicalized recording where fly0 > fly1 in bout 12 and fly0 < fly1 in bout
17 -- so pooling per slot blends the two animals. When identity is unresolved
the scale stage falls back to ONE scale shared by both flies.

The review (``id_review_reviewed_*.json``, 160 bouts) is the human answer and
the durable source of truth. ``sex.json`` is derived from it, so a fresh
run_root re-derives rather than depending on a backup directory surviving.

    python scripts/apply_id_review.py --review <id_review_reviewed_*.json> \
        --run-root <.../pose> --recording Session0/2025_10_20_13_20_04
    python scripts/apply_id_review.py --review <...> --root <.../processed/courtship>

NOTE this writes only ``sex.json``; it performs no directory swap. Every
reviewed bout in this dataset has ``applied: false`` (i.e. the tracker's
original assignment already matched the reviewer's), so a swap would be a
no-op -- but if a future review disagrees, this refuses rather than guessing.
Use ``scripts/canonicalize_session_sex.py`` for the physical swap.
"""
import argparse
import json
import os
import re

_BOUT_RE = re.compile(r"^bout_\d{5}$")


def sex_entry(rec_bout, entry):
    """Review entry -> sex.json payload, or raise if it cannot be trusted."""
    male = entry.get("reviewed_male_fly", entry.get("prior_reviewed_male_fly"))
    if isinstance(male, bool) or not isinstance(male, int) or male not in (0, 1):
        raise ValueError(f"{rec_bout}: no usable reviewed_male_fly ({male!r})")
    status = str(entry.get("status", entry.get("prior_status", "unknown")))
    if status != "confirmed":
        raise ValueError(f"{rec_bout}: review status is {status!r}, not 'confirmed'")
    orig = entry.get("original_male_fly")
    if entry.get("applied") and orig is not None and orig != male:
        # A swap was recorded as already applied to the DIRECTORIES. Re-emitting
        # sex.json against a freshly-run run_root (whose dirs were never
        # swapped) would then label the wrong animal. Refuse loudly.
        raise ValueError(
            f"{rec_bout}: review says a directory swap was applied "
            f"(original_male_fly={orig}, reviewed={male}); this script only "
            f"writes labels. Run scripts/canonicalize_session_sex.py first.")
    return {
        "male_fly": male,
        "original_male_fly": orig if orig is not None else male,
        "applied_swap": bool(entry.get("applied", False)),
        "confidence": "user",
        "method": "manual-gui",
        "note": f"fly_id_review {entry.get('reviewed_at', '?')} status={status}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--run-root", help="a single recording's pose/ dir")
    ap.add_argument("--recording", help="e.g. Session0/2025_10_20_13_20_04; "
                    "required with --run-root")
    ap.add_argument("--root", help="processed/courtship root; walks every "
                    "<Session>/<rec>/pose/bouts it finds")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing sex.json (default: leave it)")
    args = ap.parse_args()

    with open(args.review) as f:
        review = json.load(f)["bouts"]

    targets = []                       # (recording, run_root)
    if args.run_root:
        if not args.recording:
            raise SystemExit("--run-root needs --recording")
        targets.append((args.recording, args.run_root))
    elif args.root:
        for rec in sorted({"/".join(k.split("/")[:2]) for k in review}):
            rr = os.path.join(args.root, rec, "pose")
            if os.path.isdir(os.path.join(rr, "bouts")):
                targets.append((rec, rr))
    else:
        raise SystemExit("pass --run-root (+--recording) or --root")

    n_written = n_skipped = n_missing = 0
    for rec, run_root in targets:
        bouts_dir = os.path.join(run_root, "bouts")
        for name in sorted(os.listdir(bouts_dir)):
            if not _BOUT_RE.match(name):
                continue
            bout_dir = os.path.join(bouts_dir, name)
            key = f"{rec}/{name}"
            entry = review.get(key)
            if entry is None:
                # A bout the human never reviewed. Do NOT invent a label --
                # an unreviewed bout with a guessed sex.json would make
                # _determine_identity report "canonical" on a fabrication.
                print(f"  MISSING REVIEW {key}")
                n_missing += 1
                continue
            out = os.path.join(bout_dir, "sex.json")
            if os.path.exists(out) and not args.overwrite:
                n_skipped += 1
                continue
            payload = sex_entry(key, entry)
            if args.dry_run:
                print(f"  would write {out}: male_fly={payload['male_fly']}")
            else:
                tmp = out + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp, out)
            n_written += 1
        print(f"[{rec}] {n_written} written, {n_skipped} kept, {n_missing} unreviewed")

    if n_missing:
        print(f"\nWARNING: {n_missing} bout(s) had no review entry and got NO "
              f"sex.json. Identity will read 'unknown' for those recordings and "
              f"the per-fly scale will not engage -- review them or exclude "
              f"them, do not fabricate a label.")


if __name__ == "__main__":
    main()
