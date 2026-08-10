#!/usr/bin/env python3
"""Combine per-bout IK outputs into one analysis h5.

The pipeline writes one `stac_ik.h5` + `outputs.h5` per (bout, fly). Analysis
wants a single file in the shape of
`Data_analysis/analysis/v1/ik_output_combined_*.h5`: a `bout_NNN` group per
bout carrying the IK arrays, plus an `info` group of parallel metadata lists.

Written with stac_mjx's `io_dict_to_hdf5.save`, the same writer that produced
the reference files, so nested dicts/lists land in the same on-disk layout
(lists become numbered subgroups -- that is the writer's convention, not a
quirk of this script).

Metadata carried per bout, which the reference format did not have:
recording, bout index, absolute start/end frame, fly slot, the male_fly from
sex.json where a human verified it, and whether the bout passes the
reconstructability gate. That is what lets a downstream analysis exclude the
10 unusable bouts and know which fly is which without re-deriving it.

Usage:
    python scripts/export/combine_ik_outputs.py --fly 0 --out <path>.h5
    python scripts/export/combine_ik_outputs.py --fly both --skip-failing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"
for p in (str(PROJECT_DIR), str(PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship"
VIDEO_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship"

# Straight from stac_ik.h5 -- the IK solve's own outputs.
IK_KEYS = ("qpos", "qvel", "kp_data", "marker_sites", "xpos", "xquat")
# Per-file metadata that is identical across bouts; taken from the first.
SHARED_KEYS = ("kp_names", "names_qpos", "names_xpos", "offsets")


def bout_start_end(csv, session_dir, bout_idx, _cache={}):
    from jarvis_jax.predict.sam3_driver import parse_bouts, session_tag_for
    key = (csv, session_dir)
    if key not in _cache:
        _cache[key] = {int(b["bout_idx"]): (int(b["start"]), int(b["end"]))
                       for b in parse_bouts(csv, session_tag_for(session_dir))}
    return _cache[key].get(int(bout_idx), (-1, -1))


def main(argv=None):
    import h5py
    import stac_mjx.io_dict_to_hdf5 as ioh5

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--pose-dir", default="pose")
    ap.add_argument("--fly", default="0", choices=["0", "1", "both"])
    ap.add_argument("--skip-failing", action="store_true",
                    help="omit bouts that fail the reconstructability gate")
    # regeneration_report.json, not bout_reconstructable.json: the latter has
    # no `dropped_bouts` key (its verdicts live under `bouts`), and its numbers
    # predate the mask regeneration. Pointing at it silently gated nothing.
    ap.add_argument("--qc-report", default="docs/qc/regeneration_report.json")
    # The GUI's `bad` verdict is a TRACKING-quality judgement ("the female is
    # mistracked here"), separate from the reconstructability gate (a geometric
    # coverage test) and from `unsure` (identity unknown but pose may be fine).
    # A bad bout is excluded under --skip-failing; an unsure one is kept, with
    # male_fly = -1 so no analysis can silently assume a sex for it.
    ap.add_argument("--review-manifest", default=None,
                    help="id_review manifest (default: <root>/id_review_<pose-dir>.json, "
                         "falling back to <root>/id_review.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    failing = set()
    qc = Path(a.qc_report)
    if a.skip_failing:
        # Fail loudly rather than silently gating nothing. docs/qc/ is not
        # tracked (regenerable artifacts), so a missing report is the LIKELY
        # case on a fresh checkout -- and an earlier run of this script
        # reported "0 bouts skipped" against a report that had no
        # `dropped_bouts` key, quietly including all 10 unusable bouts.
        if not qc.is_file():
            raise SystemExit(
                f"--skip-failing needs {qc}, which does not exist.\n"
                f"Regenerate it with:\n"
                f"  python scripts/qc/verify_regeneration.py --new sam3_masks")
        rep = json.loads(qc.read_text())
        if "dropped_bouts" not in rep:
            raise SystemExit(
                f"{qc} has no `dropped_bouts` key -- wrong report type. "
                f"Use the output of scripts/qc/verify_regeneration.py, not "
                f"bout_reconstructable.py (whose verdicts live under `bouts`).")
        failing.update(rep["dropped_bouts"])         # "Sess/rec#bout"
        print(f"reconstructability gate: {len(failing)} bouts will be skipped")

    flies = [0, 1] if a.fly == "both" else [int(a.fly)]
    root = Path(a.root)

    # Review verdicts, keyed 'Session0/rec/bout_00001'.
    rm = Path(a.review_manifest) if a.review_manifest else None
    if rm is None:
        for cand in (root / f"id_review_{a.pose_dir}.json", root / "id_review.json"):
            if cand.is_file():
                rm = cand
                break
    review = {}
    if rm and rm.is_file():
        raw = json.loads(rm.read_text())
        review = raw.get("bouts", raw)
        print(f"review manifest: {rm} ({len(review)} bouts)")
    else:
        print("review manifest: NONE FOUND -- review_status will be 'unreviewed' "
              "for every bout and no bout can be excluded as bad")

    out = {}
    info = {k: [] for k in ("clip_lengths", "fly_ids", "source_flies", "bucket",
                            "recordings", "bout_indices", "start_frames",
                            "end_frames", "fly_slots", "male_fly",
                            "sex_verified", "reconstructable", "review_status")}
    shared = {}
    n = 0
    skipped = 0
    skipped_bad = 0

    for sess in sorted(p.name for p in root.iterdir() if p.is_dir()):
        sdir = root / sess
        if not sdir.is_dir():
            continue
        for rec in sorted(p.name for p in sdir.iterdir() if p.is_dir()):
            bouts_dir = sdir / rec / a.pose_dir / "bouts"
            if not bouts_dir.is_dir():
                continue
            csv = str(sdir / rec / "courtship_bout_summary.csv")
            sd = os.path.join(VIDEO_ROOT, sess, rec)
            for bdir in sorted(bouts_dir.glob("bout_*")):
                try:
                    bidx = int(bdir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                tag = f"{sess}/{rec}#{bidx}"
                rstat = review.get(f"{sess}/{rec}/{bdir.name}", {}).get(
                    "status", "unreviewed")
                ok = tag not in failing
                if a.skip_failing and not ok:
                    skipped += 1
                    continue
                if a.skip_failing and rstat == "bad":
                    skipped_bad += 1
                    continue
                sexp = bdir / "sex.json"
                male, verified = -1, False
                if sexp.is_file():
                    try:
                        j = json.loads(sexp.read_text())
                        male = int(j.get("male_fly", -1))
                        verified = j.get("confidence") == "user"
                    except Exception:
                        pass
                # An unreviewed/unsure bout has no trustworthy identity: report
                # -1 rather than the mask-area heuristic's guess, so downstream
                # code cannot mistake a vote for a verified sex.
                if rstat in ("unsure", "bad", "unreviewed") and not verified:
                    male = -1
                start, end = bout_start_end(csv, sd, bidx)
                for fly in flies:
                    ik = bdir / f"fly{fly}" / "stac_ik.h5"
                    outs = bdir / f"fly{fly}" / "outputs.h5"
                    if not ik.is_file():
                        continue
                    d = {}
                    with h5py.File(ik, "r") as f:
                        for k in IK_KEYS:
                            if k in f:
                                d[k] = np.asarray(f[k])
                        for k in SHARED_KEYS:
                            if k in f and k not in shared:
                                v = np.asarray(f[k])
                                if v.dtype.kind == "S":
                                    v = [x.decode() for x in v.ravel().tolist()]
                                shared[k] = v
                    if outs.is_file():
                        with h5py.File(outs, "r") as f:
                            # FK'd site positions in world mm -- the reference
                            # format's `site_xpos`.
                            if "kp3d_mm" in f:
                                d["site_xpos"] = np.asarray(f["kp3d_mm"])
                            for k in ("root_se3", "scale"):
                                if k in f:
                                    d[k] = np.asarray(f[k])
                    if "qpos" not in d:
                        continue
                    out[f"bout_{n:03d}"] = d
                    T = int(d["qpos"].shape[0])
                    info["clip_lengths"].append(T)
                    info["fly_ids"].append(f"{sess}/{rec}/bout{bidx:05d}/fly{fly}")
                    info["source_flies"].append(f"fly{fly}")
                    info["bucket"].append(sess)
                    info["recordings"].append(f"{sess}/{rec}")
                    info["bout_indices"].append(bidx)
                    info["start_frames"].append(start)
                    info["end_frames"].append(end)
                    info["fly_slots"].append(fly)
                    info["male_fly"].append(male)
                    info["sex_verified"].append(bool(verified))
                    info["reconstructable"].append(bool(ok))
                    info["review_status"].append(str(rstat))
                    n += 1

    if not n:
        raise SystemExit("no bouts found -- check --root/--pose-dir/--fly")
    info.update(shared)
    out["info"] = info

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    ioh5.save(str(outp), out)
    mb = outp.stat().st_size / 1e6
    print(f"wrote {outp}  ({n} bout-flies, {sum(info['clip_lengths'])} frames, {mb:.1f} MB)")
    print(f"  skipped (reconstructability): {skipped}")
    print(f"  skipped (review status=bad):  {skipped_bad}")
    from collections import Counter
    print(f"  review status: {dict(Counter(info['review_status']))}")
    print(f"  sex verified: {sum(info['sex_verified'])}/{n}")
    return out


if __name__ == "__main__":
    main()
