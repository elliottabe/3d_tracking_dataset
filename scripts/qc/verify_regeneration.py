#!/usr/bin/env python3
"""Verify a regenerated SAM3 mask set before promoting it over the old one.

Regeneration is NOT deterministic -- repeat runs of Session0 bout 22 tracked
fly0 in 2 cameras where the stored masks had 3, and the camera they dropped was
the one tracking a REFLECTION. So "fewer valid views" can mean better data, and
the decision to promote has to look at what changed rather than at totals.

Reports, per recording and overall:

  1. completeness   -- every expected bout produced an npz
  2. visibility     -- the in_frame split (out-of-FOV vs in-frame-and-missed),
                       which the old masks could not report at all
  3. reflections    -- cameras whose slot hugs the other fly (find_reflection_cameras)
  4. reconstructability -- the per-bout keep/drop verdict
  5. identity       -- sex_meta status recorded by canonicalize_male_fly
  6. old-vs-new     -- for recordings that had masks before, what changed

Read-only. Promotion is a separate, deliberate step.

Usage:
    python scripts/qc/verify_regeneration.py [--new sam3_masks_v2] [--old sam3_masks]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"
for p in (str(PROJECT_DIR), str(PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship"
VIDEO_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship"


def calib_for(session, recording):
    d = os.path.join(VIDEO_ROOT, session, recording, "calibration")
    return d if os.path.isdir(d) else None


def load_npz(path, cam_order):
    """centroids/valid reordered into `cam_order` via the npz's own `cameras`.

    New files carry a `cameras` array whose order is NOT the calibration's
    lexicographic one; ignoring it compares different cameras to each other.
    """
    with np.load(path, allow_pickle=True) as z:
        val = np.asarray(z["valid"], bool)
        cent = np.asarray(z["centroids"])
        names = [str(x) for x in z["cameras"]] if "cameras" in z else None
        infr = np.asarray(z["in_frame"]) if "in_frame" in z else None
        shape = np.asarray(z["shape"])[:2] if "shape" in z else None
    if names and cam_order:
        perm = [names.index(c) for c in cam_order if c in names]
        if len(perm) == len(cam_order):
            val = val[:, perm]; cent = cent[:, perm]
            if infr is not None:
                infr = infr[:, perm]
    return cent, val, infr, shape


def main(argv=None):
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.predict.sam3_driver import (
        IN_FRAME_NO, IN_FRAME_UNKNOWN, IN_FRAME_YES, find_reflection_cameras)
    from jarvis_jax.predict.sam3_driver import parse_bouts, session_tag_for
    from scripts.qc.bout_reconstructable import bout_verdict, coverage_from_valid

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--new", default="sam3_masks_v2")
    ap.add_argument("--old", default="sam3_masks")
    ap.add_argument("--min-views", type=int, default=4)
    ap.add_argument("--min-frac", type=float, default=0.8)
    ap.add_argument("--out", default="docs/qc/regeneration_report.json")
    args = ap.parse_args(argv)

    root = Path(args.root)
    report = {"recordings": {}, "totals": {}}
    tot = Counter()
    all_drop, all_refl, missing = [], [], []

    for sess in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for rec in sorted(p.name for p in (root / sess).iterdir() if p.is_dir()):
            newdir = root / sess / rec / args.new
            if not newdir.is_dir():
                continue
            csv = root / sess / rec / "courtship_bout_summary.csv"
            sd = os.path.join(VIDEO_ROOT, sess, rec)
            try:
                expected = [int(b["bout_idx"]) for b in
                            parse_bouts(str(csv), session_tag_for(sd))]
            except Exception:
                expected = []
            cal = calib_for(sess, rec)
            rt = ReprojectionTool(cal) if cal else None
            cams = list(rt.cameras.keys()) if rt else []

            got, rr = [], {"bouts": {}, "missing": [], "reflections": [],
                           "drops": [], "sex": Counter()}
            for b in expected:
                p = newdir / f"bout_{b:05d}" / "sam3_masks.npz"
                if not p.is_file():
                    rr["missing"].append(b); missing.append((sess, rec, b)); continue
                got.append(b)
                cent, val, infr, shape = load_npz(p, cams)
                A = val.shape[0]
                fr = [coverage_from_valid(val[a], args.min_views) for a in range(A)]
                keep, reason = bout_verdict(fr, args.min_frac)
                if not keep:
                    rr["drops"].append({"bout": b, "fractions": [round(f, 3) for f in fr]})
                    all_drop.append((sess, rec, b))
                inv = ~val
                vis = {}
                if infr is not None:
                    vis = {"miss": int((inv & (infr == IN_FRAME_YES)).sum()),
                           "oof": int((inv & (infr == IN_FRAME_NO)).sum()),
                           "unknown": int((inv & (infr == IN_FRAME_UNKNOWN)).sum())}
                    for k, v in vis.items():
                        tot[f"vis_{k}"] += v
                refl = find_reflection_cameras(cent, val) if A >= 2 else []
                if refl:
                    names = [cams[k] if k < len(cams) else str(k) for _, k, _, _ in refl]
                    rr["reflections"].append({"bout": b, "cameras": names})
                    all_refl.append((sess, rec, b, names))
                with zipfile.ZipFile(p) as zf:
                    has_sex = "sex_meta.npy" in zf.namelist()
                if has_sex:
                    with np.load(p, allow_pickle=True) as z:
                        try:
                            rr["sex"][json.loads(str(z["sex_meta"]))["status"]] += 1
                        except Exception:
                            rr["sex"]["unparsed"] += 1
                else:
                    rr["sex"]["absent"] += 1
                rr["bouts"][b] = {"keep": keep, "visibility": vis,
                                  "valid_views": int(val.sum())}
                tot["bouts"] += 1
            rr["sex"] = dict(rr["sex"])
            rr["n_expected"] = len(expected); rr["n_got"] = len(got)
            report["recordings"][f"{sess}/{rec}"] = rr
            print(f'{sess}/{rec}: {len(got)}/{len(expected)} bouts'
                  f'{"  MISSING " + str(rr["missing"]) if rr["missing"] else ""}'
                  f'{"  reflections:" + str(len(rr["reflections"])) if rr["reflections"] else ""}'
                  f'{"  drops:" + str(len(rr["drops"])) if rr["drops"] else ""}')

    tot["missing"] = len(missing)
    report["totals"] = dict(tot)
    report["dropped_bouts"] = [f"{s}/{r}#{b}" for s, r, b in all_drop]
    report["reflection_bouts"] = [f"{s}/{r}#{b}:{c}" for s, r, b, c in all_refl]

    print(f'\n=== TOTALS ===')
    print(f'  bouts verified      : {tot["bouts"]}')
    print(f'  MISSING npz         : {tot["missing"]}')
    if tot.get("vis_miss") is not None:
        v = tot["vis_miss"] + tot["vis_oof"] + tot["vis_unknown"]
        if v:
            print(f'  invalid views       : {v}')
            print(f'    in-frame misses   : {tot["vis_miss"]} ({100*tot["vis_miss"]/v:.1f}%)  <- recoverable')
            print(f'    out of FOV        : {tot["vis_oof"]} ({100*tot["vis_oof"]/v:.1f}%)  <- nothing to recover')
            print(f'    unknown (<3 views): {tot["vis_unknown"]} ({100*tot["vis_unknown"]/v:.1f}%)')
    print(f'  bouts with reflections: {len(all_refl)}')
    print(f'  bouts failing reconstructability: {len(all_drop)}')

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=1, default=str))
    print(f'\nreport -> {outp}')
    return report


if __name__ == "__main__":
    main()
