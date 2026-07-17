#!/usr/bin/env python3
"""Re-canonicalize existing SAM3 mask files so the MALE is fly slot 1, using the
multi-view mask-area vote (jarvis_jax.predict.sam3_driver.sex_male_by_size).

For masks generated before the step-0 sexing existed (or by the old temporal-CV
method), this fixes the fly0/fly1 ordering IN PLACE so a downstream pose run that
reuses the masks inherits male=fly1 -- without re-running SAM3.

Per bout `sam3_masks.npz`:
  * vote with sex_male_by_size on a frame subsample (cheap);
  * if the male is not already `male_slot`, swap the fly axis (axis 0) of
    packed/valid/centroids (a 2-element reverse view -- no doubled copy);
  * (re)write `sex_meta` recording the decision; save atomically (tmp + os.replace).
Idempotent: a re-run votes male already at slot 1 -> status "kept", no swap.

Usage:
  python scripts/recanonicalize_masks.py --session <SESSION_DIR> [--dry-run]
  python scripts/recanonicalize_masks.py --predictions-dir <DIR> [--dry-run]
Heavy (each packed array ~1 GB) -> run on a compute node, not the login node.
"""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party/jarvis_jax"))
from jarvis_jax.predict.sam3_driver import sex_male_by_size, _append_sex_meta_to_npz


def _bm_from_arrays(packed, valid, W, *, vote_frames=40):
    """Lightweight bm-mock for sex_male_by_size, from a frame subsample."""
    from types import SimpleNamespace
    F, C, T = packed.shape[:3]
    ts = np.linspace(0, T - 1, min(T, vote_frames)).astype(int)
    masks = []
    for c in range(C):
        frames = []
        for t in ts:
            fr = {}
            for f in range(F):
                if valid[f, c, t]:
                    fr[f] = {"mask": np.unpackbits(packed[f, c, t], axis=-1)[:, :W]}
            frames.append(fr)
        masks.append(frames)
    return SimpleNamespace(num_cameras=C, num_frames=len(ts), masks=masks,
                           identity_map=[{0: 0, 1: 1} for _ in range(C)])


def recanonicalize_npz(npz_path, *, male_slot=1, pct=75, min_pairs=6, min_cams=2,
                       vote_frames=40, dry_run=False):
    """Vote + (optionally) swap one sam3_masks.npz to male=fly{male_slot}. Returns
    a result dict (no file change if dry_run, ambiguous, or already canonical)."""
    with np.load(npz_path, allow_pickle=True) as d:
        keys = {k: d[k] for k in d.files}
    packed, valid, centroids = keys["packed"], keys["valid"], keys["centroids"]
    W = int(keys["shape"][1])
    if packed.shape[0] != 2:                       # only 2-fly bouts
        return {"status": "skip_not2", "male_fly": None}

    bm = _bm_from_arrays(packed, valid, W, vote_frames=vote_frames)
    male, info = sex_male_by_size(bm, 2, pct=pct, min_pairs=min_pairs, min_cams=min_cams)

    if male is None:
        status = "ambiguous"
    elif male == male_slot:
        status = "kept"
    else:
        status = "swapped"

    sex_meta = {"male_slot": male_slot, "status": status, **info}
    res = {"status": status, "male_fly": (male_slot if male is not None else None),
           "male_detected_slot": male, "agreement": info.get("agreement"),
           "margin": info.get("margin"), "n_cameras": info.get("n_cameras")}
    if dry_run:
        return res

    if status == "swapped":
        # fly axis changes -> full atomic rewrite (2-elem reverse view -> [1,0])
        keys["packed"] = packed[::-1]
        keys["valid"] = valid[::-1]
        keys["centroids"] = centroids[::-1]
        keys["sex_meta"] = np.array(json.dumps(sex_meta))
        tmp = npz_path + ".recanon.tmp.npz"
        np.savez_compressed(tmp, **keys)
        os.replace(tmp, npz_path)
    else:
        # kept / ambiguous: arrays unchanged -> cheap zip-append of sex_meta only
        # (reuses the driver's overwrite-safe helper; no 0.8 GB recompress)
        _append_sex_meta_to_npz(npz_path, sex_meta)
    return res


def _predictions_dirs(args):
    if args.predictions_dir:
        return [args.predictions_dir]
    dirs = sorted(glob.glob(os.path.join(args.session, "*", "Predictions_3D_sam3")))
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="session dir; processes each <rec>/Predictions_3D_sam3")
    ap.add_argument("--predictions-dir", help="a single Predictions_3D_sam3 dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pct", type=int, default=75)
    ap.add_argument("--min-pairs", type=int, default=6)
    ap.add_argument("--min-cams", type=int, default=2)
    ap.add_argument("--vote-frames", type=int, default=40)
    args = ap.parse_args()
    if not (args.session or args.predictions_dir):
        ap.error("pass --session or --predictions-dir")

    from collections import Counter
    counts = Counter()
    for pdir in _predictions_dirs(args):
        bouts = sorted(glob.glob(os.path.join(pdir, "bout_*", "sam3_masks.npz")))
        rec = os.path.basename(os.path.dirname(pdir))
        print(f"\n=== {rec} : {len(bouts)} bouts ({pdir}) dry_run={args.dry_run} ===")
        for npz in bouts:
            b = os.path.basename(os.path.dirname(npz))
            r = recanonicalize_npz(npz, pct=args.pct, min_pairs=args.min_pairs,
                                   min_cams=args.min_cams, vote_frames=args.vote_frames,
                                   dry_run=args.dry_run)
            counts[r["status"]] += 1
            print(f"  {b}: status={r['status']:9s} male_detected_slot={r.get('male_detected_slot')} "
                  f"agree={r.get('agreement')} margin={r.get('margin')} n_cams={r.get('n_cameras')}")
    print(f"\nstatus counts: {dict(counts)}")


if __name__ == "__main__":
    main()
