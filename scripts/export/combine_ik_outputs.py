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


def _quat_to_R(q):
    """(T,4) wxyz unit quaternions -> (T,3,3) body->world rotation matrices."""
    w, x, y, z = np.asarray(q, float).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def egocentric_sites(site_xpos, root_se3, kp_names, origin="Scutellum"):
    """Site positions in the fly's own body frame, in mm.

    The analysis (`utils.song_analysis`) only computes the wing extension /
    horizontal angles that Figure 4 panel E needs when a bout carries
    `xpos_egocentric`; without it `horiz_angle_L/R` stay None and the panel
    renders empty. The legacy `Data_analysis/analysis/v1` h5 had this array;
    the per-bout pipeline never writes it (neither stac_ik.h5 nor outputs.h5
    contains it), so it is derived here from data that IS written.

    Definition matches the legacy one -- FK with the root pose removed, i.e.
    the thorax frame -- as `R_root^T (p_world - p_origin)`. The origin is a
    KEYPOINT rather than `root_se3`'s translation because `site_xpos` is in mm
    while `root_se3`'s translation is in model units; subtracting those
    directly would mix units. Since the frame differs from the legacy one only
    by that rigid offset, and both consumers build their own frame from
    normalised differences, the resulting angles are identical (verified: a
    random rotation+scale+translation moves them by ~2e-11 deg).

    Returns None when `root_se3` or the origin keypoint is unavailable, so a
    bout without them is left without the key rather than given a silently
    non-egocentric array.
    """
    if root_se3 is None or site_xpos is None:
        return None
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in kp_names]
    if origin not in names:
        return None
    se3 = np.asarray(root_se3, float)
    p = np.asarray(site_xpos, float)
    if se3.ndim != 2 or se3.shape[1] < 7 or se3.shape[0] != p.shape[0]:
        return None
    R = _quat_to_R(se3[:, 3:7])                       # (T,3,3) body->world
    rel = p - p[:, names.index(origin), :][:, None, :]
    # R^T @ rel: world -> body
    return np.einsum("tji,tkj->tki", R, rel).astype(np.float32)


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
    # Per-FLY quality, from scripts/qc/per_fly_quality.py. Human review gave one
    # verdict per bout, but the defect is per fly: in 35 bouts the female is
    # unusable while the male is fine. Excluding those bouts wholesale would throw
    # away 35 good male fits, so --skip-unusable drops the FLY, not the bout.
    ap.add_argument("--quality-report", default="docs/qc/per_fly_quality.json")
    ap.add_argument("--skip-unusable", action="store_true",
                    help="omit individual flies that fail the per-fly quality gate")
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

    unusable = {}
    if a.skip_unusable:
        qp = Path(a.quality_report)
        if not qp.is_file():
            raise SystemExit(
                f"--skip-unusable needs {qp}, which does not exist.\n"
                f"Regenerate it with:\n"
                f"  python scripts/qc/per_fly_quality.py --pose-dir {a.pose_dir} "
                f"--json-out {qp}")
        qrep = json.loads(qp.read_text())
        if qrep.get("pose_dir") != a.pose_dir:
            raise SystemExit(
                f"{qp} was built for pose_dir={qrep.get('pose_dir')!r} but this "
                f"run uses {a.pose_dir!r} -- the verdicts would not match the data.")
        unusable = {k: v["reasons"] for k, v in qrep["flies"].items()
                    if not v["usable"]}
        print(f"per-fly quality gate: {len(unusable)} flies will be skipped")

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
                            "sex_verified", "reconstructable", "review_status",
                            "tracking_usable")}
    shared = {}
    n = 0
    skipped = 0
    skipped_bad = 0
    skipped_fly = 0

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
                    fkey = f"{sess}/{rec}/{bdir.name}/fly{fly}"
                    if fkey in unusable:
                        skipped_fly += 1
                        continue
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
                            # Panel E's wing angles need a body-frame array;
                            # see egocentric_sites for why it is derived here.
                            _ego = egocentric_sites(
                                d.get("site_xpos"), d.get("root_se3"),
                                shared.get("kp_names", []))
                            if _ego is not None:
                                d["xpos_egocentric"] = _ego
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
                    info["tracking_usable"].append(fkey not in unusable)
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
    print(f"  skipped (per-fly quality):    {skipped_fly}")
    from collections import Counter
    print(f"  review status: {dict(Counter(info['review_status']))}")
    print(f"  sex verified: {sum(info['sex_verified'])}/{n}")
    return out


if __name__ == "__main__":
    main()
