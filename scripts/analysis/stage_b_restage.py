"""Move aside artifacts that predate the triangulation outlier gate, so Stage B re-runs.

WHY. `detector.reproj_resid_px: 10.0` (consensus outlier-view rejection) landed
2026-08-14 in c4bdaa8, on by default. 252 of 320 courtship bout-flies were
triangulated 2026-08-10, four days earlier, and 68 free-running bout-flies on
2026-07-07 -- so most of the dataset's 3D was built WITHOUT the gate. Verified
directly on bout_00001 fly0: re-triangulating with the gate off reproduces the
stored kp3d.npz to 0.0000 mm.

That stale 3D is what today's wing investigation was actually chasing. With the
gate on, WingL_V12's reprojection error drops 15.63 -> 5.03 px and it moves on
98.3% of frames, because one camera (Cam2012861) is persistently inconsistent.

WHAT THIS DOES. run_bout resumes by artifact existence -- `bout_complete` on the
DONE marker, `stage_done` on each stage's output -- so re-running Stage B means
those artifacts must not be present. This MOVES them to a backup tree rather
than deleting, so a bad re-run is recoverable by moving them back.

kp2d.npz is KEPT: Stage A (detection) is unaffected by the gate, the 2D was
verified good (WingL_V12 sits 0.020 of fly size from the silhouette edge at conf
0.96 in 7/7 views), and re-detecting would cost far more than re-triangulating.

Recording-level `offsets.h5` / `scale.json` / `segment_scales.json` are derived
FROM kp3d and shared across bouts, so they are moved too when --with-recording
is passed; leaving a stale offsets.h5 in place would make the new 3D be fit
against marker offsets solved for the old 3D.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

# artifacts downstream of Stage B, in dependency order. kp2d.npz is NOT here.
BOUT_ARTIFACTS = ["DONE", "kp3d.npz", "kp3d_filt.npz", "stac_ik.h5", "outputs.h5",
                  "qpos_refined.npz", "qc.json", "qc_perframe.npz",
                  "unsolvable.json", "track_qc.json"]
RECORDING_ARTIFACTS = ["offsets.h5", "scale.json", "segment_scales.json"]
# Commit c4bdaa8 ("consensus outlier-view rejection in triangulation, on by
# default"). Derived from git rather than hardcoded -- a hand-typed epoch here
# was off by a YEAR (2025 not 2026), which silently matched nothing.
def _gate_epoch(default=1786853476):
    import subprocess
    try:
        return int(subprocess.run(["git", "log", "-1", "--format=%at", "c4bdaa8"],
                                  cwd=os.path.dirname(os.path.dirname(
                                      os.path.dirname(os.path.abspath(__file__)))),
                                  capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        return default


GATE_EPOCH = _gate_epoch()


def stale_bout_flies(proc_root, cutoff):
    out = []
    for root, _dirs, files in os.walk(proc_root):
        if "kp3d.npz" not in files or "/pose/bouts/" not in root:
            continue
        if os.path.getmtime(os.path.join(root, "kp3d.npz")) < cutoff:
            out.append(root)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-root",
                    default="/gscratch/portia/eabe/data/Johnson_lab/processed/courtship")
    ap.add_argument("--recording", default=None,
                    help="limit to one recording, e.g. Session0/2025_10_20_13_20_04")
    ap.add_argument("--bout", type=int, default=None, help="limit to one bout index")
    ap.add_argument("--backup", required=True, help="destination tree for the moved artifacts")
    ap.add_argument("--with-recording", action="store_true",
                    help="also move offsets.h5/scale.json/segment_scales.json")
    ap.add_argument("--cutoff", type=float, default=GATE_EPOCH)
    ap.add_argument("--apply", action="store_true", help="actually move (default: dry run)")
    args = ap.parse_args()

    dirs = stale_bout_flies(args.proc_root, args.cutoff)
    if args.recording:
        dirs = [d for d in dirs if args.recording in d]
    if args.bout is not None:
        dirs = [d for d in dirs if f"bout_{args.bout:05d}" in d]
    if not dirs:
        raise SystemExit("no stale bout-flies matched the filters")

    recs = sorted({d.split("/pose/bouts/")[0] for d in dirs})
    n_files = 0
    n_bytes = 0
    plan = []
    for d in dirs:
        for f in BOUT_ARTIFACTS:
            p = os.path.join(d, f)
            if os.path.exists(p):
                plan.append((p, os.path.join(args.backup, os.path.relpath(p, args.proc_root))))
                n_files += 1
                n_bytes += os.path.getsize(p)
    if args.with_recording:
        for r in recs:
            for f in RECORDING_ARTIFACTS:
                p = os.path.join(r, "pose", f)
                if os.path.exists(p):
                    plan.append((p, os.path.join(args.backup,
                                                 os.path.relpath(p, args.proc_root))))
                    n_files += 1
                    n_bytes += os.path.getsize(p)

    print(f"{len(dirs)} stale bout-flies across {len(recs)} recording(s)")
    for r in recs:
        print(f"   {r.replace(args.proc_root + '/', '')}: "
              f"{sum(1 for d in dirs if d.startswith(r))} bout-flies")
    print(f"{n_files} files, {n_bytes / 2**30:.2f} GB -> {args.backup}")
    print(f"KEPT in place: kp2d.npz (Stage A unaffected), overlays/, sidebyside*.mp4")
    if not args.apply:
        print("\nDRY RUN. Sample of what would move:")
        for src, dst in plan[:6]:
            print(f"   {src.replace(args.proc_root + '/', '')}")
        print("   ...")
        print("re-run with --apply to execute")
        return

    for src, dst in plan:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    manifest = os.path.join(args.backup, "restage_manifest.json")
    os.makedirs(args.backup, exist_ok=True)
    with open(manifest, "w") as f:
        json.dump({"moved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "proc_root": args.proc_root, "cutoff": args.cutoff,
                   "n_files": n_files, "bout_flies": dirs,
                   "with_recording": bool(args.with_recording),
                   "reason": "artifacts predate detector.reproj_resid_px gate (c4bdaa8, "
                             "2026-08-14); Stage B onward must recompute"}, f, indent=2)
    print(f"\nMOVED {n_files} files. Manifest: {manifest}")
    print("To undo: move the tree back over the processed root (paths are relative to it).")


if __name__ == "__main__":
    main()
