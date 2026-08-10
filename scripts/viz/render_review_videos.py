#!/usr/bin/env python3
"""Render the sidebyside.mp4 files the fly-ID review GUI serves.

The GUI (scripts/viz/fly_id_review.py) plays
`<pose_dir>/bouts/bout_<NNNNN>/fly<N>/sidebyside.mp4` for both flies of every
bout it lists. A pipeline run with `outputs.overlay=false` produces none of
them, so the GUI lists bouts it cannot show.

Calls `python -m viz sidebyside` per (bout, fly) rather than re-running
run_bout with overlays on, for two reasons:

  * run_bout's DONE marker makes it skip a finished bout entirely, so it would
    do nothing without first deleting DONE;
  * `outputs.overlay=true` also renders 7 per-camera reprojection videos the
    review GUI never opens. Skipping those is most of the cost -- 122 review
    videos took 6.2 min this way versus the ~9 min PER BOUT-FLY that the full
    overlay stage measures at.

Mirrors run_bout.py's own invocation, including JAX_PLATFORMS=cpu for the
subprocess: the render is MuJoCo/EGL and must not contend for GPU memory with
anything else on the node. Already-present videos are skipped per (bout, fly),
so this is safe to re-run and resumes after an interruption.

Usage:
    # every bout missing a video (default)
    python scripts/viz/render_review_videos.py --pose-dir pose_v3
    # only the bouts still queued for review
    python scripts/viz/render_review_videos.py --pose-dir pose_v3 \
        --statuses unsure,bad,pending
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
for p in (str(PROJECT_DIR), str(PROJECT_DIR / "third_party" / "jarvis_jax")):
    if p not in sys.path:
        sys.path.insert(0, p)

PROC = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship"
VIDEO_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship"


def main(argv=None):
    from hydra import compose, initialize_config_dir
    from scripts.run_bout import bout_start_frame

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=PROC)
    ap.add_argument("--pose-dir", default="pose_v3")
    ap.add_argument("--statuses", default="all",
                    help="comma-separated review statuses to cover, or 'all'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--frames", type=int, default=300,
                    help="frames per video (caps the MuJoCo render on long bouts)")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args(argv)

    root = Path(a.root)
    mpath = root / (f"id_review_{a.pose_dir}.json" if a.pose_dir != "pose"
                    else "id_review.json")
    if not mpath.is_file():
        raise SystemExit(f"no review manifest at {mpath} -- start the GUI once "
                         f"to create it, or pass --pose-dir")
    man = json.loads(mpath.read_text())
    man = man.get("bouts", man)
    keys = (list(man) if a.statuses == "all"
            else [k for k, v in man.items()
                  if v.get("status") in a.statuses.split(",")])
    print(f"{len(keys)} bouts in scope (--statuses {a.statuses}) from {mpath.name}")

    cfgs, jobs, no_h5 = {}, [], 0
    for key in sorted(keys):
        sess, rec, bname = key.split("/")
        bidx = int(bname.split("_")[1])
        if (sess, rec) not in cfgs:
            with initialize_config_dir(version_base=None,
                                       config_dir=str(PROJECT_DIR / "configs")):
                cfgs[(sess, rec)] = compose(config_name="pipeline", overrides=[
                    "paths=hyak",
                    f"recording={'session0' if sess == 'Session0' else 'session1'}",
                    f"recording.session_dir={os.path.join(VIDEO_ROOT, sess, rec)}",
                    f"recording.bouts_csv="
                    f"{os.path.join(a.root, sess, rec, 'courtship_bout_summary.csv')}"])
        cfg = cfgs[(sess, rec)]
        try:
            start = int(bout_start_frame(cfg, bidx))
        except Exception as e:                       # bout absent from the CSV
            print(f"  SKIP {key}: {type(e).__name__}: {e}")
            continue
        for fly in (0, 1):
            bd = Path(a.root) / sess / rec / a.pose_dir / "bouts" / bname / f"fly{fly}"
            out = bd / "sidebyside.mp4"
            if out.is_file() and out.stat().st_size > 10_000:
                continue
            if not (bd / "outputs.h5").is_file():
                no_h5 += 1
                continue
            jobs.append((key, fly, [
                sys.executable, "-m", "viz", "sidebyside",
                "--run", str(Path(a.root) / sess / rec / a.pose_dir),
                "--bout", str(bidx), "--fly", str(fly),
                "--n", str(a.frames), "--camera", "track1",
                "--conf", str(float(cfg.detector.conf_thresh)),
                "--session-dir", str(cfg.recording.session_dir),
                "--predictions-dir", str(cfg.recording.predictions_dir),
                "--start-frame", str(start),
                "--fps", str(a.fps), "--out", str(out)]))

    if no_h5:
        print(f"  {no_h5} (bout, fly) have no outputs.h5 and cannot be rendered")
    if not jobs:
        print("nothing to render -- every video in scope already exists")
        return 0
    print(f"{len(jobs)} videos to render on {a.workers} workers")

    t0 = time.time()
    done = [0]

    def run(i_job):
        i, (key, fly, cmd) = i_job
        env = {**os.environ, "JAX_PLATFORMS": "cpu",
               "CUDA_VISIBLE_DEVICES": str(i % max(a.gpus, 1)),
               "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
               "OMP_NUM_THREADS": os.environ.get("RENDER_OMP", "2")}
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=str(PROJECT_DIR), env=env)
        done[0] += 1
        el = time.time() - t0
        print(f"[{done[0]}/{len(jobs)}] {'ok  ' if r.returncode == 0 else 'FAIL'} "
              f"{key} fly{fly}  ({el/60:.1f}m elapsed, "
              f"eta {el/done[0]*(len(jobs)-done[0])/60:.0f}m)", flush=True)
        if r.returncode != 0:
            print("    " + r.stderr.strip()[-600:], flush=True)
        return r.returncode

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rcs = list(ex.map(run, enumerate(jobs)))
    nfail = sum(1 for r in rcs if r)
    print(f"FINISHED in {(time.time()-t0)/60:.1f} min; {nfail} failures")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
