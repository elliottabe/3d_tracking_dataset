#!/usr/bin/env python3
"""Prepare the clip for the explainer: output dirs + a one-bout summary CSV.

SAM3's driver (jarvis_jax.predict.sam3_driver.run_sam3_masks) discovers work
from a bouts CSV; this clip ships none. We synthesise a single bout spanning
every frame, matching the precedent in
scripts/data_prep/convert_underview_to_jarvis.py.
"""
import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))

from scripts.viz.ik_explainer import clip_io   # noqa: E402


def n_frames(clip: str) -> int:
    """Frames common to every camera; the clip is only as long as its shortest view."""
    _mats, names = clip_io.load_dlt(str(Path(clip) / "calibration"))
    # enhanced=False: this defines the bout SAM3/the detector actually ran
    # over -- keep it pinned to the raw frame count regardless of
    # clip_io.video_path's display default (see clip_io's DISPLAY SOURCE
    # docstring section). Both counts agree (921) on this clip anyway.
    counts = [clip_io.n_video_frames(clip_io.video_path(clip, n, enhanced=False))
              for n in names]
    if max(counts) - min(counts) > 1:
        raise ValueError(f"cameras disagree on frame count by >1: {dict(zip(names, counts))}")
    return min(counts)


def write_bouts_csv(clip: str = clip_io.CLIP_DEFAULT, *, n_override: int = 0,
                    name: str = "bouts.csv") -> Path:
    """Write a one-bout summary. n_override>0 truncates the bout (pilot runs).

    run_sam3_masks has NO frame limit -- its `limit` counts BOUTS -- so the only
    way to time a short slice is to hand it a bout that covers fewer frames.
    """
    d = clip_io.out_dirs(clip)
    n = int(n_override) if n_override else n_frames(clip)
    # fly_id must equal the session tag '<Session>/<timestamp>' that
    # sam3_driver.parse_bouts filters on (see its docstring).
    session_tag = "/".join(str(clip).rstrip("/").split("/")[-2:])
    out = d["root"] / name
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fly_id", "bout_idx", "start_frame", "end_frame", "n_frames"])
        w.writerow([session_tag, 0, 0, n - 1, n])
    print(f"wrote {out}  (1 bout, {n} frames, fly_id={session_tag})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    a = ap.parse_args()
    clip_io.out_dirs(a.clip)
    write_bouts_csv(a.clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
