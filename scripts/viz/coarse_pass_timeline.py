#!/usr/bin/env python3
"""Coarse-pass verification figure: gate signals across a recording, with
emitted bout windows and the known human-reviewed bouts overlaid as bands.

Expectation stated BEFORE generating (per CLAUDE.md): if the coarse pass is
correct, (1) fly1 (male)'s area ratio should sit near 1.0 through most of the
recording and rise above the wing-extension threshold mainly INSIDE the known
bout bands; (2) fly0 (female)'s area ratio should stay near 1.0 in resting
periods and fall below the 0.5 trackability line specifically where she
leaves the cameras (e.g. bout 28, offset ~1500-1590 per the spec's own
calibration evidence); (3) the emitted (green) bands should mostly sit INSIDE
the known (grey) bands, and for bout 28 specifically the green band should
end near start+1500, NOT tile the grey band all the way to start+2007 -- a
green band that matches the grey band exactly there means the trackability
gate did not actually fire and the coarse pass reproduced the defect it was
built to fix.

Usage:
    python scripts/viz/coarse_pass_timeline.py --tracks .../coarse_tracks.npz \\
        --emitted .../coarse_bout_summary.csv --ground-truth .../courtship_bout_summary.csv \\
        --session-tag Session0/2025_10_20_13_20_04 --out figures/2026-08-31-coarse-pass/timeline.png
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from coarse_pass_gates import (  # noqa: E402
    BASELINE_WINDOW, BORDER_MIN_PX, MIN_CAMS, SEP_MIN_PX, apply_gates,
    compute_gate_signals, load_ground_truth, load_tracks)

FLY0_RGB = (0.0, 1.0, 1.0)     # PALETTE['fly0'] BGR(255,255,0) cyan -> female
FLY1_RGB = (1.0, 0.647, 0.0)   # PALETTE['fly1'] BGR(0,165,255) orange -> male
KNOWN_RGB = (0.55, 0.55, 0.55)
EMITTED_RGB = (0.0, 0.75, 0.0)  # PALETTE['fit'] green


def read_windows_csv(path, session_tag=None):
    out = []
    if not path or not os.path.isfile(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if session_tag and r.get("fly_id") not in (None, "", session_tag):
                continue
            out.append((int(r["bout_idx"]), int(r["start_frame"]), int(r["end_frame"])))
    return out


def band(ax, s, e, color, alpha=0.25, ymax_frac=1.0):
    ax.axvspan(s, e, color=color, alpha=alpha, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--emitted", default=None, help="coarse-pass bout-summary CSV")
    ap.add_argument("--ground-truth", default=None, help="human-reviewed bout-summary CSV")
    ap.add_argument("--session-tag", default=None)
    ap.add_argument("--wing-ratio-min", type=float, default=1.15)
    ap.add_argument("--proximity-max-px", type=float, default=120.0)
    ap.add_argument("--baseline-window", type=int, default=BASELINE_WINDOW)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xlim", default=None, help="start,end real-frame range to zoom (else whole recording)")
    args = ap.parse_args()

    z, meta = load_tracks(args.tracks)
    sig = compute_gate_signals(z, baseline_window=args.baseline_window)
    gates = apply_gates(sig, wing_ratio_min=args.wing_ratio_min,
                        proximity_max_px=args.proximity_max_px)
    x = z["coarse_frame"]

    known = read_windows_csv(args.ground_truth, args.session_tag)
    emitted = read_windows_csv(args.emitted, args.session_tag)

    fig, axes = plt.subplots(5, 1, figsize=(16, 13), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1, 1, 1, 0.6]))
    ax_area, ax_border, ax_cams, ax_sep, ax_gate = axes

    for ax in axes:
        for bidx, s, e in known:
            band(ax, s, e, KNOWN_RGB, alpha=0.30)
        for bidx, s, e in emitted:
            band(ax, s, e, EMITTED_RGB, alpha=0.35)

    ax_area.plot(x, sig["area_ratio"][0], color=FLY0_RGB, lw=0.8, label="fly0 (female) area/baseline")
    ax_area.plot(x, sig["area_ratio"][1], color=FLY1_RGB, lw=0.8, label="fly1 (male) area/baseline")
    ax_area.axhline(0.5, color="crimson", ls="--", lw=1, label="trackability floor (0.5x)")
    ax_area.axhline(args.wing_ratio_min, color="darkgreen", ls="--", lw=1,
                    label=f"wing-extension threshold ({args.wing_ratio_min}x)")
    ax_area.set_ylabel("area / own\nrolling median")
    ax_area.set_ylim(0, max(2.0, np.nanpercentile(sig["area_ratio"], 99.5)))
    ax_area.legend(loc="upper right", fontsize=8, ncol=2)

    ax_border.plot(x, sig["border_med"][0], color=FLY0_RGB, lw=0.8)
    ax_border.plot(x, sig["border_med"][1], color=FLY1_RGB, lw=0.8)
    ax_border.axhline(BORDER_MIN_PX, color="crimson", ls="--", lw=1)
    ax_border.set_ylabel("border dist\n(px, median\nover valid cams)")

    ax_cams.plot(x, sig["n_valid_cams"][0], color=FLY0_RGB, lw=0.8, drawstyle="steps-post")
    ax_cams.plot(x, sig["n_valid_cams"][1], color=FLY1_RGB, lw=0.8, drawstyle="steps-post")
    ax_cams.axhline(MIN_CAMS, color="crimson", ls="--", lw=1)
    ax_cams.set_ylabel("n cams\nvalid")
    ax_cams.set_ylim(-0.3, 7.3)

    ax_sep.plot(x, sig["sep2d_med"], color="black", lw=0.8)
    ax_sep.axhline(SEP_MIN_PX, color="purple", ls="--", lw=1, label=f"separability floor ({SEP_MIN_PX}px)")
    ax_sep.axhline(args.proximity_max_px, color="darkgreen", ls="--", lw=1,
                  label=f"proximity threshold ({args.proximity_max_px}px)")
    ax_sep.set_ylabel("fly0-fly1\nseparation (px)")
    ax_sep.legend(loc="upper right", fontsize=8)

    ax_gate.fill_between(x, 0, gates["trackability_ok"].astype(float), step="post",
                         color="steelblue", alpha=0.6, label="trackability_ok")
    ax_gate.fill_between(x, 0, gates["behaviour_ok"].astype(float) * 0.9, step="post",
                         color="darkorange", alpha=0.5, label="behaviour_ok")
    ax_gate.fill_between(x, 0, gates["in_bout"].astype(float) * 0.8, step="post",
                         color="darkgreen", alpha=0.8, label="in_bout (both)")
    ax_gate.set_ylim(0, 1.05)
    ax_gate.set_yticks([])
    ax_gate.set_xlabel("real video frame")
    ax_gate.legend(loc="upper right", fontsize=7, ncol=3)

    if args.xlim:
        lo, hi = (int(v) for v in args.xlim.split(","))
        for ax in axes:
            ax.set_xlim(lo, hi)

    fig.suptitle(f"Coarse pass gate signals -- {meta.get('session_dir', args.tracks)}\n"
                f"grey = known human-reviewed bouts, green = coarse-pass emitted windows",
                fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
