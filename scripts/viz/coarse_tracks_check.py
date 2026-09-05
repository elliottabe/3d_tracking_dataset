#!/usr/bin/env python3
"""Figure gate 2 for the mask-free coarse pass: do the tracks explain the
reviewed bouts?

Spec: `docs/specs/2026-09-04-mvq-maskfree-frontend-design.md` §8.2.

EXPECTATION (write before looking, verbatim from the spec/brief): "reviewed
bouts coincide with close-distance, wing-extension episodes; a reviewed bout
that does not is recorded" (it decides what the §5 learned detector has to
learn). Concretely: inside a grey (reviewed) band, inter-fly distance should
dip and/or the male's wing angle should rise, roughly together with where
the green gate-derived bar row fires; a grey band with flat distance, a flat
(near-zero) wing angle AND no green bar under it is a bout the hand gates
(`jarvis_jax.tracking.bout_gates`) cannot see either -- report it by bout id,
don't average it away.

Four stacked time panels over the WHOLE recording, x in MINUTES
(`coarse_frame / 800 fps / 60`):
  1. inter-fly distance, mm (`tracks["dist"]` == `sep3d`, `* MM_PER_UNIT`)
  2. male wing angle, deg (`tracks["wing_angle_deg"][1]`, fly row 1 = male)
  3. both flies' speed, mm/s (`tracks["speed"][f] * MM_PER_UNIT * 800 / stride`,
     female cyan / male orange)
  4. existence, unitless 0-1 (`tracks["exist"][f]`, both flies)
plus a bar row of gate-derived bouts (green) below the 4 panels, reviewed
bouts (grey) shaded as `axvspan` on every panel including the bar row, and a
zoom panel on bout 28 (distance + wing angle, twin y-axis) -- the bout named
in `docs/specs/2026-08-31-pipeline-schematic-and-inventory-design.md`'s own
SAM3 calibration numbers, so this is the one bout every coarse-pass version
has been checked against. If bout 28 is not in `--reviewed-bouts` (e.g. a
synthetic check), the middle reviewed bout is used instead and the title
says so plainly -- never silently zoom on the wrong bout without saying it.

Reads `coarse_tracks.npz` (mvq schema only -- needs `exist`/`wing_angle_deg`/
`dist`, which a SAM3-schema file does not have), `--reviewed-bouts` (human
ground truth) and `--gate-bouts` (`scripts/coarse_pass_gates.py`'s emitted
CSV; optional -- the bar row is empty and says so if omitted).

Output: `<out-dir>/coarse_tracks_check.png` + a `.json` beside it recording
the reviewed/gate bout tables and the zoom bout's numbers.

Usage (see docs/benchmark/2026-09-mvq/p4-maskfree-notes.md for the exact
20_04 command line):

    python scripts/viz/coarse_tracks_check.py \\
      --tracks .../coarse_mvq/coarse_tracks.npz \\
      --reviewed-bouts .../2025_10_20_13_20_04/courtship_bouts_fly0_summary.csv \\
      --gate-bouts .../coarse_mvq/bouts_mvq_gates.csv \\
      --out-dir figures/2026-09-mvq/p4_maskfree
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SCRIPTS)
# NOTE: `scripts/` (as opposed to `scripts/viz/`) is deliberately NOT put on
# sys.path -- `scripts/viz/` has its own (empty) `__init__.py`, so with
# `scripts/` on the path `import viz` would resolve to THAT namesake package
# instead of the real `viz/` at the repo root. This script needs nothing from
# `scripts/` itself, only `third_party/jarvis_jax` and the repo root (for
# `viz.core.colors`).
for _p in (os.path.join(_REPO, "third_party", "jarvis_jax"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jarvis_jax.tracking.bout_gates import load_ground_truth  # noqa: E402
from jarvis_jax.train.train_mvq import MM_PER_UNIT  # noqa: E402
from viz.core.colors import PALETTE  # noqa: E402

FPS = 800.0
ZOOM_BOUT_IDX_DEFAULT = 28
ZOOM_PAD_FRAMES = 4000    # real frames either side of the zoomed bout (~5s at 800fps)


def _rgb01(bgr):
    b, g, r = bgr
    return (r / 255.0, g / 255.0, b / 255.0)


FEMALE_RGB = _rgb01(PALETTE["fly0"])   # cyan
MALE_RGB = _rgb01(PALETTE["fly1"])     # orange
REVIEWED_RGB = (0.55, 0.55, 0.55)      # grey
GATE_RGB = (0.0, 0.75, 0.0)            # green


def load_tracks_and_meta(tracks_path, meta_path=None):
    meta_path = meta_path or (tracks_path.rsplit(".npz", 1)[0] + ".meta.json")
    z = np.load(tracks_path, allow_pickle=True)
    if "exist" not in z:
        raise ValueError(f"{tracks_path} has no `exist` array -- this script reads the mvq "
                         f"coarse-track schema, not the SAM3 one")
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    return z, meta


def shade_bouts(axes, bouts, color, alpha, to_min):
    for _, s, e in bouts:
        lo, hi = to_min(s), to_min(e)
        for ax in axes:
            ax.axvspan(lo, hi, color=color, alpha=alpha, lw=0)


def pick_zoom_bout(reviewed, want_idx=ZOOM_BOUT_IDX_DEFAULT):
    for bidx, s, e in reviewed:
        if bidx == want_idx:
            return bidx, s, e, False
    if not reviewed:
        return None
    mid = reviewed[len(reviewed) // 2]
    return mid[0], mid[1], mid[2], True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--meta", default=None, help="default: <tracks without .npz>.meta.json")
    ap.add_argument("--reviewed-bouts", required=True, help="human-reviewed bout-summary CSV")
    ap.add_argument("--gate-bouts", default=None, help="scripts/coarse_pass_gates.py output CSV")
    ap.add_argument("--session-tag", default=None,
                    help="fly_id filter for --reviewed-bouts/--gate-bouts")
    ap.add_argument("--zoom-bout-idx", type=int, default=ZOOM_BOUT_IDX_DEFAULT)
    ap.add_argument("--zoom-pad-frames", type=int, default=ZOOM_PAD_FRAMES)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    z, meta = load_tracks_and_meta(args.tracks, args.meta)
    stride = int(meta.get("stride", z["coarse_frame"][1] - z["coarse_frame"][0]
                          if z["coarse_frame"].shape[0] > 1 else 1))
    coarse_frame = np.asarray(z["coarse_frame"], np.int64)
    to_min = lambda f: np.asarray(f, np.float64) / FPS / 60.0
    x_min = to_min(coarse_frame)

    dist_mm = np.asarray(z["dist"], np.float32) * MM_PER_UNIT
    wing_male_deg = np.asarray(z["wing_angle_deg"], np.float32)[1]
    speed_mm_s = np.asarray(z["speed"], np.float32) * MM_PER_UNIT * FPS / stride
    exist = np.asarray(z["exist"], np.float32)

    reviewed = load_ground_truth(args.reviewed_bouts, fly_id=args.session_tag)
    gate = load_ground_truth(args.gate_bouts, fly_id=args.session_tag) if args.gate_bouts else []

    # Which reviewed bouts have NO close-distance / wing-extension support at all
    # (flat signal throughout) -- reported explicitly per the EXPECTATION.
    unexplained = []
    for bidx, s, e in reviewed:
        m = (coarse_frame >= s) & (coarse_frame <= e)
        if not m.any():
            unexplained.append({"bout_idx": bidx, "start_frame": s, "end_frame": e,
                                "reason": "no coarse samples land inside this bout"})
            continue
        min_dist = float(np.nanmin(dist_mm[m])) if np.isfinite(dist_mm[m]).any() else float("nan")
        max_wing = float(np.nanmax(wing_male_deg[m])) if np.isfinite(wing_male_deg[m]).any() else float("nan")
        gated = any(overlap for _, gs, ge in gate for overlap in [max(0, min(e, ge) - max(s, gs) + 1)] if overlap > 0)
        # judgment-call thresholds, for REPORTING only (not a gate re-implementation):
        # matches bout_gates.PROXIMITY_MAX_UNITS_DEFAULT (30 units = 3mm) / WING_ANGLE_MIN_DEFAULT (30deg)
        close = np.isfinite(min_dist) and min_dist <= 30.0 * MM_PER_UNIT
        wings = np.isfinite(max_wing) and max_wing >= 30.0
        if not (close or wings):
            unexplained.append({"bout_idx": bidx, "start_frame": s, "end_frame": e,
                                "min_dist_mm": min_dist, "max_wing_deg": max_wing,
                                "gate_overlap": bool(gated),
                                "reason": "neither close-distance nor wing-extension observed"})

    fig, axes = plt.subplots(6, 1, figsize=(16, 15), sharex=False,
                             gridspec_kw=dict(height_ratios=[2, 2, 2, 1, 0.6, 2.4]))
    ax_dist, ax_wing, ax_speed, ax_exist, ax_gate, ax_zoom = axes

    shade_bouts([ax_dist, ax_wing, ax_speed, ax_exist, ax_gate], reviewed, REVIEWED_RGB, 0.30, to_min)
    shade_bouts([ax_dist, ax_wing, ax_speed, ax_exist], gate, GATE_RGB, 0.20, to_min)

    ax_dist.plot(x_min, dist_mm, color="black", lw=0.8)
    ax_dist.set_ylabel("inter-fly\ndistance (mm)")

    ax_wing.plot(x_min, wing_male_deg, color=MALE_RGB, lw=0.8)
    ax_wing.set_ylabel("male wing\nangle (deg)")

    ax_speed.plot(x_min, speed_mm_s[0], color=FEMALE_RGB, lw=0.7, label="fly0 (female)")
    ax_speed.plot(x_min, speed_mm_s[1], color=MALE_RGB, lw=0.7, label="fly1 (male)")
    ax_speed.set_ylabel("speed\n(mm/s)")
    ax_speed.legend(loc="upper right", fontsize=7)

    ax_exist.plot(x_min, exist[0], color=FEMALE_RGB, lw=0.8, drawstyle="steps-post")
    ax_exist.plot(x_min, exist[1], color=MALE_RGB, lw=0.8, drawstyle="steps-post")
    ax_exist.axhline(0.5, color="crimson", ls="--", lw=0.8)
    ax_exist.set_ylabel("existence")
    ax_exist.set_ylim(-0.05, 1.05)

    for bidx, s, e in gate:
        ax_gate.broken_barh([(to_min(s), to_min(e) - to_min(s))], (0, 1), facecolors=GATE_RGB)
    ax_gate.set_yticks([]); ax_gate.set_ylim(0, 1)
    ax_gate.set_ylabel("gate\nbouts")
    ax_gate.set_xlabel("time (min)")

    zoom = pick_zoom_bout(reviewed, args.zoom_bout_idx)
    zoom_json = None
    if zoom is not None:
        zbidx, zs, ze, is_fallback = zoom
        lo, hi = max(0, zs - args.zoom_pad_frames), ze + args.zoom_pad_frames
        m = (coarse_frame >= lo) & (coarse_frame <= hi)
        xz = coarse_frame[m]
        ax_zoom.plot(xz, dist_mm[m], color="black", lw=1.2, label="inter-fly distance (mm)")
        ax_zoom.axvspan(zs, ze, color=REVIEWED_RGB, alpha=0.30, lw=0)
        for _, gs, ge in gate:
            if max(0, min(ze, ge) - max(zs, gs)) > 0 or (gs <= hi and ge >= lo):
                ax_zoom.axvspan(max(gs, lo), min(ge, hi), color=GATE_RGB, alpha=0.20, lw=0)
        ax_zoom.set_ylabel("distance (mm)")
        ax_zoom.set_xlabel("real video frame")
        ax_zoom2 = ax_zoom.twinx()
        ax_zoom2.plot(xz, wing_male_deg[m], color=MALE_RGB, lw=1.2, ls="--",
                     label="male wing angle (deg)")
        ax_zoom2.set_ylabel("male wing angle (deg)", color=MALE_RGB)
        lines = ax_zoom.get_lines() + ax_zoom2.get_lines()
        ax_zoom.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=7)
        fallback_note = (f" (bout {args.zoom_bout_idx} not in --reviewed-bouts; showing bout "
                         f"{zbidx} instead)" if is_fallback else "")
        ax_zoom.set_title(f"Zoom: bout {zbidx} [{zs},{ze}]{fallback_note}", fontsize=9)
        zoom_json = {"bout_idx": zbidx, "start_frame": int(zs), "end_frame": int(ze),
                    "is_fallback": is_fallback, "window": [int(lo), int(hi)]}
    else:
        ax_zoom.text(0.5, 0.5, "no reviewed bouts -- nothing to zoom on",
                    ha="center", va="center", transform=ax_zoom.transAxes)
        ax_zoom.set_xticks([]); ax_zoom.set_yticks([])

    fig.suptitle(f"Coarse tracks check -- {meta.get('session_dir', args.tracks)}\n"
                f"grey = reviewed bouts, green = gate-derived bouts; "
                f"{len(unexplained)}/{len(reviewed)} reviewed bout(s) unexplained by the tracks",
                fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(args.out_dir, exist_ok=True)
    png_path = os.path.join(args.out_dir, "coarse_tracks_check.png")
    json_path = os.path.join(args.out_dir, "coarse_tracks_check.json")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    with open(json_path, "w") as f:
        json.dump({"tracks": args.tracks, "reviewed_bouts": args.reviewed_bouts,
                   "gate_bouts": args.gate_bouts, "stride": stride,
                   "reviewed": [{"bout_idx": b, "start_frame": int(s), "end_frame": int(e)}
                               for b, s, e in reviewed],
                   "gate": [{"bout_idx": b, "start_frame": int(s), "end_frame": int(e)}
                           for b, s, e in gate],
                   "unexplained_reviewed_bouts": unexplained,
                   "zoom": zoom_json}, f, indent=2)
    print(f"wrote {png_path}\nwrote {json_path}")
    if unexplained:
        print(f"[coarse_tracks_check] {len(unexplained)} reviewed bout(s) unexplained: "
              f"{[u['bout_idx'] for u in unexplained]}")
    return png_path, json_path


if __name__ == "__main__":
    main()
