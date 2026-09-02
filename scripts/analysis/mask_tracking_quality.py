#!/usr/bin/env python3
"""Rank every (bout, fly) of a recording by TRACKING QUALITY, from the SAM masks alone.

WHY. The post-STAC wing-pitch fit (Stage D2) is only meant to correct flies that
are WELL TRACKED -- the user's ruling of 2026-09-01: "our focus for correcting is
when the flies are well tracked, so if the female is not well tracked it is ok to
skip it." Choosing which bouts to run it on therefore needs a quality measure
that costs nothing, and the masks are already on disk for every bout while the
pose artifacts exist for only one.

WHAT IT MEASURES (`jarvis_jax.tracking.mask_quality.bout_fly_quality`):
per-camera validity, mask area against that camera's OWN healthy reference
(p75), the sliver fraction, how many cameras carry a usable mask per frame, and
the worst window of the bout -- because a bout average hides a bad quarter,
which is exactly how Session0 bout 28 fly0's frames 1500-2006 went unscored
through two tasks. Centroid jump is reported and deliberately NOT scored: on
bout 28 it does not separate the known good/bad split (the male, fine
throughout, jumps MORE than the collapsed female, because he moves more).

CALIBRATION, and it is not optional. `--calibrate 28` scores bout 28 in the
three windows whose truth is known from
docs/benchmark/2026-09-01-wing-mask-fit/notes.md 10.6 -- fly1 good everywhere,
fly0 good on 0-1499 and collapsed on 1500-2006 -- and PRINTS PASS/FAIL. A score
that does not reproduce that split is wrong and must be fixed before it is used
to pick anything.

CAMERAS ARE ADDRESSED BY NAME throughout. The npz's own `cameras` array is in a
DIFFERENT order from `cfg.recording.cameras` (the canonical/calibration order),
and reading one with the other's index plots one camera's data on another's --
a trap this repo has been bitten by twice. Nothing here indexes a camera axis
by integer except through a name lookup built in `_load_bout`.

Run (CPU only, no GPU, ~10 s per bout):

  python scripts/analysis/mask_tracking_quality.py \
      --masks-dir /gscratch/.../2025_10_20_13_20_04/sam3_masks \
      --calibrate 28 \
      --out figures/2026-09-01-wing-mask-fit/tracking_quality
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from jarvis_jax.tracking.mask_quality import (  # noqa: E402
    area_reference, bout_fly_quality, camera_validity, frame_keep,
    mask_areas_from_packed,
)

#: The windows bout 28's truth is stated in, and what that truth is.
#: notes.md 10.6: fly0 0-1500 100% valid / 15 862 px / 0% slivers; 1500-1710
#: 90.7% valid / 4 882 px / 40.2% slivers; 1710-2007 58% valid / 2 834 px.
CALIBRATION = {
    "fly0": [((0, 1500), "GOOD"), ((1500, 1710), "BAD"), ((1710, None), "BAD")],
    "fly1": [((0, 1500), "GOOD"), ((1500, 1710), "GOOD"), ((1710, None), "GOOD")],
}


def _bout_dirs(masks_dir, only=None):
    out = []
    for name in sorted(os.listdir(masks_dir)):
        m = re.fullmatch(r"bout_(\d+)", name)
        if not m:
            continue
        idx = int(m.group(1))
        if only is not None and idx not in only:
            continue
        npz = os.path.join(masks_dir, name, "sam3_masks.npz")
        if os.path.exists(npz):
            out.append((idx, npz))
    return out


def _load_bout(npz_path, exclude_cameras=()):
    """(cameras, per-fly dict of area/valid/centroids), cameras BY NAME.

    `exclude_cameras` drops named cameras from every array before scoring, so
    the score can be taken over exactly the cameras the stage will use
    (`wing_mask_fit.exclude_cameras` names Cam2012631, whose masks are
    truncated).
    """
    z = np.load(npz_path, allow_pickle=True)
    if "cameras" not in z.files:
        raise SystemExit(f"{npz_path} carries no `cameras` name array; its "
                         f"camera axis cannot be verified by name")
    cameras = [str(c) for c in np.asarray(z["cameras"]).tolist()]
    keep_c = [i for i, c in enumerate(cameras) if c not in set(exclude_cameras)]
    cams = [cameras[i] for i in keep_c]
    W = int(z["shape"][1])
    packed = z["packed"]
    valid = np.asarray(z["valid"], bool)                 # (A, C, T)
    cent = np.asarray(z["centroids"], np.float32)        # (A, C, T, 2)
    flies = {}
    for f in range(packed.shape[0]):
        area = mask_areas_from_packed(packed[f], width=W)        # (T, C)
        flies[f] = dict(area=area[:, keep_c],
                        valid=valid[f].T[:, keep_c],
                        centroids=cent[f].transpose(1, 0, 2)[:, keep_c])
    del packed
    sex_meta = str(z["sex_meta"]) if "sex_meta" in z.files else ""
    return cams, flies, sex_meta


def _window_report(area, valid, cams, windows, min_area_frac, min_cameras,
                   pct):
    """Per-window valid / median-area / sliver / usable-camera numbers.

    The reference is taken over the WHOLE bout (once), not per window: a window
    that is entirely collapsed would otherwise re-baseline onto its own slivers
    and report 0% slivers, which is the failure mode this whole gate exists to
    stop.
    """
    ref = area_reference(area, valid, pct=pct)
    ok = camera_validity(valid, area, ref, min_area_frac=min_area_frac)
    keep = frame_keep(ok, min_cameras=min_cameras)
    rows = []
    T = area.shape[0]
    for (lo, hi) in windows:
        hi = T if hi is None else min(int(hi), T)
        sl = slice(int(lo), hi)
        v, o = valid[sl], ok[sl]
        nv = int(v.sum())
        rows.append(dict(
            window=[int(lo), hi],
            valid_frac=float(v.mean()),
            median_area_px=float(np.median(area[sl][v])) if nv else 0.0,
            sliver_frac=float((v & ~o).sum() / nv) if nv else 1.0,
            mean_cams_ok=float(o.sum(axis=1).mean()),
            frac_frames_ok=float(keep[sl].mean()),
        ))
    return rows, dict(zip(cams, (float(x) for x in ref)))


def plot_timelines(masks_dir, bout_ids, excl, min_area_frac, min_cameras, pct,
                   out_dir, known_bad=None):
    """Per-frame tracking-quality timeline, one figure per bout, both flies.

    EXPECTATION, WRITTEN BEFORE THE FIGURE (CLAUDE.md): on the CALIBRATION bout
    28 the male (fly1, orange) should be a flat line at 7 usable cameras and an
    area ratio pinned near 1.0 for all 2007 frames, while the female (fly0,
    white) should be identical to him until ~frame 1500 and then FALL OFF A
    CLIFF -- usable cameras dropping to 2-3, area ratio to 0.1-0.3 -- inside the
    shaded stretch the notes record as her SAM-mask collapse. On a bout chosen
    as WELL TRACKED both flies should look like bout 28's male throughout, with
    no shaded region and no camera dropping out. If a chosen bout shows the
    female dipping anywhere, the ranking picked wrong.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from viz.core.colors import PALETTE

    # PALETTE is BGR (it is the cv2 palette); matplotlib wants RGB fractions.
    # fly0 = cyan / observed, fly1 = orange -- the repo's shared visual language.
    def _mpl(name):
        b, g, r = PALETTE[name]
        return (r / 255.0, g / 255.0, b / 255.0)

    col = {0: _mpl("fly0"), 1: _mpl("fly1")}
    for idx, npz in _bout_dirs(masks_dir, set(bout_ids)):
        cams, flies, _ = _load_bout(npz, exclude_cameras=excl)
        fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
        for f, d in sorted(flies.items()):
            ref = area_reference(d["area"], d["valid"], pct=pct)
            ok = camera_validity(d["valid"], d["area"], ref,
                                 min_area_frac=min_area_frac)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(d["valid"], d["area"] / ref[None, :], np.nan)
            lbl = f"fly{f}" + (" (male)" if f == 1 else " (female)")
            ax[0].plot(ok.sum(axis=1), lw=1.0, color=col[f], label=lbl)
            ax[1].plot(np.nanmedian(ratio, axis=1), lw=1.0, color=col[f], label=lbl)
        ax[0].axhline(min_cameras, color="#c04040", ls="--", lw=0.9,
                      label=f"gate: {min_cameras} usable cameras")
        ax[1].axhline(min_area_frac, color="#c04040", ls="--", lw=0.9,
                      label=f"gate: {min_area_frac:g} of own p{pct:g} area")
        for lo, hi in (known_bad or {}).get(idx, []):
            for a_ in ax:
                a_.axvspan(lo, hi, color="#c04040", alpha=0.10, lw=0)
        ax[0].set_ylabel(f"usable cameras\n(of {len(cams)})")
        ax[1].set_ylabel("median mask area\n/ own p%g" % pct)
        ax[1].set_xlabel("frame")
        ax[1].set_ylim(0, 1.6)
        ax[0].set_ylim(-0.2, len(cams) + 0.3)
        ax[0].set_title(f"SAM mask tracking quality -- Session0 bout {idx}, "
                        f"cameras {', '.join(cams)}", fontsize=9)
        for a_ in ax:
            a_.legend(fontsize=7, ncol=3, loc="lower left")
            a_.grid(alpha=0.2)
        fig.tight_layout()
        fp = os.path.join(out_dir, f"timeline_bout{idx:05d}.png")
        fig.savefig(fp, dpi=130)
        plt.close(fig)
        print(f"[mq] wrote {fp}")
        del flies


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--bouts", default="", help="comma-separated; default all")
    ap.add_argument("--exclude-cameras", default="",
                    help="comma-separated camera NAMES dropped before scoring")
    ap.add_argument("--min-area-frac", type=float, default=0.25)
    ap.add_argument("--min-cameras", type=int, default=3)
    ap.add_argument("--area-ref-pct", type=float, default=75.0)
    ap.add_argument("--window", type=int, default=200,
                    help="worst-window length in frames")
    ap.add_argument("--calibrate", type=int, default=None,
                    help="bout index whose known good/bad split must be reproduced")
    ap.add_argument("--plot-bouts", default="",
                    help="comma-separated bouts to draw a per-frame timeline for")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    only = ({int(x) for x in a.bouts.split(",") if x.strip()}
            if a.bouts.strip() else None)
    excl = [c for c in a.exclude_cameras.split(",") if c.strip()]
    bouts = _bout_dirs(a.masks_dir, only)
    if not bouts:
        raise SystemExit(f"no bout_* dirs with sam3_masks.npz under {a.masks_dir}")

    results, calib = [], None
    for idx, npz in bouts:
        cams, flies, sex_meta = _load_bout(npz, exclude_cameras=excl)
        for f, d in sorted(flies.items()):
            q = bout_fly_quality(d["area"], d["valid"], d["centroids"],
                                 min_area_frac=a.min_area_frac,
                                 min_cameras=a.min_cameras,
                                 pct=a.area_ref_pct, window=a.window)
            row = {k: v for k, v in q.items()
                   if k not in ("frame_ok", "camera_ok")}
            row.update(bout=idx, fly=f"fly{f}", cameras=cams, sex_meta=sex_meta)
            results.append(row)
            print(f"[mq] bout {idx:3d} fly{f}  score {row['score']:.3f}  "
                  f"frac_ok {row['frac_frames_ok']:.3f}  "
                  f"worst200 {row['worst_window_frac_ok']:.3f}  "
                  f"sliver {row['sliver_frac']:.3f}  "
                  f"area {row['median_area_px']:.0f}px  "
                  f"cams_ok {row['mean_cams_ok']:.2f}", flush=True)
        if a.calibrate is not None and idx == a.calibrate:
            calib = {}
            for f, d in sorted(flies.items()):
                rows, ref = _window_report(
                    d["area"], d["valid"], cams,
                    [w for w, _ in CALIBRATION[f"fly{f}"]],
                    a.min_area_frac, a.min_cameras, a.area_ref_pct)
                calib[f"fly{f}"] = dict(windows=rows, area_ref_px=ref)
        del flies

    results.sort(key=lambda r: -r["score"])
    out = dict(masks_dir=a.masks_dir, exclude_cameras=excl,
               min_area_frac=a.min_area_frac, min_cameras=a.min_cameras,
               area_ref_pct=a.area_ref_pct, worst_window=a.window,
               ranking=results, calibration=calib)

    verdict = None
    if calib is not None:
        # THE VERDICT IS TAKEN ON THE SIGNAL VECTOR, NOT ON ONE NUMBER, and
        # that is a measured decision rather than a convenience. Scored on
        # `frac_frames_ok` alone at `min_cameras=3`, bout 28 fly0's 1500-1710
        # window reads 0.519 -- because with a mean of 3.35 usable cameras,
        # about half those frames still scrape three. Every other signal calls
        # it loudly: 47% slivers against 0.1%, 3.35 usable cameras against
        # 7.00, 4 882 px against 15 862. A one-number rule that lands in the
        # middle of a case this extreme would not separate a marginal bout at
        # all, so a window is BAD when ANY signal screams and GOOD only when
        # they ALL agree. The bands are far from every measured value on both
        # sides, so the split is reproduced, not fitted.
        ok_all = True
        lines = []
        n_cam = len(results[0]["cameras"]) if results else 7
        for fly, spec in CALIBRATION.items():
            for (w, truth), row in zip(spec, calib[fly]["windows"]):
                good = (row["sliver_frac"] <= 0.05
                        and row["mean_cams_ok"] >= 0.9 * n_cam
                        and row["frac_frames_ok"] >= 0.90)
                bad = (row["sliver_frac"] >= 0.25
                       or row["mean_cams_ok"] <= 0.6 * n_cam)
                pred = "GOOD" if good else ("BAD" if bad else "MID")
                ok_all &= (pred == truth)
                row["verdict"] = pred
                row["truth"] = truth
                lines.append(
                    f"  bout {a.calibrate} {fly} {str(row['window']):>12s}  "
                    f"valid {row['valid_frac']:.3f}  area {row['median_area_px']:8.0f}px  "
                    f"sliver {row['sliver_frac']:.3f}  cams_ok {row['mean_cams_ok']:.2f}  "
                    f"frac_ok {row['frac_frames_ok']:.3f}  -> {pred:4s} (truth {truth})")
        verdict = "PASS" if ok_all else "FAIL"
        print(f"\n[calibration on bout {a.calibrate}] {verdict}")
        print("\n".join(lines))
        out["calibration_verdict"] = verdict

    jp = os.path.join(a.out, "tracking_quality.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=1)

    md = [f"| rank | bout | fly | score | frac_ok | worst{a.window} | sliver | "
          f"median area px | cams_ok | valid | T |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        md.append(f"| {i} | {r['bout']} | {r['fly']} | {r['score']:.3f} | "
                  f"{r['frac_frames_ok']:.3f} | {r['worst_window_frac_ok']:.3f} | "
                  f"{r['sliver_frac']:.3f} | {r['median_area_px']:.0f} | "
                  f"{r['mean_cams_ok']:.2f} | {r['valid_frac']:.3f} | "
                  f"{r['n_frames']} |")
    mp = os.path.join(a.out, "tracking_quality.md")
    with open(mp, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"\n[mq] wrote {jp}\n[mq] wrote {mp}")
    if a.plot_bouts.strip():
        plot_timelines(a.masks_dir,
                       [int(x) for x in a.plot_bouts.split(",") if x.strip()],
                       excl, a.min_area_frac, a.min_cameras, a.area_ref_pct,
                       a.out, known_bad={28: [(1500, 2007)]})
    if verdict == "FAIL":
        raise SystemExit("calibration FAILED -- the score does not reproduce "
                         "bout 28's known good/bad split; fix it before use")


if __name__ == "__main__":
    main()
