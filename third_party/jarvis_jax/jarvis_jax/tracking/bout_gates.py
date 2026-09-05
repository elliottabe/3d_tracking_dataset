"""Coarse-pass gates + bout-summary CSV, shared by both coarse-track schemas.

Spec: `docs/specs/2026-08-31-pipeline-schematic-and-inventory-design.md` s2
(SAM3 gates) and `docs/specs/2026-09-04-mvq-maskfree-frontend-design.md` s5.1
(mvq gates, this module's addition -- P4b task 5). `scripts/coarse_pass_gates.py`
is a thin CLI wrapper around this module; it re-exports every name here so
existing `import coarse_pass_gates as gates` callers (tests, `scripts/viz/
coarse_pass_timeline.py`) are unaffected.

Reads the `coarse_tracks.npz` written by EITHER `scripts/coarse_pass.py`
(SAM3, GPU, queued) or `jarvis_jax.tracking.coarse_track.write_coarse_tracks`
(mvq, mask-free -- see that module's docstring for the full schema) and
applies gates to emit a bout-summary CSV in TODAY'S EXACT SCHEMA (fly_id,
bout_idx, start_frame, end_frame, source_fly) -- the drop-in `run_bout.py`
already reads via `recording.bouts_csv` + `bout_ids`. This stage is pure
CPU/numpy/pandas: no GPU, no SAM3, safe to run locally and iterate on gate
thresholds without re-running the coarse pass.

Both schemas share the SAME two-gate structure -- `trackability_ok(t)` (is
each fly locatable) AND `behaviour_ok(t)` (are they doing courtship-shaped
things) -- but the SIGNALS differ, because the mvq coarse pass has no masks
(`area`/`area_med` are all-NaN by construction, see `coarse_track.py`'s
docstring) and instead carries per-fly existence, wing angle and a 3D
(not just reprojected-2D) separation:

  SAM3 trackability_ok(t) -- for EACH fly: area_med(fly,t) / baseline(fly,t)
      >= 0.5 (the exact -50% threshold from the spec's calibration evidence
      -- do NOT re-derive), border_med(fly,t) >= BORDER_MIN_PX (~50 px,
      spec), n_valid_cams(fly,t) >= MIN_CAMS (3, spec's hard floor) -- AND
      the two flies are separable (see below).

  mvq trackability_ok(t) -- for EACH fly: exist(fly,t) >= 0.5 (the typed-slot
      existence sigmoid, `coarse_track.TRACKABLE_EXIST`) AND n_valid_cams
      (fly,t) >= MIN_CAMS. NO area gate (there is no area).

  BOTH schemas -- the two flies must be separable: median per-camera 2D
      centroid separation over shared-valid cams (`sep2d_med`, computed from
      the reprojected 3D centroid even when there are no masks) exceeds
      SEP_MIN_PX, i.e. not merged. "own baseline" for the SAM3 area gate is a
      per-fly ROLLING median over BASELINE_WINDOW coarse frames (adaptive; a
      bout's own resting period is NOT used as its own baseline the way the
      spec's calibration example did locally, because a coarse pass does not
      know bout boundaries in advance).

  SAM3 behaviour_ok(t) -- wing extension (male's area_ratio > WING_RATIO_MIN,
      "above baseline", spec's language; MALE-specific because "the male's
      area is genuinely larger when extended", spec) OR close proximity
      (sep2d_med < PROXIMITY_MAX_PX px, "following").

  mvq behaviour_ok(t) -- wing extension (male's wing_angle_deg >=
      WING_ANGLE_MIN degrees, the model's own wing-vs-body-axis angle, no
      mask needed) OR close proximity (sep3d <= PROXIMITY_MAX_UNITS world
      units, the TRUE 3D distance rather than a 2D reprojection). Units in
      the flag names say which schema they belong to: SAM3 gates are in px
      (`--proximity-max-px`), mvq gates are in world units
      (`--proximity-max-units`) and degrees (`--wing-angle-min`).

  Neither schema's WING_RATIO_MIN/WING_ANGLE_MIN/PROXIMITY_MAX_* is given an
  explicit number by the spec (only the SAM3 trackability thresholds are) --
  they are tuned/defaulted here and reported as judgment calls, not measured
  constants.

in_bout(t) = trackability_ok(t) AND behaviour_ok(t); contiguous runs (with a
minimum duration and small-gap bridging, both tunable) become bout windows.

SCHEMA DETECTION (`is_mvq_schema`). An npz is mvq when its `.meta.json` says
`source == "mvq"`, OR -- so `compute_gate_signals(z)` alone, with no meta,
still dispatches correctly, e.g. when called directly on a bare `z` as
`test_coarse_track.py`'s SAM3-loader test does -- when the npz itself carries
the mvq-only `exist` array, which `scripts/coarse_pass.py` (SAM3) never
writes.
"""
import argparse
import csv
import json
import os

import numpy as np

try:
    import pandas as pd
except ImportError:  # pandas is in the 3d_tracking env; keep a numpy fallback
    pd = None

BORDER_MIN_PX = 50.0        # spec s2, verbatim (SAM3 only)
AREA_RATIO_MIN = 0.5        # spec s2, verbatim ("< 50% of own rolling median") (SAM3 only)
MIN_CAMS = 3                # spec s2, verbatim ("hard floor, unchanged") (both schemas)
SEP_MIN_PX = 15.0           # judgment call: below this, centroids likely merged (both schemas)
BASELINE_WINDOW = 3000      # coarse frames (~48,000 real frames, ~1 min); judgment call (SAM3)
EXIST_MIN = 0.5             # mvq only: `coarse_track.TRACKABLE_EXIST`, typed-slot existence

# SAM3 behaviour-gate defaults (unchanged from the pre-mvq script).
WING_RATIO_MIN_DEFAULT = 1.15
PROXIMITY_MAX_PX_DEFAULT = 120.0
# mvq behaviour-gate defaults (spec s5.1 / P4b task 5 brief).
WING_ANGLE_MIN_DEFAULT = 30.0
PROXIMITY_MAX_UNITS_DEFAULT = 30.0


def rolling_median(x, window, min_periods=50):
    if pd is not None:
        return pd.Series(x).rolling(window, center=True, min_periods=min_periods).median().to_numpy()
    # Minimal numpy fallback (slower, only used if pandas is unavailable).
    n = len(x)
    out = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = x[lo:hi]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= min_periods:
            out[i] = np.median(seg)
    return out


def load_tracks(tracks_path):
    z = np.load(tracks_path, allow_pickle=True)
    meta_path = tracks_path.rsplit(".npz", 1)[0] + ".meta.json"
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    return z, meta


def is_mvq_schema(z, meta=None):
    """True when `z`/`meta` are the mvq coarse-track schema (`jarvis_jax.
    tracking.coarse_track.write_coarse_tracks`) rather than the SAM3 one
    (`scripts/coarse_pass.py`). See the module docstring's SCHEMA DETECTION
    note: either the meta json says so, or the npz carries the mvq-only
    `exist` array."""
    if meta and meta.get("source") == "mvq":
        return True
    return "exist" in z


def compute_gate_signals(z, *, baseline_window=BASELINE_WINDOW, meta=None):
    area_med = z["area_med"]        # (2,T) -- all-NaN for mvq (no masks)
    border_med = z["border_med"]    # (2,T) -- all-NaN for mvq
    n_valid_cams = z["n_valid_cams"]  # (2,T)
    sep2d_med = z["sep2d_med"]      # (T,) -- meaningful for BOTH schemas (from the
                                    # reprojected 3D centroid, not from masks)
    sep3d = z["sep3d"] if z["sep3d"].size else None
    num_animals, T = area_med.shape

    baseline = np.stack([rolling_median(area_med[f], baseline_window) for f in range(num_animals)])
    with np.errstate(invalid="ignore", divide="ignore"):
        area_ratio = area_med / baseline

    sig = dict(area_med=area_med, border_med=border_med, n_valid_cams=n_valid_cams,
               sep2d_med=sep2d_med, sep3d=sep3d, baseline=baseline, area_ratio=area_ratio,
               mvq=is_mvq_schema(z, meta))
    if sig["mvq"]:
        sig["exist"] = np.asarray(z["exist"])
        sig["wing_angle_deg"] = np.asarray(z["wing_angle_deg"])
    return sig


def apply_gates(sig, *, wing_ratio_min=None, proximity_max_px=None,
                wing_angle_min=None, proximity_max_units=None,
                sep_min_px=SEP_MIN_PX, border_min_px=BORDER_MIN_PX,
                area_ratio_min=AREA_RATIO_MIN, min_cams=MIN_CAMS,
                exist_min=EXIST_MIN, male_slot=1):
    """Dispatches on `sig["mvq"]` (from `compute_gate_signals`): SAM3 sigs use
    `wing_ratio_min`/`proximity_max_px` (px, area-based); mvq sigs use
    `wing_angle_min`/`proximity_max_units` (deg / world units, existence
    +wing-angle-based) -- see the module docstring. The SAM3 branch is
    byte-identical to the pre-mvq script (regression-tested)."""
    area_ratio, border_med, n_valid_cams, sep2d_med = (
        sig["area_ratio"], sig["border_med"], sig["n_valid_cams"], sig["sep2d_med"])
    num_animals, T = area_ratio.shape
    mvq = bool(sig.get("mvq", False))

    with np.errstate(invalid="ignore"):
        separable = sep2d_med >= sep_min_px
    separable = np.nan_to_num(separable.astype(float), nan=0.0).astype(bool)

    if mvq:
        exist = sig["exist"]
        with np.errstate(invalid="ignore"):
            per_fly_trackable = (exist >= exist_min) & (n_valid_cams >= min_cams)
    else:
        with np.errstate(invalid="ignore"):
            per_fly_trackable = (area_ratio >= area_ratio_min) & (border_med >= border_min_px)
        per_fly_trackable = per_fly_trackable & (n_valid_cams >= min_cams)
    per_fly_trackable = np.nan_to_num(per_fly_trackable.astype(float), nan=0.0).astype(bool)
    trackable_both = per_fly_trackable.all(axis=0) if num_animals >= 2 else per_fly_trackable[0]

    trackability_ok = trackable_both & separable

    if mvq:
        wam = WING_ANGLE_MIN_DEFAULT if wing_angle_min is None else wing_angle_min
        pmu = PROXIMITY_MAX_UNITS_DEFAULT if proximity_max_units is None else proximity_max_units
        wing_angle_deg = sig["wing_angle_deg"]
        sep3d = sig["sep3d"]
        with np.errstate(invalid="ignore"):
            wing_extension = wing_angle_deg[male_slot] >= wam
            close_proximity = sep3d <= pmu
    else:
        wrm = WING_RATIO_MIN_DEFAULT if wing_ratio_min is None else wing_ratio_min
        pmp = PROXIMITY_MAX_PX_DEFAULT if proximity_max_px is None else proximity_max_px
        with np.errstate(invalid="ignore"):
            wing_extension = area_ratio[male_slot] > wrm
            close_proximity = sep2d_med < pmp
    wing_extension = np.nan_to_num(wing_extension.astype(float), nan=0.0).astype(bool)
    close_proximity = np.nan_to_num(close_proximity.astype(float), nan=0.0).astype(bool)
    behaviour_ok = wing_extension | close_proximity

    in_bout = trackability_ok & behaviour_ok
    return dict(trackability_ok=trackability_ok, behaviour_ok=behaviour_ok,
                in_bout=in_bout, wing_extension=wing_extension,
                close_proximity=close_proximity, per_fly_trackable=per_fly_trackable,
                separable=separable, mvq=mvq)


def segment_runs(mask, *, min_duration, max_gap):
    """Contiguous True runs of `mask` (coarse-frame index space), bridging gaps
    <= max_gap and dropping runs shorter than min_duration. Returns [(start,end)]
    end EXCLUSIVE."""
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []
    runs = []
    s = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i - prev - 1 > max_gap:
            runs.append((s, prev + 1))
            s = i
        prev = i
    runs.append((s, prev + 1))
    return [(s, e) for s, e in runs if (e - s) >= min_duration]


def coarse_to_real(coarse_frame_arr, s, e):
    """(start,end) coarse INDICES [s,e) -> real-frame [start_frame,end_frame]
    inclusive, matching the ground-truth CSV's inclusive-end convention."""
    start_frame = int(coarse_frame_arr[s])
    end_frame = int(coarse_frame_arr[e - 1])
    return start_frame, end_frame


def write_bout_csv(out_csv, windows, *, fly_id, source_fly="both"):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fly_id", "bout_idx", "start_frame", "end_frame", "source_fly"])
        for i, (s, e) in enumerate(windows, start=1):
            w.writerow([fly_id, i, s, e, source_fly])


def load_ground_truth(csv_path, fly_id=None):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if fly_id and r.get("fly_id") not in (None, "", fly_id):
                continue
            rows.append((int(r["bout_idx"]), int(r["start_frame"]), int(r["end_frame"])))
    return rows


def overlap(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1])
    return max(0, e - s + 1)


def validate(windows, ground_truth):
    """recall/precision (any-overlap) + boundary agreement on best-overlap pairs."""
    gt_hit = [False] * len(ground_truth)
    win_hit = [False] * len(windows)
    pairs = []  # (gt_idx, win_idx, overlap_frames)
    for gi, (bidx, gs, ge) in enumerate(ground_truth):
        best = None
        for wi, (ws, we) in enumerate(windows):
            ov = overlap((gs, ge), (ws, we))
            if ov > 0:
                gt_hit[gi] = True
                win_hit[wi] = True
                if best is None or ov > best[2]:
                    best = (gi, wi, ov)
        if best:
            pairs.append(best)
    recall = sum(gt_hit) / len(ground_truth) if ground_truth else float("nan")
    precision = sum(win_hit) / len(windows) if windows else float("nan")
    start_offsets, end_offsets = [], []
    per_bout = []
    for gi, wi, ov in pairs:
        bidx, gs, ge = ground_truth[gi]
        ws, we = windows[wi]
        start_offsets.append(ws - gs)
        end_offsets.append(we - ge)
        per_bout.append(dict(bout_idx=bidx, gt=(gs, ge), emitted=(ws, we),
                             start_offset=ws - gs, end_offset=we - ge))
    return dict(recall=recall, precision=precision, n_gt=len(ground_truth),
                n_windows=len(windows), n_matched=len(pairs),
                start_offset_median=float(np.median(start_offsets)) if start_offsets else None,
                end_offset_median=float(np.median(end_offsets)) if end_offsets else None,
                start_offset_mean=float(np.mean(start_offsets)) if start_offsets else None,
                end_offset_mean=float(np.mean(end_offsets)) if end_offsets else None,
                unmatched_gt=[ground_truth[gi][0] for gi in range(len(ground_truth)) if not gt_hit[gi]],
                unmatched_windows=[windows[wi] for wi in range(len(windows)) if not win_hit[wi]],
                per_bout=per_bout)


def run(tracks_path, out_csv, *, session_tag, wing_ratio_min, proximity_max_px,
        min_duration, max_gap, baseline_window,
        wing_angle_min=WING_ANGLE_MIN_DEFAULT,
        proximity_max_units=PROXIMITY_MAX_UNITS_DEFAULT, ground_truth_csv=None):
    z, meta = load_tracks(tracks_path)
    sig = compute_gate_signals(z, baseline_window=baseline_window, meta=meta)
    gates = apply_gates(sig, wing_ratio_min=wing_ratio_min, proximity_max_px=proximity_max_px,
                        wing_angle_min=wing_angle_min, proximity_max_units=proximity_max_units)
    runs = segment_runs(gates["in_bout"], min_duration=min_duration, max_gap=max_gap)
    coarse_frame = z["coarse_frame"]
    windows = [coarse_to_real(coarse_frame, s, e) for s, e in runs]
    write_bout_csv(out_csv, windows, fly_id=session_tag)
    print(f"[gates] {'mvq' if gates['mvq'] else 'sam3'} schema -> {len(windows)} window(s) -> {out_csv}")
    for i, (s, e) in enumerate(windows, start=1):
        print(f"  bout_idx {i}: [{s},{e}]  ({e - s + 1} frames)")

    report = None
    if ground_truth_csv:
        gt = load_ground_truth(ground_truth_csv, fly_id=session_tag)
        report = validate(windows, gt)
        print(f"[validate] recall={report['recall']:.3f} ({report['n_matched']}/{report['n_gt']}) "
              f"precision={report['precision']:.3f} ({sum(1 for w in windows if any(overlap((s,e),(w[0],w[1]))>0 for _,s,e in gt))}/{report['n_windows']})")
        print(f"[validate] start_offset median={report['start_offset_median']} "
              f"end_offset median={report['end_offset_median']}")
        print(f"[validate] unmatched GT bouts: {report['unmatched_gt']}")
        for pb in report["per_bout"]:
            if pb["bout_idx"] == 28:
                print(f"[validate] bout 28: GT={pb['gt']} EMITTED={pb['emitted']} "
                      f"start_offset={pb['start_offset']} end_offset={pb['end_offset']}")
    return dict(windows=windows, gates=gates, sig=sig, coarse_frame=coarse_frame,
                meta=meta, report=report)


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", required=True, help="coarse_tracks.npz from coarse_pass.py")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--session-tag", required=True, help="e.g. Session0/2025_10_20_13_20_04")
    ap.add_argument("--wing-ratio-min", type=float, default=WING_RATIO_MIN_DEFAULT,
                    help="SAM3 schema: male area_ratio above which counts as wing extension")
    ap.add_argument("--proximity-max-px", type=float, default=PROXIMITY_MAX_PX_DEFAULT,
                    help="SAM3 schema: sep2d_med (px) at/below which counts as close proximity")
    ap.add_argument("--wing-angle-min", type=float, default=WING_ANGLE_MIN_DEFAULT,
                    help="mvq schema: male wing_angle_deg at/above which counts as wing extension")
    ap.add_argument("--proximity-max-units", type=float, default=PROXIMITY_MAX_UNITS_DEFAULT,
                    help="mvq schema: sep3d (world units) at/below which counts as close proximity")
    ap.add_argument("--min-duration", type=int, default=3, help="coarse frames")
    ap.add_argument("--max-gap", type=int, default=2, help="coarse frames to bridge")
    ap.add_argument("--baseline-window", type=int, default=BASELINE_WINDOW)
    ap.add_argument("--ground-truth", default=None, help="known bout-summary CSV to validate against")
    return ap


def main():
    args = build_arg_parser().parse_args()
    run(args.tracks, args.out_csv, session_tag=args.session_tag,
        wing_ratio_min=args.wing_ratio_min, proximity_max_px=args.proximity_max_px,
        wing_angle_min=args.wing_angle_min, proximity_max_units=args.proximity_max_units,
        min_duration=args.min_duration, max_gap=args.max_gap,
        baseline_window=args.baseline_window, ground_truth_csv=args.ground_truth)


if __name__ == "__main__":
    main()
