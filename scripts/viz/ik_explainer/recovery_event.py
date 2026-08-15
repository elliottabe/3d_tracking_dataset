"""Data for the recovery clip: one keypoint's three 3D tracks plus the
detector's own 2D (and every camera's disagreement with the raw consensus),
over the window where detection failed.

The event was chosen by measurement (see
docs/specs/2026-08-14-recovery-clip-design.md): T1R_TaTip on Cam2012853 is
simultaneously the bout's worst detector-vs-consensus disagreement (122 px at
frame 441, confidence 0.49) and its worst raw-3D spike (1.627 mm/frame^2 at
frame 443). Nothing here is computed fresh -- every array is read from disk.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from scripts.viz.ik_explainer import clip_io   # noqa: E402

# marker_sites in the production h5 are in model units; mm = value / shared_scale.
# The EXACT value is read at load time from `06_stages.npz` (the same field
# `assemble.py:272` reads), never hardcoded: if the IK is ever re-run at a
# different body scale, a stale constant would render the IK trace in wrong mm
# while it still looked perfectly smooth -- the clip's claim would survive and
# be false. CLAUDE.md records a 38x body-scale error that residual/NaN checks
# were blind to for exactly this reason.
#
# The constant below is only a SANITY BOUND on the loaded value (the scale this
# clip was measured at): a genuinely different scale must fail loudly here
# rather than render quietly.
SHARED_SCALE_NOMINAL = 0.1261
SHARED_SCALE_TOL_FRAC = 0.01          # +-1% of nominal

EVENT = {
    "kp": "T1R_TaTip",
    "cam": "Cam2012853",
    "t0": 420,
    "t1": 465,
    "peak_2d": 441,   # worst detector-vs-consensus disagreement in the bout
    "peak_3d": 443,   # worst raw triangulation spike
}


def load_shared_scale(clip=clip_io.CLIP_DEFAULT):
    """The body scale the production IK was actually solved at, from disk.

    Read from `<clip>/ik_explainer/predictions/06_stages.npz` -- the file
    `stage_ik.py` writes it to and `assemble.py:272` already reads it from --
    and checked against `SHARED_SCALE_NOMINAL` so a different scale raises
    instead of silently rescaling the clip's IK trace.
    """
    d = clip_io.out_dirs(clip)
    with np.load(d["predictions"] / "06_stages.npz", allow_pickle=True) as z:
        scale = float(z["shared_scale"])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"06_stages.npz shared_scale is not usable: {scale!r}")
    dev = abs(scale - SHARED_SCALE_NOMINAL) / SHARED_SCALE_NOMINAL
    if dev > SHARED_SCALE_TOL_FRAC:
        raise ValueError(
            f"06_stages.npz shared_scale {scale:.6f} differs from the nominal "
            f"{SHARED_SCALE_NOMINAL} by {dev * 100:.2f}% (tol "
            f"{SHARED_SCALE_TOL_FRAC * 100:.0f}%) -- the IK was solved at a "
            "different body scale than this clip was measured at; refusing to "
            "render mm values against it")
    return scale


def accel(track):
    """(n,3) positions -> (n-2,) acceleration magnitude per frame."""
    v = np.diff(np.asarray(track, np.float64), axis=0)
    return np.linalg.norm(np.diff(v, axis=0), axis=-1)


def worst_axis(raw):
    """Index of the coordinate with the largest excursion about its median.

    Used to choose which of x/y/z to plot: the one where the failure is most
    legible. Returned rather than hardcoded so the choice stays correct if the
    window or keypoint changes.
    """
    a = np.asarray(raw, np.float64)
    return int(np.argmax(np.nanmax(np.abs(a - np.nanmedian(a, axis=0)), axis=0)))


def load_tracks(clip=clip_io.CLIP_DEFAULT, kp_name=EVENT["kp"],
                t0=EVENT["t0"], t1=EVENT["t1"], cam_name=EVENT["cam"]):
    """Three 3D tracks + the detector's 2D for one keypoint over [t0, t1).

    Keys: `raw`/`filt`/`ik` (n,3) mm; `det2d`/`rep2d` (n,2) px and `conf` (n,)
    for `cam_name`; `det2d_all`/`rep2d_all` (n, n_cams, 2) px and `conf_all`
    (n, n_cams) for EVERY camera; `cam_disagree` (n, n_cams) px
    detector-vs-raw-reprojection; `cam_names` giving the camera-axis order of
    all four `*_all` arrays; `frames` (n,) source frame indices.

    The single-camera `det2d`/`rep2d`/`conf` are slices of the `*_all` arrays
    at `cam_names.index(cam_name)`, not separately computed, so the event
    camera's panel and the extra camera panels cannot diverge.
    """
    d = clip_io.out_dirs(clip)
    kp_names = clip_io.model_kp_names()
    k = kp_names.index(kp_name)          # by NAME -- never a positional guess

    z2 = np.load(d["predictions"] / "02_kp2d.npz", allow_pickle=True)
    cam_names = [str(c) for c in z2["cam_names"]]
    if [str(n) for n in z2["kp_names"]] != kp_names:
        raise ValueError("02_kp2d.npz is not in MODEL keypoint order")
    c = cam_names.index(cam_name)        # by NAME

    z_raw = np.load(d["predictions"] / "03_kp3d.npz", allow_pickle=True)
    if [str(n) for n in z_raw["kp_names"]] != kp_names:
        raise ValueError("03_kp3d.npz is not in MODEL keypoint order")
    raw = z_raw["kp3d"]

    z_flt = np.load(d["predictions"] / "04_kp3d_filt.npz", allow_pickle=True)
    if [str(n) for n in z_flt["kp_names"]] != kp_names:
        raise ValueError("04_kp3d_filt.npz is not in MODEL keypoint order")
    flt = z_flt["kp3d"]
    shared_scale = load_shared_scale(clip)                  # from disk, not a constant
    with h5py.File(Path(clip) / "ik_production" / "stac_ik_full.h5", "r") as f:
        if [s.decode() for s in f["kp_names"][:]] != kp_names:
            raise ValueError("stac_ik_full.h5 is not in MODEL keypoint order")
        ik = f["marker_sites"][t0:t1, k] / shared_scale     # -> mm

    cam_mats, dlt_names = clip_io.load_dlt(str(Path(clip) / "calibration"))
    if dlt_names != cam_names:
        raise ValueError("calibration and kp2d disagree on camera order")

    # Reproject the FILTERED 3D into EVERY camera: what the pipeline believes,
    # seen from each view. The event camera's own `rep2d` is a slice of this
    # (below) rather than a second projection, so the panels cannot disagree.
    rep_flt_all = np.stack([clip_io.project(cam_mats, flt[t])[:, k]
                            for t in range(t0, t1)])            # (n, C, 2)

    # Per-camera detector-vs-RAW-triangulation disagreement, (n_frames, n_cams)
    # px, via the same DLT path `rep` uses. This is the quantity the on-screen
    # caveat quotes ("the other cameras also disagree here"): it says how far
    # each camera's own 2D detection sits from the consensus 3D that all seven
    # produced, so it must be measured against the RAW triangulation (the
    # consensus itself), not against the filtered track (which has already
    # absorbed part of the failure). Returned so the clip can compute the
    # caveat's number at render time instead of quoting the design doc.
    rep_raw_all = np.stack([clip_io.project(cam_mats, raw[t])[:, k]
                            for t in range(t0, t1)])            # (n, C, 2)
    det_all = z2["kp2d"][t0:t1, :, k].astype(np.float64)        # (n, C, 2)
    cam_disagree = np.linalg.norm(det_all - rep_raw_all, axis=-1)

    return {
        # real Cam20128xx names, in the column order of `cam_disagree` -- the
        # ONE camera-identity path in this clip (display "Camera N" labels are
        # derived from these, never the other way round).
        "cam_names": cam_names,
        "cam_disagree": cam_disagree.astype(np.float64),
        "raw": raw[t0:t1, k].astype(np.float64),
        "filt": flt[t0:t1, k].astype(np.float64),
        "ik": np.asarray(ik, np.float64),
        "det2d_all": det_all,
        "rep2d_all": rep_flt_all.astype(np.float64),
        "conf_all": z2["conf"][t0:t1, :, k].astype(np.float64),
        "det2d": det_all[:, c],
        "conf": z2["conf"][t0:t1, c, k].astype(np.float64),
        "rep2d": rep_flt_all[:, c].astype(np.float64),
        "frames": np.arange(t0, t1),
    }


def select_panel_cams(tracks, event=None, n_extra=2):
    """The event camera plus `n_extra` others, chosen by measurement.

    The extras bracket the caveat's claim rather than illustrating one half of
    it: the worst remaining camera (the failure is not confined to one view)
    and the cleanest remaining camera (yet some views see it correctly). Picked
    on each camera's PEAK disagreement across the whole window, not its value
    at a single frame, so a camera that fails a few frames early or late still
    counts as a bad view.

    Returns real `Cam20128xx` names -- the display "Camera N" labels are
    derived from these downstream, never the other way round.
    """
    e = EVENT if event is None else event
    cam_names = list(tracks["cam_names"])
    if e["cam"] not in cam_names:
        raise ValueError(f"event camera {e['cam']} not in {cam_names}")
    peak = np.nanmax(np.asarray(tracks["cam_disagree"], np.float64), axis=0)
    others = [n for n in cam_names if n != e["cam"]]
    if n_extra > len(others):
        raise ValueError(f"asked for {n_extra} extra cameras, only {len(others)} exist")
    ranked = sorted(others, key=lambda n: peak[cam_names.index(n)], reverse=True)
    # worst, cleanest, then second-worst, second-cleanest, ... -- so any
    # n_extra keeps the bracketing property rather than degenerating to
    # "the n worst".
    picks, lo, hi = [], 0, len(ranked) - 1
    while len(picks) < n_extra:
        picks.append(ranked[lo]); lo += 1
        if len(picks) < n_extra:
            picks.append(ranked[hi]); hi -= 1
    return [e["cam"]] + picks
