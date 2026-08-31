"""Mine hard frames for the WingL/R_V12/V13 left-right mirror-confusion defect.

THE FAILURE (see .superpowers/sdd/2026-08-29-coarse-to-fine-3d/{wing-spikes,
male-wing-mirror,wing-gating}.md). The 2D detector confidently places
`WingL_V13` (and by symmetry `WingR_V13`, and the V12 veins) on the wrong
wing structure: the landmark toggles between two different, both-visible
wing structures ~170-330px apart in a single camera, both at 0.9+
confidence. `male-wing-mirror.md` measured 94.4% of the male's large (>30px)
2D wing errors are mirror-shaped on bout 28 -- this module asks how far that
generalizes.

DETECTION RULE (Signal A -- mirror-reprojection test, validated below against
bout 28's existing hand-labelled ground truth before being trusted on
anything else): for a wing-vein keypoint K with mirror partner K', at each
(frame, camera):
    d_own    = |raw 2D detection - reprojection of the pipeline's own fitted
                3D for K|
    d_mirror = |raw 2D detection - reprojection of the fitted 3D for K's
                mirror partner K'|
    mirror_confused = d_own > 30px (this project's own "unambiguous swap"
                       threshold, TRIGGER_FACTOR*reproj_resid_px) AND
                       d_mirror < d_own AND d_mirror <= TIGHT_MIRROR_PX(20px)
A FRAME is flagged if >=2 camera views are mirror_confused for the SAME
wing-vein keypoint in that frame (>=2 chosen, not >=1 or >=3, by sweeping
against bout 28's ground truth -- see `validate_bout28` and the report),
AND that keypoint's own/mirror bone length (distance to its wing base) is
anatomically plausible for this recording (see BONE_LENGTH_MULT docstring
note) -- both refinements were added AFTER a genome-wide run of the naive
rule turned out to fire heavily on unrelated triangulation breakdowns (see
report Section 1); `mirror_confused_loose` (no tight-match requirement) is
kept alongside for the honest before/after comparison.

A second signal (B -- frame-to-frame 2D displacement > 30px, the
already-validated `wing-gating.md` Candidate-B floor) is computed alongside
for comparison/corroboration, not as the primary rule (see report for why).

Nothing here launches training, retriangulates with a different gate, or
touches jarvis_jax/data/* or scripts/run_bout.py. Everything is a read-only
pass over already-computed kp2d.npz/kp3d.npz.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_fly50_order():
    """The DETECTOR-TRAINING keypoint order (`data/fly50.json`, head-first:
    Antenna_Base, EyeL, EyeR, Scutellum, ...) -- DIFFERENT from KP_NAMES
    above (the model/kp2d.npz/kp3d.npz order, Scutellum-first). See
    `scripts/build_detector_dataset.py`'s own module docstring: "fly50
    order is NOT the same as cfg.model.KP_NAMES ... Do not cross the two."
    `build_dataset` below writes COCO `keypoints` in FLY50 order (verified
    against the real red_data_3d_v5's own `annotations/keypoint_names.json`,
    which is the plain fly50 list) precisely so this mined set is a
    drop-in-order match for what V5Dataset/the existing detector-training
    path actually expects -- writing model order here would have been a
    silent, high-consequence keypoint-order bug, the exact class this
    project has been bitten by before.
    """
    with open(os.path.join(REPO, "data", "fly50.json")) as f:
        return list(json.load(f)["node_names"])


for _p in (REPO, os.path.join(REPO, "third_party/jarvis_jax")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from viz.core import reproject  # noqa: E402

PROC_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship"
VIDEO_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship"

KP_NAMES = [
    "Scutellum", "WingL_base", "WingR_base", "Antenna_Base", "EyeL", "EyeR",
    "WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13", "Abd_A4", "Abd_tip",
    "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip",
    "T1R_ThxCx", "T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip",
    "T2L_Tro", "T2L_FeTi", "T2L_TiTa", "T2L_TaT1", "T2L_TaT3", "T2L_TaTip",
    "T2R_Tro", "T2R_FeTi", "T2R_TiTa", "T2R_TaT1", "T2R_TaT3", "T2R_TaTip",
    "T3L_Tro", "T3L_FeTi", "T3L_TiTa", "T3L_TaT1", "T3L_TaT3", "T3L_TaTip",
    "T3R_Tro", "T3R_FeTi", "T3R_TiTa", "T3R_TaT1", "T3R_TaT3", "T3R_TaTip",
]
KP_IDX = {n: i for i, n in enumerate(KP_NAMES)}

VEINS = ["WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13"]
MIRROR_OF = {
    "WingL_V12": "WingR_V12", "WingL_V13": "WingR_V13",
    "WingR_V12": "WingL_V12", "WingR_V13": "WingL_V13",
    "WingL_base": "WingR_base", "WingR_base": "WingL_base",
}

LARGE_ERR_PX = 30.0    # TRIGGER_FACTOR * reproj_resid_px(10px)
DISP_PX = 30.0         # wing-gating.md Candidate-B validated floor
FRAME_CAM_THRESH = 2   # signal-A camera-agreement floor, validated on bout 28
TIGHT_MIRROR_PX = 20.0  # visual audit (see report): genuine "sits on the
                        # OTHER real wing structure" instances measure
                        # d_mirror ~6-8px (bout 28, and an independent
                        # Session1 male event); frames where d_mirror is
                        # merely SMALLER than d_own but still 30-200px (both
                        # candidates far from the raw detection) were, on
                        # inspection, NOT this failure -- mostly female
                        # wall/occlusion tracking collapse, or a folded-wing
                        # under-constrained triangulation. This floor keeps
                        # the former and drops the latter (both signal_A_mirror's
                        # returned dict and the mined set use "mirror_confused",
                        # which already ANDs this in -- "mirror_confused_loose"
                        # is kept alongside for the honest before/after report).
BASE_OF = {"WingL_V12": "WingL_base", "WingL_V13": "WingL_base",
           "WingR_V12": "WingR_base", "WingR_V13": "WingR_base"}
BONE_LENGTH_MULT = 1.75  # plausibility cutoff = this x the recording's own
                          # clean-frame median vein-to-base bone length (see
                          # report: a genuine mirror-lock still triangulates
                          # to an anatomically normal bone length -- ~21mm on
                          # bout 28's own reference, vs 183mm on a found
                          # triangulation-breakdown frame -- so this gate
                          # separates "two visible wing structures" from
                          # "kp3d is garbage for an unrelated reason").

GROUND_TRUTH_WINDOWS_BOUT28_FLY1 = {
    "ARTIFACT": [(0, 130)],
    "REAL": [(360, 375), (680, 700), (1500, 1560), (1810, 1870), (1930, 1990)],
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def session_recordings():
    out = []
    for session in ("Session0", "Session1"):
        sdir = os.path.join(PROC_ROOT, session)
        if not os.path.isdir(sdir):
            continue
        for rec in sorted(os.listdir(sdir)):
            bouts_dir = os.path.join(sdir, rec, "pose", "bouts")
            calib = os.path.join(VIDEO_ROOT, session, rec, "calibration")
            if os.path.isdir(bouts_dir) and os.path.isdir(calib):
                out.append((session, rec, calib))
    return out


def bout_fly_dirs(session, rec):
    bouts_dir = os.path.join(PROC_ROOT, session, rec, "pose", "bouts")
    out = []
    for b in sorted(os.listdir(bouts_dir)):
        bpath = os.path.join(bouts_dir, b)
        if not os.path.isdir(bpath):
            continue
        for fly in ("fly0", "fly1"):
            fdir = os.path.join(bpath, fly)
            if os.path.exists(os.path.join(fdir, "kp2d.npz")) and \
               os.path.exists(os.path.join(fdir, "kp3d.npz")):
                out.append((b, fly, fdir, bpath))
    return out


def bout_male_fly(bpath):
    p = os.path.join(bpath, "sex.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p)).get("male_fly")
    except Exception:
        return None


def bout_start_frame(session, rec, bout_idx):
    csv_path = os.path.join(PROC_ROOT, session, rec, "courtship_bout_summary.csv")
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if int(row["bout_idx"]) == bout_idx:
                return int(row["start_frame"])
    return None


_CAM_MAT_CACHE = {}


def get_cam_mats(calib_dir):
    if calib_dir not in _CAM_MAT_CACHE:
        _CAM_MAT_CACHE[calib_dir] = reproject.camera_matrices(calib_dir)
    return _CAM_MAT_CACHE[calib_dir]


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def load_bout_fly(fdir, calib_dir):
    z2 = np.load(os.path.join(fdir, "kp2d.npz"))
    kp2d, conf = np.asarray(z2["kp2d"], np.float64), np.asarray(z2["conf"], np.float64)
    kp3d = np.asarray(np.load(os.path.join(fdir, "kp3d.npz"))["kp3d"], np.float64)
    cam_mats, cam_names = get_cam_mats(calib_dir)
    T = min(kp2d.shape[0], kp3d.shape[0])
    C = kp2d.shape[1]
    kp2d, conf, kp3d = kp2d[:T], conf[:T], kp3d[:T]
    reproj = np.stack(
        [reproject.project(cam_mats[c], kp3d.reshape(-1, 3)).reshape(T, -1, 2)
         for c in range(C)], axis=1)  # (T,C,K,2)
    return dict(kp2d=kp2d, conf=conf, kp3d=kp3d, reproj=reproj,
                cam_names=cam_names, T=T, C=C)


def signal_A_mirror(data, keypoint):
    mirror_kp = MIRROR_OF[keypoint]
    k, km = KP_IDX[keypoint], KP_IDX[mirror_kp]
    raw = data["kp2d"][:, :, k, :]
    own_r = data["reproj"][:, :, k, :]
    mir_r = data["reproj"][:, :, km, :]
    d_own = np.linalg.norm(raw - own_r, axis=-1)
    d_mirror = np.linalg.norm(raw - mir_r, axis=-1)
    finite = np.isfinite(d_own) & np.isfinite(d_mirror)
    confused_loose = finite & (d_own > LARGE_ERR_PX) & (d_mirror < d_own)
    confused = confused_loose & (d_mirror <= TIGHT_MIRROR_PX)
    return dict(d_own=d_own, d_mirror=d_mirror, mirror_confused=confused,
                mirror_confused_loose=confused_loose)


def bone_length(data, keypoint):
    """(T,) distance from this vein's fitted 3D to its own wing-base 3D."""
    k, kb = KP_IDX[keypoint], KP_IDX[BASE_OF[keypoint]]
    return np.linalg.norm(data["kp3d"][:, k] - data["kp3d"][:, kb], axis=-1)


def signal_B_displacement(data, keypoint):
    k = KP_IDX[keypoint]
    raw = data["kp2d"][:, :, k, :]
    d_kp = np.full(raw.shape[:2], np.nan)
    d_kp[1:] = np.linalg.norm(raw[1:] - raw[:-1], axis=-1)
    flagged = np.isfinite(d_kp) & (d_kp > DISP_PX)
    return dict(d_kp=d_kp, flagged=flagged)


def frame_label_bout28(offset):
    for lo, hi in GROUND_TRUTH_WINDOWS_BOUT28_FLY1["ARTIFACT"]:
        if lo <= offset <= hi:
            return "ARTIFACT"
    for lo, hi in GROUND_TRUTH_WINDOWS_BOUT28_FLY1["REAL"]:
        if lo <= offset <= hi:
            return "REAL"
    return None


def validate_bout28(out_dir):
    """Sweep the camera-agreement threshold for Signal A (and check Signal
    B / A-and-B / A-or-B) against bout 28 fly1's hand-labelled ground truth.
    Writes a JSON report; used to pick FRAME_CAM_THRESH above."""
    fdir = os.path.join(PROC_ROOT, "Session0", "2025_10_20_13_20_04",
                         "pose", "bouts", "bout_00028", "fly1")
    calib = os.path.join(VIDEO_ROOT, "Session0", "2025_10_20_13_20_04", "calibration")
    data = load_bout_fly(fdir, calib)
    labels = np.array([frame_label_bout28(t) for t in range(data["T"])], dtype=object)
    gt_artifact, gt_real, gt_unlabeled = labels == "ARTIFACT", labels == "REAL", labels == None

    per_kp_A, per_kp_A_loose, per_kp_B = {}, {}, {}
    for kp in VEINS:
        sA = signal_A_mirror(data, kp)
        per_kp_A[kp] = sA["mirror_confused"].sum(axis=1)
        per_kp_A_loose[kp] = sA["mirror_confused_loose"].sum(axis=1)
        per_kp_B[kp] = signal_B_displacement(data, kp)["flagged"].sum(axis=1)

    report = {"n_frames": int(data["T"]), "sweep": []}
    for thresh in (1, 2, 3, 4):
        frameA = np.zeros(data["T"], dtype=bool)
        frameA_loose = np.zeros(data["T"], dtype=bool)
        for kp in VEINS:
            frameA |= per_kp_A[kp] >= thresh
            frameA_loose |= per_kp_A_loose[kp] >= thresh
        report["sweep"].append(dict(
            signal="A_tight(d_mirror<=20px)", thresh=thresh, n_flagged=int(frameA.sum()),
            artifact_recall=float(frameA[gt_artifact].mean()) if gt_artifact.any() else None,
            real_fp_rate=float(frameA[gt_real].mean()) if gt_real.any() else None,
            n_unlabeled_flagged=int((frameA & gt_unlabeled).sum()),
        ))
        report["sweep"].append(dict(
            signal="A_loose(d_mirror<d_own only)", thresh=thresh, n_flagged=int(frameA_loose.sum()),
            artifact_recall=float(frameA_loose[gt_artifact].mean()) if gt_artifact.any() else None,
            real_fp_rate=float(frameA_loose[gt_real].mean()) if gt_real.any() else None,
            n_unlabeled_flagged=int((frameA_loose & gt_unlabeled).sum()),
        ))
    frameA2 = np.zeros(data["T"], dtype=bool)
    frameB2 = np.zeros(data["T"], dtype=bool)
    for kp in VEINS:
        frameA2 |= per_kp_A[kp] >= FRAME_CAM_THRESH
        frameB2 |= per_kp_B[kp] >= FRAME_CAM_THRESH
    for name, mask in (("A_only", frameA2), ("B_only", frameB2),
                        ("A_and_B", frameA2 & frameB2), ("A_or_B", frameA2 | frameB2)):
        report[name] = dict(
            n_flagged=int(mask.sum()),
            artifact_recall=float(mask[gt_artifact].mean()) if gt_artifact.any() else None,
            real_fp_rate=float(mask[gt_real].mean()) if gt_real.any() else None,
            n_unlabeled_flagged=int((mask & gt_unlabeled).sum()),
        )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "validate_bout28.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# Full mining pass
# ---------------------------------------------------------------------------

def mine(out_dir, sessions=("Session0", "Session1")):
    os.makedirs(out_dir, exist_ok=True)
    records = []          # per-frame flagged records (one row per flagged frame)
    bout_summary = []     # per (session,rec,bout,fly) counts
    bone_medians_report = []
    n_bf_done, n_bf_failed = 0, 0

    for session, rec, calib in session_recordings():
        if session not in sessions:
            continue
        try:
            get_cam_mats(calib)
        except Exception as e:
            print(f"SKIP {session}/{rec}: calib load failed: {e}")
            continue

        # Pass 1: load every bout-fly once (cached), get this recording's
        # OWN clean-frame median vein-to-base bone length per vein -- the
        # anatomical-plausibility reference (see BONE_LENGTH_MULT docstring
        # note). Pooling across all bouts/flies in the recording is robust
        # to a handful of badly-broken bout-flies as long as they are a
        # minority of the recording's frames.
        cache = []  # (b, fly, fdir, bpath, data)
        bone_vals = {kp: [] for kp in VEINS}
        for b, fly, fdir, bpath in bout_fly_dirs(session, rec):
            try:
                data = load_bout_fly(fdir, calib)
            except Exception as e:
                n_bf_failed += 1
                print(f"FAIL {session}/{rec}/{b}/{fly}: {e}")
                continue
            cache.append((b, fly, fdir, bpath, data))
            for kp in VEINS:
                bl = bone_length(data, kp)
                bone_vals[kp].append(bl[np.isfinite(bl)])
        bone_cutoff = {}
        for kp in VEINS:
            vals = np.concatenate(bone_vals[kp]) if bone_vals[kp] else np.array([np.nan])
            med = float(np.nanmedian(vals)) if vals.size else float("nan")
            bone_cutoff[kp] = med * BONE_LENGTH_MULT
        bone_medians_report.append(dict(session=session, recording=rec,
                                         bone_cutoff=bone_cutoff))

        # Pass 2: flag frames, gated on anatomical plausibility.
        for b, fly, fdir, bpath, data in cache:
            n_bf_done += 1
            bout_idx = int(b.split("_")[-1])
            fly_idx = int(fly[-1])
            male_fly = bout_male_fly(bpath)
            start_frame = bout_start_frame(session, rec, bout_idx)

            per_kp_A = {kp: signal_A_mirror(data, kp) for kp in VEINS}
            per_kp_B = {kp: signal_B_displacement(data, kp) for kp in VEINS}
            per_kp_bone = {kp: bone_length(data, kp) for kp in VEINS}

            frame_flag = np.zeros(data["T"], dtype=bool)
            frame_best_kp = np.full(data["T"], "", dtype=object)
            frame_best_cnt = np.zeros(data["T"], dtype=int)
            frame_implausible = np.zeros(data["T"], dtype=bool)
            for kp in VEINS:
                mirror_kp = MIRROR_OF[kp]
                plausible = ((per_kp_bone[kp] <= bone_cutoff[kp]) &
                             (per_kp_bone[mirror_kp] <= bone_cutoff[mirror_kp]))
                frame_implausible |= ~plausible
                cnt = per_kp_A[kp]["mirror_confused"].sum(axis=1)
                cnt_gated = np.where(plausible, cnt, 0)
                better = cnt_gated > frame_best_cnt
                frame_best_cnt = np.where(better, cnt_gated, frame_best_cnt)
                frame_best_kp = np.where(better, kp, frame_best_kp)
                frame_flag |= cnt_gated >= FRAME_CAM_THRESH

            n_flagged = int(frame_flag.sum())
            n_implausible_excluded = int((frame_implausible & ~frame_flag).sum())
            bout_summary.append(dict(
                session=session, recording=rec, bout=bout_idx, fly=fly_idx,
                male_fly=male_fly, n_frames=int(data["T"]), n_flagged=n_flagged,
                frac_flagged=n_flagged / data["T"] if data["T"] else 0.0,
                n_bone_implausible=int(frame_implausible.sum()),
            ))

            for t in np.where(frame_flag)[0]:
                kp = frame_best_kp[t]
                cams_confused = [data["cam_names"][c]
                                  for c in range(data["C"])
                                  if per_kp_A[kp]["mirror_confused"][t, c]]
                d_kp_disp = float(np.nanmax(per_kp_B[kp]["d_kp"][t])) if np.any(
                    np.isfinite(per_kp_B[kp]["d_kp"][t])) else None
                d_mirror_confused_cams = per_kp_A[kp]["d_mirror"][t][
                    per_kp_A[kp]["mirror_confused"][t]]
                records.append(dict(
                    session=session, recording=rec, bout=bout_idx, fly=fly_idx,
                    male_fly=male_fly, frame_offset=int(t),
                    abs_frame=int(start_frame + t) if start_frame is not None else None,
                    keypoint=kp, n_cams_confused=int(frame_best_cnt[t]),
                    cams_confused=cams_confused,
                    max_displacement_px=d_kp_disp,
                    bone_length_own=float(per_kp_bone[kp][t]),
                    bone_length_mirror=float(per_kp_bone[MIRROR_OF[kp]][t]),
                    d_mirror_min=float(np.min(d_mirror_confused_cams)),
                    d_mirror_max=float(np.max(d_mirror_confused_cams)),
                ))
            print(f"{session}/{rec}/{b}/{fly}: {n_flagged}/{data['T']} flagged frames "
                  f"({100*n_flagged/data['T'] if data['T'] else 0:.2f}%), "
                  f"{n_implausible_excluded} bone-implausible excluded")

    with open(os.path.join(out_dir, "flagged_frames.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(out_dir, "bout_summary.json"), "w") as f:
        json.dump(bout_summary, f, indent=2)
    with open(os.path.join(out_dir, "bone_length_medians.json"), "w") as f:
        json.dump(bone_medians_report, f, indent=2)

    totals = dict(
        n_bout_fly_processed=n_bf_done, n_bout_fly_failed=n_bf_failed,
        n_frames_total=sum(r["n_frames"] for r in bout_summary),
        n_frames_flagged_total=sum(r["n_flagged"] for r in bout_summary),
        n_bout_fly_with_any_flag=sum(1 for r in bout_summary if r["n_flagged"] > 0),
        n_bone_implausible_total=sum(r["n_bone_implausible"] for r in bout_summary),
    )
    with open(os.path.join(out_dir, "mine_totals.json"), "w") as f:
        json.dump(totals, f, indent=2)
    print(json.dumps(totals, indent=2))
    return records, bout_summary, totals


# ---------------------------------------------------------------------------
# Trainable hard-mined dataset (red_data_3d_v5-compatible, additive, separate
# root -- see report point 5). Only frames where the flagged vein's own
# camera-agreement count is a MINORITY (<MAJORITY_CONFUSED_THRESH of 7) are
# included: those are the ones where the pipeline's own kp3d (the DLT
# consensus of the OTHER, unconfused views) is still trustworthy, so the
# correct training label is derivable by GEOMETRY alone -- reproject that
# same kp3d into the confused camera -- with no hand labelling. Frames where
# a majority of cameras agree on the wrong structure (kp3d itself may follow
# them) are excluded, not guessed at; see the report's trainability section.
# ---------------------------------------------------------------------------

MAJORITY_CONFUSED_THRESH = 4     # >=4/7 cams confused -> kp3d itself untrusted
FRAMES_PER_BOUTFLY_CAP = 4        # evenly-spaced subsample per (rec,bout,fly)
CONF_VISIBLE_THRESH = 0.3         # project's own conf_thresh convention


def resolve_sex(male_fly, fly_idx):
    if male_fly is None:
        return "unknown"
    return "male" if male_fly == fly_idx else "female"


def _bbox_from_kp2d(pts_xy, img_w, img_h, margin=60.0):
    finite = np.isfinite(pts_xy).all(axis=-1)
    if not finite.any():
        return [0.0, 0.0, float(img_w), float(img_h)]
    xs, ys = pts_xy[finite, 0], pts_xy[finite, 1]
    x0, x1 = max(0.0, xs.min() - margin), min(img_w, xs.max() + margin)
    y0, y1 = max(0.0, ys.min() - margin), min(img_h, ys.max() + margin)
    return [float(x0), float(y0), float(max(1.0, x1 - x0)), float(max(1.0, y1 - y0))]


def build_dataset(mine_out_dir, out_root, frames_per_boutfly=FRAMES_PER_BOUTFLY_CAP,
                   dry_run=False):
    """Write a red_data_3d_v5-schema-compatible hard-mined set to `out_root`
    (a SEPARATE root -- the existing red_data_3d_v5 is never touched).

    Only V5Dataset's inputs are produced (manifest.json,
    annotations/instances{,_train,_val}.json, images/<rec>/<cam>/Frame_N.jpg)
    -- masks/ is intentionally omitted (V5Dataset degrades a missing mask to
    an all-zero channel rather than erroring; backfilling real SAM3 masks for
    this set is future work, noted in the report, not required for the
    schema to be consumable).
    """
    import cv2  # local import: only needed for this dataset-writing path

    fly50 = _load_fly50_order()
    # model-order index -> fly50-order index, i.e. model_to_fly50[i] is
    # where model-order slot i lands in the fly50-order array.
    model_to_fly50 = np.array([fly50.index(n) for n in KP_NAMES])

    flagged_path = os.path.join(mine_out_dir, "flagged_frames.json")
    bone_path = os.path.join(mine_out_dir, "bone_length_medians.json")
    flagged = json.load(open(flagged_path))
    bone_cutoffs = {(r["session"], r["recording"]): r["bone_cutoff"]
                    for r in json.load(open(bone_path))}

    minority = [r for r in flagged if r["n_cams_confused"] < MAJORITY_CONFUSED_THRESH]
    groups = defaultdict(list)
    for r in minority:
        groups[(r["session"], r["recording"], r["bout"], r["fly"])].append(r)

    os.makedirs(os.path.join(out_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "annotations"), exist_ok=True)

    coco_images, coco_annotations = [], []
    manifest_recordings = {}
    next_image_id, next_ann_id = 1, 1
    n_frames_considered, n_images_written, n_corrections_total = 0, 0, 0
    data_cache, video_cache = {}, {}

    def get_data(session, rec, bout, fly):
        key = (session, rec, bout, fly)
        if key not in data_cache:
            calib = os.path.join(VIDEO_ROOT, session, rec, "calibration")
            fdir = os.path.join(PROC_ROOT, session, rec, "pose", "bouts",
                                 f"bout_{bout:05d}", f"fly{fly}")
            data_cache[key] = load_bout_fly(fdir, calib)
        return data_cache[key]

    def get_frame(session, rec, cam, abs_frame):
        key = (session, rec, cam)
        if key not in video_cache:
            video_cache[key] = cv2.VideoCapture(
                os.path.join(VIDEO_ROOT, session, rec, f"{cam}.mp4"))
        cap = video_cache[key]
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
        ok, img = cap.read()
        return img if ok else None

    for (session, rec, bout, fly), recs in sorted(groups.items()):
        recs_sorted = sorted(recs, key=lambda r: r["frame_offset"])
        step = max(1, len(recs_sorted) // frames_per_boutfly)
        picked = recs_sorted[::step][:frames_per_boutfly]
        if not picked:
            continue
        data = get_data(session, rec, bout, fly)
        male_fly = picked[0]["male_fly"]
        sex = resolve_sex(male_fly, fly)
        rkey = f"{session}_{rec}"
        cutoff = bone_cutoffs.get((session, rec))
        manifest_recordings.setdefault(rkey, dict(
            source_session=session, source_recording=rec, sex_note=(
                "per-fly; see per-annotation 'sex'"),
        ))

        for row in picked:
            t = row["frame_offset"]
            abs_frame = row["abs_frame"]
            n_frames_considered += 1

            # Recompute EVERY vein's confusion at this exact frame (not just
            # the single "best" vein the mining summary kept) so a frame
            # with more than one confused vein gets ALL of them corrected.
            cams_needing = defaultdict(dict)  # cam_idx -> {kp_name: corrected_xy}
            for kp in VEINS:
                s = signal_A_mirror(data, kp)
                bl_own = bone_length(data, kp)[t]
                bl_mir = bone_length(data, MIRROR_OF[kp])[t]
                if cutoff is None or not (bl_own <= cutoff[kp] and
                                           bl_mir <= cutoff[MIRROR_OF[kp]]):
                    continue
                confused_cams = np.where(s["mirror_confused"][t])[0]
                if not (FRAME_CAM_THRESH <= len(confused_cams) < MAJORITY_CONFUSED_THRESH):
                    continue
                own_xy_all = data["reproj"][t, :, KP_IDX[kp], :]
                for c in confused_cams:
                    cams_needing[int(c)][kp] = own_xy_all[c]

            if not cams_needing:
                continue

            for cam_idx, corrections in cams_needing.items():
                cam = data["cam_names"][cam_idx]
                if dry_run:
                    h, w = 448, 1936  # known recording resolution; skip decode
                else:
                    img = get_frame(session, rec, cam, abs_frame)
                    if img is None:
                        print(f"WARN: could not read frame {abs_frame} from "
                              f"{session}/{rec}/{cam}.mp4, skipping")
                        continue
                    h, w = img.shape[:2]
                fname = f"Frame_{abs_frame}.jpg"
                rel_path = f"{rkey}/{cam}/{fname}"
                dst = os.path.join(out_root, "images", rkey, cam, fname)
                if not dry_run:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if not os.path.exists(dst):
                        cv2.imwrite(dst, img)

                kp_xy = data["kp2d"][t, cam_idx].copy()          # (50,2) raw
                conf_t = data["conf"][t, cam_idx]                # (50,)
                vis = np.where(np.isfinite(kp_xy).all(axis=-1) &
                               (conf_t >= CONF_VISIBLE_THRESH), 2, 0)
                corrected_names = []
                for kp_name, xy in corrections.items():
                    k = KP_IDX[kp_name]
                    kp_xy[k] = xy
                    vis[k] = 2 if np.isfinite(xy).all() else 0
                    corrected_names.append(kp_name)
                kp_xy = np.nan_to_num(kp_xy, nan=0.0)
                # Reorder model-order (kp2d.npz/kp3d.npz, Scutellum-first)
                # -> fly50 order (head-first) -- see _load_fly50_order.
                kp_xy_50 = np.zeros_like(kp_xy)
                vis_50 = np.zeros_like(vis)
                kp_xy_50[model_to_fly50] = kp_xy
                vis_50[model_to_fly50] = vis
                keypoints_flat = np.concatenate(
                    [kp_xy_50, vis_50[:, None].astype(np.float64)], axis=1).reshape(-1).tolist()

                bbox = _bbox_from_kp2d(data["kp2d"][t, cam_idx], w, h)

                coco_images.append(dict(id=next_image_id, file_name=rel_path,
                                         width=int(w), height=int(h)))
                coco_annotations.append(dict(
                    id=next_ann_id, image_id=next_image_id, category_id=1,
                    iscrowd=0, bbox=bbox, num_keypoints=int((vis > 0).sum()),
                    keypoints=keypoints_flat, src_ann_id=next_ann_id,
                    fly_id=fly, sex=sex, behavior="courtship",
                    category="hardmine_wingmirror",
                    corrected_keypoints=corrected_names,
                    source=dict(session=session, recording=rec, bout=bout,
                                fly=fly, frame_offset=t, abs_frame=abs_frame,
                                camera=cam),
                ))
                next_image_id += 1
                next_ann_id += 1
                n_images_written += 1
                n_corrections_total += len(corrected_names)

    # Split by (session, recording): put the smallest ~15% of recordings
    # (by annotation count) into val, the rest train -- never splitting one
    # recording's frames across both. Simple, since this additive set has
    # only a handful of distinct recordings (unlike red_data_3d_v5's
    # category-stratified split, which does not apply to a single-category
    # additive set).
    ann_count_by_rkey = defaultdict(int)
    for a in coco_annotations:
        rkey = a["source"]["session"] + "_" + a["source"]["recording"]
        ann_count_by_rkey[rkey] += 1
    ordered = sorted(ann_count_by_rkey.items(), key=lambda kv: kv[1])
    total_ann = sum(c for _, c in ordered)
    val_rkeys, val_running = set(), 0
    for rkey, c in ordered:
        if val_running / max(total_ann, 1) >= 0.15:
            break
        val_rkeys.add(rkey)
        val_running += c

    def rkey_of(a):
        return a["source"]["session"] + "_" + a["source"]["recording"]

    images_by_id = {im["id"]: im for im in coco_images}
    train_ann = [a for a in coco_annotations if rkey_of(a) not in val_rkeys]
    val_ann = [a for a in coco_annotations if rkey_of(a) in val_rkeys]
    train_img_ids = {a["image_id"] for a in train_ann}
    val_img_ids = {a["image_id"] for a in val_ann}

    categories = [dict(id=1, name="fly", keypoints=fly50, skeleton=[])]

    def write_coco(path, images, anns):
        json.dump(dict(images=images, annotations=anns, categories=categories),
                  open(path, "w"), indent=2)

    if not dry_run:
        write_coco(os.path.join(out_root, "annotations", "instances.json"),
                    coco_images, coco_annotations)
        write_coco(os.path.join(out_root, "annotations", "instances_train.json"),
                    [images_by_id[i] for i in train_img_ids], train_ann)
        write_coco(os.path.join(out_root, "annotations", "instances_val.json"),
                    [images_by_id[i] for i in val_img_ids], val_ann)
        # Plain list, matching red_data_3d_v5's own
        # annotations/keypoint_names.json format exactly (fly50 order).
        json.dump(fly50, open(os.path.join(
            out_root, "annotations", "keypoint_names.json"), "w"), indent=2)
        json.dump(dict(version="red_data_hardmine_wingmirror",
                        recordings=manifest_recordings),
                   open(os.path.join(out_root, "manifest.json"), "w"), indent=2)

    report = dict(
        source_failure="WingL/R_V12/V13 left-right mirror confusion "
                        "(.superpowers/sdd/2026-08-29-coarse-to-fine-3d/wing-hardmine.md)",
        detection_rule="signal_A_mirror: d_own>30px & d_mirror<=20px & "
                        ">=2/7 cams agree, bone-length-plausible, "
                        "n_cams_confused<4 (minority -> kp3d trustworthy)",
        n_bout_fly_groups_with_minority_flags=len(groups),
        frames_per_boutfly_cap=frames_per_boutfly,
        n_frames_considered=n_frames_considered,
        n_images_written=n_images_written,
        n_annotations=len(coco_annotations),
        n_corrections_total=n_corrections_total,
        n_train_images=len(train_img_ids), n_val_images=len(val_img_ids),
        val_recordings=sorted(val_rkeys),
        masks="OMITTED -- V5Dataset degrades a missing mask to an all-zero "
              "channel; not required for the schema to load.",
    )
    if not dry_run:
        json.dump(report, open(os.path.join(out_root, "build_report.json"), "w"),
                   indent=2)
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "figures", "2026-08-30-wing-hardmine"))
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--sessions", nargs="+", default=["Session0", "Session1"])
    ap.add_argument("--build-dataset", action="store_true",
                     help="After mining, write the red_data_3d_v5-compatible "
                          "hard-mined training set to --dataset-out.")
    ap.add_argument("--dataset-out", default=(
        "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_hardmine_wingmirror"))
    ap.add_argument("--dataset-dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    validate_bout28(args.out_dir)
    if not args.validate_only:
        mine(args.out_dir, sessions=tuple(args.sessions))
    if args.build_dataset:
        build_dataset(args.out_dir, args.dataset_out, dry_run=args.dataset_dry_run)


if __name__ == "__main__":
    main()
