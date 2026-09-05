"""Label QC for a dataset root: keypoint completeness, and whether a recording's
calibration is the one its 3D labels were triangulated with.

WHY. On 2026-09-03 two defects were found by hand in the 2026-09-02 raw export
that no existing guard could see:

  * All six `2026_04_02_*` export dirs shipped ONE byte-identical
    `calibration/`. For `15_25_51` and `17_28_34` the exported 3D reprojects
    4-15 px off the exported 2D through it, and 0.00 px through the
    calibration `general_model` had for those recordings. The builder grouped
    calibrations faithfully and the 3D trainer (`v5_3d.py`) triangulates the
    2D labels with the assigned group -- so it trained on wrong 3D.
  * Five frames (one in `12_11_50_male`, two each in `17_28_34_{female,male}`)
    carry 3D that is exactly 10x what their own 2D re-triangulates to.

The raw export is the only place the 3D labels exist (the converter emits 2D
COCO only), so the reprojection check reads the raw dirs directly. It is
run by `scripts/data_prep/red3d2jarvis.py` before shipping a calibration,
and by `build_generalmodel_split.py --raw` against every calibration group.

RAW FORMAT (see docs/benchmark/2026-09-02-courtship-labels/notes.md):
`Cam*.csv` rows are `frame, idx,u,v, idx,u,v, ...`; `v` is BOTTOM-origin
(pixel y = image_height - v); a missing keypoint is `u = v = 1e+07`.
`keypoints3d.csv` is `frame, idx,x,y,z, ...` in mm with the same sentinel.
`calibration/<Cam>_dlt.csv` = 11 DLT coefficients (+1 -> 3x4, row-major)
projecting mm straight into TOP-origin pixels; the JARVIS yaml carries the
same matrix with rows 0-1 cols 0-2 scaled by 0.1 and `scale: 10`, i.e. it
projects `10 * X_mm`.
"""
from __future__ import annotations

import collections
import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SENTINEL = 1e6           # |value| at or above this is "not labelled" (1e+07)
CONSISTENT_PX = 0.5      # median reprojection at or below this = same calibration
FRAME_BAD_PX = 1.0       # a frame whose median exceeds this is 3D-inconsistent
_REC_RE = re.compile(r"^(\d{4}(?:_\d{2}){5})")


def recording_id_of(export_dir_name: str) -> str | None:
    """`2026_04_02_15_25_51_female` -> `2026_04_02_15_25_51`; None if no stamp."""
    m = _REC_RE.match(os.path.basename(export_dir_name.rstrip("/")))
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# raw export
# --------------------------------------------------------------------------
@dataclass
class RawLabels:
    rec_dir: str
    cams: list[str]
    frames: list[int]
    kp2d: dict[str, dict[int, np.ndarray]]      # cam -> frame -> (K,2) u, v_raw
    kp3d: dict[int, np.ndarray]                  # frame -> (K,3) mm
    image_hw: dict[str, tuple[int, int]]         # cam -> (H, W) from the yaml
    n_kp: int


def _read_rows(path: Path, ndim: int) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    with open(path) as fh:
        fh.readline()                             # skeleton path line
        for line in fh:
            v = line.strip().split(",")
            if not v or not v[0].strip():
                continue                          # blank / empty-frame rows
            frame = int(v[0])
            rest, step = v[1:], 1 + ndim
            if len(rest) % step:
                raise ValueError(f"{path}: frame {frame} has {len(rest)} values, "
                                 f"not a multiple of {step}")
            n = len(rest) // step
            arr = np.full((n, ndim), np.nan)
            for k in range(n):
                blk = rest[k * step:(k + 1) * step]
                arr[int(blk[0])] = [float(x) for x in blk[1:]]
            arr[(np.abs(arr) >= SENTINEL).any(axis=1)] = np.nan
            out[frame] = arr
    return out


def _yaml_dims(path: Path) -> tuple[int, int]:
    h = w = None
    for line in path.read_text().splitlines():
        if line.startswith("image_height:"):
            h = int(line.split(":")[1])
        elif line.startswith("image_width:"):
            w = int(line.split(":")[1])
    if h is None or w is None:
        raise ValueError(f"{path}: no image_width/image_height")
    return h, w


def read_raw_labels(rec_dir: str | Path) -> RawLabels:
    rec_dir = Path(rec_dir)
    cams = sorted(p.stem for p in rec_dir.glob("Cam*.csv"))
    if not cams:
        raise ValueError(f"{rec_dir}: no Cam*.csv")
    kp2d = {c: _read_rows(rec_dir / f"{c}.csv", 2) for c in cams}
    kp3d = _read_rows(rec_dir / "keypoints3d.csv", 3)
    frames = sorted(set(kp3d) | {f for d in kp2d.values() for f in d})
    n_kp = next(iter(kp3d.values())).shape[0] if kp3d else \
        next(iter(next(iter(kp2d.values())).values())).shape[0]
    hw = {c: _yaml_dims(rec_dir / "calibration" / f"{c}.yaml") for c in cams}
    return RawLabels(str(rec_dir), cams, frames, kp2d, kp3d, hw, n_kp)


# --------------------------------------------------------------------------
# projections
# --------------------------------------------------------------------------
@dataclass
class Projection:
    P: np.ndarray            # 3x4
    scale: float = 1.0       # the matrix projects `scale * X_mm`

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """(N,3) mm -> (N,2) TOP-origin pixels."""
        Xh = np.c_[X * self.scale, np.ones(len(X))]
        p = (self.P @ Xh.T).T
        return p[:, :2] / p[:, 2:3]


def dlt_projection(calib_dir: str | Path, cams: list[str]) -> dict[str, Projection]:
    out = {}
    for c in cams:
        coefs = np.loadtxt(Path(calib_dir) / f"{c}_dlt.csv", delimiter=",").ravel()
        if coefs.shape != (11,):
            raise ValueError(f"{calib_dir}/{c}_dlt.csv: expected 11 coefficients")
        out[c] = Projection(np.append(coefs, 1.0).reshape(3, 4), 1.0)
    return out


def yaml_projection(calib_dir: str | Path, cams: list[str]) -> dict[str, Projection]:
    """Parse the JARVIS `Cam*.yaml` without cv2 (text only)."""
    out = {}
    for c in cams:
        text = (Path(calib_dir) / f"{c}.yaml").read_text()
        m = re.search(r"data:\s*\[(.*?)\]", text, re.S)
        if m is None:
            raise ValueError(f"{calib_dir}/{c}.yaml: no projectionMatrix data")
        vals = [float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
        if len(vals) != 12:
            raise ValueError(f"{calib_dir}/{c}.yaml: expected 12 values, got {len(vals)}")
        s = re.search(r"^scale:\s*([0-9.eE+-]+)", text, re.M)
        out[c] = Projection(np.array(vals).reshape(3, 4), float(s.group(1)) if s else 1.0)
    return out


# --------------------------------------------------------------------------
# 2D-vs-3D reprojection
# --------------------------------------------------------------------------
def _triangulate(Ps: list[np.ndarray], pts: list[np.ndarray]) -> np.ndarray:
    A = []
    for P, (x, y) in zip(Ps, pts):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    X = np.linalg.svd(np.asarray(A))[2][-1]
    return X[:3] / X[3]


def reprojection_report(raw: RawLabels, proj: dict[str, Projection]) -> dict:
    """Labelled 2D vs the exported 3D projected through `proj`, in px.

    Per frame the median over all cameras and labelled keypoints; overall
    numbers are over every labelled point. Frames whose median exceeds
    FRAME_BAD_PX are listed with `scale_ratio` = median |X_exported| /
    |X re-triangulated from the frame's own 2D| (10.0 = the export's known
    units slip), so the caller can tell a wrong calibration (every frame off
    by a few px) from a few broken frames (most frames at 0 px)."""
    per_cam: dict[str, list] = {c: [] for c in raw.cams}
    per_frame: dict[int, float] = {}
    scale_ratio: dict[int, float] = {}
    all_err: list[float] = []
    for fr in raw.frames:
        X = raw.kp3d.get(fr)
        if X is None:
            continue
        errs_here: list[float] = []
        for c in raw.cams:
            uv = raw.kp2d[c].get(fr)
            if uv is None:
                continue
            H = raw.image_hw[c][0]
            ok = np.isfinite(uv).all(axis=1) & np.isfinite(X).all(axis=1)
            if not ok.any():
                continue
            p = proj[c](X[ok])
            e = np.hypot(p[:, 0] - uv[ok, 0], p[:, 1] - (H - uv[ok, 1]))
            per_cam[c].extend(e.tolist())
            errs_here.extend(e.tolist())
        if not errs_here:
            continue
        per_frame[fr] = float(np.median(errs_here))
        all_err.extend(errs_here)
        if per_frame[fr] > FRAME_BAD_PX:
            # re-triangulate this frame from its own 2D to name the defect
            ratios = []
            for k in range(raw.n_kp):
                use = [c for c in raw.cams if fr in raw.kp2d[c]
                       and np.isfinite(raw.kp2d[c][fr][k]).all()]
                if len(use) < 2 or not np.isfinite(X[k]).all():
                    continue
                Xt = _triangulate(
                    [proj[c].P for c in use],
                    [np.array([raw.kp2d[c][fr][k, 0],
                               raw.image_hw[c][0] - raw.kp2d[c][fr][k, 1]])
                     for c in use]) / proj[use[0]].scale
                n = np.linalg.norm(Xt)
                if n > 0:
                    ratios.append(float(np.linalg.norm(X[k]) / n))
            scale_ratio[fr] = float(np.median(ratios)) if ratios else float("nan")
    e = np.asarray(all_err)
    bad = [{"frame": fr, "median_px": round(px, 3),
            "scale_ratio": round(scale_ratio.get(fr, float("nan")), 3)}
           for fr, px in sorted(per_frame.items()) if px > FRAME_BAD_PX]
    return {
        "median_px": float(np.median(e)) if e.size else float("nan"),
        "p95_px": float(np.percentile(e, 95)) if e.size else float("nan"),
        "median_of_frame_medians_px": (float(np.median(list(per_frame.values())))
                                       if per_frame else float("nan")),
        "n_points": int(e.size),
        "n_frames": len(per_frame),
        "per_camera_median_px": {c: (float(np.median(v)) if v else float("nan"))
                                 for c, v in per_cam.items()},
        "inconsistent_frames": bad,
    }


def calibration_consistent(report: dict) -> bool:
    """True when the labels were (to rounding) triangulated with this
    calibration: median over frames at or below CONSISTENT_PX. A few
    broken frames do not flip this; a wrong calibration (every frame off)
    does."""
    m = report.get("median_of_frame_medians_px", float("nan"))
    return bool(np.isfinite(m) and m <= CONSISTENT_PX)


def best_calibration_group(raw: RawLabels, group_dirs: dict[str, str]
                           ) -> tuple[str, dict[str, dict]]:
    """Score every candidate `Cam*.yaml` dir against the raw labels; return
    the best-fitting label and the per-group reports."""
    per = {}
    for label, d in group_dirs.items():
        per[label] = reprojection_report(raw, yaml_projection(d, raw.cams))
    best = min(per, key=lambda g: (per[g]["median_of_frame_medians_px"]
                                   if np.isfinite(per[g]["median_of_frame_medians_px"])
                                   else float("inf")))
    return best, per


# --------------------------------------------------------------------------
# completeness of the shipped COCO
# --------------------------------------------------------------------------
def completeness(images: list[dict], annotations: list[dict],
                 kp_names: list[str]) -> dict:
    """Missing-keypoint accounting from merged COCO (`images` carry
    `recording`; `annotations` carry `subset`). A slot with v == 0 is
    missing, whatever the reason (unlabelled, out of view, structurally
    absent after schema padding)."""
    K = len(kp_names)
    img = {im["id"]: im for im in images}

    def _cam(im):
        return im["file_name"].split("/")[1]

    rec_rows = collections.defaultdict(lambda: {
        "images": 0, "images_without_annotation": 0, "annotations": 0,
        "keypoint_slots": 0, "visible_keypoints": 0, "missing_keypoints": 0,
        "fully_labelled_annotations": 0, "num_keypoints_field_mismatches": 0})
    cam_rows = collections.defaultdict(lambda: collections.defaultdict(lambda: {
        "annotations": 0, "missing_keypoints": 0}))
    sub_rows = collections.defaultdict(lambda: {
        "annotations": 0, "keypoint_slots": 0, "missing_keypoints": 0})
    per_kp = np.zeros(K, dtype=int)
    annotated = set()
    for a in annotations:
        im = img[a["image_id"]]
        rec, cam = im["recording"], _cam(im)
        kp = np.asarray(a["keypoints"], dtype=float).reshape(-1, 3)
        vis = kp[:, 2] > 0
        n_vis, n_miss = int(vis.sum()), int((~vis).sum())
        per_kp += ~vis
        annotated.add(a["image_id"])
        r = rec_rows[rec]
        r["annotations"] += 1
        r["keypoint_slots"] += len(kp)
        r["visible_keypoints"] += n_vis
        r["missing_keypoints"] += n_miss
        r["fully_labelled_annotations"] += int(n_miss == 0)
        r["num_keypoints_field_mismatches"] += int(a.get("num_keypoints", n_vis) != n_vis)
        c = cam_rows[rec][cam]
        c["annotations"] += 1
        c["missing_keypoints"] += n_miss
        s = sub_rows[a.get("subset", "?")]
        s["annotations"] += 1
        s["keypoint_slots"] += len(kp)
        s["missing_keypoints"] += n_miss
    for im in images:
        r = rec_rows[im["recording"]]
        r["images"] += 1
        r["images_without_annotation"] += int(im["id"] not in annotated)
    for r in rec_rows.values():
        r["missing_fraction"] = round(r["missing_keypoints"] / max(r["keypoint_slots"], 1), 4)
    tot = {k: sum(r[k] for r in rec_rows.values())
           for k in ("images", "images_without_annotation", "annotations",
                     "keypoint_slots", "visible_keypoints", "missing_keypoints",
                     "fully_labelled_annotations", "num_keypoints_field_mismatches")}
    tot["missing_fraction"] = round(tot["missing_keypoints"] / max(tot["keypoint_slots"], 1), 4)
    return {
        "totals": tot,
        "per_recording": {k: rec_rows[k] for k in sorted(rec_rows)},
        "per_recording_camera": {r: {c: dict(v) for c, v in sorted(cam_rows[r].items())}
                                 for r in sorted(cam_rows)},
        "per_subset": {k: sub_rows[k] for k in sorted(sub_rows)},
        "per_keypoint_missing": {kp_names[i]: int(n) for i, n in enumerate(per_kp) if n},
    }


def raw_export_dirs(roots: list[str]) -> list[str]:
    """Every `<root>/<dir>` that looks like a raw recording export."""
    out = []
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "keypoints3d.csv")):
                out.append(d)
    return out
