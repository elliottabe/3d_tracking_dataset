#!/usr/bin/env python3
"""Convert ONE raw red3d label recording into a general_model-style subset.

Corrected copy of `/gscratch/portia/eabe/data/Johnson_lab/red_data/red3d2jarvis.py`
(the user's script, which lives in the data directory and is NOT modified).
Every difference from that original is listed under DEFECTS below, with the
evidence that motivated it.

    python scripts/data_prep/red3d2jarvis.py \
        -i  <raw>/coutship_label_2026_09_02/2025_10_20_13_20_04_male \
        -o  <staging>/courtship_20_04_male \
        --subset-name courtship_20_04_male \
        --recording-id 2025_10_20_13_20_04 \
        --sex male --sex-source dirname \
        --link-images-from <gm>/courtship_V2 <gm>/courtship_V3 <gm>/courtship_V4

DEFECTS IN THE ORIGINAL, and what this does instead
---------------------------------------------------
1. THE FOLDER REGEX MATCHES NOTHING.  The original scans `--label_folder` for
   children matching `^\\d{4}_\\d{2}_\\d{2}_\\d{2}_\\d{2}_\\d{2}$` and then takes
   `matching_folders[-1]`.  Every recording in the 2026-09-02 export carries a
   `_female` / `_male` / `_climbing` suffix, so the list is EMPTY and the script
   dies with IndexError before doing anything.  Worse, had it matched, it would
   have converted only ONE folder per invocation -- the alphabetically last.
   Here `-i` names the recording directory itself; one run, one recording,
   nothing implicit.

2. CALIBRATION PATH IS WRONG.  The original looks in
   `<parent-of-label_folder>/calibration`.  The real location is
   `<recording>/calibration`.  Read from the recording.

3. VIDEO PATH DOES NOT RESOLVE.  The original computes
   `video_path = Path(label_folder).parent` and opens `{video_path}/{cam}.mp4`
   purely to read frame width/height; those files do not exist, so
   `vreader.isOpened()` is False, `image_width`/`image_height` stay EMPTY, and
   the later `image_width[cam]` raises KeyError.  Note the label directory
   carries a sex suffix that the video directory does not
   (`2026_04_02_12_11_50_female` -> `.../Session1/2026_04_02_12_11_50`), so the
   mapping must be deliberate.  Here `--video-dir` is explicit, and image
   dimensions come from the calibration yaml when no video is given.

4. THE SKELETON JSON IS MISSING.  Every CSV's first line cites
   `/home/tuthill/juan/skeletons/fly50.json`, which does not exist on this
   machine, so `load_skeleton_json_format_for_jarvis` cannot run.  The 50 names
   come instead from this repo's `data/fly50.json`, and the CSV's declared
   skeleton basename is ASSERTED to be `fly50.json` so a different skeleton
   cannot be silently converted with fly50 names.  See KEYPOINT ORDER below --
   the order is not assumed, it is proven.

5. IT PERFORMS ITS OWN 90/10 SPLIT WITH seed=42.  Bypassed entirely: this
   emits ONE `annotations/instances_train.json` holding every annotation, and
   `build_generalmodel_split.py` owns train/val downstream by image CONTENT.
   (The original's split was also broken independently of the policy question:
   it seeds `np.random.default_rng(seed=42)` but then shuffles with the legacy
   global `np.random.shuffle`, so `rng` is dead code and the split is NOT
   reproducible across runs.)

6. `from keypoints import *` / `from utils import *` DO NOT RESOLVE.  Neither
   `keypoints.py` nor `utils.py` exists anywhere readable on this machine
   (searched the whole of /gscratch/portia/eabe and /mmfs1/home/eabe), so the
   original cannot be imported at all, let alone run.  Every helper it needed
   -- `csv_reader_red3d`, `get_all_cams_in_labeled_folder`, `get_skeleton_name`,
   `process_one_session`, `generate_annotation_file` -- is reimplemented here
   from the observed data format and validated by exact reproduction (below).

THE VERTICAL FLIP, which the original's missing `utils.py` hid
--------------------------------------------------------------
Raw CSV `v` is measured from the BOTTOM of the frame.  The shipped jpgs are
the native, top-left-origin video frames (verified: MAE 0.36-0.42 grey levels
between a decoded frame and its jpg, vs 35-104 for the flipped comparison), so
the conversion is

    x = floor(u_raw)                     y = floor(image_height - v_raw)

and the bbox is the min/max of the UNROUNDED flipped coordinates over visible
keypoints, with no margin.  This is not inferred: applying exactly this rule to
the 2026-09-02 raw CSVs reproduces the existing `general_model` annotations for
all seven recordings that both sources hold -- 59,331 labelled keypoints,
100.000% exact on x, y AND visibility, and bbox to 4 decimals.

Independently, the labelled 3D reprojects through the recording's own raw DLT
onto (u_raw, H - v_raw) at a MEDIAN of 0.001-0.003 px.  Two unrelated lines of
evidence, same answer.

A coordinate at |value| >= 1e6 (the CSVs use 1e+07) is the not-labelled
sentinel and becomes COCO's (0, 0, v=0).

KEYPOINT ORDER -- ESTABLISHED, NOT ASSUMED
------------------------------------------
Three independent checks, because this repo has twice produced confident,
self-consistent, completely wrong results from a keypoint-order error:

  * EXACT REPRODUCTION.  Index k of the raw CSVs equals index k of the
    authoritative `general_model` annotations, whose `keypoint_names` are
    byte-identical to `red_data_3d_v8_gm_only/annotations/keypoint_names.json`
    and to this repo's `data/fly50.json`.  59,331/59,331 points.

  * CROSS-MODAL CONSISTENCY.  Index k of `keypoints3d.csv` projects onto index
    k of every camera's 2D CSV at a median 0.001 px through the recording's own
    DLT.  A permutation between the 3D and 2D files could not survive this.

  * RIGID INVARIANTS + ANATOMY, on 677 frames of 2025_10_20_13_20_04_male:
    left/right homologous leg segments agree to 0.03-2.0% across all 15 pairs;
    wing-vein lengths are near-constant (CV 3.3-5.3%); every leg chain runs
    monotonically proximal-to-distal with T1 < T2 < T3, and
    Antenna_Base->Abd_tip is 2.32 mm.  A scrambled order breaks the symmetry
    check immediately -- which is the check that this repo's 2026-08-31
    keypoint-order bug would have failed while every jitter, confidence and
    residual metric rated it GOOD.

Names are resolved BY NAME throughout; no keypoint or camera is addressed by a
bare integer index.

--scale_10x
-----------
The original's `--scale_10x` is a REQUIRED bool that multiplies
`projectionMatrix[0:2, 0:3]` by 0.1, i.e. it re-expresses the DLT for world
coordinates 10x LARGER than the ones the raw `_dlt.csv` was fitted in.  Row 2
of these DLTs is [0, 0, 0, 1] -- they are affine, so that scaling IS a
consistent change of world units rather than the broken half-rescale it would
be for a projective DLT.  TRUE is the value that matches this corpus: with it,
the emitted yaml is byte-identical to the calibration already shipped in
`general_model` for the same recording, `scale: 10` and all.  Default True;
`--no-scale-10x` exists but should not be used with this corpus.

THE SHIPPED CALIBRATION IS VERIFIED AGAINST THE LABELS (added 2026-09-03)
-------------------------------------------------------------------------
The raw export's `calibration/` is NOT trusted to be the calibration the
labels were triangulated with. All six `2026_04_02_*` export dirs shipped one
byte-identical calibration (the 12_11_50 one); for 15_25_51 and 17_28_34 the
exported 3D reprojects 4-15 px off the exported 2D through it and 0.00 px
through the calibration `general_model` held for those recordings. The 3D
trainer triangulates the 2D labels with whatever calibration the root ships,
so this converter now refuses to write a calibration unless the recording's
own 3D reprojects onto its 2D at <= 0.5 px median (`jarvis_jax.data.label_qc`).
`--calib-from DIR` ships DIR's `Cam*.yaml` instead of the raw DLT, subject to
the same proof. Frames whose 3D is inconsistent with their own 2D (the
export carries five whose 3D is exactly 10x) are reported in the stats and
KEPT: their 2D is fine and 3D is never shipped from here.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FLY50_PATH = REPO_ROOT / "data" / "fly50.json"
_JJ = REPO_ROOT / "third_party" / "jarvis_jax"
if str(_JJ) not in sys.path:
    sys.path.insert(0, str(_JJ))
from jarvis_jax.data import label_qc  # noqa: E402

# |coord| at or above this is the raw format's not-labelled sentinel (1e+07).
MISSING = 1e6
EXPECTED_SKELETON = "fly50.json"


# --------------------------------------------------------------------------
# raw format
# --------------------------------------------------------------------------
def read_raw_csv(path: Path, ndim: int) -> tuple[str, dict[int, np.ndarray]]:
    """Line 1 is a skeleton path; each later row is
    `frame_id, kp_idx, c0..c{ndim-1}, kp_idx, ...`.

    Values are placed AT their stated kp_idx rather than in encounter order, so
    a row that omits or reorders indices cannot shift the keypoint axis."""
    out: dict[int, np.ndarray] = {}
    with open(path) as fh:
        # Line 1 is a CSV row whose only meaningful field is the path. Some
        # exports pad it out to the full column count, so
        # `2026_04_02_15_25_51_male` declares
        # "/home/user/red_data/skeleton/fly50.json,,,,,,,..." -- taking the
        # whole stripped line made basename() return "fly50.json,,,,,..." and
        # the skeleton assertion below rejected a file that is in fact fly50.
        # Take field 0. (Checked across all 9 export recordings: every one
        # declares basename fly50.json, under two different directories.)
        skeleton = fh.readline().strip().split(",")[0].strip()
        for line in fh:
            line = line.strip()
            if not line:
                continue
            v = line.split(",")
            frame = int(v[0])
            rest, step = v[1:], 1 + ndim
            if len(rest) % step:
                raise ValueError(f"{path}: frame {frame} has {len(rest)} values, "
                                 f"not a multiple of {step}")
            n = len(rest) // step
            arr = np.full((n, ndim), np.nan)
            for k in range(n):
                blk = rest[k * step:(k + 1) * step]
                idx = int(blk[0])
                if not 0 <= idx < n:
                    raise ValueError(f"{path}: frame {frame} kp index {idx} "
                                     f"out of range for {n} keypoints")
                arr[idx] = [float(x) for x in blk[1:]]
            if frame in out:
                raise ValueError(f"{path}: frame {frame} appears twice")
            out[frame] = arr
    return skeleton, out


def load_fly50() -> tuple[list[str], list[list[int]]]:
    d = json.loads(FLY50_PATH.read_text())
    return list(d["node_names"]), [list(e) for e in d["edges"]]


def jarvis_skeleton(names: list[str], edges: list[list[int]]) -> list[dict]:
    """Edge list in the general_model annotation schema."""
    return [{"keypointA": names[a], "keypointB": names[b],
             "length": 0.0, "name": f"Joint {i + 1}"}
            for i, (a, b) in enumerate(edges)]


def read_calib_yaml_dims(cal_dir: Path, cam: str) -> tuple[int, int]:
    """image_width / image_height straight out of the shipped yaml."""
    w = h = None
    for line in (cal_dir / f"{cam}.yaml").read_text().splitlines():
        if line.startswith("image_width:"):
            w = int(line.split(":")[1])
        elif line.startswith("image_height:"):
            h = int(line.split(":")[1])
    if w is None or h is None:
        raise ValueError(f"{cal_dir/f'{cam}.yaml'}: no image_width/image_height")
    return w, h


def write_calib_yaml(dst: Path, cam: str, dlt_csv: Path,
                     width: int, height: int, scale_10x: bool) -> None:
    """DLT coefficients -> the 3x4 projectionMatrix yaml JARVIS reads."""
    coefs = np.loadtxt(dlt_csv, delimiter=",")
    if coefs.shape != (11,):
        raise ValueError(f"{dlt_csv}: expected 11 DLT coefficients, got {coefs.shape}")
    P = np.append(coefs, 1.0).reshape(3, 4)
    if scale_10x:
        P[0:2, 0:3] *= 0.1
    import cv2
    s = cv2.FileStorage(str(dst / f"{cam}.yaml"), cv2.FileStorage_WRITE)
    s.write("image_width", width)
    s.write("image_height", height)
    s.write("projectionMatrix", P)
    s.write("scale", 10 if scale_10x else 1)
    s.release()


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------
def convert(rec_dir: Path, out_dir: Path, subset: str, recording_id: str,
            sex: str, sex_source: str, citation: str | None,
            scale_10x: bool, link_from: list[Path], video_dir: Path | None,
            calib_from: Path | None = None) -> dict:
    names, edges = load_fly50()
    n_kp = len(names)
    cams = sorted(p.stem for p in rec_dir.glob("Cam*.csv"))
    if not cams:
        raise SystemExit(f"{rec_dir}: no Cam*.csv")
    cal_dir = rec_dir / "calibration"          # DEFECT 2
    if not cal_dir.is_dir():
        raise SystemExit(f"{cal_dir} missing")

    skel3, kp3d = read_raw_csv(rec_dir / "keypoints3d.csv", 3)
    per_cam: dict[str, dict[int, np.ndarray]] = {}
    for cam in cams:
        skel2, per_cam[cam] = read_raw_csv(rec_dir / f"{cam}.csv", 2)
        # DEFECT 4: the cited skeleton file does not exist, so its NAME is the
        # only guard that fly50's 50 names describe this data.
        for s in (skel3, skel2):
            if os.path.basename(s) != EXPECTED_SKELETON:
                raise SystemExit(
                    f"{rec_dir}: CSV declares skeleton {s!r}, expected "
                    f"{EXPECTED_SKELETON!r}. Refusing to apply fly50 names to a "
                    f"different skeleton -- resolve the order before converting.")

    dims = {cam: read_calib_yaml_dims(cal_dir, cam) for cam in cams}
    if video_dir is not None:
        import cv2
        for cam in cams:
            cap = cv2.VideoCapture(str(video_dir / f"{cam}.mp4"))
            if not cap.isOpened():
                raise SystemExit(f"{video_dir/f'{cam}.mp4'} will not open")
            vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if (vw, vh) != dims[cam]:
                raise SystemExit(f"{cam}: video is {vw}x{vh} but calibration "
                                 f"says {dims[cam][0]}x{dims[cam][1]}")

    # ---- PROVE the calibration we are about to ship made these labels ------
    raw = label_qc.read_raw_labels(rec_dir)
    if calib_from is not None:
        for cam in cams:
            if not (calib_from / f"{cam}.yaml").exists():
                raise SystemExit(f"--calib-from {calib_from}: no {cam}.yaml")
            if read_calib_yaml_dims(calib_from, cam) != dims[cam]:
                raise SystemExit(f"--calib-from {calib_from}/{cam}.yaml: image "
                                 f"size differs from the raw calibration")
        proj = label_qc.yaml_projection(calib_from, cams)
        calib_source = str(calib_from)
    else:
        proj = label_qc.dlt_projection(cal_dir, cams)
        calib_source = str(cal_dir)
    rep = label_qc.reprojection_report(raw, proj)
    if not label_qc.calibration_consistent(rep):
        raise SystemExit(
            f"{rec_dir}: the calibration at {calib_source} does NOT reproduce "
            f"this recording's own 3D labels: median "
            f"{rep['median_of_frame_medians_px']:.2f} px over {rep['n_frames']} "
            f"frames (per camera: "
            + ", ".join(f"{c}={v:.1f}" for c, v in rep["per_camera_median_px"].items())
            + f"). Labels are triangulated to ~0.003 px by the calibration that "
            f"made them, so this is a different calibration. Pass --calib-from "
            f"<dir of Cam*.yaml> holding the right one (e.g. the same-named "
            f"general_model subset's calib_params/<rec>/).")
    if rep["inconsistent_frames"]:
        print(f"WARNING: {len(rep['inconsistent_frames'])} frames whose exported "
              f"3D does not match their own 2D (scale_ratio 10 = the export's "
              f"known 10x units slip); 2D is shipped, 3D never is:",
              file=sys.stderr)
        for row in rep["inconsistent_frames"]:
            print(f"   {row}", file=sys.stderr)

    frames = sorted(set(kp3d) | {f for d in per_cam.values() for f in d})
    images, annotations, framesets = [], [], {}
    oob: list[tuple] = []
    n_img = n_ann = 0
    stats = {"images": 0, "annotated": 0, "labelled_points": 0,
             "fully_labelled_images": 0, "frames": len(frames)}

    for frame in frames:
        ids_here = []
        for cam in cams:                       # cameras resolved BY NAME
            uv = per_cam[cam].get(frame)
            if uv is None:
                continue
            w, h = dims[cam]
            img_id = n_img
            n_img += 1
            images.append({
                "coco_url": "", "date_captured": "",
                "file_name": f"{recording_id}/{cam}/Frame_{frame}.jpg",
                "flickr_url": "", "height": h, "id": img_id, "width": w})
            ids_here.append(img_id)
            stats["images"] += 1

            vis = (np.abs(uv) < MISSING).all(axis=1) & ~np.isnan(uv).any(axis=1)
            if not vis.any():
                continue
            # THE FLIP: raw v is bottom-origin, the jpgs are top-left-origin.
            x_f = uv[:, 0]
            y_f = h - uv[:, 1]
            kps: list[int] = []
            for k in range(n_kp):
                if vis[k]:
                    # floor, then clamp into the frame. floor (not int()) so the
                    # rule is monotonic across zero; the clamp because a label
                    # may sit a hair outside the sensor -- measured over
                    # 2025_10_20_13_20_04_male, exactly ONE of 236,932 points
                    # does, by 0.004 px. Anything further out is reported below
                    # rather than silently pulled onto the edge.
                    xi = min(max(int(np.floor(x_f[k])), 0), w - 1)
                    yi = min(max(int(np.floor(y_f[k])), 0), h - 1)
                    if not (-1.0 <= x_f[k] <= w) or not (-1.0 <= y_f[k] <= h):
                        oob.append((cam, frame, names[k],
                                    round(float(x_f[k]), 3), round(float(y_f[k]), 3)))
                    kps += [xi, yi, 1]
                else:
                    kps += [0, 0, 0]
            xs, ys = x_f[vis], y_f[vis]
            annotations.append({
                "bbox": [float(xs.min()), float(ys.min()),
                         float(xs.max() - xs.min()), float(ys.max() - ys.min())],
                "category_id": 1, "id": n_ann, "image_id": img_id, "iscrowd": 0,
                "keypoints": kps, "num_keypoints": int(vis.sum()),
                "segmentation": []})
            n_ann += 1
            stats["annotated"] += 1
            stats["labelled_points"] += int(vis.sum())
            stats["fully_labelled_images"] += int(vis.all())
        if ids_here:
            framesets[f"{recording_id}/Frame_{frame}"] = {
                "datasetName": recording_id, "frames": ids_here}

    blob = {
        "keypoint_names": names,
        "skeleton": jarvis_skeleton(names, edges),
        "categories": [{"id": 0, "name": "Rat", "num_keypoints": n_kp,
                        "supercategory": "None"}],
        "annotations": annotations, "images": images,
        "calibrations": {recording_id: {
            cam: f"calib_params/{recording_id}/{cam}.yaml" for cam in cams}},
        "framesets": framesets,
    }

    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "annotations" / "instances_train.json", "w") as f:
        json.dump(blob, f)                      # DEFECT 5: no split here

    cal_out = out_dir / "calib_params" / recording_id
    cal_out.mkdir(parents=True, exist_ok=True)
    for cam in cams:
        if calib_from is not None:
            shutil.copy2(calib_from / f"{cam}.yaml", cal_out / f"{cam}.yaml")
        else:
            write_calib_yaml(cal_out, cam, cal_dir / f"{cam}_dlt.csv",
                             *dims[cam], scale_10x=scale_10x)
    stats["calibration_check"] = {
        "source": calib_source,
        "median_px": rep["median_px"], "p95_px": rep["p95_px"],
        "median_of_frame_medians_px": rep["median_of_frame_medians_px"],
        "n_points": rep["n_points"], "n_frames": rep["n_frames"],
        "per_camera_median_px": rep["per_camera_median_px"]}
    stats["frames_3d_inconsistent"] = rep["inconsistent_frames"]

    with open(out_dir / "sex.json", "w") as f:
        json.dump({"subset": subset, "sex": sex, "sex_source": sex_source,
                   "recordings": [recording_id],
                   "citation": citation,
                   "note": "Written by scripts/data_prep/red3d2jarvis.py from "
                           "the raw 2026-09-02 label export. Sex travels WITH "
                           "the subset; never re-derive it from the name."},
                  f, indent=2)

    # ---- images ----------------------------------------------------------
    linked = missing = 0
    if link_from:
        index: dict[tuple[str, int], Path] = {}
        for src_subset in link_from:
            # A general_model subset files its jpgs under train/ and val/; a raw
            # export directory holds `Cam*/Frame_*.jpg` at its own top level
            # (2025_10_12_15_06_46_male_climbing ships its frames that way, and
            # its video is a 921-frame excerpt that does not contain them). Both
            # layouts are accepted; nothing is decoded either way.
            bases = [src_subset / s for s in ("train", "val")
                     if (src_subset / s).is_dir()] or [src_subset]
            for base in bases:
                for jpg in base.rglob("Cam*/Frame_*.jpg"):
                    cam = jpg.parent.name
                    frame = int(jpg.stem.split("_")[1])
                    index.setdefault((cam, frame), jpg)
        # THE FLIP depends on image_height, which is read from the calibration
        # yaml. If the shipped jpgs are not that size the flip is wrong and the
        # keypoints land off the animal, so this is checked against the real
        # pixels rather than trusted. PIL reads the header only.
        from PIL import Image as _Image
        for (cam, _frame), jpg in sorted(index.items()):
            with _Image.open(jpg) as im_:
                if im_.size != dims[cam]:
                    raise SystemExit(
                        f"{jpg} is {im_.size[0]}x{im_.size[1]} but the "
                        f"calibration for {cam} says "
                        f"{dims[cam][0]}x{dims[cam][1]}. The vertical flip "
                        f"y = image_height - v would be wrong; refusing.")
        print(f"verified {len(index)} source jpgs match the calibration dimensions")
        for im in images:
            _, cam, fname = im["file_name"].split("/")
            frame = int(fname.split("_")[1].split(".")[0])
            src = index.get((cam, frame))
            dst = out_dir / "train" / im["file_name"]
            if src is None:
                missing += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.symlink(os.path.realpath(src), dst)
            linked += 1
    elif video_dir is not None:
        import cv2
        want: dict[str, list[int]] = {}
        for im in images:
            _, cam, fname = im["file_name"].split("/")
            want.setdefault(cam, []).append(int(fname.split("_")[1].split(".")[0]))
        for cam, wanted in want.items():
            wanted_set = set(wanted)
            cap = cv2.VideoCapture(str(video_dir / f"{cam}.mp4"))
            for frame in sorted(wanted_set):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, img = cap.read()
                if not ok:
                    missing += 1
                    continue
                dst = out_dir / "train" / recording_id / cam / f"Frame_{frame}.jpg"
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), img)
                linked += 1
            cap.release()
    stats["images_materialised"] = linked
    stats["images_missing"] = missing
    stats["out_of_frame_keypoints"] = oob
    if oob:
        print(f"WARNING: {len(oob)} labelled keypoints fall outside the frame "
              f"by more than 1 px and were clamped to the edge:", file=sys.stderr)
        for row in oob[:20]:
            print(f"   {row}", file=sys.stderr)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-i", "--recording", required=True,
                    help="ONE raw recording dir (DEFECT 1: no globbing, no "
                         "'most recent' guess)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--subset-name", required=True)
    ap.add_argument("--recording-id", required=True)
    ap.add_argument("--sex", required=True, choices=("female", "male"))
    ap.add_argument("--sex-source", required=True)
    ap.add_argument("--citation", default=None)
    ap.add_argument("--scale-10x", dest="scale_10x", action="store_true", default=True)
    ap.add_argument("--no-scale-10x", dest="scale_10x", action="store_false")
    ap.add_argument("--link-images-from", nargs="*", type=Path, default=[],
                    help="dirs holding the jpgs already, symlinked read-only "
                         "rather than re-decoded. Either a general_model subset "
                         "(jpgs under train/ and val/) or a raw export "
                         "recording dir (Cam*/Frame_*.jpg at top level).")
    ap.add_argument("--video-dir", type=Path, default=None,
                    help="DEFECT 3: the video dir has NO sex suffix while the "
                         "label dir does, so it is named explicitly")
    ap.add_argument("--calib-from", type=Path, default=None,
                    help="dir of Cam*.yaml to ship INSTEAD of the raw export's "
                         "calibration/ (e.g. general_model/<subset>/calib_params/"
                         "<rec>). Either way the shipped calibration must "
                         "reproduce the recording's own 3D labels.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = convert(Path(args.recording), out, args.subset_name,
                    args.recording_id, args.sex, args.sex_source, args.citation,
                    args.scale_10x, list(args.link_images_from), args.video_dir,
                    calib_from=args.calib_from)
    print(json.dumps(stats, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
