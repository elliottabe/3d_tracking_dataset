#!/usr/bin/env python3
"""Before/after check of the 2026-09-03 calibration fix for 15_25_51 / 17_28_34.

What the 3D trainer sees for a recording is its 2D labels triangulated with
the calibration group the root's manifest ASSIGNS it. This figure puts the
exported 3D labels through the group assigned by the OLD root (built from the
export's calibration) and by the NEW root (built from general_model's), and
compares with the exported 2D labels.

EXPECTATION if the fix is right: in the "after" panels the projected 3D
(green) sits exactly on the raw 2D labels (cyan) for every camera of 15_25_51
and 17_28_34, and the per-camera bars for "after" are at ~0 px; the "before"
projection (magenta) is displaced by 4-15 px in six of the seven cameras and
exact only on Cam2012631. 12_11_50 is the control: it must be ~0 px both
before and after, because its calibration did not change.
FAILURE MODE: green off the cyan labels anywhere = the new root still ships a
calibration those labels were not made with.

    python scripts/viz/calib_group_fix_check.py \
        --old /gscratch/.../red_data/red_data_3d_v12_export0902.pre_calibfix_0903 \
        --new /gscratch/.../red_data/red_data_3d_v12_export0902 \
        --raw /gscratch/.../red_data/courtship_label_2026_09_02
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "third_party" / "jarvis_jax"))
from jarvis_jax.data import label_qc  # noqa: E402

RECS = ["2026_04_02_12_11_50_female", "2026_04_02_12_11_50_male",
        "2026_04_02_15_25_51_female", "2026_04_02_15_25_51_male",
        "2026_04_02_17_28_34_female", "2026_04_02_17_28_34_male"]
ZOOM = [("2026_04_02_15_25_51_female", "Cam2012855"),
        ("2026_04_02_17_28_34_female", "Cam2012630"),
        ("2026_04_02_17_28_34_female", "Cam2012857")]


def assigned_projection(root: str, rec_id: str, cams):
    man = json.load(open(os.path.join(root, "manifest.json")))["recordings"]
    grp = man[rec_id]["calib_group"]
    return grp, label_qc.yaml_projection(os.path.join(root, "calibrations", grp), cams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="figures/2026-09-03-red-export-missing-kp")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = {}
    for rec in RECS:
        raw = label_qc.read_raw_labels(os.path.join(a.raw, rec))
        rid = label_qc.recording_id_of(rec)
        rows[rec] = {}
        for tag, root in (("before", a.old), ("after", a.new)):
            grp, proj = assigned_projection(root, rid, raw.cams)
            rep = label_qc.reprojection_report(raw, proj)
            rows[rec][tag] = {"group": grp, **{k: rep[k] for k in
                                               ("median_px", "p95_px", "per_camera_median_px",
                                                "median_of_frame_medians_px", "n_frames")},
                              "inconsistent_frames": rep["inconsistent_frames"]}
            print(f"{rec:36s} {tag:6s} group {grp}: median {rep['median_of_frame_medians_px']:7.3f} px  "
                  + " ".join(f"{c[-3:]}={v:5.1f}" for c, v in rep["per_camera_median_px"].items()))
    json.dump(rows, open(os.path.join(a.out, "calib_group_fix_check.json"), "w"), indent=1)

    cams = sorted(rows[RECS[0]]["before"]["per_camera_median_px"])
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(RECS) * len(cams))
    w = 0.4
    for i, (tag, col) in enumerate((("before", "magenta"), ("after", "limegreen"))):
        vals = [rows[r][tag]["per_camera_median_px"][c] for r in RECS for c in cams]
        ax.bar(x + (i - 0.5) * w, vals, w, color=col, label=f"{tag} (assigned group)")
    ax.set_xticks(x)
    ax.set_xticklabels([c[-3:] for _ in RECS for c in cams], fontsize=7)
    for j, r in enumerate(RECS):
        ax.text(j * len(cams) + len(cams) / 2 - 0.5, ax.get_ylim()[1] * 0.92,
                f"{r[11:]}\nbefore {rows[r]['before']['group']} / after {rows[r]['after']['group']}",
                ha="center", fontsize=8)
        if j:
            ax.axvline(j * len(cams) - 0.5, color="0.7", lw=0.8)
    ax.set_ylabel("median |proj(exported 3D) - raw 2D| (px)")
    ax.set_title("Exported 3D labels through the calibration group each root ASSIGNS, per camera "
                 "(0 px = the labels were triangulated with that calibration)")
    ax.legend(loc="upper left")

    for k, (rec, cam) in enumerate(ZOOM):
        axz = fig.add_subplot(gs[1, k])
        raw = label_qc.read_raw_labels(os.path.join(a.raw, rec))
        rid = label_qc.recording_id_of(rec)
        fr = raw.frames[len(raw.frames) // 2]
        X = raw.kp3d[fr]
        uv = raw.kp2d[cam][fr]
        H = raw.image_hw[cam][0]
        ok = np.isfinite(uv).all(1) & np.isfinite(X).all(1)
        img = None
        for cand in (Path(a.new) / "images" / rid / cam / f"Frame_{fr}.jpg",):
            if cand.exists():
                img = np.asarray(Image.open(cand))
        if img is not None:
            axz.imshow(img, cmap="gray")
        for tag, col, mk in (("before", "magenta", "s"), ("after", "limegreen", "o")):
            _, proj = assigned_projection(a.old if tag == "before" else a.new, rid, raw.cams)
            p = proj[cam](X[ok])
            axz.scatter(p[:, 0], p[:, 1], s=34, marker=mk, facecolors="none",
                        edgecolors=col, lw=1.2, label=f"3D through {tag} group")
        axz.scatter(uv[ok, 0], H - uv[ok, 1], s=8, c="cyan", lw=0, label="raw 2D labels")
        cx, cy = np.median(uv[ok, 0]), np.median(H - uv[ok, 1])
        axz.set_xlim(cx - 110, cx + 110)
        axz.set_ylim(min(cy + 90, H), max(cy - 90, 0))
        axz.set_xticks([]); axz.set_yticks([])
        axz.set_title(f"{rec[11:]} frame {fr} {cam}", fontsize=9)
        if k == 0:
            axz.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out = os.path.join(a.out, "calib_group_fix_check.png")
    fig.savefig(out, dpi=90)
    print("wrote", out)


if __name__ == "__main__":
    main()
