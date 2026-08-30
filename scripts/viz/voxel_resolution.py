"""Per-joint 3D error against segment length in voxels — the falsifiable gate.

STATED EXPECTATION. If the resolution hypothesis is right, per-joint error at
stage 1 (48^3, spacing 1) concentrates on the SHORT segments: the tarsal links
T1L_TiTa->T1L_TaT1, T1L_TaT1->T1L_TaT3 and T1L_TaT3->T1L_TaTip are 1.6-2.2
voxels and should show the largest error relative to their own length, while
the body (12.8 voxels) and wing (17.5 voxels) segments should be comparatively
clean. Stage 2 at spacing 0.25 should collapse that dependence.

IF ERROR IS FLAT ACROSS SEGMENT LENGTHS, THE HYPOTHESIS IS WRONG and the
remaining error lives upstream (2D) or downstream (IK). Report that outcome
rather than reaching for it.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SEGMENTS = [
    ("Scutellum", "Abd_tip"), ("WingL_base", "WingL_V12"),
    ("WingL_V12", "WingL_V13"), ("T1L_FeTi", "T1L_TiTa"),
    ("T1L_TiTa", "T1L_TaT1"), ("T1L_TaT1", "T1L_TaT3"),
    ("T1L_TaT3", "T1L_TaTip"), ("T3L_TiTa", "T3L_TaT1"), ("EyeL", "EyeR"),
]


def segment_lengths_voxels(kp3d, keypoint_names, grid_spacing: float = 1.0) -> dict:
    """Median segment length expressed in VOXELS at the given grid spacing."""
    kp3d = np.asarray(kp3d, np.float64)
    out = {}
    for a, b in SEGMENTS:
        if a not in keypoint_names or b not in keypoint_names:
            continue
        ia, ib = keypoint_names.index(a), keypoint_names.index(b)
        d = np.linalg.norm(kp3d[:, ia] - kp3d[:, ib], axis=-1)
        med = np.nanmedian(d)
        if np.isfinite(med):
            out[f"{a}->{b}"] = float(med / grid_spacing)
    return out


def plot_error_vs_segment(pred, gt, keypoint_names, *, out_png,
                          grid_spacing: float = 1.0, label: str = "") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lengths = segment_lengths_voxels(gt, keypoint_names, grid_spacing)
    err = np.nanmedian(np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=-1), 0)
    xs, ys, names = [], [], []
    for seg, L in lengths.items():
        a, b = seg.split("->")
        e = float(np.nanmean([err[keypoint_names.index(a)],
                              err[keypoint_names.index(b)]]))
        xs.append(L); ys.append(e / max(L, 1e-6)); names.append(seg)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(xs, ys, s=48, c="#00c2c7")
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axvline(2.0, ls="--", c="#888",
               label="2 voxels — below this, quantization dominates")
    ax.set_xlabel("segment length (voxels at grid_spacing=%.2f)" % grid_spacing)
    ax.set_ylabel("median 3D error / segment length")
    ax.set_title(f"Per-joint error vs segment resolution {label}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    stats = {"lengths_voxels": lengths,
             "err_over_length": dict(zip(names, ys)),
             "short_segments_mean": float(np.mean([y for x, y in zip(xs, ys) if x < 2.0]))
             if any(x < 2.0 for x in xs) else None,
             "long_segments_mean": float(np.mean([y for x, y in zip(xs, ys) if x >= 4.0]))
             if any(x >= 4.0 for x in xs) else None}
    with open(out_png.replace(".png", ".json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-npz", required=True)
    ap.add_argument("--gt-npz", required=True)
    ap.add_argument("--names-json", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--grid-spacing", type=float, default=1.0)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    names = json.load(open(a.names_json))
    stats = plot_error_vs_segment(np.load(a.pred_npz)["kp3d"],
                                  np.load(a.gt_npz)["kp3d"], names,
                                  out_png=a.out_png, grid_spacing=a.grid_spacing,
                                  label=a.label)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
