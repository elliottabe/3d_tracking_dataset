#!/usr/bin/env python
"""Per-keypoint site-residual diagnostic for a STAC IK fit.

The success metric for keypoint→model alignment is "each keypoint sits at its
model tracking-site location after IK". STAC's IK output h5 already stores the
fitted model site positions (``marker_sites``) and the target keypoints
(``kp_data``), so the residual is a direct read — no need to re-run kinematics.

This script loads an ``Fruitfly_ik_*.h5`` (or any STAC IK output), computes the
per-keypoint Euclidean residual ``‖marker_site − kp_data‖`` over all frames, and
reports the median / 90th-percentile residual per keypoint sorted worst-first,
grouped by body region (head / wings / abdomen / trunk / legs-by-segment) so the
tibia/femur, head, and wing keypoints called out in the brainstorming are easy
to read. Emits a CSV and a bar plot next to the run.

Usage:
    python scripts/diagnose_site_residual.py <ik_h5> [--tag NAME] [--out DIR]

Use it to quantify a baseline run and to compare a candidate fix (e.g. Umeyama
global scale + per-segment calibration) against it: rerun on each output and
compare the per-keypoint residual tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Body-region classification (purely from the keypoint name) for grouping the
# report. Leg keypoints keep their joint label so segments are distinguishable.
_LEG_JOINT_LABEL = {
    "ThxCx": "leg:coxa",
    "Tro": "leg:trochanter/femur",
    "FeTi": "leg:femur/tibia",
    "TiTa": "leg:tibia/tarsus",
    "TaT1": "leg:tarsus",
    "TaT3": "leg:tarsus",
    "TaTip": "leg:claw",
}


def classify(name: str) -> str:
    """Return a coarse body-region label for a keypoint name."""
    if name in ("Antenna_Base", "EyeL", "EyeR"):
        return "head"
    if name.startswith("Wing"):
        return "wing"
    if name.startswith("Abd"):
        return "abdomen"
    if name == "Scutellum":
        return "trunk"
    # Legs: T1L_FeTi -> joint label
    if len(name) > 3 and name[0] == "T" and name[2] in ("L", "R") and "_" in name:
        joint = name.split("_", 1)[1]
        return _LEG_JOINT_LABEL.get(joint, "leg:other")
    return "other"


_REGION_COLOR = {
    "head": "#d62728",
    "wing": "#9467bd",
    "abdomen": "#8c564b",
    "trunk": "#2ca02c",
    "leg:coxa": "#1f77b4",
    "leg:trochanter/femur": "#1f77b4",
    "leg:femur/tibia": "#ff7f0e",
    "leg:tibia/tarsus": "#17becf",
    "leg:tarsus": "#7f7f7f",
    "leg:claw": "#bcbd22",
}


def load_ik(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load (kp_data[F,K,3], marker_sites[F,K,3], kp_names) from a STAC IK h5."""
    with h5py.File(str(h5_path), "r") as f:
        kp = f["kp_data"][()]
        ms = f["marker_sites"][()]
        names = [n.decode("utf-8") if isinstance(n, bytes) else str(n)
                 for n in f["kp_names"][()]]
    n_kp = len(names)
    if kp.ndim == 2:                       # flattened (F, K*3) in (kp, xyz) order
        kp = kp.reshape(kp.shape[0], n_kp, 3)
    if ms.ndim == 2:
        ms = ms.reshape(ms.shape[0], n_kp, 3)
    return kp, ms, names


def body_length(ms: np.ndarray, names: List[str]) -> float:
    """Robust model body-length proxy: median ‖Scutellum − Abd_tip‖ over frames.

    Used to express residuals as a fraction of body length (unit-agnostic).
    Falls back to the trunk-marker RMS spread if those keypoints are absent.
    """
    idx = {n: i for i, n in enumerate(names)}
    if "Scutellum" in idx and "Abd_tip" in idx:
        d = np.linalg.norm(ms[:, idx["Scutellum"], :] - ms[:, idx["Abd_tip"], :], axis=-1)
        d = d[np.isfinite(d)]
        if d.size:
            return float(np.median(d))
    centered = ms - np.nanmean(ms, axis=1, keepdims=True)
    return float(np.sqrt(np.nanmean((centered ** 2).sum(-1))))


def per_keypoint_residual(kp: np.ndarray, ms: np.ndarray) -> np.ndarray:
    """Euclidean residual per (frame, keypoint), NaN where the target is NaN."""
    d = np.linalg.norm(ms - kp, axis=-1)   # (F, K)
    bad = ~np.isfinite(kp).all(axis=-1)
    d[bad] = np.nan
    return d


def summarize(d: np.ndarray, names: List[str], blen: float) -> List[Dict]:
    """Per-keypoint summary rows sorted worst-first by median residual."""
    rows = []
    for k, name in enumerate(names):
        col = d[:, k]
        col = col[np.isfinite(col)]
        if col.size == 0:
            rows.append(dict(keypoint=name, region=classify(name), n_valid=0,
                             median=np.nan, p90=np.nan, mean=np.nan, rel_pct=np.nan))
            continue
        med = float(np.median(col))
        rows.append(dict(
            keypoint=name, region=classify(name), n_valid=int(col.size),
            median=med, p90=float(np.percentile(col, 90)),
            mean=float(np.mean(col)),
            rel_pct=(100.0 * med / blen) if blen > 0 else np.nan,
        ))
    rows.sort(key=lambda r: (np.nan_to_num(r["median"], nan=-1.0)), reverse=True)
    return rows


def write_csv(rows: List[Dict], out_csv: Path) -> None:
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "keypoint", "region", "n_valid", "median", "p90", "mean", "rel_pct"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot(rows: List[Dict], out_png: Path, blen: float, tag: str) -> None:
    rows = [r for r in rows if r["n_valid"] > 0]
    names = [r["keypoint"] for r in rows]
    meds = [r["median"] for r in rows]
    p90s = [r["p90"] for r in rows]
    colors = [_REGION_COLOR.get(r["region"], "#333333") for r in rows]
    fig, ax = plt.subplots(figsize=(12, max(4, 0.28 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, meds, color=colors, label="median")
    ax.plot(p90s, y, "k.", ms=5, label="p90")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("site↔keypoint residual (model units)")
    ax.set_title(f"Per-keypoint STAC site residual — {tag}\n"
                 f"body-length proxy = {blen:.4f} (top axis = % body length)")
    secx = ax.secondary_xaxis("top", functions=(lambda x: 100 * x / blen,
                                                 lambda x: x * blen / 100))
    secx.set_xlabel("% of body length")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               dict.fromkeys(_REGION_COLOR.values())]
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ik_h5", type=Path, help="STAC IK output h5 (Fruitfly_ik_*.h5)")
    ap.add_argument("--tag", default=None, help="label for outputs (default: h5 stem)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: alongside the h5)")
    args = ap.parse_args()

    tag = args.tag or args.ik_h5.stem
    out_dir = args.out or args.ik_h5.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    kp, ms, names = load_ik(args.ik_h5)
    blen = body_length(ms, names)
    d = per_keypoint_residual(kp, ms)
    rows = summarize(d, names, blen)

    out_csv = out_dir / f"site_residual_{tag}.csv"
    out_png = out_dir / f"site_residual_{tag}.png"
    write_csv(rows, out_csv)
    plot(rows, out_png, blen, tag)

    valid = d[np.isfinite(d)]
    print(f"\n=== Site residual diagnostic: {tag} ===")
    print(f"frames={kp.shape[0]}  keypoints={len(names)}  "
          f"body-length proxy={blen:.4f} (model units)")
    print(f"overall median={np.median(valid):.5f}  "
          f"mean={np.mean(valid):.5f}  p90={np.percentile(valid,90):.5f} "
          f"({100*np.median(valid)/blen:.1f}% / {100*np.percentile(valid,90)/blen:.1f}% body len)")
    print(f"\n{'keypoint':<14}{'region':<22}{'median':>9}{'p90':>9}{'%bodylen':>10}{'n':>9}")
    print("-" * 73)
    for r in rows:
        if r["n_valid"] == 0:
            print(f"{r['keypoint']:<14}{r['region']:<22}{'--':>9}{'--':>9}{'--':>10}{0:>9}")
        else:
            print(f"{r['keypoint']:<14}{r['region']:<22}{r['median']:>9.5f}"
                  f"{r['p90']:>9.5f}{r['rel_pct']:>9.1f}%{r['n_valid']:>9}")
    print(f"\nwrote {out_csv}\nwrote {out_png}")


if __name__ == "__main__":
    main()
