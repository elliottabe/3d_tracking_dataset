#!/usr/bin/env python
"""Pre-launch gate: does the v12 manifest's per-fly sex label agree with what
the fly actually looks like, in every mixed-sex recording the mvq lifter
trains/evals on?

EXPECTATION: for each mixed recording, the fly the manifest calls MALE
(orange) is the smaller body with the dark abdomen tip; the FEMALE (cyan) is
larger with a pointed, pale-striped abdomen. In 2025_10_20_13_20_04 only the
female is labelled: the unlabelled fly in the crop must be the smaller,
darker one. If any recording shows the reverse, its `fly_sex` convention is
wrong and the export must be fixed before training.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 PYTHONPATH=third_party/jarvis_jax:. \\
        python scripts/viz/mvq_sex_label_check.py \\
        --out figures/2026-09-mvq/p3a_gates
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "jarvis_jax")); sys.path.insert(0, ROOT)
from jarvis_jax.data.v12_windows import V12WindowDataset
from viz.core.colors import PALETTE

# (recording, split) -- the four mixed-sex recordings the P3a spec names.
RECORDINGS = [
    ("2025_10_20_13_20_04", "train"),
    ("2026_04_02_17_28_34", "train"),
    ("2026_04_02_12_11_50", "val"),
    ("2026_04_02_15_25_51", "val"),
]
N_FRAMES = 3

FLY_RGB = {f: tuple(c / 255.0 for c in reversed(PALETTE[f])) for f in ("fly0", "fly1")}  # BGR->RGB


def _camera_spread(s):
    """Per-camera bbox diagonal (px) spanning every visible label of every
    labelled fly in this window -- cameras where the flies show up largest /
    most face-on score highest."""
    F, _, C, K, _ = s["kp2d"].shape
    spread = np.zeros(C)
    for c in range(C):
        if not s["cam_valid"][0, c]:
            continue
        pts = [s["kp2d"][f, 0, c][s["vis2d"][f, 0, c]] for f in range(F) if s["fly_valid"][f]]
        pts = np.concatenate(pts, axis=0) if pts else np.zeros((0, 2), np.float32)
        if pts.shape[0] >= 2:
            spread[c] = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    return spread


def _select_cameras(s):
    """Top-2 cameras for this row. When fly1 is valid, rank by how MANY of
    fly1's keypoints actually landed inside this camera's crop (`vis2d`) --
    fly0 is always well represented near the crop centre, so fly1's
    visibility is the binding constraint on whether both bodies are actually
    checkable in the panel. Falls back to overall label spread (fly0 only,
    or fly1 invalid)."""
    C = s["kp2d"].shape[2]
    valid_c = [c for c in range(C) if s["cam_valid"][0, c]]
    if s["fly_valid"][1] and s["vis2d"][1, 0].any():
        score = s["vis2d"][1, 0].sum(1).astype(float)
    else:
        score = _camera_spread(s)
    ranked = sorted(valid_c, key=lambda c: -score[c])
    return ranked[:2] if len(ranked) >= 2 else ranked


def _pick_windows(ds):
    """3 host-fly0 windows, built once and cached (`{i: sample}`) so the
    caller never re-decodes. When fly1 is labelled anywhere in the
    recording, actually build every host-fly0 window (cheap: <=44 windows
    here) and rank by how many of fly1's keypoints land inside the crop in
    their BEST camera -- `fly_valid[1]` alone (any single point inside the
    448x448 box) is routinely satisfied by one stray point while the rest of
    fly1's body sits outside the crop, which is useless for a visual sex
    check. Picks the 3 windows with the most fly1 coverage (not spread across
    the whole recording -- coverage is what makes the panel checkable at
    all); falls back to plain `fly_valid[1]`, then to any host-fly0 window,
    if nothing clears the coverage bar."""
    host0 = [i for i in range(len(ds)) if ds.windows[i][1] == 0]
    has_fly1 = any(ds.windows[i][1] == 1 for i in range(len(ds)))
    if not has_fly1:
        cand = host0 if len(host0) <= N_FRAMES else [
            host0[k] for k in np.linspace(0, len(host0) - 1, N_FRAMES).round().astype(int)]
        return [(i, ds[i]) for i in cand]
    built = {i: ds[i] for i in host0}
    best_cov = {i: int(built[i]["vis2d"][1, 0].sum(1).max()) if built[i]["fly_valid"][1] else 0
               for i in host0}
    covered = sorted([i for i in host0 if best_cov[i] > 0], key=lambda i: -best_cov[i])
    cand = covered[:N_FRAMES] if covered else host0[:N_FRAMES]
    return [(i, built[i]) for i in cand]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for rec, split in RECORDINGS:
        ds = V12WindowDataset(a.root, split, T=1, train=False, recordings=[rec])
        sex0 = ds.manifest[rec]["fly_sex"].get("fly0")
        sex1_manifest = ds.manifest[rec]["fly_sex"].get("fly1")
        for i, s in _pick_windows(ds):
            top2 = _select_cameras(s)
            cam_names = ds.camera_names(i)
            fly1_labelled = bool(s["fly_valid"][1])
            sex1 = sex1_manifest if fly1_labelled else None
            rows.append(dict(rec=rec, split=split, i=i, cams=[int(c) for c in top2],
                             cam_names=[cam_names[c] for c in top2],
                             sex0=sex0, sex1=sex1, fly1_labelled=fly1_labelled,
                             sample=s))

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(7.6, 2.6 * n_rows), squeeze=False)
    for r_i, r in enumerate(rows):
        s = r["sample"]
        for c_i, c in enumerate(r["cams"]):
            ax = axes[r_i, c_i]
            ax.imshow(s["crops"][0, c]); ax.set_xticks([]); ax.set_yticks([])
            for f, key in ((0, "fly0"), (1, "fly1")):
                if not s["fly_valid"][f]:
                    continue
                vis = s["vis2d"][f, 0, c]
                if not vis.any():
                    continue
                pts = s["kp2d"][f, 0, c][vis]
                ax.scatter(pts[:, 0], pts[:, 1], s=8, c=[FLY_RGB[key]], label=key)
            cam = r["cam_names"][c_i]
            ax.set_title(f"{r['rec']}\n{cam}  fly0={r['sex0']}  fly1={r['sex1'] or 'unlabelled'}",
                        fontsize=6.5, linespacing=1.4)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), fontsize=7, loc="upper left",
              bbox_to_anchor=(0.0, 1.0), bbox_transform=fig.transFigure)
    fig.suptitle("mvq sex-label gate  --  cyan=fly0  orange=fly1", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.2, w_pad=1.5)
    png = os.path.join(a.out, "sex_label_check.png")
    fig.savefig(png, dpi=140); plt.close(fig)
    json.dump([{k: v for k, v in r.items() if k != "sample"} for r in rows],
              open(os.path.join(a.out, "sex_label_check.json"), "w"), indent=1)
    print("wrote", png)


if __name__ == "__main__":
    main()
