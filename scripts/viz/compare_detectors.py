#!/usr/bin/env python3
"""Overlay two detectors' 2D keypoints on the same frames, side by side.

Within-bone CV says whether a detector's 3D is self-CONSISTENT, not whether it
is RIGHT: a detector that puts the whole skeleton confidently on the wrong blob
scores well. So an A/B that moves CV has to be looked at before it is believed
-- especially where CV and confidence disagree (Session0 bout 25 fly0 got a
worse CV while its confidence rose, which is the signature of confidently-wrong
output).

Draws one row per camera: the same frame twice, OLD keypoints left, NEW right,
skeleton coloured by body group so a scrambled limb is obvious. Reads frames
sync-aware (viz.core.io.read_frames_synced) so a camera that dropped a frame
still shows the instant the keypoints belong to.

Usage:
    python scripts/viz/compare_detectors.py \\
        --old <root>/old/<rec> --new <root>/new/<rec> \\
        --session-dir <videos>/<rec> --bout 8 --fly 0 --frames 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"
for p in (str(PROJECT_DIR), str(PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

GROUPS = [("head", ("Antenna", "Eye"), "#e41a1c"),
          ("trunk", ("Scutellum", "Abd"), "#377eb8"),
          ("wingL", ("WingL",), "#4daf4a"),
          ("wingR", ("WingR",), "#984ea3"),
          ("legL", ("T1L", "T2L", "T3L"), "#ff7f00"),
          ("legR", ("T1R", "T2R", "T3R"), "#a65628")]


def colour_of(name):
    for _g, pref, c in GROUPS:
        if name.startswith(pref):
            return c
    return "#999999"


def load_kp2d(root, bout, fly):
    p = Path(root) / "bouts" / f"bout_{int(bout):05d}" / f"fly{int(fly)}" / "kp2d.npz"
    if not p.is_file():
        raise FileNotFoundError(p)
    with np.load(p) as z:
        return np.asarray(z["kp2d"]), np.asarray(z["conf"])


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hydra import compose, initialize_config_dir
    from viz.core import io as vio

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--bout", type=int, required=True)
    ap.add_argument("--fly", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--frames", type=int, default=3, help="frames sampled across the bout")
    ap.add_argument("--cameras", nargs="*", default=None)
    ap.add_argument("--recording-cfg", default="session1")
    ap.add_argument("--conf-thresh", type=float, default=0.3)
    ap.add_argument("--pad", type=int, default=120)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    with initialize_config_dir(version_base=None,
                               config_dir=str(PROJECT_DIR / "configs")):
        cfg = compose(config_name="pipeline",
                      overrides=["paths=hyak", f"recording={a.recording_cfg}",
                                 f"recording.session_dir={a.session_dir}"])
    kp_names = list(cfg.model.KP_NAMES)
    cams = list(a.cameras) if a.cameras else list(cfg.recording.cameras)
    cols = [colour_of(n) for n in kp_names]

    kp_o, cf_o = load_kp2d(a.old, a.bout, a.fly)
    kp_n, cf_n = load_kp2d(a.new, a.bout, a.fly)
    T = min(kp_o.shape[0], kp_n.shape[0])
    ts = np.unique(np.linspace(0, T - 1, a.frames).astype(int)).tolist()

    start = a.start_frame
    if start is None:
        from scripts.run_bout import bout_start_frame
        start = int(bout_start_frame(cfg, a.bout))

    all_cams = list(cfg.recording.cameras)
    cam_idx = [all_cams.index(c) for c in cams]

    nrow, ncol = len(ts) * len(cams), 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(9 * ncol, 2.6 * nrow),
                             squeeze=False)
    row = 0
    for t in ts:
        frames = next(iter(vio.read_frames_synced(a.session_dir, cams, start + t, 1)))
        for ci, (cam, rgb) in enumerate(zip(cams, frames)):
            k = cam_idx[ci]
            for col, (kp, cf, label) in enumerate(
                    ((kp_o, cf_o, "OLD"), (kp_n, cf_n, "NEW"))):
                ax = axes[row][col]
                if rgb is None:
                    ax.set_title(f"{cam} t={t} {label}: frame unavailable", fontsize=8)
                    ax.axis("off"); continue
                xy = kp[t, k]; c = cf[t, k]
                good = c >= a.conf_thresh
                if good.any():
                    cx, cy = xy[good, 0].mean(), xy[good, 1].mean()
                else:
                    cx, cy = rgb.shape[1] / 2, rgb.shape[0] / 2
                x0 = int(max(0, cx - a.pad)); x1 = int(min(rgb.shape[1], cx + a.pad))
                y0 = int(max(0, cy - a.pad)); y1 = int(min(rgb.shape[0], cy + a.pad))
                ax.imshow(rgb[y0:y1, x0:x1])
                ax.scatter(xy[good, 0] - x0, xy[good, 1] - y0,
                           s=16, c=[cols[i] for i in np.where(good)[0]],
                           edgecolors="k", linewidths=0.3)
                ax.set_title(f"{cam} t={t}  {label}  median conf {np.median(c):.2f}",
                             fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            row += 1
    fig.suptitle(f"detector A/B  bout {a.bout} fly{a.fly}  "
                 f"{os.path.basename(a.session_dir)}   LEFT=old  RIGHT=new",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=100)
    print(f"wrote {a.out}  ({len(ts)} frames x {len(cams)} cameras)")


if __name__ == "__main__":
    main()
