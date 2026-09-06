#!/usr/bin/env python
"""Acceptance figure for the mvq lifter's OWN-WINDOW PREFERENCE: does the fly
carry usable per-view confidence over its WHOLE body, or only over the half
that happened to fall inside its partner's crop?

STATE THE EXPECTATION BEFORE LOOKING (CLAUDE.md). One row per run, one column
per bout frame, for one camera at a time. Each panel draws the chosen fly's
2D keypoints on the real frame, coloured by the `conf3d` written beside them
in kp3d.npz (the mean per-view visibility, which is what gates the offsets
sampler at 0.7, the rigid-edge repair at 0.5, the side-by-side display and the
pseudo-label export), with her SAM3 mask outlined in white for context and the
partner's keypoints in faint grey.

If the preference is working, the AFTER row shows the WHOLE fly in
high-confidence colour, where the BEFORE row has her head, thorax and front
legs dark (conf3d ~0.01-0.1) because that half of her body sat at or past the
edge of the crop centred on the MALE's mask. Points that stay dark after the
change are frames where her own window offered no candidate and the lifter
fell back -- the panel title says which window each read came from, so the two
cases are distinguishable rather than assumed.

A row where the dark points do NOT correspond to the crop-edge half of the
body, or where the 2D points move to a different animal between rows, would
mean the preference is picking a different instance rather than a better crop
of the same one -- check `figures/.../ownwindow_compare.json` for the 3D delta.

Usage:

    PYTHONPATH=third_party/jarvis_jax:. python scripts/viz/ownwindow_conf_check.py \\
        --session-dir <video>/courtship/Session1/2026_04_02_14_54_28 \\
        --masks-dir   <proc>/.../sam3_masks \\
        --run   after=<worktree>/OutFiles/ownwindow_check/2026_04_02_14_54_28 \\
        --compare before=<proc>/.../pose_mvq_p3b \\
        --bout 10 --fly 0 --frames 30,110,190,270 \\
        --cameras Cam2012630,Cam2012861 \\
        --out figures/2026-09-mvq/p3b_gates/ownwindow_bout10.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "jarvis_jax"))
sys.path.insert(0, ROOT)

from omegaconf import OmegaConf  # noqa: E402

from jarvis_jax.predict.synced_reader import load_plan, read_window  # noqa: E402
from jarvis_jax.tracking.lift_mvq import BoutMaskStore, resolve_bout_frames  # noqa: E402

FLY_NAME = ("fly0 (female)", "fly1 (male)")
DEF_RECORDING_CFG = os.path.join(ROOT, "configs", "recording", "session1.yaml")


def parse_run(spec):
    """`label=path`; the split is on the FIRST '=' so a label containing one
    would silently yield a nonexistent path and an empty row."""
    if "=" not in spec:
        return os.path.basename(os.path.normpath(spec)), spec
    label, path = spec.split("=", 1)
    if not os.path.isdir(path):
        raise SystemExit(f"run spec {spec!r}: {path!r} is not a directory")
    return label, path


def load_bout(run_dir, bout, cameras):
    """{fly: (kp2d (T,C,K,2) in `cameras` order BY NAME, conf3d (T,K), names)}
    plus the run's mvq_meta.json. The camera axis is permuted by NAME and a npz
    without a `cameras` array is REFUSED -- assuming it positional is the
    camera-scramble bug (CLAUDE.md)."""
    d = os.path.join(run_dir, "bouts", f"bout_{bout:05d}")
    out = {}
    for fly in (0, 1):
        with np.load(os.path.join(d, f"fly{fly}", "kp2d.npz")) as z:
            if "cameras" not in z.files:
                raise RuntimeError(f"{d}/fly{fly}/kp2d.npz carries no `cameras` array")
            have = [str(c) for c in z["cameras"].tolist()]
            missing = [c for c in cameras if c not in have]
            if missing:
                raise RuntimeError(f"{d}/fly{fly}/kp2d.npz has {have}, asked {missing}")
            kp2d = np.asarray(z["kp2d"])[:, [have.index(c) for c in cameras]]
            names2 = [str(n) for n in z["kp_names"]]
        with np.load(os.path.join(d, f"fly{fly}", "kp3d.npz")) as z:
            names3 = [str(n) for n in z["kp_names"]]
            if names3 != names2:
                raise RuntimeError(f"{d}/fly{fly}: kp2d and kp3d disagree on keypoint order")
            conf3d = np.asarray(z["conf3d"])
        out[fly] = (kp2d, conf3d, names3)
    meta = json.load(open(os.path.join(d, "mvq_meta.json")))
    return out, meta


def window_note(meta, fly, t):
    """"window 0 (own)" / "window 1 (other)" / "window 1" when the run predates
    the own_window record."""
    pf = meta.get("per_frame") or {}
    w = (pf.get("window") or [[], []])[fly][t]
    ow = pf.get("own_window")
    if w < 0:
        return "no instance"
    if not ow:
        return f"window {w}"
    o = ow[fly][t]
    return f"window {w} ({'own' if o == 1 else 'other' if o == 0 else '-'})"


def mask_outline(store, fly, cam_i, t):
    import cv2
    m = store.mask_at(fly, cam_i, t)
    if not m.any():
        return None
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea).reshape(-1, 2)
    return np.concatenate([c, c[:1]], axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--run", required=True, help="label=run_dir (the run under test)")
    ap.add_argument("--compare", action="append", default=[], help="label=run_dir")
    ap.add_argument("--bout", type=int, required=True)
    ap.add_argument("--fly", type=int, default=0, choices=(0, 1))
    ap.add_argument("--frames", default="", help="comma-separated BOUT-LOCAL frames")
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--cameras", default="Cam2012630,Cam2012861",
                    help="overhead,side BY NAME")
    ap.add_argument("--recording-cfg", default=DEF_RECORDING_CFG)
    ap.add_argument("--bouts-csv", default=None)
    ap.add_argument("--pad", type=int, default=90)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    canonical = [str(c) for c in OmegaConf.load(args.recording_cfg).cameras]
    draw_cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    bad = [c for c in draw_cams if c not in canonical]
    if bad:
        raise SystemExit(f"cameras {bad} not in canonical order {canonical}")

    runs = [parse_run(s) for s in args.compare] + [parse_run(args.run)]
    data = {label: load_bout(path, args.bout, draw_cams) for label, path in runs}
    after_label = runs[-1][0]
    T = int(data[after_label][1]["n_frames"])

    if args.frames:
        ts = [int(x) for x in args.frames.split(",") if x.strip()]
    else:
        # the hard cases, not the flattering ones: the closest-contact frame and
        # the frame where the chosen fly's BEFORE confidence is worst, plus two
        # spread over the bout
        before_label = runs[0][0]
        kb, cb, _ = data[before_label][0][args.fly]
        _kp_other, _c_other, _ = data[before_label][0][1 - args.fly]
        worst = int(np.nanargmin(np.nanmean(cb, axis=1)))
        cen = np.nanmean(np.nanmean(kb, axis=1), axis=0)  # unused, kept explicit
        ts = sorted({T // 6, worst, T // 2, 5 * T // 6})
    ts = [t for t in ts if 0 <= t < T][:max(args.n_frames, 1)]

    mask_npz = os.path.join(args.masks_dir, f"bout_{args.bout:05d}", "sam3_masks.npz")
    store = BoutMaskStore(mask_npz, canonical)
    abs_start, _e, n_csv = resolve_bout_frames(args.session_dir, args.bout,
                                               bouts_csv=args.bouts_csv)
    if n_csv != store.T:
        raise SystemExit(f"bouts CSV says {n_csv} frames, mask npz has {store.T}")
    plan = load_plan(args.session_dir)
    imgs = {}
    for t in ts:
        frames, _present = next(iter(read_window(args.session_dir, canonical, plan,
                                                 abs_start + t, 1)))
        imgs[t] = frames

    nrow = len(draw_cams) * len(runs)
    ncol = len(ts)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow), squeeze=False)
    sm = None
    for ci, cam in enumerate(draw_cams):
        cam_i = canonical.index(cam)                    # BY NAME
        for ri, (label, _p) in enumerate(runs):
            row = ci * len(runs) + ri
            kp2d, conf3d, names = data[label][0][args.fly]
            kp_o = data[label][0][1 - args.fly][0]
            meta = data[label][1]
            for col, t in enumerate(ts):
                ax = axes[row][col]
                img = imgs[t][cam_i]
                o = mask_outline(store, args.fly, cam_i, t)
                p = kp2d[t, ci]
                good = np.isfinite(p).all(-1)
                xs = list(p[good, 0]) + (list(o[:, 0]) if o is not None else [])
                ys = list(p[good, 1]) + (list(o[:, 1]) if o is not None else [])
                ax.imshow(img)
                if o is not None:
                    ax.plot(o[:, 0], o[:, 1], color="white", lw=0.8, alpha=0.8)
                po = kp_o[t, ci]
                go = np.isfinite(po).all(-1)
                ax.scatter(po[go, 0], po[go, 1], s=4, c="0.55", alpha=0.7, linewidths=0)
                sm = ax.scatter(p[good, 0], p[good, 1], s=16, c=conf3d[t][good],
                                cmap="viridis", vmin=0.0, vmax=1.0,
                                edgecolors="k", linewidths=0.25)
                if xs:
                    ax.set_xlim(max(min(xs) - args.pad, 0), min(max(xs) + args.pad,
                                                                img.shape[1]))
                    ax.set_ylim(min(max(ys) + args.pad, img.shape[0]),
                                max(min(ys) - args.pad, 0))
                ax.set_xticks([]); ax.set_yticks([])
                med = float(np.nanmedian(conf3d[t][good])) if good.any() else float("nan")
                ax.set_title(f"{label} | {cam} | frame {t}\n"
                             f"{window_note(meta, args.fly, t)}  median conf3d {med:.2f}",
                             fontsize=7)
    fig.suptitle(
        f"{os.path.basename(os.path.normpath(args.session_dir))} bout {args.bout} "
        f"-- {FLY_NAME[args.fly]} keypoints coloured by conf3d (mean per-view "
        f"visibility, 0-1)\nwhite = her SAM3 mask outline, grey = the other fly's "
        f"keypoints; own-window preference before vs after", fontsize=9)
    if sm is not None:
        fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label="conf3d")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}  frames {ts}  cameras {draw_cams}")


if __name__ == "__main__":
    main()
