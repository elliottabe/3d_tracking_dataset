#!/usr/bin/env python
"""Acceptance figure for the mvq lifter's IDENTITY assignment: is the fly the
pipeline calls fly0 actually the animal the human review calls fly0?

STATE THE EXPECTATION BEFORE LOOKING (CLAUDE.md). One bout frame, one panel
per camera per run. Each panel draws:

  * the SAM3 mask outlines the human id review labelled -- fly0 (the FEMALE)
    as a cyan outline, fly1 (the MALE) as an orange outline;
  * the keypoints the run WROTE -- fly0 in cyan dots, fly1 in orange dots,
    from `<run>/bouts/bout_<idx>/fly{0,1}/kp2d.npz` (MODEL keypoint order,
    CANONICAL camera order; both axes are taken BY NAME here, never by
    integer -- CLAUDE.md's two index traps).

If the identity assignment is correct, CYAN DOTS SIT INSIDE THE CYAN OUTLINE
and ORANGE DOTS INSIDE THE ORANGE OUTLINE, in every camera. Cyan dots absent
means the female was NaN that frame. Cyan or orange dots sitting on the OTHER
outline is an identity swap -- the exact 20_04 failure this figure gates
(`.superpowers/sdd/2026-09-04-mvq-maskfree-p4a-p4b/female-miss-diagnosis.md`:
the sex head types that female as a male, so `identity=sex` NaN'd fly0 on
97.7% of bout 25 and put fly1 on her body on 21% of frames).

MAKE IT COMPARATIVE. `--compare <label>=<run_dir>` adds a row per run, so the
same frame is drawn for the sex-head lift and the mask-identity lift side by
side. A lone render of the new behaviour is decoration.

Usage:

    MUJOCO_GL=egl PYTHONPATH=third_party/jarvis_jax:. \\
    python scripts/viz/mvq_identity_overlay.py \\
        --session-dir <video>/courtship/Session0/2025_10_20_13_20_04 \\
        --masks-dir   <proc>/.../sam3_masks \\
        --run         after=<proc>/.../mvq_identity_smoke \\
        --compare     before=<proc>/.../pose_mvq_p3a \\
        --bout 25 --frame 200 --cameras Cam2012630,Cam2012861 \\
        --out figures/2026-09-mvq/p3a_campaign_female_misses/identity_mask_bout25.png
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
from viz.core.colors import PALETTE  # noqa: E402

# PALETTE is BGR (cv2); matplotlib wants RGB 0-1.
_RGB = lambda key: tuple(c / 255.0 for c in reversed(PALETTE[key]))
FLY_COLOR = (_RGB("fly0"), _RGB("fly1"))          # cyan = fly0/female, orange = fly1/male
FLY_NAME = ("fly0 (female)", "fly1 (male)")

DEF_RECORDING_CFG = os.path.join(ROOT, "configs", "recording", "session0.yaml")


def parse_run(spec):
    """`label=path` (or a bare path, labelled by its basename).

    The label must not itself contain `=` -- the split is on the FIRST one, so
    `before (identity=sex)=/run` would silently yield the path `sex)=/run`, a
    directory that does not exist, and the row would render with mask outlines
    and NO keypoints: a figure that looks like "the run wrote nothing" when it
    is really "the label was mis-parsed". Refused here instead.
    """
    if "=" not in spec:
        return os.path.basename(os.path.normpath(spec)), spec
    label, path = spec.split("=", 1)
    if not os.path.isdir(path):
        raise SystemExit(
            f"run spec {spec!r}: {path!r} is not a directory. Use `label=path` with no "
            f"'=' inside the label (a mis-parsed label renders an empty row).")
    return label, path


def mask_outline(store, fly, cam_i, t):
    """The (N,2) px contour of one mask, or None. `cam_i` is a CANONICAL camera
    index and `store.mask_at` does the by-name permutation itself."""
    import cv2
    m = store.mask_at(fly, cam_i, t)
    if not m.any():
        return None
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea).reshape(-1, 2)
    return np.concatenate([c, c[:1]], axis=0)      # closed


def load_kp2d(run_dir, bout, cameras):
    """{fly: (T,C,K,2)} for one bout, camera axis permuted into `cameras` BY
    NAME. A run that lifted a bout with a different camera order (or a legacy
    npz with none) is REFUSED rather than assumed positional -- that
    assumption is the camera-scramble bug."""
    out = {}
    for fly in (0, 1):
        p = os.path.join(run_dir, "bouts", f"bout_{bout:05d}", f"fly{fly}", "kp2d.npz")
        if not os.path.exists(p):
            return None
        with np.load(p) as z:
            if "cameras" not in z.files:
                raise RuntimeError(f"{p} carries no `cameras` name array, so its camera "
                                   f"axis cannot be verified by name")
            have = [str(c) for c in z["cameras"].tolist()]
            missing = [c for c in cameras if c not in have]
            if missing:
                raise RuntimeError(f"{p} has cameras {have}; asked for {missing}")
            perm = [have.index(c) for c in cameras]
            out[fly] = np.asarray(z["kp2d"])[:, perm]
    return out


def meta_line(run_dir, bout):
    """One line of provenance from the run's own mvq_meta.json."""
    p = os.path.join(run_dir, "bouts", f"bout_{bout:05d}", "mvq_meta.json")
    if not os.path.exists(p):
        return ""
    m = json.load(open(p))
    T = max(int(m.get("n_frames", 0)), 1)
    nm = m.get("n_missing") or {}
    d = m.get("sex_head_disagree_frac") or {}
    return (f"identity={m.get('identity_resolved', 'sex')}  "
            f"NaN fly0 {nm.get('fly0', 0) / T:.3f} / fly1 {nm.get('fly1', 0) / T:.3f}  "
            f"sex-head disagree f0 {d.get('fly0')} f1 {d.get('fly1')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--masks-dir", required=True, help="the SAM3 mask tree")
    ap.add_argument("--run", required=True, help="label=run_dir (the run under test)")
    ap.add_argument("--compare", action="append", default=[],
                    help="label=run_dir of another lift, drawn as its own row (repeatable)")
    ap.add_argument("--bout", type=int, required=True)
    ap.add_argument("--frame", type=int, default=0, help="BOUT-LOCAL frame index")
    ap.add_argument("--cameras", default="Cam2012630,Cam2012861",
                    help="cameras to draw, by NAME (default overhead + side)")
    ap.add_argument("--recording-cfg", default=DEF_RECORDING_CFG,
                    help="defines the CANONICAL camera order the npz arrays speak")
    ap.add_argument("--bouts-csv", default=None)
    ap.add_argument("--pad", type=int, default=140,
                    help="px of context around the two masks' bounding box")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    canonical = [str(c) for c in OmegaConf.load(args.recording_cfg).cameras]
    draw_cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    bad = [c for c in draw_cams if c not in canonical]
    if bad:
        raise SystemExit(f"cameras {bad} are not in the canonical order {canonical}")

    mask_npz = os.path.join(args.masks_dir, f"bout_{args.bout:05d}", "sam3_masks.npz")
    store = BoutMaskStore(mask_npz, canonical)
    t = int(args.frame)
    if not 0 <= t < store.T:
        raise SystemExit(f"--frame {t} outside the bout's {store.T} frames")
    abs_start, _abs_end, n_csv = resolve_bout_frames(args.session_dir, args.bout,
                                                     bouts_csv=args.bouts_csv)
    if n_csv != store.T:
        raise SystemExit(f"bouts CSV says {n_csv} frames, mask npz has {store.T}")

    # the one frame, read through the SAME synced reader the lifter used
    plan = load_plan(args.session_dir)
    frames, present = next(iter(read_window(args.session_dir, canonical, plan,
                                            abs_start + t, 1)))

    runs = [parse_run(s) for s in args.compare] + [parse_run(args.run)]
    kp = {label: load_kp2d(path, args.bout, draw_cams) for label, path in runs}

    # One crop box per camera, from BOTH masks plus `--pad`, computed once so
    # every row shows the SAME pixels (a per-row box would let the two runs be
    # compared at different zooms) and so the figure can be sized to the boxes
    # instead of padded with whitespace.
    outlines, boxes = {}, {}
    for cam in draw_cams:
        cam_i = canonical.index(cam)                  # BY NAME, never by integer
        xs, ys = [], []
        for fly in (0, 1):
            o = mask_outline(store, fly, cam_i, t)
            outlines[(cam, fly)] = o
            if o is not None:
                xs += [o[:, 0].min(), o[:, 0].max()]
                ys += [o[:, 1].min(), o[:, 1].max()]
        if xs:
            boxes[cam] = (max(min(xs) - args.pad, 0), min(max(xs) + args.pad,
                                                          frames.shape[2]),
                          max(min(ys) - args.pad, 0), min(max(ys) + args.pad,
                                                          frames.shape[1]))
        else:
            boxes[cam] = (0, frames.shape[2], 0, frames.shape[1])

    nrow, ncol = len(runs), len(draw_cams)
    widths = [boxes[c][1] - boxes[c][0] for c in draw_cams]
    heights = [boxes[c][3] - boxes[c][2] for c in draw_cams]
    panel_h = 4.0
    fig_w = sum(panel_h * w / max(h, 1) for w, h in zip(widths, heights)) + 0.6
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(fig_w, panel_h * nrow + 1.5),
                             gridspec_kw={"width_ratios":
                                          [w / max(h, 1) for w, h in zip(widths, heights)]})
    for ri, (label, path) in enumerate(runs):
        for ci, cam in enumerate(draw_cams):
            ax = axes[ri][ci]
            cam_i = canonical.index(cam)
            ax.imshow(frames[cam_i])
            for fly in (0, 1):
                o = outlines[(cam, fly)]
                if o is not None:
                    ax.plot(o[:, 0], o[:, 1], color=FLY_COLOR[fly], lw=1.6, alpha=0.95,
                            label=f"mask {FLY_NAME[fly]}" if ci == 0 else None)
                k = kp.get(label)
                if k is None:
                    continue
                pts = k[fly][t, ci]                   # (K,2)
                fin = np.isfinite(pts).all(axis=-1)
                if fin.any():
                    ax.scatter(pts[fin, 0], pts[fin, 1], s=15,
                               facecolor=FLY_COLOR[fly], edgecolor="k", linewidths=0.3,
                               zorder=3,
                               label=f"written {FLY_NAME[fly]}" if ci == 0 else None)
                elif ci == 0:
                    ax.plot([], [], ls="none", marker="x", color=FLY_COLOR[fly],
                            label=f"written {FLY_NAME[fly]}: NaN this frame")
            x0, x1, y0, y1 = boxes[cam]
            ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
            ax.set_xticks([]); ax.set_yticks([])
            title = f"{label} -- {cam}{'' if present[cam_i] else ' (camera dropped)'}"
            if ci == 0:
                # the run's own numbers go in the TITLE, not under the panel: a
                # caption below the axes lands on the next row's image
                title += "\n" + meta_line(path, args.bout)
                ax.legend(loc="lower left", fontsize=7, framealpha=0.8)
            ax.set_title(title, fontsize=9)
    fig.suptitle(
        f"mvq identity check -- bout {args.bout}, bout-local frame {t} "
        f"(abs {abs_start + t})\n"
        f"EXPECT cyan written keypoints inside the cyan (human-reviewed FEMALE) mask\n"
        f"outline and orange inside the orange (MALE) one, in every camera.\n"
        f"Dots on the other outline = identity swap; missing dots = that fly was NaN.",
        fontsize=10)
    fig.tight_layout(rect=(0, 0.0, 1, 0.90))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}", flush=True)
    for label, path in runs:
        print(f"  {label}: {meta_line(path, args.bout)}", flush=True)


if __name__ == "__main__":
    main()
