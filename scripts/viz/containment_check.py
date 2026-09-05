#!/usr/bin/env python
"""Before/after acceptance check for the per-keypoint MASK-CONTAINMENT filter
(`jarvis_jax.tracking.lift_mvq.mask_containment_filter`).

Applies the filter OFFLINE to an EXISTING mvq lift -- nothing in the run dir is
touched -- recomputes the two defect metrics the filter targets, and renders the
worst frames before vs after so the drops can be seen rather than trusted.

THE METRICS (both from `scratchpad/perkp_jump_probe.py`, on the MALE = fly1):
  pose-jump  frames with >= 5 male keypoints whose own 3D step exceeds 5 units
             (0.5 mm in 1/800 s -- impossible for a real landmark)
  straddle   frames with >= 5 male keypoints nearer the FEMALE's keypoint
             centroid than his own. Blind wherever her slot is NaN, which is
             264 of the 391 pose-jump frames on 20_04 -- hence both metrics.

WHAT THE FIGURE MUST SHOW, and this is the point of running it: in the BEFORE
panel a cluster of that fly's keypoints sits inside the OTHER fly's mask
outline; in the AFTER panel those specific points are gone and the ones on its
own body are untouched, at the same pixels. If instead the filter is chasing a
bad SAM mask rather than the other fly, the after panel loses points that are
plainly on the fly's own body. Read the PNG back and check that before
believing any number in the table.

Run the female arm too (`--fly 0`). She is the fly this pipeline fails on, and
on 20_04 she has BOTH failure modes: bout 4 is a real identity leak onto the
male (correctly removed) and bout 15 is a garbage track removed by the pure
spike rule with no mask evidence at all.

    MUJOCO_GL=egl PYTHONPATH=third_party/jarvis_jax:. python \\
        scripts/viz/containment_check.py \\
        --run  <processed>/.../pose_mvq_p3a_r2 \\
        --masks <processed>/.../sam3_masks \\
        --session-dir <video>/courtship/Session0/2025_10_20_13_20_04 \\
        --bout 1 --bout 4 --fly 1 \\
        --out figures/2026-09-mvq/p3a_campaign_female_misses/containment_bout1_bout4.png

`--all-bouts` skips the render and prints the whole-recording table instead
(the "< 3% of keypoints dropped" acceptance number).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "jarvis_jax"))
sys.path.insert(0, ROOT)

from omegaconf import OmegaConf  # noqa: E402

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool  # noqa: E402
from jarvis_jax.tracking.lift_mvq import (BoutMaskStore,  # noqa: E402
                                          mask_containment_filter, project_points)
from viz.core.colors import keypoint_groups  # noqa: E402

DEF_RECORDING_CFG = os.path.join(ROOT, "configs", "recording", "session0.yaml")


def bout_indices(run_bouts):
    out = []
    for d in sorted(glob.glob(os.path.join(run_bouts, "bout_*"))):
        m = re.match(r"bout_(\d+)$", os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "fly1", "kp3d.npz")):
            out.append(int(m.group(1)))
    return out


def load_bout(run_bouts, b, cameras):
    """(kp3d (2,T,K,3), kp2d (2,T,C,K,2), conf, kp_names, frame_start).

    The npz keypoint axis is `cfg.model.KP_NAMES` (MODEL order) and its camera
    axis is the canonical one -- both asserted here, because the filter tests a
    reprojection against a mask and a camera-axis mix-up would test one
    camera's mask against another camera's point (CLAUDE.md).
    """
    d = os.path.join(run_bouts, f"bout_{b:05d}")
    kp3d, kp2d, conf, names = [], [], [], None
    for fly in (0, 1):
        with np.load(os.path.join(d, f"fly{fly}", "kp3d.npz")) as z:
            kp3d.append(np.asarray(z["kp3d"], np.float32))
            names = [str(n) for n in z["kp_names"]]
        with np.load(os.path.join(d, f"fly{fly}", "kp2d.npz")) as z:
            kp2d.append(np.asarray(z["kp2d"], np.float32))
            conf.append(np.asarray(z["conf"], np.float32))
            got = [str(c) for c in z["cameras"]]
    if got != list(cameras):
        raise RuntimeError(f"bout {b}: kp2d camera axis {got} is not the canonical order "
                           f"{list(cameras)} -- indexing it would put one camera's "
                           f"keypoints on another camera's mask")
    start = 0
    mp = os.path.join(d, "mvq_meta.json")
    if os.path.exists(mp):
        start = int(json.load(open(mp)).get("frame_start", 0))
    return np.stack(kp3d), np.stack(kp2d), np.stack(conf), names, start


def defect_metrics(kp3d, max_step=5.0):
    """(pose_jump (T,) bool, straddle (T,) bool) for the MALE (row 1)."""
    k0, k1 = kp3d[0], kp3d[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # all-NaN slices are the point
        c1, c0 = np.nanmean(k1, axis=1), np.nanmean(k0, axis=1)
    step = np.linalg.norm(np.diff(k1, axis=0), axis=-1)
    n_jump = np.r_[0, np.nansum(step > max_step, axis=1)]
    d_own = np.linalg.norm(k1 - c1[:, None], axis=-1)
    d_fem = np.linalg.norm(k1 - c0[:, None], axis=-1)
    n_str = ((d_fem < d_own) & np.isfinite(d_fem)).sum(axis=1)
    return n_jump >= 5, n_str >= 5


def run_filter(kp3d, kp2d, conf, store, cam_mats, args):
    t0 = time.time()
    out = mask_containment_filter(kp3d, kp2d, store, cam_mats,
                                  min_views=args.min_views, own_margin=args.own_margin,
                                  max_step_units=args.max_step, conf_by_fly=conf)
    out[3]["seconds"] = time.time() - t0
    return out


# --------------------------------------------------------------------- render
def _crop(masks, shape, pad=70):
    ys, xs = [], []
    for m in masks:
        y, x = np.nonzero(m)
        if y.size:
            ys.append(y)
            xs.append(x)
    if not ys:
        return 0, shape[1], 0, shape[0]
    y, x = np.concatenate(ys), np.concatenate(xs)
    return (max(x.min() - pad, 0), min(x.max() + pad, shape[1]),
            max(y.min() - pad, 0), min(y.max() + pad, shape[0]))


def render(panels, cams_shown, cam_mats, cameras, session_dir, fly, args, out_png):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jarvis_jax.predict.synced_reader import load_plan, slot_positions

    plan = load_plan(session_dir)
    ci = [list(cameras).index(c) for c in cams_shown]
    other = 1 - fly
    own_col, oth_col = ("#FFA500", "cyan") if fly == 1 else ("cyan", "0.75")
    n = len(panels)
    fig, ax = plt.subplots(n, 2 * len(cams_shown),
                           figsize=(2 * len(cams_shown) * 5.2, n * 4.0), squeeze=False)
    for r, (b, t, absf, kp3d, f3, store) in enumerate(panels):
        for k, (cname, c) in enumerate(zip(cams_shown, ci)):
            pos, _pr = slot_positions(plan, cname, absf, 1)
            cap = cv2.VideoCapture(os.path.join(session_dir, f"{cname}.mp4"))
            img = None
            if pos[0] is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos[0]))
                okf, bgr = cap.read()
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if okf else None
            cap.release()
            if img is None:
                continue
            m_own = store.mask_at(fly, c, t) if store.valid_at(fly, t)[c] else None
            m_oth = store.mask_at(other, c, t) if store.valid_at(other, t)[c] else None
            uvb = project_points(cam_mats[c:c + 1], np.nan_to_num(kp3d[fly, t]))[0]
            uvb[~np.isfinite(kp3d[fly, t]).all(-1)] = np.nan
            uva = project_points(cam_mats[c:c + 1], np.nan_to_num(f3[fly, t]))[0]
            uva[~np.isfinite(f3[fly, t]).all(-1)] = np.nan
            keep = np.isfinite(uva).all(-1)
            drop = np.isfinite(uvb).all(-1) & ~keep
            box = _crop([m for m in (m_own, m_oth) if m is not None], img.shape)
            for a_i, (arm, uv, mk) in enumerate((("before", uvb, np.isfinite(uvb).all(-1)),
                                                 ("after", uva, keep))):
                A = ax[r][2 * k + a_i]
                A.imshow(img)
                for m, col in ((m_own, own_col), (m_oth, oth_col)):
                    if m is not None:
                        A.contour(m.astype(float), levels=[0.5], colors=[col],
                                  linewidths=1.2)
                A.scatter(uv[mk, 0], uv[mk, 1], s=26, facecolors="none",
                          edgecolors=own_col, linewidths=1.4,
                          label=f"fly{fly} keypoint")
                if arm == "before" and drop.any():
                    A.scatter(uvb[drop, 0], uvb[drop, 1], s=44, c="red", marker="x",
                              linewidths=1.6, label="dropped")
                A.set_xlim(box[0], box[1])
                A.set_ylim(box[3], box[2])
                A.set_xticks([])
                A.set_yticks([])
                A.set_title(f"bout {b} f{t} (abs {absf})  {cname}  {arm}\n"
                            f"fly{fly} kp shown {int(mk.sum())}"
                            + (f"  dropped {int(drop.sum())}" if arm == "before" else ""),
                            fontsize=9)
                if r == 0 and k == 0 and a_i == 0:
                    A.legend(loc="upper right", fontsize=7, framealpha=0.8)
    fig.suptitle(
        f"mask-containment filter -- fly{fly} "
        f"({'male' if fly == 1 else 'female'}) keypoints, min_views={args.min_views}, "
        f"own dilation={args.own_margin:.0f} px, max_step={args.max_step:.0f} units "
        f"(0.5 mm/frame)\nred X = dropped; {own_col} outline = its own SAM mask, "
        f"{oth_col} = the other fly's. Expectation: the X's sit inside the OTHER "
        f"outline (or off both bodies, for a pure spike) and nothing on its own body moves.",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=105)
    print(f"wrote {out_png}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="pipeline run root (has bouts/)")
    p.add_argument("--masks", required=True, help="SAM3 mask tree (has bout_*/sam3_masks.npz)")
    p.add_argument("--session-dir", default=None,
                   help="recording dir (mp4s + calibration); required unless --no-render")
    p.add_argument("--bout", type=int, action="append", default=[])
    p.add_argument("--all-bouts", action="store_true",
                   help="every bout in the run; implies --no-render (the table only)")
    p.add_argument("--fly", type=int, default=1, choices=(0, 1),
                   help="which fly to render (1 = male, the default; 0 = the female, "
                        "the fly this pipeline fails on -- run both)")
    p.add_argument("--frames", type=int, default=3, help="frames rendered per bout")
    p.add_argument("--cameras-shown", default="Cam2012630,Cam2012855",
                   help="overhead + side, by NAME")
    p.add_argument("--min-views", type=int, default=3)
    p.add_argument("--own-margin", type=float, default=6.0)
    p.add_argument("--max-step", type=float, default=5.0)
    p.add_argument("--recording-cfg", default=DEF_RECORDING_CFG)
    p.add_argument("--calib-dir", default=None)
    p.add_argument("--out", default=None, help="figure path (.png)")
    p.add_argument("--save-npz", default=None,
                   help="directory for the FILTERED arrays (never the run dir)")
    p.add_argument("--json", default=None, help="write the before/after table here")
    p.add_argument("--no-render", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cameras = [str(c) for c in OmegaConf.load(args.recording_cfg).cameras]
    run_bouts = os.path.join(args.run, "bouts")
    bouts = bout_indices(run_bouts) if args.all_bouts else sorted(set(args.bout))
    if not bouts:
        raise SystemExit("give --bout <id> (repeatable) or --all-bouts")
    render_it = not (args.no_render or args.all_bouts)
    if render_it and not (args.session_dir and args.out):
        raise SystemExit("rendering needs --session-dir and --out (or pass --no-render)")
    calib = args.calib_dir or os.path.join(args.session_dir or args.run, "calibration")
    rt = ReprojectionTool(calib)
    if [str(k) for k in rt.cameras.keys()] != cameras:
        raise RuntimeError(f"calibration order {[str(k) for k in rt.cameras.keys()]} != "
                           f"the recording config's canonical order {cameras}")
    cam_mats = np.asarray(rt.camera_matrices, np.float32)

    rows, panels = [], []
    tot = dict(T=0, jb=0, ja=0, sb=0, sa=0, sec=0.0)
    fin_t, drop_t, grp_t = np.zeros(2), np.zeros(2), {}
    for b in bouts:
        kp3d, kp2d, conf, names, start = load_bout(run_bouts, b, cameras)
        store = BoutMaskStore(os.path.join(args.masks, f"bout_{b:05d}", "sam3_masks.npz"),
                              cameras)
        if store.T != kp3d.shape[1]:
            raise RuntimeError(f"bout {b}: masks have {store.T} frames, kp3d has "
                               f"{kp3d.shape[1]}")
        f3, f2, fc, rep = run_filter(kp3d, kp2d, conf, store, cam_mats, args)
        jb, sb = defect_metrics(kp3d, args.max_step)
        ja, sa = defect_metrics(f3, args.max_step)
        fin = np.isfinite(kp3d).all(-1)
        drop = fin & ~np.isfinite(f3).all(-1)
        T = kp3d.shape[1]
        tot["T"] += T
        tot["jb"] += int(jb.sum()); tot["ja"] += int(ja.sum())
        tot["sb"] += int(sb.sum()); tot["sa"] += int(sa.sum())
        tot["sec"] += rep["seconds"]
        fin_t += fin.sum(axis=(1, 2))
        drop_t += drop.sum(axis=(1, 2))
        for g, ix in keypoint_groups(names).items():
            if ix:
                a, c = grp_t.get(g, (0, 0))
                grp_t[g] = (a + int(drop[args.fly][:, ix].sum()),
                            c + int(fin[args.fly][:, ix].sum()))
        rows.append(dict(bout=b, T=T, jump_before=float(jb.mean()), jump_after=float(ja.mean()),
                         strad_before=float(sb.mean()), strad_after=float(sa.mean()),
                         drop_fly0=float(drop[0].sum() / max(fin[0].sum(), 1)),
                         drop_fly1=float(drop[1].sum() / max(fin[1].sum(), 1)),
                         n_other_mask=rep["n_kp_dropped_other_mask"],
                         n_spike=rep["n_kp_dropped_spike"],
                         n_frames_testable=rep["n_frames_testable"],
                         seconds=round(rep["seconds"], 2)))
        print(f"bout {b:3d} T={T:5d}  pose-jump {100 * jb.mean():6.2f}% -> "
              f"{100 * ja.mean():6.2f}%   straddle {100 * sb.mean():6.2f}% -> "
              f"{100 * sa.mean():6.2f}%   dropped fly0 "
              f"{100 * drop[0].sum() / max(fin[0].sum(), 1):5.2f}% fly1 "
              f"{100 * drop[1].sum() / max(fin[1].sum(), 1):5.2f}%   "
              f"other_mask {rep['n_kp_dropped_other_mask']} spike {rep['n_kp_dropped_spike']}"
              f"   {rep['seconds']:.1f}s", flush=True)
        if args.save_npz:
            os.makedirs(args.save_npz, exist_ok=True)
            np.savez_compressed(os.path.join(args.save_npz, f"bout_{b:05d}_filtered.npz"),
                                kp3d=f3, kp2d=f2, conf=fc, kp_names=np.array(names),
                                cameras=np.array(cameras))
        if render_it:
            nd = (np.asarray(rep["per_frame"]["n_other_mask"])[args.fly]
                  + np.asarray(rep["per_frame"]["n_spike"])[args.fly])
            picked = []
            for t in np.argsort(-nd):
                if nd[t] == 0 or len(picked) == args.frames:
                    break
                if all(abs(int(t) - q) > 5 for q in picked):     # not the same event twice
                    picked.append(int(t))
            panels += [(b, t, start + t, kp3d, f3, store) for t in picked]

    T = max(tot["T"], 1)
    summary = dict(
        bouts=rows, n_frames=int(tot["T"]),
        pose_jump_before=tot["jb"] / T, pose_jump_after=tot["ja"] / T,
        straddle_before=tot["sb"] / T, straddle_after=tot["sa"] / T,
        frac_dropped={f"fly{f}": float(drop_t[f] / max(fin_t[f], 1)) for f in (0, 1)},
        by_group={g: dict(dropped=a, finite=c, frac=a / max(c, 1)) for g, (a, c) in grp_t.items()},
        seconds=round(tot["sec"], 2),
        thresholds=dict(min_views=args.min_views, own_margin_px=args.own_margin,
                        max_step_units=args.max_step))
    print(f"\nTOTAL {len(bouts)} bout(s), {tot['T']} frames, {tot['sec']:.1f}s of filtering")
    print(f"  pose-jump frames {100 * tot['jb'] / T:.2f}% -> {100 * tot['ja'] / T:.2f}%")
    print(f"  straddle  frames {100 * tot['sb'] / T:.2f}% -> {100 * tot['sa'] / T:.2f}%")
    print(f"  keypoints dropped: fly1 (male) {100 * drop_t[1] / max(fin_t[1], 1):.2f}%, "
          f"fly0 (female) {100 * drop_t[0] / max(fin_t[0], 1):.2f}%")
    print("  fly%d by group: " % args.fly
          + "  ".join(f"{g} {100 * a / max(c, 1):.2f}% ({a}/{c})" for g, (a, c) in grp_t.items()))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1, default=float)
        print(f"wrote {args.json}")
    if render_it and panels:
        render(panels, [c.strip() for c in args.cameras_shown.split(",") if c.strip()],
               cam_mats, cameras, args.session_dir, args.fly, args, args.out)
    elif render_it:
        print("nothing dropped in these bouts -- no figure written")


if __name__ == "__main__":
    main()
