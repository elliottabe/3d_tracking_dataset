#!/usr/bin/env python
"""Pre-launch gate: does the multi-view copy-paste compositor
(`jarvis_jax.data.mv_copy_paste.composite`) produce geometrically consistent,
plausible donor/host pairs on REAL v12 frames?

EXPECTATION: in every one of the 7 cameras the pasted donor (orange labels)
sits at the same place relative to the host (cyan) -- never floating,
offset, or missing in one view -- its labels lie on its own body, host
keypoints under the donor are drawn hollow (occluded), and the contact rows
(sep <= 30 units -- amended 2026-09-04, see p3a-notes.md Ruling B: real
mounting pairs are 24-30 units apart) show the two bodies touching or
overlapping like a mounting pair.

Both host and donor are required to have a visible head landmark in at
least one camera (see `HEAD_NAMES`): an early run of this gate picked a
donor with no visible Antenna/Eye keypoint in any of its 7 cameras, and it
read as a compositor bug until traced back to that one window. It wasn't --
the manifest carries `behavior: "headless"` for 5 recordings (a real
experimental condition, e.g. `2026_06_09_15_38_35`, that donor's own
recording), so a headless-looking fly is a legitimate appearance and the
production loader/augmentation applies no such filter. This script keeps the
filter anyway, purely so this CHECK FIGURE demonstrates placement geometry
on an anatomically unambiguous pair rather than for any label-quality reason.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 PYTHONPATH=third_party/jarvis_jax:. \\
        python scripts/viz/mvq_copy_paste_check.py \\
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
from jarvis_jax.data.mv_copy_paste import CopyPasteParams
from jarvis_jax.train.matching import SEX_UNKNOWN
from viz.core.colors import PALETTE

N_CONTACT, N_FAR = 3, 3
SEX_NAME = {0: "female", 1: "male", -1: "unknown"}
FLY_RGB = {f: tuple(c / 255.0 for c in reversed(PALETTE[f])) for f in ("fly0", "fly1")}  # BGR->RGB
# Antenna/eye landmarks -- a fly with NONE of these visible in ANY camera is
# either a genuinely headless specimen (5 recordings are labelled
# `behavior: "headless"` in the manifest -- a real experimental condition,
# NOT a data defect) or has its head occluded/off-crop in every view for this
# one window. Either way this check figure skips it for legibility only: it
# exists to demonstrate placement geometry, and an anatomically ambiguous
# donor obscures that regardless of cause. The production loader/augmentation
# (V12WindowDataset.paste_window / mv_copy_paste.composite) has NO such
# filter and is not meant to.
HEAD_NAMES = ("Antenna_Base", "EyeL", "EyeR")


def _has_head(vis2d_fly, head_idx):
    return bool(np.asarray(vis2d_fly)[:, head_idx].any())


def _try_paste(ds, i, rng, head_idx):
    """One `paste_window` draw -> (row_or_None, reason). reason is 'ok',
    'rejected' (composite/paste_window itself returned None), or 'headless'
    (host or donor has no visible head landmark in any camera)."""
    r = ds.paste_window(int(i), rng)
    if r is None:
        return None, "rejected"
    sample, info = r
    if not (_has_head(sample["vis2d"][0, 0], head_idx) and _has_head(sample["vis2d"][1, 0], head_idx)):
        return None, "headless"
    row = dict(i=int(i), sample=sample, info=info,
              host_sex=int(sample["fly_sex"][0]), donor_sex=int(sample["fly_sex"][1]))
    return row, "ok"


def _collect(ds, ds_ref, seed=0, max_pool=800):
    """Draw random single-fly windows and call `ds.paste_window` until 3
    contact (sep<=30u, `CopyPasteParams.contact_sep[1]`) and 3 far pastes are
    collected, at least one same-sex, both flies with a visible head
    landmark (see `HEAD_NAMES`). `ds_ref` is a copy-paste-free
    twin (same seed/jitter) used to recover the HOST's pre-paste `vis2d`, so
    keypoints the donor occludes can be told apart from keypoints that were
    never visible."""
    head_idx = [ds.keypoint_names.index(n) for n in HEAD_NAMES]
    cand = [i for i in range(len(ds)) if ds.n_flies(i) == 1 and ds.unlabelled_sex(i) == SEX_UNKNOWN]
    order = np.random.default_rng(seed).permutation(cand)[:max_pool]
    rng = np.random.default_rng(seed + 1)
    contact, far, counts = [], [], {"tried": 0, "rejected": 0, "headless": 0}
    idx_used = 0
    for i in order:
        if len(contact) >= N_CONTACT and len(far) >= N_FAR:
            break
        idx_used += 1
        counts["tried"] += 1
        row, reason = _try_paste(ds, i, rng, head_idx)
        if row is None:
            counts[reason] += 1; continue
        row["vis_pre"] = ds_ref[int(i)]["vis2d"][0, 0]
        (contact if row["info"]["contact"] else far).append(row)
    same_sex = any(r["host_sex"] == r["donor_sex"] and r["host_sex"] in (0, 1) for r in contact + far)
    if not same_sex:
        # keep collecting (same pool, same rng state) until one same-sex pair
        # turns up, swapping it in for the LAST row of its own category so
        # counts stay 3+3 -- report if the whole pool runs out first.
        for i in order[idx_used:]:
            counts["tried"] += 1
            row, reason = _try_paste(ds, i, rng, head_idx)
            if row is None:
                counts[reason] += 1; continue
            if row["host_sex"] == row["donor_sex"] and row["host_sex"] in (0, 1):
                row["vis_pre"] = ds_ref[int(i)]["vis2d"][0, 0]
                bucket = contact if row["info"]["contact"] else far
                if bucket:
                    bucket[-1] = row
                same_sex = True
                break
    rows = contact[:N_CONTACT] + far[:N_FAR]
    return rows, dict(n_tried=counts["tried"], n_none=counts["rejected"], n_headless=counts["headless"],
                      n_pool=len(cand), same_sex_found=same_sex)


def _draw_row(axes_row, row, cam_names):
    s, info = row["sample"], row["info"]
    C = s["crops"].shape[1]
    for c in range(len(axes_row)):
        ax = axes_row[c]
        ax.set_xticks([]); ax.set_yticks([])
        if c >= C:
            ax.axis("off"); continue
        ax.imshow(s["crops"][0, c])
        if not s["cam_valid"][0, c]:
            ax.set_title(f"{cam_names[c]} absent", fontsize=6.5); continue
        vis_now = s["vis2d"][0, 0, c]
        vis_pre = row["vis_pre"][c]
        occluded = vis_pre & ~vis_now
        pts = s["kp2d"][0, 0, c]
        if vis_now.any():
            p = pts[vis_now]
            ax.scatter(p[:, 0], p[:, 1], s=7, c=[FLY_RGB["fly0"]], label="host fly0")
        if occluded.any():
            p = pts[occluded]
            ax.scatter(p[:, 0], p[:, 1], s=16, facecolors="none", edgecolors=[FLY_RGB["fly0"]],
                      linewidths=0.8, label="host occluded")
        if s["fly_valid"][1]:
            vis_d = s["vis2d"][1, 0, c]
            if vis_d.any():
                pd = s["kp2d"][1, 0, c][vis_d]
                ax.scatter(pd[:, 0], pd[:, 1], s=7, c=[FLY_RGB["fly1"]], label="donor fly1")
        ax.set_title(cam_names[c], fontsize=6.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ds = V12WindowDataset(a.root, "train", T=1, train=True,
                         copy_paste=CopyPasteParams(p=1.0, max_tries=30))
    ds_ref = V12WindowDataset(a.root, "train", T=1, train=True)   # copy_paste=None twin, same jitter
    rows, stats = _collect(ds, ds_ref, seed=a.seed)
    print(f"paste_window: pool={stats['n_pool']} tried={stats['n_tried']} "
         f"rejected(None)={stats['n_none']} headless_skipped={stats['n_headless']} "
         f"same_sex_found={stats['same_sex_found']} "
         f"collected contact={sum(r['info']['contact'] for r in rows)} "
         f"far={sum(not r['info']['contact'] for r in rows)}")
    if len(rows) < N_CONTACT + N_FAR:
        print(f"WARNING: only collected {len(rows)}/{N_CONTACT + N_FAR} pastes "
             f"(pool={stats['n_pool']}, tried={stats['n_tried']}, none={stats['n_none']})")

    C = max((r["sample"]["crops"].shape[1] for r in rows), default=7)
    fig, axes = plt.subplots(len(rows), C, figsize=(2.1 * C, 2.2 * len(rows)), squeeze=False)
    for r_i, row in enumerate(rows):
        rec = ds.windows[row["i"]][0]
        cam_names = ds.camera_names(row["i"])
        _draw_row(axes[r_i], row, cam_names)
        info = row["info"]
        tag = "CONTACT" if info["contact"] else "far"
        axes[r_i, 0].set_ylabel(
            f"#{row['i']} {rec}\nhost={SEX_NAME[row['host_sex']]} donor={SEX_NAME[row['donor_sex']]}\n"
            f"sep={info['sep']:.1f}u {tag}", fontsize=6.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for r_i in range(len(rows)):
            handles, labels = axes[r_i, 0].get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, fontsize=7, loc="upper left",
              bbox_to_anchor=(0.0, 1.0), bbox_transform=fig.transFigure)
    fig.suptitle("mvq copy-paste gate  --  cyan=host fly0 (hollow=occluded by donor)  orange=donor fly1",
                fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0, w_pad=0.6)
    png = os.path.join(a.out, "copy_paste_check.png")
    fig.savefig(png, dpi=130); plt.close(fig)

    dump = [dict(i=r["i"], rec=ds.windows[r["i"]][0], donor=r["info"]["donor"],
                D=[float(x) for x in r["info"]["D"]], sep=r["info"]["sep"],
                contact=bool(r["info"]["contact"]), host_sex=SEX_NAME[r["host_sex"]],
                donor_sex=SEX_NAME[r["donor_sex"]], cam_names=ds.camera_names(r["i"]))
           for r in rows]
    json.dump(dict(stats=stats, rows=dump), open(os.path.join(a.out, "copy_paste_check.json"), "w"), indent=1)
    print("wrote", png)


if __name__ == "__main__":
    main()
