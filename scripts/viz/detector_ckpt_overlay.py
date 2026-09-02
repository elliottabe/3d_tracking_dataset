#!/usr/bin/env python3
"""Head-to-head 2D keypoint overlay for TWO ViTPose checkpoints on the hard
cases of the red_data_3d_v5_valfix val split.

Consumes two .npz files from ``scripts/analysis/mask_channel_eval.py`` (each
already holds pred/gt/vis per annotation plus bbox + file_name, so no GPU is
needed here), reconstructs each annotation's exact 448x448 crop from the
on-disk image using the same ``crop_origin`` the loader used, and draws
ground truth against BOTH models' predictions side by side.

WHY THIS EXISTS separately from ``scripts/analysis/mask_channel_overlay.py``:
that script compares two mask CONDITIONS of ONE checkpoint. This one compares
two CHECKPOINTS, each under whichever condition is in-distribution for it --
which for a ``train.mask_ablation=true`` checkpoint is the ZEROED condition
and for a normally-trained one is POPULATED. Overlaying the wrong condition
for an arm is silently out-of-distribution (no error, just worse keypoints),
so the condition is named per arm on the command line and printed into every
panel title.

HARD-CASE SELECTION (``--case``). A grid of easy mid-bout frames overstates
any model, so the default cases are the ones this pipeline actually fails on:

  ``close``      the two-fly frames of 2026_04_07_11_33_33 ranked by SMALLEST
                 inter-fly centre distance in body lengths. This recording is
                 the whole reason valfix exists (it was TRAIN in
                 red_data_3d_v5) and it is the ONLY val recording that carries
                 both flies in the same image -- 181 of 1690 val images hold 2
                 annotations, all of them here -- so it is the only genuine
                 per-frame close-interaction signal in this split. Every other
                 val recording crops one fly per image.
  ``occlusion``  the two-fly frames ranked by LARGEST bbox overlap (IoU)
                 between the two flies -- i.e. genuine mutual occlusion, one
                 animal's body over the other's. This does NOT use GT
                 visibility: measured on this split, 1864 of 1871 val
                 annotations carry all 50 keypoints marked visible (the rest
                 carry 49), so the annotators labelled straight through
                 occlusion and the visibility flag has no occlusion signal at
                 all. Ranking by it selected fully-visible flies and would
                 have been quoted as an occlusion test that tested nothing.
  ``wall``       bbox centre closest to the image border, female only -- the
                 arena-wall poses that are out of distribution for a detector
                 trained mostly on centred flies.
  ``worst``      largest per-annotation MPJPE under the FIRST arm, female
                 only. The frames the incumbent gets most wrong.
  ``median``     the median-error female annotation, as the "is it fine
                 normally?" control against which the hard cases are read.

STATED EXPECTATION (write it down before generating, per CLAUDE.md -- a
figure you cannot be wrong about proves nothing):

  * If the two checkpoints are genuinely equivalent, the cyan (arm A) and
    orange (arm B) skeletons should sit on top of each other AND on the white
    GT on every panel, including the close-interaction and wall panels. The
    per-panel MPJPE printed in each title should agree to within ~1px.
  * If the mask channel is what carries the female through occlusion, then on
    the ``close`` and ``occlusion`` panels the MASK-OFF arm's legs should
    detach from the fly and drift -- most likely onto the OTHER fly, since the
    mask channel is the only per-pixel cue identifying which of two
    overlapping flies is the target. Whole-skeleton displacement onto the
    neighbouring fly, not a few jittery tarsi, is the signature to look for.
  * If instead the mask channel was never doing that work, both arms track the
    same fly and differ only in leg-tip scatter of a few px.
  * The ``median`` control must look good for BOTH arms. If it does not, the
    problem is not the mask channel and no comparison of the hard cases means
    anything.

A model that wins only on ``median`` and loses on ``close``/``wall`` is a
worse model for this pipeline than the numbers suggest, and that is exactly
what this figure has to be able to show.

Usage:
    python scripts/viz/detector_ckpt_overlay.py \\
        --arm maskoff=figures/2026-09-02-vitpose-maskoff-ab/v5vf_maskoff.npz:zeroed \\
        --arm maskon=figures/2026-09-02-vitpose-maskoff-ab/v5vf_maskon.npz:populated \\
        --data-root /gscratch/.../red_data/red_data_3d_v5_valfix \\
        --case close --case occlusion --case wall --case worst --case median \\
        --n 3 --out-dir figures/2026-09-02-vitpose-maskoff-ab
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"
for p in (str(PROJECT_DIR), str(PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# The two-fly recording -- resolved by NAME, and asserted to exist in the
# loaded metadata before it is used, never assumed.
TWO_FLY_RECORDING = "2026_04_07_11_33_33"

# matplotlib RGB (viz/core/colors.py's PALETTE is BGR/cv2; these are the same
# semantic colours: white = observed/GT, cyan = detector arm A, orange = arm B)
C_GT = "#ffffff"
C_A = "#00e5ff"
C_B = "#ff9a00"


def load_arm(spec):
    """'label=path.npz:condition' -> dict with everything the figure needs.

    The condition is REQUIRED in the spec (no default): picking the wrong
    condition for a checkpoint is silently out-of-distribution, so it must be
    an explicit statement by the caller, and it is echoed into every title.
    """
    import numpy as np
    label, _, rest = spec.partition("=")
    path, _, cond = rest.partition(":")
    if not (label and path and cond):
        raise SystemExit(f"--arm {spec!r} must be 'label=path.npz:condition'")
    z = np.load(path, allow_pickle=True)
    if f"pred_{cond}" not in z.files:
        have = sorted(k.split("_", 1)[1] for k in z.files if k.startswith("pred_"))
        raise SystemExit(f"--arm {spec!r}: condition {cond!r} not in {path} "
                         f"(has {have})")
    pred, gt, vis = z[f"pred_{cond}"], z[f"gt_{cond}"], z[f"vis_{cond}"]
    return {
        "label": label, "path": path, "cond": cond,
        "pred": pred, "gt": gt, "vis": vis,
        "err": np.linalg.norm(pred - gt, axis=-1),
        "kp_names": [str(x) for x in z["kp_names"]],
        "file_name": np.asarray([str(x) for x in z["file_name"]]),
        "recording": np.asarray([str(x) for x in z["recording"]]),
        "sex": np.asarray([str(x) for x in z["sex"]]),
        "ann_id": np.asarray(z["ann_id"]),
        "bbox": np.asarray(z["bbox"]),
        "img_wh": np.asarray(z["img_wh"]),
        "kp_order_verified": bool(z["kp_order_verified"])
        if "kp_order_verified" in z.files else None,
    }


def per_ann_mpjpe(err, vis):
    import numpy as np
    v = vis.astype(np.float64)
    denom = v.sum(axis=1)
    return np.where(denom > 0, (err * v).sum(axis=1) / np.maximum(denom, 1),
                    np.nan)


def body_length_px(gt, vis, kp_names):
    """Antenna_Base -> Abd_tip distance per annotation, in crop px. Resolved BY
    NAME. This is the physical scale the close-interaction distance is
    normalised by; it is also a rigid-ish invariant, so a wild value flags a
    bad annotation rather than a close frame."""
    import numpy as np
    ia = kp_names.index("Antenna_Base")
    ib = kp_names.index("Abd_tip")
    d = np.linalg.norm(gt[:, ia] - gt[:, ib], axis=-1)
    ok = vis[:, ia] & vis[:, ib]
    med = float(np.median(d[ok])) if ok.any() else 1.0
    return np.where(ok, d, med), med


def select_cases(arm, case, n):
    """Return a list of (row_index, caption) for one case name, chosen from the
    FIRST arm's metadata (both arms share the dataset order by construction --
    same root, same split, same index order out of mask_channel_eval)."""
    import numpy as np
    rec = arm["recording"]
    sex = arm["sex"]
    gt, vis, err = arm["gt"], arm["vis"], arm["err"]
    kp_names = arm["kp_names"]
    ann_mp = per_ann_mpjpe(err, vis)
    female = sex == "female"
    bl, med_bl = body_length_px(gt, vis, kp_names)

    if case == "close":
        sel = rec == TWO_FLY_RECORDING
        if not sel.any():
            raise SystemExit(
                f"case 'close': recording {TWO_FLY_RECORDING} absent from this "
                f"eval (present: {sorted(set(rec.tolist()))})")
        # group annotations by source image; the paired ones are the two-fly
        # frames. file_name is '<rec>/<cam>/Frame_N.jpg' and is shared by the
        # two annotations of one frame.
        pairs = [(v, arm["file_name"][v[0]]) for v in _two_fly_pairs(arm)]
        if not pairs:
            raise SystemExit("case 'close': no image carries 2 annotations")
        scored = []
        for (i, j), fn in pairs:
            # crop origins differ per annotation, so compare in FULL-IMAGE px
            ci = _crop_origin_of(arm, i)
            cj = _crop_origin_of(arm, j)
            gi = gt[i][vis[i]].mean(0) + np.array(ci) if vis[i].any() else None
            gj = gt[j][vis[j]].mean(0) + np.array(cj) if vis[j].any() else None
            if gi is None or gj is None:
                continue
            d_bl = float(np.linalg.norm(gi - gj) / max(bl[i], 1e-6))
            # show the FEMALE of the pair (the hard fly) when sexes differ
            pick = i if sex[i] == "female" else j
            scored.append((d_bl, pick, fn))
        scored.sort()
        return [(idx, f"close-interaction  {d:.2f} body-lengths apart")
                for d, idx, _fn in scored[:n]]

    if case == "occlusion":
        # Real mutual occlusion = the two flies' boxes overlapping. GT
        # visibility is useless here (see module docstring), so refuse to fall
        # back to it silently.
        nvis_levels = len(set(vis.sum(axis=1).tolist()))
        pairs = _two_fly_pairs(arm)
        if not pairs:
            raise SystemExit(
                "case 'occlusion': no image carries 2 annotations, and GT "
                f"visibility has only {nvis_levels} distinct level(s) across "
                "the split -- there is no occlusion signal to rank by. "
                "Refusing to emit a figure that would be read as an "
                "occlusion test.")
        scored = []
        for (i, j) in pairs:
            iou = _bbox_iou(arm["bbox"][i], arm["bbox"][j])
            pick = i if sex[i] == "female" else j
            scored.append((-iou, pick, iou))
        scored.sort()
        return [(idx, f"mutual occlusion  the two flies' bboxes overlap "
                      f"IoU={iou:.2f}") for _neg, idx, iou in scored[:n]]

    if case == "wall":
        cand = np.nonzero(female)[0]
        d_edge = []
        for i in cand:
            w, h = arm["img_wh"][i]
            x, y, bw, bh = arm["bbox"][i]
            cx, cy = x + bw / 2, y + bh / 2
            d_edge.append(min(cx, cy, w - cx, h - cy))
        cand = cand[np.argsort(d_edge)]
        return [(int(i), f"near arena wall  bbox centre "
                         f"{min(arm['bbox'][i][0] + arm['bbox'][i][2] / 2, arm['bbox'][i][1] + arm['bbox'][i][3] / 2):.0f}px "
                         f"from frame edge") for i in cand[:n]]

    if case == "worst":
        cand = np.nonzero(female & np.isfinite(ann_mp))[0]
        cand = cand[np.argsort(-ann_mp[cand])]
        return [(int(i), f"worst for {arm['label']}  {ann_mp[i]:.1f}px MPJPE")
                for i in cand[:n]]

    if case == "median":
        cand = np.nonzero(female & np.isfinite(ann_mp))[0]
        order = cand[np.argsort(ann_mp[cand])]
        mid = len(order) // 2
        lo = max(0, mid - n // 2)
        return [(int(i), f"MEDIAN female control  {ann_mp[i]:.1f}px MPJPE")
                for i in order[lo:lo + n]]

    raise SystemExit(f"unknown --case {case!r}")


def _two_fly_pairs(arm):
    """[(i, j), ...] for every source image carrying exactly 2 annotations."""
    by_img = {}
    for i, fn in enumerate(arm["file_name"]):
        by_img.setdefault(fn, []).append(i)
    return [tuple(v) for v in by_img.values() if len(v) == 2]


def _bbox_iou(a, b):
    """IoU of two COCO [x, y, w, h] boxes."""
    ax0, ay0, aw, ah = a; bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _crop_origin_of(arm, i, crop=448):
    from jarvis_jax.data.transforms import crop_origin
    w, h = arm["img_wh"][i]
    return crop_origin(arm["bbox"][i], int(w), int(h), crop)


def reconstruct_crop(data_root, arm, i, crop=448):
    """The exact 448x448 RGB window the dataset fed the model, plus its origin."""
    import numpy as np
    from PIL import Image
    x0, y0 = _crop_origin_of(arm, i, crop)
    with Image.open(os.path.join(data_root, "images", arm["file_name"][i])) as pil:
        img = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    out = np.zeros((crop, crop, 3), np.uint8)
    sub = img[y0:y0 + crop, x0:x0 + crop]
    out[:sub.shape[0], :sub.shape[1]] = sub
    return out, (x0, y0)


def draw_skeleton(ax, xy, vis, kp_names, color, label, lw=1.1, ms=9):
    """Leg chains + wing veins + body axis, resolved BY NAME via
    viz/core/colors.py::leg_chains. Never indexes a keypoint by a literal."""
    import numpy as np
    from viz.core.colors import leg_chains
    idx = {n: k for k, n in enumerate(kp_names)}
    first = True
    for _leg, chain in leg_chains(kp_names).items():
        pts = [(xy[k], vis[k]) for k in chain]
        xs = [p[0][0] if p[1] else np.nan for p in pts]
        ys = [p[0][1] if p[1] else np.nan for p in pts]
        ax.plot(xs, ys, "-", color=color, lw=lw, alpha=0.85,
                label=label if first else None)
        first = False
    for side in ("L", "R"):
        chain = [idx[f"Wing{side}_{s}"] for s in ("base", "V12", "V13")
                 if f"Wing{side}_{s}" in idx]
        xs = [xy[k][0] if vis[k] else np.nan for k in chain]
        ys = [xy[k][1] if vis[k] else np.nan for k in chain]
        ax.plot(xs, ys, "-", color=color, lw=lw + 0.6, alpha=0.95,
                label=label if first else None)
        first = False
    axis = [idx[n] for n in ("Antenna_Base", "Scutellum", "Abd_A4", "Abd_tip")
            if n in idx]
    xs = [xy[k][0] if vis[k] else np.nan for k in axis]
    ys = [xy[k][1] if vis[k] else np.nan for k in axis]
    ax.plot(xs, ys, "-", color=color, lw=lw + 0.9, alpha=0.95,
            label=label if first else None)
    v = vis.astype(bool)
    ax.plot(xy[v, 0], xy[v, 1], ".", color=color, ms=ms * 0.35, alpha=0.95)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="'label=path.npz:condition'; give exactly two")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--case", action="append", default=None,
                    help="close|occlusion|wall|worst|median (repeatable)")
    ap.add_argument("--n", type=int, default=3, help="rows per case")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    if len(a.arm) != 2:
        ap.error("--arm must be given exactly twice")
    cases = a.case or ["close", "occlusion", "wall", "worst", "median"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    A, B = load_arm(a.arm[0]), load_arm(a.arm[1])
    if A["kp_names"] != B["kp_names"]:
        raise SystemExit("the two arms have DIFFERENT keypoint name lists -- "
                         "overlaying them would compare different landmarks")
    if not np.array_equal(A["ann_id"], B["ann_id"]):
        raise SystemExit("the two arms cover different annotations / a "
                         "different order -- row i is not the same fly")
    for arm in (A, B):
        if arm["kp_order_verified"] is False:
            raise SystemExit(
                f"arm {arm['label']}: its eval ran with the keypoint-order "
                f"guard in WARN-ONLY mode. Refusing to draw named landmarks "
                f"on an unverified keypoint axis.")
    kp_names = A["kp_names"]
    print(f"arms: {A['label']}/{A['cond']} vs {B['label']}/{B['cond']}  "
          f"({len(kp_names)} keypoints, {len(A['ann_id'])} annotations, "
          f"kp-order verified: {A['kp_order_verified']}/{B['kp_order_verified']})")

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    written = []
    for case in cases:
        rows = select_cases(A, case, a.n)
        if not rows:
            print(f"case {case}: nothing selected, skipping")
            continue
        fig, axes = plt.subplots(len(rows), 3, figsize=(13.5, 4.6 * len(rows)),
                                 squeeze=False)
        for r, (i, caption) in enumerate(rows):
            crop, (x0, y0) = reconstruct_crop(a.data_root, A, i)
            gt, vis = A["gt"][i], A["vis"][i].astype(bool)
            mp_a = per_ann_mpjpe(A["err"][i:i + 1], A["vis"][i:i + 1])[0]
            mp_b = per_ann_mpjpe(B["err"][i:i + 1], B["vis"][i:i + 1])[0]
            panels = [
                ("ground truth (annotator)", [(gt, vis, C_GT, "GT")]),
                (f"{A['label']} / {A['cond']}   MPJPE {mp_a:.1f}px",
                 [(gt, vis, C_GT, "GT"), (A["pred"][i], vis, C_A, A["label"])]),
                (f"{B['label']} / {B['cond']}   MPJPE {mp_b:.1f}px",
                 [(gt, vis, C_GT, "GT"), (B["pred"][i], vis, C_B, B["label"])]),
            ]
            for c, (ttl, layers) in enumerate(panels):
                ax = axes[r][c]
                ax.imshow(crop)
                for xy, v, col, lab in layers:
                    draw_skeleton(ax, xy, v, kp_names, col, lab)
                # name the three worst keypoints on each model panel -- names,
                # never indices (CLAUDE.md: bare indices are how the
                # keypoint-order bug stayed invisible)
                if c > 0:
                    e = (A if c == 1 else B)["err"][i].copy()
                    e[~vis] = -1
                    for k in np.argsort(-e)[:3]:
                        if e[k] <= 0:
                            continue
                        p = (A if c == 1 else B)["pred"][i][k]
                        ax.annotate(f"{kp_names[k]} {e[k]:.0f}px",
                                    xy=(p[0], p[1]), fontsize=6,
                                    color="#ffe14d",
                                    xytext=(6, 6), textcoords="offset points")
                ax.set_title(ttl, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(
                        f"{A['recording'][i]}\n{Path(A['file_name'][i]).parent.name}"
                        f" / {Path(A['file_name'][i]).stem}\n"
                        f"sex={A['sex'][i]}  ann={int(A['ann_id'][i])}",
                        fontsize=7)
                if r == 0 and c == 2:
                    ax.legend(fontsize=7, loc="lower right")
            axes[r][0].text(6, 20, caption, color="#ffe14d", fontsize=8,
                            va="top")
        fig.suptitle(
            f"{case.upper()} -- {A['label']}/{A['cond']} (cyan) vs "
            f"{B['label']}/{B['cond']} (orange) vs GT (white)\n"
            f"red_data_3d_v5_valfix val, crop px, keypoints resolved by name",
            fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = Path(a.out_dir) / f"overlay_{case}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(str(path))
        print(f"wrote {path}")

    meta = Path(a.out_dir) / "overlay_manifest.json"
    meta.write_text(json.dumps(
        {"arms": [{"label": x["label"], "npz": x["path"], "cond": x["cond"],
                   "kp_order_verified": x["kp_order_verified"]} for x in (A, B)],
         "cases": cases, "n_per_case": a.n, "figures": written}, indent=2))
    print(f"wrote {meta}")


if __name__ == "__main__":
    main()
