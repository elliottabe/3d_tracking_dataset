"""Adjudication kit for SAME-FLY label pairs that disagree on identical pixels.

Six groups of recording names in red_data are the same footage ingested twice
(jarvis_jax.data.content_index). On a duplicated frame the two ingests can
carry TWO KINDS of extra annotation:

  * the SAME fly labelled twice -- median 0.00 px apart, but with a tail: mean
    1.3-1.5 px, max ~10 px, concentrated on midline landmarks and distal
    tarsi. These are copied labels with a subset re-fit, not an independent
    pass, and exactly one of the two is right.
  * a SECOND fly that exists on only one side. That is real data, never a
    duplicate, and is never reported here.

This writes what a human needs to rule on the first kind, and nothing else:
two CSVs (pair-level and keypoint-level) plus one figure per worst-N pair.

MATCHING IS SPATIAL, NEVER BY EQUALITY. Only 117 of 586 duplicated val
annotations are keypoint-identical to their train twin; a "same keypoints?"
test therefore calls the other 469 distinct and hides them.

Nor is it bbox IoU. Three of the six alias groups are the same courtship
footage labelled once for the MALE and once for the FEMALE
(`courtship_25_51_male` / `_female` -> 2026_06_18_19_23_03 /
2026_06_19_11_09_36), and a courting pair overlaps: IoU 0.35-0.64 between the
two DIFFERENT flies. An IoU matcher pairs them and reports a 325 px
"disagreement" that is really two animals -- measured, and the reason this
matcher is not IoU.

Annotations are paired by MEDIAN keypoint displacement, which separates the
two cases with a gap nothing lands in: over all 4,625 candidate matches the
best-match median is <1 px for 77% and <5 px for 84% (a copied label), jumps
to 186 px at p85, and the SECOND-best match is >=212 px at p5. Anything under
--same-fly-median-px is one fly labelled twice; anything over it is the other
fly and is dropped, not reported. A disagreeing `sex` on the two annotations
is an independent veto and any case where the two criteria disagree is
printed rather than silently resolved.

WHAT THE FIGURE MUST SHOW (state it before looking, then look).
Median disagreement is 0.00 px, so a plain two-skeleton overlay renders two
identical pictures and proves nothing. Each figure is therefore a wide crop
for context PLUS one high-magnification inset per disagreeing keypoint. If it
is correct, every inset shows TWO clearly separated markers, a few px apart,
over recognisable anatomy -- an antenna base, the scutellum, a tarsal tip --
so the reader can say which marker is on the structure. If an inset shows one
blob, the zoom is too coarse and the figure has failed its only purpose;
raise --inset-px rather than shipping it.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "jarvis_jax"))
sys.path.insert(0, _ROOT)

RED = "/gscratch/portia/eabe/data/Johnson_lab/red_data"


def corpus_of(real_path: str) -> str:
    """`red_data_unified_V3/<split>` or `general_model/<subset>/<split>` -- the
    thing the reader is actually choosing between."""
    rel = real_path.replace("/mmfs1", "")[len(RED) + 1:].split("/")
    return f"{rel[0]}/{rel[1]}" if rel[0] == "red_data_unified_V3" \
        else f"{rel[0]}/{rel[1]}/{rel[2]}"


def bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def kp_array(ann) -> np.ndarray:
    return np.asarray(ann["keypoints"], dtype=np.float64).reshape(-1, 3)


def median_disp(a, b) -> float:
    """Median px displacement over keypoints visible in BOTH. The same fly
    labelled twice reads ~0; the other fly reads hundreds."""
    ka, kb = kp_array(a), kp_array(b)
    m = (ka[:, 2] > 0) & (kb[:, 2] > 0)
    if not m.any():
        return float("inf")
    return float(np.median(np.linalg.norm(ka[m, :2] - kb[m, :2], axis=1)))


def collect_pairs(inst, image_hash, real_of, split_of, same_fly_px=20.0):
    """Every (annotation_a, annotation_b) that sit on byte-identical pixels,
    come from DIFFERENT ingests, and are the SAME animal."""
    img_by_id = {i["id"]: i for i in inst["images"]}
    ann_by_img = collections.defaultdict(list)
    for a in inst["annotations"]:
        ann_by_img[a["image_id"]].append(a)
    by_content = collections.defaultdict(list)
    for iid, h in image_hash.items():
        by_content[h].append(iid)

    pairs, unpaired, sex_conflicts = [], 0, 0
    for h, iids in by_content.items():
        if len(iids) < 2:
            continue
        iids = sorted(iids)
        for ia, ib in [(iids[i], iids[j])
                       for i in range(len(iids)) for j in range(i + 1, len(iids))]:
            A, B = ann_by_img[ia], ann_by_img[ib]
            used = set()
            for a in A:
                best, bi = same_fly_px, None
                for j, b in enumerate(B):
                    if j in used:
                        continue
                    v = median_disp(a, b)
                    if v < best:
                        best, bi = v, j
                if bi is None:
                    unpaired += 1
                    continue
                b = B[bi]
                sx = {a["sex"], b["sex"]} - {"unknown"}
                if len(sx) > 1:
                    # The two criteria disagree: co-located to <20 px median
                    # yet labelled different sexes. Never resolve this
                    # silently -- one of the two labels is wrong.
                    sex_conflicts += 1
                    continue
                used.add(bi)
                pairs.append((h, img_by_id[ia], img_by_id[ib], a, b,
                              bbox_iou(a["bbox"], b["bbox"])))
            unpaired += len(B) - len(used)
    return pairs, unpaired, sex_conflicts


def pair_stats(a, b, thr):
    ka, kb = kp_array(a), kp_array(b)
    both = (ka[:, 2] > 0) & (kb[:, 2] > 0)
    d = np.full(len(ka), np.nan)
    d[both] = np.linalg.norm(ka[both, :2] - kb[both, :2], axis=1)
    fin = d[np.isfinite(d)]
    return d, fin, np.where(np.nan_to_num(d, nan=-1) > thr)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    # v5_valfix and v6_contentsplit were both deleted 2026-09-02. The current
    # root carries the content AND the split, so both default to it.
    ap.add_argument("--src", default=f"{RED}/red_data_3d_v8_gm_only")
    ap.add_argument("--split-root", default=f"{RED}/red_data_3d_v8_gm_only")
    ap.add_argument("--hash-cache", required=True)
    ap.add_argument("--csv-dir", default=os.path.join(
        _ROOT, "docs/benchmark/2026-09-02-dataset-dedup"))
    ap.add_argument("--fig-dir", default=os.path.join(
        _ROOT, "figures/2026-09-02-dataset-dedup/label_adjudication"))
    ap.add_argument("--n-figures", type=int, default=48)
    ap.add_argument("--per-pair-figures", type=int, default=6,
                    help="guaranteed figures per recording-alias pair before "
                         "the rest of the budget goes worst-first")
    ap.add_argument("--diff-threshold", type=float, default=0.5,
                    help="px above which a keypoint counts as disagreeing. "
                         "Labels are stored to 3 dp and copied points read "
                         "exactly 0.000, so anything >0 is a real edit; 0.5 px "
                         "keeps the CSV about edits a human can see.")
    ap.add_argument("--same-fly-median-px", type=float, default=20.0,
                    help="max median keypoint displacement for two annotations "
                         "to be the same animal. The measured distribution is "
                         "bimodal with nothing between ~1.4 px (p80) and "
                         "~186 px (p85), so any value in that gap gives the "
                         "same answer.")
    ap.add_argument("--inset-px", type=int, default=28,
                    help="half-width in source px of each zoom inset")
    args = ap.parse_args()

    inst = json.load(open(os.path.join(args.src, "annotations", "instances.json")))
    names = inst["keypoint_names"]
    cache = json.load(open(args.hash_cache))
    real_of = {im["id"]: os.path.realpath(
        os.path.join(args.src, "images", im["file_name"])) for im in inst["images"]}
    image_hash = {i: cache[p] for i, p in real_of.items()}

    old = json.load(open(os.path.join(args.src, "annotations", "split.json")))
    new = json.load(open(os.path.join(args.split_root, "annotations", "split.json")))
    fs_of_ann = {}
    for k, v in inst["framesets"].items():
        for aid in v["ann_ids"]:
            if aid is not None:
                fs_of_ann[aid] = k

    def side(aid, sp):
        return sp.get(fs_of_ann.get(aid), "-")

    pairs, unpaired, sexc = collect_pairs(inst, image_hash, real_of, old,
                                          args.same_fly_median_px)
    print(f"{len(pairs)} same-fly pairs on byte-identical pixels; "
          f"{unpaired} annotations with no counterpart (the second fly, which "
          f"exists on only one side -- real data, not a duplicate); "
          f"{sexc} rejected for conflicting sex despite co-location")

    # The whole same-fly population, including the pairs that agree exactly --
    # this is the label-noise floor, and it is what an MPJPE delta competes with.
    alld = np.concatenate([pair_stats(a, b, args.diff_threshold)[1]
                           for _, _, _, a, b, _ in pairs])
    print(f"same-fly per-keypoint disagreement over ALL {len(alld)} compared "
          f"keypoints: median {np.median(alld):.2f} mean {alld.mean():.2f} "
          f"p95 {np.percentile(alld, 95):.2f} p99 {np.percentile(alld, 99):.2f} "
          f"max {alld.max():.2f} px")

    rows, kp_rows = [], []
    for h, ia, ib, a, b, iou in pairs:
        d, fin, diff = pair_stats(a, b, args.diff_threshold)
        if len(diff) == 0:
            continue
        top = sorted(diff, key=lambda i: -d[i])[:5]
        rows.append(dict(
            max_px=round(float(np.nanmax(fin)), 3),
            p95_px=round(float(np.percentile(fin, 95)), 3),
            mean_px=round(float(fin.mean()), 3),
            median_px=round(float(np.median(fin)), 3),
            n_kp_differing=len(diff), n_kp_compared=len(fin),
            content_md5=h, bbox_iou=round(iou, 4),
            recording_a=ia["recording"], recording_b=ib["recording"],
            camera=ia["file_name"].split("/")[1],
            frame=int(ia["file_name"].rsplit("Frame_", 1)[1].split(".")[0]),
            ann_id_a=a["id"], ann_id_b=b["id"],
            fly_id_a=a["fly_id"], fly_id_b=b["fly_id"],
            sex_a=a["sex"], sex_b=b["sex"],
            corpus_a=corpus_of(real_of[ia["id"]]), corpus_b=corpus_of(real_of[ib["id"]]),
            old_split_a=side(a["id"], old), old_split_b=side(b["id"], old),
            new_split=side(a["id"], new),
            path_a=real_of[ia["id"]], path_b=real_of[ib["id"]],
            top_keypoints="; ".join(f"{names[i]}={d[i]:.2f}px" for i in top),
            figure="", verdict="", notes=""))
        for i in diff:
            # recording/camera/frame are deliberately NOT repeated here -- join
            # back to the pair CSV on (content_md5, ann_id_a, ann_id_b). This
            # file has ~23k rows and the redundancy doubled it for no
            # information.
            kp_rows.append(dict(keypoint=names[i], px=round(float(d[i]), 3),
                                content_md5=h, ann_id_a=a["id"], ann_id_b=b["id"]))

    rows.sort(key=lambda r: (-r["max_px"], -r["n_kp_differing"]))
    os.makedirs(args.csv_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # Worst-first ALONE gives 48 figures from one alias pair, because
    # 2026_03_09_14_39_40 == 2026_04_08_14_59_45 supplies 87% of the rows. The
    # female headless pair (2026_06_09_15_38_35 == 2026_06_10_15_05_02) would
    # get none -- and the female fly is where this pipeline actually fails, so
    # a figure set that omits it overstates how well the labels are understood.
    # Every alias pair gets its own worst --per-pair-figures first; the
    # remaining budget goes worst-first overall.
    chosen, seen = [], set()
    by_pair = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_pair[(r["recording_a"], r["recording_b"])].append(i)
    for pr in sorted(by_pair):
        for i in by_pair[pr][:args.per_pair_figures]:
            if i not in seen:
                seen.add(i)
                chosen.append(i)
    for i in range(len(rows)):
        if len(chosen) >= args.n_figures:
            break
        if i not in seen:
            seen.add(i)
            chosen.append(i)
    chosen = sorted(chosen, key=lambda i: rows[i]["max_px"], reverse=True)
    figs = _render([rows[i] for i in chosen], inst, real_of, names,
                   args.fig_dir, args.diff_threshold, args.inset_px)
    for i, f in zip(chosen, figs):
        rows[i]["figure"] = f

    p1 = os.path.join(args.csv_dir, "same_fly_label_disagreements.csv")
    with open(p1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    p2 = os.path.join(args.csv_dir, "same_fly_label_disagreements_keypoints.csv")
    kp_rows.sort(key=lambda r: -r["px"])
    with open(p2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kp_rows[0]))
        w.writeheader()
        w.writerows(kp_rows)

    # The 50-row roll-up is the file a human actually reads to see whether one
    # landmark family dominates; the 23k-row file above is for joins.
    p3 = os.path.join(args.csv_dir, "disagreement_by_keypoint.csv")
    per = collections.defaultdict(list)
    for r in kp_rows:
        per[r["keypoint"]].append(r["px"])
    summ = sorted(({"keypoint": k, "n_pairs_differing": len(v),
                    "pct_of_pairs": round(100 * len(v) / len(rows), 1),
                    "median_px": round(float(np.median(v)), 2),
                    "p95_px": round(float(np.percentile(v, 95)), 2),
                    "max_px": round(float(max(v)), 2)} for k, v in per.items()),
                  key=lambda r: -r["n_pairs_differing"])
    with open(p3, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)

    allpx = np.array([r["px"] for r in kp_rows])
    print(f"\n{len(rows)} disagreeing pairs (of {len(pairs)} same-fly pairs); "
          f"{len(kp_rows)} disagreeing keypoint instances")
    print(f"per-keypoint px: median {np.median(allpx):.2f}  p95 "
          f"{np.percentile(allpx, 95):.2f}  max {allpx.max():.2f}")
    print("per-pair max px: median "
          f"{np.median([r['max_px'] for r in rows]):.2f}  p95 "
          f"{np.percentile([r['max_px'] for r in rows], 95):.2f}")
    dom = collections.Counter(r["keypoint"] for r in kp_rows)
    print("dominant keypoints:", ", ".join(
        f"{k} ({v}, median {np.median([r['px'] for r in kp_rows if r['keypoint']==k]):.2f}px)"
        for k, v in dom.most_common(8)))
    print("wrote", p1, "\nwrote", p2, "\nwrote", p3,
          f"\nwrote {len([f for f in figs if f])} figures to", args.fig_dir)


def _render(rows, inst, real_of, names, fig_dir, thr, inset_px):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    sys.path.insert(0, _ROOT)
    from viz.core.colors import PALETTE

    def rgb(key):
        b, g, r = PALETTE[key]
        return (r / 255, g / 255, b / 255)
    # cyan (PALETTE "detector") vs orange (PALETTE "fly1"). Both must survive
    # a blue-green arena: the first attempt used "legs", which is BGR
    # (255,140,0) = AZURE, and read as the same colour as cyan on these
    # frames -- the figure was unusable and this is why.
    CA, CB = rgb("detector"), rgb("fly1")
    ann_by_id = {a["id"]: a for a in inst["annotations"]}
    out = []
    for r in rows:
        a, b = ann_by_id[r["ann_id_a"]], ann_by_id[r["ann_id_b"]]
        ka, kb = kp_array(a), kp_array(b)
        d, _, diff = pair_stats(a, b, thr)
        img = np.asarray(Image.open(r["path_a"]).convert("RGB"))
        H, W = img.shape[:2]
        x, y, w, h = a["bbox"]
        pad = 30
        x0, x1 = max(0, int(x - pad)), min(W, int(x + w + pad))
        y0, y1 = max(0, int(y - pad)), min(H, int(y + h + pad))

        n_in = min(len(diff), 6)
        fig = plt.figure(figsize=(4 + 2.3 * n_in, 5.0))
        gs = fig.add_gridspec(2, max(n_in, 1) + 2, width_ratios=[2, 2] + [1] * max(n_in, 1))
        ax = fig.add_subplot(gs[:, :2])
        ax.imshow(img[y0:y1, x0:x1])
        ok = np.array([i for i in range(len(names)) if i not in set(diff)])
        vis = (ka[:, 2] > 0) & (kb[:, 2] > 0)
        if len(ok):
            m = ok[vis[ok]]
            ax.plot(ka[m, 0] - x0, ka[m, 1] - y0, ".", ms=2.5, color="0.75",
                    label=f"agree (<{thr}px), n={len(m)}")
        for i in diff:
            ax.plot([ka[i, 0] - x0, kb[i, 0] - x0], [ka[i, 1] - y0, kb[i, 1] - y0],
                    "-", lw=0.8, color="w", alpha=0.8)
        ax.plot(ka[diff, 0] - x0, ka[diff, 1] - y0, "o", ms=5, mfc="none", mew=1.4,
                color=CA, label=f"A  {r['corpus_a']}  ann{a['id']}")
        ax.plot(kb[diff, 0] - x0, kb[diff, 1] - y0, "s", ms=5, mfc="none", mew=1.4,
                color=CB, label=f"B  {r['corpus_b']}  ann{b['id']}")
        ax.set_title(f"{r['recording_a']} = {r['recording_b']}   {r['camera']}   "
                     f"Frame_{r['frame']}   md5 {r['content_md5'][:12]}\n"
                     f"{r['n_kp_differing']} of {r['n_kp_compared']} keypoints "
                     f"differ, max {r['max_px']:.1f} px   "
                     f"[old split A={r['old_split_a']} B={r['old_split_b']} "
                     f"-> new {r['new_split']}]", fontsize=7.5, pad=6)
        ax.legend(fontsize=6, loc="lower left", framealpha=0.75)
        ax.set_xticks([]); ax.set_yticks([])

        for j, i in enumerate(sorted(diff, key=lambda i: -d[i])[:n_in]):
            cx = (ka[i, 0] + kb[i, 0]) / 2
            cy = (ka[i, 1] + kb[i, 1]) / 2
            axi = fig.add_subplot(gs[:, 2 + j])
            # Size the window to the disagreement: a fixed box clips the
            # markers off the edge exactly when the disagreement is largest.
            half = int(max(inset_px, 0.9 * d[i]))
            ix0, ix1 = int(cx - half), int(cx + half)
            iy0, iy1 = int(cy - half), int(cy + half)
            sub = np.zeros((iy1 - iy0, ix1 - ix0, 3), np.uint8)
            sy0, sx0 = max(0, iy0), max(0, ix0)
            sy1, sx1 = min(H, iy1), min(W, ix1)
            sub[sy0 - iy0:sy1 - iy0, sx0 - ix0:sx1 - ix0] = img[sy0:sy1, sx0:sx1]
            # Shadow lift, only where the crop is actually dark. Raw crops of
            # the wing base and the trochanters are near-black at this scale
            # and the reader cannot see the structure the marker is supposed
            # to sit on. A per-channel percentile STRETCH was tried first and
            # rejected: it turned those crops into saturated false colour and
            # destroyed the anatomy it was meant to reveal. Gamma is applied
            # to all three channels equally, so hue is preserved.
            mean = float(sub.mean())
            if mean < 90:
                g = max(0.45, mean / 90.0)
                sub = (255.0 * (sub / 255.0) ** g).astype(np.uint8)
            axi.imshow(sub, interpolation="nearest")
            axi.plot([ka[i, 0] - ix0, kb[i, 0] - ix0],
                     [ka[i, 1] - iy0, kb[i, 1] - iy0], "-", lw=1.0, color="k",
                     alpha=0.6)
            axi.plot(ka[i, 0] - ix0, ka[i, 1] - iy0, "+", ms=15, mew=2.6, color="k")
            axi.plot(kb[i, 0] - ix0, kb[i, 1] - iy0, "x", ms=13, mew=2.6, color="k")
            axi.plot(ka[i, 0] - ix0, ka[i, 1] - iy0, "+", ms=14, mew=1.6, color=CA)
            axi.plot(kb[i, 0] - ix0, kb[i, 1] - iy0, "x", ms=12, mew=1.6, color=CB)
            axi.set_title(f"{names[i]}\nA(+) vs B(x)  {d[i]:.2f} px", fontsize=7)
            axi.set_xticks([]); axi.set_yticks([])
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        fn = f"{r['recording_a']}_{r['camera']}_Frame_{r['frame']}__{r['content_md5'][:8]}.png"
        fig.savefig(os.path.join(fig_dir, fn), dpi=130)
        plt.close(fig)
        out.append(fn)
    return out


if __name__ == "__main__":
    main()
