#!/usr/bin/env python3
"""Acceptance figure for SAM3 dataset masks: are they ON the labelled fly, and only it?

EXPECTATION IF THE MASKS ARE GOOD. The red overlay covers the animal that the
cyan GT keypoints sit on -- body, wings and legs -- and nothing else. Failure
modes worth catching, in order of how plausible they are here:
  * SHADOW / REFLECTION swept in with the fly. The arena floor is bright and
    flies cast a hard shadow; a box prompt that includes it can segment both.
    This is the reason the figure is sorted by foreground FRACTION: measured
    fractions run 1.8%-8.3% of the 1936x448 crop, and a fly is ~2%, so the
    high tail is where a shadow would hide.
  * THE WRONG FLY on a two-fly image (the mask should track the cyan points,
    not the other animal).
  * A mask covering the whole crop, or a sliver.

Each panel prints the foreground fraction and what share of the GT keypoints
land INSIDE the mask -- the latter is the number that says "on the right
animal", which area alone cannot.

    python scripts/viz/sam3_dataset_mask_check.py --root <v12 root> \
        --out figures/2026-09-03-sam3-dataset-masks
"""
import argparse, collections, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4, help="panels per band")
    a = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    root = a.root
    os.makedirs(a.out, exist_ok=True)

    # index annotations by (rec, cam, frame)
    per = collections.defaultdict(list)
    for split in ("train", "val"):
        p = os.path.join(root, "annotations", f"instances_{split}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        id2im = {im["id"]: im for im in d["images"]}
        for an in d["annotations"]:
            im = id2im[an["image_id"]]
            rec, cam, fn = im["file_name"].split("/")
            per[(rec, cam, os.path.splitext(fn)[0])].append((an, im))

    rows = []
    for (rec, cam, frame), anns in per.items():
        npz = os.path.join(root, "masks", rec, cam, frame + ".npz")
        if not os.path.exists(npz) or os.path.islink(npz):
            continue                      # only OUR output, not inherited links
        try:
            z = np.load(npz)
            m, ids, matched = z["masks"], z["ann_ids"], z["matched"]
        except Exception:
            continue
        for k, aid in enumerate(ids):
            an = next((x for x in anns if x[0]["id"] == int(aid)), None)
            if an is None:
                continue
            frac = float(m[k].mean())
            kp = np.asarray(an[0]["keypoints"], float).reshape(-1, 3)
            v = kp[:, 2] > 0
            H, W = m[k].shape
            xi = np.clip(kp[v, 0].astype(int), 0, W - 1)
            yi = np.clip(kp[v, 1].astype(int), 0, H - 1)
            inside = float(m[k][yi, xi].mean()) if v.any() else float("nan")
            rows.append(dict(rec=rec, cam=cam, frame=frame, ann=int(aid),
                             frac=frac, inside=inside, n_fly=len(anns),
                             img=an[1]["file_name"], mi=k, npz=npz))

    if not rows:
        raise SystemExit("no non-symlink masks found to check")
    rows.sort(key=lambda r: r["frac"])
    stats = np.array([r["frac"] for r in rows])
    ins = np.array([r["inside"] for r in rows])
    print(f"{len(rows)} masks   fg frac: min {stats.min():.4f} p50 "
          f"{np.median(stats):.4f} p95 {np.percentile(stats,95):.4f} max {stats.max():.4f}")
    print(f"keypoints-inside-mask: mean {np.nanmean(ins):.3f} "
          f"p05 {np.nanpercentile(ins,5):.3f} min {np.nanmin(ins):.3f}")

    # PER-RECORDING acceptance table. The sampled panels below can only show
    # a dozen masks; this is the check that actually covers every recording as
    # the array lands, and it is what caught that 2026_01_13_18_47_45 was 91%
    # over-inclusive while the median across the corpus looked fine.
    by = collections.defaultdict(list)
    for r in rows:
        by[r["rec"]].append(r)
    print(f"\n{'recording':26s}{'n':>6}{'fg p50':>9}{'fg max':>9}{'kp-in min':>11}"
          f"{'over-inc':>10}{'bad':>6}  verdict")
    table = {}
    for rec in sorted(by):
        f = np.array([x["frac"] for x in by[rec]])
        i = np.array([x["inside"] for x in by[rec]])
        over = int((f > 0.05).sum()); bad = int((i < 0.6).sum())
        ok = (over == 0 and bad == 0)
        table[rec] = dict(n=len(f), fg_p50=float(np.median(f)), fg_max=float(f.max()),
                          kp_in_min=float(np.nanmin(i)), over_inclusive=over,
                          badly_wrong=bad, pass_=bool(ok))
        print(f"{rec:26s}{len(f):>6}{np.median(f)*100:>8.2f}%{f.max()*100:>8.2f}%"
              f"{np.nanmin(i)*100:>10.0f}%{over:>10}{bad:>6}  {'PASS' if ok else 'FAIL'}")
    n_fail = sum(1 for v in table.values() if not v["pass_"])
    print(f"\n{len(table)} recordings, {n_fail} FAIL "
          f"(over-inclusive = fg>5%, ~2.5x a fly; bad = <60% of its own keypoints inside)")

    n = a.n
    bands = [("SMALLEST fg", rows[:n]),
             ("MEDIAN fg", rows[len(rows)//2 - n//2: len(rows)//2 - n//2 + n]),
             ("LARGEST fg  <- a swept-in shadow would be here", rows[-n:])]
    two = [r for r in rows if r["n_fly"] > 1]
    if two:
        two.sort(key=lambda r: -r["frac"])
        bands.append(("TWO-FLY images (must follow the cyan points)", two[:n]))

    fig, axes = plt.subplots(len(bands), n, figsize=(5.6 * n, 1.9 * len(bands)))
    axes = np.atleast_2d(axes)
    for bi, (label, band) in enumerate(bands):
        for ci in range(n):
            ax = axes[bi][ci]; ax.axis("off")
            if ci >= len(band):
                continue
            r = band[ci]
            img = np.asarray(Image.open(os.path.join(root, "images", r["img"])).convert("L"))
            m = np.load(r["npz"])["masks"][r["mi"]]
            rgb = np.dstack([img] * 3).astype(float) / 255.0
            rgb[m] = 0.55 * rgb[m] + 0.45 * np.array([1.0, 0.15, 0.15])
            ax.imshow(rgb)
            d = json.load(open(os.path.join(root, "annotations", "instances_train.json"))) if False else None
            ax.set_title(f"{label if ci==0 else ''}\n{r['rec'][:19]} {r['cam']} {r['frame']}"
                         f"  fg={100*r['frac']:.2f}%  kp-in-mask={100*r['inside']:.0f}%"
                         f"{'  [2 flies]' if r['n_fly']>1 else ''}",
                         fontsize=7, color=("crimson" if r["inside"] < 0.6 else "black"))
    fig.suptitle("SAM3 dataset masks (red) vs the annotation that prompted them.  "
                 "Good = red covers the labelled fly and nothing else;\n"
                 "kp-in-mask is the share of that annotation's GT keypoints inside the mask.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = os.path.join(a.out, "sam3_dataset_masks.png")
    fig.savefig(png, dpi=125)
    json.dump({"n": len(rows), "frac_p50": float(np.median(stats)),
               "frac_max": float(stats.max()),
               "inside_mean": float(np.nanmean(ins)),
               "inside_p05": float(np.nanpercentile(ins, 5)),
               "n_recordings": len(table),
               "n_failing_recordings": n_fail,
               "per_recording": table},
              open(os.path.join(a.out, "sam3_dataset_masks.json"), "w"), indent=2)
    print("wrote", png)


if __name__ == "__main__":
    main()
