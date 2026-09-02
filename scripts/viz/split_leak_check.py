"""Acceptance gate for a train/val split: does val share pixels with train?

    MUJOCO_GL=egl python scripts/viz/split_leak_check.py \
        --root  .../red_data_3d_v6_contentsplit \
        --baseline .../red_data_3d_v5_valfix \
        --hash-cache /tmp/dedup_work/hash_cache.json \
        --out figures/2026-09-02-dataset-dedup

WHAT THE FIGURE SHOULD SHOW IF THE CONTENT-KEYED SPLIT IS CORRECT.

  Panel 1 (per-recording val leak): the baseline has tall bars on
  2026_04_07_11_33_33, 2026_05_27_11_57_05, 2026_06_15_12_12_33 and
  2026_06_19_11_09_36 -- the val recordings whose alias twin sits in train --
  and a zero bar on 2026_03_18_15_31_22, the one val recording with no alias.
  The new split must be flat zero everywhere. A single non-zero new bar means
  a content group escaped its capture.

  Panel 2 (val -> nearest train image, RMS grey difference of a 64x16
  thumbnail, 0-255): the baseline must have a spike AT EXACTLY 0 holding
  ~30% of val -- those are byte-identical images. The new curve must have no
  mass at 0 and none below the dashed line at 0.39, which is what frames ONE
  apart in the same recording score; anything left of that line is a val
  image effectively already seen. If the new curve still touches 0, the split
  is not content-keyed and nothing else in this figure matters.

  Panel 3 (val composition): the new val must not have bought its cleanliness
  by throwing the female and two-fly frames away -- the coverage the old split
  claimed has to survive in a bar chart, not just in a total.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

BEHAVIOUR = {
    "2026_01_13_18_47_45": "general M", "2026_03_22_12_07_40": "general M",
    "2026_01_29_14_09_33": "general F", "2026_02_09_22_26_25": "amputation",
    "2026_02_13_13_44_49": "amputation", "2026_06_01_15_34_04": "grooming",
    "2026_07_30_13_28_99": "wall", "2026_08_26_16_05_15": "climbing F",
    # headless_* -- these fell through to the "courtship" default in the first
    # version and mislabelled 8 new-val framesets, which is exactly the sort of
    # bare-category error this repo's figures are supposed not to make.
    "2026_06_09_15_00_41": "headless", "2026_06_09_15_21_14": "headless",
    "2026_06_09_15_38_35": "headless", "2026_06_09_15_46_55": "headless",
    "2026_06_10_15_05_02": "headless",
}


def _sex(man, rec, fly):
    m = man["recordings"][rec]
    return m.get("fly_sex", {}).get(fly, m.get("sex", "unknown"))


def _load(root, inst, cache):
    sp = json.load(open(os.path.join(root, "annotations", "split.json")))
    real = {im["id"]: os.path.realpath(os.path.join(root, "images", im["file_name"]))
            for im in inst["images"]}
    return sp, {i: cache[p] for i, p in real.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--hash-cache", required=True)
    ap.add_argument("--thumbs", required=True, help="npy from the dedup pass")
    ap.add_argument("--thumb-keys", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pickle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from viz.core.colors import PALETTE

    def rgb(k):
        b, g, r = PALETTE[k]
        return (r / 255, g / 255, b / 255)
    C_OLD, C_NEW = rgb("fly1"), rgb("fit")          # orange = old, green = new

    inst = json.load(open(os.path.join(args.baseline, "annotations", "instances.json")))
    man = json.load(open(os.path.join(args.baseline, "manifest.json")))
    cache = json.load(open(args.hash_cache))
    A = np.load(args.thumbs)
    kidx = {k: i for i, k in enumerate(pickle.load(open(args.thumb_keys, "rb")))}
    fs = inst["framesets"]
    img_rec = {im["id"]: im["recording"] for im in inst["images"]}
    two_fly = {(v["recording"], k.split("/")[1]) for k, v in fs.items()}
    n_fly = collections.Counter((v["recording"], k.split("/")[1]) for k, v in fs.items())
    two_fly = {k for k in two_fly if n_fly[k] > 1}

    out = {}
    for tag, root in (("old", args.baseline), ("new", args.root)):
        sp, ih = _load(root, inst, cache)
        tr, va = set(), set()
        rec_tot, rec_leak = collections.Counter(), collections.Counter()
        for k, v in fs.items():
            s = sp.get(k)
            if s == "train":
                tr.update(ih[i] for i in v["frames"])
            elif s == "val":
                va.update(ih[i] for i in v["frames"])
        for k, v in fs.items():
            if sp.get(k) != "val":
                continue
            for i in v["frames"]:
                rec_tot[img_rec[i]] += 1
                if ih[i] in tr:
                    rec_leak[img_rec[i]] += 1
        T = A[[kidx[h] for h in sorted(tr)]]
        V = A[[kidx[h] for h in sorted(va)]]
        tn = (T * T).sum(1)
        best = np.full(len(V), np.inf)
        for s0 in range(0, len(T), 2048):
            d2 = ((V * V).sum(1)[:, None] + tn[s0:s0 + 2048][None, :]
                  - 2 * V @ T[s0:s0 + 2048].T)
            best = np.minimum(best, d2.min(1))
        sex = collections.Counter()
        beh = collections.Counter()
        n2 = set()
        for k, v in fs.items():
            if sp.get(k) != "val":
                continue
            r = v["recording"]
            sex[_sex(man, r, f"fly{v['fly_id']}")] += 1
            beh[BEHAVIOUR.get(r, "courtship")] += 1
            if (r, k.split("/")[1]) in two_fly:
                n2.add((r, k.split("/")[1]))
        out[tag] = dict(rms=np.sqrt(np.maximum(best, 0) / A.shape[1]),
                        rec_tot=rec_tot, rec_leak=rec_leak, val=len(va),
                        leak=len(tr & va), sex=sex, beh=beh, n2=len(n2))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    recs = sorted(set(out["old"]["rec_tot"]) | set(out["new"]["rec_tot"]))
    x = np.arange(len(recs))
    for k, (tag, c, off) in enumerate((("old", C_OLD, -0.2), ("new", C_NEW, 0.2))):
        o = out[tag]
        f = [100 * o["rec_leak"][r] / o["rec_tot"][r] if o["rec_tot"][r] else 0
             for r in recs]
        axes[0].bar(x + off, f, 0.38, color=c,
                    label=f"{tag}  ({o['leak']}/{o['val']} val images leaked, "
                          f"{100*o['leak']/o['val']:.1f}%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r[5:] for r in recs], rotation=90, fontsize=7)
    axes[0].set_ylabel("% of this recording's val images\nbyte-identical to a train image")
    axes[0].set_title("Per-recording val leak", fontsize=10)
    axes[0].legend(fontsize=8)

    bins = np.linspace(0, 25, 90)
    for tag, c in (("old", C_OLD), ("new", C_NEW)):
        axes[1].hist(np.clip(out[tag]["rms"], 0, 25), bins=bins, histtype="step",
                     lw=1.8, color=c, label=f"{tag}  (min {out[tag]['rms'].min():.2f})")
    axes[1].axvline(0.39, ls="--", lw=1.2, color="0.35")
    axes[1].text(0.5, axes[1].get_ylim()[1] * 0.85,
                 "0.39 = frames 1 apart,\nsame recording", fontsize=7, color="0.3")
    axes[1].set_xlabel("RMS grey difference to nearest TRAIN image "
                       "(0-255, clipped at 25)")
    axes[1].set_ylabel("val images")
    axes[1].set_title("How close is each val image to the training set?", fontsize=10)
    axes[1].legend(fontsize=8)

    cats = ["male", "female", "two-fly frames"] + sorted(
        set(out["old"]["beh"]) | set(out["new"]["beh"]))
    xs = np.arange(len(cats))
    for tag, c, off in (("old", C_OLD, -0.2), ("new", C_NEW, 0.2)):
        o = out[tag]
        vals = [o["sex"]["male"], o["sex"]["female"], o["n2"]] + \
               [o["beh"].get(b, 0) for b in cats[3:]]
        axes[2].bar(xs + off, vals, 0.38, color=c, label=tag)
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("val framesets (frames, for two-fly)")
    axes[2].set_title("Val composition", fontsize=10)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "split_leak_check.png")
    fig.savefig(p, dpi=140)
    json.dump({t: dict(val=o["val"], leak=o["leak"], min_rms=float(o["rms"].min()),
                       n_below_2=int((o["rms"] < 2).sum()),
                       sex=dict(o["sex"]), beh=dict(o["beh"]), two_fly=o["n2"])
               for t, o in out.items()},
              open(os.path.join(args.out, "split_leak_check.json"), "w"), indent=2)
    print("wrote", p)


if __name__ == "__main__":
    main()
