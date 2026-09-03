#!/usr/bin/env python3
"""Compare N detector arms on the SAME val split from mask_channel_eval.py npz files.

Acceptance criteria this figure is written against
(docs/benchmark/2026-09-03-distractor-supervision/notes.md): vs the baseline arm
(the first --npz), a winning arm shows
  * lower tarsal-tip MPJPE, most on the two courtship recordings
    (2026_04_02_15_25_51, _12_11_50) and on two-fly frames;
  * body keypoints not worse than +0.5 px; single-fly recordings unchanged;
  * a shorter tail (p99, fraction > 20 px).
Panels: per-recording MPJPE (left), per-part MPJPE (tarsal tips / other leg /
wing / body), two-fly vs single-fly frames (overall and tips), and the
per-keypoint delta vs the baseline for every other arm (negative = better).
A JSON scorecard with the same numbers lands beside the PNG.

    python scripts/viz/detector_arms_compare.py --out figures/2026-09-03-detector-arms \
        --tag plain --npz maskoff=<npz> --npz maskon=<npz> --npz armA=<npz> ...
"""
import argparse, json, os, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    conds = sorted({k.split("_", 1)[1] for k in z.files if k.startswith("pred_")})
    assert len(conds) == 1, f"{npz_path}: expected one condition, got {conds}"
    c = conds[0]
    err = np.linalg.norm(z[f"pred_{c}"] - z[f"gt_{c}"], axis=-1)
    err[~z[f"vis_{c}"]] = np.nan
    return dict(err=err, rec=z["recording"], sex=z["sex"], file=z["file_name"],
                names=list(z["kp_names"]), cond=c, kp_order_verified=bool(z["kp_order_verified"]))


def parts(names):
    out = {"tarsal tips": [], "other leg": [], "wing": [], "body": []}
    for i, n in enumerate(names):
        if n.endswith("_TaTip"): out["tarsal tips"].append(i)
        elif re.match(r"T[1-3][LR]_", n): out["other leg"].append(i)
        elif n.startswith("Wing"): out["wing"].append(i)
        else: out["body"].append(i)
    return out


def two_fly_flags(root, files):
    d = json.load(open(os.path.join(root, "annotations", "instances_val.json")))
    n_per_img = {}
    for a in d["annotations"]:
        n_per_img[a["image_id"]] = n_per_img.get(a["image_id"], 0) + 1
    fn2n = {im["file_name"]: n_per_img.get(im["id"], 0) for im in d["images"]}
    return np.asarray([fn2n[f] >= 2 for f in files])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", action="append", required=True, help="label=path; first is the baseline")
    ap.add_argument("--root", default="/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="plain")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arms = {}
    for spec in a.npz:
        lab, path = spec.split("=", 1)
        arms[lab] = load(path)
    labels = list(arms)
    base = arms[labels[0]]
    for lab, d in arms.items():
        assert list(d["file"]) == list(base["file"]), f"{lab}: different val ordering"
    names = base["names"]; P = parts(names)
    two = two_fly_flags(a.root, base["file"])
    recs = sorted(set(base["rec"]))
    score = {"tag": a.tag, "n_annotations": int(len(base["file"])), "two_fly_annotations": int(two.sum()),
             "conditions": {l: arms[l]["cond"] for l in labels}, "arms": {}}
    for lab, d in arms.items():
        e = d["err"]
        s = {"overall": float(np.nanmean(e)),
             "per_recording": {r: float(np.nanmean(e[d["rec"] == r])) for r in recs},
             "per_part": {p: float(np.nanmean(e[:, idx])) for p, idx in P.items()},
             "two_fly": float(np.nanmean(e[two])), "single": float(np.nanmean(e[~two])),
             "two_fly_tips": float(np.nanmean(e[two][:, P["tarsal tips"]])),
             "single_tips": float(np.nanmean(e[~two][:, P["tarsal tips"]])),
             "p50": float(np.nanpercentile(e, 50)), "p90": float(np.nanpercentile(e, 90)),
             "p99": float(np.nanpercentile(e, 99)), "frac_gt_20px": float(np.nanmean(e > 20)),
             "female": float(np.nanmean(e[d["sex"] == "female"])), "male": float(np.nanmean(e[d["sex"] == "male"])),
             "per_keypoint": {n: float(np.nanmean(e[:, i])) for i, n in enumerate(names)}}
        score["arms"][lab] = s
    json.dump(score, open(out / f"arms_scorecard_{a.tag}.json", "w"), indent=1)

    # ---- table to stdout
    cols = ["overall", "two_fly", "single", "two_fly_tips", "female", "male", "p99", "frac_gt_20px"]
    print(f"[{a.tag}] n={len(base['file'])} anns, two-fly {two.sum()}")
    print("arm".ljust(28) + "".join(c.rjust(14) for c in cols))
    for lab in labels:
        s = score["arms"][lab]
        print(lab.ljust(28) + "".join(f"{s[c]:14.3f}" for c in cols))
    print("per part:".ljust(28) + "".join(p.rjust(14) for p in P))
    for lab in labels:
        print(lab.ljust(28) + "".join(f"{score['arms'][lab]['per_part'][p]:14.3f}" for p in P))
    print("per recording:".ljust(28) + "".join(r[-11:].rjust(14) for r in recs))
    for lab in labels:
        print(lab.ljust(28) + "".join(f"{score['arms'][lab]['per_recording'][r]:14.3f}" for r in recs))

    # ---- figure
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    w = 0.8 / len(labels)
    ax = axes[0, 0]
    for j, lab in enumerate(labels):
        ax.bar(np.arange(len(recs)) + (j - (len(labels) - 1) / 2) * w,
               [score["arms"][lab]["per_recording"][r] for r in recs], w, label=f"{lab} ({score['arms'][lab]['overall']:.2f} px)")
    ax.set_xticks(range(len(recs))); ax.set_xticklabels([f"{r}\n(n={int((base['rec'] == r).sum())})" for r in recs], fontsize=7)
    ax.set_ylabel("MPJPE px"); ax.set_title("per val recording", fontsize=10); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax = axes[0, 1]
    for j, lab in enumerate(labels):
        ax.bar(np.arange(len(P)) + (j - (len(labels) - 1) / 2) * w, [score["arms"][lab]["per_part"][p] for p in P], w, label=lab)
    ax.set_xticks(range(len(P))); ax.set_xticklabels([f"{p}\n({len(i)} kps)" for p, i in P.items()], fontsize=8)
    ax.set_title("per body part", fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax = axes[1, 0]
    cats = ["two_fly", "single", "two_fly_tips", "single_tips"]
    for j, lab in enumerate(labels):
        ax.bar(np.arange(len(cats)) + (j - (len(labels) - 1) / 2) * w, [score["arms"][lab][c] for c in cats], w, label=lab)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"two-fly frames\n(n={two.sum()})", f"single-fly frames\n(n={(~two).sum()})", "two-fly: tarsal tips", "single: tarsal tips"], fontsize=8)
    ax.set_title("two-fly vs single-fly annotations", fontsize=10); ax.grid(axis="y", alpha=0.3); ax.set_ylabel("MPJPE px")
    ax = axes[1, 1]
    order = np.argsort([-score["arms"][labels[0]]["per_keypoint"][n] for n in names])
    for lab in labels[1:]:
        d = np.asarray([score["arms"][lab]["per_keypoint"][names[i]] - score["arms"][labels[0]]["per_keypoint"][names[i]] for i in order])
        ax.plot(d, np.arange(len(names)), "o-", ms=3, lw=0.8, label=f"{lab} - {labels[0]}")
    ax.axvline(0, color="k", lw=0.8); ax.set_yticks(range(len(names))); ax.set_yticklabels([names[i] for i in order], fontsize=6)
    ax.invert_yaxis(); ax.set_xlabel(f"per-keypoint MPJPE delta vs {labels[0]} (px; negative = better)")
    ax.set_title("per-keypoint delta (sorted by baseline error, worst on top)", fontsize=10); ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Detector arms on the v12 val split ({a.tag}); conditions: " +
                 ", ".join(f"{l}={arms[l]['cond']}" for l in labels), fontsize=11)
    fig.tight_layout(); fig.savefig(out / f"arms_compare_{a.tag}.png", dpi=120)
    print("wrote", out / f"arms_compare_{a.tag}.png")


if __name__ == "__main__":
    main()
