#!/usr/bin/env python3
"""Why v12's val MPJPE looks worse than v5vf's, on the SAME val set.

EXPECTATION IF THE GAP IS LEAKAGE RATHER THAN MODEL QUALITY. v5vf trained on
4 of the 5 recordings in v12's val (v5's held-out set was
{2026_03_18_15_31_22, 2026_06_15_12_12_33, 2026_05_27_11_57_05,
2026_04_07_11_33_33}; everything else went to train, and the courtship pairs
were filed under TWO ids so holding out one filing left the other in train).
So: on the single recording BOTH models trained on, the two bars should be the
same height; on recordings only v5vf trained on, v5vf should be far lower. If
instead v5vf were uniformly better everywhere including the shared recording,
the gap would be real model quality and this reading would be wrong.

Left: per-recording MPJPE, bars annotated with leak status.
Right: v12 per-keypoint error, coloured by body group -- says WHERE it errs.

    python scripts/viz/detector_v12_vs_v5vf.py --npz-dir figures/2026-09-03-v12-detector-ab
"""
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--root", default="/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or a.npz_dir

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.load(open(os.path.join(a.root, "annotations", "instances_val.json")))
    visv = {an["id"]: (np.asarray(an["keypoints"], float).reshape(-1, 3)[:, 2] > 0)
            for an in d["annotations"]}

    def load(name):
        z = np.load(os.path.join(a.npz_dir, name + ".npz"), allow_pickle=True)
        e = np.linalg.norm(z["pred_zeroed"] - z["gt_zeroed"], axis=-1)
        # visibility must come from the annotations: the npz stores xy only, and
        # averaging over padded-absent joints puts ~520 px "errors" on the eyes
        # of headless flies straight into the mean.
        vis = np.stack([visv[int(i)] for i in z["ann_id"]])
        return e, vis, z["recording"], z["kp_names"]

    e12, vis, rec, kpn = load("v12_bal_maskoff")
    e5, _, _, _ = load("v5vf_maskoff")
    LEAK = {"2026_01_29_14_09_33": "both trained on it",
            "2026_04_02_12_11_50": "v5vf: half",
            "2026_04_02_15_25_51": "v5vf only",
            "2026_06_09_15_21_14": "v5vf only",
            "2026_06_09_15_46_55": "v5vf only"}

    recs = sorted(set(rec.tolist()))
    m12 = [float(e12[vis & (rec == r)[:, None]].mean()) for r in recs]
    m5 = [float(e5[vis & (rec == r)[:, None]].mean()) for r in recs]
    ns = [int((rec == r).sum()) for r in recs]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16.5, 6.4),
                                   gridspec_kw={"width_ratios": [1.05, 1.35]})
    y = np.arange(len(recs))
    axL.barh(y - 0.2, m12, 0.4, color="#1f77b4", label="v12 (honest held-out)")
    axL.barh(y + 0.2, m5, 0.4, color="#d62728", label="v5vf (trained on most of this)")
    for i, r in enumerate(recs):
        tag = LEAK.get(r, "?")
        axL.text(max(m12[i], m5[i]) + 0.5, i, tag, va="center", fontsize=8,
                 color=("green" if tag.startswith("both") else "crimson"))
        if LEAK.get(r, "").startswith("both"):
            axL.axhspan(i - 0.45, i + 0.45, color="green", alpha=0.10, zorder=0)
    axL.set_yticks(y); axL.set_yticklabels([f"{r}\n(n={n})" for r, n in zip(recs, ns)], fontsize=8)
    axL.set_xlabel("val MPJPE (px), visible keypoints only")
    axL.set_title("Per-recording. GREEN = the control: the ONE recording BOTH\n"
                  "models trained on -- there the bars match (5.42 vs 5.43).", fontsize=10)
    axL.legend(fontsize=9, loc="lower right"); axL.grid(axis="x", alpha=0.3)

    pk = [(float(e12[:, i][vis[:, i]].mean()), str(kpn[i])) for i in range(e12.shape[1])
          if vis[:, i].any()]
    pk.sort()
    vals = [v for v, _ in pk]; names = [n for _, n in pk]
    def grp(n):
        if "TaTip" in n or "TaT" in n: return "#d62728"
        if n.startswith(("T1", "T2", "T3")):     return "#ff7f0e"
        if "Wing" in n:                          return "#9467bd"
        return "#1f77b4"
    axR.barh(np.arange(len(vals)), vals, color=[grp(n) for n in names])
    axR.set_yticks(np.arange(len(vals))); axR.set_yticklabels(names, fontsize=5.5)
    axR.set_xlabel("v12 per-keypoint MPJPE (px)")
    axR.set_title("v12 by keypoint: red = tarsal segments/tips, orange = other leg,\n"
                  "purple = wing, blue = body. The tips are the tail.", fontsize=10)
    axR.grid(axis="x", alpha=0.3)

    fig.suptitle("v12 vs v5vf on the IDENTICAL v12 val set (1,069 annotations, both mask-off).  "
                 "Overall 15.19 vs 7.76 px --\nbut v5vf trained on 4 of these 5 recordings, so its "
                 "number is recall, not generalisation.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = os.path.join(out, "v12_vs_v5vf.png")
    fig.savefig(p, dpi=125)
    print("wrote", p)
    print(f"\ncontrol recording (both trained): v12 {m12[recs.index('2026_01_29_14_09_33')]:.2f} "
          f"vs v5vf {m5[recs.index('2026_01_29_14_09_33')]:.2f} px")


if __name__ == "__main__":
    main()
