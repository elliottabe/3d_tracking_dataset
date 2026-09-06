"""Why is the pseudo-label census female-poor and contact-empty? (2026-09-05)

Two questions the census table itself cannot answer, both asked per FLY (the
census aggregates the two):

1. Which gate rejects the female? The census only reports rejections summed
   over both flies, and the campaign's female is the fly whose SAM3 mask fit
   is worst -- so a containment-dominated rejection would say "the gate is
   doing what it was designed to do", while an existence- or step-dominated
   one would say the female TRACK is bad.
2. Is `contact` (inter-fly centroid distance < 15 units) absent from the
   ADMITTED set because the gates drop contact frames, or because it is
   absent from the DATA? Compared here on the same bouts: the separation
   distribution over every frame where both flies are finite, versus over the
   admitted ones.

Sampling: the first `--per-rec` bouts of each recording (masks make a full
pass expensive; the per-gate rates are stable across bouts of a recording).
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "third_party", "jarvis_jax"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pseudo_labels.extract_p3b_pseudolabels import DEFAULT_ROOTS, discover_bouts  # noqa: E402

SEP_EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60, 100, 1e9])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-rec", type=int, default=3)
    ap.add_argument("--out", default="figures/2026-09-mvq/v2_pseudo/census_diagnosis.json")
    a = ap.parse_args()
    from jarvis_jax.data.pseudo_gates import GateThresholds, admit_bout, load_bout_arrays
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.tracking.lift_mvq import BoutMaskStore
    thr = GateThresholds()

    refs = discover_bouts([DEFAULT_ROOTS])
    per_rec = collections.defaultdict(list)
    for r in refs:
        per_rec[r.recording].append(r)
    chosen = [r for rr in per_rec.values() for r in rr[: a.per_rec]]

    rej = collections.defaultdict(lambda: collections.Counter())
    sep_all = np.zeros(len(SEP_EDGES) - 1, np.int64)
    sep_adm = np.zeros(len(SEP_EDGES) - 1, np.int64)
    sep_min_all, sep_min_adm = [], []
    for i, ref in enumerate(chosen, 1):
        rt = ReprojectionTool(ref.calib_dir)
        cams = list(rt.cameras.keys())
        arr = load_bout_arrays(ref.bout_dir, cameras=cams)
        res = admit_bout(arr, BoutMaskStore(ref.masks_npz, cams), rt.camera_matrices, thr)
        with np.errstate(invalid="ignore"):
            cent = np.nanmean(arr.kp3d, axis=2)
            sep = np.linalg.norm(cent[0] - cent[1], axis=-1)
            # closest KEYPOINT pair, the physical "are they touching" measure
            d = np.linalg.norm(arr.kp3d[0][:, :, None, :] - arr.kp3d[1][:, None, :, :], axis=-1)
            kmin = np.nanmin(d.reshape(d.shape[0], -1), axis=1)
        fin = np.isfinite(sep)
        sep_all += np.histogram(sep[fin], SEP_EDGES)[0]
        sep_min_all.append(kmin[np.isfinite(kmin)])
        adm = res.frame & fin
        sep_adm += np.histogram(sep[adm], SEP_EDGES)[0]
        sep_min_adm.append(kmin[res.frame & np.isfinite(kmin)])
        for f in range(2):
            sex = ref.fly_sex.get(f"fly{f}", f"fly{f}")
            c = rej[sex]
            c["_n"] += int(arr.kp3d.shape[1])
            c["_admitted"] += int(res.fly[f].sum())
            for k, v in res.reasons.items():
                c[k] += int(np.asarray(v)[f].sum() if np.asarray(v).ndim == 2 else v.sum())
        print(f"[{i}/{len(chosen)}] {ref.recording}/bout_{ref.bout:05d}", flush=True)

    out = {
        "n_bouts": len(chosen), "per_rec": a.per_rec,
        "sep_edges_units": SEP_EDGES[:-1].tolist(),
        "centroid_sep_all_frames": sep_all.tolist(),
        "centroid_sep_admitted": sep_adm.tolist(),
        "min_keypoint_dist_all": np.histogram(np.concatenate(sep_min_all), SEP_EDGES)[0].tolist(),
        "min_keypoint_dist_admitted": np.histogram(np.concatenate(sep_min_adm), SEP_EDGES)[0].tolist(),
        "per_fly_rejects": {k: dict(v) for k, v in rej.items()},
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    plot(out, a.out.replace(".json", ".png"))


def plot(d, png):
    """EXPECTATION: if the GATES were selecting against close pairs, the
    admitted (orange) distribution would sit to the RIGHT of every-frame
    (grey); if instead the 15-unit CENTROID definition of contact is simply
    unreachable, both distributions sit entirely above 15 units and the
    min-keypoint-distance panel shows the contact frames that do exist. The
    per-fly panel should show the female's rejections dominated by
    CONTAINMENT (a mask-fit problem, by design) rather than by identity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e = np.asarray(d["sep_edges_units"], float)
    x = np.arange(len(e))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, (kall, kadm, thrline, title) in enumerate([
            ("centroid_sep_all_frames", "centroid_sep_admitted", 15.0,
             "inter-fly CENTROID separation"),
            ("min_keypoint_dist_all", "min_keypoint_dist_admitted", 5.0,
             "min inter-fly KEYPOINT distance")]):
        va = np.asarray(d[kall], float); vd = np.asarray(d[kadm], float)
        ax[i].bar(x - 0.2, va / va.sum(), 0.4, color="0.6", label=f"all frames (n={int(va.sum())})")
        ax[i].bar(x + 0.2, vd / vd.sum(), 0.4, color="tab:orange",
                  label=f"admitted anchors (n={int(vd.sum())})")
        ax[i].axvline(np.searchsorted(e, thrline) - 0.5, color="tab:red", ls="--",
                      label=f"contact cut {thrline:g} units")
        ax[i].set_xticks(x)
        ax[i].set_xticklabels([f"{v:g}" for v in e], rotation=45, fontsize=7)
        ax[i].set_xlabel("units (0.1 mm), bin lower edge")
        ax[i].set_ylabel("fraction of frames")
        ax[i].set_title(title, fontsize=10)
        ax[i].legend(fontsize=7)
    gates = ["exist", "step", "reproj", "contain", "nonfinite", "identity"]
    w = 0.38
    for j, sex in enumerate(["female", "male"]):
        c = d["per_fly_rejects"].get(sex, {})
        n = max(c.get("_n", 1), 1)
        ax[2].bar(np.arange(len(gates)) + (j - 0.5) * w, [c.get(g, 0) / n for g in gates], w,
                  label=f"{sex} (admitted {c.get('_admitted', 0) / n:.0%})",
                  color=("tab:purple" if sex == "female" else "tab:green"))
    ax[2].set_xticks(np.arange(len(gates)))
    ax[2].set_xticklabels(gates, rotation=30, fontsize=8)
    ax[2].set_ylabel("fraction of fly-frames rejected")
    ax[2].set_title(f"per-gate rejection by fly sex ({d['n_bouts']} bouts)", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
