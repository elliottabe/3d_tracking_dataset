"""Can ONE conf2/conf1 threshold keep real 2nd flies and drop phantoms?

`fp_rate(ratio>=0.5)`, the metric the trainer prints, is NOT the right gate
and reading it as one produced a wrong conclusion on 2026-08-31: the focal
background term appeared to cut false positives 0.594 -> 0.375, but the
threshold-free AUC of the same statistic got WORSE (0.824 -> 0.676). A loss
change that rescales every confidence downward moves peaks under a FIXED 0.5
line without making them any more distinguishable from real ones.

What the pipeline actually needs is a decision rule. Decode the top-2 peaks
and take r = conf2/conf1. On a single-fly frame r should be low (that second
peak is a phantom); on a two-fly frame r should be high (that second peak is
a fly). This script reports the two distributions, their AUC, and the best
single threshold with BOTH of its errors -- missed real flies and admitted
phantoms -- which is what a downstream triangulation gate would live with.

Judge any CenterDetect loss/data change by AUC and by the (miss, admit) pair
at a matched miss rate. Do not judge it by fp@0.5.

    python -m jarvis_jax.scripts.eval_centerdetect_separability \
        --root .../red_data_3d_v5_valfix \
        --ckpt uniform=.../cd_bg10/ckpt/epoch_030 \
        --ckpt focal=.../cd_focal_bg10/ckpt/epoch_008 \
        --out figures/<topic>/separability
"""
import argparse
import json
import os

import numpy as np

IMAGE_SIZE = 320


def _ratios(model, ds, idx, batch=16):
    from jarvis_jax.scripts.viz_centerdetect_fp import _predict
    out = []
    for i0 in range(0, len(idx), batch):
        _, _, _, c = _predict(model, ds, list(idx[i0:i0 + batch]))
        out.append(c[:, 1] / np.maximum(c[:, 0], 1e-9))
    return np.concatenate(out)


def _auc(neg, pos):
    """P(a two-fly frame's ratio ranks above a single-fly frame's). Computed
    from the inversion count so ties and duplicates behave, and NOT folded
    with max(auc, 1-auc) -- an arm that ranks phantoms ABOVE real flies is a
    genuine failure and must be allowed to score below 0.5."""
    s = np.r_[neg, pos]
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    order = np.argsort(s, kind="mergesort")
    y = y[order]
    inversions = y.cumsum()[y == 0].sum()          # positives ranked below a negative
    return 1.0 - inversions / (len(neg) * len(pos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ckpt", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--out", default=None, help="basename; writes .json and .png")
    ap.add_argument("--max-single", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from jarvis_jax.data.v5_centerdetect import V5CenterDetectDataset
    from jarvis_jax.scripts.viz_centerdetect_fp import _restore

    ds = V5CenterDetectDataset(args.root, args.split, image_size=IMAGE_SIZE,
                               cache_images=False)
    nf = np.array([int(ds[i][2].sum()) for i in range(len(ds))])
    single, two = np.flatnonzero(nf == 1), np.flatnonzero(nf >= 2)
    rng = np.random.default_rng(args.seed)
    if len(single) > args.max_single:
        single = np.sort(rng.choice(single, args.max_single, replace=False))
    print(f"{len(single)} single-fly, {len(two)} two-fly {args.split} frames\n")
    if len(two) == 0:
        raise SystemExit("no two-fly frames in this split -- the AUC would be "
                         "undefined and every phantom number meaningless. This "
                         "is exactly the hole red_data_3d_v5's original val "
                         "split had; use a root whose val includes two-fly.")

    print(f"{'arm':<24}{'r|1fly':>9}{'r|2fly':>9}{'AUC':>8}{'thr*':>7}"
          f"{'missreal':>10}{'admitph':>9}")
    res = {}
    for spec in args.ckpt:
        lab, ck = spec.split("=", 1)
        model = _restore(ck)
        r1, r2 = _ratios(model, ds, single), _ratios(model, ds, two)
        del model
        auc = _auc(r1, r2)
        thr, keep, admit = max(((t, (r2 >= t).mean(), (r1 >= t).mean())
                                for t in np.linspace(0.02, 0.98, 97)),
                               key=lambda x: x[1] - x[2])
        print(f"{lab:<24}{np.median(r1):>9.3f}{np.median(r2):>9.3f}{auc:>8.3f}"
              f"{thr:>7.2f}{1 - keep:>10.3f}{admit:>9.3f}")
        res[lab] = dict(ckpt=ck, r1=r1.tolist(), r2=r2.tolist(), auc=float(auc),
                        thr=float(thr), miss=float(1 - keep), admit=float(admit))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(f"{args.out}.json", "w") as f:
            json.dump(res, f)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 3.6),
                                 sharey=True, squeeze=False)
        b = np.linspace(0, 1.05, 36)
        for ax, (lab, d) in zip(axes[0], res.items()):
            ax.hist(d["r1"], bins=b, color="0.45", alpha=.75, density=True,
                    label="single-fly (phantom)")
            ax.hist(d["r2"], bins=b, histtype="step", lw=2.4, color="tab:green",
                    density=True, label="two-fly (real 2nd fly)")
            ax.axvline(d["thr"], color="crimson", ls="--", lw=1.6)
            ax.set_title(f"{lab}\nAUC={d['auc']:.3f}  thr*={d['thr']:.2f}\n"
                         f"miss real {d['miss']:.1%} / admit phantom "
                         f"{d['admit']:.1%}", fontsize=9)
            ax.set_xlabel("2nd-peak confidence ratio  conf2/conf1")
        axes[0][0].set_ylabel("density")
        axes[0][0].legend(fontsize=8, loc="upper center")
        fig.suptitle("Phantom vs real 2nd fly: is there a threshold?\n"
                     "(pulled-apart grey and green = a usable gate; both "
                     "sliding left together = confidence merely rescaled)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .86))
        fig.savefig(f"{args.out}.png", dpi=140)
        print(f"\nwrote {args.out}.json / .png")


if __name__ == "__main__":
    main()
