"""Acceptance figure for class-balanced sampling on a general_model-derived root.

WHAT THIS CHECKS. `configs/sampling/balanced.yaml` (`sampling=balanced`) asks
`V3Dataset.balanced_weights` to lift rare conditions -- wall, climbing, the
courtship FEMALE -- out of the abundant ones. On every general_model-derived
root that silently could not work: `build_generalmodel_split` deliberately
leaves `behavior` as "unknown" on every annotation (writing it would re-weight
roots already trained against) and emits no `category` at all, so
`class_counts` returned {"unknown": N} and `train_keypoints.run_training`
raised rather than training uniform. `data/v5_2d._resolve_behavior` /
`_resolve_category` fix that by reading the regime from `manifest.json`, which
is where the builder actually writes it.

EXPECTATION IF THE FIX IS CORRECT.
  * `behavior` resolves to the 7 real regimes and `category` to their sex
    cross (10 classes) -- neither is a single "unknown" bucket.
  * Sampled share > data share for every class below ~5% of the data, and
    < data share for the big ones. Bars that track the grey data bars
    one-for-one mean the weights are uniform and the fix did NOT take.
  * `wall_male` (224 anns, 1.22% of train) and `courtship_female` (251,
    1.36%) both rise several-fold; `max_repeat=20` caps the per-annotation
    repeat, so no class should show a repeat factor above it.
  * key=behavior lifts `wall` but CANNOT lift `courtship_female`: under that
    axis the 251 female-courtship annotations sit inside one 5,292-annotation
    `courtship` class. That contrast is the argument for category.

Regenerate:
    python scripts/viz/sampling_balance_check.py \
        --root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v10_wall0902 \
        --out figures/2026-09-02-sampling-balance
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "third_party", "jarvis_jax"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-repeat", type=float, default=20.0)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jarvis_jax.data.v5_2d import V5Dataset

    os.makedirs(a.out, exist_ok=True)
    ds = V5Dataset(a.root, "train")
    n = len(ds)

    record = {"root": a.root, "n_train_annotations": n,
              "alpha": a.alpha, "max_repeat": a.max_repeat, "keys": {}}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, key in zip(axes, ("category", "behavior")):
        labels = getattr(ds, key)
        counts = collections.Counter(labels)
        if set(counts) == {"unknown"}:
            raise SystemExit(
                f"key={key!r} is entirely 'unknown' -- the manifest fallback "
                f"did not take; balanced sampling would be a no-op here.")
        w = ds.balanced_weights(key=key, alpha=a.alpha, max_repeat=a.max_repeat)
        share = collections.defaultdict(float)
        for wi, lab in zip(w, labels):
            share[lab] += float(wi)

        order = sorted(counts, key=lambda c: counts[c])
        data_pct = [100.0 * counts[c] / n for c in order]
        samp_pct = [100.0 * share[c] for c in order]
        repeat = [share[c] * n / counts[c] for c in order]

        record["keys"][key] = {
            c: {"annotations": counts[c],
                "data_pct": 100.0 * counts[c] / n,
                "sampled_pct": 100.0 * share[c],
                "repeat_x": share[c] * n / counts[c]}
            for c in order}

        y = np.arange(len(order))
        ax.barh(y - 0.2, data_pct, height=0.4, color="0.72",
                label="% of annotations (the data)")
        ax.barh(y + 0.2, samp_pct, height=0.4, color="#1f77b4",
                label=f"% of draws (balanced, alpha={a.alpha})")
        for i, (c, r) in enumerate(zip(order, repeat)):
            ax.text(max(data_pct[i], samp_pct[i]) + 0.4, i, f"{r:.1f}x",
                    va="center", fontsize=8, color="#1f77b4")
        # the two classes this change exists for
        for i, c in enumerate(order):
            if c in ("wall_male", "wall", "courtship_female"):
                ax.get_yticklabels()
                ax.axhspan(i - 0.45, i + 0.45, color="orange", alpha=0.13, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{c}  (n={counts[c]})" for c in order], fontsize=9)
        ax.set_xlabel("share of training annotations / of sampled draws (%)")
        ax.set_title(f"balance_key = {key}   ({len(counts)} classes, "
                     f"max_repeat={a.max_repeat})")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Class-balanced sampling -- {os.path.basename(a.root)} train, "
        f"{n} annotations\n"
        f"orange = the wall / courtship-female classes this wiring exists to "
        f"lift;  'Nx' = per-annotation repeat rate vs uniform",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    png = os.path.join(a.out, "sampling_balance.png")
    fig.savefig(png, dpi=130)
    with open(os.path.join(a.out, "sampling_balance.json"), "w") as f:
        json.dump(record, f, indent=2)

    for key, tbl in record["keys"].items():
        print(f"\n=== balance_key={key} ===")
        print(f"{'class':<22}{'anns':>7}{'data%':>8}{'draws%':>9}{'repeat':>9}")
        for c, r in sorted(tbl.items(), key=lambda kv: -kv[1]["annotations"]):
            print(f"{c:<22}{r['annotations']:>7}{r['data_pct']:>8.2f}"
                  f"{r['sampled_pct']:>9.2f}{r['repeat_x']:>8.2f}x")
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
