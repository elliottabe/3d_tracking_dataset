"""Render a labelled grid of copy-paste synthetic two-fly composites for
visual verification of ``jarvis_jax.data.copy_paste.CopyPasteCenterDetectDataset``.

Expectation, stated before generating anything (per CLAUDE.md): if the
paste/blend/separation-sampling logic is correct, each panel should show
TWO plainly-fly-shaped animals on the SAME platform/lighting as each other
(same camera by construction), with a soft edge around the pasted one (no
visible hard silhouette seam), spanning a range of separations from
near-total overlap up to the real courtship range (~267-336px) and beyond.
A visible rectangular/hard edge around the pasted fly, a fly floating off
the platform, or two panels that obviously come from different cameras
(different lighting/elevation) would mean the synthesis is teaching the
model to detect paste artifacts, not flies -- exactly the failure mode this
verification step exists to catch.

Draws round-robin across every camera in the train split, plus (when
present) at least one host fly labelled "female" in `manifest.json`, so the
grid cannot accidentally be all one viewpoint or all one sex.

Usage:
    python scripts/viz/render_copypaste_samples.py \\
        --root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5_valfix \\
        --n 12 --out-dir figures/2026-08-31-copypaste/grid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def build_samples(root, n, *, seed=0, image_size=320, p=1.0, **cp_kwargs):
    from jarvis_jax.data.copy_paste import CopyPasteCenterDetectDataset
    from jarvis_jax.data.v5_centerdetect import V5CenterDetectDataset

    train_ds = V5CenterDetectDataset(root, "train", image_size=image_size)
    cp = CopyPasteCenterDetectDataset(train_ds, p=p, seed=seed, **cp_kwargs)

    single_idx = train_ds.single_fly_indices()
    by_cam = defaultdict(list)
    for i in single_idx:
        cam = train_ds.file_names[i].split("/")[1]
        by_cam[cam].append(i)
    cams = sorted(by_cam)

    female_idx = [i for i in single_idx if train_ds.sexes[i] == ["female"]]

    hosts = []
    if female_idx:
        hosts.append(female_idx[0])
    k = 0
    while len(hosts) < n:
        cam = cams[k % len(cams)]
        pool = by_cam[cam]
        i = pool[(k // len(cams)) % len(pool)]
        if i not in hosts:
            hosts.append(i)
        k += 1
        if k > 10 * n + len(cams):
            break

    samples = []
    for j, host_idx in enumerate(hosts[:n]):
        composite_full, _img_out, _centers_out, meta = cp.sample_with_meta(
            host_idx, seed=seed * 100000 + j)
        cam = train_ds.file_names[host_idx].split("/")[1]
        is_female = host_idx in female_idx
        samples.append((composite_full, meta, cam, is_female))
    return samples


def render_grid(samples, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(samples)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.2 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, (composite, meta, cam, is_female) in zip(axes, samples):
        ax.imshow(composite)
        host = meta["host_center_xy"]
        target = meta["target_center_xy"]
        ax.plot(*host, "o", ms=10, mfc="none", mec="#2ecc71", mew=2)
        ax.text(host[0], host[1] - 12, "REAL", color="#2ecc71", fontsize=8,
               ha="center", weight="bold")
        ax.plot(*target, "o", ms=10, mfc="none", mec="#ff8c00", mew=2)
        ax.text(target[0], target[1] - 12, "SYNTH", color="#ff8c00", fontsize=8,
               ha="center", weight="bold")
        top = "donor-on-top" if meta["top_is_donor"] else "host-on-top"
        rec_note = "same-rec" if meta["same_recording"] else "cross-rec"
        title = (f"{cam} {rec_note} {top} | sep sampled={meta['sep_sampled_px']:.0f}px "
                f"achieved={meta['sep_achieved_px']:.0f}px" +
                (" | FEMALE host" if is_female else ""))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sep-low", type=float, default=None)
    ap.add_argument("--sep-high", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cp_kwargs = {}
    if args.sep_low is not None:
        cp_kwargs["sep_low"] = args.sep_low
    if args.sep_high is not None:
        cp_kwargs["sep_high"] = args.sep_high

    samples = build_samples(args.root, args.n, seed=args.seed, **cp_kwargs)
    out_png = os.path.join(args.out_dir, "copypaste_grid.png")
    render_grid(samples, out_png)

    seps = np.array([m["sep_achieved_px"] for _, m, _, _ in samples])
    cams_used = sorted({c for _, _, c, _ in samples})
    n_female = sum(1 for *_, f in samples if f)
    meta_out = {
        "n": len(samples),
        "cameras_used": cams_used,
        "n_female_hosts": n_female,
        "sep_achieved_px": seps.tolist(),
        "sep_achieved_mean_px": float(seps.mean()),
        "sep_achieved_median_px": float(np.median(seps)),
        "samples": [m for _, m, _, _ in samples],
    }
    with open(os.path.join(args.out_dir, "copypaste_grid_meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"wrote {out_png}")
    print(f"cameras used: {cams_used} ; female hosts: {n_female}/{len(samples)}")
    print(f"achieved separation px: mean={seps.mean():.1f} median={np.median(seps):.1f} "
          f"min={seps.min():.1f} max={seps.max():.1f}")


if __name__ == "__main__":
    main()
