#!/usr/bin/env python3
"""Local (no-GPU) aggregation + figures for the mask-channel-ablation task.

Reads the .npz produced by scripts/analysis/mask_channel_eval.py (per-sample,
per-keypoint pred/gt/vis for one checkpoint under one or more mask
conditions) and reports:
  * overall + female (val_recording) MPJPE per condition
  * per-keypoint MPJPE per condition
  * per-body-group MPJPE (viz/core/colors.py::keypoint_groups) per condition
  * male-vs-female MPJPE per condition (ds.sex, resolved w/ manifest fallback)
  * a behavior-tag proxy for "close interaction" (courtship vs general) --
    NOT a per-frame proximity split (see module docstring caveat below)

Usage:
    python scripts/analysis/mask_channel_report.py \\
        --npz figures/2026-08-31-mask-ablation/v5_s70_bal_augdef_full.npz \\
        --label baseline \\
        [--npz figures/.../mask_ablation_retrain.npz --label mask_ablation_retrain] \\
        --out-dir figures/2026-08-31-mask-ablation

CLOSE-INTERACTION CAVEAT (read before quoting the behavior-tag split):
red_data_3d_v5's val COCO structure crops ONE fly per annotation/image (0 of
1502 val images carry >1 annotation; every sampled val mask .npz carries
exactly 1 mask instance -- the other fly, even when present in the source
recording, is not co-segmented in what ships with this val set), and the
paired courtship recordings that look same-session (e.g.
2026_05_27_11_56_05 female / _11_57_05 male) turn out to cover disjoint frame
ranges, not simultaneous both-fly captures. So there is no per-frame
body-length proximity signal available in this val set the way
scripts/benchmark/select_bouts.py's `proximity_bl < 2.0` threshold uses on
full 3D-tracked bouts. The best available proxy here is the per-annotation
`behavior` tag ("courtship" vs "general") -- courtship recordings are where
the other fly is SOMEWHERE in the session, not necessarily touching/near in
that exact frame. Report this split as a coarse context tag, not a
close-interaction measurement, and say so every time it's quoted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def mpjpe(err, vis, mask=None):
    """Mean of err (…,) weighted by vis (bool), optionally restricted to mask
    (a boolean row-selector). NaN with an explicit note if nothing is visible."""
    import numpy as np
    e, v = (err, vis) if mask is None else (err[mask], vis[mask])
    v = v.astype(np.float64)
    denom = v.sum()
    if denom <= 0:
        return float("nan"), 0
    return float((e * v).sum() / denom), int(denom)


def load_conditions(npz_path):
    import numpy as np
    z = np.load(npz_path, allow_pickle=True)
    conditions = sorted({k.split("_", 1)[1] for k in z.files if k.startswith("pred_")})
    data = {"meta": {k: z[k] for k in
                     ("recording", "sex", "behavior", "ann_id", "file_name",
                      "kp_names", "val_recording")}}
    for cond in conditions:
        pred, gt, vis = z[f"pred_{cond}"], z[f"gt_{cond}"], z[f"vis_{cond}"]
        data[cond] = {"err": ((pred - gt) ** 2).sum(-1) ** 0.5, "vis": vis}
    return data, conditions


def report_one(npz_path, label, out_dir):
    import numpy as np
    from viz.core.colors import keypoint_groups

    data, conditions = load_conditions(npz_path)
    meta = data["meta"]
    kp_names = [str(x) for x in meta["kp_names"]]
    val_recording = str(meta["val_recording"])
    groups = keypoint_groups(kp_names)
    fem_mask = meta["recording"] == val_recording
    male_mask = meta["sex"] == "male"
    female_sex_mask = meta["sex"] == "female"
    courtship_mask = meta["behavior"] == "courtship"
    general_mask = meta["behavior"] == "general"

    out = {"label": label, "npz": str(npz_path), "conditions": {}}
    for cond in conditions:
        err, vis = data[cond]["err"], data[cond]["vis"]
        c = {}
        c["overall"], c["overall_n"] = mpjpe(err, vis)
        c["female_recording"], c["female_recording_n"] = mpjpe(err, vis, fem_mask)
        c["male_sex"], c["male_sex_n"] = mpjpe(err, vis, male_mask)
        c["female_sex"], c["female_sex_n"] = mpjpe(err, vis, female_sex_mask)
        c["courtship_behavior"], c["courtship_behavior_n"] = mpjpe(err, vis, courtship_mask)
        c["general_behavior"], c["general_behavior_n"] = mpjpe(err, vis, general_mask)
        c["per_group"] = {}
        for g, idx in groups.items():
            idx = np.asarray(idx)
            g_err, g_vis = err[:, idx], vis[:, idx]
            c["per_group"][g] = {"mpjpe": mpjpe(g_err, g_vis)[0], "n": mpjpe(g_err, g_vis)[1]}
        c["per_keypoint"] = {}
        for k, name in enumerate(kp_names):
            v, n = mpjpe(err[:, k], vis[:, k])
            c["per_keypoint"][name] = {"mpjpe": v, "n": n}
        out["conditions"][cond] = c

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json_path = Path(out_dir) / f"{label}_report.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {json_path}")
    return out, groups, kp_names


def print_summary(out):
    label = out["label"]
    print(f"\n=== {label} ===")
    for cond, c in out["conditions"].items():
        print(f"[{cond}] overall {c['overall']:.3f}px (n={c['overall_n']}) | "
              f"female-recording {c['female_recording']:.3f}px (n={c['female_recording_n']}) | "
              f"male-sex {c['male_sex']:.3f}px | female-sex {c['female_sex']:.3f}px | "
              f"courtship {c['courtship_behavior']:.3f}px | general {c['general_behavior']:.3f}px")
        for g, v in c["per_group"].items():
            print(f"    group {g:10s}: {v['mpjpe']:.3f}px (n={v['n']})")


def make_figure(all_out, out_dir):
    """Grouped bar chart: per-group MPJPE, one bar cluster per (label,
    condition), for a visual read of where mask information (if any) lives."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    series = []
    for out in all_out:
        for cond, c in out["conditions"].items():
            series.append((f"{out['label']}/{cond}", c["per_group"]))
    if not series:
        return None
    group_names = list(series[0][1].keys())
    x = np.arange(len(group_names))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, pg) in enumerate(series):
        vals = [pg[g]["mpjpe"] for g in group_names]
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(group_names)
    ax.set_ylabel("MPJPE (px)")
    ax.set_title("Per-body-group 2D MPJPE, red_data_3d_v5 val\n"
                 "(mask-channel ablation -- see mask-channel-ablation.md)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(out_dir) / "per_group_mpjpe.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", action="append", required=True,
                    help="repeatable; one .npz from mask_channel_eval.py")
    ap.add_argument("--label", action="append", required=True,
                    help="repeatable; one label per --npz, same order")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    if len(a.npz) != len(a.label):
        ap.error("--npz and --label must repeat the same number of times")

    all_out = []
    for npz_path, label in zip(a.npz, a.label):
        out, groups, kp_names = report_one(npz_path, label, a.out_dir)
        print_summary(out)
        all_out.append(out)

    make_figure(all_out, a.out_dir)


if __name__ == "__main__":
    main()
