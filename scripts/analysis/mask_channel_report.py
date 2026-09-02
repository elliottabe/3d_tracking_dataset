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
  * TAIL statistics (p50/p90/p95/p99/max, frac >10px, frac >20px) overall and
    per sex -- a mean hides the failure mode this pipeline actually suffers
    from, which is a small number of large misses on the female
  * the CLEAN-RECORDING subset: only recordings whose manifest split is "val"
    (never contributed a training annotation). The rest of the val split comes
    from "mixed" recordings whose other frames WERE trained on, so their
    numbers are optimistic for both arms.
  * a per-keypoint A/B delta table + figures, so a wing- or tarsus-specific
    regression is visible rather than averaged away

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


# viz/core/colors.py::keypoint_groups lumps Wing* in with Scutellum under
# "thorax", which is the right coarse grouping for overlays but hides exactly
# the regression this A/B has to be able to see. These extra name-derived
# groups are additive -- colors.py is untouched, so every existing figure keeps
# its grouping -- and are resolved BY NAME, never by integer index.
def extra_keypoint_groups(kp_names):
    import re
    g = {"wings": [], "tarsi": [], "leg_proximal": [], "body_axis": []}
    for i, n in enumerate(kp_names):
        if n.startswith("Wing"):
            g["wings"].append(i)
        elif re.match(r"T[1-3][LR]_(TaT1|TaT3|TaTip)$", n):
            g["tarsi"].append(i)
        elif re.match(r"T[1-3][LR]_(ThxCx|Tro|FeTi|TiTa)$", n):
            g["leg_proximal"].append(i)
        elif n in ("Antenna_Base", "EyeL", "EyeR", "Scutellum", "Abd_A4", "Abd_tip"):
            g["body_axis"].append(i)
    return {k: v for k, v in g.items() if v}


def tail_stats(err, vis, mask=None):
    """Distribution of the per-(annotation, keypoint) error, not just its mean.
    A model can win on the mean and still be the one that throws a tarsus 60px
    across the frame on the female -- that is what breaks triangulation
    downstream, so the tail is the number that matters here."""
    import numpy as np
    e, v = (err, vis) if mask is None else (err[mask], vis[mask])
    vals = e[v.astype(bool)]
    if vals.size == 0:
        return {"n": 0}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(vals.max()),
        "frac_gt_10px": float((vals > 10).mean()),
        "frac_gt_20px": float((vals > 20).mean()),
    }


def clean_recording_mask(recordings, data_root):
    """Boolean row-selector for annotations from recordings whose manifest
    split is exactly "val" -- i.e. recordings that contributed NO training
    annotation. Returns (mask, sorted_recording_names). None if the manifest
    is unreadable (then the caller reports the full-val numbers only)."""
    import json
    import os
    import numpy as np
    path = os.path.join(data_root, "manifest.json")
    if not os.path.exists(path):
        return None, []
    with open(path) as f:
        man = json.load(f)
    clean = {r for r, m in man.get("recordings", {}).items()
             if m.get("split") == "val"}
    present = sorted(clean & set(np.unique(recordings).tolist()))
    return np.isin(recordings, present), present


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
    keys = ("recording", "sex", "behavior", "ann_id", "file_name",
            "kp_names", "val_recording")
    # provenance keys added by mask_channel_eval 2026-09-02; tolerate older
    # .npz files that predate them rather than refusing to aggregate.
    opt = ("kp_order_verified", "ckpt_dir", "data_root")
    data = {"meta": {**{k: z[k] for k in keys},
                     **{k: z[k] for k in opt if k in z.files}}}
    for cond in conditions:
        pred, gt, vis = z[f"pred_{cond}"], z[f"gt_{cond}"], z[f"vis_{cond}"]
        data[cond] = {"err": ((pred - gt) ** 2).sum(-1) ** 0.5, "vis": vis}
    return data, conditions


def report_one(npz_path, label, out_dir, data_root=None):
    import numpy as np
    from viz.core.colors import keypoint_groups

    data, conditions = load_conditions(npz_path)
    meta = data["meta"]
    kp_names = [str(x) for x in meta["kp_names"]]
    val_recording = str(meta["val_recording"])
    groups = keypoint_groups(kp_names)
    groups = {**groups, **extra_keypoint_groups(kp_names)}
    male_mask = meta["sex"] == "male"
    female_sex_mask = meta["sex"] == "female"
    courtship_mask = meta["behavior"] == "courtship"
    general_mask = meta["behavior"] == "general"
    rec_mask = ((meta["recording"] == val_recording) if val_recording
                else np.zeros(len(male_mask), bool))

    root = data_root or str(meta.get("data_root", "") or "")
    clean_mask, clean_recs = (clean_recording_mask(meta["recording"], root)
                              if root else (None, []))

    subsets = {"all": None, "male": male_mask, "female": female_sex_mask,
               "courtship_behavior": courtship_mask,
               "general_behavior": general_mask}
    if val_recording:
        subsets[f"recording:{val_recording}"] = rec_mask
    if clean_mask is not None:
        subsets["clean_all"] = clean_mask
        subsets["clean_male"] = clean_mask & male_mask
        subsets["clean_female"] = clean_mask & female_sex_mask
        # The complement matters as much as the subset here: on this split the
        # only CLEAN female recording is the easiest one in the whole val set,
        # and every HARD female recording is a "mixed" one whose other frames
        # were trained on. Splitting them keeps that from being averaged into
        # a single "female" number that means two different things.
        subsets["mixed_female"] = (~clean_mask) & female_sex_mask
        subsets["mixed_male"] = (~clean_mask) & male_mask

    out = {"label": label, "npz": str(npz_path),
           "kp_order_verified": bool(meta.get("kp_order_verified", False)),
           "ckpt_dir": str(meta.get("ckpt_dir", "") or ""),
           "data_root": root,
           "clean_recordings": clean_recs,
           "n_annotations": int(len(male_mask)),
           "n_female_annotations": int(female_sex_mask.sum()),
           "n_male_annotations": int(male_mask.sum()),
           "conditions": {}}
    for cond in conditions:
        err, vis = data[cond]["err"], data[cond]["vis"]
        c = {}
        # --- headline scalars (kept under their historical key names so any
        # older consumer of this JSON keeps working)
        c["overall"], c["overall_n"] = mpjpe(err, vis)
        c["female_recording"], c["female_recording_n"] = mpjpe(err, vis, rec_mask)
        c["male_sex"], c["male_sex_n"] = mpjpe(err, vis, male_mask)
        c["female_sex"], c["female_sex_n"] = mpjpe(err, vis, female_sex_mask)
        c["courtship_behavior"], c["courtship_behavior_n"] = mpjpe(err, vis, courtship_mask)
        c["general_behavior"], c["general_behavior_n"] = mpjpe(err, vis, general_mask)
        c["subsets"] = {k: tail_stats(err, vis, m) for k, m in subsets.items()}
        c["per_group"] = {}
        c["per_group_female"] = {}
        for g, idx in groups.items():
            idx = np.asarray(idx)
            v, n = mpjpe(err[:, idx], vis[:, idx])
            c["per_group"][g] = {"mpjpe": v, "n": n}
            vf, nf = mpjpe(err[:, idx][female_sex_mask], vis[:, idx][female_sex_mask])
            c["per_group_female"][g] = {"mpjpe": vf, "n": nf}
        c["per_keypoint"] = {}
        c["per_keypoint_female"] = {}
        for k, name in enumerate(kp_names):
            v, n = mpjpe(err[:, k], vis[:, k])
            c["per_keypoint"][name] = {"mpjpe": v, "n": n}
            vf, nf = mpjpe(err[female_sex_mask, k], vis[female_sex_mask, k])
            c["per_keypoint_female"][name] = {"mpjpe": vf, "n": nf}
        out["conditions"][cond] = c

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json_path = Path(out_dir) / f"{label}_report.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {json_path}")
    return out, groups, kp_names


def print_summary(out):
    label = out["label"]
    print(f"\n=== {label} ===")
    print(f"    ckpt {out.get('ckpt_dir','?')}")
    print(f"    keypoint-order guard VERIFIED: {out.get('kp_order_verified')}"
          f"  |  {out.get('n_annotations')} anns "
          f"({out.get('n_female_annotations')} F / {out.get('n_male_annotations')} M)")
    if out.get("clean_recordings"):
        print(f"    clean (manifest split=='val') recordings: "
              f"{', '.join(out['clean_recordings'])}")
    for cond, c in out["conditions"].items():
        print(f"[{cond}] overall {c['overall']:.3f}px (n={c['overall_n']}) | "
              f"FEMALE-sex {c['female_sex']:.3f}px (n={c['female_sex_n']}) | "
              f"male-sex {c['male_sex']:.3f}px | "
              f"courtship {c['courtship_behavior']:.3f}px | "
              f"general {c['general_behavior']:.3f}px")
        for name, st in c["subsets"].items():
            if not st.get("n"):
                continue
            print(f"    {name:26s} mean {st['mean']:7.3f}  p50 {st['p50']:6.2f} "
                  f"p90 {st['p90']:7.2f} p95 {st['p95']:7.2f} p99 {st['p99']:8.2f} "
                  f"max {st['max']:8.1f}  >10px {100*st['frac_gt_10px']:5.1f}% "
                  f">20px {100*st['frac_gt_20px']:5.1f}%  (n={st['n']})")
        for g, v in c["per_group"].items():
            print(f"    group {g:14s}: all {v['mpjpe']:7.3f}px  "
                  f"female {c['per_group_female'][g]['mpjpe']:7.3f}px")


def _resolve_arm(all_out, spec):
    """'label/condition' -> (display_name, condition_dict). Errors loudly on a
    typo rather than silently picking the first arm -- this whole comparison
    exists because a silently-wrong selector produced confident numbers once."""
    label, _, cond = spec.partition("/")
    for out in all_out:
        if out["label"] == label:
            if cond not in out["conditions"]:
                raise SystemExit(
                    f"--arm {spec!r}: label {label!r} has conditions "
                    f"{sorted(out['conditions'])}, not {cond!r}")
            return spec, out["conditions"][cond], out
    raise SystemExit(f"--arm {spec!r}: no --label {label!r} among "
                     f"{[o['label'] for o in all_out]}")


def make_figure(all_out, out_dir, arms=None):
    """Figures for the head-to-head.

    STATED EXPECTATION (write it down before looking -- CLAUDE.md):
      * per_group_mpjpe.png -- if the two checkpoints are genuinely equivalent
        the bars pair up within a px in every group INCLUDING wings and tarsi.
        A bar that moves only in `wings` or only in `tarsi` is a body-part
        specific regression that the overall mean would have hidden.
      * per_keypoint_ab.png -- sorted signed delta (armB - armA) by keypoint
        NAME. If the winner wins everywhere the bars are one-signed; a
        two-signed fan means the mean is an average of a win and a loss and
        the headline number is not the whole story.
      * error_cdf.png -- the tail. Two models with the same mean can differ by
        several percent in "fraction of keypoints missed by >20px", which is
        what actually breaks triangulation downstream. Expect the curves to
        separate in the top decile if the means differ at all; if the means
        differ and the CDFs do NOT separate, the difference is a uniform
        shift, not a tail fix.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    series = []
    for out in all_out:
        for cond, c in out["conditions"].items():
            series.append((f"{out['label']}/{cond}", c))
    if not series:
        return []
    written = []

    # --- 1. per-body-group, all + female ------------------------------------
    group_names = list(series[0][1]["per_group"].keys())
    x = np.arange(len(group_names))
    width = 0.8 / max(len(series), 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=False)
    for ax, key, ttl in ((axes[0], "per_group", "all val annotations"),
                         (axes[1], "per_group_female", "FEMALE annotations only")):
        for i, (name, c) in enumerate(series):
            vals = [c[key][g]["mpjpe"] for g in group_names]
            b = ax.bar(x + i * width, vals, width, label=name)
            ax.bar_label(b, fmt="%.1f", fontsize=6, padding=1)
        ax.set_xticks(x + width * (len(series) - 1) / 2)
        ax.set_xticklabels(group_names, rotation=20, ha="right")
        ax.set_ylabel("MPJPE (px)")
        ax.set_title(ttl)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Per-body-group 2D MPJPE, red_data_3d_v5_valfix val "
                 "(groups resolved by keypoint NAME)")
    fig.tight_layout()
    path = Path(out_dir) / "per_group_mpjpe.png"
    fig.savefig(path, dpi=150); plt.close(fig); written.append(path)
    print(f"wrote {path}")

    # --- 2. per-keypoint A/B delta -----------------------------------------
    if arms and len(arms) == 2:
        (na, ca, _oa) = _resolve_arm(all_out, arms[0])
        (nb, cb, _ob) = _resolve_arm(all_out, arms[1])
        for key, tag in (("per_keypoint", "all"), ("per_keypoint_female", "female")):
            names = list(ca[key].keys())
            va = np.array([ca[key][n]["mpjpe"] for n in names])
            vb = np.array([cb[key][n]["mpjpe"] for n in names])
            delta = vb - va
            order = np.argsort(delta)
            fig, ax = plt.subplots(figsize=(13, 6))
            cols = ["tab:green" if d < 0 else "tab:red" for d in delta[order]]
            ax.barh(np.arange(len(names)), delta[order], color=cols)
            ax.set_yticks(np.arange(len(names)))
            ax.set_yticklabels([names[i] for i in order], fontsize=6)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel(f"MPJPE delta (px):  {nb}  minus  {na}"
                          f"   [green = {nb} better]")
            ax.set_title(f"Per-keypoint 2D error delta -- {tag} annotations\n"
                         f"{na} vs {nb}, red_data_3d_v5_valfix val")
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            path = Path(out_dir) / f"per_keypoint_ab_{tag}.png"
            fig.savefig(path, dpi=150); plt.close(fig); written.append(path)
            print(f"wrote {path}")

    # --- 3. error distribution / tail --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, sub, ttl in ((axes[0], "all", "all val annotations"),
                         (axes[1], "female", "FEMALE annotations only")):
        for name, c in series:
            st = c["subsets"].get(sub, {})
            if not st.get("n"):
                continue
            xs = [st["p50"], st["p90"], st["p95"], st["p99"], st["max"]]
            ax.plot([50, 90, 95, 99, 100], xs, marker="o", label=
                    f"{name}  (mean {st['mean']:.2f}px, >20px {100*st['frac_gt_20px']:.1f}%)")
        ax.set_xlabel("percentile of per-keypoint error")
        ax.set_ylabel("error (px)")
        ax.set_yscale("log")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("2D error TAIL (log px) -- the mean is not the failure mode")
    fig.tight_layout()
    path = Path(out_dir) / "error_tail.png"
    fig.savefig(path, dpi=150); plt.close(fig); written.append(path)
    print(f"wrote {path}")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", action="append", required=True,
                    help="repeatable; one .npz from mask_channel_eval.py")
    ap.add_argument("--label", action="append", required=True,
                    help="repeatable; one label per --npz, same order")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default=None,
                    help="override the data root recorded in the .npz; used "
                         "only to read manifest.json for the clean-recording "
                         "subset")
    ap.add_argument("--arm", action="append", default=None,
                    help="'label/condition'; give exactly two to emit the "
                         "per-keypoint A/B delta figures")
    a = ap.parse_args(argv)
    if len(a.npz) != len(a.label):
        ap.error("--npz and --label must repeat the same number of times")
    if a.arm and len(a.arm) != 2:
        ap.error("--arm must be given exactly twice (or not at all)")

    all_out = []
    for npz_path, label in zip(a.npz, a.label):
        out, groups, kp_names = report_one(npz_path, label, a.out_dir, a.data_root)
        print_summary(out)
        all_out.append(out)

    if a.arm:
        (na, ca, _), (nb, cb, _) = (_resolve_arm(all_out, a.arm[0]),
                                    _resolve_arm(all_out, a.arm[1]))
        print(f"\n=== HEAD-TO-HEAD  {na}  vs  {nb} ===")
        for sub in ("all", "male", "female", "clean_all", "clean_male",
                    "clean_female", "mixed_female", "mixed_male",
                    "courtship_behavior", "general_behavior"):
            sa, sb = ca["subsets"].get(sub, {}), cb["subsets"].get(sub, {})
            if not (sa.get("n") and sb.get("n")):
                continue
            print(f"  {sub:20s} mean {sa['mean']:7.3f} -> {sb['mean']:7.3f} px "
                  f"({sb['mean'] - sa['mean']:+7.3f}, "
                  f"{100*(sb['mean']/sa['mean'] - 1):+6.1f}%)   "
                  f"p95 {sa['p95']:7.2f} -> {sb['p95']:7.2f}   "
                  f">20px {100*sa['frac_gt_20px']:5.1f}% -> {100*sb['frac_gt_20px']:5.1f}%")
        worst = sorted(((cb["per_keypoint_female"][k]["mpjpe"]
                         - ca["per_keypoint_female"][k]["mpjpe"], k)
                        for k in ca["per_keypoint_female"]), reverse=True)
        print(f"  FEMALE per-keypoint, worst 8 for {nb}: " +
              ", ".join(f"{k} {d:+.2f}px" for d, k in worst[:8]))
        print(f"  FEMALE per-keypoint, best 8 for {nb}:  " +
              ", ".join(f"{k} {d:+.2f}px" for d, k in worst[-8:][::-1]))

    make_figure(all_out, a.out_dir, arms=a.arm)


if __name__ == "__main__":
    main()
