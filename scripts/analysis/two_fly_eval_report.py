#!/usr/bin/env python3
"""Two-fly (overlap) breakdown of a mask-channel eval .npz -- LOCAL, no GPU.

Why this exists: `red_data_3d_v5`'s val split contained ZERO two-fly frames, so
every detector number ever quoted -- 5.915 px, the 31% gain over v4, the mask
ablation's +21.9% -- was measured on single-fly frames only. That is precisely
blind to this detector family's documented weakness: ~6% worse where the two
flies OVERLAP, 41% BETTER when apart, and 92% of the frames it loses are
close-interaction. `red_data_3d_v5_valfix` holds `2026_04_07_11_33_33` out whole
to fix that (181 two-fly val images). This script reports the overlap subset.

CONTAMINATION WARNING: only checkpoints TRAINED on the valfix split may be
evaluated here. The old-split checkpoints (`v5_s70_bal_augdef_full`,
`mask_ablation_zeroed_v5s70`) trained WITH `2026_04_07_11_33_33` in train, so
scoring them on these frames is scoring on training data. The script refuses
unless --allow-contaminated is passed.

Usage:
    python scripts/analysis/two_fly_eval_report.py \
        --npz figures/2026-08-31-mask-ablation/v5vf_maskon.npz \
        --data-root /gscratch/.../red_data_3d_v5_valfix --condition populated
"""
from __future__ import annotations
import argparse, json, collections
import numpy as np

CONTAMINATED = {"v5_s70_bal_augdef_full", "mask_ablation_zeroed_v5s70"}


def mpjpe(err, vis):
    return float((err * vis).sum() / max(vis.sum(), 1)) if vis.sum() else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--condition", default="populated")
    ap.add_argument("--label", default=None)
    ap.add_argument("--allow-contaminated", action="store_true")
    a = ap.parse_args(argv)
    label = a.label or a.npz

    if any(c in a.npz for c in CONTAMINATED) and not a.allow_contaminated:
        raise SystemExit(
            f"REFUSING: {label} looks like an OLD-SPLIT checkpoint, which trained WITH "
            f"2026_04_07_11_33_33. Scoring it on these two-fly frames is scoring on its own "
            f"training data. Pass --allow-contaminated only if you intend that.")

    z = np.load(a.npz, allow_pickle=True)
    pred, gt, vis = z[f"pred_{a.condition}"], z[f"gt_{a.condition}"], z[f"vis_{a.condition}"]
    err = np.linalg.norm(pred - gt, axis=-1)
    ann_id = np.asarray(z["ann_id"]); sex = np.asarray(z["sex"]).astype(str)
    bbox = np.asarray(z["bbox"], float)

    d = json.load(open(f"{a.data_root}/annotations/instances_val.json"))
    a2i = {an["id"]: an["image_id"] for an in d["annotations"]}
    per_img = collections.Counter(an["image_id"] for an in d["annotations"])
    img_of = np.array([a2i.get(int(i), -1) for i in ann_id])
    n_flies = np.array([per_img.get(int(i), 0) for i in img_of])

    two, one = n_flies == 2, n_flies == 1
    print(f"\n=== {label}  [condition={a.condition}] ===")
    print(f"{'subset':<26}{'n_anns':>8}{'MPJPE px':>11}")
    for nm, m in (("TWO-FLY (overlap)", two), ("single-fly", one), ("all", np.ones_like(two))):
        print(f"{nm:<26}{int(m.sum()):>8}{mpjpe(err[m], vis[m]):>11.3f}")
    print(f"\n{'two-fly by sex':<26}{'n_anns':>8}{'MPJPE px':>11}")
    for s in ("male", "female"):
        m = two & (sex == s)
        if m.sum(): print(f"  {s:<24}{int(m.sum()):>8}{mpjpe(err[m], vis[m]):>11.3f}")

    # inter-fly separation, from the two bbox centres of each two-fly image
    cen = np.stack([bbox[:, 0] + bbox[:, 2] / 2, bbox[:, 1] + bbox[:, 3] / 2], 1)
    byimg = collections.defaultdict(list)
    for k in np.where(two)[0]:
        byimg[int(img_of[k])].append(k)
    seps = {}
    for im, ks in byimg.items():
        if len(ks) == 2:
            seps[im] = float(np.linalg.norm(cen[ks[0]] - cen[ks[1]]))
    if seps:
        sv = np.array([seps[int(img_of[k])] for k in np.where(two)[0]
                       if int(img_of[k]) in seps])
        ki = np.array([k for k in np.where(two)[0] if int(img_of[k]) in seps])
        print(f"\n{'separation bucket (px)':<26}{'n_anns':>8}{'MPJPE px':>11}")
        for lo, hi in ((0, 150), (150, 250), (250, 350), (350, 10000)):
            m = (sv >= lo) & (sv < hi)
            if m.sum():
                idx = ki[m]
                name = f"{lo}-{hi}" if hi < 10000 else f"{lo}+"
                print(f"  {name:<24}{int(m.sum()):>8}{mpjpe(err[idx], vis[idx]):>11.3f}")
        print(f"  (separation range {sv.min():.0f}-{sv.max():.0f} px, median {np.median(sv):.0f})")


if __name__ == "__main__":
    main()
