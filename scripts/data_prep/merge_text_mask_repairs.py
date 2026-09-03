#!/usr/bin/env python3
"""Repair the box-prompted masks that failed, using the text-prompted ones.

WHAT FAILED AND WHY. The box prompt produced 24 masks (of 18,242) holding under
60% of their own annotation's visible keypoints. Rendered, the failure is
unambiguous on 20 of them: the mask is a blob pinned to the FRAME EDGE while
the annotation's fly sits unmasked in the middle. The text prompt ("insect",
full frame, assigned by keypoint containment) fixes exactly those: 0-40%
containment becomes 0.72-0.96.

THE OTHER 4 ARE NOT REPAIRED AND SHOULD NOT BE. All four are
2026_06_10_15_05_02 (headless_56_42_1_female), one dark low-contrast moment
with the animal's limbs widely splayed. Full-frame text, 448-crop text,
text+box, and six prompt variants ALL land at 0.40-0.53 -- the mask was never
the variable. A SAM3 mask is the BODY silhouette (legs, wings and antennae
fall outside it, see predict/session_frameset), so `kp-in-mask` is
POSE-CONFOUNDED: a splayed fly scores low against a perfectly good mask. They
are gated to matched=False instead, so the loader returns an honest zero
channel rather than a mask we cannot vouch for either way.
(Caveat kept deliberately: on Frame_2120 the outside set includes Scutellum,
the thorax, which leg-splay does NOT explain. So "correct mask, bad metric" is
the likely reading for these 4, not a proven one -- another reason to gate
rather than adopt.)

PER ANNOTATION, NOT PER FRAME. A two-fly frame can have one good box mask and
one bad one; replacing the whole npz would throw away the good one. Each
annotation independently takes whichever source contains more of its own
keypoints, and is gated if neither clears the threshold.

Originals are copied to masks_repair_backup/ before anything is written.

    python scripts/data_prep/merge_text_mask_repairs.py --root <v12 root> \
        --frame-list <bad_frames.txt>
"""
import argparse, collections, json, os, shutil
from pathlib import Path

import numpy as np


def kp_inside(mask, kp):
    v = kp[:, 2] > 0
    if not v.any():
        return float("nan")
    H, W = mask.shape
    return float(mask[np.clip(kp[v, 1].astype(int), 0, H - 1),
                      np.clip(kp[v, 0].astype(int), 0, W - 1)].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--frame-list", required=True)
    ap.add_argument("--text-dir", default="masks_text")
    ap.add_argument("--backup-dir", default="masks_repair_backup")
    ap.add_argument("--min-inside", type=float, default=0.6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)

    ann = {}
    for split in ("train", "val"):
        p = root / "annotations" / f"instances_{split}.json"
        if p.exists():
            for x in json.loads(p.read_text())["annotations"]:
                ann[int(x["id"])] = np.asarray(x["keypoints"], float).reshape(-1, 3)

    keys = [l.strip() for l in open(a.frame_list) if l.strip()]
    stats = collections.Counter()
    rows = []
    for k in keys:
        rec, cam, fr = k.split("/")
        bp = root / "masks" / rec / cam / f"{fr}.npz"
        tp = root / a.text_dir / rec / cam / f"{fr}.npz"
        if not bp.exists() or not tp.exists():
            stats["skipped_missing"] += 1
            continue
        with np.load(bp) as z:
            bm, bids, bmt = z["masks"], z["ann_ids"], z["matched"]
        with np.load(tp) as z:
            tm, tids = z["masks"], z["ann_ids"]
        tix = {int(i): j for j, i in enumerate(tids)}

        out_m, out_id, out_mt = [], [], []
        for j, aid in enumerate(bids):
            aid = int(aid)
            kp = ann.get(aid)
            b_in = kp_inside(bm[j].astype(bool), kp) if kp is not None else float("nan")
            t_in = (kp_inside(tm[tix[aid]].astype(bool), kp)
                    if (kp is not None and aid in tix) else float("nan"))
            use, src, val = bm[j].astype(bool), "box", b_in
            if not np.isnan(t_in) and (np.isnan(b_in) or t_in > b_in):
                use, src, val = tm[tix[aid]].astype(bool), "text", t_in
            if np.isnan(val) or val < a.min_inside:
                out_m.append(np.zeros_like(use)); out_mt.append(False)
                stats["gated"] += 1; src = f"GATED (best {val:.2f})"
            else:
                out_m.append(use); out_mt.append(True)
                stats[f"kept_{src}"] += 1
            out_id.append(aid)
            rows.append((k, aid, b_in, t_in, src))

        if not a.dry_run:
            bk = root / a.backup_dir / rec / cam
            bk.mkdir(parents=True, exist_ok=True)
            if not (bk / f"{fr}.npz").exists():
                shutil.copy2(bp, bk / f"{fr}.npz")
            np.savez_compressed(bp, masks=np.stack(out_m),
                                ann_ids=np.asarray(out_id, np.int64),
                                matched=np.asarray(out_mt, bool))

    print(f"{'frame':52s}{'ann':>8}{'box':>7}{'text':>7}   chosen")
    for k, aid, b, t, src in rows:
        print(f"{k:52s}{aid:>8}{b:>7.2f}{(t if not np.isnan(t) else -1):>7.2f}   {src}")
    print(f"\n{dict(stats)}")
    if a.dry_run:
        print("DRY RUN -- nothing written")


if __name__ == "__main__":
    main()
