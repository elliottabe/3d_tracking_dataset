"""Human review gallery for gated P3b pseudo-labels (spec 2026-09-05 SS3.3).

Reads a v12-format pseudo-label export (as written by
`jarvis_jax.data.pseudo_export.write_pseudo_export` /
`scripts/pseudo_labels/extract_p3b_pseudolabels.py`) and renders a stratified
sample of its framesets -- two cameras each, keypoints coloured by anatomical
group, mask outline when available -- to a page-per-10-panels gallery plus an
editable `review.csv`. `--apply-review` reads a filled-in CSV back and gates
training on it exactly per spec SS3.3.

EXPECTATION (state it before looking, CLAUDE.md): every panel shows ONE fly's
keypoints on that fly's own body, inside its own grey mask outline, in both
cameras; the head (red) is at the antennae, the abdomen (magenta) at the
tail, and leg chains do not cross between the two animals. A panel where the
coloured skeleton straddles both bodies, or sits on the arena floor, is a
reject -- that is exactly the pseudo-label failure the 0.3 weight cannot fix.

Run (the real ~300-frameset gallery, deferred until the re-lifted export
lands -- see the task-3 report for why):

    python scripts/pseudo_labels/pseudolabel_gallery.py \\
        --export /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_pseudo_p3b_20260905 \\
        --n 300 --out figures/2026-09-mvq/v2_pseudo/gallery

Then, after Elliott fills in `verdict` in `review.csv`:

    python scripts/pseudo_labels/pseudolabel_gallery.py \\
        --export <same export> --apply-review figures/2026-09-mvq/v2_pseudo/gallery/review.csv

Everything crossing a name boundary is looked up BY NAME: keypoints via the
export's own `keypoint_names` (never a positional guess -- CLAUDE.md's
keypoint-order trap), cameras via each image's own `file_name`
(`<rec>/<cam>/Frame_<n>.jpg`), never a positional "cameras[i]" guess -- the
export does not even carry an explicit camera-order array, by design.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

import cv2
import numpy as np

# repo root, so `import viz.core.colors` resolves the same way other
# scripts/pseudo_labels/ CLIs reach into the repo (see extract_p3b_pseudolabels.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from viz.core.colors import PALETTE, keypoint_groups  # noqa: E402

OVERHEAD_CAM = "Cam2012630"
SIDE_CAM = "Cam2012855"
PANEL_W, PANEL_H = 420, 260
PER_PAGE = 10                       # framesets per page (brief: 10 rows x 2 cam columns)
REJECT_THRESHOLD = 0.03             # spec SS3.3
GATE_NAMES = ("exist", "step_units", "reproj_px", "contain_frac")
CSV_FIELDS = ["frameset", "recording", "bout", "frame", "host_fly", "host_sex", "contact",
              "wall", "page", "cell", "exist", "step_units", "reproj_px", "contain_frac",
              "verdict", "note"]


# ----------------------------------------------------------------- the export
def _load_export(export_root):
    ann_path = os.path.join(export_root, "annotations", "instances_train.json")
    coco = json.load(open(ann_path))
    kp_names = list(coco["keypoint_names"])
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_id = {a["id"]: a for a in coco["annotations"]}
    return coco, kp_names, images_by_id, anns_by_id


def _parse_key(key):
    """`"<rec>/Frame_<frame>/fly<fly>"` -> (rec, frame, fly), BY the export's
    own key schema -- never an assumed positional split of unrelated fields."""
    rec, fr, flypart = key.rsplit("/", 2)
    return rec, int(fr[len("Frame_"):]), int(flypart[len("fly"):])


def _cell_of(stratum):
    """Stratification key = (host_sex, contact, wall), per the task-3 controller
    ruling -- proportional composition of the export, not the spec's original
    (host_sex x contact/apart/mid x recording) census cell."""
    host_sex = stratum.get("host_sex", "unknown")
    contact = "contact" if stratum.get("contact") else "apart"
    wall = "wall" if stratum.get("wall") else "floor"
    return f"{host_sex}/{contact}/{wall}"


def _iter_reviewable(coco):
    """Every frameset with a real host fly -- existence negatives (`role ==
    "negative"`, `fly_id is None`) carry no keypoints and are not reviewed
    here."""
    for key, fsv in coco["framesets"].items():
        if fsv.get("role") == "negative" or fsv.get("fly_id") is None:
            continue
        yield key, fsv


def _row_info(key, fsv):
    rec, frame, fly = _parse_key(key)
    st = fsv.get("stratum") or {}
    g = fsv.get("gates") or {}
    return {
        "frameset": key, "recording": rec, "bout": fsv.get("bout"),
        "frame": frame, "host_fly": fly,
        "host_sex": st.get("host_sex", "unknown"),
        "contact": bool(st.get("contact")), "wall": bool(st.get("wall")),
        "sep_units": st.get("sep_units"), "kp_dist_units": st.get("kp_dist_units"),
        "exist": g.get("exist"), "step_units": g.get("step_units"),
        "reproj_px": g.get("reproj_px"), "contain_frac": g.get("contain_frac"),
        "cell": _cell_of(st),
    }


# --------------------------------------------------------------- stratified draw
def stratified_alloc(counts, n, rng):
    """Proportional integer allocation of `n` draws over `counts` (cell name ->
    pool size): sums to `min(n, sum(counts))`, never exceeds a cell's own
    pool, and gives every non-empty cell at least 1 when there is room for it
    (task-3 controller ruling)."""
    cells = sorted(counts)
    total = sum(counts.values())
    n = min(int(n), total)
    if n <= 0 or not cells:
        return {c: 0 for c in cells}
    raw = {c: n * counts[c] / total for c in cells}
    alloc = {c: min(int(raw[c]), counts[c]) for c in cells}
    nonempty = [c for c in cells if counts[c] > 0]
    if n >= len(nonempty):
        for c in nonempty:
            if alloc[c] == 0:
                alloc[c] = 1
    for c in cells:
        alloc[c] = min(alloc[c], counts[c])

    deficit = n - sum(alloc.values())
    perm = rng.permutation(len(cells))
    order = [cells[i] for i in perm]
    if deficit > 0:
        order_by_rem = sorted(order, key=lambda c: -(raw[c] - alloc[c]))
        i, guard = 0, 0
        while deficit > 0 and guard < 100000:
            c = order_by_rem[i % len(order_by_rem)]
            if alloc[c] < counts[c]:
                alloc[c] += 1
                deficit -= 1
            i += 1
            guard += 1
    elif deficit < 0:
        floor_by_cell = {c: (1 if n >= len(nonempty) and counts[c] > 0 else 0) for c in cells}
        order_by_rem = sorted(order, key=lambda c: (raw[c] - alloc[c]))
        i, guard = 0, 0
        while deficit < 0 and guard < 100000:
            c = order_by_rem[i % len(order_by_rem)]
            if alloc[c] > floor_by_cell[c]:
                alloc[c] -= 1
                deficit += 1
            i += 1
            guard += 1
    return alloc


def _draw(rows, n, seed):
    rng = np.random.default_rng(seed)
    counts = collections.Counter(r["cell"] for r in rows)
    alloc = stratified_alloc(counts, n, rng)
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)
    picked = []
    for cell, k in alloc.items():
        pool = by_cell[cell]
        if k <= 0:
            continue
        if k >= len(pool):
            picked.extend(pool)
        else:
            idx = rng.choice(len(pool), size=k, replace=False)
            picked.extend(pool[i] for i in idx)
    # grouped by stratum on the page (spec SS3.3), stable within a cell
    picked.sort(key=lambda r: (r["cell"], r["recording"], r["frame"], r["host_fly"]))
    return picked, dict(counts)


# --------------------------------------------------------------------- render
def _mask_outline(export_root, rec, cam, frame, ann_id):
    if ann_id is None:
        return None
    path = os.path.join(export_root, "masks", rec, cam, f"Frame_{frame}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        ids = z["ann_ids"]
        hit = np.nonzero(ids == ann_id)[0]
        if hit.size == 0 or not z["matched"][hit[0]]:
            return None
        return z["masks"][hit[0]].astype(np.uint8)


def _fmt_sep(row):
    sep = row.get("sep_units")
    if sep is None:
        sep = row.get("kp_dist_units")
    return f"{sep:.1f}" if sep is not None else "nan"


def _panel(export_root, coco, idx2group, images_by_id, anns_by_id, row, cam_name):
    fsv = coco["framesets"][row["frameset"]]
    img_path, ann, ann_id_found = None, None, None
    for img_id, ann_id in zip(fsv.get("frames", []), fsv.get("ann_ids", [])):
        im = images_by_id.get(img_id)
        if im is None:
            continue
        _, cam, _ = im["file_name"].split("/")
        if cam == cam_name:
            img_path = os.path.join(export_root, "images", im["file_name"])
            if ann_id is not None:
                ann = anns_by_id.get(ann_id)
                ann_id_found = ann_id
            break

    if img_path is None or not os.path.exists(img_path):
        canvas = np.full((PANEL_H, PANEL_W, 3), 40, np.uint8)
        cv2.putText(canvas, f"{cam_name}: missing", (8, PANEL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    img = cv2.imread(img_path)
    if img is None:
        canvas = np.full((PANEL_H, PANEL_W, 3), 40, np.uint8)
        cv2.putText(canvas, f"{cam_name}: unreadable", (8, PANEL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas
    h, w = img.shape[:2]
    canvas = cv2.resize(img, (PANEL_W, PANEL_H))
    sx, sy = PANEL_W / w, PANEL_H / h

    mask = _mask_outline(export_root, row["recording"], cam_name, row["frame"], ann_id_found)
    if mask is not None:
        mr = cv2.resize(mask * np.uint8(255), (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, PALETTE["mask"], 1)

    if ann is not None:
        kp = np.asarray(ann["keypoints"], np.float32).reshape(-1, 3)
        for i, (u, v, vflag) in enumerate(kp):
            if vflag <= 0:
                continue
            color = PALETTE.get(idx2group.get(i), (255, 255, 255))
            cv2.circle(canvas, (int(round(u * sx)), int(round(v * sy))), 3, color, -1, cv2.LINE_AA)

    host_color = PALETTE["fly1"] if row["host_sex"] == "male" else PALETTE["fly0"]
    bout = "" if row["bout"] is None else row["bout"]
    label = (f"{row['recording']} b{bout} f{row['frame']} fly{row['host_fly']} "
             f"{row['host_sex']} sep={_fmt_sep(row)}u {cam_name}")
    cv2.rectangle(canvas, (0, PANEL_H - 18), (PANEL_W, PANEL_H), (0, 0, 0), -1)
    cv2.putText(canvas, label, (4, PANEL_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                host_color, 1, cv2.LINE_AA)
    return canvas


def render_page(export_root, coco, kp_names, images_by_id, anns_by_id, rows, out_path):
    groups = keypoint_groups(kp_names)
    idx2group = {i: g for g, idxs in groups.items() for i in idxs}
    n = max(len(rows), 1)
    canvas = np.full((PANEL_H * n, PANEL_W * 2, 3), 20, np.uint8)
    for i, row in enumerate(rows):
        left = _panel(export_root, coco, idx2group, images_by_id, anns_by_id, row, OVERHEAD_CAM)
        right = _panel(export_root, coco, idx2group, images_by_id, anns_by_id, row, SIDE_CAM)
        canvas[i * PANEL_H:(i + 1) * PANEL_H, 0:PANEL_W] = left
        canvas[i * PANEL_H:(i + 1) * PANEL_H, PANEL_W:2 * PANEL_W] = right
    cv2.imwrite(out_path, canvas)


def write_review_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "frameset": r["frameset"], "recording": r["recording"],
                "bout": "" if r["bout"] is None else r["bout"],
                "frame": r["frame"], "host_fly": r["host_fly"],
                "host_sex": r["host_sex"], "contact": r["contact"], "wall": r["wall"],
                "page": r["page"], "cell": r["cell"],
                "exist": r.get("exist"), "step_units": r.get("step_units"),
                "reproj_px": r.get("reproj_px"), "contain_frac": r.get("contain_frac"),
                "verdict": "", "note": "",
            })


def generate(export_root, out_dir, n, seed):
    coco, kp_names, images_by_id, anns_by_id = _load_export(export_root)
    rows = [_row_info(k, v) for k, v in _iter_reviewable(coco)]
    if not rows:
        raise ValueError(f"{export_root}: no reviewable (non-negative) framesets")
    picked, counts = _draw(rows, n, seed)
    for i, r in enumerate(picked):
        r["page"] = i // PER_PAGE

    os.makedirs(out_dir, exist_ok=True)
    n_pages = (len(picked) + PER_PAGE - 1) // PER_PAGE if picked else 0
    for p in range(n_pages):
        page_rows = [r for r in picked if r["page"] == p]
        render_page(export_root, coco, kp_names, images_by_id, anns_by_id,
                    page_rows, os.path.join(out_dir, f"page_{p:02d}.png"))
    write_review_csv(picked, os.path.join(out_dir, "review.csv"))
    return {"n": len(picked), "n_pages": n_pages, "cells": counts}


# ------------------------------------------------------------------ apply-review
def _print_gate_breakdown(rejected_rows):
    print(f"[pseudolabel_gallery] {len(rejected_rows)} rejected rows -- per-gate "
          f"value breakdown (min / median / max) so the offending gate can be tightened:")
    for g in GATE_NAMES:
        vals = [float(r[g]) for r in rejected_rows if r.get(g) not in (None, "")]
        if not vals:
            print(f"  {g}: no values")
            continue
        vals = np.asarray(vals, np.float64)
        print(f"  {g}: min={vals.min():.4f} median={np.median(vals):.4f} "
              f"max={vals.max():.4f} n={len(vals)}")


def _cascade_partner_drop(fs, drop_keys):
    """Partners of a dropped anchor are dropped too, UNLESS another kept
    (non-dropped) frameset still references that same partner frame."""
    referenced = set()
    for key, v in fs.items():
        if key in drop_keys:
            continue
        rec, fly = v.get("recording"), v.get("fly_id")
        if fly is None:
            continue
        for pf in (v.get("partners") or {}).values():
            referenced.add(f"{rec}/Frame_{pf}/fly{fly}")

    extra = set()
    for key in list(drop_keys):
        v = fs.get(key)
        if v is None:
            continue
        rec, fly = v.get("recording"), v.get("fly_id")
        if fly is None:
            continue
        for pf in (v.get("partners") or {}).values():
            pk = f"{rec}/Frame_{pf}/fly{fly}"
            pv = fs.get(pk)
            if pv is not None and pv.get("role") == "partner" and pk not in referenced:
                extra.add(pk)
    return drop_keys | extra


def _clean_dangling_partners(fs, drop_keys):
    """Strip a surviving frameset's `partners` entries that point at a
    frameset which no longer exists -- "every remaining frameset key still
    resolves" applies to partner references too."""
    for key, v in fs.items():
        if key in drop_keys:
            continue
        rec, fly = v.get("recording"), v.get("fly_id")
        partners = v.get("partners") or {}
        if not partners or fly is None:
            continue
        kept = {}
        for d, pf in partners.items():
            pk = f"{rec}/Frame_{pf}/fly{fly}"
            if pk in drop_keys or pk not in fs:
                continue
            kept[d] = pf
        v["partners"] = kept


def apply_review(export_root, csv_path):
    """Spec SS3.3, exactly: overall reject fraction > 3% -> abort (exit 2),
    print the per-gate breakdown, write NOTHING. Else drop the rejected
    framesets, then drop every whole stratum cell whose OWN reject rate (over
    the reviewed sample) exceeds 3% -- applied to every matching frameset in
    the WHOLE export, not just the sampled ones (spec: "dropped entirely
    rather than thinned"). Rewrites `annotations/instances_train.json`,
    `manifest.json["review"]`, and a `review_summary.json` beside the CSV.
    Returns an exit code (0 ok, 2 aborted)."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    if total == 0:
        raise ValueError(f"{csv_path}: no rows to review")
    rejected_rows = [r for r in rows if (r.get("verdict") or "").strip().lower() == "reject"]
    reject_frac = len(rejected_rows) / total

    if reject_frac > REJECT_THRESHOLD:
        print(f"[pseudolabel_gallery] overall reject fraction {reject_frac:.1%} "
              f"({len(rejected_rows)}/{total}) exceeds {REJECT_THRESHOLD:.0%} -- NOT "
              f"applying the review; tighten the offending gate and regenerate.")
        _print_gate_breakdown(rejected_rows)
        return 2

    per_cell_total = collections.Counter(r["cell"] for r in rows)
    per_cell_reject = collections.Counter(r["cell"] for r in rejected_rows)
    dropped_cells = sorted(c for c, tot in per_cell_total.items()
                           if per_cell_reject.get(c, 0) / tot > REJECT_THRESHOLD)
    dropped_cell_set = set(dropped_cells)

    ann_path = os.path.join(export_root, "annotations", "instances_train.json")
    coco = json.load(open(ann_path))
    fs = coco["framesets"]

    rejected_keys = {r["frameset"] for r in rejected_rows}
    drop_keys = set()
    for key, v in fs.items():
        if key in rejected_keys:
            drop_keys.add(key)
            continue
        if v.get("role") == "negative" or v.get("fly_id") is None:
            continue
        if _cell_of(v.get("stratum") or {}) in dropped_cell_set:
            drop_keys.add(key)

    drop_keys = _cascade_partner_drop(fs, drop_keys)
    _clean_dangling_partners(fs, drop_keys)
    for key in drop_keys:
        fs.pop(key, None)
    with open(ann_path, "w") as f:
        json.dump(coco, f)

    review_block = {"file": csv_path, "n": total, "reject_frac": round(reject_frac, 4),
                    "dropped_strata": dropped_cells}
    summary = dict(review_block, n_dropped_framesets=len(drop_keys))
    with open(os.path.join(os.path.dirname(csv_path), "review_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    manifest_path = os.path.join(export_root, "manifest.json")
    manifest = json.load(open(manifest_path))
    manifest["review"] = review_block
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"[pseudolabel_gallery] applied review: {len(rejected_rows)}/{total} rejected "
          f"({reject_frac:.1%}), {len(dropped_cells)} whole strata dropped {dropped_cells}, "
          f"{len(drop_keys)} framesets removed.")
    return 0


# ------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="v12-format pseudo-label export root")
    ap.add_argument("--out", default=None,
                    help="gallery output dir (pages + review.csv); required unless --apply-review")
    ap.add_argument("--n", type=int, default=300, help="stratified sample size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apply-review", default=None, metavar="CSV",
                    help="apply a filled-in review.csv (spec SS3.3): overall reject "
                         "fraction > 3%% aborts and writes nothing; else drop rejected "
                         "framesets and any whole stratum whose own reject rate exceeds 3%%")
    args = ap.parse_args(argv)

    if args.apply_review:
        return apply_review(args.export, args.apply_review)
    if not args.out:
        ap.error("--out is required to generate a gallery")
    summary = generate(args.export, args.out, args.n, args.seed)
    print(f"[pseudolabel_gallery] wrote {summary['n']} framesets over {summary['n_pages']} "
          f"page(s) to {args.out} (cells: {summary['cells']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
