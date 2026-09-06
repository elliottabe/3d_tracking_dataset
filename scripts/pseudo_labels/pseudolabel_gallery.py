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

from viz.core.colors import PALETTE, keypoint_groups, leg_chains  # noqa: E402

OVERHEAD_CAM = "Cam2012630"
SIDE_CAM = "Cam2012855"
PANEL_W, PANEL_H = 420, 260
PER_PAGE = 10                       # framesets per page (brief: 10 rows x 4 columns:
                                     # [overhead full][overhead zoom][side full][side zoom])
REJECT_THRESHOLD = 0.03             # spec SS3.3

# ---- v2 render knobs (user feedback 2026-09-06): small dots, drawn skeleton,
# a zoomed crop per camera. None of this touches the stratified draw below --
# see the CSV-identity test and the docstring's INVARIANT.
MARKER_RADIUS_FULL = 1              # was 3; big dots were obscuring the skeleton
MARKER_RADIUS_ZOOM = 2              # slightly larger in the upsampled zoom crop
ZOOM_PAD_FRAC = 0.4                 # grow the visible-keypoint bbox by 40% (x1.4) per side
ZOOM_MIN_SIDE = 160                 # floor crop side (native px, before the x2 upsample)
ZOOM_UPSAMPLE = 2                   # cv2.INTER_CUBIC upsample factor applied to the crop
CROP_RECT_COLOR = (255, 255, 255)   # thin rectangle on the full panel marking the zoom crop
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


def _iter_reviewable(coco, include_partners=False):
    """Every frameset worth spending review budget on.

    Existence negatives (`role == "negative"`, `fly_id is None`) carry no
    keypoints and are never reviewed. By default only `role == "anchor"`
    framesets are sampled: a `role == "partner"` frameset is the SAME fly a
    few frames away (its T=2 pairing partner), and reviewing both would
    double-spend a fixed review budget on near-duplicate frames rather than
    covering more of the stratified space. Pass `include_partners=True`
    (CLI `--include-partners`) to opt into reviewing partner framesets too."""
    for key, fsv in coco["framesets"].items():
        if fsv.get("role") == "negative" or fsv.get("fly_id") is None:
            continue
        if not include_partners and fsv.get("role") != "anchor":
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


def _label_prefix(row):
    """`"<rec> [b<bout>] f<frame> fly<fly> <sex>"` -- `bout` is OMITTED
    (never rendered as a bare 'b') when the frameset carries no bout number,
    which real single-fly/free-running pseudo-labels won't."""
    parts = [row["recording"]]
    if row["bout"] is not None:
        parts.append(f"b{row['bout']}")
    parts += [f"f{row['frame']}", f"fly{row['host_fly']}", row["host_sex"]]
    return " ".join(parts)


def _placeholder_panel(row, cam_name, reason):
    """A missing/unreadable camera still carries the row's own identity --
    the reviewer needs to know WHICH frameset a placeholder belongs to, not
    just which camera failed."""
    canvas = np.full((PANEL_H, PANEL_W, 3), 40, np.uint8)
    cv2.rectangle(canvas, (0, PANEL_H - 18), (PANEL_W, PANEL_H), (0, 0, 0), -1)
    cv2.putText(canvas, f"{_label_prefix(row)} {cam_name}: {reason}", (4, PANEL_H - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _body_wing_edges(kp_names):
    """Head/wing/abdomen chain edges, ported from viz/views/sidebyside.py's
    `_skeleton_edges` -- the repo's only other place that names a body/wing
    topology (viz/core/colors.py itself defines no body-chain helper, only
    `leg_chains`/`keypoint_groups`) -- so this is REUSED, not invented. Its
    per-leg "root each chain at Scutellum" connectors are deliberately
    dropped: those are that view's own decoration, not part of this
    head/wing/abdomen chain, and leg topology here comes from `leg_chains`
    alone (see `_skeleton_edges_colored`)."""
    idx = {n: i for i, n in enumerate(kp_names)}

    def E(a, b):
        return (idx[a], idx[b]) if a in idx and b in idx else None

    edges = [E("EyeL", "Antenna_Base"), E("EyeR", "Antenna_Base"),
             E("Antenna_Base", "Scutellum"),
             E("Scutellum", "WingL_base"), E("WingL_base", "WingL_V12"), E("WingL_V12", "WingL_V13"),
             E("Scutellum", "WingR_base"), E("WingR_base", "WingR_V12"), E("WingR_V12", "WingR_V13"),
             E("Scutellum", "Abd_A4"), E("Abd_A4", "Abd_tip")]
    return [e for e in edges if e is not None]


def _skeleton_edges_colored(kp_names, idx2group):
    """Every skeleton line segment to draw: the six `leg_chains` (proximal ->
    distal) plus `_body_wing_edges`, each pre-coloured by its DISTAL
    endpoint's `keypoint_groups` colour (viz.core.colors.PALETTE). A leg-chain
    edge has both endpoints in the "legs" group so this is unambiguous there;
    a body/wing edge that crosses a group boundary (e.g. Antenna_Base(head)
    -> Scutellum(thorax)) takes the colour of the group the line enters."""
    edges = []
    for chain in leg_chains(kp_names).values():
        edges += list(zip(chain[:-1], chain[1:]))
    edges += _body_wing_edges(kp_names)
    white = (255, 255, 255)
    return [(a, b, PALETTE.get(idx2group.get(b, idx2group.get(a)), white)) for a, b in edges]


def _draw_skeleton(canvas, uv, vis, edges_colored, idx2group, radius):
    """Lines BEFORE markers (so a marker never gets painted over by a line),
    each line only between two keypoints both visible in THIS camera."""
    shown = {i: (int(round(uv[i, 0])), int(round(uv[i, 1])))
             for i in range(len(vis)) if vis[i]}
    for a, b, color in edges_colored:
        if a in shown and b in shown:
            cv2.line(canvas, shown[a], shown[b], color, 1, cv2.LINE_AA)
    for i, pt in shown.items():
        color = PALETTE.get(idx2group.get(i), (255, 255, 255))
        cv2.circle(canvas, pt, radius, color, -1, cv2.LINE_AA)


def _load_frame(export_root, coco, images_by_id, anns_by_id, row, cam_name):
    """Locate + read the native BGR frame and its keypoint annotation (if
    any) for one (frameset, camera) BY the export's own `file_name`/`ann_ids`
    -- never a positional camera guess (module docstring). Returns
    `(img, ann, ann_id, reason)`: `img` is None with a `reason`
    ("missing"/"unreadable") exactly when `_placeholder_panel` is the right
    render for both the full and the zoom slot."""
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
        return None, None, None, "missing"
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None, "unreadable"
    return img, ann, ann_id_found, None


def _zoom_crop_box(ann, img_w, img_h):
    """Square crop window (native px, NOT yet clamped to the frame) around
    the host's VISIBLE 2D keypoints: bbox padded 40% per side (x1.4), floored
    at ZOOM_MIN_SIDE so a tightly folded pose still gets a legible zoom. No
    visible keypoints (or no ann at all) falls back to the image centre at
    the floor size -- the same "nothing to centre on" fallback
    singlefly_p3b_pass.py's contact_sheet uses for an all-invisible frame."""
    cx = cy = None
    side = float(ZOOM_MIN_SIDE)
    if ann is not None:
        kp = np.asarray(ann["keypoints"], np.float32).reshape(-1, 3)
        vis = kp[:, 2] > 0
        if vis.any():
            xs, ys = kp[vis, 0], kp[vis, 1]
            w = max(float(xs.max() - xs.min()), 1.0)
            h = max(float(ys.max() - ys.min()), 1.0)
            cx, cy = float((xs.min() + xs.max()) / 2), float((ys.min() + ys.max()) / 2)
            side = max(w * (1 + ZOOM_PAD_FRAC), h * (1 + ZOOM_PAD_FRAC), float(ZOOM_MIN_SIDE))
    if cx is None:
        cx, cy = img_w / 2, img_h / 2
    side = int(round(side))
    half = side / 2
    x0 = int(np.clip(cx - half, 0, max(img_w - side, 0)))
    y0 = int(np.clip(cy - half, 0, max(img_h - side, 0)))
    return x0, y0, side


def _full_panel(img, mask, ann, edges_colored, idx2group, row, cam_name, crop_box=None):
    h, w = img.shape[:2]
    canvas = cv2.resize(img, (PANEL_W, PANEL_H))
    sx, sy = PANEL_W / w, PANEL_H / h

    if mask is not None:
        mr = cv2.resize(mask * np.uint8(255), (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, PALETTE["mask"], 1)

    if ann is not None:
        kp = np.asarray(ann["keypoints"], np.float32).reshape(-1, 3)
        uv = kp[:, :2] * np.array([sx, sy], np.float32)
        vis = kp[:, 2] > 0
        _draw_skeleton(canvas, uv, vis, edges_colored, idx2group, MARKER_RADIUS_FULL)

    if crop_box is not None:
        x0, y0, side = crop_box
        p0 = (int(round(x0 * sx)), int(round(y0 * sy)))
        p1 = (int(round((x0 + side) * sx)), int(round((y0 + side) * sy)))
        cv2.rectangle(canvas, p0, p1, CROP_RECT_COLOR, 1)

    host_color = PALETTE["fly1"] if row["host_sex"] == "male" else PALETTE["fly0"]
    label = f"{_label_prefix(row)} sep={_fmt_sep(row)}u {cam_name}"
    cv2.rectangle(canvas, (0, PANEL_H - 18), (PANEL_W, PANEL_H), (0, 0, 0), -1)
    cv2.putText(canvas, label, (4, PANEL_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                host_color, 1, cv2.LINE_AA)
    return canvas


def _zoom_panel(img, mask, ann, edges_colored, idx2group, crop_box):
    """Crop (padded, floored, clamped+letterboxed like contact_sheet's own
    crop-around-the-fly), upsample x2 with INTER_CUBIC, draw the skeleton +
    markers (radius MARKER_RADIUS_ZOOM) and mask outline at THAT resolution
    -- then, only at the very end, resize into the fixed PANEL_W x PANEL_H
    montage cell (same "any native size -> the fixed cell" convention
    `_full_panel` already uses for the whole camera frame). No label text
    here -- it stays on the adjacent full panel per the brief."""
    img_h, img_w = img.shape[:2]
    x0, y0, side = crop_box
    cw, ch = min(side, img_w - x0), min(side, img_h - y0)
    pad = np.full((side, side, 3), 20, np.uint8)
    pad[:ch, :cw] = img[y0:y0 + ch, x0:x0 + cw]
    up = cv2.resize(pad, (side * ZOOM_UPSAMPLE, side * ZOOM_UPSAMPLE), interpolation=cv2.INTER_CUBIC)

    if mask is not None:
        mcrop = np.zeros((side, side), np.uint8)
        mcrop[:ch, :cw] = mask[y0:y0 + ch, x0:x0 + cw]
        mup = cv2.resize(mcrop * np.uint8(255), (side * ZOOM_UPSAMPLE, side * ZOOM_UPSAMPLE),
                         interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mup, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(up, contours, -1, PALETTE["mask"], 1)

    if ann is not None:
        kp = np.asarray(ann["keypoints"], np.float32).reshape(-1, 3)
        uv = (kp[:, :2] - np.array([x0, y0], np.float32)) * ZOOM_UPSAMPLE
        vis = kp[:, 2] > 0
        _draw_skeleton(up, uv, vis, edges_colored, idx2group, MARKER_RADIUS_ZOOM)

    return cv2.resize(up, (PANEL_W, PANEL_H))


def _panel(export_root, coco, idx2group, edges_colored, images_by_id, anns_by_id, row, cam_name,
           crop_box=None):
    """The full-frame panel alone (no zoom) -- kept as its own entry point
    (used directly by tests) on top of the same `_load_frame`/`_full_panel`
    building blocks `_panel_pair` composes for the real gallery render."""
    img, ann, ann_id_found, reason = _load_frame(export_root, coco, images_by_id, anns_by_id,
                                                  row, cam_name)
    if img is None:
        return _placeholder_panel(row, cam_name, reason)
    mask = _mask_outline(export_root, row["recording"], cam_name, row["frame"], ann_id_found)
    return _full_panel(img, mask, ann, edges_colored, idx2group, row, cam_name, crop_box)


def _panel_pair(export_root, coco, idx2group, edges_colored, images_by_id, anns_by_id, row, cam_name):
    """(full panel, zoom panel) for one (frameset, camera) -- a single
    `_load_frame` read is shared by both so a missing/unreadable camera loads
    once and both slots get the identical placeholder."""
    img, ann, ann_id_found, reason = _load_frame(export_root, coco, images_by_id, anns_by_id,
                                                  row, cam_name)
    if img is None:
        ph = _placeholder_panel(row, cam_name, reason)
        return ph, ph.copy()
    mask = _mask_outline(export_root, row["recording"], cam_name, row["frame"], ann_id_found)
    crop_box = _zoom_crop_box(ann, img.shape[1], img.shape[0])
    full = _full_panel(img, mask, ann, edges_colored, idx2group, row, cam_name, crop_box)
    zoom = _zoom_panel(img, mask, ann, edges_colored, idx2group, crop_box)
    return full, zoom


def render_page(export_root, coco, kp_names, images_by_id, anns_by_id, rows, out_path):
    """Per frameset row: [overhead full (+ crop rect)][overhead zoom]
    [side full (+ crop rect)][side zoom] -- page width is 4 panels wide
    (was 2), i.e. doubled, so a 10-row page's canvas is
    `(PANEL_H * n, PANEL_W * 4, 3)`."""
    groups = keypoint_groups(kp_names)
    idx2group = {i: g for g, idxs in groups.items() for i in idxs}
    edges_colored = _skeleton_edges_colored(kp_names, idx2group)
    n = max(len(rows), 1)
    canvas = np.full((PANEL_H * n, PANEL_W * 4, 3), 20, np.uint8)
    for i, row in enumerate(rows):
        oh_full, oh_zoom = _panel_pair(export_root, coco, idx2group, edges_colored,
                                       images_by_id, anns_by_id, row, OVERHEAD_CAM)
        sd_full, sd_zoom = _panel_pair(export_root, coco, idx2group, edges_colored,
                                       images_by_id, anns_by_id, row, SIDE_CAM)
        r0, r1 = i * PANEL_H, (i + 1) * PANEL_H
        canvas[r0:r1, 0 * PANEL_W:1 * PANEL_W] = oh_full
        canvas[r0:r1, 1 * PANEL_W:2 * PANEL_W] = oh_zoom
        canvas[r0:r1, 2 * PANEL_W:3 * PANEL_W] = sd_full
        canvas[r0:r1, 3 * PANEL_W:4 * PANEL_W] = sd_zoom
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


def generate(export_root, out_dir, n, seed, include_partners=False):
    coco, kp_names, images_by_id, anns_by_id = _load_export(export_root)
    rows = [_row_info(k, v) for k, v in _iter_reviewable(coco, include_partners=include_partners)]
    if not rows:
        raise ValueError(f"{export_root}: no reviewable "
                         f"({'non-negative' if include_partners else 'anchor'}) framesets")
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
    (non-dropped) ANCHOR still references that same partner frame.

    `referenced` is deliberately restricted to `role == "anchor"` framesets:
    only an anchor's own `partners` dict expresses a real T=2 pairing need. A
    `role == "partner"` frameset's own `partners` field is not expected to be
    populated by the writer, but even if it were, a kept partner must not be
    able to keep ANOTHER partner alive -- that would let a chain of
    partner-referencing-partner survive a dropped anchor with nothing left
    that actually needs it."""
    referenced = set()
    for key, v in fs.items():
        if key in drop_keys or v.get("role") != "anchor":
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


def _wing_indices(kp_names):
    """Indices of `kp_names` whose name starts with `Wing` -- BY NAME, from
    the export's own `keypoint_names` (never a positional guess)."""
    return [i for i, n in enumerate(kp_names) if n.startswith("Wing")]


def _missing_wing_keys(fs, anns_by_id, wing_idx, min_cams):
    """ANCHOR frameset keys where some wing keypoint is visible (v > 0) in
    fewer than `min_cams` cameras (user decision 2026-09-06). Applied to
    EVERY anchor in the export (`fs`), not just the reviewed sample -- the
    rule is a data-quality gate at apply time, independent of the human
    review. `min_cams <= 0` (the default, "off") or no wing keypoints at all
    short-circuits to no drops."""
    if min_cams <= 0 or not wing_idx:
        return set()
    drop = set()
    for key, v in fs.items():
        if v.get("role") != "anchor":
            continue
        counts = [0] * len(wing_idx)
        for ann_id in v.get("ann_ids") or []:
            if ann_id is None:
                continue
            ann = anns_by_id.get(ann_id)
            if ann is None:
                continue
            kp = np.asarray(ann["keypoints"], np.float32).reshape(-1, 3)
            for j, widx in enumerate(wing_idx):
                if kp[widx, 2] > 0:
                    counts[j] += 1
        if any(c < min_cams for c in counts):
            drop.add(key)
    return drop


def _reverse_partner_refs(fs):
    """`partner frameset key -> {anchor keys that reference it}`, over every
    ANCHOR in `fs` (mirrors `_cascade_partner_drop`'s own `referenced`
    computation) -- used only to attribute a cascaded-partner drop back to
    the seed set(s) that caused it, for reporting."""
    refs = collections.defaultdict(set)
    for key, v in fs.items():
        if v.get("role") != "anchor":
            continue
        rec, fly = v.get("recording"), v.get("fly_id")
        if fly is None:
            continue
        for pf in (v.get("partners") or {}).values():
            refs[f"{rec}/Frame_{pf}/fly{fly}"].add(key)
    return refs


def apply_review(export_root, csv_path, *, max_reject_frac=REJECT_THRESHOLD,
                 cell_drop_frac=REJECT_THRESHOLD, override_reason=None,
                 reject_missing_wing_cams=0):
    """Spec SS3.3, by default exactly: overall reject fraction > 3% -> abort
    (exit 2), print the per-gate breakdown, write NOTHING. Else drop the
    rejected framesets, then drop every whole stratum cell whose OWN reject
    rate (over the reviewed sample) exceeds 3% -- applied to every matching
    frameset in the WHOLE export, not just the sampled ones (spec: "dropped
    entirely rather than thinned"). Rewrites `annotations/instances_train.json`,
    `manifest.json["review"]`, and a `review_summary.json` beside the CSV.
    Returns an exit code (0 ok, 2 aborted).

    `max_reject_frac`/`cell_drop_frac` override the spec's two 3% thresholds
    (`cell_drop_frac=None` disables the whole-cell drop entirely, keeping
    every cell). Either differing from `REJECT_THRESHOLD` is an OVERRIDE of
    the spec gate and requires `override_reason` (a human-readable string,
    e.g. citing the decision date) -- recorded, with the measured overall and
    per-cell reject fractions, under `overrides` in both `manifest.json` and
    `review_summary.json` so the decision is auditable after the fact.

    `reject_missing_wing_cams` (0 = off) additionally drops every ANCHOR
    frameset whose annotations have some `Wing*`-named keypoint visible in
    fewer than that many cameras, export-wide (`_missing_wing_keys`) -- an
    independent data-quality rule, not a spec-gate override, so it has no
    effect on `max_reject_frac`/`cell_drop_frac`/`overrides`."""
    is_override = (max_reject_frac != REJECT_THRESHOLD) or (cell_drop_frac != REJECT_THRESHOLD)
    if is_override and not (override_reason and override_reason.strip()):
        raise ValueError(
            "--override-reason is required whenever --max-reject-frac or --cell-drop-frac "
            f"differs from the spec default ({REJECT_THRESHOLD})")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    if total == 0:
        raise ValueError(f"{csv_path}: no rows to review")
    rejected_rows = [r for r in rows if (r.get("verdict") or "").strip().lower() == "reject"]
    reject_frac = len(rejected_rows) / total

    if reject_frac > max_reject_frac:
        print(f"[pseudolabel_gallery] overall reject fraction {reject_frac:.1%} "
              f"({len(rejected_rows)}/{total}) exceeds {max_reject_frac:.0%} -- NOT "
              f"applying the review; tighten the offending gate and regenerate.")
        _print_gate_breakdown(rejected_rows)
        return 2

    per_cell_total = collections.Counter(r["cell"] for r in rows)
    per_cell_reject = collections.Counter(r["cell"] for r in rejected_rows)
    per_cell_reject_frac = {c: per_cell_reject.get(c, 0) / tot for c, tot in per_cell_total.items()}
    if cell_drop_frac is None:
        dropped_cells = []
    else:
        dropped_cells = sorted(c for c, frac in per_cell_reject_frac.items()
                               if frac > cell_drop_frac)
    dropped_cell_set = set(dropped_cells)

    ann_path = os.path.join(export_root, "annotations", "instances_train.json")
    coco = json.load(open(ann_path))
    fs = coco["framesets"]
    anns_by_id = {a["id"]: a for a in coco["annotations"]}
    kp_names = list(coco["keypoint_names"])

    rejected_keys = {r["frameset"] for r in rejected_rows}
    review_seed = set()
    for key, v in fs.items():
        if key in rejected_keys:
            review_seed.add(key)
            continue
        if v.get("role") == "negative" or v.get("fly_id") is None:
            continue
        if _cell_of(v.get("stratum") or {}) in dropped_cell_set:
            review_seed.add(key)

    wing_idx = _wing_indices(kp_names)
    wing_seed = _missing_wing_keys(fs, anns_by_id, wing_idx, reject_missing_wing_cams)

    drop_keys = _cascade_partner_drop(fs, review_seed | wing_seed)
    cascaded_extra = drop_keys - review_seed - wing_seed
    reverse_refs = _reverse_partner_refs(fs) if cascaded_extra else {}
    review_cascaded = wing_cascaded = 0
    for pk in cascaded_extra:
        refs = reverse_refs.get(pk, set())
        from_review = bool(refs & review_seed)
        from_wing = bool(refs & wing_seed)
        review_cascaded += int(from_review)
        wing_cascaded += int(from_wing)

    _clean_dangling_partners(fs, drop_keys)
    for key in drop_keys:
        fs.pop(key, None)
    with open(ann_path, "w") as f:
        json.dump(coco, f)

    review_block = {"file": csv_path, "n": total, "reject_frac": round(reject_frac, 4),
                    "dropped_strata": dropped_cells}
    if is_override:
        review_block["overrides"] = {
            "max_reject_frac": max_reject_frac,
            "cell_drop_frac": cell_drop_frac,
            "reason": override_reason,
            "measured": {
                "overall_reject_frac": round(reject_frac, 4),
                "per_cell_reject_frac": {c: round(v, 4) for c, v in per_cell_reject_frac.items()},
            },
        }
    summary = dict(review_block, n_dropped_framesets=len(drop_keys))
    if reject_missing_wing_cams > 0:
        summary["missing_wing_dropped"] = {"anchors": len(wing_seed), "partners": wing_cascaded}
    with open(os.path.join(os.path.dirname(csv_path), "review_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    manifest_path = os.path.join(export_root, "manifest.json")
    manifest = json.load(open(manifest_path))
    manifest["review"] = review_block
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"[pseudolabel_gallery] applied review: {len(rejected_rows)}/{total} rejected "
          f"({reject_frac:.1%}), {len(dropped_cells)} whole strata dropped {dropped_cells}, "
          f"{review_cascaded} cascaded partners from review/{len(review_seed)} review-dropped "
          f"anchors; {len(wing_seed)} missing-wing anchors dropped ({wing_cascaded} cascaded "
          f"partners); {len(drop_keys)} framesets removed total.")
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
    ap.add_argument("--include-partners", action="store_true",
                    help="also sample role=='partner' framesets (the same fly a few "
                         "frames later) into the review; default is anchors only, since "
                         "a partner would double-spend the reviewer's budget on a "
                         "near-duplicate of its own anchor")
    ap.add_argument("--apply-review", default=None, metavar="CSV",
                    help="apply a filled-in review.csv (spec SS3.3): overall reject "
                         "fraction > 3%% aborts and writes nothing; else drop rejected "
                         "framesets and any whole stratum whose own reject rate exceeds 3%%")
    ap.add_argument("--max-reject-frac", type=float, default=REJECT_THRESHOLD,
                    help=f"override the spec's overall reject-fraction abort threshold "
                         f"(default {REJECT_THRESHOLD}); differing from the default is an "
                         f"OVERRIDE and requires --override-reason")
    ap.add_argument("--cell-drop-frac", default=str(REJECT_THRESHOLD), metavar="FRAC|none",
                    help=f"override the spec's per-cell reject-fraction threshold that drops "
                         f"a whole stratum cell (default {REJECT_THRESHOLD}); 'none' disables "
                         f"the whole-cell drop entirely (every cell is kept); differing from "
                         f"the default is an OVERRIDE and requires --override-reason")
    ap.add_argument("--override-reason", default=None,
                    help="required whenever --max-reject-frac or --cell-drop-frac differs "
                         "from the spec default; recorded verbatim in manifest.json[\"review\"]"
                         "/review_summary.json[\"overrides\"] for audit")
    ap.add_argument("--reject-missing-wing-cams", type=int, default=0, metavar="N",
                    help="drop anchor framesets (and cascade their partners) where some "
                         "Wing*-named keypoint is visible in fewer than N cameras, export-wide; "
                         "0 (default) disables the rule")
    args = ap.parse_args(argv)

    if args.apply_review:
        cell_drop_frac = (None if str(args.cell_drop_frac).strip().lower() == "none"
                          else float(args.cell_drop_frac))
        return apply_review(args.export, args.apply_review,
                            max_reject_frac=args.max_reject_frac,
                            cell_drop_frac=cell_drop_frac,
                            override_reason=args.override_reason,
                            reject_missing_wing_cams=args.reject_missing_wing_cams)
    if not args.out:
        ap.error("--out is required to generate a gallery")
    summary = generate(args.export, args.out, args.n, args.seed,
                       include_partners=args.include_partners)
    print(f"[pseudolabel_gallery] wrote {summary['n']} framesets over {summary['n_pages']} "
          f"page(s) to {args.out} (cells: {summary['cells']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
