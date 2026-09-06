"""`scripts/pseudo_labels/pseudolabel_gallery.py` (spec 2026-09-05 SS3.3).

The fake export is built with the SAME writer the real extractor uses
(`write_pseudo_export`), fed by `pseudo_fixtures`, so the gallery reads real
v12-format files rather than a hand-rolled stand-in. Every fixture record is
ONE fly on its own frame (host_fly=0 in a length-1 fly axis) -- the point of
these tests is the gallery's stratified draw and the `--apply-review` gate,
not multi-fly geometry (already covered by `test_pseudo_export.py`).
"""
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from pseudo_fixtures import CAMS, H, KP_NAMES, W, cam_mats, make_calib, project

MOD = Path(__file__).resolve().parents[3] / "scripts" / "pseudo_labels" / "pseudolabel_gallery.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("pseudolabel_gallery", MOD)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


FEMALE, MALE = "female", "male"


def _records(rows, rec):
    from jarvis_jax.data.pseudo_export import PseudoRecord
    cm = cam_mats()
    rng = np.random.default_rng(0)
    body = rng.normal(size=(len(KP_NAMES), 3)) * np.array([3.0, 1.5, 1.0])   # one fly, near origin
    proj = project(cm, body)                                                # (C,K,2)
    kp3d = body[None]                                                       # (1,K,3)
    kp2d = proj[None]                                                       # (1,C,K,2)
    vis = np.ones((1, len(KP_NAMES)), bool)
    out = []
    for i, spec in enumerate(rows):
        sex_code = 1 if spec["host_sex"] == MALE else 0
        out.append(PseudoRecord(
            recording=rec, frame=1000 + i, host_fly=0,
            kp3d=kp3d, kp2d=kp2d, vis=vis, sex=np.array([sex_code], np.int8),
            stratum={"host_sex": spec["host_sex"], "contact": bool(spec["contact"]),
                     "apart": not bool(spec["contact"]), "wall": bool(spec["wall"])},
            gates={"exist": 0.95, "step_units": 0.1, "reproj_px": 0.5, "contain_frac": 0.97},
            partners={}, role=spec.get("role", "anchor"), bout=1))
    return out


def _write_fake_export(tmp_path, rows, rec="rec0"):
    from jarvis_jax.data.pseudo_export import write_pseudo_export
    calib_dir = make_calib(tmp_path / "calib", cam_mats())
    out = str(tmp_path / "pseudo")
    write_pseudo_export(
        out, _records(rows, rec), export_names=KP_NAMES, cameras=CAMS,
        recordings={rec: {"calib_dir": calib_dir, "fly_sex": {"fly0": "female", "fly1": "male"}}},
        checkpoint="/fake/final", gates={"exist_min": 0.8},
        frame_reader=lambda r, f: np.full((len(CAMS), H, W, 3), 200, np.uint8))
    return out


def _fixture_12(tmp_path):
    """12 framesets, same host_sex/wall, split 6 contact / 6 apart -- an exact
    50/50 cell split so a proportional draw of --n 6 needs no rounding."""
    rows = ([{"host_sex": FEMALE, "contact": True, "wall": False} for _ in range(6)]
            + [{"host_sex": FEMALE, "contact": False, "wall": False} for _ in range(6)])
    return _write_fake_export(tmp_path, rows)


def _fixture_40(tmp_path):
    """40 framesets, an exact (host_sex x contact) 2x2 grid of 10 each, wall=False
    throughout -- --n 40 takes the whole export, so cell membership in the CSV
    is exactly cell membership in the export (no sampling noise)."""
    rows = []
    for host_sex in (FEMALE, MALE):
        for contact in (True, False):
            rows += [{"host_sex": host_sex, "contact": contact, "wall": False} for _ in range(10)]
    return _write_fake_export(tmp_path, rows)


def _fixture_cells(tmp_path, cells, rec="rec0"):
    """`cells`: a list of `(host_sex, contact, wall, n[, role])` tuples
    (role defaults to "anchor") -- builds an export with EXACTLY `n`
    framesets in each named cell, so a test's reject fractions are exact
    rationals (e.g. 3/100 == 0.03 to the bit) rather than an approximation
    that happens to round the right way."""
    rows = []
    for spec in cells:
        host_sex, contact, wall, n = spec[:4]
        role = spec[4] if len(spec) > 4 else "anchor"
        rows += [{"host_sex": host_sex, "contact": contact, "wall": wall, "role": role}
                for _ in range(n)]
    return _write_fake_export(tmp_path, rows, rec=rec)


def _rewrite_csv(m, csv_path, rows):
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=m.CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------- missing-wing-cams fixture
def _records_missing_wing(rows, rec):
    """Like `_records`, but a row may carry `missing_wing=True` (host fly's
    WingL_base -- the fixture's one `Wing*`-named keypoint -- globally
    invisible, so it is invisible in EVERY camera: 0 visible cameras, below
    any `--reject-missing-wing-cams N>=1`) and/or its own `partners` map, so
    the missing-wing rule's anchor-drop + partner-cascade can be exercised
    without touching the shared `pseudo_fixtures.py`."""
    from jarvis_jax.data.pseudo_export import PseudoRecord
    cm = cam_mats()
    rng = np.random.default_rng(0)
    body = rng.normal(size=(len(KP_NAMES), 3)) * np.array([3.0, 1.5, 1.0])
    proj = project(cm, body)
    kp3d = body[None]
    kp2d = proj[None]
    wing_i = KP_NAMES.index("WingL_base")
    out = []
    for i, spec in enumerate(rows):
        sex_code = 1 if spec["host_sex"] == MALE else 0
        vis = np.ones((1, len(KP_NAMES)), bool)
        if spec.get("missing_wing"):
            vis[0, wing_i] = False
        out.append(PseudoRecord(
            recording=rec, frame=1000 + i, host_fly=0,
            kp3d=kp3d, kp2d=kp2d, vis=vis, sex=np.array([sex_code], np.int8),
            stratum={"host_sex": spec["host_sex"], "contact": bool(spec["contact"]),
                     "apart": not bool(spec["contact"]), "wall": bool(spec["wall"])},
            gates={"exist": 0.95, "step_units": 0.1, "reproj_px": 0.5, "contain_frac": 0.97},
            partners=spec.get("partners") or {}, role=spec.get("role", "anchor"), bout=1))
    return out


def _write_fake_export_missing_wing(tmp_path, rows, rec="rec0"):
    from jarvis_jax.data.pseudo_export import write_pseudo_export
    calib_dir = make_calib(tmp_path / "calib", cam_mats())
    out = str(tmp_path / "pseudo")
    write_pseudo_export(
        out, _records_missing_wing(rows, rec), export_names=KP_NAMES, cameras=CAMS,
        recordings={rec: {"calib_dir": calib_dir, "fly_sex": {"fly0": "female", "fly1": "male"}}},
        checkpoint="/fake/final", gates={"exist_min": 0.8},
        frame_reader=lambda r, f: np.full((len(CAMS), H, W, 3), 200, np.uint8))
    return out


# ------------------------------------------ v2 render: skeleton edge topology
def test_skeleton_edges_reuse_body_wing_topology_and_colour_by_distal_group():
    """`_skeleton_edges_colored` must reuse the exact head/wing/abdomen name
    pairs (never invent new ones) and colour each edge by its DISTAL
    endpoint's `keypoint_groups` group -- checked against the fixture's own
    small `KP_NAMES` (no leg names, so `leg_chains` contributes nothing
    here, isolating the body/wing half)."""
    m = _load_cli()
    from viz.core.colors import PALETTE, keypoint_groups
    idx = {n: i for i, n in enumerate(KP_NAMES)}
    groups = keypoint_groups(KP_NAMES)
    idx2group = {i: g for g, idxs in groups.items() for i in idxs}

    edges = m._skeleton_edges_colored(KP_NAMES, idx2group)

    pairs = {(a, b) for a, b, _ in edges}
    assert pairs == {
        (idx["EyeL"], idx["Antenna_Base"]),
        (idx["EyeR"], idx["Antenna_Base"]),
        (idx["Antenna_Base"], idx["Scutellum"]),
        (idx["Scutellum"], idx["WingL_base"]),
    }
    color_by_pair = {(a, b): c for a, b, c in edges}
    assert color_by_pair[(idx["EyeL"], idx["Antenna_Base"])] == PALETTE["head"]
    # Antenna_Base(head) -> Scutellum(thorax): coloured by the DISTAL
    # (thorax) endpoint, not the proximal (head) one.
    assert color_by_pair[(idx["Antenna_Base"], idx["Scutellum"])] == PALETTE["thorax"]
    assert color_by_pair[(idx["Scutellum"], idx["WingL_base"])] == PALETTE["thorax"]


# ------------------------------------------- v2 render: zoom crop geometry
def test_zoom_crop_box_pads_bbox_40_percent_and_floors_at_min_side():
    m = _load_cli()
    kp = np.zeros((6, 3), np.float32)
    kp[0] = [100, 200, 1]
    kp[1] = [200, 205, 1]           # bbox 100x5 -> padded 140x7, floored at 160
    ann = {"keypoints": kp.reshape(-1).tolist()}

    x0, y0, side = m._zoom_crop_box(ann, img_w=1000, img_h=1000)

    assert side == m.ZOOM_MIN_SIDE
    assert (x0, y0) == (70, 122)    # centre (150, 202.5) minus half the (160) side


def test_zoom_crop_box_grows_past_the_floor_for_a_spread_out_pose():
    m = _load_cli()
    kp = np.zeros((6, 3), np.float32)
    kp[0] = [0, 0, 1]
    kp[1] = [300, 0, 1]             # bbox 300 wide -> padded 420, past the 160 floor
    ann = {"keypoints": kp.reshape(-1).tolist()}

    _, _, side = m._zoom_crop_box(ann, img_w=1000, img_h=1000)

    assert side == 420


def test_zoom_crop_box_falls_back_to_image_centre_with_no_ann():
    m = _load_cli()
    x0, y0, side = m._zoom_crop_box(None, img_w=800, img_h=600)
    assert side == m.ZOOM_MIN_SIDE
    assert (x0, y0) == (800 // 2 - 80, 600 // 2 - 80)


def test_draw_skeleton_only_connects_keypoints_both_visible_in_this_camera():
    """A line must not be drawn to/through an invisible keypoint even when an
    edge names it -- checked on a blank canvas: the pixel at the midpoint of
    an edge whose far end is invisible must stay the background colour."""
    m = _load_cli()
    canvas = np.zeros((50, 50, 3), np.uint8)
    uv = np.array([[10.0, 25.0], [40.0, 25.0], [10.0, 10.0]], np.float32)
    vis_both = np.array([True, True, False])
    edges = [(0, 1, (0, 255, 0)), (0, 2, (255, 0, 0))]     # 2nd edge's far end (2) is invisible

    m._draw_skeleton(canvas, uv, vis_both, edges, {}, radius=1)

    # LINE_AA anti-aliasing softens the exact pixel value, so check the
    # dominant channel rather than an exact (0, 255, 0) match.
    px = canvas[25, 25]
    assert px[1] > 200 and px[0] == 0 and px[2] == 0               # drawn edge 0-1 (green)
    assert tuple(int(c) for c in canvas[10, 10]) == (0, 0, 0)      # edge 0-2 skipped (kp 2 invisible)


# --------------------------------------------------------------------- (a)
def test_gallery_writes_one_page_and_a_six_row_csv_with_blank_verdicts(tmp_path):
    m = _load_cli()
    root = _fixture_12(tmp_path)
    out = str(tmp_path / "gallery")
    assert m.main(["--export", root, "--n", "6", "--out", out]) == 0
    page_path = os.path.join(out, "page_00.png")
    assert os.path.exists(page_path)
    assert not os.path.exists(os.path.join(out, "page_01.png"))       # exactly one page for 6 rows
    # panel geometry (2026-09-06 gallery-v2 render): 4 columns per frameset
    # row -- [overhead full][overhead zoom][side full][side zoom] -- so the
    # page is now PANEL_W * 4 wide (was PANEL_W * 2 before the zoom columns).
    import cv2
    page = cv2.imread(page_path)
    assert page.shape == (m.PANEL_H * 6, m.PANEL_W * 4, 3)
    with open(os.path.join(out, "review.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == m.CSV_FIELDS
    assert len(rows) == 6
    assert all(r["verdict"] == "" for r in rows)


# --------------------------------------------------------------------- (b)
def test_sampling_is_stratified_three_contact_three_apart(tmp_path):
    m = _load_cli()
    root = _fixture_12(tmp_path)
    out = str(tmp_path / "gallery")
    m.main(["--export", root, "--n", "6", "--out", out])
    with open(os.path.join(out, "review.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    contact = sum(1 for r in rows if r["contact"] == "True")
    apart = sum(1 for r in rows if r["contact"] == "False")
    assert (contact, apart) == (3, 3)


# --------------------------------------------------------------------- (b2)
def test_review_csv_unchanged_by_a_rendering_flag(tmp_path, monkeypatch):
    """INVARIANT (2026-09-06 gallery-v2 render, CLAUDE.md's own gate for a
    rendering-only change): the draw -- frameset order, page, cell -- must
    not move no matter what rendering does. Toggle rendering itself off
    (monkeypatch `render_page` to a no-op, standing in for "a rendering
    flag") and check the resulting review.csv is byte-identical to a normal
    run's: the stratified draw/CSV path must be fully decoupled from
    render_page's pixels."""
    m = _load_cli()
    root = _fixture_40(tmp_path)

    out_rendered = str(tmp_path / "gallery_rendered")
    assert m.main(["--export", root, "--n", "40", "--out", out_rendered]) == 0
    assert os.path.exists(os.path.join(out_rendered, "page_00.png"))
    rendered_csv = open(os.path.join(out_rendered, "review.csv"), "rb").read()

    monkeypatch.setattr(m, "render_page", lambda *a, **k: None)
    out_norender = str(tmp_path / "gallery_norender")
    assert m.main(["--export", root, "--n", "40", "--out", out_norender]) == 0
    assert not os.path.exists(os.path.join(out_norender, "page_00.png"))   # rendering really skipped
    norender_csv = open(os.path.join(out_norender, "review.csv"), "rb").read()

    assert norender_csv == rendered_csv


# --------------------------------------------------------------------- (c)
def test_apply_review_over_threshold_exits_nonzero_and_writes_nothing(tmp_path):
    m = _load_cli()
    root = _fixture_12(tmp_path)
    out = str(tmp_path / "gallery")
    m.main(["--export", root, "--n", "6", "--out", out])
    csv_path = os.path.join(out, "review.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["verdict"] = "reject"                          # 1/6 = 16.7% > 3%
    _rewrite_csv(m, csv_path, rows)

    manifest_before = open(os.path.join(root, "manifest.json")).read()
    ann_before = open(os.path.join(root, "annotations", "instances_train.json")).read()

    code = m.main(["--export", root, "--apply-review", csv_path])

    assert code == 2
    assert not os.path.exists(os.path.join(out, "review_summary.json"))
    assert open(os.path.join(root, "manifest.json")).read() == manifest_before
    assert open(os.path.join(root, "annotations", "instances_train.json")).read() == ann_before


# --------------------------------------------------------------------- (d)
def test_apply_review_zero_rejected_rewrites_unchanged_and_sets_manifest(tmp_path):
    m = _load_cli()
    root = _fixture_40(tmp_path)
    out = str(tmp_path / "gallery")
    m.main(["--export", root, "--n", "40", "--out", out])
    csv_path = os.path.join(out, "review.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 40

    ann_path = os.path.join(root, "annotations", "instances_train.json")
    before = json.load(open(ann_path))

    code = m.main(["--export", root, "--apply-review", csv_path])

    assert code == 0
    after = json.load(open(ann_path))
    assert after["framesets"] == before["framesets"]
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    assert manifest["review"] == {"file": csv_path, "n": 40, "reject_frac": 0.0,
                                  "dropped_strata": []}


# --------------------------------------------------------------------- (e)
def test_apply_review_drops_whole_stratum_over_its_own_threshold(tmp_path):
    m = _load_cli()
    root = _fixture_40(tmp_path)
    out = str(tmp_path / "gallery")
    m.main(["--export", root, "--n", "40", "--out", out])
    csv_path = os.path.join(out, "review.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    target_cell = f"{FEMALE}/contact/floor"
    cell_rows = [r for r in rows if r["cell"] == target_cell]
    assert len(cell_rows) == 10                             # the whole cell is in the (full) sample
    cell_rows[0]["verdict"] = "reject"                       # 1/10 = 10% (cell) > 3%; 1/40 = 2.5% overall
    _rewrite_csv(m, csv_path, rows)

    code = m.main(["--export", root, "--apply-review", csv_path])
    assert code == 0

    summary = json.load(open(os.path.join(out, "review_summary.json")))
    assert summary["dropped_strata"] == [target_cell]
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    assert manifest["review"]["dropped_strata"] == [target_cell]
    assert manifest["review"]["reject_frac"] == 0.025

    coco = json.load(open(os.path.join(root, "annotations", "instances_train.json")))
    remaining = coco["framesets"]
    assert len(remaining) == 30                             # the whole 10-row cell gone, not just 1
    for key, fsv in remaining.items():
        assert m._cell_of(fsv["stratum"]) != target_cell

    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_id = {a["id"]: a for a in coco["annotations"]}
    for key, fsv in remaining.items():                       # every remaining frameset key resolves
        for img_id, ann_id in zip(fsv["frames"], fsv["ann_ids"]):
            assert img_id in images_by_id
            if ann_id is not None:
                assert ann_id in anns_by_id
        for pf in fsv.get("partners", {}).values():
            assert f"{fsv['recording']}/Frame_{pf}/fly{fsv['fly_id']}" in remaining


# ------------------------------------------------------- exact 3% boundary
def test_apply_review_overall_exactly_3_percent_is_accepted(tmp_path):
    """spec SS3.3 says '<= 3%' passes -- 3/100 must NOT abort. Single cell, so
    this also pins that a per-cell rate of EXACTLY 3% does not drop the cell
    (its rate equals the overall rate here)."""
    m = _load_cli()
    root = _fixture_cells(tmp_path, [(FEMALE, True, False, 100)])
    out = str(tmp_path / "gallery")
    m.main(["--export", root, "--n", "100", "--out", out])
    csv_path = os.path.join(out, "review.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 100
    for r in rows[:3]:
        r["verdict"] = "reject"                              # 3/100 = 0.03 exactly
    _rewrite_csv(m, csv_path, rows)

    code = m.main(["--export", root, "--apply-review", csv_path])

    assert code == 0
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    assert manifest["review"]["reject_frac"] == 0.03
    assert manifest["review"]["dropped_strata"] == []
    coco = json.load(open(os.path.join(root, "annotations", "instances_train.json")))
    assert len(coco["framesets"]) == 97                      # only the 3 rejected rows dropped


def test_apply_review_overall_just_over_3_percent_is_rejected(tmp_path):
    """4/100 = 4% is the smallest step past the exact 3% boundary -- must
    abort and write nothing, mirroring test (c) but pinned at this exact
    fraction rather than 1/6's more generous 16.7%."""
    m = _load_cli()
    root = _fixture_cells(tmp_path, [(FEMALE, True, False, 100)])
    out = str(tmp_path / "gallery")
    m.main(["--export", root, "--n", "100", "--out", out])
    csv_path = os.path.join(out, "review.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows[:4]:
        r["verdict"] = "reject"                              # 4/100 = 0.04 > 0.03
    _rewrite_csv(m, csv_path, rows)

    manifest_before = open(os.path.join(root, "manifest.json")).read()
    ann_before = open(os.path.join(root, "annotations", "instances_train.json")).read()

    code = m.main(["--export", root, "--apply-review", csv_path])

    assert code == 2
    assert not os.path.exists(os.path.join(out, "review_summary.json"))
    assert open(os.path.join(root, "manifest.json")).read() == manifest_before
    assert open(os.path.join(root, "annotations", "instances_train.json")).read() == ann_before


def test_apply_review_per_cell_exactly_3_percent_is_not_dropped(tmp_path):
    """Isolates the per-cell check from the overall check: cell A's OWN rate
    is exactly 3% while the overall rate (3/200) is comfortably below it, so
    a per-cell check that used the overall total instead of the cell's own
    total would (wrongly) never even look at this boundary."""
    m = _load_cli()
    root = _fixture_cells(tmp_path, [(FEMALE, True, False, 100), (MALE, False, False, 100)])
    out = str(tmp_path / "gallery")
    m.main(["--export", root, "--n", "200", "--out", out])
    csv_path = os.path.join(out, "review.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 200
    cell_a = f"{FEMALE}/contact/floor"
    a_rows = [r for r in rows if r["cell"] == cell_a]
    assert len(a_rows) == 100
    for r in a_rows[:3]:
        r["verdict"] = "reject"                              # 3/100 = 0.03 exactly for cell A
    _rewrite_csv(m, csv_path, rows)

    code = m.main(["--export", root, "--apply-review", csv_path])

    assert code == 0                                          # overall 3/200 = 1.5% <= 3%
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    assert manifest["review"]["reject_frac"] == 0.015
    assert manifest["review"]["dropped_strata"] == []          # cell A's exact 3% is NOT a drop
    coco = json.load(open(os.path.join(root, "annotations", "instances_train.json")))
    assert len(coco["framesets"]) == 197                       # only the 3 rejected rows gone


def test_apply_review_per_cell_just_over_3_percent_drops_only_that_cell(tmp_path):
    """1/30 = 3.33% is the smallest step past the exact 3% boundary for a
    30-row cell -- the whole cell must be dropped while a cell with 0
    rejects (and the low overall rate) survives untouched."""
    m = _load_cli()
    root = _fixture_cells(tmp_path, [(FEMALE, True, False, 30), (MALE, False, False, 70)])
    out = str(tmp_path / "gallery")
    m.main(["--export", root, "--n", "100", "--out", out])
    csv_path = os.path.join(out, "review.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 100
    cell_a = f"{FEMALE}/contact/floor"
    a_rows = [r for r in rows if r["cell"] == cell_a]
    assert len(a_rows) == 30
    a_rows[0]["verdict"] = "reject"                           # 1/30 = 3.33% > 3%
    _rewrite_csv(m, csv_path, rows)

    code = m.main(["--export", root, "--apply-review", csv_path])

    assert code == 0                                          # overall 1/100 = 1% <= 3%
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    assert manifest["review"]["reject_frac"] == 0.01
    assert manifest["review"]["dropped_strata"] == [cell_a]
    coco = json.load(open(os.path.join(root, "annotations", "instances_train.json")))
    remaining = coco["framesets"]
    assert len(remaining) == 70                                # the whole 30-row cell gone
    for key, fsv in remaining.items():
        assert m._cell_of(fsv["stratum"]) != cell_a


# ------------------------------------------------------ anchors-only sample
def test_review_sample_is_anchors_only_by_default_include_partners_opts_in(tmp_path):
    m = _load_cli()
    root = _fixture_cells(tmp_path, [
        (FEMALE, True, False, 4, "anchor"),
        (FEMALE, True, False, 4, "partner"),
    ])
    out_default = str(tmp_path / "gallery_default")
    m.main(["--export", root, "--n", "8", "--out", out_default])
    with open(os.path.join(out_default, "review.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4                                     # only the 4 anchors, not all 8
    assert {int(r["frame"]) for r in rows} == {1000, 1001, 1002, 1003}

    out_incl = str(tmp_path / "gallery_incl")
    m.main(["--export", root, "--n", "8", "--out", out_incl, "--include-partners"])
    with open(os.path.join(out_incl, "review.csv"), newline="") as f:
        rows_incl = list(csv.DictReader(f))
    assert len(rows_incl) == 8
    assert {int(r["frame"]) for r in rows_incl} == set(range(1000, 1008))


# --------------------------------------------------- placeholder panel label
def test_label_prefix_omits_bare_bout_token_when_bout_is_none():
    m = _load_cli()
    with_bout = {"recording": "rec0", "bout": 3, "frame": 1000, "host_fly": 0, "host_sex": "female"}
    no_bout = {"recording": "rec0", "bout": None, "frame": 1000, "host_fly": 0, "host_sex": "male"}
    assert m._label_prefix(with_bout) == "rec0 b3 f1000 fly0 female"
    assert m._label_prefix(no_bout) == "rec0 f1000 fly0 male"      # no bare "b" token
    assert "b" not in m._label_prefix(no_bout).split(" ")


def test_placeholder_panel_for_a_missing_camera_carries_the_full_row_identity(tmp_path):
    m = _load_cli()
    root = _fixture_cells(tmp_path, [(FEMALE, True, False, 1)])
    coco, kp_names, images_by_id, anns_by_id = m._load_export(root)
    row = next(m._row_info(k, v) for k, v in m._iter_reviewable(coco))
    groups = m.keypoint_groups(kp_names)
    idx2group = {i: g for g, idxs in groups.items() for i in idxs}

    # edges_colored is irrelevant here -- the missing-camera branch returns
    # a placeholder before it is ever consulted -- so an empty list stands in.
    panel = m._panel(root, coco, idx2group, [], images_by_id, anns_by_id, row, "Cam_does_not_exist")

    assert panel.shape == (m.PANEL_H, m.PANEL_W, 3)
    # a real frame is the fixture's flat 200-grey; the placeholder must NOT be
    # that -- i.e. this really took the missing-camera branch, not a silent
    # fallback that drew an unrelated real frame.
    assert not np.array_equal(panel[0, 0], np.array([200, 200, 200], np.uint8))


# ----------------------------------------------------------- partner cascade
def test_cascade_drops_orphaned_partner_but_keeps_one_still_referenced():
    """Unit-level: a dropped anchor's OWN partner frame is dropped too unless
    a surviving anchor also references it -- the controller ruling's literal
    'partners of a dropped anchor: drop them too unless another kept anchor
    references them,' isolated from the CSV/export plumbing."""
    m = _load_cli()
    fs = {
        "rec/Frame_1000/fly0": {"recording": "rec", "fly_id": 0, "role": "anchor",
                                "partners": {"1": 1001}},
        "rec/Frame_1001/fly0": {"recording": "rec", "fly_id": 0, "role": "partner",
                                "partners": {}},
        "rec/Frame_2000/fly0": {"recording": "rec", "fly_id": 0, "role": "anchor",
                                "partners": {"1": 2001}},
        "rec/Frame_2001/fly0": {"recording": "rec", "fly_id": 0, "role": "partner",
                                "partners": {}},
        "rec/Frame_3000/fly0": {"recording": "rec", "fly_id": 0, "role": "anchor",
                                "partners": {"4": 2001}},          # ALSO uses 2001, and SURVIVES
    }
    dropped = m._cascade_partner_drop(fs, {"rec/Frame_1000/fly0", "rec/Frame_2000/fly0"})

    assert "rec/Frame_1001/fly0" in dropped                 # orphaned: only Frame_1000 used it
    assert "rec/Frame_2001/fly0" not in dropped             # Frame_3000 (kept) still uses it

    m._clean_dangling_partners(fs, dropped)
    assert fs["rec/Frame_3000/fly0"]["partners"] == {"4": 2001}   # untouched, still resolves


# --------------------------------------------------- --max-reject-frac / --cell-drop-frac
def test_apply_review_override_records_block_and_applies_past_default_threshold(tmp_path):
    """User-decision override (2026-09-06 real scenario): 19/300 = 6.33%
    overall rejects, well past the spec's 3% abort -- `--max-reject-frac 1.0`
    lets it through, `--cell-drop-frac none` keeps the single (violating)
    cell instead of dropping it, and the override is recorded verbatim
    (value + reason + measured fractions) in BOTH manifest.json and
    review_summary.json so the decision is auditable later."""
    m = _load_cli()
    root = _fixture_cells(tmp_path, [(FEMALE, False, False, 300)])
    out = str(tmp_path / "gallery")
    m.main(["--export", root, "--n", "300", "--out", out])
    csv_path = os.path.join(out, "review.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 300
    for r in rows[:19]:
        r["verdict"] = "reject"                              # 19/300 = 0.0633...
    _rewrite_csv(m, csv_path, rows)

    code = m.main(["--export", root, "--apply-review", csv_path,
                  "--max-reject-frac", "1.0", "--cell-drop-frac", "none",
                  "--override-reason", "user decision 2026-09-06: train with the 19 "
                  "reviewed rejects removed and no cell drop"])

    assert code == 0
    expect_frac = round(19 / 300, 4)
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    review = manifest["review"]
    assert review["reject_frac"] == expect_frac
    assert review["dropped_strata"] == []                    # cell kept despite its own > 3% rate
    ov = review["overrides"]
    assert ov["max_reject_frac"] == 1.0
    assert ov["cell_drop_frac"] is None
    assert ov["reason"].startswith("user decision 2026-09-06")
    assert ov["measured"]["overall_reject_frac"] == expect_frac
    assert ov["measured"]["per_cell_reject_frac"][f"{FEMALE}/apart/floor"] == expect_frac

    summary = json.load(open(os.path.join(out, "review_summary.json")))
    assert summary["overrides"] == ov

    coco = json.load(open(os.path.join(root, "annotations", "instances_train.json")))
    assert len(coco["framesets"]) == 300 - 19                 # only the 19 rejected rows dropped


def test_apply_review_default_call_has_no_overrides_block(tmp_path):
    """Sibling to (d): an ordinary (no-override) call must not gain an
    `overrides` key -- existing tests' exact-dict comparisons on
    `manifest["review"]` (e.g. test (d)/(e)) depend on this."""
    m = _load_cli()
    root = _fixture_40(tmp_path)
    out = str(tmp_path / "gallery")
    m.main(["--export", root, "--n", "40", "--out", out])
    csv_path = os.path.join(out, "review.csv")

    code = m.main(["--export", root, "--apply-review", csv_path])

    assert code == 0
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    assert "overrides" not in manifest["review"]
    summary = json.load(open(os.path.join(out, "review_summary.json")))
    assert "overrides" not in summary
    assert "missing_wing_dropped" not in summary


def test_apply_review_override_without_reason_raises(tmp_path):
    m = _load_cli()
    root = _fixture_cells(tmp_path, [(FEMALE, True, False, 100)])
    out = str(tmp_path / "gallery")
    m.main(["--export", root, "--n", "100", "--out", out])
    csv_path = os.path.join(out, "review.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows[:10]:
        r["verdict"] = "reject"
    _rewrite_csv(m, csv_path, rows)

    import pytest
    with pytest.raises(ValueError):
        m.main(["--export", root, "--apply-review", csv_path, "--max-reject-frac", "1.0"])
    # neither file is touched when the call raises before applying anything
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    assert "review" not in manifest


def test_apply_review_cell_drop_frac_none_disables_whole_cell_drop(tmp_path):
    """Sibling to test (e): the SAME 1/10 = 10% (cell) > 3% overrun that
    normally drops the whole cell must instead keep it, whole, when
    `--cell-drop-frac none` is passed."""
    m = _load_cli()
    root = _fixture_40(tmp_path)
    out = str(tmp_path / "gallery")
    m.main(["--export", root, "--n", "40", "--out", out])
    csv_path = os.path.join(out, "review.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    target_cell = f"{FEMALE}/contact/floor"
    cell_rows = [r for r in rows if r["cell"] == target_cell]
    assert len(cell_rows) == 10
    cell_rows[0]["verdict"] = "reject"                        # 1/10 = 10% (cell) > 3%; 1/40 overall
    _rewrite_csv(m, csv_path, rows)

    code = m.main(["--export", root, "--apply-review", csv_path,
                  "--cell-drop-frac", "none", "--override-reason", "keep all cells (test)"])

    assert code == 0
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    assert manifest["review"]["dropped_strata"] == []
    assert manifest["review"]["overrides"]["cell_drop_frac"] is None
    coco = json.load(open(os.path.join(root, "annotations", "instances_train.json")))
    remaining = coco["framesets"]
    assert len(remaining) == 39                               # only the 1 rejected row dropped
    assert sum(1 for v in remaining.values()
              if m._cell_of(v["stratum"]) == target_cell) == 9   # the other 9 of the cell survive


# --------------------------------------------------------- --reject-missing-wing-cams
def test_apply_review_reject_missing_wing_cams_drops_frame_and_cascades_partner(tmp_path):
    """The one `Wing*` keypoint (`WingL_base`) is globally invisible (0 of 7
    cameras) on the frame_1000 anchor -- below any N>=1 -- so `--reject-
    missing-wing-cams 3` must drop it even though the CSV verdict is blank
    (the rule is independent of human review), cascade its sole partner
    (frame_1001), and leave the unrelated frame_1002 anchor untouched."""
    m = _load_cli()
    rows = [
        {"host_sex": FEMALE, "contact": True, "wall": False,
         "missing_wing": True, "partners": {"1": 1001}},
        {"host_sex": FEMALE, "contact": True, "wall": False, "role": "partner"},
        {"host_sex": FEMALE, "contact": True, "wall": False},
    ]
    root = _write_fake_export_missing_wing(tmp_path, rows)
    out = str(tmp_path / "gallery")
    m.main(["--export", root, "--n", "2", "--out", out])       # samples only the 2 anchors
    csv_path = os.path.join(out, "review.csv")
    with open(csv_path, newline="") as f:
        sampled = list(csv.DictReader(f))
    assert len(sampled) == 2
    assert all((r["verdict"] or "") == "" for r in sampled)    # nobody rejected by review

    code = m.main(["--export", root, "--apply-review", csv_path,
                  "--reject-missing-wing-cams", "3"])

    assert code == 0
    coco = json.load(open(os.path.join(root, "annotations", "instances_train.json")))
    remaining = coco["framesets"]
    assert "rec0/Frame_1000/fly0" not in remaining             # missing-wing anchor dropped
    assert "rec0/Frame_1001/fly0" not in remaining             # its sole partner cascaded
    assert "rec0/Frame_1002/fly0" in remaining                 # unrelated anchor untouched

    summary = json.load(open(os.path.join(out, "review_summary.json")))
    assert summary["missing_wing_dropped"] == {"anchors": 1, "partners": 1}
    assert summary["n_dropped_framesets"] == 2                 # the anchor + its cascaded partner
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    assert "overrides" not in manifest["review"]               # not a spec-gate override


def test_reject_missing_wing_cams_zero_is_off_by_default(tmp_path):
    m = _load_cli()
    rows = [{"host_sex": FEMALE, "contact": True, "wall": False, "missing_wing": True}]
    root = _write_fake_export_missing_wing(tmp_path, rows)
    out = str(tmp_path / "gallery")
    m.main(["--export", root, "--n", "1", "--out", out])
    csv_path = os.path.join(out, "review.csv")

    code = m.main(["--export", root, "--apply-review", csv_path])

    assert code == 0
    coco = json.load(open(os.path.join(root, "annotations", "instances_train.json")))
    assert "rec0/Frame_1000/fly0" in coco["framesets"]          # default N=0 -- rule off
    summary = json.load(open(os.path.join(out, "review_summary.json")))
    assert "missing_wing_dropped" not in summary


def test_missing_wing_keys_unit_ignores_non_anchor_and_missing_ann():
    """Unit-level: `_missing_wing_keys` only ever considers `role == "anchor"`
    framesets, and a `None`/unresolvable ann_id contributes 0 (not an error)
    to that camera's visibility count."""
    m = _load_cli()
    kp_names = ["Antenna_Base", "WingL_base"]
    anns_by_id = {
        1: {"keypoints": [0, 0, 1, 5, 5, 1]},     # WingL_base visible (v=1)
        2: {"keypoints": [0, 0, 1, 5, 5, 0]},     # WingL_base NOT visible (v=0)
    }
    fs = {
        "rec/Frame_1/fly0": {"role": "anchor", "ann_ids": [1, None, 2]},   # wing seen in 1/3 cams
        "rec/Frame_2/fly0": {"role": "partner", "ann_ids": [None]},        # not an anchor: ignored
    }
    wing_idx = m._wing_indices(kp_names)
    assert wing_idx == [1]

    dropped = m._missing_wing_keys(fs, anns_by_id, wing_idx, min_cams=2)

    assert dropped == {"rec/Frame_1/fly0"}                     # 1 visible camera < 2
    assert m._missing_wing_keys(fs, anns_by_id, wing_idx, min_cams=1) == set()  # 1 >= 1, kept
    assert m._missing_wing_keys(fs, anns_by_id, wing_idx, min_cams=0) == set()  # 0 == off
