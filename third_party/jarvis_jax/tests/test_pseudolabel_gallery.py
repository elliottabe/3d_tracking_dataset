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
            partners={}, role="anchor", bout=1))
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


def _rewrite_csv(m, csv_path, rows):
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=m.CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------- (a)
def test_gallery_writes_one_page_and_a_six_row_csv_with_blank_verdicts(tmp_path):
    m = _load_cli()
    root = _fixture_12(tmp_path)
    out = str(tmp_path / "gallery")
    assert m.main(["--export", root, "--n", "6", "--out", out]) == 0
    assert os.path.exists(os.path.join(out, "page_00.png"))
    assert not os.path.exists(os.path.join(out, "page_01.png"))       # exactly one page for 6 rows
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
