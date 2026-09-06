"""End-to-end `scripts/pseudo_labels/extract_p3b_pseudolabels.py` on a fake run root.

The one thing stubbed out is the video: `SessionFrames` decodes the real mp4s,
so the tests swap it for a stub that hands back blank 448x448 frames. Every
other step is the real one -- discovery, the gates, the floor fit, the
stratified draw, the writer, and the loader reading the result back.
"""
import importlib.util
import sys
import json
import os
from pathlib import Path

import numpy as np
import pytest
from pseudo_fixtures import CAMS, make_bout, make_calib

MOD = Path(__file__).resolve().parents[3] / "scripts" / "pseudo_labels" / "extract_p3b_pseudolabels.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("extract_p3b_pseudolabels", MOD)
    m = importlib.util.module_from_spec(spec)
    # registered BEFORE exec: the module's dataclasses use PEP 563 string
    # annotations, and `dataclasses.fields` resolves them through
    # `sys.modules[cls.__module__]`.
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


class FakeFrames:
    """(C,448,448,3) blanks: >= CROP in both axes so `V12WindowDataset._build`
    can cut a real 448 crop out of them."""

    def __init__(self, sessions, cameras):
        self.n = len(cameras)

    def __call__(self, rec, frame):
        return np.zeros((self.n, 448, 448, 3), np.uint8), np.ones(self.n, bool)

    def close(self):
        pass


@pytest.fixture
def run_root(tmp_path, monkeypatch):
    """One fake recording, two bouts: one CONTACT (sep 10 units) and one APART
    (sep 40), at different absolute frames with their own mask npz."""
    d1, npz1, cm = make_bout(tmp_path, T=40, sep_units=10.0, name="bout_00001",
                             frame_start=1000, masks_path=str(tmp_path / "m1" / "sam3_masks.npz"))
    d2, npz2, _ = make_bout(tmp_path, T=40, sep_units=40.0, name="bout_00002",
                            frame_start=9000, masks_path=str(tmp_path / "m2" / "sam3_masks.npz"))
    make_calib(tmp_path / "video" / "calibration", cm)
    export_root = tmp_path / "names"
    (export_root / "annotations").mkdir(parents=True)
    from pseudo_fixtures import KP_NAMES
    (export_root / "annotations" / "keypoint_names.json").write_text(
        json.dumps(list(reversed(KP_NAMES))))          # a DIFFERENT order on purpose
    m = _load_cli()
    monkeypatch.setattr(m, "SessionFrames", FakeFrames)
    return m, str(tmp_path / "rec" / "pose_mvq_p3b"), str(export_root), cm


def _run(m, runs, names, out, *extra):
    # --figures-dir under tmp_path: its default is relative to the CWD, and a
    # test must not scribble a figures/ tree into the repo it runs from.
    return m.main(["--runs", runs, "--out", out, "--export-names-from", names,
                   "--target", "8", "--per-bout-frac", "0.5", "--min-female", "1",
                   "--figures-dir", os.path.join(out, "fig"), *extra])


def test_cli_writes_a_balanced_stratified_export(run_root, tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    m, runs, names, cm = run_root
    out = str(tmp_path / "pseudo")
    rep = _run(m, runs, names, out, "--no-masks")

    man = json.load(open(f"{out}/manifest.json"))
    assert man["source"] == "pseudo" and man["weight"] == 0.3
    assert man["gates"]["deltas"] == [1, 4, 16]
    coco = json.load(open(f"{out}/annotations/instances_train.json"))

    anchors = {k: v for k, v in coco["framesets"].items() if v["role"] == "anchor"}
    assert len(anchors) == 8
    sexes = [v["stratum"]["host_sex"] for v in anchors.values()]
    assert sexes.count("female") == 4 and sexes.count("male") == 4
    assert {v["bout"] for v in anchors.values()} == {1, 2}          # both bouts contributed
    srep = json.load(open(f"{out}/sampling_report.json"))
    assert srep["contact_frac"] >= 0.25 and srep["apart_frac"] >= 0.25
    for v in coco["framesets"].values():
        assert set(v["partners"]) <= {"1", "4", "16"}
        assert v["source"] == "pseudo" and v["weight"] == 0.3
    assert set(rep["per_role"]) <= {"anchor", "partner"} and rep["per_role"]["partner"] > 0

    ds = V12WindowDataset(out, "train", T=1, train=False)
    assert len(ds) == len(coco["framesets"])
    s = ds[0]
    assert s["kp3d_local"].shape[-2] == len(ds.keypoint_names)


def test_the_loaders_dlt_reproduces_the_gated_3d_in_EXPORT_keypoint_order(run_root, tmp_path):
    """The written 2D is the reprojection of the gated 3D, so triangulating it
    (which is all the loader ever does) must give that 3D back -- looked up BY
    NAME, since the campaign npz and the export disagree on keypoint order."""
    from jarvis_jax.data.pseudo_export import to_export_order
    from jarvis_jax.data.pseudo_gates import load_bout_arrays
    from jarvis_jax.data.v12_windows import V12WindowDataset
    m, runs, names, cm = run_root
    out = str(tmp_path / "pseudo")
    _run(m, runs, names, out, "--no-masks")
    export_names = json.load(open(f"{out}/annotations/keypoint_names.json"))
    ds = V12WindowDataset(out, "train", T=1, train=False)
    for i in range(min(4, len(ds))):
        rec, fly, frame = ds.windows[i]
        bout, start = ("bout_00001", 1000) if frame < 9000 else ("bout_00002", 9000)
        a = load_bout_arrays(f"{runs}/bouts/{bout}", cameras=CAMS)
        want = to_export_order(a.kp3d[fly, frame - start], a.kp_names, export_names)
        s = ds[i]
        got = s["kp3d_local"][0, 0] + s["center3D"]
        np.testing.assert_allclose(got, want, atol=1e-3)


def test_masks_are_written_where_the_loader_looks_for_them(run_root, tmp_path):
    """Plan B's copy-paste needs a real `prompt_mask`, which `v5_3d._load_mask`
    resolves by `src_ann_id` inside `masks/<rec>/<cam>/Frame_<n>.npz`."""
    from jarvis_jax.data.v5_3d import _load_mask
    m, runs, names, cm = run_root
    out = str(tmp_path / "pseudo")
    _run(m, runs, names, out)
    coco = json.load(open(f"{out}/annotations/instances_train.json"))
    img = {i["id"]: i for i in coco["images"]}
    ann = {a["id"]: a for a in coco["annotations"]}
    fsv = next(iter(coco["framesets"].values()))
    hits = 0
    for img_id, ann_id in zip(fsv["frames"], fsv["ann_ids"]):
        if ann_id is None:
            continue
        info = img[img_id]
        mask = _load_mask(out, info["file_name"], ann[ann_id]["src_ann_id"], ann_id,
                          info["width"], info["height"])
        hits += int(mask.any())
    assert hits >= 5, "the fly's own SAM3 mask must come back for most cameras"


def test_a_female_poor_campaign_stops_at_the_census(run_root, tmp_path, capsys):
    """Controller ruling: below `--min-female` the run writes the census and
    stops rather than quietly rebalancing onto the male fly."""
    m, runs, names, cm = run_root
    out = str(tmp_path / "pseudo_stop")
    m.main(["--runs", runs, "--out", out, "--export-names-from", names,
            "--target", "8", "--per-bout-frac", "0.5", "--min-female", "10000",
            "--figures-dir", os.path.join(out, "fig")])
    assert "STOP" in capsys.readouterr().out
    cen = json.load(open(f"{out}/census.json"))
    assert cen["by_host_sex"]["female"] > 0 and "by_stratum" in cen
    assert not (Path(out) / "annotations").exists()
