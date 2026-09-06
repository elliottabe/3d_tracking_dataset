"""End-to-end `scripts/pseudo_labels/extract_p3b_pseudolabels.py` on a fake run root.

The one thing stubbed out is the video: `SessionFrames` decodes the real mp4s,
so the tests swap it for a stub that hands back blank 448x448 frames. Every
other step is the real one -- discovery, the gates, the floor fit, the
stratified draw, the writer, and the loader reading the result back.
"""
import collections
import importlib.util
import sys
import json
import os
from pathlib import Path

import numpy as np
import pytest
from pseudo_fixtures import CAMS, make_bout, make_calib

FRAME_HW = (448, 448)      # `FakeFrames`' size; masks must match it (real: 448x1936)
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
    """(C,448,448,3) flat-grey frames: >= CROP in both axes so
    `V12WindowDataset._build` can cut a real 448 crop, and NON-ZERO so the
    CLI's own post-write check (`verify_export`, which rejects all-black
    crops -- an unwritten or misnamed JPEG) is exercised rather than tripped."""

    def __init__(self, sessions, cameras):
        self.n = len(cameras)

    def __call__(self, rec, frame):
        return np.full((self.n,) + FRAME_HW + (3,), 40, np.uint8), np.ones(self.n, bool)

    def close(self):
        pass


@pytest.fixture
def run_root(tmp_path, monkeypatch):
    """One fake recording, two bouts: one CONTACT and one APART.

    `contact` is the MINIMUM inter-fly KEYPOINT distance (ruling 2026-09-05,
    the spec's centroid rule is unreachable), so the contact bout puts the two
    flies `sep_units=10` apart -- with the fixture's ~+-4.5-unit body spread
    that leaves keypoint pairs well under the 5-unit cut -- and the apart bout
    at 40 units, where the closest keypoints are ~30 units apart.
    """
    d1, npz1, cm = make_bout(tmp_path, T=40, sep_units=10.0, name="bout_00001", hw=FRAME_HW,
                             frame_start=1000, masks_path=str(tmp_path / "m1" / "sam3_masks.npz"))
    d2, npz2, _ = make_bout(tmp_path, T=40, sep_units=40.0, name="bout_00002", hw=FRAME_HW,
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
    # --per-rec-frac 1.0: the 20 % per-recording share is meaningless on a
    # one-recording fixture (it would cap each side at ceil(0.2*4) = 1);
    # `test_the_caps_bound_a_drawn_side` exercises it on two recordings.
    return m.main(["--runs", runs, "--out", out, "--export-names-from", names,
                   "--female-n", "4", "--male-n", "4", "--per-bout-cap", "4",
                   "--per-rec-frac", "1.0", "--min-female", "1",
                   "--figures-dir", os.path.join(out, "fig"), *extra])


def test_cli_writes_a_stratified_export_with_a_balance_block(run_root, tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    m, runs, names, cm = run_root
    out = str(tmp_path / "pseudo")
    rep = _run(m, runs, names, out, "--no-masks")

    man = json.load(open(f"{out}/manifest.json"))
    assert man["source"] == "pseudo" and man["weight"] == 0.3
    assert man["gates"]["deltas"] == [1, 4, 16]
    assert man["balance"]["female_n"] == 4 and man["balance"]["male_n"] == 4
    assert man["balance"]["female_host_weight"] == 1.0
    assert man["contact_definition"]["contact_kp_units"] == 5.0
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
    assert os.path.exists(f"{out}/summary.md")
    assert rep["verify"]["windows_checked"] > 0        # loader round trip on the REAL export

    ds = V12WindowDataset(out, "train", T=1, train=False)
    assert len(ds) == len(coco["framesets"])
    s = ds[0]
    assert s["kp3d_local"].shape[-2] == len(ds.keypoint_names)


def test_the_balance_weight_is_male_over_female_not_the_reverse(run_root, tmp_path):
    """`female_host_weight` must be `male_n / female_n`. The equal-N case above
    (weight 1.0) cannot tell that ratio apart from its reciprocal; drawing an
    unequal side (female-n 2, male-n 6, both within the fixture's pool of 6
    per host sex) pins the direction: 6 / 2 == 3.0, not 2 / 6."""
    m, runs, names, cm = run_root
    out = str(tmp_path / "pseudo")
    _run(m, runs, names, out, "--no-masks", "--female-n", "2", "--male-n", "6")
    man = json.load(open(f"{out}/manifest.json"))
    assert man["balance"]["female_n"] == 2 and man["balance"]["male_n"] == 6
    assert man["balance"]["female_host_weight"] == 3.0


def test_contact_is_the_min_keypoint_distance_not_the_centroid_separation(run_root, tmp_path):
    """Ruling 2026-09-05 #1. The contact bout's flies are 10 units apart by
    CENTROID -- above the spec's 15-unit cut only in the sense that no frame
    anywhere is below it -- but their nearest keypoints are within 5 units, and
    that is what `contact` now means. The census keeps both definitions."""
    m, runs, names, cm = run_root
    out = str(tmp_path / "pseudo")
    _run(m, runs, names, out, "--no-masks")
    coco = json.load(open(f"{out}/annotations/instances_train.json"))
    b1 = [v["stratum"] for v in coco["framesets"].values() if v["bout"] == 1]
    b2 = [v["stratum"] for v in coco["framesets"].values() if v["bout"] == 2]
    assert b1 and b2
    assert all(s["contact"] and not s["apart"] for s in b1)
    assert all(s["apart"] and not s["contact"] for s in b2)
    assert all(s["kp_dist_units"] < 5.0 for s in b1)
    assert all(s["kp_dist_units"] >= 5.0 for s in b2)
    cen = json.load(open(f"{out}/census.json"))
    assert cen["contact_definition"]["used"] == "min inter-fly KEYPOINT distance"
    # the spec's centroid rule: the 10-unit bout is the ONLY thing it could
    # ever call contact, and on real data nothing at all
    assert "by_stratum_centroid" in cen and "by_host_sex_cell_centroid" in cen


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


def test_the_caps_bound_a_drawn_side(tmp_path, monkeypatch):
    """Ruling #2: a CAPPED side takes no more than `per_rec_frac` of itself
    from one recording and no more than `per_bout_cap` from one bout. (The
    female side is uncapped by construction -- it takes everything.)"""
    cm = None
    for rec in ("recA", "recB"):
        for i, fs in ((1, 1000), (2, 9000)):
            _, _, cm = make_bout(tmp_path, T=40, sep_units=40.0, name=f"bout_{i:05d}",
                                 frame_start=fs, rec=rec, hw=FRAME_HW,
                                 masks_path=str(tmp_path / f"m{rec}{i}" / "sam3_masks.npz"))
    make_calib(tmp_path / "video" / "calibration", cm)
    from pseudo_fixtures import KP_NAMES
    (tmp_path / "names" / "annotations").mkdir(parents=True)
    (tmp_path / "names" / "annotations" / "keypoint_names.json").write_text(json.dumps(KP_NAMES))
    m = _load_cli()
    monkeypatch.setattr(m, "SessionFrames", FakeFrames)
    out = str(tmp_path / "pseudo")
    m.main(["--runs", str(tmp_path / "*" / "pose_mvq_p3b"), "--out", out,
            "--export-names-from", str(tmp_path / "names"), "--female-n", "4", "--male-n", "4",
            "--per-rec-frac", "0.5", "--per-bout-cap", "1", "--min-female", "1", "--no-masks",
            "--figures-dir", os.path.join(out, "fig")])
    coco = json.load(open(f"{out}/annotations/instances_train.json"))
    anchors = [(k.split("/")[0], v["bout"]) for k, v in coco["framesets"].items()
               if v["role"] == "anchor" and v["stratum"]["host_sex"] == "male"]
    per_rec = collections.Counter(r for r, _ in anchors)
    per_bout = collections.Counter(anchors)
    assert len(anchors) == 4 and set(per_rec) == {"recA", "recB"}
    assert max(per_rec.values()) <= 2                     # 0.5 of a 4-row side
    assert max(per_bout.values()) <= 1                    # --per-bout-cap 1


def test_the_uncapped_female_side_draws_every_admissible_row(tmp_path, monkeypatch):
    """Ruling #2: `--female-n` unset means the female-host side takes EVERY
    admissible row, caps and all -- the per-bout/per-recording caps below are
    real (the capped MALE side still obeys them) but must never reach the
    female side. Same two-recording, two-bout-per-recording fixture as
    `test_the_caps_bound_a_drawn_side` above: 3 admissible anchors per
    (host_fly, bout) x 4 bout instances = a female pool of 12. A per_bout_cap
    of 1 would bound a CAPPED side at 4 total (1 per bout instance), so
    drawing all 12 -- 3 per bout, 3x the cap -- is unambiguous proof the cap
    did not apply to the female side, and `sides.female.mode == "all"` names
    the reason why."""
    cm = None
    for rec in ("recA", "recB"):
        for i, fs in ((1, 1000), (2, 9000)):
            _, _, cm = make_bout(tmp_path, T=40, sep_units=40.0, name=f"bout_{i:05d}",
                                 frame_start=fs, rec=rec, hw=FRAME_HW,
                                 masks_path=str(tmp_path / f"m{rec}{i}" / "sam3_masks.npz"))
    make_calib(tmp_path / "video" / "calibration", cm)
    from pseudo_fixtures import KP_NAMES
    (tmp_path / "names" / "annotations").mkdir(parents=True)
    (tmp_path / "names" / "annotations" / "keypoint_names.json").write_text(json.dumps(KP_NAMES))
    m = _load_cli()
    monkeypatch.setattr(m, "SessionFrames", FakeFrames)
    out = str(tmp_path / "pseudo")
    rep = m.main(["--runs", str(tmp_path / "*" / "pose_mvq_p3b"), "--out", out,
                  "--export-names-from", str(tmp_path / "names"), "--male-n", "4",
                  "--per-rec-frac", "0.5", "--per-bout-cap", "1", "--min-female", "1",
                  "--no-masks", "--figures-dir", os.path.join(out, "fig")])
    assert rep["sides"]["female"]["mode"] == "all"

    cen = json.load(open(f"{out}/census.json"))
    n_female_pool = cen["by_host_sex"]["female"]
    assert n_female_pool == 12 > 4          # 4 = what a per_bout_cap=1 side would be bounded to

    coco = json.load(open(f"{out}/annotations/instances_train.json"))
    anchors = [(k.split("/")[0], v["bout"], v["stratum"]["host_sex"])
               for k, v in coco["framesets"].items() if v["role"] == "anchor"]
    female = [(r, b) for r, b, s in anchors if s == "female"]
    male = [(r, b) for r, b, s in anchors if s == "male"]
    assert len(female) == n_female_pool == 12                      # every admissible row drawn
    assert max(collections.Counter(female).values()) == 3          # 3x --per-bout-cap 1: uncapped
    assert len(male) == 4                                          # the male side stayed capped
    assert max(collections.Counter(male).values()) <= 1            # --per-bout-cap 1
    assert max(collections.Counter(r for r, _ in male).values()) <= 2   # 0.5 of a 4-row side


def test_a_female_poor_campaign_stops_at_the_census(run_root, tmp_path, capsys):
    """Controller ruling: below `--min-female` the run writes the census and
    stops rather than quietly rebalancing onto the male fly."""
    m, runs, names, cm = run_root
    out = str(tmp_path / "pseudo_stop")
    m.main(["--runs", runs, "--out", out, "--export-names-from", names,
            "--female-n", "4", "--male-n", "4", "--min-female", "10000",
            "--figures-dir", os.path.join(out, "fig")])
    assert "STOP" in capsys.readouterr().out
    cen = json.load(open(f"{out}/census.json"))
    assert cen["by_host_sex"]["female"] > 0 and "by_stratum" in cen
    assert not (Path(out) / "annotations").exists()
