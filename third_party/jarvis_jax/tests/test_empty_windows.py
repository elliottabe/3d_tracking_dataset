# tests/test_empty_windows.py
"""`scripts/pseudo_labels/extract_empty_windows.py` (Task 5, spec §3.5).

Expectations these tests encode (CLAUDE.md, Task 5 brief Step 1):

  * (a) every emitted centre is a REAL invariant, not "looks about right": it
    is `>= 60 units` (6 mm) from BOTH tracked centroids of its OWN frame and
    projects inside `>= 5` of the 7 cameras (`coarse_centres._project_batch`,
    the same projection `coarse_pass` itself uses) -- checked directly
    against the recording's own tracked trajectory and calibration, not
    against the sampler's self-reported gate values.
  * (b) `--require-tracked` must actually change the output: a frame where a
    fly's centroid is NaN is skipped ENTIRELY with the flag, and (without it)
    such frames really are sampled from -- both directions are asserted, not
    just the flag's presence.
  * (c) the arena-edge stratum (`edge_frac`) draws from CenterDetect reads
    whose existence is low AND whose location is near the recording's own
    tracked-centroid convex hull -- exactly `edge_frac * n` of the rows,
    each stamped `stratum={"kind": "edge"}`.
  * (d) a negative record survives `pseudo_export.write_pseudo_export`
    (used AS-IS, unmodified) with the JSON contract Plan B's loader needs:
    `fly_id == -1`, `negative: true`, a 3-length `center3D`. The loader's own
    read of that contract (`V12WindowDataset`'s NEGATIVES branch) is a
    different task's concern; this only guards the WRITE side.
  * (e) T=2 partners (coordinator ruling 2026-09-06): every delta an
    anchor's `partners` dict names resolves to an actual partner row at
    `f0 + delta`, sharing the anchor's OWN `center3D` exactly, itself
    clearing the same distance/height/camera gates AT ITS OWN FRAME (never
    assumed from the anchor) -- and every partner frame is unique, never
    colliding with another negative's frame.

Synthetic recording: 2 flies on parallel lines 50 units apart (`fly1 = fly0 +
(50,0,0)`), 200 coarse frames at stride 16, real 7-camera DLT calibration
(`mvq_fixtures.cam_P`, the same fixture `test_coarse_track.py` and
`test_mvq_geometry.py` use) so the >= 5 camera check is a REAL projection,
not a fake. Two frame bands are carved out of the otherwise-fully-trackable
timeline:

  * coarse t 100-119 (video frames 1600-1919): fly1's centroid set to NaN --
    the untracked band test (b) needs.
  * coarse t 150-189 (video frames 2400-3023): fly0's read overwritten with a
    LOW-EXISTENCE (< 0.2) centre at the trajectory's own max-X point (an
    axis-aligned extreme of a point cloud is ALWAYS a vertex of its convex
    hull, so this is provably on the hull, not "probably close") plus tiny
    jitter -- the false-peak / arena-edge band test (c) needs.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "scripts", REPO / "third_party" / "jarvis_jax", REPO):
    sys.path.insert(0, str(_p))

from mvq_fixtures import CAMS, cam_P                             # noqa: E402

from pseudo_labels import extract_empty_windows as eew            # noqa: E402

W_IMG, H_IMG = 1936, 448
T_COARSE = 200
STRIDE = 16
NAN_LO, NAN_HI = 100, 120        # coarse-index band, fly1 untracked
EDGE_LO, EDGE_HI = 150, 190      # coarse-index band, fly0 -> false peak near the hull
MIN_DIST_UNITS = 60.0
MIN_HEIGHT_UNITS = 6.0
MIN_CAMS = 5
EDGE_UNITS = 5.0


def _write_calib(calib_dir, cameras, img_w=W_IMG, img_h=H_IMG):
    os.makedirs(calib_dir, exist_ok=True)
    for i, c in enumerate(cameras):
        vals = ", ".join(f"{v:.16g}" for v in cam_P(i).ravel())
        (Path(calib_dir) / f"{c}.yaml").write_text(
            "%%YAML:1.0\n---\nimage_width: %d\nimage_height: %d\n"
            "projectionMatrix: !!opencv-matrix\n   rows: 3\n   cols: 4\n   dt: d\n"
            "   data: [ %s ]\nscale: 10\n" % (img_w, img_h, vals))


@pytest.fixture
def synthetic_tracks(tmp_path):
    cameras = list(CAMS)
    session_dir = str(tmp_path / "session")
    calib_dir = str(Path(session_dir) / "calibration")
    _write_calib(calib_dir, cameras)

    coarse_frame = (np.arange(T_COARSE) * STRIDE).astype(np.int64)
    ang = np.arange(T_COARSE) / T_COARSE * 4 * np.pi
    # amplitudes picked so the tracked-centroid bounding box (inflated 20 %)
    # comfortably clears MIN_DIST_UNITS in every axis -- a box smaller than
    # the clearance in two axes (an earlier, tighter draft: y +-12, z +-6)
    # makes the >= 60-unit gate nearly unsatisfiable everywhere except the
    # box's own x-extremes, which starved the sampler (measured: 28/60 with
    # 87k/88k attempts rejected on distance alone).
    fly0 = np.stack([15 * np.sin(ang), 40 * np.cos(ang * 1.3),
                     20 + 20 * np.cos(ang * 0.7)], axis=1).astype(np.float64)
    fly1 = fly0 + np.array([50.0, 0.0, 0.0])
    X3d = np.stack([fly0.copy(), fly1.copy()], axis=0)             # (2,T,3)
    exist = np.ones((2, T_COARSE), np.float64)

    # untracked band (test b): fly1 (slot 1) goes NaN
    X3d[1, NAN_LO:NAN_HI] = np.nan
    exist[1, NAN_LO:NAN_HI] = np.nan

    # false-peak / arena-edge band (test c): fly0's slot overwritten with a
    # low-exist read at the REAL cloud's max-X point (a provable hull vertex)
    pts_real = np.concatenate([fly0, fly1], axis=0)
    vertex = pts_real[np.argmax(pts_real[:, 0])]
    n_edge = EDGE_HI - EDGE_LO
    rng = np.random.default_rng(0)
    X3d[0, EDGE_LO:EDGE_HI] = vertex[None] + rng.normal(scale=0.2, size=(n_edge, 3))
    exist[0, EDGE_LO:EDGE_HI] = rng.uniform(0.0, 0.2, size=n_edge)

    trackable = np.isfinite(X3d).all(-1) & (exist >= 0.5)

    tracks_path = str(tmp_path / "coarse_tracks.npz")
    np.savez(tracks_path, X3d=X3d.astype(np.float32), exist=exist.astype(np.float32),
             coarse_frame=coarse_frame, trackable=trackable)
    meta = {"session_dir": session_dir, "calib_dir": calib_dir, "cameras": cameras,
            "W": W_IMG, "H": H_IMG, "checkpoint": "test-ckpt"}
    (tmp_path / "coarse_tracks.meta.json").write_text(json.dumps(meta))

    return {"tracks_path": tracks_path, "cameras": cameras, "fly0": fly0, "fly1": fly1,
            "coarse_frame": coarse_frame}


def _rd(synthetic_tracks, **kw):
    return eew.build_recdata_from_tracks("test_rec", synthetic_tracks["tracks_path"],
                                         edge_units=EDGE_UNITS, exist_edge_thresh=0.2, **kw)


# --------------------------------------------------------------------------- (a)
def test_every_negative_clears_distance_and_camera_gates(synthetic_tracks):
    rd = _rd(synthetic_tracks)
    rows, prows, stats = eew.sample_negatives(
        rd, 20, edge_frac=0.0, min_dist_units=MIN_DIST_UNITS, min_height_units=MIN_HEIGHT_UNITS,
        min_cams=MIN_CAMS, rng=np.random.default_rng(1), max_tries=500)
    assert len(rows) == 20, f"expected 20 interior negatives, got {len(rows)} ({stats})"
    for row in rows:
        assert row["stratum"] == {"kind": "interior"}
        cand = np.asarray(row["center3D"], np.float64)
        cents = rd.frame_to_cent.get(int(row["frame"]), [])
        for c in cents:
            d = float(np.linalg.norm(cand - c))
            assert d >= MIN_DIST_UNITS - 1e-6, (
                f"negative at frame {row['frame']} is {d:.2f} units from a tracked centroid "
                f"{c}, below the {MIN_DIST_UNITS}-unit gate")
        n_in = eew._project_inside_count(cand, rd.cam_mats, rd.W, rd.H)
        assert n_in >= MIN_CAMS, (
            f"negative at frame {row['frame']} projects inside only {n_in} cameras, "
            f"below the {MIN_CAMS}-camera gate")


# --------------------------------------------------------------------------- (b)
def test_require_tracked_excludes_the_untracked_band(synthetic_tracks):
    rd = _rd(synthetic_tracks)
    coarse_frame = synthetic_tracks["coarse_frame"]
    nan_frames = set(int(f) for f in coarse_frame[NAN_LO:NAN_HI])

    rows_free, _, _ = eew.sample_negatives(
        rd, 60, edge_frac=0.0, min_dist_units=MIN_DIST_UNITS, min_height_units=MIN_HEIGHT_UNITS,
        min_cams=MIN_CAMS, require_tracked=False, rng=np.random.default_rng(2), max_tries=500)
    rows_req, _, _ = eew.sample_negatives(
        rd, 60, edge_frac=0.0, min_dist_units=MIN_DIST_UNITS, min_height_units=MIN_HEIGHT_UNITS,
        min_cams=MIN_CAMS, require_tracked=True, rng=np.random.default_rng(2), max_tries=500)

    assert not any(int(r["frame"]) in nan_frames for r in rows_req), (
        "--require-tracked emitted a negative inside the untracked (NaN) band")
    # the untracked band must not be VACUOUSLY excluded -- without the flag it
    # really is sampled from, or this assertion would prove nothing.
    assert any(int(r["frame"]) in nan_frames for r in rows_free), (
        "the untracked band was never drawn from even WITHOUT --require-tracked -- "
        "this test's negative assertion above would be vacuous")


# --------------------------------------------------------------------------- (c)
def test_edge_frac_draws_from_the_false_peak_band(synthetic_tracks):
    rd = _rd(synthetic_tracks)
    assert len(rd.edge_candidates) > 0, "no false-peak candidates found near the hull at all"

    n_want = 40
    edge_frac = 0.25
    rows, prows, stats = eew.sample_negatives(
        rd, n_want, edge_frac=edge_frac, min_dist_units=MIN_DIST_UNITS,
        min_height_units=MIN_HEIGHT_UNITS, min_cams=MIN_CAMS,
        rng=np.random.default_rng(3), max_tries=500)
    assert len(rows) == n_want

    edge_rows = [r for r in rows if r["stratum"]["kind"] == "edge"]
    assert len(edge_rows) == round(edge_frac * n_want) == stats["edge_taken"]
    for r in edge_rows:
        assert r["stratum"] == {"kind": "edge"}
        assert r["gates"]["hull_dist_units"] <= EDGE_UNITS
    interior_rows = [r for r in rows if r["stratum"]["kind"] == "interior"]
    assert len(interior_rows) == n_want - len(edge_rows)


# --------------------------------------------------------------------------- (d)
def test_negative_record_survives_write_pseudo_export(synthetic_tracks, tmp_path):
    """Asserts the JSON contract ONLY (fly_id == -1, negative is True, a
    3-length center3D) -- the loader's own read of a negative frameset lands
    with the parallel Plan B task. Anchors AND their T=2 partners are
    written (both are `PseudoRecord(host_fly=None)`, so both hit the same
    negative branch in `write_pseudo_export`) and the partner's `role` and
    `partners` map survive the round trip too."""
    from jarvis_jax.data.pseudo_export import PseudoRecord, write_pseudo_export

    rd = _rd(synthetic_tracks)
    rows, prows, _ = eew.sample_negatives(
        rd, 3, edge_frac=1.0 / 3, min_dist_units=MIN_DIST_UNITS,
        min_height_units=MIN_HEIGHT_UNITS, min_cams=MIN_CAMS,
        rng=np.random.default_rng(4), max_tries=500, partner_deltas=(16,))
    assert len(rows) == 3
    assert len(prows) > 0, "no delta=16 partner cleared the gates -- test would not cover partners"

    export_names = ["Antenna_Base", "EyeL", "EyeR", "Scutellum", "Abd_tip"]
    K, C = len(export_names), len(rd.cameras)

    def _mk(row, role, partners):
        return PseudoRecord(
            recording="test_rec", frame=int(row["frame"]), host_fly=None,
            kp3d=np.zeros((1, K, 3), np.float32), kp2d=np.zeros((1, C, K, 2), np.float32),
            vis=np.zeros((1, C, K), bool), sex=np.array([-1], np.int8),
            stratum=row["stratum"], gates=row["gates"], partners=partners, role=role,
            center3D=np.asarray(row["center3D"], np.float64))

    records = [_mk(row, "negative", row.get("partners", {})) for row in rows]
    records += [_mk(row, "partner", {}) for row in prows]

    def fake_reader(rec, frame):
        return np.zeros((C, H_IMG, W_IMG, 3), np.uint8), np.ones(C, bool)

    out_root = str(tmp_path / "export")
    recordings = {"test_rec": {"calib_dir": rd.calib_dir, "calib_group": "test_rec",
                               "fly_sex": {}, "kp_names": export_names, "n_flies": 2,
                               "behavior": "courtship", "sex": "mixed",
                               "sex_source": "tracks"}}
    write_pseudo_export(out_root, records, export_names=export_names, cameras=rd.cameras,
                        recordings=recordings, checkpoint="test-ckpt", gates={}, weight=0.3,
                        frame_reader=fake_reader, mask_reader=None, write_images=True,
                        progress=False)

    coco = json.load(open(os.path.join(out_root, "annotations", "instances_train.json")))
    neg_framesets = {k: v for k, v in coco["framesets"].items() if "/neg" in k}
    assert len(neg_framesets) == len(rows) + len(prows), (
        f"expected {len(rows)} anchors + {len(prows)} partners, got "
        f"{len(neg_framesets)}: {list(coco['framesets'])}")
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    n_anchor_role = n_partner_role = 0
    for key, fsv in neg_framesets.items():
        assert fsv["fly_id"] == -1, f"{key}: fly_id {fsv['fly_id']} != -1"
        assert fsv["negative"] is True, f"{key}: negative is not True"
        assert len(fsv["center3D"]) == 3, f"{key}: center3D has {len(fsv['center3D'])} entries"
        assert all(np.isfinite(v) for v in fsv["center3D"]), f"{key}: non-finite center3D"
        assert fsv["role"] in ("negative", "partner"), f"{key}: unexpected role {fsv['role']!r}"
        n_anchor_role += fsv["role"] == "negative"
        n_partner_role += fsv["role"] == "partner"
        anns = [ann_by_id[i] for i in fsv["ann_ids"]]
        assert anns, f"{key}: no annotations resolved"
        for a in anns:
            assert a["fly_id"] == -1
            assert a["num_keypoints"] == 0
            assert all(v == 0.0 for v in a["keypoints"])
    assert n_anchor_role == len(rows) and n_partner_role == len(prows), (
        "write_pseudo_export's per-role counts must separate anchors from partners "
        "(coordinator ruling 2026-09-06: count them separately)")
    # at least one anchor's OWN partners map survived, non-empty
    assert any(fsv["partners"] for fsv in neg_framesets.values()), (
        "no anchor frameset carries a non-empty partners map in the written export")


# --------------------------------------------------------------------------- (e)
def test_partners_resolve_and_share_the_anchor_centre(synthetic_tracks):
    """T=2 partners (coordinator ruling 2026-09-06, forward-only f0+delta).

    Under this fixture's source (a) (`coarse_tracks.npz`, stride 16), only
    delta=16 lands ON a coarse-sampled frame -- delta=1/4 fall BETWEEN coarse
    samples, where `frame_to_cent` has no coverage, so `_link_partners`
    OMITS them by design (never a guess). That is itself part of what this
    test checks: some deltas resolve, some are legitimately absent, and
    every one that DOES resolve shares the anchor's centre and clears its
    own gates independently.
    """
    rd = _rd(synthetic_tracks)
    rows, prows, stats = eew.sample_negatives(
        rd, 20, edge_frac=0.0, min_dist_units=MIN_DIST_UNITS, min_height_units=MIN_HEIGHT_UNITS,
        min_cams=MIN_CAMS, rng=np.random.default_rng(5), max_tries=500,
        partner_deltas=(1, 4, 16))
    assert len(rows) == 20
    assert stats["n_partners"] == len(prows) == sum(len(r["partners"]) for r in rows)
    assert stats["n_partners"] > 0, "no partner cleared any delta at all -- test would be vacuous"

    prow_by_frame = {int(p["frame"]): p for p in prows}
    seen_deltas = set()
    for row in rows:
        f0 = int(row["frame"])
        cand = np.asarray(row["center3D"], np.float64)
        for delta_str, pframe in row["partners"].items():
            delta = int(delta_str)
            seen_deltas.add(delta)
            assert int(pframe) == f0 + delta, (
                f"partner frame {pframe} != anchor {f0} + delta {delta}")
            prow = prow_by_frame.get(int(pframe))
            assert prow is not None, f"partner frame {pframe} missing from partner_rows"
            assert prow["role"] == "partner"
            assert prow["anchor_frame"] == f0
            assert prow["delta"] == delta
            assert np.allclose(np.asarray(prow["center3D"], np.float64), cand), (
                f"partner at frame {pframe} does not share anchor {f0}'s centre")
            # the partner independently clears the SAME gates, AT ITS OWN FRAME
            cents = rd.frame_to_cent.get(int(pframe), [])
            assert cents or int(pframe) in rd.frame_to_cent, (
                f"partner frame {pframe} has no tracked-centroid coverage at all -- "
                f"it should have been omitted, not written")
            for c in cents:
                d = float(np.linalg.norm(cand - c))
                assert d >= MIN_DIST_UNITS - 1e-6, (
                    f"partner at frame {pframe} is {d:.2f} units from a tracked centroid, "
                    f"below the {MIN_DIST_UNITS}-unit gate")
            n_in = eew._project_inside_count(cand, rd.cam_mats, rd.W, rd.H)
            assert n_in >= MIN_CAMS, (
                f"partner at frame {pframe} projects inside only {n_in} cameras")
    assert seen_deltas, "no delta ever appeared in any anchor's partners dict"
    assert 16 in seen_deltas, "delta=16 (the one checkable delta on this stride-16 source) never resolved"

    # every partner frame is unique and none collides with an anchor's own frame
    anchor_frames = {int(r["frame"]) for r in rows}
    partner_frames = [int(p["frame"]) for p in prows]
    assert len(partner_frames) == len(set(partner_frames)), "two partners share a frame"
    assert not (set(partner_frames) & anchor_frames), (
        "a partner frame collides with some anchor's own frame")
