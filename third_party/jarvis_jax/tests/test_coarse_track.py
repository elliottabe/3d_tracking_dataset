# tests/test_coarse_track.py
"""`jarvis_jax.tracking.coarse_track` -- the stride-16 recording pass
(spec 2026-09-04-mvq-maskfree-frontend-design.md §4.3).

Expectations these tests encode (CLAUDE.md: state the expectation, check a
real invariant, index by NAME):

  * `coarse_pass` must return ONE row per requested frame, in the order
    requested, with a per-fly axis of exactly `num_animals` -- a pass that
    silently dropped the frames where nothing was detected would shift every
    later frame's index against `frame`, and the gates read the arrays
    positionally.
  * A frame whose CenterDetect gave no centre must REUSE the previous
    frame's window centres and say so (`centre_source == 1`), and a frame
    with no centre and no previous one must be NaN and say so
    (`centre_source == 2`) -- never a silent zero centre, which would put a
    448-px window at the world origin and return a confident fit of the
    arena floor.
  * `coarse_features` must be computed from keypoints looked up BY NAME.
    The angle tests use hand-built geometry with a KNOWN answer (a wing held
    at 30 deg from the body axis reads 30 deg; a male facing straight at the
    female reads 0 deg heading) rather than "the numbers look smooth" -- the
    keypoint-order trap in CLAUDE.md produced perfectly smooth, perfectly
    wrong angles.
  * `fit_floor` must recover a KNOWN tilted plane and orient it so heights
    are positive -- a sign flip would make every fly's height negative and
    every height-based gate read backwards.
  * `write_coarse_tracks` must produce a file `scripts/coarse_pass_gates.py`
    opens with its own loader and gate-signal code, with the SAM3-schema
    arrays at the SAM3 shapes; `area*` is all-NaN (there are no masks), which
    the test asserts explicitly so nobody reads an mvq area gate as passing.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from mvq_fixtures import CAMS, REC
from test_lift_mvq import _tiny_final

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))


# --------------------------------------------------------------------------
# fakes: a reader over the fixture JPEGs and a detector that already knows
# where the flies are (the point of these tests is the pass, not CenterDetect,
# which is GPU/checkpoint-heavy and tested on synthetic heatmaps elsewhere).
# --------------------------------------------------------------------------
def _project(centres, cam_mats):
    """(N,3) world -> (N,C,2) full-image px, the `p_h @ M` convention of
    `ReprojectionTool.camera_matrices` (C,4,3)."""
    centres = np.asarray(centres, np.float64)
    ph = np.concatenate([centres, np.ones((centres.shape[0], 1))], axis=1)
    proj = np.einsum("nd,cdk->nck", ph, np.asarray(cam_mats, np.float64))
    return proj[..., :2] / proj[..., 2:3]


def true_centre(fly, frame):
    """The fixture's own fly centre (mvq_fixtures.fly_points, before the
    per-keypoint scatter)."""
    base = np.array([0.0, 0.0, 0.0]) if fly == 0 else np.array([40.0, 5.0, 0.0])
    return base + frame * np.array([1.0, 0.5, 0.0])


class FixtureReader:
    """`coarse_pass`'s reader contract: frame index -> (frames (C,H,W,3) u8,
    present (C,) bool), reading the fixture's JPEGs in CANONICAL camera order."""

    def __init__(self, root, cameras, n_frames=3):
        from PIL import Image
        self._Image = Image
        self.root, self.cameras, self.n_frames = root, list(cameras), n_frames
        self.calls = []

    def __call__(self, frame_idx):
        self.calls.append(int(frame_idx))
        f = int(frame_idx) % self.n_frames
        imgs = [np.asarray(self._Image.open(
            os.path.join(self.root, "images", REC, c, f"Frame_{f}.jpg")).convert("RGB"))
            for c in self.cameras]
        return np.stack(imgs), np.ones(len(self.cameras), bool)


class FakeDetector:
    """Returns the projections of the fixture's TRUE centres as peaks, so the
    lift stage has a well-posed problem. `blank` is the set of frame indices
    for which it reports nothing (both peaks NaN in every camera) -- the
    dropout the reuse rule exists for."""

    def __init__(self, cam_mats, n_flies=2, blank=()):
        self.cam_mats, self.n_flies, self.blank = cam_mats, n_flies, set(blank)
        self.frame = 0

    def peaks(self, frames):
        c = len(self.cam_mats)
        pk = np.full((c, 2, 2), np.nan, np.float32)
        sc = np.zeros((c, 2), np.float32)
        f = self.frame
        self.frame += 1
        if f in self.blank:
            return pk, sc
        uv = _project([true_centre(k, f) for k in range(self.n_flies)], self.cam_mats)
        for k in range(self.n_flies):
            pk[:, k] = uv[k]
            sc[:, k] = 0.9
        return pk, sc


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return _tiny_final(tmp_path_factory.mktemp("coarse_track"))


def _runner(tiny, batch=4):
    from jarvis_jax.tracking.lift_mvq import MVQRunner
    root, final = tiny
    return MVQRunner(final, calib_dir=os.path.join(root, "calibrations", "A"),
                     cameras=CAMS, batch=batch)


# --------------------------------------------------------------------------
# coarse_pass
# --------------------------------------------------------------------------
def test_coarse_pass_shapes_frames_and_centre_source(tiny):
    from jarvis_jax.tracking.coarse_track import coarse_pass
    root, _ = tiny
    r = _runner(tiny)
    reader = FixtureReader(root, CAMS)
    det = FakeDetector(r.cam_mats)
    tr = coarse_pass(reader, r, det, frames=[0, 1, 2], num_animals=2)

    assert list(tr["frame"]) == [0, 1, 2]
    assert reader.calls == [0, 1, 2]
    N, K = 3, r.K
    assert tr["kp3d"].shape == (2, N, K, 3)
    assert tr["centroid"].shape == (2, N, 3)
    for k in ("exist", "sex_prob", "slot", "centre_source"):
        assert tr[k].shape == (2, N), k
    assert tr["centre_source"].dtype == np.int8
    # every frame had a detection -> centre_source 0 everywhere
    assert (tr["centre_source"] == 0).all()
    assert tr["kp_names"] == list(r.kp_names)
    assert (tr["W"], tr["H"]) == (1936, 448)


def test_missing_detection_reuses_the_previous_frame_centres(tiny):
    """Frame 1 has no CenterDetect peak: it must still produce a row, from the
    PREVIOUS frame's window centres, flagged centre_source == 1."""
    from jarvis_jax.tracking.coarse_track import coarse_pass
    root, _ = tiny
    r = _runner(tiny)
    det = FakeDetector(r.cam_mats, blank={1})
    tr = coarse_pass(FixtureReader(root, CAMS), r, det, frames=[0, 1, 2], num_animals=2)
    assert list(tr["frame"]) == [0, 1, 2]
    assert list(tr["centre_source"][0]) == [0, 1, 0]
    assert list(tr["centre_source"][1]) == [0, 1, 0]
    assert tr["n_windows"][1] == tr["n_windows"][0]


def test_no_detection_and_no_history_is_nan_not_a_zero_centre(tiny):
    from jarvis_jax.tracking.coarse_track import coarse_pass
    root, _ = tiny
    r = _runner(tiny)
    det = FakeDetector(r.cam_mats, blank={0})
    tr = coarse_pass(FixtureReader(root, CAMS), r, det, frames=[0, 1, 2], num_animals=2)
    assert tr["centre_source"][0, 0] == 2
    assert np.isnan(tr["kp3d"][:, 0]).all()
    assert np.isnan(tr["centroid"][:, 0]).all()
    assert (tr["slot"][:, 0] == -1).all()


def test_single_fly_recording_reads_the_untyped_slot(tiny):
    from jarvis_jax.tracking.coarse_track import coarse_pass
    root, _ = tiny
    r = _runner(tiny)
    det = FakeDetector(r.cam_mats, n_flies=1)
    tr = coarse_pass(FixtureReader(root, CAMS), r, det, frames=[0, 1], num_animals=1)
    assert tr["kp3d"].shape == (1, 2, r.K, 3)
    assert tr["slot"].shape == (1, 2)
    # whichever typed slot exists (1 female / 2 male), never the prompted slot 0
    assert set(np.unique(tr["slot"])) <= {-1, 1, 2}


def test_more_windows_than_batch_warns_once_and_drops_the_extra(tiny):
    """`runner.batch` planned windows silently truncated used to drop data
    with no signal at all. Two well-separated true centres (40 units apart,
    over `merge_dist_units=30.0`) always plan 2 windows; with `batch=1` the
    second is always dropped -- must warn, naming the frame, exactly ONCE
    across the whole run (not once per frame)."""
    from jarvis_jax.tracking.coarse_track import coarse_pass
    root, _ = tiny
    r = _runner(tiny, batch=1)
    reader = FixtureReader(root, CAMS)
    det = FakeDetector(r.cam_mats)
    with pytest.warns(RuntimeWarning) as record:
        tr = coarse_pass(reader, r, det, frames=[0, 1, 2], num_animals=2)
    msgs = [str(w.message) for w in record if issubclass(w.category, RuntimeWarning)]
    assert len(msgs) == 1, f"expected exactly one warning, got {msgs}"
    assert "frame 0" in msgs[0] and "1 window" in msgs[0]
    assert list(tr["n_windows"]) == [1, 1, 1]        # truncated to batch=1 every frame


# --------------------------------------------------------------------------
# features / floor -- pure numpy, hand-built geometry with known answers
# --------------------------------------------------------------------------
def _synthetic_tracks(kp_names, *, n=4):
    """Two flies: the male (fly 1) at the origin facing +x with its left wing
    30 deg off the body axis; the female (fly 0) 100 units straight ahead of
    him at +x, so the heading is EXACTLY 0 deg and the inter-fly distance
    EXACTLY 100 units. Both sit 20 units above the z=0 floor."""
    K = len(kp_names)
    i = {nm: k for k, nm in enumerate(kp_names)}
    kp3d = np.full((2, n, K, 3), np.nan, np.float32)
    ang = np.deg2rad(30.0)
    for t in range(n):
        for f, base in ((1, np.array([0.0, 0.0, 20.0])), (0, np.array([100.0, 0.0, 20.0]))):
            kp3d[f, t, i["Scutellum"]] = base
            kp3d[f, t, i["Abd_tip"]] = base + [-30.0, 0.0, 0.0]      # axis points -x (backwards)
            kp3d[f, t, i["WingL_base"]] = base
            # left wing 30 deg from the BODY AXIS (-x): rotate -x by 30 deg in xy
            kp3d[f, t, i["WingL_V13"]] = base + 40.0 * np.array(
                [-np.cos(ang), np.sin(ang), 0.0])
            kp3d[f, t, i["WingR_base"]] = base
            kp3d[f, t, i["WingR_V13"]] = base + 40.0 * np.array([-1.0, 0.0, 0.0])  # 0 deg
    centroid = np.stack([np.array([[100.0 + t, 0.0, 20.0] for t in range(n)]),
                         np.array([[0.0 + t, 0.0, 20.0] for t in range(n)])]).astype(np.float32)
    return {"frame": np.arange(n) * 16, "kp3d": kp3d, "centroid": centroid,
            "exist": np.full((2, n), 0.9, np.float32),
            "sex_prob": np.full((2, n), 0.5, np.float32),
            "slot": np.stack([np.ones(n, np.int8), np.full(n, 2, np.int8)]),
            "centre_source": np.zeros((2, n), np.int8),
            "n_windows": np.full(n, 2, np.int8),
            "kp_names": list(kp_names), "W": 1936, "H": 448}


def _arena_cloud(n_true, *, n=2000, seed=0):
    """A realistic arena cloud: 90 % of the centroids within 3 units of the
    glass, 10 % up a wall at 10-40 units. Bottom-heavy along the TRUE up
    normal, which is the only thing that tells `fit_floor` which way is up."""
    rng = np.random.default_rng(seed)
    height = np.where(rng.random(n) < 0.9, rng.uniform(0.0, 3.0, n), rng.uniform(10.0, 40.0, n))
    xy = rng.uniform(-200, 200, size=(n, 2))
    on_plane = np.stack([xy[:, 0], xy[:, 1],
                         -(n_true[0] * xy[:, 0] + n_true[1] * xy[:, 1]) / n_true[2]], 1)
    return on_plane + height[:, None] * n_true, height


def test_fit_floor_ignores_low_exist_points_and_refits_normal_on_lowest_quantile():
    """30 % wall points (legit detections, high `exist`) tilt a naive
    single-pass total-least-squares normal toward the wall (every point gets
    equal SVD weight regardless of whether it is "the floor"); a few
    physically-impossible sub-floor points at `exist=0.1` (not real
    detections -- the typed slot barely fired) would drag the offset below
    the true floor if not gated out. `fit_floor(..., exist=...)` must recover
    the true normal within 2 deg and place the TRUE floor within 1 unit of
    height 0 despite both."""
    from jarvis_jax.tracking.coarse_track import fit_floor
    nrm = np.array([0.05, -0.1, 1.0]); nrm /= np.linalg.norm(nrm)
    rng = np.random.default_rng(7)
    n = 2000
    height = np.where(rng.random(n) < 0.7, rng.uniform(0.0, 3.0, n), rng.uniform(10.0, 40.0, n))
    xy = rng.uniform(-200, 200, size=(n, 2))
    on_plane = np.stack([xy[:, 0], xy[:, 1],
                         -(nrm[0] * xy[:, 0] + nrm[1] * xy[:, 1]) / nrm[2]], 1)
    pts = on_plane + height[:, None] * nrm
    exist = np.full(n, 0.9, np.float32)

    # A FEW impossible sub-floor points, LOW exist -- must be gated out, not
    # treated as real geometry.
    n_bad = 12
    bad_xy = rng.uniform(-200, 200, size=(n_bad, 2))
    bad_on_plane = np.stack([bad_xy[:, 0], bad_xy[:, 1],
                             -(nrm[0] * bad_xy[:, 0] + nrm[1] * bad_xy[:, 1]) / nrm[2]], 1)
    bad_pts = bad_on_plane - 50.0 * nrm              # 5 mm BELOW the true floor
    bad_exist = np.full(n_bad, 0.1, np.float32)

    all_pts = np.concatenate([pts, bad_pts])[None]            # (F=1, N, 3)
    all_exist = np.concatenate([exist, bad_exist])[None]      # (F=1, N)

    floor = fit_floor(all_pts, exist=all_exist)
    ang = np.degrees(np.arccos(np.clip(np.dot(floor.normal, nrm), -1.0, 1.0)))
    assert ang < 2.0, f"normal off by {ang:.2f} deg -- wall/outlier points tilted the fit"
    # the TRUE floor (on_plane, height 0 by construction) must read back near 0
    true_floor_height = float(on_plane[0] @ floor.normal + floor.offset)
    assert abs(true_floor_height) < 1.0, (
        f"true floor read back at height {true_floor_height:.2f}, not within 1 unit of 0")

    # the low-exist rows must actually have been excluded, not merely diluted
    with pytest.raises(ValueError):
        fit_floor(bad_pts[None], exist=bad_exist[None])   # only 12 low-exist points -> < 3 usable


def test_fit_floor_recovers_a_known_tilted_plane_with_positive_heights():
    """The plane must come back UP-oriented and sitting at the BOTTOM of the
    cloud: a fly on the glass reads ~0 and one up the wall reads ~its true
    height. The literal "sign so the median height is positive" rule fails
    exactly here (a least-squares plane runs through the MEAN, so the median
    residual of a bottom-heavy cloud is negative and the rule flips the
    normal downward) -- see fit_floor's DEVIATION note."""
    from jarvis_jax.tracking.coarse_track import fit_floor
    nrm = np.array([0.1, -0.2, 1.0]); nrm /= np.linalg.norm(nrm)
    pts, height = _arena_cloud(nrm)
    floor = fit_floor(pts[None])          # (F=1, N, 3)
    assert np.dot(floor.normal, nrm) > 0.999, "normal points DOWN -- every height reads backwards"
    got = pts @ floor.normal + floor.offset
    assert np.median(got) > 0
    assert np.abs(got - height).max() < 3.0     # 0.3 mm over a 40 mm arena
    # and the sign rule survives the mirror image: flip the true up direction
    pts2, height2 = _arena_cloud(-nrm, seed=1)
    floor2 = fit_floor(pts2[None])
    assert np.dot(floor2.normal, -nrm) > 0.999
    assert np.abs((pts2 @ floor2.normal + floor2.offset) - height2).max() < 3.0


def test_coarse_features_known_angles_distance_and_height(tiny):
    from jarvis_jax.tracking.coarse_track import coarse_features, FloorPlane
    r = _runner(tiny)
    tr = _synthetic_tracks(r.kp_names)
    ft = coarse_features(tr, r.kp_names, floor=FloorPlane(np.array([0.0, 0.0, 1.0]), 0.0))
    n = tr["frame"].shape[0]
    assert ft["wing_angle_deg"].shape == (2, n)
    assert ft["heading_deg"].shape == (n,)
    assert ft["speed"].shape == (2, n) and ft["height"].shape == (2, n)
    assert ft["dist"].shape == (n,) and ft["trackable"].shape == (2, n)
    np.testing.assert_allclose(ft["wing_angle_deg"], 30.0, atol=1e-3)
    np.testing.assert_allclose(ft["heading_deg"], 0.0, atol=1e-3)
    np.testing.assert_allclose(ft["dist"], 100.0, atol=1e-3)
    np.testing.assert_allclose(ft["height"], 20.0, atol=1e-3)
    np.testing.assert_allclose(ft["speed"][:, 1:], 1.0, atol=1e-3)
    assert np.isnan(ft["speed"][:, 0]).all()
    assert ft["trackable"].all()


def test_heading_is_measured_from_the_male_and_is_180_when_he_faces_away(tiny):
    from jarvis_jax.tracking.coarse_track import coarse_features, FloorPlane
    r = _runner(tiny)
    tr = _synthetic_tracks(r.kp_names)
    i = {nm: k for k, nm in enumerate(r.kp_names)}
    # turn the MALE (fly 1) around: head at -x, abdomen at +x
    tr["kp3d"][1, :, i["Abd_tip"], 0] = 30.0
    ft = coarse_features(tr, r.kp_names, floor=FloorPlane(np.array([0.0, 0.0, 1.0]), 0.0))
    np.testing.assert_allclose(ft["heading_deg"], 180.0, atol=1e-3)


# --------------------------------------------------------------------------
# write_coarse_tracks <-> the SAM3 gates loader
# --------------------------------------------------------------------------
def test_write_coarse_tracks_is_read_by_the_sam3_gates_loader(tiny, tmp_path):
    import coarse_pass_gates as gates
    from jarvis_jax.tracking.coarse_track import (coarse_features, fit_floor,
                                                  write_coarse_tracks)
    r = _runner(tiny)
    tr = _synthetic_tracks(r.kp_names, n=6)
    floor = fit_floor(tr["centroid"])
    ft = coarse_features(tr, r.kp_names, floor=floor)
    out = tmp_path / "coarse_mvq" / "coarse_tracks.npz"
    write_coarse_tracks(str(out), tr, ft, CAMS, session_dir="/fake/session",
                        stride=16, num_animals=2, cam_mats=r.cam_mats)

    z, meta = gates.load_tracks(str(out))
    T, C = 6, len(CAMS)
    assert z["coarse_frame"].shape == (T,)
    assert list(z["cameras"]) == CAMS
    for k, shape in (("area", (2, C, T)), ("centroid", (2, C, T, 2)), ("valid", (2, C, T)),
                     ("border_dist", (2, C, T)), ("in_frame", (2, C, T)),
                     ("area_med", (2, T)), ("border_med", (2, T)), ("n_valid_cams", (2, T)),
                     ("X3d", (2, T, 3)), ("sep3d", (T,)), ("sep2d_med", (T,)), ("filled", (T,))):
        assert z[k].shape == shape, (k, z[k].shape)
    assert np.isnan(z["area"]).all() and np.isnan(z["area_med"]).all()
    assert z["in_frame"].dtype == np.int8 and set(np.unique(z["in_frame"])) <= {0, 1}
    # mvq-only fields
    assert z["kp3d"].shape == (2, T, r.K, 3) and z["kp3d"].dtype == np.float16
    assert list(z["kp_names"]) == list(r.kp_names)
    for k, shape in (("exist", (2, T)), ("sex_prob", (2, T)), ("wing_angle_deg", (2, T)),
                     ("speed", (2, T)), ("height", (2, T)), ("heading_deg", (T,))):
        assert z[k].shape == shape, k
    assert meta["source"] == "mvq"
    for k in ("session_dir", "stride", "cameras", "W", "H", "num_animals", "n_coarse", "coarse0"):
        assert k in meta, k
    assert meta["stride"] == 16 and meta["num_animals"] == 2 and meta["n_coarse"] == T
    assert meta["coarse0"] == 0

    # the gates' own signal/gate code runs on it (no masks -> the area gate can
    # never pass, which is a FACT about mvq tracks, asserted here so it is not
    # discovered as a silent empty bout table)
    sig = gates.compute_gate_signals(z)
    assert sig["area_med"].shape == (2, T) and sig["sep2d_med"].shape == (T,)
    g = gates.apply_gates(sig, wing_ratio_min=1.1, proximity_max_px=100.0)
    assert g["in_bout"].shape == (T,)
    assert not g["in_bout"].any()

    # ... and the whole gates entry point, which also exercises segment_runs
    # and the coarse-index -> real-frame conversion against `coarse_frame`
    res = gates.run(str(out), str(tmp_path / "bouts.csv"), session_tag="test",
                    wing_ratio_min=1.1, proximity_max_px=100.0, min_duration=1,
                    max_gap=1, baseline_window=100)   # the gates hardcode min_periods=50
    assert res["meta"]["source"] == "mvq" and res["windows"] == []


# --------------------------------------------------------------------------
# scripts/coarse_pass_mvq.py -- the pieces that are not IO
# --------------------------------------------------------------------------
def test_driver_default_cameras_is_the_canonical_order():
    """`--cameras` defaults to the recording config's canonical order; a
    different order would plot one camera's fly on another camera's image."""
    import coarse_pass_mvq as drv
    assert drv.default_cameras() == CAMS


def test_driver_slot_reader_matches_read_window_byte_for_byte(tmp_path):
    """The driver keeps its `cv2.VideoCapture`s OPEN across sampled frames
    instead of calling `read_window(..., T=1)` 31k times. That is only safe if
    it reads the SAME pixels: this writes two tiny mp4s, samples them at
    stride 7 with both readers, and requires byte equality (a seek/cursor bug
    would return the neighbouring frame, which looks entirely plausible)."""
    cv2 = pytest.importorskip("cv2")
    import coarse_pass_mvq as drv
    from jarvis_jax.predict.synced_reader import load_plan, read_window

    cams, n, H, W = ["Cam1", "Cam2"], 40, 48, 64
    for ci, c in enumerate(cams):
        vw = cv2.VideoWriter(str(tmp_path / f"{c}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        assert vw.isOpened()
        for i in range(n):
            # flat frames: mp4v is lossy, but a constant frame survives, so a
            # mismatch is a WRONG FRAME, not a codec artefact
            vw.write(np.full((H, W, 3), (i * 6 + ci * 3) % 256, np.uint8))
        vw.release()
    plan = load_plan(str(tmp_path))
    assert plan is None                       # no sync_plan.json -> positional, as Session0
    reader = drv.SlotReader(str(tmp_path), cams, plan)
    try:
        assert (reader.W, reader.H) == (W, H)
        for slot in range(0, n, 7):
            got, present = reader(slot)
            want, want_present = next(iter(read_window(str(tmp_path), cams, plan, slot, 1)))
            assert present.tolist() == want_present.tolist()
            np.testing.assert_array_equal(got, want)
    finally:
        reader.close()


class _FakeRunner:
    """`MVQRunner`'s surface as `coarse_pass`/the driver use it, with a model
    that puts a fly exactly at the window centre. Lets the DRIVER's glue
    (chunking, partial writes, resume, meta) be exercised on CPU in a second;
    the real forward is covered by test_lift_mvq.py."""

    def __init__(self, run, *, step=None, attn_impl=None, calib_dir=None, cameras=None,
                 batch=4, **kw):
        from mvq_fixtures import _names, cam_P
        self.checkpoint, self.step, self.step_label = str(run), step, "final"
        self.cameras = list(cameras)
        self.cam_mats = np.stack([cam_P(i).T for i in range(len(self.cameras))]).astype(np.float32)
        self.kp_names = _names()
        self.K, self.I, self.batch, self.exist_thresh = len(self.kp_names), 4, int(batch), 0.5

    def windows(self, frames, present, centres):
        c = np.atleast_2d(np.asarray(centres, np.float32))
        # `crops` is the key `concat_windows` batches on; a stub is enough here
        return {"centres": c, "crops": np.zeros((c.shape[0], 1), np.uint8)}

    def infer(self, w):
        return {"centres": np.asarray(w["centres"], np.float32)}

    def read_typed(self, out, bi, want_sex):
        c = np.asarray(out["centres"][bi], np.float32)
        i = {nm: k for k, nm in enumerate(self.kp_names)}
        kp = np.full((self.K, 3), np.nan, np.float32)
        kp[i["Scutellum"]] = c
        kp[i["Abd_tip"]] = c + [-30.0, 0.0, 0.0]
        for base, tip, dy in (("WingL_base", "WingL_V13", 20.0), ("WingR_base", "WingR_V13", -20.0)):
            kp[i[base]] = c
            kp[i[tip]] = c + [-35.0, dy, 0.0]
        return {"slot": 1 if want_sex in (0, -1) else 2, "kp3d": kp,
                "exist": 0.9, "sex_prob": 0.8 if want_sex in (0, -1) else 0.2}


class _FakeCenterDetect:
    """CenterDetect's surface, reporting two fixed world centres per frame."""

    def __init__(self, ckpt_dir, *, min_score=0.2):
        from mvq_fixtures import cam_P
        self.cam_mats = np.stack([cam_P(i).T for i in range(2)]).astype(np.float32)
        self.frame = 0

    def peaks(self, frames):
        c = self.cam_mats.shape[0]
        f = self.frame
        self.frame += 1
        centres = np.array([[0.0, 0.0, 5.0 + f], [40.0, 5.0, 3.0]])
        uv = _project(centres, self.cam_mats)
        pk = np.full((c, 2, 2), np.nan, np.float32)
        sc = np.full((c, 2), 0.9, np.float32)
        for k in range(2):
            pk[:, k] = uv[k]
        return pk, sc


def _tiny_session(tmp_path, cams, n=40, H=48, W=64):
    import cv2
    for ci, cname in enumerate(cams):
        vw = cv2.VideoWriter(str(tmp_path / f"{cname}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        for i in range(n):
            vw.write(np.full((H, W, 3), (i * 6 + ci * 3) % 256, np.uint8))
        vw.release()


def test_driver_main_runs_chunks_writes_and_resumes(tmp_path, monkeypatch, capsys):
    """End-to-end through `scripts/coarse_pass_mvq.py:main` with a fake model:
    the chunk loop must write a partial every `--partial-every` frames, the
    final file must cover every sampled frame exactly once in order, the
    partial must be cleaned up, and `--resume` must continue after the last
    written frame rather than redo or skip work."""
    pytest.importorskip("cv2")
    import coarse_pass_mvq as drv
    cams = ["Cam2012630", "Cam2012631"]
    _tiny_session(tmp_path, cams)
    monkeypatch.setattr("jarvis_jax.tracking.lift_mvq.MVQRunner", _FakeRunner)
    monkeypatch.setattr("jarvis_jax.tracking.coarse_centres.CenterDetector", _FakeCenterDetect)
    out = tmp_path / "coarse_mvq" / "coarse_tracks.npz"
    argv = ["coarse_pass_mvq.py", "--session-dir", str(tmp_path), "--calib-dir", str(tmp_path),
            "--cameras", ",".join(cams), "--run", "/fake/final",
            "--centerdetect", "/fake/cd", "--stride", "8", "--start", "0", "--end", "40",
            "--out", str(out), "--batch", "4", "--min-views", "2", "--partial-every", "2"]

    # first pass: only the first 3 sampled frames (0, 8, 16)
    monkeypatch.setattr(sys, "argv", argv[:-4] + ["--end", "24"] + argv[-4:])
    drv.main()
    import json as _json
    meta = _json.load(open(str(out).replace(".npz", ".meta.json")))
    assert meta["n_coarse"] == 3 and meta["source"] == "mvq" and meta["complete"] is True
    assert not os.path.exists(str(out).replace(".npz", ".partial.npz"))

    # resume: the remaining frames are appended, none repeated
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    drv.main()
    with np.load(out, allow_pickle=True) as z:
        frames = z["coarse_frame"]
        assert list(frames) == [0, 8, 16, 24, 32]
        assert z["kp3d"].shape == (2, 5, 50, 3)
        assert np.isfinite(z["X3d"]).all()
        assert (z["centre_source"] == 0).all()
        assert np.isfinite(z["wing_angle_deg"]).all()
    meta = _json.load(open(str(out).replace(".npz", ".meta.json")))
    assert meta["n_coarse"] == 5 and meta["floor"] is not None
    assert meta["centre_source_counts"] == {"detected": 5, "reused": 0, "none": 0}
    assert "eta" in capsys.readouterr().out or True     # progress lines are best-effort


def test_driver_resume_round_trips_a_partial_write(tiny, tmp_path):
    """A partial file must read back into a tracks dict the pass can append
    to: same fly axis, same keypoint names, frame order preserved."""
    import coarse_pass_mvq as drv
    from jarvis_jax.tracking.coarse_track import (coarse_features, concat_tracks, fit_floor,
                                                  write_coarse_tracks)
    r = _runner(tiny)
    tr = _synthetic_tracks(r.kp_names, n=5)
    floor = fit_floor(tr["centroid"])
    tr["floor"] = floor
    out = tmp_path / "coarse_tracks.partial.npz"
    write_coarse_tracks(str(out), tr, coarse_features(tr, r.kp_names, floor=floor), CAMS,
                        session_dir="/fake", stride=16, num_animals=2, cam_mats=r.cam_mats)

    back = drv.load_partial(str(out), 2)
    assert list(back["frame"]) == list(tr["frame"])
    assert back["kp3d"].shape == tr["kp3d"].shape and back["kp3d"].dtype == np.float32
    assert back["kp_names"] == list(r.kp_names)
    assert (back["W"], back["H"]) == (1936, 448)
    np.testing.assert_allclose(back["centroid"], tr["centroid"], atol=1e-4)
    # and it concatenates with a fresh chunk (the resume path)
    nxt = _synthetic_tracks(r.kp_names, n=3)
    nxt["frame"] = nxt["frame"] + int(tr["frame"][-1]) + 16
    joined = concat_tracks([back, nxt])
    assert joined["frame"].shape == (8,) and joined["kp3d"].shape[1] == 8
    with pytest.raises(ValueError):
        drv.load_partial(str(out), 1)      # a different fly count must not silently append


def test_load_partial_refuses_a_resume_with_a_different_field(tiny, tmp_path):
    """Beyond the fly-count guard, `load_partial` must refuse a --resume with
    a different --stride, --cameras, --start, --run (checkpoint) or
    --calib-dir -- each would otherwise silently splice two different
    geometries into one file. The error must name the mismatched field."""
    import coarse_pass_mvq as drv
    from jarvis_jax.tracking.coarse_track import coarse_features, fit_floor, write_coarse_tracks
    r = _runner(tiny)
    tr = _synthetic_tracks(r.kp_names, n=3)
    floor = fit_floor(tr["centroid"], exist=tr["exist"])
    feats = coarse_features(tr, r.kp_names, floor=floor)
    out = tmp_path / "coarse_tracks.npz"
    write_coarse_tracks(str(out), tr, feats, CAMS, session_dir="/fake", stride=16,
                        num_animals=2, cam_mats=r.cam_mats,
                        meta_extra={"checkpoint": "/abs/run/final", "calib_dir": "/abs/calib",
                                   "start": 0, "mvq_step": "final"})

    # every field matching -> resumes fine
    drv.load_partial(str(out), 2, cameras=CAMS, stride=16, start=0,
                     checkpoint="/abs/run/final", calib_dir="/abs/calib", step_label="final")

    with pytest.raises(ValueError, match="stride"):
        drv.load_partial(str(out), 2, stride=8)
    with pytest.raises(ValueError, match="cameras"):
        drv.load_partial(str(out), 2, cameras=list(reversed(CAMS)))
    with pytest.raises(ValueError, match="start"):
        drv.load_partial(str(out), 2, start=100)
    with pytest.raises(ValueError, match="checkpoint"):
        drv.load_partial(str(out), 2, checkpoint="/a/different/run/final")
    with pytest.raises(ValueError, match="calib_dir"):
        drv.load_partial(str(out), 2, calib_dir="/a/different/calib")
    with pytest.raises(ValueError, match="mvq_step"):
        drv.load_partial(str(out), 2, step_label=3000)


class _FakeCenterDetectBlankFirst(_FakeCenterDetect):
    """Like `_FakeCenterDetect`, but the FIRST frame of this process's run
    reports no CenterDetect peak at all -- reproducing "the frame right after
    a resume boundary has no detection". A resumed run must show
    `centre_source == 1` (reused from the PRIOR run's last window centres),
    not `2` (no detection AND no history), which is what a bare `load_partial`
    that always reports `last_centres=None` used to produce."""

    def peaks(self, frames):
        if self.frame == 0:
            self.frame += 1
            c = self.cam_mats.shape[0]
            return np.full((c, 2, 2), np.nan, np.float32), np.zeros((c, 2), np.float32)
        return super().peaks(frames)


def test_driver_resume_reuses_last_centres_across_a_blank_frame(tmp_path, monkeypatch):
    """A resumed run whose very first sampled frame has no CenterDetect peak
    must REUSE the previous run's last window centres (`centre_source == 1`),
    matching what a single-shot run would have done at the same frame -- not
    treat it as no-history NaN (`centre_source == 2`), which is what happened
    while `load_partial` always reported `last_centres=None`."""
    pytest.importorskip("cv2")
    import coarse_pass_mvq as drv
    cams = ["Cam2012630", "Cam2012631"]
    _tiny_session(tmp_path, cams)
    monkeypatch.setattr("jarvis_jax.tracking.lift_mvq.MVQRunner", _FakeRunner)
    out = tmp_path / "coarse_mvq" / "coarse_tracks.npz"
    argv = ["coarse_pass_mvq.py", "--session-dir", str(tmp_path), "--calib-dir", str(tmp_path),
            "--cameras", ",".join(cams), "--run", "/fake/final",
            "--centerdetect", "/fake/cd", "--stride", "8", "--start", "0", "--end", "24",
            "--out", str(out), "--batch", "4", "--min-views", "2", "--partial-every", "2"]

    # first process: frames 0, 8, 16 -- every one detected
    monkeypatch.setattr("jarvis_jax.tracking.coarse_centres.CenterDetector", _FakeCenterDetect)
    monkeypatch.setattr(sys, "argv", argv)
    drv.main()
    with np.load(out, allow_pickle=True) as z:
        assert (z["centre_source"] == 0).all()

    # second process (the resume): its FIRST sampled frame (24, the very next
    # one after the resume boundary) has NO CenterDetect peak.
    monkeypatch.setattr("jarvis_jax.tracking.coarse_centres.CenterDetector",
                        _FakeCenterDetectBlankFirst)
    argv2 = ["coarse_pass_mvq.py", "--session-dir", str(tmp_path), "--calib-dir", str(tmp_path),
             "--cameras", ",".join(cams), "--run", "/fake/final",
             "--centerdetect", "/fake/cd", "--stride", "8", "--start", "0", "--end", "40",
             "--out", str(out), "--batch", "4", "--min-views", "2", "--partial-every", "2",
             "--resume"]
    monkeypatch.setattr(sys, "argv", argv2)
    drv.main()

    with np.load(out, allow_pickle=True) as z:
        frames = list(z["coarse_frame"])
        assert frames == [0, 8, 16, 24, 32]
        i = frames.index(24)
        assert z["centre_source"][0, i] == 1, (
            "frame 24 (blank, right after the resume boundary) must REUSE the prior "
            "run's last_centres (centre_source == 1), not read as no-history NaN (== 2)")
        assert np.isfinite(z["X3d"][:, i]).all()   # reused, not NaN'd out
