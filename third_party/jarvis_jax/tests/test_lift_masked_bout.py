# tests/test_lift_masked_bout.py
"""`jarvis_jax.tracking.lift_mvq.lift_masked_bout` -- the masked-bout mvq
lifter that writes pipeline-format keypoints for a whole courtship bout
(Task 7 of the mvq mask-free front end).

What is actually load-bearing here, and why each check exists:

  * WINDOW PLANNING. Two flies closer than `merge_dist_units` share ONE crop
    (the model's own two-instance case) and further apart get one each -- the
    same rule `coarse_track` uses, so the bout lifter and the coarse pass do
    not disagree about what a window is.
  * TYPED SLOT CHOICE. `fly0` is the FEMALE typed slot and `fly1` the MALE
    one, per frame from the window with the highest existence for THAT slot.
    Reading the first window instead would let a near-empty crop claim the
    fly; below `exist_thresh` the frame must be NaN, not a fallback to an
    untyped slot.
  * KEYPOINT ORDER. The written npz must speak `cfg.model.KP_NAMES`
    (model/XML order), not the model's own detector order. That is the
    CLAUDE.md trap that once measured a middle-left leg as a "collapsed right
    wing vein"; the EyeL-EyeR distance is the permutation-invariant that
    catches a botched permutation.
  * GATES. The `gates` string stamped into kp3d.npz must equal what
    `scripts/run_bout.py::stage_b_gate_signature` computes for the same
    config, or every mvq bout is refused at Stage B.
  * SEX.JSON. `canonicalize_bout` must treat the lifter's sex.json as
    authoritative -- the wing-song CV heuristic re-deciding identity after a
    typed-slot lift would swap fly dirs the model already typed.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
RUN_BOUT = REPO / "scripts" / "run_bout.py"
ANATOMY_V1 = REPO / "configs" / "anatomy" / "v1.yaml"
SLURM_ARRAY = REPO / "scripts" / "slurm_bout_array.py"

CAMS = ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855",
        "Cam2012857", "Cam2012861", "Cam2012862"]


def _model_names():
    return [str(n) for n in OmegaConf.load(ANATOMY_V1).model.KP_NAMES]


def _mvq_names():
    """The same 50 names in the model's own (detector) order -- a rotation of
    KP_NAMES, so a by-index write would be caught and a by-name one would not."""
    n = _model_names()
    return n[17:] + n[:17]


def _fake_checkpoint(tmp_path, name="final"):
    """A directory `checkpoint_sha256` accepts: it hashes an orbax
    checkpoint's METADATA files, never the parameter shards."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 1}')
    (d / "_METADATA").write_text('{"tree": "fake"}')
    return str(d)


class FakeRunner:
    """An `MVQRunner` with the forward replaced by arithmetic.

    `read_typed`, `to_pipeline` and the gate signature are the REAL methods
    (bound off the class) so the slot rule, the by-name permutation and the
    signature under test are the shipped ones; only `windows`/`infer` -- the
    448-crop DINOv3 forward -- are faked, which is what makes this file run on
    a CPU in seconds.

    `infer` returns, for window `b` and slot `s`:
        kp3d[b, s, k] = (100 * s + centre_x[b], k, 0)
    so a frame's fly can be traced back to BOTH the slot it was read from and
    the window it came from, and the EyeL-EyeR distance is a real (non-zero,
    permutation-invariant) quantity.
    """

    from jarvis_jax.tracking.lift_mvq import MVQRunner as _R
    read_typed = _R.read_typed
    to_pipeline = _R.to_pipeline
    gates_signature = _R.gates_signature
    gates_string = _R.gates_string
    del _R

    def __init__(self, checkpoint, *, kp_names, cameras=CAMS, batch=4,
                 exist_thresh=0.5, exist_fn=None, sex_prob=(0.5, 0.9, 0.1, 0.5),
                 identity="mask"):
        self.checkpoint = os.path.abspath(checkpoint)
        self.step = None
        self.step_label = "final"
        # the real MVQRunner carries the identity mode so its gates string
        # names it; a runner built for one mode may not be lifted with another
        self.identity = identity
        self.kp_names = list(kp_names)
        self.K = len(self.kp_names)
        self.I = 4
        self.cameras = list(cameras)
        self.C = len(self.cameras)
        self.batch = int(batch)
        self.exist_thresh = float(exist_thresh)
        self.meta = {"keypoint_names": list(kp_names)}
        self._gates = None
        self._sex_prob = np.asarray(sex_prob, np.float32)
        # default: only the two typed slots exist, comfortably above threshold
        self._exist_fn = exist_fn or (lambda centre: np.array([0.0, 0.9, 0.8, 0.0], np.float32))

    def windows(self, frames, present, centres, prompt_mask=None):
        centres = np.atleast_2d(np.asarray(centres, np.float64))
        b = centres.shape[0]
        return {"crops": np.zeros((b, 1, self.C, 4, 4, 3), np.uint8),
                "cam_valid": np.broadcast_to(np.asarray(present, bool),
                                             (b, self.C))[:, None].copy(),
                "M": np.zeros((b, self.C, 2, 3), np.float32),
                "t_local": np.zeros((b, 1, self.C, 2), np.float32),
                "origin": np.zeros((b, self.C, 2), np.int32),
                "centres": centres.astype(np.float32)}

    def infer(self, w, *, prompt_on=None):
        centres = np.asarray(w["centres"], np.float64)
        b = centres.shape[0]
        kp3d = np.zeros((b, self.I, self.K, 3), np.float32)
        k = np.arange(self.K, dtype=np.float32)
        for s in range(self.I):
            kp3d[:, s, :, 0] = (100.0 * s + centres[:, 0])[:, None]
            kp3d[:, s, :, 1] = k[None]
        exist = np.stack([np.asarray(self._exist_fn(centres[i]), np.float32)
                          for i in range(b)])
        return {"kp3d": kp3d,
                "kp2d": np.zeros((b, self.I, self.C, self.K, 2), np.float32),
                "vis": np.full((b, self.I, self.C, self.K), 0.8, np.float32),
                "exist": exist,
                "sex_prob": np.broadcast_to(self._sex_prob, (b, self.I)).copy(),
                "conf_raw": np.full((b, self.I, self.K), 0.05, np.float32),
                "xyz": np.zeros((b, self.I, self.K, 3), np.float32)}


def _frames(n, C=7):
    """`n` frames of (frames (C,H,W,3), present (C,)) -- the fake runner never
    looks at the pixels, only at the shapes."""
    return [(np.zeros((C, 8, 8, 3), np.uint8), np.ones(C, bool)) for _ in range(n)]


# ------------------------------------------------------------- fly-count guard
def test_lift_masked_bout_refuses_a_fly_count_other_than_two(tmp_path):
    """This lifter hardcodes exactly TWO typed slots (female fly0, male
    fly1) throughout -- a `centres`/`ok` pair with any other fly count must
    raise a clear error, not silently index (A=1) or silently drop a fly
    (A>=3)."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    r = FakeRunner(_fake_checkpoint(tmp_path), kp_names=_mvq_names())
    for a in (1, 3):
        with pytest.raises(ValueError, match="TWO typed slots"):
            lift_masked_bout(r, _frames(1), np.zeros((a, 1, 3), np.float32),
                             np.ones((a, 1), bool), out_dir=str(tmp_path / f"bout{a}"),
                             model_names=_model_names())


def test_bout_centres_3d_refuses_a_mask_store_with_a_different_fly_count(tmp_path):
    """`bout_centres_3d` reads `store.n_flies` directly -- the same guard
    belongs at the earliest point that value is used, not just inside
    `lift_masked_bout`."""
    from jarvis_jax.tracking.lift_mvq import bout_centres_3d

    class _FakeStore:
        n_flies = 3

    with pytest.raises(ValueError, match="TWO typed slots"):
        bout_centres_3d(_FakeStore(), np.zeros((7, 4, 3), np.float32), 1)


# --------------------------------------------------------------------- windows
def test_frame_windows_merge_two_close_centres_and_split_far_ones():
    """Same rule as `coarse_track`/`plan_windows`: <= 30 units (3 mm) is one
    crop holding both flies, further apart is one crop each, and a fly with no
    3D centre this frame contributes no window at all (rather than a window at
    the origin, which would crop the arena floor and come back confident)."""
    from jarvis_jax.tracking.lift_mvq import frame_windows
    close = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], np.float32)
    far = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]], np.float32)
    ok = np.ones(2, bool)

    wc, assign = frame_windows(close, ok)
    assert wc.shape == (1, 3)
    np.testing.assert_allclose(wc[0], [10.0, 0.0, 0.0])      # the midpoint
    assert list(assign) == [0, 0]

    wc, assign = frame_windows(far, ok)
    assert wc.shape == (2, 3) and sorted(assign) == [0, 1]

    wc, assign = frame_windows(far, np.array([True, False]))
    assert wc.shape == (1, 3)
    np.testing.assert_allclose(wc[0], [0.0, 0.0, 0.0])
    assert list(assign) == [0, -1]

    wc, assign = frame_windows(far, np.zeros(2, bool))
    assert wc.shape == (0, 3) and list(assign) == [-1, -1]


# ------------------------------------------------------------------ typed read
def test_typed_read_takes_the_highest_exist_window_per_slot(tmp_path):
    """Two SEPARATE windows in one frame both carry a female candidate. The
    female row must come from the window that is most confident about the
    female slot -- taking the first window instead lets a near-empty crop
    claim the fly, and nothing downstream could see it."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    mvq_names = _mvq_names()

    def exist_fn(centre):
        # window at x=0: weak female, strong male; window at x=200: the reverse
        return (np.array([0.0, 0.6, 0.95, 0.0], np.float32) if centre[0] < 100
                else np.array([0.0, 0.99, 0.55, 0.0], np.float32))

    r = FakeRunner(_fake_checkpoint(tmp_path), kp_names=mvq_names, exist_fn=exist_fn)
    centres = np.array([[[0.0, 0.0, 0.0]], [[200.0, 0.0, 0.0]]], np.float32)  # (A=2,T=1,3)
    res = lift_masked_bout(r, _frames(1), centres, np.ones((2, 1), bool),
                           out_dir=str(tmp_path / "bout"), model_names=_model_names())
    # slot 1 (female) -> fly0, read from the x=200 window (exist 0.99 > 0.6)
    assert res["slot"][0, 0] == 1 and res["exist"][0, 0] == pytest.approx(0.99)
    assert res["kp3d_mvq"][0, 0, 0, 0] == pytest.approx(100.0 + 200.0)
    # slot 2 (male) -> fly1, read from the x=0 window (exist 0.95 > 0.55)
    assert res["slot"][1, 0] == 2 and res["exist"][1, 0] == pytest.approx(0.95)
    assert res["kp3d_mvq"][1, 0, 0, 0] == pytest.approx(200.0 + 0.0)


def test_slot_below_exist_thresh_is_a_nan_frame_with_no_fallback(tmp_path):
    """`exist < exist_thresh` must NaN that fly's frame. The unprompted typed
    route has no policy fallback: a frame reported as "the female" is never
    quietly some other slot."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    r = FakeRunner(_fake_checkpoint(tmp_path), kp_names=_mvq_names(),
                   exist_fn=lambda c: np.array([0.99, 0.2, 0.9, 0.99], np.float32))
    centres = np.zeros((2, 2, 3), np.float32)
    centres[1, :, 0] = 200.0
    res = lift_masked_bout(r, _frames(2), centres, np.ones((2, 2), bool),
                           out_dir=str(tmp_path / "bout"), model_names=_model_names())
    assert (res["slot"][0] == -1).all()                       # female never clears 0.5
    assert np.isnan(res["kp3d_mvq"][0]).all()
    assert (res["slot"][1] == 2).all()                        # male does
    assert np.isfinite(res["kp3d_mvq"][1]).all()


def test_frames_with_no_centre_are_nan_and_recorded(tmp_path):
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    r = FakeRunner(_fake_checkpoint(tmp_path), kp_names=_mvq_names())
    centres = np.zeros((2, 3, 3), np.float32)
    centres[1, :, 0] = 200.0
    ok = np.ones((2, 3), bool)
    ok[:, 1] = False                                          # frame 1: neither fly
    res = lift_masked_bout(r, _frames(3), centres, ok,
                           out_dir=str(tmp_path / "bout"), model_names=_model_names())
    assert list(res["no_centre"]) == [False, True, False]
    assert np.isnan(res["kp3d_mvq"][:, 1]).all()
    assert np.isfinite(res["kp3d_mvq"][:, 0]).all() and np.isfinite(res["kp3d_mvq"][:, 2]).all()
    meta = json.load(open(tmp_path / "bout" / "mvq_meta.json"))
    assert meta["n_no_centre"] == 1


# ---------------------------------------------------------------------- writer
def test_writer_emits_model_order_canonical_identity_and_gates(tmp_path):
    """The npz the pipeline reads: MODEL keypoint order by NAME, fly0 female /
    fly1 male, `conf3d` = the mean per-view visibility, and a `gates` string
    naming the checkpoint."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout, mvq_gate_string
    model_names, mvq_names = _model_names(), _mvq_names()
    ckpt = _fake_checkpoint(tmp_path)
    # explicitly the SEX-head route: fly0 == typed slot 1, fly1 == typed slot 2
    # is what this test asserts, and with no mask sex_meta `identity="mask"`
    # would resolve to exactly the same rule but be labelled differently
    r = FakeRunner(ckpt, kp_names=mvq_names, identity="sex")
    T = 3
    centres = np.zeros((2, T, 3), np.float32)
    centres[1, :, 0] = 200.0
    out = tmp_path / "bout"
    lift_masked_bout(r, _frames(T), centres, np.ones((2, T), bool),
                     out_dir=str(out), model_names=model_names)

    for fly, slot in ((0, 1), (1, 2)):
        with np.load(out / f"fly{fly}" / "kp3d.npz") as z:
            assert list(z["kp_names"]) == model_names          # MODEL order, by name
            assert z["kp3d"].shape == (T, len(model_names), 3)
            assert z["conf3d"].shape == (T, len(model_names))
            np.testing.assert_allclose(z["conf3d"], 0.8)       # mean per-view visibility
            np.testing.assert_allclose(z["conf3d_mvq_raw"], 0.05)
            assert str(z["gates"]) == mvq_gate_string(ckpt, step=None, exist_thresh=0.5,
                                                      identity="sex")
            # fly0 IS the female typed slot (slot 1) and fly1 the male (slot 2).
            # Both flies are read out of the SAME window here (the two centres
            # are 200 units apart, so there are two windows, and the fake gives
            # both the same existence -- the tie keeps the first), so the only
            # thing separating the two files is the slot: x = 100 * slot.
            kp3d = z["kp3d"]
            assert kp3d[0, 0, 0] == pytest.approx(100.0 * slot)
            # EyeL-EyeR: a rigid, permutation-INVARIANT quantity. The fake puts
            # keypoint k at y = k in mvq order, so the distance is |i - j| there
            # and must survive the by-name permutation unchanged.
            iL, iR = mvq_names.index("EyeL"), mvq_names.index("EyeR")
            jL, jR = model_names.index("EyeL"), model_names.index("EyeR")
            d = np.linalg.norm(kp3d[:, jL] - kp3d[:, jR], axis=-1)
            np.testing.assert_allclose(d, abs(iL - iR), atol=1e-5)
        with np.load(out / f"fly{fly}" / "kp2d.npz") as z:
            assert list(z["kp_names"]) == model_names
            assert list(z["cameras"]) == CAMS
            assert z["kp2d"].shape == (T, len(CAMS), len(model_names), 2)
            assert z["conf"].shape == (T, len(CAMS), len(model_names))


def test_sex_json_matches_canonicalize_bout_schema(tmp_path):
    """`sex.json` is read by `estimate_recording_scale._determine_identity`
    (which is what lets scale.json carry a per-fly body size) and by
    `canonicalize_bout`; every key the canonicalizer writes must be present,
    with a real int `male_fly`."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    from jarvis_jax.tracking.sexing import MVQ_SEX_METHOD, canonicalize_bout
    r = FakeRunner(_fake_checkpoint(tmp_path), kp_names=_mvq_names())
    centres = np.zeros((2, 2, 3), np.float32)
    centres[1, :, 0] = 200.0
    out = tmp_path / "bout"
    lift_masked_bout(r, _frames(2), centres, np.ones((2, 2), bool),
                     out_dir=str(out), model_names=_model_names())
    sx = json.load(open(out / "sex.json"))
    assert sx["male_fly"] == 1 and isinstance(sx["male_fly"], int)
    assert sx["method"] == MVQ_SEX_METHOD and sx["applied_swap"] is False
    # per-fly means from the sex head, as the brief asks
    assert sx["sex_prob"]["fly0"] == pytest.approx(0.9)
    assert sx["sex_prob"]["fly1"] == pytest.approx(0.1)
    assert sx["exist"]["fly0"] == pytest.approx(0.9)
    assert sx["exist"]["fly1"] == pytest.approx(0.8)

    # the canonicalizer's own schema, discovered from a dry run rather than
    # hard-coded, so this test fails if that schema ever grows a key
    ref = canonicalize_bout(str(out), _model_names(), dry_run=True, verbose=False)
    missing = sorted(set(ref) - set(sx))
    assert not missing, f"sex.json is missing canonicalize_bout keys {missing}"


def test_gates_string_is_what_run_bout_stage_b_expects(tmp_path):
    """A kp3d.npz this lifter writes must be ACCEPTED by run_bout.py's Stage-B
    staleness check with `pipeline.lifter: mvq`, without `allow_stale_kp3d`."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    ckpt = _fake_checkpoint(tmp_path)
    r = FakeRunner(ckpt, kp_names=_mvq_names(), exist_thresh=0.55, identity="sex")
    centres = np.zeros((2, 1, 3), np.float32)
    centres[1, :, 0] = 200.0
    out = tmp_path / "bout"
    lift_masked_bout(r, _frames(1), centres, np.ones((2, 1), bool),
                     out_dir=str(out), model_names=_model_names())

    fn = next(n for n in ast.parse(RUN_BOUT.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "stage_b_gate_signature")
    g = {"json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<rb>", "exec"), g)
    cfg = OmegaConf.create({"detector": {"conf_thresh": 0.3},
                            "pipeline": {"lifter": "mvq"},
                            "mvq": {"checkpoint": ckpt, "step": None,
                                    "exist_thresh": 0.55, "identity": "sex"}})
    with np.load(out / "fly0" / "kp3d.npz") as z:
        assert str(z["gates"]) == g["stage_b_gate_signature"](cfg)


def test_lift_is_idempotent_and_force_reruns(tmp_path):
    from jarvis_jax.tracking.lift_mvq import bout_lift_is_current, lift_masked_bout
    r = FakeRunner(_fake_checkpoint(tmp_path), kp_names=_mvq_names(), identity="sex")
    centres = np.zeros((2, 1, 3), np.float32)
    centres[1, :, 0] = 200.0
    out = tmp_path / "bout"
    assert not bout_lift_is_current(str(out), r.gates_string())
    lift_masked_bout(r, _frames(1), centres, np.ones((2, 1), bool),
                     out_dir=str(out), model_names=_model_names())
    assert bout_lift_is_current(str(out), r.gates_string())
    # a different checkpoint (or exist_thresh) is NOT current -- that is the
    # whole point of stamping the gates string
    other = FakeRunner(_fake_checkpoint(tmp_path, "final2"), kp_names=_mvq_names(),
                       identity="sex")
    assert not bout_lift_is_current(str(out), other.gates_string())

    res = lift_masked_bout(r, _frames(1), centres, np.ones((2, 1), bool),
                           out_dir=str(out), model_names=_model_names())
    assert res["skipped"] is True
    res = lift_masked_bout(r, _frames(1), centres, np.ones((2, 1), bool),
                           out_dir=str(out), model_names=_model_names(), force=True)
    assert res["skipped"] is False


def test_lift_is_not_current_without_sex_json(tmp_path):
    """`lift_masked_bout` writes every fly's kp3d.npz BEFORE sex.json, so a
    requeued task killed in that window leaves both npz files complete and
    gated but no sex.json. That bout must NOT read as current -- otherwise a
    re-submitted array skips it forever and the recording falls back to one
    shared body scale instead of a per-fly one."""
    from jarvis_jax.tracking.lift_mvq import bout_lift_is_current, lift_masked_bout
    r = FakeRunner(_fake_checkpoint(tmp_path), kp_names=_mvq_names(), identity="sex")
    centres = np.zeros((2, 1, 3), np.float32)
    centres[1, :, 0] = 200.0
    out = tmp_path / "bout"
    lift_masked_bout(r, _frames(1), centres, np.ones((2, 1), bool),
                     out_dir=str(out), model_names=_model_names())
    assert bout_lift_is_current(str(out), r.gates_string())

    # simulate the kill: both npz are complete and gated, sex.json is gone
    os.remove(out / "sex.json")
    assert not bout_lift_is_current(str(out), r.gates_string())

    # a sex.json that exists but was NOT written by the mvq sex head (e.g. a
    # stale file from a different lifter/heuristic) must not count either
    (out / "sex.json").write_text(json.dumps({"male_fly": 1, "method": "wing_song_cv"}))
    assert not bout_lift_is_current(str(out), r.gates_string())


# --------------------------------------------------------------- canonicalize
def test_canonicalize_bout_treats_mvq_sex_head_as_authoritative(tmp_path):
    """A bout whose sex.json says `mvq_sex_head` was typed by the model's own
    sex head; re-running the wing-song CV there could only re-decide identity
    from a WORSE signal, and a swap would move fly dirs the typed slots
    already named. No swap, no heuristic, file untouched."""
    from jarvis_jax.tracking.sexing import MVQ_SEX_METHOD, canonicalize_bout
    bout = tmp_path / "bout_00028"
    (bout / "fly0").mkdir(parents=True)
    (bout / "fly1").mkdir(parents=True)
    names = _model_names()
    # a kp3d that WOULD make the CV heuristic prefer fly0 as the male
    rng = np.random.default_rng(0)
    for fly, jitter in ((0, 1.0), (1, 0.0)):
        T, K = 60, len(names)
        kp = np.zeros((T, K, 3), np.float32)
        idx = {n: i for i, n in enumerate(names)}
        kp[:, idx["Scutellum"]] = [0, 0, 0]
        kp[:, idx["Abd_tip"]] = [0, -1, 0]
        for s in "LR":
            kp[:, idx[f"Wing{s}_base"]] = [0, 0, 0]
            ang = rng.normal(size=T) * jitter
            kp[:, idx[f"Wing{s}_V13"]] = np.stack(
                [np.sin(ang), np.cos(ang), np.zeros(T)], axis=1)
        np.savez(bout / f"fly{fly}" / "kp3d.npz", kp3d=kp,
                 conf3d=np.ones((T, K), np.float32))
    payload = {"male_fly": 1, "original_male_fly": 1, "applied_swap": False,
               "confidence": "high", "method": MVQ_SEX_METHOD,
               "authority": MVQ_SEX_METHOD, "note": "typed slots"}
    (bout / "sex.json").write_text(json.dumps(payload))
    before = (bout / "sex.json").read_text()

    res = canonicalize_bout(str(bout), names, verbose=False)
    assert res["applied_swap"] is False
    assert res["method"] == MVQ_SEX_METHOD and res["authority"] == MVQ_SEX_METHOD
    assert res["male_fly"] == 1
    assert (bout / "sex.json").read_text() == before          # untouched
    assert (bout / "fly0").is_dir() and (bout / "fly1").is_dir()

    # ... and a human review still outranks it (the mvq rule must not shadow
    # the one authority that was already above the heuristic)
    res = canonicalize_bout(str(bout), names, verbose=False,
                            mask_sex_meta={"method": "human_id_review", "male_slot": 1})
    assert res["method"] == "human_id_review_masks"
# -------------------------------------------------------------- slurm plumbing
def _slurm_mod():
    spec = importlib.util.spec_from_file_location("slurm_bout_array_wt", SLURM_ARRAY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_mvq_lift_array_script_env_and_command():
    m = _slurm_mod()
    s = m.build_mvq_lift_array_script(
        job_name="mvq", partition="ckpt-all", account="portia", cpus=8, mem=48,
        gpus=1, time_limit="12:00:00", requeue=True, constraint="h200|a40",
        conda_env="3d_tracking", idxs=[1, 4, 28], session_dir="/s",
        predictions_dir="/p", run_dir="/r", checkpoint="/ckpt/final", step=None,
        exist_thresh=0.5, batch=8, anatomy_cfg="configs/anatomy/v1.yaml",
        recording_cfg="configs/recording/session0.yaml")
    assert "--array=1,4,28" in s and "--requeue" in s
    assert "scripts/mvq_lift_bout.py" in s
    assert "--bout ${SLURM_ARRAY_TASK_ID}" in s
    assert "--run /ckpt/final" in s and "--step" not in s      # final/ carries no step
    # the same JAX env lines as the IK array -- a missing `module load cuda`
    # silently runs the lifter on CPU
    for line in ("module load cuda/12.9.1", "unset LD_LIBRARY_PATH",
                 "unset JAX_PLATFORMS", "XLA_PYTHON_CLIENT_MEM_FRACTION=0.9",
                 "LD_PRELOAD"):
        assert line in s
    # the editable install points at ANOTHER checkout; without this the job
    # imports a different jarvis_jax than the one being tested
    assert "PYTHONPATH=third_party/jarvis_jax:." in s
    s2 = m.build_mvq_lift_array_script(
        job_name="mvq", partition="ckpt-all", account="portia", cpus=8, mem=48,
        gpus=1, time_limit="12:00:00", requeue=True, conda_env="3d_tracking",
        idxs=[1], session_dir="/s", predictions_dir="/p", run_dir="/r",
        checkpoint="/ckpt", step=9000, exist_thresh=0.5, batch=8,
        anatomy_cfg="configs/anatomy/v1.yaml",
        recording_cfg="configs/recording/session0.yaml")
    assert "--step 9000" in s2


def test_dry_run_with_lifter_mvq_puts_the_lift_before_precompute(tmp_path, capsys, monkeypatch):
    """`--lifter mvq` must (a) skip SAM3 entirely -- masks are an INPUT here --
    (b) run the lift array before precompute, so the body-scale precompute
    pools over mvq keypoints for the whole recording rather than one bout, and
    (c) pass `pipeline.lifter=mvq` to every run_bout command, without which
    Stage B would recompute DLT keypoints over the top of the lifted ones."""
    m = _slurm_mod()
    pred = tmp_path / "sam3_masks"
    for i in (3, 7):
        (pred / f"bout_{i:05d}").mkdir(parents=True)
        (pred / f"bout_{i:05d}" / "sam3_masks.npz").write_bytes(b"")
    monkeypatch.setattr(
        "sys.argv",
        ["slurm_bout_array.py", "--lifter", "mvq", "--dry-run", "--slurm", "ckpt_all",
         f"recording.predictions_dir={pred}", f"outputs.out={tmp_path / 'pose_mvq_p3a'}"])
    m.main()
    out = capsys.readouterr().out
    assert out.index("--- mvq_lift script") < out.index("--- precompute script")
    assert "--- sam3 script" not in out
    assert "scripts/mvq_lift_bout.py" in out
    assert "--array=3,7" in out
    assert out.count("pipeline.lifter=mvq") >= 2               # precompute + jax array
    assert "mvq=p3a" in out


def test_mvq_lift_skip_requires_the_lift_to_have_actually_happened(tmp_path, capsys,
                                                                   monkeypatch):
    """`--mvq-lift skip` (the campaign's `--local-gpus` path, which lifts on an
    interactive node to dodge a day-deep queue) must VERIFY with
    `bout_lift_is_current` -- the SAME predicate the lifter itself uses --
    rather than trust a bare `kp3d.npz` existence check: a bout whose lift
    was interrupted between writing kp3d.npz and sex.json would pass the
    weaker check and then fall back to a shared body scale in precompute
    (MEMORY scale-from-first-bout-defect), inside a run labelled mvq, which
    no later artifact distinguishes."""
    m = _slurm_mod()
    ckpt = _fake_checkpoint(tmp_path)
    pred, out = tmp_path / "sam3_masks", tmp_path / "pose_mvq_p3a"
    for i in (3, 7):
        (pred / f"bout_{i:05d}").mkdir(parents=True)
        (pred / f"bout_{i:05d}" / "sam3_masks.npz").write_bytes(b"")
    argv = ["slurm_bout_array.py", "--lifter", "mvq", "--mvq-lift", "skip", "--dry-run",
            "--slurm", "ckpt_all", f"recording.predictions_dir={pred}",
            f"outputs.out={out}", f"mvq.checkpoint={ckpt}"]
    monkeypatch.setattr("sys.argv", argv)
    m.main()
    err = capsys.readouterr().err
    assert "2 bout(s) are not a current mvq lift" in err

    # A real, GATE-MATCHING lift for both bouts (kp3d.npz per fly + sex.json).
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    r = FakeRunner(ckpt, kp_names=_mvq_names())
    centres = np.zeros((2, 1, 3), np.float32)
    centres[1, :, 0] = 200.0
    for i in (3, 7):
        lift_masked_bout(r, _frames(1), centres, np.ones((2, 1), bool),
                         out_dir=str(out / "bouts" / f"bout_{i:05d}"),
                         model_names=_model_names())

    # Bout 3's lift is "interrupted": both kp3d.npz exist but sex.json never
    # got written -- an npz-only check would call this bout done.
    os.remove(out / "bouts" / "bout_00003" / "sex.json")
    monkeypatch.setattr("sys.argv", argv)
    m.main()
    err = capsys.readouterr().err
    assert "1 bout(s) are not a current mvq lift" in err
    assert "[3]" in err

    # Restoring sex.json (a completed lift) makes bout 3 current too, and the
    # whole array reads as already-lifted.
    lift_masked_bout(r, _frames(1), centres, np.ones((2, 1), bool),
                     out_dir=str(out / "bouts" / "bout_00003"),
                     model_names=_model_names(), force=True)
    monkeypatch.setattr("sys.argv", argv)
    m.main()
    cap = capsys.readouterr()
    assert "mvq lift: skipped (2 bouts already lifted" in cap.out
    assert "--- mvq_lift script" not in cap.out
    assert cap.out.count("pipeline.lifter=mvq") >= 2


# ------------------------------------------------- run_bout viz-import recovery
def test_import_keypoint_groups_recovers_when_pythonpath_holds_the_repo_root():
    """Session0 bout 28 died at the viz stage AFTER a 36-minute STAC solve
    (job 39606799): `scripts/` is sys.path[0] under a direct invocation, so
    `import viz` binds the unrelated `scripts/viz` package, and the recovery
    path's `if _repo not in sys.path` skipped its insert because the mvq
    sbatch text ALREADY put the repo root on sys.path via
    `PYTHONPATH=third_party/jarvis_jax:.` -- just behind `scripts/`. The retry
    then resolved `viz` to `scripts/viz` a second time and raised again.

    Run in a subprocess reproducing exactly that layout (repo root present but
    behind scripts/), because the fix is about interpreter state that a
    same-process test cannot honestly recreate.
    """
    import subprocess
    src = (
        "import os, sys\n"
        f"repo = {str(REPO)!r}\n"
        # the failing layout: scripts/ first, repo root present but behind it
        "sys.path.insert(0, os.path.join(repo, 'scripts'))\n"
        "assert repo in sys.path, sys.path\n"
        "import ast\n"
        "src = open(os.path.join(repo, 'scripts', 'run_bout.py')).read()\n"
        "fn = next(n for n in ast.parse(src).body\n"
        "          if isinstance(n, ast.FunctionDef) and n.name == 'import_keypoint_groups')\n"
        "g = {'os': os, 'sys': sys, '__file__': os.path.join(repo, 'scripts', 'run_bout.py')}\n"
        "exec(compile(ast.Module(body=[fn], type_ignores=[]), '<rb>', 'exec'), g)\n"
        # the poisoned binding the real failure had by this point
        "import viz\n"
        "assert not hasattr(viz, 'core')\n"
        "kg = g['import_keypoint_groups']()\n"
        "out = kg(['Scutellum', 'EyeL', 'EyeR', 'Abd_tip', 'T1L_FeTi'])\n"
        "assert set(out) >= {'head', 'thorax', 'legs', 'abdomen'}, out\n"
        "print('OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "third_party" / "jarvis_jax"), str(REPO)])
    r = subprocess.run([__import__("sys").executable, "-c", src], env=env,
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "OK" in r.stdout


# ------------------------------------------------------------- collapse guard
def test_collapsed_frame_keeps_the_more_confident_slot_and_flags_it(tmp_path):
    """The two typed slots are chosen INDEPENDENTLY, so on a merged window both
    can read the same physical fly -- most likely on the mounting frames, and
    with nothing to show for it: a duplicated instance is smoother and more
    confident than a real one, so jitter, confidence and residual metrics all
    rate it as excellent. A frame whose two slots agree to within
    `collapse_dist_units` must keep only the more confident slot, NaN the
    other, and say so."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    from jarvis_jax.train.matching import SLOT_FEMALE, SLOT_MALE

    class Collapsing(FakeRunner):
        """Both typed slots return the SAME kp3d (the fake's per-slot offset is
        dropped), so every frame is a collapse."""

        def infer(self, w, *, prompt_on=None):
            out = FakeRunner.infer(self, w, prompt_on=prompt_on)
            out["kp3d"][:, SLOT_MALE] = out["kp3d"][:, SLOT_FEMALE]
            return out

    r = Collapsing(_fake_checkpoint(tmp_path), kp_names=_mvq_names(),
                   exist_fn=lambda c: np.array([0.0, 0.7, 0.95, 0.0], np.float32))
    centres = np.zeros((2, 2, 3), np.float32)
    centres[1, :, 0] = 10.0                       # 10 units apart -> ONE merged window
    out = tmp_path / "bout"
    res = lift_masked_bout(r, _frames(2), centres, np.ones((2, 2), bool),
                           out_dir=str(out), model_names=_model_names())
    assert list(res["collapsed"]) == [True, True]
    # the male slot is the more confident one (0.95 > 0.7), so the FEMALE row
    # is the one dropped
    assert res["n_collapsed"] == {"fly0": 2, "fly1": 0}
    assert np.isnan(res["kp3d_mvq"][0]).all() and (res["slot"][0] == -1).all()
    assert np.isfinite(res["kp3d_mvq"][1]).all() and (res["slot"][1] == SLOT_MALE).all()
    meta = json.load(open(out / "mvq_meta.json"))
    assert meta["collapsed_frac"] == pytest.approx(1.0)
    assert meta["n_collapsed"] == {"fly0": 2, "fly1": 0}
    assert meta["per_frame"]["collapsed"] == [1, 1]
    assert meta["collapse_dist_units"] == pytest.approx(3.0)
    assert meta["n_missing"]["fly0"] == 2
    with np.load(out / "fly0" / "kp3d.npz") as z:
        assert np.isnan(z["kp3d"]).all()
    assert SLOT_FEMALE == 1


def test_two_real_flies_are_not_flagged_as_collapsed(tmp_path):
    """15 units (1.5 mm) apart is two animals, not one read twice -- the guard
    must leave both rows alone. Without this it would silently halve every
    close-interaction bout."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    r = FakeRunner(_fake_checkpoint(tmp_path), kp_names=_mvq_names(),
                   exist_fn=lambda c: np.array([0.0, 0.7, 0.95, 0.0], np.float32))
    # the fake separates the two slots by 100 units in x; put the window
    # centres 15 units apart so they still MERGE into one window (< 30) --
    # merging is not what the guard keys on, AGREEMENT is.
    centres = np.zeros((2, 2, 3), np.float32)
    centres[1, :, 0] = 15.0
    out = tmp_path / "bout"
    res = lift_masked_bout(r, _frames(2), centres, np.ones((2, 2), bool),
                           out_dir=str(out), model_names=_model_names())
    assert list(res["n_windows"]) == [1, 1]        # merged, but NOT collapsed
    assert not res["collapsed"].any()
    assert res["n_collapsed"] == {"fly0": 0, "fly1": 0}
    assert np.isfinite(res["kp3d_mvq"]).all()
    assert json.load(open(out / "mvq_meta.json"))["collapsed_frac"] == 0.0


def test_canonicalize_bout_moves_an_mvq_bout_to_a_different_male_slot(tmp_path):
    """`male_slot=0` is not a convention this repo uses, but if a caller asks
    for it the mvq DECISION (which fly is the male) must be honoured by MOVING
    the dirs -- not discarded in favour of the wing-song heuristic, and not
    silently ignored, leaving sex.json claiming a slot the male is not in."""
    from jarvis_jax.tracking.sexing import MVQ_SEX_METHOD, canonicalize_bout
    bout = tmp_path / "bout_00001"
    (bout / "fly0").mkdir(parents=True)
    (bout / "fly1").mkdir(parents=True)
    (bout / "fly0" / "who.txt").write_text("female")
    (bout / "fly1" / "who.txt").write_text("male")
    (bout / "sex.json").write_text(json.dumps(
        {"male_fly": 1, "original_male_fly": 1, "applied_swap": False,
         "confidence": "high", "method": MVQ_SEX_METHOD, "authority": MVQ_SEX_METHOD,
         "sex_prob": {"fly0": 0.96, "fly1": 0.002}, "note": "typed slots"}))

    res = canonicalize_bout(str(bout), _model_names(), male_slot=0, verbose=False)
    assert res["applied_swap"] is True
    assert res["male_fly"] == 0 and res["original_male_fly"] == 1
    assert res["method"] == MVQ_SEX_METHOD          # still the mvq decision
    assert (bout / "fly0" / "who.txt").read_text() == "male"
    on_disk = json.load(open(bout / "sex.json"))
    assert on_disk["male_fly"] == 0 and on_disk["applied_swap"] is True
    assert on_disk["sex_prob"] == {"fly0": 0.96, "fly1": 0.002}   # evidence kept

    # idempotent: a second call sees the male already at slot 0 and does nothing
    again = canonicalize_bout(str(bout), _model_names(), male_slot=0, verbose=False)
    assert again["applied_swap"] is False
    assert (bout / "fly0" / "who.txt").read_text() == "male"


# ============================================================ mask identity
# The sex head is not a reliable identity source on every recording. On
# 2025_10_20_13_20_04 it calls the female a MALE: in her own mask window the
# male slot fires 0.87-1.00 on her body while the female slot reads 0.01-0.46,
# so `identity="sex"` NaN'd fly0 on 98% of bout 25 and let the male track jump
# onto her body on 21% of frames (diagnosis:
# .superpowers/sdd/2026-09-04-mvq-maskfree-p4a-p4b/female-miss-diagnosis.md).
# `identity="mask"` takes identity from the HUMAN id review the SAM3 masks
# carry -- authoritative in this pipeline's own precedence (human > mvq) --
# and asks the model only "which instance is ON this mask?".


class PlacedFake(FakeRunner):
    """A `FakeRunner` whose every instance sits at a CHOSEN world point.

    The base fake spreads keypoint `k` to `y = k`, which puts any instance's
    centroid 24.5 units off its window centre -- further than the 10-unit
    mask-assignment radius, so no mask-identity geometry could be expressed
    with it. Here keypoint `k` of slot `s` in the window centred at `cx` sits
    at `place(cx, s) + (0, 0.02 * (k - (K-1)/2), 0)`: the instance's centroid
    is EXACTLY `place(cx, s)`, and EyeL-EyeR is still a non-zero rigid
    distance (0.02 * |iL - iR|) so the by-name permutation check has something
    to measure.
    """

    def __init__(self, *a, place=None, **kw):
        super().__init__(*a, **kw)
        # default: every slot sits on its own window's centre
        self._place = place or (lambda cx, s: (float(cx), 0.0, 0.0))

    def infer(self, w, *, prompt_on=None):
        out = FakeRunner.infer(self, w, prompt_on=prompt_on)
        centres = np.asarray(w["centres"], np.float64)
        k = np.arange(self.K, dtype=np.float64) - (self.K - 1) / 2.0
        for b in range(centres.shape[0]):
            for s in range(self.I):
                out["kp3d"][b, s] = np.asarray(self._place(centres[b, 0], s), np.float32)
                out["kp3d"][b, s, :, 1] += (0.02 * k).astype(np.float32)
        return out


HUMAN_MASKS = {"method": "human_id_review", "male_slot": 1,
               "review_file": "id_review_reviewed_20260829.json"}


def _two_mask_windows():
    """(A=2,T=1,3) centres 200 units apart: the female mask at x=0, the male
    mask at x=200 -- two separate crops, one per mask fly."""
    c = np.zeros((2, 1, 3), np.float32)
    c[1, :, 0] = 200.0
    return c, np.ones((2, 1), bool)


def _female_window_reads_male():
    """The 20_04 failure, in an exist table: in the FEMALE mask's window the
    male slot fires hard (0.95) on her body and the female slot is dead (0.2);
    in the male mask's window the male slot fires a little less (0.90)."""
    return lambda c: (np.array([0.0, 0.2, 0.95, 0.0], np.float32) if c[0] < 100
                      else np.array([0.0, 0.0, 0.90, 0.0], np.float32))


def test_mask_identity_writes_the_instance_on_the_female_mask(tmp_path):
    """Case (1). The female's own mask window holds exactly one instance above
    threshold and it is the MALE slot (the sex head mistypes her). Under
    `identity="mask"` fly0 must still be written -- from slot 2, out of HER
    window -- with the disagreement recorded, rather than NaN'd because the
    female-typed slot went quiet."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    centres, ok = _two_mask_windows()
    r = PlacedFake(_fake_checkpoint(tmp_path), kp_names=_mvq_names(),
                   exist_fn=_female_window_reads_male())
    out = tmp_path / "bout"
    res = lift_masked_bout(r, _frames(1), centres, ok, out_dir=str(out),
                           model_names=_model_names(), identity="mask",
                           mask_sex_meta=HUMAN_MASKS)
    assert res["identity"] == "mask" and res["identity_source"][0] == "mask"
    # fly0 comes from slot 2 (the male-typed slot) read in the FEMALE window
    assert res["slot"][0, 0] == 2
    assert res["window"][0, 0] == 0
    np.testing.assert_allclose(res["kp3d_mvq"][0, 0, :, 0], 0.0, atol=1e-4)
    assert res["sex_head_agrees"][0, 0] == 0                  # False, and recorded
    # fly1 keeps his own window's male slot -- no swap onto her body
    assert res["slot"][1, 0] == 2 and res["sex_head_agrees"][1, 0] == 1
    np.testing.assert_allclose(res["kp3d_mvq"][1, 0, :, 0], 200.0, atol=1e-4)

    meta = json.load(open(out / "mvq_meta.json"))
    assert meta["identity"] == "mask" and meta["identity_resolved"] == "mask"
    assert meta["per_frame"]["identity_source"] == ["mask"]
    assert meta["per_frame"]["slot_used"] == [[2], [2]]
    assert meta["per_frame"]["sex_head_agrees"] == [[0], [1]]
    assert meta["sex_head_disagree_frac"] == {"fly0": 1.0, "fly1": 0.0}
    assert meta["n_missing"] == {"fly0": 0, "fly1": 0}
    sx = json.load(open(out / "sex.json"))
    assert sx["male_fly"] == 1 and sx["method"] == "mask_human_id_review"


def test_mask_identity_matches_sex_identity_when_both_typed_slots_are_right(tmp_path):
    """Case (2). When the sex head is right -- each mask window's own typed
    slot is the one that fires -- the two modes must produce the SAME bout.
    A fix that changed the good recordings too would be a different change."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    centres, ok = _two_mask_windows()
    exist_fn = lambda c: (np.array([0.0, 0.9, 0.1, 0.0], np.float32) if c[0] < 100
                          else np.array([0.0, 0.1, 0.9, 0.0], np.float32))
    res = {}
    for mode in ("sex", "mask"):
        r = PlacedFake(_fake_checkpoint(tmp_path), kp_names=_mvq_names(),
                       exist_fn=exist_fn, identity=mode)
        res[mode] = lift_masked_bout(r, _frames(1), centres, ok,
                                     out_dir=str(tmp_path / mode),
                                     model_names=_model_names(), identity=mode,
                                     mask_sex_meta=HUMAN_MASKS)
    for key in ("kp3d_mvq", "kp2d_mvq", "vis", "conf_raw", "exist", "slot", "window"):
        np.testing.assert_array_equal(np.nan_to_num(res["sex"][key], nan=-999),
                                      np.nan_to_num(res["mask"][key], nan=-999),
                                      err_msg=f"{key} differs between identity modes")
    assert res["mask"]["slot"][0, 0] == 1 and res["mask"]["slot"][1, 0] == 2
    for fly in (0, 1):
        a = np.load(tmp_path / "sex" / f"fly{fly}" / "kp3d.npz")["kp3d"]
        b = np.load(tmp_path / "mask" / f"fly{fly}" / "kp3d.npz")["kp3d"]
        np.testing.assert_array_equal(np.nan_to_num(a, nan=-999), np.nan_to_num(b, nan=-999))


def test_mask_identity_keeps_the_male_off_the_female_body(tmp_path):
    """Case (3). Same fixture as case (1), read from the male's side: the male
    slot is MORE confident in the female's window (0.95) than in his own
    (0.90). `identity="sex"` takes the highest-existence male slot anywhere and
    therefore writes fly1 on the FEMALE's body -- the 2.5%-of-20_04 swap the
    diagnosis measured. `identity="mask"` must not.

    The old behaviour is pinned here on purpose, so the difference between the
    two modes is explicit rather than assumed.
    """
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    centres, ok = _two_mask_windows()
    exist_fn = _female_window_reads_male()

    r = PlacedFake(_fake_checkpoint(tmp_path), kp_names=_mvq_names(), exist_fn=exist_fn,
                   identity="sex")
    old = lift_masked_bout(r, _frames(1), centres, ok, out_dir=str(tmp_path / "sex"),
                           model_names=_model_names(), identity="sex",
                           mask_sex_meta=HUMAN_MASKS)
    # OLD: fly1 (the male) is read out of the FEMALE mask's window -> x = 0,
    # which is her body; fly0 is NaN because her typed slot never cleared 0.5.
    assert old["window"][1, 0] == 0
    np.testing.assert_allclose(old["kp3d_mvq"][1, 0, :, 0], 0.0, atol=1e-4)
    assert old["slot"][0, 0] == -1 and np.isnan(old["kp3d_mvq"][0]).all()

    r = PlacedFake(_fake_checkpoint(tmp_path), kp_names=_mvq_names(), exist_fn=exist_fn)
    new = lift_masked_bout(r, _frames(1), centres, ok, out_dir=str(tmp_path / "mask"),
                           model_names=_model_names(), identity="mask",
                           mask_sex_meta=HUMAN_MASKS)
    # NEW: the male comes from the MALE mask's window (x = 200) and the female
    # is written at all.
    assert new["window"][1, 0] == 1
    np.testing.assert_allclose(new["kp3d_mvq"][1, 0, :, 0], 200.0, atol=1e-4)
    np.testing.assert_allclose(new["kp3d_mvq"][0, 0, :, 0], 0.0, atol=1e-4)


def test_mask_identity_assigns_by_RELATIVE_distance_not_an_absolute_radius(tmp_path):
    """Case (4a), the round-2 fix. The female's SAM mask on 20_04 is poor: over
    the 30-bout re-lift her mask's DLT centre sits 13-35 units from her body
    (3-4 valid cameras) while the male's is 5-6 units off on all 7. An
    ABSOLUTE radius of 10 units therefore threw away instances that were
    correctly detected -- fly0 went to 1.00 NaN in bout 8, 0.99 in bout 19,
    0.95 in bout 26 -- even though they were nowhere near the other mask.

    Here the one live instance sits 30 units from the FEMALE mask and 50 from
    the MALE's. It is unambiguously hers (nearer by 20 > the 8-unit margin) and
    must be written as fly0, even though 30 > any sane absolute radius.
    """
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    # masks 80 units apart -> two windows; the instance sits at x = 30
    centres = np.zeros((2, 1, 3), np.float32)
    centres[1, :, 0] = 80.0
    ok = np.ones((2, 1), bool)
    r = PlacedFake(_fake_checkpoint(tmp_path), kp_names=_mvq_names(),
                   # only the MALE-typed slot fires, and only in her window --
                   # the 20_04 failure mode
                   exist_fn=lambda c: (np.array([0.0, 0.0, 0.95, 0.0], np.float32)
                                       if c[0] < 40 else np.zeros(4, np.float32)),
                   place=lambda cx, s: (30.0, 0.0, 0.0))
    out = tmp_path / "bout"
    res = lift_masked_bout(r, _frames(1), centres, ok, out_dir=str(out),
                           model_names=_model_names(), identity="mask",
                           mask_sex_meta=HUMAN_MASKS)
    assert res["slot"][0, 0] == 2                       # her body, male-typed slot
    assert res["assign_reason"][0][0] == "nearest"
    np.testing.assert_allclose(res["kp3d_mvq"][0, 0, :, 0], 30.0, atol=1e-4)
    assert res["mask_dist"][0, 0] == pytest.approx(30.0, abs=1e-3)   # > any 10u radius
    # and the MALE gets nothing: that instance is HERS by geometry, so the
    # typed fallback must refuse it rather than put his track on her body
    assert res["slot"][1, 0] == -1 and res["assign_reason"][1][0] == "none"
    assert np.isnan(res["kp3d_mvq"][1]).all()

    meta = json.load(open(out / "mvq_meta.json"))
    assert meta["mask_assign_margin_units"] == 8.0
    assert meta["mask_assign_max_units"] == 60.0
    assert meta["assign_reason_counts"]["fly0"]["nearest"] == 1
    assert meta["assign_reason_counts"]["fly1"]["none"] == 1
    assert meta["per_frame"]["assign_reason"] == [["nearest"], ["none"]]


def test_mask_identity_abstains_when_the_masks_are_equidistant(tmp_path):
    """Case (4b). An instance that is NOT clearly nearer one mask than the
    other says nothing about identity, and guessing there is how a lifter puts
    one fly's track on the other's body. Geometry must abstain; the typed slot
    is then allowed to decide (recorded as "typed_fallback", so a reader can
    see the human review did not name that frame), and where no typed slot
    fired the fly is NaN."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    centres = np.zeros((2, 1, 3), np.float32)
    centres[1, :, 0] = 20.0                # 20 apart -> ONE merged window at x=10
    ok = np.ones((2, 1), bool)

    # (a) the one live instance sits exactly between the two masks (10 / 10):
    #     inside the 8-unit margin, so geometry abstains -- but it IS the
    #     female-typed slot, so she is written from it and the frame says so.
    r = PlacedFake(_fake_checkpoint(tmp_path), kp_names=_mvq_names(),
                   exist_fn=lambda c: np.array([0.0, 0.9, 0.0, 0.0], np.float32),
                   place=lambda cx, s: (10.0, 0.0, 0.0))
    res = lift_masked_bout(r, _frames(1), centres, ok, out_dir=str(tmp_path / "typed"),
                           model_names=_model_names(), identity="mask",
                           mask_sex_meta=HUMAN_MASKS)
    assert res["slot"][0, 0] == 1 and res["assign_reason"][0][0] == "typed_fallback"
    assert res["slot"][1, 0] == -1 and res["assign_reason"][1][0] == "none"

    # (b) same geometry, but the live instance is the UNTYPED "other" slot:
    #     nothing names it, so both flies are NaN rather than guessed.
    r = PlacedFake(_fake_checkpoint(tmp_path), kp_names=_mvq_names(),
                   exist_fn=lambda c: np.array([0.0, 0.0, 0.0, 0.9], np.float32),
                   place=lambda cx, s: (10.0, 0.0, 0.0))
    res = lift_masked_bout(r, _frames(1), centres, ok, out_dir=str(tmp_path / "none"),
                           model_names=_model_names(), identity="mask",
                           mask_sex_meta=HUMAN_MASKS)
    assert (res["slot"] == -1).all()
    assert res["assign_reason"] == [["none"], ["none"]]
    assert np.isnan(res["kp3d_mvq"]).all()


def test_mask_identity_collapse_guard_keeps_the_fly_nearer_its_own_mask(tmp_path):
    """Merged window (the flies are 10 units apart, so ONE crop holds both) and
    the two masks are equidistant enough that geometry abstains for both, so
    each fly falls back to its own typed slot -- and the model has put BOTH
    typed slots on the same body. Writing it twice would ship a duplicated fly
    that every jitter, confidence and residual metric rates as excellent, so
    the guard must keep the mask it is nearer -- here the male's, 2 units away
    vs the female's 8 -- and NaN the other.

    (Geometry alone can never do this: the margin test is exclusive, so one
    instance cannot belong to both masks. The typed fallback is the only route
    to a collision, which is why the guard is kept.)"""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    centres = np.zeros((2, 1, 3), np.float32)
    centres[1, :, 0] = 10.0                          # 10 units -> one merged window
    # both typed slots fire on ONE body at x=8 -> 8 units from the female mask
    # (x=0), 2 from the male (x=10); neither wins by the 8-unit margin
    r = PlacedFake(_fake_checkpoint(tmp_path), kp_names=_mvq_names(),
                   exist_fn=lambda c: np.array([0.0, 0.7, 0.95, 0.0], np.float32),
                   place=lambda cx, s: (8.0, 0.0, 0.0))
    res = lift_masked_bout(r, _frames(1), centres, np.ones((2, 1), bool),
                           out_dir=str(tmp_path / "bout"), model_names=_model_names(),
                           identity="mask", mask_sex_meta=HUMAN_MASKS)
    assert list(res["n_windows"]) == [1]
    assert res["collapsed"][0]
    assert res["n_collapsed"] == {"fly0": 1, "fly1": 0}      # she was the further one
    assert res["slot"][0, 0] == -1 and res["slot"][1, 0] == 2


def test_mask_identity_falls_back_to_the_sex_head_without_a_human_review(tmp_path):
    """Masks with no human id review carry no identity to honour: the mode
    must fall back to the sex head AND say so (a silent fallback would make an
    `identity=mask` run mean two different things across recordings)."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    centres, ok = _two_mask_windows()
    r = PlacedFake(_fake_checkpoint(tmp_path), kp_names=_mvq_names(),
                   exist_fn=_female_window_reads_male())
    out = tmp_path / "bout"
    with pytest.warns(RuntimeWarning, match="human_id_review"):
        res = lift_masked_bout(r, _frames(1), centres, ok, out_dir=str(out),
                               model_names=_model_names(), identity="mask",
                               mask_sex_meta={"method": "mask_area_vote", "male_slot": 1})
    assert res["identity"] == "mask" and res["identity_resolved"] == "sex"
    assert res["slot"][0, 0] == -1                        # the sex-head behaviour
    assert res["window"][1, 0] == 0
    meta = json.load(open(out / "mvq_meta.json"))
    assert meta["per_frame"]["identity_source"] == ["sex"]
    # the sex.json method names what actually decided identity
    assert json.load(open(out / "sex.json"))["method"] == "mvq_sex_head"


def test_a_fallback_bout_is_gated_as_sex_and_is_not_current_for_mask(tmp_path):
    """The fallback must be gated on the mode that ACTUALLY RAN, not the one
    asked for.

    A bout whose masks carry no human review runs the sex head even under
    `--identity mask`. If it were stamped `identity: mask`, then
    `bout_lift_is_current` would answer True for the mask gate -- and the
    moment someone canonicalizes a human review into those masks, the re-lift
    that should now produce mask identities would be SKIPPED as already
    current, leaving a sex-head bout inside a run labelled `mask` with nothing
    downstream able to tell.
    """
    from jarvis_jax.tracking.lift_mvq import (bout_lift_is_current, lift_masked_bout,
                                              mvq_gate_string, resolve_mask_identity)
    ckpt = _fake_checkpoint(tmp_path)
    centres, ok = _two_mask_windows()
    r = PlacedFake(ckpt, kp_names=_mvq_names(), exist_fn=_female_window_reads_male())
    out = tmp_path / "bout"
    with pytest.warns(RuntimeWarning, match="human_id_review"):
        res = lift_masked_bout(r, _frames(1), centres, ok, out_dir=str(out),
                               model_names=_model_names(), identity="mask",
                               mask_sex_meta={"method": "mask_area_vote", "male_slot": 1})

    # `containment="off"` throughout this file: these lifts pass no
    # `mask_store` (they build `centres`/`ok` by hand), and the containment
    # filter needs the masks themselves, so it cannot run and the lift is
    # gated as off. `tests/test_lift_mask_containment.py` covers the on arm.
    g_sex = mvq_gate_string(ckpt, step=None, exist_thresh=0.5, identity="sex",
                            containment="off")
    g_mask = mvq_gate_string(ckpt, step=None, exist_thresh=0.5, identity="mask",
                             containment="off")
    assert res["gates"] == g_sex                       # stamped as what it RAN
    with np.load(out / "fly0" / "kp3d.npz") as z:
        assert str(z["gates"]) == g_sex
    assert bout_lift_is_current(str(out), g_sex)
    assert not bout_lift_is_current(str(out), g_mask)   # the re-lift will happen

    # and once the masks DO carry the review, the resolution -- and so the gate
    # the caller compares against -- becomes "mask", which this bout fails
    assert resolve_mask_identity("mask", {"method": "mask_area_vote"})[0] == "sex"
    assert resolve_mask_identity("mask", HUMAN_MASKS)[0] == "mask"
    assert resolve_mask_identity("sex", HUMAN_MASKS)[0] == "sex"     # never upgraded
    assert resolve_mask_identity("mask", None)[1] is not None        # a message, not silence

    # re-lifting the SAME bout after the review lands is not skipped, and now
    # writes mask identities
    r2 = PlacedFake(ckpt, kp_names=_mvq_names(), exist_fn=_female_window_reads_male())
    res2 = lift_masked_bout(r2, _frames(1), centres, ok, out_dir=str(out),
                            model_names=_model_names(), identity="mask",
                            mask_sex_meta=HUMAN_MASKS)
    assert res2["skipped"] is False and res2["identity_resolved"] == "mask"
    assert res2["gates"] == g_mask and res2["slot"][0, 0] == 2
    assert json.load(open(out / "sex.json"))["method"] == "mask_human_id_review"


def test_mask_identity_refuses_masks_whose_male_is_not_slot_1(tmp_path):
    """`fly{f}` IS mask fly `f` on this route, so "male = mask slot 1" is the
    premise of writing `male_fly: 1`. A mask npz the review resolved the other
    way must raise rather than write a sex.json that names the wrong fly."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    centres, ok = _two_mask_windows()
    r = PlacedFake(_fake_checkpoint(tmp_path), kp_names=_mvq_names())
    with pytest.raises(ValueError, match="male = mask slot 0"):
        lift_masked_bout(r, _frames(1), centres, ok, out_dir=str(tmp_path / "bout"),
                         model_names=_model_names(), identity="mask",
                         mask_sex_meta={"method": "human_id_review", "male_slot": 0})


def test_gate_string_distinguishes_the_identity_modes(tmp_path):
    """Case (5). The two modes can write DIFFERENT keypoints from the same
    weights, so the Stage-B gate string has to name which one ran -- otherwise
    a run switched to `identity=mask` silently reuses the sex-head bouts it was
    meant to replace."""
    from jarvis_jax.tracking.lift_mvq import (bout_lift_is_current, lift_masked_bout,
                                              mvq_gate_string)
    ckpt = _fake_checkpoint(tmp_path)
    # containment="off": this lift passes no `mask_store`, so the filter
    # cannot run and the bout is gated as off (see the note above).
    g_mask = mvq_gate_string(ckpt, step=None, exist_thresh=0.5, identity="mask",
                             containment="off")
    g_sex = mvq_gate_string(ckpt, step=None, exist_thresh=0.5, identity="sex",
                            containment="off")
    assert g_mask != g_sex
    assert json.loads(g_mask)["identity"] == "mask"
    assert json.loads(g_sex)["identity"] == "sex"
    with pytest.raises(ValueError, match="identity"):
        mvq_gate_string(ckpt, step=None, exist_thresh=0.5, identity="whatever")

    centres, ok = _two_mask_windows()
    r = PlacedFake(ckpt, kp_names=_mvq_names())
    out = tmp_path / "bout"
    lift_masked_bout(r, _frames(1), centres, ok, out_dir=str(out),
                     model_names=_model_names(), identity="mask",
                     mask_sex_meta=HUMAN_MASKS)
    with np.load(out / "fly0" / "kp3d.npz") as z:
        assert str(z["gates"]) == g_mask
    assert bout_lift_is_current(str(out), g_mask)
    assert not bout_lift_is_current(str(out), g_sex)       # the other mode is NOT current


def test_gate_string_with_identity_is_what_run_bout_stage_b_expects(tmp_path):
    """The same contract as `test_gates_string_is_what_run_bout_stage_b_expects`,
    now with `mvq.identity` in the config: run_bout's OWN
    `stage_b_gate_signature` must reproduce the string a mask-identity lift
    stamps, or every mask-identity bout is refused at Stage B."""
    from jarvis_jax.tracking.lift_mvq import lift_masked_bout
    ckpt = _fake_checkpoint(tmp_path)
    centres, ok = _two_mask_windows()
    r = PlacedFake(ckpt, kp_names=_mvq_names(), exist_thresh=0.55)
    out = tmp_path / "bout"
    lift_masked_bout(r, _frames(1), centres, ok, out_dir=str(out),
                     model_names=_model_names(), identity="mask",
                     mask_sex_meta=HUMAN_MASKS)

    fn = next(n for n in ast.parse(RUN_BOUT.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "stage_b_gate_signature")
    g = {"json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<rb>", "exec"), g)
    cfg = OmegaConf.create({"detector": {"conf_thresh": 0.3},
                            "pipeline": {"lifter": "mvq"},
                            "mvq": {"checkpoint": ckpt, "step": None,
                                    "exist_thresh": 0.55, "identity": "mask",
                                    "containment": "off"}})
    with np.load(out / "fly0" / "kp3d.npz") as z:
        assert str(z["gates"]) == g["stage_b_gate_signature"](cfg)
    # `containment` moves the string too, so a run that turns the filter on
    # does NOT accept a bout lifted without it (and vice versa)
    cfg_on = OmegaConf.merge(cfg, {"mvq": {"containment": "on"}})
    assert g["stage_b_gate_signature"](cfg_on) != g["stage_b_gate_signature"](cfg)
    # and the shipped config carries the mode this campaign runs
    _p3a = OmegaConf.load(REPO / "configs" / "mvq" / "p3a.yaml")
    assert str(_p3a.identity) == "mask"
    assert str(_p3a.containment) == "on"


def test_canonicalize_bout_treats_mask_human_id_review_as_authoritative(tmp_path):
    """Case (6). A bout typed from the HUMAN id review carried by the masks is
    the strongest identity this pipeline has. `canonicalize_bout` must not
    re-decide it from the wing-song CV -- exactly as it already does not for
    `mvq_sex_head`."""
    from jarvis_jax.tracking.sexing import MASK_ID_SEX_METHOD, canonicalize_bout
    assert MASK_ID_SEX_METHOD == "mask_human_id_review"
    bout = tmp_path / "bout_00025"
    (bout / "fly0").mkdir(parents=True)
    (bout / "fly1").mkdir(parents=True)
    names = _model_names()
    rng = np.random.default_rng(0)
    for fly, jitter in ((0, 1.0), (1, 0.0)):            # CV would prefer fly0 as male
        T, K = 60, len(names)
        kp = np.zeros((T, K, 3), np.float32)
        idx = {n: i for i, n in enumerate(names)}
        kp[:, idx["Scutellum"]] = [0, 0, 0]
        kp[:, idx["Abd_tip"]] = [0, -1, 0]
        for s in "LR":
            kp[:, idx[f"Wing{s}_base"]] = [0, 0, 0]
            ang = rng.normal(size=T) * jitter
            kp[:, idx[f"Wing{s}_V13"]] = np.stack(
                [np.sin(ang), np.cos(ang), np.zeros(T)], axis=1)
        np.savez(bout / f"fly{fly}" / "kp3d.npz", kp3d=kp,
                 conf3d=np.ones((T, K), np.float32))
    payload = {"male_fly": 1, "original_male_fly": 1, "applied_swap": False,
               "confidence": "high", "method": MASK_ID_SEX_METHOD,
               "authority": MASK_ID_SEX_METHOD, "note": "mask human id review"}
    (bout / "sex.json").write_text(json.dumps(payload))
    before = (bout / "sex.json").read_text()

    res = canonicalize_bout(str(bout), names, verbose=False)
    assert res["applied_swap"] is False
    assert res["method"] == MASK_ID_SEX_METHOD and res["authority"] == MASK_ID_SEX_METHOD
    assert res["male_fly"] == 1
    assert (bout / "sex.json").read_text() == before          # untouched
    # `mvq_sex_head` keeps working exactly as before -- this is an addition
    (bout / "sex.json").write_text(json.dumps(dict(payload, method="mvq_sex_head",
                                                   authority="mvq_sex_head")))
    res = canonicalize_bout(str(bout), names, verbose=False)
    assert res["method"] == "mvq_sex_head" and res["applied_swap"] is False


def test_mvq_lift_cli_exposes_the_identity_mode():
    """The campaign runs this from the CLI and from the slurm array text; both
    have to be able to name the mode, and `mask` is the default."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mvq_lift_bout_wt", REPO / "scripts" / "mvq_lift_bout.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a = m.build_parser().parse_args(
        ["--session-dir", "/s", "--out", "/o", "--run", "/r", "--bout", "1"])
    assert a.identity == "mask"
    assert a.mask_assign_margin_units == 8.0 and a.mask_assign_max_units == 60.0
    a = m.build_parser().parse_args(
        ["--session-dir", "/s", "--out", "/o", "--run", "/r", "--bout", "1",
         "--identity", "sex", "--mask-assign-margin-units", "6",
         "--mask-assign-max-units", "40"])
    assert a.identity == "sex" and a.mask_assign_margin_units == 6.0
    assert a.mask_assign_max_units == 40.0

    s = _slurm_mod().build_mvq_lift_array_script(
        job_name="mvq", partition="ckpt-all", account="portia", cpus=8, mem=48,
        gpus=1, time_limit="12:00:00", requeue=True, conda_env="3d_tracking",
        idxs=[25], session_dir="/s", predictions_dir="/p", run_dir="/r",
        checkpoint="/ckpt/final", step=None, exist_thresh=0.5, batch=8,
        identity="mask", anatomy_cfg="configs/anatomy/v1.yaml",
        recording_cfg="configs/recording/session0.yaml")
    assert "--identity mask" in s
    assert "--mask-assign-margin-units 8.0" in s and "--mask-assign-max-units 60.0" in s
