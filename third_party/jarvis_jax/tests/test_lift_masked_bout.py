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
import json
import os
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
RUN_BOUT = REPO / "scripts" / "run_bout.py"
ANATOMY_V1 = REPO / "configs" / "anatomy" / "v1.yaml"

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
                 exist_thresh=0.5, exist_fn=None, sex_prob=(0.5, 0.9, 0.1, 0.5)):
        self.checkpoint = os.path.abspath(checkpoint)
        self.step = None
        self.step_label = "final"
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
    r = FakeRunner(ckpt, kp_names=mvq_names)
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
            assert str(z["gates"]) == mvq_gate_string(ckpt, step=None, exist_thresh=0.5)
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
    r = FakeRunner(ckpt, kp_names=_mvq_names(), exist_thresh=0.55)
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
                                    "exist_thresh": 0.55}})
    with np.load(out / "fly0" / "kp3d.npz") as z:
        assert str(z["gates"]) == g["stage_b_gate_signature"](cfg)


def test_lift_is_idempotent_and_force_reruns(tmp_path):
    from jarvis_jax.tracking.lift_mvq import bout_lift_is_current, lift_masked_bout
    r = FakeRunner(_fake_checkpoint(tmp_path), kp_names=_mvq_names())
    centres = np.zeros((2, 1, 3), np.float32)
    centres[1, :, 0] = 200.0
    out = tmp_path / "bout"
    assert not bout_lift_is_current(str(out), r.gates_string())
    lift_masked_bout(r, _frames(1), centres, np.ones((2, 1), bool),
                     out_dir=str(out), model_names=_model_names())
    assert bout_lift_is_current(str(out), r.gates_string())
    # a different checkpoint (or exist_thresh) is NOT current -- that is the
    # whole point of stamping the gates string
    other = FakeRunner(_fake_checkpoint(tmp_path, "final2"), kp_names=_mvq_names())
    assert not bout_lift_is_current(str(out), other.gates_string())

    res = lift_masked_bout(r, _frames(1), centres, np.ones((2, 1), bool),
                           out_dir=str(out), model_names=_model_names())
    assert res["skipped"] is True
    res = lift_masked_bout(r, _frames(1), centres, np.ones((2, 1), bool),
                           out_dir=str(out), model_names=_model_names(), force=True)
    assert res["skipped"] is False


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
