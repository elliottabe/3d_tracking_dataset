"""`scripts/pseudo_labels/singlefly_p3b_pass.py` -- the single-fly mask-free
P3b pass that writes P3b-SHAPED bout dirs for the v2 pseudo-labels (spec
2026-09-05 §3.4).

What is load-bearing here, and why each check exists:

  * ONE FLY DIR, and it says so. The masked lifter hardcodes two typed slots;
    this route writes `fly0` only, with `identity_resolved: "single_typed"` --
    a name that is deliberately NEITHER shipped identity mode, so the
    extractor's identity gate (which admits only `"mask"`) rejects these
    frames unless it is run with `--no-identity-gate`. The last test asserts
    exactly that, on the real gate stack.
  * KEYPOINT ORDER. The written npz must speak `cfg.model.KP_NAMES`
    (XML/model order), not the model's own detector order, and the EyeL-EyeR
    spacing -- a RIGID head pair, invariant under a by-name permutation and
    almost certainly changed by a by-index one -- is what proves it. That is
    CLAUDE.md's keypoint-order trap, which once read a middle-left leg as a
    "collapsed right wing vein" with perfect metrics.
  * THE SINGLE-FLY READ. `read_typed(want_sex=-1)` takes whichever typed slot
    exists and REPORTS its sex; demanding one would drop every frame the model
    happened to type the other way, which on a single-fly recording there is
    no second animal to disambiguate.
  * NO COLLAPSE, NO CONTAINMENT. Both are two-fly concepts (a second slot to
    collapse against, another fly's mask to be contained by); they must be
    written as all-zero / false by construction, not left out, because
    `pseudo_gates` reads them.
  * `--link-cameras`. Clip's mp4s are `Cam..._frames_<a>_<b>.mp4` while every
    reader resolves `<Cam>.mp4`. Two videos mapping to one camera name must
    RAISE -- picking whichever sorted first would read a different span of
    time for that camera than for the others, one level down from the
    camera-order trap.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from pseudo_fixtures import CAMS, cam_mats, project

REPO = Path(__file__).resolve().parents[3]
MOD = REPO / "scripts" / "pseudo_labels" / "singlefly_p3b_pass.py"
ANATOMY_V1 = REPO / "configs" / "anatomy" / "v1.yaml"


def _load():
    spec = importlib.util.spec_from_file_location("singlefly_p3b_pass", MOD)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _model_names():
    return [str(n) for n in OmegaConf.load(ANATOMY_V1).model.KP_NAMES]


def _mvq_names():
    """The same 50 names in the model's own (detector) order -- a rotation of
    KP_NAMES, so a by-index write is caught and a by-name one is not."""
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
    """An `MVQRunner` with the 448-crop DINOv3 forward replaced by arithmetic
    (the style of `test_lift_masked_bout.FakeRunner`).

    `read_typed`, `to_pipeline` and the gate signature are the REAL methods,
    bound off the class, so the single-fly slot rule, the by-name permutation
    and the gate string under test are the shipped ones.

    For window `b` and slot `s`, `kp3d = body + centre[b] + (100 s, 0, 0)` --
    so a frame's fly is traceable to the slot it was read from -- and `kp2d`
    is the EXACT projection of that through `cam_mats`, which is what lets the
    written bout dir be run through the real reprojection gate.
    """

    from jarvis_jax.tracking.lift_mvq import MVQRunner as _R
    read_typed = _R.read_typed
    to_pipeline = _R.to_pipeline
    gates_signature = _R.gates_signature
    gates_string = _R.gates_string
    del _R

    def __init__(self, checkpoint, *, kp_names, cm, cameras=CAMS, batch=8,
                 exist_thresh=0.5, exist_fn=None, sex_prob=(0.5, 0.99, 0.01, 0.5),
                 identity="sex", slot_offset=100.0):
        self.checkpoint = os.path.abspath(checkpoint)
        self.step = None
        self.step_label = "final"
        self.identity = identity
        self.containment = False
        self.window_pref = "own"
        self.kp_names = list(kp_names)
        self.K = len(self.kp_names)
        self.I = 4
        self.cameras = list(cameras)
        self.C = len(self.cameras)
        self.cam_mats = np.asarray(cm, np.float64)
        self.batch = int(batch)
        self.exist_thresh = float(exist_thresh)
        self.meta = {"keypoint_names": list(kp_names)}
        self._gates = None
        self._sex_prob = np.asarray(sex_prob, np.float32)
        # how far apart the four slots' skeletons sit: 100 units is a PHANTOM
        # (a second animal somewhere else), 0 is the same fly read twice
        self.slot_offset = float(slot_offset)
        # a real, asymmetric body: EyeL-EyeR is a non-zero distance, so the
        # permutation invariant has something to be invariant about
        self.body = (np.random.default_rng(0).normal(size=(self.K, 3))
                     * np.array([3.0, 1.5, 1.0]))
        self._exist_fn = exist_fn or (lambda centre: np.array([0.0, 0.9, 0.0, 0.0], np.float32))

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
        kp2d = np.zeros((b, self.I, self.C, self.K, 2), np.float32)
        for i in range(b):
            for s in range(self.I):
                kp3d[i, s] = (self.body + centres[i]
                              + np.array([self.slot_offset * s, 0.0, 0.0]))
                kp2d[i, s] = project(self.cam_mats, kp3d[i, s])
        exist = np.stack([np.asarray(self._exist_fn(centres[i]), np.float32)
                          for i in range(b)])
        return {"kp3d": kp3d, "kp2d": kp2d,
                "vis": np.full((b, self.I, self.C, self.K), 0.8, np.float32),
                "exist": exist,
                "sex_prob": np.broadcast_to(self._sex_prob, (b, self.I)).copy(),
                "conf_raw": np.full((b, self.I, self.K), 0.05, np.float32),
                "xyz": np.zeros((b, self.I, self.K, 3), np.float32)}


class FakeDetector:
    """CenterDetect stand-in: peak 0 of every camera is the exact projection
    of `centre(t)`, peak 1 is NaN (one fly). `centre(t) -> None` means the
    detector found nothing on that frame, which is the reuse path."""

    def __init__(self, cm, centre):
        self.cm = np.asarray(cm, np.float64)
        self.centre = centre
        self.t = -1

    def peaks(self, frames):
        self.t += 1
        C = self.cm.shape[0]
        pk = np.full((C, 2, 2), np.nan, np.float32)
        sc = np.zeros((C, 2), np.float32)
        c = self.centre(self.t)
        if c is not None:
            pk[:, 0] = project(self.cm, np.asarray(c, np.float64)[None])[:, 0]
            sc[:, 0] = 0.9
        return pk, sc


def _frames(n, C=len(CAMS)):
    """`(frames, present)` per frame -- the fake runner never looks at pixels."""
    return [(np.zeros((C, 8, 8, 3), np.uint8), np.ones(C, bool)) for _ in range(n)]


def _run(m, tmp_path, *, T=24, exist_fn=None, centre=None, out="bout", fly_sex="female",
         slot_offset=100.0):
    cm = cam_mats()
    r = FakeRunner(_fake_checkpoint(tmp_path), kp_names=_mvq_names(), cm=cm,
                   exist_fn=exist_fn, slot_offset=slot_offset)
    centre = centre or (lambda t: np.array([0.2 * t, 0.0, 0.0]))
    det = FakeDetector(cm, centre)
    out_dir = str(tmp_path / out)
    res = m.lift_singlefly_bout(r, det, iter(_frames(T)), frames=list(range(1000, 1000 + T)),
                                out_dir=out_dir, model_names=_model_names(),
                                fly_sex=fly_sex,
                                meta_extra={"frame_start": 1000, "session_dir": str(tmp_path)})
    return res, out_dir, cm


# --------------------------------------------------------------- what is written
def test_writes_exactly_one_fly_dir_in_model_order_flagged_single_typed(tmp_path):
    """The interface Task 2's extractor reads: one `fly0`, `kp_names` ==
    `cfg.model.KP_NAMES`, `identity_resolved` naming the single-fly rule, and
    the two-fly bookkeeping (`collapsed`, `containment`) present but empty --
    `pseudo_gates` reads all four."""
    m = _load()
    res, out_dir, _ = _run(m, tmp_path, T=24)

    flies = sorted(d for d in os.listdir(out_dir) if d.startswith("fly"))
    assert flies == ["fly0"], f"single-fly pass wrote {flies}"

    meta = json.loads(Path(out_dir, "mvq_meta.json").read_text())
    assert meta["identity_resolved"] == "single_typed"
    assert meta["single_fly"] is True and meta["want_sex"] == -1
    assert meta["containment"] is False
    assert meta["per_frame"]["collapsed"] == [0] * 24
    assert meta["n_collapsed"] == {"fly0": 0} and meta["collapsed_frac"] == 0.0
    assert meta["per_frame"]["identity_source"] == ["single_typed"] * 24
    assert meta["n_frames"] == 24 and meta["frame_start"] == 1000
    assert np.asarray(meta["per_frame"]["exist"]).shape == (1, 24)   # (F,T), F=1
    assert meta["cameras"] == CAMS
    assert meta["fly_sex"] == {"fly0": "female"}

    z3 = np.load(os.path.join(out_dir, "fly0", "kp3d.npz"), allow_pickle=True)
    z2 = np.load(os.path.join(out_dir, "fly0", "kp2d.npz"), allow_pickle=True)
    assert [str(s) for s in z3["kp_names"]] == _model_names()
    assert [str(s) for s in z2["kp_names"]] == _model_names()
    assert [str(c) for c in z2["cameras"]] == CAMS
    assert z3["kp3d"].shape == (24, len(_model_names()), 3)
    assert z2["kp2d"].shape == (24, len(CAMS), len(_model_names()), 2)
    assert str(z3["gates"]) == res["gates"] and json.loads(str(z3["gates"]))["lifter"] == "mvq"
    # the single-fly read is not a mask lift, and the gate string must not
    # claim otherwise
    assert json.loads(str(z3["gates"]))["containment"] is False


def test_the_eye_invariant_survives_the_written_permutation(tmp_path):
    """EyeL-EyeR is a RIGID pair: the by-name permutation cannot move it, and
    a by-index one would. `lift_singlefly_bout` calls `_check_eye_invariant`
    before writing -- this asserts the invariant on the FILE, and that the
    check itself is not vacuous."""
    from jarvis_jax.tracking.lift_mvq import _check_eye_invariant
    m = _load()
    res, out_dir, _ = _run(m, tmp_path, T=8)
    names, mvq = _model_names(), _mvq_names()
    kp3d = np.load(os.path.join(out_dir, "fly0", "kp3d.npz"))["kp3d"]

    d_file = np.linalg.norm(kp3d[:, names.index("EyeL")] - kp3d[:, names.index("EyeR")],
                            axis=-1)
    d_mvq = np.linalg.norm(res["kp3d_mvq"][:, mvq.index("EyeL")]
                           - res["kp3d_mvq"][:, mvq.index("EyeR")], axis=-1)
    assert np.all(d_file > 1e-3), "a degenerate eye pair would make this test vacuous"
    np.testing.assert_allclose(d_file, d_mvq, atol=1e-4)

    scrambled = res["kp3d_mvq"][:, ::-1]                 # a by-index 'permutation'
    with pytest.raises(RuntimeError, match="EyeL-EyeR"):
        _check_eye_invariant(res["kp3d_mvq"], scrambled, mvq, names)


# ------------------------------------------------------------ the single-fly read
def test_the_read_takes_whichever_typed_slot_exists_and_reports_its_sex(tmp_path):
    """`want_sex=-1` (spec §4.2): a single-fly recording has no second animal
    to disambiguate against, so demanding the female slot would drop every
    frame the model typed male. The sex is REPORTED, not assumed."""
    m = _load()
    male_only = lambda centre: np.array([0.0, 0.1, 0.95, 0.0], np.float32)   # noqa: E731
    res, out_dir, _ = _run(m, tmp_path, T=6, exist_fn=male_only, fly_sex=None)
    assert list(res["slot"]) == [2] * 6                  # SLOT_MALE
    assert res["fly_sex"] == "male" and res["sex_read"]["n_male_slot"] == 6
    meta = json.loads(Path(out_dir, "mvq_meta.json").read_text())
    assert meta["fly_sex"] == {"fly0": "male"}
    assert meta["sex_source"] == "mvq_sex_head_majority"

    female_only = lambda centre: np.array([0.0, 0.95, 0.1, 0.0], np.float32)  # noqa: E731
    res, _, _ = _run(m, tmp_path, T=6, exist_fn=female_only, fly_sex=None, out="b2")
    assert list(res["slot"]) == [1] * 6 and res["fly_sex"] == "female"


def test_a_frame_below_exist_thresh_is_nan_with_no_fallback(tmp_path):
    """Both typed slots weak -> the frame is NaN and says so (slot -1), never
    a fallback to the untyped slot 0 or 3, which have no fixed meaning."""
    m = _load()
    weak = lambda centre: np.array([0.99, 0.2, 0.2, 0.99], np.float32)      # noqa: E731
    res, out_dir, _ = _run(m, tmp_path, T=4, exist_fn=weak)
    assert list(res["slot"]) == [-1] * 4
    kp3d = np.load(os.path.join(out_dir, "fly0", "kp3d.npz"))["kp3d"]
    assert np.isnan(kp3d).all()
    meta = json.loads(Path(out_dir, "mvq_meta.json").read_text())
    assert meta["n_missing"] == {"fly0": 4}
    # the phantom-fly statistic still reports the two untyped slots
    assert meta["single_fly_check"]["frac_exactly_one_slot_over_0.5"] == 0.0


def test_a_frame_with_no_centre_reuses_the_previous_one(tmp_path):
    """The coarse pass's rule (`coarse_track` docstring): at 800 fps a fly
    moves ~0.6 mm between frames, well inside the 5.6 mm window, so a
    CenterDetect miss REUSES the last centre and says `centre_source=1`. With
    no history there is nothing to reuse and the row stays NaN -- never a
    centre at the world origin, which would crop the arena floor and come back
    confident."""
    m = _load()

    def centre(t):
        if t in (0, 3):                     # 0: no history; 3: a mid-bout miss
            return None
        return np.array([0.2 * t, 0.0, 0.0])

    res, out_dir, _ = _run(m, tmp_path, T=6, centre=centre)
    assert list(res["centre_source"]) == [2, 0, 0, 1, 0, 0]
    assert res["slot"][0] == -1, "no centre and no history must not be lifted"
    assert res["slot"][3] >= 0, "a reused centre must still be lifted"
    meta = json.loads(Path(out_dir, "mvq_meta.json").read_text())
    assert meta["n_no_centre"] == 1
    assert meta["single_fly_check"]["frac_centre_reused"] == pytest.approx(1 / 6, abs=1e-4)


def test_the_check_block_reports_the_phantom_fly_quantities(tmp_path):
    """The numbers p3b-notes.md's single-fly table is built from: how often
    EXACTLY one slot clears 0.5, the second-highest slot's mean existence, and
    the median per-keypoint step.

    Two live typed slots alone do NOT mean a phantom -- on a real single-fly
    span the model hedges on sex and lights BOTH typed slots on the SAME
    animal. What separates the two is the distance between the two slots'
    skeletons, so the check block reports that against `COLLAPSE_DIST_UNITS`
    (3 units = 0.3 mm, `pick_typed_pair`'s own same-fly rule)."""
    from jarvis_jax.tracking.lift_mvq import COLLAPSE_DIST_UNITS
    m = _load()
    two_slots = lambda centre: np.array([0.0, 0.95, 0.93, 0.0], np.float32)  # noqa: E731

    # PHANTOM: the two typed slots sit 100 units (10 mm) apart
    res, _, _ = _run(m, tmp_path, T=8, exist_fn=two_slots)
    c = res["single_fly_check"]
    assert c["frac_exactly_one_slot_over_0.5"] == 0.0
    assert c["frac_exactly_one_typed_slot_over_0.5"] == 0.0
    assert c["second_slot_exist_mean"] == pytest.approx(0.93, abs=1e-4)
    assert c["second_typed_slot_exist_mean"] == pytest.approx(0.93, abs=1e-4)
    assert c["n_both_typed_slots"] == 8
    assert c["second_typed_slot_median_dist_units"] == pytest.approx(100.0, abs=1e-2)
    assert c["n_second_typed_slot_beyond_collapse"] == 8
    assert c["collapse_dist_units"] == COLLAPSE_DIST_UNITS

    # THE SAME FLY READ TWICE: both slots live, but on one animal
    res, _, _ = _run(m, tmp_path, T=8, exist_fn=two_slots, slot_offset=0.0, out="same")
    c = res["single_fly_check"]
    assert c["n_both_typed_slots"] == 8
    assert c["second_typed_slot_median_dist_units"] == pytest.approx(0.0, abs=1e-3)
    assert c["n_second_typed_slot_beyond_collapse"] == 0

    res, _, _ = _run(m, tmp_path, T=8, out="clean")      # the default: one live slot
    c = res["single_fly_check"]
    assert c["frac_exactly_one_slot_over_0.5"] == 1.0
    assert c["frac_exactly_one_typed_slot_over_0.5"] == 1.0
    assert c["second_slot_exist_mean"] == pytest.approx(0.0, abs=1e-4)
    assert c["n_both_typed_slots"] == 0
    assert c["exist_mean_per_slot"] == [0.0, 0.9, 0.0, 0.0]
    # the fly drifts 0.2 units/frame, so every keypoint steps 0.2 units
    assert c["median_step_units"] == pytest.approx(0.2, abs=1e-3)


# ------------------------------------------------- the extractor reads these dirs
def test_the_real_gate_stack_admits_the_bout_only_without_the_identity_gate(tmp_path):
    """The whole point of the P3b SHAPE: `pseudo_gates.load_bout_arrays` +
    `admit_bout` -- what `extract_p3b_pseudolabels.scan_bout` runs -- must open
    these dirs with `n_flies=1` and `store=None` (there are no SAM3 masks for
    a free-running recording, so the containment gate cannot run) and admit
    frames. With the identity gate ON, `identity_source="single_typed"` must
    reject every frame: single-fly data enters only through
    `--no-identity-gate`, never by accident."""
    from jarvis_jax.data.pseudo_gates import GateThresholds, admit_bout, load_bout_arrays
    m = _load()
    _res, out_dir, cm = _run(m, tmp_path, T=40)

    arrays = load_bout_arrays(out_dir, cameras=CAMS, n_flies=1)
    assert list(arrays.kp_names) == _model_names()
    thr = GateThresholds()
    r = admit_bout(arrays, None, cm, thr, use_identity=False)
    assert r.fly.shape == (1, 40)
    assert r.fly.all(), "every frame of a clean synthetic bout should pass the gates"
    anchors = np.flatnonzero(r.frame)
    assert anchors.size >= 2 and np.all(np.diff(anchors) >= thr.decorrelation)

    r_id = admit_bout(arrays, None, cm, thr, use_identity=True)
    assert not r_id.frame.any(), "the identity gate must refuse a single_typed bout"


class _FakeFrames:
    """448x448 flat-grey frames, so `V12WindowDataset._build` can cut a real
    crop and the extractor's own `verify_export` (which rejects all-black
    crops) is exercised rather than tripped."""

    HW = (448, 448)

    def __init__(self, sessions, cameras):
        self.n = len(cameras)

    def __call__(self, rec, frame):
        return np.full((self.n,) + self.HW + (3,), 40, np.uint8), np.ones(self.n, bool)

    def close(self):
        pass


def test_the_extractor_exports_a_single_fly_run_end_to_end(tmp_path, monkeypatch):
    """The command this task exists to make work:

        extract_p3b_pseudolabels.py --runs <singlefly run> --num-animals 1
            --no-identity-gate --target N

    It must run the real gates with NO mask store, draw the one host sex, and
    write a v12 export whose recordings say `n_flies: 1`, `behavior:
    free_running` and `sex: female` -- claiming those frames are courtship
    pairs identified by a human mask review would be false in the manifest and
    in every annotation. `female_host_weight` must be null, not 0.0: a
    one-sided export has no host-sex ratio to restore, and 0.0 reads as
    "weight these to nothing".
    """
    from pseudo_fixtures import make_calib
    m = _load()

    rec, cm = "2026_03_03_14_32_16", cam_mats()
    root = tmp_path / rec / "pose_mvq_p3b_singlefly"
    video = tmp_path / "video"
    make_calib(video / "calibration", cm)
    r = FakeRunner(_fake_checkpoint(tmp_path), kp_names=_mvq_names(), cm=cm)
    det = FakeDetector(cm, lambda t: np.array([0.2 * t, 0.0, 0.0]))
    m.lift_singlefly_bout(r, det, iter(_frames(60)), frames=list(range(1000, 1060)),
                          out_dir=str(root / "bouts" / "bout_00000"),
                          model_names=_model_names(), fly_sex="female",
                          meta_extra={"bout": 0, "frame_start": 1000,
                                      "session_dir": str(video),
                                      "behavior": "free_running",
                                      "recording_sex": "female",
                                      "sex_source": "recording_ground_truth"})

    names_root = tmp_path / "names"
    (names_root / "annotations").mkdir(parents=True)
    (names_root / "annotations" / "keypoint_names.json").write_text(
        json.dumps(list(reversed(_model_names()))))       # a DIFFERENT order on purpose

    spec = importlib.util.spec_from_file_location(
        "extract_p3b_pseudolabels",
        REPO / "scripts" / "pseudo_labels" / "extract_p3b_pseudolabels.py")
    ex = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ex
    spec.loader.exec_module(ex)
    monkeypatch.setattr(ex, "SessionFrames", _FakeFrames)

    out = str(tmp_path / "export")
    rep = ex.main(["--runs", str(root), "--out", out,
                   "--export-names-from", str(names_root),
                   "--num-animals", "1", "--no-identity-gate", "--target", "3",
                   "--min-female", "0", "--per-rec-frac", "1.0", "--per-bout-cap", "100",
                   "--figures-dir", os.path.join(out, "fig")])

    assert rep["realised"] >= 3
    man = json.load(open(os.path.join(out, "manifest.json")))
    assert man["balance"]["female_host_weight"] is None
    assert man["balance"]["male_n"] == 0 and man["balance"]["female_n"] == 3
    info = man["recordings"][rec]
    assert info["n_flies"] == 1 and info["behavior"] == "free_running"
    assert info["sex"] == "female" and info["fly_sex"] == {"fly0": "female"}
    coco = json.load(open(os.path.join(out, "annotations", "instances_train.json")))
    assert coco["keypoint_names"] == list(reversed(_model_names()))
    assert {a["sex"] for a in coco["annotations"]} == {"female"}
    assert all(v["fly_id"] == 0 for v in coco["framesets"].values())
    # no masks were written, and none were needed to gate the bout
    assert not os.path.isdir(os.path.join(out, "masks"))
    assert rep["verify"]["windows_checked"] >= 1


def test_the_extractor_refuses_target_on_a_two_sex_pool(tmp_path, monkeypatch):
    """`--target` sizes ONE side. On the masked campaign's two-sex pool a
    single number cannot say how the sides split, and guessing would
    un-balance exactly what `female_host_weight` exists to fix -- so it must
    refuse rather than pick."""
    from pseudo_fixtures import make_bout, make_calib
    _d, _npz, cm = make_bout(tmp_path, T=40, sep_units=40.0, name="bout_00001",
                             hw=_FakeFrames.HW, frame_start=1000,
                             masks_path=str(tmp_path / "m1" / "sam3_masks.npz"))
    make_calib(tmp_path / "video" / "calibration", cm)
    names_root = tmp_path / "names"
    (names_root / "annotations").mkdir(parents=True)
    from pseudo_fixtures import KP_NAMES
    (names_root / "annotations" / "keypoint_names.json").write_text(json.dumps(KP_NAMES))

    spec = importlib.util.spec_from_file_location(
        "extract_p3b_pseudolabels",
        REPO / "scripts" / "pseudo_labels" / "extract_p3b_pseudolabels.py")
    ex = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ex
    spec.loader.exec_module(ex)
    monkeypatch.setattr(ex, "SessionFrames", _FakeFrames)
    out = str(tmp_path / "export")
    with pytest.raises(SystemExit, match="--female-n/--male-n"):
        ex.main(["--runs", str(tmp_path / "rec" / "pose_mvq_p3b"), "--out", out,
                 "--export-names-from", str(names_root), "--target", "4",
                 "--min-female", "1", "--per-rec-frac", "1.0", "--per-bout-cap", "4",
                 "--figures-dir", os.path.join(out, "fig")])


# ------------------------------------------------------------------ link-cameras
def test_link_cameras_makes_one_named_symlink_per_camera(tmp_path):
    """Clip's mp4s are `Cam..._frames_<a>_<b>.mp4`; every reader here resolves
    `<session_dir>/<Cam>.mp4`. The calibration is linked too, so the link dir
    works as a session dir end to end."""
    m = _load()
    src = tmp_path / "clip"
    (src / "calibration").mkdir(parents=True)
    for c in CAMS:
        (src / f"{c}_frames_1161383_1162303.mp4").write_bytes(b"\x00")
    made = m.link_cameras(src, tmp_path / "out" / "_cams", CAMS)

    assert len(made) == len(CAMS) == 7
    for c in CAMS:
        p = tmp_path / "out" / "_cams" / f"{c}.mp4"
        assert p.is_symlink()
        assert os.path.realpath(p) == os.path.realpath(src / f"{c}_frames_1161383_1162303.mp4")
    assert (tmp_path / "out" / "_cams" / "calibration").is_symlink()
    # idempotent: a second call replaces the links rather than raising
    m.link_cameras(src, tmp_path / "out" / "_cams", CAMS)


def test_link_cameras_refuses_two_videos_for_one_camera(tmp_path):
    """Two spans of the same camera map to the same `<Cam>.mp4`. Picking the
    sorted-first one would give that camera a DIFFERENT stretch of time than
    its neighbours -- one level down from the camera-order trap, and just as
    invisible."""
    m = _load()
    src = tmp_path / "clip"
    src.mkdir()
    for c in CAMS:
        (src / f"{c}_frames_1_2.mp4").write_bytes(b"\x00")
    (src / f"{CAMS[3]}_frames_3_4.mp4").write_bytes(b"\x00")
    with pytest.raises(ValueError, match="videos map to camera"):
        m.link_cameras(src, tmp_path / "out" / "_cams", CAMS)


def test_link_cameras_refuses_a_camera_with_no_video(tmp_path):
    m = _load()
    src = tmp_path / "clip"
    src.mkdir()
    for c in CAMS[:-1]:
        (src / f"{c}_frames_1_2.mp4").write_bytes(b"\x00")
    with pytest.raises(ValueError, match="no video matching"):
        m.link_cameras(src, tmp_path / "out" / "_cams", CAMS)


# ------------------------------------------------------------------------- CLI
def test_parse_span_and_the_num_animals_guard(tmp_path):
    m = _load()
    assert m.parse_span("25000:27000") == (25000, 27000)
    for bad in ("25000", "27000:25000", "5:5"):
        with pytest.raises(ValueError):
            m.parse_span(bad)
    with pytest.raises(SystemExit, match="single-fly"):
        m.main(["--session-dir", str(tmp_path), "--num-animals", "2",
                "--bout", "0:10", "--run", "x", "--centerdetect", "y",
                "--out", str(tmp_path / "o")])


def test_a_current_bout_is_skipped_and_force_reruns(tmp_path):
    """A re-run must cost nothing, and must NOT read a bout as current when
    the meta (written last) is missing -- that is a bout killed between the
    npz and the meta."""
    m = _load()
    _res, out_dir, _ = _run(m, tmp_path, T=6)
    gates = json.loads(Path(out_dir, "mvq_meta.json").read_text())
    gates_string = json.dumps(gates["gates"], sort_keys=True)
    assert m._bout_is_current(out_dir, gates_string)
    os.remove(os.path.join(out_dir, "mvq_meta.json"))
    assert not m._bout_is_current(out_dir, gates_string)
    assert not m._bout_is_current(out_dir, '{"lifter": "other"}')
