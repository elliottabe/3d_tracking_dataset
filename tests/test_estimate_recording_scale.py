"""TDD tests for scripts/estimate_recording_scale.py.

Regression coverage for the scale-from-first-bout defect (see
docs/.../scale-from-first-bout-defect and Task 12): scripts/run_bout.py used
to fit ONE body scale from whichever bout/fly got there first and apply it to
every bout and both flies. Measured on real frozen data, one bad bout-fly gave
a scale 38x too small (0.000338 vs a production value of 0.010994) because
only 68% of that bout-fly's frames had all 5 trunk markers finite -- the
temporal filter's gap-fill inflated the trunk point cloud. `robust_scale`'s
outlier rejection is the direct regression test for that defect.

All tests are synthetic (no GPU, no real recordings) except the model-based
tests, which load the real (small, checked-in) MuJoCo XML and are skipped if
it is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from scripts.estimate_recording_scale import (
    bout_kp3d_paths,
    per_frame_scales,
    robust_scale,
    coincident_fraction,
    estimate_run_root,
    rigid_segment_pairs,
    per_bout_segment_scale,
    segment_scale_diagnostics,
    warn_if_estimator_ignored,
    WITHIN_BONE_CV_WARN_THRESH,
    implied_body_length_mm,
    assert_plausible_body_scale,
    WORLD_UNITS_TO_MM,
    BODY_LENGTH_MM_MIN,
    BODY_LENGTH_MM_MAX,
    BODY_LENGTH_REF_PAIR,
    main as estimate_scale_main,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = REPO_ROOT / "models" / "fruitfly_v1" / "fruitfly_v1_free.xml"
TRUNK = ["Scutellum", "WingL_base", "WingR_base", "Abd_A4", "Abd_tip"]


def _require_model():
    if not MODEL_XML.exists():
        pytest.skip(f"model xml not found: {MODEL_XML}")


def _ref_trunk_positions():
    """Real model rest-pose positions for the DEFAULT_TRUNK_KEYPOINTS sites."""
    return _ref_positions_for(TRUNK)


def _site_idx():
    import mujoco

    mj = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    d = mujoco.MjData(mj)
    mujoco.mj_forward(mj, d)
    site_idx = {}
    for i in range(mj.nsite):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SITE, i)
        if name and name.startswith("tracking[") and name.endswith("]"):
            site_idx[name[len("tracking["):-1]] = i
    return site_idx, d


def _ref_positions_for(names):
    """Real model rest-pose positions for an arbitrary list of tracking-site names."""
    site_idx, d = _site_idx()
    return np.array([d.site_xpos[site_idx[n]] for n in names])


def _random_rotation(rng):
    """A random proper rotation matrix (det=+1), via QR of a random normal matrix."""
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


# A real (checked-in-model) subset of leg keypoints spanning 2 legs, one with
# a ThxCx tracking site (T1L) and one without (T2L, matching the real v1
# model -- see rigid_segment_pairs "silently drops" behaviour). Used for
# per_bout_segment_scale / segment_scale_diagnostics tests that need real
# model rest-pose geometry (rotation/translation invariance, bending, etc.).
LEG_SUBSET = [
    "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1",
    "T2L_Tro", "T2L_FeTi", "T2L_TiTa", "T2L_TaT1",
]

# A hypothetical "full" leg keypoint set (all 6 legs, all 4 chain segments,
# including ThxCx for every leg) used only to test rigid_segment_pairs'
# pure name-matching logic -- it does not need to match the real (incomplete)
# v1 model, since rigid_segment_pairs takes no model argument.
FULL_LEG_NAMES = [
    f"{leg}_{joint}"
    for leg in ("T1L", "T1R", "T2L", "T2R", "T3L", "T3R")
    for joint in ("ThxCx", "Tro", "FeTi", "TiTa", "TaT1")
]
THORAX_NAMES = ["WingL_base", "WingR_base", "Scutellum"]


# ---------------------------------------------------------------------------
# per_frame_scales
# ---------------------------------------------------------------------------

def test_per_frame_scales_recovers_known_scale():
    _require_model()
    ref = _ref_trunk_positions()
    known_scale = 0.011
    # data -> model scale is known_scale, i.e. data = ref / known_scale.
    data_frame = ref / known_scale
    kp3d = np.repeat(data_frame[None], 10, axis=0)  # (10, 5, 3)

    scales = per_frame_scales(kp3d, TRUNK, str(MODEL_XML),
                              trunk_names=TRUNK, estimator="umeyama")

    assert scales.shape == (10,)
    assert np.allclose(scales, known_scale, rtol=0.01)


def test_per_frame_scales_drops_nan_frames_instead_of_propagating():
    _require_model()
    ref = _ref_trunk_positions()
    known_scale = 0.011
    good = ref / known_scale
    bad = good.copy()
    bad[0] = np.nan  # one marker missing this frame

    kp3d = np.stack([good, bad, good, bad, good], axis=0)  # (5, 5, 3)

    scales = per_frame_scales(kp3d, TRUNK, str(MODEL_XML),
                              trunk_names=TRUNK, estimator="umeyama")

    # Only the 3 good frames should survive -- NaNs dropped, not propagated
    # as NaN entries in the output.
    assert scales.shape == (3,)
    assert np.all(np.isfinite(scales))
    assert np.allclose(scales, known_scale, rtol=0.01)


# ---------------------------------------------------------------------------
# robust_scale
# ---------------------------------------------------------------------------

def test_robust_scale_rejects_pathological_bout_regression():
    """Regression test for the measured 38x scale-from-first-bout defect.

    Three bouts cluster near the real production scale (~0.011); one bout is
    pathological (~0.0003, mirroring the real 38x-too-small bout22/fly0
    measurement). The pooled estimate must land near the good cluster and
    must flag the bad bout -- NOT be dragged toward it.
    """
    rng = np.random.default_rng(0)
    good_a = 0.0108 + rng.normal(0, 1e-5, size=30)
    good_b = 0.0110 + rng.normal(0, 1e-5, size=30)
    good_c = 0.0112 + rng.normal(0, 1e-5, size=30)
    bad = 0.0003 + rng.normal(0, 1e-6, size=30)

    per_bout = {0: good_a, 1: good_b, 2: good_c, 22: bad}

    result = robust_scale(per_bout)

    assert result["outlier_bouts"] == [22]
    # Must not be dragged >2% from the good-cluster true value.
    true_good = np.median(np.concatenate([good_a, good_b, good_c]))
    assert abs(result["scale"] - true_good) / true_good < 0.02
    assert result["n_bouts"] == 4
    assert result["n_frames"] == 120


def test_robust_scale_all_bouts_equal_has_no_outliers_and_zero_spread():
    per_bout = {0: np.full(20, 0.011), 1: np.full(20, 0.011), 2: np.full(20, 0.011)}

    result = robust_scale(per_bout)

    assert result["outlier_bouts"] == []
    assert result["spread_pct"] == pytest.approx(0.0, abs=1e-9)
    assert result["scale"] == pytest.approx(0.011)


def test_robust_scale_raises_on_no_usable_frames():
    per_bout = {0: np.array([np.nan, np.nan]), 1: np.array([-1.0, 0.0])}
    with pytest.raises(ValueError):
        robust_scale(per_bout)


# ---------------------------------------------------------------------------
# bout_kp3d_paths
# ---------------------------------------------------------------------------

def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00")


def test_bout_kp3d_paths_prefers_filt_falls_back_and_sorts(tmp_path):
    run_root = tmp_path / "run"
    # bout 2: only raw kp3d.npz
    _touch(run_root / "bouts" / "bout_00002" / "fly0" / "kp3d.npz")
    # bout 1: both present -> prefer filt
    _touch(run_root / "bouts" / "bout_00001" / "fly0" / "kp3d.npz")
    _touch(run_root / "bouts" / "bout_00001" / "fly0" / "kp3d_filt.npz")
    # bout 10: only filt
    _touch(run_root / "bouts" / "bout_00010" / "fly0" / "kp3d_filt.npz")

    paths = bout_kp3d_paths(run_root, fly=0)

    assert [p.name for p in paths] == ["kp3d_filt.npz", "kp3d.npz", "kp3d_filt.npz"]
    # sorted by bout index: 1, 2, 10
    assert [p.parent.parent.name for p in paths] == [
        "bout_00001", "bout_00002", "bout_00010"]


# ---------------------------------------------------------------------------
# estimate_run_root
# ---------------------------------------------------------------------------

def _cfg():
    return OmegaConf.create({
        "model": {"KP_NAMES": list(TRUNK)},
        "mjcf_path": str(MODEL_XML),
    })


def _write_sex_json(bout_dir: Path, male_fly: int):
    bout_dir.mkdir(parents=True, exist_ok=True)
    (bout_dir / "sex.json").write_text(json.dumps({"male_fly": male_fly}))


def _write_sex_json_raw(bout_dir: Path, obj: dict):
    """Like _write_sex_json but writes an arbitrary dict (e.g. missing the
    male_fly key entirely, or with an unusable value)."""
    bout_dir.mkdir(parents=True, exist_ok=True)
    (bout_dir / "sex.json").write_text(json.dumps(obj))


def _write_kp3d(fly_dir: Path, kp3d: np.ndarray):
    fly_dir.mkdir(parents=True, exist_ok=True)
    np.savez(fly_dir / "kp3d.npz", kp3d=kp3d, conf3d=np.ones(kp3d.shape[:2]))


_FLY_SEPARATION = np.array([10.0, 0.0, 0.0])  # ~1.3 body-lengths -- two distinct flies


def _synthetic_run_root(tmp_path, n_bouts=3, fly0_scale=0.011, fly1_scale=0.0119,
                        canonical=True, disagree=False, missing_one=False,
                        duplicate_bout=None):
    """duplicate_bout: if set, that bout index (1-based) gets fly1 kp3d ~= fly0
    kp3d (no separation offset) to simulate a SAM3 both-slots-on-one-animal bug."""
    _require_model()
    ref = _ref_trunk_positions()
    run_root = tmp_path / "run"
    rng = np.random.default_rng(1)
    for i in range(1, n_bouts + 1):
        bout_dir = run_root / "bouts" / f"bout_{i:05d}"
        f0 = ref / fly0_scale + rng.normal(0, 1e-6, size=ref.shape)
        if duplicate_bout is not None and i == duplicate_bout:
            f1 = f0 + rng.normal(0, 1e-7, size=ref.shape)  # ~= same animal
        else:
            f1 = (ref / fly1_scale + _FLY_SEPARATION
                  + rng.normal(0, 1e-6, size=ref.shape))
        kp3d0 = np.repeat(f0[None], 15, axis=0)
        kp3d1 = np.repeat(f1[None], 15, axis=0)
        _write_kp3d(bout_dir / "fly0", kp3d0)
        _write_kp3d(bout_dir / "fly1", kp3d1)
        if canonical:
            if missing_one and i == n_bouts:
                continue  # last bout has no sex.json
            male_fly = 1 if not disagree else (1 if i % 2 == 0 else 0)
            _write_sex_json(bout_dir, male_fly)
    return run_root


def test_estimate_run_root_two_flies_canonical(tmp_path):
    run_root = _synthetic_run_root(tmp_path, canonical=True)
    cfg = _cfg()

    result = estimate_run_root(run_root, cfg, estimator="umeyama",
                               scale_keypoints="trunk")

    assert result["identity"] == "canonical"
    assert result["duplicate_slot_bouts"] == []
    assert set(result["scale_by_fly"].keys()) == {"0", "1"}
    expected = float(np.median(list(result["scale_by_fly"].values())))
    assert result["scale"] == pytest.approx(expected)
    assert "0" in result["diagnostics"] and "1" in result["diagnostics"]
    assert result["scale_by_fly"]["0"] == pytest.approx(0.011, rel=0.02)
    assert result["scale_by_fly"]["1"] == pytest.approx(0.0119, rel=0.02)


def test_estimate_run_root_missing_sex_json_is_unknown_identity(tmp_path):
    run_root = _synthetic_run_root(tmp_path, canonical=True, missing_one=True)
    cfg = _cfg()

    result = estimate_run_root(run_root, cfg)

    assert result["identity"] == "unknown"
    assert result["scale_by_fly"] is None
    assert isinstance(result["scale"], float)
    assert result["scale"] > 0


def test_estimate_run_root_disagreeing_sex_json_is_unknown_identity(tmp_path):
    run_root = _synthetic_run_root(tmp_path, canonical=True, disagree=True)
    cfg = _cfg()

    result = estimate_run_root(run_root, cfg)

    assert result["identity"] == "unknown"
    assert result["scale_by_fly"] is None


# ---------------------------------------------------------------------------
# _determine_identity must treat an unusable male_fly (missing key, null,
# wrong type, out-of-range value) exactly like a missing sex.json file --
# NOT as a value that can validly agree with itself across bouts. Regression
# for review finding: `json.load(f).get("male_fly")` yields None for a
# well-formed sex.json that simply lacks the key; {None} has length 1, so the
# old code silently returned identity="canonical" for a recording whose
# identity is not actually known, defeating the entire gate.
# ---------------------------------------------------------------------------

def _run_root_with_sex_objs(tmp_path, sex_objs):
    """Every bout gets well-separated fly0/fly1 kp3d (so the pooled/per-fly
    scale still computes cleanly) and the given raw sex.json content."""
    _require_model()
    ref = _ref_trunk_positions()
    run_root = tmp_path / "run"
    rng = np.random.default_rng(7)
    for i, obj in enumerate(sex_objs, start=1):
        bout_dir = run_root / "bouts" / f"bout_{i:05d}"
        f0 = ref / 0.011 + rng.normal(0, 1e-6, size=ref.shape)
        f1 = ref / 0.0119 + _FLY_SEPARATION + rng.normal(0, 1e-6, size=ref.shape)
        _write_kp3d(bout_dir / "fly0", np.repeat(f0[None], 15, axis=0))
        _write_kp3d(bout_dir / "fly1", np.repeat(f1[None], 15, axis=0))
        _write_sex_json_raw(bout_dir, obj)
    return run_root


def test_estimate_run_root_sex_json_missing_male_fly_key_is_unknown(tmp_path):
    run_root = _run_root_with_sex_objs(
        tmp_path, [{"confidence": "user"}, {"confidence": "user"}, {"confidence": "user"}])
    cfg = _cfg()

    result = estimate_run_root(run_root, cfg)

    assert result["identity"] == "unknown"
    assert "usable male_fly" in result["identity_reason"]
    assert result["scale_by_fly"] is None
    assert isinstance(result["scale"], float) and result["scale"] > 0


def test_estimate_run_root_sex_json_null_male_fly_is_unknown(tmp_path):
    run_root = _run_root_with_sex_objs(
        tmp_path, [{"male_fly": None}, {"male_fly": None}, {"male_fly": None}])
    cfg = _cfg()

    result = estimate_run_root(run_root, cfg)

    assert result["identity"] == "unknown"
    assert "usable male_fly" in result["identity_reason"]
    assert result["scale_by_fly"] is None


def test_estimate_run_root_sex_json_string_male_fly_is_unknown(tmp_path):
    # One bout's male_fly is a string, not an int -- even though the other
    # two bouts are proper agreeing ints, the whole recording must be
    # "unknown" (a mixed-type sex.json is itself a sign something is wrong).
    run_root = _run_root_with_sex_objs(
        tmp_path, [{"male_fly": 1}, {"male_fly": "1"}, {"male_fly": 1}])
    cfg = _cfg()

    result = estimate_run_root(run_root, cfg)

    assert result["identity"] == "unknown"
    assert result["scale_by_fly"] is None


def test_estimate_run_root_sex_json_out_of_range_male_fly_is_unknown(tmp_path):
    run_root = _run_root_with_sex_objs(
        tmp_path, [{"male_fly": 1}, {"male_fly": 2}, {"male_fly": 1}])
    cfg = _cfg()

    result = estimate_run_root(run_root, cfg)

    assert result["identity"] == "unknown"
    assert result["scale_by_fly"] is None


def test_estimate_run_root_all_usable_ints_agreeing_stays_canonical(tmp_path):
    """Non-regression: the ordinary well-formed case must still work."""
    run_root = _run_root_with_sex_objs(
        tmp_path, [{"male_fly": 1}, {"male_fly": 1}, {"male_fly": 1}])
    cfg = _cfg()

    result = estimate_run_root(run_root, cfg)

    assert result["identity"] == "canonical"
    assert result["scale_by_fly"] is not None
    assert set(result["scale_by_fly"]) == {"0", "1"}


# ---------------------------------------------------------------------------
# coincident_fraction / duplicate-slot bout exclusion
#
# Regression coverage for the SAM3 both-slots-lock-onto-one-animal failure
# mode: S0 bout22/fly0 -- exactly the bout-fly that produced the 38x
# scale-from-first-bout defect -- also has its slot >20% coincident with the
# OTHER slot across the recording. MAD outlier rejection alone would not
# catch this: a duplicated slot holds a real (just wrong) fly, so its scale
# can look perfectly ordinary.
# ---------------------------------------------------------------------------

def test_coincident_fraction_low_for_well_separated_flies():
    base = np.zeros((10, 5, 3))
    base[:, :, :] = np.arange(5)[None, :, None]  # some nonzero spread per fly
    a = base
    b = base + np.array([5.0, 0.0, 0.0])  # far away relative to body spread

    frac = coincident_fraction(a, b)

    assert frac < 0.05


def test_coincident_fraction_high_when_slots_are_the_same_animal():
    rng = np.random.default_rng(2)
    base = np.zeros((10, 5, 3))
    base[:, :, :] = np.arange(5)[None, :, None]
    a = base
    b = base + rng.normal(0, 1e-4, size=base.shape)  # ~same animal, tiny noise

    frac = coincident_fraction(a, b)

    assert frac > 0.95


def test_coincident_fraction_empty_or_all_nan_returns_zero():
    empty = np.zeros((0, 5, 3))
    all_nan = np.full((10, 5, 3), np.nan)
    some = np.zeros((10, 5, 3))

    assert coincident_fraction(empty, some) == 0.0
    assert coincident_fraction(some, empty) == 0.0
    assert coincident_fraction(all_nan, some) == 0.0
    assert coincident_fraction(some, all_nan) == 0.0


def test_estimate_run_root_excludes_duplicate_slot_bout(tmp_path):
    run_root = _synthetic_run_root(tmp_path, n_bouts=3, canonical=True,
                                   duplicate_bout=2)
    cfg = _cfg()

    result = estimate_run_root(run_root, cfg)

    assert result["duplicate_slot_bouts"] == [2]
    # Both slots of bout 2 must be gone from every fly's per-bout diagnostics.
    assert 2 not in result["diagnostics"]["0"]["per_bout_median"]
    assert 2 not in result["diagnostics"]["1"]["per_bout_median"]
    # Scale still recovers close to the true per-fly values from bouts 1 & 3.
    assert result["scale_by_fly"]["0"] == pytest.approx(0.011, rel=0.02)
    assert result["scale_by_fly"]["1"] == pytest.approx(0.0119, rel=0.02)


def test_estimate_run_root_single_fly_bout_never_flagged_duplicate(tmp_path):
    _require_model()
    ref = _ref_trunk_positions()
    run_root = tmp_path / "run"
    rng = np.random.default_rng(3)
    for i in (1, 2):
        bout_dir = run_root / "bouts" / f"bout_{i:05d}"
        f0 = ref / 0.011 + rng.normal(0, 1e-6, size=ref.shape)
        _write_kp3d(bout_dir / "fly0", np.repeat(f0[None], 15, axis=0))
        if i == 1:
            f1 = ref / 0.0119 + _FLY_SEPARATION + rng.normal(0, 1e-6, size=ref.shape)
            _write_kp3d(bout_dir / "fly1", np.repeat(f1[None], 15, axis=0))
        # bout 2 has only fly0 -- must never be flagged as duplicate-slot.
        _write_sex_json(bout_dir, 1)

    cfg = _cfg()
    result = estimate_run_root(run_root, cfg)

    assert result["duplicate_slot_bouts"] == []
    assert 2 in result["diagnostics"]["0"]["per_bout_median"]


# ---------------------------------------------------------------------------
# rigid_segment_pairs
# ---------------------------------------------------------------------------

def test_rigid_segment_pairs_full_v1_yields_24_leg_pairs_no_thorax_by_default():
    pairs = rigid_segment_pairs(FULL_LEG_NAMES)

    assert len(pairs) == 24
    assert ("T1L_ThxCx", "T1L_Tro") in pairs
    assert ("T1L_Tro", "T1L_FeTi") in pairs
    assert ("T1L_FeTi", "T1L_TiTa") in pairs
    assert ("T1L_TiTa", "T1L_TaT1") in pairs
    # 4 pairs/leg x 6 legs, and nothing beyond TaT1 (no TaT1->TaT3 etc.).
    assert not any("TaT3" in a or "TaT3" in b or "TaTip" in a or "TaTip" in b
                  for a, b in pairs)


def test_rigid_segment_pairs_include_thorax_adds_3():
    names = FULL_LEG_NAMES + THORAX_NAMES

    without = rigid_segment_pairs(names, include_thorax=False)
    with_thorax = rigid_segment_pairs(names, include_thorax=True)

    assert len(without) == 24
    assert len(with_thorax) == 27
    added = set(with_thorax) - set(without)
    assert added == {
        ("WingL_base", "WingR_base"),
        ("Scutellum", "WingL_base"),
        ("Scutellum", "WingR_base"),
    }


def test_rigid_segment_pairs_silently_drops_missing_names():
    """The real v1 model only has a ThxCx tracking site for T1L/T1R -- T2/T3
    legs must silently lose their ThxCx->Tro pair, not error."""
    names = [
        "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1",
        "T2L_Tro", "T2L_FeTi", "T2L_TiTa", "T2L_TaT1",  # no T2L_ThxCx
    ]

    pairs = rigid_segment_pairs(names)

    assert ("T1L_ThxCx", "T1L_Tro") in pairs
    assert not any(a == "T2L_ThxCx" or b == "T2L_ThxCx" for a, b in pairs)
    # T1L: ThxCx->Tro, Tro->FeTi, FeTi->TiTa, TiTa->TaT1 (4)
    # T2L: Tro->FeTi, FeTi->TiTa, TiTa->TaT1 (3, no ThxCx->Tro)
    assert len(pairs) == 7


def test_rigid_segment_pairs_against_real_anatomy_config_drops_t2_t3_thxcx():
    """Non-regression against the actual checked-in anatomy config: T2/T3 legs
    have no ThxCx keypoint, so the real v1 KP_NAMES set yields 20 leg pairs,
    not the hypothetical full 24."""
    anatomy_path = REPO_ROOT / "configs" / "anatomy" / "v1.yaml"
    if not anatomy_path.exists():
        pytest.skip(f"anatomy config not found: {anatomy_path}")
    raw = OmegaConf.load(str(anatomy_path))
    kp_names = list(raw.model.KP_NAMES)

    pairs = rigid_segment_pairs(kp_names)

    assert len(pairs) == 20
    for leg in ("T2L", "T2R", "T3L", "T3R"):
        assert not any(a == f"{leg}_ThxCx" or b == f"{leg}_ThxCx" for a, b in pairs)
    for leg in ("T1L", "T1R"):
        assert (f"{leg}_ThxCx", f"{leg}_Tro") in pairs


# ---------------------------------------------------------------------------
# per_bout_segment_scale
# ---------------------------------------------------------------------------

def test_per_bout_segment_scale_recovers_known_scale_under_rotation_translation():
    """Pose (per-frame rotation + translation) must not matter -- this is the
    entire point of a rigid-segment measurement vs. a cloud-spread fit."""
    _require_model()
    ref = _ref_positions_for(LEG_SUBSET)
    known_scale = 0.0117
    rest_data = ref / known_scale
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(24):
        R = _random_rotation(rng)
        t = rng.normal(0, 50, size=3)
        frames.append(rest_data @ R.T + t)
    kp3d = np.stack(frames, axis=0)

    scales = per_bout_segment_scale(kp3d, LEG_SUBSET, str(MODEL_XML))

    assert scales.shape == (len(rigid_segment_pairs(LEG_SUBSET)),)
    assert np.all(np.isfinite(scales))
    assert np.allclose(scales, known_scale, rtol=0.01)


def test_per_bout_segment_scale_bending_isolates_only_that_pair():
    """Moving one keypoint along its own segment's axis changes ONLY that
    pair's implied scale; every other pair is untouched."""
    _require_model()
    ref = _ref_positions_for(LEG_SUBSET)
    known_scale = 0.0117
    rest_data = ref / known_scale
    n_frames = 20

    baseline = np.repeat(rest_data[None], n_frames, axis=0).copy()
    bent = baseline.copy()

    i_taT1 = LEG_SUBSET.index("T2L_TaT1")
    i_tita = LEG_SUBSET.index("T2L_TiTa")
    axis_dir = rest_data[i_taT1] - rest_data[i_tita]
    bent[:10, i_taT1] = rest_data[i_tita] + axis_dir * 1.4  # bend in half the frames

    pairs = rigid_segment_pairs(LEG_SUBSET)
    changed = pairs.index(("T2L_TiTa", "T2L_TaT1"))

    base_scales = per_bout_segment_scale(baseline, LEG_SUBSET, str(MODEL_XML))
    bent_scales = per_bout_segment_scale(bent, LEG_SUBSET, str(MODEL_XML))

    assert not np.isclose(bent_scales[changed], base_scales[changed], rtol=0.01)
    unchanged = [i for i in range(len(pairs)) if i != changed]
    assert np.allclose(bent_scales[unchanged], base_scales[unchanged], rtol=1e-6)


def test_per_bout_segment_scale_drops_nan_frames_and_unusable_pairs():
    _require_model()
    ref = _ref_positions_for(LEG_SUBSET)
    known_scale = 0.0117
    good = ref / known_scale
    kp3d = np.repeat(good[None], 5, axis=0).copy()

    # T1L_ThxCx is NaN in every frame -> its one pair (ThxCx->Tro) is dropped.
    i_thxcx = LEG_SUBSET.index("T1L_ThxCx")
    kp3d[:, i_thxcx, :] = np.nan
    # T2L_TaT1 is NaN in only one frame -> its pair survives on the rest.
    i_taT1 = LEG_SUBSET.index("T2L_TaT1")
    kp3d[0, i_taT1, :] = np.nan

    scales = per_bout_segment_scale(kp3d, LEG_SUBSET, str(MODEL_XML))

    all_pairs = rigid_segment_pairs(LEG_SUBSET)
    assert scales.shape == (len(all_pairs) - 1,)
    assert np.all(np.isfinite(scales))
    assert np.allclose(scales, known_scale, rtol=0.01)


def test_per_bout_segment_scale_raises_when_all_pairs_unusable():
    _require_model()
    kp3d = np.full((5, len(LEG_SUBSET), 3), np.nan)

    with pytest.raises(ValueError):
        per_bout_segment_scale(kp3d, LEG_SUBSET, str(MODEL_XML))


# ---------------------------------------------------------------------------
# segment_scale_diagnostics -- physics-based keypoint-quality metrics.
#
# A bone cannot change length ("within_bone_cv"); different bones must agree
# on one body size ("across_bone_cv"). Measured on real data (Session0, 4
# bouts): male within-bone CV 3.6-6.2%, across-bone CV 6.0-6.7%; female
# (known-bad keypoints) within-bone CV 20-50%, across-bone CV 29-57%. The two
# signals are independent: jitter (noise) breaks rigidity without breaking
# agreement between bones; bias (a systematically wrong length) breaks
# agreement without breaking rigidity.
# ---------------------------------------------------------------------------

def test_segment_scale_diagnostics_clean_rigid_data_has_near_zero_cvs():
    _require_model()
    ref = _ref_positions_for(LEG_SUBSET)
    known_scale = 0.0117
    rest_data = ref / known_scale
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(30):
        R = _random_rotation(rng)
        t = rng.normal(0, 50, size=3)
        frames.append(rest_data @ R.T + t)
    kp3d = np.stack(frames, axis=0)

    diag = segment_scale_diagnostics(kp3d, LEG_SUBSET, str(MODEL_XML))

    assert diag["within_bone_cv"] < 1e-6
    assert diag["across_bone_cv"] < 1e-6
    assert diag["n_pairs_used"] == len(rigid_segment_pairs(LEG_SUBSET))
    assert diag["scale"] == pytest.approx(known_scale, rel=0.01)


def test_segment_scale_diagnostics_per_frame_jitter_raises_within_not_across():
    """Per-frame (zero-mean) jitter on one pair breaks rigidity but the
    median-based implied scale for that pair is unaffected, so agreement
    between bones (across_bone_cv) stays low."""
    _require_model()
    ref = _ref_positions_for(LEG_SUBSET)
    known_scale = 0.0117
    rest_data = ref / known_scale
    n_frames = 400
    rng = np.random.default_rng(2)
    kp3d = np.repeat(rest_data[None], n_frames, axis=0).copy()

    i_taT1 = LEG_SUBSET.index("T2L_TaT1")
    jitter = rng.normal(0, rest_data.std() * 0.15, size=(n_frames, 3))
    kp3d[:, i_taT1, :] += jitter

    clean = segment_scale_diagnostics(rest_data[None].repeat(n_frames, axis=0),
                                      LEG_SUBSET, str(MODEL_XML))
    jittered = segment_scale_diagnostics(kp3d, LEG_SUBSET, str(MODEL_XML))

    assert jittered["within_bone_cv"] > clean["within_bone_cv"] + 0.01
    assert jittered["across_bone_cv"] < 0.05


def test_segment_scale_diagnostics_systematic_bias_raises_across_not_within():
    """A deterministic (per-frame constant) wrong length on one pair breaks
    agreement between bones but leaves that pair (and every other pair)
    perfectly rigid across frames."""
    _require_model()
    ref = _ref_positions_for(LEG_SUBSET)
    known_scale = 0.0117
    rest_data = ref / known_scale
    n_frames = 20
    kp3d = np.repeat(rest_data[None], n_frames, axis=0).copy()

    i_taT1 = LEG_SUBSET.index("T2L_TaT1")
    i_tita = LEG_SUBSET.index("T2L_TiTa")
    axis_dir = rest_data[i_taT1] - rest_data[i_tita]
    kp3d[:, i_taT1] = rest_data[i_tita] + axis_dir * 1.3  # constant across ALL frames

    clean = segment_scale_diagnostics(np.repeat(rest_data[None], n_frames, axis=0),
                                      LEG_SUBSET, str(MODEL_XML))
    biased = segment_scale_diagnostics(kp3d, LEG_SUBSET, str(MODEL_XML))

    assert biased["across_bone_cv"] > clean["across_bone_cv"] + 0.01
    assert biased["within_bone_cv"] < 1e-6


def test_segment_scale_diagnostics_raises_when_all_pairs_unusable():
    _require_model()
    kp3d = np.full((5, len(LEG_SUBSET), 3), np.nan)
    with pytest.raises(ValueError):
        segment_scale_diagnostics(kp3d, LEG_SUBSET, str(MODEL_XML))


# ---------------------------------------------------------------------------
# estimate_run_root(..., scale_keypoints="rigid_segment")
# ---------------------------------------------------------------------------

def _full_kp_names_and_ref():
    anatomy_path = REPO_ROOT / "configs" / "anatomy" / "v1.yaml"
    if not anatomy_path.exists() or not MODEL_XML.exists():
        pytest.skip("anatomy config or model xml not found")
    raw = OmegaConf.load(str(anatomy_path))
    kp_names = list(raw.model.KP_NAMES)
    ref = _ref_positions_for(kp_names)
    return kp_names, ref


def _synthetic_full_run_root(tmp_path, kp_names, ref, *, n_bouts=3,
                             fly0_scale=0.011, fly1_scale=0.0119,
                             fly0_leg_jitter=0.0, fly1_leg_jitter=0.0):
    """Like _synthetic_run_root but over the FULL v1 keypoint set (so
    rigid_segment mode has leg pairs to work with), with optional per-fly
    per-frame jitter injected into leg keypoints only (to simulate bad
    female-like keypoints for the within_bone_cv warning test)."""
    run_root = tmp_path / "run"
    rng = np.random.default_rng(11)
    leg_mask = np.array(["T1" in n or "T2" in n or "T3" in n for n in kp_names])
    for i in range(1, n_bouts + 1):
        bout_dir = run_root / "bouts" / f"bout_{i:05d}"
        f0 = ref / fly0_scale + rng.normal(0, 1e-6, size=ref.shape)
        f1 = ref / fly1_scale + _FLY_SEPARATION + rng.normal(0, 1e-6, size=ref.shape)
        kp3d0 = np.repeat(f0[None], 40, axis=0)
        kp3d1 = np.repeat(f1[None], 40, axis=0)
        if fly0_leg_jitter:
            kp3d0 = kp3d0.copy()
            kp3d0[:, leg_mask, :] += rng.normal(0, fly0_leg_jitter,
                                                size=kp3d0[:, leg_mask, :].shape)
        if fly1_leg_jitter:
            kp3d1 = kp3d1.copy()
            kp3d1[:, leg_mask, :] += rng.normal(0, fly1_leg_jitter,
                                                size=kp3d1[:, leg_mask, :].shape)
        _write_kp3d(bout_dir / "fly0", kp3d0)
        _write_kp3d(bout_dir / "fly1", kp3d1)
        _write_sex_json(bout_dir, 1)
    return run_root


def test_estimate_run_root_rigid_segment_mode_same_key_structure(tmp_path):
    kp_names, ref = _full_kp_names_and_ref()
    run_root = _synthetic_full_run_root(tmp_path, kp_names, ref)
    cfg = OmegaConf.create({"model": {"KP_NAMES": kp_names}, "mjcf_path": str(MODEL_XML)})

    # This fixture's near-zero genuine cross-bout spread (only 1e-6 synthetic
    # noise) is known to trip the all-bouts-MAD-outlier fallback -- expected,
    # see test_estimate_run_root_warns_when_mad_flags_all_bouts below.
    with pytest.warns(UserWarning, match="ALL"):
        result = estimate_run_root(run_root, cfg, scale_keypoints="rigid_segment")

    assert result["identity"] == "canonical"
    assert result["duplicate_slot_bouts"] == []
    assert set(result["scale_by_fly"].keys()) == {"0", "1"}
    assert result["scale_by_fly"]["0"] == pytest.approx(0.011, rel=0.02)
    assert result["scale_by_fly"]["1"] == pytest.approx(0.0119, rel=0.02)
    n_pairs_per_bout = len(rigid_segment_pairs(kp_names))
    for fly in ("0", "1"):
        diag = result["diagnostics"][fly]
        for key in ("scale", "n_bouts", "n_frames", "per_bout_median",
                    "outlier_bouts", "spread_pct", "scale_cv_across_bouts",
                    "within_bone_cv", "across_bone_cv", "n_pairs_used", "n_pairs"):
            assert key in diag, f"missing diagnostic key {key!r}"
        assert diag["within_bone_cv"] < 0.05
        # n_pairs is an alias of n_frames (robust_scale's generic key is a
        # per-PAIR sample count here, not a frame count -- see its docstring).
        assert diag["n_pairs"] == diag["n_frames"]
        # Pin against a known count: n_bouts_used bouts x n_pairs_per_bout.
        assert diag["n_pairs"] == n_pairs_per_bout * diag["n_bouts"]


def test_estimate_run_root_warns_when_mad_flags_all_bouts(tmp_path):
    """Regression: near-zero genuine cross-bout scale spread (here, only
    1e-6 synthetic per-bout noise) can make MAD flag EVERY bout an outlier --
    robust_scale falls back to pooling all bouts unfiltered, and must warn
    that this happened rather than silently discarding every bout."""
    kp_names, ref = _full_kp_names_and_ref()
    run_root = _synthetic_full_run_root(tmp_path, kp_names, ref, n_bouts=3)
    cfg = OmegaConf.create({"model": {"KP_NAMES": kp_names}, "mjcf_path": str(MODEL_XML)})

    with pytest.warns(UserWarning, match="MAD outlier rejection flagged ALL"):
        estimate_run_root(run_root, cfg, scale_keypoints="rigid_segment")


def test_estimate_run_root_rigid_segment_warns_when_estimator_not_default(tmp_path):
    kp_names, ref = _full_kp_names_and_ref()
    run_root = _synthetic_full_run_root(tmp_path, kp_names, ref, n_bouts=1)
    cfg = OmegaConf.create({"model": {"KP_NAMES": kp_names}, "mjcf_path": str(MODEL_XML)})

    with pytest.warns(UserWarning, match="ignored"):
        estimate_run_root(run_root, cfg, scale_keypoints="rigid_segment",
                          estimator="norm_ratio")


def test_estimate_run_root_rigid_segment_flags_noisy_female_like_fly(tmp_path, capsys):
    """Integration/regression for the within_bone_cv warning: a fly with
    noisy (jittered) leg keypoints must trip WITHIN_BONE_CV_WARN_THRESH and
    print a WARNING naming it, while a clean fly must not."""
    kp_names, ref = _full_kp_names_and_ref()
    # 1.0 data-unit jitter on ~5-data-unit leg bones (median, at fly1's own
    # scale) reproduces the measured female-like within_bone_cv (~0.27, well
    # above WITHIN_BONE_CV_WARN_THRESH); fly0 is left clean (within_bone_cv ~0).
    run_root = _synthetic_full_run_root(
        tmp_path, kp_names, ref, n_bouts=2, fly0_leg_jitter=0.0, fly1_leg_jitter=1.0)

    estimate_scale_main([
        "--run-root", str(run_root),
        "--scale-keypoints", "rigid_segment",
        "--dry-run",
    ])
    out = capsys.readouterr().out

    assert "WARNING" in out
    # fly1 (the jittered one) must be the one flagged.
    warn_lines = [l for l in out.splitlines() if "WARNING" in l and "within_bone_cv" in l.lower()]
    assert any("fly1" in l for l in warn_lines) or "fly1" in out.split("WARNING")[1]


# ---------------------------------------------------------------------------
# Review fixes (round 2): honesty/diagnostics-layer items.
# ---------------------------------------------------------------------------

def test_warn_if_estimator_ignored_warns_on_non_default():
    with pytest.warns(UserWarning, match="ignored"):
        warn_if_estimator_ignored("norm_ratio", caller="some_caller")


def test_warn_if_estimator_ignored_silent_on_default():
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any warning here fails the test
        warn_if_estimator_ignored("umeyama", caller="some_caller")


def test_run_bout_rigid_segment_branch_shares_estimator_warning_and_never_echoes_it():
    """scripts/run_bout.py's rigid_segment scale.json branch is production
    code (not exercised by estimate_run_root at all) -- it must use the SAME
    warn_if_estimator_ignored as estimate_run_root (regression: an earlier
    version warned only in estimate_run_root/this module's CLI, leaving the
    actual production driver silent about a stale scaling.estimator
    override), and it must never write the raw (unused) configured estimator
    value into scale.json -- that would be indistinguishable from a run
    where it was actually applied."""
    src = (REPO_ROOT / "scripts" / "run_bout.py").read_text()

    assert "warn_if_estimator_ignored" in src, (
        "run_bout.py must call the shared warn_if_estimator_ignored, not "
        "duplicate/omit the estimator-ignored check inline")
    assert '"estimator": "ignored (rigid_segment)"' in src

    start = src.index('if _scale_keypoints_mode == "rigid_segment":')
    end = src.index("\n        else:", start)
    rigid_branch = src[start:end]
    assert "warn_if_estimator_ignored" in rigid_branch
    # It's fine (expected) to pass the configured value to the shared
    # checker; the regression is writing it back into scale.json's
    # "estimator" field, which would look identical to a run where the
    # estimator was actually used.
    assert '"estimator": str(cfg.scaling.estimator)' not in rigid_branch, (
        "must not echo the ignored, unused configured estimator into scale.json")


def test_run_bout_help_exits_0_after_estimator_warning_wiring():
    """Regression for the shared-import wiring: `python scripts/run_bout.py
    --help` must still succeed (module-level imports untouched by the
    estimator-ignored-warning refactor)."""
    import subprocess
    import sys as _sys
    r = subprocess.run([_sys.executable, "scripts/run_bout.py", "--help"],
                       capture_output=True, text=True, cwd=REPO_ROOT, timeout=240)
    assert "ModuleNotFoundError" not in r.stderr, r.stderr[-2000:]
    assert r.returncode == 0, r.stderr[-2000:]


# ---------------------------------------------------------------------------
# Physical sanity gate on `scale` (implied body length in mm) -- regression
# coverage for the "scale does double duty" gap: nothing previously checked
# a fitted scale's OWN physical plausibility, so a badly wrong scale (like
# the historical 38x scale-from-first-bout defect) looked dimensionally fine
# and was absorbed by marker offsets instead of failing.
# ---------------------------------------------------------------------------

def _known_good_scale():
    """A `scale` (model_units/data_units) that implies a body length in the
    middle of the plausible band, derived from the REAL model's own
    BODY_LENGTH_REF_PAIR rest-pose distance -- exact by construction, so the
    test does not depend on any particular recording's fitted scale."""
    ref = _ref_positions_for(list(BODY_LENGTH_REF_PAIR))
    model_length = float(np.linalg.norm(ref[0] - ref[1]))
    target_mm = (BODY_LENGTH_MM_MIN + BODY_LENGTH_MM_MAX) / 2.0
    target_data_units = target_mm / WORLD_UNITS_TO_MM
    return model_length / target_data_units, target_mm


def test_implied_body_length_mm_recovers_target_length():
    _require_model()
    scale, target_mm = _known_good_scale()
    mm = implied_body_length_mm(scale, str(MODEL_XML))
    assert mm == pytest.approx(target_mm, rel=1e-9)


def test_implied_body_length_mm_nan_for_nonpositive_scale():
    _require_model()
    assert np.isnan(implied_body_length_mm(0.0, str(MODEL_XML)))
    assert np.isnan(implied_body_length_mm(-1.0, str(MODEL_XML)))
    assert np.isnan(implied_body_length_mm(float("nan"), str(MODEL_XML)))


def test_assert_plausible_body_scale_passes_for_plausible_scale():
    """A scale implying a body length inside [2.0, 3.0] mm must return that
    length and raise nothing."""
    _require_model()
    scale, target_mm = _known_good_scale()
    mm = assert_plausible_body_scale(scale, str(MODEL_XML))
    assert mm == pytest.approx(target_mm, rel=1e-9)


def test_assert_plausible_body_scale_raises_on_historical_38x_defect():
    """The historical scale-from-first-bout defect: one bad bout-fly's scale
    was 38x too small (0.000338 vs a good 0.010994). A scale too small means
    `model_dist / scale` (the implied DATA-space body length) is inflated --
    this must raise, not warn, and the message must be actionable."""
    _require_model()
    good_scale, good_mm = _known_good_scale()
    bad_scale = good_scale / 38.0

    with pytest.raises(ValueError) as excinfo:
        assert_plausible_body_scale(bad_scale, str(MODEL_XML), context="bout22 fly0")

    msg = str(excinfo.value)
    assert "bout22 fly0" in msg
    assert f"{bad_scale!r}" in msg or "scale=" in msg
    implied_mm = good_mm * 38.0
    assert f"{implied_mm:.3f}" in msg or f"{implied_mm:.1f}" in msg[:msg.find("mm")]
    assert "2.0" in msg and "3.0" in msg


def test_assert_plausible_body_scale_raises_on_implausibly_small_body():
    """The failure mode is not one-directional: a scale too LARGE implies an
    implausibly SHORT body and must also raise."""
    _require_model()
    good_scale, _ = _known_good_scale()
    too_large_scale = good_scale * 10.0

    with pytest.raises(ValueError):
        assert_plausible_body_scale(too_large_scale, str(MODEL_XML))


def test_run_bout_scale_block_calls_the_shared_physical_sanity_gate():
    """scripts/run_bout.py's scale.json block (both rigid_segment and
    trunk/all branches funnel into the SAME `scale` local below) must call
    the shared assert_plausible_body_scale after resolving per-fly-vs-shared
    scale -- not duplicate the check inline, and not skip it for either
    branch."""
    src = (REPO_ROOT / "scripts" / "run_bout.py").read_text()
    assert "assert_plausible_body_scale" in src
    idx = src.index("scale = float(_scale_data[\"scale\"])")
    tail = src[idx:idx + 1500]
    assert "assert_plausible_body_scale(" in tail
