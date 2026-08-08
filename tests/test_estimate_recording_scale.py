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
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = REPO_ROOT / "models" / "fruitfly_v1" / "fruitfly_v1_free.xml"
TRUNK = ["Scutellum", "WingL_base", "WingR_base", "Abd_A4", "Abd_tip"]


def _require_model():
    if not MODEL_XML.exists():
        pytest.skip(f"model xml not found: {MODEL_XML}")


def _ref_trunk_positions():
    """Real model rest-pose positions for the DEFAULT_TRUNK_KEYPOINTS sites."""
    import mujoco

    mj = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    d = mujoco.MjData(mj)
    mujoco.mj_forward(mj, d)
    site_idx = {}
    for i in range(mj.nsite):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SITE, i)
        if name and name.startswith("tracking[") and name.endswith("]"):
            site_idx[name[len("tracking["):-1]] = i
    return np.array([d.site_xpos[site_idx[n]] for n in TRUNK])


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
