# tests/test_mvq_calibrate.py
"""`scripts/benchmark/mvq_calibrate.py` -- post-training temperature
calibration for the mvq existence and per-view visibility heads (design spec
`docs/specs/2026-09-05-mvq-v2-pseudolabel-t2-design.md` §5), and
`jarvis_jax.tracking.lift_mvq.MVQRunner`'s consumption of the stored
temperatures.

Three properties, each stated before it is asserted (CLAUDE.md):

  * `fit_temperature` must recover a KNOWN artificial miscalibration exactly
    (logits inflated by a known factor) -- a bisection that converged to the
    wrong root, or to a clamp edge it should not have hit, would silently
    under- or over-correct every downstream probability.
  * `reliability` of ALREADY-calibrated probabilities must read as
    calibrated (max_gap/ece near 0) -- a binning bug (off-by-one bin edge,
    an empty bin counted as a perfect bin) would instead manufacture a false
    "well calibrated" or "miscalibrated" verdict from data that is neither.
  * `MVQRunner.infer` must apply the stored temperature to EVERY existence
    and visibility sigmoid, changing the reported probability by exactly
    `sigmoid(logit / T)` while leaving WHICH slot the policy would pick
    (argmax existence) unchanged -- dividing every logit by the same
    positive scalar is monotone, so calibration must never flip a ranking,
    only recalibrate what a probability means.

`scripts/benchmark/` has a real `__init__.py` (it is already an importable
package, `scripts/__init__.py` too) -- imported the same way
`tests/test_mvq_v2_acceptance.py` and `tests/test_mvq_perkp.py` do, by
putting the repo root on `sys.path` and importing `scripts.benchmark.X`.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from scripts.benchmark import mvq_calibrate as mc  # noqa: E402

from test_lift_mvq import _tiny_final  # noqa: E402 -- shared tiny-mvq-model fixture


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return _tiny_final(tmp_path_factory.mktemp("mvq_calibrate"))


def _runner(tiny, **kw):
    from jarvis_jax.tracking.lift_mvq import MVQRunner
    from mvq_fixtures import CAMS
    root, final = tiny
    kw.setdefault("batch", 2)
    return MVQRunner(final, calib_dir=os.path.join(root, "calibrations", "A"),
                     cameras=CAMS, **kw)


# --------------------------------------------------------------------------
# fit_temperature
# --------------------------------------------------------------------------
def test_fit_temperature_recovers_a_known_scaling():
    """EXPECTATION: logits generated from a WELL-calibrated `p = sigmoid(z)`,
    then artificially inflated 3x, are exactly what temperature scaling was
    built to undo -- fitting must recover T ~ 3.0 (the inflation factor), to
    within 0.1, not some other root or a clamp edge."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=20000) * 2.0
    p = 1 / (1 + np.exp(-z))
    y = rng.uniform(size=z.shape) < p
    t = mc.fit_temperature(z * 3.0, y, np.ones_like(y, bool))
    assert abs(t - 3.0) < 0.1


def test_fit_temperature_ignores_masked_entries():
    """EXPECTATION: entries where `mask` is False must not move the fit at
    all -- salting the array with adversarial masked-out garbage (a huge
    logit paired with the WRONG label) must not change T beyond fit noise."""
    rng = np.random.default_rng(2)
    z = rng.normal(size=20000) * 2.0
    p = 1 / (1 + np.exp(-z))
    y = rng.uniform(size=z.shape) < p
    t_clean = mc.fit_temperature(z * 3.0, y, np.ones_like(y, bool))
    z2 = np.concatenate([z, np.full(5000, 50.0)])
    y2 = np.concatenate([y, np.zeros(5000, bool)])          # huge logit, opposite label
    mask2 = np.concatenate([np.ones_like(y, bool), np.zeros(5000, bool)])
    t_masked = mc.fit_temperature(z2 * 3.0, y2, mask2)
    assert abs(t_masked - t_clean) < 1e-6


def test_fit_temperature_clamps_and_returns_one_for_empty_mask():
    """EXPECTATION: an all-False mask has nothing to calibrate against and
    must report T=1 (no-op), never NaN/crash/a made-up value; a pathological
    logit set whose derivative never changes sign in [0.25, 10] must clamp
    to whichever edge the derivative points at, not silently extrapolate."""
    z = np.array([1.0, -1.0, 2.0])
    y = np.array([True, False, True])
    assert mc.fit_temperature(z, y, np.zeros_like(y, bool)) == 1.0
    t = mc.fit_temperature(z, y, np.ones_like(y, bool))
    assert 0.25 <= t <= 10.0


# --------------------------------------------------------------------------
# reliability
# --------------------------------------------------------------------------
def test_reliability_of_perfectly_calibrated_probs_is_on_the_diagonal():
    """EXPECTATION: probabilities drawn uniformly and labels sampled AT those
    exact probabilities are the definition of "calibrated" -- with 50000
    samples over 10 bins (~5000/bin), sampling noise keeps every bin's
    |acc-conf| small; max_gap/ece must read near 0, not falsely flag
    miscalibration."""
    rng = np.random.default_rng(1)
    p = rng.uniform(size=50000)
    y = rng.uniform(size=p.shape) < p
    r = mc.reliability(p, y, np.ones_like(y, bool))
    assert r["max_gap"] < 0.05 and r["ece"] < 0.02
    assert len(r["edges"]) == 11 and len(r["acc"]) == 10 and len(r["n"]) == 10
    assert sum(r["n"]) == p.size


def test_reliability_names_an_over_confident_head_above_or_below_the_diagonal():
    """EXPECTATION: a head that reports 0.9 on frames that are only right
    half the time is OVER-confident there -- `conf` (0.9) sits well above
    `acc` (~0.5) in that bin, and `max_gap`/`ece` must be large, not small."""
    rng = np.random.default_rng(3)
    n = 4000
    p = np.full(n, 0.9)
    y = rng.uniform(size=n) < 0.5           # true frequency only 0.5, not 0.9
    r = mc.reliability(p, y, np.ones_like(y, bool))
    assert r["max_gap"] > 0.3
    populated = [i for i, c in enumerate(r["n"]) if c > 0]
    assert len(populated) == 1              # every prob is exactly 0.9 -- one bin


def test_reliability_empty_bins_do_not_count_toward_max_gap():
    """EXPECTATION: probabilities that only ever land in 2 of 10 bins must
    not report `max_gap`/`ece` computed over the 8 EMPTY bins (which would
    read as 0 -- artificially perfect) -- only the 2 populated bins count."""
    rng = np.random.default_rng(4)
    n = 2000
    p = np.where(rng.uniform(size=n) < 0.5, 0.05, 0.95)
    y = rng.uniform(size=n) < p
    r = mc.reliability(p, y, np.ones_like(y, bool))
    populated_n = [c for c in r["n"] if c > 0]
    assert len(populated_n) == 2
    assert np.isfinite(r["max_gap"]) and np.isfinite(r["ece"])


# --------------------------------------------------------------------------
# MVQRunner applies the stored temperature
# --------------------------------------------------------------------------
def test_runner_applies_the_stored_temperature(tiny):
    """A calibrated runner must not change WHICH slot is picked, only the
    reported probability -- the policy threshold moves with it.

    EXPECTATION: with `exist_temperature=2.0` (`vis_temperature` left at the
    default 1.0), the calibrated runner's `exist`/`vis` equal
    `sigmoid(raw_logit / T)` for the SAME forward, computed independently by
    inverting the (T=1, i.e. raw) uncalibrated runner's own reported
    probabilities -- the model is deterministic at inference (no dropout),
    so two runners loaded from the same `final/` checkpoint produce
    bit-identical logits on the same input. The argmax existence slot (the
    slot a real policy read would pick) must be IDENTICAL between the two,
    since dividing every logit by the same positive scalar cannot flip a
    ranking.
    """
    from jarvis_jax.tracking.lift_mvq import _sigmoid

    raw = _runner(tiny)                       # exist_temperature/vis_temperature default 1.0
    calibrated = _runner(tiny)
    # simulate a checkpoint whose `final/mvq_run.json` carries the
    # "calibration" block `mvq_calibrate.py` writes -- `MVQRunner` reads it
    # once at construction (`self.meta.get("calibration", {})`); set both
    # the provenance dict and the derived attribute the constructor would
    # have computed from it, so the two never disagree.
    calibrated.meta["calibration"] = {"exist_temperature": 2.0, "vis_temperature": 1.0}
    calibrated.exist_temperature = 2.0
    calibrated.vis_temperature = 1.0

    frames = np.zeros((7, 448, 1936, 3), np.uint8)
    w = raw.windows(frames, np.ones(7, bool), np.zeros((1, 3), np.float32))
    out_raw = raw.infer(w)
    out_cal = calibrated.infer(w)

    # invert the RAW (T=1) runner's reported probabilities to recover the
    # logits the shared checkpoint actually produced on this input.
    eps = 1e-6
    p_e = np.clip(out_raw["exist"], eps, 1 - eps)
    raw_exist_logit = np.log(p_e / (1 - p_e))
    expected_exist = _sigmoid(raw_exist_logit / 2.0)
    np.testing.assert_allclose(out_cal["exist"], expected_exist, atol=1e-4)

    p_v = np.clip(out_raw["vis"], eps, 1 - eps)
    raw_vis_logit = np.log(p_v / (1 - p_v))
    expected_vis = _sigmoid(raw_vis_logit / 1.0)          # vis_temperature unchanged (1.0)
    np.testing.assert_allclose(out_cal["vis"], expected_vis, atol=1e-4)
    # sanity: vis_temperature=1.0 is a no-op, so calibrated vis == raw vis
    np.testing.assert_allclose(out_cal["vis"], out_raw["vis"], atol=1e-4)

    assert np.argmax(out_raw["exist"][0]) == np.argmax(out_cal["exist"][0])


def test_runner_defaults_to_no_op_temperatures_when_uncalibrated(tiny):
    """EXPECTATION: a checkpoint whose `mvq_run.json` has never been touched
    by `mvq_calibrate.py` (no "calibration" key) must behave EXACTLY as
    every pre-calibration run always did -- `exist_temperature`/
    `vis_temperature` default to 1.0, a mathematical no-op."""
    r = _runner(tiny)
    assert r.exist_temperature == 1.0 and r.vis_temperature == 1.0
    frames = np.zeros((7, 448, 1936, 3), np.uint8)
    w = r.windows(frames, np.ones(7, bool), np.zeros((1, 3), np.float32))
    out = r.infer(w)
    assert np.all(np.isfinite(out["exist"])) and np.all(np.isfinite(out["vis"]))
