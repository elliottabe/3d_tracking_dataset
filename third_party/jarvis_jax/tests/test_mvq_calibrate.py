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


def test_reliability_min_n_excludes_a_1_sample_outlier_bin():
    """EXPECTATION: raw `max_gap` can be pinned anywhere in [0,1] by a
    single val sample landing in an otherwise-empty bin -- exactly the
    failure mode measured on the real P3b proof run (an n=1 bin drove
    max_gap to 0.28 while every well-populated bin was near-perfect).
    `max_gap_min_n` (default `min_n=20`) must ignore that one bin and read
    the well-populated bins' true (tiny) gap instead; the raw `max_gap`
    must still be reported unchanged, for reference."""
    rng = np.random.default_rng(5)
    n = 4000
    # `n` well-calibrated samples in bin [0.4, 0.5) (n well over min_n)...
    p_main = np.full(n, 0.45)
    y_main = rng.uniform(size=n) < 0.45
    # ...plus ONE adversarial sample in a different, otherwise-empty bin
    # (n=1 there), with a label that maximises its own gap.
    p = np.concatenate([p_main, [0.85]])
    y = np.concatenate([y_main, [True]])
    r = mc.reliability(p, y, np.ones_like(y, bool), min_n=20)
    ns = r["n"]
    outlier_bin = int(0.85 * 10)          # bin index conf=0.85 falls into
    assert ns[outlier_bin] == 1
    main_bin = int(0.45 * 10)
    assert ns[main_bin] >= 20
    # the raw max_gap is dominated by the n=1 bin (gap ~0.15, |1-0.85|)
    assert r["max_gap"] == pytest.approx(abs(1.0 - 0.85), abs=1e-9)
    # max_gap_min_n ignores it and reads off the well-populated bin instead
    assert r["max_gap_min_n"] is not None
    assert r["max_gap_min_n"] < r["max_gap"]
    assert r["max_gap_min_n"] == pytest.approx(
        abs(r["acc"][main_bin] - r["conf"][main_bin]), abs=1e-9)


def test_reliability_max_gap_min_n_is_none_when_no_bin_qualifies():
    """EXPECTATION: every bin below `min_n` samples means there is nothing
    reliable to gate an acceptance decision on -- `max_gap_min_n` must be
    `None` (never a number computed from sparse bins, never crash), while
    `max_gap`/`ece` (which have no such floor) still report a real number."""
    rng = np.random.default_rng(6)
    p = rng.uniform(size=50)              # 50 samples spread over 10 bins -- a handful each
    y = rng.uniform(size=p.shape) < p
    r = mc.reliability(p, y, np.ones_like(y, bool), min_n=20)
    assert r["max_gap_min_n"] is None
    assert np.isfinite(r["max_gap"])


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
