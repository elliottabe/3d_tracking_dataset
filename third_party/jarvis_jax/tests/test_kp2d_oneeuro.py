import numpy as np
from jarvis_jax.cse.kp2d_oneeuro import one_euro_1d, filter_kp2d_oneeuro


def test_one_euro_constant_is_unchanged():
    x = np.full(50, 3.0)
    out = one_euro_1d(x, min_cutoff=1.0, beta=0.0)
    assert np.allclose(out, 3.0, atol=1e-6)


def test_one_euro_reduces_noise_variance():
    rng = np.random.default_rng(1)
    x = 10.0 + rng.normal(0, 1.0, 300)
    # NOTE: min_cutoff=0.1 (not the 0.3 originally sketched in the task brief).
    # At dt=1.0, beta=0.0, min_cutoff=0.3 gives alpha ~0.65, whose steady-state
    # noise-std ratio is ~0.65-0.73 (verified empirically across 20 seeds,
    # min observed 0.648) -- it can never satisfy < 0.6. min_cutoff=0.1 gives
    # a reliable ~0.42-0.53 ratio, so it actually exercises "reduces variance".
    out = one_euro_1d(x, min_cutoff=0.1, beta=0.0)
    assert np.nanstd(out) < 0.6 * np.std(x)


def test_one_euro_resets_across_nan_gap():
    x = np.array([1.0, 1.0, 1.0, np.nan, 9.0, 9.0])
    out = one_euro_1d(x, min_cutoff=0.1, beta=0.0)
    assert np.isnan(out[3])                 # gap stays NaN
    assert out[4] == 9.0                    # segment restarts at raw value (no lag from pre-gap)


def test_filter_preserves_wings_and_smooths_legs():
    T, C, K = 200, 1, 3
    names = ["WingL_V13", "T1L_TaTip", "Scutellum"]
    rng = np.random.default_rng(2)
    base = np.zeros((T, C, K, 2))
    base[:, 0, :, 0] = 0.1 * np.arange(T)[:, None]
    kp = base + rng.normal(0, 1.5, base.shape)
    conf = np.ones((T, C, K))
    out = filter_kp2d_oneeuro(kp, conf, names, conf_thresh=0.3, min_cutoff=0.3)
    # wing (idx 0) untouched
    assert np.allclose(out[:, 0, 0], kp[:, 0, 0])
    # non-wing (idx 1,2) smoothed: lower high-freq residual than raw
    for j in (1, 2):
        raw_res = np.diff(kp[:, 0, j, 0]).std()
        flt_res = np.diff(out[:, 0, j, 0]).std()
        assert flt_res < raw_res


def test_filter_keeps_raw_on_low_conf_frame():
    T, C, K = 20, 1, 1
    kp = np.zeros((T, C, K, 2)); kp[:, 0, 0, 0] = np.arange(T)
    conf = np.ones((T, C, K)); conf[5, 0, 0] = 0.0     # below thresh
    out = filter_kp2d_oneeuro(kp, conf, ["T1L_TaTip"], conf_thresh=0.3, min_cutoff=1.0)
    assert out[5, 0, 0, 0] == kp[5, 0, 0, 0]           # low-conf frame keeps raw
