import numpy as np
from jarvis_jax.eval import decode


def _blob(h, w, k, cx, cy, sigma):
    ys = np.arange(h)[:, None]; xs = np.arange(w)[None, :]
    g = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)))
    return np.repeat(g[None, :, :, None], k, axis=3).astype(np.float32)


def test_peak_concentration_sharper_is_higher():
    sharp = _blob(128, 128, 1, 64, 64, sigma=1.5)
    diffuse = _blob(128, 128, 1, 64, 64, sigma=12.0)
    cs = decode.peak_concentration(sharp, radius=7)[0, 0]
    cd = decode.peak_concentration(diffuse, radius=7)[0, 0]
    assert cs > cd
    assert cs > 0.9        # a tight peak has nearly all its mass in-window


def test_wobble_zero_on_smooth_track_positive_on_noisy():
    T, K = 200, 3
    t = np.arange(T)
    smooth = np.stack([np.stack([0.1 * t, 0.05 * t], 1) for _ in range(K)], 1)  # (T,K,2)
    rng = np.random.default_rng(0)
    noisy = smooth + rng.normal(0, 1.5, smooth.shape)
    w_smooth = decode.wobble(smooth)
    w_noisy = decode.wobble(noisy)
    assert np.all(w_smooth < 0.05)
    assert np.all(w_noisy > 0.8)
    assert np.all(w_noisy > w_smooth)


def test_wobble_ignores_nan_frames():
    T, K = 100, 1
    v = np.stack([np.stack([0.1 * np.arange(T), np.zeros(T)], 1)], 1)  # (T,1,2)
    v[10, 0, 0] = np.nan
    out = decode.wobble(v)
    assert np.isfinite(out[0]) and out[0] < 0.05
