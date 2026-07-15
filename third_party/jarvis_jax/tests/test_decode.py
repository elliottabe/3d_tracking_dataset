import numpy as np
import pytest
from jarvis_jax.eval import decode


def _blob(h, w, k, cx, cy, sigma):
    """(1,h,w,k) heatmap: identical Gaussian blob at (cx,cy) in every channel."""
    ys = np.arange(h)[:, None]; xs = np.arange(w)[None, :]
    g = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)))
    return np.repeat(g[None, :, :, None], k, axis=3).astype(np.float32)


def test_hard_argmax_hits_peak_pixel():
    hm = _blob(224, 224, 3, cx=100, cy=60, sigma=2.0)
    kp = np.asarray(decode.decode_hard_argmax(hm, in_size=448))  # (1,3,2)
    scale = 448 / 224
    assert np.allclose(kp[0, :, 0], 100 * scale, atol=scale)     # x
    assert np.allclose(kp[0, :, 1], 60 * scale, atol=scale)      # y


def test_soft_centroid_recovers_symmetric_center():
    hm = _blob(224, 224, 2, cx=100, cy=60, sigma=3.0)
    kp = np.asarray(decode.decode_soft_centroid(hm, in_size=448, radius=7))
    scale = 448 / 224
    assert np.allclose(kp[0, :, 0], 100 * scale, atol=0.5)
    assert np.allclose(kp[0, :, 1], 60 * scale, atol=0.5)


def test_gaussian_recovers_subpixel_offset():
    hm = _blob(224, 224, 1, cx=100.4, cy=60.0, sigma=2.0)
    kp = decode.decode_gaussian(hm, in_size=448)
    scale = 448 / 224
    # subpixel decode should land nearer the true 100.4 than the integer peak 100
    assert abs(kp[0, 0, 0] / scale - 100.4) < abs(100 - 100.4)


def test_dispatcher_matches_direct_calls():
    hm = _blob(64, 64, 4, cx=30, cy=20, sigma=2.5)
    a = decode.decode_heatmaps(hm, method="soft_centroid", in_size=448, sharpen=1.0)
    b = np.asarray(decode.decode_soft_centroid(hm, in_size=448, sharpen=1.0))
    assert np.allclose(a, b)
    with pytest.raises(ValueError):
        decode.decode_heatmaps(hm, method="nope")
