"""Tests for ``jarvis_jax.eval.centerdetect_decode``: top-2 peak extraction,
full-image rescaling, and the two-peak-hit definition."""
import numpy as np

from jarvis_jax.eval.centerdetect_decode import (
    extract_top_k_peaks, peaks_to_full_image, two_peak_hit,
)


def _bump(h, w, cx, cy, amp, sigma=3.0):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    return amp * np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)))


def test_extract_top_k_peaks_recovers_two_known_peaks_in_confidence_order():
    h = w = 80
    hm = _bump(h, w, 20, 20, amp=220.0) + _bump(h, w, 60, 55, amp=90.0)
    peaks, conf = extract_top_k_peaks(hm[None], k=2, suppression_radius=15)
    assert peaks.shape == (1, 2, 2)
    assert conf.shape == (1, 2)
    # Strongest peak first.
    assert tuple(peaks[0, 0]) == (20.0, 20.0)
    assert tuple(peaks[0, 1]) == (60.0, 55.0)
    assert conf[0, 0] > conf[0, 1] > 0
    assert abs(conf[0, 0] - 220.0) < 1e-3
    assert abs(conf[0, 1] - 90.0) < 1e-3


def test_always_returns_k_peaks_even_with_one_real_peak_and_flat_background():
    """No confidence threshold anywhere: a single real peak still yields a
    second candidate, taken from whatever is left after suppression --
    matching the production PyTorch decode's unconditioned top-k."""
    h = w = 80
    rng = np.random.RandomState(0)
    hm = _bump(h, w, 40, 40, amp=230.0) + rng.uniform(0, 1.0, size=(h, w))
    peaks, conf = extract_top_k_peaks(hm[None], k=2, suppression_radius=15)
    assert peaks.shape == (1, 2, 2)
    assert conf[0, 0] > 200.0                 # the real peak
    assert conf[0, 1] < 2.0                   # just background noise, but STILL RETURNED
    assert not np.allclose(peaks[0, 0], peaks[0, 1])


def test_suppression_radius_prevents_reselecting_the_same_peak():
    h = w = 80
    hm = _bump(h, w, 40, 40, amp=255.0, sigma=5.0)
    peaks, conf = extract_top_k_peaks(hm[None], k=2, suppression_radius=15)
    dist = np.hypot(*(peaks[0, 0] - peaks[0, 1]))
    assert dist > 15.0, f"2nd peak at distance {dist:.1f}px, inside the suppression disk"


def test_batch_dimension_and_4d_singleton_channel_accepted():
    h = w = 40
    hm1 = _bump(h, w, 10, 10, amp=200.0)
    hm2 = _bump(h, w, 30, 30, amp=180.0)
    batch3d = np.stack([hm1, hm2])                     # (2,H,W)
    batch4d = batch3d[..., None]                        # (2,H,W,1)
    p3, c3 = extract_top_k_peaks(batch3d, k=2, suppression_radius=10)
    p4, c4 = extract_top_k_peaks(batch4d, k=2, suppression_radius=10)
    assert np.allclose(p3, p4) and np.allclose(c3, c4)


def test_multi_channel_heatmap_rejected():
    hm = np.zeros((1, 10, 10, 2))
    try:
        extract_top_k_peaks(hm, k=2)
        assert False, "expected a ValueError for a multi-channel heatmap"
    except ValueError:
        pass


def test_peaks_to_full_image_rescales_each_axis_independently():
    peaks_hm = np.array([[[10.0, 20.0], [30.0, 40.0]]])   # (1,2,2), heatmap_size=80
    out = peaks_to_full_image(peaks_hm, heatmap_size=80, img_w=1936, img_h=448)
    sx, sy = 1936 / 80.0, 448 / 80.0
    assert np.allclose(out[0, 0], [10.0 * sx, 20.0 * sy])
    assert np.allclose(out[0, 1], [30.0 * sx, 40.0 * sy])


def test_two_peak_hit_true_when_each_gt_fly_has_a_distinct_close_peak():
    gt = np.array([[100.0, 100.0], [400.0, 100.0]])
    peaks = np.array([[102.0, 98.0], [395.0, 105.0]])
    is_hit, dists, idx = two_peak_hit(peaks, gt, capture_radius_px=40.0)
    assert is_hit
    assert idx[0] != idx[1]
    assert (dists < 40.0).all()


def test_two_peak_hit_false_when_collapsed_to_one_peak():
    gt = np.array([[100.0, 100.0], [400.0, 100.0]])
    # Both "peaks" sit at the IDENTICAL location near the FIRST fly -- the
    # classic collapse (the 2nd extracted peak is spurious background noise
    # that happens to be nearest to the same fly as the 1st).
    peaks = np.array([[101.0, 99.0], [101.0, 99.0]])
    is_hit, dists, idx = two_peak_hit(peaks, gt, capture_radius_px=40.0)
    assert not is_hit
    assert idx[0] == idx[1]


def test_two_peak_hit_false_when_a_peak_is_too_far():
    gt = np.array([[100.0, 100.0], [400.0, 100.0]])
    peaks = np.array([[102.0, 98.0], [1000.0, 1000.0]])   # 2nd peak nowhere near fly 2
    is_hit, dists, idx = two_peak_hit(peaks, gt, capture_radius_px=40.0)
    assert not is_hit
    assert dists[1] > 40.0
