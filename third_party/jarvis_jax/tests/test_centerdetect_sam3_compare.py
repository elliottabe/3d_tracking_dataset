import numpy as np

from jarvis_jax.eval.centerdetect_sam3_compare import nearest_peak_distances, summarize


def test_nearest_peak_distances_basic():
    peaks = np.array([[100.0, 100.0], [400.0, 100.0]])
    cents = np.array([[105.0, 95.0], [1000.0, 1000.0]])
    valid = np.array([True, False])
    out = nearest_peak_distances(peaks, cents, valid)
    assert len(out) == 1
    slot, d = out[0]
    assert slot == 0
    assert abs(d - np.hypot(5, 5)) < 1e-6


def test_summarize_two_peak_rate_and_distances():
    samples = [
        {"peaks_full_xy": [[100.0, 100.0], [400.0, 100.0]],
         "sam3_centroids_xy": [[102.0, 98.0], [398.0, 103.0]],
         "sam3_valid": [True, True]},          # clean hit
        {"peaks_full_xy": [[101.0, 99.0], [101.0, 99.0]],
         "sam3_centroids_xy": [[100.0, 100.0], [400.0, 100.0]],
         "sam3_valid": [True, True]},          # collapsed
        {"peaks_full_xy": [[50.0, 50.0], [60.0, 60.0]],
         "sam3_centroids_xy": [[52.0, 48.0], [0.0, 0.0]],
         "sam3_valid": [True, False]},          # only one SAM3 slot valid -> excluded from hit rate
    ]
    out = summarize(samples)
    assert out["n_samples"] == 3
    assert out["n_both_valid"] == 2
    assert out["two_peak_rate"] == 0.5           # 1 hit / 2 both-valid samples
    assert out["n_dist_obs"] == 5                 # 2+2+1 valid SAM3 observations
    assert out["dist_median_px"] >= 0
    assert "dist_p90_px" in out


def test_summarize_handles_no_samples():
    out = summarize([])
    assert out["n_samples"] == 0
    assert np.isnan(out["two_peak_rate"])
