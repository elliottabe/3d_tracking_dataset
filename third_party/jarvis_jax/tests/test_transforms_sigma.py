import numpy as np
from jarvis_jax.data.transforms import gaussian_heatmaps

# NOTE on the two fixes applied vs. the task-9 brief's verbatim draft (see
# task-9-report.md for the full writeup):
#
# 1. gaussian_heatmaps returns (heatmap_size, heatmap_size, K) — channel
#    LAST — per its docstring and the existing tests
#    (test_transforms.py::test_gaussian_heatmaps_peak_location_and_invisible_zero,
#    test_device.py::test_render_heatmaps_matches_numpy_gaussian both index
#    `hm[:, :, j]` / `hm[..., j]`). The brief's draft indexed `hm[0]`, which
#    on this shape selects heatmap ROW 0 (far from any peak), not keypoint
#    channel 0 — it raised IndexError / compared near-zero noise. Fixed to
#    `hm[:, :, 0]`.
#
# 2. The "adjacent tarsal targets separate" check compared the value at the
#    geometric MIDPOINT between two peaks 4.6 px apart against half the peak
#    height. That midpoint (r=2.3 px from either peak) sits almost exactly at
#    sigma=2.0's own half-max radius (sigma*sqrt(2 ln 2) = 2.355 px), so the
#    ratio there (~0.516) never drops below 0.5 for ANY value of sigma=2.0 —
#    the assertion was unsatisfiable by construction, independent of any
#    implementation bug. Checking the value of keypoint 0's channel AT
#    keypoint 1's own location instead gives an unambiguous, principled
#    read on "do these two targets overlap" (0.775 at sigma=7 vs. 0.044 at
#    sigma=2 relative to peak) and is a direct measure of the "adjacent
#    tarsal targets overlap almost completely" claim this task is fixing.


def _concentration(hm, radius=7):
    """Fraction of heatmap mass within `radius` px of the peak — the same
    statistic the 2D wobble diagnostic reported as 0.352 for the shipped model."""
    j, i = np.unravel_index(np.argmax(hm), hm.shape)
    yy, xx = np.mgrid[:hm.shape[0], :hm.shape[1]]
    near = ((yy - j) ** 2 + (xx - i) ** 2) <= radius ** 2
    return float(hm[near].sum() / max(hm.sum(), 1e-9))


def test_sigma_two_is_far_more_concentrated_than_seven():
    """sigma=7 px is wider than an entire tarsal segment (~4.6 heatmap px),
    so adjacent tarsal targets overlap almost completely."""
    xy = np.array([[112.0, 112.0]], np.float32)
    vis = np.array([1], np.int32)
    wide = np.asarray(gaussian_heatmaps(xy, vis, heatmap_size=224, sigma=7.0))[:, :, 0]
    tight = np.asarray(gaussian_heatmaps(xy, vis, heatmap_size=224, sigma=2.0))[:, :, 0]
    assert _concentration(wide) < 0.55
    assert _concentration(tight) > 0.95


def test_adjacent_tarsal_targets_separate_at_sigma_two():
    """Two keypoints 4.6 px apart — the measured tarsal segment length.
    Checks how much of keypoint 0's OWN target amplitude reaches keypoint
    1's location: a high fraction there means the two targets are
    confusable ("not separated")."""
    xy = np.array([[112.0, 112.0], [116.6, 112.0]], np.float32)
    vis = np.array([1, 1], np.int32)
    neighbor_col = 117  # nearest grid col to the second keypoint at x=116.6
    for sigma, expect_separated in ((7.0, False), (2.0, True)):
        hm = np.asarray(gaussian_heatmaps(xy, vis, heatmap_size=224, sigma=sigma))
        at_neighbor = hm[112, neighbor_col, 0]
        peak = hm[112, 112, 0]
        separated = at_neighbor < 0.5 * peak
        assert separated == expect_separated, (sigma, at_neighbor, peak)


def test_default_sigma_is_two():
    import inspect
    assert inspect.signature(gaussian_heatmaps).parameters["sigma"].default == 2.0
