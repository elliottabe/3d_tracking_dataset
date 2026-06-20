import numpy as np
from jarvis_jax.data.transforms import (
    IMAGENET_MEAN, IMAGENET_STD, crop_origin, normalize_rgb,
    transform_keypoints, gaussian_heatmaps,
)


def test_crop_origin_centers_and_clamps():
    # bbox center at x=160 -> ideal x0 = 160-224 = -64 -> clamp to 0
    x0, y0 = crop_origin([100.0, 0.0, 120.0, 200.0], img_w=1936, img_h=448, crop=448)
    assert x0 == 0 and y0 == 0
    # center near right edge clamps so window stays inside width
    x0, y0 = crop_origin([1900.0, 0.0, 20.0, 200.0], img_w=1936, img_h=448, crop=448)
    assert x0 == 1936 - 448 and y0 == 0


def test_normalize_rgb_matches_formula():
    rgb = np.full((2, 2, 3), 0.5, dtype=np.float32)
    out = normalize_rgb(rgb)
    expected = (0.5 - IMAGENET_MEAN) / IMAGENET_STD
    assert np.allclose(out[0, 0], expected, atol=1e-6)
    assert out.dtype == np.float32


def test_transform_keypoints_shift_scale_and_vis():
    kps = np.zeros((50, 3), dtype=np.float32)
    kps[0] = [300.0, 100.0, 2.0]      # visible, inside
    kps[1] = [10.0, 10.0, 0.0]        # unlabelled
    kps[2] = [10000.0, 10.0, 2.0]     # out of bounds after crop
    hm_xy, vis = transform_keypoints(kps, x0=100, y0=0, crop=448, heatmap_size=224)
    # (300-100, 100-0) * (224/448) = (100, 50)
    assert np.allclose(hm_xy[0], [100.0, 50.0], atol=1e-4)
    assert vis[0] and not vis[1] and not vis[2]


def test_gaussian_heatmaps_peak_location_and_invisible_zero():
    hm_xy = np.zeros((50, 2), dtype=np.float32)
    hm_xy[0] = [100.0, 50.0]
    vis = np.zeros((50,), dtype=bool)
    vis[0] = True
    hm = gaussian_heatmaps(hm_xy, vis, heatmap_size=224, sigma=2.0)
    assert hm.shape == (224, 224, 50)
    # argmax of channel 0 is at (row=50, col=100)
    r, c = np.unravel_index(np.argmax(hm[:, :, 0]), (224, 224))
    assert (r, c) == (50, 100)
    assert abs(hm[50, 100, 0] - 1.0) < 1e-3
    assert hm[:, :, 1].max() == 0.0  # invisible channel is all zeros
