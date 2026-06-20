import numpy as np
import jax.numpy as jnp
from jarvis_jax.data.device import normalize_image, render_heatmaps
from jarvis_jax.data.transforms import (
    normalize_rgb, gaussian_heatmaps, IMAGENET_MEAN, IMAGENET_STD,
)


def test_normalize_image_matches_numpy_and_keeps_mask():
    rng = np.random.RandomState(0)
    rgb_u8 = rng.randint(0, 256, (8, 8, 3), dtype=np.uint8)
    mask = (rng.rand(8, 8) > 0.5).astype(np.uint8)
    img4 = np.concatenate([rgb_u8, mask[..., None]], axis=-1).astype(np.uint8)
    out = np.asarray(normalize_image(jnp.asarray(img4)))
    ref_rgb = normalize_rgb(rgb_u8.astype(np.float32) / 255.0)
    assert np.allclose(out[..., :3], ref_rgb, atol=1e-5)
    assert np.array_equal(out[..., 3], mask.astype(np.float32))  # 0/1 preserved
    assert out.dtype == np.float32


def test_render_heatmaps_matches_numpy_gaussian():
    # two keypoints, one invisible; compare to the NumPy renderer (sigma=7.0)
    kp = np.zeros((1, 50, 2), dtype=np.float32)
    kp[0, 0] = [100.0, 50.0]; kp[0, 1] = [60.0, 150.0]
    vis = np.zeros((1, 50), dtype=bool); vis[0, 0] = True; vis[0, 1] = True
    out = np.asarray(render_heatmaps(jnp.asarray(kp), jnp.asarray(vis)))
    ref = gaussian_heatmaps(kp[0], vis[0], heatmap_size=224, sigma=7.0)
    assert out.shape == (1, 224, 224, 50)
    assert np.allclose(out[0], ref, atol=1e-4)
    assert out[0, :, :, 2].max() == 0.0  # invisible channel stays zero


def test_render_heatmaps_invisible_all_zero():
    kp = np.zeros((2, 50, 2), dtype=np.float32)
    vis = np.zeros((2, 50), dtype=bool)
    out = np.asarray(render_heatmaps(jnp.asarray(kp), jnp.asarray(vis)))
    assert out.max() == 0.0
