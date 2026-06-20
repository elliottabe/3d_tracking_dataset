import jax.numpy as jnp
import numpy as np
from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints, mpjpe


def _gauss(h, w, cx, cy, sigma=2.0):
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * sigma * sigma))


def test_centroid_recovers_peak_and_scales_to_448():
    hm = np.zeros((1, 224, 224, 1), dtype=np.float32)
    hm[0, :, :, 0] = _gauss(224, 224, cx=100.0, cy=50.0)
    kp = np.asarray(heatmaps_to_keypoints(jnp.asarray(hm), in_size=448))
    # heatmap (100,50) -> *2 -> (200,100) in 448 coords
    assert np.allclose(kp[0, 0], [200.0, 100.0], atol=0.1)


def test_mpjpe_zero_when_equal_and_ignores_invisible():
    kp = jnp.asarray(np.random.RandomState(1).rand(2, 5, 2).astype("float32"))
    vis = jnp.ones((2, 5), dtype=bool)
    assert float(mpjpe(kp, kp, vis)) == 0.0
    pred = kp.at[0, 0, 0].add(10.0)
    vis2 = jnp.ones((2, 5), dtype=bool).at[0, 0].set(False)
    assert float(mpjpe(pred, kp, vis2)) == 0.0   # the changed kp is invisible
