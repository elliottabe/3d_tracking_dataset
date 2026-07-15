import numpy as np
from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints


def test_sharpen_reduces_diffuse_tail_bias():
    """A sharp peak at x=100 plus an asymmetric diffuse tail to the right biases
    a plain centroid (sharpen=1) rightward; sharpen>1 pulls it back to the peak."""
    h = w = 224
    ys = np.arange(h)[:, None]; xs = np.arange(w)[None, :]
    peak = np.exp(-(((xs - 100) ** 2 + (ys - 60) ** 2) / (2 * 1.5 ** 2)))
    tail = 0.25 * np.exp(-(((xs - 106) ** 2 + (ys - 60) ** 2) / (2 * 4.0 ** 2)))
    hm = (peak + tail)[None, :, :, None].astype(np.float32)     # (1,1,224,224)->(1,224,224,1)
    hm = np.moveaxis(hm, 1, -1) if hm.shape[1] == 1 else hm     # ensure (1,H,W,1)
    scale = 448 / 224
    kp_soft = np.asarray(heatmaps_to_keypoints(hm, in_size=448, sharpen=1.0))[0, 0, 0]
    kp_shrp = np.asarray(heatmaps_to_keypoints(hm, in_size=448, sharpen=4.0))[0, 0, 0]
    true_x = 100 * scale
    assert abs(kp_shrp - true_x) < abs(kp_soft - true_x)


def test_sharpen_default_is_backward_compatible():
    """Default (no sharpen kwarg) must equal sharpen=1.0 exactly."""
    rng = np.random.default_rng(0)
    hm = rng.random((2, 64, 64, 5)).astype(np.float32)
    a = np.asarray(heatmaps_to_keypoints(hm, in_size=448))
    b = np.asarray(heatmaps_to_keypoints(hm, in_size=448, sharpen=1.0))
    assert np.allclose(a, b)
