import jax.numpy as jnp
import numpy as np
from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints


def test_gate_zeroes_outside_preserves_inside():
    from jarvis_jax.densepose.gating import gate_heatmaps
    B, H, W, K = 1, 32, 32, 2
    hm = jnp.zeros((B, H, W, K))
    # ch0: a strong peak INSIDE the mask; ch1: a spurious peak OUTSIDE the mask.
    hm = hm.at[0, 10, 10, 0].set(5.0).at[0, 28, 28, 1].set(3.0)
    # mask at input res 128 covers only the top-left quadrant (dilate=0 -> raw).
    mask = jnp.zeros((B, 128, 128))
    mask = mask.at[0, :64, :64].set(1.0)                 # top-left quadrant hot
    out = np.asarray(gate_heatmaps(hm, mask, dilate=0))
    assert out.shape == (B, H, W, K)
    # the inside peak (10,10 in 32-grid == top-left quadrant) survives unchanged.
    assert out[0, 10, 10, 0] == 5.0
    # the outside peak (28,28 == bottom-right, mask=0) is zeroed.
    assert out[0, 28, 28, 1] == 0.0
    # everything outside the mask is zero.
    assert out[0, 16:, 16:, :].sum() == 0.0


def test_gate_moves_decoded_keypoint_onto_the_fly():
    from jarvis_jax.densepose.gating import gate_heatmaps
    B, H, W, K = 1, 32, 32, 1
    # a TALL spurious off-fly peak plus a smaller on-fly peak; ungated argmax picks
    # the spurious one, gated argmax picks the on-fly one.
    hm = jnp.zeros((B, H, W, K)).at[0, 28, 28, 0].set(9.0).at[0, 8, 8, 0].set(4.0)
    mask = jnp.zeros((B, 32, 32)).at[0, :16, :16].set(1.0)   # top-left is the fly
    kp_ungated = np.asarray(heatmaps_to_keypoints(hm, in_size=32))[0, 0]
    kp_gated = np.asarray(heatmaps_to_keypoints(gate_heatmaps(hm, mask, dilate=0), in_size=32))[0, 0]
    # ungated lands near the spurious (28,28); gated lands near the real (8,8).
    assert kp_ungated[0] > 20 and kp_ungated[1] > 20
    assert kp_gated[0] < 16 and kp_gated[1] < 16


def test_dilate_admits_a_peak_just_outside_the_raw_mask():
    from jarvis_jax.densepose.gating import gate_heatmaps
    B, H, W, K = 1, 16, 16, 1
    hm = jnp.zeros((B, H, W, K)).at[0, 8, 10, 0].set(2.0)   # peak at x=10
    mask = jnp.zeros((B, 16, 16)).at[0, 8, 8].set(1.0)      # raw mask covers x=8 only
    assert np.asarray(gate_heatmaps(hm, mask, dilate=0))[0, 8, 10, 0] == 0.0   # raw: zeroed
    assert np.asarray(gate_heatmaps(hm, mask, dilate=5))[0, 8, 10, 0] == 2.0   # dilated: preserved


def test_full_mask_is_identity():
    from jarvis_jax.densepose.gating import gate_heatmaps
    rng = np.random.default_rng(0)
    hm = jnp.asarray(rng.normal(size=(2, 16, 16, 3)).astype("float32"))
    mask = jnp.ones((2, 16, 16))                           # whole image is fly
    out = gate_heatmaps(hm, mask, dilate=0)
    assert np.allclose(np.asarray(out), np.asarray(hm))
