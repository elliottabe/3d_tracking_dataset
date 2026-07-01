import jax.numpy as jnp
import numpy as np
from jarvis_jax.train.losses import mask_containment


def _peak_just_outside():
    """A 1x1 heatmap peak 2px to the right of a raw 1-pixel mask.
    Raw mask covers (b=0, y=8, x=8); the peak is at (y=8, x=10)."""
    B, H, W, K = 1, 17, 17, 1
    pred = jnp.zeros((B, H, W, K)).at[0, 8, 10, 0].set(1.0)
    mask = jnp.zeros((B, H, W)).at[0, 8, 8].set(1.0)
    return pred, mask


def test_dilate0_matches_current_behaviour_exactly():
    # dilate=0 must equal the pre-Phase-5 call (no kwarg) to the bit.
    pred, mask = _peak_just_outside()
    assert float(mask_containment(pred, mask, dilate=0)) == float(mask_containment(pred, mask))
    # peak is OUTSIDE the undilated 1-pixel mask -> all positive mass outside -> 1.0
    assert abs(float(mask_containment(pred, mask, dilate=0)) - 1.0) < 1e-5


def test_penalty_drops_as_dilate_grows():
    pred, mask = _peak_just_outside()
    d0 = float(mask_containment(pred, mask, dilate=0))    # peak outside -> 1.0
    d3 = float(mask_containment(pred, mask, dilate=3))    # mask now covers x in [7,9]; peak at 10 still out -> 1.0
    d5 = float(mask_containment(pred, mask, dilate=5))    # mask covers x in [6,10]; peak at 10 now INSIDE -> 0.0
    # float32 division by (sum + eps) rounds fractionally below 1.0 (same artifact
    # the pre-existing test_mask_containment_zero_inside_one_outside tolerates).
    assert abs(d0 - 1.0) < 1e-5
    assert abs(d3 - 1.0) < 1e-5
    assert d5 < 1e-5, f"dilate=5 should bring the peak inside -> ~0 penalty, got {d5}"
    # monotone non-increasing in dilate
    ds = [float(mask_containment(pred, mask, dilate=k)) for k in (0, 1, 3, 5, 7)]
    assert all(a >= b - 1e-6 for a, b in zip(ds, ds[1:])), f"not monotone: {ds}"


def test_inside_peak_unaffected_by_dilation():
    # peak on the mask pixel -> 0 penalty regardless of dilation.
    B, H, W, K = 1, 9, 9, 1
    pred = jnp.zeros((B, H, W, K)).at[0, 4, 4, 0].set(1.0)
    mask = jnp.zeros((B, H, W)).at[0, 4, 4].set(1.0)
    for k in (0, 3, 7):
        assert float(mask_containment(pred, mask, dilate=k)) < 1e-6


def test_train_step_mask_dilate_wired():
    # make_train_step accepts mask_dilate and runs one step without error (CPU).
    from flax import nnx
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.train.train import TrainConfig, make_optimizer, make_train_step
    cfg = ViTPoseConfig(num_keypoints=3, img_size=64, heatmap_size=32)  # tiny for CPU
    m = ViTPose(cfg, rngs=nnx.Rngs(0))
    opt = make_optimizer(m, TrainConfig(total_steps=10))
    step = make_train_step(mask_weight=1.0, mask_dilate=5, heatmap_size=cfg.heatmap_size)
    img = jnp.zeros((2, 64, 64, 4)).at[..., 3].set(1.0)   # full-fly mask channel
    kp = jnp.full((2, 3, 2), 16.0)
    vis = jnp.ones((2, 3), dtype=bool)
    import jax
    loss = float(step(m, opt, jax.random.PRNGKey(0), img, kp, vis))
    assert np.isfinite(loss)


def test_traincfg_has_mask_dilate_default_zero():
    from jarvis_jax.train.train import TrainConfig
    assert TrainConfig().mask_dilate == 0
