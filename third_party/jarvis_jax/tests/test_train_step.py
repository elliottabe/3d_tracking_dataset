# tests/test_train_step.py
import jax
import numpy as np
import pytest
from flax import nnx

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.train.train import TrainConfig, make_optimizer, make_train_step

gpu = any(d.platform == "gpu" for d in jax.devices())
needs_gpu = pytest.mark.skipif(not gpu, reason="train-step test needs a GPU")


@needs_gpu
def test_train_step_reduces_loss_with_finite_grads():
    """The train step + optimizer + loss are wired correctly: a few steps
    produce finite losses and meaningfully reduce them.

    This is a deterministic wiring gate, NOT a localization gate. A full
    overfit-to-<Npx on a synthetic batch was tried and abandoned: memorizing
    random-noise images through an 86M-param ViT is intrinsically unstable under
    cross-process XLA-autotuning nondeterminism (it occasionally diverges to a
    degenerate solution, ~20% of the time), so its pass/fail is not reproducible.
    The two real failure modes that overfit once surfaced are now locked in by
    deterministic unit tests: foreground-weighted loss forming the peak
    (test_losses.test_mse_penalizes_missed_foreground_peak) and background-robust
    decoding (test_mpjpe.test_decode_robust_to_diffuse_positive_background).
    End-to-end keypoint localization is validated by the real-data smoke run in
    test_train_script (structured images + MAE init + longer training = stable).
    """
    cfg = ViTPoseConfig()
    rng = np.random.RandomState(0)
    img4 = rng.rand(2, 448, 448, 4).astype("float32")
    from jarvis_jax.data.transforms import gaussian_heatmaps
    vis = np.zeros((2, 50), dtype=bool); vis[:, 0] = True
    hm = np.zeros((2, 224, 224, 50), dtype="float32")
    for b, (cx, cy) in enumerate([(100.0, 50.0), (60.0, 150.0)]):
        xy = np.zeros((50, 2), dtype="float32"); xy[0] = [cx, cy]
        hm[b] = gaussian_heatmaps(xy, vis[b])

    model = ViTPose(cfg, rngs=nnx.Rngs(0))
    tcfg = TrainConfig(total_steps=20, lr=1e-3, warmup_steps=2)
    opt = make_optimizer(model, tcfg)
    step = make_train_step(mask_weight=0.0)

    import jax.numpy as jnp
    img4j, hmj, visj = jnp.asarray(img4), jnp.asarray(hm), jnp.asarray(vis)
    losses = [float(step(model, opt, img4j, hmj, visj)) for _ in range(20)]

    assert np.isfinite(losses).all(), f"non-finite loss encountered: {losses}"
    # A clear, robust reduction (not a marginal epsilon) proves gradients flowed
    # and the optimizer is updating; well-separated from a no-op/NaN failure.
    assert losses[-1] < 0.9 * losses[0], f"loss did not drop: {losses[0]} -> {losses[-1]}"
