# tests/test_train_step.py
import os
import jax
import numpy as np
import pytest
from flax import nnx

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.train.train import TrainConfig, make_optimizer, make_train_step
from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints, mpjpe

gpu = any(d.platform == "gpu" for d in jax.devices())
needs_gpu = pytest.mark.skipif(not gpu, reason="overfit test needs a GPU")


@needs_gpu
def test_overfit_one_batch_drops_loss_and_recovers_keypoints():
    cfg = ViTPoseConfig()
    rng = np.random.RandomState(0)
    # tiny synthetic batch of 2: random 4-ch images, one visible keypoint each
    img4 = rng.rand(2, 448, 448, 4).astype("float32")
    from jarvis_jax.data.transforms import gaussian_heatmaps
    vis = np.zeros((2, 50), dtype=bool); vis[:, 0] = True
    hm = np.zeros((2, 224, 224, 50), dtype="float32")
    centers = [(100.0, 50.0), (60.0, 150.0)]
    for b, (cx, cy) in enumerate(centers):
        xy = np.zeros((50, 2), dtype="float32"); xy[0] = [cx, cy]
        hm[b] = gaussian_heatmaps(xy, vis[b])

    model = ViTPose(cfg, rngs=nnx.Rngs(0))
    tcfg = TrainConfig(total_steps=600, lr=3e-3, warmup_steps=30)
    opt = make_optimizer(model, tcfg)
    step = make_train_step(mask_weight=0.0)

    import jax.numpy as jnp
    img4j, hmj, visj = jnp.asarray(img4), jnp.asarray(hm), jnp.asarray(vis)
    first = float(step(model, opt, img4j, hmj, visj))
    for _ in range(600):
        last = float(step(model, opt, img4j, hmj, visj))
    # balanced loss floors higher than plain MSE (background term), so use 0.3x
    assert last < 0.3 * first, f"loss did not drop: {first} -> {last}"
    # evaluate in TRAIN mode (same batch the model trained on; BN running-stat
    # convergence is Task 6's concern, not this capacity gate)
    pred = model(img4j, use_running_average=False)
    pk = heatmaps_to_keypoints(pred)
    gk = heatmaps_to_keypoints(hmj)
    err = float(mpjpe(pk, gk, visj))
    assert err < 10.0, f"overfit keypoint error too high: {err}px"
