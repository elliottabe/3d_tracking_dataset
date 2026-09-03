"""make_train_step with distractor rows: 2K keypoint rows split into targets
and the other fly's footprint; K rows behave exactly as before."""
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.train.train import TrainConfig, make_optimizer, make_train_step
from jarvis_jax.data.augment import AugParams, build_lr_swap
from jarvis_jax.data.distractor import build_part_index

NAMES = ["EyeL", "EyeR", "T1L_TaTip", "T1R_TaTip"]


def _tiny():
    cfg = ViTPoseConfig(img_size=64, patch=16, in_ch=4, embed_dim=32, depth=1, num_heads=2,
                        mlp_ratio=2, num_keypoints=4, heatmap_size=32)
    return cfg, ViTPose(cfg, rngs=nnx.Rngs(0))


def _batch(B=2, K=4, hm=32, seed=0):
    rng = np.random.RandomState(seed)
    img4 = rng.randint(0, 256, (B, 64, 64, 4), dtype=np.uint8)
    kp = rng.uniform(8, hm - 8, (B, K, 2)).astype(np.float32)
    vis = np.ones((B, K), bool)
    return img4, kp, vis


def test_step_accepts_2k_rows_and_repulsion_raises_loss():
    cfg, model = _tiny()
    tcfg = TrainConfig(total_steps=5, lr=1e-3, warmup_steps=1)
    opt = make_optimizer(model, tcfg)
    part_of_k, _ = build_part_index(NAMES)
    swap = build_lr_swap(NAMES)
    step = make_train_step(0.0, AugParams(enabled=True, cutout_n=0), swap, heatmap_size=32,
                           sigma=2.0, n_keypoints=4, part_of_k=part_of_k,
                           hardneg_k=16, hardneg_weight=0.5, repulsion_weight=1.0)
    img4, kp, vis = _batch()
    d_kp = np.full((2, 4, 2), 24.0, np.float32)            # distractor tips in-crop
    kp2 = np.concatenate([kp, d_kp], 1); vis2 = np.concatenate([vis, np.ones((2, 4), bool)], 1)
    vis2_off = np.concatenate([vis, np.zeros((2, 4), bool)], 1)
    key = jax.random.PRNGKey(0)
    # same model state, same key -> identical aug; repulsion adds a >=0 term
    gd0, st0 = nnx.split(model)
    l_on = float(step(model, opt, key, jnp.asarray(img4), jnp.asarray(kp2), jnp.asarray(vis2)))
    model2 = nnx.merge(gd0, st0); opt2 = make_optimizer(model2, tcfg)
    l_off = float(step(model2, opt2, key, jnp.asarray(img4), jnp.asarray(kp2), jnp.asarray(vis2_off)))
    assert np.isfinite(l_on) and np.isfinite(l_off) and l_on > l_off


def test_step_k_rows_unchanged_with_new_kwargs_off():
    cfg, model = _tiny()
    tcfg = TrainConfig(total_steps=5, lr=1e-3, warmup_steps=1)
    swap = build_lr_swap(NAMES)
    img4, kp, vis = _batch()
    key = jax.random.PRNGKey(1)
    gd, st = nnx.split(model)
    part_of_k, _ = build_part_index(NAMES)
    outs = []
    for kw in ({}, dict(n_keypoints=4, part_of_k=part_of_k)):
        m = nnx.merge(gd, st); o = make_optimizer(m, tcfg)
        step = make_train_step(0.0, AugParams(enabled=True), swap, heatmap_size=32, sigma=2.0, **kw)
        outs.append(float(step(m, o, key, jnp.asarray(img4), jnp.asarray(kp), jnp.asarray(vis))))
    assert outs[0] == outs[1]
