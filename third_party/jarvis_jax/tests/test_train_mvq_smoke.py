# tests/test_train_mvq_smoke.py
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from mvq_fixtures import make_v12_root


def test_two_steps_cpu_and_eval(tmp_path):
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.train_mvq import run_training, MVQTrainConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    root = make_v12_root(tmp_path)
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=2,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone_depth=1, backbone_heads=4, remat=False)
    tcfg = MVQTrainConfig(total_steps=2, batch_size=2, warmup_steps=1, eval_every=2, save_every=2,
                          log_every=1, num_workers=1, pretrained=False, window_lengths=(1, 2), smoke=True)
    res = run_training(root, out_dir=str(tmp_path / "final"), ckpt_dir=str(tmp_path / "ckpt"),
                       mcfg=mcfg, tcfg=tcfg, aug=MVAugParams(enabled=True, blur_max=0.0),
                       weights=LossWeights())
    assert np.isfinite(res["final_loss"])
    assert {"prompted", "unprompted"} <= set(res["val"])
    v = res["val"]["prompted"]
    assert {"mpjpe3d_units", "mpjpe3d_mm", "reproj_px", "cohort_female", "cohort_two_fly", "cohort_group_A"} <= set(v)
    assert os.path.isdir(tmp_path / "final") and os.path.isdir(tmp_path / "ckpt")


def test_empty_cohort_raises(tmp_path):
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.train_mvq import run_training, MVQTrainConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    root = make_v12_root(tmp_path, two_fly_frame=99)        # no two-fly frame anywhere
    mcfg = MVQConfig(embed_dim=32, n_instances=2, n_local=1, n_global=0, dec_layers_3d=2, dec_layers_2d=1,
                     dec_heads=4, backbone_depth=1, backbone_heads=4, remat=False, fourier_bands=2, patch_rgb=3)
    tcfg = MVQTrainConfig(total_steps=1, batch_size=2, pretrained=False, num_workers=1, smoke=True)
    with pytest.raises(ValueError, match="two_fly"):
        run_training(root, out_dir=str(tmp_path / "f"), ckpt_dir=None, mcfg=mcfg, tcfg=tcfg,
                     aug=MVAugParams(enabled=False), weights=LossWeights())


def test_with_ema_does_not_mutate_live_model():
    """flax 0.12.8: nnx.split(model)+nnx.merge(gdef, state) shares the SAME
    Variable objects as `model`, so nnx.update on that merged module would
    silently overwrite the LIVE training model's params too. `_with_ema` must
    `nnx.clone` first. Build a tiny model, snapshot its Param leaves, apply an
    EMA that differs from them (all-zeros), and assert the ORIGINAL model is
    untouched while the returned eval module carries the EMA values."""
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    from jarvis_jax.train.train_mvq import _with_ema
    mcfg = MVQConfig(crop=32, patch=16, embed_dim=16, num_keypoints=4, num_cameras=2, max_frames=2,
                     n_instances=1, n_local=1, n_global=0, dec_layers_3d=1, dec_layers_2d=1, dec_heads=2,
                     mlp_ratio=2.0, refine_passes=0, patch_rgb=3, fourier_bands=1, backbone_depth=1,
                     backbone_heads=2, remat=False)
    model = MVQModel(mcfg, rngs=nnx.Rngs(0))
    before = [np.array(x) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))]
    assert any(np.any(x != 0) for x in before), "fixture is meaningless if the model inits to all zeros"
    ema = jax.tree_util.tree_map(jnp.zeros_like, nnx.state(model, nnx.Param))
    em = _with_ema(model, ema)
    after = [np.array(x) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))]
    assert all(np.array_equal(a, b) for a, b in zip(before, after)), \
        "_with_ema must not mutate the live training model's params"
    em_leaves = jax.tree_util.tree_leaves(nnx.state(em, nnx.Param))
    assert all(np.all(np.asarray(x) == 0) for x in em_leaves), "the returned eval module must carry the EMA values"


def test_resume_from_checkpoint(tmp_path):
    """Resume must restore the EMA too (not re-seed it from raw params): run 2
    steps saving every step, then relaunch (fresh model/optimizer/EMA, as a
    real preempt+requeue would) with total_steps=3 against the SAME ckpt_dir
    and confirm it picks up at step 2."""
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.train_mvq import run_training, MVQTrainConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    root = make_v12_root(tmp_path)
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=2,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone_depth=1, backbone_heads=4, remat=False)
    ckpt_dir = str(tmp_path / "ckpt")
    common = dict(batch_size=2, warmup_steps=1, eval_every=100, save_every=1, log_every=1,
                 num_workers=1, pretrained=False, window_lengths=(1,), smoke=True)
    tcfg1 = MVQTrainConfig(total_steps=2, **common)
    res1 = run_training(root, out_dir=str(tmp_path / "final1"), ckpt_dir=ckpt_dir, mcfg=mcfg, tcfg=tcfg1,
                        aug=MVAugParams(enabled=False), weights=LossWeights())
    assert res1["resumed_from"] == 0
    tcfg2 = MVQTrainConfig(total_steps=3, **common)
    res2 = run_training(root, out_dir=str(tmp_path / "final2"), ckpt_dir=ckpt_dir, mcfg=mcfg, tcfg=tcfg2,
                        aug=MVAugParams(enabled=False), weights=LossWeights())
    assert res2["resumed_from"] == 2
    assert np.isfinite(res2["final_loss"])
