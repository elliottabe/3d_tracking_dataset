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
    em = _with_ema(model, ema, 0.999, 1)   # any decay/t>0: 0/(1-decay**t) is still 0
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


def test_ema_debias_matches_constant_params():
    """After n EMA updates (seeded from zero) on a model whose params never
    change (constant p), the debiased EMA must equal p to 1e-6 for n in
    {1, 10, 1000} -- the standard Adam-style bias-correction identity: a
    zero-seeded EMA of a CONSTANT p after n steps is exactly p*(1-decay**n),
    so dividing by (1-decay**n) must recover p regardless of n. Regression
    test for the EMA-bias bug: before this fix, seeding from params (not
    zero) meant a near-zero step-0 init still carried decay**t of the
    average forever -- measured on the gate-1 checkpoint as every predicted
    joint shrinking toward the crop centre, worse for distal legs (radial
    dist/GT ratio 0.68-0.83, furthest from centre) than body joints
    (0.86-0.93)."""
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    from jarvis_jax.train.train_mvq import _with_ema
    mcfg = MVQConfig(crop=32, patch=16, embed_dim=16, num_keypoints=4, num_cameras=2, max_frames=2,
                     n_instances=1, n_local=1, n_global=0, dec_layers_3d=1, dec_layers_2d=1, dec_heads=2,
                     mlp_ratio=2.0, refine_passes=0, patch_rgb=3, fourier_bands=1, backbone_depth=1,
                     backbone_heads=2, remat=False)
    model = MVQModel(mcfg, rngs=nnx.Rngs(0))
    p = jax.tree_util.tree_map(lambda x: jnp.full_like(x, 3.7), nnx.state(model, nnx.Param))
    nnx.update(model, p)
    decay = 0.999
    for n in (1, 10, 1000):
        # ema_n = p*(1-decay**n) exactly for a CONSTANT p (closed form) -- computed
        # directly rather than via an n-iteration float32 loop, whose accumulated
        # rounding (measured ~3e-5 at n=1000) would swamp the 1e-6 tolerance this
        # test is checking the DEBIASING FORMULA against, not float32 summation.
        ema = jax.tree_util.tree_map(lambda pp: pp * (1.0 - decay ** n), p)
        em = _with_ema(model, ema, decay, n)
        for leaf in jax.tree_util.tree_leaves(nnx.state(em, nnx.Param)):
            np.testing.assert_allclose(np.asarray(leaf), 3.7, atol=1e-6)


def test_evaluate_ragged_batch_matches_full_batch(tmp_path):
    """4 val windows (make_v12_root's default fixture); batch_size=3 (a clean
    batch of 3 + a ragged batch of 1, padded up to 3) must give the SAME
    per-mode mpjpe3d_units/cohort means as batch_size=2 (two clean batches of
    2, no padding needed) -- the eval-sharding padding fix (2026-09-03) must
    not let how samples happen to fall into batches change the reported
    numbers."""
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    from jarvis_jax.train.train_mvq import evaluate, _cohorts
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.distractor import build_part_index
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.sharding import data_parallel_mesh
    root = make_v12_root(tmp_path)
    ds = V12WindowDataset(root, "val", T=1, train=False)
    assert len(ds) == 4
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=2,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone_depth=1, backbone_heads=4, remat=False)
    model = MVQModel(mcfg, rngs=nnx.Rngs(0))
    part_of_k, _ = build_part_index(ds.keypoint_names)
    mesh = data_parallel_mesh()
    cohorts = _cohorts(ds)
    assert cohorts["female"].any() and cohorts["two_fly"].any()      # non-degenerate fixture
    kw = dict(cohorts=cohorts, part_of_k=part_of_k, weights=LossWeights(), mesh=mesh, num_workers=1)
    val3 = evaluate(model, ds, 3, **kw)
    val2 = evaluate(model, ds, 2, **kw)
    for mode in ("prompted", "unprompted"):
        for k in ("mpjpe3d_units", "cohort_female", "cohort_two_fly"):
            a, b = val3[mode][k], val2[mode][k]
            if np.isnan(a) and np.isnan(b):
                continue
            assert abs(a - b) < 1e-6, (mode, k, a, b)
