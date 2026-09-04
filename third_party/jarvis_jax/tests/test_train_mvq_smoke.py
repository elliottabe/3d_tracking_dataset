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
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=4,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1, backbone_heads=4, remat=False)
    tcfg = MVQTrainConfig(total_steps=2, batch_size=2, warmup_steps=1, eval_every=2, save_every=2,
                          log_every=1, num_workers=1, pretrained=False, window_lengths=(1, 2), smoke=True)
    res = run_training(root, out_dir=str(tmp_path / "final"), ckpt_dir=str(tmp_path / "ckpt"),
                       mcfg=mcfg, tcfg=tcfg, aug=MVAugParams(enabled=True, blur_max=0.0),
                       weights=LossWeights())
    assert np.isfinite(res["final_loss"])
    assert {"prompted", "unprompted"} <= set(res["val"])
    for mode in ("prompted", "unprompted"):
        v = res["val"][mode]
        assert {"mpjpe3d_units", "mpjpe3d_mm", "reproj_px", "cohort_female", "cohort_two_fly", "cohort_group_A",
                "mpjpe3d_policy_units", "mpjpe3d_policy_mm", "policy_miss_frac"} <= set(v)
        assert 0.0 <= v["policy_miss_frac"] <= 1.0
    assert os.path.isdir(tmp_path / "final") and os.path.isdir(tmp_path / "ckpt")


def test_load_mvq_model_ckpt_matches_final(tmp_path):
    """load_mvq_model must agree on the same run's two restore paths: `final/`
    (the EMA `run_training` already debiased before saving) and `ckpt/<last
    step>` (the raw EMA sum, debiased HERE by load_mvq_model using ema_meta's
    ema_updates and the run's own train.ema decay) -- same math, same
    numbers, so a figure/benchmark script reading from a mid-training
    checkpoint (no `final/` yet) gets the identical answer a completed run's
    `final/` would."""
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.models.mvq.checkpoint import load_mvq_model
    from jarvis_jax.train.train_mvq import run_training, MVQTrainConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    root = make_v12_root(tmp_path)
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=4,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1,
                     backbone_heads=4, remat=False)
    tcfg = MVQTrainConfig(total_steps=2, batch_size=2, warmup_steps=1, eval_every=2, save_every=2,
                          log_every=1, num_workers=1, pretrained=False, window_lengths=(1,), smoke=True)
    run_training(root, out_dir=str(tmp_path / "final"), ckpt_dir=str(tmp_path / "ckpt"),
                mcfg=mcfg, tcfg=tcfg, aug=MVAugParams(enabled=False), weights=LossWeights())
    m_final, meta_final = load_mvq_model(str(tmp_path / "final"))
    m_ckpt, meta_ckpt = load_mvq_model(str(tmp_path), step=2)
    assert meta_final["model"] == meta_ckpt["model"]
    pf = jax.tree_util.tree_leaves(nnx.state(m_final, nnx.Param))
    pc = jax.tree_util.tree_leaves(nnx.state(m_ckpt, nnx.Param))
    assert len(pf) == len(pc) and len(pf) > 0
    for a, c in zip(pf, pc):
        np.testing.assert_allclose(np.asarray(a), np.asarray(c), atol=1e-5)
    # "latest" resolves the same as the explicit last step
    m_latest, _ = load_mvq_model(str(tmp_path), step="latest")
    pl = jax.tree_util.tree_leaves(nnx.state(m_latest, nnx.Param))
    for a, l in zip(pf, pl):
        np.testing.assert_allclose(np.asarray(a), np.asarray(l), atol=1e-5)


def test_empty_cohort_raises(tmp_path):
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.train_mvq import run_training, MVQTrainConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    root = make_v12_root(tmp_path, two_fly_frame=99)        # no two-fly frame anywhere
    mcfg = MVQConfig(embed_dim=32, n_instances=4, n_local=1, n_global=0, dec_layers_3d=2, dec_layers_2d=1,
                     dec_heads=4, backbone="tiny", backbone_depth=1, backbone_heads=4, remat=False, fourier_bands=2, patch_rgb=3)
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
                     mlp_ratio=2.0, refine_passes=0, patch_rgb=3, fourier_bands=1, backbone="tiny", backbone_depth=1,
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
    and confirm it picks up at step 2. Also confirms the ema_updates counter
    (persisted in the new `ema_meta` Orbax item) round-trips: it must equal
    the completed step count both before and after resume, since the EMA
    updates exactly once per training step with no skips."""
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.train_mvq import run_training, MVQTrainConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    root = make_v12_root(tmp_path)
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=4,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1, backbone_heads=4, remat=False)
    ckpt_dir = str(tmp_path / "ckpt")
    common = dict(batch_size=2, warmup_steps=1, eval_every=100, save_every=1, log_every=1,
                 num_workers=1, pretrained=False, window_lengths=(1,), smoke=True)
    tcfg1 = MVQTrainConfig(total_steps=2, **common)
    res1 = run_training(root, out_dir=str(tmp_path / "final1"), ckpt_dir=ckpt_dir, mcfg=mcfg, tcfg=tcfg1,
                        aug=MVAugParams(enabled=False), weights=LossWeights())
    assert res1["resumed_from"] == 0
    assert res1["ema_updates"] == 2
    tcfg2 = MVQTrainConfig(total_steps=3, **common)
    res2 = run_training(root, out_dir=str(tmp_path / "final2"), ckpt_dir=ckpt_dir, mcfg=mcfg, tcfg=tcfg2,
                        aug=MVAugParams(enabled=False), weights=LossWeights())
    assert res2["resumed_from"] == 2
    assert res2["ema_updates"] == 3
    assert np.isfinite(res2["final_loss"])


def test_resume_refuses_checkpoint_without_ema_meta(tmp_path):
    """A checkpoint saved without the `ema_meta` item (i.e. by pre-fix-round-2
    code) must be REFUSED on resume with a ValueError naming the checkpoint
    dir, not silently resumed as though ema_zero_seeded were True -- see
    _restore_latest's docstring for why guessing here is unsafe (a pre-fix
    EMA seeded from live params would be silently rescaled by the debiasing
    divisor as though it needed the same correction, which it does not)."""
    import orbax.checkpoint as ocp
    from flax import nnx
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    from jarvis_jax.train.train_mvq import _restore_latest, make_optimizer, MVQTrainConfig
    mcfg = MVQConfig(crop=32, patch=16, embed_dim=16, num_keypoints=4, num_cameras=2, max_frames=2,
                     n_instances=1, n_local=1, n_global=0, dec_layers_3d=1, dec_layers_2d=1, dec_heads=2,
                     mlp_ratio=2.0, refine_passes=0, patch_rgb=3, fourier_bands=1, backbone="tiny", backbone_depth=1,
                     backbone_heads=2, remat=False)
    model = MVQModel(mcfg, rngs=nnx.Rngs(0))
    opt = make_optimizer(model, MVQTrainConfig())
    ema = jax.tree_util.tree_map(lambda p: p, nnx.state(model, nnx.Param))   # OLD (pre-fix) seeding: from params
    ckpt_dir = str(tmp_path / "ckpt_old_format")
    import os
    os.makedirs(ckpt_dir, exist_ok=True)
    # Old-format manager: only 3 items (no ema_meta), matching pre-fix-round-2 code.
    old_mngr = ocp.CheckpointManager(os.path.abspath(ckpt_dir),
                                     options=ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=1),
                                     item_names=("model", "opt", "ema"))
    old_mngr.save(5, args=ocp.args.Composite(
        model=ocp.args.StandardSave(nnx.split(model)[1]),
        opt=ocp.args.StandardSave(nnx.split(opt)[1]),
        ema=ocp.args.StandardSave(ema)))
    old_mngr.wait_until_finished()

    from jarvis_jax.train.train_mvq import _make_manager
    new_mngr = _make_manager(ckpt_dir)
    with pytest.raises(ValueError, match="ema_meta"):
        _restore_latest(new_mngr, model, opt, ema)


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
                     mlp_ratio=2.0, refine_passes=0, patch_rgb=3, fourier_bands=1, backbone="tiny", backbone_depth=1,
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
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=4,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1, backbone_heads=4, remat=False)
    model = MVQModel(mcfg, rngs=nnx.Rngs(0))
    part_of_k, _ = build_part_index(ds.keypoint_names)
    mesh = data_parallel_mesh()
    cohorts = _cohorts(ds)
    assert cohorts["female"].any() and cohorts["two_fly"].any()      # non-degenerate fixture
    kw = dict(cohorts=cohorts, part_of_k=part_of_k, weights=LossWeights(), mesh=mesh, num_workers=1)
    val3 = evaluate(model, ds, 3, **kw)
    val2 = evaluate(model, ds, 2, **kw)
    # mpjpe3d_units/cohort_* are computed in plain numpy from independent per-sample
    # arrays -- insensitive to batch shape, so held to a tight 1e-6. reproj_px/uv2d_px/
    # head_vs_reproj_px come from mvq_loss's own JAX batch reductions (huber means over
    # a (B,...)-shaped tensor), which legitimately differ at the ~1e-6 ABSOLUTE level
    # between batch_size=3 and 2 from ordinary floating-point non-associativity (measured
    # ~4.8e-6 here, ~4e-8 RELATIVE) -- real, harmless numerical noise, not a batch-grouping
    # dependence bug, so those three get a looser (still tight) 1e-4 tolerance.
    tight = {"mpjpe3d_units", "cohort_female", "cohort_two_fly"}
    for mode in ("prompted", "unprompted"):
        for k in ("mpjpe3d_units", "cohort_female", "cohort_two_fly",
                  "reproj_px", "uv2d_px", "head_vs_reproj_px"):
            a, b = val3[mode][k], val2[mode][k]
            if np.isnan(a) and np.isnan(b):
                continue
            tol = 1e-6 if k in tight else 1e-4
            assert abs(a - b) < tol, (mode, k, a, b)
