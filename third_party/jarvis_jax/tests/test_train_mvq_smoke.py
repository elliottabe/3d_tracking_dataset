# tests/test_train_mvq_smoke.py
import dataclasses
import json
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from mvq_fixtures import REC, make_v12_root


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
                "mpjpe3d_policy_units", "mpjpe3d_policy_mm", "policy_miss_frac", "cohort_contact_pair"} <= set(v)
        assert 0.0 <= v["policy_miss_frac"] <= 1.0
        assert {"sex_acc", "mask_containment", "cohort_contact_pair"} <= set(v)
        assert {f"exist_prec_slot{i}" for i in range(4)} | {f"exist_rec_slot{i}" for i in range(4)} <= set(v)
        assert np.isnan(v["mask_containment"]) or 0.0 <= v["mask_containment"] <= 1.0
        # the three acceptance metrics PER COHORT, incl. the new single_fly cohort
        assert {"policy_miss_frac_single_fly", "sex_acc_group_A", "mask_containment_two_fly",
                "cohort_single_fly"} <= set(v)
        assert np.isnan(v["policy_miss_frac_single_fly"]) or 0.0 <= v["policy_miss_frac_single_fly"] <= 1.0
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
    # a run whose architecture MATCHES the current code restores every leaf
    assert meta_final["_unrestored_leaves"] == [] and meta_ckpt["_unrestored_leaves"] == []
    # "latest" resolves the same as the explicit last step
    m_latest, _ = load_mvq_model(str(tmp_path), step="latest")
    pl = jax.tree_util.tree_leaves(nnx.state(m_latest, nnx.Param))
    for a, l in zip(pf, pl):
        np.testing.assert_allclose(np.asarray(a), np.asarray(l), atol=1e-5)


def test_load_mvq_model_tolerates_pre_p3a_checkpoint(tmp_path):
    """`load_mvq_model` must OPEN a pre-P3a checkpoint, not raise on it.

    The real warm-start/baseline source (`mvq_t1_b16_local8_20260904/final`,
    606 leaves) predates the sex head and has `n_instances = 3`; today's
    model has 608 leaves and 4 slots. Building the restore target from the
    fresh model -- the old behaviour -- raised, so no figure or benchmark
    script could measure the very baseline that checkpoint IS the baseline
    for. The source here emulates it exactly: a 3-slot model whose
    `decoder/heads/sex` subtree is DELETED before saving (so those leaves are
    absent, not merely differently shaped), with an `mvq_run.json` that asks
    for 4 slots. Expected unrestored list: `decoder/e_inst (partial rows
    0:3)` plus the two sex-head leaves -- exactly the three the P3a spec §8
    names for the real checkpoint."""
    import orbax.checkpoint as ocp
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    from jarvis_jax.models.mvq.checkpoint import load_mvq_model
    kw = dict(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7,
              n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
              refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1,
              backbone_heads=4, remat=False)
    src = MVQModel(MVQConfig(n_instances=3, **kw), rngs=nnx.Rngs(0))
    src.decoder.e_inst.value = src.decoder.e_inst.value + 1.0
    pure = nnx.to_pure_dict(nnx.split(src)[1])
    del pure["decoder"]["heads"]["sex"]                 # a source that predates the sex head
    final = tmp_path / "final"
    ck = ocp.StandardCheckpointer(); ck.save(str(final), pure); ck.wait_until_finished()
    json.dump({"model": dict(n_instances=4, **kw), "train": {"ema": 0.999}, "val": None,
              "keypoint_names": [f"k{i}" for i in range(50)]},
              open(final / "mvq_run.json", "w"))

    model, meta = load_mvq_model(str(final))
    assert sorted(meta["_unrestored_leaves"]) == sorted(
        ["decoder/e_inst (partial rows 0:3)", "decoder/heads/sex/bias", "decoder/heads/sex/kernel"])
    # the three source slots landed in rows 0-2; row 3 and the sex head stay fresh
    fresh = MVQModel(MVQConfig(n_instances=4, **kw), rngs=nnx.Rngs(0))
    np.testing.assert_allclose(np.asarray(model.decoder.e_inst.value[:3]),
                               np.asarray(src.decoder.e_inst.value))
    np.testing.assert_allclose(np.asarray(model.decoder.e_inst.value[3]),
                               np.asarray(fresh.decoder.e_inst.value[3]))
    np.testing.assert_allclose(np.asarray(model.decoder.heads.sex.kernel.value),
                               np.asarray(fresh.decoder.heads.sex.kernel.value))
    # a matching-architecture source restores everything (no false skips)
    full = tmp_path / "final_full"
    ck.save(str(full), nnx.split(fresh)[1]); ck.wait_until_finished()
    json.dump({"model": dict(n_instances=4, **kw), "train": {"ema": 0.999}, "val": None,
              "keypoint_names": [f"k{i}" for i in range(50)]},
              open(full / "mvq_run.json", "w"))
    _, meta_full = load_mvq_model(str(full))
    assert meta_full["_unrestored_leaves"] == []


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


def test_run_dir_json_written_at_start_and_loader_prefers_it(tmp_path):
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
    tcfg = MVQTrainConfig(total_steps=1, batch_size=2, warmup_steps=1, eval_every=1, save_every=1,
                          log_every=1, num_workers=1, pretrained=False, window_lengths=(1,), smoke=True)
    run = tmp_path / "run"
    run_training(root, out_dir=str(run / "final"), ckpt_dir=str(run / "ckpt"), mcfg=mcfg, tcfg=tcfg,
                 aug=MVAugParams(enabled=False), weights=LossWeights())
    meta = json.load(open(run / "mvq_run.json"))
    assert meta["val"] is None and meta["model"]["n_instances"] == 4 and len(meta["keypoint_names"]) == 50
    # a mid-run checkpoint loads with NO final/ present
    import shutil; shutil.rmtree(run / "final")
    m, meta2 = load_mvq_model(str(run), step="latest")
    assert meta2["model"] == meta["model"]


def test_warm_start_config_seeds_step0_from_another_run(tmp_path):
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.train_mvq import run_training, MVQTrainConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    root = make_v12_root(tmp_path)
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=4,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1,
                     backbone_heads=4, remat=False)
    base = MVQTrainConfig(total_steps=1, batch_size=2, warmup_steps=1, eval_every=1, save_every=1,
                          log_every=1, num_workers=1, pretrained=False, window_lengths=(1,), smoke=True)
    a = tmp_path / "a"
    run_training(root, out_dir=str(a / "final"), ckpt_dir=str(a / "ckpt"), mcfg=mcfg, tcfg=base,
                 aug=MVAugParams(enabled=False), weights=LossWeights())
    b = tmp_path / "b"
    res = run_training(root, out_dir=str(b / "final"), ckpt_dir=str(b / "ckpt"), mcfg=mcfg,
                       tcfg=dataclasses.replace(base, warm_start=str(a / "final")),
                       aug=MVAugParams(enabled=False), weights=LossWeights())
    assert res["resumed_from"] == 0 and np.isfinite(res["final_loss"])


def test_warm_start_is_skipped_when_the_run_already_has_a_checkpoint(tmp_path, capsys):
    """A requeue must not re-read the warm-start source only to discard it:
    resume beats warm start, so once this run's OWN ckpt/ has a step the
    source checkpoint is never opened (tens of GB of Orbax reads on every
    preemption of a long fine-tune otherwise)."""
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.train_mvq import run_training, MVQTrainConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    root = make_v12_root(tmp_path)
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=4,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1,
                     backbone_heads=4, remat=False)
    base = MVQTrainConfig(batch_size=2, warmup_steps=1, eval_every=100, save_every=1, log_every=1,
                          num_workers=1, pretrained=False, window_lengths=(1,), smoke=True)
    src = tmp_path / "src"
    run_training(root, out_dir=str(src / "final"), ckpt_dir=str(src / "ckpt"), mcfg=mcfg,
                 tcfg=dataclasses.replace(base, total_steps=1),
                 aug=MVAugParams(enabled=False), weights=LossWeights())
    ft = tmp_path / "ft"
    warm = dataclasses.replace(base, total_steps=1, warm_start=str(src / "final"))
    run_training(root, out_dir=str(ft / "final"), ckpt_dir=str(ft / "ckpt"), mcfg=mcfg, tcfg=warm,
                 aug=MVAugParams(enabled=False), weights=LossWeights())
    out = capsys.readouterr().out
    assert "warm start from" in out and "SKIPPED" not in out          # first launch: warm start ran
    # requeue: same ckpt_dir, one more step -> resume, and the source is not re-read
    res = run_training(root, out_dir=str(ft / "final"), ckpt_dir=str(ft / "ckpt"), mcfg=mcfg,
                       tcfg=dataclasses.replace(warm, total_steps=2),
                       aug=MVAugParams(enabled=False), weights=LossWeights())
    out2 = capsys.readouterr().out
    assert res["resumed_from"] == 1
    assert "SKIPPED" in out2 and "restored" not in out2


def test_train_dataset_receives_configured_jitter_units(tmp_path, monkeypatch):
    """`MVQTrainConfig.jitter_units` must reach the TRAIN `V12WindowDataset`(s)
    `run_training` builds (line ~479), not just sit unused on the config --
    P4 spec section 6's conditional retrain raises it from the dataset
    default (3.0, 0.3 mm) to 10.0 (1 mm) to make the model robust to the
    coarse-pass's off-centre windows. Wrap the real dataset class to record
    every `jitter_units` kwarg it is constructed with, run a 1-step tiny
    training with `jitter_units=7.5`, and assert that value (not the
    dataset's own default) was passed for the train split(s)."""
    import jarvis_jax.train.train_mvq as train_mvq
    from jarvis_jax.data.v12_windows import V12WindowDataset as RealV12WindowDataset
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.train_mvq import run_training, MVQTrainConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams

    seen = []

    class RecordingV12WindowDataset(RealV12WindowDataset):
        def __init__(self, *a, **kw):
            seen.append({"train": kw.get("train"), "jitter_units": kw.get("jitter_units")})
            super().__init__(*a, **kw)

    monkeypatch.setattr(train_mvq, "V12WindowDataset", RecordingV12WindowDataset)

    root = make_v12_root(tmp_path)
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=4,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1,
                     backbone_heads=4, remat=False)
    tcfg = MVQTrainConfig(total_steps=1, batch_size=2, warmup_steps=1, eval_every=1, save_every=1,
                          log_every=1, num_workers=1, pretrained=False, window_lengths=(1,), smoke=True,
                          jitter_units=7.5)
    run_training(root, out_dir=str(tmp_path / "final"), ckpt_dir=str(tmp_path / "ckpt"),
                mcfg=mcfg, tcfg=tcfg, aug=MVAugParams(enabled=False), weights=LossWeights())

    train_calls = [c for c in seen if c["train"] is True]
    assert train_calls, "no train-split V12WindowDataset was constructed"
    for c in train_calls:
        assert c["jitter_units"] == tcfg.jitter_units, (
            f"train dataset got jitter_units={c['jitter_units']!r}, expected {tcfg.jitter_units!r} "
            f"(the dataset's own default is 3.0 -- run_training must pass tcfg.jitter_units explicitly)")
    # the val split is unaffected: it is built with train=False and no jitter kwarg
    val_calls = [c for c in seen if c["train"] is False]
    assert val_calls and all(c["jitter_units"] is None for c in val_calls)


def test_female_host_weight_balances_the_sampled_host_sexes():
    """P3b `train.female_host_weight` on a FAKE manifest with the real export's
    imbalance (202 female-host windows of 2661, two behaviours).

    Expectation: 1.0 leaves `_balanced_weights` bit-identical to the P2 call
    (no silent change to any earlier run), and the documented multiplier
    `mass_male / mass_female` computed at 1.0 makes the two host sexes exactly
    equally likely -- which is the property the launch value is chosen for.
    """
    import numpy as np
    from jarvis_jax.train.train_mvq import _balanced_weights

    N_F, N_M = 202, 2459

    class FakeDS:
        """The only surface `_balanced_weights` touches: len, windows, manifest, is_female."""
        def __init__(self):
            self.windows = [(f"rec{i % 2}", 0, i) for i in range(N_F + N_M)]
            self.manifest = {"rec0": {"behavior": "courtship"}, "rec1": {"behavior": "walking"}}
            self._f = np.zeros(N_F + N_M, bool); self._f[:N_F] = True
        def __len__(self):
            return len(self.windows)
        def is_female(self, i):
            return bool(self._f[i])

    ds = FakeDS(); is_f = ds._f
    w1 = _balanced_weights(ds, 0.5, 1.0)
    np.testing.assert_allclose(_balanced_weights(ds, 0.5, 1.0, 1.0), w1, rtol=0, atol=0)
    mult = w1[~is_f].sum() / w1[is_f].sum()
    assert mult > 1.0, "the fake manifest must be female-scarce for this test to mean anything"

    w = _balanced_weights(ds, 0.5, 1.0, mult)
    assert abs(w.sum() - 1.0) < 1e-12
    assert abs(w[is_f].sum() - 0.5) < 1e-9 and abs(w[~is_f].sum() - 0.5) < 1e-9
    # and it is a pure multiplier on the female rows: their relative order is untouched
    np.testing.assert_allclose(w[is_f] / w[is_f].sum(), w1[is_f] / w1[is_f].sum(), rtol=1e-12)
    # a 2x weight moves the mass exactly 2x relative to male/other
    w2 = _balanced_weights(ds, 0.5, 1.0, 2.0)
    assert abs((w2[is_f].sum() / w2[~is_f].sum()) / (w1[is_f].sum() / w1[~is_f].sum()) - 2.0) < 1e-9


def test_p3b_knobs_reach_the_dataset_and_the_copy_paste_params(tmp_path, monkeypatch):
    """Config plumbing for the three P3b loader-side knobs: the contact-heavy
    `copy_paste_contact_sep`, the `sex_label_overrides` map and (with them)
    `copy_paste_p`/`copy_paste_contact_p` must reach the objects that consume
    them, not just sit on `MVQTrainConfig`. Same wrapping trick as
    `test_train_dataset_receives_configured_jitter_units`."""
    import dataclasses
    import jarvis_jax.train.train_mvq as train_mvq
    from jarvis_jax.data.v12_windows import V12WindowDataset as RealV12WindowDataset
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.train_mvq import run_training, MVQTrainConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    from mvq_fixtures import REC

    seen = []

    class RecordingV12WindowDataset(RealV12WindowDataset):
        def __init__(self, *a, **kw):
            seen.append({"train": kw.get("train"), "copy_paste": kw.get("copy_paste"),
                         "sex_overrides": kw.get("sex_overrides")})
            super().__init__(*a, **kw)

    monkeypatch.setattr(train_mvq, "V12WindowDataset", RecordingV12WindowDataset)

    root = make_v12_root(tmp_path)
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=4,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1,
                     backbone_heads=4, remat=False)
    tcfg = MVQTrainConfig(total_steps=1, batch_size=2, warmup_steps=1, eval_every=1, save_every=1,
                          log_every=1, num_workers=1, pretrained=False, window_lengths=(1,), smoke=True,
                          copy_paste_p=0.8, copy_paste_contact_p=0.7, copy_paste_contact_sep=(4.0, 25.0),
                          female_host_weight=4.27, sex_label_overrides={REC: {"fly1": "female"}})
    run_training(root, out_dir=str(tmp_path / "final"), ckpt_dir=str(tmp_path / "ckpt"),
                 mcfg=mcfg, tcfg=tcfg, aug=MVAugParams(enabled=False), weights=LossWeights())

    train_calls = [c for c in seen if c["train"] is True]
    assert train_calls
    for c in train_calls:
        cp = c["copy_paste"]
        assert cp is not None and cp.p == 0.8 and cp.contact_p == 0.7
        assert tuple(cp.contact_sep) == (4.0, 25.0), (
            f"contact_sep={cp.contact_sep!r}; CopyPasteParams' own default is (8.0, 30.0), so "
            f"run_training must pass tcfg.copy_paste_contact_sep explicitly")
        assert c["sex_overrides"] == {REC: {"fly1": "female"}}
    # the val split gets the overrides too (a val recording must resolve the same way)
    assert all(c["sex_overrides"] == {REC: {"fly1": "female"}} for c in seen if c["train"] is False)
    # copy_paste stays None when the rate is 0 (the P3a/P2 default path)
    seen.clear()
    run_training(root, out_dir=str(tmp_path / "f2"), ckpt_dir=str(tmp_path / "c2"), mcfg=mcfg,
                 tcfg=dataclasses.replace(tcfg, copy_paste_p=0.0, sex_label_overrides={}),
                 aug=MVAugParams(enabled=False), weights=LossWeights())
    assert seen and all(c["copy_paste"] is None for c in seen)


# ---------------------------------------------------------------- mvq v2 (T=2)
# spec 2026-09-05 §4: T=2 pairs at Delta in {1,4,16}, trained from scratch on
# the human v12 root + a pseudo-label export (weight 0.3) + a single-fly export
# (0.3) + empty-window negatives (5 % of the sampled mass).

def _rename_recording(root, old, new):
    """Rename the one recording in a v12 fixture root (images, masks, image
    `file_name`s, frameset keys, manifest). The single-fly export really is a
    DIFFERENT recording (free_running Session11 / Clip Session6, spec §3.4),
    and `ConcatWindowDataset` refuses two roots that describe the same
    recording differently -- so a copy that kept the name would not be
    testing the real layout."""
    import shutil
    for sub in ("images", "masks"):
        a = os.path.join(root, sub, old)
        if os.path.isdir(a):
            shutil.move(a, os.path.join(root, sub, new))
    for split in ("train", "val"):
        p = os.path.join(root, "annotations", f"instances_{split}.json")
        coco = json.load(open(p))
        for im in coco["images"]:
            im["file_name"] = im["file_name"].replace(f"{old}/", f"{new}/", 1)
            if im.get("recording") == old:
                im["recording"] = new
        coco["framesets"] = {k.replace(f"{old}/", f"{new}/", 1): {**v, "recording": new}
                             for k, v in coco["framesets"].items()}
        json.dump(coco, open(p, "w"))
    m = json.load(open(os.path.join(root, "manifest.json")))
    m["recordings"] = {(new if r == old else r): v for r, v in m["recordings"].items()}
    json.dump(m, open(os.path.join(root, "manifest.json"), "w"))


def _pseudo_copy(src, dst, *, weight=0.3, source="pseudo", role="anchor", keep_flies=None,
                 rename_rec=None):
    """A pseudo-flavoured copy of a v12 fixture root: the whole-manifest
    `source`/`weight`/`role` defaults the real export writes, read back by
    `V12WindowDataset.source/weight/role` (frameset -> per-recording manifest
    -> whole-manifest default). `keep_flies` (e.g. `{0}`) drops the other
    flies' framesets and `rename_rec` renames the recording, which is how the
    SINGLE-FLY root differs from the courtship one."""
    import shutil
    shutil.copytree(src, dst)
    if rename_rec is not None:
        _rename_recording(str(dst), REC, rename_rec)
    if keep_flies is not None:
        for split in ("train", "val"):
            p = os.path.join(dst, "annotations", f"instances_{split}.json")
            coco = json.load(open(p))
            coco["framesets"] = {k: v for k, v in coco["framesets"].items()
                                 if int(v["fly_id"]) in keep_flies}
            json.dump(coco, open(p, "w"))
    m = json.load(open(os.path.join(dst, "manifest.json")))
    m["source"], m["weight"], m["role"] = source, float(weight), role
    if keep_flies is not None:
        for rec in m["recordings"].values():
            rec["n_flies"] = len(keep_flies)     # else the loader claims a present-but-unlabelled fly
    json.dump(m, open(os.path.join(dst, "manifest.json"), "w"))
    return str(dst)


def _negatives_copy(src, dst, *, deltas=(1, 4)):
    """A NEGATIVES-only copy (spec §3.5): every frameset replaced by an empty
    window keyed `<rec>/Frame_<n>/neg0` with `fly_id: -1`, its own `center3D`
    and a `partners` map (delta -> the partner's ABSOLUTE frame) over the SAME
    images. `V12WindowDataset._build` discards a negative's keypoints, so
    reusing fly0's resolved slots is what the real export's placeholder rows
    amount to; without `partners` a negative contributes NO T > 1 window at all
    (the loader refuses to freeze a pair), which is the mistake this helper
    exists to not make. Weight stays 1.0: a negative exists to supervise the
    existence head and `sample_weight` multiplies every term, so a
    down-weighted negative would assert nothing."""
    import shutil
    shutil.copytree(src, dst)
    for split in ("train", "val"):
        p = os.path.join(dst, "annotations", f"instances_{split}.json")
        coco = json.load(open(p))
        frames = sorted(int(k.split("/")[1].split("_")[1]) for k, v in coco["framesets"].items()
                        if int(v["fly_id"]) == 0)
        neg = {}
        for key, fsv in coco["framesets"].items():
            rec, frame, tail = key.split("/")
            if tail != "fly0":
                continue
            f = int(frame.split("_")[1])
            neg[f"{rec}/{frame}/neg0"] = {
                **fsv, "fly_id": -1, "subset": "neg", "center3D": [0.0, 0.0, 0.0],
                "partners": {str(d): f + d for d in deltas if f + d in frames}}
        coco["framesets"] = neg
        json.dump(coco, open(p, "w"))
    m = json.load(open(os.path.join(dst, "manifest.json")))
    # 0.3, NOT 1.0: the real negatives export ships the pseudo-label weight, and
    # `run_training` is the thing that must force it back to 1.0 for training.
    m["source"], m["weight"], m["role"] = "pseudo", 0.3, "negative"
    json.dump(m, open(os.path.join(dst, "manifest.json"), "w"))
    return str(dst)


def _v2_mcfg():
    from jarvis_jax.models.mvq import MVQConfig
    return MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=4,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1,
                     backbone_heads=4, remat=False)


def test_t2_run_trains_and_mixes_roots(tmp_path, capsys):
    """The v2 recipe end to end on the tiny fixture: T=2 pairs at Delta in
    {1, 4}, a pseudo root and a single-fly root at weight 0.3, an
    empty-window negatives root at 5 % of the sampled mass, wing keypoints
    at 2x.

    Expectation if the wiring is right: a finite loss; `final/mvq_run.json`'s
    `train` block records the v2 knobs (Task 6/7 read them for the scorecard
    header); validation still runs on the REAL root only (spec §7) and
    reports a non-NaN unprompted mpjpe; the sampler's mass table names all
    four roots with the negatives at exactly `negatives_frac`; and the
    realised mix COUNTS windows drawn per root -- a root whose mass is set
    but which never actually gets drawn is the failure this catches."""
    from jarvis_jax.data.mv_augment import MVAugParams
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.train.train_mvq import MVQTrainConfig, run_training
    (tmp_path / "real").mkdir()
    real = make_v12_root(tmp_path / "real", n_frames=6)
    pseudo = _pseudo_copy(real, tmp_path / "pseudo")
    single = _pseudo_copy(real, tmp_path / "singlefly", keep_flies={0},
                          rename_rec="2026_02_02_00_00_00")
    negs = _negatives_copy(real, tmp_path / "negatives")
    tcfg = MVQTrainConfig(total_steps=2, batch_size=2, warmup_steps=1, eval_every=2, save_every=2,
                          log_every=1, num_workers=1, pretrained=False, window_lengths=(1, 2),
                          pair_deltas=(1, 4), pseudo_root=pseudo, pseudo_weight=0.3,
                          singlefly_root=single, negatives_root=negs, negatives_frac=0.05,
                          wing_kp_mult=2.0, female_host_target=0.5, smoke=True,
                          allow_calib_mismatch=True)
    run = tmp_path / "run"
    res = run_training(real, out_dir=str(run / "final"), ckpt_dir=str(run / "ckpt"), mcfg=_v2_mcfg(),
                       tcfg=tcfg, aug=MVAugParams(enabled=False), weights=LossWeights(persist=0.5))
    assert np.isfinite(res["final_loss"])
    meta = json.load(open(run / "final" / "mvq_run.json"))
    assert meta["train"]["pair_deltas"] == [1, 4] and meta["train"]["pseudo_root"] == pseudo
    assert meta["train"]["singlefly_root"] == single and meta["train"]["negatives_root"] == negs
    assert meta["train"]["negatives_frac"] == 0.05 and meta["train"]["wing_kp_mult"] == 2.0
    assert meta["val"]["unprompted"]["mpjpe3d_mm"] == meta["val"]["unprompted"]["mpjpe3d_mm"]   # not NaN
    # allow_calib_mismatch reaches the run.json config, and on this fixture (every
    # root names the ONE recording's calib_group identically -- no witness-test
    # aliasing/mismatch scenario built here) it records an empty list, not absence.
    assert meta["train"]["allow_calib_mismatch"] is True
    assert meta["train_data"]["calib_mismatches"] == []
    # the OBJECTIVE is recorded too, not just the schedule: a run.json that cannot
    # say what `persist` or `other_fly_repulsion` were is not a reproducible record
    # (Task 6/7 read this block for the scorecard header).
    assert meta["loss"]["persist"] == 0.5
    assert meta["loss"]["persist_margin_units"] == 2.0 and "other_fly_repulsion" in meta["loss"]
    assert meta["aug"]["enabled"] is False and "cam_drop_p" in meta["aug"]
    # the same objective is on disk BEFORE step 0 (the run-dir copy), so a crashed
    # run still says what it was optimising
    assert json.load(open(run / "mvq_run.json"))["loss"]["persist"] == 0.5

    out = capsys.readouterr().out
    # the mass table, per T, naming every root and pinning the negatives share
    mass = [ln for ln in out.splitlines() if "sampler mix" in ln]
    assert len(mass) == 2 and "T=1" in mass[0] and "T=2" in mass[1], mass
    assert all(all(nm in ln for nm in ("real", "pseudo", "singlefly", "negatives")) for ln in mass), mass
    import re
    assert all(re.search(r"negatives \d+ windows -> mass 0\.0500", ln) for ln in mass), mass
    # the REALISED mix, COUNTED per root (never inferred from sample_weight --
    # two roots can share one value)
    realised = [ln for ln in out.splitlines() if "realised mix" in ln]
    assert realised, out[-4000:]
    for nm in ("real", "pseudo", "singlefly", "negatives"):
        assert f"{nm}=" in realised[0], realised
    # the negatives root's exported 0.3 is overridden to 1.0, and said so
    over = [ln for ln in out.splitlines() if "OVERRIDDEN to 1.0" in ln]
    assert len(over) == 2 and all("0.3" in ln for ln in over), over        # one per T
    neg_line = [ln for ln in out.splitlines() if ln.strip().startswith("[mvq]   negatives:")]
    assert neg_line and all("mean sample_weight=1.0000" in ln for ln in neg_line), neg_line
    assert not any("WARNING" in ln for ln in neg_line), neg_line
    # and the per-root female-host multiplier was SOLVED, not taken from the hand number
    assert any("solved female-host multiplier" in ln or "UNATTAINABLE" in ln
               for ln in out.splitlines()), out[-4000:]


def test_t2_val_split_stays_the_real_root_only(tmp_path, monkeypatch):
    """Spec §7 ("validation on human labels only"): the pseudo/single-fly/
    negatives roots reach the TRAIN split and nothing else -- a negative or a
    pseudo frameset in the val set would move the very headline number the run
    is judged by. Records every `V12WindowDataset(root, split, ...)` the
    trainer builds and asserts the val split was built on `root` alone, and
    that every train root got the SAME T and the SAME spacings."""
    from jarvis_jax.data.mv_augment import MVAugParams
    from jarvis_jax.train.losses_mvq import LossWeights
    import jarvis_jax.train.train_mvq as train_mvq
    from jarvis_jax.data.v12_windows import V12WindowDataset as Real
    from jarvis_jax.train.train_mvq import MVQTrainConfig, run_training

    seen = []

    class Recording(Real):
        def __init__(self, root, split, T=1, **kw):
            seen.append((str(root), split, kw.get("pair_deltas"), int(T)))
            super().__init__(root, split, T, **kw)

    monkeypatch.setattr(train_mvq, "V12WindowDataset", Recording)
    (tmp_path / "real").mkdir()
    real = make_v12_root(tmp_path / "real", n_frames=6)
    pseudo = _pseudo_copy(real, tmp_path / "pseudo")
    negs = _negatives_copy(real, tmp_path / "negatives")
    tcfg = MVQTrainConfig(total_steps=1, batch_size=2, warmup_steps=1, eval_every=1, save_every=1,
                          log_every=1, num_workers=1, pretrained=False, window_lengths=(2,),
                          pair_deltas=(1, 4), pseudo_root=pseudo, negatives_root=negs, smoke=True)
    run_training(real, out_dir=str(tmp_path / "final"), ckpt_dir=None, mcfg=_v2_mcfg(), tcfg=tcfg,
                 aug=MVAugParams(enabled=False), weights=LossWeights())
    val = [s for s in seen if s[1] == "val"]
    assert val and all(s[0] == real for s in val), seen
    train = [s for s in seen if s[1] == "train"]
    assert sorted({s[0] for s in train}) == sorted({real, pseudo, negs}), train
    assert all(s[2] == (1, 4) and s[3] == 2 for s in train), train


def test_batch_size_must_divide_the_device_count(tmp_path, monkeypatch):
    """The v2 launch is batch 32 on 8 devices; the 7-device layout hung P3b
    (p3b-notes.md), so a batch that does not divide the device count must fail
    at once, naming both numbers, BEFORE any dataset load or model build (the
    root here does not even exist)."""
    import jax as _jax
    from jarvis_jax.data.mv_augment import MVAugParams
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.train.train_mvq import MVQTrainConfig, run_training
    monkeypatch.setattr(_jax, "devices", lambda *a, **k: [object()] * 7)
    tcfg = MVQTrainConfig(total_steps=1, batch_size=32, window_lengths=(1, 2), pair_deltas=(1, 4, 16),
                          pretrained=False, smoke=True)
    with pytest.raises(ValueError, match=r"batch_size 32 .*7"):
        run_training(str(tmp_path / "no-such-root"), out_dir=str(tmp_path / "f"), ckpt_dir=None,
                     mcfg=_v2_mcfg(), tcfg=tcfg, aug=MVAugParams(enabled=False), weights=LossWeights())


def test_loss_share_check_reports_every_term(tmp_path):
    """Step-6 launch check: `share_check_steps=N` runs N steps of the REAL
    pipeline (same sampler, augs and T alternation the 40k launch will use)
    and reports each term's WEIGHTED share of `total`, then returns without
    eval or a final checkpoint. Expectation: every term is reported with a
    finite share, the shares sum to 1.0 once the un-itemised deep-supervision
    remainder is counted, and `other_rep` carries its launch weight -- which
    is the number the 5-30 % band is judged against before spending a day of
    8 GPUs on an unmeasured loss balance."""
    from jarvis_jax.data.mv_augment import MVAugParams
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.train.train_mvq import MVQTrainConfig, run_training
    (tmp_path / "real").mkdir()
    real = make_v12_root(tmp_path / "real", n_frames=6)
    tcfg = MVQTrainConfig(total_steps=1000, batch_size=2, warmup_steps=1, log_every=1, num_workers=1,
                          pretrained=False, window_lengths=(1, 2), pair_deltas=(1, 4),
                          wing_kp_mult=2.0, smoke=True)
    res = run_training(real, out_dir=str(tmp_path / "final"), ckpt_dir=str(tmp_path / "ckpt"),
                       mcfg=_v2_mcfg(), tcfg=tcfg, aug=MVAugParams(enabled=False),
                       weights=LossWeights(other_fly_repulsion=20.0, persist=0.5), share_check_steps=2)
    sh = res["loss_shares"]
    assert res["steps"] == 2 and "val" not in res
    assert not os.path.isdir(tmp_path / "final")          # no eval, no final checkpoint
    for name in ("reproj", "l3d", "uv2d", "vis", "conf", "exist", "sex", "rep", "other_rep",
                 "persist", "deep_supervision"):
        assert name in sh["terms"], sorted(sh["terms"])
        assert np.isfinite(sh["terms"][name]["share"]), (name, sh["terms"][name])
    assert abs(sum(t["share"] for t in sh["terms"].values()) - 1.0) < 1e-4
    assert sh["terms"]["other_rep"]["weight"] == 20.0


class _CensusDS:
    """The only surface `_balanced_weights` touches (len, windows, manifest,
    is_female), with a chosen female:male host census over two behaviours."""

    def __init__(self, n_female, n_male, root="fake"):
        self.root = root
        self.windows = [(f"rec{i % 2}", 0, i) for i in range(n_female + n_male)]
        self.manifest = {"rec0": {"behavior": "courtship"}, "rec1": {"behavior": "walking"}}
        self._f = np.zeros(n_female + n_male, bool)
        self._f[:n_female] = True

    def __len__(self):
        return len(self.windows)

    def is_female(self, i):
        return bool(self._f[i])


def test_female_host_target_solves_the_multiplier_per_root(capsys):
    """`female_host_weight` is a hand number solved for ONE census (4.27 for
    `red_data_3d_v12_export0902` train). v2 balances four roots separately, each
    with its own census, so the same number applied to all of them lands the
    OVERALL female mass wherever they happen to average out to -- not on 0.5.
    `female_host_target` solves the multiplier per root instead.

    Expectation on a 1:3 female:male root: the female-host weight mass is exactly
    the target, and drawing with the sampler's own `rng.choice(..., p=w)` realises
    0.5 within 0.02. On an all-female root the target is unattainable at any
    multiplier, so the weights must come back UNCHANGED and say so in the log."""
    from jarvis_jax.train.train_mvq import _balanced_weights

    ds = _CensusDS(1000, 3000, root="one_to_three")
    is_f = ds._f
    w = _balanced_weights(ds, 0.5, 1.0, female_host_target=0.5, label="one_to_three")
    assert abs(w.sum() - 1.0) < 1e-12
    assert abs(w[is_f].sum() - 0.5) < 1e-9, w[is_f].sum()
    # realised over the SAME draw window_batches makes (rng.choice with p=w)
    idx = np.random.default_rng(0).choice(len(ds), size=20000, replace=True, p=w)
    assert abs(is_f[idx].mean() - 0.5) < 0.02, is_f[idx].mean()
    # a non-0.5 target is honoured too (the knob is a target, not a switch)
    w25 = _balanced_weights(ds, 0.5, 1.0, female_host_target=0.25, label="one_to_three")
    assert abs(w25[is_f].sum() - 0.25) < 1e-9
    out = capsys.readouterr().out
    assert "solved female-host multiplier" in out and "one_to_three" in out

    # all-female root: no multiplier can produce a male-host draw, so leave it be
    allf = _CensusDS(500, 0, root="single_fly_root")
    base = _balanced_weights(allf, 0.5, 1.0)
    tgt = _balanced_weights(allf, 0.5, 1.0, female_host_target=0.5, label="single_fly_root")
    np.testing.assert_allclose(tgt, base, rtol=0, atol=0)
    out = capsys.readouterr().out
    assert "UNATTAINABLE" in out and "single_fly_root" in out and "no male/other-host window" in out
    # ... and the mirror case (a negatives root: no female host at all)
    allm = _CensusDS(0, 500, root="negatives_root")
    np.testing.assert_allclose(
        _balanced_weights(allm, 0.5, 1.0, female_host_target=0.5, label="negatives_root"),
        _balanced_weights(allm, 0.5, 1.0), rtol=0, atol=0)
    assert "no female-host window" in capsys.readouterr().out

    # the hand-number path is untouched when no target is set (mvq.yaml/P3b runs)
    np.testing.assert_allclose(_balanced_weights(ds, 0.5, 1.0, 4.27),
                               _balanced_weights(ds, 0.5, 1.0, 4.27, None), rtol=0, atol=0)


def test_negatives_train_at_sample_weight_one(tmp_path):
    """The negatives export ships the pseudo-label weight (0.3), but
    `sample_weight` multiplies EVERY loss term per sample -- including the
    existence BCE, which is the only thing an empty window carries. Training a
    negative at 0.3 would down-weight its single purpose, so `run_training`
    forces 1.0. How OFTEN negatives are drawn is `negatives_frac`'s job, and
    that is unaffected."""
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.train.train_mvq import _ForceSampleWeight
    (tmp_path / "real").mkdir()
    real = make_v12_root(tmp_path / "real", n_frames=6)
    negs = _negatives_copy(real, tmp_path / "negatives")
    raw = V12WindowDataset(negs, "train", T=1, train=True, pair_deltas=(1,))
    assert len(raw) and raw.weight(0) == 0.3 and raw.source(0) == "pseudo"      # as exported
    assert bool(raw[0]["is_negative"]) and float(raw[0]["sample_weight"]) == pytest.approx(0.3)
    forced = _ForceSampleWeight(raw, 1.0)
    assert len(forced) == len(raw) and forced.weight(0) == 1.0
    s = forced[0]
    assert float(s["sample_weight"]) == 1.0 and bool(s["is_negative"])
    # the view is transparent: everything else still answers from the real dataset
    assert forced.keypoint_names == raw.keypoint_names and forced.T == raw.T
    assert forced.n_flies(0) == 0 and forced.root == raw.root
    forced.epoch = 7
    assert raw.epoch == 7, "the epoch write must reach the wrapped dataset"


def test_female_host_target_on_a_root_that_is_all_female_by_a_float_hair(tmp_path, capsys):
    """Regression: an all-female root's post-balance female mass comes back as
    0.9999999999999998, not 1.0 (measured on the T=2 fixture root, where only
    fly0 has labelled pairs). A bare `f >= 1.0` one-sided guard sails past that
    and "solves" a multiplier of ~4e-16, which prints as a legitimate
    `-> solved female-host multiplier 0.0000` and would zero out the female mass
    of any root that is merely NEARLY one-sided. Caught only by reading the run
    log of the CPU smoke: the guard must be on the window COUNT (exact), with the
    mass check as a toleranced backstop.

    Expectation: this root reports UNATTAINABLE and its weights are byte-identical
    to the no-target call."""
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.train.train_mvq import _balanced_weights
    (tmp_path / "real").mkdir()
    root = make_v12_root(tmp_path / "real", n_frames=6)
    ds = V12WindowDataset(root, "train", T=2, train=True, pair_deltas=(1, 4))
    assert len(ds) and all(ds.is_female(i) for i in range(len(ds))), "fixture must be all-female at T=2"
    base = _balanced_weights(ds, 0.5, 1.0)
    assert 1.0 - 1e-12 < float(base.sum()) <= 1.0 + 1e-12
    tgt = _balanced_weights(ds, 0.5, 1.0, female_host_target=0.5, label="T=2 root 'real'")
    np.testing.assert_allclose(tgt, base, rtol=0, atol=0)
    out = capsys.readouterr().out
    assert "UNATTAINABLE" in out and "no male/other-host window" in out, out
    assert "solved female-host multiplier" not in out, out
