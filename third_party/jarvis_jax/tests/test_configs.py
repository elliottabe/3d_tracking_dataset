"""Hydra config composition tests (CPU-only, no model build)."""
import os
import dataclasses
import pytest
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, build_dataclass

register_resolvers()


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides)


def test_base_config_resolves():
    cfg = _compose(["paths=hyak"])
    # Full resolution must not raise (catches missing interpolations).
    OmegaConf.resolve(cfg)
    assert cfg.paths.runs_root
    assert cfg.run_id
    # paths.cache_dir was removed 2026-09-05 (legacy cached3d cache, see
    # configs/paths/hyak.yaml); its consumers now require an explicit override.


def test_build_dataclass_filters_unknown_keys():
    @dataclasses.dataclass(frozen=True)
    class C:
        a: int = 1
        b: int = 2
    node = OmegaConf.create({"a": 10, "b": 20, "extra": 99})
    out = build_dataclass(C, node)
    assert out == C(a=10, b=20)


def test_model_data_cache_groups_resolve():
    cfg = _compose(["paths=hyak", "model=hybridnet", "data=v3", "cache=default"])
    OmegaConf.resolve(cfg)
    assert cfg.model.roi_cube == 48
    assert cfg.model.grid_spacing == 1
    assert cfg.model.num_cameras == 7
    assert cfg.model.sharpen == 3.0
    assert cfg.model.vitpose.num_keypoints == 50
    assert cfg.data.num_joints == 50
    assert cfg.cache.split == "val"


def test_model_vitpose_standalone_resolves():
    # model=vitpose puts the ViT fields directly at cfg.model.* (for 2D training).
    cfg = _compose(["paths=hyak", "model=vitpose"])
    OmegaConf.resolve(cfg)
    assert cfg.model.num_keypoints == 50
    assert cfg.model.img_size == 448


def test_model_hybridnet_composes_shared_vit():
    # model=hybridnet composes the SAME vitpose config, mounted under model.vitpose,
    # plus v2vnet + 3D params — one ViT definition, no duplication.
    cfg = _compose(["paths=hyak", "model=hybridnet"])
    OmegaConf.resolve(cfg)
    assert cfg.model.roi_cube == 48
    assert cfg.model.grid_spacing == 1
    assert cfg.model.num_cameras == 7
    assert cfg.model.sharpen == 3.0
    assert cfg.model.vitpose.num_keypoints == 50
    assert cfg.model.vitpose.img_size == 448
    assert cfg.model.v2vnet.in_channels == 50


def test_train_slurm_viz_groups_resolve():
    cfg = _compose(["paths=hyak", "train=cached3d", "slurm=ckpt_g2", "viz=default"])
    OmegaConf.resolve(cfg)
    assert cfg.train.total_steps > 0
    assert cfg.train.sharpen == 3.0
    assert cfg.train.save_every > 0
    assert cfg.slurm.partition
    assert cfg.slurm.gpus >= 1


def test_full_default_config_resolves():
    cfg = _compose([])  # all defaults from config.yaml
    OmegaConf.resolve(cfg)
    assert cfg.train and cfg.model and cfg.paths and cfg.slurm


def test_precompute_main_from_cfg_maps_config(monkeypatch):
    import importlib.util, sys, os
    path = os.path.join(os.path.dirname(CONFIG_DIR), "scripts", "precompute_repro_cache.py")
    spec = importlib.util.spec_from_file_location("precompute_repro_cache", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    captured = {}
    monkeypatch.setattr(mod, "run_precompute", lambda **kw: captured.update(kw), raising=False)
    cfg = _compose(["paths=hyak", "cache=default", "cache.split=train", "cache.batch=16",
                    "+paths.cache_dir=/tmp/test_repro_cache"])
    mod.main_from_cfg(cfg)
    assert captured["split"] == "train"
    assert captured["batch"] == 16
    assert captured["root"] == cfg.paths.data_root
    assert captured["vitpose_ckpt"] == cfg.paths.vitpose_ckpt
    assert captured["vitpose_cfg"].num_keypoints == 50
    # cache.dataset_version defaults to 'v5' (Task 15 selector) and is
    # threaded through main_from_cfg -- run_precompute picks V5FramesetDataset
    # vs V3FramesetDataset on this value.
    assert captured["dataset_version"] == "v5"


def test_precompute_main_from_cfg_dataset_version_override(monkeypatch):
    import importlib.util, os
    path = os.path.join(os.path.dirname(CONFIG_DIR), "scripts", "precompute_repro_cache.py")
    spec = importlib.util.spec_from_file_location("precompute_repro_cache", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    captured = {}
    monkeypatch.setattr(mod, "run_precompute", lambda **kw: captured.update(kw), raising=False)
    cfg = _compose(["paths=hyak", "cache=default", "cache.dataset_version=v3",
                    "+paths.cache_dir=/tmp/test_repro_cache"])
    mod.main_from_cfg(cfg)
    assert captured["dataset_version"] == "v3"


def test_cached3d_main_from_cfg_requires_cache_dir_override(monkeypatch):
    """paths.cache_dir was removed 2026-09-05 (legacy cache deleted); without
    an explicit override the legacy trainer must fail with a clear message,
    not a dangling-interpolation error."""
    import jarvis_jax.train.train_3d_cached as m
    monkeypatch.setattr(m, "run_cached_training", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("should not run without a cache_dir override")))
    cfg = _compose(["paths=hyak", "train=cached3d", "run_id=unittest"])
    with pytest.raises(SystemExit, match="cache_dir"):
        m.main_from_cfg(cfg)


def test_cached3d_main_from_cfg_maps_config(monkeypatch):
    import jarvis_jax.train.train_3d_cached as m
    captured = {}

    def fake_run(cache_dir, *, out_dir, ckpt_dir, tcfg, save_every, log_every, eval_every):
        captured.update(cache_dir=cache_dir, out_dir=out_dir, ckpt_dir=ckpt_dir,
                        tcfg=tcfg, save_every=save_every, eval_every=eval_every)
        return {"val_mpjpe_3d": 0.0}

    monkeypatch.setattr(m, "run_cached_training", fake_run)
    cfg = _compose(["paths=hyak", "train=cached3d", "run_id=unittest",
                    "train.total_steps=5", "train.sharpen=3",
                    "+paths.cache_dir=/tmp/test_cached3d_cache"])
    m.main_from_cfg(cfg)
    assert captured["tcfg"].total_steps == 5
    assert captured["tcfg"].sharpen == 3.0
    assert captured["out_dir"].endswith("unittest/final")
    assert captured["ckpt_dir"].endswith("unittest/ckpt")
    assert captured["save_every"] == cfg.train.save_every


def test_inline3d_main_from_cfg_maps_config(monkeypatch):
    import jarvis_jax.train.train_3d as m
    captured = {}
    def fake_run(root, *, out_dir, vitpose_ckpt, ckpt_dir, tcfg, save_every, log_every, eval_every):
        captured.update(root=root, out_dir=out_dir, vitpose_ckpt=vitpose_ckpt,
                        ckpt_dir=ckpt_dir, tcfg=tcfg, save_every=save_every)
        return {"val_mpjpe_3d": 0.0}
    monkeypatch.setattr(m, "run_training_3d", fake_run)
    cfg = _compose(["paths=hyak", "train=inline3d", "run_id=inl_unittest", "train.total_steps=9"])
    m.main_from_cfg(cfg)
    assert captured["tcfg"].total_steps == 9
    assert captured["root"] == cfg.paths.data_root
    assert captured["vitpose_ckpt"] == cfg.paths.vitpose_ckpt
    assert captured["out_dir"].endswith("inl_unittest/final")
    assert captured["ckpt_dir"].endswith("inl_unittest/ckpt")


def test_vit2d_main_from_cfg_maps_config(monkeypatch):
    import importlib.util, os
    path = os.path.join(CONFIG_DIR, "..", "jarvis_jax", "scripts", "train_keypoints.py")
    spec = importlib.util.spec_from_file_location("train_keypoints", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    captured = {}
    monkeypatch.setattr(mod, "run_training",
                        lambda root, **kw: captured.update(root=root, **kw))
    cfg = _compose(["paths=hyak", "model=vitpose", "train=vit2d", "run_id=vit_unittest",
                    "train.total_steps=7", "train.batch_size=16"])
    mod.main_from_cfg(cfg)
    assert captured["tcfg"].total_steps == 7
    assert captured["tcfg"].batch_size == 16
    assert captured["tcfg"].backbone_lr_mult == 0.1
    assert captured["root"] == cfg.paths.data_root
    assert captured["out_dir"].endswith("vit_unittest/final")
    assert captured["vitpose_cfg"].num_keypoints == 50
    assert captured["warm_start"] is None   # train.warm_start default '' -> None (off)


def test_vit2d_main_from_cfg_maps_warm_start(monkeypatch):
    """train.warm_start=<final dir> is threaded through to run_training's
    `warm_start` kwarg unchanged (run_training itself does the restore+fresh-
    optimizer wiring -- see tests/test_warm_start.py for that mechanism)."""
    import importlib.util, os
    path = os.path.join(CONFIG_DIR, "..", "jarvis_jax", "scripts", "train_keypoints.py")
    spec = importlib.util.spec_from_file_location("train_keypoints", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    captured = {}
    monkeypatch.setattr(mod, "run_training",
                        lambda root, **kw: captured.update(root=root, **kw))
    cfg = _compose(["paths=hyak", "model=efficienttrack_bn", "train=vit2d",
                    "run_id=ft_unittest", "train.total_steps=7",
                    "train.warm_start=/some/run/final"])
    mod.main_from_cfg(cfg)
    assert captured["warm_start"] == "/some/run/final"
    assert captured["out_dir"].endswith("ft_unittest/final")   # NEW run dir, not the source's


def test_viz_main_from_cfg_maps_config(monkeypatch):
    import importlib.util, os
    path = os.path.join(os.path.dirname(CONFIG_DIR), "scripts", "viz_compare_3d_runs.py")
    spec = importlib.util.spec_from_file_location("viz_compare_3d_runs", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    captured = {}
    monkeypatch.setattr(mod, "run_compare", lambda **kw: captured.update(kw), raising=False)
    cfg = _compose(["paths=hyak", "viz=default",
                    "viz.run1=/a/final", "viz.run2=/b/final", "viz.sharpen2=3",
                    "+paths.cache_dir=/tmp/test_viz_cache"])
    mod.main_from_cfg(cfg)
    assert captured["run1"] == "/a/final" and captured["run2"] == "/b/final"
    assert captured["sharpen2"] == 3.0
    assert captured["cache_dir"] == cfg.paths.cache_dir


def test_build_checkpoint_main_from_cfg_maps_config(monkeypatch):
    import importlib.util, os
    path = os.path.join(CONFIG_DIR, "..", "jarvis_jax", "convert", "build_checkpoint.py")
    spec = importlib.util.spec_from_file_location("build_checkpoint", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    captured = {}
    monkeypatch.setattr(mod, "run_build", lambda **kw: captured.update(kw), raising=False)
    cfg = _compose(["paths=hyak", "model=vitpose", "+convert.out=/tmp/vit_ckpt"])
    mod.main_from_cfg(cfg)
    assert captured["npz"] == cfg.paths.mae_npz
    assert captured["out"] == "/tmp/vit_ckpt"
    assert captured["vitpose_cfg"].num_keypoints == 50


def test_predict_main_from_cfg_maps_config(monkeypatch):
    import importlib.util, os
    path = os.path.join(os.path.dirname(CONFIG_DIR), "scripts", "predict_3d.py")
    spec = importlib.util.spec_from_file_location("predict_3d", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    captured = {}
    monkeypatch.setattr(mod, "run_predict", lambda **kw: captured.update(kw), raising=False)
    cfg = _compose(["paths=hyak", "model=hybridnet", "predict=default",
                    "run_id=run4", "predict.split=val", "predict.batch=8"])
    mod.main_from_cfg(cfg)
    assert captured["split"] == "val" and captured["batch"] == 8
    assert captured["vitpose_ckpt"] == cfg.paths.vitpose_ckpt
    assert captured["v2v_final"].endswith("run4/final")
    assert captured["sharpen"] == cfg.model.sharpen


def test_sam3_main_from_cfg_maps_config(monkeypatch):
    import importlib.util, os
    path = os.path.join(os.path.dirname(CONFIG_DIR), "scripts", "sam3_masks.py")
    spec = importlib.util.spec_from_file_location("sam3_masks", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    captured = {}
    monkeypatch.setattr(mod, "run_sam3_masks", lambda **kw: captured.update(kw) or {"n_bouts": 0}, raising=False)
    cfg = _compose(["paths=hyak", "sam3=default", "sam3.limit=1",
                    "sam3.session_dir=/s/rec", "sam3.num_animals=2"])
    mod.main_from_cfg(cfg)
    assert captured["session_dir"] == "/s/rec"
    assert captured["limit"] == 1
    assert captured["num_animals"] == 2
    assert captured["project"] == cfg.sam3.project
    assert captured["sam3"]["sam3_version"] == cfg.sam3.sam3_version


def test_predict_session_main_from_cfg_maps_config(monkeypatch):
    import importlib.util, os
    path = os.path.join(os.path.dirname(CONFIG_DIR), "scripts", "predict_session.py")
    spec = importlib.util.spec_from_file_location("predict_session", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    captured = {}
    monkeypatch.setattr(mod, "run_predict_session", lambda **kw: captured.update(kw) or {}, raising=False)
    cfg = _compose(["paths=hyak", "model=hybridnet", "predict_session=default",
                    "run_id=run4", "predict_session.session_dir=/s/rec",
                    "predict_session.masks_dir=/m", "predict_session.bout_ids=4"])
    mod.main_from_cfg(cfg)
    assert captured["session_dir"] == "/s/rec" and captured["masks_dir"] == "/m"
    assert captured["bout_ids"] == [4]
    assert captured["vitpose_ckpt"] == cfg.paths.vitpose_ckpt
    assert captured["v2v_final"].endswith("run4/final")
    assert captured["sharpen"] == cfg.model.sharpen


def test_train_mvq_v2_group_resolves():
    """`train=mvq_v2` (mvq-v2 spec 2026-09-05 §4) must compose AND survive the
    Hydra -> MVQTrainConfig conversion in `jarvis_jax/scripts/train_mvq.py`: the
    v2 keys are new dataclass fields, and a yaml list reaching a `tuple` field
    (pair_deltas) is exactly the kind of thing that only fails at launch."""
    from jarvis_jax.data.mv_augment import MVAugParams
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.train.train_mvq import MVQTrainConfig
    cfg = _compose(["paths=hyak", "model=mvq", "train=mvq_v2"])
    OmegaConf.resolve(cfg)
    t = OmegaConf.to_container(cfg.train, resolve=True)
    assert list(t["window_lengths"]) == [1, 2] and list(t["pair_deltas"]) == [1, 4, 16]
    assert t["female_host_target"] == 0.5          # solved per root, not the hand number
    assert t["negatives_frac"] == 0.05 and t["pseudo_weight"] == 0.3
    assert t["warm_start"] is None                 # §2 decision 1: from scratch
    assert t["prompt_p_start"] == 0.0 and t["prompt_p_end"] == 0.0
    assert t["wing_kp_mult"] == 2.0 and t["jitter_units"] == 10.0
    assert t["batch_size"] == 32 and t["total_steps"] == 40000
    assert t["loss"]["other_fly_repulsion"] == 20.0 and t["loss"]["persist"] == 0.5
    assert t["mv_aug"]["cam_drop_p"] == 0.1
    # every root path is absolute and outside red_data/ (the export location)
    for k in ("pseudo_root", "singlefly_root", "negatives_root"):
        assert t[k].startswith("/gscratch/") and "/red_data/" not in t[k], (k, t[k])
    # the same conversion the entrypoint does -- unknown keys filtered, lists to tuples
    t["window_lengths"] = tuple(t["window_lengths"]); t["val_cohorts"] = tuple(t["val_cohorts"])
    t["pair_deltas"] = tuple(int(v) for v in t["pair_deltas"])
    t["copy_paste_contact_sep"] = tuple(float(v) for v in t["copy_paste_contact_sep"])
    tcfg = MVQTrainConfig(**{k: v for k, v in t.items() if k in MVQTrainConfig.__dataclass_fields__})
    assert tcfg.pair_deltas == (1, 4, 16) and tcfg.female_host_target == 0.5
    assert LossWeights(**t["loss"]).persist == 0.5      # no unknown key in the loss block
    assert MVAugParams(**t["mv_aug"]).cam_drop_p == 0.1
