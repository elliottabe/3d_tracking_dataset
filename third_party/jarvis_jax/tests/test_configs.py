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
    assert cfg.paths.cache_dir
    assert cfg.run_id


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
    cfg = _compose(["paths=hyak", "cache=default", "cache.split=train", "cache.batch=16"])
    mod.main_from_cfg(cfg)
    assert captured["split"] == "train"
    assert captured["batch"] == 16
    assert captured["root"] == cfg.paths.data_root
    assert captured["vitpose_ckpt"] == cfg.paths.vitpose_ckpt
    assert captured["vitpose_cfg"].num_keypoints == 50


def test_cached3d_main_from_cfg_maps_config(monkeypatch):
    import jarvis_jax.train.train_3d_cached as m
    captured = {}

    def fake_run(cache_dir, *, out_dir, ckpt_dir, tcfg, save_every, log_every, eval_every):
        captured.update(cache_dir=cache_dir, out_dir=out_dir, ckpt_dir=ckpt_dir,
                        tcfg=tcfg, save_every=save_every, eval_every=eval_every)
        return {"val_mpjpe_3d": 0.0}

    monkeypatch.setattr(m, "run_cached_training", fake_run)
    cfg = _compose(["paths=hyak", "train=cached3d", "run_id=unittest",
                    "train.total_steps=5", "train.sharpen=3"])
    m.main_from_cfg(cfg)
    assert captured["tcfg"].total_steps == 5
    assert captured["tcfg"].sharpen == 3.0
    assert captured["out_dir"].endswith("unittest/final")
    assert captured["ckpt_dir"].endswith("unittest/ckpt")
    assert captured["save_every"] == cfg.train.save_every
