"""CPU-fast unit tests for jarvis_jax/scripts/eval_keypoints_2d.py -- no real
30k-step checkpoint, no GPU. Proves the load+eval path is wired:

  1. ``restore_model`` round-trips a tiny ViTPose through a real Orbax
     ``StandardCheckpointer`` save/restore (the same eval_shape + nnx.merge
     pattern as ``build_checkpoint.load_vitpose``, generalised over arch).
  2. ``evaluate`` -- the function the Hydra `main` just calls -- wires
     restore_model + V3Dataset-shaped subsetting + eval_mpjpe together and
     returns finite MPJPE numbers, using a tiny in-memory dataset stand-in
     (injected via `dataset_cls`) instead of a real V3Dataset/data root.
  3. ``main_from_cfg`` maps a composed Hydra config onto `evaluate`'s kwargs
     (same monkeypatch pattern as the other scripts' tests in
     tests/test_configs.py).
"""
import os

import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest
from flax import nnx
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.models.efficienttrack import EfficientTrack
from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers
from jarvis_jax.scripts.eval_keypoints_2d import (
    restore_model, evaluate, main_from_cfg,
)

register_resolvers()

NUM_KEYPOINTS = 5
IMG_SIZE = 64
HEATMAP_SIZE = 32   # ClassicDecoder does a fixed 8x upsample: (64/16)**2 tokens -> 4*8=32


def _tiny_vitpose_cfg():
    return ViTPoseConfig(img_size=IMG_SIZE, patch=16, embed_dim=48, depth=2,
                         num_heads=4, num_keypoints=NUM_KEYPOINTS,
                         heatmap_size=HEATMAP_SIZE)


class _TinyDS:
    """Minimal V3Dataset stand-in: exposes exactly what evaluate()/eval_mpjpe
    read -- heatmap_size, __len__, __getitem__ -> (img4_u8, kp_xy, vis), and
    file_names (for evaluate()'s per-recording recording-name discovery)."""

    def __init__(self, imgs, kps, viss, file_names, heatmap_size):
        self.imgs, self.kps, self.viss = imgs, kps, viss
        self.file_names = file_names
        self.heatmap_size = heatmap_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.kps[i], self.viss[i]


def _make_tiny_dataset_cls(img_size, heatmap_size, num_keypoints, n_per_rec=3):
    """Return a `dataset_cls(root, split, recordings=None)` factory backed by a
    small fixed pool spanning two fake recordings ("recA", "recB"), filtering
    by `recordings` the same way V3Dataset does (startswith rec + "/") -- so
    evaluate()'s overall/per-recording/female-subset wiring is exercised for
    real, not stubbed out."""
    rng = np.random.default_rng(0)
    recs = ["recA", "recB"]
    file_names, imgs, kps, viss = [], [], [], []
    for r in recs:
        for i in range(n_per_rec):
            file_names.append(f"{r}/Cam0/Frame_{i}.jpg")
            imgs.append(rng.integers(0, 256, size=(img_size, img_size, 4), dtype=np.uint8))
            kps.append(rng.uniform(0, heatmap_size, size=(num_keypoints, 2)).astype(np.float32))
            viss.append(np.ones((num_keypoints,), dtype=np.float32))
    imgs, kps, viss = np.stack(imgs), np.stack(kps), np.stack(viss)

    def dataset_cls(root, split, recordings=None):
        if recordings is None:
            sel = list(range(len(file_names)))
        else:
            sel = [i for i, fn in enumerate(file_names)
                  if any(fn.startswith(r + "/") for r in recordings)]
        return _TinyDS(imgs[sel], kps[sel], viss[sel],
                       [file_names[i] for i in sel], heatmap_size)

    return dataset_cls


def test_restore_model_roundtrips_vitpose(tmp_path):
    cfg = _tiny_vitpose_cfg()
    model = ViTPose(cfg, rngs=nnx.Rngs(0))

    ckpt_dir = str(tmp_path / "vitpose_ckpt")
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(ckpt_dir, nnx.split(model)[1])
    ckptr.wait_until_finished()

    restored = restore_model(ckpt_dir, "vitpose", cfg)

    orig_state = dict(nnx.split(model)[1].flat_state())
    restored_state = dict(nnx.split(restored)[1].flat_state())
    assert set(orig_state.keys()) == set(restored_state.keys())
    for k in orig_state:
        np.testing.assert_array_equal(
            np.asarray(orig_state[k].value), np.asarray(restored_state[k].value))


def test_restore_model_roundtrips_efficienttrack(tmp_path):
    model = EfficientTrack(num_joints=NUM_KEYPOINTS, in_channels=4, rngs=nnx.Rngs(0))

    ckpt_dir = str(tmp_path / "efftrack_ckpt")
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(ckpt_dir, nnx.split(model)[1])
    ckptr.wait_until_finished()

    cfg = ViTPoseConfig(num_keypoints=NUM_KEYPOINTS, in_ch=4)
    restored = restore_model(ckpt_dir, "efficienttrack", cfg)

    orig_state = dict(nnx.split(model)[1].flat_state())
    restored_state = dict(nnx.split(restored)[1].flat_state())
    assert set(orig_state.keys()) == set(restored_state.keys())
    for k in orig_state:
        np.testing.assert_array_equal(
            np.asarray(orig_state[k].value), np.asarray(restored_state[k].value))


def test_restore_model_rejects_unknown_arch():
    with pytest.raises(ValueError):
        restore_model("/nonexistent", "not_a_real_arch", ViTPoseConfig())


def test_evaluate_wires_restore_and_eval_mpjpe(tmp_path):
    """End-to-end (minus real V3Dataset/GPU): save a tiny ViTPose, then call
    `evaluate` exactly as `main_from_cfg` would, with a tiny in-memory
    dataset_cls standing in for V3Dataset. Confirms overall + per-recording +
    female-subset MPJPE all come back finite."""
    cfg = _tiny_vitpose_cfg()
    model = ViTPose(cfg, rngs=nnx.Rngs(0))
    ckpt_dir = str(tmp_path / "final")
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(ckpt_dir, nnx.split(model)[1])
    ckptr.wait_until_finished()

    dataset_cls = _make_tiny_dataset_cls(IMG_SIZE, HEATMAP_SIZE, NUM_KEYPOINTS)

    result = evaluate(
        ckpt_dir, "vitpose", "/fake/root", vitpose_cfg=cfg, batch_size=2,
        val_recording="recA", per_recording=True, dataset_cls=dataset_cls,
        quiet=True)

    assert result["ckpt_dir"] == ckpt_dir
    assert result["arch"] == "vitpose"
    assert np.isfinite(result["overall"]) and result["overall"] >= 0.0
    assert np.isfinite(result["female"]) and result["female"] >= 0.0
    assert set(result["per_recording"].keys()) == {"recA", "recB"}
    for v in result["per_recording"].values():
        assert np.isfinite(v) and v >= 0.0


def test_evaluate_missing_recording_skips_female_subset(tmp_path):
    """A `val_recording` absent from the (fake) val set must not crash --
    matches the real training loop's I4 empty-held-out-val behaviour."""
    cfg = _tiny_vitpose_cfg()
    model = ViTPose(cfg, rngs=nnx.Rngs(0))
    ckpt_dir = str(tmp_path / "final")
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(ckpt_dir, nnx.split(model)[1])
    ckptr.wait_until_finished()

    dataset_cls = _make_tiny_dataset_cls(IMG_SIZE, HEATMAP_SIZE, NUM_KEYPOINTS)

    result = evaluate(
        ckpt_dir, "vitpose", "/fake/root", vitpose_cfg=cfg, batch_size=2,
        val_recording="no_such_recording", per_recording=False,
        dataset_cls=dataset_cls, quiet=True)

    assert result["female"] is None
    assert result["per_recording"] == {}


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides)


def test_main_from_cfg_maps_config(monkeypatch):
    import jarvis_jax.scripts.eval_keypoints_2d as m
    captured = {}

    def fake_evaluate(ckpt_dir, arch, root, **kw):
        captured.update(ckpt_dir=ckpt_dir, arch=arch, root=root, **kw)
        return {"overall": 0.0}

    monkeypatch.setattr(m, "evaluate", fake_evaluate)
    cfg = _compose([
        "paths=hyak", "model=vitpose",
        "+eval.ckpt_dir=/some/final", "+eval.batch_size=4",
    ])
    main_from_cfg(cfg)

    assert captured["ckpt_dir"] == "/some/final"
    assert captured["arch"] == "vitpose"
    assert captured["root"] == cfg.paths.data_root
    assert captured["batch_size"] == 4
    assert captured["vitpose_cfg"].num_keypoints == 50


def test_main_from_cfg_efficienttrack_arch(monkeypatch):
    import jarvis_jax.scripts.eval_keypoints_2d as m
    captured = {}

    def fake_evaluate(ckpt_dir, arch, root, **kw):
        captured.update(ckpt_dir=ckpt_dir, arch=arch, root=root, **kw)
        return {"overall": 0.0}

    monkeypatch.setattr(m, "evaluate", fake_evaluate)
    cfg = _compose([
        "paths=hyak", "model=efficienttrack",
        "+eval.ckpt_dir=/some/eff_final",
    ])
    main_from_cfg(cfg)

    assert captured["arch"] == "efficienttrack"
    assert captured["ckpt_dir"] == "/some/eff_final"
