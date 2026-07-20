"""CPU-fast unit tests for the front-end SELECTOR added to
scripts/precompute_repro_cache.py -- proves the wiring without building a
real cache (no GPU, no full frameset sweep):

  1. ``_build_front_end`` returns a ViTPose for front_end='vitpose' (default,
     unchanged path) and a restored EfficientTrackBN for
     front_end='efficienttrack_bn' -- same eval_shape + StandardCheckpointer
     + nnx.merge restore pattern as eval_keypoints_2d.restore_model.
  2. ``_cache_is_valid`` (the idempotency/staleness check) treats a
     front_end/ckpt MISMATCH as stale (forces rebuild, never silently reuses
     a different-front-end cache), while a legacy pre-selector cache (only
     the bare 'vitpose_ckpt' key, no 'front_end' key) is still recognized as
     a valid front_end='vitpose' cache (back-compat).
  3. THE critical correctness point: a HybridNet3D built with
     ``normalize_frontend=True`` feeds its front-end ImageNet-normalized
     input (matching how EfficientTrackBN was trained by the 2-D trainer's
     ``loss_fn``, which normalizes before calling the model) -- NOT the raw
     float passthrough used by the default (byte-identical) EfficientTrack
     path. ``normalize_frontend=False`` (default) is verified to leave the
     raw-passthrough behavior untouched.
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest
from flax import nnx

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.data.device import normalize_image
from jarvis_jax.hybridnet.model import HybridNet3D
from jarvis_jax.hybridnet.v2vnet import V2VNet
from jarvis_jax.hydra_utils import CONFIG_DIR
from jarvis_jax.models.efficienttrack import EfficientTrackBN
from jarvis_jax.models.vitpose import ViTPose

# precompute_repro_cache.py lives under scripts/, not the jarvis_jax package
# -- load it the same way tests/test_configs.py does.
_SCRIPT_PATH = os.path.join(os.path.dirname(CONFIG_DIR), "scripts",
                            "precompute_repro_cache.py")
_spec = importlib.util.spec_from_file_location("precompute_repro_cache", _SCRIPT_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

NUM_KEYPOINTS = 5


def _tiny_vitpose_cfg():
    # Small enough to construct/restore fast on CPU; mirrors
    # tests/test_eval_keypoints_2d.py's _tiny_vitpose_cfg.
    return ViTPoseConfig(img_size=64, patch=16, embed_dim=48, depth=2,
                         num_heads=4, num_keypoints=NUM_KEYPOINTS,
                         heatmap_size=32, in_ch=4)


# ---------------------------------------------------------------------------
# 1. _build_front_end selection
# ---------------------------------------------------------------------------

def test_build_front_end_vitpose_default(tmp_path):
    cfg = _tiny_vitpose_cfg()
    model = ViTPose(cfg, rngs=nnx.Rngs(0))
    ckpt_dir = str(tmp_path / "vitpose_ckpt")
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(ckpt_dir, nnx.split(model)[1])
    ckptr.wait_until_finished()

    front, ckpt_used, normalize_frontend = mod._build_front_end(
        "vitpose", vitpose_ckpt=ckpt_dir, vitpose_cfg=cfg, frontend_ckpt=None)

    assert isinstance(front, ViTPose)
    assert ckpt_used == ckpt_dir
    # ViTPose is ALWAYS normalized by HybridNet3D's own _is_vitpose branch,
    # independent of this flag -- the selector must not claim otherwise.
    assert normalize_frontend is False


def test_build_front_end_efficienttrack_bn(tmp_path):
    cfg = _tiny_vitpose_cfg()
    model = EfficientTrackBN(num_joints=cfg.num_keypoints, in_channels=cfg.in_ch,
                             rngs=nnx.Rngs(0))
    ckpt_dir = str(tmp_path / "etbn_ckpt")
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(ckpt_dir, nnx.split(model)[1])
    ckptr.wait_until_finished()

    front, ckpt_used, normalize_frontend = mod._build_front_end(
        "efficienttrack_bn", vitpose_ckpt=None, vitpose_cfg=cfg,
        frontend_ckpt=ckpt_dir)

    assert isinstance(front, EfficientTrackBN)
    assert ckpt_used == ckpt_dir
    # Cache-build-time correctness requirement: ET-BN was trained on
    # normalize_image'd input (see train/train.py's loss_fn) -- the selector
    # MUST turn normalization on for this front-end.
    assert normalize_frontend is True

    # Weights actually round-tripped (not just random re-init).
    orig_state = dict(nnx.split(model)[1].flat_state())
    restored_state = dict(nnx.split(front)[1].flat_state())
    assert set(orig_state.keys()) == set(restored_state.keys())
    for k in orig_state:
        np.testing.assert_array_equal(
            np.asarray(orig_state[k].value), np.asarray(restored_state[k].value))


def test_build_front_end_rejects_unknown():
    with pytest.raises(ValueError):
        mod._build_front_end("not_a_real_front_end", vitpose_ckpt=None,
                             vitpose_cfg=_tiny_vitpose_cfg(), frontend_ckpt=None)


def test_build_front_end_vitpose_requires_ckpt():
    with pytest.raises(ValueError):
        mod._build_front_end("vitpose", vitpose_ckpt=None,
                             vitpose_cfg=_tiny_vitpose_cfg(), frontend_ckpt=None)


def test_build_front_end_etbn_requires_ckpt():
    with pytest.raises(ValueError):
        mod._build_front_end("efficienttrack_bn", vitpose_ckpt=None,
                             vitpose_cfg=_tiny_vitpose_cfg(), frontend_ckpt=None)


# ---------------------------------------------------------------------------
# 2. _cache_is_valid staleness / back-compat
# ---------------------------------------------------------------------------

def _write_fake_cache(cache_dir, split, meta, n):
    os.makedirs(cache_dir, exist_ok=True)
    meta = dict(meta)
    meta["n"] = n
    meta["split"] = split
    with open(os.path.join(cache_dir, f"{split}_meta.json"), "w") as f:
        json.dump(meta, f)
    expected_bytes = n * mod._NUM_JOINTS * mod._GRID * mod._GRID * mod._GRID * 2
    with open(os.path.join(cache_dir, f"{split}_volumes.f16"), "wb") as f:
        f.write(b"\x00" * expected_bytes)


def test_cache_is_valid_rejects_front_end_mismatch(tmp_path):
    cache_dir = str(tmp_path / "cache")
    n = 2
    meta = dict(mod._META_GRID_PARAMS)
    meta["front_end"] = "vitpose"
    meta["frontend_ckpt"] = "/fake/vitpose_ckpt"
    _write_fake_cache(cache_dir, "val", meta, n)

    # Exact match -> valid, no rebuild needed.
    assert mod._cache_is_valid(cache_dir, "val", "vitpose", "/fake/vitpose_ckpt", n)

    # front_end mismatch -> stale: MUST force a rebuild, never silently reuse
    # a cache built with a different front-end.
    assert not mod._cache_is_valid(cache_dir, "val", "efficienttrack_bn",
                                   "/fake/et_ckpt", n)

    # Same front_end, different ckpt path -> also stale.
    assert not mod._cache_is_valid(cache_dir, "val", "vitpose", "/other/ckpt", n)


def test_cache_is_valid_back_compat_legacy_vitpose_only_key(tmp_path):
    """A cache written by the pre-selector code only ever had a bare
    'vitpose_ckpt' key (no 'front_end' key at all). It must still be
    recognized as a valid front_end='vitpose' cache -- but NOT be silently
    reused for a different front_end request."""
    cache_dir = str(tmp_path / "legacy_cache")
    n = 2
    meta = dict(mod._META_GRID_PARAMS)
    meta["vitpose_ckpt"] = "/fake/legacy_ckpt"  # no 'front_end' key
    _write_fake_cache(cache_dir, "val", meta, n)

    assert mod._cache_is_valid(cache_dir, "val", "vitpose", "/fake/legacy_ckpt", n)
    assert not mod._cache_is_valid(cache_dir, "val", "efficienttrack_bn",
                                   "/fake/legacy_ckpt", n)


# ---------------------------------------------------------------------------
# 3. HybridNet3D normalize_frontend wiring (the critical correctness check)
# ---------------------------------------------------------------------------

class _CaptureFrontEnd:
    """Records whatever HybridNet3D.predict_heatmaps hands it and echoes it
    straight back, so the test can inspect the EXACT tensor the front-end
    would have received -- no need to run a real forward pass."""

    def __init__(self):
        self.last_input = None

    def predict_heatmaps(self, x):
        self.last_input = x
        return x


class _Cfg:
    num_keypoints = 3


def _tiny_crops():
    rng = np.random.RandomState(0)
    # (B=1, num_cam=2, H=8, W=8, C=4) uint8 RGB+mask -- deliberately tiny;
    # this test never calls reproject_volume, so no real grid/camera math is
    # exercised (see module docstring point 3).
    return jnp.asarray(rng.randint(0, 255, (1, 2, 8, 8, 4), dtype=np.uint8))


def test_normalize_frontend_false_is_raw_passthrough_default():
    front = _CaptureFrontEnd()
    v2v = V2VNet(3, 3, rngs=nnx.Rngs(0))
    model = HybridNet3D(front, v2v, _Cfg())  # default: normalize_frontend=False

    crops = _tiny_crops()
    out = model.predict_heatmaps(crops)

    assert model.normalize_frontend is False
    np.testing.assert_array_equal(np.asarray(out), np.asarray(crops).astype(np.float32))


def test_normalize_frontend_true_applies_normalize_image():
    front = _CaptureFrontEnd()
    v2v = V2VNet(3, 3, rngs=nnx.Rngs(0))
    model = HybridNet3D(front, v2v, _Cfg(), normalize_frontend=True)

    crops = _tiny_crops()
    out = model.predict_heatmaps(crops)

    assert model.normalize_frontend is True
    expected = normalize_image(crops)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected))

    # Sanity: normalization must actually have changed the values (otherwise
    # this test would pass vacuously even with a no-op flag).
    assert not np.allclose(np.asarray(out), np.asarray(crops).astype(np.float32))


def test_vitpose_frontend_ignores_normalize_frontend_flag():
    """ViTPose is ALWAYS normalized via HybridNet3D's own _is_vitpose branch
    (predict_heatmaps checks self._is_vitpose before ever looking at
    self.normalize_frontend) -- passing normalize_frontend=True for a ViTPose
    front-end must not double-normalize or otherwise change its output.

    Uses the full-size default ViTPoseConfig() (img_size=448) rather than the
    tiny cfg used elsewhere in this file: HybridNet3D.predict_heatmaps
    hardcodes a reshape to the production (224, 224) heatmap size (see
    jarvis_jax/hybridnet/model.py), which only matches a real 448-input
    ViTPose -- same config used by the existing
    test_hybridnet_efficienttrack_parity.py::test_vitpose_frontend_unchanged
    regression guard this test parallels."""
    cfg = ViTPoseConfig()
    rngs = nnx.Rngs(0)
    vitpose = ViTPose(cfg, rngs=rngs)
    v2v = V2VNet(cfg.num_keypoints, cfg.num_keypoints, rngs=rngs)

    crops = jnp.asarray(
        np.random.RandomState(0).randint(0, 255, (1, 1, 448, 448, 4), dtype=np.uint8))

    model_default = HybridNet3D(vitpose, v2v, cfg)
    model_flagged = HybridNet3D(vitpose, v2v, cfg, normalize_frontend=True)

    hm_default = model_default.predict_heatmaps(crops)
    hm_flagged = model_flagged.predict_heatmaps(crops)

    np.testing.assert_array_equal(np.asarray(hm_default), np.asarray(hm_flagged))
