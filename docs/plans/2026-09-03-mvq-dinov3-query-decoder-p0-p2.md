# MVQ (DINOv3 + query decoder) Implementation Plan — Phases P0–P2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train, on human labels only (T=1 first, T=2 loader ready), a multi-view query lifter that emits ROI-local 3D and per-view 2D keypoints from 7 calibrated affine cameras, initialised from DINOv3 ViT-B/16, and pass figure gate 1 plus a val comparison against DLT.

**Architecture:** Shared DINOv3 ViT-B/16 per crop -> affine "ray token" geometry embeddings -> interleaved local/global fusion -> D4RT-style cross-attention decoder with promptable instance queries (3D heavy path with query self-attention, 2D light path, one refinement pass). Losses are pixel-unit: reprojection primary, 3D L1 auxiliary, plus 2D-head, visibility, D4RT confidence, existence, repulsion; instances matched by enumerated assignment.

**Tech Stack:** JAX 0.11 / Flax NNX 0.12.x, optax 0.2.8, orbax 0.11, safetensors 0.7, huggingface_hub 1.21, numpy, PIL, Hydra; pytest (`third_party/jarvis_jax/tests`, marker `gpu`).

**Spec:** `docs/specs/2026-09-03-mvq-dinov3-query-decoder-design.md` — read it first; this plan implements its Phases P0, P1, P2 (sections 4–7, 9, gate 1 of 10). P3 (ablations) and P4 (pipeline lifter) get their own plans after P2 results exist.

## Global Constraints

- Package root for all new code: `third_party/jarvis_jax/jarvis_jax/`; tests in `third_party/jarvis_jax/tests/`; run pytest from `third_party/jarvis_jax/` (conftest registers the `gpu` marker). CPU tests use tiny shapes.
- Cameras are AFFINE: every `Cam*.yaml` projection row 3 is `[0,0,0,1]`; `ReprojectionTool.camera_matrices` is `(C,4,3)` = `P.T` float32. Never assume a pinhole/camera centre.
- 3D labels are DLT of human 2D (`rt.reconstruct_point`), only for joints with >= 2 labelled views. Reprojection into labelled views is the primary loss.
- Keypoint order = `annotations/keypoint_names.json` of the data root; assert it, never index by hand. Camera order = `rt.cameras` insertion order (sorted glob); place framesets' cameras BY NAME (see `data/v5_3d.py` docstring).
- Data root: `paths.data_root` = `/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902` (images 1936x448, crop 448 full-height). Val recording cohorts must be non-empty or the trainer raises.
- World units: 0.1 mm per unit; ROI offset scale 24 units; mean projection scale ~8 px/unit (computed per sample from M, never hardcoded).
- Loss weights (spec §5): reproj 1.0 (Huber δ=8 px), 3D L1 0.5, 2D head 0.5 (Huber 8 px), vis BCE 0.1, confidence 0.2, existence 1.0, repulsion 0.5 (20 px / 2.5 units), pass-1 deep supervision 0.5, intermediate-layer 0.3.
- Compute: never train or render on the Hyak login node. The node `g3102` is currently BUSY with the user's training arms (all 8 GPUs ~41 GB used, 2026-09-03) — submit GPU work via `scripts/slurm/submit_task.sh` (4 GPUs for training). JAX jobs need `module load cuda` and `unset JAX_PLATFORMS`.
- HF auth: `.claude/settings.local.json` sets `HF_TOKEN=""` so `huggingface_hub` uses the stored token; DINOv3 weights are gated and access is granted. Set `HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3` (already the user's default).
- Licences: vendored DINOv3 JAX code derives from `jax-ml/bonsai` (Apache-2.0) — keep the attribution header.
- Commits: end messages with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; commit only the task's files (the working tree has unrelated uncommitted changes).

---

## File map (P0–P2)

| file | responsibility |
|---|---|
| `jarvis_jax/models/dinov3.py` | DINOv3 ViT (NNX), config presets, safetensors loader, backbone registration |
| `jarvis_jax/models/mvq/__init__.py` | exports `MVQConfig`, `MVQModel` |
| `jarvis_jax/models/mvq/geometry.py` | affine rows from `camera_matrices`, local offsets, projection, ray tokens, augmentation camera updates |
| `jarvis_jax/models/mvq/fusion.py` | local/global interleaved fusion blocks with validity mask |
| `jarvis_jax/models/mvq/decoder.py` | instance/keypoint/view queries, cross-attn blocks, heads, refinement gather |
| `jarvis_jax/models/mvq/model.py` | `MVQConfig`, `MVQModel.__call__` (bank -> outputs), assembly to world/full-frame |
| `jarvis_jax/data/v12_windows.py` | window index + `V12WindowDataset` (T=1, T=2), threaded batcher |
| `jarvis_jax/data/mv_augment.py` | exact multi-view augmentation (per-view affine + camera update, world rotation, global mirror, camera dropout, photometric) |
| `jarvis_jax/train/matching.py` | enumerated instance assignment |
| `jarvis_jax/train/losses_mvq.py` | the seven loss terms + deep supervision |
| `jarvis_jax/train/train_mvq.py` | `MVQTrainConfig`, optimizer (2 LR groups + EMA), jitted step per T, eval cohorts, checkpointing |
| `jarvis_jax/scripts/train_mvq.py` | Hydra entrypoint |
| `configs/model/mvq.yaml`, `configs/train/mvq.yaml`, `configs/paths/hyak.yaml` (+`mvq_runs_root`) | config |
| `scripts/viz/mvq_overlay.py` (repo root `scripts/`) | figure gate 1 (and later gates 2, 4) |
| `tests/test_dinov3.py`, `tests/test_mvq_geometry.py`, `tests/test_v12_windows.py`, `tests/test_mv_augment.py`, `tests/test_mvq_model.py`, `tests/test_mvq_losses.py`, `tests/test_train_mvq_smoke.py` | tests |

Shared test fixture (Task 3 creates it, later tasks import it): `tests/mvq_fixtures.py` builds a synthetic v12-shaped root under `tmp_path` with 7 affine cameras, 1936x448 JPEGs, one recording, 3 consecutive frames, two flies in frame 1.

---

### Task 1: DINOv3 ViT-B/16 backbone in NNX + safetensors loader

**Files:**
- Create: `jarvis_jax/models/dinov3.py`
- Modify: `jarvis_jax/models/backbone.py` (register `"dinov3_b16"`, `"dinov3_l16"`)
- Test: `tests/test_dinov3.py`

**Interfaces:**
- Consumes: `flax.nnx`, `safetensors.numpy`, `huggingface_hub.snapshot_download`.
- Produces:
  - `DINOv3Config(patch=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, num_registers=4, rope_base=100.0, layer_norm_eps=1e-5, in_ch=3, rope_dtype="bfloat16")`, classmethods `vitb16()`, `vitl16()`.
  - `class DINOv3(nnx.Module)`: `__call__(x: (B,H,W,in_ch) float, already ImageNet-normalised) -> (B, N, D)` patch tokens only (cls + registers dropped); `forward_all(x) -> (B, 1+R+N, D)`.
  - `load_dinov3_safetensors(model: DINOv3, ckpt_dir: str) -> DINOv3` (raises if any checkpoint key is unconsumed or any model param is unfilled).
  - `dinov3_snapshot(repo_id: str) -> str` (HF snapshot dir with `*.safetensors`).
  - `HF_REPOS = {"dinov3_b16": "facebook/dinov3-vitb16-pretrain-lvd1689m", "dinov3_l16": "facebook/dinov3-vitl16-pretrain-lvd1689m"}`.
  - Backbone registry entries `"dinov3_b16"`, `"dinov3_l16"` whose factory `(cfg: ViTPoseConfig, *, rngs)` builds the preset and slices input to the first 3 channels (so the existing ViTPose 4-channel path can A/B it later).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dinov3.py
import numpy as np
import jax, jax.numpy as jnp
import pytest
from flax import nnx

TINY = dict(patch=16, embed_dim=32, depth=2, num_heads=4, mlp_ratio=2.0,
            num_registers=4, rope_base=100.0, layer_norm_eps=1e-5, in_ch=3)


def test_forward_shapes_patch_tokens_only():
    from jarvis_jax.models.dinov3 import DINOv3, DINOv3Config
    cfg = DINOv3Config(**TINY)
    m = DINOv3(cfg, rngs=nnx.Rngs(0))
    x = jnp.zeros((2, 64, 48, 3), jnp.float32)          # non-square on purpose
    toks = m(x)
    assert toks.shape == (2, (64 // 16) * (48 // 16), 32)
    assert m.forward_all(x).shape == (2, 1 + 4 + 12, 32)


def test_rope_angles_shape_and_tile():
    from jarvis_jax.models.dinov3 import rope_cos_sin
    cos, sin = rope_cos_sin(h=4, w=3, head_dim=8, base=100.0)
    assert cos.shape == sin.shape == (12, 8)
    # tile(2): second half of the last axis repeats the first half
    np.testing.assert_allclose(np.asarray(cos)[:, :4], np.asarray(cos)[:, 4:])


def test_presets_match_official_sizes():
    from jarvis_jax.models.dinov3 import DINOv3Config
    b, l = DINOv3Config.vitb16(), DINOv3Config.vitl16()
    assert (b.embed_dim, b.depth, b.num_heads, b.num_registers) == (768, 12, 12, 4)
    assert (l.embed_dim, l.depth, l.num_heads, l.num_registers) == (1024, 24, 16, 4)


def _write_hf_style_safetensors(tmp_path, cfg, seed=0):
    """Emit a checkpoint in the HuggingFace DINOv3ViTModel key layout with
    random values, and return (dir, dict_of_numpy)."""
    from safetensors.numpy import save_file
    rng = np.random.default_rng(seed)
    D, H = cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio)
    t = {}
    t["embeddings.cls_token"] = rng.normal(size=(1, 1, D)).astype(np.float32)
    t["embeddings.mask_token"] = rng.normal(size=(1, 1, D)).astype(np.float32)
    t["embeddings.register_tokens"] = rng.normal(size=(1, cfg.num_registers, D)).astype(np.float32)
    t["embeddings.patch_embeddings.weight"] = rng.normal(size=(D, cfg.in_ch, cfg.patch, cfg.patch)).astype(np.float32)
    t["embeddings.patch_embeddings.bias"] = rng.normal(size=(D,)).astype(np.float32)
    for i in range(cfg.depth):
        p = f"layer.{i}."
        for n in ("q_proj", "v_proj", "o_proj"):
            t[p + f"attention.{n}.weight"] = rng.normal(size=(D, D)).astype(np.float32)
            t[p + f"attention.{n}.bias"] = rng.normal(size=(D,)).astype(np.float32)
        t[p + "attention.k_proj.weight"] = rng.normal(size=(D, D)).astype(np.float32)   # no k bias
        t[p + "mlp.up_proj.weight"] = rng.normal(size=(H, D)).astype(np.float32)
        t[p + "mlp.up_proj.bias"] = rng.normal(size=(H,)).astype(np.float32)
        t[p + "mlp.down_proj.weight"] = rng.normal(size=(D, H)).astype(np.float32)
        t[p + "mlp.down_proj.bias"] = rng.normal(size=(D,)).astype(np.float32)
        t[p + "layer_scale1.lambda1"] = rng.normal(size=(D,)).astype(np.float32)
        t[p + "layer_scale2.lambda1"] = rng.normal(size=(D,)).astype(np.float32)
        for n in ("norm1", "norm2"):
            t[p + f"{n}.weight"] = rng.normal(size=(D,)).astype(np.float32)
            t[p + f"{n}.bias"] = rng.normal(size=(D,)).astype(np.float32)
    t["norm.weight"] = rng.normal(size=(D,)).astype(np.float32)
    t["norm.bias"] = rng.normal(size=(D,)).astype(np.float32)
    d = tmp_path / "ckpt"; d.mkdir()
    save_file(t, str(d / "model.safetensors"))
    return str(d), t


def test_loader_consumes_every_key_and_maps_layouts(tmp_path):
    from jarvis_jax.models.dinov3 import DINOv3, DINOv3Config, load_dinov3_safetensors
    cfg = DINOv3Config(**TINY)
    d, t = _write_hf_style_safetensors(tmp_path, cfg)
    m = load_dinov3_safetensors(DINOv3(cfg, rngs=nnx.Rngs(0)), d)
    # Linear: torch (out,in) -> flax kernel (in,out)
    np.testing.assert_allclose(np.asarray(m.layer[0].attention.q_proj.kernel[...]),
                               t["layer.0.attention.q_proj.weight"].T)
    # Conv: torch (O,I,kh,kw) -> flax (kh,kw,I,O)
    np.testing.assert_allclose(np.asarray(m.embeddings.patch_embeddings.kernel[...]),
                               t["embeddings.patch_embeddings.weight"].transpose(2, 3, 1, 0))
    # LayerNorm weight -> scale
    np.testing.assert_allclose(np.asarray(m.norm.scale[...]), t["norm.weight"])
    assert m.layer[0].attention.k_proj.bias is None   # k has no bias in DINOv3


def test_loader_refuses_unconsumed_key(tmp_path):
    from safetensors.numpy import save_file
    from jarvis_jax.models.dinov3 import DINOv3, DINOv3Config, load_dinov3_safetensors
    cfg = DINOv3Config(**TINY)
    d, t = _write_hf_style_safetensors(tmp_path, cfg)
    t["layer.0.attention.extra.weight"] = np.zeros((1,), np.float32)
    save_file(t, str(tmp_path / "ckpt" / "model.safetensors"))
    with pytest.raises(RuntimeError, match="unconsumed"):
        load_dinov3_safetensors(DINOv3(cfg, rngs=nnx.Rngs(0)), d)


def test_registered_backbone_slices_to_rgb():
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.models.backbone import build_backbone, BACKBONES
    assert "dinov3_b16" in BACKBONES and "dinov3_l16" in BACKBONES
    # tiny override path: factory honours cfg.embed_dim/depth/num_heads when
    # they are SMALLER than the preset (test-only knob) so CPU tests stay fast
    cfg = ViTPoseConfig(img_size=64, patch=16, in_ch=4, embed_dim=32, depth=2, num_heads=4)
    bb = build_backbone("dinov3_b16", cfg, rngs=nnx.Rngs(0))
    toks = bb(jnp.zeros((1, 64, 64, 4)))
    assert toks.shape == (1, 16, 32)


@pytest.mark.gpu
def test_parity_against_transformers_reference():
    """Runs only where `transformers` (>=4.56, DINOv3ViTModel) is installed:
    `pip install transformers` in the 3d_tracking env. Same random weights,
    same input, outputs agree to 3e-3 (RoPE runs in bf16 in both)."""
    transformers = pytest.importorskip("transformers")
    import torch, tempfile
    from safetensors.torch import save_file
    from jarvis_jax.models.dinov3 import DINOv3, DINOv3Config, load_dinov3_safetensors
    hfcfg = transformers.DINOv3ViTConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                         num_attention_heads=4, hidden_act="gelu",
                                         use_gated_mlp=False, num_register_tokens=4, patch_size=16)
    ref = transformers.DINOv3ViTModel(hfcfg).eval()
    cfg = DINOv3Config(patch=16, embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.0, num_registers=4)
    with tempfile.TemporaryDirectory() as d:
        save_file(ref.state_dict(), f"{d}/model.safetensors")
        m = load_dinov3_safetensors(DINOv3(cfg, rngs=nnx.Rngs(0)), d)
    x = np.random.default_rng(0).normal(size=(2, 3, 64, 64)).astype(np.float32)
    with torch.inference_mode():
        y_ref = ref(pixel_values=torch.tensor(x)).last_hidden_state.numpy()
    y = np.asarray(m.forward_all(jnp.asarray(x.transpose(0, 2, 3, 1))))
    np.testing.assert_allclose(y, y_ref, atol=3e-3, rtol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd third_party/jarvis_jax && python -m pytest tests/test_dinov3.py -v -m "not gpu"`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.models.dinov3'`

- [ ] **Step 3: Write the backbone module**

```python
# jarvis_jax/models/dinov3.py
"""DINOv3 ViT (patch 16, RoPE, registers, LayerScale) in Flax NNX.

Adapted from jax-ml/bonsai `bonsai/models/dinov3/{modeling,params}.py`
(Copyright 2026 The JAX Authors, Apache-2.0), with three changes for this
repo: NHWC input, patch-tokens-only default output, and a generic
safetensors loader that mirrors the HuggingFace `DINOv3ViTModel` key layout
(module attribute names below are chosen to MATCH those keys, so the loader
needs no regex table). RoPE is applied in bf16 exactly as the reference does,
which is where the ~2e-3 parity tolerance comes from.
"""
from __future__ import annotations

import dataclasses
import glob
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

HF_REPOS = {
    "dinov3_b16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_l16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


@dataclasses.dataclass(frozen=True)
class DINOv3Config:
    patch: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_registers: int = 4
    rope_base: float = 100.0
    layer_norm_eps: float = 1e-5
    in_ch: int = 3
    rope_dtype: str = "bfloat16"

    @classmethod
    def vitb16(cls):
        return cls(embed_dim=768, depth=12, num_heads=12)

    @classmethod
    def vitl16(cls):
        return cls(embed_dim=1024, depth=24, num_heads=16)

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


def rope_cos_sin(h: int, w: int, head_dim: int, base: float):
    """Axial 2D RoPE tables for an h x w patch grid: (h*w, head_dim) each.
    Coordinates are pixel-centre aligned and normalised per axis to [-1, 1]
    ("separate" normalisation, the DINOv3 default)."""
    ch = (jnp.arange(0.5, h, dtype=jnp.float32) / h) * 2.0 - 1.0
    cw = (jnp.arange(0.5, w, dtype=jnp.float32) / w) * 2.0 - 1.0
    coords = jnp.stack(jnp.meshgrid(ch, cw, indexing="ij"), axis=-1).reshape(-1, 2)   # (HW,2)
    inv_freq = 1.0 / base ** jnp.arange(0.0, 1.0, 4.0 / head_dim, dtype=jnp.float32)  # (D/4,)
    ang = 2.0 * jnp.pi * coords[:, :, None] * inv_freq[None, None, :]                # (HW,2,D/4)
    ang = jnp.tile(ang.reshape(coords.shape[0], -1), (1, 2))                          # (HW,D)
    return jnp.cos(ang), jnp.sin(ang)


def _rotate_half(x):
    d = x.shape[-1] // 2
    return jnp.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def _apply_rope(q, k, cos, sin, n_prefix, dtype):
    """Rotate the PATCH positions of q/k (B,heads,N,hd); prefix tokens untouched."""
    q = q.astype(dtype); k = k.astype(dtype)
    c = cos.astype(dtype)[None, None]; s = sin.astype(dtype)[None, None]
    qp, qk = q[:, :, :n_prefix], q[:, :, n_prefix:]
    kp, kk = k[:, :, :n_prefix], k[:, :, n_prefix:]
    qk = qk * c + _rotate_half(qk) * s
    kk = kk * c + _rotate_half(kk) * s
    return (jnp.concatenate([qp, qk], 2).astype(jnp.float32),
            jnp.concatenate([kp, kk], 2).astype(jnp.float32))


class _Embeddings(nnx.Module):
    def __init__(self, cfg: DINOv3Config, *, rngs):
        D = cfg.embed_dim
        self.cls_token = nnx.Param(jnp.zeros((1, 1, D), jnp.float32))
        self.mask_token = nnx.Param(jnp.zeros((1, 1, D), jnp.float32))   # unused; consumes the ckpt key
        self.register_tokens = nnx.Param(jnp.zeros((1, cfg.num_registers, D), jnp.float32))
        self.patch_embeddings = nnx.Conv(cfg.in_ch, D, kernel_size=(cfg.patch, cfg.patch),
                                         strides=(cfg.patch, cfg.patch), padding="VALID", rngs=rngs)
        self.num_registers = cfg.num_registers

    def __call__(self, x):                       # x: (B,H,W,in_ch) float
        p = self.patch_embeddings(x)             # (B,h,w,D)
        b, h, w, d = p.shape
        p = p.reshape(b, h * w, d)
        cls = jnp.broadcast_to(self.cls_token[...], (b, 1, d))
        reg = jnp.broadcast_to(self.register_tokens[...], (b, self.num_registers, d))
        return jnp.concatenate([cls, reg, p], axis=1), (h, w)


class _Attention(nnx.Module):
    def __init__(self, cfg: DINOv3Config, *, rngs):
        D = cfg.embed_dim
        self.q_proj = nnx.Linear(D, D, use_bias=True, rngs=rngs)
        self.k_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)   # DINOv3: k bias is masked to 0
        self.v_proj = nnx.Linear(D, D, use_bias=True, rngs=rngs)
        self.o_proj = nnx.Linear(D, D, use_bias=True, rngs=rngs)
        self.num_heads, self.head_dim = cfg.num_heads, cfg.head_dim
        self.rope_dtype = jnp.dtype(cfg.rope_dtype)

    def __call__(self, x, cos, sin, n_prefix):
        b, n, d = x.shape
        split = lambda t: t.reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        q, k = _apply_rope(q, k, cos, sin, n_prefix, self.rope_dtype)
        att = jax.nn.softmax((q @ k.transpose(0, 1, 3, 2)) * (self.head_dim ** -0.5), axis=-1)
        out = (att @ v).transpose(0, 2, 1, 3).reshape(b, n, d)
        return self.o_proj(out)


class _LayerScale(nnx.Module):
    def __init__(self, dim, *, init=1.0):
        self.lambda1 = nnx.Param(jnp.full((dim,), init, jnp.float32))

    def __call__(self, x):
        return x * self.lambda1[...]


class _MLP(nnx.Module):
    def __init__(self, cfg: DINOv3Config, *, rngs):
        D, H = cfg.embed_dim, int(round(cfg.embed_dim * cfg.mlp_ratio))
        self.up_proj = nnx.Linear(D, H, rngs=rngs)
        self.down_proj = nnx.Linear(H, D, rngs=rngs)

    def __call__(self, x):
        return self.down_proj(jax.nn.gelu(self.up_proj(x), approximate=False))


class _Layer(nnx.Module):
    def __init__(self, cfg: DINOv3Config, *, rngs):
        D = cfg.embed_dim
        self.norm1 = nnx.LayerNorm(D, epsilon=cfg.layer_norm_eps, rngs=rngs)
        self.attention = _Attention(cfg, rngs=rngs)
        self.layer_scale1 = _LayerScale(D)
        self.norm2 = nnx.LayerNorm(D, epsilon=cfg.layer_norm_eps, rngs=rngs)
        self.mlp = _MLP(cfg, rngs=rngs)
        self.layer_scale2 = _LayerScale(D)

    def __call__(self, x, cos, sin, n_prefix):
        x = x + self.layer_scale1(self.attention(self.norm1(x), cos, sin, n_prefix))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class DINOv3(nnx.Module):
    """Input (B,H,W,in_ch) float32, ImageNet-normalised. Extra input channels
    beyond `in_ch` are sliced off (lets the 4-channel ViTPose path use it)."""

    def __init__(self, cfg: DINOv3Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embeddings = _Embeddings(cfg, rngs=rngs)
        self.layer = nnx.List([_Layer(cfg, rngs=rngs) for _ in range(cfg.depth)])
        self.norm = nnx.LayerNorm(cfg.embed_dim, epsilon=cfg.layer_norm_eps, rngs=rngs)

    def forward_all(self, x, *, remat: bool = False):
        x = x[..., : self.cfg.in_ch]
        toks, (h, w) = self.embeddings(x)
        cos, sin = rope_cos_sin(h, w, self.cfg.head_dim, self.cfg.rope_base)
        n_prefix = 1 + self.cfg.num_registers
        for lyr in self.layer:
            fn = nnx.remat(lambda m, t: m(t, cos, sin, n_prefix)) if remat else (lambda m, t: m(t, cos, sin, n_prefix))
            toks = fn(lyr, toks)
        return self.norm(toks)

    def __call__(self, x, *, remat: bool = False):
        return self.forward_all(x, remat=remat)[:, 1 + self.cfg.num_registers:]


# ---------------------------------------------------------------- loading
def dinov3_snapshot(repo_id: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, allow_patterns=["*.safetensors", "config.json"])


def _assign(state: dict, path: list, value):
    node = state
    for k in path[:-1]:
        node = node[k]
    if path[-1] not in node:
        raise KeyError(".".join(map(str, path)))
    tgt = node[path[-1]]
    if tuple(tgt.shape) != tuple(value.shape):
        raise ValueError(f"shape {value.shape} vs {tgt.shape} at {'.'.join(map(str, path))}")
    node[path[-1]] = jnp.asarray(value, dtype=tgt.dtype)


def load_dinov3_safetensors(model: DINOv3, ckpt_dir: str) -> DINOv3:
    """Fill `model` from HF-layout safetensors in `ckpt_dir`. Every checkpoint
    key must map onto a model parameter and every model parameter must be
    filled, else RuntimeError -- a partially loaded backbone is worse than
    a loud failure."""
    from safetensors.numpy import load_file
    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {ckpt_dir}")
    gdef, st = nnx.split(model)
    state = nnx.to_pure_dict(st)
    filled, unconsumed = set(), []
    for f in files:
        for key, val in load_file(f).items():
            parts = key.split(".")
            leaf = parts[-1]
            is_ln = parts[-2].startswith("norm") or parts[-2] == "norm"
            if leaf == "weight" and val.ndim == 4:                  # conv (O,I,kh,kw)->(kh,kw,I,O)
                parts[-1], val = "kernel", val.transpose(2, 3, 1, 0)
            elif leaf == "weight" and val.ndim == 2:                # linear (out,in)->(in,out)
                parts[-1], val = "kernel", val.T
            elif leaf == "weight" and is_ln:
                parts[-1] = "scale"
            path = [int(p) if p.isdigit() else p for p in parts]
            try:
                _assign(state, path, val)
            except KeyError:
                unconsumed.append(key)
                continue
            filled.add(".".join(map(str, path)))
    if unconsumed:
        raise RuntimeError(f"{len(unconsumed)} unconsumed checkpoint keys, e.g. {unconsumed[:5]}")
    all_leaves = {"/".join(map(str, jax.tree_util.keystr(p, simple=True, separator="/").split("/")))
                  for p, _ in jax.tree_util.tree_leaves_with_path(state)}
    missing = sorted(l for l in all_leaves if l.replace("/", ".") not in filled)
    if missing:
        raise RuntimeError(f"{len(missing)} model params not filled, e.g. {missing[:5]}")
    return nnx.merge(gdef, state)


def load_pretrained(name: str = "dinov3_b16", *, rngs=None) -> DINOv3:
    cfg = DINOv3Config.vitb16() if name == "dinov3_b16" else DINOv3Config.vitl16()
    model = DINOv3(cfg, rngs=rngs or nnx.Rngs(0))
    return load_dinov3_safetensors(model, dinov3_snapshot(HF_REPOS[name]))


# ---------------------------------------------------------------- registry
def _factory(preset):
    def make(cfg, *, rngs):
        base = preset()
        # test-only shrink knob: honour smaller dims from a ViTPoseConfig
        kw = {}
        if cfg.embed_dim < base.embed_dim:
            kw = dict(embed_dim=cfg.embed_dim, depth=cfg.depth, num_heads=cfg.num_heads)
        return DINOv3(dataclasses.replace(base, **kw), rngs=rngs)
    return make


def register():
    from jarvis_jax.models.backbone import register_backbone
    register_backbone("dinov3_b16", _factory(DINOv3Config.vitb16))
    register_backbone("dinov3_l16", _factory(DINOv3Config.vitl16))
```

Then in `jarvis_jax/models/backbone.py`, after `BACKBONES = {"vit": _vit_factory}` add:

```python
def _register_optional():
    # DINOv3 lives in its own module; registering here keeps `BACKBONES`
    # the single lookup table without importing it at package import.
    from jarvis_jax.models import dinov3
    dinov3.register()

_register_optional()
```

Note on the `missing` check: `jax.tree_util.keystr(..., simple=True, separator="/")` exists in jax 0.11; if it raises `TypeError` on this env, replace with `"".join(str(k) for k in p).replace("[","/").replace("]","").replace("'","").lstrip("/")`. Verify with the test in Step 4 rather than assuming.

- [ ] **Step 4: Run tests**

Run: `cd third_party/jarvis_jax && python -m pytest tests/test_dinov3.py -v -m "not gpu"`
Expected: 6 PASS (the `gpu` parity test is deselected).

- [ ] **Step 5: Parity against the PyTorch reference (one-off, on a compute node)**

`transformers` is not in the env. Install it (pure Python; torch 2.11 is present) and run the parity test once via the queue, not on the login node:

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
pip install "transformers>=4.56" --no-deps 2>&1 | tail -1 && pip install tokenizers regex 2>&1 | tail -1
scripts/slurm/submit_task.sh --gpus 1 --time 0:20:00 mvq_parity \
  'module load cuda; unset JAX_PLATFORMS; cd third_party/jarvis_jax && python -m pytest tests/test_dinov3.py -k parity -v -m gpu'
```
Expected in `slurm_logs/mvq_parity-*.out`: `1 passed`. If it fails with atol > 3e-3 only in the patch-token rows, the RoPE dtype differs: confirm `rope_dtype="bfloat16"` and that the prefix count is `1 + num_registers`. Record the measured max abs diff in the commit message.

- [ ] **Step 6: Download the real weights and check they load (compute node)**

```bash
scripts/slurm/submit_task.sh --gpus 1 --time 0:20:00 mvq_dl \
  'module load cuda; unset JAX_PLATFORMS; cd third_party/jarvis_jax && python -c "
from jarvis_jax.models.dinov3 import load_pretrained
import jax, jax.numpy as jnp
m = load_pretrained(\"dinov3_b16\")
print(m(jnp.zeros((1,448,448,3))).shape)"'
```
Expected: `(1, 784, 768)` and no `unconsumed`/`not filled` error. (If it raises `unconsumed` for `embeddings.mask_token` or `layer.N.attention.k_proj.bias`, the HF layout has drifted: add the key to the module rather than to an ignore list, so coverage stays exact.)

- [ ] **Step 7: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/models/dinov3.py third_party/jarvis_jax/jarvis_jax/models/backbone.py third_party/jarvis_jax/tests/test_dinov3.py
git commit -m "feat(models): DINOv3 ViT-B/16 backbone in NNX with HF safetensors loader

Vendored from jax-ml/bonsai (Apache-2.0); NHWC, patch tokens out, exact key
coverage on load; registered as dinov3_b16 / dinov3_l16. Parity vs
transformers.DINOv3ViTModel: max abs diff <measured>.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Affine camera geometry for the model (`models/mvq/geometry.py`)

**Files:**
- Create: `jarvis_jax/models/mvq/__init__.py` (empty for now), `jarvis_jax/models/mvq/geometry.py`
- Test: `tests/test_mvq_geometry.py`

**Interfaces:**
- Consumes: `ReprojectionTool.camera_matrices` `(C,4,3)` float32 (= `P.T`).
- Produces (all `jnp`, batch-agnostic leading dims where noted):
  - `affine_rows(cam_mats: (C,4,3)) -> (M: (C,2,3), t: (C,2))` — raises `ValueError` unless `P[2] == [0,0,0,1]` within 1e-6.
  - `local_offset(M, t, center3D: (3,), crop_origin: (C,2)) -> t_local: (C,2)` = `M @ center3D + t - crop_origin` (crop px).
  - `project_local(X: (...,3), M: (C,2,3), t_local: (C,2)) -> (...,C,2)` crop px.
  - `ray_from_pixel(M: (C,2,3), t_local: (C,2), uv: (C,N,2)) -> (p0: (C,N,3), d: (C,3))` — `p0` least-norm solution of `M p0 = uv - t_local`; `d` unit null vector of `M` with sign such that `dot(d, cross(M[0], M[1])) > 0`.
  - `token_pixel_centres(h: int, w: int, patch: int) -> (h*w, 2)` (x,y) crop px of token centres, row-major to match the backbone's flatten.
  - `warp_cameras(M, t_local, A: (C,2,2), b: (C,2)) -> (A@M, A@t_local + b)`.
  - `rotate_world(M, R: (3,3)) -> M @ R.T`.
  - `mirror_world(M, t_local, crop: int) -> (M', t')` for flipping ALL views horizontally AND reflecting the world with `S = diag(-1,1,1)`: `A = diag(-1,1)`, `b = (crop-1, 0)`, `M' = A @ M @ S`, `t' = A @ t_local + b`.
  - `px_scale(M: (C,2,3)) -> scalar` = mean over cameras of `sqrt(sum(M**2)/2)` (px per world unit).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mvq_geometry.py
import numpy as np
import jax.numpy as jnp
import pytest

# a real telecentric calibration (Cam2012630, 2026_03_18_15_31_22), see tests/test_affine_camera.py
P_REAL = np.array([
    [8.1001, 0.0074869, -0.031773, -2.828],
    [0.0093308, -8.0788, -0.17912, 462.78],
    [0.0, 0.0, 0.0, 1.0]], np.float64)


def _cam_mats(n=3, seed=0):
    """(n,4,3) P.T stacks: P_REAL rotated about z by different yaws."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        th = rng.uniform(0, 2 * np.pi)
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        P = P_REAL.copy(); P[:2, :3] = P[:2, :3] @ Rz
        out.append(P.T.astype(np.float32))
    return np.stack(out)


def test_affine_rows_and_projection_match_reprojection_tool_formula():
    from jarvis_jax.models.mvq.geometry import affine_rows, project_local, local_offset
    cm = _cam_mats()
    M, t = affine_rows(jnp.asarray(cm))
    assert M.shape == (3, 2, 3) and t.shape == (3, 2)
    X = np.array([1.5, -0.7, 12.0], np.float32)
    ph = np.concatenate([X, [1.0]])
    uv_ref = np.stack([(ph @ cm[c])[:2] / (ph @ cm[c])[2] for c in range(3)])   # ReprojectionTool math
    center = jnp.zeros(3); origin = jnp.zeros((3, 2))
    uv = project_local(jnp.asarray(X), M, local_offset(M, t, center, origin))
    np.testing.assert_allclose(np.asarray(uv), uv_ref, atol=1e-3)


def test_affine_rows_rejects_pinhole():
    from jarvis_jax.models.mvq.geometry import affine_rows
    cm = _cam_mats(1); cm[0, 2, 2] = 0.5    # P[2,2] != 0
    with pytest.raises(ValueError):
        affine_rows(jnp.asarray(cm))


def test_ray_from_pixel_projects_back_for_every_depth():
    from jarvis_jax.models.mvq.geometry import affine_rows, ray_from_pixel, project_local, local_offset
    M, t = affine_rows(jnp.asarray(_cam_mats()))
    tl = local_offset(M, t, jnp.array([3.0, -2.0, 5.0]), jnp.full((3, 2), 100.0))
    uv = jnp.asarray(np.random.default_rng(1).uniform(0, 448, size=(3, 5, 2)).astype(np.float32))
    p0, d = ray_from_pixel(M, tl, uv)
    assert p0.shape == (3, 5, 3) and d.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(d), axis=-1), 1.0, atol=1e-5)
    for s in (-30.0, 0.0, 17.5):
        X = p0 + s * d[:, None, :]                                    # (C,N,3)
        for c in range(3):
            back = project_local(X[c], M, tl)[:, c]                   # (N,2) in camera c
            np.testing.assert_allclose(np.asarray(back), np.asarray(uv[c]), atol=1e-2)


def test_warp_cameras_is_exact_for_affine_image_warp():
    from jarvis_jax.models.mvq.geometry import affine_rows, project_local, local_offset, warp_cameras
    M, t = affine_rows(jnp.asarray(_cam_mats()))
    tl = local_offset(M, t, jnp.zeros(3), jnp.zeros((3, 2)))
    th = 0.3; A = jnp.asarray(np.tile(0.9 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]), (3, 1, 1)))
    b = jnp.asarray(np.array([[5.0, -3.0]] * 3))
    X = jnp.asarray(np.random.default_rng(2).normal(size=(7, 3)).astype(np.float32) * 5)
    uv = project_local(X, M, tl)                                       # (7,C,2)
    uv_w = jnp.einsum("cij,ncj->nci", A, uv) + b                       # warp the 2D labels
    M2, tl2 = warp_cameras(M, tl, A, b)
    np.testing.assert_allclose(np.asarray(project_local(X, M2, tl2)), np.asarray(uv_w), atol=1e-3)


def test_rotate_world_keeps_projection():
    from jarvis_jax.models.mvq.geometry import affine_rows, project_local, local_offset, rotate_world
    M, t = affine_rows(jnp.asarray(_cam_mats()))
    tl = local_offset(M, t, jnp.zeros(3), jnp.zeros((3, 2)))
    a = 0.7; R = jnp.asarray(np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32))
    X = jnp.asarray(np.random.default_rng(3).normal(size=(4, 3)).astype(np.float32))
    np.testing.assert_allclose(np.asarray(project_local(X @ R.T, rotate_world(M, R), tl)),
                               np.asarray(project_local(X, M, tl)), atol=1e-3)


def test_mirror_world_flips_every_view():
    from jarvis_jax.models.mvq.geometry import affine_rows, project_local, local_offset, mirror_world
    M, t = affine_rows(jnp.asarray(_cam_mats()))
    tl = local_offset(M, t, jnp.zeros(3), jnp.zeros((3, 2)))
    X = jnp.asarray(np.random.default_rng(4).normal(size=(4, 3)).astype(np.float32))
    uv = np.asarray(project_local(X, M, tl))
    S = jnp.asarray(np.diag([-1.0, 1.0, 1.0]).astype(np.float32))
    M2, tl2 = mirror_world(M, tl, 448)
    uv2 = np.asarray(project_local(X @ S, M2, tl2))                    # reflected world, mirrored cams
    np.testing.assert_allclose(uv2[..., 0], 447.0 - uv[..., 0], atol=1e-3)
    np.testing.assert_allclose(uv2[..., 1], uv[..., 1], atol=1e-3)


def test_token_centres_and_px_scale():
    from jarvis_jax.models.mvq.geometry import token_pixel_centres, px_scale, affine_rows
    c = np.asarray(token_pixel_centres(28, 28, 16))
    assert c.shape == (784, 2) and tuple(c[0]) == (8.0, 8.0) and tuple(c[1]) == (24.0, 8.0)
    M, _ = affine_rows(jnp.asarray(_cam_mats()))
    assert 7.5 < float(px_scale(M)) < 8.5          # ~8.1 px per world unit for this rig
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd third_party/jarvis_jax && python -m pytest tests/test_mvq_geometry.py -v`
Expected: FAIL, `No module named 'jarvis_jax.models.mvq'`

- [ ] **Step 3: Implement**

```python
# jarvis_jax/models/mvq/geometry.py
"""Affine (telecentric) camera geometry for the multi-view query model.

Every calibration in this pipeline has projection row 3 == [0,0,0,1]
(`tracking/affine_camera.py`), so projection is uv = M X + t with no camera
centre. Everything here is linear, which makes reprojection losses and the
geometric augmentations exact. Coordinates: X is ROI-LOCAL (world minus
center3D, world units); uv is CROP pixels (full-frame minus crop origin).
"""
from __future__ import annotations

import jax.numpy as jnp


def affine_rows(cam_mats):
    """(C,4,3) P.T -> M (C,2,3), t (C,2). Raises unless the cameras are affine."""
    P = jnp.swapaxes(jnp.asarray(cam_mats, jnp.float32), 1, 2)          # (C,3,4)
    row3 = P[:, 2, :]
    ok = jnp.all(jnp.abs(row3 - jnp.array([0.0, 0.0, 0.0, 1.0])) < 1e-6)
    if not bool(ok):
        raise ValueError("camera is not affine: projection row 3 != [0,0,0,1]")
    return P[:, :2, :3], P[:, :2, 3]


def local_offset(M, t, center3D, crop_origin):
    """t_local (C,2) so that uv_crop = M @ X_local + t_local."""
    return jnp.einsum("cij,j->ci", M, jnp.asarray(center3D, jnp.float32)) + t - crop_origin


def project_local(X, M, t_local):
    """X (...,3) local -> (...,C,2) crop px."""
    return jnp.einsum("cij,...j->...ci", M, X) + t_local


def ray_from_pixel(M, t_local, uv):
    """Back-project crop pixels to lines. p0 (C,N,3) least-norm point, d (C,3) unit direction."""
    Mt = jnp.swapaxes(M, 1, 2)                                            # (C,3,2)
    MMt_inv = jnp.linalg.inv(jnp.einsum("cij,ckj->cik", M, M))            # (C,2,2)
    pinv = jnp.einsum("cij,cjk->cik", Mt, MMt_inv)                        # (C,3,2)
    p0 = jnp.einsum("cij,cnj->cni", pinv, uv - t_local[:, None, :])
    d = jnp.cross(M[:, 0, :], M[:, 1, :])
    d = d / jnp.linalg.norm(d, axis=-1, keepdims=True)
    return p0, d


def token_pixel_centres(h: int, w: int, patch: int):
    ys, xs = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    return jnp.stack([(xs + 0.5) * patch, (ys + 0.5) * patch], -1).reshape(-1, 2).astype(jnp.float32)


def warp_cameras(M, t_local, A, b):
    return jnp.einsum("cij,cjk->cik", A, M), jnp.einsum("cij,cj->ci", A, t_local) + b


def rotate_world(M, R):
    return jnp.einsum("cij,kj->cik", M, R)          # M @ R.T


def mirror_world(M, t_local, crop: int):
    C = M.shape[0]
    A = jnp.broadcast_to(jnp.diag(jnp.array([-1.0, 1.0], jnp.float32)), (C, 2, 2))
    b = jnp.broadcast_to(jnp.array([crop - 1.0, 0.0], jnp.float32), (C, 2))
    S = jnp.diag(jnp.array([-1.0, 1.0, 1.0], jnp.float32))
    M2, t2 = warp_cameras(M, t_local, A, b)
    return jnp.einsum("cij,jk->cik", M2, S), t2


def px_scale(M):
    return jnp.mean(jnp.sqrt(jnp.sum(M ** 2, axis=(1, 2)) / 2.0))
```

- [ ] **Step 4: Run tests**

Run: `cd third_party/jarvis_jax && python -m pytest tests/test_mvq_geometry.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/models/mvq/__init__.py third_party/jarvis_jax/jarvis_jax/models/mvq/geometry.py third_party/jarvis_jax/tests/test_mvq_geometry.py
git commit -m "feat(mvq): affine camera geometry -- ray tokens, exact warp/rotate/mirror camera updates

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Synthetic v12 fixture + window dataset (`data/v12_windows.py`)

**Files:**
- Create: `tests/mvq_fixtures.py`, `jarvis_jax/data/v12_windows.py`
- Test: `tests/test_v12_windows.py`

**Interfaces:**
- Consumes: `ReprojectionTool`, `iter_resolved_slots`, `_resolve_sex` (from `data/v5_3d.py`), `crop_origin` (`data/transforms.py`), `affine_rows`/`local_offset` (Task 2, numpy-compatible via `np.asarray`).
- Produces:
  - `WINDOW_KEYS = ("crops","cam_valid","M","t_local","center3D","kp3d_local","has3d","kp2d","vis2d","fly_valid","px_scale","is_female","prompt_mask")`.
  - `class V12WindowDataset(root, split, T=1, *, max_flies=2, jitter_units=3.0, seed=0, train=True, recordings=None)` with `__len__`, `__getitem__(i) -> dict`, attributes `keypoint_names: list[str]`, `windows: list[tuple[rec, fly_id, frame0]]`, `calib_group(i) -> str`, `is_female(i) -> bool`, `n_flies(i) -> int`.
  - Sample dict shapes (K=50, C=7): `crops (T,C,448,448,3) uint8`, `cam_valid (T,C) bool`, `M (C,2,3) f32`, `t_local (T,C,2) f32`, `center3D (3,) f32`, `kp3d_local (F,T,K,3) f32`, `has3d (F,T,K) bool`, `kp2d (F,T,C,K,2) f32` crop px, `vis2d (F,T,C,K) bool`, `fly_valid (F,) bool` (fly 0 = host), `px_scale () f32`, `is_female () bool`, `prompt_mask (T,C,448,448) bool` (host's SAM mask in the crop, all-False when absent).
  - `window_batches(ds, batch_size, *, shuffle, seed, weights=None, num_workers=8, drop_last=True)` -> iterator of dicts of stacked numpy arrays (same keys), drawing indices with replacement from `weights` when given (mirrors `data/v3.py` balanced epochs).
  - `tests/mvq_fixtures.py: make_v12_root(tmp_path, *, n_frames=3, two_fly_frame=1, img_w=1936, img_h=448) -> str` and `CAMS` (7 names).

Behaviour to implement (spec §6.1): host fly defines `center3D` = midrange of its DLT 3D over visible joints (`quantize_center3d` NOT used — keep it continuous) plus uniform jitter in `[-jitter_units, +jitter_units]` per axis when `train=True`; crop origin per (frame, camera) = `crop_origin([px, py, 0, 0], w, h, 448)` at the projection of `center3D`; every OTHER labelled fly in the same (rec, frame) becomes instance 1..F-1 if any of its keypoints lands in any crop; `vis2d` = `v>0` AND inside the crop; `has3d` = DLT possible (>=2 views with v>0), 3D computed on full-frame 2D BEFORE cropping; camera rows placed BY NAME; None slots -> `cam_valid False`. T=2 windows exist only where frames `f, f+1` both have a frameset for the same `(rec, fly)`.

- [ ] **Step 1: Write the fixture**

```python
# tests/mvq_fixtures.py
"""Synthetic v12-shaped root: 7 affine cameras (real calibration rotated per
camera), 1936x448 JPEGs with a bright blob per fly, framesets keyed
'<rec>/Frame_<n>/fly<k>', frames 0..n_frames-1 consecutive, and a second
fly labelled only in `two_fly_frame`."""
import json
import os
import numpy as np
from PIL import Image

CAMS = ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855",
        "Cam2012857", "Cam2012861", "Cam2012862"]
REC = "2026_01_01_00_00_00"
K = 50
P_REAL = np.array([[8.1001, 0.0074869, -0.031773, 900.0],
                   [0.0093308, -8.0788, -0.17912, 300.0],
                   [0.0, 0.0, 0.0, 1.0]], np.float64)


def _names():
    legs = [f"{s}{i}{lr}_{p}" for lr in "LR" for i in (1, 2, 3) for s in "T"
            for p in ("Tro", "FeTi", "TiTa", "TaT1", "TaT3", "TaTip")]
    head = ["Antenna_Base", "EyeL", "EyeR", "Scutellum", "Abd_A4", "Abd_tip",
            "WingL_base", "WingL_V12", "WingL_V13", "T1L_ThxCx"]
    right = ["WingR_base", "WingR_V12", "WingR_V13", "T1R_ThxCx"]
    names = head + legs[:18] + right + legs[18:]
    assert len(names) == K, len(names)
    return names


def cam_P(i):
    th = 2 * np.pi * i / 7
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    P = P_REAL.copy(); P[:2, :3] = P[:2, :3] @ Rz
    return P


def fly_points(fly, frame, rng):
    """(K,3) world points: a 20-unit body around a per-fly centre that drifts 1 unit/frame."""
    centre = np.array([0.0, 0.0, 0.0]) if fly == 0 else np.array([40.0, 5.0, 0.0])
    centre = centre + frame * np.array([1.0, 0.5, 0.0])
    return centre + rng.normal(size=(K, 3)) * np.array([10.0, 4.0, 2.0])


def make_v12_root(tmp_path, *, n_frames=3, two_fly_frame=1, img_w=1936, img_h=448):
    root = tmp_path / "v12"; root.mkdir()
    names = _names()
    (root / "calibrations" / "A").mkdir(parents=True)
    for i, c in enumerate(CAMS):
        vals = ", ".join(f"{v:.16g}" for v in cam_P(i).ravel())
        (root / "calibrations" / "A" / f"{c}.yaml").write_text(
            "%YAML:1.0\n---\nimage_width: %d\nimage_height: %d\n"
            "projectionMatrix: !!opencv-matrix\n   rows: 3\n   cols: 4\n   dt: d\n"
            "   data: [ %s ]\nscale: 10\n" % (img_w, img_h, vals))
    rng = np.random.default_rng(0)
    images, anns, framesets = [], [], {}
    img_id = ann_id = 0
    for f in range(n_frames):
        flies = [0, 1] if f == two_fly_frame else [0]
        pts = {fly: fly_points(fly, f, rng) for fly in flies}
        per_fly = {fly: {"frames": [], "ann_ids": []} for fly in flies}
        for i, c in enumerate(CAMS):
            P = cam_P(i)
            img = np.zeros((img_h, img_w, 3), np.uint8)
            fn = f"{REC}/{c}/Frame_{f}.jpg"
            img_id += 1
            images.append({"id": img_id, "width": img_w, "height": img_h,
                           "recording": REC, "file_name": fn})
            for fly in flies:
                X = np.concatenate([pts[fly], np.ones((K, 1))], 1)
                uv = (X @ P.T)[:, :2]
                kps = np.stack([uv[:, 0], uv[:, 1], np.ones(K)], 1)
                cx, cy = int(uv[:, 0].mean()), int(uv[:, 1].mean())
                img[max(cy - 12, 0):cy + 12, max(cx - 12, 0):cx + 12] = 255
                anns.append({"id": ann_id, "image_id": img_id,
                             "bbox": [float(uv[:, 0].min()), float(uv[:, 1].min()),
                                      float(np.ptp(uv[:, 0])), float(np.ptp(uv[:, 1]))],
                             "keypoints": [float(v) for v in kps.ravel()], "num_keypoints": K,
                             "sex": "female" if fly == 0 else "male", "fly_id": fly,
                             "subset": "synthetic", "src_ann_id": ann_id})
                per_fly[fly]["frames"].append(img_id); per_fly[fly]["ann_ids"].append(ann_id)
                ann_id += 1
            os.makedirs(root / "images" / REC / c, exist_ok=True)
            Image.fromarray(img).save(root / "images" / REC / c / f"Frame_{f}.jpg", quality=90)
        for fly in flies:
            framesets[f"{REC}/Frame_{f}/fly{fly}"] = {"recording": REC, "fly_id": fly,
                                                     "subset": "synthetic", **per_fly[fly]}
    coco = {"keypoint_names": names, "skeleton": [], "categories": [{"id": 1, "name": "fly"}],
            "images": images, "annotations": anns, "framesets": framesets}
    (root / "annotations").mkdir()
    for split in ("train", "val"):
        json.dump(coco, open(root / "annotations" / f"instances_{split}.json", "w"))
    json.dump(names, open(root / "annotations" / "keypoint_names.json", "w"))
    json.dump({"version": "synthetic", "recordings": {REC: {
        "calib_group": "A", "sex": "mixed", "behavior": "courtship",
        "fly_sex": {"fly0": "female", "fly1": "male"}, "split": "train"}}},
              open(root / "manifest.json", "w"))
    return str(root)
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_v12_windows.py
import json, os
import numpy as np
import pytest
from tests.mvq_fixtures import make_v12_root, CAMS, REC, K


def test_window_census_t1_and_t2(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path, n_frames=3, two_fly_frame=1)
    d1 = V12WindowDataset(root, "train", T=1)
    d2 = V12WindowDataset(root, "train", T=2)
    assert len(d1) == 4          # fly0 x3 frames + fly1 x1
    assert len(d2) == 2          # fly0: (0,1), (1,2); fly1 has no consecutive pair
    assert d1.keypoint_names == json.load(open(os.path.join(root, "annotations", "keypoint_names.json")))


def test_sample_shapes_and_instances(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset, WINDOW_KEYS
    root = make_v12_root(tmp_path)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    i = ds.windows.index((REC, 0, 1))              # host fly0, frame 1 (two-fly frame)
    s = ds[i]
    assert set(s) == set(WINDOW_KEYS)
    assert s["crops"].shape == (1, 7, 448, 448, 3) and s["crops"].dtype == np.uint8
    assert s["cam_valid"].shape == (1, 7) and s["cam_valid"].all()
    assert s["M"].shape == (7, 2, 3) and s["t_local"].shape == (1, 7, 2)
    assert s["kp3d_local"].shape == (2, 1, K, 3) and s["kp2d"].shape == (2, 1, 7, K, 2)
    assert s["fly_valid"].tolist() == [True, True]
    assert ds.n_flies(i) == 2 and ds.is_female(i)
    j = ds.windows.index((REC, 0, 0))
    assert ds[j]["fly_valid"].tolist() == [True, False]


def test_labels_are_consistent_with_geometry(tmp_path):
    """GT 3D (local) reprojected through (M, t_local) must land on the GT 2D
    crop coords -- the invariant the reprojection loss relies on."""
    import jax.numpy as jnp
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.models.mvq.geometry import project_local
    root = make_v12_root(tmp_path)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    s = ds[0]
    uv = np.asarray(project_local(jnp.asarray(s["kp3d_local"][0, 0]), jnp.asarray(s["M"]),
                                  jnp.asarray(s["t_local"][0])))            # (K,C,2)
    vis = s["vis2d"][0, 0]                                                    # (C,K)
    gt = s["kp2d"][0, 0]                                                      # (C,K,2)
    err = np.linalg.norm(uv.transpose(1, 0, 2) - gt, axis=-1)[vis]
    assert err.max() < 0.5
    assert s["has3d"][0, 0].all()
    # the host's visible keypoints lie inside the crop
    assert (gt[vis] >= 0).all() and (gt[vis] <= 447).all()
    assert 7.5 < float(s["px_scale"]) < 8.5


def test_jitter_only_in_train_mode_and_bounded(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path)
    a = V12WindowDataset(root, "train", T=1, train=False)[0]["center3D"]
    b = V12WindowDataset(root, "train", T=1, train=False)[0]["center3D"]
    c = V12WindowDataset(root, "train", T=1, train=True, jitter_units=3.0, seed=1)[0]["center3D"]
    np.testing.assert_array_equal(a, b)
    assert 0 < np.abs(c - a).max() <= 3.0


def test_none_slot_marks_camera_invalid(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path)
    p = os.path.join(root, "annotations", "instances_train.json")
    coco = json.load(open(p))
    fs = coco["framesets"][f"{REC}/Frame_0/fly0"]
    fs["ann_ids"][2] = None                         # third listed camera unresolved
    json.dump(coco, open(p, "w"))
    ds = V12WindowDataset(root, "train", T=1, train=False)
    s = ds[ds.windows.index((REC, 0, 0))]
    assert s["cam_valid"].sum() == 6
    assert (~s["vis2d"][0, 0][~s["cam_valid"][0]]).all()


def test_t2_window_shares_center_and_has_both_frames(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path)
    ds = V12WindowDataset(root, "train", T=2, train=False)
    s = ds[ds.windows.index((REC, 0, 0))]
    assert s["crops"].shape[0] == 2 and s["has3d"][0].all()
    # frame 1 has the second fly, frame 0 does not: fly 1 valid, with vis only in frame 1
    assert s["fly_valid"][1] and not s["vis2d"][1, 0].any() and s["vis2d"][1, 1].any()


def test_batches_stack_and_weights(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset, window_batches
    root = make_v12_root(tmp_path)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    w = np.array([1.0, 0.0, 0.0, 0.0])
    b = next(window_batches(ds, 2, shuffle=True, seed=0, weights=w, num_workers=2))
    assert b["crops"].shape == (2, 1, 7, 448, 448, 3)
    np.testing.assert_array_equal(b["center3D"][0], b["center3D"][1])   # only index 0 has weight
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd third_party/jarvis_jax && python -m pytest tests/test_v12_windows.py -v`
Expected: FAIL, `No module named 'jarvis_jax.data.v12_windows'`

- [ ] **Step 4: Implement the dataset**

```python
# jarvis_jax/data/v12_windows.py
"""Window dataset over the v12 root for the multi-view query model (mvq).

A sample is (recording, host fly, start frame, T): T consecutive labelled
frames of the host, all 7 cameras, cropped at 448 around the projection of
ONE window-level center3D -- the inference convention of
predict/session_frameset.build_frameset, not the per-camera bbox crop of
data/v5_3d.py. Every other labelled fly inside the crops is an extra
instance (fly_valid), so the model can be trained as a set predictor.

Camera order is rt.cameras' sorted-glob order and slots are placed BY NAME
(see data/v5_3d.py docstring for why). Keypoint order is asserted against
annotations/keypoint_names.json.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from jarvis_jax.data.build_v5 import iter_resolved_slots
from jarvis_jax.data.transforms import crop_origin
from jarvis_jax.data.v5_3d import _load_mask, _resolve_sex, _frameset_own_sex
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

CROP = 448
WINDOW_KEYS = ("crops", "cam_valid", "M", "t_local", "center3D", "kp3d_local", "has3d",
               "kp2d", "vis2d", "fly_valid", "px_scale", "is_female", "prompt_mask")


def _parse_key(key):
    rec, frame, fly = key.split("/")
    return rec, int(frame.split("_")[1]), int(fly[3:])


def _affine_np(cam_mats):
    P = np.swapaxes(np.asarray(cam_mats, np.float64), 1, 2)          # (C,3,4)
    if not np.allclose(P[:, 2, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("non-affine calibration")
    return P[:, :2, :3], P[:, :2, 3]


class V12WindowDataset:
    def __init__(self, root, split, T=1, *, max_flies=2, jitter_units=3.0, seed=0,
                 train=True, recordings=None):
        self.root, self.split, self.T = root, split, int(T)
        self.max_flies, self.jitter, self.train = int(max_flies), float(jitter_units), bool(train)
        self.rng = np.random.default_rng(seed)
        coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
        self.manifest = json.load(open(os.path.join(root, "manifest.json")))["recordings"]
        self.keypoint_names = list(coco["keypoint_names"])
        canon = json.load(open(os.path.join(root, "annotations", "keypoint_names.json")))
        if self.keypoint_names != canon:
            raise ValueError("instances keypoint_names != annotations/keypoint_names.json")
        self.K = len(self.keypoint_names)
        self._img = {i["id"]: i for i in coco["images"]}
        self._ann = {a["id"]: a for a in coco["annotations"]}
        self._fs = {}                                   # (rec, frame, fly) -> frameset
        self._tools = {}
        for key, fsv in coco["framesets"].items():
            rec, frame, fly = _parse_key(key)
            if recordings is not None and rec not in recordings:
                continue
            grp = self.manifest[rec]["calib_group"]
            if grp not in self._tools:
                self._tools[grp] = ReprojectionTool(os.path.join(root, "calibrations", str(grp)))
            self._fs[(rec, frame, fly)] = fsv
        self.windows = []
        for (rec, frame, fly) in sorted(self._fs):
            if all((rec, frame + k, fly) in self._fs for k in range(self.T)):
                self.windows.append((rec, fly, frame))
        self._sex = {}
        for (rec, frame, fly), fsv in self._fs.items():
            self._sex[(rec, fly)] = _resolve_sex(_frameset_own_sex(fsv, self._ann), fly,
                                                 self.manifest.get(rec, {}))

    def __len__(self):
        return len(self.windows)

    def calib_group(self, i):
        return self.manifest[self.windows[i][0]]["calib_group"]

    def is_female(self, i):
        rec, fly, _ = self.windows[i]
        return self._sex[(rec, fly)] == "female"

    def n_flies(self, i):
        rec, fly, f0 = self.windows[i]
        others = {k[2] for k in self._fs if k[0] == rec and f0 <= k[1] < f0 + self.T and k[2] != fly}
        return 1 + min(len(others), self.max_flies - 1)

    # ------------------------------------------------------------------ helpers
    def _rt(self, rec):
        return self._tools[self.manifest[rec]["calib_group"]]

    def _labels_full(self, fsv, rt):
        """Per camera (BY NAME) full-frame (K,3) labels, image infos, ann infos."""
        C = rt.num_cameras
        cam_to_row = {n: i for i, n in enumerate(rt.cameras.keys())}
        kp = np.zeros((C, self.K, 3), np.float32)
        infos = [None] * C
        for img_id, ann_id in iter_resolved_slots(fsv):
            info, ann = self._img[img_id], self._ann[ann_id]
            c = cam_to_row.get(info["file_name"].split("/")[1])
            if c is None:
                continue
            k = np.asarray(ann["keypoints"], np.float32)
            if k.size == self.K * 3:
                kp[c] = k.reshape(-1, 3)
            infos[c] = (info, ann)
        return kp, infos

    def _dlt(self, kp, rt):
        C = rt.num_cameras
        X = np.zeros((self.K, 3), np.float32); has = np.zeros(self.K, bool)
        for j in range(self.K):
            use = [c for c in range(C) if kp[c, j, 2] > 0]
            if len(use) >= 2:
                pts = np.zeros((C, 2)); pts[use] = kp[use, j, :2]
                X[j] = rt.reconstruct_point(pts, cams_to_use=use); has[j] = True
        return X, has

    def _decode(self, info):
        with Image.open(os.path.join(self.root, "images", info["file_name"])) as im:
            return np.asarray(im.convert("RGB"), np.uint8)

    # ------------------------------------------------------------------ sample
    def __getitem__(self, i):
        rec, host, f0 = self.windows[i]
        rt = self._rt(rec); C = rt.num_cameras; T, K, F = self.T, self.K, self.max_flies
        cams = list(rt.cameras.keys())
        M, t = _affine_np(rt.camera_matrices)                         # (C,2,3),(C,2) float64

        # --- labels per frame per fly (full-frame), 3D via DLT, host first
        frames = [f0 + k for k in range(T)]
        others = sorted({k[2] for k in self._fs if k[0] == rec and k[1] in frames and k[2] != host})
        flies = [host] + others[: F - 1]
        kp_full = np.zeros((F, T, C, K, 3), np.float32)
        X3 = np.zeros((F, T, K, 3), np.float32); has3d = np.zeros((F, T, K), bool)
        infos = {}
        for fi, fly in enumerate(flies):
            for ti, f in enumerate(frames):
                fsv = self._fs.get((rec, f, fly))
                if fsv is None:
                    continue
                kp, inf = self._labels_full(fsv, rt)
                kp_full[fi, ti] = kp
                X3[fi, ti], has3d[fi, ti] = self._dlt(kp, rt)
                for c in range(C):
                    if inf[c] is not None:
                        infos.setdefault((ti, c), inf[c])
        cam_valid = np.zeros((T, C), bool)
        for (ti, c) in infos:
            cam_valid[ti, c] = True
        # host frameset None-slots: camera absent for this window frame
        for ti, f in enumerate(frames):
            fsv = self._fs[(rec, f, host)]
            present = {self._img[img]["file_name"].split("/")[1] for img, _ in iter_resolved_slots(fsv)}
            for c, name in enumerate(cams):
                if name not in present:
                    cam_valid[ti, c] = False

        # --- window centre from the host's 3D (frame 0), jittered in train mode
        vis0 = has3d[0, 0]
        pts = X3[0, 0][vis0] if vis0.any() else np.zeros((1, 3), np.float32)
        center = 0.5 * (pts.max(0) + pts.min(0))
        if self.train and self.jitter > 0:
            center = center + self.rng.uniform(-self.jitter, self.jitter, size=3)
        center = center.astype(np.float32)

        # --- crops around the projection of center (same origin for all frames of the window)
        origin = np.zeros((C, 2), np.int32)
        for c in range(C):
            info = next((infos[(ti, c)][0] for ti in range(T) if (ti, c) in infos), None)
            w, h = (info["width"], info["height"]) if info else (1936, 448)
            u, v = M[c] @ center + t[c]
            origin[c] = crop_origin([u, v, 0, 0], w, h, CROP)
        crops = np.zeros((T, C, CROP, CROP, 3), np.uint8)
        prompt = np.zeros((T, C, CROP, CROP), bool)
        for (ti, c), (info, ann) in infos.items():
            if not cam_valid[ti, c]:
                continue
            img = self._decode(info)
            x0, y0 = origin[c]
            crops[ti, c] = img[y0:y0 + CROP, x0:x0 + CROP]
            # host mask (may be absent -> zeros)
            fsv = self._fs[(rec, frames[ti], host)]
            for img_id, ann_id in iter_resolved_slots(fsv):
                if self._img[img_id]["file_name"] == info["file_name"]:
                    a = self._ann[ann_id]
                    m = _load_mask(self.root, info["file_name"], a.get("src_ann_id", ann_id),
                                   ann_id, info["width"], info["height"])
                    prompt[ti, c] = m[y0:y0 + CROP, x0:x0 + CROP].astype(bool)

        # --- to crop/local coordinates
        t_local = np.zeros((T, C, 2), np.float32)
        for ti in range(T):
            t_local[ti] = (M @ center + t - origin).astype(np.float32)
        kp2d = kp_full[..., :2] - origin[None, None, :, None, :]
        inside = ((kp2d >= 0) & (kp2d <= CROP - 1)).all(-1)
        vis2d = (kp_full[..., 2] > 0) & inside & cam_valid[None, :, :, None]
        fly_valid = np.array([fi < len(flies) and vis2d[fi].any() for fi in range(F)])
        fly_valid[0] = True
        px_scale = float(np.mean(np.sqrt((M ** 2).sum((1, 2)) / 2.0)))
        return {
            "crops": crops, "cam_valid": cam_valid,
            "M": M.astype(np.float32), "t_local": t_local, "center3D": center,
            "kp3d_local": (X3 - center).astype(np.float32) * has3d[..., None],
            "has3d": has3d, "kp2d": kp2d.astype(np.float32), "vis2d": vis2d,
            "fly_valid": fly_valid, "px_scale": np.float32(px_scale),
            "is_female": np.bool_(self.is_female(i)), "prompt_mask": prompt,
        }


def window_batches(ds, batch_size, *, shuffle=True, seed=0, weights=None, num_workers=8,
                   drop_last=True):
    rng = np.random.default_rng(seed)
    n = len(ds)
    if weights is not None:
        w = np.asarray(weights, np.float64); w = w / w.sum()
        idx = rng.choice(n, size=n, replace=True, p=w)
    else:
        idx = rng.permutation(n) if shuffle else np.arange(n)
    stop = (n // batch_size) * batch_size if drop_last else n
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        for s in range(0, stop, batch_size):
            samples = list(pool.map(ds.__getitem__, [int(i) for i in idx[s:s + batch_size]]))
            yield {k: np.stack([smp[k] for smp in samples]) for k in WINDOW_KEYS}
```

`_frameset_own_sex` and `_resolve_sex` are imported from `data/v5_3d.py` deliberately (same fallback chain as the 3D loader).

- [ ] **Step 5: Run tests**

Run: `cd third_party/jarvis_jax && python -m pytest tests/test_v12_windows.py -v`
Expected: 7 PASS. If `test_labels_are_consistent_with_geometry` fails with errors ~1 px, the crop origin was applied with a half-pixel offset: `t_local` must subtract the INTEGER origin, and `kp2d` must subtract the same integer.

- [ ] **Step 6: Census on the real root (login node is fine: JSON only, no images)**

```bash
cd third_party/jarvis_jax && python -c "
from jarvis_jax.data.v12_windows import V12WindowDataset as D
r='/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902'
for T in (1,2):
    for s in ('train','val'):
        d=D(r,s,T=T,train=False); print(T,s,len(d), sum(d.n_flies(i)>1 for i in range(len(d))),'two-fly')"
```
Expected: T=1 train 2661 / val 153; T=2 train ≈735 / val ≈15; two-fly counts ≈44 train / 37 val at T=1. Paste the numbers into the commit message.

- [ ] **Step 7: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/data/v12_windows.py third_party/jarvis_jax/tests/mvq_fixtures.py third_party/jarvis_jax/tests/test_v12_windows.py
git commit -m "feat(data): v12 window dataset for mvq -- inference-convention crops, multi-instance labels, T=1/T=2

Census on red_data_3d_v12_export0902: <numbers>.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Exact multi-view augmentation (`data/mv_augment.py`)

**Files:**
- Create: `jarvis_jax/data/mv_augment.py`
- Test: `tests/test_mv_augment.py`

**Interfaces:**
- Consumes: `affine_warp_image`, `photometric_batch`, `gaussian_blur_batch`, `gaussian_noise_batch`, `per_channel_multiply_batch`, `build_lr_swap` from `data/augment.py`; `warp_cameras`, `rotate_world`, `mirror_world` from Task 2.
- Produces:
  - `@dataclasses.dataclass class MVAugParams: enabled=True; rot_deg=30.0; scale_min=0.8; scale_max=1.25; translate_frac=0.1; world_yaw=True; world_tilt_deg=30.0; mirror_p=0.5; cam_drop_p=0.3; cam_drop_max=2; brightness=0.2; contrast=0.2; gamma=0.2; blur_max=0.5; noise_scale=0.02; pc_color=0.2`
  - `augment_window(key, batch: dict, params: MVAugParams, lr_swap: (K,) int) -> dict` — same keys as the input batch (Task 3 `WINDOW_KEYS` after device transfer), operating on `crops (B,T,C,448,448,3) uint8`, `M (B,C,2,3)`, `t_local (B,T,C,2)`, `kp3d_local (B,F,T,K,3)`, `kp2d (B,F,T,C,K,2)`, `vis2d`, `cam_valid`, `prompt_mask`. jit-safe (all randomness from `key`).
  - Order: per-view affine -> world rotation -> mirror -> camera dropout -> photometric/blur/noise/colour (RGB only).

Design notes: per-view affine samples `(theta, s, tx, ty)` per (sample, view) about the crop centre `(223.5, 223.5)`; image warped with `affine_warp_image(img, theta, s, tx*448, ty*448, 223.5, 223.5)` **(check its argument convention in `data/augment.py:47-73` and match it exactly: it maps OUTPUT pixels to input pixels, so the label transform must be the forward map — write the forward `A, b` explicitly and test them against a keypoint drawn as a bright pixel)**; cameras updated with `warp_cameras(M, t_local, A, b)`; `kp2d` updated as `A @ kp2d + b`; `vis2d` ANDed with in-bounds. World rotation: `R = Rz(yaw) @ Rx(tilt)`, one per sample (shared across frames and flies), `kp3d_local <- kp3d_local @ R.T`, `M <- rotate_world(M, R)`. Mirror per sample with prob `mirror_p`: crops flipped along width, `M, t_local <- mirror_world`, `kp3d_local <- kp3d_local * [-1,1,1]`, `kp2d[...,0] <- 447 - kp2d[...,0]`, keypoint axis permuted by `lr_swap` for kp3d_local/has3d/kp2d/vis2d. Camera dropout: per sample with prob `cam_drop_p`, zero `cam_valid` for `k ~ U{1..cam_drop_max}` random valid cameras (also zero their `vis2d`); never drop below 3 valid.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_augment.py
import numpy as np
import jax, jax.numpy as jnp
import pytest
from tests.mvq_fixtures import make_v12_root


def _batch(tmp_path, B=2):
    from jarvis_jax.data.v12_windows import V12WindowDataset, window_batches
    ds = V12WindowDataset(make_v12_root(tmp_path), "train", T=1, train=False)
    b = next(window_batches(ds, B, shuffle=False, num_workers=1))
    return {k: jnp.asarray(v) for k, v in b.items()}, ds.keypoint_names


def _reproj_err(b):
    from jarvis_jax.models.mvq.geometry import project_local
    errs = []
    for i in range(b["crops"].shape[0]):
        for t in range(b["crops"].shape[1]):
            uv = project_local(b["kp3d_local"][i, 0, t], b["M"][i], b["t_local"][i, t])  # (K,C,2)
            d = jnp.linalg.norm(jnp.swapaxes(uv, 0, 1) - b["kp2d"][i, 0, t], axis=-1)   # (C,K)
            m = b["vis2d"][i, 0, t] & b["has3d"][i, 0, t][None]
            errs.append(float(jnp.where(m, d, 0.0).max()))
    return max(errs)


def test_identity_when_disabled(tmp_path):
    from jarvis_jax.data.mv_augment import augment_window, MVAugParams
    from jarvis_jax.data.augment import build_lr_swap
    b, names = _batch(tmp_path)
    out = augment_window(jax.random.PRNGKey(0), b, MVAugParams(enabled=False), build_lr_swap(names))
    for k in b:
        np.testing.assert_array_equal(np.asarray(out[k]), np.asarray(b[k]))


@pytest.mark.parametrize("params", [
    dict(rot_deg=30.0, world_yaw=False, world_tilt_deg=0.0, mirror_p=0.0, cam_drop_p=0.0),
    dict(rot_deg=0.0, scale_min=1.0, scale_max=1.0, translate_frac=0.0, world_yaw=True, world_tilt_deg=30.0, mirror_p=0.0, cam_drop_p=0.0),
    dict(rot_deg=0.0, scale_min=1.0, scale_max=1.0, translate_frac=0.0, world_yaw=False, world_tilt_deg=0.0, mirror_p=1.0, cam_drop_p=0.0),
], ids=["per-view-affine", "world-rotation", "mirror"])
def test_each_geometric_aug_keeps_labels_consistent(tmp_path, params):
    from jarvis_jax.data.mv_augment import augment_window, MVAugParams
    from jarvis_jax.data.augment import build_lr_swap
    b, names = _batch(tmp_path)
    p = MVAugParams(brightness=0, contrast=0, gamma=0, blur_max=0, noise_scale=0, pc_color=0, **params)
    out = augment_window(jax.random.PRNGKey(3), b, p, build_lr_swap(names))
    assert _reproj_err(out) < 0.05          # 3D->2D consistency survives exactly


def test_mirror_swaps_left_right_and_flips_pixels(tmp_path):
    from jarvis_jax.data.mv_augment import augment_window, MVAugParams
    from jarvis_jax.data.augment import build_lr_swap
    b, names = _batch(tmp_path)
    swap = build_lr_swap(names)
    p = MVAugParams(rot_deg=0, scale_min=1, scale_max=1, translate_frac=0, world_yaw=False,
                    world_tilt_deg=0, mirror_p=1.0, cam_drop_p=0, brightness=0, contrast=0,
                    gamma=0, blur_max=0, noise_scale=0, pc_color=0)
    out = augment_window(jax.random.PRNGKey(0), b, p, swap)
    np.testing.assert_array_equal(np.asarray(out["crops"]), np.asarray(b["crops"][..., ::-1, :]))
    l = names.index("EyeL"); r = names.index("EyeR")
    np.testing.assert_allclose(np.asarray(out["kp2d"][..., l, 0]), 447.0 - np.asarray(b["kp2d"][..., r, 0]), atol=1e-4)
    np.testing.assert_allclose(np.asarray(out["kp3d_local"][..., l, :]),
                               np.asarray(b["kp3d_local"][..., r, :]) * np.array([-1, 1, 1]), atol=1e-5)


def test_per_view_affine_moves_pixels_with_labels(tmp_path):
    """Paint the EyeL label pixel white in one view; after warping, the
    transformed label must land on a white pixel."""
    from jarvis_jax.data.mv_augment import augment_window, MVAugParams
    from jarvis_jax.data.augment import build_lr_swap
    b, names = _batch(tmp_path, B=1)
    l = names.index("EyeL")
    crops = np.asarray(b["crops"]).copy(); crops[:] = 0
    u, v = np.round(np.asarray(b["kp2d"][0, 0, 0, :, l])).astype(int).T        # (C,)
    for c in range(7):
        crops[0, 0, c, max(v[c]-2, 0):v[c]+3, max(u[c]-2, 0):u[c]+3] = 255
    b["crops"] = jnp.asarray(crops)
    p = MVAugParams(rot_deg=25, scale_min=0.9, scale_max=1.1, translate_frac=0.05, world_yaw=False,
                    world_tilt_deg=0, mirror_p=0, cam_drop_p=0, brightness=0, contrast=0, gamma=0,
                    blur_max=0, noise_scale=0, pc_color=0)
    out = augment_window(jax.random.PRNGKey(7), b, p, build_lr_swap(names))
    uv = np.asarray(out["kp2d"][0, 0, 0, :, l]); vis = np.asarray(out["vis2d"][0, 0, 0, :, l])
    img = np.asarray(out["crops"][0, 0])
    for c in np.where(vis)[0]:
        x, y = np.round(uv[c]).astype(int)
        assert img[c, y, x].max() > 128, f"cam {c}: label off the painted pixel"


def test_camera_dropout_never_below_three(tmp_path):
    from jarvis_jax.data.mv_augment import augment_window, MVAugParams
    from jarvis_jax.data.augment import build_lr_swap
    b, names = _batch(tmp_path)
    p = MVAugParams(cam_drop_p=1.0, cam_drop_max=6, rot_deg=0, scale_min=1, scale_max=1, translate_frac=0,
                    world_yaw=False, world_tilt_deg=0, mirror_p=0, brightness=0, contrast=0, gamma=0,
                    blur_max=0, noise_scale=0, pc_color=0)
    out = augment_window(jax.random.PRNGKey(1), b, p, build_lr_swap(names))
    cv = np.asarray(out["cam_valid"])
    assert (cv.sum(-1) >= 3).all() and (cv.sum(-1) < 7).any()
    assert not np.asarray(out["vis2d"])[..., ~cv[0, 0], :][0].any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd third_party/jarvis_jax && python -m pytest tests/test_mv_augment.py -v`
Expected: FAIL, `No module named 'jarvis_jax.data.mv_augment'`

- [ ] **Step 3: Implement**

```python
# jarvis_jax/data/mv_augment.py
"""Multi-view augmentation for mvq windows. Every geometric op updates the
affine cameras so GT 3D still reprojects onto GT 2D exactly (tested)."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from jarvis_jax.data.augment import (affine_warp_image, photometric_batch, gaussian_blur_batch,
                                     gaussian_noise_batch, per_channel_multiply_batch)
from jarvis_jax.models.mvq.geometry import warp_cameras, rotate_world, mirror_world

CROP = 448


@dataclasses.dataclass
class MVAugParams:
    enabled: bool = True
    rot_deg: float = 30.0
    scale_min: float = 0.8
    scale_max: float = 1.25
    translate_frac: float = 0.1
    world_yaw: bool = True
    world_tilt_deg: float = 30.0
    mirror_p: float = 0.5
    cam_drop_p: float = 0.3
    cam_drop_max: int = 2
    brightness: float = 0.2
    contrast: float = 0.2
    gamma: float = 0.2
    blur_max: float = 0.5
    noise_scale: float = 0.02
    pc_color: float = 0.2


def _forward_affine(theta, s, tx, ty, c=(CROP - 1) / 2.0):
    """Forward map p' = A p + b for a rotation by theta and scale s about the
    crop centre followed by a translation (tx, ty) in px."""
    ct, st = jnp.cos(theta), jnp.sin(theta)
    A = s * jnp.array([[ct, -st], [st, ct]])
    b = jnp.array([c, c]) - A @ jnp.array([c, c]) + jnp.array([tx, ty])
    return A, b


def _warp_rgb(img, A, b):
    """Warp (H,W,3) uint8 by the FORWARD map (A, b): output pixel q samples input A^-1 (q - b)."""
    H, W = img.shape[:2]
    Ai = jnp.linalg.inv(A)
    ys, xs = jnp.meshgrid(jnp.arange(H, dtype=jnp.float32), jnp.arange(W, dtype=jnp.float32), indexing="ij")
    q = jnp.stack([xs, ys], -1) - b
    src = q @ Ai.T                                                    # (H,W,2) x,y
    coords = [src[..., 1], src[..., 0]]
    out = jnp.stack([jax.scipy.ndimage.map_coordinates(img[..., ch].astype(jnp.float32), coords,
                                                       order=1, mode="constant", cval=0.0)
                     for ch in range(img.shape[-1])], -1)
    return jnp.clip(jnp.round(out), 0, 255).astype(jnp.uint8)


def _per_view_affine(key, b, p):
    B, T, C = b["crops"].shape[:3]
    k1, k2, k3, k4 = jax.random.split(key, 4)
    theta = jnp.deg2rad(jax.random.uniform(k1, (B, C), minval=-p.rot_deg, maxval=p.rot_deg))
    s = jax.random.uniform(k2, (B, C), minval=p.scale_min, maxval=p.scale_max)
    tx = jax.random.uniform(k3, (B, C), minval=-p.translate_frac, maxval=p.translate_frac) * CROP
    ty = jax.random.uniform(k4, (B, C), minval=-p.translate_frac, maxval=p.translate_frac) * CROP
    A, bb = jax.vmap(jax.vmap(_forward_affine))(theta, s, tx, ty)      # (B,C,2,2), (B,C,2)
    warp = jax.vmap(jax.vmap(jax.vmap(_warp_rgb, in_axes=(0, 0, 0)), in_axes=(0, None, None)))
    crops = warp(b["crops"], A, bb)                                    # over B, T, C
    pm = jax.vmap(jax.vmap(jax.vmap(lambda m, A_, b_: _warp_rgb(m[..., None].astype(jnp.uint8) * 255, A_, b_)[..., 0] > 127,
                                    in_axes=(0, 0, 0)), in_axes=(0, None, None)))(b["prompt_mask"], A, bb)
    # warp_cameras(M, t, A, b) per sample; t_local is per (sample, frame), so write it out:
    M = jnp.einsum("bcij,bcjk->bcik", A, b["M"])
    tl = jnp.einsum("bcij,btcj->btci", A, b["t_local"]) + bb[:, None, :, :]
    kp = jnp.einsum("bcij,bftckj->bftcki", A, b["kp2d"]) + bb[:, None, None, :, None, :]
    inb = ((kp >= 0) & (kp <= CROP - 1)).all(-1)
    return {**b, "crops": crops, "prompt_mask": pm, "M": M, "t_local": tl, "kp2d": kp,
            "vis2d": b["vis2d"] & inb}
```

Continue the module:

```python
def _rot_mats(key, B, p):
    ky, kt = jax.random.split(key)
    yaw = jax.random.uniform(ky, (B,), minval=0.0, maxval=2 * jnp.pi) if p.world_yaw else jnp.zeros((B,))
    tilt = jnp.deg2rad(jax.random.uniform(kt, (B,), minval=-p.world_tilt_deg, maxval=p.world_tilt_deg))
    cz, sz, cx, sx = jnp.cos(yaw), jnp.sin(yaw), jnp.cos(tilt), jnp.sin(tilt)
    Rz = jnp.stack([jnp.stack([cz, -sz, 0 * cz], -1), jnp.stack([sz, cz, 0 * cz], -1),
                    jnp.stack([0 * cz, 0 * cz, 1 + 0 * cz], -1)], -2)
    Rx = jnp.stack([jnp.stack([1 + 0 * cx, 0 * cx, 0 * cx], -1), jnp.stack([0 * cx, cx, -sx], -1),
                    jnp.stack([0 * cx, sx, cx], -1)], -2)
    return Rz @ Rx                                                         # (B,3,3)


def _world_rotation(key, b, p):
    R = _rot_mats(key, b["crops"].shape[0], p)
    M = jax.vmap(rotate_world)(b["M"], R)
    X = jnp.einsum("bij,bftkj->bftki", R, b["kp3d_local"])
    return {**b, "M": M, "kp3d_local": X}


def _mirror(key, b, p, lr_swap):
    B = b["crops"].shape[0]
    do = jax.random.bernoulli(key, p.mirror_p, (B,))
    M2, tl2 = jax.vmap(lambda M_, t_: mirror_world(M_, t_, CROP), in_axes=(0, 0))(
        b["M"], b["t_local"][:, 0])
    # t_local is identical across frames only if origins are shared per window (they are);
    # apply the same b-shift to every frame:
    tl2 = jnp.broadcast_to(tl2[:, None], b["t_local"].shape)
    sel = lambda a, m: jnp.where(do.reshape((B,) + (1,) * (a.ndim - 1)), m, a)
    X = b["kp3d_local"][..., lr_swap, :] * jnp.array([-1.0, 1.0, 1.0])
    kp = b["kp2d"][..., lr_swap, :].at[..., 0].set(CROP - 1 - b["kp2d"][..., lr_swap, 0])
    return {**b,
            "crops": sel(b["crops"], b["crops"][..., ::-1, :]),
            "prompt_mask": sel(b["prompt_mask"], b["prompt_mask"][..., ::-1]),
            "M": sel(b["M"], M2), "t_local": sel(b["t_local"], tl2),
            "kp3d_local": sel(b["kp3d_local"], X), "has3d": sel(b["has3d"], b["has3d"][..., lr_swap]),
            "kp2d": sel(b["kp2d"], kp), "vis2d": sel(b["vis2d"], b["vis2d"][..., lr_swap])}


def _camera_dropout(key, b, p):
    B, T, C = b["cam_valid"].shape
    k1, k2, k3 = jax.random.split(key, 3)
    do = jax.random.bernoulli(k1, p.cam_drop_p, (B,))
    n_drop = jax.random.randint(k2, (B,), 1, p.cam_drop_max + 1)
    score = jax.random.uniform(k3, (B, C)) + (~b["cam_valid"][:, 0]).astype(jnp.float32)   # invalid sort last
    order = jnp.argsort(score, axis=1)
    rank = jnp.argsort(order, axis=1)                                     # rank of each cam
    n_valid = b["cam_valid"][:, 0].sum(1)
    n_drop = jnp.minimum(n_drop, jnp.maximum(n_valid - 3, 0))
    drop = (rank < n_drop[:, None]) & do[:, None]                          # (B,C)
    cv = b["cam_valid"] & ~drop[:, None, :]
    return {**b, "cam_valid": cv, "vis2d": b["vis2d"] & cv[:, None, :, :, None]}


def _photometric(key, b, p):
    B, T, C = b["crops"].shape[:3]
    flat = b["crops"].reshape((B * T * C,) + b["crops"].shape[3:])
    flat4 = jnp.concatenate([flat, jnp.zeros(flat.shape[:-1] + (1,), flat.dtype)], -1)   # reuse 4-ch helpers
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x = photometric_batch(k1, flat4, p.brightness, p.contrast, p.gamma)
    x = gaussian_blur_batch(k2, x, p.blur_max)
    x = gaussian_noise_batch(k3, x, p.noise_scale)
    x = per_channel_multiply_batch(k4, x, p.pc_color)
    return {**b, "crops": x[..., :3].reshape(b["crops"].shape)}


def augment_window(key, b, params: MVAugParams, lr_swap):
    if not params.enabled:
        return b
    lr_swap = jnp.asarray(lr_swap)
    ka, kr, km, kd, kp = jax.random.split(key, 5)
    if params.rot_deg > 0 or params.scale_min != 1 or params.scale_max != 1 or params.translate_frac > 0:
        b = _per_view_affine(ka, b, params)
    if params.world_yaw or params.world_tilt_deg > 0:
        b = _world_rotation(kr, b, params)
    if params.mirror_p > 0:
        b = _mirror(km, b, params, lr_swap)
    if params.cam_drop_p > 0:
        b = _camera_dropout(kd, b, params)
    if any(v > 0 for v in (params.brightness, params.contrast, params.gamma, params.blur_max,
                           params.noise_scale, params.pc_color)):
        b = _photometric(kp, b, params)
    return b
```

`data/augment.py`'s helpers take `(B,H,W,4)` uint8 and touch channels 0-2 only; the 4th channel is padded with zeros and dropped again. Check `gaussian_blur_batch`/`gaussian_noise_batch` signatures at `data/augment.py:398,419` if a positional-argument error appears.

- [ ] **Step 4: Run tests**

Run: `cd third_party/jarvis_jax && python -m pytest tests/test_mv_augment.py -v`
Expected: 7 PASS. The per-view-affine consistency test is the one that catches a wrong forward/inverse convention: if it fails with errors of tens of px, `_warp_rgb` and `_forward_affine` disagree — the label map is the forward `A p + b` and the image is sampled at `A^-1 (q - b)`.

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/data/mv_augment.py third_party/jarvis_jax/tests/test_mv_augment.py
git commit -m "feat(data): exact multi-view augmentation for mvq (per-view affine w/ camera update, world rotation, global mirror, camera dropout)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Fusion, decoder, and the MVQ model (`models/mvq/{fusion,decoder,model}.py`)

**Files:**
- Create: `jarvis_jax/models/mvq/fusion.py`, `jarvis_jax/models/mvq/decoder.py`, `jarvis_jax/models/mvq/model.py`; update `jarvis_jax/models/mvq/__init__.py` to export `MVQConfig, MVQModel`
- Test: `tests/test_mvq_model.py`

**Interfaces:**
- Consumes: `DINOv3`, `DINOv3Config` (Task 1); `ray_from_pixel`, `token_pixel_centres`, `project_local` (Task 2).
- Produces:
  - `MVQConfig` (frozen dataclass): `crop=448, patch=16, embed_dim=768, num_keypoints=50, num_cameras=7, max_frames=8, n_instances=3, n_local=2, n_global=2, global_pool=2, dec_layers_3d=8, dec_layers_2d=4, dec_heads=12, mlp_ratio=4.0, refine_passes=1, patch_rgb=9, fourier_bands=8, roi_scale=24.0, camera_slot_embed=True, backbone="dinov3_b16", backbone_depth=12, backbone_heads=12, remat=True`.
  - `MVQModel(cfg, *, rngs)` with attribute `backbone` (a `DINOv3`), and
    `__call__(crops_norm: (B,T,C,H,W,3) f32, cam_valid: (B,T,C) bool, M: (B,C,2,3), t_local: (B,T,C,2), prompt_mask: (B,T,C,H,W) bool | None, *, prompt_on: (B,) bool | None) -> dict` with keys:
    - `xyz (B,I,T,K,3)` ROI-local world units; `conf_logit (B,I,T,K)`; `exist_logit (B,I)`;
    - `uv (B,I,T,C,K,2)` crop px; `vis_logit (B,I,T,C,K)`;
    - `aux_pass1: dict | None` — the pass-1 readout (same five keys) when `refine_passes>0`, else `None`;
    - `aux_layers: list[dict]` — intermediate 3D readouts (`xyz`, `conf_logit`, `exist_logit`) from every second 3D-decoder layer of every pass, for deep supervision.
  - `fusion.py: FusionStack(cfg, rngs)`, `__call__(tokens (B,T*C,N,D), valid (B,T*C) bool) -> (B,T*C,N,D)`.
  - `decoder.py: QueryDecoder(cfg, rngs)`, `__call__(bank (B,L,D), bank_valid (B,L) bool, geom_cam (B,C,D), frame_emb (T,D), prompt_tok (B,D)|None, prompt_on (B,)|None, refine_ctx=None) -> (out dict, per_layer list)`; `gather_refine_context(...)`.
  - `model.py: build_bank(...)` internal; `MVQModel.assemble(out, center3D, crop_origin, exist_thresh=0.5) -> (kp3d_world (B,I,T,K,3) with NaN, conf3d, kp2d_full (B,I,T,C,K,2))` (numpy-side helper for later phases; unit-tested for NaN policy).

Design (spec §4.3–4.6), stated so the implementer does not have to re-derive it:
1. **Bank.** Backbone on `crops_norm` reshaped to `(B*T*C, H, W, 3)` (remat per block) -> `(B,T,C,N,D)`, N = (448/16)^2 = 784. Add: `geom = Linear(Fourier([p0, d]))` where `(p0, d) = ray_from_pixel(M, t_local[t], token_pixel_centres)` per (t, c) (p0 divided by `roi_scale` before the Fourier features so values are O(1)); `frame_emb[t]` (learned, `max_frames` rows); optional learned `cam_slot[c]`. Invalid cameras: tokens zeroed and masked in every attention.
2. **Fusion.** `n_local + n_global` blocks alternating local/global. Local block: standard pre-norm self-attention over each view's N tokens (batch = B*T*C). Global block: average-pool each view's 28x28 grid by `global_pool` (-> 196 tokens), self-attend over all `T*C*196` tokens with key mask from `cam_valid`, then add the per-pooled-token update back to its `global_pool^2` children (nearest upsample). Both blocks use `LayerScale` initialised to 0.
3. **Queries.** `E_inst (I,D)`, `E_kp (K,D)`, `E_frame (max_frames, D)` (shared with the bank's frame embedding), `E_pass (2,D)`. 3D query `q[i,t,k] = E_inst[i] + E_kp[k] + E_frame[t] (+ prompt_tok if i==0 and prompt_on)`. Prompt token = masked mean of bank tokens under the 28x28-downsampled `prompt_mask` over valid views -> `Linear`.
4. **3D path.** `dec_layers_3d` blocks: self-attn over all `I*T*K` queries -> cross-attn to bank (key mask) -> MLP; pre-norm, width D, `dec_heads`. Heads (shared across layers for deep supervision): `xyz = roi_scale * Linear(D->3)`, `conf_logit = Linear(D->1)`; `exist_logit = Linear(mean over (t,k) of instance features)`.
5. **2D path.** view query `q[i,t,c,k] = Linear(h3d[i,t,k]) + geom_cam[c] + E_frame[t]` where `geom_cam[c] = Linear(Fourier(M[c] rows flattened / px_scale-free: use M[c].ravel() and t_local[t,c]/crop))`; `dec_layers_2d` cross-attn-only blocks; heads `uv = crop * sigmoid(Linear(D->2))`, `vis_logit = Linear(D->1)`.
6. **Refinement pass** (`refine_passes=1`): from pass-1 `xyz`, reproject with `project_local` to each valid view -> bilinear-sample the (28x28) token grid (4-corner gather) and a `patch_rgb x patch_rgb` RGB patch (nearest-centre gather on the un-normalised crop is fine; pass the uint8 crops through as `crops_u8` for this, or use `crops_norm`), average over valid views, `Linear` -> add to the 3D query together with `Fourier(uv/crop)` mean and `E_pass[1]`; rerun the SAME 3D and 2D decoder weights; outputs are `pass1 + residual` for `xyz`/`uv`, and replace for logits. Pass-1 readout goes into `aux`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mvq_model.py
import numpy as np
import jax, jax.numpy as jnp
import pytest
from flax import nnx

TINY = dict(crop=64, patch=16, embed_dim=32, num_keypoints=5, num_cameras=3, max_frames=4,
            n_instances=2, n_local=1, n_global=1, global_pool=2, dec_layers_3d=2, dec_layers_2d=1,
            dec_heads=4, mlp_ratio=2.0, refine_passes=1, patch_rgb=3, fourier_bands=4,
            roi_scale=24.0, backbone_depth=1, backbone_heads=4, remat=False)


def _inputs(B=2, T=2, C=3, H=64, seed=0):
    rng = np.random.default_rng(seed)
    P = np.array([[8.1, 0.0, 0.0, 30.0], [0.0, -8.1, -0.2, 30.0], [0, 0, 0, 1.0]])
    Ms, ts = [], []
    for c in range(C):
        th = 2 * np.pi * c / C
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        Ms.append(P[:2, :3] @ Rz); ts.append(P[:2, 3])
    M = jnp.asarray(np.broadcast_to(np.stack(Ms), (B, C, 2, 3)).astype(np.float32))
    tl = jnp.asarray(np.broadcast_to(np.stack(ts), (B, T, C, 2)).astype(np.float32))
    crops = jnp.asarray(rng.normal(size=(B, T, C, H, H, 3)).astype(np.float32))
    cam_valid = jnp.ones((B, T, C), bool).at[1, :, 2].set(False)
    pm = jnp.zeros((B, T, C, H, H), bool).at[:, :, :, 20:40, 20:40].set(True)
    return crops, cam_valid, M, tl, pm


def test_output_shapes_and_aux():
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    cfg = MVQConfig(**TINY)
    m = MVQModel(cfg, rngs=nnx.Rngs(0))
    crops, cv, M, tl, pm = _inputs()
    out = m(crops, cv, M, tl, pm, prompt_on=jnp.array([True, False]))
    B, T, C, I, K = 2, 2, 3, 2, 5
    assert out["xyz"].shape == (B, I, T, K, 3) and out["conf_logit"].shape == (B, I, T, K)
    assert out["exist_logit"].shape == (B, I)
    assert out["uv"].shape == (B, I, T, C, K, 2) and out["vis_logit"].shape == (B, I, T, C, K)
    assert out["aux_pass1"] is not None and set(out["aux_pass1"]) >= {"xyz", "uv", "conf_logit", "vis_logit", "exist_logit"}
    assert isinstance(out["aux_layers"], list) and all("xyz" in a for a in out["aux_layers"])
    assert bool(jnp.isfinite(out["xyz"]).all()) and bool((out["uv"] >= 0).all()) and bool((out["uv"] <= 64).all())


def test_invalid_camera_does_not_influence_outputs():
    """Zero/garbage in a masked view must not change the 3D output."""
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    m = MVQModel(MVQConfig(**TINY), rngs=nnx.Rngs(0))
    crops, cv, M, tl, pm = _inputs()
    cv = cv.at[:, :, 1].set(False)
    a = m(crops, cv, M, tl, pm)["xyz"]
    crops2 = crops.at[:, :, 1].set(crops[:, :, 1] * 100.0 + 7.0)
    b = m(crops2, cv, M, tl, pm)["xyz"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-4)


def test_prompt_changes_only_instance_zero_when_on():
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    m = MVQModel(MVQConfig(**TINY), rngs=nnx.Rngs(0))
    crops, cv, M, tl, pm = _inputs()
    off = m(crops, cv, M, tl, pm, prompt_on=jnp.array([False, False]))["xyz"]
    on = m(crops, cv, M, tl, pm, prompt_on=jnp.array([True, True]))["xyz"]
    assert not np.allclose(np.asarray(off[:, 0]), np.asarray(on[:, 0]))
    # instance 1 sees the prompt only through self-attention; with fresh weights the
    # difference must be much smaller than instance 0's
    d0 = float(jnp.abs(off[:, 0] - on[:, 0]).mean()); d1 = float(jnp.abs(off[:, 1] - on[:, 1]).mean())
    assert d1 < d0


def test_zero_global_layers_is_approach_b():
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    cfg = MVQConfig(**{**TINY, "n_global": 0})
    m = MVQModel(cfg, rngs=nnx.Rngs(0))
    crops, cv, M, tl, pm = _inputs()
    assert m(crops, cv, M, tl, pm)["xyz"].shape == (2, 2, 2, 5, 3)


def test_fusion_layerscale_zero_init_is_identity():
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.models.mvq.fusion import FusionStack
    cfg = MVQConfig(**TINY)
    fs = FusionStack(cfg, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 6, 16, 32)).astype(np.float32))
    valid = jnp.ones((2, 6), bool)
    np.testing.assert_allclose(np.asarray(fs(x, valid)), np.asarray(x), atol=1e-6)


def test_assemble_nan_policy():
    from jarvis_jax.models.mvq.model import assemble
    B, I, T, K, C = 1, 2, 1, 3, 2
    out = {"xyz": np.zeros((B, I, T, K, 3), np.float32), "conf_logit": np.zeros((B, I, T, K), np.float32),
           "exist_logit": np.array([[3.0, -3.0]], np.float32),
           "uv": np.zeros((B, I, T, C, K, 2), np.float32), "vis_logit": np.zeros((B, I, T, C, K), np.float32)}
    kp3d, conf3d, kp2d = assemble(out, center3D=np.array([[1.0, 2.0, 3.0]]),
                                  crop_origin=np.zeros((B, C, 2)), exist_thresh=0.5)
    assert np.isnan(kp3d[0, 1]).all() and (conf3d[0, 1] == 0).all()
    np.testing.assert_allclose(kp3d[0, 0, 0, 0], [1.0, 2.0, 3.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd third_party/jarvis_jax && python -m pytest tests/test_mvq_model.py -v`
Expected: FAIL, `cannot import name 'MVQConfig'`

- [ ] **Step 3: Implement `fusion.py`**

```python
# jarvis_jax/models/mvq/fusion.py
"""Interleaved per-view / global fusion over the multi-view token bank."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


def masked_attention(q, k, v, key_valid, num_heads):
    """q (B,Nq,D), k/v (B,Nk,D), key_valid (B,Nk) bool -> (B,Nq,D)."""
    B, Nq, D = q.shape; Nk = k.shape[1]; hd = D // num_heads
    qh = q.reshape(B, Nq, num_heads, hd).transpose(0, 2, 1, 3)
    kh = k.reshape(B, Nk, num_heads, hd).transpose(0, 2, 1, 3)
    vh = v.reshape(B, Nk, num_heads, hd).transpose(0, 2, 1, 3)
    logits = (qh @ kh.transpose(0, 1, 3, 2)) * (hd ** -0.5)
    logits = jnp.where(key_valid[:, None, None, :], logits, -1e9)
    att = jax.nn.softmax(logits, axis=-1)
    return (att @ vh).transpose(0, 2, 1, 3).reshape(B, Nq, D)


class Attn(nnx.Module):
    def __init__(self, D, heads, *, rngs):
        self.q, self.k, self.v, self.o = (nnx.Linear(D, D, rngs=rngs) for _ in range(4))
        self.heads = heads

    def __call__(self, x, ctx, ctx_valid):
        return self.o(masked_attention(self.q(x), self.k(ctx), self.v(ctx), ctx_valid, self.heads))


class MLP(nnx.Module):
    def __init__(self, D, ratio, *, rngs):
        self.fc1 = nnx.Linear(D, int(D * ratio), rngs=rngs); self.fc2 = nnx.Linear(int(D * ratio), D, rngs=rngs)

    def __call__(self, x):
        return self.fc2(jax.nn.gelu(self.fc1(x)))


class SelfBlock(nnx.Module):
    """Pre-norm self-attention + MLP with LayerScale (init 0 => identity at step 0)."""
    def __init__(self, D, heads, ratio, *, rngs, ls_init=0.0):
        self.n1 = nnx.LayerNorm(D, rngs=rngs); self.attn = Attn(D, heads, rngs=rngs)
        self.n2 = nnx.LayerNorm(D, rngs=rngs); self.mlp = MLP(D, ratio, rngs=rngs)
        self.ls1 = nnx.Param(jnp.full((D,), ls_init)); self.ls2 = nnx.Param(jnp.full((D,), ls_init))

    def __call__(self, x, valid):
        h = self.n1(x)
        x = x + self.ls1[...] * self.attn(h, h, valid)
        return x + self.ls2[...] * self.mlp(self.n2(x))


class FusionStack(nnx.Module):
    def __init__(self, cfg, *, rngs):
        D, H = cfg.embed_dim, cfg.dec_heads
        self.cfg = cfg
        n = max(cfg.n_local, cfg.n_global)
        self.blocks = nnx.List([])
        self.kinds = []
        for i in range(n):
            if i < cfg.n_local:
                self.blocks.append(SelfBlock(D, H, cfg.mlp_ratio, rngs=rngs)); self.kinds.append("local")
            if i < cfg.n_global:
                self.blocks.append(SelfBlock(D, H, cfg.mlp_ratio, rngs=rngs)); self.kinds.append("global")

    def __call__(self, tokens, valid):
        """tokens (B,V,N,D) with V = T*C views, valid (B,V)."""
        B, V, N, D = tokens.shape
        g = int(round(N ** 0.5)); p = self.cfg.global_pool
        for kind, blk in zip(self.kinds, self.blocks):
            if kind == "local":
                x = tokens.reshape(B * V, N, D)
                ok = jnp.broadcast_to(valid.reshape(B * V, 1), (B * V, N))
                tokens = blk(x, ok).reshape(B, V, N, D)
            else:
                grid = tokens.reshape(B, V, g, g, D)
                pooled = grid.reshape(B, V, g // p, p, g // p, p, D).mean(axis=(3, 5))     # (B,V,g/p,g/p,D)
                q = (g // p) ** 2
                flat = pooled.reshape(B, V * q, D)
                ok = jnp.repeat(valid, q, axis=1)
                upd = blk(flat, ok) - flat                                                  # (B,V*q,D)
                upd = upd.reshape(B, V, g // p, 1, g // p, 1, D)
                upd = jnp.broadcast_to(upd, (B, V, g // p, p, g // p, p, D)).reshape(B, V, N, D)
                tokens = tokens + upd
            tokens = tokens * valid[:, :, None, None]
        return tokens
```

- [ ] **Step 4: Implement `decoder.py`**

```python
# jarvis_jax/models/mvq/decoder.py
"""D4RT-style query decoder: heavy 3D path (query self-attn + cross-attn),
light 2D path (cross-attn only), shared heads, refinement context gather."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jarvis_jax.models.mvq.fusion import Attn, MLP, masked_attention


def fourier(x, bands):
    """(...,d) in ~[-1,1] -> (..., d*(2*bands+1))."""
    freqs = (2.0 ** jnp.arange(bands)) * jnp.pi
    ang = x[..., None] * freqs
    return jnp.concatenate([x[..., None], jnp.sin(ang), jnp.cos(ang)], -1).reshape(x.shape[:-1] + (-1,))


class CrossBlock(nnx.Module):
    def __init__(self, D, heads, ratio, *, rngs, self_attn: bool):
        self.self_attn = self_attn
        if self_attn:
            self.ns = nnx.LayerNorm(D, rngs=rngs); self.sa = Attn(D, heads, rngs=rngs)
        self.nq = nnx.LayerNorm(D, rngs=rngs); self.nk = nnx.LayerNorm(D, rngs=rngs)
        self.ca = Attn(D, heads, rngs=rngs)
        self.nm = nnx.LayerNorm(D, rngs=rngs); self.mlp = MLP(D, ratio, rngs=rngs)

    def __call__(self, q, bank, bank_valid):
        if self.self_attn:
            h = self.ns(q); ok = jnp.ones(q.shape[:2], bool)
            q = q + self.sa(h, h, ok)
        q = q + self.ca(self.nq(q), self.nk(bank), bank_valid)
        return q + self.mlp(self.nm(q))


class Heads(nnx.Module):
    def __init__(self, D, *, rngs):
        self.xyz = nnx.Linear(D, 3, rngs=rngs, kernel_init=nnx.initializers.zeros)
        self.conf = nnx.Linear(D, 1, rngs=rngs)
        self.exist = nnx.Linear(D, 1, rngs=rngs)
        self.uv = nnx.Linear(D, 2, rngs=rngs, kernel_init=nnx.initializers.zeros)
        self.vis = nnx.Linear(D, 1, rngs=rngs)


class QueryDecoder(nnx.Module):
    def __init__(self, cfg, *, rngs):
        D, H, R = cfg.embed_dim, cfg.dec_heads, cfg.mlp_ratio
        self.cfg = cfg
        self.e_inst = nnx.Param(jax.random.normal(rngs.params(), (cfg.n_instances, D)) * 0.02)
        self.e_kp = nnx.Param(jax.random.normal(rngs.params(), (cfg.num_keypoints, D)) * 0.02)
        self.e_pass = nnx.Param(jnp.zeros((2, D)))
        self.blocks3d = nnx.List([CrossBlock(D, H, R, rngs=rngs, self_attn=True) for _ in range(cfg.dec_layers_3d)])
        self.blocks2d = nnx.List([CrossBlock(D, H, R, rngs=rngs, self_attn=False) for _ in range(cfg.dec_layers_2d)])
        self.to_view = nnx.Linear(D, D, rngs=rngs)
        nf = 2 * cfg.fourier_bands + 1
        self.geom_cam = nnx.Linear(8 * nf, D, rngs=rngs)          # Fourier(M.ravel()/10 (6), t_local/crop (2))
        self.refine_in = nnx.Linear(D + 3 * cfg.patch_rgb ** 2 + 2 * nf, D, rngs=rngs,
                                    kernel_init=nnx.initializers.zeros)
        self.prompt_proj = nnx.Linear(D, D, rngs=rngs)
        self.heads = Heads(D, rngs=rngs)

    # ------------------------------------------------------------ readouts
    def _read3d(self, h, I, T, K):
        B = h.shape[0]
        h4 = h.reshape(B, I, T, K, -1)
        return {"xyz": self.cfg.roi_scale * self.heads.xyz(h4),
                "conf_logit": self.heads.conf(h4)[..., 0],
                "exist_logit": self.heads.exist(h4.mean(axis=(2, 3)))[..., 0]}

    def _path2d(self, h3d, bank, bank_valid, gcam, femb, I, T, K):
        B, C = h3d.shape[0], gcam.shape[1]
        base = self.to_view(h3d).reshape(B, I, T, 1, K, -1)
        q = base + gcam[:, None, None, :, None, :] + femb[None, None, :, None, None, :]
        q = q.reshape(B, I * T * C * K, -1)
        for blk in self.blocks2d:
            q = blk(q, bank, bank_valid)
        q = q.reshape(B, I, T, C, K, -1)
        return {"uv": self.cfg.crop * jax.nn.sigmoid(self.heads.uv(q)),
                "vis_logit": self.heads.vis(q)[..., 0]}

    def __call__(self, bank, bank_valid, M, t_local, femb, prompt_tok=None, prompt_on=None,
                 refine_ctx=None, pass_idx=0):
        """bank (B,L,D); M (B,C,2,3); t_local (B,T,C,2); femb (T,D)."""
        cfg = self.cfg; B = bank.shape[0]; I, K = cfg.n_instances, cfg.num_keypoints
        T, C = t_local.shape[1], t_local.shape[2]
        q = (self.e_inst[...][None, :, None, None, :] + self.e_kp[...][None, None, None, :, :]
             + femb[None, None, :, None, :])                                  # (B?,I,T,K,D) broadcast
        q = jnp.broadcast_to(q, (B, I, T, K, q.shape[-1]))
        if prompt_tok is not None:
            on = prompt_on if prompt_on is not None else jnp.ones((B,), bool)
            add = self.prompt_proj(prompt_tok) * on[:, None]
            q = q.at[:, 0].add(add[:, None, None, :])
        q = q + self.e_pass[...][pass_idx]
        if refine_ctx is not None:
            q = q + self.refine_in(refine_ctx)
        q = q.reshape(B, I * T * K, -1)
        per_layer = []
        for li, blk in enumerate(self.blocks3d):
            q = blk(q, bank, bank_valid)
            if li % 2 == 1 and li != len(self.blocks3d) - 1:
                per_layer.append(self._read3d(q, I, T, K))
        out = self._read3d(q, I, T, K)
        gfeat = jnp.concatenate([M.reshape(B, C, 6) / 10.0,
                                 t_local.mean(axis=1) / cfg.crop], -1)            # (B,C,8)
        gcam = self.geom_cam(fourier(gfeat, cfg.fourier_bands))
        out.update(self._path2d(q, bank, bank_valid, gcam, femb, I, T, K))
        return out, per_layer, q


def gather_refine_context(xyz, M, t_local, cam_valid, grid_tokens, crops, patch, bands, crop):
    """xyz (B,I,T,K,3) -> context (B,I,T,K, D + 3*patch^2 + 2*(2*bands+1)).
    grid_tokens (B,T,C,g,g,D); crops (B,T,C,H,W,3) float."""
    from jarvis_jax.models.mvq.geometry import project_local
    B, I, T, K, _ = xyz.shape; C = M.shape[1]; g = grid_tokens.shape[3]; H = crops.shape[3]
    uv = jax.vmap(lambda X, m, tl: jax.vmap(lambda Xt, tlt: project_local(Xt, m, tlt), in_axes=(1, 0), out_axes=1)(X, tl))(
        xyz, M, t_local)                                                       # (B,I,T,K,C,2)
    uv = jnp.moveaxis(uv, 4, 3)                                                # (B,I,T,C,K,2)
    scale = g / crop
    gx = jnp.clip(uv[..., 0] * scale - 0.5, 0, g - 1); gy = jnp.clip(uv[..., 1] * scale - 0.5, 0, g - 1)
    x0 = jnp.floor(gx).astype(jnp.int32); y0 = jnp.floor(gy).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, g - 1); y1 = jnp.minimum(y0 + 1, g - 1)
    wx, wy = gx - x0, gy - y0
    def take(ix, iy):                                                          # (B,I,T,C,K) idx -> (B,I,T,C,K,D)
        flat = grid_tokens.reshape(B, T, C, g * g, -1)
        idx = (iy * g + ix)                                                    # (B,I,T,C,K)
        idx_b = jnp.moveaxis(idx, 1, -1)                                       # (B,T,C,K,I)
        got = jnp.take_along_axis(flat[:, :, :, :, None, :], idx_b.reshape(B, T, C, K * I, 1, 1), axis=3)
        return jnp.moveaxis(got.reshape(B, T, C, K, I, -1), 4, 1)              # (B,I,T,C,K,D)
    tok = ((1 - wx) * (1 - wy))[..., None] * take(x0, y0) + (wx * (1 - wy))[..., None] * take(x1, y0) \
        + ((1 - wx) * wy)[..., None] * take(x0, y1) + (wx * wy)[..., None] * take(x1, y1)
    # RGB patch: nearest centre, offsets -r..r
    r = patch // 2
    cu = jnp.clip(jnp.round(uv[..., 0]).astype(jnp.int32), r, H - 1 - r)
    cv = jnp.clip(jnp.round(uv[..., 1]).astype(jnp.int32), r, H - 1 - r)
    offs = jnp.arange(-r, r + 1)
    pix = crops.reshape(B, T, C, H * H, 3)
    pidx = (cv[..., None, None] + offs[None, None, None, None, None, :, None]) * H \
         + (cu[..., None, None] + offs[None, None, None, None, None, None, :])       # (B,I,T,C,K,p,p)
    pidx = jnp.moveaxis(pidx.reshape(B, I, T, C, K * patch * patch), 1, -1).reshape(B, T, C, -1)   # (B,T,C,K*p*p*I)
    rgb = jnp.take_along_axis(pix, pidx[..., None], axis=3)                    # (B,T,C,K*p*p*I,3)
    rgb = rgb.reshape(B, T, C, K, patch * patch, I, 3)
    rgb = jnp.moveaxis(rgb, 5, 1).reshape(B, I, T, C, K, 3 * patch * patch)
    vw = cam_valid[:, None, :, :, None, None].astype(jnp.float32)              # (B,1,T,C,1,1)
    den = jnp.maximum(vw.sum(3), 1.0)
    tok_m = (tok * vw).sum(3) / den; rgb_m = (rgb * vw).sum(3) / den           # (B,I,T,K,·)
    fuv = (fourier((uv / crop) * 2 - 1, bands) * vw).sum(3) / den
    return jnp.concatenate([tok_m, rgb_m, fuv], -1)
```

- [ ] **Step 5: Implement `model.py` and `__init__.py`**

```python
# jarvis_jax/models/mvq/model.py
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jarvis_jax.models.dinov3 import DINOv3, DINOv3Config
from jarvis_jax.models.mvq.decoder import QueryDecoder, fourier, gather_refine_context
from jarvis_jax.models.mvq.fusion import FusionStack
from jarvis_jax.models.mvq.geometry import ray_from_pixel, token_pixel_centres


@dataclasses.dataclass(frozen=True)
class MVQConfig:
    crop: int = 448
    patch: int = 16
    embed_dim: int = 768
    num_keypoints: int = 50
    num_cameras: int = 7
    max_frames: int = 8
    n_instances: int = 3
    n_local: int = 2
    n_global: int = 2
    global_pool: int = 2
    dec_layers_3d: int = 8
    dec_layers_2d: int = 4
    dec_heads: int = 12
    mlp_ratio: float = 4.0
    refine_passes: int = 1
    patch_rgb: int = 9
    fourier_bands: int = 8
    roi_scale: float = 24.0
    camera_slot_embed: bool = True
    backbone: str = "dinov3_b16"
    backbone_depth: int = 12
    backbone_heads: int = 12
    remat: bool = True

    @property
    def grid(self):
        return self.crop // self.patch


class MVQModel(nnx.Module):
    def __init__(self, cfg: MVQConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        D = cfg.embed_dim
        base = DINOv3Config.vitl16() if cfg.backbone == "dinov3_l16" else DINOv3Config.vitb16()
        self.backbone = DINOv3(dataclasses.replace(base, embed_dim=D, depth=cfg.backbone_depth,
                                                   num_heads=cfg.backbone_heads), rngs=rngs)
        nf = 2 * cfg.fourier_bands + 1
        self.geom = nnx.Linear(6 * nf, D, rngs=rngs)
        self.e_frame = nnx.Param(jax.random.normal(rngs.params(), (cfg.max_frames, D)) * 0.02)
        self.e_cam = nnx.Param(jnp.zeros((cfg.num_cameras, D)))
        self.fusion = FusionStack(cfg, rngs=rngs)
        self.decoder = QueryDecoder(cfg, rngs=rngs)
        self.prompt_ln = nnx.LayerNorm(D, rngs=rngs)

    def _bank(self, crops, cam_valid, M, t_local):
        cfg = self.cfg; B, T, C, H, W, _ = crops.shape; g = cfg.grid; N = g * g
        toks = self.backbone(crops.reshape(B * T * C, H, W, 3), remat=cfg.remat).reshape(B, T, C, N, -1)
        centres = token_pixel_centres(g, g, cfg.patch)                                    # (N,2)
        def rays(Mb, tlb):                                                                # per sample, per frame
            p0, d = ray_from_pixel(Mb, tlb, jnp.broadcast_to(centres, (C, N, 2)))
            return jnp.concatenate([p0 / cfg.roi_scale, jnp.broadcast_to(d[:, None], p0.shape)], -1)  # (C,N,6)
        feats = jax.vmap(lambda Mb, tl: jax.vmap(lambda tlt: rays(Mb, tlt))(tl))(M, t_local)      # (B,T,C,N,6)
        toks = toks + self.geom(fourier(jnp.clip(feats, -3, 3) / 3.0, cfg.fourier_bands))
        toks = toks + self.e_frame[...][:T][None, :, None, None, :]
        if cfg.camera_slot_embed:
            toks = toks + self.e_cam[...][None, None, :, None, :]
        toks = toks * cam_valid[..., None, None]
        toks = self.fusion(toks.reshape(B, T * C, N, -1), cam_valid.reshape(B, T * C))
        return toks.reshape(B, T, C, N, -1)

    def _prompt(self, grid_tokens, prompt_mask, cam_valid):
        cfg = self.cfg; B, T, C, N, D = grid_tokens.shape; g = cfg.grid
        m = prompt_mask.reshape(B, T, C, g, cfg.patch, g, cfg.patch).mean(axis=(4, 6)) > 0.5   # (B,T,C,g,g)
        m = (m.reshape(B, T, C, N) & cam_valid[..., None]).astype(jnp.float32)
        num = (grid_tokens * m[..., None]).sum(axis=(1, 2, 3)); den = jnp.maximum(m.sum(axis=(1, 2, 3)), 1.0)
        return self.prompt_ln(num / den[:, None])

    def __call__(self, crops, cam_valid, M, t_local, prompt_mask=None, *, prompt_on=None):
        cfg = self.cfg; B, T, C = crops.shape[:3]; N = cfg.grid ** 2
        grid_tokens = self._bank(crops, cam_valid, M, t_local)                               # (B,T,C,N,D)
        bank = grid_tokens.reshape(B, T * C * N, -1)
        bank_valid = jnp.repeat(cam_valid.reshape(B, T * C), N, axis=1)
        femb = self.e_frame[...][:T]
        ptok = self._prompt(grid_tokens, prompt_mask, cam_valid) if prompt_mask is not None else None
        out, per_layer, _ = self.decoder(bank, bank_valid, M, t_local, femb, ptok, prompt_on, None, 0)
        aux_layers, aux_pass1 = list(per_layer), None
        for p in range(cfg.refine_passes):
            aux_pass1 = out
            ctx = gather_refine_context(out["xyz"], M, t_local, cam_valid,
                                        grid_tokens.reshape(B, T, C, cfg.grid, cfg.grid, -1), crops,
                                        cfg.patch_rgb, cfg.fourier_bands, cfg.crop)
            ctx = jax.lax.stop_gradient(ctx)
            out2, per2, _ = self.decoder(bank, bank_valid, M, t_local, femb, ptok, prompt_on, ctx, 1)
            out2["xyz"] = out["xyz"] + out2["xyz"]
            out2["uv"] = jnp.clip(out["uv"] + (out2["uv"] - cfg.crop / 2.0), 0.0, cfg.crop)
            aux_layers.extend(per2); out = out2
        out["aux_pass1"], out["aux_layers"] = aux_pass1, aux_layers
        return out


def assemble(out, center3D, crop_origin, exist_thresh=0.5):
    """numpy: model outputs -> world kp3d (NaN where absent), conf3d, full-frame kp2d."""
    xyz, conf = np.asarray(out["xyz"]), 1 / (1 + np.exp(-np.asarray(out["conf_logit"])))
    exist = 1 / (1 + np.exp(-np.asarray(out["exist_logit"]))) >= exist_thresh                 # (B,I)
    kp3d = xyz + np.asarray(center3D)[:, None, None, None, :]
    kp3d = np.where(exist[:, :, None, None, None], kp3d, np.nan)
    conf3d = np.where(exist[:, :, None, None], conf, 0.0)
    kp2d = np.asarray(out["uv"]) + np.asarray(crop_origin)[:, None, None, :, None, :]
    return kp3d.astype(np.float32), conf3d.astype(np.float32), kp2d.astype(np.float32)
```

```python
# jarvis_jax/models/mvq/__init__.py
from jarvis_jax.models.mvq.model import MVQConfig, MVQModel, assemble  # noqa: F401
```

Note the refinement-pass `uv` residual: the 2D head emits `crop*sigmoid(.)`, so `(out2["uv"] - crop/2)` is a residual centred at zero when the head is at its zero-init. The `xyz` head is zero-initialised, so pass 2 starts as an identity refinement.

- [ ] **Step 6: Run tests**

Run: `cd third_party/jarvis_jax && python -m pytest tests/test_mvq_model.py -v`
Expected: 6 PASS. `test_invalid_camera_does_not_influence_outputs` is the important one: if it fails, a masked camera leaks through the pooled global block's mean (pool BEFORE zeroing) or through the refine gather (weights `vw` not applied).

- [ ] **Step 7: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/models/mvq/ third_party/jarvis_jax/tests/test_mvq_model.py
git commit -m "feat(mvq): fusion stack, D4RT-style query decoder with promptable instances and refinement pass, MVQModel

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Instance matching and the loss (`train/matching.py`, `train/losses_mvq.py`)

**Files:**
- Create: `jarvis_jax/train/matching.py`, `jarvis_jax/train/losses_mvq.py`
- Test: `tests/test_mvq_losses.py`

**Interfaces:**
- Consumes: model output dict (Task 5), batch labels (Task 3 keys), `project_local` (Task 2), `build_part_index` (`data/distractor.py`).
- Produces:
  - `matching.py: enumerate_assignments(n_inst: int, n_flies: int) -> np.ndarray (n_cand, n_flies)` int, each row = instance index per fly (injective), static; `match(cost: (B,I,F), fly_valid: (B,F), pin_first: (B,) bool) -> assign (B,F) int32` (instance per fly; entries for invalid flies are -1) and `inst_matched (B,I) bool`.
  - `losses_mvq.py: LossWeights` dataclass with the spec values (`reproj=1.0, l3d=0.5, uv2d=0.5, vis=0.1, conf=0.2, exist=1.0, rep=0.5, pass1=0.5, aux=0.3, huber_px=8.0, rep_px=20.0, rep_units=2.5`); `mvq_loss(out, batch, w: LossWeights, part_of_k: np.ndarray) -> (total: scalar, metrics: dict[str, scalar])` where `batch` holds `M, t_local, kp3d_local, has3d, kp2d, vis2d, fly_valid, px_scale, cam_valid, prompt_on (B,) bool`; metrics include every term, `match_reproj_px` (mean reprojection px of matched instances, final pass), `mpjpe3d_units`, `exist_acc`.

Formulas (all pixel units; `s = px_scale (B,)`):
- Cost for matching (final pass, no grad): `cost[b,i,f] = mean_{t,c,k: vis2d} Huber(|proj(xyz[i]) - kp2d[f]|) + s * mean_{t,k: has3d} |xyz[i]-kp3d[f]|_1` (with the two means over their own masks; use large cost when a fly has no labels).
- After matching, gather per-fly predictions `P_f = out[..., assign[b,f], ...]` and compute terms 1–7 of spec §5 exactly as listed; `Huber(d, δ)` elementwise on the 2D error vector then mean over the 2 coords; term 5 is `c * e_reproj_perkp − w.conf * log c` with `c = sigmoid(conf_logit)` and `e_reproj_perkp` the mean-over-views Huber of that keypoint (detached inside the confidence term's error factor is NOT done: D4RT lets it flow; keep it flowing); term 7 for fly f uses the OTHER matched flies' labels of the same part: 2D hinge `relu(rep_px − |uv_pred − kp2d_other|)` over labelled other-keypoints with `part_of_k` equal, and 3D hinge `s * relu(rep_units − |xyz_pred − kp3d_other|)`.
- Deep supervision: terms 1–3 on `out["aux_pass1"]` (when not `None`) at weight `w.pass1`; terms 1–2 (no 2D head in intermediate readouts) on each entry of `out["aux_layers"]` at weight `w.aux`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mvq_losses.py
import numpy as np
import jax, jax.numpy as jnp
import pytest


def test_enumerate_assignments_is_injective_and_complete():
    from jarvis_jax.train.matching import enumerate_assignments
    a = enumerate_assignments(3, 2)
    assert a.shape == (6, 2) and all(len(set(r)) == 2 for r in a)
    assert enumerate_assignments(2, 2).shape == (2, 2)


def test_match_agrees_with_scipy_hungarian():
    from scipy.optimize import linear_sum_assignment
    from jarvis_jax.train.matching import match
    rng = np.random.default_rng(0)
    cost = rng.uniform(size=(16, 3, 2)).astype(np.float32)
    fv = np.ones((16, 2), bool)
    assign, inst_m = match(jnp.asarray(cost), jnp.asarray(fv), jnp.zeros(16, bool))
    for b in range(16):
        rows, cols = linear_sum_assignment(cost[b].T)          # flies x instances
        ref = np.full(2, -1); ref[rows] = cols
        assert list(np.asarray(assign[b])) == list(ref)
    assert np.asarray(inst_m).sum(1).tolist() == [2] * 16


def test_match_pins_fly0_to_instance0_and_ignores_invalid_flies():
    from jarvis_jax.train.matching import match
    cost = jnp.asarray([[[0.0, 5.0], [1.0, 0.1], [9.0, 9.0]]])        # instance 1 is best for fly 0
    assign, inst_m = match(cost, jnp.asarray([[True, False]]), jnp.asarray([True]))
    assert np.asarray(assign).tolist() == [[0, -1]]
    assert np.asarray(inst_m).tolist() == [[True, False, False]]


def _perfect_batch(B=2, I=3, T=1, C=3, K=5, seed=0):
    """Labels + an output that reproduces them exactly (instance 0 <- fly 0, 2 <- fly 1)."""
    from jarvis_jax.models.mvq.geometry import project_local
    rng = np.random.default_rng(seed)
    M = np.stack([np.array([[8.0, 0.1 * c, 0.0], [0.0, -8.0, 0.2 * c]]) for c in range(C)]).astype(np.float32)
    M = np.broadcast_to(M, (B, C, 2, 3)).copy()
    tl = np.broadcast_to(np.array([[224.0, 224.0]] * C, np.float32), (B, T, C, 2)).copy()
    X = rng.normal(size=(B, 2, T, K, 3)).astype(np.float32) * 5                    # flies' 3D
    kp2d = np.stack([np.stack([np.asarray(project_local(jnp.asarray(X[b, f, t]), jnp.asarray(M[b]), jnp.asarray(tl[b, t])))
                               for t in range(T)]) for f in range(2)]) for b in range(B)])  # (B,F,T,K,C,2)
    kp2d = np.moveaxis(kp2d, 4, 3)                                                  # (B,F,T,C,K,2)
    batch = {"M": M, "t_local": tl, "kp3d_local": X, "has3d": np.ones((B, 2, T, K), bool),
             "kp2d": kp2d.astype(np.float32), "vis2d": np.ones((B, 2, T, C, K), bool),
             "fly_valid": np.ones((B, 2), bool), "px_scale": np.full((B,), 8.0, np.float32),
             "cam_valid": np.ones((B, T, C), bool), "prompt_on": np.zeros((B,), bool)}
    xyz = np.zeros((B, I, T, K, 3), np.float32); xyz[:, 0] = X[:, 0]; xyz[:, 2] = X[:, 1]; xyz[:, 1] = 50.0
    uv = np.zeros((B, I, T, C, K, 2), np.float32); uv[:, 0] = kp2d[:, 0]; uv[:, 2] = kp2d[:, 1]
    big = np.full((B, I, T, K), 6.0, np.float32)
    out = {"xyz": xyz, "conf_logit": big, "exist_logit": np.array([[6.0, -6.0, 6.0]] * B, np.float32),
           "uv": uv, "vis_logit": np.full((B, I, T, C, K), 6.0, np.float32), "aux_pass1": None, "aux_layers": []}
    out["aux_pass1"] = {k: v for k, v in out.items() if k not in ("aux_pass1", "aux_layers")}
    j = lambda d: {k: (jnp.asarray(v) if not isinstance(v, (dict, list)) and v is not None else v) for k, v in d.items()}
    out = j(out); out["aux_pass1"] = j(out["aux_pass1"])
    return out, j(batch)


def test_loss_near_zero_at_ground_truth_and_metrics():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    out, batch = _perfect_batch()
    part_of_k = np.arange(5, dtype=np.int32)
    total, m = mvq_loss(out, batch, LossWeights(), part_of_k)
    assert float(m["reproj"]) < 1e-3 and float(m["l3d"]) < 1e-3 and float(m["uv2d"]) < 1e-3
    assert float(m["rep"]) == 0.0 and float(m["exist_acc"]) == 1.0
    assert float(m["match_reproj_px"]) < 1e-2 and float(m["mpjpe3d_units"]) < 1e-3
    # only the -log c and BCE floors remain
    assert float(total) < 0.05


def test_masked_entries_do_not_contribute():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    out, batch = _perfect_batch()
    out["xyz"] = out["xyz"].at[:, 0, :, 0].add(100.0)                    # corrupt keypoint 0 of fly 0
    _, m_bad = mvq_loss(out, batch, LossWeights(), np.arange(5, dtype=np.int32))
    batch["vis2d"] = batch["vis2d"].at[:, 0, :, :, 0].set(False)
    batch["has3d"] = batch["has3d"].at[:, 0, :, 0].set(False)
    _, m_masked = mvq_loss(out, batch, LossWeights(), np.arange(5, dtype=np.int32))
    assert float(m_bad["reproj"]) > 1.0 and float(m_masked["reproj"]) < 1e-3
    assert float(m_masked["l3d"]) < 1e-3


def test_repulsion_fires_only_near_other_fly_same_part():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    out, batch = _perfect_batch()
    # move fly-0 prediction keypoint 1 onto fly-1's keypoint 1 (same part) in 3D and 2D
    out["xyz"] = out["xyz"].at[:, 0, :, 1].set(batch["kp3d_local"][:, 1, :, 1])
    out["uv"] = out["uv"].at[:, 0, :, :, 1].set(batch["kp2d"][:, 1, :, :, 1])
    _, m_same = mvq_loss(out, batch, LossWeights(), np.arange(5, dtype=np.int32))
    _, m_diff = mvq_loss(out, batch, LossWeights(), np.array([0, 9, 2, 3, 4], np.int32))   # part 9 unique
    assert float(m_same["rep"]) > 0.0 and float(m_diff["rep"]) == 0.0


def test_confidence_term_prefers_low_c_on_bad_points():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    out, batch = _perfect_batch()
    out["xyz"] = out["xyz"].at[:, 0].add(3.0)                             # ~24 px reprojection error
    hi = dict(out); hi["conf_logit"] = jnp.full_like(out["conf_logit"], 4.0)
    lo = dict(out); lo["conf_logit"] = jnp.full_like(out["conf_logit"], -4.0)
    _, m_hi = mvq_loss(hi, batch, LossWeights(), np.arange(5, dtype=np.int32))
    _, m_lo = mvq_loss(lo, batch, LossWeights(), np.arange(5, dtype=np.int32))
    assert float(m_lo["conf"]) < float(m_hi["conf"])


def test_loss_is_differentiable_and_jittable():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    out, batch = _perfect_batch()
    pk = np.arange(5, dtype=np.int32)
    f = jax.jit(lambda xyz: mvq_loss({**out, "xyz": xyz}, batch, LossWeights(), pk)[0])
    g = jax.grad(f)(out["xyz"])
    assert g.shape == out["xyz"].shape and bool(jnp.isfinite(g).all())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd third_party/jarvis_jax && python -m pytest tests/test_mvq_losses.py -v`
Expected: FAIL, `No module named 'jarvis_jax.train.matching'`

- [ ] **Step 3: Implement `matching.py`**

```python
# jarvis_jax/train/matching.py
"""Instance <-> labelled-fly assignment by enumeration (I<=4, F<=2)."""
from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np


def enumerate_assignments(n_inst: int, n_flies: int) -> np.ndarray:
    return np.asarray(list(itertools.permutations(range(n_inst), n_flies)), np.int32)


def match(cost, fly_valid, pin_first):
    """cost (B,I,F) -> assign (B,F) instance index (-1 for invalid flies), inst_matched (B,I)."""
    B, I, F = cost.shape
    cand = jnp.asarray(enumerate_assignments(I, F))                          # (n,F)
    c = jnp.take_along_axis(jnp.broadcast_to(cost[:, None], (B, cand.shape[0], I, F)),
                            cand[None, :, None, :], axis=2)[:, :, 0, :]      # (B,n,F)
    c = jnp.where(fly_valid[:, None, :], c, 0.0).sum(-1)                     # (B,n)
    pinned_ok = (cand[:, 0] == 0)[None, :] | ~pin_first[:, None]
    c = jnp.where(pinned_ok, c, jnp.inf)
    best = jnp.argmin(c, axis=1)                                             # (B,)
    assign = cand[best]                                                      # (B,F)
    assign = jnp.where(fly_valid, assign, -1)
    inst_matched = (jnp.arange(I)[None, :, None] == assign[:, None, :]).any(-1)
    return assign.astype(jnp.int32), inst_matched
```

- [ ] **Step 4: Implement `losses_mvq.py`**

```python
# jarvis_jax/train/losses_mvq.py
"""Loss for the multi-view query lifter (spec 2026-09-03 §5). All terms in px."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from jarvis_jax.models.mvq.geometry import project_local
from jarvis_jax.train.matching import match


@dataclasses.dataclass(frozen=True)
class LossWeights:
    reproj: float = 1.0
    l3d: float = 0.5
    uv2d: float = 0.5
    vis: float = 0.1
    conf: float = 0.2
    exist: float = 1.0
    rep: float = 0.5
    pass1: float = 0.5
    aux: float = 0.3
    huber_px: float = 8.0
    rep_px: float = 20.0
    rep_units: float = 2.5


def _huber(d, delta):
    a = jnp.abs(d)
    return jnp.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


def _mmean(x, m):
    m = m.astype(x.dtype)
    return (x * m).sum() / jnp.maximum(m.sum(), 1.0)


def _reproject(xyz, M, t_local):
    """xyz (B,N,T,K,3) -> (B,N,T,C,K,2)."""
    uv = jax.vmap(lambda X, Mb, tl: jax.vmap(lambda Xt, tlt: project_local(Xt, Mb, tlt), in_axes=(1, 0), out_axes=1)(X, tl))(
        xyz, M, t_local)                                                        # (B,N,T,K,C,2)
    return jnp.moveaxis(uv, 4, 3)


def _gather_inst(x, assign):
    """x (B,I,...) , assign (B,F) -> (B,F,...) (index clamped; invalid flies masked by caller)."""
    idx = jnp.clip(assign, 0, x.shape[1] - 1)
    return jnp.take_along_axis(x, idx.reshape(idx.shape + (1,) * (x.ndim - 2)), axis=1)


def _geo_terms(pred, batch, w, valid_f):
    """Terms 1-3 for a readout `pred` already gathered per fly: dict with xyz (B,F,T,K,3), uv (B,F,T,C,K,2)."""
    s = batch["px_scale"][:, None, None, None]
    uv_re = _reproject(pred["xyz"], batch["M"], batch["t_local"])
    e2 = _huber(uv_re - batch["kp2d"], w.huber_px).mean(-1)                    # (B,F,T,C,K)
    m2 = batch["vis2d"] & valid_f[:, :, None, None, None]
    reproj = _mmean(e2, m2)
    m3 = batch["has3d"] & valid_f[:, :, None, None]
    l3d = _mmean(s * jnp.abs(pred["xyz"] - batch["kp3d_local"]).mean(-1), m3)
    uv2d = _mmean(_huber(pred["uv"] - batch["kp2d"], w.huber_px).mean(-1), m2)
    per_kp = (e2 * m2).sum(3) / jnp.maximum(m2.sum(3), 1.0)                     # (B,F,T,K) mean over views
    return reproj, l3d, uv2d, per_kp, (m2.sum(3) > 0)


def mvq_loss(out, batch, w: LossWeights, part_of_k):
    B, I = out["xyz"].shape[:2]; F = batch["kp3d_local"].shape[1]
    fv = batch["fly_valid"]
    # ---------------- matching on the final pass (no gradient)
    xyz_sg = jax.lax.stop_gradient(out["xyz"])
    uv_all = _reproject(xyz_sg, batch["M"], batch["t_local"])                   # (B,I,T,C,K,2)
    e2 = _huber(uv_all[:, :, None] - batch["kp2d"][:, None], w.huber_px).mean(-1)   # (B,I,F,T,C,K)
    m2 = batch["vis2d"][:, None]
    c2 = (e2 * m2).sum((3, 4, 5)) / jnp.maximum(m2.sum((3, 4, 5)), 1.0)
    e3 = batch["px_scale"][:, None, None] * jnp.abs(xyz_sg[:, :, None] - batch["kp3d_local"][:, None]).mean(-1)
    m3 = batch["has3d"][:, None]
    c3 = (e3 * m3).sum((3, 4)) / jnp.maximum(m3.sum((3, 4)), 1.0)
    cost = jnp.where(fv[:, None, :], c2 + c3, 1e6)
    assign, inst_matched = match(cost, fv, batch["prompt_on"])
    # ---------------- gather per fly
    g = lambda d: {k: _gather_inst(d[k], assign) for k in ("xyz", "uv", "conf_logit", "vis_logit")}
    pf = g(out)
    reproj, l3d, uv2d, per_kp, per_kp_m = _geo_terms(pf, batch, w, fv)
    # term 4 visibility BCE (per view query, target v flag; absent cams masked)
    vis_t = batch["vis2d"].astype(jnp.float32)
    bce_v = jnp.maximum(pf["vis_logit"], 0) - pf["vis_logit"] * vis_t + jnp.log1p(jnp.exp(-jnp.abs(pf["vis_logit"])))
    vis = _mmean(bce_v, batch["cam_valid"][:, None, :, :, None] & fv[:, :, None, None, None])
    # term 5 confidence (D4RT): c*err - lambda*log c
    c = jnp.clip(jax.nn.sigmoid(pf["conf_logit"]), 1e-4, 1 - 1e-4)
    conf = _mmean(c * per_kp - w.conf * jnp.log(c), per_kp_m & fv[:, :, None, None])
    # term 6 existence
    tgt = inst_matched.astype(jnp.float32)
    bce_e = jnp.maximum(out["exist_logit"], 0) - out["exist_logit"] * tgt + jnp.log1p(jnp.exp(-jnp.abs(out["exist_logit"])))
    exist = bce_e.mean()
    exist_acc = ((out["exist_logit"] > 0) == inst_matched).astype(jnp.float32).mean()
    # term 7 repulsion against OTHER flies' same-part labels
    pok = jnp.asarray(np.asarray(part_of_k))
    same_part = (pok[:, None] == pok[None, :]).astype(jnp.float32)              # (K,K')
    rep = jnp.zeros(())
    if F > 1:
        for f in range(F):
            for o in range(F):
                if o == f:
                    continue
                ok = fv[:, f] & fv[:, o]
                d2 = jnp.linalg.norm(pf["uv"][:, f][..., :, None, :] - batch["kp2d"][:, o][..., None, :, :], axis=-1)  # (B,T,C,K,K')
                h2 = jax.nn.relu(w.rep_px - d2) * same_part * batch["vis2d"][:, o][..., None, :]
                d3 = jnp.linalg.norm(pf["xyz"][:, f][..., :, None, :] - batch["kp3d_local"][:, o][..., None, :, :], axis=-1)  # (B,T,K,K')
                h3 = batch["px_scale"][:, None, None, None] * jax.nn.relu(w.rep_units - d3) * same_part * batch["has3d"][:, o][..., None, :]
                rep = rep + (h2.sum((1, 2, 3, 4)) * ok).sum() / jnp.maximum(ok.sum() * h2.shape[1] * h2.shape[2] * h2.shape[3], 1.0) \
                          + (h3.sum((1, 2, 3)) * ok).sum() / jnp.maximum(ok.sum() * h3.shape[1] * h3.shape[2], 1.0)
    total = (w.reproj * reproj + w.l3d * l3d + w.uv2d * uv2d + w.vis * vis + conf
             + w.exist * exist + w.rep * rep)
    # deep supervision
    if out.get("aux_pass1") is not None:
        r1, l1, u1, _, _ = _geo_terms(g(out["aux_pass1"]), batch, w, fv)
        total = total + w.pass1 * (w.reproj * r1 + w.l3d * l1 + w.uv2d * u1)
    for a in out.get("aux_layers", []):                 # intermediate 3D-only readouts: terms 1-2
        xyz_f = _gather_inst(a["xyz"], assign)
        uv_re = _reproject(xyz_f, batch["M"], batch["t_local"])
        r_ = _mmean(_huber(uv_re - batch["kp2d"], w.huber_px).mean(-1), batch["vis2d"] & fv[:, :, None, None, None])
        l_ = _mmean(batch["px_scale"][:, None, None, None] * jnp.abs(xyz_f - batch["kp3d_local"]).mean(-1),
                    batch["has3d"] & fv[:, :, None, None])
        total = total + w.aux * (w.reproj * r_ + w.l3d * l_)
    mp = _mmean(jnp.linalg.norm(pf["xyz"] - batch["kp3d_local"], axis=-1), batch["has3d"] & fv[:, :, None, None])
    px = _mmean(jnp.linalg.norm(_reproject(pf["xyz"], batch["M"], batch["t_local"]) - batch["kp2d"], axis=-1),
                batch["vis2d"] & fv[:, :, None, None, None])
    metrics = {"total": total, "reproj": reproj, "l3d": l3d, "uv2d": uv2d, "vis": vis, "conf": conf,
               "exist": exist, "rep": rep, "exist_acc": exist_acc, "match_reproj_px": px,
               "mpjpe3d_units": mp}
    return total, metrics
```

- [ ] **Step 5: Run tests**

Run: `cd third_party/jarvis_jax && python -m pytest tests/test_mvq_losses.py -v`
Expected: 8 PASS.

- [ ] **Step 6: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/train/matching.py third_party/jarvis_jax/jarvis_jax/train/losses_mvq.py third_party/jarvis_jax/tests/test_mvq_losses.py
git commit -m "feat(mvq): enumerated instance matching and pixel-unit loss (reproj primary, 3D L1, 2D, vis, D4RT confidence, existence, repulsion, deep supervision)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Trainer, configs, and CPU smoke test (`train/train_mvq.py`, `scripts/train_mvq.py`)

**Files:**
- Create: `jarvis_jax/train/train_mvq.py`, `jarvis_jax/scripts/train_mvq.py`, `configs/model/mvq.yaml`, `configs/train/mvq.yaml`
- Modify: `configs/paths/hyak.yaml` (add `mvq_runs_root`)
- Test: `tests/test_train_mvq_smoke.py`

**Interfaces:**
- Consumes: `MVQModel/MVQConfig` (Task 5), `V12WindowDataset/window_batches` (Task 3), `augment_window/MVAugParams` (Task 4), `mvq_loss/LossWeights` (Task 6), `load_dinov3_safetensors/dinov3_snapshot/HF_REPOS` (Task 1), `build_lr_swap`, `build_part_index`, `data_parallel_mesh/replicate`, `prefetch`, `make_manager/save_step/restore_latest`, `IMAGENET_MEAN/STD` from `data/transforms.py`.
- Produces:
  - `MVQTrainConfig` dataclass: `lr=3e-4, weight_decay=0.05, warmup_steps=500, total_steps=30000, batch_size=32, backbone_lr_mult=0.1, grad_clip=1.0, ema=0.999, seed=0, window_lengths=(1,), prompt_p_start=1.0, prompt_p_end=0.5, prompt_anneal_steps=2000, female_weight=1.0, balance_alpha=0.5, log_every=50, eval_every=2000, save_every=1000, num_workers=16, pretrained=True, smoke=False, val_cohorts=("female","group_C","two_fly")`.
  - `normalize_crops(u8: (...,3) uint8) -> float32` ImageNet-normalised.
  - `make_train_step(model_cfg, aug, lr_swap, part_of_k, weights) -> step(model, opt, ema_state, key, batch, prompt_p) -> (loss, metrics)`; `nnx.jit`-compiled, one instance per window length.
  - `evaluate(model, ds, batch_size, *, prompted: bool) -> dict` with `mpjpe3d_units`, `mpjpe3d_mm`, `reproj_px`, `uv2d_px`, `exist_prec`, `exist_rec`, `rigid_spread_mm`, per-cohort `mpjpe3d_units` for cohorts female / calib group / two-fly.
  - `run_training(root, out_dir, ckpt_dir, mcfg: MVQConfig, tcfg: MVQTrainConfig, aug: MVAugParams, weights: LossWeights) -> dict`.
  - Hydra: `python -m jarvis_jax.scripts.train_mvq run_id=<name> [overrides]` with `model=mvq train=mvq paths.runs_root=${paths.mvq_runs_root}` applied inside `main`.

Details:
- Optimizer: reuse the two-group pattern from `train/train.py::make_optimizer` (label leaves by `"backbone"` in the keypath), plus `optax.clip_by_global_norm(grad_clip)` chained BEFORE adamw; EMA kept as a separate `nnx.State` of Params updated `ema = d*ema + (1-d)*p` each step; eval and final save use the EMA weights.
- Balanced sampling: weights `w_i ∝ (1/n_c)^alpha` over `c = "<behavior>_<sex>"` read from the manifest (`ds.manifest[rec]["behavior"]`, `ds.is_female`), times `female_weight` for female windows; passed to `window_batches(weights=...)`.
- Prompt schedule: per step `prompt_p = start + (end-start)*min(step/anneal,1)`; each sample's `prompt_on ~ Bernoulli(prompt_p)`; when a sample has no mask (`prompt_mask` all False) force `prompt_on=False`.
- Eval cohorts: `female` (`ds.is_female`), `group_<X>` for every calib group present, `two_fly` (`ds.n_flies(i)>1`); raise `ValueError` at startup if `female` or `two_fly` is empty in val. `rigid_spread_mm` = std over the batch of `|EyeL-EyeR|` and the six femur lengths (`T?_Tro`→`T?_FeTi`) in mm (x0.1), reported per segment.
- Pretrained init: `pretrained=True` loads `dinov3_snapshot(HF_REPOS[mcfg.backbone])` into `model.backbone`; `False` (tests) keeps random init.
- Checkpointing/resume identical to `train_keypoints.run_training` (Orbax manager on `ckpt_dir`, `restore_latest`, `replicate` after restore, final `StandardCheckpointer.save(out_dir)` of the EMA state).

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_train_mvq_smoke.py
import os
import numpy as np
import pytest
from tests.mvq_fixtures import make_v12_root


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd third_party/jarvis_jax && python -m pytest tests/test_train_mvq_smoke.py -v`
Expected: FAIL, `No module named 'jarvis_jax.train.train_mvq'`

- [ ] **Step 3: Implement the trainer**

```python
# jarvis_jax/train/train_mvq.py
"""Training loop for the multi-view query lifter (mvq)."""
from __future__ import annotations

import dataclasses
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.data.augment import build_lr_swap
from jarvis_jax.data.distractor import build_part_index
from jarvis_jax.data.mv_augment import MVAugParams, augment_window
from jarvis_jax.data.prefetch import prefetch
from jarvis_jax.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from jarvis_jax.data.v12_windows import V12WindowDataset, window_batches, WINDOW_KEYS
from jarvis_jax.models.dinov3 import HF_REPOS, dinov3_snapshot, load_dinov3_safetensors
from jarvis_jax.models.mvq import MVQConfig, MVQModel
from jarvis_jax.sharding import data_parallel_mesh, replicate
from jarvis_jax.train.checkpoint import make_manager, restore_latest, save_step
from jarvis_jax.train.losses_mvq import LossWeights, mvq_loss

MM_PER_UNIT = 0.1


@dataclasses.dataclass
class MVQTrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    total_steps: int = 30000
    batch_size: int = 32
    backbone_lr_mult: float = 0.1
    grad_clip: float = 1.0
    ema: float = 0.999
    seed: int = 0
    window_lengths: tuple = (1,)
    prompt_p_start: float = 1.0
    prompt_p_end: float = 0.5
    prompt_anneal_steps: int = 2000
    female_weight: float = 1.0
    balance_alpha: float = 0.5
    log_every: int = 50
    eval_every: int = 2000
    save_every: int = 1000
    num_workers: int = 16
    pretrained: bool = True
    smoke: bool = False
    val_cohorts: tuple = ("female", "two_fly")


_MEAN = jnp.asarray(IMAGENET_MEAN); _STD = jnp.asarray(IMAGENET_STD)


def normalize_crops(u8):
    return (u8.astype(jnp.float32) / 255.0 - _MEAN) / _STD


def _labels(params):
    return jax.tree_util.tree_map_with_path(
        lambda path, _: "backbone" if "backbone" in jax.tree_util.keystr(path) else "head", params)


def make_optimizer(model, tcfg):
    decay = max(tcfg.total_steps, tcfg.warmup_steps + 1)
    sched = lambda peak: optax.warmup_cosine_decay_schedule(0.0, peak, tcfg.warmup_steps, decay, 0.0)
    def group(peak):
        return optax.chain(optax.clip_by_global_norm(tcfg.grad_clip),
                           optax.adamw(sched(peak), weight_decay=tcfg.weight_decay))
    tx = optax.multi_transform({"backbone": group(tcfg.lr * tcfg.backbone_lr_mult),
                                "head": group(tcfg.lr)}, _labels)
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def _batch_to_model(batch):
    return dict(crops=normalize_crops(batch["crops"]), cam_valid=batch["cam_valid"], M=batch["M"],
                t_local=batch["t_local"], prompt_mask=batch["prompt_mask"])


def make_train_step(aug: MVAugParams, lr_swap, part_of_k, weights: LossWeights, ema_decay: float):
    swap = jnp.asarray(lr_swap); pok = np.asarray(part_of_k)

    def loss_fn(model, batch):
        out = model(**_batch_to_model(batch), prompt_on=batch["prompt_on"])
        return mvq_loss(out, batch, weights, pok)

    @nnx.jit
    def step(model, optimizer, ema, key, batch, prompt_p):
        k_aug, k_p = jax.random.split(key)
        batch = augment_window(k_aug, batch, aug, swap)
        has_mask = batch["prompt_mask"].reshape(batch["prompt_mask"].shape[0], -1).any(-1)
        batch["prompt_on"] = jax.random.bernoulli(k_p, prompt_p, has_mask.shape) & has_mask
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
        optimizer.update(model, grads)
        params = nnx.state(model, nnx.Param)
        ema = jax.tree_util.tree_map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema, params)
        return loss, metrics, ema
    return step


@nnx.jit
def _fwd(model, crops, cam_valid, M, t_local, prompt_mask, prompt_on):
    return model(crops, cam_valid, M, t_local, prompt_mask, prompt_on=prompt_on)


def evaluate(model, ds, batch_size, *, prompted: bool, cohorts: dict, part_of_k, weights):
    """cohorts: name -> bool array over ds indices."""
    from jarvis_jax.models.mvq.geometry import project_local
    names = ds.keypoint_names
    ii = lambda n: names.index(n)
    segs = [("EyeL", "EyeR")] + [(f"T{i}{s}_Tro", f"T{i}{s}_FeTi") for i in (1, 2, 3) for s in "LR"]
    seg_idx = [(ii(a), ii(b)) for a, b in segs if a in names and b in names]
    per_sample = []       # (mpjpe_units, n_joints, reproj_px, n_views, exist_pred, exist_true, seg_lengths)
    idx_all = np.arange(len(ds))
    for s0 in range(0, len(ds), batch_size):
        idx = idx_all[s0:s0 + batch_size]
        b = {k: np.stack([ds[int(i)][k] for i in idx]) for k in WINDOW_KEYS}
        jb = {k: jnp.asarray(v) for k, v in b.items()}
        on = jnp.full((len(idx),), prompted) & jb["prompt_mask"].reshape(len(idx), -1).any(-1)
        out = _fwd(model, normalize_crops(jb["crops"]), jb["cam_valid"], jb["M"], jb["t_local"],
                   jb["prompt_mask"], on)
        jb["prompt_on"] = on
        _, m = mvq_loss(out, jb, weights, part_of_k)          # reuses the matching
        xyz = np.asarray(out["xyz"])                            # (B,I,T,K,3)
        # per-sample numbers from the matched instance (fly 0 = host): redo the cheap host match
        for bi, i_ds in enumerate(idx):
            gt = b["kp3d_local"][bi, 0]; has = b["has3d"][bi, 0]
            d = np.linalg.norm(xyz[bi][:, :, :, :] - gt[None], axis=-1)      # (I,T,K)
            inst = int(np.argmin(np.where(has[None], d, 0).sum((1, 2)) / max(has.sum(), 1)))
            e = d[inst][has]
            L = [np.linalg.norm(xyz[bi, inst, 0, a] - xyz[bi, inst, 0, c]) for a, c in seg_idx]
            exist = 1 / (1 + np.exp(-np.asarray(out["exist_logit"][bi]))) > 0.5
            per_sample.append((float(e.mean()) if e.size else np.nan, int(e.size), float(m["match_reproj_px"]),
                               int(exist.sum()), int(b["fly_valid"][bi].sum()), L, int(i_ds)))
    mp = np.array([p[0] for p in per_sample]); n = np.array([p[1] for p in per_sample])
    ok = np.isfinite(mp)
    res = {"mpjpe3d_units": float(np.average(mp[ok], weights=n[ok])),
           "reproj_px": float(np.mean([p[2] for p in per_sample])),
           "exist_prec": float(np.sum([min(p[3], p[4]) for p in per_sample]) / max(np.sum([p[3] for p in per_sample]), 1)),
           "exist_rec": float(np.sum([min(p[3], p[4]) for p in per_sample]) / max(np.sum([p[4] for p in per_sample]), 1))}
    res["mpjpe3d_mm"] = res["mpjpe3d_units"] * MM_PER_UNIT
    Ls = np.array([p[5] for p in per_sample])                   # (n, n_seg)
    for (a, c), col in zip(segs, Ls.T):
        res[f"rigid_spread_mm/{a}-{c}"] = float(np.std(col) * MM_PER_UNIT)
    for name, mask in cohorts.items():
        sel = np.array([mask[p[6]] for p in per_sample]) & ok
        res[f"cohort_{name}"] = float(np.average(mp[sel], weights=n[sel])) if sel.any() else float("nan")
    return res


def _cohorts(ds):
    n = len(ds)
    c = {"female": np.array([ds.is_female(i) for i in range(n)]),
         "two_fly": np.array([ds.n_flies(i) > 1 for i in range(n)])}
    for g in sorted({ds.calib_group(i) for i in range(n)}):
        c[f"group_{g}"] = np.array([ds.calib_group(i) == g for i in range(n)])
    return c


def _balanced_weights(ds, alpha, female_weight):
    cats = [f"{ds.manifest[ds.windows[i][0]].get('behavior', 'unknown')}_{'female' if ds.is_female(i) else 'other'}"
            for i in range(len(ds))]
    counts = {c: cats.count(c) for c in set(cats)}
    w = np.array([(1.0 / counts[c]) ** alpha for c in cats])
    w *= np.where([ds.is_female(i) for i in range(len(ds))], female_weight, 1.0)
    return w / w.sum()


def run_training(root, *, out_dir, ckpt_dir, mcfg: MVQConfig, tcfg: MVQTrainConfig,
                 aug: MVAugParams, weights: LossWeights):
    n_dev = len(jax.devices())
    if tcfg.batch_size % n_dev:
        raise ValueError(f"batch_size {tcfg.batch_size} not divisible by {n_dev} devices")
    model = MVQModel(mcfg, rngs=nnx.Rngs(tcfg.seed))
    if tcfg.pretrained:
        model.backbone = load_dinov3_safetensors(model.backbone, dinov3_snapshot(HF_REPOS[mcfg.backbone]))
        print(f"[mvq] loaded {HF_REPOS[mcfg.backbone]}")
    opt = make_optimizer(model, tcfg)
    ema = jax.tree_util.tree_map(lambda p: p, nnx.state(model, nnx.Param))
    mngr = make_manager(ckpt_dir) if ckpt_dir else None
    start = 0
    if mngr is not None:
        model, opt, start = restore_latest(mngr, model, opt)
        if start:
            ema = jax.tree_util.tree_map(lambda p: p, nnx.state(model, nnx.Param)); print(f"resume @ {start}")
    mesh = data_parallel_mesh()
    gm, sm = nnx.split(model); model = nnx.merge(gm, replicate(sm, mesh))
    go, so = nnx.split(opt); opt = nnx.merge(go, replicate(so, mesh))
    ema = replicate(ema, mesh)

    train_sets = {T: V12WindowDataset(root, "train", T=T, train=True, seed=tcfg.seed) for T in tcfg.window_lengths}
    val_ds = V12WindowDataset(root, "val", T=1, train=False)
    names = train_sets[tcfg.window_lengths[0]].keypoint_names
    lr_swap = build_lr_swap(names); part_of_k, _ = build_part_index(names)
    cohorts = _cohorts(val_ds)
    for c in tcfg.val_cohorts:
        if not cohorts.get(c, np.zeros(1, bool)).any():
            raise ValueError(f"val cohort '{c}' is empty on {root}")
    step_fns = {T: make_train_step(aug, lr_swap, part_of_k, weights, tcfg.ema) for T in tcfg.window_lengths}
    streams = {}
    for T, ds in train_sets.items():
        w = _balanced_weights(ds, tcfg.balance_alpha, tcfg.female_weight)
        def epochs(ds=ds, w=w, T=T):
            e = 0
            while True:
                yield from window_batches(ds, tcfg.batch_size, shuffle=True, seed=tcfg.seed + 1000 * e + T,
                                          weights=w, num_workers=tcfg.num_workers)
                e += 1
        streams[T] = prefetch((tuple(bt[k] for k in WINDOW_KEYS) for bt in epochs()), mesh, depth=2)
    key = jax.random.PRNGKey(tcfg.seed)
    Ts = list(tcfg.window_lengths)
    loss = float("nan"); t0 = time.time()
    for i in range(start, tcfg.total_steps):
        T = Ts[i % len(Ts)]
        batch = dict(zip(WINDOW_KEYS, next(streams[T])))
        pp = tcfg.prompt_p_start + (tcfg.prompt_p_end - tcfg.prompt_p_start) * min(i / max(tcfg.prompt_anneal_steps, 1), 1.0)
        loss, metrics, ema = step_fns[T](model, opt, ema, jax.random.fold_in(key, i), batch, jnp.float32(pp))
        loss = float(loss)
        if (i + 1) % tcfg.log_every == 0:
            ms = " ".join(f"{k}={float(v):.4f}" for k, v in metrics.items() if k != "total")
            print(f"step {i+1}/{tcfg.total_steps} T={T} loss {loss:.4f} {ms} ({time.time()-t0:.0f}s)", flush=True)
        if (i + 1) % tcfg.eval_every == 0 or i + 1 == tcfg.total_steps:
            em = _with_ema(model, ema)
            for mode in ("prompted", "unprompted"):
                r = evaluate(em, val_ds, min(tcfg.batch_size, 8), prompted=(mode == "prompted"),
                             cohorts=cohorts, part_of_k=part_of_k, weights=weights)
                print(f"  val[{mode}] " + " ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)
        if mngr is not None and (i + 1) % tcfg.save_every == 0:
            save_step(mngr, i + 1, model, opt)
    if mngr is not None:
        save_step(mngr, tcfg.total_steps, model, opt); mngr.wait_until_finished()
    em = _with_ema(model, ema)
    val = {mode: evaluate(em, val_ds, min(tcfg.batch_size, 8), prompted=(mode == "prompted"),
                          cohorts=cohorts, part_of_k=part_of_k, weights=weights)
           for mode in ("prompted", "unprompted")}
    os.makedirs(out_dir, exist_ok=True)
    ckptr = ocp.StandardCheckpointer(); ckptr.save(out_dir, nnx.split(em)[1], force=True); ckptr.wait_until_finished()
    json.dump({"model": dataclasses.asdict(mcfg), "train": dataclasses.asdict(tcfg), "val": val,
               "keypoint_names": names}, open(os.path.join(out_dir, "mvq_run.json"), "w"), indent=1)
    return {"final_loss": loss, "val": val, "steps": tcfg.total_steps}


def _with_ema(model, ema):
    gdef, state = nnx.split(model)
    em = nnx.merge(gdef, state)
    nnx.update(em, ema)
    em.eval()
    return em
```

`tcfg.smoke` is accepted for parity with the 2D trainer but the test sets the tiny values explicitly; no special-casing is needed inside `run_training`.

- [ ] **Step 4: Configs and Hydra entry**

`configs/model/mvq.yaml`:
```yaml
# @package model
arch: mvq
crop: 448
patch: 16
embed_dim: 768
num_keypoints: 50
num_cameras: 7
max_frames: 8
n_instances: 3
n_local: 2
n_global: 2
global_pool: 2
dec_layers_3d: 8
dec_layers_2d: 4
dec_heads: 12
mlp_ratio: 4.0
refine_passes: 1
patch_rgb: 9
fourier_bands: 8
roi_scale: 24.0
camera_slot_embed: true
backbone: dinov3_b16
backbone_depth: 12
backbone_heads: 12
remat: true
```

`configs/train/mvq.yaml`:
```yaml
# @package train
lr: 3.0e-4
weight_decay: 0.05
warmup_steps: 500
total_steps: 30000
batch_size: 32          # 8 per GPU x 4 GPUs
backbone_lr_mult: 0.1
grad_clip: 1.0
ema: 0.999
seed: 0
window_lengths: [1]     # P2 = T=1 only; P3 adds 2
prompt_p_start: 1.0
prompt_p_end: 0.5
prompt_anneal_steps: 2000
female_weight: 1.0
balance_alpha: 0.5
log_every: 50
eval_every: 2000
save_every: 1000
num_workers: 16
pretrained: true
smoke: false
val_cohorts: [female, two_fly]
# loss weights (spec 2026-09-03 §5) -- starting points, swept in P3
loss:
  reproj: 1.0
  l3d: 0.5
  uv2d: 0.5
  vis: 0.1
  conf: 0.2
  exist: 1.0
  rep: 0.5
  pass1: 0.5
  aux: 0.3
  huber_px: 8.0
  rep_px: 20.0
  rep_units: 2.5
mv_aug:
  enabled: true
  rot_deg: 30.0
  scale_min: 0.8
  scale_max: 1.25
  translate_frac: 0.1
  world_yaw: true
  world_tilt_deg: 30.0
  mirror_p: 0.5
  cam_drop_p: 0.3
  cam_drop_max: 2
  brightness: 0.2
  contrast: 0.2
  gamma: 0.2
  blur_max: 0.5
  noise_scale: 0.02
  pc_color: 0.2
```

Append to `configs/paths/hyak.yaml`:
```yaml
mvq_runs_root:       /gscratch/portia/${paths.user}/data/Johnson_lab/jax_mvq_runs
```

`jarvis_jax/scripts/train_mvq.py`:
```python
"""Hydra entrypoint: python -m jarvis_jax.scripts.train_mvq run_id=<name> [overrides]"""
import os
import hydra
from omegaconf import OmegaConf

from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, build_dataclass, run_dir_for

register_resolvers()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.train_mvq import MVQTrainConfig, run_training
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    if cfg.model.get("arch") != "mvq":
        raise ValueError("run with model=mvq train=mvq")
    mcfg = build_dataclass(MVQConfig, cfg.model)
    tnode = OmegaConf.to_container(cfg.train, resolve=True)
    tnode["window_lengths"] = tuple(tnode["window_lengths"]); tnode["val_cohorts"] = tuple(tnode["val_cohorts"])
    tcfg = MVQTrainConfig(**{k: v for k, v in tnode.items() if k in MVQTrainConfig.__dataclass_fields__})
    weights = LossWeights(**tnode["loss"]); aug = MVAugParams(**tnode["mv_aug"])
    run_dir = run_dir_for(cfg)
    return run_training(cfg.paths.data_root, out_dir=os.path.join(run_dir, "final"),
                        ckpt_dir=os.path.join(run_dir, "ckpt"), mcfg=mcfg, tcfg=tcfg, aug=aug, weights=weights)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the smoke test and a config compose check**

Run: `cd third_party/jarvis_jax && python -m pytest tests/test_train_mvq_smoke.py -v` — Expected: 2 PASS (CPU, a few minutes: 7 crops x 448 through a 1-layer width-32 ViT).
Run: `cd third_party/jarvis_jax && python -c "
from hydra import initialize_config_dir, compose
from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers; register_resolvers()
with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
    c = compose('config', overrides=['model=mvq','train=mvq','paths=hyak','run_id=x','paths.runs_root=\${paths.mvq_runs_root}'])
print(c.model.backbone, c.train.batch_size, c.paths.runs_root)"` — Expected: `dinov3_b16 32 /gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs`.

- [ ] **Step 6: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/train/train_mvq.py third_party/jarvis_jax/jarvis_jax/scripts/train_mvq.py third_party/jarvis_jax/configs/model/mvq.yaml third_party/jarvis_jax/configs/train/mvq.yaml third_party/jarvis_jax/configs/paths/hyak.yaml third_party/jarvis_jax/tests/test_train_mvq_smoke.py
git commit -m "feat(mvq): trainer with 2-group AdamW + EMA, prompt annealing, balanced windows, cohort eval; Hydra configs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Figure gate 1, the first real T=1 run, and the val comparison against DLT

**Files:**
- Create: `scripts/viz/mvq_overlay.py` (repo root), `docs/benchmark/2026-09-mvq/p2-t1-notes.md`
- Test: none beyond a `--help`/dry import; the figure is the test (CLAUDE.md: state the expectation before rendering, read the PNG back).

**Interfaces:**
- Consumes: a `final/` dir written by Task 7 (`mvq_run.json` + Orbax state), `V12WindowDataset`, `assemble`, `project_local`.
- Produces: `python scripts/viz/mvq_overlay.py --run <final_dir> --split val --n 6 --cases female,two_fly,worst --out figures/2026-09-mvq/<run_id>/` writing one PNG per case: rows = samples, columns = cameras (all 7), white = human 2D, cyan = model 2D head, green = reprojected model 3D, with a per-panel reprojection px number; plus `summary.json` (per-sample mpjpe units/mm, reproj px, exist probs, calib group, sex).

- [ ] **Step 1: Write the overlay script**

```python
#!/usr/bin/env python
"""Figure gate 1 for the mvq lifter.

EXPECTATION (write before looking): after 1k+ steps the GREEN reprojected 3D
and the CYAN 2D head both sit on the fly in every valid camera, within a
few px of the WHITE human labels on easy male frames. If green is offset
by the same vector in all cameras, center3D/t_local bookkeeping is wrong
(crop origin or local offset); if green is right in some cameras and
rotated/mirrored in others, the per-camera geometry tokens or the camera
order by name is wrong; if cyan is fine and green is not, the 3D path is
broken independently of the encoder. On female wall/contact frames the
expectation is looser: points stay on HER body, never on the male.
"""
import argparse, json, os, sys
import numpy as np
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from flax import nnx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "jarvis_jax")); sys.path.insert(0, ROOT)
from jarvis_jax.models.mvq import MVQConfig, MVQModel, assemble
from jarvis_jax.models.mvq.geometry import project_local
from jarvis_jax.data.v12_windows import V12WindowDataset, WINDOW_KEYS
from jarvis_jax.train.train_mvq import normalize_crops


def load_model(final_dir):
    meta = json.load(open(os.path.join(final_dir, "mvq_run.json")))
    cfg = MVQConfig(**meta["model"])
    model = MVQModel(cfg, rngs=nnx.Rngs(0))
    gdef, state = nnx.split(model)
    target = jax.tree_util.tree_map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), state)
    restored = ocp.StandardCheckpointer().restore(final_dir, target=target)
    model = nnx.merge(gdef, restored); model.eval()
    return model, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="val"); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cases", default="female,two_fly,worst"); ap.add_argument("--out", required=True)
    ap.add_argument("--prompted", action="store_true")
    a = ap.parse_args()
    root = a.root or "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902"
    model, meta = load_model(a.run)
    ds = V12WindowDataset(root, a.split, T=1, train=False)
    names = ds.keypoint_names
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for i in range(len(ds)):
        s = ds[i]; b = {k: jnp.asarray(v)[None] for k, v in s.items()}
        on = jnp.array([a.prompted and bool(s["prompt_mask"].any())])
        out = model(normalize_crops(b["crops"]), b["cam_valid"], b["M"], b["t_local"], b["prompt_mask"], prompt_on=on)
        xyz = np.asarray(out["xyz"][0]); gt = s["kp3d_local"][0, 0]; has = s["has3d"][0, 0]
        d = np.linalg.norm(xyz[:, 0] - gt[None], axis=-1); inst = int(np.argmin((d * has).sum(1)))
        uv3 = np.asarray(project_local(jnp.asarray(xyz[inst, 0]), b["M"][0], b["t_local"][0, 0]))   # (K,C,2)
        uv2 = np.asarray(out["uv"][0, inst, 0])                                                     # (C,K,2)
        vis = s["vis2d"][0, 0]                                                                     # (C,K)
        re = np.linalg.norm(uv3.transpose(1, 0, 2) - s["kp2d"][0, 0], axis=-1)
        rows.append(dict(i=i, mpjpe_units=float(d[inst][has].mean()) if has.any() else np.nan,
                         reproj_px=float(re[vis].mean()) if vis.any() else np.nan,
                         exist=[float(x) for x in 1 / (1 + np.exp(-np.asarray(out["exist_logit"][0])))],
                         female=bool(ds.is_female(i)), two_fly=ds.n_flies(i) > 1, group=ds.calib_group(i),
                         sample=s, uv3=uv3, uv2=uv2, inst=inst))
    json.dump([{k: v for k, v in r.items() if k not in ("sample", "uv3", "uv2")} for r in rows],
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for case in a.cases.split(","):
        if case == "female": sel = [r for r in rows if r["female"]]
        elif case == "two_fly": sel = [r for r in rows if r["two_fly"]]
        else: sel = sorted(rows, key=lambda r: -np.nan_to_num(r["reproj_px"], nan=1e9))
        sel = sel[: a.n]
        if not sel:
            print(f"[{case}] no samples"); continue
        C = 7; fig, axes = plt.subplots(len(sel), C, figsize=(2.2 * C, 2.2 * len(sel)), squeeze=False)
        for r_i, r in enumerate(sel):
            s = r["sample"]
            for c in range(C):
                ax = axes[r_i, c]; ax.imshow(s["crops"][0, c]); ax.set_xticks([]); ax.set_yticks([])
                if not s["cam_valid"][0, c]:
                    ax.set_title(f"cam{c+1} absent", fontsize=7); continue
                vis = s["vis2d"][0, 0, c]; g2 = s["kp2d"][0, 0, c]
                ax.scatter(g2[vis, 0], g2[vis, 1], s=6, c="white", label="human 2D")
                ax.scatter(r["uv2"][c, :, 0], r["uv2"][c, :, 1], s=6, c="cyan", label="model 2D head")
                ax.scatter(r["uv3"][:, c, 0], r["uv3"][:, c, 1], s=6, c="lime", label="model 3D reprojected")
                e = np.linalg.norm(r["uv3"][:, c] - g2, axis=-1)[vis]
                ax.set_title(f"cam{c+1} {e.mean():.1f}px" if e.size else f"cam{c+1}", fontsize=7)
            axes[r_i, 0].set_ylabel(f"#{r['i']} {'F' if r['female'] else 'M'} grp{r['group']}\n"
                                    f"{r['mpjpe_units']*0.1:.2f}mm", fontsize=7)
        axes[0, 0].legend(fontsize=6, loc="lower left")
        fig.suptitle(f"mvq {os.path.basename(os.path.dirname(a.run.rstrip('/')))} — {case} — white=human, cyan=2D head, green=reprojected 3D")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"gate1_{case}.png"), dpi=130); plt.close(fig)
        print("wrote", os.path.join(a.out, f"gate1_{case}.png"))


if __name__ == "__main__":
    main()
```

Colours follow `viz/core/colors.py`'s shared language (white = observed/human, cyan = detector/2D head, green = fit/3D) and are written literally here so the script has no dependency on the repo-root `viz` package's import path from a queued job; the keypoint-group skeleton colouring from `viz.core.colors.keypoint_groups` is a P4 addition when this script is promoted alongside `python -m viz overlay --compare`.

- [ ] **Step 2: Launch the P2 T=1 run (queue, 4 GPUs)**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
scripts/slurm/submit_task.sh --gpus 4 --cpus 32 --mem 200 --time 36:00:00 mvq_t1_b16 \
  'module load cuda; unset JAX_PLATFORMS; export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3; cd third_party/jarvis_jax && python -m jarvis_jax.scripts.train_mvq model=mvq train=mvq paths=hyak run_id=mvq_t1_b16_20260903 "paths.runs_root=\${paths.mvq_runs_root}"'
```
Before submitting, check `scripts/slurm/submit_task.sh` sets `--gres=gpu:$GPUS` and `--cpus-per-task=$CPUS` (read lines 40-60); if it hardcodes 1 GPU, copy it to `scripts/slurm/submit_task_multi.sh` with the flags wired through and use that. Watch the first 200 steps in `slurm_logs/mvq_t1_b16-*.out`: loss must fall from its step-1 value and `exist_acc` must reach 1.0 quickly on single-fly batches; `match_reproj_px` should be under ~40 px by step 1k. If step time exceeds ~1.5 s/step at batch 32, the loader is the bottleneck (JPEG decode of 224 images/step): raise `train.num_workers=32` first, and if still bound, that is the trigger for the strip cache deferred from spec §6.2 (record the measured s/step in the notes either way).

- [ ] **Step 3: Gate 1 at the 2k-step checkpoint (queue, 1 GPU)**

The trainer writes `final/` only at the end; for the gate, save an interim EMA checkpoint by running a short resume: launch a second job with `train.total_steps=2000` and a different `run_id` (`mvq_t1_b16_gate1`) sharing nothing, OR wait for the first `eval_every` print and use the main run's numbers plus the overlay on `ckpt/<step>` — simplest: run the gate job as its own 2000-step run.

```bash
scripts/slurm/submit_task.sh --gpus 4 --cpus 32 --mem 200 --time 4:00:00 mvq_gate1 \
  'module load cuda; unset JAX_PLATFORMS; export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3; cd third_party/jarvis_jax && python -m jarvis_jax.scripts.train_mvq model=mvq train=mvq paths=hyak run_id=mvq_t1_b16_gate1 train.total_steps=2000 train.eval_every=1000 "paths.runs_root=\${paths.mvq_runs_root}"'
# then
scripts/slurm/submit_task.sh --gpus 1 --time 0:30:00 mvq_gate1_fig \
  'module load cuda; unset JAX_PLATFORMS; cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset && python scripts/viz/mvq_overlay.py --run /gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_gate1/final --split val --n 6 --out figures/2026-09-mvq/mvq_t1_b16_gate1'
```
Then **open every PNG with the Read tool** and write what is seen against the docstring's expectation into `docs/benchmark/2026-09-mvq/p2-t1-notes.md` (female, two-fly, worst cases; all 7 cameras; name cameras and keypoints, never indices). A run that fails gate 1 stops here: fix geometry before spending the 30k-step budget.

- [ ] **Step 4: Val comparison against DLT and the A4 arm (after the 30k run)**

Numbers to put in `docs/benchmark/2026-09-mvq/p2-t1-notes.md`, all on v12 val T=1 (153 framesets), prompted and unprompted:

| metric | source |
|---|---|
| mvq 3D MPJPE (units, mm), overall / female / group_A / group_C / two_fly | `final/mvq_run.json["val"]` |
| mvq reprojection px, 2D-head px | same |
| DLT-of-GT baseline: 0 by construction — instead report the current pipeline's robust DLT of the **ViTPose v5vf_maskoff 2D** on the same framesets: run `triangulate_keypoints` (`jarvis_jax/tracking/triangulate.py:136`) on `predict_2d`-style crops of each val frameset and compute MPJPE vs the DLT-of-GT 3D | new script `scripts/benchmark/mvq_val_baselines.py` (30–60 lines: loop val framesets, ViTPose forward on the 7 crops from `V12WindowDataset` (`normalize_image` needs a 4th zero channel), `heatmaps_to_keypoints`, `triangulate_keypoints`, MPJPE with `mpjpe_3d`) |
| A4 c2f arm 0.56 units | `docs/benchmark/2026-08-30-arm-results/notes.md` — label clearly as a DIFFERENT (v5) val split, not like-for-like |
| rigid spread mm per segment | `mvq_run.json` |

Expectation to write down first: mvq prompted beats the ViTPose+DLT baseline on the two_fly and female cohorts; on the easy male cohort it is allowed to be within noise; unprompted is worse than prompted on two_fly by a margin that shrinks when copy-paste is enabled in P3. If mvq is worse than DLT everywhere, the first suspects are the 2D precision gap (check `uv2d_px` vs the detector's 5.3 px val MPJPE) and the refinement pass (compare `aux_pass1` vs final reprojection px in an eval print).

- [ ] **Step 5: Commit the script and the notes**

```bash
git add scripts/viz/mvq_overlay.py scripts/benchmark/mvq_val_baselines.py docs/benchmark/2026-09-mvq/p2-t1-notes.md
git commit -m "feat(viz): mvq figure gate 1 overlay + val baseline script; P2 T=1 notes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## What P3 and P4 need from this plan (not done here)

- P3 ablations run through `train.window_lengths=[1,2]`, `model.n_global=0`, `model.refine_passes=0`, `train.prompt_p_end`, copy-paste (new loader wrapper), and a loss-weight sweep; each is a Hydra override on Task 7's entrypoint, one arm per 4-GPU job.
- P4 (`tracking/lift_mvq.py`) consumes `MVQModel`, `assemble`, and the `mvq_run.json` metadata; the pipeline's `kp3d.npz` gate signature must include the checkpoint hash, T, and prompt mode.
