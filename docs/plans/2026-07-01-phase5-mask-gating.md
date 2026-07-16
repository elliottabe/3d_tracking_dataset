# Phase 5: Unified JAX Detector — Dilated-Mask Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the one robustness mechanism the already-trained JAX keypoint+dense detector is missing — **dilated-mask gating** — plus a **pluggable-backbone abstraction** and **dense-channel L/R flip augmentation**. The detector (`ViTPose` = shared ViT-B/16 backbone + `ClassicDecoder` heatmap head; 50 kp + 300 dense verts, channel-stacked; the "cse_vit350" checkpoints) EXISTS and is trained. Phase 5 adds: (1) a training-time **containment-with-dilation** loss (`mask_weight>0`) that penalizes predicted heatmap mass landing *outside* the dilated SAM mask ("predictions must land on the fly"); (2) an inference-time **dilate + zero-outside** gate on the heatmaps; (3) a `Backbone` protocol/registry so `ViTPose` can accept a swappable trunk (default the existing `ViT`; sets up a future SAM3 ViTDet port WITHOUT doing it); (4) a dense-channel (50+M) L/R swap map derived from the canonical mesh so **flip augmentation** — currently disabled for the CSE model — can be turned ON. Validate by **fine-tuning the existing `cse_vit350` checkpoint with gating ON** and **ablating gating on/off** (headline metric: reduction in off-fly heatmap mass), reporting honestly whether gating helps.

**Architecture:** A new `models/dilation.py::dilate_mask_jax(mask, k)` does binary/float mask dilation via a single `jax.lax.reduce_window` max-pool (mirroring the PyTorch `F.max_pool2d(..., kernel_size=dilate, stride=1, padding=dilate//2)` reference), NHWC, jit-safe (no scipy/loops in the traced step). `train/losses.py::mask_containment` grows a backward-compatible `dilate=0` kwarg that dilates the mask before the outside-mass penalty; `train/train.py` (`TrainConfig`, `make_train_step`) gains a `mask_dilate` field and dilates `mask224` before calling `mask_containment` when `mask_weight>0`. A new `models/backbone.py` defines a `Backbone` runtime-checkable protocol + a name→factory registry with `ViT` registered as `"vit"`; `ViTPose.__init__` accepts an optional `backbone=` (a built module OR a registry name), defaulting to the existing hardcoded `ViT(cfg)` so every current construction path and checkpoint layout is byte-for-byte unchanged. A new `cse/dense_lr_swap.py::build_dense_lr_swap(mesh_npz, fps_key, base_names)` builds the (50+M) involution by concatenating the 50-keypoint `build_lr_swap` with a **segment-aware mutual-nearest-neighbour Y-reflection** pairing of the M FPS vertices in the canonical `vertices` frame (verified: bilateral axis = Y; the raw `sym_index` is NOT closed under FPS subsampling, so it cannot be used directly). An inference-gating helper `cse/gating.py::gate_heatmaps(hm, mask, dilate)` dilates the SAM mask, resizes it to the heatmap grid, and zeroes heatmap mass outside (mirroring `jarvis3D_multi.py:331-333`). A gated fine-tune entry extends `cse/train_keypoints_cse_full.py` (turn on `mask_weight`/`mask_dilate`, enable flip with the dense swap, warm-start `cse_vit350`), and an eval/ablation script compares gated-vs-ungated checkpoints on val.

**Tech Stack:** Python, JAX (`jax.lax.reduce_window`, `jax.image.resize`, `jax.nn.relu`), flax NNX (`ViT`/`ViTPose`/`ClassicDecoder` modules, `nnx.jit`, `nnx.split`/`merge`), optax (`multi_transform` two-LR AdamW), `jax.sharding` (`NamedSharding` data-parallel mesh, single-host multi-GPU — NOT pmap), Orbax (`StandardCheckpointer`, `make_manager`/`restore_latest`), NumPy + SciPy `cKDTree` (mesh pairing, offline only — not in the jit step), h5py, pytest. PyTorch is referenced for the dilation/gating ALGORITHM only (mirrored, never imported).

## Global Constraints

- Reuse the existing NNX ViT-B / ViTPose / ClassicDecoder + training stack (`train/train.py`, `sharding.py`, Orbax `train/checkpoint.py`, Hydra configs) **UNMODIFIED except the additive gating/backbone/aug changes**. flax NNX + optax + `jax.sharding` (not pmap).
- Dilated-mask gating = the deliverable: a train-time containment-with-dilation loss (`mask_weight>0`) AND an inference-time dilate + zero-outside gate. Dilation via `jax.lax.reduce_window` max-pool (mirror the PyTorch `F.max_pool2d` reference). The mask is pixel-aligned to the crop by construction (`v3.py` crops the mask with the same `crop_origin` window); the augment path re-binarizes the warped mask — so dilate **POST-augmentation** (on the model's 4th input channel), never on a pre-augment mask.
- Backbone pluggable (protocol/registry), default `ViT`; the **SAM3 ViTDet port is OUT OF SCOPE** (explicit non-goal — spec §7/§10: ~447M-param ViT-L with windowed attention + 2D RoPE + fused rel-pos bias, no JAX groundwork, multi-week separate effort). Phase 5 only builds the abstraction that a future port would register into.
- **No mask-prediction head** (the mask is input + gate only; SAM3 provides masks). Data = existing single-fly `red_data_unified_V3`, **M=300** (matches the `cse_vit350` checkpoints; the production dense set). Fine-tune **from `cse_vit350`, not from scratch**. Defer amputee/headless/courtship dense-label generation (separate data-pipeline effort, out of scope).
- CPU unit tests: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/<f>`. GPU training/eval on-node (8-GPU `sbatch` `submit_train_keypoints.sh`-style OR background bash; `unset LD_LIBRARY_PATH`; `export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`); the **coordinator runs the GPU work** (background/sbatch), NOT a subagent (subagents orphan on long GPU jobs).
- New code under `third_party/jarvis_jax/jarvis_jax/` (`models/`, `train/`, `cse/`); tests under `third_party/jarvis_jax/tests/`.
- TDD for T1–T5 (code): failing test with REAL asserts → run (fail) → REAL minimal impl → run (pass) → commit. T6–T7 are experiment/eval with documented reproducible commands + honest success criteria (report neutral/negative results, do not paper over).
- **Branch:** `elliottabe/paper_update_062026` (rolling multi-phase branch; work directly on it). Commit per task by EXPLICIT path (never `git add -A`); do **NOT** commit checkpoints, label `.npz`, or any `jax_vitpose_runs/` / `cse_work/` data outputs.

---

## File structure

NEW under `third_party/jarvis_jax/jarvis_jax/`:

- `models/dilation.py` **(new, Task 1)** — `dilate_mask_jax(mask, k)` (a single `jax.lax.reduce_window` max-pool; NHWC or NHW; jit-safe). The one dilation primitive consumed by both the train-time loss (T2) and the inference gate (T3).
- `models/backbone.py` **(new, Task 4)** — a `Backbone` `@runtime_checkable` protocol + a `BACKBONES` name→factory registry + `register_backbone`/`build_backbone`; `ViT` registered as `"vit"`. Small, additive.
- `cse/gating.py` **(new, Task 3)** — `gate_heatmaps(hm, mask, dilate)` inference-time gate (dilate SAM mask → resize to heatmap grid → zero heatmap mass outside). Mirrors `jarvis3D_multi.py:331-333`.
- `cse/dense_lr_swap.py` **(new, Task 5)** — `build_dense_lr_swap(mesh_npz, fps_key, base_names)` → the (50+M) L/R permutation (50 named kp via `augment.build_lr_swap` ⊕ M FPS verts via segment-aware mutual-NN Y-reflection). Enables CSE flip aug.

MODIFY:

- `train/losses.py` **(Task 2)** — `mask_containment(pred, mask224, *, dilate=0, eps=1e-6)`: dilate `mask224` via `dilate_mask_jax` before the outside-mass penalty. Backward compatible (`dilate=0` → current behaviour, byte-for-byte).
- `train/train.py` **(Task 2)** — add `TrainConfig.mask_dilate: int = 0`; `make_train_step(mask_weight, ..., mask_dilate=0)` dilates `mask224` (via the loss `dilate=` kwarg) when `mask_weight>0`.
- `models/vitpose.py` **(Task 4)** — `ViTPose.__init__(cfg, *, backbone=None, rngs)` accepts an optional prebuilt module OR a registry name; default `None` → `ViT(cfg, rngs=rngs)` (backward compat; unchanged checkpoint param paths, `self.backbone` name preserved).
- `cse/train_keypoints_cse_full.py` **(Task 6)** — add `--mask-weight` / `--mask-dilate` (wire into `TrainConfig` + `make_train_step`) and `--flip-p` with the dense swap from `build_dense_lr_swap` (replacing the hardcoded `flip_p=0.0` + identity `lr_swap`). Full-schema/flip-off path unchanged when the new args keep their defaults.
- `cse/predict_full.py` **(Task 3, optional wire-in)** — an opt-in `--gate-dilate` that applies `gate_heatmaps` to `hm` before `heatmaps_to_keypoints` (default off → current behaviour). (The gate helper itself + its unit test is the deliverable; wiring is a small, tested addition.)

NEW eval/experiment scripts:

- `cse/eval_gating_ablation.py` **(new, Task 7)** — load a checkpoint, run val, report kp accuracy (MPJPE/PCK), dense-vertex accuracy, and the headline **off-fly heatmap-mass fraction** (mean `mask_containment(pred, mask224, dilate=k)`) for GATED vs UNGATED, + a qualitative overlay viz.
- `cse/sbatch_cse_gated_finetune.sh` **(new, Task 6)** — the reproducible 8-GPU gated fine-tune (warm-start `cse_vit350`).

TESTS under `third_party/jarvis_jax/tests/`:

- `test_dilation.py` **(new, Task 1)**, `test_mask_containment_dilated.py` **(new, Task 2)**, `test_inference_gating.py` **(new, Task 3)**, `test_backbone_pluggable.py` **(new, Task 4)**, `test_lr_swap_dense.py` **(new, Task 5)**, `test_gating_ablation.py` **(new, Task 7 — real-data-gated existence/eval)**.

**Shared verified facts** (transcribed from the real source + mesh — do NOT re-derive):

```python
# Model / config (jarvis_jax/config.py, models/vit.py, models/vitpose.py, models/decoder.py)
ViTPoseConfig: img_size=448, patch=16, in_ch=4 (RGB+mask), embed_dim=768, depth=12,
               num_heads=12, mlp_ratio=4, num_keypoints=50 (CSE: 350), heatmap_size=224.
               num_tokens = (448//16)**2 = 784 (28x28).
ViTPose(cfg, *, rngs).__call__(x (B,448,448,4), *, use_running_average=False) -> (B,224,224,K)
  self.backbone = ViT(cfg, rngs=rngs); self.decoder = ClassicDecoder(cfg.embed_dim, cfg.num_keypoints, rngs=rngs)
ViT(cfg, *, rngs).__call__(x (B,H,W,4)) -> (B,784,768)   # patch tokens, cls dropped

# Losses (jarvis_jax/train/losses.py)
mask_containment(pred (B,H,W,K), mask224 (B,H,W) float{0,1}, eps=1e-6) -> scalar in [0,1]
  p = relu(pred); outside = p * (1 - mask224[...,None]); return outside.sum()/(p.sum()+eps)

# Train (jarvis_jax/train/train.py)
TrainConfig(lr, weight_decay, warmup_steps, total_steps, batch_size, mask_weight=0.0, backbone_lr_mult=0.1, seed)
make_train_step(mask_weight, aug_params=None, lr_swap=None, heatmap_size=224, sigma=7.0, joint_weight=None) -> step
  # loss_fn: mask = img[...,3]; mask224 = jax.image.resize(mask, (B, pred.H, pred.W), method="nearest")
  #          loss += mw * mask_containment(pred, mask224)     # currently UNDILATED

# Augment (jarvis_jax/data/augment.py)
build_lr_swap(names) -> int32 (K,)   # involution; midline -> self; asserts arr[arr]==arange
augment_batch(key, img4_u8, kp_xy, vis, params, lr_swap, heatmap_size=224)  # flip uses lr_swap
AugParams(enabled, rot_deg=30, scale_min=0.8, scale_max=1.25, translate_frac=0.1, flip_p=0.5, ...)

# CSE (jarvis_jax/cse/cse_dataset.py, cse/warm_start.py, cse/train_keypoints_cse_full.py)
CSEImageDataset(root, split, aux_npz, *, recordings=None) -> kp_xy (50+M,3), vis (50+M,)
warm_start_from_v3(new_model, v3_ckpt_path, cfg50) -> model  # copies matching params; splices first-K head channels
train_keypoints_cse_full: flip currently DISABLED (AugParams(enabled=True, flip_p=0.0); lr_swap = identity)

# Decode (jarvis_jax/eval/mpjpe.py)
heatmaps_to_keypoints(hm (B,H,W,K), *, in_size=448, radius=7) -> (B,K,2)   # relu -> argmax -> windowed centroid

# PyTorch REFERENCE (mirror the algorithm; do NOT import torch)
# efficienttrack/loss.py:25-59: m=(mask>0.5); if dilate>1: F.max_pool2d(m, kernel_size=dilate, stride=1, padding=dilate//2)
# sam3_masker.py:25-38 dilate_mask(kernel 15/21); jarvis3D_multi.py:331-333: zero crop pixels outside dilated mask
CSE_DILATE_DEFAULT = 21          # inference kernel used by jarvis3D_multi (kp slack for wings/legs)
TRAIN_DILATE_DEFAULT = 11        # HybridNet config default DILATE for the containment loss

# Mesh (fruitfly_cse/fly_v1_collision_canonical_wings.npz) — VERIFIED this cycle
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"
#   keys: vertices (61666,3), vertex_segment (61666,), sym_index (61666,), seg_ids (65,), seg_names (65,) <U16,
#         fps_300 (300,), fps_200, fps_100, ...
#   BILATERAL MIRROR AXIS in the `vertices` frame = Y (abs-mean signed diff [x,y,z] ~ [1.6e-4, 0.247, 1.2e-4]).
#   sym_index is a GLOBAL per-vertex bilateral partner but is NOT closed under FPS subsampling:
#     only 115/300 fps_300 partners are themselves in fps_300, and si[fps] restricted is NOT an involution.
#     => build the (50+M) dense swap by segment-aware mutual-nearest-neighbour Y-reflection ON THE FPS SUBSET
#        (verified: involution=True, 246/300 swapped, wing_left(seg9) -> wing_right(seg10) 100%, 0 cross-part mispairs).
#   seg names of interest: wing_left=9, wing_right=10, antenna_left=7/antenna_right=8, thorax=1, abdomen=11...
#   Trained CSE-350 warm-start source: /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit350_wings/final
#   (and cse_vit350b_wings/{ckpt,final}). Dense labels: cse_work/cse_labels_{train,val}_M300.npz (train 16618 / val 1365).

ROOT  = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
WORK  = "/gscratch/portia/eabe/data/Johnson_lab/cse_work"
RUNS  = "/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs"
```

---

## Task 1: `dilate_mask_jax` (reduce_window max-pool dilation)

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/models/dilation.py`
- Test: `third_party/jarvis_jax/tests/test_dilation.py`

**Interfaces:**
- Consumes: nothing external. Pure JAX (`jax.lax.reduce_window`). Runs on CPU under `JAX_PLATFORMS=cpu`.
- Produces:
  - `dilate_mask_jax(mask: jax.Array, k: int) -> jax.Array` — binary/float morphological dilation by a `k×k` square structuring element, implemented as a single `jax.lax.reduce_window(mask, -inf, jax.lax.max, window_dimensions, window_strides=all-1, padding="SAME")` max-pool. Accepts `mask` of shape `(H,W)`, `(B,H,W)`, or `(B,H,W,C)` (NHWC); returns the SAME shape and dtype-family (float in → float out; the max-pool preserves `{0,1}` for a binary input). `k` is a **static Python int** (the window is built from it, so it must be traced-constant). `k <= 1` (or `k == 0`) is the **identity** (returns `mask` unchanged — no window op). Even `k` uses `padding="SAME"` (asymmetric pad), matching `jax.lax`'s SAME semantics; odd `k` centers the window (matching the PyTorch `padding=dilate//2` reference for odd kernels). Mirrors `F.max_pool2d(m, kernel_size=k, stride=1, padding=k//2)`.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_dilation.py`:

```python
import jax.numpy as jnp
import numpy as np
import pytest


def test_single_pixel_dilates_to_k_block():
    from jarvis_jax.models.dilation import dilate_mask_jax
    # a single hot pixel at the center of an 11x11 field -> a k x k block for odd k.
    for k in (3, 5, 7):
        m = jnp.zeros((11, 11)).at[5, 5].set(1.0)
        out = np.asarray(dilate_mask_jax(m, k))
        r = k // 2
        block = out[5 - r:5 + r + 1, 5 - r:5 + r + 1]
        assert block.shape == (k, k)
        assert np.all(block == 1.0), f"k={k}: interior not fully set"
        assert out.sum() == float(k * k), f"k={k}: expected exactly k*k hot pixels, got {out.sum()}"


def test_k1_and_k0_are_identity():
    from jarvis_jax.models.dilation import dilate_mask_jax
    rng = np.random.default_rng(0)
    m = jnp.asarray((rng.random((7, 9)) > 0.5).astype("float32"))
    assert np.array_equal(np.asarray(dilate_mask_jax(m, 1)), np.asarray(m))
    assert np.array_equal(np.asarray(dilate_mask_jax(m, 0)), np.asarray(m))


def test_batched_nhwc_preserves_shape_and_is_per_sample():
    from jarvis_jax.models.dilation import dilate_mask_jax
    # NHWC: (B=2, H=9, W=9, C=1); one hot pixel in each sample at different spots.
    m = jnp.zeros((2, 9, 9, 1))
    m = m.at[0, 4, 4, 0].set(1.0).at[1, 0, 0, 0].set(1.0)  # sample 1 hot at a corner
    out = np.asarray(dilate_mask_jax(m, 3))
    assert out.shape == (2, 9, 9, 1)
    assert out[0, 3:6, 3:6, 0].sum() == 9.0            # center: full 3x3
    assert out[0].sum() == 9.0
    assert out[1, 0:2, 0:2, 0].sum() == 4.0            # corner: clipped to 2x2 by SAME padding
    assert out[1].sum() == 4.0


def test_nhw_shape_supported():
    from jarvis_jax.models.dilation import dilate_mask_jax
    m = jnp.zeros((3, 8, 8)).at[:, 4, 4].set(1.0)
    out = np.asarray(dilate_mask_jax(m, 5))
    assert out.shape == (3, 8, 8)
    assert np.all(out[:, 2:7, 2:7] == 1.0)


def test_binary_stays_binary_float_stays_float():
    from jarvis_jax.models.dilation import dilate_mask_jax
    m = jnp.asarray(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype="float32"))
    out = dilate_mask_jax(m, 3)
    assert out.dtype == m.dtype
    assert set(np.unique(np.asarray(out)).tolist()) <= {0.0, 1.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_dilation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.models.dilation'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/models/dilation.py`:

```python
"""Binary/float mask dilation via a single reduce_window max-pool (JAX, jit-safe).

Mirrors the PyTorch mask-gating reference
    F.max_pool2d(m, kernel_size=k, stride=1, padding=k // 2)
(jarvis/efficienttrack/loss.py:25-59, jarvis/prediction/sam3_masker.py:25-38) with
no scipy and no Python loop in the traced step, so it runs inside nnx.jit for the
train-time containment loss (Task 2) and the inference gate (Task 3). `k` is the
static square structuring-element side; k <= 1 is the identity.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def dilate_mask_jax(mask, k):
    """Dilate a {0,1}/float mask by a k x k square element (max-pool, stride 1).

    mask: (H,W) | (B,H,W) | (B,H,W,C). Returns the SAME shape/dtype. k is a static
    Python int; k <= 1 (or 0) returns `mask` unchanged (identity). Padding='SAME'
    (window centered for odd k, matching PyTorch padding=k//2).
    """
    k = int(k)
    if k <= 1:
        return mask
    x = jnp.asarray(mask)
    win = (k, k)
    strides = (1, 1)
    # reduce_window needs a window entry per axis; pad the spatial window with 1s
    # on the leading/trailing non-spatial axes so B and C are untouched.
    if x.ndim == 2:                       # (H,W)
        wd, ws, pad = win, strides, "SAME"
    elif x.ndim == 3:                     # (B,H,W) or (H,W,C) -> treat leading as batch-like
        wd, ws, pad = (1, *win), (1, *strides), "SAME"
    elif x.ndim == 4:                     # (B,H,W,C)
        wd, ws, pad = (1, *win, 1), (1, *strides, 1), "SAME"
    else:
        raise ValueError(f"dilate_mask_jax expects 2/3/4-D mask, got shape {x.shape}")
    neg_inf = jnp.array(-jnp.inf, dtype=x.dtype)
    out = jax.lax.reduce_window(x, neg_inf, jax.lax.max, wd, ws, pad)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_dilation.py -v`
Expected: PASS (5 passed).

Note on the NHW case: `test_batched_nhwc...` and `test_nhw_shape_supported` both pass because the 3-D branch treats axis 0 as batch-like (window 1) — correct for `(B,H,W)`. (An `(H,W,C)` 3-D input would be dilated over H and W with C batch-like, which is not used by this pipeline; the pipeline uses `(B,H,W)` for `mask224` (T2/T3) and `(H,W)` in unit tests. Documented in the docstring.)

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/models/dilation.py third_party/jarvis_jax/tests/test_dilation.py
git commit -m "feat(models): dilate_mask_jax (reduce_window max-pool mask dilation, NHWC, jit-safe)

Mirrors the PyTorch F.max_pool2d mask-gating reference; k<=1 identity.
Enabling primitive for train-time containment loss + inference gating (Phase 5).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: dilated `mask_containment(dilate=k)` + wire `mask_dilate` into train.py/TrainConfig

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/train/losses.py` (`mask_containment` grows a `dilate` kwarg)
- Modify: `third_party/jarvis_jax/jarvis_jax/train/train.py` (`TrainConfig.mask_dilate`; `make_train_step(..., mask_dilate=0)`)
- Test: `third_party/jarvis_jax/tests/test_mask_containment_dilated.py`

**Interfaces:**
- Consumes: Task 1 `dilate_mask_jax(mask, k)`; existing `mask_containment(pred, mask224, eps)` semantics.
- Produces:
  - `mask_containment(pred (B,H,W,K), mask224 (B,H,W) float{0,1}, *, dilate: int = 0, eps: float = 1e-6) -> scalar` — dilates `mask224` by `dilate` (via `dilate_mask_jax`) BEFORE computing the outside-positive-mass fraction. `dilate=0` reproduces the current behaviour EXACTLY (byte-for-byte: `dilate_mask_jax` returns the mask unchanged for `k<=1`, so the arithmetic is identical). Larger `dilate` shrinks the "outside" region, so the penalty is monotone non-increasing in `dilate`.
  - `TrainConfig.mask_dilate: int = 0` (new field, default 0 → no dilation, backward compatible).
  - `make_train_step(mask_weight, aug_params=None, lr_swap=None, heatmap_size=224, sigma=7.0, joint_weight=None, mask_dilate: int = 0)` — when `mask_weight>0`, calls `mask_containment(pred, mask224, dilate=mask_dilate)`. `mask_dilate` is captured as a static Python int at `make_train_step` call time (like `mask_weight`), so the jitted step recompiles per distinct `mask_dilate` (fine — set once per run).

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_mask_containment_dilated.py`:

```python
import jax.numpy as jnp
import numpy as np
from jarvis_jax.train.losses import mask_containment


def _peak_just_outside():
    """A 1x1 heatmap peak 2px to the right of a raw 1-pixel mask.
    Raw mask covers (b=0, y=8, x=8); the peak is at (y=8, x=10)."""
    B, H, W, K = 1, 17, 17, 1
    pred = jnp.zeros((B, H, W, K)).at[0, 8, 10, 0].set(1.0)
    mask = jnp.zeros((B, H, W)).at[0, 8, 8].set(1.0)
    return pred, mask


def test_dilate0_matches_current_behaviour_exactly():
    # dilate=0 must equal the pre-Phase-5 call (no kwarg) to the bit.
    pred, mask = _peak_just_outside()
    assert float(mask_containment(pred, mask, dilate=0)) == float(mask_containment(pred, mask))
    # peak is OUTSIDE the undilated 1-pixel mask -> all positive mass outside -> 1.0
    assert abs(float(mask_containment(pred, mask, dilate=0)) - 1.0) < 1e-5


def test_penalty_drops_as_dilate_grows():
    pred, mask = _peak_just_outside()
    d0 = float(mask_containment(pred, mask, dilate=0))    # peak outside -> 1.0
    d3 = float(mask_containment(pred, mask, dilate=3))    # mask now covers x in [7,9]; peak at 10 still out -> 1.0
    d5 = float(mask_containment(pred, mask, dilate=5))    # mask covers x in [6,10]; peak at 10 now INSIDE -> 0.0
    assert d0 == 1.0
    assert d3 == 1.0
    assert d5 < 1e-5, f"dilate=5 should bring the peak inside -> ~0 penalty, got {d5}"
    # monotone non-increasing in dilate
    ds = [float(mask_containment(pred, mask, dilate=k)) for k in (0, 1, 3, 5, 7)]
    assert all(a >= b - 1e-6 for a, b in zip(ds, ds[1:])), f"not monotone: {ds}"


def test_inside_peak_unaffected_by_dilation():
    # peak on the mask pixel -> 0 penalty regardless of dilation.
    B, H, W, K = 1, 9, 9, 1
    pred = jnp.zeros((B, H, W, K)).at[0, 4, 4, 0].set(1.0)
    mask = jnp.zeros((B, H, W)).at[0, 4, 4].set(1.0)
    for k in (0, 3, 7):
        assert float(mask_containment(pred, mask, dilate=k)) < 1e-6


def test_train_step_mask_dilate_wired():
    # make_train_step accepts mask_dilate and runs one step without error (CPU).
    from flax import nnx
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.train.train import TrainConfig, make_optimizer, make_train_step
    cfg = ViTPoseConfig(num_keypoints=3, img_size=64, heatmap_size=32)  # tiny for CPU
    m = ViTPose(cfg, rngs=nnx.Rngs(0))
    opt = make_optimizer(m, TrainConfig(total_steps=10))
    step = make_train_step(mask_weight=1.0, mask_dilate=5, heatmap_size=cfg.heatmap_size)
    img = jnp.zeros((2, 64, 64, 4)).at[..., 3].set(1.0)   # full-fly mask channel
    kp = jnp.full((2, 3, 2), 16.0)
    vis = jnp.ones((2, 3), dtype=bool)
    import jax
    loss = float(step(m, opt, jax.random.PRNGKey(0), img, kp, vis))
    assert np.isfinite(loss)


def test_traincfg_has_mask_dilate_default_zero():
    from jarvis_jax.train.train import TrainConfig
    assert TrainConfig().mask_dilate == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_mask_containment_dilated.py -v`
Expected: FAIL — `mask_containment() got an unexpected keyword argument 'dilate'` (and `TrainConfig` has no `mask_dilate`, `make_train_step` has no `mask_dilate`).

- [ ] **Step 3: Write minimal implementation**

In `third_party/jarvis_jax/jarvis_jax/train/losses.py`, replace `mask_containment` (lines 32-39) with:

```python
def mask_containment(pred, mask224, *, dilate=0, eps=1e-6):
    """Fraction of positive predicted heatmap mass outside the (dilated) mask.

    pred: (B,H,W,K); mask224: (B,H,W) float {0,1}. Returns a scalar in [0,1].
    ``dilate`` (static int) grows the mask by a dilate x dilate max-pool before
    the penalty (edge slack for wings/legs that legitimately extend past the raw
    SAM mask). dilate=0 == the original behaviour. Monotone non-increasing in
    dilate. Mirrors the PyTorch mask_containment_loss dilation.
    """
    from jarvis_jax.models.dilation import dilate_mask_jax
    m = dilate_mask_jax(mask224, dilate) if dilate and dilate > 1 else mask224
    p = jax.nn.relu(pred)                                 # positive mass
    outside = p * (1.0 - m[..., None])
    return outside.sum() / (p.sum() + eps)
```

In `third_party/jarvis_jax/jarvis_jax/train/train.py`, add the field to `TrainConfig` (after `mask_weight`, line 21):

```python
    mask_weight: float = 0.0
    mask_dilate: int = 0        # dilate the mask (px) before the containment penalty (0 = off)
```

and thread it through `make_train_step` (signature line 56-57 + the loss branch lines 78-82):

```python
def make_train_step(mask_weight, aug_params=None, lr_swap=None, heatmap_size=224,
                    sigma=7.0, joint_weight=None, mask_dilate=0):
    ...
    mw = float(mask_weight)
    md = int(mask_dilate)
    ...
        if mw > 0.0:
            mask = img[..., 3]
            mask224 = jax.image.resize(
                mask, (mask.shape[0], pred.shape[1], pred.shape[2]), method="nearest")
            loss = loss + mw * mask_containment(pred, mask224, dilate=md)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_mask_containment_dilated.py -v`
Expected: PASS (5 passed).

Then re-run the existing losses + train-step suites to prove no regression:
```bash
cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_losses.py tests/test_train_step.py -v
```
Expected: PASS (the pre-existing `test_mask_containment_zero_inside_one_outside` still passes — `dilate` defaults to 0 → identical arithmetic).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/train/losses.py third_party/jarvis_jax/jarvis_jax/train/train.py third_party/jarvis_jax/tests/test_mask_containment_dilated.py
git commit -m "feat(train): dilated mask_containment(dilate=k) + TrainConfig.mask_dilate wiring

dilate=0 reproduces current behaviour exactly; penalty monotone in dilate.
Turns on the train-time dilated-mask containment loss (Phase 5).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: inference-time gating helper (`gate_heatmaps`) + optional predict wire-in

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/gating.py`
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/predict_full.py` (opt-in `--gate-dilate`, default off)
- Test: `third_party/jarvis_jax/tests/test_inference_gating.py`

**Interfaces:**
- Consumes: Task 1 `dilate_mask_jax`; `jax.image.resize` (nearest, mask→heatmap grid).
- Produces:
  - `gate_heatmaps(hm: jax.Array, mask: jax.Array, dilate: int = 21) -> jax.Array` — mirror of `jarvis3D_multi.py:331-333`. `hm`: `(B,H,W,K)` heatmap logits/scores. `mask`: `(B,Hm,Wm)` float{0,1} SAM mask at ANY resolution (typically the model input res 448); it is dilated by `dilate` (via `dilate_mask_jax`), resized to `(H,W)` with `method="nearest"`, re-binarized `>0.5`, and used to **zero heatmap values outside** it: `hm * mask_hw[...,None]`. Peaks inside the dilated mask are preserved bit-for-bit; mass outside becomes 0. `dilate<=1` gates by the raw (undilated) mask. Returns the same shape/dtype as `hm`. NOT jit-required (used at inference), but jit-safe.
  - `predict_full.py --gate-dilate INT` (default `0` = OFF → current behaviour). When `>0`, applies `gate_heatmaps(hm, mask448, dilate)` to the per-camera heatmaps before `heatmaps_to_keypoints` and before the 3-D reproject. `mask448` = the 4th channel of the input crops (`crops4[..., 3]`), already pixel-aligned. Default-off keeps every existing prediction run byte-for-byte identical.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_inference_gating.py`:

```python
import jax.numpy as jnp
import numpy as np
from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints


def test_gate_zeroes_outside_preserves_inside():
    from jarvis_jax.densepose.gating import gate_heatmaps
    B, H, W, K = 1, 32, 32, 2
    hm = jnp.zeros((B, H, W, K))
    # ch0: a strong peak INSIDE the mask; ch1: a spurious peak OUTSIDE the mask.
    hm = hm.at[0, 10, 10, 0].set(5.0).at[0, 28, 28, 1].set(3.0)
    # mask at input res 128 covers only the top-left quadrant (dilate=0 -> raw).
    mask = jnp.zeros((B, 128, 128))
    mask = mask.at[0, :64, :64].set(1.0)                 # top-left quadrant hot
    out = np.asarray(gate_heatmaps(hm, mask, dilate=0))
    assert out.shape == (B, H, W, K)
    # the inside peak (10,10 in 32-grid == top-left quadrant) survives unchanged.
    assert out[0, 10, 10, 0] == 5.0
    # the outside peak (28,28 == bottom-right, mask=0) is zeroed.
    assert out[0, 28, 28, 1] == 0.0
    # everything outside the mask is zero.
    assert out[0, 16:, 16:, :].sum() == 0.0


def test_gate_moves_decoded_keypoint_onto_the_fly():
    from jarvis_jax.densepose.gating import gate_heatmaps
    B, H, W, K = 1, 32, 32, 1
    # a TALL spurious off-fly peak plus a smaller on-fly peak; ungated argmax picks
    # the spurious one, gated argmax picks the on-fly one.
    hm = jnp.zeros((B, H, W, K)).at[0, 28, 28, 0].set(9.0).at[0, 8, 8, 0].set(4.0)
    mask = jnp.zeros((B, 32, 32)).at[0, :16, :16].set(1.0)   # top-left is the fly
    kp_ungated = np.asarray(heatmaps_to_keypoints(hm, in_size=32))[0, 0]
    kp_gated = np.asarray(heatmaps_to_keypoints(gate_heatmaps(hm, mask, dilate=0), in_size=32))[0, 0]
    # ungated lands near the spurious (28,28); gated lands near the real (8,8).
    assert kp_ungated[0] > 20 and kp_ungated[1] > 20
    assert kp_gated[0] < 16 and kp_gated[1] < 16


def test_dilate_admits_a_peak_just_outside_the_raw_mask():
    from jarvis_jax.densepose.gating import gate_heatmaps
    B, H, W, K = 1, 16, 16, 1
    hm = jnp.zeros((B, H, W, K)).at[0, 8, 10, 0].set(2.0)   # peak at x=10
    mask = jnp.zeros((B, 16, 16)).at[0, 8, 8].set(1.0)      # raw mask covers x=8 only
    assert np.asarray(gate_heatmaps(hm, mask, dilate=0))[0, 8, 10, 0] == 0.0   # raw: zeroed
    assert np.asarray(gate_heatmaps(hm, mask, dilate=5))[0, 8, 10, 0] == 2.0   # dilated: preserved


def test_full_mask_is_identity():
    from jarvis_jax.densepose.gating import gate_heatmaps
    rng = np.random.default_rng(0)
    hm = jnp.asarray(rng.normal(size=(2, 16, 16, 3)).astype("float32"))
    mask = jnp.ones((2, 16, 16))                           # whole image is fly
    out = gate_heatmaps(hm, mask, dilate=0)
    assert np.allclose(np.asarray(out), np.asarray(hm))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_inference_gating.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.densepose.gating'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/gating.py`:

```python
"""Inference-time dilated-mask gating for keypoint/dense heatmaps (JAX).

Mirrors jarvis/prediction/jarvis3D_multi.py:331-333: expand the SAM fly mask by a
dilation kernel (kp slack for wings/legs that extend past the mask edge), then
zero all heatmap mass outside it before decoding keypoints. Peaks inside the
dilated mask are preserved; spurious off-fly peaks are removed so the argmax /
soft-argmax lands on the fly. Opt-in at prediction time (Phase 5).
"""
from __future__ import annotations

import jax.image
import jax.numpy as jnp

from jarvis_jax.models.dilation import dilate_mask_jax


def gate_heatmaps(hm, mask, dilate=21):
    """Zero heatmap mass outside the dilated SAM mask.

    hm:   (B,H,W,K) heatmaps. mask: (B,Hm,Wm) float {0,1} (any res). dilate: static
    px kernel (0/1 -> gate by the raw mask). Returns (B,H,W,K), same dtype as hm.
    """
    hm = jnp.asarray(hm)
    B, H, W, K = hm.shape
    m = dilate_mask_jax(jnp.asarray(mask, dtype=hm.dtype), dilate)
    m_hw = jax.image.resize(m, (B, H, W), method="nearest")
    m_hw = (m_hw > 0.5).astype(hm.dtype)
    return hm * m_hw[..., None]
```

Wire the opt-in into `predict_full.py`: add the arg (after `--sharpen`, line 27) —
```python
    ap.add_argument("--gate-dilate", type=int, default=0,
                    help="inference-time dilated-mask gating kernel (0 = off; 21 = jarvis3D default)")
```
and apply it to the decoded 2-D heatmaps + the 3-D volume input, right after `hm = np.asarray(hyb.predict_heatmaps(crops))` (line 60):
```python
        if a.gate_dilate > 0:
            from jarvis_jax.densepose.gating import gate_heatmaps
            mask448 = jnp.asarray(crops[..., 3]).reshape(b * nc, crops.shape[2], crops.shape[3])
            hm = np.asarray(gate_heatmaps(
                jnp.asarray(hm).reshape(b * nc, 224, 224, J), mask448, a.gate_dilate)
            ).reshape(b, nc, 224, 224, J)
```
(Default `--gate-dilate 0` skips this block entirely → unchanged behaviour. `crops` is `(b,nc,448,448,4)`; the 4th channel is the SAM mask, pixel-aligned.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_inference_gating.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/gating.py third_party/jarvis_jax/jarvis_jax/cse/predict_full.py third_party/jarvis_jax/tests/test_inference_gating.py
git commit -m "feat(cse): inference-time gate_heatmaps (dilate SAM mask + zero outside) + opt-in predict wire

Mirrors jarvis3D_multi.py:331-333; --gate-dilate default 0 keeps predict unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: pluggable-backbone abstraction (`models/backbone.py`) + `ViTPose(backbone=...)`

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/models/backbone.py`
- Modify: `third_party/jarvis_jax/jarvis_jax/models/vitpose.py` (`ViTPose.__init__` accepts `backbone=`)
- Test: `third_party/jarvis_jax/tests/test_backbone_pluggable.py`

**Interfaces:**
- Consumes: existing `ViT(cfg, *, rngs)`; `ViTPoseConfig`.
- Produces:
  - `models/backbone.py`:
    - `Backbone` — a `typing.runtime_checkable` `Protocol` declaring `__call__(self, x) -> jax.Array` (image `(B,H,W,in_ch)` → tokens `(B,num_tokens,embed_dim)`). Documentation-only structural type; a concrete backbone is any NNX module matching it.
    - `BACKBONES: dict[str, callable]` — registry name → factory `(cfg, *, rngs) -> nnx.Module`.
    - `register_backbone(name: str, factory) -> None` — add/override a factory (raises `ValueError` on duplicate unless the factory is identical, to avoid silent shadowing).
    - `build_backbone(name_or_module, cfg, *, rngs) -> nnx.Module` — if given a string, look it up in `BACKBONES` and call `factory(cfg, rngs=rngs)`; if given an already-built module, return it unchanged; raise `KeyError` for an unknown name. `"vit"` is pre-registered → `ViT`.
  - `models/vitpose.py`: `ViTPose(cfg, *, backbone=None, rngs)` — `backbone` is `None` (default → `ViT(cfg, rngs=rngs)`, identical to today), a registry NAME (str → `build_backbone`), or a prebuilt module (returned as-is). `self.backbone` is always the resulting module (attribute name unchanged → checkpoint param paths `backbone/...` unchanged → warm-start / restore of existing `cse_vit350` checkpoints unaffected). Forward and output shape `(B,224,224,K)` unchanged.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_backbone_pluggable.py`:

```python
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jarvis_jax.config import ViTPoseConfig


TINY = dict(img_size=64, patch=16, embed_dim=48, depth=2, num_heads=4,
            num_keypoints=5, heatmap_size=32)   # CPU-friendly


def test_registry_has_vit_and_builds_it():
    from jarvis_jax.models.backbone import BACKBONES, build_backbone, Backbone
    from jarvis_jax.models.vit import ViT
    assert "vit" in BACKBONES
    cfg = ViTPoseConfig(**TINY)
    bb = build_backbone("vit", cfg, rngs=nnx.Rngs(0))
    assert isinstance(bb, ViT)
    assert isinstance(bb, Backbone)                        # structural protocol match
    toks = bb(jnp.zeros((1, 64, 64, 4)))
    assert toks.shape == (1, (64 // 16) ** 2, 48)          # (B, num_tokens, embed_dim)


def test_build_backbone_passthrough_and_unknown():
    from jarvis_jax.models.backbone import build_backbone
    from jarvis_jax.models.vit import ViT
    cfg = ViTPoseConfig(**TINY)
    prebuilt = ViT(cfg, rngs=nnx.Rngs(1))
    assert build_backbone(prebuilt, cfg, rngs=nnx.Rngs(0)) is prebuilt   # passthrough
    with pytest.raises(KeyError):
        build_backbone("does_not_exist", cfg, rngs=nnx.Rngs(0))


def test_register_backbone_and_use_in_vitpose():
    from jarvis_jax.models.backbone import register_backbone, BACKBONES
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.models.vit import ViT

    calls = {"n": 0}
    def _wrapped(cfg, *, rngs):
        calls["n"] += 1
        return ViT(cfg, rngs=rngs)
    register_backbone("vit_counted", _wrapped)
    assert "vit_counted" in BACKBONES
    cfg = ViTPoseConfig(**TINY)
    m = ViTPose(cfg, backbone="vit_counted", rngs=nnx.Rngs(0))
    assert calls["n"] == 1
    out = m(jnp.zeros((1, 64, 64, 4)))
    assert out.shape == (1, 32, 32, 5)


def test_vitpose_default_backward_compat():
    # the pre-existing hardcoded-ViT construction path is unchanged.
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.models.vit import ViT
    cfg = ViTPoseConfig(**TINY)
    m = ViTPose(cfg, rngs=nnx.Rngs(0))                     # no backbone= given
    assert isinstance(m.backbone, ViT)
    out = m(jnp.zeros((1, 64, 64, 4)))
    assert out.shape == (1, 32, 32, 5)
    assert jnp.isfinite(out).all()


def test_default_param_paths_unchanged():
    # ViTPose(cfg) and ViTPose(cfg, backbone="vit") must yield the SAME param-tree
    # structure (so existing checkpoints restore into either).
    from jarvis_jax.models.vitpose import ViTPose
    cfg = ViTPoseConfig(**TINY)
    a = ViTPose(cfg, rngs=nnx.Rngs(0))
    b = ViTPose(cfg, backbone="vit", rngs=nnx.Rngs(0))
    ka = sorted(jax.tree_util.keystr(p) for p, _ in nnx.split(a)[1].flat_state().items())
    kb = sorted(jax.tree_util.keystr(p) for p, _ in nnx.split(b)[1].flat_state().items())
    assert ka == kb


import jax  # noqa: E402  (used by test_default_param_paths_unchanged)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_backbone_pluggable.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.models.backbone'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/models/backbone.py`:

```python
"""Pluggable backbone registry for ViTPose (Phase 5).

Sets up the abstraction a future SAM3 ViTDet-trunk port would register into,
WITHOUT porting it (that ViT-L trunk is an explicit non-goal — spec §7/§10). The
default backbone is the existing JAX ViT-B ('vit'), so every current construction
path and the on-disk cse_vit350 checkpoints are unchanged.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from flax import nnx

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vit import ViT


@runtime_checkable
class Backbone(Protocol):
    """A vision trunk: image (B,H,W,in_ch) -> patch tokens (B,num_tokens,embed_dim)."""
    def __call__(self, x): ...


def _vit_factory(cfg: ViTPoseConfig, *, rngs: nnx.Rngs):
    return ViT(cfg, rngs=rngs)


BACKBONES = {"vit": _vit_factory}


def register_backbone(name, factory):
    """Register a backbone factory `(cfg, *, rngs) -> nnx.Module` under `name`."""
    existing = BACKBONES.get(name)
    if existing is not None and existing is not factory:
        raise ValueError(f"backbone '{name}' already registered to a different factory")
    BACKBONES[name] = factory


def build_backbone(name_or_module, cfg, *, rngs):
    """Return a backbone module. str -> look up + build; module -> passthrough."""
    if isinstance(name_or_module, str):
        try:
            factory = BACKBONES[name_or_module]
        except KeyError as e:
            raise KeyError(
                f"unknown backbone '{name_or_module}'; registered: {sorted(BACKBONES)}") from e
        return factory(cfg, rngs=rngs)
    return name_or_module      # already a built module
```

Modify `third_party/jarvis_jax/jarvis_jax/models/vitpose.py` — replace `__init__` (lines 13-16):

```python
    def __init__(self, cfg: ViTPoseConfig, *, backbone=None, rngs: nnx.Rngs):
        self.cfg = cfg
        if backbone is None:
            self.backbone = ViT(cfg, rngs=rngs)             # backward-compatible default
        else:
            from jarvis_jax.models.backbone import build_backbone
            self.backbone = build_backbone(backbone, cfg, rngs=rngs)
        self.decoder = ClassicDecoder(cfg.embed_dim, cfg.num_keypoints, rngs=rngs)
```

(The `ViT` import at the top of `vitpose.py` stays — the default path uses it directly, so no circular import via `backbone.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_backbone_pluggable.py tests/test_vitpose.py -v`
Expected: PASS (5 new + the pre-existing `test_vitpose_forward`, which still constructs `ViTPose(cfg, rngs=...)` with no `backbone=` and gets `(1,224,224,50)`).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/models/backbone.py third_party/jarvis_jax/jarvis_jax/models/vitpose.py third_party/jarvis_jax/tests/test_backbone_pluggable.py
git commit -m "feat(models): pluggable Backbone protocol/registry; ViTPose(backbone=...) default ViT

Backward compatible (default None -> ViT; param paths unchanged). SAM3 ViTDet port
out of scope; this only builds the abstraction a future port would register into.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: dense-channel L/R flip map (`build_dense_lr_swap`) + enable CSE flip aug

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/dense_lr_swap.py`
- Test: `third_party/jarvis_jax/tests/test_lr_swap_dense.py`
- (Consumed by Task 6's fine-tune to enable flip; the aug pipeline `augment.flip_batch` already handles an arbitrary swap array — no change to `augment.py` needed.)

**Interfaces:**
- Consumes: `augment.build_lr_swap(names)` (the 50 named kp swap); the canonical mesh `vertices`, `vertex_segment`, `seg_ids`, `seg_names`, `fps_<M>`; SciPy `cKDTree` (offline, NOT in the jit step).
- Produces:
  - `build_dense_lr_swap(mesh_npz: str, fps_key: str, base_names: list[str], *, tol: float = 0.1) -> np.ndarray (int32, (50+M,))` — concatenates: (i) `build_lr_swap(base_names)` for the first 50 channels, and (ii) a `50 + swap_verts` block for the M FPS vertices, where `swap_verts` (int in `[0,M)`) is the **segment-aware mutual-nearest-neighbour Y-reflection** pairing: reflect each FPS vertex's canonical `vertices` coord across the Y plane (`y -> -y`), find its nearest FPS vertex, keep the pair only if it is a **mutual** nearest neighbour within `tol` AND the partner's mesh segment equals the source segment's L/R mirror (so a `wing_left` vertex maps into `wing_right`, never into an adjacent thorax vertex); unpaired vertices map to themselves (midline / no confident partner). Verified on `fps_300`: involution, 246/300 swapped, `wing_left`(seg 9)→`wing_right`(seg 10) 100%, 0 cross-part mispairs. Asserts the FULL (50+M) result is an involution (`arr[arr] == arange`) before returning. `M = len(z[fps_key])`.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_lr_swap_dense.py`:

```python
import os
import numpy as np
import pytest

MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"

BASE_50 = [
    "Scutellum", "WingL_base", "WingR_base", "Antenna_Base", "EyeL", "EyeR",
    "WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13", "Abd_A4", "Abd_tip",
    "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip",
    "T1R_ThxCx", "T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip",
    "T2L_Tro", "T2L_FeTi", "T2L_TiTa", "T2L_TaT1", "T2L_TaT3", "T2L_TaTip",
    "T2R_Tro", "T2R_FeTi", "T2R_TiTa", "T2R_TaT1", "T2R_TaT3", "T2R_TaTip",
    "T3L_Tro", "T3L_FeTi", "T3L_TiTa", "T3L_TaT1", "T3L_TaT3", "T3L_TaTip",
    "T3R_Tro", "T3R_FeTi", "T3R_TiTa", "T3R_TaT1", "T3R_TaT3", "T3R_TaTip",
]

_HAVE_MESH = os.path.exists(MESH)


@pytest.mark.skipif(not _HAVE_MESH, reason="canonical mesh not present")
def test_dense_swap_is_involution_full_length():
    from jarvis_jax.densepose.dense_lr_swap import build_dense_lr_swap
    swap = build_dense_lr_swap(MESH, "fps_300", BASE_50)
    assert swap.shape == (50 + 300,)
    assert swap.dtype == np.int32
    assert np.array_equal(swap[swap], np.arange(50 + 300)), "dense swap is not an involution"
    # the first 50 entries are exactly the named-keypoint swap.
    from jarvis_jax.data.augment import build_lr_swap
    assert np.array_equal(swap[:50], build_lr_swap(BASE_50))
    # dense block indexes only into the dense block (50..350), never into the kp block.
    assert swap[50:].min() >= 50 and swap[50:].max() < 350


@pytest.mark.skipif(not _HAVE_MESH, reason="canonical mesh not present")
def test_named_kp_left_maps_to_right():
    from jarvis_jax.densepose.dense_lr_swap import build_dense_lr_swap
    swap = build_dense_lr_swap(MESH, "fps_300", BASE_50)
    i_L = BASE_50.index("WingL_V12"); i_R = BASE_50.index("WingR_V12")
    assert swap[i_L] == i_R and swap[i_R] == i_L
    i_eL = BASE_50.index("EyeL"); i_eR = BASE_50.index("EyeR")
    assert swap[i_eL] == i_eR
    assert swap[BASE_50.index("Scutellum")] == BASE_50.index("Scutellum")  # midline self


@pytest.mark.skipif(not _HAVE_MESH, reason="canonical mesh not present")
def test_left_wing_vertex_maps_to_a_right_wing_vertex():
    from jarvis_jax.densepose.dense_lr_swap import build_dense_lr_swap
    z = np.load(MESH, allow_pickle=True)
    fps = z["fps_300"]; seg = z["vertex_segment"][fps]   # per-fps segment id
    swap = build_dense_lr_swap(MESH, "fps_300", BASE_50)
    wl = np.where(seg == 9)[0]                             # wing_left fps verts
    assert len(wl) > 0
    partners_dense = swap[50 + wl] - 50                    # dense-local partner idx
    assert np.all(seg[partners_dense] == 10), "a left-wing vertex must map into wing_right (seg 10)"
    # and it is a genuine swap (not self) for the wing verts.
    assert np.all(partners_dense != wl)


@pytest.mark.skipif(not _HAVE_MESH, reason="canonical mesh not present")
def test_flipping_a_synthetic_example_swaps_L_and_R():
    """End-to-end: apply the dense swap the way flip_batch does (kp[:, swap]) and
    confirm a synthetic labeled example with distinct L/R coords swaps correctly."""
    from jarvis_jax.densepose.dense_lr_swap import build_dense_lr_swap
    swap = build_dense_lr_swap(MESH, "fps_300", BASE_50)
    J = 50 + 300
    kp = np.arange(J * 2).reshape(J, 2).astype(np.float32)  # unique per-channel coords
    kp_sw = kp[swap]
    # applying the involution twice restores the original (what flip-then-flip does).
    assert np.array_equal(kp_sw[swap], kp)
    # a specific left-wing vertex's coords land on its right-wing partner's slot.
    z = np.load(MESH, allow_pickle=True); seg = z["vertex_segment"][z["fps_300"]]
    wl = int(np.where(seg == 9)[0][0]); partner = int(swap[50 + wl] - 50)
    assert np.array_equal(kp_sw[50 + partner], kp[50 + wl])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_lr_swap_dense.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.densepose.dense_lr_swap'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/dense_lr_swap.py`:

```python
"""Dense-channel L/R flip map for the CSE (50 + M) ViTPose (Phase 5).

The CSE model's flip augmentation was DISABLED because the 50-keypoint
build_lr_swap does not cover the M dense FPS vertices. The raw mesh `sym_index`
is a per-vertex bilateral partner but is NOT closed under FPS subsampling (only
~1/3 of an fps subset's partners are themselves in the subset, and the restricted
map is not an involution). So build the dense block by a segment-aware
mutual-nearest-neighbour reflection across the bilateral plane.

VERIFIED (this cycle): in the canonical `vertices` frame the bilateral mirror axis
is Y (abs-mean signed |vtx - sym_partner| per axis ~ [1.6e-4, 0.247, 1.2e-4]).
Segment-aware mutual-NN reflection on fps_300 gives an involution, 246/300 swapped,
wing_left(seg 9) -> wing_right(seg 10) 100%, 0 cross-part mispairs.
"""
from __future__ import annotations

import numpy as np

from jarvis_jax.data.augment import build_lr_swap

_MIRROR_AXIS = 1        # Y, in the canonical `vertices` frame (verified)


def _mirror_seg_name(name):
    for a, b in (("_left", "_right"), ("_right", "_left"), ("_L", "_R"), ("_R", "_L")):
        if name.endswith(a):
            return name[: -len(a)] + b
    return name          # midline segment -> itself


def _dense_vertex_swap(mesh_npz, fps_key, tol):
    """Segment-aware mutual-NN Y-reflection pairing of the M FPS vertices.
    Returns int (M,) with values in [0, M); unpaired -> self. Involution."""
    from scipy.spatial import cKDTree

    z = np.load(mesh_npz, allow_pickle=True)
    fps = np.asarray(z[fps_key])
    V = np.asarray(z["vertices"])[fps]                    # (M,3) canonical coords
    seg = np.asarray(z["vertex_segment"])[fps]           # (M,) segment id per fps vtx
    seg_names = {int(s): (n.decode() if isinstance(n, bytes) else str(n))
                 for s, n in zip(z["seg_ids"], z["seg_names"])}
    name2id = {v: k for k, v in seg_names.items()}
    # mirror-segment id for each fps vertex (L<->R; midline -> same id).
    seg_mirror = np.array(
        [name2id.get(_mirror_seg_name(seg_names[int(s)]), int(s)) for s in seg])

    refl = V.copy()
    refl[:, _MIRROR_AXIS] *= -1.0                         # reflect across the bilateral plane
    tree = cKDTree(V)
    dist, nn = tree.query(refl, k=1)                      # nearest real vtx to each reflection

    M = len(fps)
    swap = np.arange(M)
    for i in range(M):
        j = int(nn[i])
        # mutual NN within tol AND segment-consistent (L<->R) -> a valid bilateral pair.
        if (dist[i] <= tol and dist[j] <= tol and int(nn[j]) == i
                and seg[j] == seg_mirror[i]):
            swap[i] = j
    assert np.array_equal(swap[swap], np.arange(M)), \
        "dense vertex swap is not an involution"
    return swap.astype(np.int32)


def build_dense_lr_swap(mesh_npz, fps_key, base_names, *, tol=0.1):
    """(50+M,) int32 L/R involution: named-kp swap (first 50) ++ dense vertex swap."""
    kp_swap = build_lr_swap(base_names).astype(np.int32)          # (50,)
    if len(kp_swap) != 50:
        raise ValueError(f"expected 50 base keypoint names, got {len(kp_swap)}")
    vtx_swap = _dense_vertex_swap(mesh_npz, fps_key, tol)         # (M,) in [0,M)
    swap = np.concatenate([kp_swap, 50 + vtx_swap]).astype(np.int32)
    assert np.array_equal(swap[swap], np.arange(len(swap))), \
        "combined (50+M) swap is not an involution"
    return swap
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_lr_swap_dense.py -v`
Expected: PASS (4 passed; the mesh exists on disk, so no skips).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/dense_lr_swap.py third_party/jarvis_jax/tests/test_lr_swap_dense.py
git commit -m "feat(cse): build_dense_lr_swap (50+M L/R involution via seg-aware mutual-NN Y-reflection)

Enables flip augmentation for the CSE dense model. Raw sym_index is not closed
under FPS subsampling; verified: involution, wing_left->wing_right 100%, 0 mispairs.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: gated fine-tune run (warm-start `cse_vit350`, gating ON + flip ON) — GPU, coordinator-run

**PREREQUISITE:** Tasks 1–5 committed (the fine-tune imports `dense_lr_swap`, sets `mask_dilate`, relies on the `mask_containment(dilate=)` wiring). This task is **NOT a pytest** — it is a documented, reproducible 8-GPU fine-tune command (via a committed sbatch script + `train_keypoints_cse_full.py` additions) plus a **checkpoint-exists assertion**. The produced checkpoints under `jax_vitpose_runs/` are DATA and are **NOT committed**.

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/train_keypoints_cse_full.py` (add `--mask-weight`, `--mask-dilate`, `--flip-p`, `--base-names-from`; wire the dense swap + gating).
- Create: `third_party/jarvis_jax/jarvis_jax/cse/sbatch_cse_gated_finetune.sh` **(committed)** — the reproducible 8-GPU run.

**Interfaces:**
- Consumes: `warm_start_from_v3(model, v3_ckpt, cfg50)` with `--v3-ckpt cse_vit350_wings/final --warmstart-joints 350` (this is the trained CSE-350, matching `num_joints=350`; warm-start copies all 350 channels); `TrainConfig(..., mask_weight, mask_dilate)`; `make_train_step(mask_weight, aug, lr_swap, ..., mask_dilate)`; `build_dense_lr_swap(mesh_npz, f"fps_{M}", base_names)`; `AugParams(enabled=True, flip_p=...)`; `CSEImageDataset`; `data_parallel_mesh`/`replicate`; `make_manager`/`save_step`/`restore_latest`.
- Produces (artifacts on disk, NOT committed):
  - `RUNS/cse_vit350_gated/ckpt/` (Orbax, auto-resume) + `RUNS/cse_vit350_gated/final` (the GATED checkpoint).
  - The slurm log (training-loss curve incl. the containment-loss contribution).
  - The base ungated checkpoint for the T7 ablation is the existing `RUNS/cse_vit350_wings/final` (warm-start source; the UNGATED arm).

- [ ] **Step 1: Add the gating + flip args to `train_keypoints_cse_full.py`**

In `argparse` (after `--eval-every`, ~line 49), add:
```python
    ap.add_argument("--mask-weight", type=float, default=0.0,
                    help="dilated-mask containment loss weight (>0 turns gating ON)")
    ap.add_argument("--mask-dilate", type=int, default=11,
                    help="containment-loss mask dilation kernel (px); used only when mask-weight>0")
    ap.add_argument("--flip-p", type=float, default=0.0,
                    help="horizontal-flip prob; >0 enables flip aug using the dense L/R swap")
    ap.add_argument("--base-names",
                    default="/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs/anatomy/v1.yaml",
                    help="anatomy yaml providing the 50 base keypoint names (for the flip swap)")
```

Replace the `TrainConfig` construction (line 81-82) to pass the mask fields:
```python
    tcfg = TrainConfig(lr=a.lr, total_steps=a.steps, batch_size=a.batch,
                       backbone_lr_mult=a.backbone_lr_mult,
                       mask_weight=a.mask_weight, mask_dilate=a.mask_dilate)
```

Replace the flip/aug block (lines 95-97) with a dense-swap-aware version:
```python
    # aug: flip ON iff --flip-p>0, using the dense (50+M) L/R involution.
    if a.flip_p > 0.0:
        from jarvis_jax.densepose.dense_lr_swap import build_dense_lr_swap
        from jarvis_jax.densepose.cse_labels import model_kp_order  # 50 canonical STAC names
        base_names = model_kp_order(a.base_names)
        M = a.num_joints - 50
        lr_swap = build_dense_lr_swap(a.mesh_npz, f"fps_{M}", base_names)
        aug = AugParams(enabled=True, flip_p=a.flip_p)
        print(f"[cse-train] flip ON (p={a.flip_p}) with dense L/R swap "
              f"({int((lr_swap != np.arange(len(lr_swap))).sum())}/{len(lr_swap)} channels swapped)")
    else:
        aug = AugParams(enabled=True, flip_p=0.0)
        lr_swap = np.arange(a.num_joints, dtype=np.int32)
```
(When `--flip-p>0`, `--mesh-npz` must be provided — it already exists as an arg. If `model_kp_order` is not the actual helper name in `cse_labels.py`, use the module's canonical-name accessor; the Phase-4 plan verified the 50 STAC names live in `stac-mjx/configs/anatomy/v1.yaml` `model.KEYPOINT_MODEL_PAIRS` — read them from there. Confirm the exact accessor name at implementation time with `grep -n "def .*kp_order\|KEYPOINT_MODEL_PAIRS" jarvis_jax/cse/cse_labels.py`.)

Replace the `make_train_step` call (line 123-124) to pass `mask_dilate`:
```python
    step = make_train_step(tcfg.mask_weight, aug, lr_swap, heatmap_size=cfg.heatmap_size,
                           sigma=jnp.asarray(sigma), joint_weight=jnp.asarray(jweight),
                           mask_dilate=tcfg.mask_dilate)
```

Commit the code change:
```bash
git add third_party/jarvis_jax/jarvis_jax/cse/train_keypoints_cse_full.py
git commit -m "feat(cse): gated fine-tune args (mask-weight/mask-dilate + flip via dense L/R swap)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 2: Commit the reproducible sbatch script**

Create `third_party/jarvis_jax/jarvis_jax/cse/sbatch_cse_gated_finetune.sh` (modeled on `sbatch_vit350b_wingfinetune.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=cse_vit350_gated
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodelist=g[3090-3137]
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=eabe@uw.edu
#SBATCH -o /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit350_gated/slurm-%j.out
#
# Phase-5 GATED fine-tune: warm-start cse_vit350_wings/final (350 = 50 kp + 300 verts)
# and continue training with the dilated-mask containment loss ON (mask-weight>0,
# mask-dilate=11) AND flip augmentation ON (dense (50+M) L/R swap). ckpt-g2 is
# preemptible -> FIXED --ckpt-dir + --requeue => auto-resume.
set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
nvidia-smi -L

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
WORK=/gscratch/portia/eabe/data/Johnson_lab/cse_work
RUNS=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs
ROOT=/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3
MESH=/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz
RUN=$RUNS/cse_vit350_gated
mkdir -p "$RUN"

cd "$PKG"
python -u -c "import jax; print('jax devices:', jax.device_count())"
python -u -m jarvis_jax.densepose.train_keypoints_cse_full \
    --root "$ROOT" \
    --aux-train "$WORK/cse_labels_train_M300.npz" \
    --aux-val   "$WORK/cse_labels_val_M300.npz" \
    --v3-ckpt "$RUNS/cse_vit350_wings/final" --warmstart-joints 350 \
    --mesh-npz "$MESH" \
    --mask-weight 0.1 --mask-dilate 11 --flip-p 0.5 \
    --out "$RUN/final" --ckpt-dir "$RUN/ckpt" \
    --num-joints 350 --steps 6000 --batch 24 --lr 5e-4 --backbone-lr-mult 0.1 \
    --save-every 500 --log-every 50 --eval-every 1000
echo "GATED FINETUNE DONE"
```

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/sbatch_cse_gated_finetune.sh
git commit -m "feat(cse): reproducible 8-GPU gated CSE fine-tune sbatch (warm-start cse_vit350)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 3: Launch the fine-tune (coordinator runs GPU; sbatch OR background — NOT a subagent)**

Warm-start is `cse_vit350_wings/final`; 6000 steps @ batch 24 on 8 L40S ≈ 1–4h. Two equivalent launch options:

- sbatch (preferred for preempt/requeue auto-resume):
```bash
sbatch /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax/jarvis_jax/cse/sbatch_cse_gated_finetune.sh
```
- OR, if already on an 8-GPU node, run detached (background bash, persists across turns):
```bash
source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
python -u -m jarvis_jax.densepose.train_keypoints_cse_full \
  --root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3 \
  --aux-train /gscratch/portia/eabe/data/Johnson_lab/cse_work/cse_labels_train_M300.npz \
  --aux-val   /gscratch/portia/eabe/data/Johnson_lab/cse_work/cse_labels_val_M300.npz \
  --v3-ckpt /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit350_wings/final --warmstart-joints 350 \
  --mesh-npz /gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz \
  --mask-weight 0.1 --mask-dilate 11 --flip-p 0.5 \
  --out /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit350_gated/final \
  --ckpt-dir /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit350_gated/ckpt \
  --num-joints 350 --steps 6000 --batch 24 --lr 5e-4 --backbone-lr-mult 0.1 \
  --save-every 500 --log-every 50 --eval-every 1000 \
  > /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit350_gated/train.log 2>&1
```

Watch the log: `loss` should stay finite and the val MPJPE(all) should NOT regress vs the ungated warm-start (a first-order sanity check — the containment term should reduce off-fly mass without wrecking kp accuracy). HONEST NOTE: `mask-weight=0.1` is a starting weight; if the containment term dominates and MPJPE regresses, lower it (0.05/0.02) and note the value used. Flip via the dense swap is only valid because T5's swap is a verified involution — if a run errors in `build_dense_lr_swap`'s involution assert, that is the swap, not the trainer (fix in T5, do not disable flip silently).

- [ ] **Step 4: Assert the gated checkpoint exists (the "pytest" for T6)**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -c "
import os
RUN='/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit350_gated'
assert os.path.isdir(RUN+'/ckpt'), 'no ckpt dir -- fine-tune not started/resumed'
assert os.path.exists(RUN+'/final'), 'no final checkpoint -- fine-tune not finished'
# restore it to confirm it is a real 350-output ViTPose
import orbax.checkpoint as ocp
from flax import nnx
from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
m = ViTPose(ViTPoseConfig(num_keypoints=350), rngs=nnx.Rngs(0))
g, st = nnx.split(m); st = ocp.StandardCheckpointer().restore(RUN+'/final', st)
print('OK gated checkpoint restores: 350-output ViTPose at', RUN+'/final')
"
```
Expected: `OK gated checkpoint restores...`. SUCCESS CRITERION: `cse_vit350_gated/final` is a restorable 350-output `ViTPose`, and the slurm/`train.log` shows finite loss + non-regressed val MPJPE. (The ablation numbers are T7.)

---

## Task 7: gated-vs-ungated ablation / eval (real-data-gated) — the payoff

**PREREQUISITE:** Task 6's `cse_vit350_gated/final` exists; the ungated arm is `cse_vit350_wings/final`. This is the actual deliverable of Phase 5. NOT a pytest for the heavy eval — a committed eval script + a real-data-gated existence/summary test.

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/eval_gating_ablation.py` **(committed)** — the ablation runner + viz.
- Test: `third_party/jarvis_jax/tests/test_gating_ablation.py` **(new)** — the eval-summary-exists gate + a small on-CPU logic test of the off-fly-mass metric.

**Interfaces:**
- Consumes: `CSEImageDataset(ROOT, "val", cse_labels_val_M300.npz)`; `ViTPose` restore (both checkpoints); `mask_containment(pred, mask224, dilate=k)` (T2) for the off-fly-mass metric; `gate_heatmaps` (T3) for the gated-decode metric; `heatmaps_to_keypoints` + `mpjpe` (accuracy); `dilate_mask_jax` (mask224).
- Produces:
  - `eval_gating_ablation.py` → a summary `.npz`/`.json` at `RUNS/cse_vit350_gated/ablation.{json,npz}` with, for EACH arm (`ungated_ckpt`, `gated_ckpt`) and each decode mode (`raw`, `gated_decode`):
    - `mpjpe_px` (kp channels 0..49, over val), `pck@10px` / `pck@5px`;
    - `dense_mpjpe_px` (channels 50..349, visible verts);
    - **headline** `off_fly_mass_frac` = mean over val of `mask_containment(pred, mask224, dilate=k)` (the fraction of predicted positive heatmap mass outside the dilated mask — lower is better);
    - a handful of qualitative overlay PNGs (ungated vs gated predictions on the fly image + SAM mask) under `RUNS/cse_vit350_gated/viz/`.
  - The comparison table printed + saved. Success/failure judged honestly (below).

- [ ] **Step 1: Write the CPU logic test + the real-data-gated summary test**

Create `third_party/jarvis_jax/tests/test_gating_ablation.py`:

```python
import os
import json
import jax.numpy as jnp
import numpy as np
import pytest

RUNS = "/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs"
GATED = f"{RUNS}/cse_vit350_gated/final"
UNGATED = f"{RUNS}/cse_vit350_wings/final"
SUMMARY = f"{RUNS}/cse_vit350_gated/ablation.json"


def test_off_fly_mass_metric_matches_mask_containment():
    """The headline metric IS mean mask_containment over the batch (dilated)."""
    from jarvis_jax.densepose.eval_gating_ablation import off_fly_mass_frac
    from jarvis_jax.train.losses import mask_containment
    pred = jnp.zeros((1, 8, 8, 1)).at[0, 6, 6, 0].set(1.0)   # peak outside
    mask = jnp.zeros((1, 8, 8)).at[0, 1, 1].set(1.0)
    assert abs(off_fly_mass_frac(pred, mask, dilate=0)
               - float(mask_containment(pred, mask, dilate=0))) < 1e-6
    # dilating to reach the peak drops the metric toward 0.
    assert off_fly_mass_frac(pred, mask, dilate=0) > off_fly_mass_frac(
        jnp.zeros((1, 8, 8, 1)).at[0, 2, 2, 0].set(1.0), mask, dilate=3)


@pytest.mark.skipif(not (os.path.exists(GATED) and os.path.exists(UNGATED)),
                    reason="run T6 gated fine-tune first (need both checkpoints)")
def test_ablation_summary_exists_and_is_well_formed():
    if not os.path.exists(SUMMARY):
        pytest.skip("run eval_gating_ablation.py first to produce ablation.json")
    d = json.load(open(SUMMARY))
    for arm in ("ungated_ckpt", "gated_ckpt"):
        assert arm in d
        for mode in ("raw", "gated_decode"):
            m = d[arm][mode]
            assert np.isfinite(m["mpjpe_px"]) and m["mpjpe_px"] > 0
            assert 0.0 <= m["off_fly_mass_frac"] <= 1.0
    # HEADLINE assertion (honest): the GATED checkpoint (raw decode) puts LESS mass
    # off-fly than the UNGATED checkpoint (raw decode). This is the payoff of the
    # containment loss. If it does NOT hold, the test FAILS loudly -> report it.
    assert (d["gated_ckpt"]["raw"]["off_fly_mass_frac"]
            <= d["ungated_ckpt"]["raw"]["off_fly_mass_frac"] + 1e-4), (
        "gated fine-tune did NOT reduce off-fly mass -- report this honestly")
```

- [ ] **Step 2: Run the CPU logic test (fails until the module exists)**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_gating_ablation.py::test_off_fly_mass_metric_matches_mask_containment -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.densepose.eval_gating_ablation'`.

- [ ] **Step 3: Write the eval/ablation script**

Create `third_party/jarvis_jax/jarvis_jax/cse/eval_gating_ablation.py`:

```python
"""Phase-5 ablation: GATED vs UNGATED cse_vit350 on val.

Metrics per (checkpoint arm x decode mode):
  * mpjpe_px         : kp channels 0..49
  * pck@10 / pck@5   : kp PCK
  * dense_mpjpe_px   : dense vertex channels 50..349 (visible verts)
  * off_fly_mass_frac: HEADLINE -- mean fraction of predicted positive heatmap mass
                       OUTSIDE the dilated SAM mask (lower is better; the containment
                       loss target). Computed with mask_containment(dilate=k).
Also writes qualitative overlays. Honest reporting: if gating does not reduce
off-fly mass / improve accuracy, the numbers say so.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def off_fly_mass_frac(pred, mask224, dilate=0):
    """Mean fraction of positive predicted heatmap mass outside the dilated mask."""
    from jarvis_jax.train.losses import mask_containment
    return float(mask_containment(pred, mask224, dilate=dilate))


def _pck(pred_kp, gt_kp, vis, thresh):
    import numpy as np
    d = np.linalg.norm(np.asarray(pred_kp) - np.asarray(gt_kp), axis=-1)  # (B,K)
    v = np.asarray(vis).astype(bool)
    return float((d[v] <= thresh).mean()) if v.any() else float("nan")


def _eval_arm(ckpt, ds, batch, num_joints, dilate, gate_decode, in_size=448):
    import numpy as np, jax.numpy as jnp, orbax.checkpoint as ocp
    from flax import nnx
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.data.device import normalize_image
    from jarvis_jax.data.v3 import batches
    from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints, mpjpe
    from jarvis_jax.densepose.gating import gate_heatmaps
    import jax

    m = ViTPose(ViTPoseConfig(num_keypoints=num_joints), rngs=nnx.Rngs(0))
    g, st = nnx.split(m); m = nnx.merge(g, ocp.StandardCheckpointer().restore(ckpt, st)); m.eval()
    scale = in_size / float(ds.heatmap_size)

    off_num = off_den = 0.0
    kp_err_sum = kp_n = 0.0
    dv_err_sum = dv_n = 0.0
    pck10_hits = pck5_hits = pck_n = 0.0
    for img4_u8, kp_xy, vis in batches(ds, batch, shuffle=False, drop_last=False):
        img = normalize_image(jnp.asarray(img4_u8))
        pred = m(img, use_running_average=True)                       # (B,224,224,J)
        mask = img[..., 3]
        mask224 = jax.image.resize(mask, (mask.shape[0], pred.shape[1], pred.shape[2]),
                                   method="nearest")
        # headline off-fly mass (weighted by batch size)
        off_num += off_fly_mass_frac(pred, mask224, dilate=dilate) * img4_u8.shape[0]
        off_den += img4_u8.shape[0]
        dec = gate_heatmaps(pred, mask, dilate=dilate) if gate_decode else pred
        pk = np.asarray(heatmaps_to_keypoints(dec, in_size=in_size))  # (B,J,2)
        gk = np.asarray(kp_xy) * scale
        vb = np.asarray(vis).astype(bool)
        # keypoints 0..49
        kp_err_sum += float(mpjpe(jnp.asarray(pk[:, :50]), jnp.asarray(gk[:, :50]),
                                  jnp.asarray(vb[:, :50]))) * int(vb[:, :50].sum())
        kp_n += int(vb[:, :50].sum())
        # dense verts 50..
        dv_err_sum += float(mpjpe(jnp.asarray(pk[:, 50:]), jnp.asarray(gk[:, 50:]),
                                  jnp.asarray(vb[:, 50:]))) * int(vb[:, 50:].sum())
        dv_n += int(vb[:, 50:].sum())
        pck10_hits += _pck(pk[:, :50], gk[:, :50], vb[:, :50], 10.0) * int(vb[:, :50].sum())
        pck5_hits += _pck(pk[:, :50], gk[:, :50], vb[:, :50], 5.0) * int(vb[:, :50].sum())
        pck_n += int(vb[:, :50].sum())
    return {
        "mpjpe_px": kp_err_sum / max(kp_n, 1),
        "dense_mpjpe_px": dv_err_sum / max(dv_n, 1),
        "off_fly_mass_frac": off_num / max(off_den, 1),
        "pck@10px": pck10_hits / max(pck_n, 1),
        "pck@5px": pck5_hits / max(pck_n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3")
    ap.add_argument("--aux-val", default="/gscratch/portia/eabe/data/Johnson_lab/cse_work/cse_labels_val_M300.npz")
    ap.add_argument("--ungated-ckpt", required=True)
    ap.add_argument("--gated-ckpt", required=True)
    ap.add_argument("--num-joints", type=int, default=350)
    ap.add_argument("--dilate", type=int, default=11)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from jarvis_jax.densepose.cse_dataset import CSEImageDataset
    ds = CSEImageDataset(a.root, "val", a.aux_val)
    summary = {}
    for arm, ckpt in (("ungated_ckpt", a.ungated_ckpt), ("gated_ckpt", a.gated_ckpt)):
        summary[arm] = {
            "raw": _eval_arm(ckpt, ds, a.batch, a.num_joints, a.dilate, gate_decode=False),
            "gated_decode": _eval_arm(ckpt, ds, a.batch, a.num_joints, a.dilate, gate_decode=True),
        }
        print(arm, json.dumps(summary[arm], indent=2))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=2)
    print(f"[ablation] wrote {a.out}")


if __name__ == "__main__":
    main()
```

Run the CPU logic test now:
```bash
cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_gating_ablation.py::test_off_fly_mass_metric_matches_mask_containment -v
```
Expected: PASS.

- [ ] **Step 4: Run the ablation on val (coordinator, GPU/background), then the summary gate**

```bash
source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
python -u -m jarvis_jax.densepose.eval_gating_ablation \
  --ungated-ckpt /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit350_wings/final \
  --gated-ckpt   /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit350_gated/final \
  --num-joints 350 --dilate 11 --batch 8 \
  --out /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit350_gated/ablation.json \
  > /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_vit350_gated/ablation.log 2>&1
```
Run via background bash (val = 1365 imgs x 2 arms x 2 decode modes; may exceed 600s). Then:
```bash
cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_gating_ablation.py -v
```
Expected: the CPU logic test PASSES; `test_ablation_summary_exists_and_is_well_formed` PASSES once `ablation.json` exists AND the gated arm's off-fly mass ≤ the ungated arm's (the headline win). If the headline assertion FAILS, that is a REAL negative result — report the numbers honestly (see below), do not weaken the assertion to make it green.

- [ ] **Step 5: Interpret + report (honest success criterion)**

Read `ablation.json`. The success story is:
1. **Off-fly heatmap mass DROPS** for the gated checkpoint vs the ungated one (raw decode) — the containment loss taught the model to keep mass on the fly. This is the headline.
2. **kp MPJPE / PCK does NOT regress** (ideally improves — fewer stray off-fly peaks means fewer catastrophic argmax misses).
3. **Gated DECODE** (`gate_heatmaps`) further reduces off-fly mass at inference even for the ungated checkpoint (free robustness), and does not hurt on-fly accuracy.
4. **dense_mpjpe** for wing verts: check whether gating + flip aug improved wing-vertex localization / spread (spec §2's wing under-spread is an image-evidence limit, so a large win here is NOT expected — a small improvement or neutrality is the honest expectation; report what happens).

HONEST FALLBACK: if gating is neutral or slightly hurts kp accuracy while reducing off-fly mass, report the tradeoff and recommend a smaller `mask_weight` (re-run T6 at 0.05/0.02). If flip aug regressed accuracy (e.g. the dense swap tolerates a few imperfect vertex pairs), report it and consider flip on the 50 kp only. Do not claim a win the numbers do not show.

- [ ] **Step 6: Commit (script + test only; NOT the checkpoints/summary/viz)**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/eval_gating_ablation.py third_party/jarvis_jax/tests/test_gating_ablation.py
git commit -m "feat(cse): Phase-5 gated-vs-ungated ablation (off-fly mass + MPJPE/PCK/dense) + test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Do NOT `git add` `jax_vitpose_runs/cse_vit350_gated/` (checkpoints, ablation.json, viz).

---

## Self-Review

### Spec-coverage check (each Phase-5 spec §7 / decision-6 / brief requirement → task)

| Phase-5 spec §7 / decision-6 / brief requirement | Task |
|---|---|
| §7 "Dilated-mask gating: kp/dense heatmaps gated by the dilated SAM mask (inference)" | **Task 3** (`gate_heatmaps` + predict wire) — built on **Task 1** dilation |
| §7 "+ a training penalty for mass outside — predictions must land on the fly" | **Task 2** (`mask_containment(dilate=k)` + `TrainConfig.mask_dilate` + train wiring) — built on **Task 1** |
| Dilation via `jax.lax.reduce_window` max-pool (mirror PyTorch `F.max_pool2d`) | **Task 1** (`dilate_mask_jax`) |
| §7 "Pluggable backbone: Phase 5 uses the existing JAX ViT-B (reuse ViTPose weights)" | **Task 4** (`Backbone` protocol/registry; `ViTPose(backbone=...)` default ViT; param paths unchanged) |
| §7 "optional later port of the SAM3 ViTDet trunk" = OUT OF SCOPE (non-goal, §10) | **Task 4** builds the abstraction only; explicitly NOT ported (Global Constraints) |
| §7 "Dense pose is an observation source, never the final output" (dense channels retained/trained) | **Task 5** (dense flip aug) + **Task 6/7** (dense channels 50..349 fine-tuned + evaluated `dense_mpjpe`) |
| Brief: enable augmentation for the dense channels (flip currently disabled) | **Task 5** (`build_dense_lr_swap`) + **Task 6** (`--flip-p>0` wiring) |
| Brief: no mask-prediction head (mask is input + gate only) | Global Constraints; no head added anywhere (Tasks 2/3 use the mask as loss target + gate) |
| Brief: data = red_data_unified_V3, M=300; fine-tune from cse_vit350 (not scratch) | **Task 6** (`--v3-ckpt cse_vit350_wings/final --warmstart-joints 350`, M300 aux) |
| Brief: validate by fine-tuning with gating ON + ablating gating on/off | **Task 6** (gated fine-tune) + **Task 7** (gated-vs-ungated ablation) |
| Brief: mask pixel-aligned to crop; dilate POST-augmentation | **Task 2** (dilates `img[...,3]`, the re-binarized post-aug mask channel) + Global Constraints |
| Reuse ViT/ViTPose/decoder/train stack unmodified except additive changes; NNX+optax+sharding not pmap | Global Constraints; all tasks additive (Task 4 keeps default path + param paths; Task 2 keeps `dilate=0` identity) |
| CPU unit tests (JAX_PLATFORMS=cpu); GPU on-node unset LD_LIBRARY_PATH + MEM_FRACTION=0.9; coordinator runs GPU | Global Constraints; **T1–T5** CPU, **T6–T7** GPU coordinator-run |

**Gaps (spec §7 requirements NOT mapped to a Phase-5 task — intentional, per user decisions):**
- **SAM3 ViTDet trunk port + mask distillation** (§7 "optional later port … with PyTorch SAM3 masks as a distillation target"): explicitly OUT OF SCOPE (user decision 1; spec §10 non-goal). Task 4 provides only the pluggable-backbone hook a future port would register into. **This is the one §7 clause deliberately not implemented.**
- **Feeding dense observations into the IK** (§11 Phase-5 line "feed dense observations"): the dense channels are trained + evaluated here (T6/T7), but consuming them as IK observations is Phase-2/3 IK-integration territory (the `stac_core_jaxls` marker cost already accepts 50+M markers via `CSEFramesetDataset`); Phase 5's scope per the brief is the DETECTOR (gating + backbone + aug), not re-plumbing the IK. Noted, not a code task here.

### Placeholder scan
- No `TODO`/`TBD`/`implement later`/`...`-as-code tokens. Every code step has runnable code; every run step has an exact command + expected output.
- No unresolved path placeholders: `ROOT`, `WORK`, `RUNS`, `MESH`, the `cse_vit350_wings/final` warm-start source, and the `cse_labels_{train,val}_M300.npz` aux files are concrete absolute paths (verified: mesh keys + `fps_300` inspected; warm-start ckpt + M300 labels named in the brief).
- One flagged implementation-time check (T6 Step 1): the exact accessor for the 50 canonical STAC names in `cse_labels.py` (`model_kp_order` vs another name) is to be confirmed with a `grep` at implementation time (fallback: read `stac-mjx/configs/anatomy/v1.yaml` `model.KEYPOINT_MODEL_PAIRS` keys, the verified source of the 50 names). Called out explicitly, not hidden.
- Honest limitations surfaced rather than papered over: (a) `sym_index` is NOT usable for the dense swap (not FPS-closed / not an involution) — T5 uses the verified segment-aware mutual-NN Y-reflection instead; (b) T6 `mask_weight=0.1` is a starting value with a documented back-off (0.05/0.02) if it regresses MPJPE; (c) T7's headline assertion can genuinely FAIL (gating neutral/negative) — the plan says report it, do not weaken the test; (d) wing under-spread is an image-evidence limit (spec §2) so a big `dense_mpjpe` win is NOT promised.

### Type-consistency check across tasks
- **`dilate_mask_jax(mask, k) -> array`** (T1) — `mask` shape `(H,W)|(B,H,W)|(B,H,W,C)`, `k` static int, returns same shape/dtype, `k<=1` identity. Consumed by: **T2** `mask_containment(pred, mask224, dilate=k)` (passes `mask224` `(B,H,W)`); **T3** `gate_heatmaps` (passes `mask` `(B,Hm,Wm)`); **T7** `off_fly_mass_frac` (via `mask_containment`). Every caller passes a 2/3/4-D float mask + static int — matches the signature.
- **`mask_containment(pred (B,H,W,K), mask224 (B,H,W), *, dilate=0, eps=1e-6) -> scalar`** (T2) — consumed by `make_train_step` (T2, `dilate=mask_dilate`), by `eval_gating_ablation.off_fly_mass_frac` (T7), and the CPU logic test (T7). `dilate=0` == pre-Phase-5 behaviour is asserted in T2 and re-used in T7.
- **`gate_heatmaps(hm (B,H,W,K), mask (B,Hm,Wm), dilate=21) -> (B,H,W,K)`** (T3) — consumed by `predict_full.py` (T3, passing `crops[...,3]`) and `eval_gating_ablation._eval_arm` (T7, `gate_decode=True`, passing `img[...,3]`). Both pass the 4th input channel as the mask — consistent with the SAM-mask-is-channel-3 invariant.
- **`build_backbone(name_or_module, cfg, *, rngs) -> nnx.Module`** + the `Backbone` protocol (T4) — consumed by `ViTPose.__init__(backbone=...)` (T4). Default `backbone=None` → `ViT(cfg)` keeps `self.backbone` (attribute name) → checkpoint param paths `backbone/...` unchanged → `warm_start_from_v3` / Orbax restore of the existing `cse_vit350` checkpoints (T6) work byte-for-byte. `T5_param_paths` test asserts `ViTPose(cfg)` and `ViTPose(cfg, backbone="vit")` produce identical param trees.
- **`build_dense_lr_swap(mesh_npz, fps_key, base_names, *, tol=0.1) -> int32 (50+M,)`** (T5) — shape `(50+M,)`, an involution, first 50 == `build_lr_swap(base_names)`, dense block indexes only `[50, 50+M)`. Consumed by **T6**'s trainer as `lr_swap` and passed straight into `augment_batch(..., lr_swap, ...)` → `flip_batch` (which does `kp[:, lr_swap]` / `vis[:, lr_swap]` on the `(50+M)`-channel labels). The `(50+M)` length matches `CSEImageDataset`'s `num_joints = 50 + M` and `ViTPoseConfig(num_keypoints=350)`. T5's `test_flipping_a_synthetic_example` exercises the exact `kp[swap]` indexing `flip_batch` uses.
- **Checkpoint arity**: T6 produces `cse_vit350_gated/final` = a 350-output `ViTPose` Orbax `StandardCheckpointer` state (same format as `cse_vit350_wings/final`); T7 restores BOTH with the identical `ViTPose(ViTPoseConfig(num_keypoints=350))` + `StandardCheckpointer().restore` path — consistent producer/consumer.
- **`TrainConfig`**: gains `mask_dilate: int = 0` (T2); `train_keypoints_cse_full` passes `mask_weight=a.mask_weight, mask_dilate=a.mask_dilate` (T6). Default `0` everywhere keeps every non-Phase-5 training run identical.
