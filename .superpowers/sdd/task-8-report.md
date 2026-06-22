### Task 8 Report: 3D training loop (`train/train_3d.py`)

**Status:** DONE — both tests pass; no regressions.

**Commit:** `1b87d91` — `feat(jax-C): 3D training loop (frozen ViTPose + v2vNet), 3D-MPJPE eval, checkpoint resume`

---

#### Freeze mechanism

Two-part freeze, both required:

1. **stop_gradient on heatmaps** (`train_3d.py:loss_fn`): after `model.predict_heatmaps()` produces `(B, nc, 224, 224, J)` float32 tensors, `jax.lax.stop_gradient(hm)` is applied before the reprojection → v2vNet path. This prevents any gradient from flowing back through the 86M-param ViTPose backbone, even if the optimizer were to somehow include those params.

2. **Path-based wrt-filter restricts optimizer to v2vnet subtree**: `_is_v2vnet_param(path, var)` checks `isinstance(var, nnx.Param) and path[0] == "v2vnet"`. This filter is passed as `wrt=` to both `nnx.Optimizer(model, tx, wrt=_is_v2vnet_param)` and `nnx.value_and_grad(..., argnums=DiffState(0, _is_v2vnet_param))`. The nnx path format inside these transforms is a flat tuple of attribute strings, e.g. `('v2vnet', 'encoder_decoder', ...)`, NOT the `jax.tree_util.keystr` bracket format `"['v2vnet']['encoder_decoder']..."` — a critical distinction.

**Freeze test result (CPU):**
- `vitpose_delta = 0.000000e+00` (ViTPose params unchanged)
- `v2vnet_delta = 2.607703e-08` (v2vNet params updated)

---

#### Loss trajectory (8-step smoke gate, GPU, batch_size=2)

| Step | Loss |
|------|------|
| 1    | 0.39309 (first_loss) |
| 8    | 0.31204 (final_loss) |

Loss decreased 20.6% over 8 steps. Val 3D MPJPE = **12.239 mm** (world units).

---

#### Full test suite

`56 passed, 3 skipped, 0 failed` (598 s, 8 GPU)

Skips are pre-existing GPU/data/checkpoint guards in other test files.

---

#### Fix note (2026-06-22): wire graph-Laplacian skeleton edges + train_3d cleanup

**Skeleton edge count:** 44 edges (from V3 COCO `keypoint_names` × `skeleton` fields).

**Changes made:**

1. `jarvis_jax/train/losses_3d.py` — `build_skeleton_edges`: added dict-format support (`keypointA`/`keypointB` keys) alongside existing tuple/list format; the V3 COCO `skeleton` entries are dicts, so the prior was silently producing 0 edges before.

2. `jarvis_jax/data/v3_3d.py` — `V3FramesetDataset.__init__`: added `self.keypoint_names` and `self.skeleton` attributes from the COCO json (raw lists).

3. `jarvis_jax/train/train_3d.py`:
   - Replaced hardcoded empty `ei/ej` with `build_skeleton_edges(train_ds.keypoint_names, train_ds.skeleton)`.
   - Added `print(f"skeleton edges wired: {len(ei)} ...")` in `run_training_3d`.
   - Fixed `laplacian_weight` default from `1.0` to `0.0` (prior stays OFF by default; enable via `--laplacian-weight`).
   - Moved inline imports (`reproject_heatmaps`, `soft_argmax_3d`) from inside `loss_fn` to top-of-file.
   - Added sync comment in `loss_fn` noting parallel forward path with `HybridNet3D.__call__`.

4. `jarvis_jax/hybridnet/model.py` — `HybridNet3D.__call__`: added sync comment noting parallel forward path in `train_3d.py::loss_fn`.

**Test results (2026-06-22):** `6 passed` in `tests/test_train_3d.py tests/test_v3_3d.py` (167 s).
Smoke test output confirmed `skeleton edges wired: 44`.

---

#### Files created

- `third_party/jarvis_jax/jarvis_jax/train/train_3d.py` — `HybridNetConfig`, `make_v2v_optimizer`, `make_train_step_3d`, `eval_mpjpe_3d`, `run_training_3d`, `main()`
- `third_party/jarvis_jax/tests/test_train_3d.py` — freeze test (CPU) + smoke gate (GPU)
