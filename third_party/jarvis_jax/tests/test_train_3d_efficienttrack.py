"""Tests for the EfficientTrack + fusion-aware 3D training loop (Task 8).

Extends the existing `train_3d.py` machinery (`make_v2v_optimizer`,
`make_train_step_3d`) with a `mode` in {'3D_only', 'all'} and optional
`masks`/fusion-mode support, rather than duplicating it under new names (see
task-8-brief.md's reconciliation note).

Tests (all CPU, tiny synthetic model -- no real checkpoints needed):
  (a) test_one_train_step_decreases_loss_3d_only -- mode='3D_only': loss goes
      down over a few steps; front-end params stay frozen.
  (b) test_all_mode_trains_front_end -- mode='all': the front-end's own
      trainable param changes after a step (freeze boundary lifted).
  (c) test_carve_fusion_smoke -- one step with masks + fusion_mode='carve'
      runs without error and yields a finite loss (Task 6/7 fusion wiring).

The front-end used here is a tiny custom nnx.Module (NOT ViTPose, NOT a real
EfficientTrack) implementing only the pluggable-front-end contract
(`predict_heatmaps(crops_float) -> (B, num_cam, 224, 224, J)`), matching the
non-ViTPose dispatch branch in `HybridNet3D.predict_heatmaps` (the same
branch a real `EfficientTrack` goes through -- see
`jarvis_jax/hybridnet/model.py`'s `_is_vitpose` check). This keeps the test
CPU-fast (<10s) while genuinely exercising the generic/pluggable front-end
code path exercised by `make_train_step_3d`/`make_v2v_optimizer`; full
EfficientTrack parity through this same non-ViTPose branch is separately
covered (fixture-gated) by `tests/test_hybridnet_efficienttrack_parity.py`.
"""
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# ---------------------------------------------------------------------------
# Tiny synthetic front-end: NOT ViTPose (isinstance check in HybridNet3D
# routes it through the generic/pluggable non-ViTPose branch, same as a real
# EfficientTrack), with exactly one trainable nnx.Param so tests can directly
# check whether gradients/updates reach it.
# ---------------------------------------------------------------------------

class TinyFrontEnd(nnx.Module):
    """Cheap stand-in 2-D front-end: a single learnable (224, 224, J) bias,
    broadcast over batch/camera, completely ignoring the input crops (so no
    heavy conv work is needed -- keeps the test CPU-fast) while still being
    genuinely trainable (the returned heatmap depends directly on the Param,
    so its gradient is well-defined and generically nonzero)."""

    def __init__(self, num_joints: int, *, rngs: nnx.Rngs):
        self.num_joints = num_joints
        self.bias = nnx.Param(
            jax.random.normal(rngs.params(), (224, 224, num_joints)) * 0.05)

    def predict_heatmaps(self, crops_float: jnp.ndarray) -> jnp.ndarray:
        # crops_float: (B, num_cam, H, W, C) float32 -- shape only used to
        # determine (B, num_cam); content intentionally ignored.
        B, num_cam = crops_float.shape[0], crops_float.shape[1]
        return jnp.broadcast_to(
            self.bias.value, (B, num_cam, 224, 224, self.num_joints))


class _Cfg:
    """Bare cfg object -- HybridNet3D reads all fields via getattr() with
    backward-compatible defaults, so only num_keypoints is required."""

    def __init__(self, num_keypoints: int, fusion_mode: str = "none"):
        self.num_keypoints = num_keypoints
        self.fusion_mode = fusion_mode
        self.gate_temperature = 1.0
        self.gate_floor = 0.0
        self.sharpen = 1.0


# ---------------------------------------------------------------------------
# Shared tiny synthetic batch builder
# ---------------------------------------------------------------------------

def _build_model_and_batch(fusion_mode: str = "none", seed: int = 0):
    from jarvis_jax.hybridnet.model import HybridNet3D
    from jarvis_jax.hybridnet.v2vnet import V2VNet

    J = 4
    B, nc = 2, 2
    cfg = _Cfg(num_keypoints=J, fusion_mode=fusion_mode)

    front = TinyFrontEnd(J, rngs=nnx.Rngs(seed))
    v2vnet = V2VNet(J, J, rngs=nnx.Rngs(seed + 1))
    model = HybridNet3D(front, v2vnet, cfg)

    rng = np.random.RandomState(seed + 100)
    batch = {
        "crops4": jnp.asarray(
            rng.randint(0, 255, (B, nc, 448, 448, 3), dtype=np.uint8)),
        "centerHM": jnp.zeros((B, nc, 2), dtype=jnp.float32),
        "center3D": jnp.zeros((B, 3), dtype=jnp.float32),
        "cameraMatrices": jnp.broadcast_to(
            jnp.eye(4, 3, dtype=jnp.float32)[None, None], (B, nc, 4, 3)),
        # GT keypoints well inside the 48-unit cube (|local| < 24).
        "kp3d": jnp.asarray(
            rng.uniform(-15.0, 15.0, size=(B, J, 3)).astype(np.float32)),
        "vis": jnp.ones((B, J), dtype=bool),
    }
    if fusion_mode == "carve":
        # masks at the padded heatmap resolution (226x226), per
        # HybridNet3D.__call__'s 'carve' docstring convention.
        batch["masks"] = jnp.asarray(
            rng.uniform(0.0, 1.0, size=(B, nc, 226, 226)).astype(np.float32))

    ei = np.zeros((0,), dtype=np.int32)
    ej = np.zeros((0,), dtype=np.int32)
    return model, batch, ei, ej


# ---------------------------------------------------------------------------
# (a) 3D_only: loss decreases, front-end frozen
# ---------------------------------------------------------------------------

def test_one_train_step_decreases_loss_3d_only():
    from jarvis_jax.train.train_3d import (
        HybridNetConfig, make_v2v_optimizer, make_train_step_3d,
    )

    model, batch, ei, ej = _build_model_and_batch(fusion_mode="none")

    front_before = np.array(model.front_end.bias.value)

    tcfg = HybridNetConfig(total_steps=10, batch_size=2, warmup_steps=0, lr=1e-2)
    opt = make_v2v_optimizer(model, tcfg, mode="3D_only")
    step = make_train_step_3d(
        laplacian_weight=0.0, ei=ei, ej=ej,
        grid_spacing=1, roi_cube=48, sigma=2.0, mode="3D_only",
    )

    l0 = float(step(model, opt, batch))
    assert np.isfinite(l0)
    loss = l0
    for _ in range(5):
        loss = float(step(model, opt, batch))

    assert loss < l0, f"loss did not decrease: first={l0:.6f} final={loss:.6f}"

    front_after = np.array(model.front_end.bias.value)
    delta = float(np.abs(front_after - front_before).max())
    assert delta == 0.0, (
        f"front-end param changed under mode='3D_only' (delta={delta:.2e}); "
        f"freeze boundary is broken.")


# ---------------------------------------------------------------------------
# (b) all: front-end trains too
# ---------------------------------------------------------------------------

def test_all_mode_trains_front_end():
    from jarvis_jax.train.train_3d import (
        HybridNetConfig, make_v2v_optimizer, make_train_step_3d,
    )

    model, batch, ei, ej = _build_model_and_batch(fusion_mode="none", seed=1)

    front_before = np.array(model.front_end.bias.value)
    v2v_before = np.array(model.v2vnet.front_basic.conv.kernel.value)

    tcfg = HybridNetConfig(total_steps=1, batch_size=2, warmup_steps=0, lr=1e-2)
    opt = make_v2v_optimizer(model, tcfg, mode="all")
    step = make_train_step_3d(
        laplacian_weight=0.0, ei=ei, ej=ej,
        grid_spacing=1, roi_cube=48, sigma=2.0, mode="all",
    )

    loss = float(step(model, opt, batch))
    assert np.isfinite(loss)

    front_after = np.array(model.front_end.bias.value)
    v2v_after = np.array(model.v2vnet.front_basic.conv.kernel.value)

    front_delta = float(np.abs(front_after - front_before).max())
    v2v_delta = float(np.abs(v2v_after - v2v_before).max())

    print(f"\nAll-mode: front_end_delta={front_delta:.6e}  v2vnet_delta={v2v_delta:.6e}")
    assert front_delta > 0.0, (
        "front-end param did NOT change under mode='all' -- freeze was not lifted.")
    assert v2v_delta > 0.0, "v2vnet param did NOT change under mode='all'."


def test_invalid_mode_rejected():
    from jarvis_jax.train.train_3d import HybridNetConfig, make_v2v_optimizer, make_train_step_3d

    model, _, ei, ej = _build_model_and_batch()
    tcfg = HybridNetConfig(total_steps=1, batch_size=2, warmup_steps=0)
    try:
        make_v2v_optimizer(model, tcfg, mode="bogus")
        assert False, "expected ValueError for invalid mode"
    except ValueError:
        pass
    try:
        make_train_step_3d(laplacian_weight=0.0, ei=ei, ej=ej, mode="bogus")
        assert False, "expected ValueError for invalid mode"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# (c) fusion smoke test: 'carve' + masks runs, finite loss
# ---------------------------------------------------------------------------

def test_carve_fusion_smoke():
    from jarvis_jax.train.train_3d import (
        HybridNetConfig, make_v2v_optimizer, make_train_step_3d,
    )

    model, batch, ei, ej = _build_model_and_batch(fusion_mode="carve", seed=2)
    assert "masks" in batch

    tcfg = HybridNetConfig(total_steps=1, batch_size=2, warmup_steps=0)
    opt = make_v2v_optimizer(model, tcfg, mode="3D_only")
    step = make_train_step_3d(
        laplacian_weight=0.0, ei=ei, ej=ej,
        grid_spacing=1, roi_cube=48, sigma=2.0, mode="3D_only",
    )

    loss = float(step(model, opt, batch))
    assert np.isfinite(loss), f"carve-fusion loss not finite: {loss}"


# ---------------------------------------------------------------------------
# (d) regression guard: fusion_mode='none' + no masks key is unaffected by
# the masks-plumbing change (loss identical whether or not batch.get("masks")
# is exercised, since predict_heatmaps/loss_fn never touch it when 'none').
# ---------------------------------------------------------------------------

def test_none_fusion_identical_with_or_without_masks_key():
    from jarvis_jax.train.train_3d import (
        HybridNetConfig, make_v2v_optimizer, make_train_step_3d,
    )

    model, batch, ei, ej = _build_model_and_batch(fusion_mode="none", seed=3)
    tcfg = HybridNetConfig(total_steps=1, batch_size=2, warmup_steps=0)

    # Run 1: no masks key at all.
    opt1 = make_v2v_optimizer(model, tcfg, mode="3D_only")
    step1 = make_train_step_3d(
        laplacian_weight=0.0, ei=ei, ej=ej, mode="3D_only")
    loss_no_masks = float(step1(model, opt1, batch))

    # Run 2 (fresh model copy): masks key present but fusion_mode='none' --
    # must be numerically identical since predict_heatmaps/loss_fn never read
    # masks unless fusion_mode is 'carve'/'input_mask'.
    model2, batch2, ei2, ej2 = _build_model_and_batch(fusion_mode="none", seed=3)
    batch2 = dict(batch2)
    batch2["masks"] = jnp.zeros((2, 2, 226, 226), dtype=jnp.float32)
    opt2 = make_v2v_optimizer(model2, tcfg, mode="3D_only")
    step2 = make_train_step_3d(
        laplacian_weight=0.0, ei=ei2, ej=ej2, mode="3D_only")
    loss_with_masks_key = float(step2(model2, opt2, batch2))

    assert loss_no_masks == loss_with_masks_key, (
        f"fusion_mode='none' behavior changed when a (unused) masks key is "
        f"present: {loss_no_masks} vs {loss_with_masks_key}")
