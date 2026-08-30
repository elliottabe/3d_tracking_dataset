"""Tests for the heatmap-cache trainer's gradient-clipping + NaN-guard
additions (train_3d_cached_hm.py). Added after A7_c2f_aug_femwt diverged to
NaN at step ~850/1500 under unclipped AdamW (see HMCachedConfig.grad_clip_norm's
docstring and the 2026-08-30 restart report for the full investigation).

No GPU needed: everything here operates on tiny synthetic pytrees / CPU-sized
nnx modules, not the real cache or V2VNet.

Tests:
  (a) test_disabled_clip_is_byte_identical_to_plain_adamw -- grad_clip_norm=None
      (or <=0) must build the EXACT same optax transform as the pre-clipping
      code (same opt_state tree structure + bit-identical updates across
      several steps), so existing/older checkpoints stay restorable and the
      six already-running arms stay comparable to anything trained later.
  (b) test_grad_clip_engages_on_large_gradient_and_is_noop_on_small -- the
      configured `_build_tx` chain must actually clip a synthetic gradient
      whose global norm exceeds HMCachedConfig's default threshold, and must
      leave a small gradient (norm well under threshold) completely
      unchanged before it reaches AdamW.
  (c) test_nan_guard_* -- `_check_nan_guard` must report loudly and return
      True the step a loss goes non-finite, be a no-op on a finite loss, and
      be a no-op when disabled even on a non-finite loss.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jarvis_jax.train.train_3d_cached_hm import (
    HMCachedConfig, _build_tx, _check_nan_guard,
)


# ---------------------------------------------------------------------------
# (a) Disabled clip == plain AdamW, byte-identical, several steps
# ---------------------------------------------------------------------------

def _reference_adamw(cfg: HMCachedConfig):
    """Reconstructs the schedule + AdamW exactly as `_build_tx` does, without
    going through the clip-aware branch at all -- the pre-clipping code."""
    decay_steps = max(cfg.total_steps, cfg.warmup_steps + 1)
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=cfg.lr, warmup_steps=cfg.warmup_steps,
        decay_steps=decay_steps, end_value=0.0)
    return optax.adamw(sched, weight_decay=cfg.weight_decay)


@pytest.mark.parametrize("grad_clip_norm", [None, 0.0, -1.0])
def test_disabled_clip_is_byte_identical_to_plain_adamw(grad_clip_norm):
    cfg = HMCachedConfig(total_steps=200, warmup_steps=10, lr=1e-3,
                          weight_decay=0.05, grad_clip_norm=grad_clip_norm)
    tx = _build_tx(cfg)
    ref_tx = _reference_adamw(cfg)

    params = {"w": jnp.array([1.0, -2.0, 3.0]), "b": jnp.array(0.5)}
    rng = np.random.RandomState(0)

    state = tx.init(params)
    ref_state = ref_tx.init(params)
    # Same opt_state PYTREE STRUCTURE -- required for Orbax `StandardRestore`
    # against a checkpoint saved by the pre-clipping code to keep working.
    assert jax.tree_util.tree_structure(state) == jax.tree_util.tree_structure(ref_state)

    for _ in range(5):
        grads = {"w": jnp.asarray(rng.randn(3).astype("float32")),
                 "b": jnp.asarray(rng.randn(1)[0].astype("float32"))}

        updates, state = tx.update(grads, state, params)
        ref_updates, ref_state = ref_tx.update(grads, ref_state, params)

        assert jnp.array_equal(updates["w"], ref_updates["w"])
        assert jnp.array_equal(updates["b"], ref_updates["b"])

        params = optax.apply_updates(params, updates)


# ---------------------------------------------------------------------------
# (b) Enabled clip: engages on a large synthetic gradient, no-op on a small one
# ---------------------------------------------------------------------------

def test_grad_clip_engages_on_large_gradient_and_is_noop_on_small():
    threshold = HMCachedConfig().grad_clip_norm   # the shipped default
    assert threshold is not None and threshold > 0

    cfg = HMCachedConfig(total_steps=200, warmup_steps=10, grad_clip_norm=threshold)
    tx = _build_tx(cfg)
    ref_tx = _reference_adamw(cfg)
    params = {"w": jnp.zeros((50,), jnp.float32)}
    state = tx.init(params)
    ref_state = ref_tx.init(params)

    # Warm up BOTH optimizers on identical small (well-under-threshold, so
    # `tx`'s clip stage is a no-op here) random gradients first -- AdamW's
    # bias-corrected first step is magnitude-invariant from a fresh zero
    # moment state (m_hat/sqrt(v_hat) cancels any uniform gradient rescale),
    # so comparing raw vs. clipped on step 1 alone can't show a difference.
    # A few steps of shared history gives both optimizers non-trivial,
    # IDENTICAL second-moment estimates to diverge from.
    rng = np.random.RandomState(1)
    for _ in range(5):
        g = {"w": jnp.asarray((rng.randn(50) * 0.05).astype("float32"))}
        _, state = tx.update(g, state, params)
        _, ref_state = ref_tx.update(g, ref_state, params)

    # --- Large gradient: global norm far exceeds the clip threshold. -------
    # Scaled off `threshold` itself (not a fixed constant) so this stays
    # comfortably over it regardless of where the default is tuned.
    large_grad = {"w": jnp.asarray((rng.randn(50) * threshold).astype("float32"))}
    assert float(optax.tree.norm(large_grad)) > threshold * 1.5

    clipped_updates, _ = tx.update(large_grad, state, params)
    ref_updates, _ = ref_tx.update(large_grad, ref_state, params)
    # Clipping actually changed what reaches AdamW -- the resulting update
    # must differ from what unclipped AdamW would have produced for the SAME
    # raw gradient (proves the clip stage is really wired in, not a no-op).
    assert not jnp.allclose(clipped_updates["w"], ref_updates["w"])

    # And directly: the clip sub-transform alone clamps this gradient's
    # global norm down to (approximately) the threshold.
    clip_only = optax.clip_by_global_norm(threshold)
    clip_state = clip_only.init(large_grad)
    clipped_grad, _ = clip_only.update(large_grad, clip_state)
    assert float(optax.tree.norm(clipped_grad)) == pytest.approx(threshold, rel=1e-4)

    # --- Small gradient: global norm well under the threshold. -------------
    small_grad = {"w": jnp.full((50,), 1e-4, jnp.float32)}
    assert float(optax.tree.norm(small_grad)) < threshold * 0.01
    clipped_small, _ = clip_only.update(small_grad, clip_state)
    # No-op: the gradient that reaches AdamW is untouched.
    assert jnp.array_equal(clipped_small["w"], small_grad["w"])

    # And end-to-end: a small gradient produces the IDENTICAL AdamW update
    # whether or not the clip stage sits in front of it.
    small_updates, _ = tx.update(small_grad, state, params)
    small_ref_updates, _ = ref_tx.update(small_grad, ref_state, params)
    assert jnp.array_equal(small_updates["w"], small_ref_updates["w"])


# ---------------------------------------------------------------------------
# (c) NaN guard
# ---------------------------------------------------------------------------

def test_nan_guard_triggers_and_reports_the_step(capsys):
    triggered = _check_nan_guard(float("nan"), step_idx=849, total_steps=1500, enabled=True)
    assert triggered is True
    out = capsys.readouterr().out
    assert "NON-FINITE LOSS" in out
    assert "850/1500" in out          # 1-indexed step number, matching training-loop logging


def test_nan_guard_triggers_on_inf_too():
    assert _check_nan_guard(float("inf"), step_idx=10, total_steps=100, enabled=True) is True


def test_nan_guard_is_noop_on_finite_loss(capsys):
    triggered = _check_nan_guard(0.02313, step_idx=799, total_steps=1500, enabled=True)
    assert triggered is False
    assert capsys.readouterr().out == ""


def test_nan_guard_disabled_is_noop_even_on_nan(capsys):
    triggered = _check_nan_guard(float("nan"), step_idx=849, total_steps=1500, enabled=False)
    assert triggered is False
    assert capsys.readouterr().out == ""
