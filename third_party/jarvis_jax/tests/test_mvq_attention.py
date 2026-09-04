"""CPU tests for `flash_attention`'s padding/mask construction (the cudnn
kernel itself needs a GPU -- see `@pytest.mark.gpu` parity tests below, and
the dinov3/mvq model tests for the attn_impl='cudnn' integration path).

Each CPU test passes `_impl="xla"` so `jax.nn.dot_product_attention` runs
its pure-XLA reference implementation (works on CPU, no bf16 cast) and
compares against an existing fp32 reference computation to a tight (1e-5)
tolerance -- this isolates flash_attention's OWN logic (even-length padding,
mask construction) from bf16-rounding, which is a separate, much looser
concern covered on GPU."""
import numpy as np
import jax, jax.numpy as jnp
import pytest


def test_matches_masked_attention_with_nontrivial_key_mask():
    """30 queries, 17 keys (both odd -> both axes get padded by one), a
    non-trivial key_valid mask -> must match fusion.masked_attention
    (q_chunk=None) to 1e-5: same math, same mask, different padding path."""
    from jarvis_jax.models.mvq.attention import flash_attention
    from jarvis_jax.models.mvq.fusion import masked_attention
    rng = np.random.default_rng(0)
    B, Nq, Nk, H, D = 2, 30, 17, 4, 32
    hd = D // H
    q = jnp.asarray(rng.normal(size=(B, Nq, D)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(B, Nk, D)).astype(np.float32))
    v = jnp.asarray(rng.normal(size=(B, Nk, D)).astype(np.float32))
    key_valid = jnp.asarray(rng.uniform(size=(B, Nk)) > 0.3)
    key_valid = key_valid.at[:, 0].set(True)          # keep >=1 valid key per row

    ref = masked_attention(q, k, v, key_valid, H, q_chunk=None)
    qh, kh, vh = (t.reshape(t.shape[0], t.shape[1], H, hd) for t in (q, k, v))
    out = flash_attention(qh, kh, vh, key_valid, _impl="xla").reshape(B, Nq, D)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-5)


def test_odd_token_count_backbone_shape_no_mask_is_exact():
    """13 tokens (odd, like a small backbone N), key_valid=None -- the REAL
    backbone call signature (no camera ever invalidates a patch token).
    Fix-round-1: this used to be an approximation (the pad key was left
    unmasked); it is now exact, excluded via cuDNN's native
    `key_value_seq_lengths` padding-mask (no bias tensor at all, see
    attention.py's module docstring), so it must match
    `masked_attention`'s all-True-mask reference to 1e-5 -- not a loose
    bound -- and must agree with the explicit-key_valid path too (both are
    mathematically the same computation via two different exclusion
    mechanisms)."""
    from jarvis_jax.models.mvq.attention import flash_attention
    from jarvis_jax.models.mvq.fusion import masked_attention
    rng = np.random.default_rng(1)
    B, T, H, D = 2, 13, 4, 32
    hd = D // H
    q = jnp.asarray(rng.normal(size=(B, T, D)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(B, T, D)).astype(np.float32))
    v = jnp.asarray(rng.normal(size=(B, T, D)).astype(np.float32))
    key_valid = jnp.ones((B, T), bool)

    ref = masked_attention(q, k, v, key_valid, H, q_chunk=None)
    qh, kh, vh = (t.reshape(B, T, H, hd) for t in (q, k, v))
    out_none = flash_attention(qh, kh, vh, None, _impl="xla").reshape(B, T, D)
    out_mask = flash_attention(qh, kh, vh, key_valid, _impl="xla").reshape(B, T, D)
    np.testing.assert_allclose(np.asarray(out_none), np.asarray(ref), atol=1e-5)
    np.testing.assert_allclose(np.asarray(out_mask), np.asarray(ref), atol=1e-5)
    # and the output shape has the padding sliced back off
    assert out_none.shape == (B, T, D)


def test_pads_odd_axes_to_even_internally():
    from jarvis_jax.models.mvq import attention as attn_mod
    x = jnp.zeros((2, 13, 4, 8))
    x_pad, n = attn_mod._pad_even_tokens(x)
    assert n == 13 and x_pad.shape[1] == 14
    x2 = jnp.zeros((2, 14, 4, 8))
    x2_pad, n2 = attn_mod._pad_even_tokens(x2)
    assert n2 == 14 and x2_pad.shape[1] == 14 and x2_pad is x2


def test_cudnn_missing_raises_clear_runtime_error():
    """On a host without a compatible GPU (this CPU test env), the default
    `_impl="cudnn"` call must fail with a RuntimeError that names attn_impl,
    not a bare JAX NotImplementedError."""
    from jarvis_jax.models.mvq.attention import flash_attention
    q = jnp.zeros((1, 4, 2, 8))
    with pytest.raises(RuntimeError, match="attn_impl"):
        flash_attention(q, q, q)


@pytest.mark.gpu
def test_cudnn_matches_explicit_and_gradients_agree():
    """cudnn vs an explicit fp32 reference at the backbone shape (2, 789, 12,
    64): outputs agree to 2e-2 (bf16 output scale) and d(loss)/d(q,k,v) are
    finite with cosine similarity > 0.99 to the explicit path's gradient."""
    from jarvis_jax.models.mvq.attention import flash_attention
    B, T, H, D = 2, 789, 12, 64
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.normal(size=(B, T, H, D)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(B, T, H, D)).astype(np.float32))
    v = jnp.asarray(rng.normal(size=(B, T, H, D)).astype(np.float32))

    def explicit(q, k, v):
        qh, kh, vh = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
        a = jax.nn.softmax((qh @ kh.transpose(0, 1, 3, 2)) * (D ** -0.5), axis=-1)
        return (a @ vh).transpose(0, 2, 1, 3)

    loss_ref = lambda q, k, v: jnp.sum(explicit(q, k, v) ** 2)
    loss_fa = lambda q, k, v: jnp.sum(flash_attention(q, k, v).astype(jnp.float32) ** 2)

    out_ref = explicit(q, k, v)
    out_fa = flash_attention(q, k, v)
    scale = float(jnp.abs(out_ref).max())
    assert float(jnp.abs(out_fa - out_ref).max()) < 2e-2 * max(scale, 1.0)

    g_ref = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
    g_fa = jax.grad(loss_fa, argnums=(0, 1, 2))(q, k, v)
    for name, gr, gf in zip("qkv", g_ref, g_fa):
        gr, gf = np.asarray(gr).ravel(), np.asarray(gf).ravel()
        assert np.isfinite(gf).all(), name
        cos = np.dot(gr, gf) / (np.linalg.norm(gr) * np.linalg.norm(gf) + 1e-12)
        assert cos > 0.99, (name, cos)
