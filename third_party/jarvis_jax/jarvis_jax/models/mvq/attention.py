"""cuDNN flash-attention wrapper for the backbone and decoder.

`jax.nn.dot_product_attention(..., implementation="cudnn")` needs bf16/fp16
inputs, head_dim a multiple of 8 (<=128 on compute capability 8.9), and an
EVEN token count under training -- even with no mask at all (measured:
T=789 fails, T=790 passes; see docs/benchmark/2026-09-mvq/ notes). This
helper pads the token axis of q and of k/v to even length unconditionally,
builds the boolean key mask when `key_valid` is given, calls the cudnn
kernel, and slices the padding back off.

Layout is (B, N, heads, hd) ("BTNH") -- the shape `jax.nn.dot_product_attention`
expects -- NOT the (B, heads, N, hd) layout `dinov3.py`'s explicit path uses
internally, nor the flat (B, N, D) layout `fusion.py::masked_attention` takes.
Callers transpose/reshape into BTNH before calling and back out after.

Caveat: when `key_valid` is None (the backbone's "all tokens valid" case) and
the token count is odd, the one zero-valued pad key is NOT masked out of the
softmax -- it contributes a neutral (all-zero) key/value pair with no `mask`
argument to suppress it. At the backbone's real N (~789 prefix+patch tokens)
this is a ~1/N nudge, judged negligible against the 2x speed cost of a mask
(4.7 vs 9.6 ms/iter, see the brief); it is NOT negligible at small N, which
is why the CPU test below only checks this path against an explicit
reference at an ODD count with a MASK supplied (all tokens marked valid),
not with `key_valid=None`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _pad_even_tokens(x):
    """Pad axis 1 (tokens) with one zero row if its length is odd.

    Returns (padded_x, original_length).
    """
    n = x.shape[1]
    if n % 2 == 0:
        return x, n
    pad_width = [(0, 0)] * x.ndim
    pad_width[1] = (0, 1)
    return jnp.pad(x, pad_width), n


def flash_attention(q, k, v, key_valid=None, *, _impl: str = "cudnn"):
    """q (B,Tq,heads,hd), k/v (B,Tk,heads,hd), key_valid (B,Tk) bool or None
    -> (B,Tq,heads,hd) fp32.

    Pads the token axis of q and of k/v to an even length (always, per the
    even-length rule above), builds a (B,1,Tq_pad,Tk_pad) boolean mask from
    `key_valid` when given (padded key columns False, padded query rows True
    -- those rows are sliced off below, but must stay unmasked or an
    all-False row softmaxes to NaN), casts to bf16 (the cudnn kernel's
    required dtype) and calls
    `jax.nn.dot_product_attention(..., mask=mask, implementation=_impl)`,
    slices back to the input token counts, and returns fp32.

    `_impl` is a private escape hatch (default "cudnn") so CPU tests can pass
    `_impl="xla"` to exercise the padding/mask construction above without a
    GPU. The bf16 cast is skipped for `_impl="xla"`: that path exists to
    isolate the padding/mask logic to fp32 precision (1e-5 parity against the
    explicit path), not to re-test bf16 rounding, which the GPU parity test
    covers directly on the real cudnn kernel. On a host where the cudnn
    kernel is unavailable, the default call raises `NotImplementedError` from
    JAX; this wraps it in a `RuntimeError` naming `attn_impl` so the failure
    is legible from model config, not a bare JAX internals trace.
    """
    q_pad, Tq = _pad_even_tokens(q)
    k_pad, Tk = _pad_even_tokens(k)
    v_pad, _ = _pad_even_tokens(v)
    Tq_p, Tk_p = q_pad.shape[1], k_pad.shape[1]

    mask = None
    if key_valid is not None:
        B = key_valid.shape[0]
        kv = jnp.pad(key_valid, ((0, 0), (0, Tk_p - Tk)), constant_values=False)
        qv = jnp.pad(jnp.ones((B, Tq), bool), ((0, 0), (0, Tq_p - Tq)), constant_values=True)
        mask = qv[:, None, :, None] & kv[:, None, None, :]                       # (B,1,Tq_p,Tk_p)

    if _impl == "cudnn":
        bf = jnp.bfloat16
        q_pad, k_pad, v_pad = q_pad.astype(bf), k_pad.astype(bf), v_pad.astype(bf)
    try:
        out = jax.nn.dot_product_attention(
            q_pad, k_pad, v_pad, mask=mask, implementation=_impl,
        )
    except NotImplementedError as e:
        raise RuntimeError(
            "flash_attention: cuDNN attention is unavailable on this host "
            "(no compatible GPU, or an unsupported shape/dtype). Set the "
            "model config's attn_impl to 'xla' to use the explicit path "
            f"instead. Original error: {e}"
        ) from e
    return out[:, :Tq].astype(jnp.float32)
