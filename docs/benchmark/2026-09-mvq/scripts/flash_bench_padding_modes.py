"""Fix-round-1 empirical check (Task 9 review, Critical finding #1): does the
cudnn flash-attention kernel actually require an even T/S when NO mask/bias
is passed? Reading jax/_src/cudnn/fused_attention_stablehlo.py directly (this
env's installed version) shows the even-length raise at line ~392 is gated
`is_training and has_bias`, and `has_bias` comes from
`combine_bias_and_mask(bias, mask, dtype) is not None` -- i.e. only True when
a bias/bool-mask tensor is actually supplied. Passing `query_seq_lengths`/
`key_value_seq_lengths` alone (MaskType.PADDING) does NOT set a bias tensor
(mask stays None all the way through), so it should ALSO skip the even-length
check while still letting cudnn exclude the padded tail natively.

Runs under jax.grad (mirrors real training), bf16, at the shipped backbone
shape (2, 789, 12, 64) exactly as flash_bench2.py did:
  (a) mask=None, T=789 (odd), UNPADDED -- tests whether odd T ever needed
      padding at all when there's no bias.
  (b) T padded to 790, mask=None, query_seq_lengths=key_value_seq_lengths=789
      (MaskType.PADDING via seq-length args, no bias tensor) -- tests the
      "exclude the pad exactly, no bias" path.
  (c) T padded to 790, explicit bool key-mask (790 keys, last one False) --
      today's implemented path, has_bias=True, expected to need padding.
Each variant: does it run at all (NotImplementedError bucketed), and if so
its fwd+bwd wall time.
"""
import time, jax, jax.numpy as jnp
N, Hh, T, D = 56, 12, 789, 64
key = jax.random.PRNGKey(0); k1, k2, k3 = jax.random.split(key, 3)
q = jax.random.normal(k1, (N, T, Hh, D), jnp.float32)
k = jax.random.normal(k2, (N, T, Hh, D), jnp.float32)
v = jax.random.normal(k3, (N, T, Hh, D), jnp.float32)
bf = lambda x: x.astype(jnp.bfloat16)


def bench(name, fn, *args, reps=10):
    loss = lambda *a: jnp.sum(fn(*a).astype(jnp.float32) ** 2)
    g = jax.jit(jax.grad(loss))
    try:
        out = g(*args)
        jax.block_until_ready(out)
    except Exception as e:
        print(f"{name:45s} FAILED: {type(e).__name__}: {str(e)[:160]}")
        return False
    dev = jax.local_devices()[0]
    t0 = time.perf_counter()
    for _ in range(reps):
        out = g(*args)
    jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / reps
    print(f"{name:45s} PASSED  {dt*1000:8.2f} ms/iter  peak {dev.memory_stats()['peak_bytes_in_use']/2**30:6.2f} GiB")
    return True


# (a) mask=None, T=789 (odd), unpadded -- no bias at all
def variant_a(q, k, v):
    return jax.nn.dot_product_attention(q, k, v, mask=None, implementation="cudnn")


# (b) T padded to 790, mask=None, seq_lengths=789 (native PADDING, no bias)
Tp = T + 1
qp = jnp.pad(q, ((0, 0), (0, 1), (0, 0), (0, 0)))
kp = jnp.pad(k, ((0, 0), (0, 1), (0, 0), (0, 0)))
vp = jnp.pad(v, ((0, 0), (0, 1), (0, 0), (0, 0)))
seqlen = jnp.full((N,), T, dtype=jnp.int32)


def variant_b(q, k, v):
    return jax.nn.dot_product_attention(
        q, k, v, mask=None,
        query_seq_lengths=seqlen, key_value_seq_lengths=seqlen,
        implementation="cudnn",
    )


# (c) T padded to 790, explicit bool key-mask (today's implemented path)
mvalid = jnp.arange(Tp) < T                                    # (790,) last False
mask_bc = jnp.broadcast_to(mvalid[None, None, None, :], (N, 1, Tp, Tp))


def variant_c(q, k, v):
    return jax.nn.dot_product_attention(q, k, v, mask=mask_bc, implementation="cudnn")


print("=== fix-round-1 empirical check: bf16 (2,789,12,64)-style, under jax.grad ===")
ok_a = bench("(a) mask=None T=789 odd UNPADDED", variant_a, bf(q), bf(k), bf(v))
ok_b = bench("(b) T=790 padded, seq_lengths=789 (no bias)", variant_b, bf(qp), bf(kp), bf(vp))
ok_c = bench("(c) T=790 padded, explicit bool mask (has_bias)", variant_c, bf(qp), bf(kp), bf(vp))
print(f"\nRESULT: (a)={'OK' if ok_a else 'FAIL'} (b)={'OK' if ok_b else 'FAIL'} (c)={'OK' if ok_c else 'FAIL'}")
