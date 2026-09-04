"""cuDNN flash attention feasibility for the DINOv3 backbone shape (N=56 crops, 12 heads, 789 tokens, hd=64).
jax's check: with a bias/mask AND training, T and S must be EVEN; without a mask odd lengths are fine on cc 8.9.
Measures fwd+bwd wall time and peak memory: explicit fp32, explicit bf16, cudnn bf16 (no mask, T=789),
cudnn bf16 with a key mask at T padded to 790 and 832, and a 12-layer stack of each attention-only op."""
import time, jax, jax.numpy as jnp, numpy as np
N,Hh,T,D=56,12,789,64
key=jax.random.PRNGKey(0); k1,k2,k3=jax.random.split(key,3)
q=jax.random.normal(k1,(N,T,Hh,D),jnp.float32); k=jax.random.normal(k2,(N,T,Hh,D),jnp.float32); v=jax.random.normal(k3,(N,T,Hh,D),jnp.float32)
def explicit(q,k,v):
    qh,kh,vh=(x.transpose(0,2,1,3) for x in (q,k,v))
    a=jax.nn.softmax((qh@kh.transpose(0,1,3,2))*(D**-0.5),axis=-1)
    return (a@vh).transpose(0,2,1,3)
def cudnn(q,k,v,mask=None):
    return jax.nn.dot_product_attention(q,k,v,mask=mask,implementation="cudnn")
def bench(name, fn, *args, reps=10):
    loss=lambda *a: jnp.sum(fn(*a).astype(jnp.float32)**2)
    g=jax.jit(jax.grad(loss))
    try:
        out=g(*args); jax.block_until_ready(out)
    except Exception as e:
        print(f"{name:40s} FAILED: {type(e).__name__}: {str(e)[:120]}"); return
    dev=jax.local_devices()[0]
    t0=time.perf_counter()
    for _ in range(reps): out=g(*args)
    jax.block_until_ready(out); dt=(time.perf_counter()-t0)/reps
    print(f"{name:40s} {dt*1000:8.2f} ms/iter  peak {dev.memory_stats()['peak_bytes_in_use']/2**30:6.2f} GiB")
bf=lambda x: x.astype(jnp.bfloat16)
bench("explicit fp32 T=789", explicit, q,k,v)
bench("explicit bf16 T=789", explicit, bf(q),bf(k),bf(v))
bench("cudnn bf16 no-mask T=789", cudnn, bf(q),bf(k),bf(v))
bench("cudnn bf16 no-mask T=790 (pad 1)", cudnn, bf(jnp.pad(q,((0,0),(0,1),(0,0),(0,0)))),bf(jnp.pad(k,((0,0),(0,1),(0,0),(0,0)))),bf(jnp.pad(v,((0,0),(0,1),(0,0),(0,0)))))
for Tp in (790, 832):
    qp=bf(jnp.pad(q,((0,0),(0,Tp-T),(0,0),(0,0)))); kp=bf(jnp.pad(k,((0,0),(0,Tp-T),(0,0),(0,0)))); vp=bf(jnp.pad(v,((0,0),(0,Tp-T),(0,0),(0,0))))
    m=jnp.arange(Tp)<T; mask=jnp.broadcast_to(m[None,None,None,:],(N,1,Tp,Tp))
    bench(f"cudnn bf16 key-mask T={Tp}", lambda a,b,c: cudnn(a,b,c,mask), qp,kp,vp)
# 12-layer attention-only stacks (residual) to mimic the backbone's attention share
def stack(fn):
    def f(q,k,v):
        x=q
        for _ in range(12): x=x+fn(x,k,v)
        return x
    return f
bench("12x explicit fp32", stack(explicit), q,k,v, reps=3)
bench("12x explicit bf16", stack(explicit), bf(q),bf(k),bf(v), reps=3)
bench("12x cudnn bf16 no-mask", stack(cudnn), bf(q),bf(k),bf(v), reps=3)
