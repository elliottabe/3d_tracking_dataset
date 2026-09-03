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
