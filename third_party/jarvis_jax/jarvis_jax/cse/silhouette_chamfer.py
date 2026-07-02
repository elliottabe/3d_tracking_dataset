"""Differentiable one-directional sparse Chamfer residual (pure JAX).

For each SAM-boundary point p, r_p = softmin_v ||p - proj_v|| over the
projected mesh vertices v. Softmin = -1/beta * logsumexp(-beta * d) is a smooth
lower bound on the hard min (beta -> inf recovers the hard min); the nearest
vertex receives (almost) all the gradient, so this pulls the mesh outline OUT to
the mask boundary WITHOUT needing an explicit differentiable mesh contour.

NaN-safe (NaN target rows -> 0 residual, no gradient), mirroring stac
marker_cost's finite-mask convention. Optional Huber robustification.

MEMORY BOUND (efficiency Global Constraint): the per-target softmin over the M
verts is computed inside a jax.lax.map over CHUNKS of the N target points, so
the peak intermediate is O(chunk_size * M) -- the full (N, M) distance matrix is
NEVER materialized. This matters because the silhouette factor (Task 3) calls
this once per camera and jaxls vmaps the whole factor over T frames; a naive
(N, M) matrix would peak at O(T * C * N * M).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _huber_sqrt(d, delta):
    """sqrt of the Huber loss so that (return)^2 == huber(d).

    huber(d) = d^2 for |d|<=delta, else 2*delta*|d| - delta^2. Returning its
    sqrt lets the LM least-squares objective (sum of residual^2) behave as a
    Huber loss on the raw distance d.
    """
    quad = d
    lin = jnp.sqrt(jnp.clip(2.0 * delta * jnp.abs(d) - delta ** 2, a_min=0.0))
    return jnp.where(jnp.abs(d) <= delta, quad, lin)


def chamfer_residual(target_pts, proj_pts, *, beta: float = 8.0,
                     huber_delta: float = 0.0, chunk_size: int = 32):
    """One-directional sparse Chamfer: SAM boundary -> nearest projected vertex.

    Args:
        target_pts: (N, 2) sampled SAM boundary points (NaN rows allowed).
        proj_pts:   (M, 2) projected mesh vertices (function of qpos).
        beta:       softmin sharpness (larger -> closer to hard min).
        huber_delta: >0 enables Huber on the per-target distance; 0 disables.
            MUST be a static Python float (branched with a plain `if`, NOT
            jnp.where) -- jnp.where would trace the unused Huber branch, whose
            sqrt(clip(.,0)) has a NaN gradient at 0 and would poison grads even
            when huber_delta==0 (verified). It is closed over the inner _one, so
            it stays static under jax.lax.map.
        chunk_size: batch size for jax.lax.map over the N targets. Bounds the
            peak intermediate at O(chunk_size * M); result is chunk-invariant.

    Returns:
        (N,) per-target residual; NaN target rows -> 0.0.
    """
    target_pts = jnp.asarray(target_pts)
    proj_pts = jnp.asarray(proj_pts)
    finite = jnp.isfinite(target_pts).all(axis=-1)              # (N,)
    tgt = jnp.where(finite[:, None], target_pts, 0.0)           # sanitize NaNs

    # Per-target softmin over the M verts. Peak intermediate here is O(M) (the
    # per-vertex distance vector); jax.lax.map batches this by chunk_size, so the
    # whole call peaks at O(chunk_size * M), never the full (N, M) matrix. No
    # host callbacks -> jit-fusable. huber_delta is branched with a STATIC `if`
    # (never jnp.where) so the unused sqrt branch is not traced.
    def _one(p):
        d = jnp.sqrt(jnp.sum((proj_pts - p[None, :]) ** 2, axis=-1) + 1e-12)  # (M,)
        soft = -(1.0 / beta) * logsumexp(-beta * d)                          # scalar
        if huber_delta > 0.0:
            soft = _huber_sqrt(soft, huber_delta)
        return soft

    soft = jax.lax.map(_one, tgt, batch_size=chunk_size)        # (N,)
    return jnp.where(finite, soft, 0.0)
