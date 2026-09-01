"""Differentiable one-directional sparse Chamfer residual (pure JAX), plus a
target sampler for WING PITCH refinement.

Recovered from `silhouette_chamfer.py` (deleted in `0bc36fe` with the
silhouette-polish stage) and rehomed here: `chamfer_residual` is renamed
`coverage_residual` and paired with the one genuinely new piece,
`wing_target_points`, which supplies its targets -- mask pixels the BODY does
not already explain.

This module supplies the WING-COVERAGE term that opposes `mask_containment`.
`containment_residual` is one-sided by design (a vertex inside the mask costs
nothing), so minimising it alone would be solved perfectly by tucking the
wing entirely inside the body silhouette -- the "wing rotates through the
abdomen" bug this refinement exists to fix. `coverage_residual`, driven by
`wing_target_points`, instead penalises mask area that the body silhouette
does NOT cover and that the wing therefore ought to explain: for each such
pixel, r_p = softmin_v ||p - proj_v|| over the projected WING vertices v.
Softmin = -1/beta * logsumexp(-beta * d) is a smooth lower bound on the hard
min (beta -> inf recovers the hard min); the nearest vertex receives (almost)
all the gradient, so this pulls the wing OUT to cover that mask area without
needing an explicit differentiable wing contour. Together, containment
(pull wing verts that stray outside the mask back in) and coverage (pull
wing-attributable mask area toward the nearest wing vert) pin the wing where
the mask actually says it is.

NaN-safe (NaN target rows -> 0 residual, no gradient), mirroring stac
marker_cost's finite-mask convention. Optional Huber robustification.

MEMORY BOUND (efficiency Global Constraint): the per-target softmin over the M
verts is computed inside a jax.lax.map over CHUNKS of the N target points, so
the peak intermediate is O(chunk_size * M) -- the full (N, M) distance matrix
is NEVER materialized.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def wing_target_points(mask, body_uv, *, n_points, dilate_px=3, seed=0):
    """Sample mask pixels the BODY silhouette does not already explain.

    Rasterises `body_uv` (projected BODY vertices, ORIGINAL pixels, (x, y))
    into a boolean image the size of `mask`, optionally dilated by
    `dilate_px` so a sparse vertex cloud covers the body area it stands in
    for, then samples `n_points` pixels from `mask & ~body` uniformly at
    random (seeded). NaN-padded to `n_points` rows if fewer pixels survive.

    Args:
        mask: (H, W) bool SAM mask for this (frame, camera).
        body_uv: (V, 2) projected body vertices, (x=col, y=row), original px.
        n_points: number of target rows to return.
        dilate_px: isotropic dilation radius (pixels) applied to the
            rasterised body footprint before subtracting it from the mask.
        seed: RNG seed for the sample (reproducible target sets).

    Returns:
        (n_points, 2) float32, columns (x, y); rows beyond the number of
        surviving pixels are NaN.
    """
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    body_uv = np.asarray(body_uv)

    body = np.zeros((H, W), dtype=bool)
    xy = np.round(body_uv).astype(np.int64)
    x, y = xy[:, 0], xy[:, 1]
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    body[y[valid], x[valid]] = True

    if dilate_px and dilate_px > 0:
        from scipy import ndimage
        body = ndimage.binary_dilation(body, iterations=int(dilate_px))

    uncovered = mask & ~body
    ys, xs = np.where(uncovered)
    n_avail = xs.shape[0]

    out = np.full((n_points, 2), np.nan, dtype=np.float32)
    if n_avail == 0:
        return out

    rng = np.random.default_rng(seed)
    n_take = min(n_points, n_avail)
    idx = rng.choice(n_avail, size=n_take, replace=False)
    out[:n_take, 0] = xs[idx].astype(np.float32)
    out[:n_take, 1] = ys[idx].astype(np.float32)
    return out


def _huber_sqrt(d, delta):
    """sqrt of the Huber loss so that (return)^2 == huber(d).

    huber(d) = d^2 for |d|<=delta, else 2*delta*|d| - delta^2. Returning its
    sqrt lets the LM least-squares objective (sum of residual^2) behave as a
    Huber loss on the raw distance d.
    """
    quad = d
    lin = jnp.sqrt(jnp.clip(2.0 * delta * jnp.abs(d) - delta ** 2, min=0.0))
    return jnp.where(jnp.abs(d) <= delta, quad, lin)


def coverage_residual(target_pts, proj_wing_uv, *, beta: float = 8.0,
                      huber_delta: float = 0.0, chunk_size: int = 32):
    """One-directional sparse Chamfer: wing-attributable mask pixel -> nearest
    projected wing vertex.

    Args:
        target_pts: (N, 2) mask pixels the body does not explain, from
            `wing_target_points` (NaN rows allowed).
        proj_wing_uv: (M, 2) projected wing mesh vertices (function of qpos).
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
    proj_wing_uv = jnp.asarray(proj_wing_uv)
    finite = jnp.isfinite(target_pts).all(axis=-1)              # (N,)
    tgt = jnp.where(finite[:, None], target_pts, 0.0)           # sanitize NaNs

    # Per-target softmin over the M verts. Peak intermediate here is O(M) (the
    # per-vertex distance vector); jax.lax.map batches this by chunk_size, so the
    # whole call peaks at O(chunk_size * M), never the full (N, M) matrix. No
    # host callbacks -> jit-fusable. huber_delta is branched with a STATIC `if`
    # (never jnp.where) so the unused sqrt branch is not traced.
    def _one(p):
        d = jnp.sqrt(jnp.sum((proj_wing_uv - p[None, :]) ** 2, axis=-1) + 1e-12)  # (M,)
        soft = -(1.0 / beta) * logsumexp(-beta * d)                              # scalar
        if huber_delta > 0.0:
            soft = _huber_sqrt(soft, huber_delta)
        return soft

    soft = jax.lax.map(_one, tgt, batch_size=chunk_size)        # (N,)
    return jnp.where(finite, soft, 0.0)
