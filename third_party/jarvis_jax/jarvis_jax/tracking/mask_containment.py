"""Differentiable mesh->mask CONTAINMENT residual (pure JAX).

Recovered verbatim from `silhouette_containment.py` (deleted in `0bc36fe`,
which removed the silhouette-polish stage). Rehomed here for WING PITCH
refinement only: after the STAC solve, this residual is used to pull a
projected wing vertex that has drifted outside the fly's SAM mask back
toward the silhouette.

For each projected mesh vertex, bilinearly sample the mask's cropped signed
distance field (negative inside, positive outside, original px) and penalize the
outside part: r = conf * relu(d + margin). This is the coverage-ASYMMETRIC term:
verts outside the mask are pulled in hard; verts inside contribute nothing (so
the SAM shadow halo, which only inflates the mask, cannot drag the fit outward).

ONE-SIDED BY DESIGN: relu(d + margin) means a vertex already inside the mask
costs nothing, so minimising this residual ALONE would happily tuck the wing
INSIDE the body silhouette -- which is exactly the "wing rotates through the
abdomen" bug this refinement is meant to fix, not a variant of it. This module
supplies only the one-sided pull-in-from-outside term; a separate wing-COVERAGE
term (added downstream, Task 4) supplies the opposing pull that keeps the wing
from collapsing into the body, and the two together are what makes the
refinement well posed.

An out-of-crop OVERFLOW distance (converted to original px via grid_scale) is
added so verts projecting beyond the cropped SDF still receive an inward
gradient instead of a clamped zero-gradient plateau.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp


def bilinear_sample(img, xy):
    """Clamped bilinear sample of img (H,W) at xy (M,2) grid coords (x=col,y=row)."""
    H, W = img.shape
    x = jnp.clip(xy[:, 0], 0.0, W - 1.0)
    y = jnp.clip(xy[:, 1], 0.0, H - 1.0)
    x0 = jnp.floor(x).astype(jnp.int32); y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, W - 1); y1 = jnp.clip(y0 + 1, 0, H - 1)
    wx = x - x0; wy = y - y0
    Ia = img[y0, x0]; Ib = img[y0, x1]; Ic = img[y1, x0]; Id = img[y1, x1]
    return (Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy)
            + Ic * (1 - wx) * wy + Id * wx * wy)


def containment_residual(proj_pts, sdf, grid_scale, grid_offset, conf, *,
                         margin=0.0, present=True):
    """conf * relu(d + margin), gated by `present`. d = SDF(bilinear) + overflow.

    proj_pts (M,2) original px; grid_xy = (proj - grid_offset) * grid_scale.
    overflow = distance the point lies OUTSIDE the SDF grid box, back in original
    px, so far-outside verts keep a finite inward gradient (no clamp plateau).
    """
    H, W = sdf.shape
    grid = (proj_pts - grid_offset[None, :]) * grid_scale[None, :]        # (M,2)
    d_edge = bilinear_sample(sdf, grid)                                   # (M,) orig px
    ox = jnp.maximum(jnp.maximum(-grid[:, 0], grid[:, 0] - (W - 1.0)), 0.0)
    oy = jnp.maximum(jnp.maximum(-grid[:, 1], grid[:, 1] - (H - 1.0)), 0.0)
    overflow = jnp.sqrt((ox / grid_scale[0]) ** 2 + (oy / grid_scale[1]) ** 2 + 1e-12)
    d = d_edge + overflow
    r = conf * jax.nn.relu(d + margin)
    return jnp.where(present, r, jnp.zeros_like(r))
