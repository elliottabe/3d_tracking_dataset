"""Instance <-> labelled-fly assignment by enumeration (I<=4, F<=2)."""
from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np


def enumerate_assignments(n_inst: int, n_flies: int) -> np.ndarray:
    return np.asarray(list(itertools.permutations(range(n_inst), n_flies)), np.int32)


def match(cost, fly_valid, pin_first):
    """cost (B,I,F) -> assign (B,F) instance index (-1 for invalid flies), inst_matched (B,I)."""
    B, I, F = cost.shape
    cand = jnp.asarray(enumerate_assignments(I, F))                          # (n,F)
    c = jnp.take_along_axis(jnp.broadcast_to(cost[:, None], (B, cand.shape[0], I, F)),
                            cand[None, :, None, :], axis=2)[:, :, 0, :]      # (B,n,F)
    c = jnp.where(fly_valid[:, None, :], c, 0.0).sum(-1)                     # (B,n)
    pinned_ok = (cand[:, 0] == 0)[None, :] | ~pin_first[:, None]
    c = jnp.where(pinned_ok, c, jnp.inf)
    best = jnp.argmin(c, axis=1)                                             # (B,)
    assign = cand[best]                                                      # (B,F)
    assign = jnp.where(fly_valid, assign, -1)
    inst_matched = (jnp.arange(I)[None, :, None] == assign[:, None, :]).any(-1)
    return assign.astype(jnp.int32), inst_matched
