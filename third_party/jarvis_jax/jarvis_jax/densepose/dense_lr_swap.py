"""Dense-channel L/R flip map for the CSE (50 + M) ViTPose (Phase 5).

The CSE model's flip augmentation was DISABLED because the 50-keypoint
build_lr_swap does not cover the M dense FPS vertices. The raw mesh `sym_index`
is a per-vertex bilateral partner but is NOT closed under FPS subsampling (only
~1/3 of an fps subset's partners are themselves in the subset, and the restricted
map is not an involution). So build the dense block by a reflection-distance
mutual-mirror pairing across the bilateral plane.

VERIFIED (this cycle): in the canonical `vertices` frame the bilateral mirror axis
is Y (abs-mean signed |vtx - sym_partner| per axis ~ [1.6e-4, 0.247, 1.2e-4]), and
the vertex cloud spans Y in [-0.305, 0.305] (std ~0.15). On fps_300, |Y| has a
natural gap: 48/300 verts sit at |Y| < 0.02 (true midline, median |Y| ~0.006) and
the remaining 252/300 are clearly lateral (|Y| >= 0.02, i.e. off the bilateral
plane) -- so midline_tol=0.02 cleanly separates "sits on the midline" from
"has a mirror partner on the other side" (default kept below).

FIXED (final review, Critical): the previous mutual-NN + same-segment-only gate
left 26/300 dense verts self-mapped even though they were lateral -- flip_batch
mirrors x for every channel unconditionally, so those 26 self-mapped lateral
targets landed on the WRONG side of the midline on every flipped batch (~50% of
batches), corrupting keypoint-head gradients. Now: midline verts (|Y| < tol)
self-map; every lateral vert is guaranteed a mutual mirror partner via a global
minimum-weight perfect matching over reflection distance (`networkx`, exact
blossom algorithm) -- an involution by construction, with no greedy-matching
"stranded vertex" pathology (verified: naive nearest-first greedy left one
abdominal vertex at residual 0.28; min-weight matching over the same candidate
graph gets 0.070 max, 0.0051 mean). If a lateral vertex still cannot be matched
(non-even lateral count, disconnected graph, etc.) this raises -- it never
silently self-maps.
"""
from __future__ import annotations

import numpy as np

from jarvis_jax.data.augment import build_lr_swap

_MIRROR_AXIS = 1                  # Y, in the canonical `vertices` frame (verified)
_DEFAULT_MIDLINE_TOL = 0.02        # canonical units; see module docstring


def _match_lateral_mirrors(V, lateral_idx, mirror_axis):
    """Pair every lateral vertex with its mutual mirror partner.

    Builds the complete graph on `lateral_idx` weighted by reflection distance
    ``||reflect(V_i) - V_j||`` (i != j) and returns the exact global minimum-weight
    maximum-cardinality matching (networkx blossom algorithm). Because the graph
    is complete and `lateral_idx` has even size, the maximum-cardinality matching
    is a perfect matching -- i.e. every lateral vertex is paired, by construction.
    This avoids the failure mode of naive nearest-first greedy matching, where an
    early greedy pick can strand a later vertex with only a poor, distant option
    left (observed on this mesh: greedy stranded one vertex at residual 0.28;
    the optimal matching pairs the same vertex at 0.025).

    Returns dict {i: j} with i, j in `lateral_idx`, symmetric (j in dict maps to i).
    Raises RuntimeError if a perfect matching cannot be found (should not happen
    on a symmetric mesh with an even lateral count).
    """
    import networkx as nx

    n = len(lateral_idx)
    if n == 0:
        return {}
    if n % 2 != 0:
        raise RuntimeError(
            f"cannot pair lateral dense verts: odd count ({n}) with no midline "
            "partner; check midline_tol / mesh symmetry")

    refl = V[lateral_idx].copy()
    refl[:, mirror_axis] *= -1.0
    P = V[lateral_idx]
    # reflection is an isometry, so this distance matrix is symmetric:
    # ||reflect(V_i) - V_j|| == ||V_i - reflect(V_j)||.
    D = np.linalg.norm(refl[:, None, :] - P[None, :, :], axis=-1)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    iu, ju = np.triu_indices(n, k=1)
    for a, b in zip(iu.tolist(), ju.tolist()):
        graph.add_edge(a, b, weight=float(D[a, b]))

    matching = nx.algorithms.matching.min_weight_matching(graph)
    if len(matching) * 2 != n:
        unmatched = n - 2 * len(matching)
        raise RuntimeError(
            f"failed to pair {unmatched} lateral dense vert(s) into mirror "
            "partners -- refusing to silently self-map a lateral vertex")

    pairs = {}
    for a, b in matching:
        i, j = int(lateral_idx[a]), int(lateral_idx[b])
        pairs[i] = j
        pairs[j] = i
    return pairs


def _dense_vertex_swap(mesh_npz, fps_key, tol, midline_tol=_DEFAULT_MIDLINE_TOL):
    """Mirror-reflection pairing of the M FPS vertices.

    Midline verts (|Y| < midline_tol) self-map. Every lateral vert (|Y| >=
    midline_tol) is paired with a mutual mirror partner via global min-weight
    matching on reflection distance -- see `_match_lateral_mirrors`. Segment
    identity (`vertex_segment` / `seg_names`) is not needed to build a valid
    involution: the matching already favours anatomically-correct L<->R pairs
    since those have the smallest reflection distance.

    `tol` is accepted for backward compatibility with the previous mutual-NN
    gate's signature but is no longer used to reject pairs: this function now
    guarantees every lateral vertex is paired, so silently discarding a pair for
    exceeding `tol` is exactly the bug being fixed. It is kept as an unused
    parameter (rather than removed) to preserve the call signature.

    Returns int32 (M,) with values in [0, M). Involution by construction.
    """
    del tol  # kept for signature compatibility; see docstring.

    z = np.load(mesh_npz, allow_pickle=True)
    fps = np.asarray(z[fps_key])
    V = np.asarray(z["vertices"])[fps].astype(np.float64)  # (M,3) canonical coords
    M = len(fps)

    is_midline = np.abs(V[:, _MIRROR_AXIS]) < midline_tol
    lateral_idx = np.where(~is_midline)[0]

    swap = np.arange(M)
    pairs = _match_lateral_mirrors(V, lateral_idx, _MIRROR_AXIS)
    for i, j in pairs.items():
        swap[i] = j

    assert np.array_equal(swap[swap], np.arange(M)), \
        "dense vertex swap is not an involution"
    self_mapped = np.where(swap == np.arange(M))[0]
    assert np.all(np.abs(V[self_mapped, _MIRROR_AXIS]) < midline_tol), \
        "a lateral (off-midline) dense vertex is self-mapped -- would corrupt " \
        "flip-augmented targets by leaving it on the wrong side of the midline"
    return swap.astype(np.int32)


def build_dense_lr_swap(mesh_npz, fps_key, base_names, *, tol=0.1,
                         midline_tol=_DEFAULT_MIDLINE_TOL):
    """(50+M,) int32 L/R involution: named-kp swap (first 50) ++ dense vertex swap."""
    kp_swap = build_lr_swap(base_names).astype(np.int32)          # (50,)
    if len(kp_swap) != 50:
        raise ValueError(f"expected 50 base keypoint names, got {len(kp_swap)}")
    vtx_swap = _dense_vertex_swap(mesh_npz, fps_key, tol, midline_tol)  # (M,) in [0,M)
    swap = np.concatenate([kp_swap, 50 + vtx_swap]).astype(np.int32)
    assert np.array_equal(swap[swap], np.arange(len(swap))), \
        "combined (50+M) swap is not an involution"

    # Final-review guardrail: every self-mapped DENSE channel must be a true
    # midline vertex. A self-mapped lateral vertex would silently corrupt every
    # flip-augmented batch (flip_batch mirrors x unconditionally for all
    # channels), so this must hold or building the swap map fails loudly.
    z = np.load(mesh_npz, allow_pickle=True)
    fps = np.asarray(z[fps_key])
    V = np.asarray(z["vertices"])[fps]
    dense = swap[50:] - 50
    self_mapped_dense = np.where(dense == np.arange(len(dense)))[0]
    assert np.all(np.abs(V[self_mapped_dense, _MIRROR_AXIS]) < midline_tol), \
        "build_dense_lr_swap: a self-mapped dense channel is lateral, not midline"
    return swap
