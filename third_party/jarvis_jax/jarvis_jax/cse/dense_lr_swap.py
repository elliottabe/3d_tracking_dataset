"""Dense-channel L/R flip map for the CSE (50 + M) ViTPose (Phase 5).

The CSE model's flip augmentation was DISABLED because the 50-keypoint
build_lr_swap does not cover the M dense FPS vertices. The raw mesh `sym_index`
is a per-vertex bilateral partner but is NOT closed under FPS subsampling (only
~1/3 of an fps subset's partners are themselves in the subset, and the restricted
map is not an involution). So build the dense block by a segment-aware
mutual-nearest-neighbour reflection across the bilateral plane.

VERIFIED (this cycle): in the canonical `vertices` frame the bilateral mirror axis
is Y (abs-mean signed |vtx - sym_partner| per axis ~ [1.6e-4, 0.247, 1.2e-4]).
Segment-aware mutual-NN reflection on fps_300 gives an involution, 246/300 swapped,
wing_left(seg 9) -> wing_right(seg 10) 100%, 0 cross-part mispairs.
"""
from __future__ import annotations

import numpy as np

from jarvis_jax.data.augment import build_lr_swap

_MIRROR_AXIS = 1        # Y, in the canonical `vertices` frame (verified)


def _mirror_seg_name(name):
    for a, b in (("_left", "_right"), ("_right", "_left"), ("_L", "_R"), ("_R", "_L")):
        if name.endswith(a):
            return name[: -len(a)] + b
    return name          # midline segment -> itself


def _dense_vertex_swap(mesh_npz, fps_key, tol):
    """Segment-aware mutual-NN Y-reflection pairing of the M FPS vertices.
    Returns int (M,) with values in [0, M); unpaired -> self. Involution."""
    from scipy.spatial import cKDTree

    z = np.load(mesh_npz, allow_pickle=True)
    fps = np.asarray(z[fps_key])
    V = np.asarray(z["vertices"])[fps]                    # (M,3) canonical coords
    seg = np.asarray(z["vertex_segment"])[fps]           # (M,) segment id per fps vtx
    seg_names = {int(s): (n.decode() if isinstance(n, bytes) else str(n))
                 for s, n in zip(z["seg_ids"], z["seg_names"])}
    name2id = {v: k for k, v in seg_names.items()}
    # mirror-segment id for each fps vertex (L<->R; midline -> same id).
    seg_mirror = np.array(
        [name2id.get(_mirror_seg_name(seg_names[int(s)]), int(s)) for s in seg])

    refl = V.copy()
    refl[:, _MIRROR_AXIS] *= -1.0                         # reflect across the bilateral plane
    tree = cKDTree(V)
    dist, nn = tree.query(refl, k=1)                      # nearest real vtx to each reflection

    M = len(fps)
    swap = np.arange(M)
    for i in range(M):
        j = int(nn[i])
        # mutual NN within tol AND segment-consistent (L<->R) -> a valid bilateral pair.
        if (dist[i] <= tol and dist[j] <= tol and int(nn[j]) == i
                and seg[j] == seg_mirror[i]):
            swap[i] = j
    assert np.array_equal(swap[swap], np.arange(M)), \
        "dense vertex swap is not an involution"
    return swap.astype(np.int32)


def build_dense_lr_swap(mesh_npz, fps_key, base_names, *, tol=0.1):
    """(50+M,) int32 L/R involution: named-kp swap (first 50) ++ dense vertex swap."""
    kp_swap = build_lr_swap(base_names).astype(np.int32)          # (50,)
    if len(kp_swap) != 50:
        raise ValueError(f"expected 50 base keypoint names, got {len(kp_swap)}")
    vtx_swap = _dense_vertex_swap(mesh_npz, fps_key, tol)         # (M,) in [0,M)
    swap = np.concatenate([kp_swap, 50 + vtx_swap]).astype(np.int32)
    assert np.array_equal(swap[swap], np.arange(len(swap))), \
        "combined (50+M) swap is not an involution"
    return swap
