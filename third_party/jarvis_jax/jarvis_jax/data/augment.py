"""On-device (jitted) train-time data augmentation for ViTPose 2D.

Geometric affine + horizontal flip (with a left/right keypoint-index swap) +
cutout + photometric, applied to a (img4_u8, kp_xy, vis) batch before heatmap
rendering. Keypoints are transformed in lockstep with the image. Train-only;
`AugParams.enabled == False` is an exact identity.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


def _mirror_name(n):
    """The left/right mirror of a keypoint name, or None if it is midline.
    The side token is the last char of the segment before the first '_'
    (e.g. 'EyeL'->'EyeR', 'WingL_base'->'WingR_base', 'T1L_TaTip'->'T1R_TaTip')."""
    head = n.split("_", 1)[0]
    if head.endswith("L"):
        return n.replace(head, head[:-1] + "R", 1)
    if head.endswith("R"):
        return n.replace(head, head[:-1] + "L", 1)
    return None


def build_lr_swap(names):
    """Permutation index array mapping each keypoint to its L/R mirror (midline
    -> itself). Returns int32 (K,). Asserts the result is an involution."""
    idx = {n: i for i, n in enumerate(names)}
    swap = list(range(len(names)))
    for i, n in enumerate(names):
        m = _mirror_name(n)
        if m is not None and m in idx:
            swap[i] = idx[m]
    arr = np.asarray(swap, dtype=np.int32)
    assert np.array_equal(arr[arr], np.arange(len(names))), \
        "lr_swap is not an involution — check keypoint L/R naming"
    return arr
