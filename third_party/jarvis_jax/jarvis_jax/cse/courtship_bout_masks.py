"""Courtship SAM3 mask adapter: bit-packed per-bout masks -> full-frame bool.

`sam3_masks.npz` stores `packed (A, C, T, H, ceil(W/8)) uint8` (np.packbits along
width), `valid (A,C,T) bool`, `shape [H,W]`, `centroids (A,C,T,2)`. Unpacking
gives the FULL-FRAME (H,W) mask in the calibration pixel frame (verified: fly0/fly1
land at the correct full-frame columns), so no centroid offset is needed.
"""
from __future__ import annotations
import numpy as np


def unpack_one(packed, fly: int, cam: int, frame: int, W: int) -> np.ndarray:
    """(H,W) bool full-frame mask for one (fly,cam,frame)."""
    return np.unpackbits(packed[fly, cam, frame], axis=-1)[:, :W].astype(bool)


def load_bout_masks(npz_path: str, fly: int) -> dict:
    """Full-frame masks for one fly across the bout: masks (T,C,H,W) bool + valid (T,C)."""
    z = np.load(npz_path)
    packed = z["packed"]; H, W = int(z["shape"][0]), int(z["shape"][1])
    A, C, T = packed.shape[0], packed.shape[1], packed.shape[2]
    valid = np.asarray(z["valid"])[fly].transpose(1, 0)          # (T,C)
    masks = np.zeros((T, C, H, W), bool)
    for c in range(C):
        for t in range(T):
            masks[t, c] = np.unpackbits(packed[fly, c, t], axis=-1)[:, :W].astype(bool)
    return dict(masks=masks, valid=valid, T=T, C=C, H=H, W=W)
