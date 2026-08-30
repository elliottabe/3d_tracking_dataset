"""Bounded rotation augmentation of the world-locked voxel grid.

WHY. hybridnet/reproject.py builds the grid axis-aligned to the WORLD frame
(`grid = base_grid + center3D`) and mean-pools the cameras, so the V2VNet's
only geometric dependency is world-frame ORIENTATION -- it learns "dorsal is
roughly +Z" for whichever calibration dominates training. There are three
distinct rig calibrations in the labeled data. Rotating the grid basis during
training, with the labels rotated by the same R, removes that dependence.

At 3,800 fly-samples the bigger win is simply that this is a free data
multiplier for a 3D conv net.

WHY BOUNDED, NOT FULL SO(3). The world frame carries a real physical prior:
flies are usually upright on the arena floor, so "legs point down in world -Z"
is genuine free information. Full SO(3) throws it away. Expect this to HELP the
female on walls (where the upright prior fails anyway) and cost a little on the
easy male -- which is the trade we want, and which arms A1/A2 and A3/A4 measure
rather than assume.
"""
from __future__ import annotations

import numpy as np


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def gravity_axis(cam_mats: np.ndarray) -> np.ndarray:
    """World axis most nearly shared as 'up' across the rig.

    The arena floor is level and every camera looks roughly inward, so the
    world axis with the most consistent sign across camera image-y directions
    is the gravity axis. Falls back to +Z, which is correct for this rig.

    Uses the UN-centered SVD of the (sign-agnostic) image-y directions: its
    top right-singular-vector is the direction of maximum shared AGREEMENT
    across cameras. (Mean-centering first and taking the residual SVD would
    instead return the axis of maximum *disagreement* between cameras, which
    is wrong for near-identical rig setups.) Since image-y can point either
    way per camera, each row is sign-flipped toward the running consensus
    before the SVD so opposite-facing cameras don't cancel out.
    """
    M = np.asarray(cam_mats, np.float64)
    if M.ndim != 3:
        return np.array([0.0, 0.0, 1.0])
    ydir = M[:, :3, 1]                                   # (C,3)
    ydir = ydir / np.maximum(np.linalg.norm(ydir, axis=1, keepdims=True), 1e-12)
    # Make signs consistent (axis, not direction, is what we want to agree on).
    ref = ydir[0]
    signs = np.sign(ydir @ ref)
    signs[signs == 0] = 1.0
    ydir = ydir * signs[:, None]
    u, s, vt = np.linalg.svd(ydir)
    axis = vt[0] if s[0] > 0 else np.array([0.0, 0.0, 1.0])
    if axis[2] < 0:
        axis = -axis
    return axis / max(np.linalg.norm(axis), 1e-12)


def sample_rotation(rng: np.random.Generator, *, yaw_range=(0.0, 2 * np.pi),
                    tilt_deg: float = 30.0, axis=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Full yaw about `axis`, plus a bounded tilt away from it."""
    axis = np.asarray(axis, np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    yaw = rng.uniform(*yaw_range)
    R = _rodrigues(axis, yaw)
    if tilt_deg > 0:
        # A random axis perpendicular to `axis`, tilted by <= tilt_deg.
        perp = np.cross(axis, rng.normal(size=3))
        n = np.linalg.norm(perp)
        if n > 1e-9:
            tilt = np.radians(rng.uniform(-tilt_deg, tilt_deg))
            R = _rodrigues(perp / n, tilt) @ R
    return R.astype(np.float32)


def augment_sample(sample: dict, R: np.ndarray) -> dict:
    """Rotate kp3d about center3D by R. Invisible keypoints are left untouched."""
    out = dict(sample)
    R = np.asarray(R, np.float32)
    c = np.asarray(sample["center3D"], np.float32)
    kp = np.asarray(sample["kp3d"], np.float32).copy()
    vis = np.asarray(sample["vis"], bool)
    kp[vis] = ((kp[vis] - c) @ R.T) + c
    out["kp3d"] = kp
    out["rotation"] = R
    return out
