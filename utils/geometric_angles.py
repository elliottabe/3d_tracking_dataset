"""Compute geometric joint angles from 3D keypoint positions (anipose-compatible).

Replicates the angles_chain algorithm from anipose to compute joint angles
using local coordinate frames built from limb segment vectors, decomposed
into ZYX Euler angles. This provides angles comparable to anipose output,
independent of MuJoCo joint parameterization.

Reference: https://github.com/lambdaloop/anipose/blob/5e7e56e/anipose/compute_angles.py
"""

import numpy as np
from scipy.spatial.transform import Rotation


def _proj(v, u):
    """Project vector v onto u (row-wise for (N, 3) arrays)."""
    return u * (np.sum(v * u, axis=1) / np.sum(u * u, axis=1))[:, None]


def _ortho(u, v):
    """Orthogonalize u with respect to v: remove component of u along v."""
    return u - _proj(v, u)


def _normalize(u):
    """Normalize rows of (N, 3) array to unit length."""
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return u / norms


def _angles_flex(keypoints_dict, chain_triple):
    """Compute flexion angle between 3 consecutive keypoints (anipose method)."""
    a = keypoints_dict[chain_triple[0]]
    b = keypoints_dict[chain_triple[1]]
    c = keypoints_dict[chain_triple[2]]
    v1 = _normalize(a - b)
    v2 = _normalize(c - b)
    ang_rad = np.arccos(np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0))
    return np.degrees(ang_rad)


def angles_chain(keypoints_dict, chain_list):
    """Compute joint angles along a kinematic chain using the anipose method.

    Builds local coordinate frames at each joint from segment vectors,
    computes relative rotations between consecutive frames, and decomposes
    into ZYX Euler angles.

    Args:
        keypoints_dict: dict mapping keypoint name -> (T, 3) array of positions
        chain_list: ordered list of keypoint names forming the chain.
            Append "/" to a name to flip the flex direction (anipose convention).

    Returns:
        dict: angle arrays keyed by "{joint_name}_flex", "{joint_name}_rot",
              and "{first_joint}_abduct". Angles in degrees.
    """
    # Parse flex_type flags from chain_list
    chain = []
    flex_type = []
    for c in chain_list:
        if c.endswith("/"):
            chain.append(c[:-1])
            flex_type.append(-1)
        else:
            chain.append(c)
            flex_type.append(1)

    n_joints = len(chain)
    # Stack keypoints: (n_joints, T, 3)
    keypoints = np.array([keypoints_dict[c] for c in chain])
    n_points = keypoints.shape[1]

    # Build rotation frames at each joint
    xfs = []
    xfs.append(Rotation.identity(n_points))

    for i in range(n_joints - 1):
        pos = keypoints[i + 1]
        z_dir = _normalize(pos - keypoints[i])

        if i == n_joints - 2:
            # Last segment: use fallback axis for x_dir
            x_dir = _ortho(
                np.tile([1.0, 0.0, 0.0], (n_points, 1)), z_dir
            )
            # If z_dir is nearly parallel to [1,0,0], use [0,1,0]
            small_norm = np.linalg.norm(x_dir, axis=1) < 1e-5
            if np.any(small_norm):
                fallback = _ortho(
                    np.tile([0.0, 1.0, 0.0], (n_points, 1)), z_dir
                )
                x_dir[small_norm] = fallback[small_norm]
        else:
            x_dir = _ortho(keypoints[i + 2] - pos, z_dir)
            x_dir *= flex_type[i + 1]

        x_dir = _normalize(x_dir)
        y_dir = np.cross(z_dir, x_dir)

        M = np.stack([x_dir, y_dir, z_dir], axis=-1)  # (T, 3, 3)
        rot = Rotation.from_matrix(M)
        xfs.append(rot)

    # Compute relative rotations and extract Euler angles
    angles = []
    for i in range(n_joints - 1):
        try:
            rot = xfs[i].inv() * xfs[i + 1]
        except ValueError:
            angles.append(np.full((n_points, 3), np.nan))
            continue

        ang = rot.as_euler("zyx", degrees=True)

        # Correct flex sign ambiguity (anipose convention)
        if i != 0:
            flex = _angles_flex(keypoints_dict, chain[i - 1 : i + 2]) * flex_type[i]
            test = ~np.isclose(flex, ang[:, 1])
            ang[:, 0] += 180 * test
            ang[:, 1] = test * np.mod(-(ang[:, 1] + 180), 360) + (1 - test) * ang[:, 1]
            ang = np.mod(np.array(ang) + 180, 360) - 180

        angles.append(ang)

    # Package output
    outdict = {}
    for i, (name, ang) in enumerate(zip(chain, angles)):
        outdict[name + "_flex"] = ang[:, 1]
        if i != len(angles) - 1:
            outdict[name + "_rot"] = ang[:, 0]
        if i == 0:
            outdict[name + "_abduct"] = ang[:, 2]

    return outdict


# Default leg keypoint chains for the fruitfly model
LEG_CHAINS = {
    "T1L": ["T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa"],
    "T1R": ["T1R_ThxCx", "T1R_Tro", "T1R_FeTi", "T1R_TiTa"],
    "T2L": ["T2L_Tro", "T2L_FeTi", "T2L_TiTa"],
    "T2R": ["T2R_Tro", "T2R_FeTi", "T2R_TiTa"],
    "T3L": ["T3L_Tro", "T3L_FeTi", "T3L_TiTa"],
    "T3R": ["T3R_Tro", "T3R_FeTi", "T3R_TiTa"],
}


def compute_geometric_angles_all_legs(
    marker_sites, kp_names, leg_chains=None
):
    """Compute anipose-compatible geometric angles for all legs.

    Args:
        marker_sites: (T, N_kp, 3) array of 3D keypoint positions
        kp_names: list of N_kp keypoint names
        leg_chains: dict mapping leg name -> list of keypoint names.
            Defaults to LEG_CHAINS.

    Returns:
        dict: nested dict {leg_name: {angle_name: (T,) array in degrees}}
    """
    if leg_chains is None:
        leg_chains = LEG_CHAINS

    marker_sites = np.asarray(marker_sites)

    # Build keypoints_dict from marker_sites
    keypoints_dict = {}
    for i, name in enumerate(kp_names):
        keypoints_dict[name] = marker_sites[:, i, :]

    result = {}
    for leg_name, chain in leg_chains.items():
        # Check all keypoints exist
        missing = [kp for kp in chain if kp not in keypoints_dict]
        if missing:
            continue
        try:
            result[leg_name] = angles_chain(keypoints_dict, chain)
        except Exception as e:
            print(f"Warning: geometric angle computation failed for {leg_name}: {e}")
            continue

    return result
