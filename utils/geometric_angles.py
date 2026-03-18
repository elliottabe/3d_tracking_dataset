"""Geometric joint angle computation from 3D keypoints.

Implements anipose-style angle calculation without any body model or constraints:
  - flex angles: angle at the vertex keypoint between two bone vectors [0, π]
  - dihedral angles: signed rotation around a bone axis (-π, π]

These unconstrained geometric angles can be compared against STAC's body-model
joint angles (which are clipped to XML-defined ranges) to identify joints where
the ROM limits are too tight.

Typical usage
-------------
    kp, node_names = load_csv_keypoints("data3D.csv")
    angles_df = compute_geometric_angles(kp, node_names)
    suggestions = suggest_xml_ranges(angles_df)
    update_xml_ranges("fruitfly_force_free.xml", suggestions, dry_run=True)

Notes
-----
Flex angles are body-frame invariant — they capture actual bending regardless
of body rotation. Dihedral angles (for coxa decomposition) are computed in the
world frame and are only approximate when the fly's body orientation varies; for
clean decomposition into STAC's twist/flexion/extend DOFs, transform keypoints
into the body frame first (see `get_body_frame_keypoints`).

T2 and T3 legs lack a ThxCx keypoint, so only a combined coxa flex angle
(Scutellum→Tro→FeTi) is computed for those legs. It reflects total coxa bending
but cannot be decomposed into separate DOFs.
"""

from __future__ import annotations

import difflib
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── fly50 node names and their indices ────────────────────────────────────────
# Matches the ordering in data/fly50.json exactly.
FLY50_NODE_NAMES: list[str] = [
    "Antenna_Base", "EyeL", "EyeR", "Scutellum", "Abd_A4", "Abd_tip",        # 0-5
    "WingL_base", "WingL_V12", "WingL_V13",                                    # 6-8
    "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa",                          # 9-12
    "T1L_TaT1", "T1L_TaT3", "T1L_TaTip",                                      # 13-15
    "T2L_Tro", "T2L_FeTi", "T2L_TiTa",                                        # 16-18
    "T2L_TaT1", "T2L_TaT3", "T2L_TaTip",                                      # 19-21
    "T3L_Tro", "T3L_FeTi", "T3L_TiTa",                                        # 22-24
    "T3L_TaT1", "T3L_TaT3", "T3L_TaTip",                                      # 25-27
    "WingR_base", "WingR_V12", "WingR_V13",                                    # 28-30
    "T1R_ThxCx", "T1R_Tro", "T1R_FeTi", "T1R_TiTa",                          # 31-34
    "T1R_TaT1", "T1R_TaT3", "T1R_TaTip",                                      # 35-37
    "T2R_Tro", "T2R_FeTi", "T2R_TiTa",                                        # 38-40
    "T2R_TaT1", "T2R_TaT3", "T2R_TaTip",                                      # 41-43
    "T3R_Tro", "T3R_FeTi", "T3R_TiTa",                                        # 44-46
    "T3R_TaT1", "T3R_TaT3", "T3R_TaTip",                                      # 47-49
]

# ── Angle definitions ──────────────────────────────────────────────────────────
# Each entry: (angle_name, kind, node_names_tuple, xml_class, description)
#
# kind = 'flex':     arccos(dot(unit(n1-n2), unit(n3-n2))), always [0, π]
#                    node_names_tuple = (prox, vertex/joint, dist)
#
# kind = 'dihedral': signed angle of kp_moving around axis kp_axis_prox→kp_axis_dist,
#                    measured relative to kp_ref. Range (-π, π].
#                    node_names_tuple = (axis_prox, axis_dist, ref, moving)
#                    Result is comparable to STAC joint angles (signed, body-frame approx.)
#
# xml_class: the MuJoCo <default class="..."> that owns the range= attribute for
#            this DOF. None means no direct XML counterpart.

ANGLE_DEFINITIONS: list[tuple] = [
    # ── Left T1 leg ────────────────────────────────────────────────────────────
    # Coxa overall bending (T1 has ThxCx so we can form the triplet)
    ("T1L_coxa_flex",    "flex",
     ("Scutellum", "T1L_ThxCx", "T1L_Tro"),
     None, "Overall coxa bending at ThxCx (combined DOFs)"),

    # Coxa extend (elevation/depression): dihedral of Tro around the
    # Scutellum→ThxCx axis, relative to the EyeL→Scutellum direction as zero.
    # Approximates extend_coxa (axis X in body frame) in world frame.
    ("T1L_coxa_extend",  "dihedral",
     ("Scutellum", "T1L_ThxCx", "EyeL", "T1L_Tro"),
     "extend_coxa_T1",
     "Coxa elevation/depression (approx. extend_coxa_T1, world-frame dihedral)"),

    # Coxa flexion (protraction/retraction): dihedral of FeTi around the
    # ThxCx→Tro axis, relative to Scutellum as zero.
    # Approximates flexion_coxa (axis Z in body frame).
    ("T1L_coxa_flexion", "dihedral",
     ("T1L_ThxCx", "T1L_Tro", "Scutellum", "T1L_FeTi"),
     "flexion_coxa_T1",
     "Coxa protraction/retraction (approx. flexion_coxa_T1, world-frame dihedral)"),

    # Coxa twist (abduction/adduction): dihedral of TiTa around Tro→FeTi axis,
    # relative to ThxCx as zero. Approximates twist_coxa (axis Y in body frame).
    ("T1L_coxa_twist",   "dihedral",
     ("T1L_Tro", "T1L_FeTi", "T1L_ThxCx", "T1L_TiTa"),
     "twist_coxa_T1",
     "Coxa twist/abduction (approx. twist_coxa_T1, world-frame dihedral)"),

    # Femur extension (bending at the Tro/coxa-femur joint)
    ("T1L_femur_flex",   "flex",
     ("T1L_ThxCx", "T1L_Tro", "T1L_FeTi"),
     "extend_femur",
     "Femur-coxa bending angle (maps to extend_femur)"),

    # Femur twist: dihedral of TiTa around Tro→FeTi, relative to ThxCx
    ("T1L_femur_twist",  "dihedral",
     ("T1L_Tro", "T1L_FeTi", "T1L_ThxCx", "T1L_TiTa"),
     "twist_femur",
     "Femur torsion around its long axis (approx. twist_femur)"),

    # Tibia extension (bending at FeTi joint)
    ("T1L_tibia_flex",   "flex",
     ("T1L_Tro", "T1L_FeTi", "T1L_TiTa"),
     "extend_tibia",
     "Femur-tibia bending angle (maps to extend_tibia)"),

    # Tarsus extension (bending at TiTa joint)
    ("T1L_tarsus_flex",  "flex",
     ("T1L_FeTi", "T1L_TiTa", "T1L_TaT1"),
     "extend_tarsus_T1",
     "Tibia-tarsus bending angle (maps to extend_tarsus_T1)"),

    # ── Right T1 leg ───────────────────────────────────────────────────────────
    ("T1R_coxa_flex",    "flex",
     ("Scutellum", "T1R_ThxCx", "T1R_Tro"),
     None, "Overall coxa bending at ThxCx (combined DOFs)"),

    ("T1R_coxa_extend",  "dihedral",
     ("Scutellum", "T1R_ThxCx", "EyeR", "T1R_Tro"),
     "extend_coxa_T1",
     "Coxa elevation/depression (approx. extend_coxa_T1, world-frame dihedral)"),

    ("T1R_coxa_flexion", "dihedral",
     ("T1R_ThxCx", "T1R_Tro", "Scutellum", "T1R_FeTi"),
     "flexion_coxa_T1",
     "Coxa protraction/retraction (approx. flexion_coxa_T1, world-frame dihedral)"),

    ("T1R_coxa_twist",   "dihedral",
     ("T1R_Tro", "T1R_FeTi", "T1R_ThxCx", "T1R_TiTa"),
     "twist_coxa_T1",
     "Coxa twist/abduction (approx. twist_coxa_T1, world-frame dihedral)"),

    ("T1R_femur_flex",   "flex",
     ("T1R_ThxCx", "T1R_Tro", "T1R_FeTi"),
     "extend_femur",
     "Femur-coxa bending angle"),

    ("T1R_femur_twist",  "dihedral",
     ("T1R_Tro", "T1R_FeTi", "T1R_ThxCx", "T1R_TiTa"),
     "twist_femur", "Femur torsion"),

    ("T1R_tibia_flex",   "flex",
     ("T1R_Tro", "T1R_FeTi", "T1R_TiTa"),
     "extend_tibia", "Femur-tibia bending angle"),

    ("T1R_tarsus_flex",  "flex",
     ("T1R_FeTi", "T1R_TiTa", "T1R_TaT1"),
     "extend_tarsus_T1", "Tibia-tarsus bending angle"),

    # ── Left T2 leg (no ThxCx → combined coxa flex only) ──────────────────────
    ("T2L_coxa_flex",    "flex",
     ("Scutellum", "T2L_Tro", "T2L_FeTi"),
     None,
     "Overall coxa bending (Scutellum-Tro-FeTi); T2 has no ThxCx keypoint"),

    ("T2L_femur_flex",   "flex",
     ("T2L_Tro", "T2L_FeTi", "T2L_TiTa"),
     "extend_femur",
     "Femur-coxa bending angle"),

    ("T2L_femur_twist",  "dihedral",
     ("T2L_Tro", "T2L_FeTi", "Scutellum", "T2L_TiTa"),
     "twist_femur_T2", "Femur torsion (uses Scutellum as ref since no ThxCx)"),

    ("T2L_tibia_flex",   "flex",
     ("T2L_Tro", "T2L_FeTi", "T2L_TiTa"),
     "extend_tibia",
     "Femur-tibia bending angle"),

    ("T2L_tarsus_flex",  "flex",
     ("T2L_FeTi", "T2L_TiTa", "T2L_TaT1"),
     "extend_tarsus_T2", "Tibia-tarsus bending angle"),

    # ── Right T2 leg ───────────────────────────────────────────────────────────
    ("T2R_coxa_flex",    "flex",
     ("Scutellum", "T2R_Tro", "T2R_FeTi"),
     None,
     "Overall coxa bending (Scutellum-Tro-FeTi); T2 has no ThxCx keypoint"),

    ("T2R_femur_flex",   "flex",
     ("T2R_Tro", "T2R_FeTi", "T2R_TiTa"),
     "extend_femur", "Femur-coxa bending angle"),

    ("T2R_femur_twist",  "dihedral",
     ("T2R_Tro", "T2R_FeTi", "Scutellum", "T2R_TiTa"),
     "twist_femur_T2", "Femur torsion"),

    ("T2R_tibia_flex",   "flex",
     ("T2R_Tro", "T2R_FeTi", "T2R_TiTa"),
     "extend_tibia", "Femur-tibia bending angle"),

    ("T2R_tarsus_flex",  "flex",
     ("T2R_FeTi", "T2R_TiTa", "T2R_TaT1"),
     "extend_tarsus_T2", "Tibia-tarsus bending angle"),

    # ── Left T3 leg ────────────────────────────────────────────────────────────
    ("T3L_coxa_flex",    "flex",
     ("Scutellum", "T3L_Tro", "T3L_FeTi"),
     None, "Overall coxa bending; T3 has no ThxCx keypoint"),

    ("T3L_femur_flex",   "flex",
     ("T3L_Tro", "T3L_FeTi", "T3L_TiTa"),
     "extend_femur_T3", "Femur-coxa bending angle"),

    ("T3L_femur_twist",  "dihedral",
     ("T3L_Tro", "T3L_FeTi", "Scutellum", "T3L_TiTa"),
     "twist_femur_T3", "Femur torsion"),

    ("T3L_tibia_flex",   "flex",
     ("T3L_Tro", "T3L_FeTi", "T3L_TiTa"),
     "extend_tibia", "Femur-tibia bending angle"),

    ("T3L_tarsus_flex",  "flex",
     ("T3L_FeTi", "T3L_TiTa", "T3L_TaT1"),
     "extend_tarsus_T3", "Tibia-tarsus bending angle"),

    # ── Right T3 leg ───────────────────────────────────────────────────────────
    ("T3R_coxa_flex",    "flex",
     ("Scutellum", "T3R_Tro", "T3R_FeTi"),
     None, "Overall coxa bending; T3 has no ThxCx keypoint"),

    ("T3R_femur_flex",   "flex",
     ("T3R_Tro", "T3R_FeTi", "T3R_TiTa"),
     "extend_femur_T3", "Femur-coxa bending angle"),

    ("T3R_femur_twist",  "dihedral",
     ("T3R_Tro", "T3R_FeTi", "Scutellum", "T3R_TiTa"),
     "twist_femur_T3", "Femur torsion"),

    ("T3R_tibia_flex",   "flex",
     ("T3R_Tro", "T3R_FeTi", "T3R_TiTa"),
     "extend_tibia", "Femur-tibia bending angle"),

    ("T3R_tarsus_flex",  "flex",
     ("T3R_FeTi", "T3R_TiTa", "T3R_TaT1"),
     "extend_tarsus_T3", "Tibia-tarsus bending angle"),
]

# ── Current XML joint ranges (from fruitfly_force_free.xml) ───────────────────
# Used by suggest_xml_ranges() to show what is currently set.
# Format: xml_class → (lower_rad, upper_rad)
XML_JOINT_RANGES: dict[str, tuple[float, float]] = {
    "twist_coxa_T1":    (-0.80,  0.80),
    "flexion_coxa_T1":  (-1.00,  0.70),
    "extend_coxa_T1":   (-0.20,  1.70),
    "twist_coxa_T2":    (-0.75,  0.80),
    "flexion_coxa_T2":  (-0.50,  0.30),
    "extend_coxa_T2":   (-0.20,  0.90),
    "twist_coxa_T3":    (-0.15,  0.80),
    "flexion_coxa_T3":  (-0.90,  0.25),
    "extend_coxa_T3":   (-0.30,  1.30),
    "twist_femur":      (-1.00,  1.00),
    "twist_femur_T2":   (-1.00,  1.00),  # inherits from twist_femur
    "twist_femur_T3":   (-1.00,  1.00),  # inherits from twist_femur
    "extend_femur":     (-0.15,  2.00),
    "extend_femur_T3":  (-0.70,  1.50),
    "extend_tibia":     (-1.35,  1.30),
    "extend_tarsus_T1": (-0.70,  1.20),
    "extend_tarsus_T2": (-1.00,  1.80),
    "extend_tarsus_T3": (-0.80,  1.20),
}


# ── Low-level angle primitives ─────────────────────────────────────────────────

def _flex_angle(kp: np.ndarray, i1: int, i2: int, i3: int) -> np.ndarray:
    """Flex angle at vertex i2 between bones i1-i2 and i2-i3.

    Parameters
    ----------
    kp : (T, N, 3) array
    i1, i2, i3 : node indices; i2 is the joint vertex

    Returns
    -------
    (T,) array in radians [0, π]. NaN wherever any input keypoint is NaN.
    """
    v1 = kp[:, i1, :] - kp[:, i2, :]  # (T, 3)
    v2 = kp[:, i3, :] - kp[:, i2, :]  # (T, 3)

    norm1 = np.linalg.norm(v1, axis=1, keepdims=True)  # (T, 1)
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True)

    # Avoid divide-by-zero; will become NaN after masking
    with np.errstate(invalid="ignore", divide="ignore"):
        u1 = v1 / norm1
        u2 = v2 / norm2
        dot = np.sum(u1 * u2, axis=1)

    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    # Propagate NaN wherever any keypoint is missing
    missing = np.any(np.isnan(kp[:, [i1, i2, i3], :]), axis=(1, 2))
    angle[missing] = np.nan
    return angle


def _dihedral_angle(
    kp: np.ndarray,
    i_axis_prox: int,
    i_axis_dist: int,
    i_ref: int,
    i_moving: int,
) -> np.ndarray:
    """Signed dihedral angle of `i_moving` around axis `i_axis_prox→i_axis_dist`.

    The zero reference is the plane defined by the axis and the vector to `i_ref`.
    Positive angle = right-hand rule around the axis (CCW viewed from distal end).

    Parameters
    ----------
    kp : (T, N, 3) array
    i_axis_prox, i_axis_dist : define the rotation axis
    i_ref    : reference keypoint (defines the zero plane)
    i_moving : keypoint whose angular position is measured

    Returns
    -------
    (T,) array in radians (-π, π]. NaN wherever any input keypoint is NaN.
    """
    axis = kp[:, i_axis_dist, :] - kp[:, i_axis_prox, :]  # (T, 3)
    v_ref = kp[:, i_ref, :] - kp[:, i_axis_prox, :]
    v_mov = kp[:, i_moving, :] - kp[:, i_axis_dist, :]

    with np.errstate(invalid="ignore", divide="ignore"):
        axis_norm = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12)

        # Project onto plane perpendicular to axis
        u_ref = v_ref - np.sum(v_ref * axis_norm, axis=1, keepdims=True) * axis_norm
        u_mov = v_mov - np.sum(v_mov * axis_norm, axis=1, keepdims=True) * axis_norm

        # atan2(cross·axis, dot) for signed angle
        cross = np.cross(u_ref, u_mov)  # (T, 3)
        dot = np.sum(u_ref * u_mov, axis=1)
        cross_dot_axis = np.sum(cross * axis_norm, axis=1)

    angle = np.arctan2(cross_dot_axis, dot)

    missing = np.any(
        np.isnan(kp[:, [i_axis_prox, i_axis_dist, i_ref, i_moving], :]),
        axis=(1, 2),
    )
    angle[missing] = np.nan
    return angle


# ── Public API ─────────────────────────────────────────────────────────────────

def load_walking_frame_indices(
    bout_summary_path: str | Path,
    fly_id: Optional[str] = None,
) -> np.ndarray:
    """Return a sorted array of frame indices belonging to walking bouts.

    Reads a walking_bouts_summary.csv (generated by the walking bout detector)
    and returns every frame index covered by at least one bout.

    Parameters
    ----------
    bout_summary_path : path to walking_bouts_summary.csv
    fly_id : if the CSV contains multiple flies, restrict to this fly_id.
        If None, all bouts in the file are used.

    Returns
    -------
    (M,) int array of frame indices, sorted and deduplicated.
    """
    bouts = pd.read_csv(bout_summary_path)
    if fly_id is not None:
        bouts = bouts[bouts["fly_id"] == fly_id]
        if len(bouts) == 0:
            raise ValueError(
                f"No bouts found for fly_id={fly_id!r} in {bout_summary_path}"
            )

    indices = []
    for _, row in bouts.iterrows():
        # start_frame and end_frame are both inclusive
        indices.append(np.arange(int(row["start_frame"]), int(row["end_frame"]) + 1))

    all_indices = np.unique(np.concatenate(indices))
    print(
        f"[geometric_angles] Walking bouts: {len(bouts)} bouts, "
        f"{len(all_indices)} frames total"
    )
    return all_indices


def load_csv_keypoints(
    csv_path: str | Path,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
    confidence_threshold: float = 0.0,
    walking_bouts_path: Optional[str | Path] = None,
    fly_id: Optional[str] = None,
) -> tuple[np.ndarray, list[str]]:
    """Load a JARVIS/RED data3D.csv into a (T, N, 3) keypoint array.

    When `walking_bouts_path` is provided, only frames that fall within
    walking bouts are returned. This is strongly recommended to avoid
    analysing noisy non-walking frames.

    Parameters
    ----------
    csv_path : path to CSV with two header rows (node_name, coordinate)
    frame_start : first frame index to include (applied before bout filter)
    frame_end   : last frame index (exclusive); None = all frames
    confidence_threshold : keypoints with confidence below this value are set
        to NaN. Set to e.g. 0.5 to mask uncertain predictions.
    walking_bouts_path : optional path to walking_bouts_summary.csv.
        When given, only frames inside walking bouts are loaded.
    fly_id : if walking_bouts_path covers multiple flies, restrict to this one.

    Returns
    -------
    kp : (T, N, 3) float array, NaN for missing/low-confidence keypoints
    node_names : list of N keypoint names (order matches axis 1 of kp)
    """
    df = pd.read_csv(csv_path, header=[0, 1])

    # Flatten multi-level columns: (NodeName, x) → "NodeName_x"
    df.columns = [
        "_".join(str(c).strip() for c in col) if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]

    # Apply frame_start/end first
    df = df.iloc[frame_start:frame_end]

    # Filter to walking bout frames if requested
    if walking_bouts_path is not None:
        bout_indices = load_walking_frame_indices(walking_bouts_path, fly_id=fly_id)
        # Intersect with the frame_start/end range
        valid_global = bout_indices[
            (bout_indices >= frame_start)
            & (bout_indices < (frame_end if frame_end is not None else len(df) + frame_start))
        ]
        # Convert to local (0-based) indices within the sliced df
        local_indices = valid_global - frame_start
        local_indices = local_indices[local_indices < len(df)]
        df = df.iloc[local_indices]
        print(f"[geometric_angles] After walking-bout filter: {len(df)} frames")

    df = df.reset_index(drop=True)

    # Extract ordered node names from _x columns
    xyz_cols = [c for c in df.columns if c.endswith("_x")]
    node_names = [c[:-2] for c in xyz_cols]  # strip "_x"

    T = len(df)
    N = len(node_names)
    kp = np.full((T, N, 3), np.nan, dtype=np.float64)

    for i, name in enumerate(node_names):
        for j, coord in enumerate(["x", "y", "z"]):
            col = f"{name}_{coord}"
            if col in df.columns:
                kp[:, i, j] = df[col].to_numpy(dtype=float)

    # Mask low-confidence keypoints
    if confidence_threshold > 0.0:
        for i, name in enumerate(node_names):
            conf_col = f"{name}_confidence"
            if conf_col in df.columns:
                low_conf = df[conf_col].to_numpy(dtype=float) < confidence_threshold
                kp[low_conf, i, :] = np.nan

    return kp, node_names


def compute_geometric_angles(
    kp: np.ndarray,
    node_names: list[str],
) -> pd.DataFrame:
    """Compute all geometric joint angles for every frame.

    Uses `ANGLE_DEFINITIONS` to compute flex and dihedral angles from 3D
    keypoint positions. No body model or joint limits are applied.

    Parameters
    ----------
    kp : (T, N, 3) array of 3D keypoints (NaN for missing)
    node_names : list of N keypoint names (must include the nodes referenced
        in ANGLE_DEFINITIONS; extra nodes are ignored)

    Returns
    -------
    DataFrame of shape (T, n_angles) with angle names as columns.
    All values are in radians. Rows are NaN wherever required keypoints
    are missing.
    """
    name2idx = {name: i for i, name in enumerate(node_names)}
    missing_nodes: set[str] = set()
    results = {}

    for ang_name, kind, nodes, xml_class, desc in ANGLE_DEFINITIONS:
        # Check that all required nodes are available
        if any(n not in name2idx for n in nodes):
            unavailable = [n for n in nodes if n not in name2idx]
            missing_nodes.update(unavailable)
            results[ang_name] = np.full(len(kp), np.nan)
            continue

        idxs = [name2idx[n] for n in nodes]

        if kind == "flex":
            results[ang_name] = _flex_angle(kp, *idxs)
        elif kind == "dihedral":
            results[ang_name] = _dihedral_angle(kp, *idxs)
        else:
            raise ValueError(f"Unknown angle kind: {kind!r}")

    if missing_nodes:
        print(
            f"[geometric_angles] Warning: {len(missing_nodes)} nodes not found in "
            f"node_names; affected angles set to NaN: {sorted(missing_nodes)}"
        )

    return pd.DataFrame(results)


def get_body_frame_keypoints(
    kp: np.ndarray,
    node_names: list[str],
) -> np.ndarray:
    """Rotate keypoints into an approximate body-aligned frame.

    The body frame is defined per frame using:
      - origin: Scutellum
      - X axis: Scutellum → Abd_A4 (anterior-posterior, pointing posterior)
      - Z axis: cross(X, Scutellum→EyeL) (dorsal)
      - Y axis: cross(Z, X) (lateral, right-hand rule)

    This transforms world-frame keypoints into a frame that is stable with
    respect to the fly's body orientation, making dihedral angles more
    directly comparable to STAC's body-frame DOFs.

    Parameters
    ----------
    kp : (T, N, 3) world-frame keypoints
    node_names : list of N node names

    Returns
    -------
    (T, N, 3) body-frame keypoints
    """
    n2i = {n: i for i, n in enumerate(node_names)}
    scut = kp[:, n2i["Scutellum"], :]      # (T, 3)
    abd = kp[:, n2i["Abd_A4"], :]
    eyeleft = kp[:, n2i["EyeL"], :]

    x_ax = abd - scut
    x_ax /= np.linalg.norm(x_ax, axis=1, keepdims=True) + 1e-12

    tmp = eyeleft - scut
    z_ax = np.cross(x_ax, tmp)
    z_ax /= np.linalg.norm(z_ax, axis=1, keepdims=True) + 1e-12

    y_ax = np.cross(z_ax, x_ax)
    y_ax /= np.linalg.norm(y_ax, axis=1, keepdims=True) + 1e-12

    # Rotation matrix: rows are body axes in world coords → R.T rotates world→body
    # R[t] = [[x_ax], [y_ax], [z_ax]] as (3,3); apply as kp_body = (kp - scut) @ R.T
    T, N, _ = kp.shape
    R = np.stack([x_ax, y_ax, z_ax], axis=1)   # (T, 3, 3) — rows are body axes

    kp_centered = kp - scut[:, np.newaxis, :]   # (T, N, 3)
    # R[t] @ v rotates world→body (R has body-axes as rows, so it's already R^T)
    kp_body = np.einsum("tij,tnj->tni", R, kp_centered)
    return kp_body


def suggest_xml_ranges(
    angles_df: pd.DataFrame,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
    margin_rad: float = 0.05,
) -> pd.DataFrame:
    """Compare observed geometric angle distributions to current XML limits.

    For dihedral angles (which are in the same signed-radian space as STAC
    joint angles), the comparison is direct. For flex angles (which are in
    [0, π] regardless of neutral pose), only the SPAN of observed motion is
    compared to the XML range span — a wider observed span suggests the XML
    limits may be too tight.

    Parameters
    ----------
    angles_df : output of `compute_geometric_angles()`
    lo_pct, hi_pct : percentiles defining the "observed" range (after dropping NaN)
    margin_rad : safety margin added to each side of the suggested range

    Returns
    -------
    DataFrame with one row per angle, columns:
      angle_name, kind, xml_class,
      observed_lo, observed_hi, observed_span,
      xml_lo, xml_hi, xml_span,
      suggested_lo, suggested_hi,  (only for dihedral angles)
      span_ratio,                  (observed_span / xml_span; > 1 → potentially clipped)
      potentially_clipped
    """
    rows = []
    for ang_name, kind, nodes, xml_class, desc in ANGLE_DEFINITIONS:
        if ang_name not in angles_df.columns:
            continue

        vals = angles_df[ang_name].dropna().to_numpy()
        if len(vals) < 10:
            continue

        obs_lo = float(np.percentile(vals, lo_pct))
        obs_hi = float(np.percentile(vals, hi_pct))
        obs_span = obs_hi - obs_lo

        xml_lo, xml_hi = XML_JOINT_RANGES.get(xml_class, (np.nan, np.nan))
        xml_span = xml_hi - xml_lo if not np.isnan(xml_lo) else np.nan

        # For dihedral angles: direct comparison to signed XML range
        if kind == "dihedral" and not np.isnan(xml_lo):
            sugg_lo = round(obs_lo - margin_rad, 3)
            sugg_hi = round(obs_hi + margin_rad, 3)
        else:
            sugg_lo = sugg_hi = np.nan

        span_ratio = obs_span / xml_span if not np.isnan(xml_span) and xml_span > 0 else np.nan
        potentially_clipped = bool(span_ratio > 1.0) if not np.isnan(span_ratio) else False

        rows.append(
            dict(
                angle_name=ang_name,
                kind=kind,
                xml_class=xml_class,
                observed_lo=round(obs_lo, 4),
                observed_hi=round(obs_hi, 4),
                observed_span=round(obs_span, 4),
                xml_lo=xml_lo,
                xml_hi=xml_hi,
                xml_span=round(xml_span, 4) if not np.isnan(xml_span) else np.nan,
                suggested_lo=sugg_lo,
                suggested_hi=sugg_hi,
                span_ratio=round(span_ratio, 3) if not np.isnan(span_ratio) else np.nan,
                potentially_clipped=potentially_clipped,
            )
        )

    return pd.DataFrame(rows)


def update_xml_ranges(
    xml_path: str | Path,
    suggestions: pd.DataFrame,
    dry_run: bool = True,
    backup: bool = True,
) -> None:
    """Apply suggested joint ranges to the MuJoCo XML file.

    Only dihedral-based suggestions (where suggested_lo/hi are not NaN) are
    applied, since these are in the same signed-radian space as STAC DOFs.

    Parameters
    ----------
    xml_path : path to fruitfly_force_free.xml (or similar)
    suggestions : output of `suggest_xml_ranges()`
    dry_run : if True, print a unified diff but do NOT write the file
    backup : if True (and dry_run=False), save a .xml.bak before writing
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Build map: class_name → joint element (searching recursively)
    class_to_joint: dict[str, ET.Element] = {}

    def _collect_defaults(elem: ET.Element) -> None:
        cls = elem.get("class")
        if cls is not None:
            for child in elem:
                if child.tag == "joint" and "range" in child.attrib:
                    class_to_joint[cls] = child
            for child in elem:
                if child.tag == "default":
                    _collect_defaults(child)

    for defaults_block in root.iter("default"):
        if defaults_block.get("class") is None:
            for child in defaults_block:
                if child.tag == "default":
                    _collect_defaults(child)
        else:
            _collect_defaults(defaults_block)

    # Collect proposed edits
    edits: list[tuple[str, str, str]] = []  # (xml_class, old_range, new_range)
    for _, row in suggestions.iterrows():
        if pd.isna(row.get("suggested_lo")) or pd.isna(row.get("suggested_hi")):
            continue
        cls = row["xml_class"]
        if cls not in class_to_joint:
            print(f"  [update_xml_ranges] class {cls!r} not found in XML — skipping")
            continue
        joint_elem = class_to_joint[cls]
        old_range = joint_elem.get("range", "")
        new_range = f"{row['suggested_lo']:.3f} {row['suggested_hi']:.3f}"
        if old_range != new_range:
            edits.append((cls, old_range, new_range))

    if not edits:
        print("[update_xml_ranges] No changes needed — all suggested ranges match current XML.")
        return

    print(f"[update_xml_ranges] {'DRY RUN — ' if dry_run else ''}Proposed range changes:\n")
    print(f"  {'XML class':<30}  {'current':<18}  {'suggested'}")
    print("  " + "-" * 70)
    for cls, old, new in edits:
        print(f"  {cls:<30}  {old:<18}  {new}")

    if dry_run:
        print(
            "\n[update_xml_ranges] DRY RUN: no file written. "
            "Call with dry_run=False to apply."
        )
        return

    # Apply edits
    for cls, old_range, new_range in edits:
        class_to_joint[cls].set("range", new_range)

    # Serialise back to string (ET doesn't preserve formatting perfectly, but is safe)
    orig_text = xml_path.read_text()
    ET.indent(tree, space="  ")
    new_text = ET.tostring(root, encoding="unicode", xml_declaration=False)
    new_text = '<?xml version="1.0" encoding="utf-8"?>\n' + new_text

    # Unified diff for reference
    diff = list(
        difflib.unified_diff(
            orig_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=str(xml_path),
            tofile=str(xml_path) + " (updated)",
        )
    )
    if diff:
        print("\nUnified diff:")
        print("".join(diff[:80]))
        if len(diff) > 80:
            print(f"  ... ({len(diff) - 80} more lines)")

    if backup:
        bak = xml_path.with_suffix(".xml.bak")
        shutil.copy2(xml_path, bak)
        print(f"\n[update_xml_ranges] Backup saved to: {bak}")

    xml_path.write_text(new_text, encoding="utf-8")
    print(f"[update_xml_ranges] Written: {xml_path}")
