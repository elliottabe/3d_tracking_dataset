#!/usr/bin/env python3
"""
Preprocess 3D keypoint data for STAC IK solver.

This script:
1. Loads CSV keypoint data and skeleton definition
2. Matches CSV keypoints to skeleton nodes
3. Reorders to match MuJoCo XML site order (required by STAC)
4. Optionally applies Procrustes alignment and scaling
5. Saves preprocessed data to HDF5 format

Usage:
    python preprocess_keypoints_for_ik.py paths=workstation dataset=free_walking
    python preprocess_keypoints_for_ik.py paths=hyak dataset=courtship preprocessing.apply_alignment=false
    python preprocess_keypoints_for_ik.py paths=workstation preprocessing.frame_start=100 preprocessing.frame_end=300
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# Note: jax_persistent_cache_enable_xla_caches may not be available in all JAX versions
try:
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
except AttributeError:
    pass  # Skip if not available in this JAX version

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import jax.numpy as jnp
import mujoco
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Import utilities
try:
    import utils.io_dict_to_hdf5 as ioh5
    from utils.path_utils import load_config_with_path_template, convert_dict_to_path
    from utils.optimized_floor_alignment import jit_vectorized_procrustes_with_scaling
    from utils.io import (
        match_csv_to_skeleton,
        reorder_keypoints_array,
        reorder_skeleton_edges)
except ModuleNotFoundError as e:
    print(f"Error: Could not import utilities. Make sure you're running from the project root.")
    print(f"Current directory: {Path.cwd()}")
    print(f"Project root: {project_root}")
    raise


# Helper functions (extracted from notebook workflow)
# Note: match_csv_to_skeleton, reorder_keypoints_array, and reorder_skeleton_edges
# are now imported from utils.io

def load_csv_data(csv_path: Path, frame_indices: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load CSV keypoint data.
    
    Args:
        csv_path: Path to CSV file with multi-level header (node_name, coordinate)
        frame_indices: Optional array of frame indices to keep (for selecting specific bouts)
    
    Returns:
        df: DataFrame with xyz columns
        csv_kp_names: List of keypoint names from CSV
    """
    print(f"Loading CSV data from: {csv_path}")
    
    # Load CSV with multi-level header
    df = pd.read_csv(csv_path, header=[0, 1])
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
    
    # Extract xyz columns
    xyz_columns = [col for col in df.columns if col.endswith(('_x', '_y', '_z'))]
    kp_data = df[xyz_columns]
    
    # Select frame subset if specified
    if frame_indices is not None:
        print(f"Selecting {len(frame_indices)} frames from data")
        kp_data = kp_data.iloc[frame_indices]
    
    # Get unique CSV keypoint names (without _x, _y, _z suffix)
    csv_kp_names = []
    for col in xyz_columns:
        if col.endswith('_x'):
            csv_kp_names.append(col[:-2])
    
    print(f"Found {len(csv_kp_names)} keypoints in CSV: {csv_kp_names[:5]}...")
    print(f"Data shape: {kp_data.shape}")
    
    return kp_data, csv_kp_names


def load_skeleton(skeleton_path: Path) -> Tuple[Dict, np.ndarray]:
    """
    Load skeleton definition from JSON.
    
    Args:
        skeleton_path: Path to skeleton JSON file
    
    Returns:
        skeleton: Skeleton dictionary with node_names and edges
        edges: Edge array (N_edges, 2)
    """
    print(f"Loading skeleton from: {skeleton_path}")
    
    with open(skeleton_path, 'r') as f:
        skeleton = json.load(f)
    
    edges = np.array(skeleton['edges'])
    print(f"Skeleton: {len(skeleton['node_names'])} nodes, {len(edges)} edges")
    
    return skeleton, edges


def match_and_filter_skeleton(csv_kp_names: List[str], 
                               skeleton: Dict,
                               edges: np.ndarray) -> Tuple[List[str], np.ndarray, Dict]:
    """
    Match CSV keypoints to skeleton nodes and filter skeleton.
    
    Args:
        csv_kp_names: List of keypoint names from CSV
        skeleton: Skeleton dictionary
        edges: Original skeleton edges
    
    Returns:
        filtered_node_names: List of matched node names in filtered order
        filtered_edges: Filtered edge array with remapped indices
        csv_to_filtered_idx: Mapping from CSV name to filtered index
    """
    print("\nMatching CSV keypoints to skeleton nodes...")
    
    csv_to_skel_map, unmatched = match_csv_to_skeleton(csv_kp_names, skeleton['node_names'])
    print(f"\nMatched {len(csv_to_skel_map)}/{len(csv_kp_names)} CSV keypoints to skeleton nodes:")
    for csv_name, (skel_idx, skel_name) in sorted(csv_to_skel_map.items(), key=lambda x: x[1][0]):
        match_symbol = '✓' if csv_name == skel_name else '~'
        print(f"  {match_symbol} CSV '{csv_name}' -> Skeleton[{skel_idx:2d}] '{skel_name}'")

    if unmatched:
        print(f"⚠ Unmatched: {unmatched}")
    
    # Filter skeleton to matched nodes
    matched_skel_indices = sorted([idx for idx, name in csv_to_skel_map.values()])
    filtered_node_names = [skeleton['node_names'][idx] for idx in matched_skel_indices]
    
    # Create index mapping: old skeleton index -> new filtered index
    old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(matched_skel_indices)}
    
    # Filter edges
    filtered_edges = []
    for edge in edges:
        start_idx, end_idx = edge
        if start_idx in old_to_new_idx and end_idx in old_to_new_idx:
            filtered_edges.append([old_to_new_idx[start_idx], old_to_new_idx[end_idx]])
    filtered_edges = np.array(filtered_edges)
    
    # Create CSV to filtered index mapping
    csv_to_filtered_idx = {}
    for csv_name, (old_skel_idx, skel_name) in csv_to_skel_map.items():
        if old_skel_idx in old_to_new_idx:
            csv_to_filtered_idx[csv_name] = old_to_new_idx[old_skel_idx]
    
    print(f"Filtered skeleton: {len(filtered_node_names)} nodes, {len(filtered_edges)} edges")
    
    return filtered_node_names, filtered_edges, csv_to_filtered_idx


def reorder_csv_to_skeleton(kp_data: pd.DataFrame,
                             csv_kp_names: List[str],
                             csv_to_filtered_idx: Dict,
                             filtered_node_names: List[str]) -> np.ndarray:
    """
    Reorder CSV data to match filtered skeleton node order.
    
    Args:
        kp_data: DataFrame with xyz columns
        csv_kp_names: List of CSV keypoint names
        csv_to_filtered_idx: Mapping from CSV name to filtered skeleton index
        filtered_node_names: List of filtered node names in order
    
    Returns:
        kp_array: Reordered keypoint array (T, N, 3)
    """
    print("\nReordering CSV data to match skeleton...")
    
    # Create reordered column list
    reordered_cols = [''] * len(filtered_node_names) * 3
    for csv_name, new_idx in csv_to_filtered_idx.items():
        reordered_cols[new_idx * 3] = f"{csv_name}_x"
        reordered_cols[new_idx * 3 + 1] = f"{csv_name}_y"
        reordered_cols[new_idx * 3 + 2] = f"{csv_name}_z"
    
    # Reorder dataframe
    kp_data_reordered = kp_data[reordered_cols]
    kp_array = np.array(kp_data_reordered.values).reshape(-1, len(filtered_node_names), 3)
    
    print(f"Reordered data shape: {kp_array.shape}")
    
    return kp_array


def match_to_mujoco_sites(filtered_node_names: List[str], 
                           xml_path: Path) -> Tuple[Dict, object, List[str]]:
    """
    Match filtered skeleton nodes to MuJoCo tracking sites.
    
    Args:
        filtered_node_names: List of filtered node names
        xml_path: Path to MuJoCo XML file
    
    Returns:
        skeleton_to_mujoco: Mapping from node name to MuJoCo site index
        mj_model: Compiled MuJoCo model
        all_site_names: List of all site names in model
    """
    print(f"\nLoading MuJoCo model from: {xml_path}")
    
    spec = mujoco.MjSpec.from_file(str(xml_path))
    mj_model = spec.compile()
    
    # Extract ALL site names and TRACKING site names separately
    all_site_names = [site.name for site in spec.sites]
    
    # CRITICAL FIX: Only match against tracking sites, not aligned sites
    tracking_site_names = [name for name in all_site_names if 'tracking[' in name]
    tracking_names_clean = [name.replace('tracking[', '').replace(']', '') for name in tracking_site_names]
    
    print(f"Found {len(all_site_names)} total sites in model ({len(tracking_site_names)} tracking sites)")
    
    # Match skeleton nodes to tracking sites
    skeleton_to_mujoco = {}
    matched_count = 0
    unmatched = []
    
    for node_name in filtered_node_names:
        if node_name in tracking_names_clean:
            # Find position in tracking list
            tracking_idx = tracking_names_clean.index(node_name)
            # Get the actual site index in the full model
            actual_site_name = tracking_site_names[tracking_idx]
            actual_site_idx = all_site_names.index(actual_site_name)
            skeleton_to_mujoco[node_name] = actual_site_idx
            matched_count += 1
        else:
            unmatched.append(node_name)
    
    print(f"Matched {matched_count}/{len(filtered_node_names)} nodes to MuJoCo sites")
    if unmatched:
        print(f"⚠ Unmatched nodes: {unmatched}")
    
    return skeleton_to_mujoco, mj_model, all_site_names


def reorder_to_xml_site_order(kp_array: np.ndarray,
                               filtered_node_names: List[str],
                               filtered_edges: np.ndarray,
                               skeleton_to_mujoco: Dict) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Reorder keypoints by XML site index (required for STAC IK).
    
    Args:
        kp_array: Keypoint array (T, N, 3) in filtered skeleton order
        filtered_node_names: List of filtered node names
        filtered_edges: Filtered skeleton edges
        skeleton_to_mujoco: Mapping from node name to site index
    
    Returns:
        kp_array_xml_order: Keypoint array reordered by XML site index
        xml_node_names_ordered: Node names in XML order
        xml_edges: Edges with indices remapped to XML order
    """
    print("\nReordering to XML site index order (STAC requirement)...")
    
    # Sort by site index
    xml_nodes_sorted = sorted(skeleton_to_mujoco.items(), key=lambda x: x[1])
    xml_node_names_ordered = [name for name, site_idx in xml_nodes_sorted]
    
    print("XML order (by site index):")
    for i, (name, site_idx) in enumerate(xml_nodes_sorted[:10]):
        print(f"  {i}: {name:15s} (site {site_idx})")
    if len(xml_nodes_sorted) > 10:
        print(f"  ... ({len(xml_nodes_sorted) - 10} more)")
    
    # Reorder keypoint array
    kp_array_xml_order, _ = reorder_keypoints_array(
        kp_array,
        filtered_node_names,
        xml_node_names_ordered
    )
    
    # Reorder edges
    xml_edges = reorder_skeleton_edges(
        filtered_edges,
        filtered_node_names,
        xml_node_names_ordered
    )
    
    print(f"Final shape: {kp_array_xml_order.shape}")
    
    return kp_array_xml_order, xml_node_names_ordered, xml_edges


def apply_procrustes_alignment(kp_array: np.ndarray,
                                mj_model: object,
                                xml_node_names: List[str],
                                skeleton_to_mujoco: Dict,
                                exclude_indices: Optional[np.ndarray] = None,
                                apply_scaling: bool = True,
                                preserve_translation: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Apply Procrustes scaling to match MuJoCo model's size.
    
    This function extracts the reference pose from the MuJoCo model at its rest
    configuration and scales the keypoint data to match the model's dimensions.
    
    IMPORTANT: This preserves the original position and orientation of the data,
    only adjusting the scale to match the model. This ensures:
    - Keypoints maintain their original spatial relationships and motion dynamics
    - Data is sized correctly to match the physical model dimensions
    - Original global position and orientation are preserved for analysis
    
    The transformation parameters (rotation, scale, translation) are computed and
    saved so the alignment can be inverted or reapplied if needed.
    
    Args:
        kp_array: Keypoint array (T, N, 3) in XML order
        mj_model: MuJoCo model (reference pose extracted from rest configuration)
        xml_node_names: Node names in XML order
        skeleton_to_mujoco: Mapping from node name to site index
        exclude_indices: Keypoint indices to exclude from alignment computation (e.g., wings, antenna)
        apply_scaling: Whether to apply scaling (should be True for size matching)
        preserve_translation: Whether to preserve original translation (position/orientation)
    Returns:
        aligned_kp: Scaled keypoint array (T, N, 3) with original position/orientation
        alignment_info: Dictionary with transformation parameters (rotation, scale, translation)
    """
    print(f"\nApplying Procrustes alignment (scaling={apply_scaling})...")
    
    # Get reference pose from MuJoCo model's rest configuration
    # This ensures keypoints are scaled and positioned to match the model
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)  # Compute positions at rest pose
    
    # Extract tracking site positions in the same order as keypoints
    site_subset = []
    missing_sites = []
    for name in xml_node_names:
        if name in skeleton_to_mujoco:
            site_subset.append(skeleton_to_mujoco[name])
        else:
            missing_sites.append(name)
    
    if missing_sites:
        print(f"⚠ WARNING: {len(missing_sites)} keypoints not found in skeleton_to_mujoco mapping:")
        for name in missing_sites:
            print(f"   - {name}")
    
    if len(site_subset) != len(xml_node_names):
        raise ValueError(f"Reference pose extraction failed: expected {len(xml_node_names)} sites, got {len(site_subset)}")
    
    ref_pose = mj_data.site_xpos[site_subset]
    
    ref_center = np.mean(ref_pose, axis=0)
    ref_span = np.max(ref_pose, axis=0) - np.min(ref_pose, axis=0)
    print(f"Reference pose shape: {ref_pose.shape}")
    print(f"Reference center: {ref_center}")
    print(f"Reference span: {ref_span}")
    
    # Convert to JAX arrays
    kp_jax = jnp.array(kp_array)
    ref_pose_jax = jnp.array(ref_pose)

    # Apply alignment
    if exclude_indices is not None:
        print(f"Excluding {len(exclude_indices)} keypoints from alignment computation")
    
    # Apply Procrustes scaling to match model size
    # preserve_translation=True keeps original position/orientation (only scales)
    # use_clip_average=True computes transformation from average pose (temporal consistency)
    # NOTE: Pass data directly - Procrustes handles centering internally when preserve_translation=True
    if preserve_translation:
        aligned_kp, procrustes_info = jit_vectorized_procrustes_with_scaling(
            kp_jax,
            ref_pose_jax,
            use_clip_average=True,
            exclude_indices=exclude_indices,
            preserve_translation=True  # Let Procrustes handle centering internally
        )
        
        # Apply only the scale to original keypoints (Procrustes already computed correct scale)
        aligned_kp = kp_jax * procrustes_info['scales'][:, None, None]  # Scale only
    else: 
        aligned_kp, procrustes_info = jit_vectorized_procrustes_with_scaling(
            kp_jax,
            ref_pose_jax,
            use_clip_average=True,
            exclude_indices=exclude_indices,
            preserve_translation=False  # Allow full transformation (rotation, scale, translation)
        )
    # Convert back to numpy
    aligned_kp = np.array(aligned_kp)
    
    # Extract alignment info
    alignment_info = {
        'scales': float(procrustes_info['scales'][0]) if apply_scaling else 1.0,
        'rotation': np.array(procrustes_info['rotations'][0]),
        'translation': np.array(procrustes_info['translations'][0]),
    }
    
    # Only add exclude_indices if it exists (avoid None in HDF5)
    if exclude_indices is not None:
        alignment_info['exclude_indices'] = exclude_indices.tolist()
    
    print(f"Scale factor applied: {alignment_info['scales']:.4f}")
    
    # Check scaling quality
    orig_center = np.mean(kp_array, axis=(0, 1))
    scaled_center = np.mean(aligned_kp, axis=(0, 1))
    data_span = np.max(aligned_kp[0], axis=0) - np.min(aligned_kp[0], axis=0)
    scale_ratio = data_span / ref_span
    print(f"Original data center: {orig_center}")
    print(f"Scaled data center: {scaled_center} (should match original)")
    print(f"Data/Model span ratio: {scale_ratio} (should be ~1.0)")
    
    # Return the scaled keypoints (original position/orientation preserved)
    return aligned_kp, alignment_info


def load_bouts_from_csv(bouts_csv_path: Path) -> List[Dict]:
    """
    Load bout information from CSV file.
    
    Expected CSV columns:
        - bout_idx: Bout identifier
        - start_frame: Start frame index
        - end_frame: End frame index (inclusive)
    
    Args:
        bouts_csv_path: Path to CSV file with bout information
    
    Returns:
        List of dicts with keys: 'bout_idx', 'start_frame', 'end_frame'
    """
    print(f"\nLoading bout information from: {bouts_csv_path}")
    
    df = pd.read_csv(bouts_csv_path)
    
    # Check required columns
    required_cols = ['bout_idx', 'start_frame', 'end_frame']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    # Convert to list of dicts
    bouts = []
    for _, row in df.iterrows():
        bout_info = {
            'bout_idx': int(row['bout_idx']-1),
            'start_frame': int(row['start_frame']),
            'end_frame': int(row['end_frame'])
        }
        bouts.append(bout_info)
    
    print(f"Loaded {len(bouts)} bouts")
    print(f"  First bout: idx={bouts[0]['bout_idx']}, frames {bouts[0]['start_frame']}-{bouts[0]['end_frame']}")
    if len(bouts) > 1:
        print(f"  Last bout:  idx={bouts[-1]['bout_idx']}, frames {bouts[-1]['start_frame']}-{bouts[-1]['end_frame']}")
    
    return bouts


def load_concatenated_bouts(csv_path: Path, 
                            bouts: List[Dict],
                            csv_kp_names: List[str],
                            csv_to_filtered_idx: Dict,
                            filtered_node_names: List[str]) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load multiple bouts as a single concatenated array for efficient batch processing.
    
    Args:
        csv_path: Path to CSV file with keypoint data
        bouts: List of bout information dicts
        csv_kp_names: List of CSV keypoint names
        csv_to_filtered_idx: Mapping from CSV name to filtered skeleton index
        filtered_node_names: List of filtered node names in order
    
    Returns:
        concatenated_kp: Concatenated keypoint array (T_total, N, 3)
        clip_info: List of dicts with 'bout_idx', 'start_idx', 'end_idx' for each bout
    """
    print(f"\nLoading {len(bouts)} bouts as concatenated array...")
    
    # Load full CSV once
    df = pd.read_csv(csv_path, header=[0, 1])
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
    xyz_columns = [col for col in df.columns if col.endswith(('_x', '_y', '_z'))]
    kp_data = df[xyz_columns]
    
    # Create reordered column list (from reorder_csv_to_skeleton)
    reordered_cols = [''] * len(filtered_node_names) * 3
    for csv_name, new_idx in csv_to_filtered_idx.items():
        reordered_cols[new_idx * 3] = f"{csv_name}_x"
        reordered_cols[new_idx * 3 + 1] = f"{csv_name}_y"
        reordered_cols[new_idx * 3 + 2] = f"{csv_name}_z"
    
    # Extract and concatenate bout data
    bout_arrays = []
    clip_info = []
    current_idx = 0
    
    for bout in bouts:
        frame_indices = np.arange(bout['start_frame'], bout['end_frame'])
        bout_data = kp_data.iloc[frame_indices][reordered_cols]
        bout_array = np.array(bout_data.values).reshape(-1, len(filtered_node_names), 3)
        
        bout_arrays.append(bout_array)
        
        clip_info.append({
            'bout_idx': bout['bout_idx'],
            'start_idx': current_idx,
            'end_idx': current_idx + len(bout_array)
        })
        current_idx += len(bout_array)
    
    concatenated_kp = np.concatenate(bout_arrays, axis=0)
    
    print(f"Concatenated shape: {concatenated_kp.shape}")
    print(f"  Total frames: {concatenated_kp.shape[0]}")
    print(f"  Bouts: {len(bouts)}")
    
    return concatenated_kp, clip_info


def save_to_hdf5(output_path: Path,
                 kp_data: np.ndarray,
                 orig_kp_data: np.ndarray,
                 xml_node_names: List[str],
                 xml_edges: np.ndarray,
                 alignment_info: Optional[Dict] = None):
    """
    Save preprocessed data to HDF5 format.
    
    Args:
        output_path: Path to output HDF5 file
        kp_data: Preprocessed keypoint array (T, N, 3)
        orig_kp_data: Original (unaligned) keypoint array
        xml_node_names: Node names in XML order
        xml_edges: Skeleton edges in XML order
        alignment_info: Optional alignment information
    """
    print(f"\nSaving to HDF5: {output_path}")
    
    data_dict = {
        'keypoints': kp_data,
        'orig_keypoints': orig_kp_data,
        'kp_names': xml_node_names,
        'skeleton_edges': xml_edges,
    }
    
    if alignment_info is not None:
        data_dict['alignment_info'] = alignment_info
    
    ioh5.save(output_path, data_dict)
    
    print(f"✓ Saved preprocessed data:")
    print(f"  - keypoints: {kp_data.shape}")
    print(f"  - kp_names: {len(xml_node_names)}")
    print(f"  - skeleton_edges: {xml_edges.shape}")


def process_single_bout(csv_path: Path,
                        skeleton_path: Path,
                        xml_path: Path,
                        frame_start: Optional[int] = None,
                        frame_end: Optional[int] = None,
                        apply_alignment: bool = False,
                        apply_scaling: bool = False,
                        exclude_antenna: bool = False,
                        exclude_wings: bool = False) -> Optional[Dict]:
    """
    Process a single bout of keypoint data.
    
    Args:
        csv_path: Path to CSV file with keypoint data
        skeleton_path: Path to skeleton JSON file
        xml_path: Path to MuJoCo XML file
        frame_start: Start frame index (None for all frames)
        frame_end: End frame index (None for all frames)
        apply_alignment: Whether to apply Procrustes alignment
        apply_scaling: Whether to apply scaling during alignment
        exclude_antenna: Whether to exclude antenna from alignment
        exclude_wings: Whether to exclude wings from alignment
    
    Returns:
        Dictionary with bout data if successful, None otherwise
    """
    try:
        # 1. Load CSV data
        frame_indices = None
        if frame_start is not None and frame_end is not None:
            frame_indices = np.arange(frame_start, frame_end)
            print(f"\nExtracting frames {frame_start} to {frame_end}")
        
        kp_data_df, csv_kp_names = load_csv_data(csv_path, frame_indices)
        
        # 2. Load skeleton
        skeleton, edges = load_skeleton(skeleton_path)
        
        # 3. Match and filter
        filtered_node_names, filtered_edges, csv_to_filtered_idx = match_and_filter_skeleton(
            csv_kp_names, skeleton, edges
        )

        # 4. Reorder CSV to skeleton order
        kp_array = reorder_csv_to_skeleton(
            kp_data_df, csv_kp_names, csv_to_filtered_idx, filtered_node_names
        )
        orig_kp_array = kp_array.copy()  # Keep original for reference
        
        # 5. Match to MuJoCo sites
        skeleton_to_mujoco, mj_model, all_site_names = match_to_mujoco_sites(
            filtered_node_names, xml_path
        )
        
        # 6. Reorder to XML site order (REQUIRED FOR STAC)
        kp_array_xml, xml_node_names, xml_edges = reorder_to_xml_site_order(
            kp_array, filtered_node_names, filtered_edges, skeleton_to_mujoco
        )
        # IMPORTANT: Copy AFTER reordering so orig_keypoints are in same order as keypoints
        orig_kp_xml = kp_array_xml.copy()
        
        # 7. Optional: Apply Procrustes alignment
        alignment_info = None
        if apply_alignment:
            # Determine which keypoints to exclude from alignment
            exclude_indices = []
            if exclude_antenna:
                # Antenna is typically index 0 after reordering
                antenna_idx = [i for i, name in enumerate(xml_node_names) if 'Antenna' in name]
                exclude_indices.extend(antenna_idx)
            
            if exclude_wings:
                # Wing keypoints
                wing_idx = [i for i, name in enumerate(xml_node_names) if 'Wing' in name]
                exclude_indices.extend(wing_idx)
            
            exclude_arr = jnp.array(exclude_indices) if exclude_indices else None
            
            kp_array_xml, alignment_info = apply_procrustes_alignment(
                kp_array_xml, mj_model, xml_node_names, skeleton_to_mujoco,
                exclude_indices=exclude_arr, apply_scaling=apply_scaling, preserve_translation=True
            )
        
        # 8. Build data dictionary
        bout_data = {
            'keypoints': kp_array_xml,
            'orig_keypoints': orig_kp_xml,
            'kp_names': xml_node_names,
            'skeleton_edges': xml_edges,
        }
        
        if alignment_info is not None:
            bout_data['alignment_info'] = alignment_info
        
        print(f"✓ Processed bout data:")
        print(f"  - keypoints: {kp_array_xml.shape}")
        print(f"  - kp_names: {len(xml_node_names)}")
        print(f"  - skeleton_edges: {xml_edges.shape}")
        
        return bout_data
        
    except Exception as e:
        print(f"\n✖ Error processing bout: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_bouts_batch(csv_path: Path,
                       skeleton_path: Path,
                       xml_path: Path,
                       bouts: List[Dict],
                       apply_alignment: bool = False,
                       apply_scaling: bool = False,
                       exclude_antenna: bool = False,
                       exclude_wings: bool = False) -> Optional[Dict]:
    """
    Efficiently process multiple bouts by loading skeleton/model once and 
    processing all data as a concatenated array.
    
    This is much faster than processing each bout individually because:
    - Skeleton and MuJoCo model loaded only once
    - Matching/filtering operations done once
    - JAX operations benefit from batch processing larger arrays
    
    Args:
        csv_path: Path to CSV file with keypoint data
        skeleton_path: Path to skeleton JSON file
        xml_path: Path to MuJoCo XML file
        bouts: List of bout information dicts
        apply_alignment: Whether to apply Procrustes alignment
        apply_scaling: Whether to apply scaling during alignment
        exclude_antenna: Whether to exclude antenna from alignment
        exclude_wings: Whether to exclude wings from alignment
    
    Returns:
        Dictionary with all bout data keyed by 'bout_<idx>' if successful, None otherwise
    """
    try:
        print("\n" + "="*80)
        print("BATCH PREPROCESSING - LOADING COMMON DATA (done once)")
        print("="*80)
        
        # 1. Load CSV and get keypoint names (just for matching)
        print(f"\nLoading CSV header from: {csv_path}")
        df_header = pd.read_csv(csv_path, header=[0, 1], nrows=0)
        df_header.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                            for col in df_header.columns.values]
        xyz_columns = [col for col in df_header.columns if col.endswith(('_x', '_y', '_z'))]
        csv_kp_names = [col[:-2] for col in xyz_columns if col.endswith('_x')]
        print(f"Found {len(csv_kp_names)} keypoints in CSV")
        
        # 2. Load skeleton (once)
        skeleton, edges = load_skeleton(skeleton_path)
        
        # 3. Match and filter (once)
        filtered_node_names, filtered_edges, csv_to_filtered_idx = match_and_filter_skeleton(
            csv_kp_names, skeleton, edges
        )
        
        # 4. Load all bouts as concatenated array
        concatenated_kp, clip_info = load_concatenated_bouts(
            csv_path, bouts, csv_kp_names, csv_to_filtered_idx, filtered_node_names
        )
        orig_concatenated = concatenated_kp.copy()
        
        # 5. Match to MuJoCo sites (once)
        skeleton_to_mujoco, mj_model, all_site_names = match_to_mujoco_sites(
            filtered_node_names, xml_path
        )
        
        # 6. Reorder to XML site order (once for all data)
        print("\nReordering concatenated data to XML site order...")
        kp_array_xml, xml_node_names, xml_edges = reorder_to_xml_site_order(
            concatenated_kp, filtered_node_names, filtered_edges, skeleton_to_mujoco
        )
        # IMPORTANT: Copy AFTER reordering so orig_keypoints match keypoints order
        orig_xml = kp_array_xml.copy()
        
        # 7. Determine which keypoints to exclude from alignment (do this once)
        exclude_indices = []
        if apply_alignment:
            if exclude_antenna:
                antenna_idx = [i for i, name in enumerate(xml_node_names) if 'Antenna' in name]
                exclude_indices.extend(antenna_idx)
                print(f"Excluding antenna keypoints: {antenna_idx}")
            
            if exclude_wings:
                wing_idx = [i for i, name in enumerate(xml_node_names) if 'Wing' in name]
                exclude_indices.extend(wing_idx)
                print(f"Excluding wing keypoints: {wing_idx}")
        
        exclude_arr = jnp.array(exclude_indices) if exclude_indices else None
        
        # 8. Split back into individual bouts and apply alignment PER BOUT
        print("\n" + "="*80)
        print("PROCESSING INDIVIDUAL BOUTS WITH PER-BOUT ALIGNMENT")
        print("="*80)
        
        all_bouts_dict = {}
        for clip in clip_info:
            bout_idx = clip['bout_idx']
            start_idx = clip['start_idx']
            end_idx = clip['end_idx']
            
            # Extract this bout's data
            bout_kp = kp_array_xml[start_idx:end_idx]
            bout_orig = orig_xml[start_idx:end_idx]
            
            # Apply Procrustes alignment PER BOUT (like the notebook does)
            alignment_info = None
            if apply_alignment:
                print(f"\n  Processing bout_{bout_idx:03d} ({bout_kp.shape[0]} frames)...")
                aligned_bout_kp, alignment_info = apply_procrustes_alignment(
                    bout_kp, mj_model, xml_node_names, skeleton_to_mujoco,
                    exclude_indices=exclude_arr, apply_scaling=apply_scaling, preserve_translation=True
                )
            else:
                aligned_bout_kp = bout_kp
            
            bout_data = {
                'keypoints': aligned_bout_kp,
                'orig_keypoints': bout_orig,
                'kp_names': xml_node_names,
                'skeleton_edges': xml_edges,
            }
            
            if alignment_info is not None:
                bout_data['alignment_info'] = alignment_info
            
            all_bouts_dict[f'bout_{bout_idx:03d}'] = bout_data
            
            scale_str = f", scale={alignment_info['scales']:.6f}" if alignment_info else ""
            print(f"  ✓ bout_{bout_idx:03d}: {bout_data['keypoints'].shape[0]} frames{scale_str}")
        
        print(f"\n✓ Successfully split into {len(all_bouts_dict)} bouts")
        
        return all_bouts_dict
        
    except Exception as e:
        print(f"\n✖ Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        return None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main preprocessing function using Hydra configuration.
    
    Usage:
        python preprocess_keypoints_for_ik.py paths=workstation dataset=free_walking
        python preprocess_keypoints_for_ik.py paths=hyak preprocessing.apply_alignment=false
    """
    # Print configuration
    print("=" * 80)
    print("KEYPOINT PREPROCESSING FOR STAC IK")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print()
    
    # Convert path strings to Path objects
    cfg.paths = convert_dict_to_path(cfg.paths)
    
    # Resolve paths (handle both absolute and relative)
    data_dir = Path(cfg.paths.data_dir)
    csv_path = Path(cfg.preprocessing.csv_path)
    if not csv_path.is_absolute():
        csv_path = data_dir / csv_path
    
    skeleton_path = Path(cfg.preprocessing.skeleton_path)
    if not skeleton_path.is_absolute():
        skeleton_path = Path(project_root) / skeleton_path
    
    xml_path = Path(cfg.preprocessing.xml_path)
    if not xml_path.is_absolute():
        xml_path = Path(project_root) / xml_path
    
    output_dir = data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Resolved paths:")
    print(f"  CSV: {csv_path}")
    print(f"  Skeleton: {skeleton_path}")
    print(f"  XML: {xml_path}")
    print(f"  Output dir: {output_dir}")
    print()
    
    # Determine processing mode: batch (from CSV) or single bout
    if cfg.preprocessing.bouts_csv is not None:
        # Batch processing mode - EFFICIENT VERSION
        bouts_csv_path = Path(cfg.preprocessing.bouts_csv)
        if not bouts_csv_path.is_absolute():
            bouts_csv_path = data_dir / bouts_csv_path
        
        if not bouts_csv_path.exists():
            print(f"❌ Error: Bouts CSV file not found: {bouts_csv_path}")
            sys.exit(1)
        
        bouts = load_bouts_from_csv(bouts_csv_path)
        
        print(f"\nProcessing {len(bouts)} bouts in EFFICIENT BATCH mode:")
        print("  ✓ Loading skeleton/model once")
        print("  ✓ Processing all frames together")
        print("  ✓ Splitting back into individual bouts\n")
        
        # Process all bouts efficiently in one go
        all_bouts_dict = process_bouts_batch(
            csv_path=csv_path,
            skeleton_path=skeleton_path,
            xml_path=xml_path,
            bouts=bouts,
            apply_alignment=cfg.preprocessing.apply_alignment,
            apply_scaling=cfg.preprocessing.apply_scaling,
            exclude_antenna=cfg.preprocessing.exclude_antenna,
            exclude_wings=cfg.preprocessing.exclude_wings
        )
        
        if all_bouts_dict is not None:
            # Save all bouts as nested HDF5
            output_path = output_dir / f"{cfg.preprocessing.bout_name}.h5"
            print(f"\nSaving {len(all_bouts_dict)} bouts to: {output_path}")
            ioh5.save(output_path, all_bouts_dict)
            print(f"✓ Saved nested HDF5 with {len(all_bouts_dict)} bouts")
            
            # Summary
            print("\n" + "=" * 80)
            print("BATCH PREPROCESSING COMPLETE")
            print("=" * 80)
            print(f"\n✓ Successfully processed: {len(all_bouts_dict)}/{len(bouts)} bouts")
            print(f"\nOutput saved to: {output_path}")
            print(f"Structure: bout_<idx>/{{keypoints, orig_keypoints, kp_names, skeleton_edges}}")
            print(f"Keypoint order matches STAC config KP_NAMES")
            print(f"Ready for STAC IK solver!")
        else:
            print("\n❌ Batch preprocessing failed")
            sys.exit(1)
        
    else:
        # Single bout mode (original behavior)
        if cfg.preprocessing.frame_start is None or cfg.preprocessing.frame_end is None:
            print("\n⚠ Warning: Processing entire CSV (no frame range specified)")
            print("   Use preprocessing.frame_start and preprocessing.frame_end to extract a specific bout")
            print("   Or use preprocessing.bouts_csv to process multiple bouts\n")
        
        bout_data = process_single_bout(
            csv_path=csv_path,
            skeleton_path=skeleton_path,
            xml_path=xml_path,
            frame_start=cfg.preprocessing.frame_start,
            frame_end=cfg.preprocessing.frame_end,
            apply_alignment=cfg.preprocessing.apply_alignment,
            apply_scaling=cfg.preprocessing.apply_scaling,
            exclude_antenna=cfg.preprocessing.exclude_antenna,
            exclude_wings=cfg.preprocessing.exclude_wings
        )
        
        if bout_data is not None:
            # Wrap in nested dictionary for consistency
            output_dict = {'bout_0': bout_data}
            output_path = output_dir / f"{cfg.preprocessing.bout_name}.h5"
            
            print(f"\nSaving to: {output_path}")
            ioh5.save(output_path, output_dict)
            
            print("\n" + "=" * 80)
            print("✓ PREPROCESSING COMPLETE")
            print("=" * 80)
            print(f"\nOutput saved to: {output_path}")
            print(f"Structure: bout_0/{{keypoints, orig_keypoints, kp_names, skeleton_edges}}")
            print(f"Keypoint order matches STAC config KP_NAMES")
            print(f"Ready for STAC IK solver!")
        else:
            print("\n❌ Preprocessing failed")
            sys.exit(1)


if __name__ == '__main__':
    main()
