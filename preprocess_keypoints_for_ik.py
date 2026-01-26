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
    python preprocess_keypoints_for_ik.py \
        --csv_path /path/to/data3D.csv \
        --skeleton_path data/fly50.json \
        --xml_path assets/fruitfly_v1/fruitfly_v1_free.xml \
        --output_dir /path/to/output \
        --bout_name example_bout_0 \
        --apply_alignment \
        --apply_scaling
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import jax.numpy as jnp
import mujoco

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# Import utilities
try:
    import utils.io_dict_to_hdf5 as ioh5
    from utils.optimized_floor_alignment import jit_vectorized_procrustes_with_scaling
except ModuleNotFoundError as e:
    print(f"Error: Could not import utilities. Make sure you're running from the project root.")
    print(f"Current directory: {Path.cwd()}")
    print(f"Project root: {project_root}")
    raise


# Helper functions (extracted from notebook workflow)
def match_csv_to_skeleton(csv_names: List[str], skeleton_names: List[str]) -> Tuple[Dict, List[str]]:
    """
    Match CSV keypoint names to skeleton node names using exact and fuzzy matching.
    
    Returns:
        matched: Dict mapping csv_name -> (skeleton_index, skeleton_name)
        unmatched: List of unmatched CSV names
    """
    from difflib import SequenceMatcher
    
    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    matched = {}
    unmatched = []
    
    for csv_name in csv_names:
        # Try exact match first
        if csv_name in skeleton_names:
            idx = skeleton_names.index(csv_name)
            matched[csv_name] = (idx, csv_name)
            continue
        
        # Try fuzzy match
        best_match = None
        best_score = 0.0
        
        for skel_idx, skel_name in enumerate(skeleton_names):
            score = similarity(csv_name, skel_name)
            if score > best_score:
                best_score = score
                best_match = (skel_idx, skel_name)
        
        if best_score > 0.8:  # Threshold for fuzzy matching
            matched[csv_name] = best_match
        else:
            unmatched.append(csv_name)
    
    return matched, unmatched


def reorder_keypoints_array(kp_array: np.ndarray, 
                             current_order: List[str], 
                             target_order: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Reorder keypoint array from current order to target order.
    
    Args:
        kp_array: (T, N, 3) array
        current_order: List of current node names
        target_order: List of target node names
    
    Returns:
        reordered_array: (T, N, 3) array in target order
        reordered_names: Node names in target order
    """
    # Create mapping from name to current index
    name_to_idx = {name: i for i, name in enumerate(current_order)}
    
    # Get indices in target order
    reorder_indices = [name_to_idx[name] for name in target_order]
    
    # Reorder array
    reordered_array = kp_array[:, reorder_indices, :]
    
    return reordered_array, target_order


def reorder_skeleton_edges(edges: np.ndarray,
                           current_order: List[str],
                           target_order: List[str]) -> np.ndarray:
    """
    Reorder skeleton edges to match new node indices.
    
    Args:
        edges: (E, 2) array of edge indices in current order
        current_order: List of current node names
        target_order: List of target node names
    
    Returns:
        reordered_edges: (E, 2) array with updated indices
    """
    # Create mapping: old index -> new index
    old_to_new = {}
    for new_idx, name in enumerate(target_order):
        old_idx = current_order.index(name)
        old_to_new[old_idx] = new_idx
    
    # Remap edges
    reordered_edges = []
    for edge in edges:
        start, end = edge
        if start in old_to_new and end in old_to_new:
            reordered_edges.append([old_to_new[start], old_to_new[end]])
    
    return np.array(reordered_edges)


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
    
    print(f"Matched {len(csv_to_skel_map)}/{len(csv_kp_names)} keypoints")
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
    
    # Extract tracking site names
    all_site_names = [site.name for site in spec.sites]
    mj_site_names_clean = [name.replace('tracking[', '').replace(']', '') for name in all_site_names]
    
    print(f"Found {len(all_site_names)} sites in model")
    
    # Match skeleton nodes to sites
    skeleton_to_mujoco = {}
    matched_count = 0
    unmatched = []
    
    for node_name in filtered_node_names:
        if node_name in mj_site_names_clean:
            idx = mj_site_names_clean.index(node_name)
            skeleton_to_mujoco[node_name] = idx
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
                                apply_scaling: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Apply Procrustes alignment to keypoint data.
    
    Args:
        kp_array: Keypoint array (T, N, 3) in XML order
        mj_model: MuJoCo model
        xml_node_names: Node names in XML order
        skeleton_to_mujoco: Mapping from node name to site index
        exclude_indices: Keypoint indices to exclude from alignment computation (e.g., wings, antenna)
        apply_scaling: Whether to apply scaling
    
    Returns:
        aligned_kp: Aligned keypoint array
        alignment_info: Dictionary with alignment information
    """
    print(f"\nApplying Procrustes alignment (scaling={apply_scaling})...")
    
    # Get reference pose from MuJoCo model
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    
    site_subset = [skeleton_to_mujoco[name] for name in xml_node_names if name in skeleton_to_mujoco]
    ref_pose = mj_data.site_xpos[site_subset]
    
    print(f"Reference pose shape: {ref_pose.shape}")
    print(f"Reference origin: {ref_pose[0]}")
    
    # Convert to JAX arrays
    kp_jax = jnp.array(kp_array)
    ref_pose_jax = jnp.array(ref_pose)
    
    # Apply alignment
    if exclude_indices is not None:
        print(f"Excluding {len(exclude_indices)} keypoints from alignment computation")
    
    aligned_kp, procrustes_info = jit_vectorized_procrustes_with_scaling(
        kp_jax,
        ref_pose_jax,
        use_clip_average=True,
        exclude_indices=exclude_indices,
        preserve_translation=True
    )
    
    # Convert back to numpy
    aligned_kp = np.array(aligned_kp)
    
    # Extract alignment info
    alignment_info = {
        'scales': float(procrustes_info['scales'][0]) if apply_scaling else 1.0,
        'rotation': np.array(procrustes_info['rotations'][0]),
        'translation': np.array(procrustes_info['translations'][0]),
        'exclude_indices': exclude_indices.tolist() if exclude_indices is not None else None
    }
    
    print(f"Alignment scale: {alignment_info['scales']:.4f}")
    
    # Apply scale to original data (if using scaling)
    if apply_scaling:
        kp_array = alignment_info['scales'] * kp_array
    
    return kp_array, alignment_info


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


def main():
    parser = argparse.ArgumentParser(description='Preprocess keypoints for STAC IK')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file with keypoint data')
    parser.add_argument('--skeleton_path', type=str, required=True,
                        help='Path to skeleton JSON file (e.g., fly50.json)')
    parser.add_argument('--xml_path', type=str, required=True,
                        help='Path to MuJoCo XML file (e.g., fruitfly_v1_free.xml)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save preprocessed data')
    parser.add_argument('--bout_name', type=str, default='preprocessed_bout',
                        help='Name for output file (default: preprocessed_bout)')
    parser.add_argument('--frame_start', type=int, default=None,
                        help='Start frame index for bout extraction')
    parser.add_argument('--frame_end', type=int, default=None,
                        help='End frame index for bout extraction')
    parser.add_argument('--apply_alignment', action='store_true',
                        help='Apply Procrustes alignment')
    parser.add_argument('--apply_scaling', action='store_true',
                        help='Apply scaling during alignment')
    parser.add_argument('--exclude_antenna', action='store_true',
                        help='Exclude antenna from alignment computation')
    parser.add_argument('--exclude_wings', action='store_true',
                        help='Exclude wings from alignment computation')
    
    args = parser.parse_args()
    
    # Convert paths
    csv_path = Path(args.csv_path)
    skeleton_path = Path(args.skeleton_path)
    xml_path = Path(args.xml_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("KEYPOINT PREPROCESSING FOR STAC IK")
    print("=" * 80)
    
    # 1. Load CSV data
    frame_indices = None
    if args.frame_start is not None and args.frame_end is not None:
        frame_indices = np.arange(args.frame_start, args.frame_end)
        print(f"Extracting frames {args.frame_start} to {args.frame_end}")
    
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
    orig_kp_xml = orig_kp_array.copy()
    
    # 7. Optional: Apply Procrustes alignment
    alignment_info = None
    if args.apply_alignment:
        # Determine which keypoints to exclude from alignment
        exclude_indices = []
        if args.exclude_antenna:
            # Antenna is typically index 0 after reordering
            antenna_idx = [i for i, name in enumerate(xml_node_names) if 'Antenna' in name]
            exclude_indices.extend(antenna_idx)
        
        if args.exclude_wings:
            # Wing keypoints
            wing_idx = [i for i, name in enumerate(xml_node_names) if 'Wing' in name]
            exclude_indices.extend(wing_idx)
        
        exclude_arr = jnp.array(exclude_indices) if exclude_indices else None
        
        kp_array_xml, alignment_info = apply_procrustes_alignment(
            kp_array_xml, mj_model, xml_node_names, skeleton_to_mujoco,
            exclude_indices=exclude_arr, apply_scaling=args.apply_scaling
        )
    
    # 8. Save to HDF5
    output_path = output_dir / f"{args.bout_name}.h5"
    save_to_hdf5(
        output_path, kp_array_xml, orig_kp_xml, xml_node_names, xml_edges, alignment_info
    )
    
    print("\n" + "=" * 80)
    print("✓ PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nOutput saved to: {output_path}")
    print(f"Keypoint order matches STAC config KP_NAMES")
    print(f"Ready for STAC IK solver!")


if __name__ == '__main__':
    main()
