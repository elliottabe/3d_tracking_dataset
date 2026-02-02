"""Utilities for reorganizing STAC output data into bout-based structures."""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Union, Optional


def reorganize_stac_by_bouts(
    stac_data: Dict[str, Union[np.ndarray, jnp.ndarray, List[str]]],
    clip_lengths: List[int],
    bout_names: Optional[List[str]] = None,
    data_keys: Optional[List[str]] = None,
    name_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Union[np.ndarray, jnp.ndarray, List[str]]]]:
    """
    Reorganize STAC output data from concatenated vectors into bout-based structure.
    
    STAC outputs data as single concatenated vectors along the time axis. This function
    splits those vectors back into individual bouts/clips and creates a nested dictionary
    structure for easier access and processing.
    
    **Handles both concatenated and padded data formats:**
    - Concatenated: Total frames = sum(clip_lengths)
    - Padded: Total frames = max(clip_lengths) × n_bouts (each bout padded to max length)
    
    The function automatically detects which format is present and extracts only the 
    actual data (excluding padding) for each bout.
    
    Args:
        stac_data: Dictionary containing STAC output with concatenated data arrays.
            Expected keys include:
            - 'qpos': Joint positions/angles (T, n_qpos)
            - 'qvel': Joint velocities (T, n_qvel)  
            - 'xpos': Body positions (T, n_bodies, 3)
            - 'xquat': Body quaternions (T, n_bodies, 4)
            - 'marker_sites': Marker site positions (T, n_markers, 3)
            - 'kp_data': Keypoint data (T, n_kp, 3)
            - 'offsets': Marker offsets (n_markers, 3)
            - 'egocentric_site_pos': Egocentric site positions (T, n_sites, 3)
            - 'names_qpos': List of qpos names
            - 'names_xpos': List of body names
            - 'kp_names': List of keypoint names
            - 'egocentric_site_names': List of site names
        clip_lengths: List of frame counts for each bout/clip. 
            - For concatenated data: Must sum to total frames in arrays
            - For padded data: Will extract clip_lengths[i] frames from each padded block
        bout_names: Optional list of names for each bout. If None, uses 'bout_0', 'bout_1', etc.
        data_keys: Optional list of keys to split by time. If None, auto-detects array keys.
        name_keys: Optional list of name/metadata keys to copy to all bouts. If None, 
            auto-detects list/string keys.
    
    Returns:
        bout_dict: Nested dictionary with structure:
            {
                'info': {
                    'names_qpos': list of names,
                    'kp_names': list of names,
                    'offsets': array,
                    'egocentric_site_names': list of names,
                    ...
                },
                'bout_0': {
                    'qpos': array (T0, n_qpos),
                    'qvel': array (T0, n_qvel),
                    'xpos': array (T0, n_bodies, 3),
                    ...
                },
                'bout_1': {...},
                ...
            }
            where T0, T1, ... are the clip lengths for each bout.
            Metadata is stored once in 'info' key instead of being duplicated in each bout.
    
    Examples:
        >>> # Load STAC output from HDF5
        >>> stac_data = ioh5.load('Fruitfly_ik_V1_free.h5', enable_jax=True)
        >>> 
        >>> # Define clip lengths from original preprocessing
        >>> clip_lengths = [100, 150, 200]  # 3 bouts with different lengths
        >>> 
        >>> # Reorganize into bout-based structure
        >>> bout_dict = reorganize_stac_by_bouts(
        ...     stac_data,
        ...     clip_lengths,
        ...     bout_names=['walk_1', 'walk_2', 'walk_3']
        ... )
        >>> 
        >>> # Access specific bout data
        >>> bout_0_qpos = bout_dict['walk_1']['qpos']  # Shape: (100, n_qpos)
        >>> bout_0_kp = bout_dict['walk_1']['kp_data']  # Shape: (100, n_kp, 3)
    
    Raises:
        ValueError: If clip_lengths don't sum to total frames in data arrays.
        KeyError: If expected keys are missing from stac_data.
    """
    # Validate clip_lengths
    total_frames_unpadded = sum(clip_lengths)
    max_clip_length = max(clip_lengths)
    n_bouts = len(clip_lengths)
    total_frames_padded = max_clip_length * n_bouts
    
    # Generate bout names if not provided
    if bout_names is None:
        bout_names = [f'bout_{i:03d}' for i in range(n_bouts)]
    elif len(bout_names) != n_bouts:
        raise ValueError(
            f"Number of bout_names ({len(bout_names)}) must match "
            f"number of clip_lengths ({n_bouts})"
        )
    
    # Detect if data is padded or concatenated
    is_padded = False
    sample_array = None
    for value in stac_data.values():
        if isinstance(value, (np.ndarray, jnp.ndarray)) and len(value.shape) > 0:
            sample_array = value
            break
    
    if sample_array is not None:
        if sample_array.shape[0] == total_frames_padded:
            is_padded = True
            print(f"Detected PADDED data: {total_frames_padded} frames "
                  f"({n_bouts} bouts × {max_clip_length} max_length)")
        elif sample_array.shape[0] == total_frames_unpadded:
            is_padded = False
            print(f"Detected CONCATENATED data: {total_frames_unpadded} frames "
                  f"(sum of clip_lengths)")
        else:
            print(f"Warning: Data shape {sample_array.shape[0]} doesn't match "
                  f"expected padded ({total_frames_padded}) or concatenated ({total_frames_unpadded})")
    
    # Auto-detect data keys (arrays with time dimension) if not provided
    if data_keys is None:
        data_keys = []
        expected_frames = total_frames_padded if is_padded else total_frames_unpadded
        for key, value in stac_data.items():
            if isinstance(value, (np.ndarray, jnp.ndarray)):
                # Check if first dimension matches expected frames
                if len(value.shape) > 0 and value.shape[0] == expected_frames:
                    data_keys.append(key)
        print(f"Auto-detected {len(data_keys)} temporal data keys: {data_keys}")
    
    # Auto-detect name/metadata keys if not provided
    if name_keys is None:
        name_keys = []
        expected_frames = total_frames_padded if is_padded else total_frames_unpadded
        for key, value in stac_data.items():
            if key not in data_keys:
                # Include lists, strings, and arrays that don't match temporal dimension
                if isinstance(value, (list, str)):
                    name_keys.append(key)
                elif isinstance(value, (np.ndarray, jnp.ndarray)):
                    # Small arrays that don't vary with time (like offsets)
                    if len(value.shape) == 0 or value.shape[0] != expected_frames:
                        name_keys.append(key)
        print(f"Auto-detected {len(name_keys)} metadata keys: {name_keys}")
    
    # Validate that at least one data key has the expected frames
    if data_keys:
        sample_key = data_keys[0]
        expected_frames = total_frames_padded if is_padded else total_frames_unpadded
        if stac_data[sample_key].shape[0] != expected_frames:
            raise ValueError(
                f"Expected frames ({expected_frames}) doesn't match data shape "
                f"({stac_data[sample_key].shape[0]}) for key '{sample_key}'"
            )
    
    # Create bout dictionary with 'info' at top level for metadata
    bout_dict = {'info': {}}
    
    # Store metadata in 'info' key (shared across all bouts)
    for key in name_keys:
        bout_dict['info'][key] = stac_data[key]
    
    # Store clip_lengths in 'info' for future reference
    bout_dict['info']['clip_lengths'] = clip_lengths
    
    # Compute frame ranges for each bout based on padding mode
    frame_ranges = []
    if is_padded:
        # Padded: each bout occupies max_clip_length frames
        for bout_idx in range(n_bouts):
            start_idx = bout_idx * max_clip_length
            end_idx = start_idx + clip_lengths[bout_idx]  # Only take actual length, not padding
            frame_ranges.append((start_idx, end_idx))
    else:
        # Concatenated: bouts are back-to-back
        start_idx = 0
        for length in clip_lengths:
            end_idx = start_idx + length
            frame_ranges.append((start_idx, end_idx))
            start_idx = end_idx
    
    # Split temporal data for each bout
    for bout_idx, (bout_name, (start_frame, end_frame)) in enumerate(
        zip(bout_names, frame_ranges)
    ):
        bout_dict[bout_name] = {}
        
        # Split temporal data only (metadata is in 'info')
        for key in data_keys:
            data = stac_data[key]
            bout_dict[bout_name][key] = data[start_frame:end_frame]
        
        padding_note = f" (padded from {max_clip_length})" if is_padded and clip_lengths[bout_idx] < max_clip_length else ""
        print(
            f"Created {bout_name}: {clip_lengths[bout_idx]} frames "
            f"(indices {start_frame}-{end_frame}){padding_note}"
        )
    
    padding_mode = "PADDED" if is_padded else "CONCATENATED"
    print(f"\n✓ Successfully reorganized {padding_mode} data into {n_bouts} bouts")
    print(f"✓ Metadata stored in 'info' key: {list(bout_dict['info'].keys())}")
    return bout_dict


def concatenate_bouts(
    bout_dict: Dict[str, Dict[str, Union[np.ndarray, jnp.ndarray, List[str]]]],
    bout_order: Optional[List[str]] = None,
) -> Dict[str, Union[np.ndarray, jnp.ndarray, List[str]]]:
    """
    Concatenate bout-based data back into single vectors (inverse of reorganize_stac_by_bouts).
    
    Useful for preparing data to pass back into STAC or for creating continuous trajectories.
    
    Args:
        bout_dict: Nested dictionary with bout-based structure (output of reorganize_stac_by_bouts).
            Expected to have 'info' key at top level with metadata.
        bout_order: Optional list specifying order of bouts to concatenate. If None, uses sorted keys.
    
    Returns:
        concatenated_data: Dictionary with concatenated arrays along time axis and metadata from 'info'.
    
    Examples:
        >>> # Concatenate bouts in custom order
        >>> stac_data = concatenate_bouts(
        ...     bout_dict,
        ...     bout_order=['walk_2', 'walk_1', 'walk_3']  # Custom order
        ... )
    """
    if not bout_dict:
        raise ValueError("bout_dict is empty")
    
    # Determine bout order (exclude 'info' key)
    if bout_order is None:
        bout_order = sorted([k for k in bout_dict.keys() if k != 'info'])
    else:
        # Validate all specified bouts exist
        missing = set(bout_order) - set(bout_dict.keys())
        if missing:
            raise ValueError(f"Specified bouts not found in bout_dict: {missing}")
    
    # Get reference bout to determine keys
    ref_bout = bout_dict[bout_order[0]]
    
    # Separate data keys (arrays) from metadata keys
    data_keys = []
    metadata_keys = []
    for key, value in ref_bout.items():
        if isinstance(value, (np.ndarray, jnp.ndarray)) and len(value.shape) > 0:
            # Check if first dimension varies across bouts (temporal data)
            lengths = [bout_dict[bout][key].shape[0] for bout in bout_order]
            if len(set(lengths)) > 1 or all(
                bout_dict[bout][key].shape[0] > 1 for bout in bout_order
            ):
                data_keys.append(key)
            else:
                metadata_keys.append(key)
        else:
            metadata_keys.append(key)
    
    concatenated_data = {}
    
    # Concatenate temporal data
    for key in data_keys:
        arrays_to_concat = [bout_dict[bout][key] for bout in bout_order]
        concatenated_data[key] = np.concatenate(arrays_to_concat, axis=0)
    
    # Copy metadata from 'info' key if it exists, otherwise from first bout
    if 'info' in bout_dict:
        for key, value in bout_dict['info'].items():
            concatenated_data[key] = value
    else:
        # Fallback for old format (metadata in each bout)
        for key in metadata_keys:
            concatenated_data[key] = ref_bout[key]
    
    total_frames = concatenated_data[data_keys[0]].shape[0] if data_keys else 0
    print(
        f"✓ Concatenated {len(bout_order)} bouts into {total_frames} total frames"
    )
    
    return concatenated_data


def get_bout_summary(
    bout_dict: Dict[str, Dict[str, Union[np.ndarray, jnp.ndarray, List[str]]]]
) -> Dict[str, Dict[str, any]]:
    """
    Get summary statistics for each bout in the dictionary.
    
    Args:
        bout_dict: Nested dictionary with bout-based structure.
    
    Returns:
        summary: Dictionary with summary information for each bout:
            {
                'info': {
                    'metadata_keys': list of keys in 'info'
                },
                'bout_0': {
                    'n_frames': int,
                    'data_keys': list of keys,
                    'array_shapes': dict of shapes,
                },
                ...
            }
    """
    summary = {}
    
    # Add info summary if it exists
    if 'info' in bout_dict:
        summary['info'] = {
            'metadata_keys': list(bout_dict['info'].keys()),
            'n_metadata_keys': len(bout_dict['info'].keys())
        }
    
    for bout_name, bout_data in bout_dict.items():
        # Skip 'info' key
        if bout_name == 'info':
            continue
        bout_summary = {
            'data_keys': [],
            'metadata_keys': [],
            'array_shapes': {}
        }
        
        for key, value in bout_data.items():
            if isinstance(value, (np.ndarray, jnp.ndarray)):
                bout_summary['data_keys'].append(key)
                bout_summary['array_shapes'][key] = value.shape
                
                # Get number of frames from first temporal array
                if 'n_frames' not in bout_summary and len(value.shape) > 0:
                    bout_summary['n_frames'] = value.shape[0]
            else:
                bout_summary['metadata_keys'].append(key)
        
        summary[bout_name] = bout_summary
    
    return summary


def print_bout_dict_structure(
    bout_dict: Dict[str, Dict[str, Union[np.ndarray, jnp.ndarray, List[str]]]],
    max_depth: int = 2,
    show_values: bool = False
) -> None:
    """
    Print the structure, shapes, and types of data in a nested bout dictionary.
    
    Displays a hierarchical view of the dictionary structure with array shapes,
    data types, and optionally list/string values for metadata.
    
    Args:
        bout_dict: Nested dictionary with bout-based structure (output of reorganize_stac_by_bouts).
        max_depth: Maximum nesting depth to print (default: 2).
        show_values: If True, show first few elements of lists and strings (default: False).
    
    Examples:
        >>> bout_dict = reorganize_stac_by_bouts(stac_data, [100, 150])
        >>> print_bout_dict_structure(bout_dict)
        
        >>> # Show with metadata values
        >>> print_bout_dict_structure(bout_dict, show_values=True)
    """
    def _format_value(value, indent=""):
        """Format value based on type."""
        if isinstance(value, (np.ndarray, jnp.ndarray)):
            array_type = "jax" if isinstance(value, jnp.ndarray) else "numpy"
            return f"<{array_type} array: shape={value.shape}, dtype={value.dtype}>"
        elif isinstance(value, list):
            if len(value) == 0:
                return "[]"
            elif show_values:
                if len(value) <= 3:
                    return f"{value}"
                else:
                    return f"[{value[0]}, {value[1]}, ..., {value[-1]}] (len={len(value)})"
            else:
                return f"<list: len={len(value)}>"
        elif isinstance(value, str):
            if show_values:
                return f"'{value}'"
            else:
                return f"<str: len={len(value)}>"
        elif isinstance(value, dict):
            return f"<dict: {len(value)} keys>"
        elif isinstance(value, (int, float)):
            return f"{value}"
        else:
            return f"<{type(value).__name__}>"
    
    def _print_dict(d, prefix="", depth=0):
        """Recursively print dictionary structure."""
        if depth > max_depth:
            return
        
        keys = list(d.keys())
        for i, key in enumerate(keys):
            is_last = (i == len(keys) - 1)
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "
            
            value = d[key]
            value_str = _format_value(value)
            
            print(f"{prefix}{connector}{key}: {value_str}")
            
            # Recursively print nested dicts
            if isinstance(value, dict) and depth < max_depth:
                _print_dict(value, prefix + extension, depth + 1)
    
    # Print header
    print("=" * 80)
    print("BOUT DICTIONARY STRUCTURE")
    print("=" * 80)
    
    # Count bouts and info
    n_bouts = len([k for k in bout_dict.keys() if k != 'info'])
    has_info = 'info' in bout_dict
    
    print(f"Total keys: {len(bout_dict)} ({'info' + ' + ' if has_info else ''}{n_bouts} bouts)")
    print()
    
    # Print structure
    _print_dict(bout_dict)
    
    # Print summary statistics
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Info summary
    if 'info' in bout_dict:
        print(f"📋 Metadata ('info' key): {len(bout_dict['info'])} items")
        for key in bout_dict['info'].keys():
            value_str = _format_value(bout_dict['info'][key])
            print(f"   - {key}: {value_str}")
        print()
    
    # Bout summaries
    bout_keys = sorted([k for k in bout_dict.keys() if k != 'info'])
    if bout_keys:
        print(f"🎬 Bouts: {len(bout_keys)} total")
        for bout_name in bout_keys:
            bout_data = bout_dict[bout_name]
            
            # Get frame count from first array
            n_frames = None
            for value in bout_data.values():
                if isinstance(value, (np.ndarray, jnp.ndarray)) and len(value.shape) > 0:
                    n_frames = value.shape[0]
                    break
            
            # Count data types
            array_keys = [k for k, v in bout_data.items() 
                         if isinstance(v, (np.ndarray, jnp.ndarray))]
            other_keys = [k for k in bout_data.keys() if k not in array_keys]
            
            frame_str = f"{n_frames} frames" if n_frames else "no frames"
            print(f"   - {bout_name}: {frame_str}, {len(array_keys)} arrays, {len(other_keys)} other")
    
    print("=" * 80)
