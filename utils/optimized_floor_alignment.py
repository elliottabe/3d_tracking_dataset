import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Optional
import numpy as np
import time

def procrustes_align_jax_optimized_with_scaling(points, reference, exclude_indices=None, preserve_translation=False):
    """
    JIT-optimized Procrustes alignment with scaling.

    Args:
        points: (N, 3) source keypoints
        reference: (N, 3) reference pose
        exclude_indices: Optional array of indices to exclude from alignment computation
        preserve_translation: If True, preserve original spatial translation (only apply rotation/scale)

    Returns:
        aligned_points: (N, 3) - All points transformed (including excluded)
        R: (3, 3) - Rotation matrix
        scale: float - Scale factor
        translation: (3,) - Translation vector
    """
    # Create inclusion mask
    if exclude_indices is None:
        # Use all points
        points_for_alignment = points
        ref_for_alignment = reference
    elif len(exclude_indices) == 0:
        # Empty exclusion list - use all points
        points_for_alignment = points
        ref_for_alignment = reference
    else:
        # Create integer array of all indices
        N = points.shape[0]
        all_indices = jnp.arange(N)

        # Find indices to include using setdiff1d (JIT-compatible)
        include_indices = jnp.setdiff1d(all_indices, exclude_indices, size=N-len(exclude_indices))

        # Select included keypoints only using integer indexing
        points_for_alignment = points[include_indices]
        ref_for_alignment = reference[include_indices]

    # Center both point sets (using subset)
    points_centroid = jnp.mean(points_for_alignment, axis=0)
    ref_centroid = jnp.mean(ref_for_alignment, axis=0)

    points_centered = points_for_alignment - points_centroid
    ref_centered = ref_for_alignment - ref_centroid

    # Calculate scale factor (using subset)
    points_scale = jnp.sqrt(jnp.sum(points_centered**2))
    ref_scale = jnp.sqrt(jnp.sum(ref_centered**2))
    scale = jnp.where(points_scale > 1e-10, ref_scale / points_scale, 1.0)
    points_scaled = points_centered * scale

    # Find optimal rotation using SVD (on subset)
    H = points_scaled.T @ ref_centered
    U, _, Vt = jnp.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    det_R = jnp.linalg.det(R)
    # If determinant is negative, flip the last column of Vt
    Vt_corrected = jnp.where(det_R < 0,
                             Vt.at[-1, :].multiply(-1),
                             Vt)
    R = jnp.where(det_R < 0,
                  Vt_corrected.T @ U.T,
                  R)

    # Apply transformation to ALL N original points (not just subset)
    points_all_centered = points - points_centroid  # Center using subset centroid
    points_all_scaled = points_all_centered * scale  # Scale all points
    
    # When preserve_translation=True: only scale (no rotation)
    # When preserve_translation=False: apply rotation + translation to reference frame
    aligned_points = jax.lax.cond(
        preserve_translation,
        lambda ps, pc: ps + pc,  # Only scale, keep original position
        lambda ps, pc: (R @ ps.T).T + ref_centroid,  # Rotate + translate to reference
        points_all_scaled, points_centroid
    )
    
    translation = jax.lax.cond(
        preserve_translation,
        lambda: jnp.zeros(3),  # No translation applied
        lambda: ref_centroid - points_centroid  # Translation to reference
    )

    return aligned_points, R, scale, translation


def procrustes_align_jax_optimized_no_scaling(points, reference, exclude_indices=None, preserve_translation=False):
    """
    JIT-optimized Procrustes alignment without scaling.

    Args:
        points: (N, 3) source keypoints
        reference: (N, 3) reference pose
        exclude_indices: Optional array of indices to exclude from alignment computation
        preserve_translation: If True, preserve original spatial translation (only apply rotation)

    Returns:
        aligned_points: (N, 3) - All points transformed (including excluded)
        R: (3, 3) - Rotation matrix
        scale: float - Scale factor (always 1.0 for no scaling)
        translation: (3,) - Translation vector
    """
    # Create inclusion mask
    if exclude_indices is None:
        # Use all points
        points_for_alignment = points
        ref_for_alignment = reference
    elif len(exclude_indices) == 0:
        # Empty exclusion list - use all points
        points_for_alignment = points
        ref_for_alignment = reference
    else:
        # Create integer array of all indices
        N = points.shape[0]
        all_indices = jnp.arange(N)

        # Find indices to include using setdiff1d (JIT-compatible)
        include_indices = jnp.setdiff1d(all_indices, exclude_indices, size=N-len(exclude_indices))

        # Select included keypoints only using integer indexing
        points_for_alignment = points[include_indices]
        ref_for_alignment = reference[include_indices]

    # Center both point sets (using subset)
    points_centroid = jnp.mean(points_for_alignment, axis=0)
    ref_centroid = jnp.mean(ref_for_alignment, axis=0)

    points_centered = points_for_alignment - points_centroid
    ref_centered = ref_for_alignment - ref_centroid

    # No scaling
    scale = 1.0
    points_scaled = points_centered

    # Find optimal rotation using SVD (on subset)
    H = points_scaled.T @ ref_centered
    U, _, Vt = jnp.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    det_R = jnp.linalg.det(R)
    # If determinant is negative, flip the last column of Vt
    Vt_corrected = jnp.where(det_R < 0,
                             Vt.at[-1, :].multiply(-1),
                             Vt)
    R = jnp.where(det_R < 0,
                  Vt_corrected.T @ U.T,
                  R)

    # Apply transformation to ALL N original points (not just subset)
    points_all_centered = points - points_centroid  # Center using subset centroid
    points_all_scaled = points_all_centered * scale  # No scaling, but keep for consistency
    
    # Use conditional to choose centroid based on preserve_translation
    # Keep original spatial position OR align to reference centroid
    final_centroid = jax.lax.cond(
        preserve_translation,
        lambda: points_centroid,  # Keep original position
        lambda: ref_centroid      # Align to reference
    )
    aligned_points = (R @ points_all_scaled.T).T + final_centroid

    translation = ref_centroid - points_centroid

    return aligned_points, R, scale, translation


# JIT compile both versions with static argnames
jit_procrustes_with_scaling = jax.jit(
    procrustes_align_jax_optimized_with_scaling,
    static_argnames=['preserve_translation']
)
jit_procrustes_no_scaling = jax.jit(
    procrustes_align_jax_optimized_no_scaling,
    static_argnames=['preserve_translation']
)


# Vectorized Procrustes alignment
def vectorized_procrustes_alignment(kp_clip, ref_pose, allow_scaling=True, use_clip_average=False, exclude_indices=None, preserve_translation=False):
    """
    Vectorized Procrustes alignment using vmap.

    Args:
        kp_clip: (T, N, 3) keypoint trajectories
        ref_pose: (N, 3) reference pose
        allow_scaling: Whether to allow scaling
        use_clip_average: If True, compute average of all frames, align that to ref_pose,
                         and apply the same transformation to all frames
        exclude_indices: Optional array of keypoint indices to exclude from alignment
        preserve_translation: If True, preserve original spatial translation (only apply rotation/scale)

    Returns:
        aligned_frames: (T, N, 3) aligned keypoints (ALL points transformed)
        alignment_info: dict with rotation, scale, translation, errors
    """
    if allow_scaling:
        align_func = lambda pts: jit_procrustes_with_scaling(pts, ref_pose, exclude_indices, preserve_translation)
    else:
        align_func = lambda pts: jit_procrustes_no_scaling(pts, ref_pose, exclude_indices, preserve_translation)

    if use_clip_average:
        # Compute average pose across all frames
        avg_pose = jnp.mean(kp_clip, axis=0)  # (N, 3)

        # Perform Procrustes alignment on the average pose
        _, R, scale, translation = align_func(avg_pose)

        # Apply the same transformation to all frames
        def apply_transform(frame):
            frame_centroid = jnp.mean(frame, axis=0)
            points_centered = frame - frame_centroid
            points_scaled = points_centered * scale
            aligned = (R @ points_scaled.T).T + jnp.mean(ref_pose, axis=0)
            return aligned

        aligned_frames = jax.vmap(apply_transform)(kp_clip)

        # Broadcast transformation parameters to match per-frame format
        rotations = jnp.broadcast_to(R[None, ...], (kp_clip.shape[0], R.shape[0], R.shape[1]))
        scales = jnp.broadcast_to(scale, (kp_clip.shape[0],))
        translations = jnp.broadcast_to(translation[None, ...], (kp_clip.shape[0], translation.shape[0]))

    else:
        # Original per-frame alignment
        def align_single_frame(frame):
            return align_func(frame)

        # Apply vmap over all frames
        aligned_frames, rotations, scales, translations = jax.vmap(align_single_frame)(kp_clip)

    # Calculate alignment errors
    errors = jax.vmap(lambda frame: jnp.mean(jnp.linalg.norm(frame - ref_pose, axis=1)))(aligned_frames)

    return aligned_frames, {
        'rotations': rotations,
        'scales': scales,
        'translations': translations,
        'errors': errors,
        'mean_error': jnp.mean(errors),
        'max_error': jnp.max(errors),
        'use_clip_average': use_clip_average
    }

# JIT compile the vectorized version for both scaling options and averaging modes
jit_vectorized_procrustes_with_scaling = jax.jit(
    lambda kp_clip, ref_pose, use_clip_average=False, exclude_indices=None, preserve_translation=False: vectorized_procrustes_alignment(
        kp_clip, ref_pose, allow_scaling=True, use_clip_average=use_clip_average,
        exclude_indices=exclude_indices, preserve_translation=preserve_translation
    ),
    static_argnames=['use_clip_average', 'preserve_translation']
)
jit_vectorized_procrustes_no_scaling = jax.jit(
    lambda kp_clip, ref_pose, use_clip_average=False, exclude_indices=None, preserve_translation=False: vectorized_procrustes_alignment(
        kp_clip, ref_pose, allow_scaling=False, use_clip_average=use_clip_average,
        exclude_indices=exclude_indices, preserve_translation=preserve_translation
    ),
    static_argnames=['use_clip_average', 'preserve_translation']
)


def align_to_ground_plane_with_contact(aligned_points, end_eff_indices=None, percentile=10, target_z=-0.125):
    """
    Align keypoints ensuring all end effectors touch the ground at target_z.
    
    Two-step process:
    1. Rotate all points so the fitted ground plane becomes horizontal
    2. Translate vertically so the LOWEST end effector touches target_z, then clip any below-ground points
    
    Args:
        aligned_points: (T, N, 3) keypoint trajectories
        end_eff_indices: Indices of end effectors (default: [4, 9, 14, 19, 24, 29])
        percentile: Percentile threshold for selecting ground points
        target_z: Target z-coordinate for the ground plane (default: 0.0)
    
    Returns:
        ground_aligned_points: (T, N, 3) ground plane aligned keypoints
        alignment_info: Dict with alignment information
    """
    if end_eff_indices is None:
        end_eff_indices = jnp.array([4, 9, 14, 19, 24, 29])
    
    # Extract end effector positions
    endeff_xpos = aligned_points[:, end_eff_indices]  # (T, n_endeff, 3)
    
    # Calculate percentile threshold for each end effector
    z_threshold = jnp.percentile(endeff_xpos[..., 2], percentile, axis=0)  # (n_endeff,)
    
    # Collect ground points for each end effector
    ground_points_list = []
    for n in range(endeff_xpos.shape[1]):
        # Get points below threshold for this end effector
        mask = endeff_xpos[:, n, 2] <= z_threshold[n]
        ground_points_endeff = endeff_xpos[mask, n, :]
        ground_points_list.append(ground_points_endeff)
    
    # Combine all ground points
    ground_points = jnp.concatenate(ground_points_list, axis=0)  # (n_ground_points, 3)
    
    # Fit plane to ground points using SVD
    centroid = jnp.mean(ground_points, axis=0)
    centered_points = ground_points - centroid
    
    # Handle case where we have too few points or no variation
    if ground_points.shape[0] < 3:
        # Not enough points, use identity transformation
        rotation_matrix = jnp.eye(3)
        ground_normal = jnp.array([0.0, 0.0, 1.0])
        rotated_points = aligned_points
        rotated_centroid = centroid
    else:
        # SVD to find plane normal
        try:
            U, s, Vt = jnp.linalg.svd(centered_points, full_matrices=False)
            ground_normal = Vt[-1, :]  # Normal vector (smallest singular vector)
            
            # Ensure normal points upward (positive z component)
            if ground_normal[2] < 0:
                ground_normal = -ground_normal
        except:
            # Fallback if SVD fails
            ground_normal = jnp.array([0.0, 0.0, 1.0])
            rotation_matrix = jnp.eye(3)
            rotated_points = aligned_points
            rotated_centroid = centroid
        else:
            # Step 1: Calculate rotation matrix to align ground normal with [0, 0, 1]
            target_normal = jnp.array([0.0, 0.0, 1.0])
            
            # Calculate rotation using Rodrigues' formula
            cross_product = jnp.cross(ground_normal, target_normal)
            dot_product = jnp.dot(ground_normal, target_normal)
            
            if jnp.abs(dot_product - 1.0) < 1e-6:
                # Already aligned
                rotation_matrix = jnp.eye(3)
            elif jnp.abs(dot_product + 1.0) < 1e-6:
                # Opposite direction, rotate 180 degrees around x-axis
                rotation_matrix = jnp.array([
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0]
                ])
            else:
                # General case using Rodrigues' rotation formula
                axis = cross_product / jnp.linalg.norm(cross_product)
                angle = jnp.arccos(jnp.clip(dot_product, -1.0, 1.0))
                
                # Rodrigues' rotation formula
                K = jnp.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                
                rotation_matrix = (jnp.eye(3) + 
                                 jnp.sin(angle) * K + 
                                 (1 - jnp.cos(angle)) * jnp.dot(K, K))
            #TODO: align to coxa zero
            # Step 1: Apply rotation to all points
            rotated_points = jnp.einsum('ij,tnj->tni', rotation_matrix, aligned_points)
            rotated_centroid = rotation_matrix @ centroid
    
    # Step 2: Ensure all end effectors touch the ground
    # Find the lowest end effector after rotation
    rotated_points = rotated_points - rotated_points[:,:1] # Center around first point 
    rotated_end_effs = rotated_points[:, end_eff_indices]  # (T, n_endeff, 3)
    min_end_eff_z = jnp.mean((jnp.mean(rotated_end_effs[:, :, 2],axis=0)))  # Lowest point across all frames and end effectors
    
    # Calculate translation to bring the lowest end effector to target_z
    translation = jnp.array([0.0, 0.0, target_z - min_end_eff_z])
    
    # Step 2a: Apply translation
    translated_points = rotated_points + translation
    
    # Step 2b: Clip any points that go below the floor
    # Only clip z-coordinates of end effectors, leave other keypoints unchanged
    clipped_points = translated_points.copy()
    clipped_points = clipped_points.at[:, end_eff_indices, 2].set(
        jnp.maximum(translated_points[:, end_eff_indices, 2], target_z)
    )
    
    # Calculate metrics for reporting
    final_end_eff_heights = clipped_points[:, end_eff_indices, 2]
    n_contacts = jnp.sum(jnp.abs(final_end_eff_heights - target_z) < 1e-6)  # Points exactly at target_z
    n_clipped = jnp.sum(translated_points[:, end_eff_indices, 2] < target_z)  # Points that were clipped
    
    # Calculate final centroids
    final_centroid = rotated_centroid + translation
    
    # Prepare alignment info
    alignment_info = {
        'ground_points': ground_points,
        'ground_normal': ground_normal,
        'rotation_matrix': rotation_matrix,
        'translation': translation,
        'original_centroid': centroid,
        'rotated_centroid': rotated_centroid,
        'final_centroid': final_centroid,
        'z_thresholds': z_threshold,
        'n_ground_points': ground_points.shape[0],
        'percentile': percentile,
        'target_z': target_z,
        'min_end_eff_z_before_translation': min_end_eff_z,
        'min_end_eff_z_after_translation': jnp.min(final_end_eff_heights),
        'n_contacts': n_contacts,
        'n_clipped': n_clipped,
        'contact_ratio': n_contacts / (final_end_eff_heights.shape[0] * final_end_eff_heights.shape[1]),
        'steps': {
            'step1_rotation': 'Applied rotation to make ground plane horizontal',
            'step2a_translation': f'Applied vertical translation of {translation[2]:.4f} to bring lowest end effector to target_z={target_z}',
            'step2b_clipping': f'Clipped {n_clipped} end effector points that went below floor'
        }
    }
    
    return clipped_points, alignment_info

def complete_alignment_pipeline_with_ground_contact(kp_clip, ref_pose,
                                                  end_eff_indices=None,
                                                  percentile=10.0,
                                                  target_z=0.0,
                                                  allow_scaling=True,
                                                  use_clip_average=False,
                                                  exclude_indices=None,
                                                  preserve_translation=False):
    """
    Complete alignment pipeline: Procrustes → Ground Contact Alignment

    Args:
        kp_clip: Input keypoint trajectories
        ref_pose: Reference pose for Procrustes alignment
        end_eff_indices: End effector indices
        percentile: Percentile for ground point selection
        target_z: Target ground plane height
        allow_scaling: Whether to allow scaling in Procrustes alignment
        use_clip_average: If True, compute average of all frames for Procrustes alignment
                         and apply the same transformation to all frames
        exclude_indices: Optional array of keypoint indices to exclude from Procrustes alignment
        preserve_translation: If True, preserve original spatial translation in Procrustes step

    Returns:
        final_clip: Final aligned keypoint trajectories
        pipeline_info: Dictionary with alignment information
    """
    if end_eff_indices is None:
        end_eff_indices = jnp.array([4, 9, 14, 19, 24, 29])

    kp_clip = kp_clip - kp_clip[:,:1] # Center around first point
    # Stage 1: Procrustes alignment
    if allow_scaling:
        procrustes_clip, procrustes_info = jit_vectorized_procrustes_with_scaling(
            kp_clip, ref_pose, use_clip_average, exclude_indices, preserve_translation
        )
    else:
        procrustes_clip, procrustes_info = jit_vectorized_procrustes_no_scaling(
            kp_clip, ref_pose, use_clip_average, exclude_indices, preserve_translation
        )
    procrustes_clip = procrustes_clip - procrustes_clip[:,:1] # Center around first point
    # Stage 2: compute ground plane alignment
    final_clip, ground_contact_info = align_to_ground_plane_with_contact(
        procrustes_clip, end_eff_indices, percentile
    )


    pipeline_info = {
        'procrustes': procrustes_info,
        'ground_contact': ground_contact_info,
        'pipeline_params': {
            'end_eff_indices': end_eff_indices,
            'percentile': percentile,
            'target_z': target_z,
            'allow_scaling': allow_scaling,
            'use_clip_average': use_clip_average,
            'exclude_indices': exclude_indices,
            'preserve_translation': preserve_translation
        }
    }

    return final_clip, pipeline_info


# Note: The complete pipeline cannot be fully JIT-compiled due to the
# pre-computation step, but the core alignment operations are JIT-optimized


def batch_process_with_ground_contact(bout_dict, ref_pose,
                                    end_eff_indices=None,
                                    percentile=10.0,
                                    target_z=0.0,
                                    max_clips=None,
                                    verbose=True,
                                    use_clip_average=False,
                                    exclude_indices=None,
                                    preserve_translation=False):
    """
    Optimized batch processing with ground contact alignment.

    Args:
        bout_dict: Dictionary of walking bout data
        ref_pose: Reference pose for Procrustes alignment
        end_eff_indices: End effector indices
        percentile: Percentile for ground point selection
        target_z: Target ground plane height
        max_clips: Maximum number of clips to process
        verbose: Whether to print progress
        use_clip_average: If True, compute average of all frames for Procrustes alignment
                         and apply the same transformation to all frames
        exclude_indices: Optional array of keypoint indices to exclude from Procrustes alignment
        preserve_translation: If True, preserve original spatial translation in Procrustes step

    Returns:
        processed_bouts: Dictionary with aligned data
        summary: Processing summary statistics
    """
    if end_eff_indices is None:
        end_eff_indices = jnp.array([4, 9, 14, 19, 24, 29])

    ref_pose_jax = jnp.array(ref_pose)
    bout_keys = list(bout_dict.keys())
    if max_clips is not None:
        bout_keys = bout_keys[:max_clips]

    print(f"Processing {len(bout_keys)} walking bouts with ground contact alignment...")
    print("✅ Starting batch processing with optimized ground contact alignment...")

    # Process clips
    processed_bouts = {}
    stats = {
        'processing_times': [],
        'contact_counts': [],
        'procrustes_errors': [],
        'clipped_points': [],
        'clip_lengths': []
    }

    start_time = time.time()

    for i, bout_key in enumerate(bout_keys):
        bout_start = time.time()

        orig_kp = jnp.array(bout_dict[bout_key]['orig_kp'])

        # Apply optimized pipeline
        aligned_clip, pipeline_info = complete_alignment_pipeline_with_ground_contact(
            orig_kp, ref_pose_jax, end_eff_indices, percentile, target_z,
            use_clip_average=use_clip_average, exclude_indices=exclude_indices,
            preserve_translation=preserve_translation
        )

        processing_time = time.time() - bout_start

        # Store results
        processed_bouts[bout_key] = {
            'aligned_kp': aligned_clip,
            'scaled_kp': orig_kp*pipeline_info['scales'],
            'pipeline_info': pipeline_info
        }

        # Collect stats
        stats['processing_times'].append(processing_time)
        stats['contact_counts'].append(float(pipeline_info['ground_contact']['mean_contacts']))
        stats['procrustes_errors'].append(float(pipeline_info['procrustes']['mean_error']))
        stats['clipped_points'].append(float(pipeline_info['ground_contact']['total_clipped']))
        stats['clip_lengths'].append(orig_kp.shape[0])

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            eta = avg_time * (len(bout_keys) - i - 1)
            print(f"  Processed {i+1}/{len(bout_keys)} | "
                  f"Avg: {avg_time:.3f}s/clip | "
                  f"ETA: {eta/60:.1f}min | "
                  f"Contacts: {np.mean(stats['contact_counts'][-50:]):.1f}")

    total_time = time.time() - start_time

    summary = {
        'total_clips': len(bout_keys),
        'total_time': total_time,
        'avg_time_per_clip': total_time / len(bout_keys),
        'clips_per_second': len(bout_keys) / total_time,
        'mean_contact_count': np.mean(stats['contact_counts']),
        'mean_procrustes_error': np.mean(stats['procrustes_errors']),
        'mean_clipped_points': np.mean(stats['clipped_points']),
        'total_frames': sum(stats['clip_lengths'])
    }

    print(f"\n🚀 GROUND CONTACT PROCESSING COMPLETE!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Speed: {summary['clips_per_second']:.1f} clips/second")
    print(f"   Throughput: {summary['total_frames']/total_time:.0f} frames/second")
    print(f"   Mean contacts: {summary['mean_contact_count']:.2f}/{len(end_eff_indices)}")
    print(f"   Mean clipped points: {summary['mean_clipped_points']:.1f}")

    return processed_bouts, summary


print("✅ Ground contact alignment with JIT optimization integrated")
