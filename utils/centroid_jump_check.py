"""
Centroid-jump detector for identity switch detection in multi-fly tracking.

Flags frames where the body centroid (mean of trunk keypoints) jumps by more
than a threshold distance between consecutive frames, which indicates a likely
identity swap between tracked flies.
"""

import numpy as np
from typing import List, Optional, Tuple


def detect_centroid_jumps(
    kp_array: np.ndarray,
    kp_names: List[str],
    trunk_keypoints: List[str],
    threshold_mm: float = 2.0,
) -> Tuple[np.ndarray, dict]:
    """
    Detect frames where the body centroid jumps abnormally between consecutive frames.

    Computes the centroid from trunk keypoints each frame, then flags frames where
    the frame-to-frame displacement exceeds ``threshold_mm``.

    Args:
        kp_array: (T, N, 3) keypoint positions in mm.
        kp_names: List of N keypoint names matching axis 1 of kp_array.
        trunk_keypoints: Keypoint names to average for centroid (e.g. ['Scutellum', 'Postnotum']).
        threshold_mm: Maximum allowed frame-to-frame centroid displacement in mm.

    Returns:
        bad_mask: (T,) boolean array — True for frames flagged as jumps.
        report: Summary dict with detection statistics.
    """
    T, N, _ = kp_array.shape

    # Find indices of trunk keypoints (substring match)
    trunk_idx = []
    matched_names = []
    for i, name in enumerate(kp_names):
        if any(tk in name for tk in trunk_keypoints):
            trunk_idx.append(i)
            matched_names.append(name)

    if len(trunk_idx) == 0:
        raise ValueError(
            f"No trunk keypoints found. Looked for substrings {trunk_keypoints} "
            f"in {kp_names}"
        )

    trunk_idx = np.array(trunk_idx)

    # Compute centroid per frame, ignoring NaN keypoints
    trunk_kp = kp_array[:, trunk_idx, :]  # (T, n_trunk, 3)
    centroid = np.nanmean(trunk_kp, axis=1)  # (T, 3)

    # Frame-to-frame displacement
    displacement = np.full(T, 0.0)
    diff = np.diff(centroid, axis=0)  # (T-1, 3)
    displacement[1:] = np.linalg.norm(diff, axis=1)

    # Flag frames exceeding threshold
    bad_mask = displacement > threshold_mm

    n_flagged = int(bad_mask.sum())
    report = {
        'n_frames_flagged': n_flagged,
        'pct_flagged': 100.0 * n_flagged / T if T > 0 else 0.0,
        'threshold_mm': threshold_mm,
        'trunk_keypoints_used': matched_names,
        'max_displacement_mm': float(np.nanmax(displacement)) if T > 1 else 0.0,
        'median_displacement_mm': float(np.nanmedian(displacement[1:])) if T > 1 else 0.0,
    }

    return bad_mask, report


def mask_centroid_jumps(
    kp_array: np.ndarray,
    kp_names: List[str],
    trunk_keypoints: List[str],
    threshold_mm: float = 2.0,
    window: int = 1,
) -> Tuple[np.ndarray, dict]:
    """
    Detect centroid jumps and set all keypoints in flagged frames to NaN.

    Also masks ``window`` frames on either side of each jump to catch the
    transition period.

    Args:
        kp_array: (T, N, 3) keypoint positions in mm.
        kp_names: List of N keypoint names.
        trunk_keypoints: Keypoint names to average for centroid.
        threshold_mm: Maximum allowed displacement in mm.
        window: Number of extra frames to mask on each side of a jump.

    Returns:
        masked_kp: (T, N, 3) with flagged frames set to NaN.
        report: Detection summary dict.
    """
    bad_mask, report = detect_centroid_jumps(
        kp_array, kp_names, trunk_keypoints, threshold_mm
    )

    # Expand mask by window on each side
    if window > 0 and bad_mask.any():
        expanded = bad_mask.copy()
        for offset in range(1, window + 1):
            expanded[offset:] |= bad_mask[:-offset]
            expanded[:-offset] |= bad_mask[offset:]
        bad_mask = expanded

    n_masked = int(bad_mask.sum())
    report['n_frames_masked'] = n_masked
    report['pct_masked'] = 100.0 * n_masked / len(bad_mask) if len(bad_mask) > 0 else 0.0
    report['window'] = window

    masked_kp = kp_array.copy()
    masked_kp[bad_mask] = np.nan

    return masked_kp, report
