"""SAM3 mask-centroid triangulation for female fly center-of-mass.

Loads per-camera 2D centroids from a ``sam3_masks.npz`` file, loads 11-parameter
DLT calibration for each camera, and linearly triangulates a 3D world-frame
centroid per frame. Used to recover a robust female COM trajectory in frames
where the pose model (and therefore 3D keypoints) is unreliable (e.g. on the
arena wall).
"""
# Note: utils/fly_detection.py has a sibling _triangulate_dlt that consumes
# OpenCV-YAML (C, 3, 4) projection matrices. This module is separate because
# it loads 11-parameter DLT CSV coefficients (via _dlt_load from
# utils/courtship_figure_panels.py). If the calibration pipeline is ever
# unified on a single format, these two helpers should be collapsed.
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from utils.courtship_figure_panels import _dlt_load


def _triangulate_point(
    coeffs: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    min_cams: int = 2,
) -> np.ndarray:
    """Linear DLT triangulation of a single 3D point from multiple cameras.

    Parameters
    ----------
    coeffs : (n_cams, 11) array of DLT coefficients.
    uv     : (n_cams, 2) array of (u, v) pixel coords.
    valid  : (n_cams,) bool mask of cameras whose observation is usable.
    min_cams : minimum valid cameras to attempt triangulation.

    Returns
    -------
    (3,) world-frame point, or (3,) NaN if fewer than ``min_cams`` are valid.
    """
    mask = np.asarray(valid, dtype=bool)
    if int(mask.sum()) < int(min_cams):
        return np.full(3, np.nan, dtype=float)
    L = np.asarray(coeffs, dtype=float)[mask]
    p = np.asarray(uv, dtype=float)[mask]
    u = p[:, 0]
    v = p[:, 1]
    # Two rows per camera of the standard DLT back-projection:
    #   (L1 - u L9) X + (L2 - u L10) Y + (L3 - u L11) Z = u - L4
    #   (L5 - v L9) X + (L6 - v L10) Y + (L7 - v L11) Z = v - L8
    A_u = np.stack(
        [L[:, 0] - u * L[:, 8], L[:, 1] - u * L[:, 9], L[:, 2] - u * L[:, 10]],
        axis=1,
    )
    A_v = np.stack(
        [L[:, 4] - v * L[:, 8], L[:, 5] - v * L[:, 9], L[:, 6] - v * L[:, 10]],
        axis=1,
    )
    A = np.concatenate([A_u, A_v], axis=0)
    b = np.concatenate([u - L[:, 3], v - L[:, 7]], axis=0)
    xyz, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.asarray(xyz, dtype=float).reshape(3)


def triangulate_sam3_female_com(
    npz_path: str | Path,
    calib_dir: str | Path,
    fly_idx: int = 0,
    camera_order: Optional[Sequence[str]] = None,
    min_cams: int = 2,
    verbose: bool = True,
) -> np.ndarray:
    """Triangulate per-frame female COM from SAM3 mask centroids.

    Parameters
    ----------
    npz_path : path to ``sam3_masks.npz`` with ``valid`` (n_flies, n_cams, T)
        and ``centroids`` (n_flies, n_cams, T, 2) arrays.
    calib_dir : directory containing ``Cam*_dlt.csv`` files (sorted by filename
        to define camera ordering unless ``camera_order`` is given).
    fly_idx : which fly in the SAM3 arrays corresponds to the female (default 0).
    camera_order : optional explicit filename order (e.g. ``['Cam2012630_dlt.csv',
        ...]``); when ``None``, ``sorted(glob('Cam*_dlt.csv'))`` is used.
    min_cams : minimum valid cameras to attempt triangulation (default 2).
    verbose : if True (default), print the resolved camera order to stdout.
        Set to False to suppress the side-effect in library/batch use.

    Returns
    -------
    (T, 3) array of world-frame female COM; NaN on frames with
    fewer than ``min_cams`` valid cameras.
    """
    npz_path = Path(npz_path)
    calib_dir = Path(calib_dir)
    if camera_order is None:
        dlt_files = sorted(calib_dir.glob('Cam*_dlt.csv'))
    else:
        dlt_files = [calib_dir / name for name in camera_order]
    if not dlt_files:
        raise FileNotFoundError(f'no Cam*_dlt.csv files found in {calib_dir}')

    coeffs = np.stack([_dlt_load(f) for f in dlt_files], axis=0)  # (n_cams, 11)
    if verbose:
        print(f'[sam3_female_com] camera order:')
        for idx, f in enumerate(dlt_files):
            print(f'  cam {idx}: {f.name}')

    with np.load(npz_path) as npz:
        valid = np.asarray(npz['valid'])                            # (n_flies, n_cams, T)
        # Promote centroids to float64 so triangulation precision is not
        # bottlenecked by on-disk float32 storage.
        centroids = np.asarray(npz['centroids'], dtype=np.float64)  # (n_flies, n_cams, T, 2)

    if valid.shape[1] != coeffs.shape[0]:
        raise ValueError(
            f'SAM3 n_cams ({valid.shape[1]}) != n DLT files ({coeffs.shape[0]})'
        )

    v_fly = valid[fly_idx]                 # (n_cams, T)
    c_fly = centroids[fly_idx]             # (n_cams, T, 2)
    T = v_fly.shape[1]
    out = np.full((T, 3), np.nan, dtype=float)
    for t in range(T):
        uv_t = c_fly[:, t, :]              # (n_cams, 2)
        valid_t = v_fly[:, t]              # (n_cams,)
        finite_t = np.isfinite(uv_t).all(axis=1)
        out[t] = _triangulate_point(coeffs, uv_t, valid_t & finite_t,
                                    min_cams=min_cams)
    return out


def sam3_camera_index(calib_dir: str | Path, cam_csv_name: str) -> int:
    """Return the SAM3-axis index for ``cam_csv_name`` in ``sorted(calib_dir)``.

    The SAM3 ``packed``/``valid``/``centroids`` arrays are stored with cameras
    in ``sorted(glob('Cam*_dlt.csv'))`` order (same convention as
    :func:`triangulate_sam3_female_com`). This helper converts a camera CSV
    filename (e.g. ``'Cam2012630_dlt.csv'``) into its SAM3 camera-axis slot.
    """
    files = sorted(Path(calib_dir).glob('Cam*_dlt.csv'))
    names = [f.name for f in files]
    if cam_csv_name not in names:
        raise ValueError(f'{cam_csv_name!r} not in {names}')
    return names.index(cam_csv_name)


def unpack_sam3_masks_for_frames(
    npz_path: str | Path,
    cam_idx: int,
    fly_indices: Sequence[int],
    frame_indices: Sequence[int],
) -> list:
    """Return one unpacked mask stack per fly, indexed by ``frame_indices``.

    Output is ``[mask_fly0, mask_fly1, ...]`` where each element has shape
    ``(len(frame_indices), H_full, W_full)`` of ``bool``. Invalid frames (per
    the npz ``valid`` array) return an all-False mask for that slot.
    """
    with np.load(npz_path) as npz:
        packed = npz['packed']           # (n_flies, n_cams, T, H, W/8)
        valid = npz['valid']             # (n_flies, n_cams, T)
        shape = tuple(int(x) for x in npz['shape'])  # (H, W_full)
    H, W = shape
    fidx = np.asarray(frame_indices, dtype=int)
    out = []
    for fi in fly_indices:
        stack = np.zeros((len(fidx), H, W), dtype=bool)
        for k, t in enumerate(fidx):
            if not bool(valid[int(fi), int(cam_idx), int(t)]):
                continue
            bits = np.unpackbits(
                packed[int(fi), int(cam_idx), int(t)], axis=-1,
            )
            stack[k] = bits[:H, :W].astype(bool)
        out.append(stack)
    return out
