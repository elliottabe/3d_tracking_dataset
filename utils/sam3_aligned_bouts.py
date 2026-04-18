"""Per-bout male-pitch alignment across all bouts in a ``sam3_aligned.h5`` file.

``sam3_aligned.h5`` holds SAM3 mask-centroid triangulated male/female
COMs concatenated across all courtship bouts of a session, plus
``bout_boundaries`` / ``bout_frames`` tables that map concat-array slices
to per-bout video frames. The stored ``derived/male_heading`` is
horizontal (yaw only), so body **pitch** must still come from per-bout
keypoint CSVs. This module matches the h5 bout index to on-disk
``bout_XXXXX/fly*.csv`` folders and returns pitch-alignment traces.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


# Male KP CSV is conventionally ``fly1.csv`` in these per-bout folders; the
# fly-id mapping in this session (female in slot 0 at IK time) is the same as
# the notebook's session-level override.
MALE_CSV_NAME_DEFAULT = 'fly1.csv'


def _bout_slice(boundaries: np.ndarray, i: int) -> slice:
    """Return the inclusive-end slice for concat-array access.

    ``bout_boundaries`` stores half-open ``[start, end)`` but the concat
    arrays carry one extra frame per bout (the CSV endpoint), so the
    full per-bout slice is ``[start, end+1)``.
    """
    return slice(int(boundaries[i, 0]), int(boundaries[i, 1]) + 1)


def load_sam3_aligned(h5_path: str | Path) -> Dict[str, np.ndarray]:
    """Load derived arrays + bout bookkeeping from ``sam3_aligned.h5``.

    Returns a dict with keys ``male_centroid``, ``female_centroid``,
    ``bout_boundaries``, ``bout_frames``, ``sex_swaps`` (concat arrays in
    whatever units the file stores; DLT convention is cm in this repo).
    """
    with h5py.File(str(h5_path), 'r') as f:
        out = {
            'male_centroid':   f['derived/male_centroid'][:],
            'female_centroid': f['derived/female_centroid'][:],
            'bout_boundaries': f['bout_boundaries'][:],
            'bout_frames':     f['bout_frames'][:],
            'sex_swaps':       f['sex_swaps'][:],
        }
    return out


def load_per_bout_male_kp(
    root: str | Path,
    bout_name: str,
    male_csv: str = MALE_CSV_NAME_DEFAULT,
) -> Tuple[np.ndarray, List[str]]:
    """Load a per-bout male-KP CSV and return ``(T, K, 3)`` + keypoint names.

    CSV has a two-row header (keypoint name, then ``x/y/z/conf``); the
    first column is the absolute video frame. This helper drops the
    frame column and the ``conf`` slot to return XYZ only.
    """
    path = Path(root) / bout_name / male_csv
    df = pd.read_csv(path, header=[0, 1])
    cols = df.columns.tolist()
    if cols[0][0].lower() != 'frame':
        raise ValueError(f'{path}: first column is not "frame"')
    body = df.iloc[:, 1:]
    kp_names: List[str] = []
    for name, axis in body.columns:
        if axis == 'x':
            kp_names.append(str(name))
    arr = body.to_numpy(dtype=float).reshape(len(df), len(kp_names), 4)
    return arr[:, :, :3], kp_names


def _body_pitch_deg(head: np.ndarray, scut: np.ndarray) -> np.ndarray:
    vec = head - scut
    n = np.linalg.norm(vec, axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.degrees(np.arcsin(np.divide(
            vec[..., 2], n,
            out=np.full_like(n, np.nan), where=n > 0)))


def _target_pitch_deg(
    male_scut: np.ndarray, female_com: np.ndarray,
) -> np.ndarray:
    vec = female_com - male_scut
    n = np.linalg.norm(vec, axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.degrees(np.arcsin(np.divide(
            vec[..., 2], n,
            out=np.full_like(n, np.nan), where=n > 0)))


def compute_per_bout_pitch_alignment(
    h5_path: str | Path,
    bouts_root: str | Path,
    kp_scale: float = 0.1,
    male_csv: str = MALE_CSV_NAME_DEFAULT,
    head_name: str = 'Antenna_Base',
    scut_name: str = 'Scutellum',
) -> Dict[str, object]:
    """Return per-bout alignment traces (body_pitch − target_pitch, degrees).

    The h5 stores male/female COMs in DLT units (cm); ``kp_scale``
    converts those to KP units (mm) so they subtract cleanly from the
    per-bout male Scutellum coordinate. The order of bouts in the
    returned list matches ``bout_boundaries`` row order (i.e. the h5
    bout index, 0..N-1).
    """
    meta = load_sam3_aligned(h5_path)
    bb = meta['bout_boundaries']
    n_bouts = bb.shape[0]
    bouts_root = Path(bouts_root)
    bout_names = sorted(p.name for p in bouts_root.glob('bout_*')
                        if (p / male_csv).exists())
    if len(bout_names) < n_bouts:
        raise FileNotFoundError(
            f'found {len(bout_names)} bout_* folders with {male_csv}; '
            f'h5 expects {n_bouts}'
        )
    bout_names = bout_names[:n_bouts]

    per_bout: List[np.ndarray] = []
    summaries: List[float] = []
    for i, name in enumerate(bout_names):
        sl = _bout_slice(bb, i)
        male_com_mm   = meta['male_centroid'][sl]   / kp_scale
        female_com_mm = meta['female_centroid'][sl] / kp_scale
        kp, kp_names = load_per_bout_male_kp(bouts_root, name, male_csv)
        if head_name not in kp_names or scut_name not in kp_names:
            raise KeyError(
                f'{name}: missing keypoints {head_name!r}/{scut_name!r} '
                f'(have {kp_names[:6]}...)'
            )
        T = min(kp.shape[0], male_com_mm.shape[0])
        head = kp[:T, kp_names.index(head_name), :]
        scut = kp[:T, kp_names.index(scut_name), :]
        body_pitch = _body_pitch_deg(head, scut)
        target_pitch = _target_pitch_deg(scut, female_com_mm[:T])
        alignment = body_pitch - target_pitch
        per_bout.append(alignment)
        finite = np.isfinite(alignment)
        summaries.append(
            float(np.median(np.abs(alignment[finite]))) if finite.any()
            else float('nan')
        )

    return {
        'alignment_per_bout': per_bout,
        'bout_names':         bout_names,
        'bout_frames':        meta['bout_frames'],
        'bout_boundaries':    bb,
        'median_abs_alignment_deg': np.asarray(summaries, dtype=float),
    }
