"""Loader for non-courtship free-walking scutellum z-position.

Mirrors the relevant slice of :mod:`utils.courtship_loader` but only extracts
the named keypoint's z-coordinate; no song / sex / locomotion analysis. Used
by the consolidated courtship figure to compare singing male body height
against a population of freely walking flies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from utils.io_dict_to_hdf5 import load as h5_load


def _resolve_kp_idx(info: dict, kp_name: str) -> int:
    raw = info.get('kp_names', info.get('site_names_egocentric', []))
    if isinstance(raw, dict):
        names = [raw[k] for k in sorted(raw.keys(), key=lambda x: int(x))]
    else:
        names = list(raw)
    if kp_name not in names:
        raise KeyError(f'keypoint {kp_name!r} not in info kp_names: {names}')
    return names.index(kp_name)


def load_free_walking_scutellum_z(
    h5_path: str | Path,
    kp_name: str = 'Scutellum',
    bout_keys: Optional[Sequence[str]] = None,
    enable_jax: bool = False,
    per_bout: bool = False,
    min_frames: int = 1,
) -> np.ndarray:
    """Return scutellum z (mm) for a free-walking combined h5.

    Parameters
    ----------
    h5_path : path to a free-walking combined h5 with per-bout
        ``kp_data`` arrays of shape (T, N, 3) (or flat (T, N*3)).
    kp_name : keypoint to extract; default 'Scutellum'.
    bout_keys : optional subset of bout keys to load; default = all bouts
        in the file (excluding 'info').
    per_bout : if True, return one ``np.nanmean`` per bout instead of the
        concatenated frame-level array. Bouts with fewer than ``min_frames``
        finite samples are skipped.
    min_frames : minimum number of finite z samples a bout must have to
        contribute when ``per_bout=True``.

    Returns
    -------
    np.ndarray
        Shape ``(sum_T,)`` of scutellum z values (NaNs dropped) when
        ``per_bout=False`` (default), or ``(n_bouts,)`` of per-bout means
        when ``per_bout=True``.
    """
    data = h5_load(str(h5_path), enable_jax=enable_jax)
    info = data.get('info', {}) or {}
    idx = _resolve_kp_idx(info, kp_name)

    keys: List[str] = (
        list(bout_keys) if bout_keys is not None
        else sorted(k for k in data.keys() if k != 'info')
    )

    if per_bout:
        means: List[float] = []
        for k in keys:
            bout = data[k]
            kp = np.asarray(bout['kp_data'])
            if kp.ndim == 2:
                kp = kp.reshape(kp.shape[0], -1, 3)
            z = kp[:, idx, 2].astype(float)
            z = z[np.isfinite(z)]
            if z.size >= int(min_frames):
                means.append(float(np.mean(z)))
        return np.asarray(means, dtype=float)

    chunks: List[np.ndarray] = []
    for k in keys:
        bout = data[k]
        kp = np.asarray(bout['kp_data'])
        if kp.ndim == 2:
            kp = kp.reshape(kp.shape[0], -1, 3)
        z = kp[:, idx, 2].astype(float)
        chunks.append(z[np.isfinite(z)])
    if not chunks:
        return np.zeros(0, dtype=float)
    return np.concatenate(chunks)
