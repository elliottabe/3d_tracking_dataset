"""Cached Pslow / Pfast classification across all per-pair song results.

Pools waveforms from every pair, fits one :class:`PulseTypeModel`, then
classifies each pair's pulses (per side). Also returns the pooled raw
waveforms per type so the caller can plot mean +/- std. Cached to a
pickle so the notebook does not refit on every run.

Return shape::

    {
      'labels':   {pair_idx: {'L': labels, 'R': labels}},
      'centroids': {'Pslow': (W,), 'Pfast': (W,)},
      'counts':    {'Pslow': int,  'Pfast': int},
      'pooled_waveforms': {'Pslow': (N_slow, W), 'Pfast': (N_fast, W)},
      'fs':        float,
    }
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from utils.pulse_types import (
    PulseTypeConfig,
    PulseTypeModel,
    classify_pulses,
    fit_pulse_type_model,
)


CACHE_VERSION = 2  # bump when the cached dict schema changes


def _gather_waveforms(results: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (waveform, symmetry) arrays across all pairs and sides."""
    wfs, syms = [], []
    for r in results:
        for side in ('L', 'R'):
            pf = r.get('song0', {}).get('sides', {}).get(side, {}).get('pulse_features')
            if not pf:
                continue
            w = np.asarray(pf.get('waveforms', np.zeros((0, 0))))
            s = np.asarray(pf.get('symmetry', np.zeros(0)))
            if w.ndim != 2 or w.shape[0] == 0 or s.shape[0] != w.shape[0]:
                continue
            wfs.append(w)
            syms.append(s)
    if not wfs:
        return np.zeros((0, 0)), np.zeros(0)
    W = max(w.shape[1] for w in wfs)
    if any(w.shape[1] != W for w in wfs):
        padded = []
        for w in wfs:
            if w.shape[1] == W:
                padded.append(w)
            else:
                pad = np.full((w.shape[0], W - w.shape[1]), np.nan)
                padded.append(np.concatenate([w, pad], axis=1))
        wfs = padded
    waveforms = np.concatenate(wfs, axis=0)
    symmetry = np.concatenate(syms, axis=0)
    finite = np.isfinite(waveforms).all(axis=1) & np.isfinite(symmetry)
    return waveforms[finite], symmetry[finite]


def get_pulse_type_labels(
    results: Sequence[dict],
    cache_path: Optional[str | Path] = None,
    force: bool = False,
    cfg: Optional[PulseTypeConfig] = None,
    fs: float = 800.0,
) -> Dict[str, Any]:
    """Fit one PulseTypeModel pooled across all pairs and classify each side.

    Returns a dict with keys ``labels``, ``centroids``, ``counts``,
    ``pooled_waveforms``, ``fs`` (see module docstring).
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and cached.get('version') == CACHE_VERSION:
                return cached
            # stale schema — fall through to refit

    if cfg is None:
        cfg = PulseTypeConfig()
    pooled_w, pooled_s = _gather_waveforms(results)
    empty: Dict[str, Any] = {
        'version':   CACHE_VERSION,
        'labels':    {},
        'centroids': {'Pslow': np.zeros(0), 'Pfast': np.zeros(0)},
        'counts':    {'Pslow': 0, 'Pfast': 0},
        'pooled_waveforms': {'Pslow': np.zeros((0, 0)), 'Pfast': np.zeros((0, 0))},
        'fs':        float(fs),
    }
    if pooled_w.shape[0] < cfg.n_components:
        return empty
    model: PulseTypeModel = fit_pulse_type_model(pooled_w, pooled_s, cfg)

    # Classify each pulse in the pooled set so we can stratify waveforms.
    pooled_labels = classify_pulses(pooled_w, model)
    pooled_by_type = {
        name: pooled_w[pooled_labels == name]
        for name in ('Pslow', 'Pfast')
    }

    # Centroids: rebuild from pooled-classified data (matches counts).
    centroids = {
        name: (arr.mean(axis=0) if arr.shape[0] else np.zeros(pooled_w.shape[1]))
        for name, arr in pooled_by_type.items()
    }
    counts = {name: int(arr.shape[0]) for name, arr in pooled_by_type.items()}

    labels_out: Dict[int, Dict[str, np.ndarray]] = {}
    for r in results:
        pair_idx = int(r.get('pair_idx', -1))
        side_labels: Dict[str, np.ndarray] = {}
        for side in ('L', 'R'):
            pf = r.get('song0', {}).get('sides', {}).get(side, {}).get('pulse_features')
            if not pf:
                continue
            w = np.asarray(pf.get('waveforms', np.zeros((0, 0))))
            if w.ndim != 2 or w.shape[0] == 0:
                continue
            finite = np.isfinite(w).all(axis=1)
            if not finite.any():
                continue
            labels_finite = classify_pulses(w[finite], model)
            labels = np.empty(w.shape[0], dtype=object)
            labels[:] = ''
            labels[finite] = labels_finite
            side_labels[side] = labels
        if side_labels:
            labels_out[pair_idx] = side_labels

    out: Dict[str, Any] = {
        'version':          CACHE_VERSION,
        'labels':           labels_out,
        'centroids':        centroids,
        'counts':           counts,
        'pooled_waveforms': pooled_by_type,
        'fs':               float(fs),
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(out, f)
    return out
