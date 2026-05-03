"""Loader for non-courtship free-running scutellum z-position.

Mirrors the relevant slice of :mod:`utils.courtship_loader` but only extracts
the named keypoint's z-coordinate; no song / sex / locomotion analysis. Used
by the consolidated courtship figure to compare singing male body height
against a population of freely running flies.
"""
from __future__ import annotations

import collections
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils.io_dict_to_hdf5 import load as h5_load, save as h5_save


def _resolve_kp_idx(info: dict, kp_name: str) -> int:
    raw = info.get('kp_names', info.get('site_names_egocentric', []))
    if isinstance(raw, dict):
        names = [raw[k] for k in sorted(raw.keys(), key=lambda x: int(x))]
    else:
        names = list(raw)
    if kp_name not in names:
        raise KeyError(f'keypoint {kp_name!r} not in info kp_names: {names}')
    return names.index(kp_name)


def load_free_running_scutellum_z(
    h5_path: str | Path,
    kp_name: str = 'Scutellum',
    bout_keys: Optional[Sequence[str]] = None,
    enable_jax: bool = False,
    per_bout: bool = False,
    min_frames: int = 1,
) -> np.ndarray:
    """Return scutellum z (mm) for a free-running combined h5.

    Parameters
    ----------
    h5_path : path to a free-running combined h5 with per-bout
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


_FREE_WALK_PER_BOUT_FIELDS = (
    'kp_data', 'marker_sites', 'xpos_egocentric', 'qpos', 'qvel',
    'xpos', 'xquat', 'site_xpos',
)
_FREE_WALK_GLOBAL_INFO_KEYS = (
    'kp_names', 'names_qpos', 'names_xpos', 'offsets', 'site_names_egocentric',
)
_FREE_WALK_PER_BOUT_INFO_KEYS = (
    'fly_ids', 'clip_lengths', 'source_flies',
)


def _info_seq(info: dict, key: str) -> list:
    """Coerce an info entry to a plain list (handles dict / ndarray / list)."""
    v = info.get(key)
    if v is None:
        return []
    if isinstance(v, dict):
        return [v[k] for k in sorted(v.keys(), key=lambda x: int(x))]
    if isinstance(v, np.ndarray):
        return v.tolist()
    return list(v)


def _build_csv_index(
    bout_summary_csvs: Sequence[str | Path],
) -> Dict[str, List[dict]]:
    """Return ``{fly_id: [csv_row_dict, ...]}`` from one or more bout summary csvs.

    Each row is a plain dict with all CSV columns. Order is preserved within
    each csv. Multiple csvs are concatenated.
    """
    import pandas as pd  # heavy import; keep lazy

    out: Dict[str, List[dict]] = collections.defaultdict(list)
    for p in bout_summary_csvs:
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            fid = str(row['fly_id'])
            out[fid].append({k: row[k] for k in df.columns})
    return dict(out)


def _build_preproc_index(
    preproc_h5_paths: Sequence[str | Path],
) -> Dict[str, List[Tuple[np.ndarray, int]]]:
    """Return ``{fly_id: [(orig_keypoints_array, n_frames), ...]}``.

    Order is preserved within each preprocessing h5. ``n_frames`` comes from
    ``info['clip_lengths']`` when present, otherwise from the orig array's
    leading dimension.
    """
    out: Dict[str, List[Tuple[np.ndarray, int]]] = collections.defaultdict(list)
    for p in preproc_h5_paths:
        try:
            d = h5_load(str(p))
        except (OSError, KeyError):
            continue
        info = d.get('info', {}) or {}
        fids = _info_seq(info, 'fly_ids')
        cls  = _info_seq(info, 'clip_lengths')
        bout_keys = sorted(k for k in d.keys() if k != 'info')
        for i, k in enumerate(bout_keys):
            if i >= len(fids):
                break
            bout = d[k]
            okp = bout.get('orig_keypoints')
            if okp is None:
                continue
            okp = np.asarray(okp)
            if okp.ndim == 2 and okp.shape[-1] != 3:
                okp = okp.reshape(okp.shape[0], -1, 3)
            n = int(cls[i]) if i < len(cls) else int(okp.shape[0])
            out[str(fids[i])].append((okp, n))
    return dict(out)


def _match_by_n_frames(
    candidates: list,
    n_frames: int,
    *,
    used_flags: list,
    n_frames_getter,
) -> Optional[int]:
    """Greedy unique-match: find the next unused candidate whose n_frames
    equals ``n_frames``. Returns the candidate index on a hit, otherwise None."""
    for i, cand in enumerate(candidates):
        if used_flags[i]:
            continue
        if int(n_frames_getter(cand)) == int(n_frames):
            used_flags[i] = True
            return i
    return None


def export_raw_free_running_h5(
    combined_h5_path: str | Path,
    out_path: str | Path,
    *,
    bout_summary_csvs: Optional[Sequence[str | Path]] = None,
    preproc_h5_paths: Optional[Sequence[str | Path]] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict:
    """Export a "raw-data only" free-running h5 from an existing combined h5.

    The input is the already-merged ``ik_output_combined_v1_free_running.h5``.
    All bouts are kept (no song or pair filter — free running has no notion of
    pairs). For each bout, copies the standard raw arrays
    (``kp_data, marker_sites, xpos_egocentric, qpos, qvel, xpos, xquat,
    site_xpos``) but drops ``geometric_angles``.

    When ``bout_summary_csvs`` is provided, each CSV row is matched to a bout
    by ``(fly_id, n_frames == clip_lengths)`` using greedy unique matching.
    All CSV columns of the matched row are added as per-bout info arrays.

    When ``preproc_h5_paths`` is provided, ``orig_keypoints`` from the matching
    preprocessing h5 bout (matched by the same ``(fly_id, n_frames)`` rule) is
    attached. Bouts with no match get no ``orig_keypoints`` field.

    Parameters
    ----------
    combined_h5_path : path
        Source merged free-running h5.
    out_path : path
        Destination path for the exported h5.
    bout_summary_csvs : sequence of paths, optional
        ``free_running_bouts_summary.csv`` files (one per Predictions_3D dir).
    preproc_h5_paths : sequence of paths, optional
        ``preprocessed_bout_*_free_running.h5`` files for ``orig_keypoints``.
    overwrite : bool
        If False (default), raise ``FileExistsError`` when ``out_path`` exists.
    verbose : bool
        Print progress + match stats.

    Returns
    -------
    dict
        ``{n_input_bouts, n_kept_bouts, n_csv_matched, n_orig_attached,
        out_path, missing_fields}``.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists; pass overwrite=True to replace."
        )

    if verbose:
        print(f'loading combined h5: {combined_h5_path}')
    data = h5_load(str(combined_h5_path))
    info = data.get('info', {}) or {}
    bout_keys = sorted(k for k in data.keys() if k != 'info')
    n_input = len(bout_keys)
    if not bout_keys:
        raise ValueError(f'no bouts in {combined_h5_path}')

    fly_ids_seq    = _info_seq(info, 'fly_ids')
    clip_lens_seq  = _info_seq(info, 'clip_lengths')
    if len(fly_ids_seq) != n_input or len(clip_lens_seq) != n_input:
        raise RuntimeError(
            f"info length mismatch: fly_ids={len(fly_ids_seq)}, "
            f"clip_lengths={len(clip_lens_seq)}, n_bouts={n_input}"
        )

    # Build matching indices.
    csv_index: Dict[str, List[dict]] = {}
    if bout_summary_csvs:
        csv_index = _build_csv_index(bout_summary_csvs)
        if verbose:
            n_rows = sum(len(v) for v in csv_index.values())
            print(f'  csv index: {n_rows} rows across {len(csv_index)} fly_ids '
                  f'(from {len(list(bout_summary_csvs))} csv file(s))')

    preproc_index: Dict[str, List[Tuple[np.ndarray, int]]] = {}
    if preproc_h5_paths:
        preproc_index = _build_preproc_index(preproc_h5_paths)
        if verbose:
            n_rows = sum(len(v) for v in preproc_index.values())
            print(f'  preproc index: {n_rows} bouts across {len(preproc_index)} '
                  f'fly_ids (from {len(list(preproc_h5_paths))} preproc h5(s))')

    # Per-fly_id used flags so we never double-match a CSV/preproc row.
    csv_used: Dict[str, list] = {fid: [False] * len(rows) for fid, rows in csv_index.items()}
    pre_used: Dict[str, list] = {fid: [False] * len(rows) for fid, rows in preproc_index.items()}

    # Per-bout payload.
    pad = max(3, len(str(max(n_input - 1, 0))))
    out_data: dict = {}
    missing_field_counts: Dict[str, int] = {f: 0 for f in _FREE_WALK_PER_BOUT_FIELDS}
    matched_csv_rows: List[Optional[dict]] = []
    n_orig_attached = 0

    for new_idx, old_key in enumerate(bout_keys):
        bout = data[old_key]
        new_key = f'bout_{new_idx:0{pad}d}'
        new_bout: dict = {}
        for field in _FREE_WALK_PER_BOUT_FIELDS:
            if field in bout:
                v = bout[field]
                new_bout[field] = v if isinstance(v, dict) else np.asarray(v)
            else:
                missing_field_counts[field] += 1

        fid = str(fly_ids_seq[new_idx])
        n_frames = int(clip_lens_seq[new_idx])

        # CSV match.
        csv_row: Optional[dict] = None
        if csv_index and fid in csv_index:
            j = _match_by_n_frames(
                csv_index[fid], n_frames,
                used_flags=csv_used[fid],
                n_frames_getter=lambda r: r['n_frames'],
            )
            if j is not None:
                csv_row = csv_index[fid][j]
        matched_csv_rows.append(csv_row)

        # orig_keypoints match.
        if preproc_index and fid in preproc_index:
            j = _match_by_n_frames(
                preproc_index[fid], n_frames,
                used_flags=pre_used[fid],
                n_frames_getter=lambda r: r[1],
            )
            if j is not None:
                okp = preproc_index[fid][j][0]
                new_bout['orig_keypoints'] = okp
                n_orig_attached += 1

        out_data[new_key] = new_bout

    n_csv_matched = sum(r is not None for r in matched_csv_rows)

    # Build out_info: copy globals + per-bout info, plus attached CSV columns.
    out_info: dict = {}
    for k in _FREE_WALK_GLOBAL_INFO_KEYS:
        if k in info:
            out_info[k] = info[k]
    for k in _FREE_WALK_PER_BOUT_INFO_KEYS:
        seq = _info_seq(info, k)
        if seq:
            if len(seq) != n_input:
                raise RuntimeError(
                    f"info['{k}'] length {len(seq)} != n_input {n_input}"
                )
            out_info[k] = list(seq)

    # CSV columns -> per-bout info arrays. Use sentinel '' for ints/floats and
    # NaN for floats? Simpler: use None when no match; ioh5.save handles
    # int/float/str scalars uniformly.
    if csv_index:
        # Collect the union of CSV columns (in stable order from first matched row).
        col_order: List[str] = []
        for row in matched_csv_rows:
            if row is None:
                continue
            for c in row.keys():
                if c not in col_order:
                    col_order.append(c)
        for c in col_order:
            arr: List = []
            for row in matched_csv_rows:
                if row is None:
                    arr.append('')  # missing-row sentinel; consumer can filter on csv_matched.
                else:
                    arr.append(row[c])
            out_info[f'csv__{c}'] = arr
        out_info['csv_matched'] = [r is not None for r in matched_csv_rows]

    if verbose:
        missing_summary = {f: c for f, c in missing_field_counts.items() if c}
        if missing_summary:
            print(f'missing per-bout fields (skipped): {missing_summary}')
        if csv_index:
            print(f'csv metadata attached to {n_csv_matched}/{n_input} bouts')
        if preproc_index:
            print(f'orig_keypoints attached to {n_orig_attached}/{n_input} bouts')
        print(f'writing {n_input} bouts to {out_path} ...')

    h5_save(str(out_path), {**out_data, 'info': out_info})

    if verbose:
        print(f'done: {n_input} bouts -> {out_path}')

    return {
        'n_input_bouts': n_input,
        'n_kept_bouts': n_input,
        'n_csv_matched': n_csv_matched,
        'n_orig_attached': n_orig_attached,
        'out_path': str(out_path),
        'missing_fields': {f: c for f, c in missing_field_counts.items() if c},
    }
