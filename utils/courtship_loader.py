"""Shared data loading for courtship analysis notebooks.

Loads combined h5 files, pairs fly0/fly1 bouts, applies despiking,
runs song analysis + sex ID, and reorders so slot-0 = male, slot-1 = female.

Usage::

    from utils.courtship_loader import load_courtship_h5, pair_bouts, get_fields, analyze_pair

    data, info, kp_names, bout_keys = load_courtship_h5(h5_path)
    pairs = pair_bouts(bout_keys, info)
    for k0, k1 in pairs:
        kp0, xp0, q0 = get_fields(data[k0])
        kp1, xp1, q1 = get_fields(data[k1])
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from utils.io_dict_to_hdf5 import load as h5_load
from utils.keypoint_filter import despike_isolated_spikes, medfilt_despike
from utils.pair_validity import PairValidityConfig, compute_pair_validity
from utils.song_analysis import SongAnalysisConfig, analyze_fly_song
from utils.sex_id import SexIdConfig, identify_male_female
from utils.locomotion import (
    LocomotionConfig,
    compute_centroid_velocity,
    compute_com_height,
    classify_walking_state,
    summarize_by_song,
)


def load_courtship_h5(
    h5_path: str | Path,
    enable_jax: bool = False,
) -> Tuple[dict, dict, List[str], List[str]]:
    """Load a combined courtship h5 and extract metadata.

    Returns
    -------
    data : dict
        Full h5 contents (bout dicts + 'info').
    info : dict
        The ``data['info']`` sub-dict.
    kp_names : list[str]
        Keypoint names from ``info['kp_names']``.
    bout_keys : list[str]
        Sorted bout key names (excluding 'info').
    """
    data = h5_load(str(h5_path), enable_jax=enable_jax)
    info = data.get('info', {}) or {}

    # kp_names may be stored as list or dict
    raw_names = info.get('kp_names', info.get('site_names_egocentric', []))
    if isinstance(raw_names, dict):
        kp_names = [raw_names[k] for k in sorted(raw_names.keys(), key=lambda x: int(x))]
    else:
        kp_names = list(raw_names)

    bout_keys = sorted(k for k in data.keys() if k != 'info')
    return data, info, kp_names, bout_keys


def pair_bouts(
    bout_keys: Sequence[str],
    info: dict,
) -> List[Tuple[str, str]]:
    """Pair consecutive fly0/fly1 bout keys using info['source_flies'].

    Falls back to simple even/odd pairing when source_flies is absent.
    """
    src = list(info.get('source_flies', []))
    bucket = list(info.get('bucket', []))
    pairs: List[Tuple[str, str]] = []

    if src and len(src) == len(bout_keys):
        i = 0
        while i + 1 < len(bout_keys):
            if (src[i] == 'fly0' and src[i + 1] == 'fly1'
                    and (not bucket or bucket[i] == 'both' == bucket[i + 1])):
                pairs.append((bout_keys[i], bout_keys[i + 1]))
                i += 2
            else:
                i += 1
    else:
        for i in range(0, len(bout_keys) - 1, 2):
            pairs.append((bout_keys[i], bout_keys[i + 1]))
    return pairs


def get_fields(
    bout: dict,
    despike: bool = True,
    despike_iterations: int = 1,
    medfilt_clean: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract kp_data, xpos_egocentric, qpos from a single bout dict.

    Parameters
    ----------
    bout : dict
        A single bout's data (e.g. ``data['bout_000']``).
    despike : bool
        Apply velocity-reversal spike removal.
    despike_iterations : int
        Number of despiking passes.  1 = single-frame only (safe for male
        song).  Use higher values (e.g. 6) for non-singing flies where
        multi-frame tracking glitches are common.
    medfilt_clean : bool
        After despiking, also apply a median-filter pass to catch
        multi-frame tracking excursions that lack a velocity reversal.
        **Not safe for signals with fast oscillations (male wing song).**
        Only enable for non-singing flies.

    Returns
    -------
    kp : ndarray, shape (T, N, 3)
        World-frame keypoints.
    xpos_ego : ndarray or None, shape (T, N, 3)
        Egocentric site positions (if present).
    qpos : ndarray or None, shape (T, D)
        Joint angles (if present).
    """
    kp = np.asarray(bout['kp_data'])
    if kp.ndim == 2:
        kp = kp.reshape(kp.shape[0], -1, 3)
    if despike:
        kp, _ = despike_isolated_spikes(kp, max_iterations=despike_iterations)
    if medfilt_clean:
        kp, _ = medfilt_despike(kp)

    xp = bout.get('xpos_egocentric')
    if xp is not None:
        xp = np.asarray(xp)
        if xp.ndim == 2:
            xp = xp.reshape(xp.shape[0], -1, 3)
        if despike:
            xp, _ = despike_isolated_spikes(xp, max_iterations=despike_iterations)
        if medfilt_clean:
            xp, _ = medfilt_despike(xp)

    qp = bout.get('qpos')
    if qp is not None:
        qp = np.asarray(qp)
        if despike:
            qp, _ = despike_isolated_spikes(qp, max_iterations=despike_iterations)
        if medfilt_clean:
            qp, _ = medfilt_despike(qp)

    return kp, xp, qp


def analyze_pair(
    key0: str,
    key1: str,
    bout0: dict,
    bout1: dict,
    kp_names: Sequence[str],
    *,
    song_cfg: Optional[SongAnalysisConfig] = None,
    sex_cfg: Optional[SexIdConfig] = None,
    loc_cfg: Optional[LocomotionConfig] = None,
    pair_cfg: Optional[PairValidityConfig] = None,
    despike: bool = True,
) -> dict:
    """Run full per-pair analysis: song, sex ID, male-first reorder, locomotion.

    After this call, slot-0 is always male and slot-1 is always female.
    Tracker-slot fields are preserved for provenance.

    Parameters
    ----------
    key0, key1 : str
        Bout keys (e.g. 'bout_000', 'bout_001').
    bout0, bout1 : dict
        Bout data dicts from the h5.
    kp_names : sequence of str
        Keypoint names.
    song_cfg, sex_cfg, loc_cfg, pair_cfg : config objects, optional
        Analysis configs. Defaults are created if None.
    despike : bool
        Apply spike removal when loading fields.

    Returns
    -------
    dict with keys: key0, key1, T, valid_fly0, valid_fly1, colocated,
        song0, song1, sex, tracker_key0, tracker_key1, tracker_male_id,
        tracker_song_fraction_fly0, tracker_song_fraction_fly1,
        male_labels, male_valid, kin, com_z, floor_z, walking_state, by_song.
    """
    if song_cfg is None:
        song_cfg = SongAnalysisConfig()
    if sex_cfg is None:
        sex_cfg = SexIdConfig()
    if loc_cfg is None:
        loc_cfg = LocomotionConfig()
    if pair_cfg is None:
        pair_cfg = PairValidityConfig()

    kp0, xp0, q0 = get_fields(bout0, despike=despike)
    kp1, xp1, q1 = get_fields(bout1, despike=despike)

    # Clip to common length
    T = min(len(kp0), len(kp1))
    kp0, kp1 = kp0[:T], kp1[:T]
    if xp0 is not None: xp0 = xp0[:T]
    if xp1 is not None: xp1 = xp1[:T]
    if q0  is not None: q0  = q0[:T]
    if q1  is not None: q1  = q1[:T]

    # Pair validity
    pv = compute_pair_validity(kp0, kp1, kp_names, cfg=pair_cfg)
    v0 = np.asarray(pv['valid_fly0']).astype(bool)
    v1 = np.asarray(pv['valid_fly1']).astype(bool)
    coloc = np.asarray(pv['pair_colocated']).astype(bool)

    # Song analysis
    song0 = analyze_fly_song(kp0, xp0, q0, kp_names, cfg=song_cfg, valid_mask=v0)
    song1 = analyze_fly_song(kp1, xp1, q1, kp_names, cfg=song_cfg, valid_mask=v1)

    # Sex ID
    sex = identify_male_female(song0, song1, kp0, kp1, kp_names, cfg=sex_cfg)

    # Preserve tracker-slot assignment before swapping
    tracker_song_fraction_fly0 = float(song0['summary']['song_fraction'])
    tracker_song_fraction_fly1 = float(song1['summary']['song_fraction'])
    tracker_key0 = key0
    tracker_key1 = key1
    tracker_male_id = sex['male_id']

    # Reorder so slot-0 = male, slot-1 = female
    if sex['male_id'] == 'fly1':
        key0, key1 = key1, key0
        kp0, kp1 = kp1, kp0
        xp0, xp1 = xp1, xp0
        q0, q1 = q1, q0
        v0, v1 = v1, v0
        song0, song1 = song1, song0
        sex['male_id'] = 'fly0'
        sex['female_id'] = 'fly1'

    # Locomotion for male (slot-0)
    bl = sex['body_length_male'] if np.isfinite(sex['body_length_male']) else None
    kin = compute_centroid_velocity(kp0, kp_names, loc_cfg, body_length=bl)
    com_z, floor_z = compute_com_height(kp0, kp_names, loc_cfg)
    speed_bl = kin.get('speed_bl', kin['speed'])
    wstate = classify_walking_state(np.asarray(speed_bl), loc_cfg)

    # Dominant-wing frame labels
    dw = str(song0.get('dominant_wing', 'L')).upper()
    side_key = 'L' if dw.startswith('L') else 'R'
    song_labels = np.asarray(song0['sides'][side_key]['frame_labels'])

    # Song-conditioned aggregates
    metrics = {
        'forward_speed_bl': np.asarray(kin.get('forward_speed_bl', kin['forward_speed'])),
        'speed_bl':         np.asarray(speed_bl),
        'turn_rate':        np.asarray(kin['turn_rate']),
        'com_z':            np.asarray(com_z),
    }
    by_song = summarize_by_song(song_labels, metrics, valid_mask=v0)

    return {
        'key0': key0, 'key1': key1,
        'T': T,
        'valid_fly0': v0, 'valid_fly1': v1, 'colocated': coloc,
        'song0': song0, 'song1': song1,
        'sex': sex,
        'tracker_key0': tracker_key0,
        'tracker_key1': tracker_key1,
        'tracker_male_id': tracker_male_id,
        'tracker_song_fraction_fly0': tracker_song_fraction_fly0,
        'tracker_song_fraction_fly1': tracker_song_fraction_fly1,
        'male_labels': song_labels,
        'male_valid':  v0,
        'kin': kin,
        'com_z': com_z, 'floor_z': floor_z,
        'walking_state': wstate,
        'by_song': by_song,
    }


def analyze_all_pairs(
    data: dict,
    pairs: List[Tuple[str, str]],
    kp_names: Sequence[str],
    *,
    cache_path: Optional[str | Path] = None,
    force: bool = False,
    song_cfg: Optional[SongAnalysisConfig] = None,
    sex_cfg: Optional[SexIdConfig] = None,
    loc_cfg: Optional[LocomotionConfig] = None,
    pair_cfg: Optional[PairValidityConfig] = None,
    despike: bool = True,
    verbose: bool = True,
) -> List[dict]:
    """Analyze all pairs with optional pickle caching.

    Parameters
    ----------
    data : dict
        Full h5 data dict.
    pairs : list of (key0, key1)
        Output of :func:`pair_bouts`.
    kp_names : sequence of str
        Keypoint names.
    cache_path : path, optional
        Pickle cache location. Loads from cache if it exists and force=False.
    force : bool
        Re-run analysis even if cache exists.
    verbose : bool
        Print progress every 25 pairs.

    Returns
    -------
    list of dict
        Per-pair result dicts from :func:`analyze_pair`.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            if verbose:
                print(f'loading cache: {cache_path}')
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

    results = []
    for i, (k0, k1) in enumerate(pairs):
        try:
            r = analyze_pair(
                k0, k1, data[k0], data[k1], kp_names,
                song_cfg=song_cfg, sex_cfg=sex_cfg,
                loc_cfg=loc_cfg, pair_cfg=pair_cfg,
                despike=despike,
            )
            r['pair_idx'] = i
            results.append(r)
        except Exception as e:
            if verbose:
                print(f'  pair {i} ({k0}/{k1}): {type(e).__name__}: {e}')
        if verbose and (i + 1) % 25 == 0:
            print(f'  processed {i + 1}/{len(pairs)}')

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
        if verbose:
            print(f'cached {len(results)} pair results -> {cache_path}')

    return results
