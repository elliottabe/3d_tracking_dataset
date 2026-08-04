"""Tests for `trim_bouts_to_valid_span` (Task 9b).

Background: `filtering.interpolation.max_edge_extrap_frames` caps edge
gap-filling during preprocessing; frames beyond that cap stay NaN. The
comment in `configs/preprocessing/default.yaml` claims those frames get
"dropped by pair_validity", but `pair_validity.enabled` is False for
single-fly `free_running`, so nothing drops them. They survive into the STAC
output and reach `interpolate_trajectory`, whose `scipy.interp1d(kind='cubic')`
is a *global* solve -- a single leftover NaN poisons the entire interpolated
array for that bout.

`trim_bouts_to_valid_span` trims each bout to its first/last valid (non-NaN)
frame before interpolation, so edge NaN runs are dropped instead of poisoning
the whole bout. Interior NaN gaps (not fixable by trimming) must be reported,
never silently passed through.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.postprocess_stac_data import trim_bouts_to_valid_span

TEMPORAL_KEYS = ['qpos', 'qvel', 'xpos', 'xquat', 'kp_data', 'marker_sites', 'site_xpos']


def _make_bout(T, nq=6, nkp=4, seed=0):
    """A synthetic bout dict with every temporal array the real pipeline
    carries at the trim call-site, plus one non-temporal array ('offsets')
    whose leading dim never matches T."""
    rng = np.random.default_rng(seed)
    bout = {
        'qpos': rng.normal(size=(T, nq)),
        'qvel': rng.normal(size=(T, nq - 1)),
        'xpos': rng.normal(size=(T, 3, 3)),
        'xquat': rng.normal(size=(T, 3, 4)),
        'kp_data': rng.normal(size=(T, nkp, 3)),
        'marker_sites': rng.normal(size=(T, nkp, 3)),
        'site_xpos': rng.normal(size=(T, 5, 3)),
        # Non-temporal: leading dim (nkp+7) deliberately never equals T.
        'offsets': rng.normal(size=(nkp + 7, 3)),
    }
    return bout


def _nan_frames(bout, key, frame_idxs):
    arr = bout[key]
    arr = arr.copy()
    for i in frame_idxs:
        arr[i, ...] = np.nan
    bout[key] = arr


def test_leading_only_nan_is_trimmed():
    T = 20
    bout = _make_bout(T)
    _nan_frames(bout, 'kp_data', range(0, 4))  # frames 0..3 bad
    bout_dict = {'bout_000': bout, 'info': {}}

    summary = trim_bouts_to_valid_span(bout_dict, ['bout_000'], verbose=False)

    assert summary['bouts_trimmed'] == 1
    assert summary['frames_dropped'] == 4
    assert summary['interior_nan_bouts'] == []
    assert bout_dict['bout_000']['qpos'].shape[0] == T - 4
    assert not np.isnan(bout_dict['bout_000']['kp_data']).any()


def test_trailing_only_nan_is_trimmed():
    T = 20
    bout = _make_bout(T)
    _nan_frames(bout, 'kp_data', range(T - 3, T))  # last 3 frames bad
    bout_dict = {'bout_000': bout, 'info': {}}

    summary = trim_bouts_to_valid_span(bout_dict, ['bout_000'], verbose=False)

    assert summary['bouts_trimmed'] == 1
    assert summary['frames_dropped'] == 3
    assert summary['interior_nan_bouts'] == []
    assert bout_dict['bout_000']['qpos'].shape[0] == T - 3
    assert not np.isnan(bout_dict['bout_000']['kp_data']).any()


def test_both_ends_nan_is_trimmed():
    T = 20
    bout = _make_bout(T)
    _nan_frames(bout, 'kp_data', [0, 1])
    _nan_frames(bout, 'kp_data', [T - 1])
    bout_dict = {'bout_000': bout, 'info': {}}

    summary = trim_bouts_to_valid_span(bout_dict, ['bout_000'], verbose=False)

    assert summary['bouts_trimmed'] == 1
    assert summary['frames_dropped'] == 3
    assert summary['interior_nan_bouts'] == []
    assert bout_dict['bout_000']['qpos'].shape[0] == T - 3
    assert not np.isnan(bout_dict['bout_000']['kp_data']).any()


def test_no_nan_is_a_true_noop():
    """Clean bouts must be returned unchanged -- same arrays, not copies."""
    T = 20
    bout = _make_bout(T)
    orig_qpos = bout['qpos']
    orig_kp = bout['kp_data']
    bout_dict = {'bout_000': bout, 'info': {}}

    summary = trim_bouts_to_valid_span(bout_dict, ['bout_000'], verbose=False)

    assert summary['bouts_trimmed'] == 0
    assert summary['frames_dropped'] == 0
    assert summary['interior_nan_bouts'] == []
    # Identity, not just equality: no-op means no slicing/reassignment happened.
    assert bout_dict['bout_000']['qpos'] is orig_qpos
    assert bout_dict['bout_000']['kp_data'] is orig_kp


def test_interior_nan_is_reported_not_silently_trimmed():
    T = 20
    bout = _make_bout(T)
    _nan_frames(bout, 'kp_data', [10])  # a gap in the middle, clean edges
    bout_dict = {'bout_000': bout, 'info': {}}

    summary = trim_bouts_to_valid_span(bout_dict, ['bout_000'], verbose=False)

    assert summary['bouts_trimmed'] == 0
    assert summary['frames_dropped'] == 0
    assert summary['interior_nan_bouts'] == ['bout_000']
    # Left untouched -- length unchanged and the interior NaN still present
    # (trimming cannot fix it; we must not pretend otherwise).
    assert bout_dict['bout_000']['qpos'].shape[0] == T
    assert np.isnan(bout_dict['bout_000']['kp_data'][10]).all()


def test_all_nan_bout_is_untouched_and_reported():
    T = 20
    bout = _make_bout(T)
    bout['kp_data'][:] = np.nan
    bout_dict = {'bout_000': bout, 'info': {}}

    summary = trim_bouts_to_valid_span(bout_dict, ['bout_000'], verbose=False)

    assert summary['bouts_trimmed'] == 0
    assert summary['frames_dropped'] == 0
    assert summary['interior_nan_bouts'] == ['bout_000']
    # Not a zero-length bout, not modified at all.
    assert bout_dict['bout_000']['qpos'].shape[0] == T
    assert np.isnan(bout_dict['bout_000']['kp_data']).all()


def test_every_temporal_array_sliced_consistently():
    T = 20
    bout = _make_bout(T)
    _nan_frames(bout, 'kp_data', [0, 1, 2])
    _nan_frames(bout, 'kp_data', [T - 1])
    bout_dict = {'bout_000': bout, 'info': {}}

    trim_bouts_to_valid_span(bout_dict, ['bout_000'], verbose=False)

    new_len = T - 4
    for key in TEMPORAL_KEYS:
        assert bout_dict['bout_000'][key].shape[0] == new_len, (
            f"{key} not sliced to consistent length"
        )


def test_non_temporal_array_not_sliced():
    T = 20
    bout = _make_bout(T)
    orig_offsets_shape = bout['offsets'].shape
    _nan_frames(bout, 'kp_data', [0, 1, T - 1])
    bout_dict = {'bout_000': bout, 'info': {}}

    trim_bouts_to_valid_span(bout_dict, ['bout_000'], verbose=False)

    assert bout_dict['bout_000']['offsets'].shape == orig_offsets_shape


def test_qpos_only_nan_with_clean_kp_data_still_trims():
    T = 20
    bout = _make_bout(T)
    _nan_frames(bout, 'qpos', [0])  # kp_data stays fully clean
    bout_dict = {'bout_000': bout, 'info': {}}

    summary = trim_bouts_to_valid_span(bout_dict, ['bout_000'], verbose=False)

    assert summary['bouts_trimmed'] == 1
    assert summary['frames_dropped'] == 1
    assert summary['interior_nan_bouts'] == []
    assert bout_dict['bout_000']['qpos'].shape[0] == T - 1
    assert not np.isnan(bout_dict['bout_000']['qpos']).any()


def test_multiple_bouts_summary_aggregates():
    bout_a = _make_bout(15, seed=1)
    _nan_frames(bout_a, 'kp_data', [0])
    bout_b = _make_bout(25, seed=2)
    _nan_frames(bout_b, 'kp_data', [23, 24])
    bout_c = _make_bout(10, seed=3)  # clean, no-op

    bout_dict = {'bout_000': bout_a, 'bout_001': bout_b, 'bout_002': bout_c, 'info': {}}
    summary = trim_bouts_to_valid_span(
        bout_dict, ['bout_000', 'bout_001', 'bout_002'], verbose=False
    )

    assert summary['bouts_trimmed'] == 2
    assert summary['frames_dropped'] == 3
    assert summary['interior_nan_bouts'] == []
