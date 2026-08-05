"""Tests for `trim_bout_to_valid_span` (Task 11).

Background: this is the PRE-STAC counterpart of
`trim_bouts_to_valid_span` in `scripts/postprocess_stac_data.py`. That
downstream trim fixed `interpolate_trajectory`'s NaN-poisoning problem, but
left STAC itself reading NaN keypoints. In the v2_3 run this had two
measured consequences: 7 of 23 directories had their offsets fit land on
NaN frames, writing an all-NaN `offsets` array (every leg joint frozen at
0 across 113 clips); 10 further bouts in otherwise-healthy directories had
scattered NaN keypoints and froze individually. 123 of 387 clips (32%)
were unusable. Measured root cause: 79 bouts had NaN keypoints, all
edge-only with zero interior gaps -- 508 leading + 569 trailing NaN frames
out of 140,809 total (0.76%).

`trim_bout_to_valid_span` trims a single bout dict -- the same shape
`process_bouts_batch` builds right before storing it into
`all_bouts_dict` (keys like `keypoints`, `orig_keypoints`, `kp_names`,
`skeleton_edges`, `edge_nan`, `alignment_info`, and, when pair_validity is
enabled, `valid_fly`/`filter_ok`/`ground_ok`/`floor_z`/`swap_state`) --
to its first/last valid (non-NaN, per `keypoints`) frame. This is a
DIFFERENT dict shape than the postprocess-side bout (which has
`qpos`/`kp_data`, not `keypoints`/`orig_keypoints`); do not confuse the two
sibling implementations.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.preprocess_keypoints_for_ik import trim_bout_to_valid_span

# Every array in a real bout_data whose leading dim equals the bout's frame
# count T (see preprocess_keypoints_for_ik.py ~1420-1512).
TEMPORAL_KEYS = ['keypoints', 'orig_keypoints', 'edge_nan',
                  'valid_fly', 'filter_ok', 'ground_ok', 'swap_state']


def _make_bout_data(T, N=6, E=5, seed=0):
    """A synthetic bout_data dict with every key the real pipeline carries
    at the trim call-site (~line 1512), including non-temporal arrays
    (`kp_names`: length N; `skeleton_edges`: length E) whose leading dims
    are deliberately different from T, plus the dict-valued `alignment_info`
    and the scalar `floor_z`."""
    rng = np.random.default_rng(seed)
    return {
        'keypoints': rng.normal(size=(T, N, 3)),
        'orig_keypoints': rng.normal(size=(T, N, 3)),
        'kp_names': [f'kp_{i}' for i in range(N)],
        'skeleton_edges': np.arange(E * 2).reshape(E, 2),
        'edge_nan': np.zeros((T, N), dtype=bool),
        'alignment_info': {
            'scales': 1.0,
            'rotation': np.eye(3),
            'translation': np.zeros(3),
        },
        'valid_fly': np.ones(T, dtype=bool),
        'filter_ok': np.ones(T, dtype=bool),
        'ground_ok': np.ones(T, dtype=bool),
        'floor_z': -1.23,
        'swap_state': np.zeros(T, dtype=bool),
    }


def _nan_frames(bout_data, key, frame_idxs):
    arr = bout_data[key].copy()
    for i in frame_idxs:
        arr[i, ...] = np.nan
    bout_data[key] = arr


def test_leading_only_nan_is_trimmed():
    T = 20
    bout_data = _make_bout_data(T)
    _nan_frames(bout_data, 'keypoints', range(0, 4))  # frames 0..3 bad

    trimmed, n_dropped = trim_bout_to_valid_span(bout_data, verbose=False)

    assert n_dropped == 4
    assert trimmed['keypoints'].shape[0] == T - 4
    assert not np.isnan(trimmed['keypoints']).any()


def test_trailing_only_nan_is_trimmed():
    T = 20
    bout_data = _make_bout_data(T)
    _nan_frames(bout_data, 'keypoints', range(T - 3, T))  # last 3 frames bad

    trimmed, n_dropped = trim_bout_to_valid_span(bout_data, verbose=False)

    assert n_dropped == 3
    assert trimmed['keypoints'].shape[0] == T - 3
    assert not np.isnan(trimmed['keypoints']).any()


def test_both_ends_nan_is_trimmed():
    T = 20
    bout_data = _make_bout_data(T)
    _nan_frames(bout_data, 'keypoints', [0, 1])
    _nan_frames(bout_data, 'keypoints', [T - 1])

    trimmed, n_dropped = trim_bout_to_valid_span(bout_data, verbose=False)

    assert n_dropped == 3
    assert trimmed['keypoints'].shape[0] == T - 3
    assert not np.isnan(trimmed['keypoints']).any()


def test_no_nan_is_a_true_noop():
    """A clean bout must be returned unchanged -- same dict object, same
    arrays (identity, not just equality): a true no-op does no slicing."""
    T = 20
    bout_data = _make_bout_data(T)

    trimmed, n_dropped = trim_bout_to_valid_span(bout_data, verbose=False)

    assert n_dropped == 0
    assert trimmed is bout_data
    assert trimmed['keypoints'] is bout_data['keypoints']
    assert trimmed['orig_keypoints'] is bout_data['orig_keypoints']


def test_interior_nan_is_reported_not_silently_trimmed(capsys):
    T = 20
    bout_data = _make_bout_data(T)
    _nan_frames(bout_data, 'keypoints', [10])  # a gap in the middle, clean edges

    trimmed, n_dropped = trim_bout_to_valid_span(bout_data, verbose=True)

    assert n_dropped == 0
    # Left completely untouched -- trimming cannot fix an interior gap.
    assert trimmed is bout_data
    assert trimmed['keypoints'].shape[0] == T
    assert np.isnan(trimmed['keypoints'][10]).all()
    out = capsys.readouterr().out
    assert 'WARNING' in out
    assert '1' in out  # count of interior NaN frames


def test_all_nan_bout_is_untouched_and_reported(capsys):
    T = 20
    bout_data = _make_bout_data(T)
    bout_data['keypoints'][:] = np.nan

    trimmed, n_dropped = trim_bout_to_valid_span(bout_data, verbose=True)

    assert n_dropped == 0
    # Not a zero-length bout, not modified at all.
    assert trimmed is bout_data
    assert trimmed['keypoints'].shape[0] == T
    assert np.isnan(trimmed['keypoints']).all()
    out = capsys.readouterr().out
    assert 'WARNING' in out


def test_every_temporal_array_sliced_consistently():
    T = 20
    bout_data = _make_bout_data(T)
    _nan_frames(bout_data, 'keypoints', [0, 1, 2])
    _nan_frames(bout_data, 'keypoints', [T - 1])

    trimmed, n_dropped = trim_bout_to_valid_span(bout_data, verbose=False)

    new_len = T - 4
    assert n_dropped == 4
    for key in TEMPORAL_KEYS:
        assert trimmed[key].shape[0] == new_len, f"{key} not sliced to consistent length"


def test_keypoints_and_orig_keypoints_sliced_to_the_same_span():
    """orig_keypoints has no NaN of its own but must be sliced to the exact
    same [first_valid, last_valid] span computed from `keypoints` (they are
    the same frames, pre/post-filter, and must stay index-aligned)."""
    T = 20
    bout_data = _make_bout_data(T)
    _nan_frames(bout_data, 'keypoints', [0, 1])
    _nan_frames(bout_data, 'keypoints', [T - 1, T - 2, T - 3])
    orig_before = bout_data['orig_keypoints'][2:T - 3].copy()

    trimmed, n_dropped = trim_bout_to_valid_span(bout_data, verbose=False)

    assert n_dropped == 5
    assert trimmed['orig_keypoints'].shape[0] == T - 5
    np.testing.assert_array_equal(trimmed['orig_keypoints'], orig_before)


def test_non_temporal_fields_left_unsliced():
    T = 20
    bout_data = _make_bout_data(T)
    orig_kp_names = list(bout_data['kp_names'])
    orig_edges_shape = bout_data['skeleton_edges'].shape
    orig_alignment_info = bout_data['alignment_info']
    orig_floor_z = bout_data['floor_z']
    _nan_frames(bout_data, 'keypoints', [0, 1, T - 1])

    trimmed, n_dropped = trim_bout_to_valid_span(bout_data, verbose=False)

    assert n_dropped == 3
    assert trimmed['kp_names'] == orig_kp_names
    assert trimmed['skeleton_edges'].shape == orig_edges_shape
    assert trimmed['alignment_info'] is orig_alignment_info
    assert trimmed['floor_z'] == orig_floor_z


def test_returned_dropped_frame_count_matches_shape_delta():
    T = 30
    bout_data = _make_bout_data(T)
    _nan_frames(bout_data, 'keypoints', [0, 1, 2, 3, 4])
    _nan_frames(bout_data, 'keypoints', [T - 1, T - 2])

    trimmed, n_dropped = trim_bout_to_valid_span(bout_data, verbose=False)

    assert n_dropped == 7
    assert trimmed['keypoints'].shape[0] == T - n_dropped


def test_warning_names_the_bout_when_bout_name_given(capsys):
    T = 20
    bout_data = _make_bout_data(T)
    bout_data['keypoints'][:] = np.nan

    trim_bout_to_valid_span(bout_data, verbose=True, bout_name='bout_007')

    out = capsys.readouterr().out
    assert 'bout_007' in out


def test_quiet_by_default_no_warning_printed(capsys):
    """verbose=False (the default) must not print anything, even when the
    bout is entirely NaN."""
    T = 20
    bout_data = _make_bout_data(T)
    bout_data['keypoints'][:] = np.nan

    trim_bout_to_valid_span(bout_data)

    out = capsys.readouterr().out
    assert out == ''
