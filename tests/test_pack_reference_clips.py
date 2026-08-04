"""Padding/stacking invariants for the reference-clip packer.

Two conventions matter and are easy to get wrong:
  * padding repeats the FINAL frame (np.tile of arr[-1:]), not zeros
  * clip_lengths holds the TRUE unpadded length. The v1 reference file stores
    the padded length for every clip, which fly_mimic works around by sniffing
    repeated trailing frames; we do not reproduce that bug.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.export.pack_reference_clips import pack_clips

NAMES = [f'j{i}' for i in range(5)]


def _bout(T, seed):
    rng = np.random.default_rng(seed)
    return {
        'qpos': rng.normal(size=(T, 5)).astype(np.float32),
        'qvel': rng.normal(size=(T, 4)).astype(np.float32),
        'xpos': rng.normal(size=(T, 3, 3)).astype(np.float32),
        'xquat': rng.normal(size=(T, 3, 4)).astype(np.float32),
        'kp_data': rng.normal(size=(T, 6)).astype(np.float32),
    }


def test_shapes_are_stacked_to_max_length():
    out = pack_clips([_bout(10, 0), _bout(25, 1), _bout(17, 2)], NAMES)
    assert out['qpos'].shape == (3, 25, 5)
    assert out['qvel'].shape == (3, 25, 4)
    assert out['xpos'].shape == (3, 25, 3, 3)
    assert out['xquat'].shape == (3, 25, 3, 4)
    assert out['kp_data'].shape == (3, 25, 6)


def test_clip_lengths_are_true_unpadded_lengths():
    out = pack_clips([_bout(10, 0), _bout(25, 1), _bout(17, 2)], NAMES)
    np.testing.assert_array_equal(out['clip_lengths'], [10, 25, 17])
    assert out['clip_lengths'].dtype == np.int32


def test_padding_repeats_the_final_frame():
    bouts = [_bout(10, 0), _bout(25, 1)]
    out = pack_clips(bouts, NAMES)
    last = bouts[0]['qpos'][-1]
    for t in range(10, 25):
        np.testing.assert_allclose(out['qpos'][0, t], last)


def test_real_frames_are_untouched():
    bouts = [_bout(10, 0), _bout(25, 1)]
    out = pack_clips(bouts, NAMES)
    np.testing.assert_allclose(out['qpos'][0, :10], bouts[0]['qpos'])
    np.testing.assert_allclose(out['xquat'][1], bouts[1]['xquat'])


def test_longest_clip_is_not_padded():
    bouts = [_bout(10, 0), _bout(25, 1)]
    out = pack_clips(bouts, NAMES)
    np.testing.assert_allclose(out['qpos'][1], bouts[1]['qpos'])


def test_qpos_names_passed_through():
    out = pack_clips([_bout(4, 0)], NAMES)
    assert out['qpos_names'] == NAMES


def test_rejects_names_length_mismatch():
    """qpos_names must be per-COLUMN; fly_mimic keys its root-offset logic on it."""
    with pytest.raises(ValueError, match='qpos_names'):
        pack_clips([_bout(4, 0)], NAMES[:3])


def test_single_clip_needs_no_padding():
    out = pack_clips([_bout(7, 0)], NAMES)
    assert out['qpos'].shape == (1, 7, 5)
    np.testing.assert_array_equal(out['clip_lengths'], [7])
