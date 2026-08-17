"""Bout-key numbering and ordering must survive >999 bouts.

concatenate_bout_dicts named bouts with a hardcoded :03d, so a 2203-bout
combine produced mixed-width keys (bout_999, bout_1000) whose LEXICOGRAPHIC
sort disagrees with the numeric write order of the info arrays -- measured on
the NewBouts combined h5: 2099/2203 bouts misaligned for every consumer that
does sorted(keys) (combine's unpad step, pack_reference_clips, the raw
exporter).
"""
from __future__ import annotations

import numpy as np

from utils.stac_data_utils import (
    bout_key_pad, sorted_bout_keys, concatenate_bout_dicts)
from utils import io_dict_to_hdf5 as ioh5


def test_bout_key_pad_scales_with_count():
    assert bout_key_pad(5) == 3          # never narrower than the legacy 3
    assert bout_key_pad(1000) == 3       # max index 999 still fits 3 digits
    assert bout_key_pad(1001) == 4       # bout_0999, bout_1000 stay aligned
    assert bout_key_pad(2203) == 4
    assert bout_key_pad(10001) == 5


def test_sorted_bout_keys_is_numeric():
    keys = ["bout_999", "bout_1000", "bout_100", "bout_0", "bout_1001"]
    assert sorted_bout_keys(keys) == [
        "bout_0", "bout_100", "bout_999", "bout_1000", "bout_1001"]


def _tiny_source(path, n_bouts, start0):
    d = {f"bout_{i:03d}": {"qpos": np.full((2 + (i % 3), 4), float(i))}
         for i in range(n_bouts)}
    d["info"] = {
        "clip_lengths": np.array([2 + (i % 3) for i in range(n_bouts)]),
        "start_frames": np.array([start0 + 10 * i for i in range(n_bouts)]),
        "fly_ids": [f"s/x{start0}"] * n_bouts,
    }
    ioh5.save(str(path), d)
    return path


def test_concatenate_pads_uniformly_and_stays_aligned(tmp_path):
    # 600 + 500 = 1100 bouts -> crosses the 999 boundary that broke :03d.
    f1 = _tiny_source(tmp_path / "a.h5", 600, start0=0)
    f2 = _tiny_source(tmp_path / "b.h5", 500, start0=100000)
    out = concatenate_bout_dicts([str(f1), str(f2)], enable_jax=False,
                                 verbose=False)
    keys = [k for k in out if k != "info"]
    assert len(keys) == 1100
    # uniform 4-digit width -> plain sorted() == numeric order
    assert all(len(k) == len("bout_0000") for k in keys)
    assert sorted(keys) == sorted_bout_keys(keys)
    # data stays aligned with the concatenated info arrays under sorted order
    cl = np.asarray(out["info"]["clip_lengths"])
    for i, k in enumerate(sorted(keys)):
        assert out[k]["qpos"].shape[0] == cl[i], (i, k)
