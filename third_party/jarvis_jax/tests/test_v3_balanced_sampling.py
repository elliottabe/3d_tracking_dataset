"""Tests for V3Dataset.balanced_weights / class_counts.

Class-balanced sampling replaces the single-class `sampling_weights` boost once
a dataset spans many under-represented conditions at very different sizes (V4
train: wall=33 anns .. grooming=4592). These tests are synthetic -- they build a
bare object with the label lists attached rather than touching data on disk --
plus one guard that runs against the real V4 root when it is present.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from jarvis_jax.data.v3 import V3Dataset


def make_ds(categories, sexes=None, behaviors=None):
    """A V3Dataset with only the fields balanced_weights/class_counts read."""
    ds = object.__new__(V3Dataset)
    ds.category = list(categories)
    ds.sex = list(sexes) if sexes is not None else ["unknown"] * len(ds.category)
    ds.behavior = (list(behaviors) if behaviors is not None
                   else ["unknown"] * len(ds.category))
    ds.file_names = ["f"] * len(ds.category)
    return ds


# ---------------------------------------------------------------------------
# alpha endpoints
# ---------------------------------------------------------------------------

def test_alpha_zero_is_uniform():
    ds = make_ds(["big"] * 100 + ["small"] * 4)
    w = ds.balanced_weights(alpha=0.0)
    assert np.allclose(w, 1.0 / len(ds.category))


def test_alpha_one_gives_every_class_equal_total_mass():
    ds = make_ds(["big"] * 100 + ["mid"] * 20 + ["small"] * 4)
    w = ds.balanced_weights(alpha=1.0)
    share = {}
    for wi, c in zip(w, ds.category):
        share[c] = share.get(c, 0.0) + wi
    assert np.allclose(list(share.values()), 1.0 / 3.0)


def test_alpha_half_is_between_uniform_and_parity():
    ds = make_ds(["big"] * 100 + ["small"] * 4)
    uniform_small_share = 4 / len(ds.category)       # 0.038; alpha=1 would give 0.5
    w = ds.balanced_weights(alpha=0.5)
    small = sum(wi for wi, c in zip(w, ds.category) if c == "small")
    assert uniform_small_share < small < 0.5


def test_weights_always_sum_to_one():
    ds = make_ds(["a"] * 7 + ["b"] * 3 + ["c"] * 91)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        for cap in (None, 1.0, 5.0, 50.0):
            w = ds.balanced_weights(alpha=alpha, max_repeat=cap)
            assert np.isclose(w.sum(), 1.0), (alpha, cap)
            assert (w > 0).all()


def test_rarer_class_never_gets_a_lower_per_sample_weight():
    ds = make_ds(["big"] * 60 + ["small"] * 5)
    w = ds.balanced_weights(alpha=0.5)
    big = w[np.array(ds.category) == "big"]
    small = w[np.array(ds.category) == "small"]
    assert small.min() >= big.max()


# ---------------------------------------------------------------------------
# max_repeat cap
# ---------------------------------------------------------------------------

def test_max_repeat_caps_the_rarest_class():
    # 1 annotation vs 999: full parity would repeat it ~500x.
    ds = make_ds(["big"] * 999 + ["tiny"])
    n = len(ds.category)
    w = ds.balanced_weights(alpha=1.0, max_repeat=10.0)
    assert w.max() <= 10.0 / n + 1e-12


def test_max_repeat_is_enforced_after_renormalisation():
    # Clipping then renormalising can lift entries back over the ceiling; the
    # implementation iterates, so NO entry may exceed the cap in the result.
    ds = make_ds(["a"] * 500 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2)
    n = len(ds.category)
    cap = 3.0
    w = ds.balanced_weights(alpha=1.0, max_repeat=cap)
    assert w.max() <= cap / n + 1e-12
    assert np.isclose(w.sum(), 1.0)


def test_cap_of_one_forces_uniform():
    # max_repeat=1 means no sample may be drawn more often than uniform, and
    # the weights must still sum to 1 -- the only solution is uniform.
    ds = make_ds(["big"] * 50 + ["small"] * 5)
    w = ds.balanced_weights(alpha=1.0, max_repeat=1.0)
    assert np.allclose(w, 1.0 / len(ds.category))


def test_cap_below_one_is_unsatisfiable_and_raises():
    ds = make_ds(["a"] * 10 + ["b"] * 2)
    with pytest.raises(ValueError, match="max_repeat"):
        ds.balanced_weights(max_repeat=0.5)


def test_generous_cap_is_inert():
    ds = make_ds(["a"] * 100 + ["b"] * 25)
    uncapped = ds.balanced_weights(alpha=0.5, max_repeat=None)
    capped = ds.balanced_weights(alpha=0.5, max_repeat=1000.0)
    assert np.allclose(uncapped, capped)


# ---------------------------------------------------------------------------
# grouping key / validation
# ---------------------------------------------------------------------------

def test_can_group_by_sex_instead_of_category():
    ds = make_ds(["x"] * 12, sexes=["male"] * 10 + ["female"] * 2)
    w = ds.balanced_weights(key="sex", alpha=1.0)
    fem = sum(wi for wi, s in zip(w, ds.sex) if s == "female")
    assert np.isclose(fem, 0.5)


def test_unknown_key_raises():
    ds = make_ds(["a", "b"])
    with pytest.raises(ValueError, match="unknown grouping key"):
        ds.balanced_weights(key="not_a_field")


def test_alpha_out_of_range_raises():
    ds = make_ds(["a", "b"])
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            ds.balanced_weights(alpha=bad)


def test_empty_dataset_raises():
    ds = make_ds([])
    with pytest.raises(ValueError, match="empty"):
        ds.balanced_weights()


def test_single_class_is_uniform():
    ds = make_ds(["only"] * 9)
    assert np.allclose(ds.balanced_weights(alpha=1.0), 1.0 / 9)


def test_class_counts():
    ds = make_ds(["a", "a", "b"])
    assert ds.class_counts() == {"a": 2, "b": 1}


# ---------------------------------------------------------------------------
# real V4 root
# ---------------------------------------------------------------------------

V4 = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V4"


@pytest.mark.skipif(not os.path.isdir(V4), reason="V4 data root not present")
def test_v4_carries_category_tags():
    # The first V4 build dropped sex/behavior/category, which silently degraded
    # every weighted-sampling path to uniform. Guard against a rebuild that
    # regresses it.
    ds = V3Dataset(V4, "train")
    counts = ds.class_counts("category")
    assert set(counts) != {"unknown"}, "V4 annotations carry no category tag"
    assert "wall" in counts and "headless" in counts and "amputated" in counts
    assert set(ds.class_counts("sex")) >= {"male", "female"}


@pytest.mark.skipif(not os.path.isdir(V4), reason="V4 data root not present")
def test_v4_balanced_sampling_lifts_the_rare_classes():
    ds = V3Dataset(V4, "train")
    n = len(ds)
    counts = ds.class_counts("category")
    w = ds.balanced_weights("category", alpha=0.5, max_repeat=20.0)
    share = {}
    for wi, c in zip(w, ds.category):
        share[c] = share.get(c, 0.0) + wi
    rare = min(counts, key=lambda c: counts[c])
    big = max(counts, key=lambda c: counts[c])
    # Rare class share must rise above its raw data share; big class must fall.
    assert share[rare] > counts[rare] / n
    assert share[big] < counts[big] / n
    # And the cap must hold.
    assert w.max() <= 20.0 / n + 1e-12
