# tests/test_v3_dataset.py
import os
import numpy as np
import pytest
from jarvis_jax.data.v3 import V3Dataset, batches

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
have_data = os.path.isdir(ROOT)
skip = pytest.mark.skipif(not have_data, reason="V3 data root not present")


@skip
def test_getitem_shapes_and_dtypes():
    ds = V3Dataset(ROOT, "val")
    assert len(ds) > 0
    img4, kp, vis = ds[0]
    assert img4.shape == (448, 448, 4) and img4.dtype == np.uint8
    assert kp.shape == (50, 2) and kp.dtype == np.float32
    assert vis.shape == (50,) and vis.dtype == np.bool_
    assert set(np.unique(img4[:, :, 3])).issubset({0, 1})  # mask binary uint8


@skip
def test_visible_keypoints_in_bounds():
    ds = V3Dataset(ROOT, "val")
    for i in range(min(len(ds), 50)):
        img4, kp, vis = ds[i]
        if vis.any():
            v = kp[vis]
            assert (v >= 0).all() and (v <= 223).all()  # heatmap-coord bounds
            return
    pytest.skip("no visible keypoints in first 50 samples")


@skip
def test_recordings_filter_keeps_only_requested():
    ds = V3Dataset(ROOT, "val", recordings=["2026_05_27_11_56_05"])
    assert len(ds) > 0
    assert all(r.startswith("2026_05_27_11_56_05/") for r in ds.file_names)


@skip
def test_batches_stacks_and_is_deterministic():
    ds = V3Dataset(ROOT, "val")
    it = batches(ds, batch_size=2, shuffle=True, seed=0)
    img4, kp, vis = next(it)
    assert img4.shape == (2, 448, 448, 4) and img4.dtype == np.uint8
    assert kp.shape == (2, 50, 2)
    assert vis.shape == (2, 50)
    img4b, _, _ = next(batches(ds, batch_size=2, shuffle=True, seed=0))
    assert np.array_equal(img4, img4b)


# --- weighted sampling of the female-courtship class -----------------------------

class _StubDS:
    """Minimal stand-in exercising V3Dataset.sampling_weights without any I/O."""
    def __init__(self, sex, behavior):
        self.sex = sex; self.behavior = behavior
    def __len__(self):
        return len(self.sex)
    sampling_weights = V3Dataset.sampling_weights


def test_sampling_weights_boosts_target_class():
    s = _StubDS(sex=["female", "male", "female", "unknown"],
               behavior=["courtship", "courtship", "general", "grooming"])
    w = s.sampling_weights("female", "courtship", 11)          # only ann0 matches
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w[0], 11 / 14) and np.allclose(w[1:], 1 / 14)   # 11 + 1 + 1 + 1 = 14


def test_sampling_weights_none_axis_matches_any():
    s = _StubDS(sex=["female", "male", "female"],
               behavior=["courtship", "courtship", "general"])
    w = s.sampling_weights("female", None, 5)                  # any behavior, sex=female
    # anns 0 and 2 are female -> weight 5 each; ann1 male -> 1. total 11
    assert np.allclose(w, [5 / 11, 1 / 11, 5 / 11])


@skip
def test_train_ds_loads_sex_behavior_tags():
    ds = V3Dataset(ROOT, "train")
    assert len(ds.sex) == len(ds) and len(ds.behavior) == len(ds)
    assert "female" in set(ds.sex) and "courtship" in set(ds.behavior)


@skip
def test_weighted_batches_oversample_female_courtship():
    ds = V3Dataset(ROOT, "train")
    tgt = np.array([s == "female" and b == "courtship"
                    for s, b in zip(ds.sex, ds.behavior)])
    base = tgt.mean()                                     # ~0.022 uniform
    w = ds.sampling_weights("female", "courtship", 11)
    idx = np.random.default_rng(0).choice(len(ds), size=len(ds), replace=True, p=w)
    boosted = tgt[idx].mean()
    assert boosted > 5 * base                             # ~0.2 vs ~0.022
