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
    img4, hm, vis = ds[0]
    assert img4.shape == (448, 448, 4) and img4.dtype == np.float32
    assert hm.shape == (224, 224, 50) and hm.dtype == np.float32
    assert vis.shape == (50,) and vis.dtype == np.bool_
    # mask channel is binary
    m = img4[:, :, 3]
    assert set(np.unique(m)).issubset({0.0, 1.0})


@skip
def test_heatmap_peaks_align_with_visible_keypoints():
    ds = V3Dataset(ROOT, "val")
    # find a sample with at least one visible keypoint
    for i in range(min(len(ds), 50)):
        img4, hm, vis = ds[i]
        if vis.any():
            j = int(np.argmax(vis))
            assert hm[:, :, j].max() > 0.9        # a real peak exists
            assert hm[:, :, ~vis].max(initial=0.0) == 0.0  # invisible channels empty
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
    img4, hm, vis = next(it)
    assert img4.shape == (2, 448, 448, 4)
    assert hm.shape == (2, 224, 224, 50)
    assert vis.shape == (2, 50)
    # same seed -> same first batch order
    img4b, _, _ = next(batches(ds, batch_size=2, shuffle=True, seed=0))
    assert np.array_equal(img4, img4b)
