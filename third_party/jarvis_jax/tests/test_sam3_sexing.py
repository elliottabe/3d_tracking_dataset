import numpy as np
from types import SimpleNamespace
from jarvis_jax.predict.sam3_driver import sex_male_by_size


def _mask(area_px):
    """(100,100) bool mask with exactly `area_px` True pixels (or None)."""
    if area_px is None:
        return None
    m = np.zeros((100, 100), bool)
    flat = m.reshape(-1)
    flat[:int(area_px)] = True
    return m


def _bm(area_by_cam, num_frames=10):
    """area_by_cam[cam] = (area0, area1); fly0 uses oid 10, fly1 uses oid 20.
    A None area omits that fly's mask in every frame of that camera."""
    masks, idmap = [], []
    for (a0, a1) in area_by_cam:
        frames = []
        for _ in range(num_frames):
            fr = {}
            m0 = _mask(a0); m1 = _mask(a1)
            if m0 is not None:
                fr[10] = {"mask": m0}
            if m1 is not None:
                fr[20] = {"mask": m1}
            frames.append(fr)
        masks.append(frames)
        idmap.append({10: 0, 20: 1})
    return SimpleNamespace(num_cameras=len(area_by_cam), num_frames=num_frames,
                           masks=masks, identity_map=idmap)


def test_male_is_larger_majority_fly0():
    bm = _bm([(90, 50)] * 5 + [(50, 90)] * 2)      # fly0 larger in 5 of 7 cams
    male, info = sex_male_by_size(bm, 2)
    assert male == 0
    assert info["method"] == "mask_area_vote"
    assert info["n_cameras"] == 7
    assert info["male_detected_slot"] == 0
    assert 0.0 < info["agreement"] <= 1.0
    assert info["margin"] > 0.0


def test_male_is_larger_fly1():
    bm = _bm([(50, 90)] * 6 + [(90, 50)] * 1)
    male, info = sex_male_by_size(bm, 2)
    assert male == 1 and info["male_detected_slot"] == 1


def test_insufficient_cameras_returns_none():
    # only cam 0 has BOTH flies; cams 1..6 omit fly0 -> no pairs there
    bm = _bm([(90, 50)] + [(None, 50)] * 6)
    male, info = sex_male_by_size(bm, 2)
    assert male is None
    assert info["n_cameras"] <= 1


def test_min_pairs_gate_returns_none():
    # both flies present but fewer than min_pairs frames -> every camera skipped
    bm = _bm([(90, 50)] * 7, num_frames=3)
    male, _ = sex_male_by_size(bm, 2)
    assert male is None


def test_num_animals_not_two():
    bm = _bm([(90, 50)] * 7)
    assert sex_male_by_size(bm, 3) == (None, {})
