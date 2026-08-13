import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.viz.ik_explainer import clip_io


def test_the_two_orders_are_different_but_same_set():
    det, mod = clip_io.detector_kp_names(), clip_io.model_kp_names()
    assert len(det) == len(mod) == 50
    assert set(det) == set(mod)
    assert det != mod, "orders must differ — see configs comments"
    assert det[:4] == ["Antenna_Base", "EyeL", "EyeR", "Scutellum"]
    assert mod[:3] == ["Scutellum", "WingL_base", "WingR_base"]


def test_index_maps_detector_array_to_model_order():
    det, mod = clip_io.detector_kp_names(), clip_io.model_kp_names()
    idx = clip_io.detector_to_model_index()
    assert idx.shape == (50,)
    assert [det[i] for i in idx] == mod


def test_reordering_is_a_permutation_not_a_reshape():
    idx = clip_io.detector_to_model_index()
    assert sorted(idx.tolist()) == list(range(50))


def test_shipped_csv_is_in_detector_order():
    xyz, conf, names = clip_io.load_shipped_kp3d_mm(
        clip_io.shipped_csv_path(clip_io.CLIP_DEFAULT))
    assert names == clip_io.detector_kp_names()
