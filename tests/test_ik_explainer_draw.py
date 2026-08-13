import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.viz.ik_explainer import draw


def test_fade_endpoints_and_midpoint():
    a = np.zeros((10, 10, 3), np.uint8)
    b = np.full((10, 10, 3), 200, np.uint8)
    assert np.array_equal(draw.fade(a, b, 0.0), a)
    assert np.array_equal(draw.fade(a, b, 1.0), b)
    assert abs(int(draw.fade(a, b, 0.5)[0, 0, 0]) - 100) <= 1


def test_alpha_zero_leaves_image_untouched():
    img = np.full((60, 60, 3), 40, np.uint8)
    uv = np.full((50, 2), 30.0)
    names = ["Antenna_Base"] * 50
    assert np.array_equal(draw.draw_keypoints(img, uv, names, alpha=0.0), img)


def test_drawing_does_not_mutate_the_input():
    img = np.full((60, 60, 3), 40, np.uint8)
    before = img.copy()
    uv = np.full((50, 2), 30.0)
    draw.draw_keypoints(img, uv, ["EyeL"] * 50, alpha=1.0)
    assert np.array_equal(img, before), "helper must return a copy"


def test_nonfinite_keypoints_are_skipped_not_crashed():
    img = np.zeros((60, 60, 3), np.uint8)
    uv = np.full((50, 2), np.nan)
    out = draw.draw_keypoints(img, uv, ["EyeL"] * 50, alpha=1.0)
    assert np.array_equal(out, img)
