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


def test_all_helpers_return_copies_and_never_mutate_input():
    """Copy semantics are load-bearing: acts composite one base frame at
    several alphas, so an in-place helper would smear across a fade."""
    names = ["Antenna_Base", "EyeL", "Scutellum", "T1L_FeTi"] * 12 + ["Abd_tip"] * 2
    uv = np.full((50, 2), 30.0)
    base = np.full((60, 60, 3), 40, np.uint8)
    calls = [
        lambda im: draw.draw_keypoints(im, uv, names, alpha=1.0),
        lambda im: draw.draw_leg_chains(im, uv, names, alpha=1.0),
        lambda im: draw.label(im, "T1L_FeTi", (5, 20)),
        lambda im: draw.stage_title(im, "root_optimization", "residual 0.42 mm"),
        lambda im: draw.scale_bar_mm(im, 80.7, mm=1.0),
    ]
    for fn in calls:
        img = base.copy()
        before = img.copy()
        out = fn(img)
        assert np.array_equal(img, before), f"{fn} mutated its input"
        assert out is not img


def test_leg_chains_skip_non_finite_without_crashing():
    img = np.zeros((60, 60, 3), np.uint8)
    names = ["T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1",
             "T1L_TaT3", "T1L_TaTip"] + ["EyeL"] * 43
    uv = np.full((50, 2), np.nan)
    out = draw.draw_leg_chains(img, uv, names, alpha=1.0)
    assert np.array_equal(out, img)


def test_alpha_one_reproduces_the_drawn_layer_exactly():
    """Guards endpoint rounding: a faint ghost at alpha=1.0 would show in every act."""
    names = ["EyeL"] * 50
    uv = np.full((50, 2), 30.0)
    img = np.full((60, 60, 3), 40, np.uint8)
    a = draw.draw_keypoints(img, uv, names, alpha=1.0)
    assert not np.array_equal(a, img), "alpha=1.0 drew nothing — test is vacuous"
    b = draw.draw_keypoints(img, uv, names, alpha=1.0)
    assert np.array_equal(a, b)
