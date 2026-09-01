"""The SDF's ORIGINAL-pixel scaling must be exact, not averaged.

The recovered silhouette_sdf averaged the two axis scale factors into one, which
cannot be right for a non-square crop: measured 27% error on a 40x60 box. These
tests pin the fix.
"""
import numpy as np
import pytest

from jarvis_jax.tracking.mask_sdf import (mask_bbox, mask_to_sdf_crop,
                                          sdf_stack_from_masks, BIG)


def _box(h=40, w=60, H=200, W=300):
    m = np.zeros((H, W), bool)
    y0, x0 = (H - h) // 2, (W - w) // 2
    m[y0:y0 + h, x0:x0 + w] = True
    return m, (y0 + h / 2, x0 + w / 2)


def test_sign_convention_negative_inside_positive_outside():
    m, (cy, cx) = _box()
    bb = mask_bbox(m, 0.4)
    sdf, gs, go = mask_to_sdf_crop(m, bb, (64, 64))
    at = lambda y, x: float(sdf[int(round((y - go[1]) * gs[1])),
                               int(round((x - go[0]) * gs[0]))])
    assert at(cy, cx) < 0, "inside must be negative"
    assert float(sdf[0, 0]) > 0, "the crop corner is outside the box"


def test_distance_is_exact_original_pixels_on_an_ANISOTROPIC_crop():
    """THE REGRESSION. A 40-tall x 60-wide box has inradius 20 px. The old code
    averaged the axis scales and read 25.3 px here -- a 27% error."""
    m, (cy, cx) = _box(h=40, w=60)
    bb = mask_bbox(m, 0.4)
    sdf, gs, go = mask_to_sdf_crop(m, bb, (64, 64))
    assert abs(gs[0] / gs[1] - 1.0) > 0.2, "this crop must actually be anisotropic"
    centre = float(sdf[int(round((cy - go[1]) * gs[1])),
                       int(round((cx - go[0]) * gs[0]))])
    assert abs(abs(centre) - 20.0) < 1.2, f"inradius should read ~20.0 px, got {centre}"


def test_a_square_crop_agrees_with_the_anisotropic_path():
    """Sanity: where the old averaging was harmless (square crop), the fixed code
    gives the same answer, so the fix is not a behaviour change in that case."""
    m, (cy, cx) = _box(h=50, w=50)
    bb = mask_bbox(m, 0.4)
    sdf, gs, go = mask_to_sdf_crop(m, bb, (64, 64))
    centre = float(sdf[int(round((cy - go[1]) * gs[1])),
                       int(round((cx - go[0]) * gs[0]))])
    assert abs(abs(centre) - 25.0) < 1.5, f"inradius should read ~25.0 px, got {centre}"


def test_distance_grows_linearly_away_from_the_edge_in_original_px():
    """Walk out along +x from the box edge; each original pixel of travel must add
    ~1 to the SDF, in BOTH axes. An averaged scale fails one of them."""
    m, (cy, cx) = _box(h=40, w=60)
    bb = mask_bbox(m, 0.4)
    sdf, gs, go = mask_to_sdf_crop(m, bb, (64, 64))
    x_edge = cx + 30.0
    y_edge = cy + 20.0
    samp = lambda y, x: float(sdf[np.clip(int(round((y - go[1]) * gs[1])), 0, 63),
                                  np.clip(int(round((x - go[0]) * gs[0])), 0, 63)])
    for axis, (a0, step) in (("x", (x_edge, (0.0, 1.0))), ("y", (y_edge, (1.0, 0.0)))):
        d5 = samp(cy + 5 * step[0] if axis == "y" else cy,
                  cx if axis == "y" else a0 + 5)
        d15 = samp(cy + 15 * step[0] if axis == "y" else cy,
                   cx if axis == "y" else a0 + 15)
        if axis == "y":
            d5 = samp(a0 + 5, cx); d15 = samp(a0 + 15, cx)
        slope = (d15 - d5) / 10.0
        assert 0.75 < slope < 1.25, f"{axis}: {slope:.2f} original px per px"


def test_empty_and_invalid_frames_are_marked_absent_not_filled_with_garbage():
    T, C, H, W = 3, 2, 40, 60
    masks = np.zeros((T, C, H, W), bool)
    masks[0, 0, 10:30, 20:40] = True
    valid = np.ones((T, C), bool)
    valid[1, :] = False                       # invalid views
    sdf, gs, go, present = sdf_stack_from_masks(masks, valid, out_hw=(32, 32))
    assert present[0, 0] and not present[0, 1], "empty mask must be absent"
    assert not present[1].any(), "invalid views must be absent"
    assert np.allclose(sdf[~present], BIG), "absent entries must be BIG, not 0"
    assert sdf[0, 0].min() < 0, "the real mask must produce an interior"


def test_bbox_margin_grows_the_box():
    m, _ = _box(h=40, w=60)
    a = mask_bbox(m, 0.0)
    b = mask_bbox(m, 0.5)
    assert (b[2] - b[0]) > (a[2] - a[0]) and (b[3] - b[1]) > (a[3] - a[1])
    assert mask_bbox(np.zeros((10, 10), bool)) is None
