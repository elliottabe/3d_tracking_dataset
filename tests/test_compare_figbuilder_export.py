"""The M0 comparison harness."""
from __future__ import annotations

import shutil

import numpy as np
import pytest
from PIL import Image

from scripts.viz.compare_figbuilder_export import (
    render_svg_to_png, side_by_side,
)

MINIMAL_SVG = (
    b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
    b'width="72pt" height="36pt" viewBox="0 0 72 36">'
    b'<rect x="4" y="4" width="20" height="10" fill="#38bdf8"/></svg>'
)


@pytest.mark.skipif(shutil.which("rsvg-convert") is None,
                    reason="rsvg-convert not on PATH")
def test_render_svg_to_png_produces_an_opaque_image(tmp_path):
    s = tmp_path / "a.svg"
    s.write_bytes(MINIMAL_SVG)
    p = render_svg_to_png(s, tmp_path / "a.png", dpi=100)
    im = Image.open(p)
    assert im.mode == "RGB"
    assert im.size[0] > 0


def test_side_by_side_width_is_the_sum_plus_the_gutter(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.fromarray(np.zeros((40, 30, 3), np.uint8)).save(a)
    Image.fromarray(np.zeros((50, 20, 3), np.uint8)).save(b)
    out = side_by_side(a, b, tmp_path / "c.png", labels=("old", "new"), gutter=10)
    im = Image.open(out)
    assert im.size[0] == 30 + 10 + 20
    assert im.size[1] >= 50
