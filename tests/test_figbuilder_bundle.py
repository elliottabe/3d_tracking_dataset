# tests/test_figbuilder_bundle.py
"""Round-trip tests for the figbuilder bundle format."""
from __future__ import annotations

import numpy as np

from figbuilder.bundle import (
    PanelData, array_to_segments, read_bundle, segments_to_array, write_bundle,
)


def test_bundle_roundtrip_preserves_arrays_and_meta(tmp_path):
    p = tmp_path / "b.h5"
    panels = {
        "wing": PanelData(
            type="courtship.wing_z",
            data={"t_ms": np.linspace(0, 100, 51),
                  "wingL_z": np.random.default_rng(0).normal(size=51)},
            assets={},
            attrs={"fs": 800.0},
        ),
    }
    write_bundle(p, meta={"fig_width_mm": 183.0, "fig_height_mm": 140.0}, panels=panels)
    b = read_bundle(p)

    assert b.meta["fig_width_mm"] == 183.0
    assert set(b.panels) == {"wing"}
    assert b.panels["wing"].type == "courtship.wing_z"
    assert b.panels["wing"].attrs["fs"] == 800.0
    np.testing.assert_allclose(b.panels["wing"].data["t_ms"], panels["wing"].data["t_ms"])


def test_bundle_roundtrip_preserves_uint8_image_assets(tmp_path):
    p = tmp_path / "b.h5"
    img = np.random.default_rng(1).integers(0, 255, (12, 10, 3), dtype=np.uint8)
    panels = {"vid": PanelData(type="image", data={}, assets={"frame": img}, attrs={})}
    write_bundle(p, meta={}, panels=panels)
    b = read_bundle(p)
    got = b.panels["vid"].assets["frame"]
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, img)


def test_segment_table_roundtrip():
    segs = [{"start": 0, "end": 12, "type": "pulse"},
            {"start": 40, "end": 91, "type": "sine"}]
    arr = segments_to_array(segs)
    assert arr.dtype.names == ("start", "end", "type")
    assert array_to_segments(arr) == segs


def test_empty_segment_table_roundtrips_to_empty_list():
    arr = segments_to_array([])
    assert arr.shape == (0,)
    assert array_to_segments(arr) == []
