"""Root-figure rect extraction from nested subfigures."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.figures.seed_fig4_layout import root_rect, seed_layout


def test_root_rect_differs_from_get_position_for_nested_subfigures():
    fig = plt.figure(figsize=(6, 4))
    top, _ = fig.subfigures(2, 1)
    left, _ = top.subfigures(1, 2)
    ax = left.subplots(1, 1)
    try:
        got = root_rect(fig, ax)
        assert got != pytest.approx(ax.get_position().bounds)
        # left half, top half
        assert got[0] < 0.5 and got[1] > 0.5
    finally:
        plt.close(fig)


def test_root_rect_matches_get_position_for_a_plain_axes():
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0.2, 0.3, 0.5, 0.4])
    try:
        assert root_rect(fig, ax) == pytest.approx((0.2, 0.3, 0.5, 0.4), abs=1e-6)
    finally:
        plt.close(fig)


def test_root_rects_are_all_inside_the_unit_square():
    spec = seed_layout()
    for p in spec.panels:
        x, y, w, h = p.rect
        assert 0.0 <= x < 1.0 and 0.0 <= y < 1.0
        assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def test_seed_layout_emits_one_panel_per_fig4_axes():
    spec = seed_layout(n_video_frames=4, n_render_strip=4)
    ids = {p.id for p in spec.panels}
    assert {"wing", "scut", "sine_phase", "wing_phase_polar", "angle_2d",
            "pulse_class", "zheight", "pitch", "align_violin"} <= ids
    assert {"video_0", "video_3", "render_0", "render_3"} <= ids


def test_seed_layout_creates_groups_for_the_two_strips():
    spec = seed_layout()
    gids = {g.id for g in spec.groups}
    assert {"video_strip", "render_strip"} <= gids
    assert all(p.group == "video_strip"
               for p in spec.panels if p.id.startswith("video_"))


def test_seed_layout_creates_a_panel_letter_annotation_per_lettered_panel():
    spec = seed_layout()
    letters = {a["text"] for a in spec.annotations
               if a.get("managed") == "panel_letter"}
    assert {"A", "B", "C", "D"} <= letters


def test_seed_layout_seeds_render_params_into_panel_spec():
    """Bundle attrs are metadata only, so fs must land in figure.json spec."""
    spec = seed_layout()
    by_id = {p.id: p for p in spec.panels}
    assert by_id["wing"].spec["fs"] == pytest.approx(800.0)
    assert by_id["wing"].spec["time_unit"] == "s"
    assert by_id["wing_phase_polar"].spec["center_stat"] == "median"
    assert by_id["zheight"].spec["kind"] == "violin"


def test_seed_layout_uses_the_requested_canvas_size():
    spec = seed_layout(width_mm=183.0, height_mm=140.0)
    assert spec.width_mm == pytest.approx(183.0)
    assert spec.height_mm == pytest.approx(140.0)


def test_seed_layout_wires_panel_data_refs_from_a_real_bundle(tmp_path):
    """Regression: real run (Task 12) found every panel's ``data`` empty.

    ``seed_layout`` built rects only and left ``PanelSpec.data == {}`` for
    every panel, so `figbuilder export` KeyErrors on the first panel that
    reads `data["..."]` (e.g. WingZPanel needs `data["t_ms"]`). When a real
    bundle path is supplied, the seeded refs must point at that bundle's
    actual dataset/asset names so `panel_data()` can resolve them.
    """
    from figbuilder.bundle import PanelData, write_bundle

    bundle_path = tmp_path / "fig4_bundle.h5"
    write_bundle(bundle_path, meta={}, panels={
        "wing": PanelData(type="courtship.wing_z", data={
            "t_ms": np.arange(10.0), "wingL_z": np.zeros(10),
            "wingR_z": np.zeros(10),
        }),
        "video_0": PanelData(type="image",
                             assets={"img": np.zeros((4, 4, 3), np.uint8)}),
    })

    spec = seed_layout(n_video_frames=2, n_render_strip=2,
                       bundle_path=bundle_path)
    by_id = {p.id: p for p in spec.panels}

    wing_data = by_id["wing"].data
    assert wing_data["t_ms"] == {"dataset": "/panels/wing/data/t_ms"}
    assert wing_data["wingL_z"] == {"dataset": "/panels/wing/data/wingL_z"}
    assert wing_data["wingR_z"] == {"dataset": "/panels/wing/data/wingR_z"}

    video0_data = by_id["video_0"].data
    assert video0_data["img"] == {"dataset": "/panels/video_0/assets/img"}

    # No bundle path (e.g. plain unit tests): unchanged, data stays empty —
    # not a regression for every other test in this file.
    assert seed_layout().panel("wing").data == {}
