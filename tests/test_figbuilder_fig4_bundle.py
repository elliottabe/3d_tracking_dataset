"""build_fig4_panels shapes the bundle correctly from synthetic analysis output."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import figbuilder.panels  # noqa: F401
from figbuilder.panels.base import get_panel_type
from scripts.figures.export_fig4_bundle import build_fig4_panels

FS = 800.0
T = 300


def _synthetic():
    rng = np.random.default_rng(0)
    seg = [{"start": 10, "end": 80, "type": "pulse"},
           {"start": 120, "end": 220, "type": "sine"}]
    ex = {
        "T": T,
        "com_z": rng.normal(size=T),
        "song0": {
            "wing_data": {"WingL_V13": {"z": rng.normal(size=T)},
                          "WingR_V13": {"z": rng.normal(size=T)}},
            "sides": {"L": {"segments": seg}, "R": {"segments": seg}},
            "angle_L": rng.normal(size=T), "angle_R": rng.normal(size=T),
            "dominant_wing": "L",
        },
    }
    results = [{"male_valid": np.ones(T, bool),
                "male_labels": np.array(["pulse"] * 150 + ["sine"] * 150),
                "com_z": rng.normal(size=T),
                "song0": ex["song0"]} for _ in range(4)]
    extras = {
        "fs": FS, "start_frame": 0, "end_frame": T,
        "walking_z": rng.normal(size=20),
        "phase_diffs": rng.uniform(-np.pi, np.pi, 40),
        "per_bout_align": np.abs(rng.normal(size=30)) * 15,
        "male_pitch": rng.normal(size=T) * 10,
        "target_pitch": rng.normal(size=T) * 10,
        "pulse_centroids": {"Pslow": rng.normal(size=48),
                            "Pfast": rng.normal(size=48)},
        "pulse_pooled": {"Pslow": rng.normal(size=(9, 48)),
                         "Pfast": rng.normal(size=(7, 48))},
        "pulse_counts": {"Pslow": 9, "Pfast": 7},
        "ext_pulse": np.abs(rng.normal(size=200)) * 30,
        "ext_sine": np.abs(rng.normal(size=200)) * 30,
        "video_frames": [np.zeros((16, 16, 3), np.uint8) for _ in range(4)],
        "render_frames": [np.zeros((16, 16, 3), np.uint8) for _ in range(4)],
        "exemplar_bout_idx": 3,
    }
    return results, ex, extras


def test_bundle_contains_a_panel_for_every_fig4_element():
    panels = build_fig4_panels(*_synthetic())
    expected = {"wing", "scut", "sine_phase", "wing_phase_polar", "angle_2d",
                "pulse_class", "zheight", "pitch", "align_violin"}
    assert expected <= set(panels)


def test_video_and_render_frames_become_individual_image_panels():
    panels = build_fig4_panels(*_synthetic())
    assert {"video_0", "video_1", "video_2", "video_3"} <= set(panels)
    assert {"render_0", "render_1", "render_2", "render_3"} <= set(panels)
    assert panels["video_0"].assets["img"].dtype == np.uint8


def test_pulse_class_panel_stores_flat_per_type_arrays():
    """Ruling 5: the bundle is flat; the adapter re-nests into Pslow/Pfast."""
    panels = build_fig4_panels(*_synthetic())
    d = panels["pulse_class"].data
    assert {"centroid_Pslow", "centroid_Pfast",
            "pooled_Pslow", "pooled_Pfast"} <= set(d)
    assert d["centroid_Pslow"].shape == (48,)
    assert d["pooled_Pfast"].shape == (7, 48)


def test_pulse_class_adapter_renders_from_the_bundled_arrays():
    """End-to-end: flat datasets -> nested pulse_type_results -> a drawn axes."""
    import matplotlib.pyplot as plt
    panels = build_fig4_panels(*_synthetic())
    pt = get_panel_type("courtship.pulse_class")
    fig, ax = plt.subplots()
    try:
        pt.draw(ax, panels["pulse_class"].data,
                {"fs": FS, "show_std": True, "count_Pslow": 9, "count_Pfast": 7})
        labels = [ln.get_label() for ln in ax.lines]
        assert any("Pslow (n=9)" == lb for lb in labels), labels
        assert any("Pfast (n=7)" == lb for lb in labels), labels
    finally:
        plt.close(fig)


def test_segment_tables_are_stored_as_structured_arrays():
    panels = build_fig4_panels(*_synthetic())
    assert panels["wing"].data["seg_L"].dtype.names == ("start", "end", "type")


@pytest.mark.parametrize("pid", ["wing", "scut", "pitch", "align_violin",
                                 "zheight", "angle_2d", "wing_phase_polar",
                                 "sine_phase", "pulse_class"])
def test_each_panel_satisfies_its_types_declared_needs(pid):
    """Covers EVERY data-bearing panel, including sine_phase and pulse_class.

    This is the test that catches an adapter whose `needs` drifts from what
    build_fig4_panels actually emits — exactly the pulse_class defect found in
    the pre-flight audit (Ruling 5).
    """
    panels = build_fig4_panels(*_synthetic())
    pd = panels[pid]
    pt = get_panel_type(pd.type)
    assert set(pt.needs) <= set(pd.data), f"{pid} missing {set(pt.needs) - set(pd.data)}"
