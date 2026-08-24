"""Smoke tests: every courtship adapter draws from its declared `needs`."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import figbuilder.panels  # noqa: F401  (registers types)
from figbuilder.bundle import segments_to_array
from figbuilder.panels.base import get_panel_type

RNG = np.random.default_rng(0)
T = 400
FS = 800.0


def _t_ms():
    return np.arange(T) / FS * 1000.0


def _segs():
    return segments_to_array([{"start": 20, "end": 90, "type": "pulse"},
                              {"start": 150, "end": 260, "type": "sine"}])


CASES = {
    "courtship.wing_z": (
        {"t_ms": _t_ms(), "wingL_z": RNG.normal(size=T), "wingR_z": RNG.normal(size=T),
         "seg_L": _segs(), "seg_R": _segs()},
        {"fs": FS, "frame_range": [0, T], "time_unit": "s"},
    ),
    "courtship.scutellum_z": (
        {"t_ms": _t_ms(), "scutellum_z": RNG.normal(size=T), "segments": _segs()},
        {"fs": FS, "frame_range": [0, T], "time_unit": "s"},
    ),
    "courtship.male_pitch": (
        {"t_ms": _t_ms(), "male_pitch": RNG.normal(size=T) * 10,
         "target_pitch": RNG.normal(size=T) * 10},
        {"fs": FS, "frame_range": [0, T], "time_unit": "s"},
    ),
    "courtship.pitch_violin": (
        {"per_bout": np.abs(RNG.normal(size=37)) * 20},
        {"exemplar_idx": 3},
    ),
    "courtship.zheight": (
        {"pulse_z": RNG.normal(size=25), "sine_z": RNG.normal(size=25),
         "walking_z": RNG.normal(size=25)},
        {"kind": "violin"},
    ),
    "courtship.angle_density": (
        {"ext_pulse": np.abs(RNG.normal(size=500)) * 30,
         "ext_sine": np.abs(RNG.normal(size=500)) * 30},
        {"range_deg": [0.0, 90.0]},
    ),
    "courtship.wing_polar": (
        {"phase_diffs": RNG.uniform(-np.pi, np.pi, size=80)},
        {"center_stat": "median"},
    ),
}


@pytest.mark.parametrize("type_id", sorted(CASES))
def test_adapter_draws_without_error(type_id):
    data, spec = CASES[type_id]
    pt = get_panel_type(type_id)
    projection = "polar" if type_id.endswith("polar") else None
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projection)
    try:
        pt.draw(ax, data, spec)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("type_id", sorted(CASES))
def test_adapter_declares_the_data_it_uses(type_id):
    data, _ = CASES[type_id]
    pt = get_panel_type(type_id)
    assert set(pt.needs) <= set(data), f"{type_id} needs {pt.needs}"
    assert pt.needs, f"{type_id} must declare its needs"


def test_wing_z_adapter_converts_segment_arrays_to_dicts():
    """The panel functions expect list-of-dicts, the bundle stores structs."""
    from figbuilder.panels.courtship import _segs_arg
    out = _segs_arg(_segs())
    assert out[0] == {"start": 20, "end": 90, "type": "pulse"}


def test_polar_panel_is_flagged_so_the_renderer_uses_a_polar_axes():
    assert get_panel_type("courtship.wing_polar").projection == "polar"
    assert get_panel_type("courtship.wing_z").projection is None


def test_zheight_relabels_free_walk_to_free_running():
    """The assay is FREE RUNNING, not free walking (user correction); the
    underlying panel function hardcodes a 'free walk' tick label, so the
    adapter must relabel it for real, preserving the '(n=...)' count."""
    data, spec = CASES["courtship.zheight"]
    pt = get_panel_type("courtship.zheight")
    fig, ax = plt.subplots()
    try:
        pt.draw(ax, data, spec)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert not any("free walk" in lb for lb in labels), labels
        running = [lb for lb in labels if "free running" in lb]
        assert running, labels
        assert "(n=" in running[0]
    finally:
        plt.close(fig)
