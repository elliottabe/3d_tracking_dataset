"""Adapters exposing utils.courtship_figure_panels as registry panel types.

Each adapter is a thin shim: convert bundle-stored structured arrays back to
the list-of-dicts the panel functions expect, then forward. The panel
functions themselves are consumed unmodified.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from utils import courtship_figure_panels as cfp

from figbuilder.bundle import array_to_segments
from figbuilder.panels.base import PanelType, register

#: The assay is FREE RUNNING, not free walking. `panel_z_height_singing_vs_
#: walking` (utils/, consumed unmodified) hardcodes a 'free walk\n(n=...)'
#: tick label; `ZHeightPanel.draw` substitutes this string in for the
#: 'free walk' portion after drawing, preserving the '(n=...)' suffix.
#: Module-level so it is greppable and the schema default can reference it.
FREE_LABEL = "free running"


def _segs_arg(value: Any) -> List[dict]:
    """Accept a structured segment array or an already-decoded list."""
    if value is None:
        return []
    arr = np.asarray(value)
    if arr.dtype.names:
        return array_to_segments(arr)
    return list(value)


def _frame_range(spec: Dict[str, Any]) -> Optional[tuple]:
    fr = spec.get("frame_range")
    return None if fr is None else (int(fr[0]), int(fr[1]))


_TIME_SCHEMA = {
    "fs": {"type": "number", "default": 800.0, "title": "Sample rate (Hz)"},
    "frame_range": {"type": "array", "items": {"type": "integer"},
                    "title": "Frame range"},
    "time_unit": {"type": "string", "enum": ["ms", "s"], "default": "s"},
}


@register
class WingZPanel(PanelType):
    id = "courtship.wing_z"
    label = "Wing V13 z + song segments"
    needs = ["t_ms", "wingL_z", "wingR_z", "seg_L", "seg_R"]
    schema = {"type": "object", "properties": {
        **_TIME_SCHEMA,
        "min_segment_ms": {"type": "number", "default": 0.0},
    }}

    def draw(self, ax, data, spec):
        cfp.panel_wing_z_traces(
            ax, np.asarray(data["t_ms"]), np.asarray(data["wingL_z"]),
            np.asarray(data["wingR_z"]),
            _segs_arg(data.get("seg_L")), _segs_arg(data.get("seg_R")),
            fs=float(spec.get("fs", 800.0)),
            frame_range=_frame_range(spec),
            min_segment_ms=float(spec.get("min_segment_ms", 0.0)),
            time_unit=spec.get("time_unit", "s"),
        )


@register
class ScutellumZPanel(PanelType):
    id = "courtship.scutellum_z"
    label = "Scutellum (body) z"
    needs = ["t_ms", "scutellum_z"]
    schema = {"type": "object", "properties": {
        **_TIME_SCHEMA,
        "line_color": {"type": "string", "format": "color", "default": "k"},
    }}

    def draw(self, ax, data, spec):
        cfp.panel_scutellum_z_trace(
            ax, np.asarray(data["t_ms"]), np.asarray(data["scutellum_z"]),
            segments=_segs_arg(data.get("segments")),
            fs=float(spec.get("fs", 800.0)),
            frame_range=_frame_range(spec),
            line_color=spec.get("line_color", "k"),
            time_unit=spec.get("time_unit", "s"),
        )


@register
class MalePitchPanel(PanelType):
    id = "courtship.male_pitch"
    label = "Male body pitch vs target pitch"
    needs = ["t_ms", "male_pitch", "target_pitch"]
    schema = {"type": "object", "properties": {
        **_TIME_SCHEMA,
        "male_color": {"type": "string", "format": "color", "default": "#d62728"},
        "target_color": {"type": "string", "format": "color", "default": "#1f77b4"},
    }}

    def draw(self, ax, data, spec):
        cfp.panel_male_pitch(
            ax, np.asarray(data["t_ms"]), np.asarray(data["male_pitch"]),
            np.asarray(data["target_pitch"]),
            segments=None, fs=float(spec.get("fs", 800.0)),
            frame_range=_frame_range(spec),
            male_color=spec.get("male_color", "#d62728"),
            target_color=spec.get("target_color", "#1f77b4"),
            time_unit=spec.get("time_unit", "s"),
        )


@register
class PitchViolinPanel(PanelType):
    id = "courtship.pitch_violin"
    label = "Per-bout pitch alignment (violin)"
    needs = ["per_bout"]
    schema = {"type": "object", "properties": {
        "exemplar_idx": {"type": ["integer", "null"], "title": "Exemplar index"},
    }}

    def draw(self, ax, data, spec):
        cfp.panel_pitch_alignment_violin(
            ax, np.asarray(data["per_bout"]),
            exemplar_idx=spec.get("exemplar_idx"),
        )


@register
class ZHeightPanel(PanelType):
    id = "courtship.zheight"
    label = "Body height: singing vs walking"
    needs = ["pulse_z", "sine_z", "walking_z"]
    schema = {"type": "object", "properties": {
        "kind": {"type": "string", "enum": ["violin", "box"], "default": "violin"},
        "title": {"type": "string", "default": ""},
        "free_label": {"type": "string", "default": FREE_LABEL,
                       "title": "Control-condition label"},
    }}

    def draw(self, ax, data, spec):
        cfp.panel_z_height_singing_vs_walking(
            ax, np.asarray(data["pulse_z"]), np.asarray(data["sine_z"]),
            np.asarray(data["walking_z"]),
            kind=spec.get("kind", "violin"),
            point_kwargs={"s": 3, "alpha": 0.25},
            title=spec.get("title", ""),
        )
        # The assay is FREE RUNNING, not free walking; the panel function
        # above (utils/, consumed unmodified) hardcodes a 'free walk' tick
        # label, so relabel it here at the figbuilder layer, preserving the
        # '(n=...)' suffix.
        free_label = spec.get("free_label", FREE_LABEL)
        ax.set_xticks(ax.get_xticks())  # avoid FixedFormatter/FixedLocator warning
        new_labels = [t.get_text().replace("free walk", free_label)
                     for t in ax.get_xticklabels()]
        ax.set_xticklabels(new_labels)


@register
class AngleDensityPanel(PanelType):
    id = "courtship.angle_density"
    label = "Wing angle density: pulse vs sine"
    needs = ["ext_pulse", "ext_sine"]
    schema = {"type": "object", "properties": {
        "range_deg": {"type": "array", "items": {"type": "number"},
                      "default": [0.0, 90.0]},
        "title": {"type": "string", "default": ""},
    }}

    def draw(self, ax, data, spec):
        rd = spec.get("range_deg", [0.0, 90.0])
        cfp.panel_joint_angle_density(
            ax, np.asarray(data["ext_pulse"]), np.asarray(data["ext_sine"]),
            range_deg=(float(rd[0]), float(rd[1])),
            title=spec.get("title", ""),
        )


@register
class WingPolarPanel(PanelType):
    id = "courtship.wing_polar"
    label = "L-R wing phase difference (polar)"
    needs = ["phase_diffs"]
    projection = "polar"
    schema = {"type": "object", "properties": {
        "center_stat": {"type": "string", "enum": ["median", "mean"],
                        "default": "median"},
        "title": {"type": "string", "default": ""},
    }}

    def draw(self, ax, data, spec):
        cfp.panel_wing_phase_polar(
            ax, np.asarray(data["phase_diffs"]),
            center_stat=spec.get("center_stat", "median"),
            title=spec.get("title", ""),
        )


@register
class SineInPhasePanel(PanelType):
    id = "courtship.sine_inphase"
    label = "Sine song: extended vs folded wing in phase"
    needs = ["t_ms", "ext_z", "fold_z"]
    schema = {"type": "object", "properties": {
        **_TIME_SCHEMA, "title": {"type": "string", "default": ""}}}

    def draw(self, ax, data, spec):
        cfp.panel_sine_wing_inphase(
            ax, np.asarray(data["t_ms"]), np.asarray(data["ext_z"]),
            np.asarray(data["fold_z"]), fs=float(spec.get("fs", 800.0)),
            frame_range=_frame_range(spec),
            sine_segments=_segs_arg(data.get("sine_segments")),
            title=spec.get("title", ""),
        )


_PULSE_TYPES = ("Pslow", "Pfast")


@register
class PulseClassPanel(PanelType):
    """Reassembles the nested `pulse_type_results` dict from flat datasets.

    `panel_pulse_classification` wants {'centroids': {'Pslow': wf, 'Pfast': wf},
    'counts': {...}, 'pooled_waveforms': {'Pslow': (n, W), ...}, 'fs': float}.
    The bundle stores flat named arrays, so per-type arrays live as
    `centroid_<T>` / `pooled_<T>` and are re-nested here. Std shading is derived
    by the panel function from `pooled_waveforms` — there is no `stds` input.
    """

    id = "courtship.pulse_class"
    label = "Pslow / Pfast typed centroids"
    needs = ["centroid_Pslow", "centroid_Pfast"]
    schema = {"type": "object", "properties": {
        "show_std": {"type": "boolean", "default": True},
        "fs": {"type": "number", "default": 800.0},
        "count_Pslow": {"type": "integer", "default": 0},
        "count_Pfast": {"type": "integer", "default": 0},
        "title": {"type": "string", "default": ""}}}

    def draw(self, ax, data, spec):
        results = {
            "centroids": {t: np.asarray(data[f"centroid_{t}"], dtype=float)
                          for t in _PULSE_TYPES if f"centroid_{t}" in data},
            "counts": {t: int(spec.get(f"count_{t}", 0)) for t in _PULSE_TYPES},
            "pooled_waveforms": {t: np.asarray(data[f"pooled_{t}"], dtype=float)
                                 for t in _PULSE_TYPES if f"pooled_{t}" in data},
            "fs": float(spec.get("fs", 800.0)),
        }
        cfp.panel_pulse_classification(
            ax, results, show_std=bool(spec.get("show_std", True)),
            title=spec.get("title", ""),
        )
