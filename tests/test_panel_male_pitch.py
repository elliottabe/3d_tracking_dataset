"""Tests for utils.courtship_figure_panels.panel_male_pitch."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.courtship_figure_panels import panel_male_pitch


def test_panel_male_pitch_plots_two_lines_and_shades_segments():
    fs = 800.0
    T = 1600
    t_ms = np.arange(T) / fs * 1000.0
    body = np.linspace(-5.0, 25.0, T)
    target = body + np.sin(np.linspace(0, 6.28, T)) * 4.0
    segs = [
        {'type': 'pulse', 'start': 100, 'end': 300},
        {'type': 'sine',  'start': 400, 'end': 700},
    ]
    fig, ax = plt.subplots()
    panel_male_pitch(ax, t_ms, body, target,
                     segments=segs, fs=fs, frame_range=(0, T),
                     min_segment_ms=10.0)
    # Two Line2D artists (solid + dashed) and at least 2 shaded spans.
    lines = ax.get_lines()
    assert len(lines) >= 2
    assert any(l.get_linestyle() == '-' for l in lines)
    assert any(l.get_linestyle() == '--' for l in lines)
    patches = [p for p in ax.patches if getattr(p, 'get_xy', None) is not None]
    assert len(patches) >= 2
    assert ax.get_ylabel() == 'Pitch (°)'
    plt.close(fig)


def test_panel_male_pitch_handles_nan_gaps():
    fs = 800.0
    T = 200
    t_ms = np.arange(T) / fs * 1000.0
    body = np.full(T, 10.0)
    body[50:60] = np.nan
    target = np.full(T, 5.0)
    fig, ax = plt.subplots()
    panel_male_pitch(ax, t_ms, body, target, fs=fs, frame_range=(0, T))
    # Nothing raised; solid line exists.
    assert len(ax.get_lines()) >= 2
    plt.close(fig)
