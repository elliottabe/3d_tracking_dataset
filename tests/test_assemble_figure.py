"""Tests for utils.courtship_figure_panels.assemble_figure Row 4 split."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.courtship_figure_panels import (
    DEFAULT_PANEL_LETTERS,
    assemble_figure,
)


def test_assemble_figure_default_render_strip_is_4_and_pitch_axes_exists():
    fig, axd = assemble_figure()
    try:
        assert 'render' in axd
        assert 'pitch' in axd
        assert isinstance(axd['render'], list)
        assert len(axd['render']) == 4
        assert isinstance(axd['pitch'], plt.Axes)
    finally:
        plt.close(fig)


def test_assemble_figure_respects_custom_n_render_strip():
    fig, axd = assemble_figure(n_render_strip=6)
    try:
        assert len(axd['render']) == 6
    finally:
        plt.close(fig)


def test_default_panel_letters_include_pitch_j():
    keys = [k for k, _ in DEFAULT_PANEL_LETTERS]
    letters = [ch for _, ch in DEFAULT_PANEL_LETTERS]
    assert 'pitch' in keys
    assert letters[keys.index('pitch')] == 'J'
