"""Single source of truth for render style.

Every renderer — tile, export, and test — routes through `apply_style`, so a
figure rendered in the browser and one exported headlessly cannot drift.
"""
from __future__ import annotations

from typing import Any, Dict

import matplotlib

#: Non-negotiable: text must export as real <text>, and PDF must embed Type-42.
FORCED_RCPARAMS: Dict[str, Any] = {
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

DEFAULT_RCPARAMS: Dict[str, Any] = {
    "font.family": "sans-serif",
    # Liberation Sans is metric-compatible with Arial and is what fontconfig
    # substitutes when Arial is absent. Including it keeps matplotlib's LAYOUT
    # font identical to the font the SVG renderer actually DRAWS with, so
    # ink_box measurements stay truthful on machines without Arial. Ruling 7.
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 6.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
}


def apply_style(style: Dict[str, Any] | None = None) -> None:
    """Apply default rcParams, then the figure's overrides, then the forced set."""
    matplotlib.rcParams.update(DEFAULT_RCPARAMS)
    if style:
        matplotlib.rcParams.update(style)
    matplotlib.rcParams.update(FORCED_RCPARAMS)
