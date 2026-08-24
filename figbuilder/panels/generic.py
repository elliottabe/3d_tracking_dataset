"""Generic, data-agnostic panel types."""
from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from figbuilder.panels.base import PanelType, register


def _decorate(ax: plt.Axes, spec: Dict[str, Any]) -> None:
    if spec.get("xlabel"):
        ax.set_xlabel(spec["xlabel"])
    if spec.get("ylabel"):
        ax.set_ylabel(spec["ylabel"])
    if spec.get("title"):
        ax.set_title(spec["title"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


@register
class LinePanel(PanelType):
    id = "line"
    label = "Line / trace"
    needs = []
    schema = {
        "type": "object",
        "properties": {
            "series": {"type": "array", "title": "Series", "items": {
                "type": "object", "properties": {
                    "x": {"type": "string", "title": "x dataset"},
                    "y": {"type": "string", "title": "y dataset"},
                    "label": {"type": "string"},
                    "color": {"type": "string", "format": "color"},
                    "lw": {"type": "number", "default": 0.7},
                }}},
            "xlabel": {"type": "string"},
            "ylabel": {"type": "string"},
            "title": {"type": "string"},
            "legend": {"type": "boolean", "default": False},
        },
    }

    def draw(self, ax, data, spec):
        for s in spec.get("series", []):
            y = np.asarray(data[s["y"]], dtype=float)
            x = np.asarray(data[s["x"]], dtype=float) if s.get("x") else np.arange(y.size)
            ax.plot(x, y, label=s.get("label", ""),
                    color=s.get("color"), lw=float(s.get("lw", 0.7)))
        if spec.get("legend"):
            ax.legend()
        _decorate(ax, spec)


@register
class ImagePanel(PanelType):
    id = "image"
    label = "Image / rendered frame"
    needs = []
    schema = {
        "type": "object",
        "properties": {
            "asset": {"type": "string", "title": "Asset name"},
            "interpolation": {"type": "string", "default": "nearest"},
        },
    }

    def draw(self, ax, data, spec):
        ax.imshow(np.asarray(data[spec["asset"]]),
                  interpolation=spec.get("interpolation", "nearest"))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)


@register
class BlankPanel(PanelType):
    id = "blank"
    label = "Blank (spacer / hand graphic)"
    needs = []
    schema = {"type": "object", "properties": {}}

    def draw(self, ax, data, spec):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.patch.set_alpha(0.0)
