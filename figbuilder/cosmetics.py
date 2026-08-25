"""Per-panel cosmetics applied AFTER a panel type has drawn.

The panel `draw` methods delegate to the researcher's analysis code in
`utils/`, which is consumed unmodified. Anything we want to change about the
resulting axes — spines, tick labels, where a legend sits — therefore has to
happen afterwards, on the axes object, rather than by passing new arguments
down into that code.

Applying these before `_ink_box` measures the tile matters: hiding tick labels
shrinks the ink box, which is what stops a panel's labels from overflowing the
canvas onto its neighbour.

Repositioning deliberately does NOT rebuild the legend. `utils`'
`_colored_text_legend` draws colour-matched text with no handles; calling
`ax.legend(...)` again would discard that styling and put the markers back.
So we move the legend object that is already there, and never recreate it.
"""
from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt

#: Legend positions matplotlib accepts, for the schema-driven properties form.
LEGEND_LOCS = [
    "best", "upper right", "upper left", "lower left", "lower right",
    "right", "center left", "center right", "lower center", "upper center",
    "center",
]

#: Merged into every panel type's schema, so the front-end properties form
#: offers these controls on every panel without each type restating them.
SHARED_SCHEMA: Dict[str, Any] = {
    "spines": {
        "type": "object",
        "title": "Spines",
        "properties": {
            "top": {"type": "boolean", "title": "Top"},
            "right": {"type": "boolean", "title": "Right"},
            "bottom": {"type": "boolean", "title": "Bottom"},
            "left": {"type": "boolean", "title": "Left"},
        },
    },
    "hide_xticklabels": {
        "type": "boolean", "default": False, "title": "Hide x tick labels",
    },
    "hide_yticklabels": {
        "type": "boolean", "default": False, "title": "Hide y tick labels",
    },
    "legend": {
        "type": "object",
        "title": "Legend",
        "properties": {
            "hide": {"type": "boolean", "title": "Hide legend"},
            "loc": {"type": "string", "enum": LEGEND_LOCS, "title": "Position"},
            "bbox_to_anchor": {
                "type": "array",
                "items": {"type": "number"},
                "title": "Anchor (x, y[, w, h]) in axes fractions",
            },
        },
    },
}


def _apply_spines(ax: plt.Axes, spines: Dict[str, Any]) -> None:
    for name, visible in spines.items():
        # Polar axes have a single 'polar' spine, image panels may have none;
        # silently skip names this projection does not have rather than raising.
        spine = ax.spines.get(name)
        if spine is not None:
            spine.set_visible(bool(visible))


def _apply_legend(ax: plt.Axes, legend: Dict[str, Any]) -> None:
    leg = ax.get_legend()
    if leg is None:
        return
    if legend.get("hide"):
        leg.remove()
        return
    bbox = legend.get("bbox_to_anchor")
    if bbox is not None:
        # A 2-tuple anchors a point; a 4-tuple anchors a box. Both are valid.
        leg.set_bbox_to_anchor(tuple(float(v) for v in bbox))
    loc = legend.get("loc")
    if loc is not None:
        leg.set_loc(loc)


def apply_cosmetics(ax: plt.Axes, spec: Dict[str, Any] | None) -> None:
    """Apply a panel's shared cosmetic options to an already-drawn axes.

    Every option is absent-by-default: a spec that sets none of them leaves
    the axes exactly as the panel type drew it.
    """
    if not spec:
        return

    spines = spec.get("spines")
    if isinstance(spines, dict):
        _apply_spines(ax, spines)

    if spec.get("hide_xticklabels"):
        ax.tick_params(bottom=False, labelbottom=False)
    if spec.get("hide_yticklabels"):
        ax.tick_params(left=False, labelleft=False)

    legend = spec.get("legend")
    if isinstance(legend, dict):
        _apply_legend(ax, legend)
