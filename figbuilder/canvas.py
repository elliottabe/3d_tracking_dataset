"""Canvas sizing for paper figures.

Journal figures are sized to a column width, so the canvas is an explicit
property rather than an afterthought. Sizes are stored in millimetres (the
unit journals specify) but may be entered in mm, cm, inches, or points.

The max-width guard is deliberately a WARNING, never an error: house styles
vary, posters and talk slides are legitimately wider, and a tool that refuses
to render is worse than one that tells you what it noticed.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

MM_PER_INCH = 25.4
MM_PER_PT = MM_PER_INCH / 72.0

#: US Letter width — the widest a figure can be and still print on a page.
DEFAULT_MAX_WIDTH_MM = 8.5 * MM_PER_INCH  # 215.9

_UNIT_TO_MM: Dict[str, float] = {
    "mm": 1.0,
    "cm": 10.0,
    "in": MM_PER_INCH,
    "pt": MM_PER_PT,
}

#: (width_mm, height_mm | None). None height means "you choose".
CANVAS_PRESETS: Dict[str, Tuple[float, Optional[float]]] = {
    "nature-single":   (89.0, None),
    "nature-double":   (183.0, None),
    "cell-1.5col":     (120.0, None),
    "elsevier-single": (90.0, None),
    "elsevier-double": (190.0, None),
    "plos-single":     (83.0, None),
    "plos-double":     (173.0, None),
    "us-letter":       (215.9, 279.4),
    "a4":              (210.0, 297.0),
}


def to_mm(value: float, unit: str = "mm") -> float:
    try:
        return float(value) * _UNIT_TO_MM[unit]
    except KeyError:
        raise ValueError(
            f"unknown unit {unit!r}; expected one of {sorted(_UNIT_TO_MM)}"
        ) from None


def from_mm(value_mm: float, unit: str = "mm") -> float:
    try:
        return float(value_mm) / _UNIT_TO_MM[unit]
    except KeyError:
        raise ValueError(
            f"unknown unit {unit!r}; expected one of {sorted(_UNIT_TO_MM)}"
        ) from None


def resolve_preset(name: str) -> Tuple[float, Optional[float]]:
    try:
        return CANVAS_PRESETS[name]
    except KeyError:
        raise KeyError(
            f"unknown canvas preset {name!r}; known presets: "
            f"{', '.join(sorted(CANVAS_PRESETS))}"
        ) from None


def check_canvas(spec) -> List[str]:
    """Return human-readable warnings about a figure's canvas size.

    Never raises for an oversized canvas — the caller decides what to do.
    """
    out: List[str] = []
    if spec.width_mm <= 0 or spec.height_mm <= 0:
        out.append(
            f"canvas size must be positive, got "
            f"{spec.width_mm:g} x {spec.height_mm:g} mm"
        )
        return out
    if spec.max_width_mm is not None and spec.width_mm > spec.max_width_mm:
        out.append(
            f"canvas width {spec.width_mm:g} mm "
            f"({from_mm(spec.width_mm, 'in'):.2f} in) exceeds the "
            f"{spec.max_width_mm:g} mm ({from_mm(spec.max_width_mm, 'in'):.2f} in) "
            f"limit; set \"max_width_mm\": null to silence this"
        )
    return out
