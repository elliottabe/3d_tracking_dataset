"""Annotation layer: objects -> SVG elements.

This is the Python half of a deliberately duplicated emitter; the TypeScript
half drives live editing in the browser. A conformance test (Task 14) asserts
the two agree. Duplication is preferred over making headless export depend on
a browser.

Positions are in millimetres. Figure space has y growing UPWARD from the
bottom-left (matching matplotlib); SVG has y growing downward, so every y is
flipped here exactly once.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from lxml import etree

from figbuilder.svgutil import qname

PT_PER_MM = 72.0 / 25.4


def _num(v: float) -> str:
    """Format a number identically in Python and TypeScript.

    Six decimals, trailing zeros trimmed. `:g` is NOT usable here: it gives
    6 SIGNIFICANT figures, so 141.73228346 becomes "141.732" — a 2e-6 relative
    error that breaks both pytest.approx and character-exact agreement with the
    TS emitter. See Ruling 3 in the SDD ledger.
    """
    out = f"{float(v):.6f}".rstrip("0").rstrip(".")
    # Negative zero must render as "0": JS toFixed(-0.0) has no sign, and a
    # coordinate of exactly -0.0 is reachable (a panel at the canvas edge).
    return "0" if out.lstrip("-") in ("", "0") else out


ANNOTATION_KINDS = frozenset({
    "text", "line", "arrow", "leader", "rect", "ellipse", "bracket",
    "scalebar", "image",
})

_ARROW_MARKER_ID = "fb_arrowhead"


def _style(d: Dict[str, Any]) -> str:
    parts: List[str] = []
    if "color" in d:
        parts.append(f"fill:{d['color']}")
    if "stroke" in d:
        parts.append(f"stroke:{d['stroke']}")
    if "lw_pt" in d:
        parts.append(f"stroke-width:{_num(d['lw_pt'])}")
    if "font_size_pt" in d:
        parts.append(f"font-size:{_num(d['font_size_pt'])}px")
    if d.get("weight") in ("bold", 700):
        parts.append("font-weight:700")
    if "font" in d:
        parts.append(f"font-family:{d['font']}")
    if "opacity" in d:
        parts.append(f"opacity:{_num(d['opacity'])}")
    return ";".join(parts)


class _Frame:
    """Converts annotation mm coordinates into SVG point coordinates."""

    def __init__(self, fig_spec, parent_id: Optional[str]):
        self.w_mm = fig_spec.width_mm
        self.h_mm = fig_spec.height_mm
        self.ox_mm = 0.0
        self.oy_mm = 0.0
        if parent_id:
            x, y, _, _ = fig_spec.panel(parent_id).rect
            self.ox_mm = x * self.w_mm
            self.oy_mm = y * self.h_mm

    def x(self, mm: float) -> float:
        return (self.ox_mm + float(mm)) * PT_PER_MM

    def y(self, mm: float) -> float:
        return (self.h_mm - (self.oy_mm + float(mm))) * PT_PER_MM

    def d(self, mm: float) -> float:
        return float(mm) * PT_PER_MM


def _pt(seq: Sequence[float], default=(0.0, 0.0)):
    return tuple(seq) if seq is not None else default


def emit_annotations(fig_spec, annotations: List[dict]) -> List:
    """Convert annotation objects into SVG elements, in document order."""
    out: List = []
    needs_marker = False

    for ann in annotations:
        kind = ann.get("kind")
        if kind not in ANNOTATION_KINDS:
            raise ValueError(
                f"unknown annotation kind {kind!r}; expected one of "
                f"{sorted(ANNOTATION_KINDS)}")
        f = _Frame(fig_spec, ann.get("parent"))
        st = dict(ann.get("style", {}))
        px, py = _pt(ann.get("pos_mm"))

        if kind == "text":
            el = etree.Element(qname("text"))
            el.set("x", _num(f.x(px)))
            el.set("y", _num(f.y(py)))
            st.setdefault("font", "Arial, Helvetica, sans-serif")
            el.set("style", _style(st))
            el.text = str(ann.get("text", ""))

        elif kind in ("line", "leader"):
            tx, ty = _pt(ann.get("to_mm"))
            el = etree.Element(qname("line"))
            el.set("x1", _num(f.x(px))); el.set("y1", _num(f.y(py)))
            el.set("x2", _num(f.x(tx))); el.set("y2", _num(f.y(ty)))
            st.setdefault("stroke", "#000000"); st.setdefault("lw_pt", 0.8)
            el.set("style", _style(st))

        elif kind == "arrow":
            tx, ty = _pt(ann.get("to_mm"))
            el = etree.Element(qname("path"))
            el.set("d", f"M {_num(f.x(px))},{_num(f.y(py))} "
                        f"L {_num(f.x(tx))},{_num(f.y(ty))}")
            st.setdefault("stroke", "#000000"); st.setdefault("lw_pt", 0.8)
            el.set("style", _style(st) + f";fill:none;marker-end:url(#{_ARROW_MARKER_ID})")
            needs_marker = True

        elif kind == "rect":
            w, h = _pt(ann.get("size_mm"), (1.0, 1.0))
            el = etree.Element(qname("rect"))
            el.set("x", _num(f.x(px)))
            el.set("y", _num(f.y(py + h)))
            el.set("width", _num(f.d(w))); el.set("height", _num(f.d(h)))
            st.setdefault("stroke", "#000000"); st.setdefault("lw_pt", 0.8)
            el.set("style", _style(st) + ";fill:none" if "color" not in st else _style(st))

        elif kind == "ellipse":
            w, h = _pt(ann.get("size_mm"), (1.0, 1.0))
            el = etree.Element(qname("ellipse"))
            el.set("cx", _num(f.x(px + w / 2)))
            el.set("cy", _num(f.y(py + h / 2)))
            el.set("rx", _num(f.d(w / 2))); el.set("ry", _num(f.d(h / 2)))
            st.setdefault("stroke", "#000000"); st.setdefault("lw_pt", 0.8)
            el.set("style", _style(st) + ";fill:none" if "color" not in st else _style(st))

        elif kind == "bracket":
            tx, ty = _pt(ann.get("to_mm"))
            tick = f.d(float(ann.get("tick_mm", 1.0)))
            x1, y1, x2, y2 = f.x(px), f.y(py), f.x(tx), f.y(ty)
            el = etree.Element(qname("path"))
            el.set("d", f"M {_num(x1)},{_num(y1 + tick)} "
                        f"L {_num(x1)},{_num(y1)} "
                        f"L {_num(x2)},{_num(y2)} "
                        f"L {_num(x2)},{_num(y2 + tick)}")
            st.setdefault("stroke", "#000000"); st.setdefault("lw_pt", 0.8)
            el.set("style", _style(st) + ";fill:none")

        elif kind == "scalebar":
            length = float(ann.get("length_mm", 5.0))
            el = etree.Element(qname("g"))
            ln = etree.SubElement(el, qname("line"))
            ln.set("x1", _num(f.x(px))); ln.set("y1", _num(f.y(py)))
            ln.set("x2", _num(f.x(px + length))); ln.set("y2", _num(f.y(py)))
            ln.set("style", _style({"stroke": st.get("stroke", "#000000"),
                                    "lw_pt": st.get("lw_pt", 1.2)}))
            tx_el = etree.SubElement(el, qname("text"))
            tx_el.set("x", _num(f.x(px + length / 2)))
            tx_el.set("y", _num(f.y(py) + f.d(2.0)))
            tx_el.set("style", _style({"font_size_pt": st.get("font_size_pt", 6),
                                       "font": "Arial, Helvetica, sans-serif"})
                      + ";text-anchor:middle")
            tx_el.text = str(ann.get("label", ""))

        elif kind == "image":
            w, h = _pt(ann.get("size_mm"), (10.0, 10.0))
            el = etree.Element(qname("image"))
            el.set("x", _num(f.x(px)))
            el.set("y", _num(f.y(py + h)))
            el.set("width", _num(f.d(w))); el.set("height", _num(f.d(h)))
            el.set("{http://www.w3.org/1999/xlink}href", ann.get("href", ""))

        if ann.get("id"):
            el.set("id", str(ann["id"]))
        out.append(el)

    if needs_marker:
        defs = etree.Element(qname("defs"))
        m = etree.SubElement(defs, qname("marker"))
        m.set("id", _ARROW_MARKER_ID)
        m.set("viewBox", "0 0 10 10"); m.set("refX", "9"); m.set("refY", "5")
        m.set("markerWidth", "5"); m.set("markerHeight", "5")
        m.set("orient", "auto-start-reverse")
        p = etree.SubElement(m, qname("path"))
        p.set("d", "M 0,0 L 10,5 L 0,10 z")
        out.insert(0, defs)

    return out
