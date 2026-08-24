"""Compose per-panel tiles and an annotation layer into one SVG document."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from lxml import etree

from figbuilder.figure import FigureSpec
from figbuilder.svgutil import NSMAP, SVG_NS, namespace_ids, qname

PT_PER_MM = 72.0 / 25.4

#: Tile children that must not be copied into the composed document.
_SKIP = {"metadata"}


def compose(fig_spec: FigureSpec,
            tiles: Sequence[Tuple[str, bytes]],
            annotation_elements: Iterable = ()) -> bytes:
    """Overlay `tiles` (in order) and append the annotation layer."""
    w_pt = fig_spec.width_mm * PT_PER_MM
    h_pt = fig_spec.height_mm * PT_PER_MM

    root = etree.Element(qname("svg"), nsmap=NSMAP)
    # Six decimals, matching matplotlib's own pt formatting in the tiles this
    # document overlays — see Ruling 4 in the SDD ledger.
    root.set("width", f"{w_pt:.6f}pt")
    root.set("height", f"{h_pt:.6f}pt")
    root.set("viewBox", f"0 0 {w_pt:.6f} {h_pt:.6f}")
    root.set("version", "1.1")

    for panel_id, svg in tiles:
        tile_root = namespace_ids(svg, panel_id)
        g = etree.SubElement(root, qname("g"))
        g.set("id", f"panel_{panel_id}")
        for child in tile_root:
            if etree.QName(child).localname in _SKIP:
                continue
            g.append(child)

    ann: List = list(annotation_elements)
    layer = etree.SubElement(root, qname("g"))
    layer.set("id", "annotations")
    for el in ann:
        layer.append(el)

    return etree.tostring(root, xml_declaration=True, encoding="utf-8",
                          pretty_print=True)
