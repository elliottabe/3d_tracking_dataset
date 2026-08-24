"""SVG id namespacing and document composition."""
from __future__ import annotations

import re

import matplotlib
matplotlib.use('Agg')
import numpy as np
from lxml import etree

import figbuilder.panels  # noqa: F401
from figbuilder.compose import compose
from figbuilder.figure import FigureSpec, PanelSpec
from figbuilder.render import render_tile
from figbuilder.svgutil import XLINK_NS, namespace_ids


def _fig():
    return FigureSpec(width_mm=100.0, height_mm=60.0)


def _tile(pid, rect, color):
    p = PanelSpec(id=pid, type="line", rect=rect,
                  data={"y": {"dataset": f"/panels/{pid}/data/y"}},
                  spec={"series": [{"y": "y", "color": color}], "ylabel": "z"})
    return render_tile(_fig(), p, {"y": np.linspace(0, 1, 40)}).svg


def test_namespace_ids_rewrites_definitions_and_references():
    svg = _tile("a", (0.2, 0.2, 0.6, 0.6), "#000000")
    root = namespace_ids(svg, "a")
    ids = [e.get("id") for e in root.iter() if e.get("id")]
    assert ids and all(i.startswith("a__") for i in ids)
    blob = etree.tostring(root)
    assert b'href="#a__' in blob or b"#a__" in blob or b"url(#a__" in blob


def test_compose_produces_no_duplicate_ids_across_tiles():
    out = compose(_fig(), [("a", _tile("a", (0.1, 0.55, 0.8, 0.35), "#38bdf8")),
                           ("b", _tile("b", (0.1, 0.12, 0.8, 0.35), "#000000"))])
    ids = re.findall(rb'id="([^"]+)"', out)
    assert len(ids) == len(set(ids)), "duplicate ids in composed document"


def test_compose_wraps_each_tile_in_a_named_group():
    out = compose(_fig(), [("a", _tile("a", (0.1, 0.55, 0.8, 0.35), "#38bdf8")),
                           ("b", _tile("b", (0.1, 0.12, 0.8, 0.35), "#000000"))])
    root = etree.fromstring(out)
    top = [g.get("id") for g in root if g.get("id")]
    assert "panel_a" in top and "panel_b" in top


def test_composed_root_carries_figure_size_in_points():
    out = compose(_fig(), [("a", _tile("a", (0.2, 0.2, 0.6, 0.6), "#000000"))])
    root = etree.fromstring(out)
    assert root.get("width") == f"{100.0 / 25.4 * 72:.6f}pt"
    assert root.get("viewBox") == (
        f"0 0 {100.0 / 25.4 * 72:.6f} {60.0 / 25.4 * 72:.6f}")


def test_annotation_elements_are_appended_after_tiles():
    ann = etree.Element("{http://www.w3.org/2000/svg}text")
    ann.set("id", "letter_a")
    ann.text = "A"
    out = compose(_fig(), [("a", _tile("a", (0.2, 0.2, 0.6, 0.6), "#000000"))],
                  annotation_elements=[ann])
    root = etree.fromstring(out)
    ids = [c.get("id") for c in root]
    assert ids[-1] == "annotations"
    assert b">A<" in out


def test_compose_with_no_tiles_still_emits_a_valid_document():
    out = compose(_fig(), [])
    root = etree.fromstring(out)
    assert root.tag.endswith("svg")


def test_namespace_ids_does_not_touch_text_content():
    svg = (b'<svg xmlns="http://www.w3.org/2000/svg" '
           b'xmlns:xlink="http://www.w3.org/1999/xlink">'
           b'<defs><marker id="a"/></defs>'
           b'<text>f(#a)</text>'
           b'</svg>')
    root = namespace_ids(svg, "p")
    marker = root.find(".//{http://www.w3.org/2000/svg}marker")
    assert marker.get("id") == "p__a"
    text = root.find(".//{http://www.w3.org/2000/svg}text")
    assert text.text == "f(#a)", (
        f"text content must be untouched, got {text.text!r}")


def test_namespace_ids_rewrites_clip_path_and_xlink_href():
    svg = (b'<svg xmlns="http://www.w3.org/2000/svg" '
           b'xmlns:xlink="http://www.w3.org/1999/xlink">'
           b'<defs><clipPath id="p1"/><marker id="m1"/></defs>'
           b'<g clip-path="url(#p1)"><use xlink:href="#m1"/></g>'
           b'</svg>')
    root = namespace_ids(svg, "t")
    g = root.find(".//{http://www.w3.org/2000/svg}g")
    assert g.get("clip-path") == "url(#t__p1)"
    use = root.find(".//{http://www.w3.org/2000/svg}use")
    assert use.get(f"{{{XLINK_NS}}}href") == "#t__m1"


def test_namespace_ids_rewrites_url_ref_embedded_in_style_attribute():
    svg = (b'<svg xmlns="http://www.w3.org/2000/svg">'
           b'<defs><marker id="arrow"/></defs>'
           b'<path style="fill:none;marker-end:url(#arrow);stroke:#000"/>'
           b'</svg>')
    root = namespace_ids(svg, "s")
    path = root.find(".//{http://www.w3.org/2000/svg}path")
    assert path.get("style") == "fill:none;marker-end:url(#s__arrow);stroke:#000"


def test_namespace_ids_no_double_prefixing_with_prefix_colliding_ids():
    svg = (b'<svg xmlns="http://www.w3.org/2000/svg" '
           b'xmlns:xlink="http://www.w3.org/1999/xlink">'
           b'<defs>'
           b'<marker id="a"/><marker id="a1"/>'
           b'<marker id="line2d_1"/><marker id="line2d_11"/>'
           b'</defs>'
           b'<use xlink:href="#a"/><use xlink:href="#a1"/>'
           b'<use xlink:href="#line2d_1"/><use xlink:href="#line2d_11"/>'
           b'</svg>')
    root = namespace_ids(svg, "z")
    ids = {e.get("id") for e in root.iter() if e.get("id")}
    assert ids == {"z__a", "z__a1", "z__line2d_1", "z__line2d_11"}
    hrefs = [e.get(f"{{http://www.w3.org/1999/xlink}}href")
             for e in root.iter() if e.get(f"{{http://www.w3.org/1999/xlink}}href")]
    assert set(hrefs) == {"#z__a", "#z__a1", "#z__line2d_1", "#z__line2d_11"}


def test_compose_has_no_dangling_references_across_two_real_tiles():
    out = compose(_fig(), [("a", _tile("a", (0.1, 0.55, 0.8, 0.35), "#38bdf8")),
                           ("b", _tile("b", (0.1, 0.12, 0.8, 0.35), "#000000"))])
    root = etree.fromstring(out)
    ids = {e.get("id") for e in root.iter() if e.get("id")}

    url_refs = set()
    for e in root.iter():
        for attr, value in e.attrib.items():
            if value is None:
                continue
            for m in re.finditer(r"url\(#([^)]+)\)", value):
                url_refs.add(m.group(1))
            if value.startswith("#"):
                url_refs.add(value[1:])

    dangling = url_refs - ids
    assert not dangling, f"dangling references not defined in the document: {dangling}"
