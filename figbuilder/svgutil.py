"""SVG helpers, chiefly id namespacing.

matplotlib emits document-local ids (`figure_1`, `axes_1`, `line2d_1`,
`C3_0_ad4e761f83`) that are identical across independently rendered figures.
Overlaying two tiles without rewriting them silently corrupts clip paths and
marker definitions, because `url(#...)` resolves to the first match.
"""
from __future__ import annotations

import re

from typing import Set

from lxml import etree

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
NSMAP = {None: SVG_NS, "xlink": XLINK_NS}

SEP = "__"

_URL_REF = re.compile(r"url\(#([^)]+)\)")


def qname(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def _collect_ids(root) -> Set[str]:
    return {e.get("id") for e in root.iter() if e.get("id")}


def namespace_ids(svg: bytes, prefix: str):
    """Return the parsed tile with every id (and reference) prefixed.

    Rewriting happens on the parsed ATTRIBUTE tree, never on serialized
    bytes or element text/tail: a `<text>` label that merely contains a
    substring shaped like a reference (e.g. `f(#a)`) must survive
    untouched, only `id` attributes and genuine `#id` / `url(#id)`
    references in attribute values are candidates. Only ids that are
    actually DEFINED in this tile are rewritten, so a reference to an id
    defined elsewhere (outside this tile) is left alone. Because every
    match is against the full, exact id string (never a substring), there
    is no risk of `line2d_1` corrupting `line2d_11` and no need for a
    longest-first pass.
    """
    root = etree.fromstring(svg)
    ids = _collect_ids(root)
    if not ids:
        return root

    def _prefixed(old: str) -> str:
        return f"{prefix}{SEP}{old}"

    for el in root.iter():
        old_id = el.get("id")
        if old_id is not None:
            el.set("id", _prefixed(old_id))

        for attr, value in list(el.attrib.items()):
            if value is None:
                continue

            # Bare fragment reference, e.g. xlink:href="#m2e9b37ff53".
            if value.startswith("#") and value[1:] in ids:
                el.set(attr, f"#{_prefixed(value[1:])}")
                continue

            # url(#X) occurrences anywhere in the value: covers
            # clip-path="url(#p1)" and embedded refs like a `style` of
            # "...;marker-end:url(#arrow)".
            if "url(#" in value:
                def _sub(m: "re.Match[str]") -> str:
                    target = m.group(1)
                    if target in ids:
                        return f"url(#{_prefixed(target)})"
                    return m.group(0)

                new_value = _URL_REF.sub(_sub, value)
                if new_value != value:
                    el.set(attr, new_value)

    return root
