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


def qname(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def _collect_ids(root) -> Set[str]:
    return {e.get("id") for e in root.iter() if e.get("id")}


def namespace_ids(svg: bytes, prefix: str):
    """Return the parsed tile with every id (and reference) prefixed."""
    root = etree.fromstring(svg)
    ids = _collect_ids(root)
    if not ids:
        return root
    blob = etree.tostring(root)
    # Longest first, so `line2d_1` cannot corrupt `line2d_11`.
    for old in sorted(ids, key=len, reverse=True):
        new = f"{prefix}{SEP}{old}"
        blob = blob.replace(f'id="{old}"'.encode(), f'id="{new}"'.encode())
        blob = blob.replace(f'#{old}"'.encode(), f'#{new}"'.encode())
        blob = blob.replace(f'#{old})'.encode(), f'#{new})'.encode())
    return etree.fromstring(blob)
