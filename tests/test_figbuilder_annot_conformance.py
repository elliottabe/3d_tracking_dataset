"""The Python and TypeScript annotation emitters must agree.

The emitter is duplicated on purpose (60 fps live editing in the browser,
headless export in Python). This test is what makes that duplication safe.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from lxml import etree

from figbuilder.annot import emit_annotations
from figbuilder.figure import FigureSpec, PanelSpec

FIXTURES = Path("web/src/annot/fixtures.json")


def _fig() -> FigureSpec:
    return FigureSpec(width_mm=100.0, height_mm=50.0, panels=[
        PanelSpec(id="wing", type="line", rect=(0.1, 0.5, 0.8, 0.4))])


def _normalize(svg: str) -> str:
    """Collapse whitespace and round numbers so the two emitters can be compared."""
    svg = re.sub(r"(\d+\.\d{3})\d+", r"\1", svg)
    return re.sub(r"\s+", " ", svg).strip()


def test_fixture_file_covers_every_annotation_kind():
    from figbuilder.annot import ANNOTATION_KINDS
    anns = json.loads(FIXTURES.read_text())
    assert {a["kind"] for a in anns} == set(ANNOTATION_KINDS)


@pytest.mark.skipif(not (Path("web/node_modules").exists()),
                    reason="web deps not installed (run `npm ci` in web/)")
def test_python_and_typescript_emitters_agree():
    anns = json.loads(FIXTURES.read_text())
    py = _normalize("".join(
        etree.tostring(e, encoding="unicode") for e in emit_annotations(_fig(), anns)))
    ts = _normalize(subprocess.run(
        ["npx", "tsx", "src/annot/emitCli.ts"], cwd="web",
        capture_output=True, text=True, check=True).stdout)
    assert py == ts
