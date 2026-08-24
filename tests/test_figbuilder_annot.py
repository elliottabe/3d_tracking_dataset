"""Annotation layer emission."""
from __future__ import annotations

import pytest
from lxml import etree

from figbuilder.annot import ANNOTATION_KINDS, emit_annotations
from figbuilder.figure import FigureSpec, PanelSpec


def _fig():
    return FigureSpec(width_mm=100.0, height_mm=50.0, panels=[
        PanelSpec(id="wing", type="line", rect=(0.1, 0.5, 0.8, 0.4)),
    ])


def test_text_annotation_becomes_real_svg_text():
    els = emit_annotations(_fig(), [
        {"id": "t1", "kind": "text", "pos_mm": [5.0, 10.0], "text": "C",
         "style": {"font_size_pt": 8, "weight": "bold"}}])
    assert len(els) == 1
    assert etree.QName(els[0]).localname == "text"
    assert els[0].text == "C"
    assert "font-weight:700" in els[0].get("style")


def test_y_is_flipped_from_figure_space_to_svg_space():
    """Figure y grows upward; SVG y grows downward."""
    els = emit_annotations(_fig(), [
        {"id": "t", "kind": "text", "pos_mm": [0.0, 0.0], "text": "x"}])
    # y=0 mm from the figure bottom -> y = height in pt
    assert float(els[0].get("y")) == pytest.approx(50.0 * 72 / 25.4)
    # ...and it is written at full precision, not 6 significant figures.
    assert els[0].get("y") == "141.732283"


def test_num_formatter_trims_trailing_zeros_and_keeps_six_decimals():
    from figbuilder.annot import _num
    assert _num(0.8) == "0.8"
    assert _num(8) == "8"
    assert _num(141.73228346456693) == "141.732283"
    assert _num(-0.0) == "0"   # must match JS toFixed, which has no -0


def test_panel_parented_annotation_offsets_from_that_panel_corner():
    els = emit_annotations(_fig(), [
        {"id": "L", "kind": "text", "parent": "wing", "pos_mm": [0.0, 0.0],
         "text": "A"}])
    # panel rect x=0.1 of 100 mm -> 10 mm -> in pt
    assert float(els[0].get("x")) == pytest.approx(10.0 * 72 / 25.4)


def test_line_and_rect_and_ellipse_emit_expected_tags():
    els = emit_annotations(_fig(), [
        {"id": "l", "kind": "line", "pos_mm": [1, 1], "to_mm": [9, 9]},
        {"id": "r", "kind": "rect", "pos_mm": [2, 2], "size_mm": [5, 4]},
        {"id": "e", "kind": "ellipse", "pos_mm": [3, 3], "size_mm": [6, 2]},
    ])
    assert [etree.QName(e).localname for e in els] == ["line", "rect", "ellipse"]


def test_arrow_emits_a_path_with_a_marker_reference():
    els = emit_annotations(_fig(), [
        {"id": "a", "kind": "arrow", "pos_mm": [1, 1], "to_mm": [9, 5]}])
    # A <defs> with the arrowhead marker is prepended (index 0) whenever an
    # arrow is present, so the path itself is found by tag, not by position.
    paths = [e for e in els if etree.QName(e).localname == "path"]
    assert len(paths) == 1
    assert "marker-end" in paths[0].get("style", "")


def test_scalebar_emits_a_group_with_a_line_and_a_label():
    els = emit_annotations(_fig(), [
        {"id": "sb", "kind": "scalebar", "pos_mm": [2, 2], "length_mm": 10.0,
         "label": "1 mm"}])
    kids = [etree.QName(c).localname for c in els[0]]
    assert "line" in kids and "text" in kids


def test_every_declared_kind_is_emittable():
    for kind in ANNOTATION_KINDS:
        ann = {"id": "x", "kind": kind, "pos_mm": [1, 1], "to_mm": [5, 5],
               "size_mm": [3, 3], "text": "t", "label": "l", "length_mm": 4.0}
        assert emit_annotations(_fig(), [ann]), f"{kind} emitted nothing"


def test_unknown_kind_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown annotation kind"):
        emit_annotations(_fig(), [{"id": "z", "kind": "hologram"}])


def test_annotations_carry_their_id_for_reediting():
    els = emit_annotations(_fig(), [
        {"id": "letter_wing", "kind": "text", "pos_mm": [1, 1], "text": "A"}])
    assert els[0].get("id") == "letter_wing"
