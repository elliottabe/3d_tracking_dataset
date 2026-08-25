"""Per-panel cosmetics: spines, tick labels, legend placement.

These options exist because the drawing itself lives in `utils/`, which is
consumed unmodified — so they are applied to the axes afterwards.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from figbuilder.cosmetics import SHARED_SCHEMA, apply_cosmetics  # noqa: E402


@pytest.fixture
def ax():
    fig = plt.figure(figsize=(3, 2))
    a = fig.add_subplot(111)
    a.plot([0, 1, 2], [0, 1, 0], label="left")
    a.plot([0, 1, 2], [1, 0, 1], label="right")
    a.legend()
    yield a
    plt.close(fig)


def test_no_spec_changes_nothing(ax):
    before = {k: v.get_visible() for k, v in ax.spines.items()}
    labels = [t.get_text() for t in ax.get_xticklabels()]
    apply_cosmetics(ax, None)
    apply_cosmetics(ax, {})
    assert {k: v.get_visible() for k, v in ax.spines.items()} == before
    assert [t.get_text() for t in ax.get_xticklabels()] == labels
    assert ax.get_legend() is not None


def test_spines_toggle_individually(ax):
    for s in ax.spines.values():
        s.set_visible(True)
    apply_cosmetics(ax, {"spines": {"top": False, "right": False}})
    assert ax.spines["top"].get_visible() is False
    assert ax.spines["right"].get_visible() is False
    # Untouched names keep their state rather than being reset.
    assert ax.spines["left"].get_visible() is True
    assert ax.spines["bottom"].get_visible() is True


def test_spines_can_be_turned_back_on(ax):
    ax.spines["top"].set_visible(False)
    apply_cosmetics(ax, {"spines": {"top": True}})
    assert ax.spines["top"].get_visible() is True


def test_unknown_spine_name_is_ignored_not_raised(ax):
    # A polar axes has only a 'polar' spine; a shared schema must not explode
    # when a name is absent for this projection.
    apply_cosmetics(ax, {"spines": {"polar": False, "nonsense": True}})


def test_polar_axes_survives_shared_spine_options():
    fig = plt.figure()
    a = fig.add_subplot(111, projection="polar")
    try:
        apply_cosmetics(a, {"spines": {"top": False, "right": False}})
    finally:
        plt.close(fig)


def test_hide_xticklabels(ax):
    ax.figure.canvas.draw()
    # Guard against passing for the wrong reason: there must be labels to hide.
    assert any(t.get_text() for t in ax.get_xticklabels())
    assert any(t.get_visible() for t in ax.get_xticklabels())
    apply_cosmetics(ax, {"hide_xticklabels": True})
    ax.figure.canvas.draw()
    assert all(not t.get_visible() for t in ax.get_xticklabels())


def test_hide_yticklabels(ax):
    apply_cosmetics(ax, {"hide_yticklabels": True})
    ax.figure.canvas.draw()
    assert all(not t.get_visible() for t in ax.get_yticklabels())


def test_hiding_xticklabels_shrinks_the_ink_box(ax):
    """The point of the option: a shorter tile stops overflowing its neighbour."""
    from figbuilder.render import _ink_box

    ax.figure.canvas.draw()
    before = _ink_box(ax.figure, ax)
    apply_cosmetics(ax, {"hide_xticklabels": True})
    ax.figure.canvas.draw()
    after = _ink_box(ax.figure, ax)
    # y0 rises and the height shrinks: the labels below the axes are gone.
    assert after[1] > before[1]
    assert after[3] < before[3]


def test_legend_reposition_keeps_the_same_legend_object(ax):
    """Repositioning must not rebuild: utils' colour-matched text legend would
    lose its styling and regain handles if we called ax.legend() again."""
    leg = ax.get_legend()
    texts = [t.get_text() for t in leg.get_texts()]
    apply_cosmetics(ax, {"legend": {"loc": "lower left"}})
    assert ax.get_legend() is leg
    assert [t.get_text() for t in ax.get_legend().get_texts()] == texts


def test_legend_bbox_moves_it(ax):
    ax.figure.canvas.draw()
    before = ax.get_legend().get_window_extent().x0
    apply_cosmetics(ax, {"legend": {"loc": "lower left",
                                    "bbox_to_anchor": [1.05, 0.0]}})
    ax.figure.canvas.draw()
    assert ax.get_legend().get_window_extent().x0 > before


def test_legend_hide_removes_it(ax):
    apply_cosmetics(ax, {"legend": {"hide": True}})
    assert ax.get_legend() is None


def test_legend_options_on_axes_without_a_legend_do_not_raise():
    fig = plt.figure()
    a = fig.add_subplot(111)
    a.plot([0, 1], [0, 1])
    try:
        apply_cosmetics(a, {"legend": {"loc": "upper left", "hide": True}})
    finally:
        plt.close(fig)


def test_shared_schema_is_merged_into_every_panel_type():
    from figbuilder.panels.base import list_panel_types

    for t in list_panel_types():
        props = t["schema"]["properties"]
        for key in SHARED_SCHEMA:
            assert key in props, f"{t['id']} is missing shared option {key!r}"


def test_panel_own_options_win_over_shared_ones():
    """A panel type that declares a name itself keeps its own definition."""
    from figbuilder.panels.base import PanelType, full_schema

    class Fake(PanelType):
        id = "fake"
        schema = {"type": "object",
                  "properties": {"legend": {"type": "string", "title": "mine"}}}

    merged = full_schema(Fake())
    assert merged["properties"]["legend"]["title"] == "mine"
    assert "spines" in merged["properties"]


def test_default_style_despines_top_and_right():
    from figbuilder.style import DEFAULT_RCPARAMS

    assert DEFAULT_RCPARAMS["axes.spines.top"] is False
    assert DEFAULT_RCPARAMS["axes.spines.right"] is False
