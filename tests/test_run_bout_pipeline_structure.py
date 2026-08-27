"""Structural guards on scripts/run_bout.py's stage pipeline.

These exist because of a real, silent regression (2026-08-27): adding the
NaN-robust STAC helpers pasted their `def`s into the MIDDLE of
`process_bout_fly`, which truncated that function after the offsets fit and
left Stage C (STAC), Stage D (polish), Stage E (outputs/qc), the overlays and
`mark_done` stranded as unreachable code after `joints_frozen`'s unconditional
`return`. The pipeline then ran to completion, printed no error and exited 0
while producing no stac_ik.h5 for ANY bout -- invisible to every runtime check
we have, because nothing raises when a stage simply never executes.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

RUN_BOUT = Path(__file__).resolve().parents[1] / "scripts" / "run_bout.py"


@pytest.fixture(scope="module")
def tree():
    return ast.parse(RUN_BOUT.read_text())


def _functions(tree):
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def test_no_unreachable_code_after_return():
    """No statement may follow an unconditional return/raise in the same block.

    This is the exact shape of the regression: a stage block that parses fine,
    reads fine, and never runs.
    """
    src = RUN_BOUT.read_text()
    tree = ast.parse(src)
    dead = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for node in ast.walk(fn):
            for field in ("body", "orelse", "finalbody"):
                block = getattr(node, field, None)
                if not isinstance(block, list):
                    continue
                for i, stmt in enumerate(block[:-1]):
                    if isinstance(stmt, (ast.Return, ast.Raise)):
                        nxt = block[i + 1]
                        dead.append(
                            f"{fn.name}: line {nxt.lineno} is unreachable "
                            f"(follows {type(stmt).__name__.lower()} at line {stmt.lineno})")
    assert not dead, "unreachable code:\n  " + "\n  ".join(dead)


@pytest.mark.parametrize("marker", [
    "Stage C: STAC",
    "Stage D: silhouette",
    "Stage E: outputs",
    "stac_ik.h5",
    "mark_done(",
])
def test_every_stage_lives_inside_process_bout_fly(tree, marker):
    """The stages must be in the function the bout loop actually calls."""
    src = RUN_BOUT.read_text()
    fn = _functions(tree)["process_bout_fly"]
    body = ast.get_source_segment(src, fn)
    assert marker in body, (
        f"{marker!r} is not inside process_bout_fly -- it was orphaned out of "
        f"the function body, so the bout loop will never execute it")


def test_joints_frozen_is_only_a_predicate(tree):
    """It returns a bool; it must not have absorbed pipeline stages."""
    fn = _functions(tree)["joints_frozen"]
    assert fn.end_lineno - fn.lineno < 30, (
        f"joints_frozen spans {fn.end_lineno - fn.lineno} lines -- it has "
        f"swallowed code that belongs to another function")


def test_nan_helpers_are_module_level(tree):
    """The STAC NaN helpers must be siblings of process_bout_fly, not nested
    inside it (nesting them there is what truncated it)."""
    top = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("finite_frame_mask", "fill_short_gaps", "contiguous_segments",
                 "_solve_segments_into", "_restore_unsolved_nan", "joints_frozen"):
        assert name in top, f"{name} is not defined at module level"
