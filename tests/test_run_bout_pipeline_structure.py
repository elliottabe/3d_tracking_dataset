"""Structural guards on scripts/run_bout.py's stage pipeline.

These exist because of a real, silent regression (2026-08-27): adding the
NaN-robust STAC helpers pasted their `def`s into the MIDDLE of
`process_bout_fly`, which truncated that function after the offsets fit and
left Stage C (STAC), Stage D (model->mm bridge), Stage E (outputs/qc), the overlays and
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
    "Stage D: model->mm bridge",
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


def test_stage_b_locals_are_not_bound_only_inside_stage_a(tree):
    """Names Stage B reads must be bound OUTSIDE the `if not stage_done(kp2d_path)`
    guard, or a resume raises UnboundLocalError.

    Real regression (2026-08-29): the Stage-B mask-agreement gate (commit
    0aaccec) reads `centroids`, but `centroids = masks_dict["centroids"]` was
    bound inside the Stage A block. Every fresh run was fine -- Stage A runs, so
    the name exists -- and every fully-DONE bout was fine too, because
    `bout_complete` skips the whole fly. It only fires on the path where kp2d.npz
    exists and kp3d.npz does not, which is exactly the re-run needed after
    invalidating triangulation that predates the outlier gate:

        UnboundLocalError: cannot access local variable 'centroids'

    That path had never been exercised, so nothing caught it.
    """
    fn = _functions(tree)["process_bout_fly"]

    def _guard_is_stage_a(node):
        return (isinstance(node, ast.If)
                and "kp2d_path" in ast.dump(node.test)
                and "stage_done" in ast.dump(node.test))

    stage_a = [n for n in ast.walk(fn) if _guard_is_stage_a(n)]
    assert stage_a, "could not locate the Stage A guard in process_bout_fly"

    def _names(target):
        """All Names bound by an assignment target, incl. tuple/list unpacking.

        `kp2d, conf = z["kp2d"], z["conf"]` rebinds kp2d AFTER Stage A; missing
        tuple targets here made this test flag kp2d as a false positive.
        """
        out = set()
        for n in ast.walk(target):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
        return out

    def _assigned(node):
        out = set()
        for n in ast.walk(node):
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    out |= _names(t)
            elif isinstance(n, (ast.For, ast.withitem)):
                tgt = getattr(n, "target", None) or getattr(n, "optional_vars", None)
                if tgt is not None:
                    out |= _names(tgt)
        return out

    bound_in_stage_a = set()
    for blk in stage_a:
        bound_in_stage_a |= _assigned(blk)

    stage_a_nodes = {id(x) for blk in stage_a for x in ast.walk(blk)}
    bound_outside = set()
    for n in ast.walk(fn):
        if id(n) in stage_a_nodes:
            continue
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.withitem)):
            bound_outside |= _assigned(n)

    # names read by Stage B onward (everything after the Stage A guard ends)
    a_end = max(getattr(n, "lineno", 0) for blk in stage_a for n in ast.walk(blk))
    read_later = {n.id for n in ast.walk(fn)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                  and getattr(n, "lineno", 0) > a_end}

    only_in_stage_a = (bound_in_stage_a & read_later) - bound_outside
    assert not only_in_stage_a, (
        "these names are bound ONLY inside the Stage A block but read by a later "
        f"stage, so resuming with kp2d.npz present raises UnboundLocalError: "
        f"{sorted(only_in_stage_a)}")


def test_stage_b_gate_signature_changes_with_every_gate():
    """Stage B's gates only run when kp3d.npz is (re)computed, so a resumed run
    with a stale file silently ignores the current config. The signature is what
    makes that detectable, so it must actually move when any gate moves.
    Measured cause: on Session0 bout 28, fly1 re-ran and got rigid-repair while
    fly0 kept a pre-gate kp3d.npz and was never reprocessed, with no warning.
    """
    import importlib.util
    from omegaconf import OmegaConf
    spec = importlib.util.spec_from_file_location("_rb", str(RUN_BOUT))
    # run_bout imports heavy deps at module level; read the function out instead
    src = RUN_BOUT.read_text()
    ns = {}
    import ast as _ast
    tree = _ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, _ast.FunctionDef) and n.name == "stage_b_gate_signature")
    exec(compile(_ast.Module(body=[fn], type_ignores=[]), "<sig>", "exec"),
         {"json": __import__("json")}, ns)
    sig = ns["stage_b_gate_signature"]

    base = OmegaConf.create({
        "detector": {"conf_thresh": 0.3, "view_conf_thresh": 0.6,
                     "reproj_resid_px": 10.0},
        "masks": {"kp_mask_agree_fly_lengths": 3.0, "min_views": 3},
        "wing_collapse": {"enabled": False},
        "rigid_repair": {"enabled": False},
    })
    b = sig(base)
    assert isinstance(b, str) and "wing_collapse" in b

    variants = {
        "conf": {"detector": {"conf_thresh": 0.4}},
        "resid": {"detector": {"reproj_resid_px": 12.0}},
        "maskagree": {"masks": {"kp_mask_agree_fly_lengths": 4.0}},
        "collapse_on": {"wing_collapse": {"enabled": True, "abs_floor": 0.6,
                                          "rel_frac": 0.35, "min_views_kept": 4}},
        "repair_on": {"rigid_repair": {"enabled": True, "edges_containing": ["Wing"],
                                       "rel_tol": 0.5, "max_gap": 5, "max_cv": 0.2,
                                       "min_conf": 0.5, "min_frames": 50}},
    }
    for name, patch in variants.items():
        c = OmegaConf.merge(base, OmegaConf.create(patch))
        assert sig(c) != b, f"signature did not change for {name}"

    # and a gate's PARAMETER must move it too, not just its enabled flag
    on = OmegaConf.merge(base, OmegaConf.create(variants["repair_on"]))
    tweaked = OmegaConf.merge(on, OmegaConf.create(
        {"rigid_repair": {"rel_tol": 0.7}}))
    assert sig(on) != sig(tweaked), "rel_tol change must invalidate kp3d.npz"


def test_run_bout_refuses_a_kp3d_written_under_other_gates():
    """The refusal must name the artifacts to delete and must NOT auto-delete a
    12-minute STAC solve -- same contract as the stac_ik.h5 T-mismatch check."""
    src = RUN_BOUT.read_text()
    # the CALL site, not the def -- an earlier version of this test matched the
    # definition and passed vacuously
    i = src.index("_gate_sig = stage_b_gate_signature(cfg)")
    blk = src[i:i + 2600]
    assert "raise RuntimeError" in blk, "a mismatch must refuse, not warn"
    assert "rm -f" in blk, "the refusal must tell the operator what to delete"
    assert "allow_stale_kp3d" in blk, "there must be a documented escape hatch"
    # it must not silently delete anything itself
    assert "os.remove" not in blk and "shutil.rmtree" not in blk, \
        "must not auto-delete downstream artifacts"
