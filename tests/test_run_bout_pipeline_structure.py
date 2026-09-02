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
PIPELINE_YAML = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"


def _pipeline_wing_mask_fit():
    """The shipped `wing_mask_fit:` block, read straight out of configs/pipeline.yaml.

    Read as raw YAML rather than composed by hydra so this stays a cheap,
    import-free structure test; the block has no ${} interpolations.
    """
    import yaml
    cfg = yaml.safe_load(PIPELINE_YAML.read_text())
    assert "wing_mask_fit" in cfg, (
        "configs/pipeline.yaml has no `wing_mask_fit:` block -- the opt-in "
        "wing-pitch refinement stage is unreachable from the shipped config")
    return cfg["wing_mask_fit"]


# The shipped block with the opt-in flag flipped: what a real enabling run sees.
WING_MASK_FIT_ON = {**_pipeline_wing_mask_fit(), "enabled": True}


def _run_bout_helpers(*names):
    """exec the named module-level functions out of run_bout.py, without
    importing it, sharing one globals dict so they can call each other.

    run_bout imports jax/mujoco/hydra/stac_mjx at module level; these guards
    must stay cheap and AST-based (same trick as test_stage_b_gate_signature_*).
    """
    import ast as _ast
    import json as _json
    from pathlib import Path as _P
    import numpy as _np
    body = [n for n in _ast.parse(RUN_BOUT.read_text()).body
            if isinstance(n, _ast.FunctionDef) and n.name in names]
    got = {n.name for n in body}
    assert got == set(names), f"run_bout.py is missing {sorted(set(names) - got)}"
    # run_bout imports these from jarvis_jax.tracking.resume at module level;
    # `stage_done` is literally "exists and non-empty", so reproduce it here
    # rather than importing the package for an AST-level test.
    import os as _os
    g = {"json": _json, "np": _np,
         "stage_done": lambda p: _os.path.exists(p) and _os.path.getsize(p) > 0,
         "atomic_save_json": lambda p, o: _P(p).write_text(_json.dumps(o))}
    exec(compile(_ast.Module(body=body, type_ignores=[]), "<rb>", "exec"), g)
    return tuple(g[n] for n in names)


def _run_bout_helper(name):
    return _run_bout_helpers(name)[0]


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
    "[wing-mask-fit]",
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
    # the wing-mask fit is covered by the same signature, so its artifact must
    # be named too -- deleting everything else and leaving qpos_wingfit.npz
    # behind would silently reuse a fit made under the old gates.
    assert "qpos_wingfit.npz" in blk, \
        "the refusal must name qpos_wingfit.npz among the artifacts to delete"


# ---------------------------------------------------------------------------
# Task 6: the opt-in post-STAC wing-pitch refinement against the SAM masks
# ---------------------------------------------------------------------------

def test_wing_mask_fit_is_off_and_absent_is_a_strict_no_op():
    """`wing_mask_fit` is opt-in, and a config that has never heard of it must
    behave exactly as before -- no crash, no stage, no signature change."""
    from omegaconf import OmegaConf
    enabled = _run_bout_helper("wing_mask_fit_enabled")

    assert _pipeline_wing_mask_fit()["enabled"] is False, \
        "the shipped default must be OFF -- this stage is opt-in"

    # absent block: the pre-feature config
    assert enabled(OmegaConf.create({"detector": {}})) is False
    assert enabled({}) is False
    # present but off
    assert enabled(OmegaConf.create({"wing_mask_fit": {"enabled": False}})) is False
    # explicit null block (hydra `wing_mask_fit: null`)
    assert enabled(OmegaConf.create({"wing_mask_fit": None})) is False
    # on
    assert enabled(OmegaConf.create({"wing_mask_fit": WING_MASK_FIT_ON})) is True


def test_wing_mask_fit_never_enters_the_stage_b_gate_signature():
    """The Stage-B signature is "every setting that changes what kp3d.npz
    contains", it is stored INSIDE kp3d.npz, and a mismatch REFUSES the bout --
    telling the operator to delete kp3d.npz, stac_ik.h5 and the rest.

    The wing fit rewrites qpos AFTER the bridges and provably cannot alter
    kp3d.npz or stac_ik.h5, so enrolling it charged a re-triangulation plus a
    12-minute STAC solve per bout-fly per sweep point (~5.2 h on the 13-bout
    benchmark) for nothing -- and the only escape, allow_stale_kp3d, disables
    the comparison altogether, leaving the fit with NO staleness protection.
    Its provenance lives in its own artifact instead (wing_mask_fit_signature).
    Toggling or retuning the block must therefore leave this string untouched.
    """
    from omegaconf import OmegaConf
    sig = _run_bout_helper("stage_b_gate_signature")
    base = OmegaConf.create({
        "detector": {"conf_thresh": 0.3, "view_conf_thresh": 0.6,
                     "reproj_resid_px": 10.0},
        "masks": {"kp_mask_agree_fly_lengths": 3.0, "min_views": 3},
        "wing_collapse": {"enabled": False},
        "rigid_repair": {"enabled": False},
    })
    absent = sig(base)
    assert "wing_mask_fit" not in absent
    for label, patch in (("off", {"enabled": False}),
                         ("on", WING_MASK_FIT_ON),
                         ("retuned", {**WING_MASK_FIT_ON, "coverage_weight": 0.03})):
        got = sig(OmegaConf.merge(base, OmegaConf.create({"wing_mask_fit": patch})))
        assert got == absent, (
            f"wing_mask_fit={label} moved the Stage-B gate signature; that "
            f"refuses every kp3d.npz on disk and forces a STAC re-solve for a "
            f"stage that cannot change either artifact")


def test_every_wing_mask_fit_yaml_key_reaches_the_refinement():
    """No YAML key may be silently dropped, and none may fall back to a module
    default.

    `refine_wing_pitch`'s own defaults are NOT the measured-best configuration
    -- `huber_delta` defaults to 0.0, at which the coverage term is an
    unrobustified L2 that measurably degrades the fit (Task 5). So every knob in
    the block is passed explicitly, and this pins that: the YAML keys must
    partition exactly into (the refine_wing_pitch kwargs) + (the keys the caller
    consumes itself) + enabled.
    """
    import ast as _ast
    import importlib.util

    wf = _pipeline_wing_mask_fit()
    refine_kwargs = _run_bout_helper("wing_mask_fit_refine_kwargs")(wf)

    # 1. every kwarg we pass is a real parameter of refine_wing_pitch (parsed
    #    from source: importing it would pull in jax).
    origin = importlib.util.find_spec(
        "jarvis_jax.tracking.wing_mask_refine").origin
    fn = next(n for n in _ast.parse(Path(origin).read_text()).body
              if isinstance(n, _ast.FunctionDef) and n.name == "refine_wing_pitch")
    params = {a.arg for a in fn.args.kwonlyargs} | {a.arg for a in fn.args.args}
    unknown = sorted(set(refine_kwargs) - params)
    assert not unknown, f"not parameters of refine_wing_pitch: {unknown}"

    # 2. the values reaching it are the YAML's, not the module's defaults.
    for k, v in refine_kwargs.items():
        assert v == wf[k], f"{k}: passed {v!r}, YAML says {wf[k]!r}"
    assert refine_kwargs["huber_delta"] == 8.0, (
        "huber_delta must reach the module as the measured 8.0; its default "
        "0.0 makes the coverage term an unrobustified L2")
    assert refine_kwargs["coverage_normalize"] is True

    # 3. the keys the caller consumes itself (they configure the SDF stack, the
    #    body-vertex basis and the per-frame camera gate, not the refiner) must
    #    each be read in the stage body -- nothing may be quietly ignored.
    src = RUN_BOUT.read_text()
    stage = _ast.get_source_segment(src, next(
        n for n in _ast.parse(src).body
        if isinstance(n, _ast.FunctionDef) and n.name == "wing_mask_fit_bout"))
    caller_keys = {"out_hw", "bbox_margin", "body_vertex_stride",
                   "min_present_cameras", "exclude_cameras",
                   # the validity gate: the stage builds camera_ok/frame_keep
                   # itself and hands the refiner an array, not these knobs
                   "gate_enabled", "gate_min_area_frac",
                   "gate_min_body_inside", "gate_area_ref_pct"}
    for k in caller_keys:
        assert f'"{k}"' in stage or f"'{k}'" in stage, \
            f"wing_mask_fit.{k} is in the YAML but never read by the stage"

    # 4. and the partition is exact -- a new YAML key cannot be added without
    #    being wired somewhere.
    assert set(wf) == set(refine_kwargs) | caller_keys | {"enabled"}, (
        f"wing_mask_fit YAML keys are not fully wired: "
        f"unwired={sorted(set(wf) - set(refine_kwargs) - caller_keys - {'enabled'})}, "
        f"wired-but-absent-from-YAML="
        f"{sorted((set(refine_kwargs) | caller_keys) - set(wf))}")


def test_wing_mask_fit_runs_after_the_bridges_and_reuses_them():
    """ORDER: the refinement maps model units to mm with the per-frame bridge,
    so it cannot run before `compute_bridges`. And it must REUSE those bridges
    rather than triggering a refit."""
    import ast as _ast
    src = RUN_BOUT.read_text()
    fn = next(n for n in _ast.parse(src).body
              if isinstance(n, _ast.FunctionDef) and n.name == "process_bout_fly")
    body = _ast.get_source_segment(src, fn)
    i_bridge = body.index("compute_bridges(")
    i_fit = body.index("[wing-mask-fit]")
    i_outputs = body.index("Stage E: outputs")
    assert i_bridge < i_fit < i_outputs, (
        "the wing-mask fit must sit between compute_bridges and Stage E "
        f"(compute_bridges@{i_bridge}, fit@{i_fit}, Stage E@{i_outputs})")
    assert body.count("compute_bridges(") == 1, \
        "the bridges must be reused, not refitted after the wing fit"

    # and a stale resumed qpos_wingfit.npz must be refused, not silently fed to
    # Stage E -- same contract as the stac_ik.h5 T-mismatch check.
    blk = body[i_fit - 3000:i_outputs]
    assert "qpos_wingfit.npz qpos T=" in blk and "raise RuntimeError" in blk, \
        "the wing-fit stage must refuse a T-mismatched resumed artifact"


def _fake_refine(q, delta, honour_present, **kw):
    """Stand-in for `refine_wing_pitch`, faking BOTH refiners and the gate.

    `honour_present=True` is `param_mode='free'`, whose frame gate lives on the
    parameter update; False is `spline`/`lowpass`, where a no-evidence frame is
    interpolated and MOVES. `frame_keep` is honoured in both, because that is
    the contract the real module now guarantees regardless of mode -- and the
    only reason the gate is worth having in a band-limited mode at all.
    """
    import numpy as np
    q = np.array(q, np.float32)
    d = delta * (np.asarray(kw["present"], bool).any(axis=1)[:, None]
                 if honour_present else 1.0)
    out = np.clip(q + d, np.asarray(kw["lb"]), np.asarray(kw["ub"]))
    keep = kw.get("frame_keep")
    if keep is not None:
        out = np.where(np.asarray(keep, bool)[:, None], out, q)
    lim = kw.get("max_dpitch_deg")
    if lim is not None:
        dd = np.clip(out - q, -np.deg2rad(float(lim)), np.deg2rad(float(lim)))
        out = q + dd
    return out


class _Rec(dict):
    """A recorder that stands in for one jarvis_jax entry point."""

    def __init__(self, ret):
        super().__init__()
        self._ret = ret

    def __call__(self, *a, **kw):
        self["args"], self["kwargs"] = a, kw
        return self._ret(*a, **kw) if callable(self._ret) else self._ret


def _stub_wing_mask_fit(monkeypatch, n_frames, cameras, honour_present=True,
                        limits=None):
    """Install fake jarvis_jax/mujoco entry points for wing_mask_fit_bout.

    Only the WIRING is under test here. The real `refine_wing_pitch` runs
    `mjx.kinematics` on an 87-joint model, measured at 1739 s to COMPILE on the
    XLA CPU backend against 6.4 s on an L40S (Task 5), so calling it for real is
    a GPU-only Task-7 concern -- but the config plumbing, the camera axes, the
    per-frame gating and the stats are all exercised for real below.

    `honour_present` picks WHICH refiner is being faked, and the two are not
    interchangeable. True reproduces `param_mode='free'`, whose frame gate lives
    on the parameter update, so a frame with no present camera comes back at
    q_init. False reproduces `spline`/`lowpass`, where one knot spans many frames
    and a no-evidence frame is interpolated -- it moves. The first version of
    this stub ignored `present` unconditionally while the tests asserted the
    free-mode counts, so it was silently testing the OLD predicate rather than
    either refiner's behaviour.
    """
    import sys
    import types
    import numpy as np
    # Resolved BEFORE the monkeypatch below replaces the module in sys.modules:
    # a lazy `from ... import` inside the stub would re-enter the stub itself.
    # `mask_quality` is pure numpy, so importing it keeps this guard cheap.
    from jarvis_jax.tracking.mask_quality import wing_fit_validity_gate

    C = len(cameras)
    rec = {}

    def _sdf(masks, valid, *, out_hw, bbox_margin):
        rec["sdf"] = {"valid": np.array(valid, bool, copy=True),
                      "out_hw": out_hw, "bbox_margin": bbox_margin}
        T = masks.shape[0]
        return (np.zeros((T, C) + tuple(out_hw), np.float32),
                np.ones((T, C, 2), np.float32),
                np.zeros((T, C, 2), np.float32),
                np.array(valid, bool, copy=True))       # present == valid

    rec["sdf_fn"] = _sdf
    # DELIBERATE: joint 0 is wing_pitch_RIGHT at qpos 9 and joint 1 is
    # wing_pitch_LEFT at qpos 12, i.e. left comes SECOND in address order. A
    # stats read that takes np.flatnonzero(opt_mask) positionally instead of
    # resolving each joint by name therefore reports the two wings swapped, and
    # the assertions below catch it.
    fake_m = types.SimpleNamespace(jnt_qposadr=np.array([9, 12], np.int32))
    delta = np.zeros(14, np.float32)
    delta[9], delta[12] = 0.5, 0.25          # right +0.5 rad, left +0.25 rad
    stubs = {
        "jarvis_jax.tracking.mask_sdf": {"sdf_stack_from_masks": _sdf},
        "jarvis_jax.tracking.fk": {
            "load_anatomy": lambda xml, npz: {"m": fake_m, "xml": xml, "npz": npz},
            "make_fk_repose": lambda anat: "FK"},
        "jarvis_jax.tracking.appendage_dof": {
            "appendage_vertex_indices": _Rec(np.arange(4, dtype=np.int32))},
        # The gate's two entry points. `body_uv_track` fakes the projection
        # (the real one FKs an 87-joint model); `wing_fit_validity_gate` is the
        # REAL function, fed the fixture's masks, so the wiring under test is
        # the wiring that ships -- only the FK is stubbed.
        "jarvis_jax.tracking.mask_quality": {
            "wing_fit_validity_gate": _Rec(wing_fit_validity_gate)},
        "jarvis_jax.tracking.wing_mask_refine": {
            "affine_cameras_by_name": _Rec(
                (np.zeros((C, 2, 3), np.float32), np.zeros((C, 2), np.float32))),
            "body_uv_track": _Rec(
                lambda q, *a, **k: np.zeros((len(q), C, 4, 2), np.float32)),
            "body_vertex_indices": _Rec(np.arange(8, dtype=np.int32)),
            "qpos_limits": lambda m: (
                np.full(14, -np.inf, np.float32) if limits is None
                else np.full(14, limits[0], np.float32),
                np.full(14, np.inf, np.float32) if limits is None
                else np.full(14, limits[1], np.float32)),
            "wing_pitch_dof_mask": lambda m: np.isin(np.arange(14), [9, 12]),
            "refine_wing_pitch": _Rec(
                lambda q, **kw: _fake_refine(q, delta, honour_present, **kw))},
        "mujoco": {"mjtObj": types.SimpleNamespace(mjOBJ_JOINT=3),
                   "mj_name2id": lambda m, obj, nm: {"wing_pitch_left": 1,
                                                     "wing_pitch_right": 0}[nm]},
    }
    for name, attrs in stubs.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
            if isinstance(v, _Rec):
                rec[k] = v
        monkeypatch.setitem(sys.modules, name, mod)
    return rec


def _wing_fit_fixture(n_frames=5):
    """(cfg, masks_dict, qpos, bridge arrays) for wing_mask_fit_bout."""
    import numpy as np
    from omegaconf import OmegaConf
    cameras = ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855", "Cam2012857"]
    cfg = OmegaConf.create({
        "recording": {"calib_dir": "/nonexistent/calib", "cameras": cameras},
        "ik": {"xml": "/nonexistent/model.xml", "mesh_npz": "/nonexistent/mesh.npz",
               "mesh_subset": "fps_300"},
        "wing_mask_fit": WING_MASK_FIT_ON,
    })
    # Masks are all-TRUE, not all-false: the validity gate is pose-aware and
    # asks how much of the projected body lands inside the mask, so an empty
    # mask is (correctly) rejected as badly tracked and nothing downstream of
    # the gate would ever be exercised. The stubbed `body_uv_track` projects to
    # (0, 0), which is inside this mask.
    masks_dict = {"masks": np.ones((n_frames, len(cameras), 4, 4), bool),
                  "valid": np.ones((n_frames, len(cameras)), bool),
                  "cameras": list(cameras), "T": n_frames, "C": len(cameras)}
    qpos = np.zeros((n_frames, 14), np.float32)
    bs = np.ones(n_frames, np.float32)
    bR = np.broadcast_to(np.eye(3, dtype=np.float32), (n_frames, 3, 3)).copy()
    bt = np.zeros((n_frames, 3), np.float32)
    bok = np.ones(n_frames, bool)
    bok[0] = False                       # frame 0: STAC could not solve it
    qpos[1] = np.nan                     # frame 1: non-finite pose, bridge fine
    return cfg, cameras, masks_dict, qpos, bs, bR, bt, bok


def test_wing_mask_fit_bout_forwards_every_config_knob(monkeypatch):
    """Behavioural counterpart to the structural key test: the YAML values must
    actually ARRIVE at refine_wing_pitch / the SDF stack / the body basis, not
    merely be assembled into a dict, and the module's own defaults must not win.
    """
    import numpy as np
    cfg, cameras, masks_dict, qpos, bs, bR, bt, bok = _wing_fit_fixture()
    rec = _stub_wing_mask_fit(monkeypatch, len(qpos), cameras)
    fit = _run_bout_helpers("wing_mask_fit_bout", "wing_mask_fit_refine_kwargs")[0]

    q_ref, stats = fit(cfg, qpos, bs, bR, bt, bok, masks_dict, cameras)

    kw = rec["refine_wing_pitch"]["kwargs"]
    assert kw["containment_weight"] == 0.3
    assert kw["coverage_weight"] == 0.3
    assert kw["coverage_normalize"] is True
    assert kw["huber_delta"] == 8.0, "the module default 0.0 must not win"
    assert kw["smooth_weight"] == 0.005
    assert kw["limit_weight"] == 10.0
    assert kw["n_steps"] == 300
    assert kw["lr"] == 0.01
    assert kw["frame_chunk"] == 256, "the module default 64 must not win"
    assert kw["n_target_points"] == 128
    assert kw["dilate_px"] == 8, "the module default 3 must not win"
    # the caller-side knobs
    assert rec["sdf"]["out_hw"] == (128, 128) and rec["sdf"]["bbox_margin"] == 0.4
    assert rec["body_vertex_indices"]["kwargs"]["stride"] == 4
    assert rec["appendage_vertex_indices"]["kwargs"] == {
        "subset": "fps_300", "include": ("wing",)}
    # and the pose actually handed on is the refined one -- on the frames that
    # HAVE evidence. Frame 0 has bridge_ok=False, so in `free` mode (the default
    # this fixture uses) its `present` row is all-False and it is returned at
    # q_init; asserting over every finite frame would only pass against a stub
    # that ignores the frame gate.
    fin = np.isfinite(qpos).all(axis=1) & np.asarray(bok, bool)
    assert np.allclose(q_ref[fin, 9], qpos[fin, 9] + 0.5)
    assert np.allclose(q_ref[fin, 12], qpos[fin, 12] + 0.25)


def test_the_shipped_parameterisation_is_the_song_safe_one():
    """The default `param_mode` must not be the mode that destroys the song.

    `free` optimises an independent pitch correction per frame, and is MEASURED
    to collapse the bilaterally phase-locked courtship song in wing pitch --
    left/right coherence at the song frequency 0.418/0.852 -> 0.038/0.070, with
    pulses buried (168 detected where 43 are real). `spline` preserves it by
    construction. The stage ships OFF, so whoever turns it on gets whatever this
    default says: it must be the safe one, and changing it must be deliberate.

    `knot_spacing` 64 over 32 because a C0 basis leaks 1/f^2 kink energy at the
    knot rate; 64 measured better on the hard fly on every axis (comb 1.06-1.11x
    vs 1.13-1.56x, mask-failure excursion +47.4 deg vs +76.6).

    The multiple-of constraint is real and enforced at wing_mask_refine.py:829.
    """
    wf = _pipeline_wing_mask_fit()
    assert wf["enabled"] is False, "the stage must ship OFF"
    assert wf["param_mode"] == "spline", (
        f"shipped param_mode is {wf['param_mode']!r}; `free` destroys the song")
    assert wf["knot_spacing"] == 64
    assert wf["frame_chunk"] % wf["knot_spacing"] == 0, (
        "spline mode requires frame_chunk % knot_spacing == 0 to keep the knot "
        "grid uniform across chunks")


def test_wing_mask_fit_bout_gates_frames_and_cameras(monkeypatch):
    """exclude_cameras, bridge_ok and min_present_cameras must each actually
    remove evidence -- an all-False `present` row is what leaves a frame at its
    STAC pose inside refine_wing_pitch."""
    import numpy as np
    cfg, cameras, masks_dict, qpos, bs, bR, bt, bok = _wing_fit_fixture()
    rec = _stub_wing_mask_fit(monkeypatch, len(qpos), cameras)
    fit = _run_bout_helpers("wing_mask_fit_bout", "wing_mask_fit_refine_kwargs")[0]

    _, stats = fit(cfg, qpos, bs, bR, bt, bok, masks_dict, cameras)

    valid = rec["sdf"]["valid"]
    i631 = cameras.index("Cam2012631")
    assert not valid[:, i631].any(), "wing_mask_fit.exclude_cameras had no effect"
    assert not valid[0].any(), "a frame with bridge_ok=False has no model->mm map"
    present = rec["refine_wing_pitch"]["kwargs"]["present"]
    assert not present[0].any(), "frame 0 must be frozen at its STAC pose"
    assert present[1:].sum(axis=1).tolist() == [4] * 4
    # "STAC could not solve it" and "too few mask cameras" are DIFFERENT
    # diagnoses -- one sends you to the solver, the other to SAM coverage -- so
    # frame 0 (bridge_ok=False) must not be counted as camera-starved.
    # left is at qpos 12 (+0.25 rad) and right at qpos 9 (+0.5 rad) -- reported
    # BY JOINT NAME, so a positional read of opt_mask would swap these two.
    assert stats == {"n_frames": 5, "n_refined": 3, "n_skipped": 2,
                     # free mode: the frame gate is on the UPDATE, so a
                     # no-evidence frame cannot move and nothing is interpolated
                     "n_interpolated": 0, "n_no_evidence_frames": 2,
                     "n_clamp_hits": 0,
                     "n_thin_frames": 0, "n_no_bridge_frames": 1,
                     # the validity gate is ON but the fixture's masks are
                     # healthy and agree with the (stubbed) pose, so it removes
                     # nothing -- a gate that fired here would be a false
                     # positive, and the count is how you would see it
                     "n_gated_frames": 0, "n_sliver_camera_frames": 0,
                     "n_pose_reject_camera_frames": 0, "gate_enabled": True,
                     "n_at_dpitch_bound": 0,
                     "n_nonfinite_pose_frames": 1, "min_present_cameras": 3,
                     # which PARAMETERISATION the pose came from -- `free`
                     # (per-frame) is the arm measured to destroy the song, so a
                     # reader of qpos_wingfit.npz must not have to reconstruct
                     # the config to find out which one they are holding.
                     # read from the shipped config rather than hardcoded, so
                     # this test asserts "the stats RECORD the parameterisation"
                     # and does not rot every time the default is retuned. The
                     # default itself is pinned deliberately, and separately, by
                     # test_the_shipped_parameterisation_is_the_song_safe_one.
                     "param_mode": WING_MASK_FIT_ON["param_mode"],
                     "knot_spacing": WING_MASK_FIT_ON["knot_spacing"],
                     "dpitch_left_deg": pytest.approx(np.rad2deg(0.25)),
                     "dpitch_right_deg": pytest.approx(np.rad2deg(0.5))}
    # the log accounts for n_skipped by bucket, so the buckets must SUM to it --
    # `moved` also drops non-finite qpos rows, a third bucket the first version
    # counted in neither, so the printed numbers could silently fail to add up
    assert (stats["n_no_bridge_frames"] + stats["n_thin_frames"]
            + stats["n_gated_frames"]
            + stats["n_nonfinite_pose_frames"]) == stats["n_skipped"]

    # raise the bar above what the kept cameras can supply -> nothing is refined
    cfg.wing_mask_fit.min_present_cameras = 5
    _, stats2 = fit(cfg, qpos, bs, bR, bt, bok, masks_dict, cameras)
    # 4 camera-starved + the 1 bridge-less frame = 5 skipped, still disjoint
    assert stats2["n_refined"] == 0
    assert stats2["n_thin_frames"] == 4 and stats2["n_no_bridge_frames"] == 1
    assert stats2["n_nonfinite_pose_frames"] == 0, (
        "frame 1 is now camera-starved as well; the buckets must stay disjoint")
    assert stats2["n_gated_frames"] == 0, (
        "these frames are starved of VALID cameras, which is a different "
        "diagnosis from 'their masks disagree with the pose'")
    assert (stats2["n_no_bridge_frames"] + stats2["n_thin_frames"]
            + stats2["n_gated_frames"]
            + stats2["n_nonfinite_pose_frames"]) == stats2["n_skipped"]


def test_wing_mask_fit_bout_counts_the_frames_that_ACTUALLY_moved(monkeypatch):
    """The frame accounting must be a MEASUREMENT, not a predicate.

    `moved = present.any(1) & finite_pose` is an assumption that holds only in
    `param_mode='free'`, where the frame gate lives on the parameter update. In
    `spline`/`lowpass` one knot spans many frames and the gate lives on the COST
    instead, so a frame with no mask evidence is INTERPOLATED by its neighbouring
    knots -- it genuinely moves. Under the old predicate the log line then said
    "N left at the STAC pose" about frames that had changed, and the reported
    `dpitch` medians were taken over a subset that excluded them.

    That is not cosmetic: the per-frame evidence gate is the only frame-level
    safety the stage has, and the remedy for the SAM-mask collapse on fly0's
    frames 1500-2006 is to widen it. A gate whose effect is invisible in the
    telemetry cannot be tuned.

    The stub moves EVERY finite frame, which is exactly what a band-limited mode
    does, so the two accountings disagree here by construction.
    """
    import numpy as np
    cfg, cameras, masks_dict, qpos, bs, bR, bt, bok = _wing_fit_fixture()
    cfg.wing_mask_fit.param_mode = "spline"
    # The gate is OFF here on purpose: this test documents the hazard the gate
    # exists to remove, so with the gate on it would no longer be visible.
    # `test_the_validity_gate_HOLDS...` below is the with-gate counterpart.
    cfg.wing_mask_fit.gate_enabled = False
    _stub_wing_mask_fit(monkeypatch, len(qpos), cameras, honour_present=False)
    fit = _run_bout_helpers("wing_mask_fit_bout", "wing_mask_fit_refine_kwargs")[0]
    _q, stats = fit(cfg, qpos, bs, bR, bt, bok, masks_dict, cameras)

    # frame 0 has no bridge (no evidence) but the fit moved it; frame 1 is NaN
    # in and NaN out, so it did not.
    assert stats["n_refined"] == 4, (
        f"n_refined {stats['n_refined']} -- frames 0,2,3,4 all changed; a frame "
        f"with no mask evidence still moves in a band-limited mode")
    assert stats["n_skipped"] == 1
    assert stats["n_interpolated"] == 1, (
        "frame 0 changed WITHOUT direct mask evidence -- that has to be visible "
        "in the stats or the evidence gate cannot be tuned")
    # the evidence diagnosis is unchanged and still disjoint
    assert stats["n_no_bridge_frames"] == 1
    assert stats["n_thin_frames"] == 0
    assert stats["n_nonfinite_pose_frames"] == 1
    # and the medians are taken over what MOVED, resolved BY JOINT NAME
    assert stats["dpitch_left_deg"] == pytest.approx(np.rad2deg(0.25))
    assert stats["dpitch_right_deg"] == pytest.approx(np.rad2deg(0.5))
    assert stats["n_clamp_hits"] == 0


def test_wing_mask_fit_bout_counts_joint_clamp_hits(monkeypatch):
    """A clamp hit VOIDS the band-limited guarantee, so it has to be counted.

    The hard `np.clip` in `refine_wing_pitch` is a per-frame nonlinearity applied
    AFTER the parameterisation. Measured on bout 28 fly0, 17 clamped frames took
    a `spline` correction 11.45 deg -- 0.146 of its own amplitude -- out of the
    knot span it lies in to 4.5e-08 everywhere else. "Band-limited by
    construction" is therefore conditional on the clamp not firing, and a reader
    of the log has to be able to see whether it did.
    """
    import numpy as np
    cfg, cameras, masks_dict, qpos, bs, bR, bt, bok = _wing_fit_fixture()
    cfg.wing_mask_fit.param_mode = "spline"
    cfg.wing_mask_fit.gate_enabled = False   # the clamp in isolation
    # +0.5 / +0.25 rad against a 0.2 rad ceiling: every moved frame clamps.
    _stub_wing_mask_fit(monkeypatch, len(qpos), cameras, honour_present=False,
                        limits=(-0.2, 0.2))
    fit = _run_bout_helpers("wing_mask_fit_bout", "wing_mask_fit_refine_kwargs")[0]
    q_ref, stats = fit(cfg, qpos, bs, bR, bt, bok, masks_dict, cameras)
    assert stats["n_clamp_hits"] == 4, (
        f"n_clamp_hits {stats['n_clamp_hits']} -- frames 0,2,3,4 were pushed to "
        f"the joint stop and every one of them voids the knot-span guarantee")
    fin = np.isfinite(qpos).all(axis=1)
    assert np.allclose(q_ref[fin, 9], 0.2) and np.allclose(q_ref[fin, 12], 0.2)


def test_the_validity_gate_HOLDS_no_evidence_frames_at_stac_in_a_band_limited_mode(monkeypatch):
    """The with-gate counterpart of the ACTUALLY_moved test above.

    Without the gate, `spline`/`lowpass` INTERPOLATE a no-evidence frame -- the
    knots span it -- which is how Session0 bout 28 fly0's collapsed masks became
    a smooth 130 deg swing. `frame_keep` holds such a frame at exactly its STAC
    pose, and `n_interpolated` going to zero is how you can see it happened.
    """
    import numpy as np
    cfg, cameras, masks_dict, qpos, bs, bR, bt, bok = _wing_fit_fixture()
    cfg.wing_mask_fit.param_mode = "spline"
    _stub_wing_mask_fit(monkeypatch, len(qpos), cameras, honour_present=False)
    fit = _run_bout_helpers("wing_mask_fit_bout", "wing_mask_fit_refine_kwargs")[0]
    q_ref, stats = fit(cfg, qpos, bs, bR, bt, bok, masks_dict, cameras)

    assert stats["n_interpolated"] == 0, (
        f"{stats['n_interpolated']} no-evidence frames still moved -- the gate "
        f"is not holding them at the STAC pose, which is the ONLY thing it is "
        f"for in a band-limited mode")
    assert np.array_equal(q_ref[0], qpos[0]), \
        "frame 0 has no bridge; it must come back bit-identical"
    assert stats["n_refined"] == 3 and stats["n_skipped"] == 2


def test_the_validity_gate_skips_sliver_and_pose_disagreeing_frames(monkeypatch):
    """The two rejection reasons, end to end through the stage, and they are
    counted separately because they send a reader to different places.

    A SLIVER is a truncated mask -- SAM found the fly but only a fragment of it.
    A POSE-DISAGREEING mask is full-size and in the wrong place: the wrong fly,
    or two flies merged. No area test can see the second one, which is why the
    gate projects the body at all.
    """
    import numpy as np
    cfg, cameras, masks_dict, qpos, bs, bR, bt, bok = _wing_fit_fixture(n_frames=6)
    bok[:] = True
    qpos[:] = 0.0                                   # every frame solvable
    cfg.wing_mask_fit.param_mode = "spline"
    C = len(cameras)
    masks = np.ones((6, C, 8, 8), bool)
    masks[3] = False
    masks[3, :, 0:1, 0:1] = True                    # frame 3: slivers everywhere
    masks_dict["masks"] = masks
    rec = _stub_wing_mask_fit(monkeypatch, 6, cameras, honour_present=False)
    # frame 4: full-size masks, but the body projects OUTSIDE them
    uv = np.zeros((6, C, 4, 2), np.float32)
    uv[4] = 100.0
    rec_uv = {"body_uv_track": lambda q, *a, **k: uv}
    import sys
    mod = sys.modules["jarvis_jax.tracking.wing_mask_refine"]
    monkeypatch.setattr(mod, "body_uv_track", rec_uv["body_uv_track"])
    fit = _run_bout_helpers("wing_mask_fit_bout", "wing_mask_fit_refine_kwargs")[0]
    q_ref, stats = fit(cfg, qpos, bs, bR, bt, bok, masks_dict, cameras)

    # Cam2012631 is excluded for wings, so 4 of the 5 fixture cameras count
    assert stats["n_sliver_camera_frames"] == 4, stats
    assert stats["n_pose_reject_camera_frames"] == 4, stats
    assert stats["n_gated_frames"] == 2, (
        f"frames 3 (sliver) and 4 (wrong place) must be gated, got "
        f"{stats['n_gated_frames']}")
    assert np.array_equal(q_ref[3], qpos[3]) and np.array_equal(q_ref[4], qpos[4])
    assert not np.array_equal(q_ref[0], qpos[0]), \
        "the healthy frames must still be fitted"
    assert stats["n_refined"] == 4


def test_max_dpitch_deg_reaches_the_refiner_and_is_counted(monkeypatch):
    """The MODEL's joint limits are no bound -- wing pitch is legal over
    -72.8..+167.3 deg and the optimiser has been measured travelling most of it
    on collapsed masks. `max_dpitch_deg` is the physiological one, and a
    non-zero `n_at_dpitch_bound` is a pointer at the MASKS, not a defect."""
    import numpy as np
    cfg, cameras, masks_dict, qpos, bs, bR, bt, bok = _wing_fit_fixture()
    cfg.wing_mask_fit.max_dpitch_deg = 5.0        # the stub moves 14.3 / 28.6 deg
    rec = _stub_wing_mask_fit(monkeypatch, len(qpos), cameras)
    fit = _run_bout_helpers("wing_mask_fit_bout", "wing_mask_fit_refine_kwargs")[0]
    q_ref, stats = fit(cfg, qpos, bs, bR, bt, bok, masks_dict, cameras)
    assert rec["refine_wing_pitch"]["kwargs"]["max_dpitch_deg"] == 5.0
    fin = np.isfinite(qpos).all(axis=1) & np.asarray(bok, bool)
    assert np.allclose(np.abs(q_ref[fin][:, [9, 12]] - qpos[fin][:, [9, 12]]),
                       np.deg2rad(5.0))
    assert stats["n_at_dpitch_bound"] == 3, stats

    cfg.wing_mask_fit.max_dpitch_deg = None       # null == no bound
    rec2 = _stub_wing_mask_fit(monkeypatch, len(qpos), cameras)
    _q, stats2 = fit(cfg, qpos, bs, bR, bt, bok, masks_dict, cameras)
    assert rec2["refine_wing_pitch"]["kwargs"]["max_dpitch_deg"] is None
    assert stats2["n_at_dpitch_bound"] == 0


def test_the_gate_and_the_bound_have_SEPARATE_switches_and_both_can_be_off(monkeypatch):
    """THE ESCAPE HATCH. Every acceptance number in
    docs/benchmark/2026-09-01-wing-mask-fit/ sections 1-11 was measured before
    the validity gate and the |dpitch| bound existed, and the spec's own
    criterion thresholds -- including the null-derived CRIT1F_PEAK_TOL = 2.0 --
    were calibrated against them. If no config state reproduced those arms they
    would stop being re-derivable from the committed config.

    So the bound gets its OWN key rather than living under `gate_enabled`, and
    `gate_enabled=false max_dpitch_deg=null` must reach the refiner as
    `frame_keep=None, max_dpitch_deg=None` -- the pre-gate call, exactly.
    (`apply_gate_and_bound` is then a bit-exact no-op; pinned in
    third_party/jarvis_jax/tests/test_mask_quality.py.)

    Turning ONLY the gate off is NOT the escape hatch, and this pins that too:
    measured on bout 28 fly0, the bound alone fires on 34 frames and moves her
    left wing by up to 28.2 deg.
    """
    import numpy as np
    cfg, cameras, masks_dict, qpos, bs, bR, bt, bok = _wing_fit_fixture()
    cfg.wing_mask_fit.gate_enabled = False
    cfg.wing_mask_fit.max_dpitch_deg = None
    rec = _stub_wing_mask_fit(monkeypatch, len(qpos), cameras)
    fit = _run_bout_helpers("wing_mask_fit_bout", "wing_mask_fit_refine_kwargs")[0]
    q_ref, stats = fit(cfg, qpos, bs, bR, bt, bok, masks_dict, cameras)

    kw = rec["refine_wing_pitch"]["kwargs"]
    assert kw["frame_keep"] is None, "gate_enabled=false must not pass a frame gate"
    assert kw["max_dpitch_deg"] is None, "max_dpitch_deg=null must not become 0.0"
    assert stats["gate_enabled"] is False
    assert stats["n_gated_frames"] == 0 and stats["n_at_dpitch_bound"] == 0
    # the stub moves 14.3 / 28.6 deg; neither switch may touch that
    fin = np.isfinite(qpos).all(axis=1) & np.asarray(bok, bool)
    assert np.allclose(q_ref[fin, 9], qpos[fin, 9] + 0.5)
    assert np.allclose(q_ref[fin, 12], qpos[fin, 12] + 0.25)
    # and the SDF stack saw the ungated validity -- the gate did not run at all
    assert rec["sdf"]["valid"][1:, [0, 2, 3, 4]].all()

    # gate off but bound ON is a DIFFERENT state, and it must bite
    cfg.wing_mask_fit.max_dpitch_deg = 5.0
    rec2 = _stub_wing_mask_fit(monkeypatch, len(qpos), cameras)
    q2, stats2 = fit(cfg, qpos, bs, bR, bt, bok, masks_dict, cameras)
    assert rec2["refine_wing_pitch"]["kwargs"]["max_dpitch_deg"] == 5.0
    assert stats2["n_at_dpitch_bound"] == 3
    assert not np.allclose(q2[fin, 9], qpos[fin, 9] + 0.5)


def test_wing_mask_fit_bout_refuses_a_non_canonical_mask_camera_axis(monkeypatch):
    """The camera-order trap. `refine_wing_pitch` only checks the camera COUNT,
    so a PERMUTATION is invisible to it -- it would project each camera's wing
    onto another camera's mask and still produce a plausible residual. Both axes
    are built by name off `cfg.recording.cameras`; this pins the check that says
    so, and the refusal of a legacy npz that carries no camera names at all."""
    cfg, cameras, masks_dict, qpos, bs, bR, bt, bok = _wing_fit_fixture()
    _stub_wing_mask_fit(monkeypatch, len(qpos), cameras)
    fit = _run_bout_helpers("wing_mask_fit_bout", "wing_mask_fit_refine_kwargs")[0]

    permuted = dict(masks_dict, cameras=list(reversed(cameras)))
    with pytest.raises(RuntimeError, match="canonical"):
        fit(cfg, qpos, bs, bR, bt, bok, permuted, cameras)

    legacy = {k: v for k, v in masks_dict.items() if k != "cameras"}
    with pytest.raises(RuntimeError, match="cannot be verified by name"):
        fit(cfg, qpos, bs, bR, bt, bok, legacy, cameras)

    cfg.wing_mask_fit.exclude_cameras = ["CamNotOnTheRig"]
    with pytest.raises(ValueError, match="CamNotOnTheRig"):
        fit(cfg, qpos, bs, bR, bt, bok, masks_dict, cameras)


# ---------------------------------------------------------------------------
# Task 6 fix round 1: the wing fit carries its OWN provenance
# ---------------------------------------------------------------------------

def test_wing_mask_fit_signature_covers_the_semantic_keys_and_only_those():
    """The fit's provenance must move for anything that changes the POSE and
    stay put for anything that only changes speed.

    `frame_chunk` is documented in the YAML as a pure performance knob (frames
    per device call), so enrolling it would invalidate a perfectly good fit
    merely for moving to a smaller GPU. A blanket `sorted(wf)` would also
    auto-enrol any key added later, including a future `prefetch` -- hence the
    exact partition below, which fails if a new YAML key is neither semantic nor
    declared performance-only.
    """
    import json as _json
    from omegaconf import OmegaConf
    sig = _run_bout_helper("wing_mask_fit_signature")
    wf = _pipeline_wing_mask_fit()

    on = OmegaConf.create({"wing_mask_fit": WING_MASK_FIT_ON})
    base = sig(on)
    keys = set(_json.loads(base))

    perf_only = {"frame_chunk"}
    assert keys | perf_only | {"enabled"} == set(wf), (
        f"wing_mask_fit keys are neither in the signature nor declared "
        f"performance-only: {sorted(set(wf) - keys - perf_only - {'enabled'})}")
    assert not (keys & perf_only), f"performance-only keys enrolled: {keys & perf_only}"

    def _moved(patch):
        return sig(OmegaConf.merge(on, OmegaConf.create({"wing_mask_fit": patch}))) != base

    # performance knobs must NOT move it
    assert not _moved({"frame_chunk": 64}), \
        "frame_chunk is a perf knob; moving GPU must not invalidate the fit"
    # every semantic knob must
    for k, v in (("coverage_weight", 0.03), ("huber_delta", 0.0), ("n_steps", 50),
                 ("lr", 0.05), ("containment_weight", 1.0), ("smooth_weight", 0.5),
                 ("limit_weight", 1.0), ("dilate_px", 3), ("body_vertex_stride", 20),
                 ("bbox_margin", 0.2), ("n_target_points", 64),
                 ("min_present_cameras", 5), ("coverage_normalize", False),
                 ("out_hw", [64, 64]), ("exclude_cameras", [])):
        assert _moved({k: v}), f"{k} changes the fitted pose but not its signature"


def test_wing_fit_action_reuses_and_refits_but_never_refuses():
    """The staleness contract for qpos_wingfit.npz, as a pure function.

    Three reachable routes make this necessary even though the Stage-B gate
    exists, and in ALL THREE that gate never fires: deleting kp3d.npz by hand
    (the ordinary "just re-triangulate this bout"), pipeline.allow_stale_kp3d,
    and scripts/analysis/stage_b_restage.py.

    There is deliberately NO 'refuse'. An earlier round raised on
    "stage disabled but a fit and an outputs.h5 are both on disk", which was the
    wrong tool three ways: it cannot fire on the common case (a completed bout
    has DONE and returns before the check), its blast radius is the whole SLURM
    array task because nothing wraps process_bout_fly in try/except, and it
    assumed on exactly the missing information that `reuse` assumed away. The
    pose STAMP makes that state decidable instead -- rebuild once from the STAC
    pose and stamp "none" -- so the action set is off/fit/reuse.
    """
    from omegaconf import OmegaConf
    action, sig = _run_bout_helpers("wing_fit_action", "wing_mask_fit_signature",
                                    "wing_mask_fit_enabled")[:2]

    off = OmegaConf.create({"wing_mask_fit": {"enabled": False}})
    on = OmegaConf.create({"wing_mask_fit": WING_MASK_FIT_ON})
    retuned = OmegaConf.merge(on, OmegaConf.create(
        {"wing_mask_fit": {"coverage_weight": 0.03}}))
    faster = OmegaConf.merge(on, OmegaConf.create(
        {"wing_mask_fit": {"frame_chunk": 64}}))
    s_on = sig(on)

    assert action(off, None) == ("off", None)
    assert action(OmegaConf.create({}), None) == ("off", None)
    # disabled with a fit on disk is NOT an error -- the stamp decides
    assert action(off, s_on) == ("off", None)
    assert action(on, None) == ("fit", s_on)          # nothing on disk
    assert action(on, s_on) == ("reuse", s_on)        # matching provenance
    assert action(retuned, s_on)[0] == "fit"          # different weights
    assert action(faster, s_on) == ("reuse", s_on)    # perf-only change


def test_pose_provenance_is_read_from_each_artifact_and_defaults_to_none(tmp_path):
    """The rebuild decision must live ON DISK, one stamp per derived artifact.

    N1 (the reason): with an in-memory `_wing_fit_rebuilt` flag, a preemption
    between build_fly_outputs and qc_report left QC PERMANENTLY stale. On the
    next run the stored wing-fit signature matched, so the action was 'reuse',
    the flag was False, and qc.json/qc_perframe.npz both existed -- so both
    guards skipped forever while outputs.h5 held the new pose. We run on
    preemptible ckpt nodes. Per-artifact stamps make that state self-describing.
    """
    import json as _json
    import numpy as np
    import h5py
    (outp, qcp, pfp, side) = _run_bout_helpers(
        "outputs_pose_source", "json_pose_source", "npz_pose_source",
        "pose_artifact_stale")

    # an UNSTAMPED artifact reads as the pre-wing-fit pose, which is what keeps
    # enabling nothing a strict no-op on every bout already on disk
    assert side(None, "none") is False
    assert side("none", "none") is False
    assert side(None, "SIG") is True
    assert side("OLD", "SIG") is True
    assert side("SIG", "SIG") is False

    h5p = tmp_path / "outputs.h5"
    with h5py.File(h5p, "w") as f:
        f["mesh_mm"] = np.zeros((2, 3, 3), np.float32)
    assert outp(str(h5p)) is None, "an outputs.h5 predating the stamp reads None"
    with h5py.File(h5p, "w") as f:
        f["mesh_mm"] = np.zeros((2, 3, 3), np.float32)
        f["pose_source"] = np.asarray("SIG").astype("S")
    assert outp(str(h5p)) == "SIG"
    assert outp(str(tmp_path / "missing.h5")) is None

    jp = tmp_path / "qc.json"
    jp.write_text(_json.dumps({"reproj": 1.0}))
    assert qcp(str(jp)) is None
    jp.write_text(_json.dumps({"reproj": 1.0, "pose_source": "OLD"}))
    assert qcp(str(jp)) == "OLD"

    np.savez(tmp_path / "qc_perframe.npz", soft_iou=np.zeros(3))
    assert pfp(str(tmp_path / "qc_perframe.npz")) is None
    np.savez(tmp_path / "qc_perframe.npz", soft_iou=np.zeros(3), pose_source="OLD")
    assert pfp(str(tmp_path / "qc_perframe.npz")) == "OLD"

    # THE N1 SCENARIO: outputs.h5 rebuilt and stamped, QC killed mid-write.
    # outputs.h5 is current; qc.json is not; the two decisions must diverge.
    assert side(outp(str(h5p)), "SIG") is False
    assert side(qcp(str(jp)), "SIG") is True


def test_the_wing_fit_records_its_signature_in_its_own_artifact():
    """It is written into qpos_wingfit.npz, read back on resume, and the read
    is what drives the decision -- not the file's mere existence."""
    import ast as _ast
    src = RUN_BOUT.read_text()
    fn = next(n for n in _ast.parse(src).body
              if isinstance(n, _ast.FunctionDef) and n.name == "process_bout_fly")
    body = _ast.get_source_segment(src, fn)
    assert "atomic_save_npz(wingfit_path" in body
    i = body.index("atomic_save_npz(wingfit_path")
    assert "wing_mask_fit_sig=" in body[i:i + 300], \
        "the fit must record the config it was made under"
    assert "stored_wing_fit_signature(wingfit_path)" in body, \
        "the decision must read the stored provenance, not just os.path.exists"
    assert "stage_done(wingfit_path)" not in body, \
        "existence alone must not gate the fit -- that is the stale-reuse bug"
    # N2: the pose provenance handed to Stage E must come from the ACTION, not
    # from "did we recompute this run". On the `reuse` path the fit is loaded
    # and qpos_refined replaced, so an outputs.h5 that was NOT built from the
    # fit still has to be rebuilt -- reachable with nothing but config flags
    # (run on -> delete outputs.h5/qc.json and re-run OFF for the before-picture
    # -> re-enable unchanged -> 'reuse', and outputs.h5 keeps the STAC pose
    # while the log prints a plausible [wing-mask-fit] line).
    assert "_wing_fit_rebuilt" not in body, (
        "the rebuild decision must be the on-disk stamp, not an in-memory flag: "
        "the flag is False on the `reuse` path and does not survive a preemption")
    assert '_pose_sig = _wf_sig if _action in ("fit", "reuse") else "none"' in body, \
        "the pose provenance must be decided by the action, including `reuse`"


def test_every_pose_derived_artifact_is_gated_on_the_pose_stamp():
    """N3: the invalidation must reach every artifact built from the fitted pose.

    The previous version of this test enumerated `If` nodes mentioning three
    consumer names, which FROZE AN INCOMPLETE LIST into a test -- it asserted
    that outputs.h5/qc.json/qc_perframe.npz were "the ONLY consumers" while the
    per-camera reprojection overlays (FK'd from outputs.h5's mesh_mm) and
    sidebyside.mp4 were skipped by their own `stage_done` guards and kept
    showing the previous wings.

    So this now works the other way round: it collects EVERY `stage_done(...)`
    guard in process_bout_fly and requires each one to be classified -- either
    upstream of the pose, or gated on the pose stamp. A guard added later cannot
    quietly default to "not my problem".
    """
    import ast as _ast
    src = RUN_BOUT.read_text()
    fn = next(n for n in _ast.parse(src).body
              if isinstance(n, _ast.FunctionDef) and n.name == "process_bout_fly")

    # artifacts that exist BEFORE the pose is fitted, or are the fit's own input
    upstream = {"coverage_path", "kp2d_path", "kp3d_path", "kp3d_filt_path",
                "offsets_path", "stac_h5_path", "qpos_path", "scale_path",
                "seg_scales_path", "wingfit_path"}
    # artifacts built FROM the fitted qpos -- every one must honour the stamp
    pose_derived = {"outputs_h5_path", "qc_json_path", "qc_perframe_path",
                    "overlay_path", "sbs_path"}

    guarded = set()
    for node in _ast.walk(fn):
        if (isinstance(node, _ast.Call)
                and getattr(node.func, "id", None) == "stage_done"
                and node.args and isinstance(node.args[0], _ast.Name)):
            guarded.add(node.args[0].id)
    unclassified = guarded - upstream - pose_derived
    assert not unclassified, (
        f"stage_done guards on artifacts nobody has classified as upstream or "
        f"pose-derived: {sorted(unclassified)} -- decide, do not default")
    missing = pose_derived - guarded
    assert not missing, f"expected a stage_done guard on {sorted(missing)}"

    # ... and every guard over a pose-derived artifact must also test staleness
    unstamped = []
    for node in _ast.walk(fn):
        if not isinstance(node, _ast.If):
            continue
        dump = _ast.dump(node.test)
        if "stage_done" not in dump:
            continue
        hit = sorted(a for a in pose_derived if f"id='{a}'" in dump)
        # A guard whose BODY writes the stamp sidecar is a completeness check
        # ("did every camera render before I declare the group current?"), not a
        # skip -- it must not be required to test staleness itself.
        writes_stamp = any(isinstance(c, _ast.Call)
                           and getattr(c.func, "id", None) == "atomic_save_json"
                           for st in node.body for c in _ast.walk(st))
        if hit and not writes_stamp and "stale" not in dump:
            unstamped.append((node.lineno, hit))
    assert not unstamped, (
        f"these guards skip a pose-derived artifact without comparing the pose "
        f"stamp, so a refit never reaches them: {unstamped}")

    # ... and each staleness term must come from THAT artifact's OWN stamp.
    # Sharing one term is precisely the N1 bug in a new costume: a preemption
    # between build_fly_outputs and qc_report leaves outputs.h5 current and QC
    # not, so a QC guard reading outputs.h5's stamp declares QC fresh forever.
    flat = "".join(_ast.get_source_segment(src, fn).split())
    for stale, reader, path in (
            ("_outputs_stale", "outputs_pose_source", "outputs_h5_path"),
            ("_qc_stale", "json_pose_source", "qc_json_path"),
            ("_qcpf_stale", "npz_pose_source", "qc_perframe_path"),
            ("_sbs_stale", "json_pose_source", "sbs_stamp_path")):
        want = f"{stale}=pose_artifact_stale({reader}({path}),_pose_sig)"
        assert want in flat, (
            f"{stale} must be read from {path}'s own stamp via {reader}; "
            f"sharing another artifact's term reintroduces the preemption hole")

    # the overlays are the one PER-CAMERA case: an .mp4 cannot carry a stamp and
    # a group stamp made one permanently-failing camera re-render the others on
    # every resume, so the sidecar maps camera -> pose and the skip is per camera
    assert "_overlay_srcs=overlay_pose_sources(overlay_stamp_path)" in flat
    assert "_overlay_stale=_overlay_srcs.get(cam)!=_pose_sig" in flat, \
        "the overlay skip must be decided per camera, from that camera's stamp"

    # every stamp that is READ must also be WRITTEN, or the artifact rebuilds on
    # every single run instead of once
    for write in ("pose_source=_pose_sig",                       # build_fly_outputs
                  "stamp_json_pose_source(qc_json_path,_pose_sig)",
                  "atomic_save_npz(qc_perframe_path,pose_source=_pose_sig",
                  'atomic_save_json(overlay_stamp_path,{"cameras":_overlay_srcs})',
                  'atomic_save_json(sbs_stamp_path,{"pose_source":_pose_sig})'):
        assert write in flat, f"nothing writes the stamp: {write}"


def test_the_renderers_are_told_which_pose_to_draw():
    """N3, second half: the pipeline's default visual QC artifact must be able
    to show this stage's effect. `python -m viz sidebyside` loads the pose
    itself, so run_bout must name the source explicitly rather than let the
    renderer guess -- and CLAUDE.md's bar is that a change is not done until a
    figure shows it on real frames."""
    import ast as _ast
    src = RUN_BOUT.read_text()
    fn = next(n for n in _ast.parse(src).body
              if isinstance(n, _ast.FunctionDef) and n.name == "process_bout_fly")
    body = _ast.get_source_segment(src, fn)
    i = body.index('"-m", "viz", "sidebyside"')
    cmd = body[i:i + 2500]
    assert "sidebyside_pose_args(" in cmd, (
        "the sidebyside subprocess must name the pose source; without it the "
        "renderer loads qpos_refined.npz and shows the PRE-FIT pose")

    # The value depends on the RIGHT panel too, so the decision is a helper the
    # binding tests in viz/tests/test_cli.py drive through the real argparse and
    # the real check_pose_mode. Here just pin that both arms are reachable.
    pose_args, = _run_bout_helpers("sidebyside_pose_args")
    assert pose_args("rigcam", "fit") == ["--pose", "wingfit"]
    assert pose_args("rigcam", "off") == ["--pose", "refined"]
    assert pose_args("mujoco", "fit") == [], (
        "the mujoco panel renders stac_ik.h5's pre-bridge pose and viz refuses "
        "a --pose there; emitting one made every Stage-F subprocess exit 2, "
        "non-fatally, so sidebyside.mp4 silently stopped being produced")


def test_stage_b_restage_moves_the_wing_fit_aside():
    """F4: stage_b_restage.py moves the STAC solve aside to force a re-run. A
    qpos_wingfit.npz left behind was solved against the stac_ik.h5 it just
    removed, and outputs.h5 would be rebuilt from it."""
    import ast as _ast
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parents[1] / "scripts" / "analysis"
           / "stage_b_restage.py").read_text()
    node = next(n for n in _ast.parse(src).body
                if isinstance(n, _ast.Assign)
                and any(getattr(t, "id", None) == "BOUT_ARTIFACTS" for t in n.targets))
    artifacts = [e.value for e in node.value.elts]
    assert "qpos_wingfit.npz" in artifacts, (
        f"BOUT_ARTIFACTS moves the STAC solve aside but leaves the wing fit "
        f"solved against it: {artifacts}")
    # it must sit with the other pose artifacts, not after DONE-only entries
    assert artifacts.index("qpos_wingfit.npz") > artifacts.index("stac_ik.h5")


def test_outputs_h5_carries_the_pose_it_was_built_from():
    """The stamp is INTRINSIC to outputs.h5, not a sidecar beside it.

    outputs.h5 is what every external consumer reads -- the benchmark, the
    renderers, the analysis scripts -- and a sidecar can be moved or deleted
    away from it. `pose_source` travels with the pose.
    """
    import ast as _ast
    import importlib.util
    from pathlib import Path as _P
    origin = importlib.util.find_spec("jarvis_jax.tracking.outputs").origin
    tree = _ast.parse(_P(origin).read_text())
    fns = {n.name: n for n in tree.body if isinstance(n, _ast.FunctionDef)}
    for name in ("build_fly_outputs", "write_outputs_h5"):
        args = {a.arg for a in fns[name].args.kwonlyargs} | {a.arg for a in fns[name].args.args}
        assert "pose_source" in args, f"{name} cannot record the pose provenance"
    src = _ast.get_source_segment(_P(origin).read_text(), fns["write_outputs_h5"])
    assert '"pose_source"' in src, "the key must actually reach the h5 dict"
    # and run_bout must hand it the value it just decided
    body = _ast.get_source_segment(
        RUN_BOUT.read_text(),
        next(n for n in _ast.parse(RUN_BOUT.read_text()).body
             if isinstance(n, _ast.FunctionDef) and n.name == "process_bout_fly"))
    i = body.index("build_fly_outputs(")
    assert "pose_source=_pose_sig" in body[i:i + 500], \
        "Stage E must stamp the pose provenance it is writing"


# ---------------------------------------------------------------------------
# Task 6 fix round 3
# ---------------------------------------------------------------------------

def test_overlay_stamps_are_per_camera_not_per_group(tmp_path):
    """F2 (cost): the group sidecar was written only when EVERY camera's mp4
    existed, and overlay failures are caught non-fatally -- so one camera that
    keeps erroring left the group stale forever and re-rendered the six good
    cameras on every resumed run, minutes each time, where previously each
    existing mp4 was simply skipped."""
    import ast as _ast
    import json as _json
    read, = _run_bout_helpers("overlay_pose_sources")

    assert read(str(tmp_path / "nope.json")) == {}
    p = tmp_path / "pose_source.json"
    p.write_text(_json.dumps({"cameras": {"Cam1": "SIG"}}))
    assert read(str(p)) == {"Cam1": "SIG"}
    p.write_text(_json.dumps({"pose_source": "SIG"}))    # the old group format
    assert read(str(p)) == {}, "the old group stamp must not read as any camera"

    src = RUN_BOUT.read_text()
    fn = next(n for n in _ast.parse(src).body
              if isinstance(n, _ast.FunctionDef) and n.name == "process_bout_fly")
    flat = "".join(_ast.get_source_segment(src, fn).split())
    assert "_overlay_srcs.get(cam)" in flat.replace('"', "'") or \
           "_overlay_srcs.get(cam)" in flat, "the skip must be decided per camera"
    assert "all(stage_done(os.path.join(overlay_dir" not in flat, \
        "the all-cameras group gate is what made one bad camera cost every run"

    # ... and a camera may only be stamped if it ACTUALLY rendered. Stamping
    # every job unconditionally is the same bug pointing the other way: a camera
    # whose render failed would be recorded as current and never retried, so its
    # missing or stale mp4 is declared fine forever.
    loops = [n for n in _ast.walk(fn)
             if isinstance(n, _ast.For)
             and getattr(n.iter, "id", None) == "jobs"
             and any(getattr(t, "value", None) is not None
                     and getattr(getattr(t, "value", None), "id", None) == "_overlay_srcs"
                     for a in _ast.walk(n) if isinstance(a, _ast.Assign)
                     for t in a.targets)]
    assert loops, "could not find the loop that stamps the rendered cameras"
    def _both_conditions_required(test):
        """An `and` of a render-succeeded check and an output-exists check.

        Substring-matching the dump would pass an `and` -> `or` swap, which
        would stamp a camera whose render failed as soon as a stale mp4 happened
        to exist -- so require the BoolOp and check each operand separately.
        """
        if not (isinstance(test, _ast.BoolOp) and isinstance(test.op, _ast.And)):
            return False
        dumps = [_ast.dump(v) for v in test.values]
        return (any("_errs" in d for d in dumps)
                and any("stage_done" in d for d in dumps))

    guarded = [any(isinstance(st, _ast.If) and _both_conditions_required(st.test)
                   for st in _ast.walk(lp))
               for lp in loops]
    assert all(guarded), (
        "a camera is stamped without checking that its render succeeded and its "
        "mp4 exists -- a failing camera would then never be retried")


def test_backfill_qc_perframe_stamps_the_pose_it_rebuilt_from():
    """F5: `_backfill_qc_perframe` writes qc_perframe.npz from outputs.h5, and
    it lives OUTSIDE process_bout_fly, so the AST guard above cannot see it. It
    was the one writer of a stamped artifact that did not stamp."""
    import ast as _ast
    src = RUN_BOUT.read_text()
    fn = next(n for n in _ast.parse(src).body
              if isinstance(n, _ast.FunctionDef) and n.name == "_backfill_qc_perframe")
    flat = "".join(_ast.get_source_segment(src, fn).split())
    assert "atomic_save_npz(qc_perframe_path,pose_source=" in flat, (
        "the backfill must stamp the pose it rebuilt from -- it is rebuilding "
        "from outputs.h5, whose own stamp says which pose that is")
    assert "outputs_pose_source(outputs_h5_path)" in flat, \
        "and that value must come from outputs.h5, not be invented"


def test_the_ab_render_script_pins_the_pose_it_patched():
    """F1, second half. scripts/viz/render_prior_sidebyside.py builds each A/B
    arm by COPYING outputs.h5/qpos_refined.npz/stac_ik.h5 and SYMLINKING
    everything else, then patches only qpos_refined.npz and
    outputs.h5[qpos, kp3d_mm]. A qpos_wingfit.npz would follow the symlink into
    both arms, and it invoked `viz sidebyside` with no --pose -- so once any
    bout has a wing fit, both arms render the SAME wing-fit pose and the
    comparison silently shows nothing. Latent only because the stage has never
    run, and Task 7 is about to run it."""
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parents[1] / "scripts" / "viz"
           / "render_prior_sidebyside.py").read_text()
    flat = "".join(src.split())
    assert '"--pose","refined"' in flat, (
        "each arm's pose is the qpos_refined.npz this script patched, so the "
        "render must be pinned to it")
    assert '"pose_source"' in flat, \
        "the patched outputs.h5 must not keep a stamp that lies about its contents"
    assert '"qpos_wingfit.npz"' in flat, \
        "a wing fit must not be symlinked into an arm whose pose was patched"
