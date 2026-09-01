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
    import numpy as _np
    body = [n for n in _ast.parse(RUN_BOUT.read_text()).body
            if isinstance(n, _ast.FunctionDef) and n.name in names]
    got = {n.name for n in body}
    assert got == set(names), f"run_bout.py is missing {sorted(set(names) - got)}"
    g = {"json": _json, "np": _np}
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
                   "min_present_cameras", "exclude_cameras"}
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


class _Rec(dict):
    """A recorder that stands in for one jarvis_jax entry point."""

    def __init__(self, ret):
        super().__init__()
        self._ret = ret

    def __call__(self, *a, **kw):
        self["args"], self["kwargs"] = a, kw
        return self._ret(*a, **kw) if callable(self._ret) else self._ret


def _stub_wing_mask_fit(monkeypatch, n_frames, cameras):
    """Install fake jarvis_jax/mujoco entry points for wing_mask_fit_bout.

    Only the WIRING is under test here. The real `refine_wing_pitch` runs
    `mjx.kinematics` on an 87-joint model, measured at 1739 s to COMPILE on the
    XLA CPU backend against 6.4 s on an L40S (Task 5), so calling it for real is
    a GPU-only Task-7 concern -- but the config plumbing, the camera axes, the
    per-frame gating and the stats are all exercised for real below.
    """
    import sys
    import types
    import numpy as np

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
        "jarvis_jax.tracking.wing_mask_refine": {
            "affine_cameras_by_name": _Rec(
                (np.zeros((C, 2, 3), np.float32), np.zeros((C, 2), np.float32))),
            "body_vertex_indices": _Rec(np.arange(8, dtype=np.int32)),
            "qpos_limits": lambda m: (np.full(14, -np.inf, np.float32),
                                      np.full(14, np.inf, np.float32)),
            "wing_pitch_dof_mask": lambda m: np.isin(np.arange(14), [9, 12]),
            "refine_wing_pitch": _Rec(
                lambda q, **kw: np.array(q, np.float32) + delta)},
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
    masks_dict = {"masks": np.zeros((n_frames, len(cameras), 4, 4), bool),
                  "valid": np.ones((n_frames, len(cameras)), bool),
                  "cameras": list(cameras), "T": n_frames, "C": len(cameras)}
    qpos = np.zeros((n_frames, 14), np.float32)
    bs = np.ones(n_frames, np.float32)
    bR = np.broadcast_to(np.eye(3, dtype=np.float32), (n_frames, 3, 3)).copy()
    bt = np.zeros((n_frames, 3), np.float32)
    bok = np.ones(n_frames, bool)
    bok[0] = False                       # frame 0: STAC could not solve it
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
    # and the pose actually handed on is the refined one
    assert np.allclose(q_ref[:, 9], qpos[:, 9] + 0.5)
    assert np.allclose(q_ref[:, 12], qpos[:, 12] + 0.25)


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
    assert stats == {"n_frames": 5, "n_refined": 4, "n_skipped": 1,
                     "n_thin_frames": 0, "n_no_bridge_frames": 1,
                     "min_present_cameras": 3,
                     "dpitch_left_deg": pytest.approx(np.rad2deg(0.25)),
                     "dpitch_right_deg": pytest.approx(np.rad2deg(0.5))}

    # raise the bar above what the kept cameras can supply -> nothing is refined
    cfg.wing_mask_fit.min_present_cameras = 5
    _, stats2 = fit(cfg, qpos, bs, bR, bt, bok, masks_dict, cameras)
    # 4 camera-starved + the 1 bridge-less frame = 5 skipped, still disjoint
    assert stats2["n_refined"] == 0
    assert stats2["n_thin_frames"] == 4 and stats2["n_no_bridge_frames"] == 1


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


def test_wing_fit_action_reuses_refits_and_refuses():
    """The whole staleness contract for qpos_wingfit.npz, as a pure function.

    Three reachable routes make this necessary even though the Stage-B gate
    exists, and in ALL THREE that gate never fires: deleting kp3d.npz by hand
    (the ordinary "just re-triangulate this bout"), pipeline.allow_stale_kp3d,
    and scripts/analysis/stage_b_restage.py.
    """
    from omegaconf import OmegaConf
    action, sig = (_run_bout_helpers("wing_fit_action", "wing_mask_fit_signature",
                                     "wing_mask_fit_enabled"))[:2]

    off = OmegaConf.create({"wing_mask_fit": {"enabled": False}})
    on = OmegaConf.create({"wing_mask_fit": WING_MASK_FIT_ON})
    retuned = OmegaConf.merge(on, OmegaConf.create(
        {"wing_mask_fit": {"coverage_weight": 0.03}}))
    faster = OmegaConf.merge(on, OmegaConf.create(
        {"wing_mask_fit": {"frame_chunk": 64}}))
    s_on = sig(on)

    # off, nothing on disk -> strict no-op
    assert action(off, None, False) == ("off", None)
    assert action(OmegaConf.create({}), None, True) == ("off", None)
    # off, an orphan fit but no outputs.h5 -> still a no-op (nothing consumed it)
    assert action(off, s_on, False) == ("off", None)
    # off, a fit AND an outputs.h5 -> nothing records which pose outputs.h5 holds
    assert action(off, s_on, True)[0] == "refuse"
    # on, nothing on disk -> fit
    assert action(on, None, False) == ("fit", s_on)
    # on, a file with NO provenance (written before this record existed) -> refit
    assert action(on, None, True) == ("fit", s_on)
    # on, matching provenance -> reuse
    assert action(on, s_on, True) == ("reuse", s_on)
    # on, provenance from a different weight -> refit
    assert action(retuned, s_on, True)[0] == "fit"
    # on, same weights but a different frame_chunk -> reuse (perf-only)
    assert action(faster, s_on, True) == ("reuse", s_on)


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


def test_a_new_or_changed_wing_fit_invalidates_outputs_and_qc():
    """F3: outputs.h5/qc.json/qc_perframe.npz are the ONLY consumers of the
    refined qpos and every one is guarded by `stage_done`. Enabling the stage on
    a bout that already has them would otherwise compute the fit, write it, log
    a plausible `[wing-mask-fit]` line, and change nothing anyone reads.

    Checked structurally over EVERY such guard, so a guard added later without
    the flag is caught too.
    """
    import ast as _ast
    src = RUN_BOUT.read_text()
    fn = next(n for n in _ast.parse(src).body
              if isinstance(n, _ast.FunctionDef) and n.name == "process_bout_fly")
    consumers = ("outputs_h5_path", "qc_json_path", "qc_perframe_path")
    guards = []
    for node in _ast.walk(fn):
        if not isinstance(node, _ast.If):
            continue
        test = _ast.dump(node.test)
        if "stage_done" in test and any(c in test for c in consumers):
            guards.append((node.lineno, test))
    assert guards, "could not find the Stage E / QC stage_done guards"
    missing = [ln for ln, t in guards if "_wing_fit_rebuilt" not in t]
    assert not missing, (
        f"these stage_done guards on the refined-qpos consumers do not honour "
        f"_wing_fit_rebuilt, so a newly-run wing fit never reaches them: "
        f"lines {missing}")

    # ... and the flag must actually be RAISED by the branch that (re)fits,
    # otherwise every guard above honours a flag that is always False.
    fit_branch = [n for n in _ast.walk(fn)
                  if isinstance(n, _ast.If) and "'fit'" in _ast.dump(n.test)
                  and "_action" in _ast.dump(n.test)]
    assert fit_branch, "could not find the `if _action == 'fit':` branch"
    raised = any(isinstance(st, _ast.Assign)
                 and any(getattr(t, "id", None) == "_wing_fit_rebuilt"
                         for t in st.targets)
                 and getattr(st.value, "value", None) is True
                 for b in fit_branch for st in _ast.walk(b))
    assert raised, ("the branch that computes the fit must set "
                    "_wing_fit_rebuilt = True, or the guards never fire")


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
