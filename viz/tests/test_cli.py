from viz import cli

def test_parser_has_all_subcommands():
    p = cli.build_parser()
    sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
    for name in ("overlay","legskel","kp-qc","fit-check","reproj-video","clip"):
        assert name in sub.choices

def test_overlay_parses_shared_flags():
    p = cli.build_parser()
    ns = p.parse_args(["overlay","--run","/r","--bout","1","--fly","0","--frame","250"])
    assert ns.run == "/r" and ns.bout == 1 and ns.fly == 0 and ns.frame == 250
    assert ns.func == "_overlay"   # name-based dispatch handle

def test_main_dispatches(monkeypatch):
    called = {}
    import viz.cli as c
    monkeypatch.setattr(c, "_overlay", lambda args: (called.setdefault("frame", args.frame), 0)[1])
    assert c.main(["overlay","--run","/r","--bout","1","--fly","0","--frame","7"]) == 0
    assert called["frame"] == 7   # main resolved c._overlay at call time (name-based)

def test_kp_qc_parses_paired_recs():
    p = cli.build_parser()
    ns = p.parse_args(["kp-qc","--run-dir","/r","--female-vs-male",
                        "--female-rec","F","--male-rec","M"])
    assert ns.female_vs_male is True
    assert ns.female_rec == "F" and ns.male_rec == "M"
    assert ns.func == "_kp_qc"

def test_kp_qc_n_defaults_to_none():
    # kp-qc's --n must NOT hardcode 8 (viz_keypoints.py's single-mode default):
    # the view applies faithful per-mode defaults (8 single / 5 paired) itself,
    # keyed off `args.n is None`. If this default ever becomes non-None again,
    # paired mode silently reverts to the wrong (single-mode) frame count.
    p = cli.build_parser()
    ns = p.parse_args(["kp-qc","--run-dir","/r"])
    assert ns.n is None

def test_fit_check_n_default_unaffected():
    # fit-check has its own unrelated --n (default 10); guard against a
    # shared-parser refactor accidentally zeroing it out too.
    p = cli.build_parser()
    ns = p.parse_args(["fit-check","dummy.h5"])
    assert ns.n == 10


def test_pose_flag_parses_for_every_view_that_draws_a_fitted_pose():
    """F4: run_bout shells out to `python -m viz sidebyside ... --pose <x>`, and
    Stage F's failure is NON-FATAL -- so if the flag name or its choices drift,
    argparse exits 2, the message is swallowed, and the pipeline just silently
    stops producing sidebyside.mp4. Nothing bound the two sides together."""
    import pytest
    p = cli.build_parser()
    for view, extra in (("sidebyside", []), ("rigcam", ["--model-xml", "/m.xml"])):
        base = [view, "--run", "/r", "--bout", "1", "--fly", "0"] + extra
        assert p.parse_args(base).pose == "auto", "the default must stay 'auto'"
        for want in ("auto", "wingfit", "refined"):
            assert p.parse_args(base + ["--pose", want]).pose == want
        with pytest.raises(SystemExit):
            p.parse_args(base + ["--pose", "nonsense"])


def _run_bout_fn(name):
    """exec one self-contained module-level function out of scripts/run_bout.py.

    run_bout imports jax/mujoco/stac_mjx/hydra at module level, so it cannot be
    imported here; the AST trick keeps this a cheap parser-level test.
    """
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "scripts" / "run_bout.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<rb>", "exec"), {}, ns)
    return ns[name]


# every value cfg.outputs.sidebyside_right can take, and every Stage-D2 action
RIGHT_MODES = ("rigcam", "reproj", "mujoco")
WING_ACTIONS = ("off", "fit", "reuse")


def test_run_bout_emits_a_pose_value_this_parser_accepts():
    """The value half of the binding: every literal run_bout can pass to --pose
    must be in this parser's choices."""
    pose_args = _run_bout_fn("sidebyside_pose_args")
    emitted = set()
    for right in RIGHT_MODES:
        for action in WING_ACTIONS:
            a = pose_args(right, action)
            assert a == [] or (len(a) == 2 and a[0] == "--pose"), \
                f"unexpected argv fragment for {right}/{action}: {a}"
            emitted |= set(a[1:])
    assert emitted, "could not read the --pose values run_bout emits"
    p = cli.build_parser()
    sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
    choices = next(a.choices for a in sub.choices["sidebyside"]._actions
                   if a.dest == "pose")
    assert emitted <= set(choices), (
        f"run_bout emits --pose {sorted(emitted)} but the viz parser accepts "
        f"{sorted(choices)} -- the subprocess would exit 2, and Stage F's "
        f"failure is non-fatal, so sidebyside.mp4 would silently stop appearing")


def test_run_bout_never_emits_a_right_pose_combination_the_renderer_refuses():
    """The COMBINATION half, which the value check could not catch -- and a
    combination is exactly what broke.

    `outputs.sidebyside_right` is a live config knob, and run_bout used to emit
    an explicit, never-'auto' --pose unconditionally. With `mujoco` selected,
    `check_pose_mode` then refused every Stage-F subprocess, on every bout,
    whether or not wing_mask_fit was enabled -- and Stage F is non-fatal, so
    nothing crashed: sidebyside.mp4 just silently stopped being produced. That
    is the very failure the value-binding test was added to prevent, one level
    up. So bind the PAIR, through the real guard, and parse it with the real
    parser.
    """
    from viz.views import sidebyside
    pose_args = _run_bout_fn("sidebyside_pose_args")
    p = cli.build_parser()
    for right in RIGHT_MODES:
        for action in WING_ACTIONS:
            argv = (["sidebyside", "--run", "/r", "--bout", "1", "--fly", "0",
                     "--right", right] + pose_args(right, action))
            ns = p.parse_args(argv)               # would SystemExit on a bad value
            assert ns.right == right
            # must not raise -- run_bout may never ask for what viz refuses
            sidebyside.check_pose_mode(ns.right, ns.pose)

    # and the guard is still doing its job when a HUMAN asks for it
    import pytest
    with pytest.raises(ValueError):
        sidebyside.check_pose_mode("mujoco", "wingfit")
    # the rigcam/reproj panels must still be told the pose explicitly, or the
    # renderer falls back to 'auto' and Task 7 cannot pin an arm
    assert pose_args("rigcam", "fit") == ["--pose", "wingfit"]
    assert pose_args("rigcam", "off") == ["--pose", "refined"]
    assert pose_args("reproj", "reuse") == ["--pose", "wingfit"]
    assert pose_args("mujoco", "fit") == [], \
        "the mujoco panel renders stac_ik.h5; --pose there is meaningless"


def test_check_pose_mode_runs_before_any_expensive_work():
    """A rejected combination must not first load masks, score the left camera,
    makedirs the output, or compose Hydra."""
    import ast
    import inspect
    from viz.views import sidebyside
    fn = next(n for n in ast.parse(inspect.getsource(sidebyside)).body
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    checks = [n.lineno for n in ast.walk(fn)
              if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "check_pose_mode"]
    assert checks, "run() never validates the --right/--pose combination"
    costly = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
              and ast.unparse(n.func) in ("os.makedirs", "vio.load_masks",
                                          "vio.load_outputs", "compose")]
    assert costly, "expected some expensive calls in run() to order against"
    assert min(checks) < min(costly), (
        f"check_pose_mode is called at line {min(checks)}, after work at line "
        f"{min(costly)} -- a rejected call pays for all of it before raising")


def test_sidebyside_rejects_a_pose_override_the_mujoco_panel_cannot_honour():
    """F3: `--right mujoco` renders qpos straight out of stac_ik.h5 -- the
    PRE-BRIDGE STAC pose -- so it cannot show a wing fit at all. Silently
    ignoring --pose there is the worst of the three options."""
    import pytest
    from viz.views import sidebyside
    assert sidebyside.check_pose_mode("rigcam", "wingfit") is None
    assert sidebyside.check_pose_mode("mujoco", "auto") is None
    with pytest.raises(ValueError, match="stac_ik.h5"):
        sidebyside.check_pose_mode("mujoco", "wingfit")
    with pytest.raises(ValueError, match="--right rigcam"):
        sidebyside.check_pose_mode("mujoco", "refined")
