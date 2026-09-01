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


def test_run_bout_emits_a_pose_value_this_parser_accepts():
    """The actual binding: every literal run_bout can pass to --pose must be in
    this parser's choices."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "scripts" / "run_bout.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "process_bout_fly")
    emitted = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.List):
            continue
        for i, el in enumerate(node.elts[:-1]):
            if not (isinstance(el, ast.Constant) and el.value == "--pose"):
                continue
            val = node.elts[i + 1]
            # only the VALUE branches -- an `X if _action in ("fit","reuse")
            # else Y` also mentions the action names in its test, which are not
            # things that reach argparse
            parts = ([val.body, val.orelse] if isinstance(val, ast.IfExp) else [val])
            for part in parts:
                emitted |= {c.value for c in ast.walk(part)
                            if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    assert emitted, "could not read the --pose values run_bout emits"
    p = cli.build_parser()
    sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
    choices = next(a.choices for a in sub.choices["sidebyside"]._actions
                   if a.dest == "pose")
    assert emitted <= set(choices), (
        f"run_bout emits --pose {sorted(emitted)} but the viz parser accepts "
        f"{sorted(choices)} -- the subprocess would exit 2, and Stage F's "
        f"failure is non-fatal, so sidebyside.mp4 would silently stop appearing")


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
