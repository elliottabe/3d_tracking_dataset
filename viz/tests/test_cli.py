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
