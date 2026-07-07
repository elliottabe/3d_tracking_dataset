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
