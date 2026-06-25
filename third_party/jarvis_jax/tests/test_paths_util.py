from jarvis_jax.predict.paths_util import dataset_for, processed_dir_for, resolve_auto


def test_dataset_for():
    assert dataset_for("/d/Johnson_lab/Video_recordings/courtship/Session1/2026_04_02_11_52_43") == "courtship"
    assert dataset_for("/x/free_running/Session0/rec/") == "free_running"   # trailing slash
    assert dataset_for("too/short", default="courtship") == "courtship"      # <3 comps -> default
    assert dataset_for("a/b") is None                                        # <3 comps, no default


def test_processed_dir_for():
    pr = "/d/Johnson_lab/processed"
    sd = "/d/Johnson_lab/Video_recordings/courtship/Session1/2026_04_02_11_52_43"
    assert processed_dir_for(pr, sd) == "/d/Johnson_lab/processed/courtship/Session1/2026_04_02_11_52_43"
    # explicit dataset overrides derivation
    assert processed_dir_for(pr, sd, dataset="free_running") == \
        "/d/Johnson_lab/processed/free_running/Session1/2026_04_02_11_52_43"


def test_resolve_auto():
    assert resolve_auto("auto", "/fb") == "/fb"
    assert resolve_auto("AUTO", "/fb") == "/fb"
    assert resolve_auto("", "/fb") == "/fb"
    assert resolve_auto(None, "/fb") == "/fb"
    assert resolve_auto("/explicit/path", "/fb") == "/explicit/path"
