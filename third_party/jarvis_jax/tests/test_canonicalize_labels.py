import os, sys, json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))


def _bout(root, bi):
    for fly in (0, 1):
        d = os.path.join(root, "bouts", f"bout_{bi:05d}", f"fly{fly}")
        os.makedirs(d)
        open(os.path.join(d, f"orig{fly}"), "w").close()


def test_apply_manual_labels(tmp_path):
    import canonicalize_session_sex as css
    root = str(tmp_path / "pose")
    _bout(root, 1); _bout(root, 2)
    res = css.apply_manual_labels(root, {1: 0, 2: 1})     # bout1 male=fly0, bout2 male=fly1
    # bout1: male was fly0 -> swapped -> the dir that held fly0's marker is now fly1
    assert os.path.exists(os.path.join(root, "bouts", "bout_00001", "fly1", "orig0"))
    sj = json.load(open(os.path.join(root, "bouts", "bout_00001", "sex.json")))
    assert sj["method"] == "manual" and sj["male_fly"] == 1 and sj["applied_swap"] is True
    # bout2: male already fly1 -> no swap
    assert os.path.exists(os.path.join(root, "bouts", "bout_00002", "fly1", "orig1"))
    assert json.load(open(os.path.join(root, "bouts", "bout_00002", "sex.json")))["applied_swap"] is False


def test_apply_manual_labels_idempotent(tmp_path):
    import canonicalize_session_sex as css
    root = str(tmp_path / "pose")
    _bout(root, 1)
    css.apply_manual_labels(root, {1: 0})                 # swaps
    css.apply_manual_labels(root, {1: 0})                 # already manual -> skip
    assert os.path.exists(os.path.join(root, "bouts", "bout_00001", "fly1", "orig0"))  # not re-swapped


def test_apply_manual_labels_dry_run(tmp_path):
    import canonicalize_session_sex as css
    root = str(tmp_path / "pose")
    _bout(root, 1)
    css.apply_manual_labels(root, {1: 0}, dry_run=True)
    assert os.path.exists(os.path.join(root, "bouts", "bout_00001", "fly0", "orig0"))  # NOT moved
    assert not os.path.exists(os.path.join(root, "bouts", "bout_00001", "sex.json"))   # NOT written
