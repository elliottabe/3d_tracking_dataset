import os, numpy as np
from jarvis_jax.cse.courtship_resume import (atomic_save_npz, atomic_save_json,
                                             stage_done, mark_done, bout_complete)


def test_atomic_save_and_stage_done(tmp_path):
    p = str(tmp_path / "kp2d.npz")
    assert not stage_done(p)
    atomic_save_npz(p, a=np.zeros(3))
    assert stage_done(p) and not os.path.exists(p + ".tmp")
    atomic_save_json(str(tmp_path / "qc.json"), {"x": 1})
    assert stage_done(str(tmp_path / "qc.json"))


def test_done_marker(tmp_path):
    d = str(tmp_path / "bout_00001" / "fly0"); os.makedirs(d)
    assert not bout_complete(d)
    mark_done(d)
    assert bout_complete(d)


def test_stage_skip_logic(tmp_path):
    # a present artifact is "done"; a leftover .tmp is NOT
    p = str(tmp_path / "stac_ik.h5"); open(p + ".tmp", "w").write("partial")
    assert not stage_done(p)
    open(p, "w").write("real")
    assert stage_done(p)
