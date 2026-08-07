"""Tests for scripts/benchmark/run_variant.py (no GPU: build + command emission)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.benchmark.freeze_inputs import freeze
from scripts.benchmark.run_variant import build_variant_root, variant_commands
from tests.test_benchmark_freeze import make_source_bout, manifest_for


def _frozen(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    make_source_bout(src, 1, flies=(0, 1), sex_json={"male_fly": 1})
    m = manifest_for(src)
    freeze(m, dest)
    return m, dest


def test_build_variant_root_links_inputs_not_shared_artifacts(tmp_path):
    m, dest = _frozen(tmp_path)
    # simulate stale shared artifacts in frozen tree root: must NOT propagate
    (dest / "frozen" / "courtship_r" / "scale.json").write_text("{}")
    vroot = build_variant_root(m, dest, "all_norm")
    assert vroot == dest / "variants" / "all_norm"
    fly0 = vroot / "courtship_r" / "bouts" / "bout_00001" / "fly0"
    assert (fly0 / "kp2d.npz").exists()
    assert (fly0 / "kp3d.npz").exists()
    assert not (vroot / "courtship_r" / "scale.json").exists()


def test_variant_commands_shape(tmp_path):
    m, dest = _frozen(tmp_path)
    vroot = build_variant_root(m, dest, "v")
    cmds = variant_commands(m, vroot,
                            ["scaling.scale_keypoints=all",
                             "scaling.estimator=norm_ratio"])
    assert len(cmds) == 1
    c = cmds[0]
    assert c.startswith("python scripts/run_bout.py ")
    assert "recording=session1" in c
    assert "recording.session_dir=/nope" in c
    assert f"outputs.out={vroot}/courtship_r" in c
    assert "+bout_ids=1" in c
    assert "scaling.scale_keypoints=all" in c and "scaling.estimator=norm_ratio" in c
