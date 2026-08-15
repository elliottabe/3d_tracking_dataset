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


def test_build_variant_root_freeze_subset_recomputes_downstream(tmp_path):
    # A Stage-B treatment (e.g. detector.reproj_resid_px) must see frozen kp2d
    # but RECOMPUTE kp3d/kp3d_filt: linking them would make run_bout.py's
    # stage-skipping silently serve the baseline triangulation for every
    # variant, and the A/B would compare a run against itself.
    m, dest = _frozen(tmp_path)
    vroot = build_variant_root(m, dest, "stageb",
                               frozen_inputs=("kp2d.npz",))
    fly0 = vroot / "courtship_r" / "bouts" / "bout_00001" / "fly0"
    assert (fly0 / "kp2d.npz").exists()
    assert not (fly0 / "kp3d.npz").exists()
    assert not (fly0 / "kp3d_filt.npz").exists()


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
    assert "bout_ids=1" in c
    assert "scaling.scale_keypoints=all" in c and "scaling.estimator=norm_ratio" in c


def test_variant_commands_merges_bouts_per_run_key(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    # Create two bouts with the same run_key
    make_source_bout(src, 3, flies=(0, 1), sex_json={"male_fly": 1})
    make_source_bout(src, 8, flies=(0, 1), sex_json={"male_fly": 1})
    # Build manifest with both bouts in the same run_key
    m = {
        "benchmark_root": "",
        "proximity_threshold_bl": 2.0,
        "bouts": [
            {
                "run_key": "courtship_r",
                "layout": "canonical",
                "source_root": str(src),
                "recording_cfg": "session1",
                "session_dir": "/nope",
                "assay": "courtship",
                "bout": 3,
                "flies": [0, 1],
                "tags": [],
                "render_frames": [],
            },
            {
                "run_key": "courtship_r",
                "layout": "canonical",
                "source_root": str(src),
                "recording_cfg": "session1",
                "session_dir": "/nope",
                "assay": "courtship",
                "bout": 8,
                "flies": [0, 1],
                "tags": [],
                "render_frames": [],
            },
        ],
    }
    freeze(m, dest)
    vroot = build_variant_root(m, dest, "v")
    cmds = variant_commands(m, vroot, [])
    # Should emit ONE command per run_key, not per bout
    assert len(cmds) == 1
    c = cmds[0]
    # Bouts must be merged with escaped quotes (hydra sweep syntax)
    assert "bout_ids=\\'3,8\\'" in c


def test_render_bout_smoke(tmp_path):
    import mujoco  # noqa: F401  (skip if unavailable)
    from scripts.benchmark.render_grid import render_bout
    xml = tmp_path / "m.xml"
    xml.write_text("""<mujoco><worldbody><body name="root">
      <joint type="free"/><geom size="0.02"/></body></worldbody></mujoco>""")
    qpos = np.zeros((10, 7)); qpos[:, 3] = 1.0     # identity quat
    out = tmp_path / "grid.png"
    render_bout(qpos, [0, 5, 9], str(xml), out, size=(64, 64))
    assert out.exists() and out.stat().st_size > 0
